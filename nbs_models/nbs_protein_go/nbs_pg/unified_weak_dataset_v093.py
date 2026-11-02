"""CPU construction of immutable frequency-routed weak supervision for NBS.

This module deliberately avoids importing nbs_pg, torch, labels from independent
tests, or model code. All source arrays remain read-only. Thresholds are fixed
defaults unless an explicit, provenance-recorded policy file is supplied.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'nbs-unified-dataset-v093-r1'
GROUPS = [('zero', 0, 0), ('1-5', 1, 5), ('6-9', 6, 9),
          ('10-99', 10, 99), ('100-999', 100, 999), ('1000+', 1000, None)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.asarray(value, dtype='<i8').tobytes()).hexdigest()


def resolve(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def file_identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def load_unified_manifest(path, verify=True):
    path = Path(path).resolve()
    result = read_json(path)
    if result.get('schema') != SCHEMA or result.get('complete') is not True:
        raise ValueError('Use a complete unified v093 dataset manifest')
    if verify:
        for name, expected in result['hashes'].items():
            if sha256(resolve(path.parent, name)) != expected:
                raise ValueError('Unified dataset file hash mismatch: ' + name)
    return result


def load_registry(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'protein_idx', 'role', 'role_row_idx'} <= set(reader.fieldnames or []):
            raise ValueError('Invalid protein registry schema')
        records = list(reader)
    if not records or any(int(r['protein_idx']) != i for i, r in enumerate(records)):
        raise ValueError('Protein global IDs must be contiguous in registry row order')
    roles = np.asarray([r['role'] for r in records])
    local = np.asarray([int(r['role_row_idx']) for r in records], np.int64)
    role_ids = {}
    for role in set(roles):
        ids = np.flatnonzero(roles == role)
        ids = ids[np.argsort(local[ids], kind='stable')]
        if not np.array_equal(local[ids], np.arange(len(ids))):
            raise ValueError('Role-local IDs must be contiguous: ' + role)
        role_ids[role] = ids
    if 'core' not in role_ids or 'weak' not in role_ids:
        raise ValueError('Both core and weak protein roles are required')
    return records, roles, local, role_ids


def read_gold(spec, base, num_proteins, num_go):
    ptr_path, idx_path = resolve(base, spec['indptr']), resolve(base, spec['go_idx'])
    ptr, idx = [np.load(p, mmap_mode='r', allow_pickle=False) for p in (ptr_path, idx_path)]
    if (int(spec['num_go']) != num_go or ptr.dtype.kind not in 'iu' or idx.dtype.kind not in 'iu'
            or ptr.shape != (num_proteins + 1,) or idx.ndim != 1 or ptr[0] != 0
            or ptr[-1] != len(idx) or (ptr[1:] < ptr[:-1]).any()):
        raise ValueError('Invalid global protein-major gold CSR')
    for start in range(0, len(idx), 1_000_000):
        v = idx[start:start + 1_000_000]
        if len(v) and (v.min() < 0 or v.max() >= num_go):
            raise ValueError('Gold GO outside the immutable classifier columns')
    return ptr, idx, ptr_path, idx_path


def training_core_split(core_ids, ptr, ft):
    labeled = np.asarray(core_ids)[np.diff(ptr)[core_ids] > 0]
    n = ft.get('holdout_core_count', 0)
    if type(n) is not int or not 0 <= n < len(labeled):
        raise ValueError('Invalid core holdout size')
    validation = np.sort(np.random.default_rng(int(ft.get('holdout_seed', 8080))).permutation(labeled)[:n])
    return np.setdiff1d(labeled, validation), validation


def count_gold(ptr, idx, core_ids, num_go):
    counts = np.zeros(num_go, np.int64)
    for protein in core_ids:
        labels = np.unique(idx[int(ptr[protein]):int(ptr[protein + 1])])
        counts[labels] += 1
    return counts


def frequency_groups(counts):
    result = np.full(len(counts), -1, np.int8)
    for index, (_, lower, upper) in enumerate(GROUPS):
        result[(counts >= lower) & (True if upper is None else counts <= upper)] = index
    if (result < 0).any():
        raise ValueError('Invalid negative core frequency')
    return result


def make_seen_mask(counts, mapping, fraction=0., seed=1337):
    """Whole canonical groups; natural zero groups never receive BCE negatives."""
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('label holdout fraction must be in [0,1)')
    mapping = np.asarray(mapping)
    if mapping.shape != counts.shape or mapping.dtype.kind not in 'iu' or (mapping < 0).any():
        raise ValueError('Invalid task-to-ontology mapping')
    positive_rows = np.unique(mapping[counts > 0])
    if not len(positive_rows):
        raise ValueError('At least one training-positive canonical GO is required')
    heldout = np.empty(0, np.int64)
    if fraction:
        if len(positive_rows) < 2:
            raise ValueError('Canonical label holdout needs at least two positive groups')
        n = max(1, min(len(positive_rows)-1, int(np.floor(len(positive_rows) * fraction))))
        heldout = np.sort(np.random.default_rng(int(seed)).permutation(positive_rows)[:n])
    seen_rows = np.setdiff1d(positive_rows, heldout)
    seen = np.isin(mapping, seen_rows)
    return seen, heldout


def threshold_policy(path=None):
    thresholds = {name: .5 for name, _, _ in GROUPS}
    info = {'source': 'fixed_default_0.5', 'independently_calibrated': False,
            'labels_used_for_selection': False,
            'note': 'Fixed exploratory thresholds; not fitted or optimal thresholds.'}
    if path is not None:
        value = read_json(path)
        supplied = value.get('thresholds', value)
        if not isinstance(supplied, dict) or set(supplied) - set(thresholds):
            raise ValueError('Threshold JSON requires known frequency group names')
        thresholds.update(supplied)
        info = {'source': 'explicit_thresholds_json', 'file': file_identity(path),
                'provenance': value.get('provenance', {}),
                'independently_calibrated': value.get('independently_calibrated') is True,
                'note': 'Caller-supplied provenance is recorded; it is not independently audited.'}
    for name, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Threshold must be finite and in [0,1]: ' + name)
    return thresholds, info


def choose_candidate_rows(go, attrs, counts, seen, *, mode='frequency', topk=16):
    """Budget 4 rare + 4 medium + 8 global, filled by ranked unused columns.

    Filtering occurs before selection. Original reciprocal-rank/probability/GO
    ordering is retained inside each quota; output columns are never duplicated.
    """
    go, attrs = np.asarray(go), np.asarray(attrs)
    if (go.ndim != 2 or attrs.shape != (*go.shape, 3) or go.dtype.kind not in 'iu'
            or attrs.dtype.kind != 'f' or not np.isfinite(attrs).all()):
        raise ValueError('Candidate rows require integer [N,K] and finite [N,K,3]')
    if mode not in ('frequency', 'global') or topk != 16:
        raise ValueError('Use frequency/global with the fixed 16-candidate budget')
    n_go = len(counts)
    if (go < -1).any() or (go >= n_go).any():
        raise ValueError('Source candidate GO outside immutable columns')
    safe_go = np.maximum(go, 0)
    valid = (go >= 0) & seen[safe_go]
    rank = np.where(valid, attrs[..., 2], -np.inf)
    order = np.lexsort((go, -attrs[..., 0], -rank), axis=1)
    ordered = np.take_along_axis(go, order, axis=1)
    ordered_attr = np.take_along_axis(attrs, order[..., None], axis=1)
    allowed = np.take_along_axis(valid, order, axis=1)
    output = np.full((len(go), topk), -1, np.int32)
    output_attr = np.zeros((len(go), topk, 3), np.float32)
    for row in range(len(go)):
        chosen, used = [], set()
        if mode == 'frequency':
            for lower, upper, budget in [(1, 9, 4), (10, 999, 4)]:
                ordered_count = counts[np.maximum(ordered[row], 0)]
                positions = np.flatnonzero(allowed[row] & (ordered_count >= lower)
                                           & (ordered_count <= upper))
                for pos in positions:
                    label = ordered[row, pos]
                    label = int(label)
                    if label not in used:
                        chosen.append(int(pos)); used.add(label)
                        budget -= 1
                        if not budget:
                            break
        for pos, label in enumerate(ordered[row]):
            if len(chosen) == topk:
                break
            if allowed[row, pos] and int(label) not in used:
                chosen.append(pos); used.add(int(label))
        if chosen:
            output[row, :len(chosen)] = ordered[row, chosen]
            output_attr[row, :len(chosen)] = ordered_attr[row, chosen]
    return output, output_attr


def _load_go_mapping(project_root, config, weak, weak_path, num_go):
    folder = Path(project_root) / 'nbs_models/nbs_protein_go/nbs_pg'
    spec = importlib.util.spec_from_file_location('_v093_source_go_registry', folder/'go_registry_v093.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = resolve(project_root, config['data']['root'])
    task_path = root/'gg_relations/go_registry.tsv'
    if not task_path.is_file():
        task_path = resolve(weak_path.parent, weak['go_registry']['path'])
    full_path = resolve(project_root, config['data']['full_go_box_manifest'])
    alignment_path = resolve(project_root, config['go_boxsqel']['alignment_manifest'])
    registry = module.GORegistryV093.from_manifests(full_path, task_path, alignment_path)
    expected_task_hash = weak.get('go_registry', {}).get('sha256')
    if expected_task_hash is not None and sha256(task_path) != expected_task_hash:
        raise ValueError('Source exporter GO registry hash differs from the configured task columns')
    mapping = np.asarray(registry.task_to_ontology, np.int64)
    if mapping.shape != (num_go,):
        raise ValueError('GO registry does not preserve all task columns')
    full = read_json(full_path)
    registry_path = resolve(full_path.parent, full['registry']['file'])
    aligned = read_json(alignment_path)
    mapping_path = resolve(alignment_path.parent, aligned['arrays']['source_row']['file'])
    return mapping, registry, {'task_registry': task_path, 'full_ontology_manifest': full_path,
                              'ontology_registry': registry_path, 'alignment_manifest': alignment_path,
                              'alignment_mapping': mapping_path}


def _probability_path(weak, weak_path, role, name, explicit):
    if explicit is not None:
        return Path(explicit).resolve()
    record = next((item for item in weak.get('roles', []) if item.get('role') == role), None)
    if record is None or not record.get(name):
        raise FileNotFoundError('Weak manifest has no saved ' + name + '; re-export aligned probabilities or provide an explicit file')
    return resolve(weak_path.parent, record[name])


def _validate_prob_block(value, shape):
    if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all() or (value < 0).any() or (value > 1).any():
        raise ValueError('Teacher probabilities must be finite, aligned, floating and within [0,1]')


def _save_npy(directory, name, value):
    np.save(directory/name, value)
    return name


def build_dataset(args, project_root=ROOT):
    started = time.monotonic()
    project_root = Path(project_root).resolve()
    work = resolve(project_root, args.source_work_dir)
    config_path = work/'resolved_config.json'
    config = read_json(config_path)
    if config['full_task'].get('label_split', 'all_seen') not in (None, 'all_seen'):
        raise ValueError('Build from the original all_seen source config')
    root = resolve(project_root, config['data']['root'])
    weak_path = resolve(root, config['data']['weak_graph_predictions_manifest'])
    inverted_path = resolve(root, config['data']['go_protein_inverted_index_manifest'])
    weak, inverted = read_json(weak_path), read_json(inverted_path)
    if weak.get('task') != config.get('task'):
        raise ValueError('Source task mismatch')
    num_go = int(weak['go_registry']['num_terms'])
    registry_path = root/'features/protein_registry.csv'
    expected_registry_hash = weak.get('protein_registry', {}).get('sha256')
    if expected_registry_hash is not None and sha256(registry_path) != expected_registry_hash:
        raise ValueError('Source exporter protein registry hash differs from current registry')
    records, roles, role_rows, role_ids = load_registry(registry_path)
    gold_ptr, gold_go, gold_ptr_path, gold_go_path = read_gold(
        inverted['indices']['gold']['protein_major'], inverted_path.parent, len(roles), num_go)
    core_ids, validation_ids = training_core_split(role_ids['core'], gold_ptr, config['full_task'])
    counts = count_gold(gold_ptr, gold_go, core_ids, num_go)
    mapping, go_registry, go_sources = _load_go_mapping(project_root, config, weak, weak_path, num_go)
    seen, heldout_rows = make_seen_mask(counts, mapping, args.label_holdout_fraction, args.seed)
    thresholds, threshold_info = threshold_policy(args.thresholds_json)
    if args.thresholds_json is not None:
        calibration = read_json(args.thresholds_json)
        if calibration.get('schema') == 'nbs-backbone-group-thresholds-v093-r1':
            calibration_identity = calibration.get('identity', {})
            required_identity = {'training_core_ids_sha256': array_sha256(core_ids),
                                 'core_counts_sha256': array_sha256(counts),
                                 'seen_task_mask_content_sha256_int64': array_sha256(seen),
                                 'low_frequency_max': args.low_frequency_max}
            if (calibration.get('complete') is not True
                    or any(calibration_identity.get(key) != value for key,value in required_identity.items())
                    or calibration_identity.get('sources', {}).get('weak_manifest', {}).get('sha256') != sha256(weak_path)):
                raise ValueError('Backbone calibration source/supervision split differs from the dataset being built')
    groups = frequency_groups(counts)
    # Match CPU calibration exactly: stored teacher values are compared against
    # the float64 JSON/grid threshold, without rounding that threshold to f32.
    threshold_vector = np.asarray([thresholds[GROUPS[int(g)][0]] for g in groups], np.float64)
    backbone_path = _probability_path(weak, weak_path, 'weak', 'backbone_dense_file', args.backbone_prob)
    modelout_path = _probability_path(weak, weak_path, 'weak', 'modelout_dense_file', args.modelout_prob)
    backbone = np.load(backbone_path, mmap_mode='r', allow_pickle=False)
    modelout = np.load(modelout_path, mmap_mode='r', allow_pickle=False)
    expected_shape = (len(role_ids['weak']), num_go)
    if backbone.shape != expected_shape or modelout.shape != expected_shape:
        raise ValueError('Weak teacher matrices must use registry role-local order and all immutable columns')
    weak_record = next((r for r in weak['roles'] if r.get('role') == 'weak'), {})
    if weak_record.get('rows', expected_shape[0]) != expected_shape[0]:
        raise ValueError('Source exporter row count differs from weak registry')
    old_spec = weak['weak_pseudo_targets']
    old_ptr_path = resolve(weak_path.parent, old_spec['csr_indptr_file'])
    old_go_path = resolve(weak_path.parent, old_spec['csr_indices_file'])
    old_probability_path = resolve(weak_path.parent, old_spec['csr_probability_file'])
    old_ptr, old_go, old_probability = [np.load(path, mmap_mode='r', allow_pickle=False)
        for path in (old_ptr_path, old_go_path, old_probability_path)]
    if (old_ptr.shape != (expected_shape[0]+1,) or old_go.ndim != 1 or old_ptr.dtype.kind not in 'iu'
            or old_go.dtype.kind not in 'iu' or old_probability.shape != old_go.shape
            or old_ptr[0] != 0 or old_ptr[-1] != len(old_go) or (old_ptr[1:] < old_ptr[:-1]).any()):
        raise ValueError('Invalid original weak membership CSR')
    default_modelout = weak_record.get('modelout_dense_file')
    restore_quantized_membership = bool(default_modelout and
        modelout_path == resolve(weak_path.parent, default_modelout)
        and old_spec.get('comparison') == '>' and old_spec.get('threshold') == .5)
    if records[0].get('protein_id') is not None and weak_record.get('protein_ids_sha256'):
        digest = hashlib.sha256()
        for protein in role_ids['weak']:
            digest.update(records[int(protein)]['protein_id'].encode()); digest.update(b'\n')
        if digest.hexdigest() != weak_record['protein_ids_sha256']:
            raise ValueError('Weak exporter protein-ID order differs from registry')
    candidate_spec = weak.get('backbone_candidate_edges') or weak.get('backbone_rare_edges')
    if not candidate_spec:
        raise ValueError('Source manifest has no candidate evidence')
    edge_path = resolve(weak_path.parent, candidate_spec['edge_index_file'])
    attr_path = resolve(weak_path.parent, candidate_spec['edge_attr_file'])
    candidate_info = inverted['indices']['candidate']
    degree = candidate_info.get('fixed_degree')
    source_start = candidate_info.get('source_protein_start')
    if type(degree) is not int or degree < 1 or type(source_start) is not int:
        raise ValueError('Source candidate index must declare fixed degree and global start')
    sources = {'weak_manifest': weak_path, 'inverted_manifest': inverted_path,
               'protein_registry': registry_path, 'gold_indptr': gold_ptr_path, 'gold_go_idx': gold_go_path,
               'backbone_weak': backbone_path, 'modelout_weak': modelout_path,
               'old_pseudo_indptr': old_ptr_path, 'old_pseudo_go_idx': old_go_path,
               'old_pseudo_probability': old_probability_path,
               'candidate_edge_index': edge_path, 'candidate_edge_attr': attr_path, **go_sources}
    print('[dataset] hashing source inputs', flush=True)
    policies = {'low_frequency_max': args.low_frequency_max, 'low_frequency_teacher': 'backbone',
                'high_frequency_teacher': 'source_stage1_modelout', 'thresholds': thresholds,
                'threshold_provenance': threshold_info, 'comparison': '>',
                'candidate_mode': args.candidate_mode, 'candidate_topk': 16,
                'candidate_quotas': {'1-9': 4, '10-999': 4, 'global_fill': 8} if args.candidate_mode == 'frequency' else None,
                'label_holdout_fraction': args.label_holdout_fraction, 'label_holdout_seed': args.seed,
                'zero_supervision_policy': 'exclude_whole_canonical_groups_without_training_core_positive',
                'quantization_policy': 'For the unchanged source modelout at threshold 0.5, restore original CSR positives rounded to dense 0.5; no rethresholding of old membership.',
                'restore_quantized_modelout_membership': restore_quantized_membership,
                'strict_pipeline_zero_shot': False}
    identity = {'source_config': file_identity(config_path),
                'sources': {name: file_identity(path) for name, path in sources.items()},
                'policies': policies, 'builder_sha256': sha256(__file__),
                'go_registry_implementation_sha256': sha256(Path(__file__).with_name('go_registry_v093.py')),
                'go_registry_contract_sha256': hashlib.sha256(json.dumps(
                    go_registry.contract(), sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                'training_core_ids_sha256': array_sha256(core_ids),
                'core_counts_sha256': array_sha256(counts),
                'task_to_ontology_sha256': array_sha256(mapping)}
    out = resolve(project_root, args.output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_name(out.name+'.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'unified_dataset_manifest.json').is_file():
            existing = load_unified_manifest(out/'unified_dataset_manifest.json')
            if existing['identity'] != identity:
                raise ValueError('Existing dataset has a different source/policy identity; use a new output directory')
            print('[dataset] complete identical dataset reused: ' + str(out), flush=True)
            return existing
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Refusing to replace a nonempty incomplete dataset directory')
        stage = Path(tempfile.mkdtemp(prefix=out.name+'.building.', dir=out.parent))
        try:
            ptr = np.zeros(expected_shape[0]+1, np.int64)
            pseudo_counts = np.zeros(num_go, np.int64)
            old_counts = np.zeros(num_go, np.int64)
            restored_boundary_memberships = 0
            source_vector = np.where(counts <= args.low_frequency_max, 1, 2).astype(np.uint8)
            use_b = source_vector == 1
            raw_paths = {key: stage/(key+'.raw') for key in ['go', 'probability', 'source']}
            with raw_paths['go'].open('wb') as gi, raw_paths['probability'].open('wb') as pr, raw_paths['source'].open('wb') as sr:
                for start in range(0, expected_shape[0], args.chunk_rows):
                    stop = min(start+args.chunk_rows, expected_shape[0])
                    b = np.asarray(backbone[start:stop], np.float32)
                    m = np.asarray(modelout[start:stop], np.float32)
                    _validate_prob_block(b, (stop-start, num_go)); _validate_prob_block(m, b.shape)
                    chosen = np.where(use_b[None], b, m)
                    positive = (chosen > threshold_vector[None]) & seen[None]
                    old_start, old_stop = int(old_ptr[start]), int(old_ptr[stop])
                    old_labels = np.asarray(old_go[old_start:old_stop], np.int64)
                    old_p = np.asarray(old_probability[old_start:old_stop], np.float32)
                    if len(old_labels) and ((old_labels < 0).any() or (old_labels >= num_go).any()
                            or not np.isfinite(old_p).all() or (old_p < 0).any() or (old_p > 1).any()):
                        raise ValueError('Invalid original weak GO/probability payload')
                    old_counts += np.bincount(old_labels, minlength=num_go)
                    if restore_quantized_membership and len(old_labels):
                        old_row = np.repeat(np.arange(stop-start), np.diff(old_ptr[start:stop+1]))
                        if not np.array_equal(m[old_row, old_labels], old_p):
                            raise ValueError('Original modelout dense probabilities disagree with its CSR payload')
                        restore = (~use_b[old_labels] & (threshold_vector[old_labels] == .5)
                                   & seen[old_labels] & (m[old_row, old_labels] == .5))
                        restored_boundary_memberships += int(restore.sum())
                        positive[old_row[restore], old_labels[restore]] = True
                    row, label = np.nonzero(positive)
                    label.astype(np.int32).tofile(gi)
                    chosen[row, label].astype(np.float32).tofile(pr)
                    source_vector[label].tofile(sr)
                    ptr[start+1:stop+1] = ptr[start] + np.cumsum(positive.sum(1, dtype=np.int64))
                    pseudo_counts += np.bincount(label, minlength=num_go)
                    if start == 0 or stop == expected_shape[0] or start//16384 != stop//16384:
                        print(f'[weak labels] {stop}/{expected_shape[0]} rows; positives={int(ptr[stop])}', flush=True)
            pseudo_files = {'indptr': 'weak_indptr.i64.npy', 'go_idx': 'weak_go_idx.i32.npy',
                            'probability': 'weak_probability.f32.npy', 'source': 'weak_source.u8.npy'}
            np.save(stage/pseudo_files['indptr'], ptr)
            for key, dtype in [('go_idx', np.int32), ('probability', np.float32), ('source', np.uint8)]:
                raw_key = 'go' if key == 'go_idx' else key
                target = np.lib.format.open_memmap(stage/pseudo_files[key], mode='w+', dtype=dtype, shape=(int(ptr[-1]),))
                source = np.memmap(raw_paths[raw_key], mode='r', dtype=dtype) if ptr[-1] else np.empty(0, dtype)
                for start in range(0, len(target), 1_000_000):
                    target[start:start+1_000_000] = source[start:start+1_000_000]
                target.flush(); del target, source
                raw_paths[raw_key].unlink()
            active_weak_ids = role_ids['weak'][np.diff(ptr) > 0]
            if not len(active_weak_ids):
                raise ValueError('Threshold policy leaves no trainable weak proteins')
            edge = np.load(edge_path, mmap_mode='r', allow_pickle=False)
            attr = np.load(attr_path, mmap_mode='r', allow_pickle=False)
            if (edge.dtype.kind not in 'iu' or edge.ndim != 2 or edge.shape[0] != 2
                    or edge.shape[1] % degree or attr.shape != (edge.shape[1], 3)):
                raise ValueError('Invalid fixed-degree candidate arrays')
            source_count = edge.shape[1]//degree
            if source_start < 0 or source_start+source_count > len(roles):
                raise ValueError('Candidate source protein range outside registry')
            candidate_files = {'go': 'candidate_go.i32.npy', 'attr': 'candidate_attr.f32.npy',
                               'topk': 16, 'mode': args.candidate_mode, 'rows': len(roles)}
            cg = np.lib.format.open_memmap(stage/candidate_files['go'], mode='w+', dtype=np.int32, shape=(len(roles), 16))
            ca = np.lib.format.open_memmap(stage/candidate_files['attr'], mode='w+', dtype=np.float32, shape=(len(roles), 16, 3))
            cg[:] = -1; ca[:] = 0
            selected_counts = np.zeros(num_go, np.int64)
            for start in range(0, source_count, args.chunk_rows):
                stop = min(start+args.chunk_rows, source_count)
                global_ids = source_start + np.arange(start, stop)
                source_ids = np.asarray(edge[0, start*degree:stop*degree]).reshape(-1, degree)
                if not np.array_equal(source_ids, np.broadcast_to(global_ids[:, None], source_ids.shape)):
                    raise ValueError('Source candidates are not contiguous global protein-major rows')
                go = np.asarray(edge[1, start*degree:stop*degree]).reshape(-1, degree)
                attrs = np.asarray(attr[start*degree:stop*degree]).reshape(-1, degree, 3)
                new_go, new_attr = choose_candidate_rows(go, attrs, counts, seen, mode=args.candidate_mode)
                cg[global_ids] = new_go; ca[global_ids] = new_attr
                selected_counts += np.bincount(new_go[new_go >= 0], minlength=num_go)
                if start == 0 or stop == source_count or start//16384 != stop//16384:
                    print(f'[candidate selection] {stop}/{source_count} protein rows', flush=True)
            cg.flush(); ca.flush(); del cg, ca
            files = {name: _save_npy(stage, filename, values) for name, filename, values in [
                ('core_counts', 'core_counts.i64.npy', counts), ('seen_task_mask', 'seen_task_mask.bool.npy', seen),
                ('task_to_ontology', 'task_to_ontology.i64.npy', mapping),
                ('training_core_ids', 'training_core_ids.i64.npy', core_ids),
                ('validation_ids', 'validation_ids.i64.npy', validation_ids),
                ('training_weak_ids', 'training_weak_ids.i64.npy', active_weak_ids),
                ('heldout_ontology_rows', 'heldout_ontology_rows.i64.npy', heldout_rows)]}
            shutil.copyfile(config_path, stage/'source_config_snapshot.json')
            files['source_config_snapshot'] = 'source_config_snapshot.json'
            bins = []
            for i, (name, _, _) in enumerate(GROUPS):
                mask = groups == i
                bins.append({'frequency_bin': name, 'GO_columns': int(mask.sum()),
                             'seen_GO_columns': int(seen[mask].sum()),
                             'old_weak_positive_memberships': int(old_counts[mask].sum()),
                             'old_weak_covered_GO_columns': int((old_counts[mask] > 0).sum()),
                             'weak_positive_memberships': int(pseudo_counts[mask].sum()),
                             'weak_covered_GO_columns': int((pseudo_counts[mask] > 0).sum()),
                             'retained_candidate_pairs_all_roles': int(selected_counts[mask].sum())})
            output_names = list(files.values()) + list(pseudo_files.values()) + [candidate_files['go'], candidate_files['attr']]
            manifest = {'schema': SCHEMA, 'complete': True, 'task': config['task'],
                        'identity': identity, 'source_config': identity['source_config'], 'sources': identity['sources'],
                        'num_task_go': num_go, 'num_registry_proteins': len(roles),
                        'num_weak_registry_proteins': expected_shape[0], 'num_training_weak_proteins': len(active_weak_ids),
                        'num_training_core_proteins': len(core_ids), 'num_validation_proteins': len(validation_ids),
                        'files': files, 'core_counts_file': files['core_counts'],
                        'seen_task_mask_file': files['seen_task_mask'], 'pseudo': pseudo_files, 'candidates': candidate_files,
                        'hashes': {name: sha256(stage/name) for name in output_names}, 'policies': policies,
                        'training_ids_sha256': {'core': array_sha256(core_ids), 'validation': array_sha256(validation_ids),
                                                'weak': array_sha256(active_weak_ids)},
                        'num_seen_task_columns': int(seen.sum()), 'num_unseen_task_columns': int((~seen).sum()),
                        'restored_quantized_boundary_memberships': restored_boundary_memberships,
                        'GO_registry_contract': go_registry.contract(), 'frequency_strata': bins,
                        'probability_payload': 'unaltered selected teacher probability in float32; membership threshold applied before storage',
                        'teacher_source_codes': {'1': 'backbone', '2': 'source_stage1_modelout'},
                        'claim': 'Stage2 supervision isolation only; inherited Stage1 features/teacher/proposals may have seen held-out GO',
                        'zero_shot_claimed': False, 'elapsed_seconds': round(time.monotonic()-started, 3)}
            write_json(stage/'unified_dataset_manifest.json', manifest)
            if out.exists():
                out.rmdir()
            os.replace(stage, out)
            print('[dataset complete] ' + str(out/'unified_dataset_manifest.json'), flush=True)
            return manifest
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-work-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v092/geometry_single12800_scratch')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v093/unified_dataset')
    p.add_argument('--backbone-prob', type=Path)
    p.add_argument('--modelout-prob', type=Path)
    p.add_argument('--thresholds-json', type=Path)
    p.add_argument('--low-frequency-max', type=int, default=999)
    p.add_argument('--candidate-mode', choices=['frequency', 'global'], default='frequency')
    p.add_argument('--label-holdout-fraction', type=float, default=0.)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--chunk-rows', type=int, default=256)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.chunk_rows < 1 or args.low_frequency_max < 0 or args.seed < 0:
        raise SystemExit('chunk-rows must be positive; low-frequency-max and seed nonnegative')
    build_dataset(args)
