"""Backbone-threshold hard labels and independent modelout soft supervision.

Reuses the audited v093 CPU readers, GO aliases, and global top-16 selector.
Source files are read-only. Empty hard-label weak rows remain in the training
population because their dense teacher probabilities still carry supervision.
"""
from __future__ import annotations
import argparse
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
SCHEMA = 'nbs-weak-dataset-v094-r1'
CALIBRATION_SCHEMA = 'nbs-backbone-group-thresholds-v094-r1'
_spec = importlib.util.spec_from_file_location('_v094_audited_cpu_helpers', Path(__file__).with_name('unified_weak_dataset_v093.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
# Export shared pure-CPU helpers for calibration/runtime and preserve one
# implementation of immutable raw columns, role ordering, and GO resolution.
for _name in ('sha256', 'array_sha256', 'resolve', 'read_json', 'write_json',
              'file_identity', 'load_registry', 'read_gold', 'training_core_split',
              'count_gold', 'frequency_groups', 'make_seen_mask', 'threshold_policy',
              '_load_go_mapping', '_probability_path', '_validate_prob_block', '_save_npy'):
    globals()[_name] = getattr(_base, _name)
GROUPS = _base.GROUPS


def load_unified_manifest(path, verify=True):
    path = Path(path).resolve()
    result = read_json(path)
    if result.get('schema') != SCHEMA or result.get('complete') is not True:
        raise ValueError('Use a complete v094 backbone-hard/modelout-soft dataset')
    if verify:
        for name, expected in result['hashes'].items():
            if sha256(resolve(path.parent, name)) != expected:
                raise ValueError('v094 dataset file hash mismatch: ' + name)
    return result


def choose_candidate_rows(go, attrs, counts, seen, *, mode='global', topk=16):
    if mode != 'global':
        raise ValueError('v094 candidate selection is global; frequency quotas are removed')
    return _base.choose_candidate_rows(go, attrs, counts, seen, mode='global', topk=topk)


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
    if args.thresholds_json is None:
        raise ValueError('v094 requires --thresholds-json from all-frequency backbone calibration')
    thresholds, threshold_info = threshold_policy(args.thresholds_json)
    if args.thresholds_json is not None:
        calibration = read_json(args.thresholds_json)
        if calibration.get('schema') != CALIBRATION_SCHEMA:
            raise ValueError('v094 requires its own all-frequency backbone calibration schema')
        if calibration.get('schema') == CALIBRATION_SCHEMA:
            calibration_identity = calibration.get('identity', {})
            required_identity = {'training_core_ids_sha256': array_sha256(core_ids),
                                 'core_counts_sha256': array_sha256(counts),
                                 'seen_task_mask_content_sha256_int64': array_sha256(seen),
                                 'membership_teacher': 'backbone_all_frequencies'}
            if (calibration.get('complete') is not True
                    or any(calibration_identity.get(key) != value for key,value in required_identity.items())
                    or calibration_identity.get('sources', {}).get('weak_manifest', {}).get('sha256') != sha256(weak_path)):
                raise ValueError('Backbone calibration source/supervision split differs from the dataset being built')
            for name, identity in calibration_identity['sources'].items():
                if sha256(identity['path']) != identity['sha256']:
                    raise ValueError('Backbone calibration source changed after fitting: ' + name)
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
    policies = {'membership_teacher': 'backbone_all_frequencies',
                'confidence_teacher': 'source_stage1_modelout',
                'thresholds': thresholds, 'threshold_provenance': threshold_info, 'comparison': '>',
                'candidate_mode': 'global', 'candidate_topk': 16, 'candidate_quotas': None,
                'label_holdout_fraction': args.label_holdout_fraction, 'label_holdout_seed': args.seed,
                'zero_supervision_policy': 'exclude_whole_canonical_groups_without_training_core_positive',
                'weak_population': 'all_registry_weak_including_empty_hard_membership',
                'soft_supervision_scope': 'all_seen_GO',
                'probability_payload_source': 'modelout_at_backbone_selected_pairs',
                'quantization_policy': 'Compare stored backbone probabilities strictly against float64 thresholds; no old modelout membership restoration.',
                'strict_pipeline_zero_shot': False}
    identity = {'source_config': file_identity(config_path),
                'sources': {name: file_identity(path) for name, path in sources.items()},
                'policies': policies, 'builder_sha256': sha256(__file__),
                'helper_implementation_sha256': sha256(Path(__file__).with_name('unified_weak_dataset_v093.py')),
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
            hard_positive_modelout_low = np.zeros(num_go, np.int64)
            hard_negative_modelout_high = np.zeros(num_go, np.int64)
            positive_modelout_sum = np.zeros(num_go, np.float64)
            source_vector = np.full(num_go, 2, np.uint8)
            raw_paths = {key: stage/(key+'.raw') for key in ['go', 'probability', 'source']}
            with raw_paths['go'].open('wb') as gi, raw_paths['probability'].open('wb') as pr, raw_paths['source'].open('wb') as sr:
                for start in range(0, expected_shape[0], args.chunk_rows):
                    stop = min(start+args.chunk_rows, expected_shape[0])
                    b = np.asarray(backbone[start:stop], np.float32)
                    m = np.asarray(modelout[start:stop], np.float32)
                    _validate_prob_block(b, (stop-start, num_go)); _validate_prob_block(m, b.shape)
                    positive = (b.astype(np.float64) > threshold_vector[None]) & seen[None]
                    hard_positive_modelout_low += (positive & (m < .5)).sum(0, dtype=np.int64)
                    hard_negative_modelout_high += ((~positive) & seen[None] & (m > .5)).sum(0, dtype=np.int64)
                    positive_modelout_sum += np.where(positive, m, 0.).sum(0, dtype=np.float64)
                    old_start, old_stop = int(old_ptr[start]), int(old_ptr[stop])
                    old_labels = np.asarray(old_go[old_start:old_stop], np.int64)
                    old_p = np.asarray(old_probability[old_start:old_stop], np.float32)
                    if len(old_labels) and ((old_labels < 0).any() or (old_labels >= num_go).any()
                            or not np.isfinite(old_p).all() or (old_p < 0).any() or (old_p > 1).any()):
                        raise ValueError('Invalid original weak GO/probability payload')
                    old_counts += np.bincount(old_labels, minlength=num_go)
                    row, label = np.nonzero(positive)
                    label.astype(np.int32).tofile(gi)
                    m[row, label].astype(np.float32).tofile(pr)
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
            # Empty hard-label rows still carry dense modelout soft supervision.
            active_weak_ids = role_ids['weak']
            edge = np.load(edge_path, mmap_mode='r', allow_pickle=False)
            attr = np.load(attr_path, mmap_mode='r', allow_pickle=False)
            if (edge.dtype.kind not in 'iu' or edge.ndim != 2 or edge.shape[0] != 2
                    or edge.shape[1] % degree or attr.shape != (edge.shape[1], 3)):
                raise ValueError('Invalid fixed-degree candidate arrays')
            source_count = edge.shape[1]//degree
            if source_start < 0 or source_start+source_count > len(roles):
                raise ValueError('Candidate source protein range outside registry')
            candidate_files = {'go': 'candidate_go.i32.npy', 'attr': 'candidate_attr.f32.npy',
                               'topk': 16, 'mode': 'global', 'rows': len(roles)}
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
                new_go, new_attr = choose_candidate_rows(go, attrs, counts, seen, mode='global')
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
                             'backbone_hard_positive_modelout_below_0_5_pairs': int(hard_positive_modelout_low[mask].sum()),
                             'backbone_hard_negative_modelout_above_0_5_pairs': int(hard_negative_modelout_high[mask].sum()),
                             'mean_modelout_on_backbone_hard_positives': (float(positive_modelout_sum[mask].sum()/pseudo_counts[mask].sum())
                                                                        if pseudo_counts[mask].sum() else None),
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
                        'empty_hard_membership_weak_proteins': int((np.diff(ptr) == 0).sum()),
                        'soft_teacher': {'role': 'weak', 'path': str(modelout_path),
                                         'sha256': identity['sources']['modelout_weak']['sha256'],
                                         'shape': list(expected_shape), 'source': 'modelout',
                                         'row_order': 'registry_role_local', 'scope': 'all_seen_GO'},
                        'teacher_disagreement': {
                            'scope': 'all weak proteins and seen raw task columns; strict modelout <0.5 or >0.5 only for diagnostics',
                            'backbone_hard_positive_modelout_below_0_5_pairs': int(hard_positive_modelout_low.sum()),
                            'backbone_hard_negative_modelout_above_0_5_pairs': int(hard_negative_modelout_high.sum()),
                            'mean_modelout_on_backbone_hard_positives': (float(positive_modelout_sum.sum()/pseudo_counts.sum())
                                                                       if pseudo_counts.sum() else None)},
                        'GO_registry_contract': go_registry.contract(), 'frequency_strata': bins,
                        'probability_payload': 'unaltered modelout probability in float32 at backbone-threshold selected pairs; no modelout rethresholding',
                        'teacher_source_codes': {'2': 'source_stage1_modelout_probability_only'},
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
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v094/backbone_hard_modelout_soft')
    p.add_argument('--backbone-prob', type=Path)
    p.add_argument('--modelout-prob', type=Path)
    p.add_argument('--thresholds-json', type=Path, required=True)
    p.add_argument('--label-holdout-fraction', type=float, default=0.)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--chunk-rows', type=int, default=256)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.chunk_rows < 1 or args.seed < 0:
        raise SystemExit('chunk-rows must be positive and seed nonnegative')
    build_dataset(args)


if __name__ == '__main__':
    main()
