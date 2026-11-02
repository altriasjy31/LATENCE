#!/usr/bin/env python3
"""Read-only, NumPy-only audit of NBS core/weak membership and candidate coverage."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'nbs-weak-dataset-audit-r1'
BINS = [('zero', 0, 0), ('1-5', 1, 5), ('6-9', 6, 9),
        ('10-99', 10, 99), ('100-999', 100, 999), ('1000+', 1000, None)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(base, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.',
                                     delete=False, encoding='utf-8') as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_registry(path):
    roles, role_rows = [], []
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'protein_idx', 'role', 'role_row_idx'} <= set(reader.fieldnames or []):
            raise ValueError('Registry requires protein_idx, role, role_row_idx')
        for expected, row in enumerate(reader):
            if int(row['protein_idx']) != expected:
                raise ValueError('Registry protein_idx must be contiguous and row-aligned')
            roles.append(row['role'])
            role_rows.append(int(row['role_row_idx']))
    if not roles:
        raise ValueError('Empty protein registry')
    local = np.asarray(role_rows, np.int64)
    role_array = np.asarray(roles)
    ordered = {}
    for role in set(roles):
        ids = np.flatnonzero(role_array == role)
        ids = ids[np.argsort(local[ids], kind='stable')]
        if not np.array_equal(local[ids], np.arange(len(ids))):
            raise ValueError(f'Noncontiguous or repeated role_row_idx for {role}')
        ordered[role] = ids
    if 'core' not in ordered or 'weak' not in ordered:
        raise ValueError('Both core and weak roles are required')
    return role_array, local, ordered


def load_csr(ptr_path, go_path, rows, num_go):
    ptr = np.load(ptr_path, mmap_mode='r', allow_pickle=False)
    go = np.load(go_path, mmap_mode='r', allow_pickle=False)
    if (ptr.ndim != 1 or go.ndim != 1 or ptr.dtype.kind not in 'iu' or go.dtype.kind not in 'iu'
            or len(ptr) != rows + 1 or ptr[0] != 0 or ptr[-1] != len(go)
            or np.any(ptr[1:] < ptr[:-1])):
        raise ValueError(f'Invalid CSR dimensions/boundaries: {ptr_path}')
    for start in range(0, len(go), 1_000_000):
        value = go[start:start + 1_000_000]
        if len(value) and (value.min() < 0 or value.max() >= num_go):
            raise ValueError(f'GO outside immutable task columns: {go_path}')
    return ptr, go


def reconstruct_split(core_ids, gold_ptr, options):
    """Exactly match FullTaskData; core_ids are in role-local row order."""
    labeled = np.asarray(core_ids, np.int64)[np.diff(gold_ptr)[core_ids] > 0]
    holdout = options.get('holdout_core_count', 0)
    if type(holdout) is not int or not 0 <= holdout < len(labeled):
        raise ValueError('Invalid holdout_core_count')
    seed = int(options.get('holdout_seed', 8080))
    validation = np.sort(np.random.default_rng(seed).permutation(labeled)[:holdout])
    return np.setdiff1d(labeled, validation), validation


def csr_rows(ptr, go, rows):
    """Vectorized gather: returned row indices address the supplied rows list."""
    rows = np.asarray(rows, np.int64)
    lengths = np.asarray(ptr[rows + 1] - ptr[rows], np.int64)
    total = int(lengths.sum())
    if not total:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    row = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
    within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    positions = np.repeat(np.asarray(ptr[rows], np.int64), lengths) + within
    return row, np.asarray(go[positions], np.int64), positions


def frequency_groups(counts):
    groups = np.full(len(counts), -1, np.int8)
    for index, (_, lower, upper) in enumerate(BINS):
        groups[(counts >= lower) & (True if upper is None else counts <= upper)] = index
    if (groups < 0).any():
        raise ValueError('Negative gold frequency')
    return groups


def membership_counts(ptr, go, rows, num_go, *, groups=None, chunk_edges=1_000_000,
                      probability=None):
    """Count unique memberships; chunk size controls work, not CSR membership."""
    rows = np.asarray(rows, np.int64)
    counts = np.zeros(num_go, np.int64)
    row_bins = np.zeros((len(rows), len(BINS)), np.int32) if groups is not None else None
    degree = np.zeros(len(rows), np.int32)
    duplicate_edges = 0
    probability_info = {'min': None, 'max': None, 'stored_exactly_0_5': 0,
                        'below_0_5': 0, 'read_as_soft_targets': False} if probability is not None else None
    if probability is not None and (probability.shape != go.shape or probability.dtype.kind != 'f'):
        raise ValueError('Pseudo probabilities must be floating and aligned with CSR indices')
    start = 0
    while start < len(rows):
        # A single unusually large row is allowed and reported; never truncate it.
        stop = min(start + 8192, len(rows))
        lengths = np.asarray(ptr[rows[start:stop] + 1] - ptr[rows[start:stop]], np.int64)
        cumulative = np.cumsum(lengths)
        if len(cumulative) and cumulative[-1] > chunk_edges:
            stop = start + max(1, int(np.searchsorted(cumulative, chunk_edges, side='right')))
        local, labels, positions = csr_rows(ptr, go, rows[start:stop])
        if probability is not None and len(positions):
            p = np.asarray(probability[positions], np.float32)
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError('Pseudo probabilities must be finite in [0,1]')
            probability_info['min'] = min(probability_info['min'] if probability_info['min'] is not None else 1., float(p.min()))
            probability_info['max'] = max(probability_info['max'] if probability_info['max'] is not None else 0., float(p.max()))
            probability_info['stored_exactly_0_5'] += int((p == .5).sum())
            probability_info['below_0_5'] += int((p < .5).sum())
        key = np.unique(local * num_go + labels)
        duplicate_edges += len(labels) - len(key)
        label = key % num_go
        local = key // num_go
        counts += np.bincount(label, minlength=num_go)
        degree[start:stop] = np.bincount(local, minlength=stop-start)
        if groups is not None:
            cell = local * len(BINS) + groups[label]
            row_bins[start:stop] = np.bincount(cell, minlength=(stop-start)*len(BINS)).reshape(-1, len(BINS))
        start = stop
    return {'counts': counts, 'degree': degree, 'row_bins': row_bins,
            'duplicate_edges': duplicate_edges, 'probability': probability_info}


def quantiles(array):
    array = np.asarray(array)
    if not len(array):
        return {'rows': 0, 'mean': None, 'min': None, 'p25': None, 'median': None,
                'p75': None, 'p90': None, 'p99': None, 'max': None, 'zero_rows': 0}
    q = np.quantile(array, [0, .25, .5, .75, .9, .99, 1])
    return {'rows': len(array), 'mean': float(array.mean()),
            **dict(zip(['min', 'p25', 'median', 'p75', 'p90', 'p99', 'max'], map(float, q))),
            'zero_rows': int((array == 0).sum())}


def candidate_order(go, attrs, topk, num_go):
    """Exact v084/v088 ordering: reciprocal rank, probability, then GO ID."""
    valid = (go >= 0) & (go < num_go)
    rank = np.where(valid, attrs[..., 2], -np.inf)
    order = np.lexsort((go, -attrs[..., 0], -rank), axis=1)[:, :min(topk, go.shape[1])]
    chosen = np.take_along_axis(go, order, axis=1)
    return chosen


def scan_candidates(edge, attrs, degree, source_start, num_proteins, role_names, role_rows,
                    pseudo_ptr, pseudo_go, groups, topk, *, chunk_edges=1_000_000):
    """Full scan bounded by chunk_edges; all candidate input files stay mmap-backed."""
    num_go = len(groups)
    if (type(degree) is not int or degree <= 0 or edge.ndim != 2 or edge.shape[0] != 2
            or edge.dtype.kind not in 'iu' or edge.shape[1] % degree
            or attrs.shape != (edge.shape[1], 3) or attrs.dtype.kind != 'f'):
        raise ValueError('Candidates must be fixed-degree [2,E] edges with [E,3] attrs')
    num_sources = edge.shape[1] // degree
    if source_start < 0 or source_start + num_sources > num_proteins:
        raise ValueError('Candidate source range outside registry')
    if topk < 1:
        raise ValueError('candidate_topk must be positive')
    stats = {key: np.zeros(num_go, np.int64) for key in
             ('export_all', 'retained_all', 'export_weak', 'retained_weak',
              'export_active_weak', 'retained_active_weak', 'pseudo_in_export', 'pseudo_in_retained')}
    chunk_rows = max(1, min(8192, chunk_edges // degree))
    weak_sources = active_sources = 0
    for start in range(0, num_sources, chunk_rows):
        stop = min(start + chunk_rows, num_sources)
        global_ids = source_start + np.arange(start, stop)
        offset, end = start * degree, stop * degree
        source = np.asarray(edge[0, offset:end]).reshape(-1, degree)
        if not np.array_equal(source, np.broadcast_to(global_ids[:, None], source.shape)):
            raise ValueError('Candidate layout is not contiguous global protein-major fixed-degree; refusing an incorrect audit')
        go = np.asarray(edge[1, offset:end], np.int64).reshape(-1, degree)
        attr = np.asarray(attrs[offset:end], np.float32).reshape(-1, degree, 3)
        if (go < 0).any() or (go >= num_go).any() or not np.isfinite(attr).all():
            raise ValueError('Invalid candidate GO IDs or nonfinite attributes')
        # The exporter produces distinct GO candidates. Reject duplicates rather
        # than reporting edge counts as if they were unique protein-GO coverage.
        sorted_go = np.sort(go, axis=1)
        if (sorted_go[:, 1:] == sorted_go[:, :-1]).any():
            raise ValueError('Duplicate GO candidates within a protein row')
        retained = candidate_order(go, attr, topk, num_go)
        weak = role_names[global_ids] == 'weak'
        local_weak = role_rows[global_ids[weak]]
        active = np.zeros(len(global_ids), bool)
        active[weak] = pseudo_ptr[local_weak + 1] > pseudo_ptr[local_weak]
        weak_sources += int(weak.sum())
        active_sources += int(active.sum())
        for prefix, values in [('export', go), ('retained', retained)]:
            stats[prefix + '_all'] += np.bincount(values.ravel(), minlength=num_go)
            stats[prefix + '_weak'] += np.bincount(values[weak].ravel(), minlength=num_go)
            stats[prefix + '_active_weak'] += np.bincount(values[active].ravel(), minlength=num_go)
        if len(local_weak):
            row, labels, _ = csr_rows(pseudo_ptr, pseudo_go, local_weak)
            pseudo_key = np.unique(row * num_go + labels)
            for name, values in [('pseudo_in_export', go[weak]), ('pseudo_in_retained', retained[weak])]:
                keys = np.arange(len(values), dtype=np.int64)[:, None] * num_go + values
                found = np.isin(keys, pseudo_key, assume_unique=False)
                stats[name] += np.bincount(values[found], minlength=num_go)
        if start == 0 or stop == num_sources or start // 50000 != stop // 50000:
            print(f'[candidate audit] {stop}/{num_sources} protein rows', flush=True)
    return stats, {'source_protein_start': source_start, 'source_proteins': num_sources,
                   'weak_source_proteins': weak_sources, 'active_weak_source_proteins': active_sources,
                   'export_degree': degree, 'retained_topk': min(topk, degree),
                   'ordering': 'descending reciprocal_rank, descending backbone_probability, ascending GO column',
                   'output_GO_space_is_not_limited_by_candidate_topk': True}


def summary_check(path, core, counts, num_go):
    value = read_json(path)
    if value.get('num_GO_columns') != num_go:
        raise ValueError('Source summary GO width differs')
    if value.get('num_training_core_proteins') != len(core):
        raise ValueError('Source summary training core count differs')
    if value.get('training_core_ids_sha256') != array_hash(core):
        raise ValueError('Source summary training core IDs differ')
    if value.get('core_frequency_sha256') != array_hash(counts):
        raise ValueError('Source summary core frequencies differ')
    return {'path': str(path), 'sha256': sha256(path),
            'checks': ['num_GO_columns', 'num_training_core_proteins',
                       'training_core_ids_sha256', 'core_frequency_sha256'],
            'epoch_not_used_as_training_input': True}


def bind_prediction_contract(summary_path, file_identities):
    """Bind current training CSR bytes to an existing evaluation's data contract."""
    path = summary_path.parent.parent / 'nbs_full_task_prediction_manifest.json'
    if not path.is_file():
        return {'weak_source_identity_bound': False, 'path': str(path),
                'reason': 'Prediction manifest unavailable: only training IDs/frequencies were checked; weak-source identity is unbound'}
    summary, prediction = read_json(summary_path), read_json(path)
    digest = sha256(path)
    if digest != summary.get('prediction_manifest_sha256'):
        raise ValueError('Prediction manifest hash differs from source summary')
    contract = prediction.get('contract', {}).get('data', {})
    mapping = {'protein_registry': 'registry_sha256',
               'weak_manifest': 'weak_graph_predictions_manifest_sha256',
               'inverted_manifest': 'go_protein_inverted_index_manifest_sha256',
               'gold_indptr': 'gold_indptr_sha256', 'gold_go_idx': 'gold_go_idx_sha256',
               'pseudo_indptr': 'pseudo_indptr_sha256', 'pseudo_go_idx': 'pseudo_go_idx_sha256',
               'pseudo_probability': 'pseudo_probability_sha256'}
    for name, key in mapping.items():
        expected = contract.get(key)
        if not isinstance(expected, str) or file_identities[name]['sha256'] != expected:
            raise ValueError(f'Current training input differs from source prediction: {key}')
    return {'weak_source_identity_bound': True, 'path': str(path), 'sha256': digest,
            'checks': list(mapping.values()),
            'candidate_payload_bytes_not_bound_by_original_data_contract': True}


def _write_csv(path, names, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def audit(args, project_root=ROOT):
    started = time.monotonic()
    work = resolve(project_root, args.work_dir)
    config_path = work / 'resolved_config.json'
    config = read_json(config_path)
    ft, data = config['full_task'], config['data']
    if ft.get('label_split', 'all_seen') not in ('all_seen', None):
        raise ValueError('r1 audit requires all_seen label split; held-out GO need explicit filtering')
    root = resolve(project_root, data['root'])
    weak_path = resolve(root, data['weak_graph_predictions_manifest'])
    inverted_path = resolve(root, data['go_protein_inverted_index_manifest'])
    weak, inverted = read_json(weak_path), read_json(inverted_path)
    if weak.get('task') != config.get('task'):
        raise ValueError('Weak manifest task differs from config')
    num_go = int(weak['go_registry']['num_terms'])
    if num_go <= 0 or inverted.get('num_go', num_go) != num_go:
        raise ValueError('Inconsistent GO width')
    registry_path = root / 'features/protein_registry.csv'
    role_names, role_rows, role_ids = load_registry(registry_path)
    gold_spec = inverted['indices']['gold']['protein_major']
    if int(gold_spec['num_go']) != num_go:
        raise ValueError('Gold CSR GO width differs')
    gold_ptr_path = resolve(inverted_path.parent, gold_spec['indptr'])
    gold_go_path = resolve(inverted_path.parent, gold_spec['go_idx'])
    gold_ptr, gold_go = load_csr(gold_ptr_path, gold_go_path, len(role_names), num_go)
    core, validation = reconstruct_split(role_ids['core'], gold_ptr, ft)
    gold_stats = membership_counts(gold_ptr, gold_go, core, num_go, chunk_edges=args.chunk_edges)
    counts = gold_stats['counts']
    groups = frequency_groups(counts)
    spec = weak['weak_pseudo_targets']
    if spec.get('role') != 'weak':
        raise ValueError('Pseudo CSR must address role-local weak rows')
    pseudo_ptr_path = resolve(weak_path.parent, spec['csr_indptr_file'])
    pseudo_go_path = resolve(weak_path.parent, spec['csr_indices_file'])
    pseudo_prob_path = resolve(weak_path.parent, spec['csr_probability_file'])
    ptr, go = load_csr(pseudo_ptr_path, pseudo_go_path, len(role_ids['weak']), num_go)
    probability = np.load(pseudo_prob_path, mmap_mode='r', allow_pickle=False)
    ws = membership_counts(ptr, go, np.arange(len(role_ids['weak'])), num_go,
                           groups=groups, chunk_edges=args.chunk_edges, probability=probability)
    active = ws['degree'] > 0
    n_active, n_weak = int(active.sum()), len(active)
    sources = {'resolved_config': config_path, 'weak_manifest': weak_path, 'inverted_manifest': inverted_path,
               'protein_registry': registry_path, 'gold_indptr': gold_ptr_path, 'gold_go_idx': gold_go_path,
               'pseudo_indptr': pseudo_ptr_path, 'pseudo_go_idx': pseudo_go_path, 'pseudo_probability': pseudo_prob_path}
    summary_path = resolve(project_root, args.source_summary) if args.source_summary else work/'eval_step2880/full/metrics_v092/summary.json'
    if summary_path.exists():
        reference = summary_check(summary_path, core, counts, num_go)
    elif args.source_summary:
        raise FileNotFoundError(summary_path)
    else:
        reference = {'available': False, 'path': str(summary_path), 'reason': 'Optional reference summary absent; frequencies still reconstructed from training inputs'}
    candidate_stats = None
    candidate_info = {'available': False, 'reason': '--skip-candidates' if args.skip_candidates else 'Candidate declaration/files unavailable'}
    candidate_spec = weak.get('backbone_candidate_edges') or weak.get('backbone_rare_edges')
    if not args.skip_candidates and candidate_spec:
        edge_path = resolve(weak_path.parent, candidate_spec['edge_index_file'])
        attr_path = resolve(weak_path.parent, candidate_spec['edge_attr_file'])
        if edge_path.is_file() and attr_path.is_file():
            inv_candidate = inverted['indices'].get('candidate', {})
            degree = inv_candidate.get('fixed_degree')
            if degree is None:
                means = {int(r.get('candidate_degree_mean', r.get('rare_degree_mean', -1))) for r in weak.get('roles', [])}
                degree = means.pop() if len(means) == 1 else None
            source_start = inv_candidate.get('source_protein_start')
            if source_start is None:
                raise ValueError('Candidate source_protein_start missing; cannot safely infer mapping')
            candidate_stats, candidate_info = scan_candidates(
                np.load(edge_path, mmap_mode='r', allow_pickle=False),
                np.load(attr_path, mmap_mode='r', allow_pickle=False), degree, int(source_start),
                len(role_names), role_names, role_rows, ptr, go, groups,
                int(ft.get('sampler', {}).get('candidate_topk', 16)), chunk_edges=args.chunk_edges)
            candidate_info['available'] = True
            sources.update(candidate_edge_index=edge_path, candidate_edge_attr=attr_path)
        else:
            candidate_info.update(expected_edge_path=str(edge_path), expected_attr_path=str(attr_path))
    print('[audit] hashing read input files for provenance', flush=True)
    identity = {'schema': SCHEMA, 'task': config['task'],
                'files': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in sources.items()},
                'tool_sha256': sha256(__file__), 'training_core_ids_sha256': array_hash(core),
                'validation_ids_sha256': array_hash(validation), 'core_frequency_sha256': array_hash(counts),
                'skip_candidates': bool(args.skip_candidates), 'reference': reference}
    identity['source_prediction_binding'] = (bind_prediction_contract(summary_path, identity['files'])
        if summary_path.is_file() else {'weak_source_identity_bound': False,
            'reason': 'No source summary/manifest: current training inputs only; weak-source identity is unbound'})
    bins = []
    for index, (name, lower, upper) in enumerate(BINS):
        mask = groups == index
        row = {'frequency_bin': name, 'core_count_min': lower, 'core_count_max': upper,
               'GO_columns': int(mask.sum()), 'gold_positive_memberships': int(counts[mask].sum()),
               'weak_positive_memberships': int(ws['counts'][mask].sum()),
               'weak_covered_GO_columns': int((ws['counts'][mask] > 0).sum()),
               'weak_zero_positive_GO_columns': int((ws['counts'][mask] == 0).sum()),
               'potential_negative_pairs_before_alias_exclusion': int(n_active * int(mask.sum()) - ws['counts'][mask].sum()),
               'weak_rows_with_bin_positive': int((ws['row_bins'][:, index] > 0).sum()),
               'weak_labels_per_row_all': quantiles(ws['row_bins'][:, index]),
               'weak_labels_per_row_active': quantiles(ws['row_bins'][active, index]),
               'weak_prevalence_per_GO_active': quantiles(ws['counts'][mask] / n_active) if n_active else quantiles([])}
        if candidate_stats is not None:
            for key, value in candidate_stats.items():
                row['candidate_' + key + '_pairs'] = int(value[mask].sum())
            denom = row['candidate_export_weak_pairs']
            row['candidate_weak_pair_retention_fraction'] = row['candidate_retained_weak_pairs'] / denom if denom else None
            for key in ['export', 'retained']:
                denom = row['weak_positive_memberships']
                row['weak_pseudo_fraction_in_candidate_' + key] = row['candidate_pseudo_in_' + key + '_pairs'] / denom if denom else None
        bins.append(row)
    report = {'schema': SCHEMA, 'complete': True, 'identity': identity, 'task': config['task'],
              'num_GO_columns': num_go, 'num_registry_proteins': len(role_names),
              'num_training_core_proteins': len(core), 'num_holdout_core_proteins': len(validation),
              'num_weak_registry_proteins': n_weak, 'num_training_weak_proteins': n_active,
              'weak_empty_rows_excluded_from_training': n_weak - n_active,
              'weak_labels_per_row_all': quantiles(ws['degree']),
              'weak_labels_per_row_active': quantiles(ws['degree'][active]),
              'duplicate_gold_CSR_edges': gold_stats['duplicate_edges'],
              'duplicate_weak_CSR_edges': ws['duplicate_edges'], 'pseudo_probability': ws['probability'],
              'pseudo_label_contract': config.get('supervision_contract', {}).get('weak_pseudo'),
              'exported_pseudo_label_contract': spec,
              'teacher_semantics': weak.get('model_semantics', {}).get('modelout'),
              'candidate': candidate_info, 'frequency_strata': bins,
              'frequency_definition': 'unique gold memberships among reconstructed training core IDs; holdout excluded; mutually exclusive bins',
              'weak_target_definition': 'CSR membership, independent of stored probability; no rethresholding',
              'weak_GOs_without_positive_membership_usually_receive_negative_BCE': True,
              'weak_precision_or_recall_estimated': False,
              'dense_teacher_probabilities_read': False,
              'limitations': ['No weak gold labels are read: label coverage is not precision, recall or proof of missing biological functions.',
                  'No teacher thresholds are selected; dense backbone/expert/modelout signals and calibration quality remain unassessed.',
                  'Candidate topK limits graph evidence, not the all-GO prediction output space.',
                  'No role-population prevalence equivalence is assumed; core and weak proteins can differ biologically.',
                  'Potential negative pairs count all active-weak nonmembers before alias exclusions; these are not final loss-mask counts or established biological negatives.',
                  'No optimizer, model forward, training cache, original data or checkpoint is modified.',
                  'Raw task columns are retained; canonical GO aliases are not merged.'],
              'elapsed_seconds': round(time.monotonic() - started, 3)}
    out = resolve(project_root, args.output_dir) if args.output_dir else work/'weak_dataset_audit_r1'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'analysis.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Audit output is in use by another process') from None
        if (out/'report.json').exists() and read_json(out/'report.json').get('identity') != identity:
            raise ValueError('Existing output belongs to different data/code/options; use a new --output-dir')
        prevalence = ws['counts'] / n_active if n_active else np.zeros(num_go)
        go_rows = []
        for go_id in range(num_go):
            row = {'GO_column': go_id, 'frequency_bin': BINS[int(groups[go_id])][0],
                   'training_core_positive_count': int(counts[go_id]),
                   'weak_positive_count': int(ws['counts'][go_id]),
                   'potential_negative_pairs_before_alias_exclusion': int(n_active - ws['counts'][go_id]),
                   'weak_prevalence_among_training_weak': float(prevalence[go_id]) if n_active else None,
                   'weak_prevalence_among_all_weak': float(ws['counts'][go_id] / n_weak) if n_weak else None,
                   'zero_weak_positive': bool(ws['counts'][go_id] == 0)}
            if candidate_stats is not None:
                row.update({'candidate_' + key + '_pairs': int(value[go_id]) for key, value in candidate_stats.items()})
            go_rows.append(row)
        _write_csv(out/'go_coverage.csv', list(go_rows[0]), go_rows)
        flat_bins = [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in bins]
        _write_csv(out/'frequency_strata.csv', list(flat_bins[0]), flat_bins)
        atomic_json(out/'report.json', report)
    print(f'[audit complete] {out / "report.json"}', flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v092/geometry_single12800_scratch')
    p.add_argument('--source-summary', type=Path, help='Optional v092 summary used only to verify training IDs/frequencies')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--skip-candidates', action='store_true', help='Quick membership-only audit; do not scan candidate files')
    p.add_argument('--chunk-edges', type=int, default=1_000_000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.chunk_edges <= 0:
        raise SystemExit('--chunk-edges must be positive')
    audit(args)


if __name__ == '__main__':
    main()
