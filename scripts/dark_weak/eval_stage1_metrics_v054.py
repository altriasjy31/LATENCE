#!/usr/bin/env python3
"""Comprehensive, read-only evaluation of frozen Stage-1 / NBS probabilities.

Both Dark–Weak complete manifests and explicit ind_test exports are supported.
No inference, score fusion, ontology propagation or training takes place.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
for directory in (PROJECT, PROJECT / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import eval_dark_weak_predictions as ranking
from dark_weak import eval_stage1_frequency_v053 as common
from dark_weak.complete_metric_pack_v054 import (
    comprehensive_metrics, load_kernels, metric_protocol,
)

SCHEMA = 'latence-comprehensive-frozen-metrics-v054'
KEYS = ['split', 'family', 'bin', 'model']
DEFAULT_FAMILIES = ['stage1_raw_classic', 'historical_classic', 'historical_fixed']


def read_labels(metadata, split, namespace, width, metric_field, rank_field):
    group = metadata[split][ranking.NS[namespace]]
    names = list(map(str, group['proteins']))
    if not names or len(names) != len(set(names)):
        raise ValueError(f'{split}: empty cohort or duplicate protein identifiers')
    if rank_field == 'auto':
        rank_field = 'annotations' if 'annotations' in group else 'prop_annotations'
    for field in (metric_field, rank_field):
        if field not in group or len(group[field]) != len(names):
            raise ValueError(f'{split}: missing/misaligned label field {field}')
    y = np.zeros((len(names), width), dtype=np.uint8)
    ranks = []
    for i, name in enumerate(names):
        cols = common.np_unique_idx(group[metric_field][i], width, f'{split}/{name}/{metric_field}')
        y[i, cols] = 1
        ranks.append(common.np_unique_idx(group[rank_field][i], width, f'{split}/{name}/{rank_field}'))
    return names, y, ranks, rank_field


def load_explicit(args, gos):
    predictions = ranking.parse_kv(args.prediction, '--prediction')
    ids = ranking.parse_kv(args.prediction_ids, '--prediction-ids')
    goids = ranking.parse_kv(args.prediction_go_ids, '--prediction-go-ids')
    if not predictions or set(predictions) != set(ids) or set(predictions) != set(goids):
        raise ValueError('Each model requires --prediction, --prediction-ids and --prediction-go-ids')
    result, provenance = {}, {}
    for model, path in predictions.items():
        pp, ip, gp = (Path(x).resolve(strict=True) for x in (path, ids[model], goids[model]))
        if ranking.read_go_list(gp) != gos:
            raise ValueError(f'{model}: GO order differs from frozen unified vocabulary')
        array = ranking.load_prediction(pp, len(gos))
        order = ranking.load_protein_order(ip, array.shape[0])
        result[model] = (array, order, pp, ip, gp)
        provenance[model] = {'probability': str(pp), 'probability_sha256': common.sha256(pp),
                             'protein_ids': str(ip), 'protein_ids_sha256': common.sha256(ip),
                             'go_ids': str(gp), 'go_ids_sha256': common.sha256(gp)}
    return result, provenance


def make_families(args, raw, counts, inverse, canon, gos):
    raw_bins, raw_meta = common.classic_bins(raw)
    raw_meta['semantics'] = 'Original Stage-1 quantile rules on unmerged historical output-column counts'
    canon_bins, canon_meta = common.classic_bins(counts)
    # Each family supplies raw classifier-column masks. Only canonical families
    # supply canonical rank-positive masks: an alias can cross raw-bin borders.
    catalog = {
        'overall': {'columns': {'all': np.ones(len(gos), bool)},
                    'rank_masks': {'all': np.ones(len(canon), bool)},
                    'counts': counts, 'axis': 'all_original_classifier_columns', 'metadata': {}},
        'stage1_raw_classic': {'columns': raw_bins, 'rank_masks': None,
                    'counts': raw, 'axis': 'raw_output_column_frequency', 'metadata': raw_meta},
        'historical_classic': {'columns': {b: m[inverse] for b, m in canon_bins.items()},
                    'rank_masks': canon_bins, 'counts': counts,
                    'axis': 'canonical_unique_training_protein_frequency', 'metadata': canon_meta},
        'historical_fixed': {'columns': {b: m[inverse] for b, m in common.fixed_bins(counts).items()},
                    'rank_masks': common.fixed_bins(counts), 'counts': counts,
                    'axis': 'canonical_unique_training_protein_frequency',
                    'metadata': {'intervals': common.FIXED_GROUPS}},
    }
    native_info = None
    if args.v094_unified_manifest:
        c, _, native_info = common.load_v094_core_frequency(
            args.v094_unified_manifest, gos, inverse, len(canon))
        bins = common.fixed_bins(c)
        catalog['v094_native_core_fixed'] = {
            'columns': {b: m[inverse] for b, m in bins.items()}, 'rank_masks': bins,
            'counts': c, 'axis': 'v094_native_alias_max_core_frequency',
            'metadata': {'intervals': common.FIXED_GROUPS}}
    missing = set(args.families) - set(catalog)
    if missing:
        raise ValueError(f'Unavailable frequency families {sorted(missing)}; v094 needs its unified manifest')
    return {name: catalog[name] for name in ['overall', *args.families]}, native_info


def positive_records(p, labels, names, canon, inverse, alias_sizes, root_id,
                     families, model, split, sequence_hash, ks):
    allowed = np.flatnonzero(np.array(canon) != root_id)
    records = []
    for row, (protein, raw_cols) in enumerate(zip(names, labels)):
        positive = np.unique(inverse[raw_cols])
        positive = positive[np.array(canon)[positive] != root_id]
        if not len(positive):
            continue
        combined = np.bincount(inverse, weights=p[row].astype(np.float64),
                               minlength=len(canon)) / alias_sizes
        scores = combined[allowed]
        local = np.searchsorted(allowed, positive)
        stable = ranking.positive_ranks(scores, local)
        sorted_scores = np.sort(scores)
        optimistic = 1 + len(scores) - np.searchsorted(sorted_scores, scores[local], side='right')
        pessimistic = len(scores) - np.searchsorted(sorted_scores, scores[local], side='left')
        for col, rank, best, worst in zip(positive, stable, optimistic, pessimistic):
            for family, spec in families.items():
                if spec['rank_masks'] is None:
                    continue
                for bin_name, mask in spec['rank_masks'].items():
                    if not mask[col]:
                        continue
                    record = {'split': split, 'family': family, 'bin': bin_name, 'model': model,
                              'protein': protein, 'sequence_hash': sequence_hash.get(protein, protein),
                              'go_id': canon[col], 'train_frequency': int(spec['counts'][col]),
                              'rank': int(rank), 'rank_optimistic': int(best),
                              'rank_pessimistic': int(worst), 'tie_size': int(worst - best + 1),
                              'score': float(combined[col]), 'reciprocal_rank': 1. / rank,
                              'rank_vocabulary_size': len(allowed)}
                    for k in ks:
                        record[f'hit_at_{k}'] = int(rank <= k)
                        record[f'hit_at_{k}_lower'] = int(worst <= k)
                        record[f'hit_at_{k}_upper'] = int(best <= k)
                    records.append(record)
    return records


def rank_summary(frame, ks):
    if frame.empty:
        return {'n_rank_positive_pairs': 0, 'n_rank_positive_proteins': 0, 'n_rank_positive_GO': 0,
                'rank_status': 'no_rank_positives'}
    result = common.rank_summary(frame, ks)
    for old, new in [('n_direct_pairs', 'n_rank_positive_pairs'),
                     ('n_direct_proteins', 'n_rank_positive_proteins'), ('n_direct_GO', 'n_rank_positive_GO')]:
        result[new] = result.pop(old)
    result.update(rank_status='ok', positive_tie_fraction=float((frame.tie_size > 1).mean()),
                  mean_tie_size=float(frame.tie_size.mean()),
                  pairwise_mrr_lower=float((1. / frame.rank_pessimistic).mean()),
                  pairwise_mrr_upper=float((1. / frame.rank_optimistic).mean()))
    for k in ks:
        result[f'micro_recall_at_{k}_lower'] = float(frame[f'hit_at_{k}_lower'].mean())
        result[f'micro_recall_at_{k}_upper'] = float(frame[f'hit_at_{k}_upper'].mean())
    return result


def run(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    unified = Path(args.unified_root).resolve(strict=True)
    original = None
    if args.complete_manifest:
        if args.metadata or args.namespace or args.prediction or args.prediction_ids or args.prediction_go_ids:
            raise ValueError('Use either --complete-manifest or explicit metadata/namespace/predictions')
        original, metadata_path, models = common.load_reference_manifest(args.complete_manifest, unified, True)
        namespace = original['namespace']
        cases = args.splits or original['cases']
        provenance = original['input_files']
    else:
        if not args.metadata or not args.namespace or not args.splits:
            raise ValueError('Explicit mode needs --metadata, --namespace, --splits and prediction files')
        namespace, cases = args.namespace, args.splits
        metadata_path = Path(args.metadata).resolve(strict=True)
        models = None
    gos, canon, inverse, sizes = common.read_vocabulary(unified, namespace)
    if models is None:
        models, provenance = load_explicit(args, gos)
    # Earlier complete manifests do not hash ID files. Freeze their current
    # bytes now, without implying retrospective verification against that run.
    provenance = {model: dict(info) for model, info in provenance.items()}
    for model, (_array, _order, _pp, ip, gp) in models.items():
        provenance[model]['protein_ids_sha256'] = common.sha256(ip)
        provenance[model]['go_ids_sha256'] = common.sha256(gp)
        provenance[model]['id_hash_status'] = ('frozen_current_bytes; original manifest did not bind ID-file hashes'
            if original is not None else 'frozen_current_explicit_inputs')
    if args.reference not in models:
        raise ValueError(f'Reference model {args.reference!r} has no prediction input')
    branches = ranking.parse_kv(args.model_branch, '--model-branch')
    if set(branches) - set(models):
        raise ValueError('A --model-branch entry refers to a missing model')
    train_path = Path(args.train_metadata).resolve(strict=True)
    raw, counts, ntrain = common.get_train_counts(train_path, namespace, args.train_split,
                                args.train_label_field, len(gos), inverse, len(canon))
    families, v094_info = make_families(args, raw, counts, inverse, canon, gos)
    ic = None
    if args.ic_vector:
        ic = np.load(args.ic_vector, allow_pickle=False)
        if ic.shape != (len(gos),) or not np.isfinite(ic).all() or (ic < 0).any():
            raise ValueError('--ic-vector must be a finite, nonnegative vector in original GO column order')
        if args.skip_stage1_official:
            raise ValueError('--ic-vector requires the Stage-1 official helper')
    kernels = load_kernels(PROJECT, official=not args.skip_stage1_official,
                           official_source=args.stage1_helper_source)
    cohort_path = unified / 'tables/full_dark_weak_cohort.tsv'
    sequence_hash = {}
    if cohort_path.is_file():
        cohort = pd.read_csv(cohort_path, sep='\t', usecols=['protein_22', 'sequence_sha256']).dropna()
        sequence_hash = dict(zip(cohort.protein_22.astype(str), cohort.sequence_sha256.astype(str)))
    with metadata_path.open('rb') as f:
        metadata = pickle.load(f)  # Only trusted local LATENCE pickle files.
    rows, pair_rows, group_rows, label_policy = [], [], [], {}
    sequence_hash_coverage = {}
    for family, spec in families.items():
        for name, colmask in spec['columns'].items():
            freq = raw if family == 'stage1_raw_classic' else spec['counts'][inverse]
            group_rows.append({'family': family, 'bin': name, 'grouping_axis': spec['axis'],
                               'n_model_columns': int(colmask.sum()),
                               'n_unique_canonical_GO': int(np.unique(inverse[colmask]).size),
                               'training_freq_min': int(freq[colmask].min()) if colmask.any() else None,
                               'training_freq_max': int(freq[colmask].max()) if colmask.any() else None})
    for split in cases:
        print(f'[v054] {namespace}/{split}: comprehensive frozen-score metrics', flush=True)
        names, y, ranking_labels, rank_field = read_labels(metadata, split, namespace, len(gos),
                                                         args.metric_label_field, args.ranking_label_field)
        matched_hashes = sum(name in sequence_hash for name in names)
        sequence_hash_coverage[split] = {'evaluated_proteins': len(names),
            'with_sequence_hash': matched_hashes, 'protein_id_fallback': len(names) - matched_hashes}
        label_policy[split] = {'metric_label_field': args.metric_label_field,
                               'ranking_label_field': rank_field,
                               'ranking_target': 'supplied annotations; future-direct only when metadata is a future-direct cohort'}
        for model, (array, order, *_paths) in models.items():
            missing = [name for name in names if name not in order]
            if missing:
                raise ValueError(f'{model}/{split}: missing prediction IDs {missing[:3]}')
            p = np.asarray(array[[order[name] for name in names]], dtype=np.float32)
            if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError(f'{model}/{split}: invalid probability matrix')
            pair_rows.extend(positive_records(p, ranking_labels, names, canon, inverse, sizes,
                             ranking.ROOT_IDS[namespace], families, model, split, sequence_hash, args.ks))
            for family, spec in families.items():
                for bin_name, mask in spec['columns'].items():
                    yy, pp = y[:, mask], p[:, mask]
                    if yy.size > args.max_metric_cells and not args.allow_large_metrics:
                        raise MemoryError(f'{split}/{family}/{bin_name}: {yy.size:,} cells; review RAM then use --allow-large-metrics')
                    entry = {'split': split, 'family': family, 'bin': bin_name, 'model': model,
                             'grouping_axis': spec['axis'], 'scoring_axis': 'original_classifier_columns',
                             'model_branch': branches.get(model, 'not_supplied'),
                             'source_probability_dtype': str(array.dtype),
                             'num_rows': len(names), 'n_model_columns': int(mask.sum()),
                             'n_metric_positive_columns': int(yy.sum()),
                             'n_metric_positive_proteins': int(np.count_nonzero(yy.sum(axis=1))),
                             'metric_label_field': args.metric_label_field,
                             'status': 'empty_group' if not mask.any() else ('no_positives' if not yy.any() else 'ok')}
                    entry.update(comprehensive_metrics(yy, pp, native=kernels['native'],
                                 legacy=kernels['legacy'], official=kernels['official'],
                                 threshold_step=args.stage1_threshold_step,
                                 compute_sample_fmax=True, verify_fmax=True,
                                 ic=ic[mask] if ic is not None else None))
                    rows.append(entry)
    summary = pd.DataFrame(rows)
    pair_columns = KEYS + ['protein', 'sequence_hash', 'go_id', 'train_frequency', 'rank',
        'rank_optimistic', 'rank_pessimistic', 'tie_size', 'score', 'reciprocal_rank', 'rank_vocabulary_size']
    for k in args.ks:
        pair_columns.extend([f'hit_at_{k}', f'hit_at_{k}_lower', f'hit_at_{k}_upper'])
    pairs = pd.DataFrame(pair_rows, columns=pair_columns)
    ranked = []
    if not pairs.empty:
        for key, frame in pairs.groupby(KEYS, sort=False):
            ranked.append(dict(zip(KEYS, key), **rank_summary(frame, args.ks)))
    if ranked:
        summary = summary.merge(pd.DataFrame(ranked), on=KEYS, how='left')
    if 'rank_status' not in summary:
        summary['rank_status'] = 'no_rank_positives'
    else:
        summary['rank_status'] = summary['rank_status'].fillna('no_rank_positives')
    for count_field in ('n_rank_positive_pairs', 'n_rank_positive_proteins', 'n_rank_positive_GO'):
        if count_field not in summary:
            summary[count_field] = np.nan
        canonical_rows = summary.family != 'stage1_raw_classic'
        summary.loc[canonical_rows, count_field] = summary.loc[canonical_rows, count_field].fillna(0)
    summary.loc[summary.family == 'stage1_raw_classic', 'rank_status'] = 'not_defined_for_alias_crossing_raw_bins'
    # Paired comparisons preserve the same cohort/columns and metric definitions.
    metric_names = [c for c in summary if c.startswith(('stage1_hist__', 'stage1_official__', 'standard_'))
                    and ('fmax' in c or 'auprc' in c or c.endswith(('micro_ap', 'micro_pr_auc', '__smin')))
                    and not any(s in c for s in ('threshold', 'eligible', 'step', 'min_included_score'))]
    metric_names += [c for c in summary if c.startswith(('micro_recall_at_', 'pairwise_mrr'))]
    deltas = []
    for key, frame in summary.groupby(['split', 'family', 'bin'], sort=False):
        baseline = frame[frame.model == args.reference].iloc[0]
        for _, row in frame[frame.model != args.reference].iterrows():
            for metric in metric_names:
                a, b = row[metric], baseline[metric]
                value = float(a - b) if pd.notna(a) and pd.notna(b) else None
                deltas.append(dict(zip(['split', 'family', 'bin'], key), model=row['model'],
                              reference=args.reference, metric=metric, delta=value,
                              delta_unit='fraction' if metric.startswith(('micro_recall', 'pairwise_mrr'))
                              else ('IC_units' if metric.endswith('__smin') else 'percentage_points')))
    boot = []
    if not pairs.empty:
        for key, frame in pairs.groupby(['split', 'family', 'bin'], sort=False):
            baseline = frame[frame.model == args.reference]
            for model in models:
                if model == args.reference:
                    continue
                result = common.boot_pair_delta(frame[frame.model == model], baseline,
                                                args.ks, args.bootstrap, args.seed)
                for metric, value in result.items():
                    boot.append(dict(zip(['split', 'family', 'bin'], key), model=model,
                                     reference=args.reference, metric=metric, **value))
    manifest = {'schema': SCHEMA, 'complete': True, 'namespace': namespace,
                'complete_manifest': str(Path(args.complete_manifest).resolve()) if args.complete_manifest else None,
                'complete_manifest_sha256': common.sha256(args.complete_manifest) if args.complete_manifest else None,
                'metadata': str(metadata_path), 'metadata_sha256': common.sha256(metadata_path),
                'train_metadata': str(train_path), 'train_metadata_sha256': common.sha256(train_path),
                'train_split': args.train_split, 'train_label_field': args.train_label_field,
                'training_proteins': ntrain, 'num_model_columns': len(gos), 'num_canonical_GO': len(canon),
                'vocabulary_sha256': common.sha256(unified / 'labels' / namespace / 'go_ids.txt'),
                'vocabulary_mapping_sha256': common.sha256(unified / 'labels' / namespace / 'vocabulary_mapping.tsv'),
                'input_files': provenance, 'model_branches': branches,
                'original_export_provenance': {key: original[key] for key in
                    ('stage1_branch', 'modelout_branch', 'native_nbs_provenance')
                    if original is not None and key in original},
                'software_versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                    'torch': sys.modules['torch'].__version__ if 'torch' in sys.modules else None,
                    'sklearn': sys.modules['sklearn'].__version__ if 'sklearn' in sys.modules else None},
                'families': {f: {'grouping_axis': s['axis'], **s['metadata']} for f, s in families.items()},
                'metric_protocol': metric_protocol(kernels, threshold_step=args.stage1_threshold_step,
                                                   compute_sample_fmax=True),
                'ic_vector': {'path': str(Path(args.ic_vector).resolve()), 'sha256': common.sha256(args.ic_vector),
                              'provenance': args.ic_provenance} if args.ic_vector else None,
                'label_policy': label_policy, 'cases': cases, 'ks': args.ks,
                'bootstrap_repetitions': args.bootstrap, 'bootstrap_seed': args.seed,
                'ranking_protocol': {'axis': 'all canonical GO, root excluded', 'alias_score': 'mean',
                    'ties': 'stable original canonical column order; optimistic/pessimistic bounds also reported',
                    'stratification': 'filter positive GO after full-vocabulary ranking',
                    'raw_family': 'threshold/AP only; ranks intentionally undefined because aliases may cross raw bins',
                    'bounds': 'tie-position bounds only, not statistical confidence intervals',
                    'bootstrap_unit': 'exact sequence SHA256 when available; otherwise protein ID',
                    'sequence_hash_coverage_by_split': sequence_hash_coverage},
                'interpretation': ['Unobserved GO entries are treated as negatives for proxy Fmax/AP.',
                    'All threshold optima use the evaluated labels and are descriptive oracle maxima.',
                    'No labels/predictions are propagated here; supplied labels must already have the intended closure.',
                    'Raw group parity requires the same historical metadata field, vocabulary, branch and filtering as the old run.',
                    'Canonical frequency families retain original classifier columns in threshold/AP metrics.',
                    'Original source storage dtype is recorded; fp32 casting cannot restore precision lost in fp16.',
                    'Original official helper runs on CPU; historical GPU/version tie ordering can change its grid-trapezoid AuPRC.',
                    'zero_train is not a guarantee of model-wide zero-shot.',
                    'Bootstrap intervals apply to stable-rank deltas only; evidence cohorts may overlap.'],
                'original_reports_modified': False, 'model_inference_performed': False}
    out.mkdir(parents=True, exist_ok=False)
    summary.to_csv(out / 'comprehensive_metrics.tsv', sep='\t', index=False)
    summary[summary.family == 'overall'].to_csv(out / 'overall_metrics.tsv', sep='\t', index=False)
    pd.DataFrame(group_rows).to_csv(out / 'frequency_groups.tsv', sep='\t', index=False)
    pairs.to_csv(out / 'positive_global_ranks.tsv', sep='\t', index=False)
    pd.DataFrame(deltas, columns=['split', 'family', 'bin', 'model', 'reference', 'metric', 'delta', 'delta_unit']).to_csv(
        out / 'metric_deltas_vs_reference.tsv', sep='\t', index=False)
    pd.DataFrame(boot, columns=['split', 'family', 'bin', 'model', 'reference', 'metric', 'delta', 'ci95',
                              'bootstrap_unit', 'n_clusters', 'repetitions']).to_csv(
        out / 'paired_rank_bootstrap_deltas.tsv', sep='\t', index=False)
    lookup = pd.DataFrame({'model_column': np.arange(len(gos)), 'model_go_id': gos,
                           'canonical_go_id': [canon[i] for i in inverse],
                           'historical_raw_count': raw, 'historical_canonical_count': counts[inverse]})
    for family, spec in families.items():
        for name, mask in spec['columns'].items():
            lookup[f'{family}__{name}'] = mask
    lookup.to_csv(out / 'go_frequency_lookup.tsv', sep='\t', index=False)
    (out / 'metrics_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                                          allow_nan=False) + '\n', encoding='utf-8')
    print(f'[v054] Completed: {out}', flush=True)
    return summary, manifest


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--complete-manifest', type=Path)
    ap.add_argument('--metadata', type=Path)
    ap.add_argument('--namespace', choices=tuple(ranking.NS))
    ap.add_argument('--prediction', action='append', default=[], metavar='MODEL=PATH')
    ap.add_argument('--prediction-ids', action='append', default=[], metavar='MODEL=PATH')
    ap.add_argument('--prediction-go-ids', action='append', default=[], metavar='MODEL=PATH')
    ap.add_argument('--model-branch', action='append', default=[], metavar='MODEL=BRANCH')
    ap.add_argument('--unified-root', type=Path, default=Path('data/dark_weak_test/unified'))
    ap.add_argument('--train-metadata', type=Path, default=Path('data/unidata_with_exp_train_pseudo.pkl'))
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--train-label-field', choices=('annotations', 'prop_annotations'), default='prop_annotations')
    ap.add_argument('--metric-label-field', choices=('annotations', 'prop_annotations'), default='prop_annotations')
    ap.add_argument('--ranking-label-field', choices=('auto', 'annotations', 'prop_annotations'), default='auto')
    ap.add_argument('--v094-unified-manifest', type=Path)
    ap.add_argument('--families', nargs='+', default=DEFAULT_FAMILIES,
                    choices=tuple(DEFAULT_FAMILIES) + ('v094_native_core_fixed',))
    ap.add_argument('--splits', nargs='+')
    ap.add_argument('--reference', default='backbone')
    ap.add_argument('--ks', nargs='+', type=int, default=[10, 50, 100, 512, 1024])
    ap.add_argument('--bootstrap', type=int, default=500)
    ap.add_argument('--seed', type=int, default=6061)
    ap.add_argument('--stage1-threshold-step', type=float, default=.01)
    ap.add_argument('--skip-stage1-official', action='store_true', help='Omit original torch helper metrics')
    ap.add_argument('--stage1-helper-source', type=Path)
    ap.add_argument('--ic-vector', type=Path, help='Optional nonnegative historical-training IC .npy in original GO order')
    ap.add_argument('--ic-provenance', help='Required with --ic-vector: historical source/formula; never use future test labels')
    ap.add_argument('--max-metric-cells', type=int, default=80_000_000)
    ap.add_argument('--allow-large-metrics', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.bootstrap < 0 or args.max_metric_cells < 1:
        ap.error('--bootstrap must be >=0; --max-metric-cells must be positive')
    if not args.ks or any(k < 1 for k in args.ks) or len(set(args.ks)) != len(args.ks):
        ap.error('--ks must be distinct positive integers')
    if len(args.families) != len(set(args.families)):
        ap.error('--families must be distinct')
    if not np.isfinite(args.stage1_threshold_step) or not 0 < args.stage1_threshold_step <= 1:
        ap.error('--stage1-threshold-step must divide one exactly')
    intervals = round(1 / args.stage1_threshold_step)
    if intervals > 100000 or not np.isclose(intervals * args.stage1_threshold_step, 1, atol=1e-12, rtol=0):
        ap.error('--stage1-threshold-step must divide one exactly (<=100000 intervals)')
    if args.ic_vector and not args.ic_provenance:
        ap.error('--ic-vector requires --ic-provenance')
    run(args)


if __name__ == '__main__':
    main()
