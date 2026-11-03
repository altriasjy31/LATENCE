#!/usr/bin/env python3
"""Four-view Functional Observability Atlas evaluation from frozen probabilities.

Evidence memberships intentionally overlap. Direct canonical GO relationships
define recovery; ancestor closure is retained for optional classification metrics.
The program never runs a model, propagates predictions, or changes training data.
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
for path in (PROJECT, PROJECT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_dark_weak_predictions as ranking
from dark_weak import eval_stage1_frequency_v053 as common
from dark_weak.complete_metric_pack_v054 import comprehensive_metrics, load_kernels, metric_protocol
from dark_weak.atlas_evidence_v060 import (
    canonical_sets, evidence_membership, rank_positive_pairs, summarize_ranks, recovery_expansion, RECOVERY_COLUMNS,
)

SCHEMA = 'latence-functional-observability-atlas-evaluation-v060'
SPLITS = {
    'future_union': 'dark_weak_test_direct',
    'future_exp': 'dark_weak_test_exp_direct',
    'future_ipr': 'dark_weak_test_ipr2go_direct',
    'history': 'dark_weak_test_2022_ipr2go_direct',
}
FLAGS = {'future_union': 'in_future_union', 'future_exp': 'in_future_exp',
         'future_ipr': 'in_future_ipr', 'history': 'in_history'}


def write_tsv(path, frame, columns=()):
    if frame.empty and not len(frame.columns):
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, sep='\t', index=False)


def read_views(metadata, namespace, width):
    views = {}
    for layer, split in SPLITS.items():
        if split not in metadata or ranking.NS[namespace] not in metadata[split]:
            raise KeyError(f'Missing {split}/{namespace}; build future/history metadata views first')
        data = metadata[split][ranking.NS[namespace]]
        ids = list(map(str, data['proteins']))
        if len(ids) != len(set(ids)):
            raise ValueError(f'{split}/{namespace}: duplicate proteins')
        for field in ('annotations', 'prop_annotations'):
            if field not in data or len(data[field]) != len(ids):
                raise ValueError(f'{split}/{namespace}: missing/misaligned {field}')
        direct, propagated = {}, {}
        for protein, a, b in zip(ids, data['annotations'], data['prop_annotations']):
            x = common.np_unique_idx(a, width, f'{split}/{protein}/direct')
            y = common.np_unique_idx(b, width, f'{split}/{protein}/propagated')
            if not np.isin(x, y).all():
                raise ValueError(f'{split}/{protein}: ancestor closure omits direct labels')
            direct[protein], propagated[protein] = x, y
        views[layer] = dict(split=split, ids=ids, direct=direct, propagated=propagated)
    return views


def inventory_rows(namespace, views, sets):
    result = []
    for layer, view in views.items():
        pairs = sets[layer]
        result.append(dict(namespace=namespace, evidence_layer=layer, split=view['split'],
            n_proteins=len(view['ids']), n_raw_direct_pairs=sum(len(x) for x in view['direct'].values()),
            n_ancestor_pairs=sum(len(x) for x in view['propagated'].values()),
            n_canonical_direct_pairs_without_root=len(pairs),
            n_canonical_positive_proteins=len({p for p, _ in pairs}),
            n_canonical_positive_GO=len({g for _, g in pairs})))
    return result


def load_config(args):
    config = {}
    origin = {}
    if args.prediction_config:
        if args.complete_manifest:
            raise ValueError('Use --prediction-config or --complete-manifest, not both')
        path = Path(args.prediction_config).resolve(strict=True)
        config = common.read_json(path)
        if not isinstance(config, dict):
            raise TypeError('Prediction config must map namespace to model records')
        # Config-relative filenames are explicit and portable.
        for namespace, records in config.items():
            for model, record in records.items():
                for field in ('probability', 'protein_ids', 'go_ids'):
                    p = Path(record[field])
                    record[field] = str(p if p.is_absolute() else path.parent / p)
        origin['prediction_config'] = {'path': str(path), 'sha256': common.sha256(path)}
    for item in args.complete_manifest:
        if '=' not in item:
            raise ValueError('--complete-manifest needs NAMESPACE=PATH')
        ns, filename = item.split('=', 1)
        if ns not in ranking.NS or ns in config:
            raise ValueError(f'Duplicate/invalid complete-manifest namespace: {ns}')
        path = Path(filename).resolve(strict=True)
        record = common.read_json(path)
        if record.get('namespace') != ns:
            raise ValueError('Complete manifest namespace differs from argument')
        config[ns] = record['input_files']
        origin[ns] = {'path': str(path), 'sha256': common.sha256(path)}
    return config, origin


def load_models(records, gos, required, namespace, inventory_only=False):
    models, provenance, coverage = {}, {}, []
    for model, rec in records.items():
        pp, ip, gp = [Path(rec[field]).resolve(strict=True) for field in ('probability', 'protein_ids', 'go_ids')]
        if ranking.read_go_list(gp) != gos:
            raise ValueError(f'{namespace}/{model}: GO order differs from frozen vocabulary')
        array = ranking.load_prediction(pp, len(gos))
        order = ranking.load_protein_order(ip, array.shape[0])
        digest = common.sha256(pp)
        if rec.get('probability_sha256') and rec['probability_sha256'] != digest:
            raise ValueError(f'{namespace}/{model}: frozen prediction hash differs')
        id_hash, go_hash = common.sha256(ip), common.sha256(gp)
        for key, actual in [('protein_ids_sha256', id_hash), ('go_ids_sha256', go_hash)]:
            if rec.get(key) and rec[key] != actual:
                raise ValueError(f'{namespace}/{model}: frozen {key} differs')
        missing = sorted(required - set(order))
        coverage.append(dict(namespace=namespace, model=model, prediction_rows=len(order),
                             required_proteins=len(required), covered=len(required) - len(missing),
                             missing=len(missing), missing_examples=';'.join(missing[:5])))
        provenance[model] = {'probability': str(pp), 'probability_sha256': digest,
                            'protein_ids': str(ip), 'protein_ids_sha256': id_hash,
                            'go_ids': str(gp), 'go_ids_sha256': go_hash,
                            'branch': rec.get('branch', 'not_supplied'), 'storage_dtype': str(array.dtype),
                            'native_adapter': rec.get('native_adapter'),
                            'native_adapter_verified_by_this_program': False}
        if missing and not inventory_only:
            raise ValueError(f'{namespace}/{model}: {len(missing)} of {len(required)} required proteins missing '
                             f'(e.g. {missing[:3]}). Export the four-view inference union first; no cohort is silently reduced.')
        models[model] = (array, order)
    return models, provenance, coverage


def evidence_counts(namespace, sets):
    h, f, e, i = (sets[x] for x in ('history', 'future_union', 'future_exp', 'future_ipr'))
    return dict(namespace=namespace, historical_pairs=len(h), future_union_pairs=len(f),
        future_exp_pairs=len(e), future_ipr_pairs=len(i), both_future_sources_pairs=len(e & i),
        future_history_overlap=len(f & h), novel_future_pairs=len(f - h), atlas_observed_union_pairs=len(h | f),
        observed_expansion_fraction=len(f - h) / len(h) if h else None,
        historical_proteins=len({p for p, _ in h}), future_proteins=len({p for p, _ in f}),
        future_proteins_without_historical_direct_GO=len({p for p, _ in f} - {p for p, _ in h}))


def paired_bootstrap(frame, reference, target, ks, reps, seed):
    """Sequence-cluster paired intervals on positive-only ranking, per one view."""
    keys = ['protein', 'go_id', 'sequence_hash']
    a = frame[frame.model.eq(target)].sort_values(keys).reset_index(drop=True)
    b = frame[frame.model.eq(reference)].sort_values(keys).reset_index(drop=True)
    if a.empty or b.empty:
        return []
    if not a[keys].equals(b[keys]):
        raise ValueError('Paired ranking comparison needs identical canonical positive pairs')
    clusters, inverse = np.unique(a.sequence_hash.to_numpy(), return_inverse=True)
    cluster_counts = np.bincount(inverse, minlength=len(clusters))
    rng = np.random.default_rng(seed)
    fields = ['reciprocal_rank'] + [f'hit_at_{k}' for k in ks] + [f'hit_at_{k}_conservative' for k in ks]
    rows = []
    for field in fields:
        differences = a[field].to_numpy(float) - b[field].to_numpy(float)
        sums = np.bincount(inverse, weights=differences, minlength=len(clusters))
        estimates = []
        # One draw per repetition avoids bootstrap_reps x N cluster allocations.
        for _ in range(reps):
            draw = rng.integers(0, len(clusters), size=len(clusters))
            estimates.append(sums[draw].sum() / cluster_counts[draw].sum())
        lo, hi = np.quantile(estimates, [.025, .975]).tolist() if estimates else (None, None)
        rows.append(dict(reference=reference, model=target, metric=field, delta=float(differences.mean()),
                         ci95_low=lo, ci95_high=hi, n_clusters=len(clusters), repetitions=reps,
                         bootstrap_unit='exact sequence SHA256, protein ID fallback'))
    return rows


def incremental_pair_rows(frame, evidence, target, references, ks, namespace):
    """Concrete gained/lost relations; one membership record retains all sources."""
    if target not in set(frame.model):
        return []
    target_rows = frame[frame.model.eq(target)].set_index(['protein', 'go_id'])
    memberships = evidence.set_index(['protein', 'go_id'])
    records = []
    for k in ks:
        for policy in ('stable', 'conservative'):
            field = f'hit_at_{k}' + ('_conservative' if policy == 'conservative' else '')
            reference_hits = {}
            for model in references:
                sub = frame[frame.model.eq(model) & frame[field].gt(0)]
                for pair in zip(sub.protein, sub.go_id):
                    reference_hits.setdefault(pair, []).append(model)
            target_hits = set(target_rows.index[target_rows[field].gt(0)])
            ref_hits = set(reference_hits)
            for pair in sorted(target_hits ^ ref_hits):
                row, membership = target_rows.loc[pair], memberships.loc[pair]
                records.append(dict(namespace=namespace, protein=pair[0], go_id=pair[1],
                    k=k, hit_policy=policy, state='gained' if pair in target_hits else 'lost',
                    target_model=target, target_score=float(row.score), target_rank=int(row['rank']),
                    target_rank_pessimistic=int(row.rank_pessimistic),
                    stage1_recovering_models='|'.join(reference_hits.get(pair, [])),
                    **{flag: bool(membership[flag]) for flag in FLAGS.values()},
                    novel_vs_history=bool(membership.in_future_union and not membership.in_history)))
    return records


def run(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    root = Path(args.unified_root).resolve(strict=True)
    metadata_path = Path(args.metadata).resolve(strict=True)
    with metadata_path.open('rb') as f:
        metadata = pickle.load(f)  # Trusted local LATENCE pickle only.
    config, config_origin = load_config(args)
    if not args.inventory_only and not config:
        raise ValueError('Evaluation needs --prediction-config or --complete-manifest; use --inventory-only for a label inventory')
    if not args.inventory_only:
        for ns in args.namespaces:
            if not config.get(ns):
                raise ValueError(f'No frozen predictions supplied for requested namespace {ns}')
            if args.target_model in config[ns]:
                missing_refs = set(args.stage1_models) - set(config[ns])
                if missing_refs:
                    raise ValueError(f'{ns}: target model comparison needs all Stage-1 references: {sorted(missing_refs)}')
    seqhash = {}
    cohort = root / 'tables/full_dark_weak_cohort.tsv'
    if cohort.is_file():
        data = pd.read_csv(cohort, sep='\t', usecols=['protein_22', 'sequence_sha256']).dropna()
        seqhash = dict(zip(data.protein_22.astype(str), data.sequence_sha256.astype(str)))
    all_inventory, all_evidence, all_ranks, all_metrics = [], [], [], []
    all_expansion, all_counts, all_classification, all_bootstrap, all_coverage = [], [], [], [], []
    all_incremental_pairs = []
    audits, inputs, frequency_audit = {}, {}, {}
    kernels = None
    if not args.inventory_only and args.classification != 'none':
        kernels = load_kernels(PROJECT, official=not args.skip_stage1_official,
                               official_source=args.stage1_helper_source)
    cross_namespace_proteins = {key: set() for key in SPLITS}
    for ns in args.namespaces:
        print(f'[atlas v060] {ns}: four nested evidence views', flush=True)
        gos, canon, inverse, sizes = common.read_vocabulary(root, ns)
        views = read_views(metadata, ns, len(gos))
        root_col = canon.index(ranking.ROOT_IDS[ns]) if ranking.ROOT_IDS[ns] in canon else None
        sets, canonical_audit = canonical_sets({k: v['direct'] for k, v in views.items()}, inverse, root_col=root_col)
        evidence, audit = evidence_membership(sets, canon, strict_union=True, strict_novel=not args.allow_history_overlap)
        evidence['namespace'] = ns
        all_evidence.append(evidence)
        all_inventory.extend(inventory_rows(ns, views, sets))
        all_counts.append(evidence_counts(ns, sets))
        audits[ns] = {'canonicalization': canonical_audit, 'memberships': audit,
                      'vocabulary_sha256': common.sha256(root / 'labels' / ns / 'go_ids.txt'),
                      'vocabulary_mapping_sha256': common.sha256(root / 'labels' / ns / 'vocabulary_mapping.tsv')}
        for layer in SPLITS:
            cross_namespace_proteins[layer].update(views[layer]['ids'])
        required = set().union(*(set(v['ids']) for v in views.values()))
        models, inputs[ns], coverage = load_models(config.get(ns, {}), gos, required, ns, args.inventory_only)
        all_coverage.extend(coverage)
        if args.inventory_only:
            continue
        if not models:
            raise ValueError(f'No frozen predictions supplied for requested namespace {ns}')
        freq = None
        bins = {'all': np.ones(len(canon), bool)}
        if args.train_metadata:
            _, freq, ntrain = common.get_train_counts(args.train_metadata, ns, args.train_split,
                args.train_label_field, len(gos), inverse, len(canon))
            classic, bin_info = common.classic_bins(freq)
            bins.update(classic)
            frequency_audit[ns] = {'training_proteins': ntrain, 'groups': bin_info}
        ranked = []
        positive_pairs = set().union(*sets.values())
        for model, (array, order) in models.items():
            # Kernel uses row indexing/memmaps and validates each scored row;
            # no NxGO all-model copy is made for direct-pair ranking.
            ids = [None] * len(order)
            for name, row in order.items():
                ids[row] = name
            frame = rank_positive_pairs(array, ids, canon, inverse, sizes, positive_pairs,
                                       model=model, ks=args.ks, root_id=ranking.ROOT_IDS[ns])
            if not frame.empty:
                frame = frame.merge(evidence.drop(columns='namespace'), on=['protein', 'canonical_index', 'go_id'],
                                    how='left', validate='many_to_one')
                frame['namespace'] = ns
                frame['sequence_hash'] = frame.protein.map(lambda p: seqhash.get(p, p))
                frame['train_frequency'] = frame.canonical_index.map(lambda c: int(freq[c])) if freq is not None else np.nan
            ranked.append(frame)
            for layer, view in views.items():
                if args.classification == 'none' or (args.classification == 'future' and layer == 'history'):
                    continue
                cells = len(view['ids']) * len(gos)
                if cells > args.max_metric_cells and not args.allow_large_metrics:
                    raise MemoryError(f'{ns}/{layer}: {cells:,} classification cells; ranking is unaffected. '
                                      'Use --classification future/none, or review RAM and add --allow-large-metrics.')
                if not view['ids']:
                    continue
                y = np.zeros((len(view['ids']), len(gos)), np.uint8)
                for row, protein in enumerate(view['ids']):
                    y[row, view['propagated'][protein]] = 1
                p = np.asarray(array[[order[protein] for protein in view['ids']]], dtype=np.float32)
                stats = comprehensive_metrics(y, p, native=kernels['native'], legacy=kernels['legacy'],
                    official=kernels['official'], threshold_step=.01, compute_sample_fmax=True)
                all_classification.append(dict(namespace=ns, evidence_layer=layer, split=view['split'],
                    model=model, label_field='prop_annotations', **stats))
        rank_frame = pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()
        all_ranks.append(rank_frame)
        for layer, flag in FLAGS.items():
            for name, mask in bins.items():
                layer_frame = rank_frame[rank_frame[flag].eq(True)] if not rank_frame.empty else rank_frame
                if not layer_frame.empty:
                    layer_frame = layer_frame[layer_frame.canonical_index.map(lambda c: bool(mask[c]))]
                for model in models:
                    sub = layer_frame[layer_frame.model.eq(model)] if not layer_frame.empty else layer_frame
                    all_metrics.append(dict(namespace=ns, evidence_layer=layer, split=SPLITS[layer],
                        frequency_bin=name, model=model, **summarize_ranks(sub, args.ks)))
                if not layer_frame.empty and args.reference in models:
                    for model in models:
                        if model != args.reference:
                            for record in paired_bootstrap(layer_frame, args.reference, model, args.ks,
                                                           args.bootstrap, args.seed):
                                all_bootstrap.append(dict(namespace=ns, evidence_layer=layer,
                                                           frequency_bin=name, **record))
        if args.target_model in models:
            expansion = recovery_expansion(rank_frame, evidence, ks=args.ks,
                stage1_models=tuple(args.stage1_models), target_model=args.target_model)
            expansion['namespace'] = ns
            all_expansion.append(expansion)
            all_incremental_pairs.extend(incremental_pair_rows(rank_frame, evidence,
                args.target_model, args.stage1_models, args.ks, ns))
        audits[ns]['target_present'] = args.target_model in models
        audits[ns]['sequence_hash_coverage'] = sum(p in seqhash for p in required)
    inventory = pd.DataFrame(all_inventory)
    observed = pd.concat(all_evidence, ignore_index=True) if all_evidence else pd.DataFrame()
    ranks = pd.concat(all_ranks, ignore_index=True) if all_ranks else pd.DataFrame()
    totals = {layer: {'unique_proteins_across_namespaces': len(proteins),
               'canonical_direct_pairs_without_root': 0,
               'raw_direct_pairs': int(inventory[inventory.evidence_layer.eq(layer)].n_raw_direct_pairs.sum())}
              for layer, proteins in cross_namespace_proteins.items()}
    for layer in SPLITS:
        totals[layer]['canonical_direct_pairs_without_root'] = int(
            inventory[inventory.evidence_layer.eq(layer)].n_canonical_direct_pairs_without_root.sum())
    manifest = dict(schema=SCHEMA, complete=True, inventory_only=args.inventory_only,
        metadata=str(metadata_path), metadata_sha256=common.sha256(metadata_path),
        namespace_names=ranking.NS, namespaces=args.namespaces, split_profile=SPLITS,
        config_sources=config_origin, input_files=inputs, audit=audits, unique_counts=totals,
        train_metadata=({'path': str(Path(args.train_metadata).resolve()), 'sha256': common.sha256(args.train_metadata),
                         'split': args.train_split, 'label_field': args.train_label_field,
                         'canonical_frequency_audit': frequency_audit} if args.train_metadata else None),
        ks=args.ks, reference=args.reference, stage1_models=args.stage1_models, target_model=args.target_model,
        classification=args.classification, classification_protocol=metric_protocol(kernels) if kernels else None,
        bootstrap_repetitions=args.bootstrap, bootstrap_seed=args.seed,
        protocol={'nested_membership': 'preserved as evidence expansion; views are not independent replicates',
            'atlas_observed_union': 'historical canonical direct evidence UNION future confirmed direct evidence',
            'prediction_axis': 'ALL canonical GO in frozen model vocabulary, namespace root excluded',
            'alias_scores': 'mean of original output columns',
            'headline_recovery': 'positive-score AND pessimistic tied-group rank <= K',
            'stable_ranks': 'original canonical GO column order breaks equal-score ties, diagnostic only',
            'new_stage2_recovery': 'target recovered positives minus union of supplied Stage-1 recovered positives at same K',
            'not_probability_fusion': 'Stage-1 hit-set union is a stronger comparator with up to number_of_references times K candidates; not a same-budget deployed ensemble',
            'history_role': 'historical computational evidence baseline, not prospective discoveries',
            'closure_role': 'classification proxy / hierarchy, never counted as independent discoveries',
            'unvalidated_predictions': 'not exported or labeled wrong; observed edge export alone is not the full Atlas universe',
            'unseen_GO': 'not evaluated in this release; see docs/UNSEEN_GO_EVALUATION_PLAN.md',
            'zero_train': 'zero historical chosen-field positive count, not pipeline zero-shot',
            'native_NBS_provenance': 'paths recorded only; validate native exports with existing v094 adapter before use'},
        software={'numpy': np.__version__, 'pandas': pd.__version__},
        original_files_modified=False, model_inference_performed=False)
    out.mkdir(parents=True, exist_ok=False)
    write_tsv(out/'dataset_inventory.tsv', inventory)
    write_tsv(out/'prediction_coverage.tsv', pd.DataFrame(all_coverage),
              ['namespace', 'model', 'prediction_rows', 'required_proteins', 'covered', 'missing', 'missing_examples'])
    write_tsv(out/'evidence_expansion.tsv', pd.DataFrame(all_counts))
    write_tsv(out/'atlas_observed_relations.tsv', observed)
    write_tsv(out/'atlas_pair_ranks.tsv', ranks, ['namespace', 'model', 'protein', 'go_id', 'rank'])
    write_tsv(out/'atlas_metrics.tsv', pd.DataFrame(all_metrics), ['namespace', 'evidence_layer', 'frequency_bin', 'model'])
    write_tsv(out/'classification_metrics.tsv', pd.DataFrame(all_classification), ['namespace', 'evidence_layer', 'model'])
    expansion = pd.concat(all_expansion, ignore_index=True) if all_expansion else pd.DataFrame()
    write_tsv(out/'stage2_incremental_recovery.tsv', expansion,
              ['namespace', *RECOVERY_COLUMNS])
    write_tsv(out/'stage2_incremental_pairs.tsv', pd.DataFrame(all_incremental_pairs),
              ['namespace', 'protein', 'go_id', 'k', 'hit_policy', 'state', 'target_model',
               'target_score', 'target_rank', 'target_rank_pessimistic', 'stage1_recovering_models',
               *FLAGS.values(), 'novel_vs_history'])
    write_tsv(out/'paired_rank_bootstrap.tsv', pd.DataFrame(all_bootstrap),
              ['namespace', 'evidence_layer', 'frequency_bin', 'reference', 'model', 'metric', 'delta', 'ci95_low', 'ci95_high'])
    (out/'atlas_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(inventory.to_string(index=False), flush=True)
    print(f'[atlas v060] Completed: {out}', flush=True)
    return inventory, manifest


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata', type=Path, default=Path('data/unidata_with_dark_weak_eval.pkl'))
    ap.add_argument('--unified-root', type=Path, default=Path('data/dark_weak_test/unified'))
    ap.add_argument('--namespaces', nargs='+', choices=tuple(ranking.NS), default=['bp', 'mf', 'cc'])
    ap.add_argument('--prediction-config', type=Path)
    ap.add_argument('--complete-manifest', action='append', default=[], metavar='NS=PATH')
    ap.add_argument('--inventory-only', action='store_true')
    ap.add_argument('--classification', choices=('none', 'future', 'all'), default='future')
    ap.add_argument('--skip-stage1-official', action='store_true')
    ap.add_argument('--stage1-helper-source', type=Path)
    ap.add_argument('--train-metadata', type=Path)
    ap.add_argument('--train-split', default='train')
    ap.add_argument('--train-label-field', choices=('annotations', 'prop_annotations'), default='prop_annotations')
    ap.add_argument('--reference', default='backbone')
    ap.add_argument('--stage1-models', nargs='+', default=['backbone', 'expert', 'modelout'])
    ap.add_argument('--target-model', default='nbs')
    ap.add_argument('--ks', nargs='+', type=int, default=[10, 50, 100, 512, 1024])
    ap.add_argument('--bootstrap', type=int, default=500)
    ap.add_argument('--seed', type=int, default=6061)
    ap.add_argument('--max-metric-cells', type=int, default=80_000_000)
    ap.add_argument('--allow-large-metrics', action='store_true')
    ap.add_argument('--allow-history-overlap', action='store_true', help='Diagnostic only: record novelty violations instead of failing')
    ap.add_argument('--out', type=Path, required=True)
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if len(set(args.namespaces)) != len(args.namespaces) or len(set(args.ks)) != len(args.ks) or any(k < 1 for k in args.ks):
        ap.error('Namespaces and K values must be distinct; K must be positive')
    if args.bootstrap < 0 or args.max_metric_cells < 1:
        ap.error('Bootstrap must be >=0 and memory cell limit must be positive')
    if len(set(args.stage1_models)) != len(args.stage1_models) or args.target_model in args.stage1_models:
        ap.error('Stage-1 models must be distinct and cannot include target model')
    run(args)


if __name__ == '__main__':
    main()
