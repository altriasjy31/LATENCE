#!/usr/bin/env python3
"""Evaluate dense external scores against the existing unseen-GO test manifest.

No LATENCE model identity is assigned to an external prediction. Candidate
coverage is reported separately and paired comparisons use the same GO axis.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from eval_unseen_go_v095 import (
    checked_ids, exact_threshold_metrics, load_inputs, ranking_metrics,
    read_manifest, sha256, write_tsv,
)
from eval_unseen_go_retrieval_v095 import evaluate_retrieval

SCHEMA = 'external-unseen-go-prediction-v1'
GO_ID = re.compile(r'GO:\d{7}')


def read_test(path, cohort_name):
    test = read_manifest(path, 'latence-unseen-go-test-v1')
    candidates = checked_ids(test['candidate_go_ids'], 'candidate GO axis')
    if any(GO_ID.fullmatch(go) is None for go in candidates):
        raise ValueError('Candidate axis must use GO:NNNNNNN identifiers')
    if cohort_name not in test['cohorts']:
        raise ValueError(f'Unknown cohort {cohort_name!r}; choices: {list(test["cohorts"])}')
    cohort = test['cohorts'][cohort_name]
    ids = checked_ids(cohort['protein_ids'], 'complete cohort protein axis')
    eligible = checked_ids(cohort['evaluable_protein_ids'], 'eligible protein axis', allow_empty=True)
    eset = set(eligible)
    if eligible != [p for p in ids if p in eset]:
        raise ValueError('Eligible proteins must be an ordered subset of the complete cohort')
    positives = defaultdict(dict)
    for pair in cohort['positives']:
        p, go = pair['protein_id'], pair['go_id']
        if p not in eset or go not in candidates:
            raise ValueError(f'Positive outside eligible test axes: {p}/{go}')
        if go in positives[p]:
            raise ValueError(f'Duplicate positive pair: {p}/{go}')
        positives[p][go] = pair
    return test, cohort, candidates, positives


def verified_artifact(folder, manifest, key):
    filename = manifest[key]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f'Invalid artifact filename for {key}')
    path = folder / filename
    digest = manifest.get('sha256', {}).get(filename)
    if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise ValueError(f'Missing SHA256 for {filename} in external manifest')
    if sha256(path) != digest:
        raise ValueError(f'Artifact SHA256 mismatch: {filename}')
    return path


def load_external(folder, cohort_ids, candidates, *, test_path=None, cohort_name=None):
    path = folder / 'external_prediction_manifest.json'
    manifest = read_manifest(path, SCHEMA)
    bound_test = manifest.get('test_manifest_sha256')
    if test_path is not None and bound_test is not None and bound_test != sha256(test_path):
        raise ValueError('External export test-manifest binding differs from evaluation test')
    bound_cohort = manifest.get('cohort')
    bound_data = read_manifest(test_path, 'latence-unseen-go-test-v1') if test_path is not None else None
    if bound_data is not None and manifest.get('namespace') not in (None, bound_data['namespace']):
        raise ValueError('External export namespace differs from evaluation test')
    if 'requested_go_ids' in manifest and manifest['requested_go_ids'] != candidates:
        raise ValueError('External export requested GO axis differs from the complete test candidates')
    protein_path = verified_artifact(folder, manifest, 'protein_ids_file')
    go_path = verified_artifact(folder, manifest, 'go_ids_file')
    probability_path = verified_artifact(folder, manifest, 'probability_file')
    ids = checked_ids(protein_path.read_text().splitlines(), 'external protein axis')
    if cohort_name is not None and bound_cohort not in (None, cohort_name):
        if bound_data is None or bound_test != sha256(test_path):
            raise ValueError('Reusing an export across evidence cohorts requires the same bound test manifest')
        origin = bound_data['cohorts'].get(bound_cohort)
        if origin is None:
            raise ValueError('External export origin cohort is absent from the bound test manifest')
        origin_ids = checked_ids(origin['protein_ids'], 'origin cohort protein axis')
        if set(origin_ids) != set(ids):
            raise ValueError('External export origin cohort has a different protein axis')
    gos = checked_ids(go_path.read_text().splitlines(), 'external GO axis', allow_empty=True)
    if any(GO_ID.fullmatch(go) is None for go in gos):
        raise ValueError('External GO axis must use GO:NNNNNNN identifiers')
    if set(ids) != set(cohort_ids):
        missing, extra = set(cohort_ids) - set(ids), set(ids) - set(cohort_ids)
        raise ValueError(f'External predictions must cover the complete cohort: {len(missing)} missing '
                         f'({sorted(missing)[:5]}), {len(extra)} extra ({sorted(extra)[:5]})')
    if set(gos) - set(candidates):
        raise ValueError('External GO axis includes terms outside the requested test candidates')
    scores = np.load(probability_path, mmap_mode='r', allow_pickle=False)
    if scores.dtype != np.float32 or scores.shape != (len(ids), len(gos)):
        raise ValueError('Expected float32 probabilities shaped (all cohort proteins, supported GO)')
    for start in range(0, len(ids), 4096):
        block = scores[start:start + 4096]
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
            raise ValueError('Probabilities must be finite and in [0, 1]')
    return dict(kind='external', manifest=manifest, manifest_path=path, ids=ids, gos=gos,
                scores=scores, name=manifest.get('model', manifest.get('model_name', folder.name)))


def load_model(folder, test_path, cohort_name, cohort_ids, candidates):
    if (folder / 'external_prediction_manifest.json').is_file():
        return load_external(folder, cohort_ids, candidates, test_path=test_path, cohort_name=cohort_name)
    result = load_inputs(test_path, folder, cohort_name, with_full_scores=True,
                         materialize_positive_scores=False)
    test, cohort, pred, pred_path, gos, positives, _, _, _, _, ids, scores = result
    return dict(kind='latence-v095', manifest=pred, manifest_path=pred_path, ids=ids, gos=gos,
                scores=scores, name='LATENCE-v095')


def classification_undefined():
    return {key: None for key in (
        'protein_fmax_observed_proxy', 'protein_fmax_threshold',
        'protein_precision_at_fmax_observed_proxy', 'protein_recall_at_fmax_observed',
        'protein_coverage_at_fmax', 'micro_ap_observed_proxy',
        'micro_auprc_trapezoid_observed_proxy', 'protein_auprc_trapezoid_observed_proxy',
        'distinct_probability_thresholds')}


def evaluate(model, selected_gos, cohort, all_positives, pool_mode, top_k, retrieval_k):
    selected = set(selected_gos)
    positives = defaultdict(dict)
    for p, terms in all_positives.items():
        for go, pair in terms.items():
            if go in selected:
                positives[p][go] = pair
    eligible = cohort['evaluable_protein_ids']
    ids = cohort['protein_ids'] if pool_mode == 'full' else eligible
    if not ids:
        raise ValueError('Selected retrieval protein pool is empty')
    positive_ids = [p for p in eligible if positives[p]]
    original_pairs = sum(map(len, all_positives.values()))
    retained_pairs = sum(map(len, positives.values()))
    old_positive_ids = {p for p, terms in all_positives.items() if terms}
    counts = dict(candidate_go_count=len(selected_gos), cohort_protein_count=len(cohort['protein_ids']),
                  identity_evaluable_protein_count=len(eligible), retrieval_protein_count=len(ids),
                  original_observed_positive_pairs=original_pairs, retained_observed_positive_pairs=retained_pairs,
                  excluded_positive_pairs_due_to_GO_coverage=original_pairs - retained_pairs,
                  original_observed_positive_proteins=len(old_positive_ids),
                  retained_observed_positive_proteins=len(positive_ids),
                  positive_proteins_without_supported_positive=len(old_positive_ids - set(positive_ids)),
                  retained_observed_positive_GO=len({g for terms in positives.values() for g in terms}),
                  eligible_without_supported_observed_positive=len(eligible) - len(positive_ids))
    status = 'ok' if retained_pairs else 'no_observed_positive_metrics_undefined'
    if not selected_gos:
        return dict(status='no_supported_GO_metrics_undefined', counts=counts,
                    protein_to_GO={'ranking': None, 'classification_positive_proteins': classification_undefined()},
                    GO_to_protein=None), {'protein_positive_pairs': [], 'positive_proteins': [],
                                         'protein_per_GO': [], 'retrieval_per_GO': [], 'retrieval_positive_pairs': []}
    row_index = {p: i for i, p in enumerate(model['ids'])}
    col_index = {g: i for i, g in enumerate(model['gos'])}
    columns = np.asarray([col_index[g] for g in selected_gos], dtype=np.intp)
    pos_rows = np.asarray([row_index[p] for p in positive_ids], dtype=np.intp)
    scores = np.asarray(model['scores'][np.ix_(pos_rows, columns)])
    truth = np.zeros(scores.shape, dtype=np.bool_)
    for row, p in enumerate(positive_ids):
        truth[row] = [g in positives[p] for g in selected_gos]
    ranking, pairs, proteins, per_go = ranking_metrics(selected_gos, positive_ids, scores, truth, positives,
                                                      top_k, len(eligible))
    for k in top_k:
        effective = min(k, len(selected_gos))
        for protein in proteins:
            protein[f'f1_observed_proxy@{k}'] = (2 * protein[f'expected_hits@{k}'] /
                                                (effective + protein['observed_positives']))
        ranking[f'protein_macro_f1_observed_proxy@{k}'] = (
            float(np.mean([p[f'f1_observed_proxy@{k}'] for p in proteins])) if proteins else None)
    classification = exact_threshold_metrics(scores, truth) if positive_ids else classification_undefined()
    pool_rows = np.asarray([row_index[p] for p in ids], dtype=np.intp)
    by_go = defaultdict(list)
    for p, terms in positives.items():
        for g in terms:
            by_go[g].append(p)
    retrieval, retrieval_go, retrieval_pairs = evaluate_retrieval(
        selected_gos, ids, by_go, lambda i, g: model['scores'][pool_rows, col_index[g]], retrieval_k)
    for k in retrieval_k:
        effective = min(k, len(ids))
        for row in retrieval_go:
            if row['n_observed_positives']:
                row[f'observed_f1_proxy@{k}'] = (2 * row[f'observed_recall@{k}'] *
                    row['n_observed_positives'] / (effective + row['n_observed_positives']))
            else:
                row[f'observed_f1_proxy@{k}'] = None
        defined = [row[f'observed_f1_proxy@{k}'] for row in retrieval_go if row['n_observed_positives']]
        retrieval[f'go_macro_observed_f1_proxy@{k}'] = float(np.mean(defined)) if defined else None
    return dict(status=status, counts=counts, protein_to_GO=dict(ranking=ranking,
                classification_positive_proteins=classification), GO_to_protein=retrieval), {
        'protein_positive_pairs': pairs, 'positive_proteins': proteins, 'protein_per_GO': per_go,
        'retrieval_per_GO': retrieval_go, 'retrieval_positive_pairs': retrieval_pairs}


def paired_rows(external, reference):
    ref = {(r['protein_id'], r['go_id']): r for r in reference}
    if set(ref) != {(r['protein_id'], r['go_id']) for r in external}:
        raise ValueError('Paired comparison positive axes differ')
    result = []
    for row in external:
        other = ref[row['protein_id'], row['go_id']]
        output = {'protein_id': row['protein_id'], 'go_id': row['go_id']}
        for name, values in [('external', row), ('reference', other)]:
            output.update({f'{name}_{k}': v for k, v in values.items()
                           if k not in ('protein_id', 'go_id', 'evidence_codes')})
        output['external_rank_improvement'] = other['midrank'] - row['midrank']
        result.append(output)
    return result


def numeric_delta(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return {key: numeric_delta(value, b[key]) for key, value in a.items() if key in b}
    if a is None or b is None:
        return None
    if type(a) in (int, float) and type(b) in (int, float):
        return a - b
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-manifest', type=Path, required=True)
    parser.add_argument('--cohort', default='ind_test_exp')
    parser.add_argument('--predictions-dir', type=Path, required=True)
    parser.add_argument('--reference-predictions-dir', type=Path)
    parser.add_argument('--protein-pool', choices=['full', 'evaluable'], default='full')
    parser.add_argument('--top-k', type=int, nargs='+', default=[1, 3, 5, 10, 20, 50])
    parser.add_argument('--retrieval-top-k', type=int, nargs='+', default=[10, 50, 100, 500, 1000])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    for ks in (args.top_k, args.retrieval_top_k):
        if min(ks) < 1 or len(set(ks)) != len(ks):
            parser.error('Top-K values must be distinct positive integers')
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f'Choose a new output directory: {args.output_dir}')
    test, cohort, candidates, positives = read_test(args.test_manifest, args.cohort)
    external = load_external(args.predictions_dir, cohort['protein_ids'], candidates,
                             test_path=args.test_manifest, cohort_name=args.cohort)
    reference = load_model(args.reference_predictions_dir, args.test_manifest, args.cohort,
                           cohort['protein_ids'], candidates) if args.reference_predictions_dir else None
    external_gos = [g for g in candidates if g in set(external['gos'])]
    reference_gos = [g for g in candidates if g in set(reference['gos'])] if reference else []
    common = [g for g in external_gos if g in set(reference_gos)] if reference else []
    result = dict(schema='external-unseen-go-evaluation-v1', complete=True, cohort=args.cohort,
                  namespace=test['namespace'], protein_pool=args.protein_pool,
                  test_manifest_sha256=sha256(args.test_manifest),
                  strict_zero_shot_claimed=False,
                  definitions={
        'unobserved': 'Unannotated pairs are unlabeled; precision, F1, AP and Fmax are observed-label proxies.',
        'protein_ranking': 'Ranks supported GO for each protein with at least one supported observed positive. Complete-cohort and eligible denominators are reported separately.',
        'classification_positive_proteins': 'Exactly matches original v095 evaluator: Fmax/AP use eligible proteins with supported observed positives. Not a full-population specificity measurement.',
        'protein_precision_all_evaluable_rows_proxy': 'Observed top-K hits divided by all eligible proteins times effective K; zero-positive proteins stay in this denominator.',
        'retrieval': 'Ranks every protein in the selected complete pool for each supported GO. No-positive GO are listed; macro metrics average only over GO with observed positives.',
        'tie_policy': 'Exact ties use expected top-K credit and reciprocal first rank under uniform within-tie order. AP uses grouped threshold precision.',
        'comparison': 'Common GO axis is the supported GO intersection in test order. No missing column is zero-filled; removed positive pairs and proteins are counted.',
        'rank_improvement': 'reference midrank minus external midrank; positive means external ranks the observed pair earlier.',
        'zero_shot': 'Model score evaluation does not certify training-label, protein, domain, or ontology exposure; inspect each export provenance.',
        'threshold': 'Fmax is a descriptive threshold sweep on this test; do not reuse the selected threshold as a validated deployment threshold.'},
                  models={}, scopes={})
    details = {}
    plans = [('external_supported', external, external_gos)]
    if reference:
        plans += [('reference_supported', reference, reference_gos), ('external_common', external, common),
                  ('reference_common', reference, common)]
    for label, model in [('external', external), ('reference', reference)]:
        if model:
            result['models'][label] = dict(name=model['name'], kind=model['kind'],
                prediction_manifest_sha256=sha256(model['manifest_path']),
                supported_candidate_GO=len(model['gos']), missing_candidate_GO=len(candidates) - len(model['gos']),
                supported_go_ids=[g for g in candidates if g in set(model['gos'])],
                missing_go_ids=[g for g in candidates if g not in set(model['gos'])],
                protein_rows_reordered=model['ids'] != cohort['protein_ids'],
                prediction_provenance=model['manifest'])
    for scope, model, gos in plans:
        result['scopes'][scope], details[scope] = evaluate(
            model, gos, cohort, positives, args.protein_pool, args.top_k, args.retrieval_top_k)
    if reference:
        result['common_candidate_go_ids'] = common
        result['external_minus_reference_common'] = numeric_delta(result['scopes']['external_common'],
                                                                   result['scopes']['reference_common'])
    coverage = []
    for go in candidates:
        coverage.append(dict(go_id=go, observed_positive_pairs=sum(go in t for t in positives.values()),
                             external_supported=go in external_gos,
                             reference_supported=(go in reference_gos) if reference else None,
                             common_supported=(go in common) if reference else None))
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.external_unseen_eval.', dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / 'output'
        stage.mkdir()
        (stage / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                                       allow_nan=False) + '\n')
        write_tsv(stage / 'coverage.tsv', coverage, ['go_id'])
        for scope, tables in details.items():
            dest = stage / scope
            dest.mkdir()
            for name, rows in tables.items():
                write_tsv(dest / f'{name}.tsv', rows, ['protein_id', 'go_id'])
        if reference:
            for direction, name in [('protein_to_GO', 'protein_positive_pairs'),
                                    ('GO_to_protein', 'retrieval_positive_pairs')]:
                rows = paired_rows(details['external_common'][name], details['reference_common'][name])
                write_tsv(stage / f'paired_{direction}.tsv', rows, ['protein_id', 'go_id'])
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(args.output_dir)
        stage.rename(args.output_dir)
    print(args.output_dir / 'metrics.json')


if __name__ == '__main__':
    main()
