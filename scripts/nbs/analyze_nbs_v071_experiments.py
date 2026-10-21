#!/usr/bin/env python3
"""Analyze available staged outputs without selecting an independent-test epoch."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def analyze_smoke(directory: Path):
    directory = Path(directory)
    history_path, config_path = directory / 'training_history.json', directory / 'resolved_config.json'
    if not history_path.is_file() or not config_path.is_file():
        return {'status': 'missing', 'directory': str(directory), 'reason': 'completed training_history.json and resolved_config.json required'}
    history, config = read(history_path), read(config_path)
    if not isinstance(history, list) or not history:
        return {'status': 'failed', 'directory': str(directory), 'reason': 'empty or invalid training history'}
    last = history[-1]
    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)
    checks = {
        'twenty_steps_per_rank': last.get('steps_per_rank') == 20,
        'finite_total_loss': finite(last.get('total')),
        'no_cap_drop': last.get('avg_candidate_dropped_by_cap') == 0,
        'all_sampled_supervision_retained': last.get('avg_full_supervision_retained') == 1,
        'positive_evidence_present': last.get('avg_candidate_evidence_positive_pairs', 0) > 0,
        'unknown_evidence_present': last.get('avg_candidate_evidence_unknown_pairs', 0) > 0,
        'saved_epoch1': (directory / 'nbs_epoch1.pt').is_file(),
    }
    column = config.get('loss', {}).get('weights', {}).get('protein_column', 0) > 0
    if column:
        checks.update(
            column_loss_active=last.get('contrib_protein_column_asl', 0) > 0,
            multi_positive_columns=last.get('avg_weak_primary_positive_go_per_anchor_mean', 0) > 1,
            hard_column_pairs_present=last.get('avg_weak_primary_total_hard_pairs', 0) > 0,
        )
    gradients = {}
    for rank in range(int(last.get('world_size', 1))):
        file = directory / f'gradient_probe_rank{rank}.json'
        if not file.is_file():
            checks[f'rank{rank}_gradient_log_present'] = False
            continue
        records = read(file).get('records', [])
        checks[f'rank{rank}_gradient_records'] = len(records) == 20
        checks[f'rank{rank}_gradients_finite'] = bool(records) and all(p.get('grad_finite', False) for r in records for p in r.get('parameters', {}).values())
        for suffix in ('graph_delta_scale', 'candidate_evidence_residual.2.weight', 'candidate_evidence_scale'):
            norms = [p.get('grad_l2', 0.) for r in records for name, p in r.get('parameters', {}).items() if name == 'matcher.' + suffix]
            peak = max(norms, default=0.)
            gradients[f'rank{rank}/{suffix}'] = peak
            checks[f'rank{rank}_{suffix}_gradient_started'] = finite(peak) and peak > 0
    return {'status': 'passed' if all(checks.values()) else 'needs_attention',
            'directory': str(directory), 'checks': checks, 'peak_parameter_gradient_l2': gradients,
            'training': last,
            'note': 'Semantic/gradient gate only; does not establish generalization or a performance gain.'}


def metric(mapping, key):
    value = mapping.get(key, mapping.get({'fmax': 'Fmax', 'auprc': 'AuPRC'}.get(key, key)))
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def evaluation_rows(summary, name):
    rows = []
    for record in summary.get('records', []):
        primary = record.get('primary', {})
        nbs, base = primary.get('NBS_final', {}), primary.get('backbone_base', {})
        f, a, bf, ba = metric(nbs, 'fmax'), metric(nbs, 'auprc'), metric(base, 'fmax'), metric(base, 'auprc')
        row = {'experiment': name, 'epoch': record.get('epoch'), 'num_samples': record.get('num_samples'),
               'fmax': f, 'auprc': a, 'backbone_fmax': bf, 'backbone_auprc': ba,
               'delta_fmax_pp': None if f is None or bf is None else f - bf,
               'delta_auprc_pp': None if a is None or ba is None else a - ba,
               'candidate_positive_recall': record.get('candidate_coverage', {}).get('positive_label_recall')}
        for group in ('candidate', 'non_candidate'):
            values = record.get('candidate_analysis', {})
            n = values.get('NBS_final', {}).get(group, {}).get('auprc_micro_hist')
            b = values.get('backbone_base', {}).get(group, {}).get('auprc_micro_hist')
            row[f'delta_{group}_auprc_micro_hist_pp'] = None if n is None or b is None else n - b
        for group in ('rare', 'rare_le_5', 'medium', 'common'):
            values = record.get('rare_analysis', {})
            n = values.get('NBS_final', {}).get(group, {}).get('auprc_micro_hist')
            b = values.get('backbone_base', {}).get(group, {}).get('auprc_micro_hist')
            row[f'delta_{group}_auprc_micro_hist_pp'] = None if n is None or b is None else n - b
        deltas = record.get('diagnostics', {}).get('probability_delta', {})
        row['unlabelled_probability_delta_mean'] = deltas.get('unlabelled_positions', {}).get('mean')
        row['unlabelled_probability_increase_fraction'] = deltas.get('unlabelled_positions', {}).get('positive_fraction')
        ranking = record.get('ranking_analysis', {}).get('paired_delta_NBS_minus_backbone', {})
        for k in (10, 50, 100):
            for metric_name, short in (('precision_at_k', 'p'), ('recall_at_k', 'r')):
                row[f'delta_{short}{k}_pp'] = ranking.get(str(k), {}).get(metric_name, {}).get('mean')
        keys = [f'delta_{short}{k}_pp' for k in (10, 50, 100) for short in ('p', 'r')]
        if row['delta_auprc_pp'] is None or any(row[k] is None for k in keys):
            row['interpretation'] = 'insufficient_ranking_diagnostics'
        elif row['delta_auprc_pp'] < 0 or any(row[k] < 0 for k in keys):
            row['interpretation'] = 'ranking_not_fully_recovered_vs_backbone'
        else:
            row['interpretation'] = 'ranking_recovered_check_uncertainty_and_strata'
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.work_root.resolve()
    output = root / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    audits = {}
    for file in sorted((root / 'audit').glob('*/alignment_summary.json')):
        data = read(file)
        audits[file.parent.name] = {key: data.get(key) for key in (
            'cohort_mode', 'anchors_audited', 'anchor_rates', 'anchor_metrics',
            'cross_epoch_repeated_proteins', 'cross_epoch_new_positive_go_after_first',
            'cross_epoch_adjacent_positive_jaccard')}
        mismatches = [b['evidence_mismatched_pairs'] for b in data.get('batch_metrics', []) if b.get('evidence_mismatched_pairs') is not None]
        audits[file.parent.name]['evidence_mismatched_pairs'] = sum(mismatches) if mismatches else None
    smokes = {d.name: analyze_smoke(d) for d in sorted((root / 'train').glob('*_smoke20')) if d.is_dir()}
    rows = []
    for part in ('baseline', 'eval'):
        for file in sorted((root / part).glob('*/nbs_epoch_series_summary.json')):
            rows.extend(evaluation_rows(read(file), f'{part}/{file.parent.name}'))
    result = {'schema_version': 1, 'audits': audits, 'smokes': smokes, 'evaluations': rows,
              'decision_order': ['input/evidence contract', 'per-protein positive and hard coverage',
                                 'finite nonzero gradient chain', 'AUPRC and top-k vs backbone',
                                 'candidate/non-candidate and rare strata', 'neighborhood/source ablations'],
              'limitations': 'No best checkpoint selection. Independent-test-guided repeated changes make this set diagnostic; confirm final claims on an untouched evaluation.'}
    (output / 'experiment_analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    if rows:
        with (output / 'experiment_comparison.tsv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({'audits': list(audits), 'smokes': {k: v['status'] for k, v in smokes.items()}, 'evaluations': len(rows)}, indent=2))
    print(output / 'experiment_analysis.json')


if __name__ == '__main__':
    main()
