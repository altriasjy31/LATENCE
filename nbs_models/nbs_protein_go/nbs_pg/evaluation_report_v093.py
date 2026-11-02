"""Exact overall/rare/Stage2-unseen metrics for the v093 prediction artifact."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'nbs-v093-unified-evaluation-v1'
BINS = ((0, 0), (1, 5), (1, 9), (10, 99), (100, 999), (1000, None))


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def per_go_average_precision(labels, probabilities, block_columns=128):
    """Return percent AP; no-positive columns are NaN, score ties stay grouped."""
    y, p = np.asarray(labels, bool), np.asarray(probabilities)
    if y.ndim != 2 or y.shape != p.shape or block_columns < 1:
        raise ValueError('AP requires matching protein-by-GO matrices')
    result = np.full(y.shape[1], np.nan, np.float64)
    eligible = np.flatnonzero(y.any(0))
    for start in range(0, len(eligible), block_columns):
        columns = eligible[start:start+block_columns]
        scores = np.asarray(p[:, columns], np.float64)
        order = np.argsort(-scores, axis=0, kind='stable')
        sorted_scores = np.take_along_axis(scores, order, axis=0)
        sorted_labels = np.take_along_axis(y[:, columns], order, axis=0)
        positives = sorted_labels.cumsum(0)
        for local, column in enumerate(columns):
            ends = np.r_[np.flatnonzero(sorted_scores[:-1, local] != sorted_scores[1:, local]), len(y)-1]
            cumulative = positives[ends, local]
            gains = np.diff(np.r_[0, cumulative])
            result[column] = 100 * np.sum(gains * cumulative / (ends+1)) / cumulative[-1]
    return result


def summarize_subset(labels, probabilities, mask, metrics, per_go_ap):
    y, p = np.asarray(labels, bool), np.asarray(probabilities)
    selected = np.asarray(mask, bool)
    if selected.shape != (y.shape[1],):
        raise ValueError('Evaluation subset must address immutable GO columns')
    positives = int(y[:, selected].sum())
    result = {'GO_columns': int(selected.sum()), 'test_positive_pairs': positives,
              'GO_columns_with_test_positive': int(np.isfinite(per_go_ap[selected]).sum())}
    if not selected.any() or positives == 0:
        result.update(fmax=None, threshold=None, precision=None, recall=None,
                      micro_ap=None, macro_ap=None,
                      reason='No positive test labels in this fixed subset')
        return result
    sy, sp = metrics._validate(y[:, selected], p[:, selected])
    thresholds = metrics._thresholds(.001)
    f1, precision, recall, eligible = metrics._protein_curve(sy, sp, thresholds)
    best = int(np.argmax(f1))
    exact = metrics._exact_micro(sy, sp)
    result.update(fmax=100*float(f1[best]), threshold=float(thresholds[best]),
                  precision=100*float(precision[best]), recall=100*float(recall[best]),
                  micro_ap=float(exact['standard_micro_ap']),
                  macro_ap=float(np.nanmean(per_go_ap[selected])),
                  protein_fmax_eligible=int(eligible))
    return result


def decision_counts(labels, probabilities, mask, threshold):
    """Pair counts at one fixed global threshold, including no-positive bins."""
    y, p = np.asarray(labels, bool), np.asarray(probabilities)
    selected = np.asarray(mask, bool)
    if selected.shape != (y.shape[1],) or p.shape != y.shape:
        raise ValueError('Decision-count axes differ from the fixed GO subset')
    positives = int(y[:, selected].sum())
    if threshold is None:
        return {'threshold': None, 'tp': None, 'predicted_pairs': None,
                'test_positive_pairs': positives, 'pair_precision': None, 'pair_recall': None}
    predicted = p[:, selected] > float(threshold)
    tp = int((predicted & y[:, selected]).sum())
    count = int(predicted.sum())
    return {'threshold': float(threshold), 'tp': tp, 'true_positive_pairs': tp,
            'predicted_pairs': count, 'test_positive_pairs': positives,
            'pair_precision': 100*tp/count if count else None,
            'pair_recall': 100*tp/positives if positives else None}


def clipped_backbone_reference(probabilities, probability_clip):
    clip = float(probability_clip)
    if not np.isfinite(clip) or not 0 < clip < .5:
        raise ValueError('Loaded-backbone clipping must be finite within (0,.5)')
    values = np.asarray(probabilities, np.float32)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Backbone reference must contain probabilities in [0,1]')
    return np.clip(values, clip, 1-clip)


def summarize_predictions(labels, methods, counts, seen_task_mask, metrics):
    y = np.asarray(labels)
    if y.ndim != 2 or not len(y) or not ((y == 0) | (y == 1)).all():
        raise ValueError('Evaluation labels must have nonempty binary protein-by-GO shape')
    y = y.astype(bool, copy=False)
    counts, seen = np.asarray(counts), np.asarray(seen_task_mask)
    if (counts.shape != (y.shape[1],) or counts.dtype.kind not in 'iu'
            or (counts < 0).any() or seen.shape != counts.shape or seen.dtype != np.bool_):
        raise ValueError('Frequency and seen-mask axes must match immutable GO columns')
    if not methods:
        raise ValueError('At least one prediction method is required')
    result = {'num_proteins': len(y), 'num_GO_columns': y.shape[1],
              'test_positive_pairs': int(y.sum()), 'methods': {},
              'frequency_strata': [], 'supervision_subsets': {}}
    per_go = {}
    for name, values in methods.items():
        _, probabilities = metrics._validate(y, values)
        per_go[name] = per_go_average_precision(y, probabilities)
        result['methods'][name] = summarize_subset(y, probabilities, np.ones(y.shape[1], bool), metrics, per_go[name])
    backbone_threshold = result['methods'].get('backbone_base', {}).get('threshold')

    def subset_at_working_points(name, values, mask):
        subset = summarize_subset(y, values, mask, metrics, per_go[name])
        subset['at_overall_fmax'] = decision_counts(y, values, mask, result['methods'][name]['threshold'])
        subset['at_backbone_overall_threshold'] = decision_counts(y, values, mask, backbone_threshold)
        return subset

    for name, values in methods.items():
        overall = result['methods'][name]
        overall['at_overall_fmax'] = decision_counts(y, values, np.ones(y.shape[1], bool), overall['threshold'])
        overall['at_backbone_overall_threshold'] = decision_counts(y, values, np.ones(y.shape[1], bool), backbone_threshold)
    for low, high in BINS:
        mask = (counts >= low) & (True if high is None else counts <= high)
        result['frequency_strata'].append({
            'train_core_count_min': low, 'train_core_count_max': high,
            'GO_columns': int(mask.sum()),
            'methods': {name: subset_at_working_points(name, values, mask)
                        for name, values in methods.items()}})
    for name, mask in [('seen', seen), ('unseen', ~seen)]:
        result['supervision_subsets'][name] = {
            'GO_columns': int(mask.sum()),
            'methods': {method: subset_at_working_points(method, values, mask)
                        for method, values in methods.items()}}
    result.update(score_unit='percent; thresholds in probability units',
                  AP_definition='exact non-interpolated average precision with grouped ties',
                  macro_AP_definition='mean per-GO AP over fixed test-positive GO columns',
                  frequency_definition='training core gold membership; original protein holdout excluded; raw task columns retained',
                  frequency_bins_overlap=True,
                  frequency_strata_note='1-5 is nested in 1-9; these rows must not be added',
                  threshold_selection='descriptive subset Fmax; not deployment calibration or training label selection',
                  working_point_definition='at_overall_fmax uses each method overall threshold; at_backbone_overall_threshold uses raw backbone overall threshold; all decisions use strict >',
                  unseen_scope='Stage2 supervision holdout/natural-zero only; Stage1 exposure not excluded',
                  strict_pipeline_zero_shot_claimed=False)
    return result


def write_extended_report(args, config, data, *, saved=None):
    """Read the completed v093 export; never run another GNN forward."""
    if saved is None:
        import torch
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    checkpoint = Path(args.checkpoint).resolve()
    folder = Path(args.work_dir).resolve()/f"eval_step{saved['step']}"/'full'
    probability_path = folder/'nbs_ind_test_prob.f32.npy'
    manifest_path = folder/'nbs_full_task_prediction_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    input_dir = Path(args.input_dir).resolve()
    checkpoint_sha, probability_sha = _sha256(checkpoint), _sha256(probability_path)
    input_sha = _sha256(input_dir/'ind_test_input_manifest.json')
    if (manifest.get('checkpoint_sha256') != checkpoint_sha
            or manifest.get('output_probability_sha256') != probability_sha
            or manifest.get('input_manifest_sha256') != input_sha
            or manifest.get('step') != saved['step']):
        raise ValueError('Completed v093 prediction identity differs from report inputs')
    helper_path = ROOT/'scripts/nbs/diagnose_nbs_knn_v090.py'
    metrics_path = ROOT/'nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py'
    helpers = _module(helper_path, '_nbs_v093_report_inputs')
    metrics = _module(metrics_path, '_nbs_v093_original_metric_kernels')
    test = helpers.load_test_inputs(input_dir, Path(args.metadata_file).resolve(), config['task'],
                                   data._signature['representation_manifest_sha256'])
    # v092's frequency reader explicitly excludes core protein holdout and
    # counts original gold memberships, including held-out GO labels for bins.
    from .evaluation_report_v092 import core_counts
    counts = core_counts(data)
    probabilities = np.load(probability_path, mmap_mode='r', allow_pickle=False)
    clip = float(data.stores.episode_sampler.base_logit_store.probability_clip)
    loaded_backbone = clipped_backbone_reference(test['backbone'], clip)
    report = summarize_predictions(test['labels'], {'NBS_final': probabilities, 'backbone_base': test['backbone'],
                                                   'backbone_loaded_clip': loaded_backbone},
                                   counts, np.asarray(data.seen_task_mask, bool), metrics)
    report.update(schema=SCHEMA, complete=True, task=config['task'],
                  checkpoint_epoch=saved['epoch'], checkpoint_step=saved['step'],
                  checkpoint_sha256=checkpoint_sha, probability_sha256=probability_sha,
                  input_manifest_sha256=input_sha, prediction_manifest_sha256=_sha256(manifest_path),
                  source_prediction=manifest,
                  loaded_backbone_probability_clip=clip,
                  loaded_backbone_reference_definition='clip(raw backbone probability, loader clip, 1-loader clip); float32; exact logit/sigmoid round-trip may add floating-point differences',
                  num_training_core_proteins=len(data.core_ids),
                  core_frequency_sha256=hashlib.sha256(np.asarray(counts, dtype='<i8').tobytes()).hexdigest(),
                  seen_task_mask_sha256=hashlib.sha256(np.asarray(data.seen_task_mask, bool).tobytes()).hexdigest(),
                  metric_kernel_sha256=_sha256(metrics_path))
    output = folder/'metrics_v093'/'summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=output.parent, delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'[v093 metrics] {output}', flush=True)
    return output
