"""Explicit supplementary metrics; never replace the historical Stage-1 fields.

Scores are percentages, thresholds are probabilities. Protein Fmax excludes
rows without gold annotations, averages precision over eligible rows with at
least one prediction, and averages recall over *all* eligible rows. Every task
GO column is retained. No GO propagation or calibration is performed here.

The grid uses strict ``score > threshold`` with thresholds represented in
float64. Micro metrics include every row (also empty-gold rows). The exact
micro curve groups equal scores before computing a point, so ties cannot be
split to improve the result. The exact threshold may be just below zero to
include zero-score predictions; protein-grid thresholds remain in [0, 1].

Memory: protein curves use per-row histograms, not a threshold x protein x GO
tensor. Exact micro metrics require one intp argsort of the flattened scores;
all other ranking work is in bounded chunks, rather than materializing several
full sorted/cumulative arrays (about 304 MB of indices for 38 million pairs on
a 64-bit platform, in addition to the caller's inputs).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

METRIC_SCHEMA = "0.8.5-standard-metrics-v1"
_RANK_CHUNK = 1_000_000


def metrics_implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def metric_definitions() -> dict:
    return {
        "schema": METRIC_SCHEMA,
        "score_unit": "percent; thresholds remain in probability units",
        "protein_fmax": (
            "Harmonic mean of mean per-protein precision and mean per-protein "
            "recall, maximized on the stated uniform [0,1] grid; strict >. "
            "Exclude empty-gold proteins from both means; precision includes "
            "only eligible proteins with predictions, recall includes all "
            "eligible proteins. Earliest threshold wins an exact F tie."
        ),
        "micro_ap": "Exact non-interpolated AP over all protein-GO pairs; equal scores grouped.",
        "micro_pr_auc": (
            "Exact trapezoidal area over the empirical micro PR curve, including "
            "the precision=1, recall=0 endpoint; equal scores grouped."
        ),
        "micro_fmax_exact": (
            "Best micro F1 among complete equal-score groups. Strict > threshold "
            "is next representable score below the minimum included score; "
            "can be negative when zero-score pairs are included."
        ),
        "no_positive_labels": "All standardized scores are defined as zero if no positive gold exists.",
        "task_columns": "All supplied columns; no zero-positive-class filtering or ontology propagation.",
        "legacy_fields": "Historical Stage-1 metric fields are preserved and are not overwritten.",
    }


def _validate(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.ndim != 2 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be equally shaped 2-D arrays")
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("metric arrays must have at least one protein and GO column")
    if labels.dtype.kind not in "biuf" or probabilities.dtype.kind not in "biuf":
        raise ValueError("metric arrays must contain real numeric values")
    # Bound temporary masks, including for memory-mapped full-task predictions.
    for start in range(0, labels.shape[0], 16):
        y, p = labels[start:start + 16], probabilities[start:start + 16]
        if np.any((y != 0) & (y != 1)):
            raise ValueError("gold labels must be binary 0/1")
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must be finite and lie in [0, 1]")
    return labels, probabilities


def _thresholds(step: float) -> np.ndarray:
    if not np.isfinite(step) or not 0 < step <= 1:
        raise ValueError("threshold_step must be positive and divide 1 exactly")
    intervals = int(round(1 / step))
    if intervals > 100_000 or not np.isclose(intervals * step, 1, rtol=0, atol=1e-12):
        raise ValueError("threshold_step must divide 1 exactly (at most 100000 intervals)")
    return np.arange(intervals + 1, dtype=np.float64) / intervals


def _protein_curve(labels: np.ndarray, probabilities: np.ndarray,
                   thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    count = len(thresholds)
    precision_sum = np.zeros(count, dtype=np.float64)
    precision_rows = np.zeros(count, dtype=np.int64)
    recall_sum = np.zeros(count, dtype=np.float64)
    eligible = 0
    for row_y, row_p in zip(labels, probabilities):
        positive = int(row_y.sum())
        if not positive:
            continue
        eligible += 1
        # b is the number of thresholds strictly below each score. A score
        # contributes to indices 0..b-1; scores equal to threshold are excluded.
        bins = np.searchsorted(thresholds, row_p, side="left")
        predicted_hist = np.bincount(bins, minlength=count + 1)
        true_hist = np.bincount(bins[row_y != 0], minlength=count + 1)
        predicted = len(row_p) - np.cumsum(predicted_hist, dtype=np.int64)[:count]
        true_positive = positive - np.cumsum(true_hist, dtype=np.int64)[:count]
        active = predicted > 0
        precision_sum[active] += true_positive[active] / predicted[active]
        precision_rows += active
        recall_sum += true_positive / positive
    precision = np.divide(precision_sum, precision_rows, out=np.zeros(count), where=precision_rows > 0)
    recall = recall_sum / eligible if eligible else recall_sum
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(count), where=(precision + recall) > 0)
    return f1, precision, recall, eligible


def _exact_micro(labels: np.ndarray, probabilities: np.ndarray,
                 rank_chunk: int = _RANK_CHUNK) -> dict[str, float]:
    y, p = labels.reshape(-1), probabilities.reshape(-1)
    total_positive = int(y.sum())
    empty = {
        "standard_micro_ap": 0., "standard_micro_pr_auc": 0.,
        "standard_micro_fmax_exact": 0., "standard_micro_fmax_exact_threshold": 1.,
        "standard_micro_fmax_exact_min_included_score": 1.,
    }
    if not total_positive:
        return empty
    if rank_chunk < 1:
        raise ValueError("rank_chunk must be positive")
    order = np.argsort(p, kind="quicksort")[::-1]
    accumulated_tp = 0
    last_recall, last_precision = 0., 1.
    ap, area, best_f1, best_score = 0., 0., 0., 1.
    for start in range(0, p.size, rank_chunk):
        stop = min(p.size, start + rank_chunk)
        indices = order[start:stop]
        scores = p[indices]
        tp = np.cumsum(y[indices], dtype=np.int64) + accumulated_tp
        accumulated_tp = int(tp[-1])
        final_group = stop == p.size or scores[-1] != p[order[stop]]
        group_end = np.flatnonzero(np.r_[scores[:-1] != scores[1:], final_group])
        if not group_end.size:
            continue
        recall = tp[group_end] / total_positive
        precision = tp[group_end] / (start + group_end + 1)
        previous_recall = np.r_[last_recall, recall[:-1]]
        previous_precision = np.r_[last_precision, precision[:-1]]
        increase = recall - previous_recall
        ap += float(np.sum(increase * precision))
        area += float(np.sum(increase * (precision + previous_precision) / 2))
        f1 = 2 * tp[group_end] / (total_positive + start + group_end + 1)
        best = int(np.argmax(f1))
        if f1[best] > best_f1:
            best_f1, best_score = float(f1[best]), float(scores[group_end[best]])
        last_recall, last_precision = float(recall[-1]), float(precision[-1])
    score_dtype = p.dtype if p.dtype.kind == "f" else np.dtype("float64")
    threshold = float(np.nextafter(np.asarray(best_score, dtype=score_dtype),
                                   np.asarray(-np.inf, dtype=score_dtype)))
    return {
        "standard_micro_ap": 100 * ap,
        "standard_micro_pr_auc": 100 * area,
        "standard_micro_fmax_exact": 100 * best_f1,
        "standard_micro_fmax_exact_threshold": threshold,
        "standard_micro_fmax_exact_min_included_score": best_score,
    }


def compute_standard_metrics(labels: np.ndarray, probabilities: np.ndarray,
                             threshold_step: float = .001) -> dict[str, float | int]:
    """Return supplementary metric values; no legacy names are reused.

    The default protein grid has 1001 thresholds. A 101-point 0.01 grid is also
    reported for a direct grid-resolution check. Exact AP and exact PR AUC
    depend only on score ordering/ties, not on the protein threshold grid.
    """
    y, p = _validate(labels, probabilities)
    thresholds = _thresholds(float(threshold_step))
    f1, precision, recall, eligible = _protein_curve(y, p, thresholds)
    best = int(np.argmax(f1))
    coarse = _thresholds(.01)
    # Reuse exactly matching fine-grid points only when the ratio is integral.
    ratio = .01 / threshold_step
    if ratio >= 1 and np.isclose(ratio, round(ratio), rtol=0, atol=1e-10):
        coarse_f1 = f1[::int(round(ratio))]
    else:
        coarse_f1 = _protein_curve(y, p, coarse)[0]
    coarse_best = int(np.argmax(coarse_f1))
    output = {
        "standard_num_proteins": int(y.shape[0]),
        "standard_num_go": int(y.shape[1]),
        "standard_num_positive_labels": int(y.sum()),
        "standard_num_protein_fmax_eligible": eligible,
        "standard_num_empty_gold_proteins": int(y.shape[0]) - eligible,
        "standard_protein_fmax": 100 * float(f1[best]),
        "standard_protein_fmax_threshold": float(thresholds[best]),
        "standard_protein_precision_at_fmax": 100 * float(precision[best]),
        "standard_protein_recall_at_fmax": 100 * float(recall[best]),
        "standard_protein_fmax_threshold_step": float(threshold_step),
        "standard_protein_fmax_grid_0p01": 100 * float(coarse_f1[coarse_best]),
        "standard_protein_fmax_grid_0p01_threshold": float(coarse[coarse_best]),
    }
    output.update(_exact_micro(y, p))
    return output
