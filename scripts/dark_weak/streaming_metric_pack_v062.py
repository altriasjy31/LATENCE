"""Bounded-memory equivalents of the complete Atlas classification metrics.

All supplied proteins and original GO columns are evaluated. Dense gold and
probability cohort copies, flattened argsorts, and threshold-by-cohort tensors
are never built. Grid statistics use per-row histograms. Exact AP, PR-AUC and
micro Fmax use complete tied positive-score groups and two background counts
(scores strictly above / at least each positive score). Negative-only score
groups contribute zero recall increment, but their precision immediately before
the next positive group is retained in the trapezoidal area. Thus neither AP
nor PR-AUC is histogram-approximated.

Native v085, original Stage-1 histogram, and original Stage-1 official metrics
have distinct threshold arithmetic and averaging rules. Those distinctions are
preserved deliberately and covered by dense-reference regression tests.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np


_REFERENCE_FUNCTION_HASHES = {
    'native_kernel': {
        '_thresholds': 'b2e066ccca1526ea58038bc6b10a0ed9f846349873d1e333c0ed2e9d2c9a2196',
        '_protein_curve': 'f59a77782c54533c0800b4abcb23b77d0987e377881c1ae565c8e228aff5aa4d',
        '_exact_micro': 'b569558ba5b2363c3482a7c5bed7ff6fdcb2805cf72e8b9e06c998d073da3584',
        'compute_standard_metrics': '8a93186aeaf16874a07c32d488bf36486f989d8dd14fa50a8e20bdf6e513f29e',
    },
    'stage1_hist': {
        'fmax_from_histograms': '85d5b9119a820615153f5a9f076ecea056aab612f94d3db4b0295b1f6fb2e0bf',
        'compute_metric_pack': '55a80350c229edd2f1e1cafa940d66a3948913bfcd9a7dbfb014cb081d671e6e',
    },
    'stage1_official': {
        'evalperf_torch': 'cba3f6ef6ab2c038ccb90291148d64920213365bace5fb28816b2dedacde2685',
    },
}


def _normalized_function_ast_sha256(node):
    if node is None:
        return None
    # Python 3.12 added an empty type_params field to function definitions.
    # Ignore this absent semantic content so identical 3.10+ code hashes alike.
    for child in ast.walk(node):
        if 'type_params' in child._fields and not getattr(child, 'type_params', None):
            child._fields = tuple(key for key in child._fields if key != 'type_params')
    return hashlib.sha256(ast.dump(node, annotate_fields=True,
                                   include_attributes=False).encode()).hexdigest()


def verify_reference_contract(kernels):
    """Fail clearly if a numerical reference function has changed.

    AST hashes ignore source formatting, comments, and unrelated functions.
    Call once after loading the original kernels; record this returned audit
    alongside the streaming implementation SHA rather than claiming that the
    original dense functions themselves executed during a streaming run.
    """
    sources = kernels.get('sources', {})
    checked = {}
    for kind, expected in _REFERENCE_FUNCTION_HASHES.items():
        if kind == 'stage1_official' and kernels.get('official') is None:
            continue
        source = sources.get(kind)
        if not source or not source.get('path'):
            raise RuntimeError(f'Streaming metric reference provenance missing: {kind}')
        path = Path(source['path'])
        parsed = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        functions = {node.name: node for node in parsed.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        observed = {}
        for name, digest in expected.items():
            node = functions.get(name)
            actual = _normalized_function_ast_sha256(node)
            if actual != digest:
                raise RuntimeError(f'Streaming metric reference contract changed: {kind}/{name} '
                    f'in {path}. Review the numerical change before evaluating; no approximate fallback is used.')
            observed[name] = actual
        checked[kind] = {'path': str(path.resolve()), 'normalized_function_ast_sha256': observed}
    return {'checked': True, 'reference_functions': checked,
            'streaming_implementation_sha256': implementation_sha256(),
            'implementation': 'bounded-memory numerical equivalents; original dense functions not called'}


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _curve_f1(precision, recall):
    return np.divide(2 * precision * recall, precision + recall,
                     out=np.zeros_like(precision), where=(precision + recall) > 0)


def _tail(hist):
    return np.cumsum(hist[::-1], dtype=np.int64)[::-1]


def _strict_counts(scores, positive_scores, thresholds):
    """Count strict score>threshold for the precise supplied grid dtype."""
    n = len(thresholds)
    bins = np.searchsorted(thresholds, scores, side='left')
    pos_bins = np.searchsorted(thresholds, positive_scores, side='left')
    predicted = len(scores) - np.cumsum(np.bincount(bins, minlength=n + 1), dtype=np.int64)[:n]
    true = len(positive_scores) - np.cumsum(np.bincount(pos_bins, minlength=n + 1), dtype=np.int64)[:n]
    return predicted, true


def _exact_positive_groups(raw_scores, row_indices, score_groups, group_positive,
                           *, chunk_rows):
    """Exact pooled ranking with memory bounded by chunk and positive labels."""
    total_positive = int(group_positive.sum())
    if not total_positive:
        return dict(standard_micro_ap=0., standard_micro_pr_auc=0.,
                    standard_micro_fmax_exact=0., standard_micro_fmax_exact_threshold=1.,
                    standard_micro_fmax_exact_min_included_score=1.)
    count = len(score_groups)
    # For a score p, left=#positive-score-groups<p and right=#groups<=p.
    # Prefix counts then give background <=s and <s, respectively.
    left_hist = np.zeros(count + 1, dtype=np.int64)
    right_hist = np.zeros(count + 1, dtype=np.int64)
    total_cells = len(row_indices) * raw_scores.shape[1]
    for start in range(0, len(row_indices), chunk_rows):
        rows = row_indices[start:start + chunk_rows]
        values = np.asarray(raw_scores[rows], dtype=np.float32).reshape(-1)
        left_hist += np.bincount(np.searchsorted(score_groups, values, side='left'), minlength=count + 1)
        right_hist += np.bincount(np.searchsorted(score_groups, values, side='right'), minlength=count + 1)
    greater = total_cells - np.cumsum(left_hist, dtype=np.int64)[:count]
    greater_equal = total_cells - np.cumsum(right_hist, dtype=np.int64)[:count]
    tp_above = total_positive - np.cumsum(group_positive, dtype=np.int64)
    tp_including = tp_above + group_positive
    precision_after = tp_including / greater_equal
    precision_before = np.divide(tp_above, greater, out=np.ones(count), where=greater > 0)
    recall_increment = group_positive / total_positive
    ap = float(np.sum(recall_increment * precision_after))
    area = float(np.sum(recall_increment * (precision_before + precision_after) / 2))
    # Descending score order matches the original earliest-maximizer tie rule.
    f1 = (2 * tp_including / (total_positive + greater_equal))[::-1]
    best_desc = int(np.argmax(f1))
    best = count - 1 - best_desc
    score = np.float32(score_groups[best])
    threshold = float(np.nextafter(score, np.float32(-np.inf)))
    return dict(standard_micro_ap=100 * ap, standard_micro_pr_auc=100 * area,
                standard_micro_fmax_exact=100 * float(f1[best_desc]),
                standard_micro_fmax_exact_threshold=threshold,
                standard_micro_fmax_exact_min_included_score=float(score))


def comprehensive_metrics_streaming(raw_scores, row_indices, positive_columns, *,
                                    kernels, chunk_rows=32, scratch_dir=None,
                                    threshold_step=.01):
    """Score a sparse-gold cohort against every original probability column.

    ``positive_columns[i]`` lists the gold GO columns for selected row i.
    ``raw_scores`` may be an np.load(..., mmap_mode='r') array. ``scratch_dir``
    is accepted for API consistency; no disk spill is necessary. Numerical
    metrics preserve the v054 complete pack's default unfiltered semantics.
    """
    del scratch_dir
    scores = np.asarray(raw_scores)
    indices = np.asarray(row_indices)
    if scores.ndim != 2 or scores.dtype.kind not in 'biuf':
        raise ValueError('raw_scores must be a two-dimensional real numeric array')
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise ValueError('row_indices must be a one-dimensional integer array')
    indices = indices.astype(np.int64, copy=False)
    if len(indices) != len(positive_columns):
        raise ValueError('row_indices and positive_columns differ in length')
    if len(set(indices.tolist())) != len(indices):
        raise ValueError('Selected protein rows must be unique')
    if indices.size and (indices.min() < 0 or indices.max() >= scores.shape[0]):
        raise ValueError('Selected protein row is outside raw_scores')
    if not isinstance(chunk_rows, (int, np.integer)) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    step = float(threshold_step)
    if (not np.isfinite(step) or not 0 < step <= 1 or
            not np.isclose(round(1 / step) * step, 1, rtol=0, atol=1e-12)):
        raise ValueError('threshold_step must divide 1 exactly')
    nrows, ncols = len(indices), scores.shape[1]
    if not nrows or not ncols:
        raise ValueError('Streaming classification requires a nonempty protein and GO axis')
    positive = []
    for values in positive_columns:
        values = np.asarray(list(values))
        if values.size and (values.ndim != 1 or values.dtype.kind not in 'iu'):
            raise ValueError('Gold GO columns must be one-dimensional integer indices')
        values = np.unique(values.astype(np.int64, copy=False))
        if values.size and (values.min() < 0 or values.max() >= ncols):
            raise ValueError('Gold GO column is outside raw_scores')
        positive.append(values)

    # Native thresholds are strict float64; original official Torch thresholds
    # are strict float32 scalar comparisons. Histogram Stage1 uses floor bins.
    native_thresholds = np.arange(1001, dtype=np.float64) / 1000
    official_thresholds = np.arange(101, dtype=np.float64) / 100
    official_comparison = official_thresholds.astype(np.float32)
    nbins = int(round(1 / step)) + 1
    hist_thresholds = np.arange(nbins, dtype=np.float32) * step
    hist_thresholds[-1] = 1.
    native_precision_sum = np.zeros(1001, dtype=np.float64)
    native_precision_rows = np.zeros(1001, dtype=np.int64)
    native_recall_sum = np.zeros(1001, dtype=np.float64)
    hist_pred_total = np.zeros(nbins, dtype=np.int64)
    hist_tp_total = np.zeros(nbins, dtype=np.int64)
    hist_precision_sum = np.zeros(nbins, dtype=np.float64)
    hist_precision_rows = np.zeros(nbins, dtype=np.int64)
    hist_recall_sum = np.zeros(nbins, dtype=np.float64)
    official_pred_total = np.zeros(101, dtype=np.int64)
    official_tp_total = np.zeros(101, dtype=np.int64)
    positive_scores = []
    total_positive = sum(len(cols) for cols in positive)
    eligible = sum(bool(len(cols)) for cols in positive)
    zero, one, positive_zero, equal_rows = 0, 0, 0, 0
    minimum, maximum, first_value, all_equal = np.inf, -np.inf, None, True
    for start in range(0, nrows, chunk_rows):
        block = np.asarray(scores[indices[start:start + chunk_rows]], dtype=np.float32)
        if not np.isfinite(block).all() or np.any((block < 0) | (block > 1)):
            raise ValueError('Probabilities must be finite in [0,1] after float32 conversion')
        zero += int(np.count_nonzero(block == 0))
        one += int(np.count_nonzero(block == 1))
        minimum = min(minimum, float(block.min()))
        maximum = max(maximum, float(block.max()))
        if first_value is None:
            first_value = block.flat[0]
        all_equal = all_equal and bool(np.all(block == first_value))
        equal_rows += int(np.count_nonzero(np.all(block == block[:, :1], axis=1)))
        for local, row in enumerate(block):
            cols = positive[start + local]
            pos = row[cols]
            positive_scores.append(pos)
            positive_zero += int(np.count_nonzero(pos == 0))
            if len(cols):
                predicted, true = _strict_counts(row, pos, native_thresholds)
                active = predicted > 0
                native_precision_sum[active] += true[active] / predicted[active]
                native_precision_rows += active
                native_recall_sum += true / len(cols)
            bins = np.clip(np.floor(row / step).astype(np.int32), 0, nbins - 1)
            predicted = _tail(np.bincount(bins, minlength=nbins))
            true = _tail(np.bincount(bins[cols], minlength=nbins))
            hist_pred_total += predicted
            hist_tp_total += true
            active = predicted > 0
            hist_precision_sum[active] += true[active] / predicted[active]
            hist_precision_rows += active
            if len(cols):
                hist_recall_sum += true / len(cols)
            if kernels.get('official') is not None:
                predicted, true = _strict_counts(row, pos, official_comparison)
                official_pred_total += predicted
                official_tp_total += true

    all_positive_scores = np.concatenate(positive_scores)
    groups, group_positive = np.unique(all_positive_scores, return_counts=True)
    exact = _exact_positive_groups(scores, indices, groups, group_positive,
                                   chunk_rows=chunk_rows)
    native_precision = np.divide(native_precision_sum, native_precision_rows,
                                 out=np.zeros(1001), where=native_precision_rows > 0)
    native_recall = native_recall_sum / eligible if eligible else native_recall_sum
    native_f1 = _curve_f1(native_precision, native_recall)
    native_best = int(np.argmax(native_f1))
    coarse = native_f1[::10]
    coarse_best = int(np.argmax(coarse))
    out = dict(standard_num_proteins=nrows, standard_num_go=ncols,
        standard_num_positive_labels=total_positive,
        standard_num_protein_fmax_eligible=eligible,
        standard_num_empty_gold_proteins=nrows - eligible,
        standard_protein_fmax=100 * float(native_f1[native_best]),
        standard_protein_fmax_threshold=float(native_thresholds[native_best]),
        standard_protein_precision_at_fmax=100 * float(native_precision[native_best]),
        standard_protein_recall_at_fmax=100 * float(native_recall[native_best]),
        standard_protein_fmax_threshold_step=.001,
        standard_protein_fmax_grid_0p01=100 * float(coarse[coarse_best]),
        standard_protein_fmax_grid_0p01_threshold=float(official_thresholds[coarse_best]),
        **exact)
    out.update(Fmax_percent=out['standard_protein_fmax'],
        AuPRC_percent=out['standard_micro_pr_auc'], AP_percent=out['standard_micro_ap'],
        AuPRC_native_key='standard_micro_pr_auc', AP_native_key='standard_micro_ap',
        observed_positive_prevalence_percent=100 * total_positive / (nrows * ncols),
        evaluation_backend='v062_bounded_memory_equivalent', standard_threshold_step_used=.001,
        score_zero_fraction=zero / (nrows * ncols), score_one_fraction=one / (nrows * ncols),
        score_all_scores_equal=bool(all_equal), score_per_row_all_equal_fraction=equal_rows / nrows,
        score_positive_zero_fraction=positive_zero / total_positive if total_positive else None,
        score_min=minimum, score_max=maximum, metric_has_positive_labels=bool(total_positive),
        metric_empty_axis=False, stage1_hist__threshold_step=step,
        stage1_hist__num_proteins=nrows, stage1_hist__num_go=ncols,
        stage1_hist__num_positives=total_positive,
        metric_stream_chunk_rows=int(chunk_rows), metric_exact_background_passes=2,
        metric_stream_implementation_sha256=implementation_sha256())

    hist_precision = hist_tp_total / np.maximum(hist_pred_total, 1)
    hist_recall = hist_tp_total / max(total_positive, 1)
    hist_f1 = 2 * hist_precision * hist_recall / np.maximum(hist_precision + hist_recall, 1e-12)
    hist_best = int(np.nanargmax(hist_f1))
    rec_desc, prec_desc = hist_recall[::-1], hist_precision[::-1]
    hist_ap = float(np.sum(np.diff(np.r_[0., rec_desc]) * prec_desc))
    sample_precision = hist_precision_sum / np.maximum(hist_precision_rows, 1)
    sample_recall = hist_recall_sum / max(eligible, 1)
    sample_f1 = 2 * sample_precision * sample_recall / np.maximum(sample_precision + sample_recall, 1e-12)
    sample_best = int(np.nanargmax(sample_f1))
    out.update(stage1_hist__fmax_micro=100 * float(hist_f1[hist_best]),
        stage1_hist__threshold_micro=float(hist_thresholds[hist_best]),
        stage1_hist__precision_micro=100 * float(hist_precision[hist_best]),
        stage1_hist__recall_micro=100 * float(hist_recall[hist_best]),
        stage1_hist__auprc_micro_hist=100 * hist_ap,
        stage1_hist__auprc_micro_exact=exact['standard_micro_ap'] if total_positive else None,
        stage1_hist__fmax_sample=100 * float(sample_f1[sample_best]),
        stage1_hist__threshold_sample=float(hist_thresholds[sample_best]),
        stage1_hist__precision_sample=100 * float(sample_precision[sample_best]),
        stage1_hist__recall_sample=100 * float(sample_recall[sample_best]))
    if kernels.get('official') is not None:
        # Only tiny 101-point tensors are used. This retains the original
        # Torch unstable tie ordering and float32 trapezoidal accumulation.
        import torch
        precision = official_tp_total / np.maximum(official_pred_total, 1)
        recall = official_tp_total / max(total_positive, 1)
        f1 = np.asarray([100 * 2 * float(a) * float(b) / (float(a) + float(b) or 1.)
                         for a, b in zip(precision, recall)])
        best = int(np.argmax(f1))
        prs = torch.from_numpy(precision.astype(np.float32))
        rcs = torch.from_numpy(recall.astype(np.float32))
        ordering = torch.argsort(rcs)
        auc = float((torch.trapz(prs[ordering], rcs[ordering]) * 100).item())
        out.update(stage1_official__fmax=float(f1[best]),
            stage1_official__threshold=float(official_thresholds[best]),
            stage1_official__auprc=auc)
    return out
