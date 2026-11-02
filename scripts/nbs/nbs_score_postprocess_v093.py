"""Label-free score transforms, fixed fusion rules and exact offline metrics.

The caller owns provenance/alignment checks and supplies the original v085
metric module. No model, training labels, fitted calibration or GO propagation
is introduced here. All reported metric values are percentages.
"""
from __future__ import annotations

import math
import numpy as np

_ROWS = 32
_MACRO_COLUMNS = 128
_LOGIT_LOWER = np.nextafter(np.float64(0), np.float64(1))
_LOGIT_UPPER = np.nextafter(np.float64(1), np.float64(0))
_MIX_LOGIT_EPSILON = 1e-6
_BINS = ((0, 0), (1, 5), (1, 9), (10, 99), (100, 999), (1000, None))


def _validate_scores(scores, name):
    array = np.asarray(scores)
    if array.ndim != 2 or min(array.shape) == 0 or array.dtype.kind not in 'biuf':
        raise ValueError(f'{name} must be a nonempty real numeric matrix')
    for start in range(0, len(array), _ROWS):
        block = array[start:start + _ROWS]
        if not np.isfinite(block).all() or np.any((block < 0) | (block > 1)):
            raise ValueError(f'{name} must contain finite probabilities in [0,1]')
    return array


def _validate_counts(counts, width):
    counts = np.asarray(counts)
    if counts.shape != (width,) or counts.dtype.kind not in 'iu' or np.any(counts < 0):
        raise ValueError('counts must be a nonnegative integer vector matching GO columns')
    return counts


def _endpoint_counts(scores):
    zero = one = 0
    for start in range(0, len(scores), _ROWS):
        block = scores[start:start + _ROWS]
        zero += int(np.count_nonzero(block == 0))
        one += int(np.count_nonzero(block == 1))
    return {'zero_scores': zero, 'one_scores': one}


def _stable_logit(probability):
    # Endpoint-only clipping retains all representable interior probabilities.
    p = np.clip(np.asarray(probability, dtype=np.float64), _LOGIT_LOWER, _LOGIT_UPPER)
    return np.log(p) - np.log1p(-p)


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(under='ignore'):
        result = np.exp(-np.abs(values))
    result /= 1 + result
    positive = values >= 0
    result[positive] = 1 - result[positive]
    return result


def _unary(scores, operation):
    result = np.empty(scores.shape, dtype=np.float64)
    for start in range(0, len(scores), _ROWS):
        result[start:start + _ROWS] = operation(scores[start:start + _ROWS])
    return result


def _probability_mix(backbone, graph, alpha):
    result = np.empty(backbone.shape, dtype=np.float64)
    for start in range(0, len(backbone), _ROWS):
        b = np.asarray(backbone[start:start + _ROWS], dtype=np.float64)
        g = np.asarray(graph[start:start + _ROWS], dtype=np.float64)
        result[start:start + _ROWS] = (1 - alpha) * b + alpha * g
    return result


def _logit_mix(backbone, graph, alpha):
    result = np.empty(backbone.shape, dtype=np.float64)
    for start in range(0, len(backbone), _ROWS):
        # Symmetric finite logits are essential for opposite exact endpoints.
        # Nearest-float endpoint clipping is asymmetric around probability .5
        # and would make a 0/1 disagreement arbitrarily favor the zero score.
        b = np.clip(np.asarray(backbone[start:start + _ROWS], dtype=np.float64),
                    _MIX_LOGIT_EPSILON, 1 - _MIX_LOGIT_EPSILON)
        g = np.clip(np.asarray(graph[start:start + _ROWS], dtype=np.float64),
                    _MIX_LOGIT_EPSILON, 1 - _MIX_LOGIT_EPSILON)
        b = np.log(b) - np.log1p(-b)
        g = np.log(g) - np.log1p(-g)
        result[start:start + _ROWS] = _sigmoid((1 - alpha) * b + alpha * g)
    return result


def _mix_clipping_counts(scores):
    low = high = 0
    for start in range(0, len(scores), _ROWS):
        block = np.asarray(scores[start:start + _ROWS], dtype=np.float64)
        low += int(np.count_nonzero(block < _MIX_LOGIT_EPSILON))
        high += int(np.count_nonzero(block > 1 - _MIX_LOGIT_EPSILON))
    return {'below_epsilon': low, 'above_one_minus_epsilon': high,
            'total_clipped': low + high}


def _protected_mix(backbone, graph, counts, cutoff):
    # Preserve both dtype and original stored values for floating inputs.
    dtype = backbone.dtype if backbone.dtype.kind == 'f' else np.float64
    result = np.array(backbone, dtype=dtype, copy=True)
    blend = counts > cutoff
    for start in range(0, len(backbone), _ROWS):
        b = np.asarray(backbone[start:start + _ROWS, blend], dtype=np.float64)
        g = np.asarray(graph[start:start + _ROWS, blend], dtype=np.float64)
        result[start:start + _ROWS, blend] = .5 * b + .5 * g
    return result


def _go_logit_zscore(scores):
    """Streaming population moments over this unlabeled inference cohort."""
    width = scores.shape[1]
    mean = np.zeros(width, dtype=np.float64)
    second = np.zeros(width, dtype=np.float64)
    n = 0
    for start in range(0, len(scores), _ROWS):
        block = _stable_logit(scores[start:start + _ROWS])
        count = len(block)
        block_mean = block.mean(axis=0)
        delta = block_mean - mean
        total = n + count
        second += np.sum((block - block_mean) ** 2, axis=0) + delta ** 2 * n * count / total
        mean += delta * count / total
        n = total
    std = np.sqrt(np.maximum(second / n, 0))
    scale = np.maximum(std, 1e-3)
    result = _unary(scores, lambda p: _sigmoid((_stable_logit(p) - mean) / scale))
    return result, {'statistics_source': 'current_unlabeled_inference_cohort',
                    'cohort_proteins': n, 'population_std_floor': 1e-3,
                    'columns_below_std_floor': int(np.count_nonzero(std < 1e-3)),
                    'transductive_cohort_diagnostic': True,
                    'independent_calibration_set_used': False}


def build_methods(mode, B, G, counts):
    """Yield one method at a time; alpha always denotes the graph weight.

    Inputs are probabilities that have already passed through their model's
    sigmoid. The two originals are yielded without copies; callers must not
    mutate yielded matrices. No test labels are accepted by this API.
    """
    B, G = _validate_scores(B, 'B'), _validate_scores(G, 'G')
    if B.shape != G.shape:
        raise ValueError('B and G must have identical protein/GO alignment and shape')
    counts = _validate_counts(counts, B.shape[1])
    if mode not in ('transform', 'mix'):
        raise ValueError('mode must be transform or mix')
    common = {'input_score_type': 'already_sigmoid_probabilities',
              'calibration_labels_used': False, 'automatic_test_set_selection': False}
    endpoints = {'B': _endpoint_counts(B), 'G': _endpoint_counts(G)}
    logit_metadata = {'logit_clipping': 'endpoints_only_float64_nextafter',
                      'logit_lower_probability': float(_LOGIT_LOWER),
                      'logit_upper_probability': float(_LOGIT_UPPER),
                      'input_endpoint_counts': endpoints,
                      'rounding_note': 'endpoint clipping and finite precision may create ties; metrics use actual transformed scores'}
    yield 'B', B, {**common, 'rule': 'backbone_original'}
    yield 'G', G, {**common, 'rule': 'graph_original'}
    if mode == 'transform':
        yield 'G_sigmoid_again', _unary(G, _sigmoid), {
            **common, 'rule': 'sigmoid_of_probability', 'monotone_scope': 'global',
            'negative_control': True,
            'interpretation': 'theoretical ranking invariance; fixed threshold grid and floating ties may change measured metrics'}
        lo, hi = float(np.min(G)), float(np.max(G))
        normalized = _unary(G, lambda p: (np.asarray(p, dtype=np.float64) - lo) / (hi - lo)) \
            if hi > lo else np.zeros(G.shape, dtype=np.float64)
        yield 'G_global_minmax', normalized, {
            **common, 'rule': 'global_minmax', 'monotone_scope': 'global',
            'negative_control': True, 'minimum': lo, 'maximum': hi,
            'constant_matrix_policy': 'all_zero',
            'interpretation': 'global ranking invariance except floating ties; Fmax may change on the fixed grid'}
        del normalized
        def row_max(block):
            block = np.asarray(block, dtype=np.float64)
            scale = np.max(block, axis=1, keepdims=True)
            return np.divide(block, scale, out=np.zeros_like(block), where=scale > 0)
        yield 'G_row_max', _unary(G, row_max), {
            **common, 'rule': 'divide_each_protein_row_by_its_maximum',
            'monotone_scope': 'within_protein_row', 'zero_row_policy': 'all_zero',
            'interpretation': 'changes cross-protein scales; removes row confidence and need not improve calibration'}
        normalized, stats = _go_logit_zscore(G)
        yield 'G_go_logit_zscore', normalized, {
            **common, **logit_metadata, **stats,
            'rule': 'sigmoid_of_GO_wise_standardized_logit',
            'monotone_scope': 'within_GO_column',
            'interpretation': 'GO-wise monotonicity should preserve per-GO AP in exact arithmetic; actual macro AP may change due to finite ties and endpoint clipping'}
        return
    for alpha in (.25, .5, .75):
        yield f'prob_mix_g{round(100 * alpha):03d}', _probability_mix(B, G, alpha), {
            **common, 'rule': '(1-alpha)*B+alpha*G', 'graph_weight': alpha}
    yield 'logit_mix_g050', _logit_mix(B, G, .5), {
        **common, 'rule': 'sigmoid(0.5*logit(B)+0.5*logit(G))',
        'graph_weight': .5, 'logit_clipping': 'symmetric_fixed_epsilon',
        'logit_clip_epsilon': _MIX_LOGIT_EPSILON,
        'input_clipping_counts': {'B': _mix_clipping_counts(B), 'G': _mix_clipping_counts(G)},
        'input_endpoint_counts': endpoints,
        'rounding_note': 'symmetric clipping and finite precision can create ties; metrics use actual mixed scores'}
    yield 'max_BG', np.maximum(B, G), {
        **common, 'rule': 'maximum(B,G)',
        'interpretation': 'union-like scoring may increase recall and false positives'}
    for cutoff in (9, 99):
        protected = counts <= cutoff
        yield f'freq_protect_{cutoff}', _protected_mix(B, G, counts, cutoff), {
            **common, 'rule': 'B if core_count<=cutoff else 0.5*B+0.5*G',
            'core_count_cutoff': cutoff, 'protected_columns': int(protected.sum()),
            'protected_scores': 'copied_from_B_unchanged', 'graph_weight_elsewhere': .5,
            'interpretation': 'preserves protected B scores/AP; recall can still change at this method own overall Fmax threshold'}


def per_go_average_precision(labels, scores):
    """Tie-grouped AP in percent; no-positive columns are NaN.

    Only bounded blocks of positive GO columns are sorted. The same eligible
    label set is determined by gold alone and is shared across every method.
    """
    labels, scores = np.asarray(labels), np.asarray(scores)
    positive = np.asarray(labels.sum(axis=0), dtype=np.int64)
    eligible = np.flatnonzero(positive > 0)
    result = np.full(labels.shape[1], np.nan, dtype=np.float64)
    ranks = np.arange(1, len(labels) + 1, dtype=np.float64)[:, None]
    for start in range(0, len(eligible), _MACRO_COLUMNS):
        columns = eligible[start:start + _MACRO_COLUMNS]
        p = np.take(scores, columns, axis=1)
        y = np.take(labels, columns, axis=1)
        order = np.argsort(p, axis=0, kind='quicksort')[::-1]
        p = np.take_along_axis(p, order, axis=0)
        y = np.take_along_axis(y, order, axis=0)
        tp = np.cumsum(y, axis=0, dtype=np.int64)
        ends = np.empty(p.shape, dtype=bool)
        ends[:-1] = p[:-1] != p[1:]
        ends[-1] = True
        completed_tp = np.maximum.accumulate(np.where(ends, tp, 0), axis=0)
        previous = np.empty(tp.shape, dtype=np.int64)
        previous[0] = 0
        previous[1:] = completed_tp[:-1]
        increase = np.where(ends, tp - previous, 0) / positive[columns][None, :]
        result[columns] = 100 * np.sum(increase * tp / ranks, axis=0)
    return result


def _macro(values):
    finite = np.isfinite(values)
    return (float(values[finite].mean()) if finite.any() else None, int(finite.sum()))


def _pair_stats(labels, scores, threshold, columns=None):
    if not isinstance(threshold, (int, float, np.integer, np.floating)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('working-point threshold must be finite and in [0,1]')
    width = labels.shape[1] if columns is None else len(columns)
    predicted_columns = np.zeros(width, dtype=bool)
    predicted = true_positive = positives = 0
    for start in range(0, len(labels), _ROWS):
        y, p = labels[start:start + _ROWS], scores[start:start + _ROWS]
        if columns is not None:
            y, p = np.take(y, columns, axis=1), np.take(p, columns, axis=1)
        active = p > threshold
        predicted += int(np.count_nonzero(active))
        positives += int(np.count_nonzero(y))
        true_positive += int(np.count_nonzero(active & (y != 0)))
        predicted_columns |= np.any(active, axis=0)
    return {'threshold': float(threshold), 'tp': true_positive, 'true_positive_pairs': true_positive,
            'predicted_pairs': predicted, 'test_positive_pairs': positives,
            'pair_precision': 100 * true_positive / predicted if predicted else None,
            'pair_recall': 100 * true_positive / positives if positives else None,
            'columns_predicted': int(predicted_columns.sum())}


def summarize_method(labels, scores, counts, metric_module, backbone_threshold=None):
    """Measure actual transformed scores using the production metric kernels."""
    y, p = metric_module._validate(labels, scores)
    counts = _validate_counts(counts, y.shape[1])
    thresholds = metric_module._thresholds(.001)
    f1, precision, recall, eligible = metric_module._protein_curve(y, p, thresholds)
    best = int(np.argmax(f1))
    threshold = float(thresholds[best])
    per_go_ap = per_go_average_precision(y, p)
    macro_ap, macro_count = _macro(per_go_ap)
    positives = int(np.count_nonzero(y))
    micro_ap = float(metric_module._exact_micro(y, p)['standard_micro_ap']) if positives else None
    result = {'num_proteins': int(y.shape[0]), 'num_GO_columns': int(y.shape[1]),
              'test_positive_pairs': positives, 'fmax': 100 * float(f1[best]),
              'threshold': threshold, 'precision': 100 * float(precision[best]),
              'recall': 100 * float(recall[best]), 'eligible_proteins': int(eligible),
              'micro_ap': micro_ap, 'macro_ap': macro_ap,
              'macro_ap_eligible_go': macro_count,
              'macro_ap_definition': 'mean per-GO exact non-interpolated AP over the fixed gold-positive GO columns; ties grouped',
              'at_method_fmax': _pair_stats(y, p, threshold),
              '_per_go_ap': per_go_ap,
              'frequency_bins_overlap': True,
              'frequency_strata_note': '1-5 is nested in 1-9; rows and AP values must not be added',
              'frequency_strata': []}
    if backbone_threshold is not None:
        fixed = _pair_stats(y, p, backbone_threshold)
        fixed_f, fixed_p, fixed_r, fixed_n = metric_module._protein_curve(
            y, p, np.asarray([backbone_threshold], dtype=np.float64))
        fixed.update(protein_f1=100 * float(fixed_f[0]),
                     protein_precision=100 * float(fixed_p[0]),
                     protein_recall=100 * float(fixed_r[0]), eligible_proteins=int(fixed_n))
        result['at_backbone_threshold'] = fixed
    positive_columns = np.asarray(y.sum(axis=0)) > 0
    for lower, upper in _BINS:
        mask = counts >= lower
        if upper is not None:
            mask &= counts <= upper
        columns = np.flatnonzero(mask)
        total_positive = int(np.sum(y[:, mask]))
        if total_positive:
            subset_y, subset_p = np.take(y, columns, axis=1), np.take(p, columns, axis=1)
            ap = float(metric_module._exact_micro(subset_y, subset_p)['standard_micro_ap'])
            del subset_y, subset_p
        else:
            ap = None
        macro, macro_n = _macro(per_go_ap[mask])
        item = {'train_core_count_min': lower, 'train_core_count_max': upper,
                'columns': int(mask.sum()), 'test_positive_pairs': total_positive,
                'columns_with_test_positive': int(np.count_nonzero(positive_columns & mask)),
                'micro_ap': ap, 'macro_ap': macro, 'macro_ap_eligible_go': macro_n,
                'at_method_fmax': _pair_stats(y, p, threshold, columns)}
        if backbone_threshold is not None:
            item['at_backbone_threshold'] = _pair_stats(y, p, backbone_threshold, columns)
        result['frequency_strata'].append(item)
    return result
