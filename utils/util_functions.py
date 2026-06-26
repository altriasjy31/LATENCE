from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

def make_frequency_bins(counts: np.ndarray) -> Dict[str, np.ndarray]:
    counts = np.asarray(counts, dtype=np.float64)
    pos = counts > 0
    bins: Dict[str, np.ndarray] = {}
    bins["zero_train"] = (counts == 0)


    if pos.sum() == 0:
        bins["rare"] = np.zeros_like(pos, dtype=bool)
        bins["medium"] = np.zeros_like(pos, dtype=bool)
        bins["common"] = np.zeros_like(pos, dtype=bool)
        return bins


    pos_counts = counts[pos]
    q33, q66 = np.quantile(pos_counts, [0.33, 0.66])
    bins["rare"] = pos & (counts <= q33)
    bins["medium"] = pos & (counts > q33) & (counts <= q66)
    bins["common"] = pos & (counts > q66)
    bins["rare_le_5"] = pos & (counts <= 5)
    bins["common_ge_50"] = counts >= 50
    return bins

def compute_simulated_ic(preds: np.ndarray, step=0.01) -> np.ndarray:
    """
    Compute simulated IC per GO term from prediction matrix.
    preds: [num_samples, num_classes] float32 in [0,1]
    step: threshold step
    Returns:
        ic_sim: [num_classes] float32
    """
    thresholds = np.arange(0.0, 1.0 + step, step)
    num_classes = preds.shape[1]
    counts = np.zeros((len(thresholds), num_classes), dtype=np.float32)

    for i, t in enumerate(thresholds):
        binarized = (preds >= t).astype(np.float32)  # [num_samples, num_classes]
        counts[i] = binarized.sum(axis=0) / preds.shape[0]  # frequency fraction

    ic_sim = counts.mean(axis=0)  # average frequency over thresholds
    return ic_sim

def compute_ic_from_pred_map(pred_map: Dict[str, np.ndarray], source_key: str, num_bins: int = 100) -> Dict[str, np.ndarray]:
    """
    根据 pred_map 中指定来源的预测概率矩阵，计算 rare/medium/common term

    Args:
        pred_map: 包含不同预测来源的 dict，如 "backbone_base"、"external_only"
        source_key: 使用哪个 key 来统计 IC
        num_bins: 离散化步数

    Returns:
        bins: Dict[str, np.ndarray], 对每个 GO term 标注 rare/medium/common
    """
    if source_key not in pred_map:
        raise KeyError(f"{source_key} not found in pred_map")
    y_pred = pred_map[source_key]  # shape [N, C]

    N, C = y_pred.shape
    term_freqs = np.zeros(C, dtype=np.float64)

    thresholds = np.linspace(0, 1, num_bins)
    for thr in thresholds:
        term_freqs += (y_pred >= thr).sum(axis=0)
    term_freqs /= num_bins

    # 使用原版 make_frequency_bins 逻辑
    pos = term_freqs > 0
    bins: Dict[str, np.ndarray] = {}
    bins["zero_train"] = term_freqs == 0
    if pos.sum() == 0:
        bins["rare"] = np.zeros_like(pos, dtype=bool)
        bins["medium"] = np.zeros_like(pos, dtype=bool)
        bins["common"] = np.zeros_like(pos, dtype=bool)
        return bins

    pos_counts = term_freqs[pos]
    q33, q66 = np.quantile(pos_counts, [0.33, 0.66])
    bins["rare"] = pos & (term_freqs <= q33)
    bins["medium"] = pos & (term_freqs > q33) & (term_freqs <= q66)
    bins["common"] = pos & (term_freqs > q66)
    bins["rare_le_5"] = pos & (term_freqs <= 5/N)
    bins["common_ge_50"] = term_freqs >= 0.5
    return bins

def resolve_pred_map_source_key(source_key: str, pred_map: Dict[str, np.ndarray]) -> str:
    """
    Resolve short aliases to actual pred_map keys.

    Examples:
        base        -> backbone_base
        expert      -> external_only
        external    -> external_only
        probmix_a08 -> prob_mix::expert_base::alpha_0.8
    """

    key = str(source_key)

    alias = {
        "base": "backbone_base",
        "backbone": "backbone_base",
        "backbone_base": "backbone_base",

        "expert": "external_only",
        "external": "external_only",
        "external_only": "external_only",

        "probmix_a05": "prob_mix::expert_base::alpha_0.5",
        "probmix_a07": "prob_mix::expert_base::alpha_0.7",
        "probmix_a08": "prob_mix::expert_base::alpha_0.8",
        "probmix_a09": "prob_mix::expert_base::alpha_0.9",
        "probmix_a095": "prob_mix::expert_base::alpha_0.95",

        "modelout_a05": "modelout::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.5",
        "modelout_a08": "modelout::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.8",
        "modelout_a09": "modelout::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.9",

        "modelanchor_a05": "modelanchor::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.5",
        "modelanchor_a08": "modelanchor::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.8",
        "modelanchor_a09": "modelanchor::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.9",
    }

    resolved = alias.get(key, key)

    if resolved not in pred_map:
        available = "\n  ".join(sorted(pred_map.keys())[:200])
        raise KeyError(
            f"simulated_ic_source_key={source_key!r} resolved to {resolved!r}, "
            f"but this key is not in pred_map.\n"
            f"Available pred_map keys include:\n  {available}"
        )

    return resolved


def compute_simulated_counts_from_scores(
    scores: np.ndarray,
    threshold_min: float = 0.01,
    threshold_max: float = 1.0,
    threshold_step: float = 0.01,
    include_zero_threshold: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute simulated GO-term counts from prediction scores.

    For each threshold t:
        binary_pred = scores >= t
        count_i(t) = number of samples predicting GO_i above t

    simulated_count_i:
        average_t count_i(t)

    This follows the proposed threshold-sweep idea while keeping the final
    rare/medium/common split identical to make_frequency_bins(counts).

    Important:
        Do not include threshold 0 by default.
        If threshold=0 is included, every class receives at least N / num_thresholds
        artificial count, which makes rare_le_5 almost impossible on N=1800.
    """

    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"scores must be [N, C], got shape={scores.shape}")

    if threshold_step <= 0:
        raise ValueError(f"threshold_step must be positive, got {threshold_step}")

    if include_zero_threshold:
        start = 0.0
    else:
        start = max(float(threshold_min), float(threshold_step))

    thresholds = np.arange(
        start,
        float(threshold_max) + 0.5 * float(threshold_step),
        float(threshold_step),
        dtype=np.float32,
    )
    thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]

    if thresholds.size == 0:
        raise ValueError(
            f"No thresholds generated from min={threshold_min}, "
            f"max={threshold_max}, step={threshold_step}, "
            f"include_zero={include_zero_threshold}"
        )

    counts = np.zeros(scores.shape[1], dtype=np.float64)

    # Threshold loop is memory-stable and explicit.
    for thr in thresholds:
        counts += (scores >= float(thr)).sum(axis=0).astype(np.float64)

    counts /= float(len(thresholds))

    meta = {
        "num_samples": int(scores.shape[0]),
        "num_classes": int(scores.shape[1]),
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
        "threshold_step": float(threshold_step),
        "num_thresholds": int(len(thresholds)),
        "include_zero_threshold": bool(include_zero_threshold),
        "counts_min": float(counts.min()),
        "counts_max": float(counts.max()),
        "counts_mean": float(counts.mean()),
        "counts_median": float(np.median(counts)),
    }

    return counts, meta


def compute_simulated_bins_from_pred_map(
    pred_map: Dict[str, np.ndarray],
    source_key: str,
    threshold_min: float = 0.01,
    threshold_max: float = 1.0,
    threshold_step: float = 0.01,
    include_zero_threshold: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Build make_frequency_bins-compatible bins from pred_map[source_key].
    """

    resolved_key = resolve_pred_map_source_key(source_key, pred_map)
    scores = pred_map[resolved_key]

    sim_counts, meta = compute_simulated_counts_from_scores(
        scores=scores,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        include_zero_threshold=include_zero_threshold,
    )

    bins = make_frequency_bins(sim_counts)

    meta = dict(meta)
    meta["source_key"] = str(source_key)
    meta["resolved_source_key"] = str(resolved_key)
    meta["count_source_key"] = f"simulated_ic::{resolved_key}"

    meta["bin_sizes"] = {
        k: int(np.asarray(v).sum())
        for k, v in bins.items()
        if isinstance(v, np.ndarray) and v.dtype == bool
    }

    return bins, sim_counts, meta

def sanitize_key_for_filename(key: str) -> str:
    s = str(key)
    for ch in [":", "/", "\\", " ", "\t", "\n", "|", "*", "?", "<", ">", '"']:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")