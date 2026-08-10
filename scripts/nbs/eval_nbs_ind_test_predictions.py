#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _task_key(task: str) -> str:
    mapping = {"bp": "biological_process", "mf": "molecular_function", "cc": "cellular_component"}
    task = task.strip().lower()
    if task not in mapping:
        raise ValueError(f"unknown task={task!r}")
    return mapping[task]


def _task_dict(data: dict[str, Any], mode: str, task: str) -> dict[str, Any]:
    block = data.get(mode)
    if not isinstance(block, dict):
        raise KeyError(f"metadata lacks mode={mode!r}")
    for key in (task, _task_key(task)):
        value = block.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError(f"metadata mode={mode!r} lacks task={task!r}")


def _dense_labels(annotation: Any, num_classes: int, expected_rows: int) -> np.ndarray:
    if isinstance(annotation, np.ndarray):
        arr = annotation
        if arr.ndim == 2 and arr.shape == (expected_rows, num_classes):
            return (arr > 0).astype(np.bool_, copy=False)
    try:
        rows = list(annotation)
    except Exception as exc:
        raise TypeError("unsupported ind_test annotation object") from exc
    if len(rows) != expected_rows:
        raise ValueError(f"annotation rows={len(rows)} != proteins={expected_rows}")
    out = np.zeros((expected_rows, num_classes), dtype=np.bool_)
    for i, item in enumerate(rows):
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        arr = np.asarray(item)
        if arr.ndim == 1 and arr.size == num_classes and arr.dtype.kind in "biuf":
            out[i] = arr > 0
            continue
        idx = np.asarray(list(item) if isinstance(item, (set, tuple, list)) else item, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < num_classes)]
        out[i, idx] = True
    return out


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_ids(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _summary(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return {"count": 0}
    count = 0
    total = 0.0
    total_sq = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    positive = negative = zero = 0
    for start in range(0, arr.size, 1_000_000):
        chunk = np.asarray(arr[start : start + 1_000_000], dtype=np.float32)
        chunk = chunk[np.isfinite(chunk)]
        if not chunk.size:
            continue
        count += int(chunk.size)
        total += float(chunk.sum(dtype=np.float64))
        total_sq += float(np.square(chunk, dtype=np.float64).sum(dtype=np.float64))
        minimum = min(minimum, float(chunk.min()))
        maximum = max(maximum, float(chunk.max()))
        positive += int(np.count_nonzero(chunk > 0.0))
        negative += int(np.count_nonzero(chunk < 0.0))
        zero += int(np.count_nonzero(chunk == 0.0))
    if count == 0:
        return {"count": 0}
    # Exact quantiles for ordinary subsets; deterministic stride sampling keeps
    # full [N,GO] diagnostics memory bounded.
    stride = max(1, int(np.ceil(arr.size / 1_000_000)))
    sample = np.asarray(arr[::stride], dtype=np.float32)
    sample = sample[np.isfinite(sample)]
    q = np.quantile(sample, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    mean = total / count
    variance = max(0.0, total_sq / count - mean * mean)
    return {
        "count": count,
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "min": minimum,
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": maximum,
        "positive_fraction": positive / count,
        "negative_fraction": negative / count,
        "zero_fraction": zero / count,
        "quantile_sample_size": int(sample.size),
    }


def _column_mask(path: Path, num_classes: int) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
    if values.size and (int(values.min()) < 0 or int(values.max()) >= num_classes):
        raise IndexError(f"GO index outside [0,{num_classes}) in {path}")
    mask = np.zeros(num_classes, dtype=bool)
    mask[values] = True
    return mask


def _candidate_mask(path: Path, rows: int, num_classes: int) -> np.ndarray:
    candidate = np.asarray(np.load(path, mmap_mode="r"), dtype=np.int64)
    if candidate.ndim != 2 or candidate.shape[0] != rows:
        raise ValueError("candidate GO index must be [N_test,K]")
    valid = candidate >= 0
    if np.any(candidate[valid] >= num_classes):
        raise IndexError("candidate GO index is outside the task classifier space")
    mask = np.zeros((rows, num_classes), dtype=bool)
    row = np.broadcast_to(np.arange(rows, dtype=np.int64)[:, None], candidate.shape)
    mask[row[valid], candidate[valid]] = True
    return mask


def _masked_metric(
    compute_metric_pack: Any,
    y_true: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    threshold_step: float,
    auprc_mode: str,
    metrics_are_percent: bool,
) -> dict[str, Any]:
    selected_y = np.asarray(y_true[mask], dtype=np.bool_).reshape(1, -1)
    selected_p = np.asarray(prediction[mask], dtype=np.float32).reshape(1, -1)
    if selected_y.size == 0:
        return {"num_positions": 0, "num_positives": 0}
    result = compute_metric_pack(
        selected_y,
        selected_p,
        threshold_step=threshold_step,
        auprc_mode=auprc_mode,
        compute_sample_fmax=False,
        no_empty_labels=False,
        no_zero_classes=False,
        metrics_are_percent=metrics_are_percent,
    )
    result["num_positions"] = int(selected_y.size)
    result["num_positives"] = int(selected_y.sum())
    return result


def _make_frequency_bins(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Exact copy of the Stage-1 training-count bin definition."""
    counts = np.asarray(counts, dtype=np.float64)
    positive = counts > 0
    bins: dict[str, np.ndarray] = {"zero_train": counts == 0}
    if positive.sum() == 0:
        bins["rare"] = np.zeros_like(positive, dtype=bool)
        bins["medium"] = np.zeros_like(positive, dtype=bool)
        bins["common"] = np.zeros_like(positive, dtype=bool)
        return bins
    q33, q66 = np.quantile(counts[positive], [0.33, 0.66])
    bins["rare"] = positive & (counts <= q33)
    bins["medium"] = positive & (counts > q33) & (counts <= q66)
    bins["common"] = positive & (counts > q66)
    bins["rare_le_5"] = positive & (counts <= 5)
    bins["common_ge_50"] = counts >= 50
    return bins


def _fmax_from_histograms(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold_step: float,
) -> dict[str, float]:
    y = y_true.astype(bool, copy=False)
    probability = np.asarray(y_prob, dtype=np.float32)
    step = float(threshold_step)
    if not 0.0 < step <= 1.0:
        raise ValueError("threshold-step must lie in (0,1]")
    nbins = int(round(1.0 / step)) + 1
    thresholds = np.arange(nbins, dtype=np.float32) * step
    thresholds[-1] = 1.0
    index = np.floor(probability / step).astype(np.int32)
    index = np.clip(index, 0, nbins - 1)
    flat_index = index.ravel()
    flat_y = y.ravel()
    pred_hist = np.bincount(flat_index, minlength=nbins).astype(np.float64)
    tp_hist = np.bincount(flat_index[flat_y], minlength=nbins).astype(np.float64)
    pred_cum = np.cumsum(pred_hist[::-1])[::-1]
    tp_cum = np.cumsum(tp_hist[::-1])[::-1]
    total_positive = float(flat_y.sum())
    precision = tp_cum / np.maximum(pred_cum, 1.0)
    recall = tp_cum / max(total_positive, 1.0)
    fscore = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.nanargmax(fscore))
    order = np.arange(nbins - 1, -1, -1)
    previous_recall = 0.0
    approximate_ap = 0.0
    for current_recall, current_precision in zip(recall[order], precision[order]):
        if current_recall > previous_recall:
            approximate_ap += (current_recall - previous_recall) * current_precision
            previous_recall = float(current_recall)
    return {
        "fmax_micro": float(fscore[best]),
        "threshold_micro": float(thresholds[best]),
        "precision_micro": float(precision[best]),
        "recall_micro": float(recall[best]),
        "auprc_micro_hist": float(approximate_ap),
    }


def _compute_metric_pack(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold_step: float,
    auprc_mode: str,
    compute_sample_fmax: bool = False,
    no_empty_labels: bool = False,
    no_zero_classes: bool = False,
    metrics_are_percent: bool = True,
) -> dict[str, float]:
    del compute_sample_fmax  # NBS reporting is micro-oriented by contract.
    y = y_true
    probability = y_prob
    if no_empty_labels:
        keep = y.sum(axis=1) > 0
        y, probability = y[keep], probability[keep]
    if no_zero_classes:
        keep = y.sum(axis=0) > 0
        y, probability = y[:, keep], probability[:, keep]
    output = _fmax_from_histograms(
        y, probability, threshold_step=threshold_step
    )
    if auprc_mode == "exact":
        try:
            from sklearn.metrics import average_precision_score
        except ImportError:
            pass
        else:
            flat_y = y.astype(np.int8, copy=False).ravel()
            if flat_y.sum() > 0:
                output["auprc_micro_exact"] = float(
                    average_precision_score(
                        flat_y,
                        np.asarray(probability, dtype=np.float32).ravel(),
                    )
                )
    elif auprc_mode not in {"hist", "none"}:
        raise ValueError(f"unknown auprc-mode={auprc_mode!r}")
    if metrics_are_percent:
        for key in list(output):
            if key.startswith(("fmax", "precision", "recall", "auprc")):
                output[key] = 100.0 * float(output[key])
    return output


def _label_counts(annotation: Any, num_classes: int) -> tuple[np.ndarray, int]:
    counts = np.zeros(num_classes, dtype=np.float64)
    if annotation is None:
        return counts, 0
    if hasattr(annotation, "detach"):
        annotation = annotation.detach().cpu().numpy()
    if isinstance(annotation, np.ndarray):
        if annotation.ndim == 2 and annotation.shape[1] == num_classes:
            return annotation.astype(np.float64, copy=False).sum(axis=0), int(annotation.shape[0])
        if annotation.ndim == 1 and annotation.shape[0] == num_classes:
            return annotation.astype(np.float64, copy=False), 1
    rows = list(annotation)
    for item in rows:
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        array = np.asarray(item)
        if array.ndim == 1 and array.size == num_classes and array.dtype.kind in "biuf":
            counts += array.astype(np.float64, copy=False)
        else:
            index = np.asarray(list(item) if isinstance(item, (list, tuple, set)) else item, dtype=np.int64).reshape(-1)
            index = index[(index >= 0) & (index < num_classes)]
            counts[index] += 1.0
    return counts, len(rows)


def _load_train_label_counts(
    metadata_file: Path, task: str, num_classes: int
) -> tuple[np.ndarray, int, str]:
    with metadata_file.open("rb") as handle:
        metadata = pickle.load(handle)
    train = _task_dict(metadata, "train", task)
    for key in ("annotations", "labels", "prop_annotations"):
        if key in train:
            counts, rows = _label_counts(train[key], num_classes)
            if rows > 0:
                return counts, rows, key
    raise KeyError("metadata train block has no usable labels")


def _rare_analysis_metrics(
    *,
    y_true: np.ndarray,
    pred_map: Mapping[str, np.ndarray],
    class_bins: Mapping[str, np.ndarray],
    threshold_step: float,
    auprc_mode: str,
    metrics_are_percent: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, prediction in pred_map.items():
        output[key] = {}
        for bin_name, raw_mask in class_bins.items():
            mask = np.asarray(raw_mask, dtype=bool)
            if not mask.any():
                continue
            labels = y_true[:, mask]
            if labels.size == 0 or labels.sum() == 0:
                continue
            result = _compute_metric_pack(
                labels,
                prediction[:, mask],
                threshold_step=threshold_step,
                auprc_mode=auprc_mode,
                metrics_are_percent=metrics_are_percent,
            )
            result["num_classes"] = int(mask.sum())
            result["num_positives"] = int(labels.sum())
            output[key][bin_name] = result
    return output


def _align_rows(
    predictions: np.ndarray,
    *,
    prediction_ids: list[str] | None,
    metadata_ids: list[str],
) -> np.ndarray:
    if prediction_ids is None:
        if predictions.shape[0] != len(metadata_ids):
            raise ValueError(
                "prediction rows do not match metadata ind_test rows; provide --protein-ids for strict alignment"
            )
        return predictions
    if len(prediction_ids) != predictions.shape[0]:
        raise ValueError("prediction protein-id count does not match prediction rows")
    if len(set(prediction_ids)) != len(prediction_ids) or len(set(metadata_ids)) != len(metadata_ids):
        raise ValueError("protein IDs must be unique for independent-test alignment")
    row_by_id = {protein: i for i, protein in enumerate(prediction_ids)}
    missing = [protein for protein in metadata_ids if protein not in row_by_id]
    extra = sorted(set(prediction_ids) - set(metadata_ids))
    if missing or extra:
        raise ValueError(
            f"ind_test protein sets differ: missing={missing[:5]}, extra={extra[:5]}"
        )
    order = np.asarray([row_by_id[p] for p in metadata_ids], dtype=np.int64)
    return predictions[order]


def _load_prediction(path: Path, num_classes: int) -> np.ndarray:
    value = np.load(path, mmap_mode="r")
    if value.ndim != 2 or value.shape[1] != num_classes:
        raise ValueError(f"{path} shape={value.shape} does not match num_classes={num_classes}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate precomputed NBS independent-test predictions with the same "
            "Fmax/AUPRC and frequency-bin metric implementation used by the Stage-1 evaluator."
        )
    )
    parser.add_argument("--task", choices=["bp", "mf", "cc"], required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--nbs-prob", type=Path, required=True)
    parser.add_argument("--backbone-prob", type=Path, default=None)
    parser.add_argument("--modelout-prob", type=Path, default=None)
    parser.add_argument("--protein-ids", type=Path, default=None)
    parser.add_argument("--train-counts", type=Path, default=None)
    parser.add_argument("--candidate-go-index", type=Path, default=None)
    parser.add_argument("--eligible-go-index", type=Path, default=None)
    parser.add_argument("--applied-logit-delta", type=Path, default=None)
    parser.add_argument("--delta-gate", type=Path, default=None)
    parser.add_argument("--routing-source-weights", type=Path, default=None)
    parser.add_argument("--routing-null-weight", type=Path, default=None)
    parser.add_argument("--routing-source-names", type=Path, default=None)
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--prediction-manifest", type=Path, default=None)
    parser.add_argument("--go-registry", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--auprc-mode", choices=["hist", "exact", "none"], default="exact")
    parser.add_argument("--metrics-are-percent", action="store_true", default=True)
    args = parser.parse_args()

    num_classes = {"bp": 21312, "mf": 7038, "cc": 2903}[args.task]
    with args.metadata_file.open("rb") as handle:
        metadata = pickle.load(handle)
    task_data = _task_dict(metadata, "ind_test", args.task)
    proteins = [str(x) for x in task_data.get("proteins", [])]
    if not proteins:
        raise ValueError("ind_test metadata has no proteins")
    label_key = next(
        (
            key
            for key in ("prop_annotations", "prop_annotation", "annotations", "labels")
            if key in task_data
        ),
        None,
    )
    if label_key is None:
        raise KeyError("ind_test metadata has no recognized annotation field")
    y_true = _dense_labels(task_data[label_key], num_classes, len(proteins))

    input_manifest = _load_json(args.input_manifest)
    prediction_manifest = _load_json(args.prediction_manifest)
    metadata_ids_hash = _hash_ids(proteins)
    if input_manifest is not None:
        recorded = input_manifest.get("protein_ids_sha256")
        if recorded and str(recorded) != metadata_ids_hash:
            raise ValueError(
                "ind-test metadata order differs from the prepared input manifest"
            )
        leakage = input_manifest.get("leakage_contract", {})
        forbidden_true = [
            key
            for key in (
                "ind_test_labels_in_representation",
                "ind_test_labels_in_candidate_selection",
                "ind_test_labels_in_pp_retrieval",
                "expert_probability_in_forward_inputs",
                "test_to_test_edges",
            )
            if bool(leakage.get(key, False))
        ]
        if forbidden_true:
            raise RuntimeError(f"input manifest violates inductive contract: {forbidden_true}")
    if prediction_manifest is not None:
        if int(prediction_manifest.get("num_task_go", -1)) != num_classes:
            raise ValueError("prediction manifest task-space width is incorrect")
        if bool(prediction_manifest.get("uses_expert_probability_in_nbs_forward", True)):
            raise RuntimeError("prediction manifest reports expert probability in NBS forward")
        if prediction_manifest.get("prediction_space") != "complete_task_classifier_columns":
            raise ValueError("prediction manifest does not report the complete classifier space")
        recorded_prediction_hash = prediction_manifest.get("output_probability_sha256")
        if recorded_prediction_hash and str(recorded_prediction_hash) != _sha256(args.nbs_prob):
            raise ValueError("NBS probability hash differs from the prediction manifest")
        if args.input_manifest is not None:
            recorded_input_hash = prediction_manifest.get("input_manifest_sha256")
            if recorded_input_hash and str(recorded_input_hash) != _sha256(args.input_manifest):
                raise ValueError("prediction and input manifests do not belong to the same run")

    prediction_ids = None if args.protein_ids is None else _read_ids(args.protein_ids)
    if prediction_ids is not None and _hash_ids(prediction_ids) != metadata_ids_hash:
        # Set equality is checked by _align_rows below; a differing hash simply
        # means the rows will be explicitly reordered to metadata order.
        pass
    pred_map: dict[str, np.ndarray] = {}
    for key, path in (
        ("NBS_final", args.nbs_prob),
        ("backbone_base", args.backbone_prob),
        ("modelout_reference", args.modelout_prob),
    ):
        if path is None:
            continue
        arr = _load_prediction(path, num_classes)
        arr = _align_rows(arr, prediction_ids=prediction_ids, metadata_ids=proteins)
        pred_map[key] = np.asarray(arr, dtype=np.float32)

    metrics = {
        key: _compute_metric_pack(
            y_true,
            pred,
            threshold_step=float(args.threshold_step),
            auprc_mode=str(args.auprc_mode),
            compute_sample_fmax=False,
            no_empty_labels=False,
            no_zero_classes=False,
            metrics_are_percent=bool(args.metrics_are_percent),
        )
        for key, pred in pred_map.items()
    }

    if args.train_counts is not None:
        counts = np.asarray(np.load(args.train_counts, mmap_mode="r"), dtype=np.float64)
        if counts.shape != (num_classes,):
            raise ValueError("train-count vector does not match task GO space")
        count_source = str(args.train_counts)
        n_train = None
    else:
        counts, n_train, count_key = _load_train_label_counts(args.metadata_file, args.task, num_classes)
        count_source = f"metadata:train.{count_key}"
    bins = _make_frequency_bins(counts)
    rare = _rare_analysis_metrics(
        y_true=y_true,
        pred_map=pred_map,
        class_bins=bins,
        threshold_step=float(args.threshold_step),
        auprc_mode=("hist" if args.auprc_mode != "none" else "none"),
        metrics_are_percent=bool(args.metrics_are_percent),
    )

    candidate_analysis: dict[str, Any] = {}
    candidate_mask = None
    if args.candidate_go_index is not None:
        candidate_mask = _candidate_mask(
            args.candidate_go_index, len(proteins), num_classes
        )
        for key, prediction in pred_map.items():
            candidate_analysis[key] = {
                "candidate": _masked_metric(
                    _compute_metric_pack,
                    y_true,
                    prediction,
                    candidate_mask,
                    threshold_step=float(args.threshold_step),
                    auprc_mode=str(args.auprc_mode),
                    metrics_are_percent=bool(args.metrics_are_percent),
                ),
                "non_candidate": _masked_metric(
                    _compute_metric_pack,
                    y_true,
                    prediction,
                    ~candidate_mask,
                    threshold_step=float(args.threshold_step),
                    auprc_mode=str(args.auprc_mode),
                    metrics_are_percent=bool(args.metrics_are_percent),
                ),
            }

    query_scope_analysis: dict[str, Any] = {}
    eligible_mask = None
    if args.eligible_go_index is not None:
        eligible_mask = _column_mask(args.eligible_go_index, num_classes)
        for key, prediction in pred_map.items():
            query_scope_analysis[key] = {}
            for scope_name, scope_mask in (
                ("eligible_direct_query", eligible_mask),
                ("context_only", ~eligible_mask),
            ):
                if not scope_mask.any():
                    continue
                query_scope_analysis[key][scope_name] = _compute_metric_pack(
                    y_true[:, scope_mask],
                    prediction[:, scope_mask],
                    threshold_step=float(args.threshold_step),
                    auprc_mode=str(args.auprc_mode),
                    compute_sample_fmax=False,
                    no_empty_labels=False,
                    no_zero_classes=False,
                    metrics_are_percent=bool(args.metrics_are_percent),
                )
                query_scope_analysis[key][scope_name].update(
                    {
                        "num_classes": int(scope_mask.sum()),
                        "num_positives": int(y_true[:, scope_mask].sum()),
                    }
                )

    diagnostics: dict[str, Any] = {}
    base_prediction = pred_map.get("backbone_base")
    nbs_prediction = pred_map.get("NBS_final")
    if base_prediction is not None and nbs_prediction is not None:
        probability_delta = nbs_prediction - base_prediction
        diagnostics["probability_delta"] = {
            "overall": _summary(probability_delta),
            "positive_labels": _summary(probability_delta[y_true]),
            "unlabelled_positions": _summary(probability_delta[~y_true]),
        }
        for bin_name, mask in bins.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.any():
                diagnostics["probability_delta"][f"frequency/{bin_name}"] = _summary(
                    probability_delta[:, mask]
                )
        if candidate_mask is not None:
            diagnostics["probability_delta"]["candidate"] = _summary(
                probability_delta[candidate_mask]
            )
            diagnostics["probability_delta"]["non_candidate"] = _summary(
                probability_delta[~candidate_mask]
            )
        if eligible_mask is not None:
            diagnostics["probability_delta"]["eligible_direct_query"] = _summary(
                probability_delta[:, eligible_mask]
            )
            diagnostics["probability_delta"]["context_only"] = _summary(
                probability_delta[:, ~eligible_mask]
            )

    if args.applied_logit_delta is not None:
        applied = _load_prediction(args.applied_logit_delta, num_classes)
        applied = _align_rows(
            applied, prediction_ids=prediction_ids, metadata_ids=proteins
        )
        diagnostics["applied_logit_delta"] = {
            "overall": _summary(applied),
            "positive_labels": _summary(np.asarray(applied)[y_true]),
            "unlabelled_positions": _summary(np.asarray(applied)[~y_true]),
        }
        for bin_name, mask in bins.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.any():
                diagnostics["applied_logit_delta"][f"frequency/{bin_name}"] = _summary(
                    np.asarray(applied)[:, mask]
                )
        if candidate_mask is not None:
            diagnostics["applied_logit_delta"]["candidate"] = _summary(
                np.asarray(applied)[candidate_mask]
            )
            diagnostics["applied_logit_delta"]["non_candidate"] = _summary(
                np.asarray(applied)[~candidate_mask]
            )
        if eligible_mask is not None:
            diagnostics["applied_logit_delta"]["eligible_direct_query"] = _summary(
                np.asarray(applied)[:, eligible_mask]
            )
            diagnostics["applied_logit_delta"]["context_only"] = _summary(
                np.asarray(applied)[:, ~eligible_mask]
            )
    if args.delta_gate is not None:
        gate = _load_prediction(args.delta_gate, num_classes)
        gate = _align_rows(gate, prediction_ids=prediction_ids, metadata_ids=proteins)
        diagnostics["delta_gate"] = {"overall": _summary(gate)}
        for bin_name, mask in bins.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.any():
                diagnostics["delta_gate"][f"frequency/{bin_name}"] = _summary(
                    np.asarray(gate)[:, mask]
                )
        if candidate_mask is not None:
            diagnostics["delta_gate"]["candidate"] = _summary(
                np.asarray(gate)[candidate_mask]
            )
            diagnostics["delta_gate"]["non_candidate"] = _summary(
                np.asarray(gate)[~candidate_mask]
            )
    if args.routing_source_weights is not None:
        source_weight = np.load(args.routing_source_weights, mmap_mode="r")
        if source_weight.ndim != 2 or source_weight.shape[0] != num_classes:
            raise ValueError("routing source weights must be [num_classes,num_sources]")
        names = None
        if args.routing_source_names is not None:
            names = json.loads(args.routing_source_names.read_text(encoding="utf-8"))
            if not isinstance(names, list) or len(names) != source_weight.shape[1]:
                raise ValueError("routing source names do not align with source weights")
        diagnostics["routing_source_weights"] = {
            "shape": list(source_weight.shape),
            "source_names": names,
            "per_source": {
                (str(names[i]) if names is not None else str(i)): _summary(source_weight[:, i])
                for i in range(source_weight.shape[1])
            },
        }
    if args.routing_null_weight is not None:
        null_weight = np.load(args.routing_null_weight, mmap_mode="r")
        if null_weight.shape != (num_classes,):
            raise ValueError("routing null weight must be [num_classes]")
        diagnostics["routing_null_weight"] = _summary(null_weight)

    go_registry_contract = None
    if args.go_registry is not None:
        registry_hash = _sha256(args.go_registry)
        if input_manifest is not None:
            expected_registry_hash = input_manifest.get("registries", {}).get(
                "go_registry_sha256"
            )
            if expected_registry_hash and str(expected_registry_hash) != registry_hash:
                raise ValueError("GO registry differs from the prepared Stage-1 task order")
        go_registry_contract = {
            "path": str(args.go_registry),
            "sha256": registry_hash,
        }

    result = {
        "schema_version": 2,
        "task": args.task,
        "mode": "ind_test",
        "num_samples": len(proteins),
        "num_classes": num_classes,
        "label_key": label_key,
        "metrics": metrics,
        "frequency": {
            "count_source": count_source,
            "num_train_samples": n_train,
            "bin_sizes": {name: int(np.asarray(mask, dtype=bool).sum()) for name, mask in bins.items()},
        },
        "rare_analysis": rare,
        "candidate_analysis": candidate_analysis,
        "query_scope_analysis": query_scope_analysis,
        "diagnostics": diagnostics,
        "alignment_contract": {
            "metadata_protein_ids_sha256": metadata_ids_hash,
            "prediction_protein_ids_sha256": (
                None if prediction_ids is None else _hash_ids(prediction_ids)
            ),
            "rows_reordered_to_metadata": (
                prediction_ids is not None and prediction_ids != proteins
            ),
            "go_registry": go_registry_contract,
            "input_manifest": (
                None if args.input_manifest is None else str(args.input_manifest)
            ),
            "prediction_manifest": (
                None if args.prediction_manifest is None else str(args.prediction_manifest)
            ),
        },
        "prediction_sources": {
            "NBS_final": str(args.nbs_prob),
            "backbone_base": None if args.backbone_prob is None else str(args.backbone_prob),
            "modelout_reference": None if args.modelout_prob is None else str(args.modelout_prob),
        },
        "note": (
            "modelout_reference may contain expert information and is a strong reference only; "
            "NBS_final must be produced without expert probability in the NBS forward pass."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "nbs_ind_test_metrics.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    print(f"[Saved] {out}")


if __name__ == "__main__":
    main()