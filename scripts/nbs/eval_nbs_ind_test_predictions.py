#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import pickle
import re
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping

import numpy as np


EVALUATION_VERSION = "0.6.1"
EXPECTED_EXTERNAL_COMPARISONS = (
    "BLAST_best_hit",
    "label_propagation",
    "NetGO3.0",
    "SPROF-GO",
    "PANDA2",
)


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


def _load_stage1_metric_backend() -> tuple[Callable[..., Any], Path]:
    """Load the exact Stage-1 metric implementation used by eval_ind_test.py."""
    msa_root = _project_root() / "msa_models"
    helper_path = msa_root / "helper_functions" / "helper.py"
    if not helper_path.is_file():
        raise RuntimeError(
            "The formal metric backend requires "
            f"{helper_path}, but it is missing. Restore the Stage-1 msa_models tree "
            "or explicitly use --metric-backend local_micro for diagnostics only."
        )
    text = str(msa_root)
    if text not in sys.path:
        sys.path.insert(0, text)
    module = importlib.import_module("helper_functions.helper")
    evaluator = getattr(module, "evalperf_torch", None)
    if evaluator is None:
        raise RuntimeError(f"{helper_path} does not export evalperf_torch")
    return evaluator, helper_path


def _compute_stage1_metric_pack(
    evaluator: Callable[..., Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Run the official project metric without reimplementing its Fmax semantics."""
    import torch

    labels = torch.from_numpy(
        np.ascontiguousarray(np.asarray(y_true, dtype=np.float32))
    )
    probabilities = torch.from_numpy(
        np.ascontiguousarray(np.asarray(y_prob, dtype=np.float32))
    )
    result = evaluator(
        targs=labels,
        preds=probabilities,
        threshold=True,
        auprc=True,
        no_empty_labels=False,
        no_zero_classes=False,
    )
    if not isinstance(result, Mapping):
        raise TypeError("Stage-1 evalperf_torch must return a metric mapping")
    return {str(key): float(value) for key, value in result.items()}


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


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


def _safe_comparison_name(value: str) -> str:
    name = value.strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(
            f"invalid comparison name={value!r}; use letters, digits, '.', '_' or '-'"
        )
    return name


def _load_external_comparisons(
    manifest_path: Path | None,
    *,
    num_classes: int,
    metadata_ids: list[str],
    go_registry_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if manifest_path is None:
        return {}, {"manifest": None, "methods": {}}
    raw = _load_json(manifest_path)
    assert raw is not None
    methods = raw.get("predictions", raw.get("methods", raw))
    if not isinstance(methods, dict):
        raise TypeError("comparison manifest must contain a predictions/methods object")
    reserved = {"NBS_final", "backbone_base", "modelout_reference"}
    predictions: dict[str, np.ndarray] = {}
    expected_go_registry_sha256 = (
        None if go_registry_path is None else _sha256(go_registry_path)
    )
    contract: dict[str, Any] = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "methods": {},
    }
    for raw_name, raw_spec in methods.items():
        name = _safe_comparison_name(str(raw_name))
        if name in reserved:
            raise ValueError(f"comparison manifest may not override reserved name {name!r}")
        spec = {"probability": raw_spec} if isinstance(raw_spec, str) else raw_spec
        if not isinstance(spec, dict):
            raise TypeError(f"comparison specification for {name!r} must be a path or object")
        probability_value = spec.get("probability", spec.get("prob", spec.get("path")))
        if not probability_value:
            raise KeyError(f"comparison {name!r} has no probability path")
        probability_path = _resolve_manifest_path(manifest_path, str(probability_value))
        ids_value = spec.get("protein_ids")
        ids_path = (
            None
            if not ids_value
            else _resolve_manifest_path(manifest_path, str(ids_value))
        )
        comparison_ids = None if ids_path is None else _read_ids(ids_path)
        method_registry_value = spec.get("go_registry")
        method_registry_path = (
            None
            if not method_registry_value
            else _resolve_manifest_path(manifest_path, str(method_registry_value))
        )
        method_registry_sha256 = (
            str(spec["go_registry_sha256"])
            if spec.get("go_registry_sha256")
            else (
                None
                if method_registry_path is None
                else _sha256(method_registry_path)
            )
        )
        if (
            expected_go_registry_sha256 is not None
            and method_registry_sha256 is not None
            and method_registry_sha256 != expected_go_registry_sha256
        ):
            raise ValueError(
                f"comparison {name!r} uses a different GO registry/order"
            )
        values = _load_prediction(probability_path, num_classes)
        values = _align_rows(
            values,
            prediction_ids=comparison_ids,
            metadata_ids=metadata_ids,
        )
        predictions[name] = np.asarray(values, dtype=np.float32)
        contract["methods"][name] = {
            "role": str(spec.get("role", "external_comparison")),
            "probability": str(probability_path),
            "probability_sha256": _sha256(probability_path),
            "protein_ids": None if ids_path is None else str(ids_path),
            "protein_ids_sha256": (
                None if comparison_ids is None else _hash_ids(comparison_ids)
            ),
            "protein_order_verified": comparison_ids is not None,
            "rows_reordered_to_metadata": (
                comparison_ids is not None and comparison_ids != metadata_ids
            ),
            "go_registry": (
                None if method_registry_path is None else str(method_registry_path)
            ),
            "go_registry_sha256": method_registry_sha256,
            "go_order_verified": (
                expected_go_registry_sha256 is not None
                and method_registry_sha256 == expected_go_registry_sha256
            ),
            "deployable": bool(spec.get("deployable", True)),
            "description": spec.get("description"),
        }
    return predictions, contract


def _read_delimited_rows(path: Path) -> list[dict[str, str]]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"table has no header: {path}")
        rows = []
        for raw in reader:
            rows.append(
                {
                    str(key).strip(): "" if value is None else str(value).strip()
                    for key, value in raw.items()
                    if key is not None
                }
            )
    return rows


def _first_column(row: Mapping[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name, "")
        if value != "":
            return value
    return ""


def _load_protein_strata(
    path: Path | None,
    *,
    proteins: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Load leakage-safe evaluator-only protein strata aligned by protein ID."""
    if path is None:
        return {}, {"path": None, "status": "not_provided"}
    rows = _read_delimited_rows(path)
    row_by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        protein = _first_column(row, ("protein_id", "protein", "id"))
        if not protein:
            raise ValueError("protein-strata table requires a protein_id column")
        if protein in row_by_id:
            raise ValueError(f"duplicate protein in strata table: {protein}")
        row_by_id[protein] = row
    missing = [protein for protein in proteins if protein not in row_by_id]
    extra = sorted(set(row_by_id) - set(proteins))
    if missing or extra:
        raise ValueError(
            "protein-strata IDs must exactly match ind_test: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    dimensions: dict[str, np.ndarray] = {}
    definitions: dict[str, Any] = {}
    ordered = [row_by_id[protein] for protein in proteins]

    def categorical(column: str, aliases: tuple[str, ...] = ()) -> np.ndarray | None:
        values = [
            _first_column(row, (column, *aliases))
            for row in ordered
        ]
        if not any(values):
            return None
        if not all(values):
            raise ValueError(f"strata column {column!r} is only partially populated")
        return np.asarray(values, dtype=object)

    identity_bin = categorical("sequence_identity_bin", ("seq_identity_bin",))
    identity_raw: np.ndarray | None = None
    if identity_bin is None:
        raw = [
            _first_column(
                row,
                ("sequence_identity", "max_sequence_identity", "seq_identity"),
            )
            for row in ordered
        ]
        if any(raw):
            if not all(raw):
                raise ValueError("sequence_identity is only partially populated")
            identity_raw = np.asarray([float(value) for value in raw], dtype=np.float64)
            if np.any(~np.isfinite(identity_raw)) or np.any(
                (identity_raw < 0.0) | (identity_raw > 1.0)
            ):
                raise ValueError("sequence_identity must be finite and lie in [0,1]")
            identity_bin = np.full(len(proteins), "ge_0.7", dtype=object)
            identity_bin[identity_raw < 0.7] = "0.5_to_0.7"
            identity_bin[identity_raw < 0.5] = "0.3_to_0.5"
            identity_bin[identity_raw < 0.3] = "lt_0.3"
            definitions["sequence_identity"] = {
                "source": "sequence_identity",
                "unit": "fraction",
                "bins": ["lt_0.3", "0.3_to_0.5", "0.5_to_0.7", "ge_0.7"],
            }
    if identity_bin is not None:
        dimensions["sequence_identity"] = identity_bin
        definitions.setdefault(
            "sequence_identity",
            {"source": "sequence_identity_bin", "bins": "provided_categorical"},
        )

    neff_bin = categorical("msa_neff_bin", ("neff_bin",))
    if neff_bin is None:
        raw = [_first_column(row, ("msa_neff", "neff")) for row in ordered]
        if any(raw):
            if not all(raw):
                raise ValueError("msa_neff is only partially populated")
            neff = np.asarray([float(value) for value in raw], dtype=np.float64)
            if np.any(~np.isfinite(neff)) or np.any(neff < 0.0):
                raise ValueError("msa_neff must be finite and non-negative")
            neff_bin = np.full(len(proteins), "gt_100", dtype=object)
            neff_bin[neff <= 100.0] = "10_to_100"
            neff_bin[neff <= 10.0] = "1_to_10"
            neff_bin[neff <= 1.0] = "le_1"
            definitions["msa_neff"] = {
                "source": "msa_neff",
                "bins": ["le_1", "1_to_10", "10_to_100", "gt_100"],
            }
    if neff_bin is not None:
        dimensions["msa_neff"] = neff_bin
        definitions.setdefault(
            "msa_neff", {"source": "msa_neff_bin", "bins": "provided_categorical"}
        )

    foldseek_bin = categorical("foldseek_bin", ("structure_bin",))
    foldseek_raw: np.ndarray | None = None
    if foldseek_bin is None:
        raw = [
            _first_column(row, ("foldseek_score", "max_foldseek_score"))
            for row in ordered
        ]
        if any(raw):
            foldseek_raw = np.asarray(
                [np.nan if value == "" else float(value) for value in raw],
                dtype=np.float64,
            )
            finite = np.isfinite(foldseek_raw)
            if np.any((foldseek_raw[finite] < 0.0) | (foldseek_raw[finite] > 1.0)):
                raise ValueError("foldseek_score must lie in [0,1]; blank means no hit")
            foldseek_bin = np.full(len(proteins), "no_hit", dtype=object)
            foldseek_bin[finite & (foldseek_raw < 0.3)] = "lt_0.3"
            foldseek_bin[finite & (foldseek_raw >= 0.3) & (foldseek_raw < 0.5)] = "0.3_to_0.5"
            foldseek_bin[finite & (foldseek_raw >= 0.5)] = "ge_0.5"
            definitions["foldseek"] = {
                "source": "foldseek_score",
                "unit": "precomputed_similarity_fraction",
                "bins": ["no_hit", "lt_0.3", "0.3_to_0.5", "ge_0.5"],
            }
    if foldseek_bin is not None:
        dimensions["foldseek"] = foldseek_bin
        definitions.setdefault(
            "foldseek", {"source": "foldseek_bin", "bins": "provided_categorical"}
        )

    evidence_tier = categorical("evidence_tier")
    if evidence_tier is not None:
        dimensions["evidence_tier"] = evidence_tier
        definitions["evidence_tier"] = {
            "source": "evidence_tier",
            "bins": "provided_categorical",
            "claim_policy": "strong claims are restricted to EXP/IDA-backed Tier 0",
        }

    if identity_raw is not None and foldseek_raw is not None:
        remote = (identity_raw < 0.3) & (
            ~np.isfinite(foldseek_raw) | (foldseek_raw < 0.5)
        )
        dimensions["compound_remote"] = np.where(
            remote, "seq_lt_0.3_and_foldseek_lt_0.5_or_no_hit", "other"
        ).astype(object)
        definitions["compound_remote"] = {
            "source": "derived",
            "remote_definition": (
                "sequence_identity < 0.3 and (Foldseek score < 0.5 or no hit)"
            ),
        }

    masks: dict[str, dict[str, np.ndarray]] = {}
    for dimension, values in dimensions.items():
        masks[dimension] = {
            str(value): np.asarray(values == value, dtype=bool)
            for value in sorted(set(str(value) for value in values))
        }
    return masks, {
        "path": str(path),
        "sha256": _sha256(path),
        "status": "loaded",
        "num_proteins": len(proteins),
        "definitions": definitions,
    }


def _row_slice_metric_bundle(
    *,
    y_true: np.ndarray,
    pred_map: Mapping[str, np.ndarray],
    row_mask: np.ndarray,
    threshold_step: float,
    auprc_mode: str,
    metrics_are_percent: bool,
    stage1_evaluator: Callable[..., Any] | None,
) -> dict[str, Any]:
    selected_y = y_true[row_mask]
    output: dict[str, Any] = {
        "num_proteins": int(np.count_nonzero(row_mask)),
        "num_positive_labels": int(np.count_nonzero(selected_y)),
        "metrics": {},
        "micro_diagnostics": {},
    }
    if selected_y.shape[0] == 0 or not np.any(selected_y):
        output["status"] = "insufficient_positive_labels"
        return output
    for name, prediction in pred_map.items():
        selected_prediction = prediction[row_mask]
        output["micro_diagnostics"][name] = _compute_metric_pack(
            selected_y,
            selected_prediction,
            threshold_step=threshold_step,
            auprc_mode=auprc_mode,
            metrics_are_percent=metrics_are_percent,
        )
        output["metrics"][name] = (
            output["micro_diagnostics"][name]
            if stage1_evaluator is None
            else _compute_stage1_metric_pack(
                stage1_evaluator, selected_y, selected_prediction
            )
        )
    output["status"] = "ok"
    return output


def _protein_strata_analysis(
    *,
    y_true: np.ndarray,
    pred_map: Mapping[str, np.ndarray],
    strata_masks: Mapping[str, Mapping[str, np.ndarray]],
    threshold_step: float,
    auprc_mode: str,
    metrics_are_percent: bool,
    stage1_evaluator: Callable[..., Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension, bins in strata_masks.items():
        result[dimension] = {}
        for bin_name, row_mask in bins.items():
            result[dimension][bin_name] = _row_slice_metric_bundle(
                y_true=y_true,
                pred_map=pred_map,
                row_mask=np.asarray(row_mask, dtype=bool),
                threshold_step=threshold_step,
                auprc_mode=auprc_mode,
                metrics_are_percent=metrics_are_percent,
                stage1_evaluator=stage1_evaluator,
            )
    return result


def _parse_codes(value: str) -> tuple[str, ...]:
    codes = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not codes:
        raise ValueError("strong-evidence-codes must contain at least one code")
    return codes


def _load_label_evidence(
    path: Path | None,
    *,
    proteins: list[str],
    y_true: np.ndarray,
    strong_codes: tuple[str, ...],
    pred_map: Mapping[str, np.ndarray],
    threshold_step: float,
    auprc_mode: str,
    metrics_are_percent: bool,
    stage1_evaluator: Callable[..., Any] | None,
) -> dict[str, Any]:
    """Evaluate EXP/IDA-backed positives without feeding evidence into inference."""
    if path is None:
        return {
            "status": "not_provided",
            "strong_codes": list(strong_codes),
            "claim_policy": (
                "strong EXP/IDA claims are unavailable until label evidence is provided"
            ),
        }
    row_by_protein = {protein: row for row, protein in enumerate(proteins)}
    rows = _read_delimited_rows(path)
    seen: set[tuple[int, int, str]] = set()
    by_code: dict[str, list[tuple[int, int]]] = {}
    for item in rows:
        protein = _first_column(item, ("protein_id", "protein", "id"))
        if protein not in row_by_protein:
            raise ValueError(f"label-evidence protein is outside ind_test: {protein!r}")
        go_value = _first_column(item, ("go_index", "go_idx", "class_index"))
        code = _first_column(item, ("evidence_code", "evidence", "code")).upper()
        if not go_value or not code:
            raise ValueError(
                "label-evidence table requires protein_id, go_index and evidence_code"
            )
        go_index = int(go_value)
        row_index = row_by_protein[protein]
        if go_index < 0 or go_index >= y_true.shape[1]:
            raise IndexError(f"label-evidence GO index outside task space: {go_index}")
        if not bool(y_true[row_index, go_index]):
            raise ValueError(
                "label-evidence row does not correspond to a positive evaluation label: "
                f"protein={protein!r}, go_index={go_index}"
            )
        key = (row_index, go_index, code)
        if key in seen:
            continue
        seen.add(key)
        by_code.setdefault(code, []).append((row_index, go_index))

    strong_positive = np.zeros_like(y_true, dtype=bool)
    code_counts: dict[str, int] = {}
    code_protein_slices: dict[str, Any] = {}
    for code, pairs in sorted(by_code.items()):
        code_counts[code] = len(pairs)
        if code in strong_codes:
            row_mask = np.zeros(len(proteins), dtype=bool)
            for row_index, go_index in pairs:
                strong_positive[row_index, go_index] = True
                row_mask[row_index] = True
            code_protein_slices[code] = _row_slice_metric_bundle(
                y_true=y_true,
                pred_map=pred_map,
                row_mask=row_mask,
                threshold_step=threshold_step,
                auprc_mode=auprc_mode,
                metrics_are_percent=metrics_are_percent,
                stage1_evaluator=stage1_evaluator,
            )

    strong_rows = np.any(strong_positive, axis=1)
    total_positive = int(np.count_nonzero(y_true))
    strong_count = int(np.count_nonzero(strong_positive))
    # Exclude known non-EXP/IDA positives from this flattened diagnostic so
    # that they are not incorrectly counted as false positives.
    evaluable_positions = (~y_true) | strong_positive
    position_diagnostics = {
        name: _masked_metric(
            _compute_metric_pack,
            y_true,
            prediction,
            evaluable_positions,
            threshold_step=threshold_step,
            auprc_mode=auprc_mode,
            metrics_are_percent=metrics_are_percent,
        )
        for name, prediction in pred_map.items()
    }
    return {
        "status": "loaded",
        "path": str(path),
        "sha256": _sha256(path),
        "strong_codes": list(strong_codes),
        "code_positive_counts": code_counts,
        "num_positive_labels": total_positive,
        "num_positive_labels_with_any_evidence_row": int(len({(r, g) for r, g, _ in seen})),
        "num_strong_positive_labels": strong_count,
        "strong_positive_label_coverage": (
            0.0 if total_positive == 0 else float(strong_count / total_positive)
        ),
        "strong_evidence_protein_slice": _row_slice_metric_bundle(
            y_true=y_true,
            pred_map=pred_map,
            row_mask=strong_rows,
            threshold_step=threshold_step,
            auprc_mode=auprc_mode,
            metrics_are_percent=metrics_are_percent,
            stage1_evaluator=stage1_evaluator,
        ),
        "per_code_protein_slices": code_protein_slices,
        "strong_positive_position_micro_diagnostics": position_diagnostics,
        "claim_policy": (
            "strong conclusions are restricted to the EXP/IDA-backed protein slice; "
            "the position-level result is a flattened diagnostic that excludes known "
            "non-strong positives and is not the official Stage-1 metric"
        ),
    }


def _calibration_curve(
    y_true: np.ndarray,
    prediction: np.ndarray,
    *,
    mask: np.ndarray | None,
    num_bins: int,
) -> dict[str, Any]:
    if num_bins < 2:
        raise ValueError("calibration-bins must be at least 2")
    count = np.zeros(num_bins, dtype=np.int64)
    confidence_sum = np.zeros(num_bins, dtype=np.float64)
    positive_sum = np.zeros(num_bins, dtype=np.float64)
    squared_error_sum = 0.0
    selected_count = 0
    for row in range(y_true.shape[0]):
        row_probability = np.asarray(prediction[row], dtype=np.float32)
        row_truth = np.asarray(y_true[row], dtype=np.bool_)
        if mask is not None:
            row_mask = mask if mask.ndim == 1 else mask[row]
            row_probability = row_probability[row_mask]
            row_truth = row_truth[row_mask]
        if row_probability.size == 0:
            continue
        finite = np.isfinite(row_probability)
        if not np.all(finite):
            row_probability = row_probability[finite]
            row_truth = row_truth[finite]
        clipped = np.clip(row_probability, 0.0, 1.0)
        index = np.minimum((clipped * num_bins).astype(np.int64), num_bins - 1)
        count += np.bincount(index, minlength=num_bins)
        confidence_sum += np.bincount(
            index, weights=clipped.astype(np.float64), minlength=num_bins
        )
        positive_sum += np.bincount(
            index, weights=row_truth.astype(np.float64), minlength=num_bins
        )
        squared_error_sum += float(
            np.square(clipped.astype(np.float64) - row_truth).sum(dtype=np.float64)
        )
        selected_count += int(clipped.size)
    nonzero = count > 0
    mean_confidence = np.zeros(num_bins, dtype=np.float64)
    observed_rate = np.zeros(num_bins, dtype=np.float64)
    mean_confidence[nonzero] = confidence_sum[nonzero] / count[nonzero]
    observed_rate[nonzero] = positive_sum[nonzero] / count[nonzero]
    gap = np.abs(mean_confidence - observed_rate)
    ece = (
        0.0
        if selected_count == 0
        else float(np.sum(count * gap, dtype=np.float64) / selected_count)
    )
    return {
        "num_positions": selected_count,
        "num_positive_labels": int(positive_sum.sum()),
        "positive_rate": (
            0.0 if selected_count == 0 else float(positive_sum.sum() / selected_count)
        ),
        "ece_equal_width": ece,
        "mce_equal_width": 0.0 if not np.any(nonzero) else float(gap[nonzero].max()),
        "brier_score": (
            0.0 if selected_count == 0 else float(squared_error_sum / selected_count)
        ),
        "bins": [
            {
                "index": index,
                "lower": float(index / num_bins),
                "upper": float((index + 1) / num_bins),
                "count": int(count[index]),
                "mean_confidence": (
                    None if count[index] == 0 else float(mean_confidence[index])
                ),
                "observed_positive_rate": (
                    None if count[index] == 0 else float(observed_rate[index])
                ),
                "absolute_gap": None if count[index] == 0 else float(gap[index]),
            }
            for index in range(num_bins)
        ],
    }


def _calibration_analysis(
    *,
    y_true: np.ndarray,
    pred_map: Mapping[str, np.ndarray],
    candidate_mask: np.ndarray | None,
    eligible_mask: np.ndarray | None,
    num_bins: int,
) -> dict[str, Any]:
    scopes: dict[str, np.ndarray | None] = {"overall": None}
    if candidate_mask is not None:
        scopes["candidate_evidence"] = candidate_mask
        scopes["non_candidate"] = ~candidate_mask
    if eligible_mask is not None:
        scopes["eligible_direct_query"] = eligible_mask
        scopes["context_only"] = ~eligible_mask
    return {
        "contract": {
            "binning": "equal_width_probability_bins",
            "num_bins": num_bins,
            "scope_note": (
                "candidate/query scopes are evaluator-only evidence-channel slices; "
                "ECE is descriptive under severe multilabel class imbalance and "
                "inherits the benchmark convention that unlabelled positions are "
                "treated as negatives despite possible annotation incompleteness"
            ),
        },
        "models": {
            model_name: {
                scope_name: _calibration_curve(
                    y_true, prediction, mask=scope_mask, num_bins=num_bins
                )
                for scope_name, scope_mask in scopes.items()
            }
            for model_name, prediction in pred_map.items()
        },
    }


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def _write_reliability_outputs(
    calibration: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "nbs_reliability_diagram.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "scope",
                "bin_index",
                "lower",
                "upper",
                "count",
                "mean_confidence",
                "observed_positive_rate",
                "absolute_gap",
            ]
        )
        for model_name, scopes in calibration.get("models", {}).items():
            for scope_name, curve in scopes.items():
                for item in curve.get("bins", []):
                    writer.writerow(
                        [
                            model_name,
                            scope_name,
                            item["index"],
                            item["lower"],
                            item["upper"],
                            item["count"],
                            item["mean_confidence"],
                            item["observed_positive_rate"],
                            item["absolute_gap"],
                        ]
                    )

    svg_paths: list[str] = []
    colors = ["#0068B5", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
    for model_name, scopes in calibration.get("models", {}).items():
        width = height = 620
        left, top, plot = 78, 48, 480
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">Reliability: {_slug(model_name)}</text>',
            f'<line x1="{left}" y1="{top+plot}" x2="{left+plot}" y2="{top}" stroke="#999" stroke-dasharray="5,5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot}" stroke="black"/>',
            f'<line x1="{left}" y1="{top+plot}" x2="{left+plot}" y2="{top+plot}" stroke="black"/>',
            f'<text x="{left+plot/2}" y="{top+plot+48}" text-anchor="middle" font-family="sans-serif" font-size="13">Mean predicted probability</text>',
            f'<text x="20" y="{top+plot/2}" text-anchor="middle" transform="rotate(-90 20 {top+plot/2})" font-family="sans-serif" font-size="13">Observed positive rate</text>',
        ]
        for tick in range(6):
            value = tick / 5
            x = left + value * plot
            y = top + (1.0 - value) * plot
            lines.extend(
                [
                    f'<line x1="{x}" y1="{top+plot}" x2="{x}" y2="{top+plot+5}" stroke="black"/>',
                    f'<text x="{x}" y="{top+plot+20}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.1f}</text>',
                    f'<line x1="{left-5}" y1="{y}" x2="{left}" y2="{y}" stroke="black"/>',
                    f'<text x="{left-10}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="10">{value:.1f}</text>',
                ]
            )
        legend_y = top + 14
        for scope_index, (scope_name, curve) in enumerate(scopes.items()):
            points = [
                (float(item["mean_confidence"]), float(item["observed_positive_rate"]))
                for item in curve.get("bins", [])
                if item.get("mean_confidence") is not None
                and item.get("observed_positive_rate") is not None
            ]
            color = colors[scope_index % len(colors)]
            if points:
                coordinates = " ".join(
                    f"{left+x*plot:.2f},{top+(1.0-y)*plot:.2f}" for x, y in points
                )
                lines.append(
                    f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
                )
                for x, y in points:
                    lines.append(
                        f'<circle cx="{left+x*plot:.2f}" cy="{top+(1.0-y)*plot:.2f}" r="2.5" fill="{color}"/>'
                    )
            legend_x = left + plot - 178
            lines.extend(
                [
                    f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+18}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>',
                    f'<text x="{legend_x+24}" y="{legend_y+4}" font-family="sans-serif" font-size="10">{_slug(scope_name)}</text>',
                ]
            )
            legend_y += 15
        lines.append("</svg>")
        svg_path = output_dir / f"reliability_{_slug(model_name)}.svg"
        svg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        svg_paths.append(str(svg_path))
    return {"csv": str(csv_path), "svg": svg_paths}


def _pearson_from_sums(
    count: int,
    sum_x: float,
    sum_y: float,
    sum_xx: float,
    sum_yy: float,
    sum_xy: float,
) -> float | None:
    if count < 2:
        return None
    covariance = sum_xy - sum_x * sum_y / count
    variance_x = sum_xx - sum_x * sum_x / count
    variance_y = sum_yy - sum_y * sum_y / count
    denominator = math.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
    return None if denominator <= 0.0 else float(covariance / denominator)


def _pearson(values_x: np.ndarray, values_y: np.ndarray) -> float | None:
    x = np.asarray(values_x, dtype=np.float64).reshape(-1)
    y = np.asarray(values_y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        return None
    return _pearson_from_sums(
        int(x.size),
        float(x.sum()),
        float(y.sum()),
        float(np.square(x).sum()),
        float(np.square(y).sum()),
        float((x * y).sum()),
    )


def _rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _student_expert_error_correlation(
    *,
    y_true: np.ndarray,
    student: np.ndarray | None,
    expert: np.ndarray | None,
    candidate_mask: np.ndarray | None,
) -> dict[str, Any]:
    if student is None or expert is None:
        return {
            "status": "not_available",
            "reason": "both backbone_base and modelout_reference/3B expert are required",
        }
    scope_stats: dict[str, dict[str, list[float]]] = {
        "overall": {"signed_error": [0, 0, 0, 0, 0, 0], "squared_error": [0, 0, 0, 0, 0, 0]},
        "positive_labels": {"signed_error": [0, 0, 0, 0, 0, 0], "squared_error": [0, 0, 0, 0, 0, 0]},
        "unlabelled_positions": {"signed_error": [0, 0, 0, 0, 0, 0], "squared_error": [0, 0, 0, 0, 0, 0]},
    }
    if candidate_mask is not None:
        scope_stats["candidate_evidence"] = {
            "signed_error": [0, 0, 0, 0, 0, 0],
            "squared_error": [0, 0, 0, 0, 0, 0],
        }
        scope_stats["non_candidate"] = {
            "signed_error": [0, 0, 0, 0, 0, 0],
            "squared_error": [0, 0, 0, 0, 0, 0],
        }
    student_brier = np.zeros(y_true.shape[0], dtype=np.float64)
    expert_brier = np.zeros(y_true.shape[0], dtype=np.float64)
    student_mae = np.zeros(y_true.shape[0], dtype=np.float64)
    expert_mae = np.zeros(y_true.shape[0], dtype=np.float64)

    def update(accumulator: list[float], x: np.ndarray, y: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64).reshape(-1)
        y64 = np.asarray(y, dtype=np.float64).reshape(-1)
        accumulator[0] += int(x64.size)
        accumulator[1] += float(x64.sum())
        accumulator[2] += float(y64.sum())
        accumulator[3] += float(np.square(x64).sum())
        accumulator[4] += float(np.square(y64).sum())
        accumulator[5] += float((x64 * y64).sum())

    for row in range(y_true.shape[0]):
        truth = np.asarray(y_true[row], dtype=np.float32)
        student_error = np.asarray(student[row], dtype=np.float32) - truth
        expert_error = np.asarray(expert[row], dtype=np.float32) - truth
        student_squared = np.square(student_error, dtype=np.float32)
        expert_squared = np.square(expert_error, dtype=np.float32)
        student_brier[row] = float(student_squared.mean(dtype=np.float64))
        expert_brier[row] = float(expert_squared.mean(dtype=np.float64))
        student_mae[row] = float(np.abs(student_error).mean(dtype=np.float64))
        expert_mae[row] = float(np.abs(expert_error).mean(dtype=np.float64))
        row_scopes: dict[str, np.ndarray | slice] = {
            "overall": slice(None),
            "positive_labels": y_true[row],
            "unlabelled_positions": ~y_true[row],
        }
        if candidate_mask is not None:
            row_scopes["candidate_evidence"] = candidate_mask[row]
            row_scopes["non_candidate"] = ~candidate_mask[row]
        for scope_name, selection in row_scopes.items():
            update(
                scope_stats[scope_name]["signed_error"],
                student_error[selection],
                expert_error[selection],
            )
            update(
                scope_stats[scope_name]["squared_error"],
                student_squared[selection],
                expert_squared[selection],
            )

    pointwise: dict[str, Any] = {}
    for scope_name, kinds in scope_stats.items():
        pointwise[scope_name] = {}
        for kind, values in kinds.items():
            count, sum_x, sum_y, sum_xx, sum_yy, sum_xy = values
            pointwise[scope_name][kind] = {
                "num_positions": int(count),
                "pearson": _pearson_from_sums(
                    int(count), sum_x, sum_y, sum_xx, sum_yy, sum_xy
                ),
            }
    return {
        "status": "ok",
        "student": "backbone_base",
        "expert": "modelout_reference",
        "per_protein": {
            "brier": {
                "pearson": _pearson(student_brier, expert_brier),
                "spearman": _pearson(
                    _rank_values(student_brier), _rank_values(expert_brier)
                ),
                "student": _summary(student_brier),
                "expert": _summary(expert_brier),
            },
            "mae": {
                "pearson": _pearson(student_mae, expert_mae),
                "spearman": _pearson(
                    _rank_values(student_mae), _rank_values(expert_mae)
                ),
                "student": _summary(student_mae),
                "expert": _summary(expert_mae),
            },
        },
        "pointwise": pointwise,
        "interpretation": (
            "high positive correlation means the student and 3B expert tend to fail "
            "on the same proteins/positions; it is not evidence of causal dependence"
        ),
    }


def _histogram_metric_from_counts(
    pred_hist: np.ndarray,
    tp_hist: np.ndarray,
    *,
    total_positive: float,
    threshold_step: float,
) -> tuple[float, float]:
    pred_cum = np.cumsum(pred_hist[::-1], dtype=np.float64)[::-1]
    tp_cum = np.cumsum(tp_hist[::-1], dtype=np.float64)[::-1]
    precision = tp_cum / np.maximum(pred_cum, 1.0)
    recall = tp_cum / max(float(total_positive), 1.0)
    fscore = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    order = np.arange(pred_hist.size - 1, -1, -1)
    previous_recall = 0.0
    approximate_ap = 0.0
    for current_recall, current_precision in zip(recall[order], precision[order]):
        if current_recall > previous_recall:
            approximate_ap += (current_recall - previous_recall) * current_precision
            previous_recall = float(current_recall)
    return float(np.max(fscore)), float(approximate_ap)


def _per_protein_histograms(
    y_true: np.ndarray,
    prediction: np.ndarray,
    *,
    threshold_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nbins = int(round(1.0 / threshold_step)) + 1
    pred_hist = np.zeros((y_true.shape[0], nbins), dtype=np.float64)
    tp_hist = np.zeros((y_true.shape[0], nbins), dtype=np.float64)
    positives = np.zeros(y_true.shape[0], dtype=np.float64)
    for row in range(y_true.shape[0]):
        probability = np.asarray(prediction[row], dtype=np.float32)
        index = np.floor(probability / threshold_step).astype(np.int32)
        index = np.clip(index, 0, nbins - 1)
        truth = np.asarray(y_true[row], dtype=bool)
        pred_hist[row] = np.bincount(index, minlength=nbins)
        tp_hist[row] = np.bincount(index[truth], minlength=nbins)
        positives[row] = float(np.count_nonzero(truth))
    return pred_hist, tp_hist, positives


def _distribution_interval(
    values: np.ndarray, *, alpha: float, scale: float
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "replicates": int(values.size),
        "estimate_mean": float(values.mean() * scale),
        "standard_error": float(values.std(ddof=1) * scale),
        "ci_lower": float(lower * scale),
        "ci_upper": float(upper * scale),
    }


def _paired_bootstrap_micro_ci(
    *,
    y_true: np.ndarray,
    nbs_prediction: np.ndarray | None,
    backbone_prediction: np.ndarray | None,
    threshold_step: float,
    replicates: int,
    seed: int,
    alpha: float,
    target_power: float,
    metrics_are_percent: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    if nbs_prediction is None or backbone_prediction is None or replicates <= 0:
        return {
            "status": "not_available",
            "reason": "NBS, backbone and a positive bootstrap replicate count are required",
        }, None
    if replicates < 2:
        raise ValueError("bootstrap-replicates must be 0 (disabled) or at least 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("ci-alpha must lie in (0,1)")
    if not 0.5 < target_power < 1.0:
        raise ValueError("target-power must lie in (0.5,1)")
    nbs_hist = _per_protein_histograms(
        y_true, nbs_prediction, threshold_step=threshold_step
    )
    base_hist = _per_protein_histograms(
        y_true, backbone_prediction, threshold_step=threshold_step
    )
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        y_true.shape[0],
        np.full(y_true.shape[0], 1.0 / y_true.shape[0]),
        size=replicates,
    ).astype(np.float64)
    distributions: dict[str, dict[str, np.ndarray]] = {}
    for name, (pred_hist, tp_hist, positives) in {
        "NBS_final": nbs_hist,
        "backbone_base": base_hist,
    }.items():
        aggregate_pred = weights @ pred_hist
        aggregate_tp = weights @ tp_hist
        aggregate_positive = weights @ positives
        fmax = np.zeros(replicates, dtype=np.float64)
        auprc = np.zeros(replicates, dtype=np.float64)
        for index in range(replicates):
            fmax[index], auprc[index] = _histogram_metric_from_counts(
                aggregate_pred[index],
                aggregate_tp[index],
                total_positive=aggregate_positive[index],
                threshold_step=threshold_step,
            )
        distributions[name] = {"fmax_micro": fmax, "auprc_micro_hist": auprc}
    scale = 100.0 if metrics_are_percent else 1.0
    output: dict[str, Any] = {
        "status": "ok",
        "contract": {
            "unit": "protein-row paired nonparametric bootstrap",
            "replicates": replicates,
            "seed": seed,
            "alpha": alpha,
            "target_power": target_power,
            "metric_scope": (
                "flattened-position histogram micro diagnostics, not an official "
                "evalperf_torch confidence interval"
            ),
        },
        "models": {},
        "paired_delta_NBS_minus_backbone": {},
        "minimum_detectable_delta": {},
    }
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(target_power)
    for metric_name in ("fmax_micro", "auprc_micro_hist"):
        nbs_values = distributions["NBS_final"][metric_name]
        base_values = distributions["backbone_base"][metric_name]
        delta = nbs_values - base_values
        output["models"].setdefault("NBS_final", {})[metric_name] = _distribution_interval(
            nbs_values, alpha=alpha, scale=scale
        )
        output["models"].setdefault("backbone_base", {})[metric_name] = _distribution_interval(
            base_values, alpha=alpha, scale=scale
        )
        output["paired_delta_NBS_minus_backbone"][metric_name] = {
            **_distribution_interval(delta, alpha=alpha, scale=scale),
            "probability_delta_gt_zero": float(np.mean(delta > 0.0)),
        }
        standard_error = float(delta.std(ddof=1) * scale)
        output["minimum_detectable_delta"][metric_name] = {
            "two_sided_normal_approximation": float(
                (z_alpha + z_power) * standard_error
            ),
            "unit": "percentage_points" if metrics_are_percent else "fraction",
            "note": (
                "observed-data sensitivity diagnostic; not a substitute for a "
                "prospective sample-size calculation"
            ),
        }
    return output, weights


def _parse_precision_k(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    values = tuple(sorted(set(int(item.strip()) for item in value.split(",") if item.strip())))
    if not values or values[0] <= 0:
        raise ValueError("precision-k must be a comma-separated list of positive integers")
    return values


def _per_protein_ranking_metrics(
    y_true: np.ndarray, prediction: np.ndarray, ks: tuple[int, ...]
) -> dict[int, dict[str, np.ndarray]]:
    if not ks:
        return {}
    if ks[-1] > prediction.shape[1]:
        raise ValueError("precision-k exceeds the task classifier width")
    output = {
        k: {
            "precision": np.zeros(y_true.shape[0], dtype=np.float64),
            "recall": np.zeros(y_true.shape[0], dtype=np.float64),
        }
        for k in ks
    }
    for row in range(y_true.shape[0]):
        probability = np.asarray(prediction[row], dtype=np.float32)
        top = np.argpartition(probability, -ks[-1])[-ks[-1] :]
        top = top[np.argsort(probability[top], kind="mergesort")[::-1]]
        hits = np.asarray(y_true[row, top], dtype=np.int64)
        cumulative = np.cumsum(hits)
        positives = int(np.count_nonzero(y_true[row]))
        for k in ks:
            output[k]["precision"][row] = float(cumulative[k - 1] / k)
            output[k]["recall"][row] = (
                0.0 if positives == 0 else float(cumulative[k - 1] / positives)
            )
    return output


def _ranking_analysis(
    *,
    y_true: np.ndarray,
    pred_map: Mapping[str, np.ndarray],
    ks: tuple[int, ...],
    bootstrap_weights: np.ndarray | None,
    alpha: float,
    metrics_are_percent: bool,
) -> dict[str, Any]:
    if not ks:
        return {
            "status": "not_requested",
            "note": "predeclare precision@k values with --precision-k before evaluation",
        }
    scale = 100.0 if metrics_are_percent else 1.0
    per_model = {
        model_name: _per_protein_ranking_metrics(y_true, prediction, ks)
        for model_name, prediction in pred_map.items()
    }
    output: dict[str, Any] = {
        "status": "ok",
        "contract": {
            "aggregation": "macro mean across proteins",
            "ks": list(ks),
            "zero_positive_protein_recall": 0.0,
            "ci": (
                "same paired protein-bootstrap draws as micro CI"
                if bootstrap_weights is not None
                else "not computed"
            ),
        },
        "models": {},
        "paired_delta_NBS_minus_backbone": {},
    }
    for model_name, by_k in per_model.items():
        output["models"][model_name] = {}
        for k, values in by_k.items():
            output["models"][model_name][str(k)] = {}
            for metric_name, per_protein in values.items():
                entry: dict[str, Any] = {
                    "mean": float(per_protein.mean() * scale),
                    "num_proteins": int(per_protein.size),
                }
                if bootstrap_weights is not None:
                    distribution = (
                        bootstrap_weights @ per_protein / y_true.shape[0]
                    )
                    entry.update(
                        _distribution_interval(distribution, alpha=alpha, scale=scale)
                    )
                output["models"][model_name][str(k)][f"{metric_name}_at_k"] = entry
    if "NBS_final" in per_model and "backbone_base" in per_model:
        for k in ks:
            output["paired_delta_NBS_minus_backbone"][str(k)] = {}
            for metric_name in ("precision", "recall"):
                delta = (
                    per_model["NBS_final"][k][metric_name]
                    - per_model["backbone_base"][k][metric_name]
                )
                entry = {"mean": float(delta.mean() * scale)}
                if bootstrap_weights is not None:
                    distribution = bootstrap_weights @ delta / y_true.shape[0]
                    entry.update(
                        {
                            **_distribution_interval(
                                distribution, alpha=alpha, scale=scale
                            ),
                            "probability_delta_gt_zero": float(
                                np.mean(distribution > 0.0)
                            ),
                        }
                    )
                output["paired_delta_NBS_minus_backbone"][str(k)][
                    f"{metric_name}_at_k"
                ] = entry
    return output


def _metric_deltas(
    metrics: Mapping[str, Mapping[str, float]], *, baseline: str
) -> dict[str, Any]:
    if baseline not in metrics:
        return {}
    base = metrics[baseline]
    output: dict[str, Any] = {}
    for model_name, values in metrics.items():
        if model_name == baseline:
            continue
        output[model_name] = {
            key: float(value) - float(base[key])
            for key, value in values.items()
            if key in base
            and isinstance(value, (int, float))
            and isinstance(base[key], (int, float))
            and key.lower().startswith(("fmax", "auprc", "precision", "recall"))
        }
    return output


def _write_primary_table(
    *,
    metrics: Mapping[str, Mapping[str, float]],
    output_dir: Path,
) -> str:
    keys = sorted({key for values in metrics.values() for key in values})
    path = output_dir / "nbs_primary_comparison.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["model", *keys])
        for model_name, values in metrics.items():
            writer.writerow([model_name, *(values.get(key, "") for key in keys)])
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate precomputed NBS independent-test predictions. Formal overall "
            "metrics call the exact Stage-1 evalperf_torch implementation; separately "
            "named micro metrics are retained for masks and diagnostics."
        )
    )
    parser.add_argument("--task", choices=["bp", "mf", "cc"], required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--nbs-prob", type=Path, required=True)
    parser.add_argument("--backbone-prob", type=Path, default=None)
    parser.add_argument("--modelout-prob", type=Path, default=None)
    parser.add_argument(
        "--expert-3b-prob",
        type=Path,
        default=None,
        help="explicit alias for the non-deployable 3B/modelout reference",
    )
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        default=None,
        help=(
            "JSON mapping external method names to complete-task probability arrays "
            "and optional protein-id files"
        ),
    )
    parser.add_argument(
        "--expected-comparisons",
        default=",".join(EXPECTED_EXTERNAL_COMPARISONS),
        help="comma-separated external controls expected in the formal report",
    )
    parser.add_argument("--require-expected-comparisons", action="store_true")
    parser.add_argument("--protein-ids", type=Path, default=None)
    parser.add_argument(
        "--protein-strata",
        type=Path,
        default=None,
        help=(
            "TSV/CSV keyed by protein_id with raw or categorical sequence-identity, "
            "MSA-Neff, Foldseek and evidence-tier columns"
        ),
    )
    parser.add_argument(
        "--label-evidence",
        type=Path,
        default=None,
        help="TSV/CSV positive-label table: protein_id, go_index, evidence_code",
    )
    parser.add_argument("--strong-evidence-codes", default="EXP,IDA")
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
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=6061)
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument(
        "--precision-k",
        default="",
        help=(
            "predeclared comma-separated k values; empty disables ranking metrics "
            "until k is fixed without looking at ind_test"
        ),
    )
    parser.add_argument(
        "--auprc-mode",
        choices=["hist", "exact", "none"],
        default="hist",
        help=(
            "micro-diagnostic AUPRC backend; formal overall AUPRC comes from "
            "evalperf_torch when --metric-backend=stage1"
        ),
    )
    parser.add_argument(
        "--metric-backend",
        choices=["stage1", "local_micro"],
        default="stage1",
        help=(
            "stage1 is required for the primary report; local_micro is an explicitly "
            "different flattened-position diagnostic backend"
        ),
    )
    parser.add_argument("--metrics-are-percent", action="store_true", default=True)
    args = parser.parse_args()

    if args.modelout_prob is not None and args.expert_3b_prob is not None:
        if args.modelout_prob.resolve() != args.expert_3b_prob.resolve():
            raise ValueError("--modelout-prob and --expert-3b-prob name different files")
    expert_probability_path = args.expert_3b_prob or args.modelout_prob
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        ("modelout_reference", expert_probability_path),
    ):
        if path is None:
            continue
        arr = _load_prediction(path, num_classes)
        arr = _align_rows(arr, prediction_ids=prediction_ids, metadata_ids=proteins)
        pred_map[key] = np.asarray(arr, dtype=np.float32)

    external_predictions, comparison_contract = _load_external_comparisons(
        args.comparison_manifest,
        num_classes=num_classes,
        metadata_ids=proteins,
        go_registry_path=args.go_registry,
    )
    pred_map.update(external_predictions)
    expected_comparisons = tuple(
        item.strip() for item in args.expected_comparisons.split(",") if item.strip()
    )
    missing_comparisons = [
        name for name in expected_comparisons if name not in external_predictions
    ]
    comparison_contract.update(
        {
            "expected_external_methods": list(expected_comparisons),
            "present_external_methods": sorted(external_predictions),
            "missing_external_methods": missing_comparisons,
            "complete": not missing_comparisons,
        }
    )
    if args.require_expected_comparisons and missing_comparisons:
        raise RuntimeError(
            "formal comparison manifest is incomplete: " + ", ".join(missing_comparisons)
        )
    if args.require_expected_comparisons:
        unverified = [
            name
            for name in expected_comparisons
            if not comparison_contract["methods"][name]["protein_order_verified"]
            or not comparison_contract["methods"][name]["go_order_verified"]
        ]
        if unverified:
            raise RuntimeError(
                "formal comparison rows/GO order are not fully verified for: "
                + ", ".join(unverified)
            )

    micro_metrics = {
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
    stage1_helper_path = None
    stage1_helper_sha256 = None
    stage1_evaluator: Callable[..., Any] | None = None
    if args.metric_backend == "stage1":
        stage1_evaluator, stage1_helper_path = _load_stage1_metric_backend()
        stage1_helper_sha256 = _sha256(stage1_helper_path)
        metrics = {
            key: _compute_stage1_metric_pack(stage1_evaluator, y_true, pred)
            for key, pred in pred_map.items()
        }
    else:
        metrics = micro_metrics

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

    strata_masks, strata_contract = _load_protein_strata(
        args.protein_strata,
        proteins=proteins,
    )
    protein_strata_analysis = _protein_strata_analysis(
        y_true=y_true,
        pred_map=pred_map,
        strata_masks=strata_masks,
        threshold_step=float(args.threshold_step),
        auprc_mode=("hist" if args.auprc_mode != "none" else "none"),
        metrics_are_percent=bool(args.metrics_are_percent),
        stage1_evaluator=stage1_evaluator,
    )
    label_evidence_analysis = _load_label_evidence(
        args.label_evidence,
        proteins=proteins,
        y_true=y_true,
        strong_codes=_parse_codes(args.strong_evidence_codes),
        pred_map=pred_map,
        threshold_step=float(args.threshold_step),
        auprc_mode=("hist" if args.auprc_mode != "none" else "none"),
        metrics_are_percent=bool(args.metrics_are_percent),
        stage1_evaluator=stage1_evaluator,
    )

    candidate_analysis: dict[str, Any] = {}
    candidate_coverage: dict[str, Any] = {}
    candidate_mask = None
    if args.candidate_go_index is not None:
        candidate_mask = _candidate_mask(
            args.candidate_go_index, len(proteins), num_classes
        )
        total_positive = int(np.count_nonzero(y_true))
        selected_positive = int(np.count_nonzero(y_true & candidate_mask))
        candidate_positions = int(np.count_nonzero(candidate_mask))
        total_positions = int(y_true.size)
        full_space_positive_rate = (
            0.0 if total_positions == 0 else float(total_positive / total_positions)
        )
        candidate_positive_rate = (
            0.0 if candidate_positions == 0 else float(selected_positive / candidate_positions)
        )
        candidate_coverage = {
            "selector_scope": (
                None
                if input_manifest is None
                else input_manifest.get("artifacts", {})
                .get("candidate_evidence", {})
                .get("selector_scope")
            ),
            "num_candidate_positions": candidate_positions,
            "candidate_position_fraction": float(candidate_mask.mean()),
            "num_positive_labels": total_positive,
            "num_positive_labels_in_candidates": selected_positive,
            "positive_label_recall": (
                0.0
                if total_positive == 0
                else float(selected_positive / total_positive)
            ),
            "candidate_positive_rate": candidate_positive_rate,
            "full_space_positive_rate": full_space_positive_rate,
            "candidate_positive_enrichment": (
                None
                if full_space_positive_rate == 0.0
                else float(candidate_positive_rate / full_space_positive_rate)
            ),
            "selected_and_positive_fraction_of_full_space": (
                0.0
                if total_positions == 0
                else float(selected_positive / total_positions)
            ),
            "selection_bias_note": (
                "candidate_positive_rate is conditional on deterministic top-K "
                "selection and is not an unbiased population confirmation rate; "
                "coverage and the complete-task result must be reported alongside it"
            ),
            "frequency": {},
            "protein_strata": {},
        }
        for bin_name, column_mask in bins.items():
            column_mask = np.asarray(column_mask, dtype=bool)
            bin_truth = y_true[:, column_mask]
            bin_candidate = candidate_mask[:, column_mask]
            bin_total_positive = int(np.count_nonzero(bin_truth))
            bin_selected_positive = int(
                np.count_nonzero(bin_truth & bin_candidate)
            )
            candidate_coverage["frequency"][bin_name] = {
                "num_positive_labels": bin_total_positive,
                "num_positive_labels_in_candidates": bin_selected_positive,
                "positive_label_recall": (
                    0.0
                    if bin_total_positive == 0
                    else float(bin_selected_positive / bin_total_positive)
                ),
                "num_candidate_positions": int(np.count_nonzero(bin_candidate)),
            }
        for dimension, dimension_bins in strata_masks.items():
            candidate_coverage["protein_strata"][dimension] = {}
            for bin_name, row_mask in dimension_bins.items():
                stratum_truth = y_true[row_mask]
                stratum_candidate = candidate_mask[row_mask]
                stratum_total = int(np.count_nonzero(stratum_truth))
                stratum_selected = int(
                    np.count_nonzero(stratum_truth & stratum_candidate)
                )
                candidate_coverage["protein_strata"][dimension][bin_name] = {
                    "num_proteins": int(np.count_nonzero(row_mask)),
                    "num_positive_labels": stratum_total,
                    "num_positive_labels_in_candidates": stratum_selected,
                    "positive_label_recall": (
                        0.0
                        if stratum_total == 0
                        else float(stratum_selected / stratum_total)
                    ),
                    "num_candidate_positions": int(
                        np.count_nonzero(stratum_candidate)
                    ),
                }
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

    calibration = _calibration_analysis(
        y_true=y_true,
        pred_map=pred_map,
        candidate_mask=candidate_mask,
        eligible_mask=eligible_mask,
        num_bins=int(args.calibration_bins),
    )
    reliability_artifacts = _write_reliability_outputs(
        calibration, args.output_dir
    )
    student_expert_correlation = _student_expert_error_correlation(
        y_true=y_true,
        student=pred_map.get("backbone_base"),
        expert=pred_map.get("modelout_reference"),
        candidate_mask=candidate_mask,
    )
    confidence_intervals, bootstrap_weights = _paired_bootstrap_micro_ci(
        y_true=y_true,
        nbs_prediction=pred_map.get("NBS_final"),
        backbone_prediction=pred_map.get("backbone_base"),
        threshold_step=float(args.threshold_step),
        replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed),
        alpha=float(args.ci_alpha),
        target_power=float(args.target_power),
        metrics_are_percent=bool(args.metrics_are_percent),
    )
    ranking_analysis = _ranking_analysis(
        y_true=y_true,
        pred_map=pred_map,
        ks=_parse_precision_k(args.precision_k),
        bootstrap_weights=bootstrap_weights,
        alpha=float(args.ci_alpha),
        metrics_are_percent=bool(args.metrics_are_percent),
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

    metric_deltas_vs_backbone = _metric_deltas(metrics, baseline="backbone_base")
    primary_table_path = _write_primary_table(
        metrics=metrics,
        output_dir=args.output_dir,
    )
    prediction_sources: dict[str, Any] = {
        "NBS_final": str(args.nbs_prob),
        "backbone_base": (
            None if args.backbone_prob is None else str(args.backbone_prob)
        ),
        "modelout_reference": (
            None if expert_probability_path is None else str(expert_probability_path)
        ),
    }
    prediction_sources.update(
        {
            name: value["probability"]
            for name, value in comparison_contract.get("methods", {}).items()
        }
    )

    result = {
        "schema_version": 4,
        "evaluation_version": EVALUATION_VERSION,
        "task": args.task,
        "mode": "ind_test",
        "num_samples": len(proteins),
        "num_classes": num_classes,
        "label_key": label_key,
        "metrics": metrics,
        "metric_deltas_vs_backbone": metric_deltas_vs_backbone,
        "metric_contract": {
            "primary_backend": str(args.metric_backend),
            "primary_scope": "overall_N_by_task_GO_matrix",
            "stage1_helper": (
                None if stage1_helper_path is None else str(stage1_helper_path)
            ),
            "stage1_helper_sha256": stage1_helper_sha256,
            "micro_diagnostics": (
                "flattened label-position threshold-grid Fmax and micro-AUPRC; "
                "not a substitute for Stage-1 evalperf_torch in the primary comparison"
            ),
            "confidence_interval_scope": (
                "paired protein bootstrap is reported only for histogram micro "
                "diagnostics; it is not presented as an evalperf_torch CI"
            ),
        },
        "micro_diagnostics": micro_metrics,
        "frequency": {
            "count_source": count_source,
            "num_train_samples": n_train,
            "bin_sizes": {name: int(np.asarray(mask, dtype=bool).sum()) for name, mask in bins.items()},
        },
        "rare_analysis": rare,
        "protein_strata_contract": strata_contract,
        "protein_strata_analysis": protein_strata_analysis,
        "label_evidence_analysis": label_evidence_analysis,
        "candidate_analysis": candidate_analysis,
        "candidate_coverage": candidate_coverage,
        "query_scope_analysis": query_scope_analysis,
        "calibration": calibration,
        "confidence_intervals": confidence_intervals,
        "ranking_analysis": ranking_analysis,
        "student_expert_error_correlation": student_expert_correlation,
        "comparison_contract": comparison_contract,
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
        "prediction_sources": prediction_sources,
        "artifacts": {
            "primary_comparison_tsv": primary_table_path,
            "reliability": reliability_artifacts,
        },
        "claim_contract": {
            "primary_checkpoint_selection": (
                "must be fixed without using ind_test; ind_test is not an early-stop "
                "or best-epoch selection set"
            ),
            "strong_evidence": (
                "strong claims are restricted to EXP/IDA-backed Tier 0 when the "
                "label-evidence contract is available"
            ),
            "weaker_evidence": (
                "IEA/ISS/IBA-only slices are descriptive and do not support the "
                "same strength of claim"
            ),
            "candidate_space": (
                "top-512 candidate evidence is a sparse input channel, never the "
                "prediction vocabulary; all task GO columns are evaluated"
            ),
            "external_baselines": (
                "missing external controls are reported explicitly and may be made "
                "fatal with --require-expected-comparisons"
            ),
        },
        "note": (
            "modelout_reference may contain expert information and is a strong reference only; "
            "NBS_final must be produced without expert probability in the NBS forward pass."
        ),
    }
    out = args.output_dir / "nbs_ind_test_metrics.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    print(f"[Saved] {out}")


if __name__ == "__main__":
    main()
