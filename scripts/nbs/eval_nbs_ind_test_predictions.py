#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--auprc-mode", choices=["hist", "exact", "none"], default="exact")
    parser.add_argument("--metrics-are-percent", action="store_true", default=True)
    args = parser.parse_args()

    root = _project_root()
    sys.path.insert(0, str(root))
    from experiments.eval_weak_ind_test_detr_diagnostics import (  # pylint: disable=import-outside-toplevel
        compute_metric_pack,
        rare_analysis_metrics,
        load_train_label_counts,
    )
    from utils.util_functions import make_frequency_bins  # pylint: disable=import-outside-toplevel

    num_classes = {"bp": 21312, "mf": 7038, "cc": 2903}[args.task]
    with args.metadata_file.open("rb") as handle:
        metadata = pickle.load(handle)
    task_data = _task_dict(metadata, "ind_test", args.task)
    proteins = [str(x) for x in task_data.get("proteins", [])]
    if not proteins:
        raise ValueError("ind_test metadata has no proteins")
    label_key = "prop_annotations" if "prop_annotations" in task_data else "annotations"
    if label_key not in task_data:
        raise KeyError("ind_test metadata has neither prop_annotations nor annotations")
    y_true = _dense_labels(task_data[label_key], num_classes, len(proteins))

    prediction_ids = None if args.protein_ids is None else _read_ids(args.protein_ids)
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
        key: compute_metric_pack(
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
        counts, n_train, count_key = load_train_label_counts(args.metadata_file, args.task, num_classes)
        count_source = f"metadata:train.{count_key}"
    bins = make_frequency_bins(counts)
    rare = rare_analysis_metrics(
        y_true=y_true,
        pred_map=pred_map,
        class_bins=bins,
        keys=list(pred_map),
        threshold_step=float(args.threshold_step),
        auprc_mode=("hist" if args.auprc_mode != "none" else "none"),
        metrics_are_percent=bool(args.metrics_are_percent),
    )

    result = {
        "schema_version": 1,
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
