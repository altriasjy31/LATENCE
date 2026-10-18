#!/usr/bin/env python3
"""Score already-audited native v088/v089 predictions with v085 Fmax/PR-AUC/AP.

This NEVER edits the acceptance JSON. It requires an original Fmax-only
comparison report (scope + schema) and validates the prediction/metadata hashes
before producing a separate supplementary JSON/TSV.

Run from LATENCE project root, after the original native evaluation:
  python scripts/nbs/eval_nbs_v085_supplementary.py \
    --comparison /path/to/eval_step1920/full/metrics/nbs_w2s_comparison.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NBS_ROOT = ROOT / "nbs_models" / "nbs_protein_go"
if str(NBS_ROOT) not in sys.path:
    sys.path.insert(0, str(NBS_ROOT))

try:
    from nbs_pg.full_task_metrics_v085 import (
        METRIC_SCHEMA, compute_standard_metrics, metric_definitions,
        metrics_implementation_sha256,
    )
except ImportError as exc:
    raise RuntimeError("Requires the original nbs_pg/full_task_metrics_v085.py in the project") from exc

WIDTH = {"bp": 21312, "mf": 7038, "cc": 2903}
TASK = {"bp": "biological_process", "mf": "molecular_function", "cc": "cellular_component"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def required_file(path, description):
    if path is None:
        raise ValueError(f"Missing {description}")
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{description} is missing: {p}")
    return p


def verify_hash(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label}: no valid SHA256 in original report")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label}: SHA256 mismatch (original report vs current file): {path}")
    return actual


def read_ids(path):
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Protein IDs must be unique and nonempty: {path}")
    return ids


def annotation_matrix(meta, task, n_classes):
    split = meta.get("ind_test")
    if not isinstance(split, dict):
        raise ValueError("Original acceptance metadata lacks ind_test split")
    d = split.get(TASK[task], split.get(task))
    if not isinstance(d, dict):
        raise ValueError(f"Original acceptance metadata lacks {task} under ind_test")
    proteins = list(map(str, d["proteins"]))
    key = next((k for k in ("prop_annotations", "prop_annotation", "annotations", "labels") if k in d), None)
    if key is None:
        raise ValueError("Original metadata has no GO labels")
    annotations = list(d[key])
    if len(annotations) != len(proteins) or len(proteins) != len(set(proteins)):
        raise ValueError("Original metadata labels/protein lengths disagree, or protein IDs repeat")
    y = np.zeros((len(proteins), n_classes), dtype=np.uint8)
    for i, terms in enumerate(annotations):
        term_idx = np.asarray(list(terms), dtype=np.int64)
        if term_idx.size and (term_idx.min() < 0 or term_idx.max() >= n_classes):
            raise ValueError(f"Original gold GO indices out of range at protein {proteins[i]}")
        y[i, term_idx] = 1
    return proteins, y


def verify_report(report, report_path):
    if report.get("schema_version") != 3 or report.get("evaluation_scope") != "standard_protein_fmax_only":
        raise ValueError(
            "Supply a freshly recomputed ORIGINAL v088/v089 Fmax-only acceptance report "
            "(schema_version=3, evaluation_scope=standard_protein_fmax_only), not the v0.4.x upgraded report"
        )
    version = report.get("evaluation_version")
    if version not in ("0.8.8", "0.8.9.1"):
        raise ValueError(f"Unsupported evaluation version: {version}")
    scope = report.get("standardized_metric_contract", {})
    if scope.get("implementation_sha256") != metrics_implementation_sha256():
        raise ValueError("The installed native v085 metric kernel differs from the original acceptance kernel")
    if scope.get("threshold_step") != 0.001:
        raise ValueError("Original acceptance did not use the 0.001 protein Fmax grid")
    pred = report.get("prediction_provenance", {})
    metadata = report.get("metadata_provenance", {})
    paths = {
        "nbs": required_file(pred.get("probability_path"), "NBS predictions"),
        "backbone": required_file(pred.get("backbone_probability_path"), "backbone predictions"),
        "input_manifest": required_file(pred.get("input_manifest_path"), "input manifest"),
        "prediction_manifest": required_file(pred.get("prediction_manifest_path"), "prediction manifest"),
        "protein_ids": required_file(pred.get("protein_ids_path"), "input protein IDs"),
        "metadata": required_file(metadata.get("path"), "acceptance metadata"),
    }
    for name, expected in (
        ("nbs", pred.get("probability_sha256")),
        ("backbone", pred.get("backbone_probability_sha256")),
        ("input_manifest", pred.get("input_manifest_sha256")),
        ("prediction_manifest", pred.get("prediction_manifest_sha256")),
        ("protein_ids", pred.get("protein_ids_file_sha256")),
        ("metadata", metadata.get("sha256")),
    ):
        verify_hash(paths[name], expected, name)
    checkpoint_path = pred.get("checkpoint_path")
    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_absolute():
            checkpoint = paths["prediction_manifest"].parent / checkpoint
        if checkpoint.is_file():
            verify_hash(checkpoint, pred.get("checkpoint_sha256"), "checkpoint")
    predicted = json.loads(paths["prediction_manifest"].read_text(encoding="utf-8"))
    input_manifest = json.loads(paths["input_manifest"].read_text(encoding="utf-8"))
    if predicted.get("runner_version") != version:
        raise ValueError("Prediction manifest runner_version differs from acceptance report")
    if predicted.get("output_probability_sha256") != pred.get("probability_sha256"):
        raise ValueError("Prediction manifest probability hash differs")
    if predicted.get("input_manifest_sha256") != pred.get("input_manifest_sha256"):
        raise ValueError("Prediction manifest input hash differs")
    if predicted.get("checkpoint_sha256") != pred.get("checkpoint_sha256"):
        raise ValueError("Prediction manifest checkpoint hash differs")
    if predicted.get("prediction_space") != "complete_task_classifier_columns":
        raise ValueError("Prediction space is not the full original GO task axis")
    if predicted.get("uses_expert_probability_in_nbs_forward") is not False:
        raise ValueError("NBS forward used Expert probabilities")
    if predicted.get("output_fusion") != "none":
        raise ValueError("Expected standalone NBS probabilities (no output fusion)")
    if predicted.get("num_task_go") != WIDTH[report["task"]]:
        raise ValueError("Prediction task width differs from acceptance report")
    go_registry = pred.get("input_go_registry_path")
    if go_registry is not None:
        go_registry = required_file(go_registry, "GO registry")
        verify_hash(go_registry, pred.get("input_go_registry_sha256"), "GO registry")
        with go_registry.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if len(rows) != WIDTH[report["task"]] or [int(r["go_idx"]) for r in rows] != list(range(len(rows))):
            raise ValueError("GO registry original classifier order invalid")
    return paths, input_manifest, predicted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True, help="Original Fmax-only JSON, untouched by supplementary evaluation")
    parser.add_argument("--out", type=Path, default=None, help="Supplementary JSON; defaults alongside acceptance JSON")
    parser.add_argument("--threshold-step", type=float, default=0.001)
    args = parser.parse_args(argv)
    if args.threshold_step != .001:
        parser.error("The frozen v085 acceptance protocol requires threshold-step=0.001")
    comparison = required_file(args.comparison, "original acceptance comparison")
    report = json.loads(comparison.read_text(encoding="utf-8"))
    paths, _, prediction_manifest = verify_report(report, comparison)
    task = report["task"]
    if task not in WIDTH:
        raise ValueError(f"Unknown task: {task}")
    width = WIDTH[task]
    source_ids = read_ids(paths["protein_ids"])
    with paths["metadata"].open("rb") as stream:
        metadata = pickle.load(stream)
    meta_ids, y = annotation_matrix(metadata, task, width)
    if len(source_ids) != len(meta_ids) or set(source_ids) != set(meta_ids):
        raise ValueError("Original prediction protein IDs differ from metadata proteins")
    if len(meta_ids) != report["num_samples"] or report["num_classes"] != width:
        raise ValueError("Original acceptance report labels/shape differ")
    lookup = {protein: i for i, protein in enumerate(source_ids)}
    rows = np.fromiter((lookup[p] for p in meta_ids), dtype=np.int64, count=len(meta_ids))
    is_identity = np.array_equal(rows, np.arange(len(rows)))
    result = {}
    for name, label in (("NBS_final", "nbs"), ("backbone_base", "backbone")):
        arr = np.load(paths[label], mmap_mode="r", allow_pickle=False)
        if arr.shape != (len(meta_ids), width):
            raise ValueError(f"{name} probability shape differs: {arr.shape}")
        # Do not copy the entire array when no row reordering is needed.
        aligned = arr if is_identity else np.asarray(arr[rows], dtype=np.float32)
        scored = compute_standard_metrics(y, aligned, threshold_step=.001)
        original_fmax = report["methods"][name]["standardized"]["standard_protein_fmax"]
        if not np.isclose(scored["standard_protein_fmax"], original_fmax, rtol=0, atol=1e-9):
            raise ValueError(f"{name} Fmax disagrees with untouched acceptance report: "
                             f"{scored['standard_protein_fmax']} vs {original_fmax}")
        result[name] = scored
    scored_keys = ("standard_protein_fmax", "standard_micro_pr_auc", "standard_micro_ap")
    deltas = {key: float(result["NBS_final"][key] - result["backbone_base"][key]) for key in scored_keys}
    output = args.out.resolve() if args.out else comparison.with_name("nbs_supplementary_v085.json")
    if output == comparison:
        raise ValueError("Supplementary output must never overwrite original acceptance comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "evaluation_scope": "supplementary_v085_Fmax_PR_AUC_AP",
        "source_acceptance_comparison": str(comparison),
        "source_acceptance_comparison_sha256": sha256(comparison),
        "metric_schema": METRIC_SCHEMA,
        "metric_kernel_sha256": metrics_implementation_sha256(),
        "metric_definitions": metric_definitions(),
        "task": task,
        "num_samples": len(meta_ids),
        "num_classes": width,
        "checkpoint_sha256": prediction_manifest["checkpoint_sha256"],
        "methods": result,
        "G_minus_B": deltas,
        "note": "This file is supplementary; it must NOT be passed to check_nbs_v088_acceptance.py",
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for model, metrics in result.items():
        print(f"[{model}] Fmax={metrics['standard_protein_fmax']:.6f} "
              f"AuPRC={metrics['standard_micro_pr_auc']:.6f} AP={metrics['standard_micro_ap']:.6f}")
    print(f"[Saved] supplementary report: {output}")
    print(f"[UNCHANGED] original acceptance report: {comparison}")
    return summary


if __name__ == "__main__":
    main()
