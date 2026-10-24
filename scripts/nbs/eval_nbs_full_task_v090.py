#!/usr/bin/env python3
"""v0.9.0: source-bound, unchanged protein-centric Fmax only."""
from __future__ import annotations

import argparse

import contextlib

import csv

import hashlib

import importlib.util

import json

from pathlib import Path

import sys

from typing import Any

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[2] / "nbs_models/nbs_protein_go"

if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from nbs_pg.full_task_metrics_v085 import (compute_standard_metrics, metric_definitions, metrics_implementation_sha256,
                                          _validate, _thresholds, _protein_curve)

from nbs_pg.full_task_phase_v090 import PhasePlanV090

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ABLATIONS = ("full",)

SOURCE_KEYS = ("query_candidate", "neighbor_candidate", "neighbor_pseudo", "core", "pp")

def expected_source_flags(ablation):
    if ablation != "full":
        raise ValueError("v0.9.0 supports full evaluation only; ablations are paused")
    return dict.fromkeys(SOURCE_KEYS, True)

def _validate_source_flags(manifest):
    flags = manifest.get("source_flags")
    if (flags != expected_source_flags(manifest.get("ablation"))
            or any(type(value) is not bool for value in flags.values())):
        raise ValueError("v0.9.0 requires all graph source flags enabled")
    return dict(flags)

ENCODER_IDENTITY_KEYS = ("encoder_variant", "model_config", "sampler_config",
                         "model_architecture_version", "data_architecture_version", "prediction_mode",
                         "uses_dense_backbone_logits_in_forward", "output_fusion")

PHASE_IDENTITY_KEYS = ("phase_step", "phase_epoch", "phase_step_in_epoch", "epoch",
                       "lifetime_weak_exposures", "lifetime_core_exposures",
                       "lifetime_optimizer_updates", "cumulative_completed_epochs")

def _phase_identity(manifest):
    """Validate recorded phase/lifetime counts without loading model tensors.

    A source checkpoint is bound by path and SHA in the phase origin. The
    trainer verifies its model and data contracts when initializing the phase;
    this evaluator does not claim to rerun or authenticate that training.
    """
    values = manifest.get("phase_plan")
    if not isinstance(values, dict):
        raise ValueError("v090 prediction requires a phase_plan")
    try:
        plan = PhasePlanV090(**values)
    except TypeError as exc:
        raise ValueError("v090 phase_plan has an incomplete or unknown schema") from exc
    progress = plan.progress(manifest.get("step"))
    identity = {}
    for key in PHASE_IDENTITY_KEYS:
        expected = progress["cumulative_epoch"] if key == "epoch" else progress[key]
        value = manifest.get(key)
        if type(value) is not int or value != expected:
            raise ValueError(f"v090 prediction phase bookkeeping differs: {key}")
        identity[key] = value
    # Optional detailed counts also must be consistent if a runner writes them.
    for key, expected in progress.items():
        if key in identity or key not in manifest:
            continue
        value = manifest[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"v090 prediction phase bookkeeping differs: {key}")
        identity[key] = value
    origin = manifest.get("phase_origin")
    if not isinstance(origin, dict):
        raise ValueError("v090 prediction requires a phase_origin")
    for key in ("origin_epochs", "origin_weak_exposures", "origin_core_exposures", "origin_optimizer_updates"):
        if type(origin.get(key)) is not int or origin[key] != getattr(plan, key):
            raise ValueError(f"v090 phase_origin differs from phase_plan: {key}")
    if (origin.get("source_optimizer_state_reused") is not False
            or origin.get("source_rng_state_reused") is not False):
        raise ValueError("v090 phase origin must declare optimizer and RNG state reset")
    if type(origin.get("new_training_seed")) is not int or origin["new_training_seed"] < 0:
        raise ValueError("v090 phase_origin requires a nonnegative new_training_seed")
    mode = origin.get("initialization")
    if mode == "fresh":
        if plan.origin_epochs != 0:
            raise ValueError("v090 fresh phase cannot inherit completed epochs")
        if any(origin.get(key) is not None for key in ("source_checkpoint", "source_checkpoint_sha256", "source_runner_version")):
            raise ValueError("v090 fresh phase cannot identify a source checkpoint")
        for key in ("source_phase_step", "source_world_size"):
            if type(origin.get(key)) is not int or origin[key] != 0:
                raise ValueError(f"v090 fresh phase requires zero {key}")
    else:
        raise ValueError("v090 phase_origin initialization must be fresh (optional encoder transfer is separate lineage)")
    transfer = origin.get("encoder_transfer")
    if transfer is not None:
        if not isinstance(transfer, dict) or transfer.get("mode") != "encoder_only":
            raise ValueError("v090 encoder transfer lineage must be encoder_only")
        if transfer.get("source_runner_version") not in ("0.8.8", "0.8.9.1"):
            raise ValueError("v090 encoder source must retain its original runner identity")
        source, digest = transfer.get("source_checkpoint"), transfer.get("source_checkpoint_sha256")
        if (not isinstance(source, str) or not Path(source).is_absolute() or
                not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("v090 encoder transfer needs an absolute source path and SHA256")
        for name in ("source_encoder_epochs", "source_encoder_weak_exposures", "source_encoder_core_exposures", "source_encoder_optimizer_updates"):
            if type(transfer.get(name)) is not int or transfer[name] <= 0:
                raise ValueError("v090 encoder source counters must be separately recorded positive integers")
        for name in ("old_head_transferred", "optimizer_state_reused", "rng_state_reused"):
            if transfer.get(name) is not False:
                raise ValueError("v090 encoder transfer cannot reuse old head, optimizer or RNG")
        if type(transfer.get("new_scorer_training_exposures_at_initialization")) is not int or transfer["new_scorer_training_exposures_at_initialization"] != 0:
            raise ValueError("v090 new scorer cannot inherit source training exposures")
    return {"phase_plan": dict(values), "phase_origin": dict(origin), **identity}

def _encoder_identity(manifest):
    for key, expected in (("model_architecture_version", "0.9.0"), ("data_architecture_version", "0.8.8")):
        if manifest.get(key) != expected:
            raise ValueError(f"v0.9.0 prediction requires {key} {expected}")
    model, sampler = manifest.get("model_config"), manifest.get("sampler_config")
    if manifest.get("encoder_variant") != "hetero_tuned" or not isinstance(model, dict) or model.get("encoder_variant") != "hetero_tuned":
        raise ValueError("v0.9.0 requires explicit hetero_tuned encoder identity")
    if manifest.get("prediction_mode") != "shared_bilinear" or model.get("prediction_mode") != "shared_bilinear":
        raise ValueError("v0.9.0 requires shared_bilinear prediction mode")
    if not isinstance(sampler, dict) or sampler.get("mode") != manifest.get("variant"):
        raise ValueError("v0.9.0 sampler_config differs from variant")
    if manifest.get("uses_dense_backbone_logits_in_forward") is not False or manifest.get("output_fusion") != "none":
        raise ValueError("v0.9.0 requires standalone model logits with no output fusion")
    return {key: manifest[key] for key in ENCODER_IDENTITY_KEYS}

def _legacy():
    path = Path(__file__).with_name("eval_nbs_ind_test_predictions.py")
    spec = importlib.util.spec_from_file_location("_nbs_existing_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=("bp", "mf", "cc"))
    for key in ("metadata-file", "nbs-prob", "backbone-prob", "output-dir", "input-manifest", "prediction-manifest", "protein-ids"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--go-registry", type=Path)
    p.add_argument("--fmax-only", action="store_true", default=True,
                   help="Original protein-centric Fmax, also the default")
    p.add_argument("--metric-backend", choices=("stage1",), default="stage1")
    p.add_argument("--auprc-mode", choices=("exact",), default="exact",
                   help="Compatibility value only; PR-AUC is not computed")
    p.add_argument("--standard-threshold-step", type=float, default=.001)
    return p


def _prediction_provenance(args: argparse.Namespace, legacy: Any) -> dict[str, Any]:
    """Bind the evaluated matrix to the checkpoint and prepared input before evaluation.

    A manifest's existence alone does not associate a metrics JSON with its
    claimed checkpoint. Both array and input hashes are checked, then copied
    into the compact comparison used for analysis.
    """
    if args.prediction_manifest is None or args.input_manifest is None:
        raise ValueError("v0.9.0 evaluation requires prediction and input manifests")
    manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    if manifest.get("runner_version") != "0.9.0" or manifest.get("version") != "0.9.0":
        raise ValueError("Prediction manifest is not a v0.9.0 output; use its matching evaluator")
    variant, branch = manifest.get("variant"), manifest.get("branch")
    if variant not in ("fixed", "dynamic"):
        raise ValueError("Prediction manifest has an unknown v0.9.0 variant")
    encoder_identity = _encoder_identity(manifest)
    phase_identity = _phase_identity(manifest)
    source_flags = _validate_source_flags(manifest)
    if branch != "final":
        raise ValueError("v0.9.0 has a single final matching output")
    if manifest.get("ablation") not in ABLATIONS:
        raise ValueError("Prediction manifest has an unknown v0.9.0 ablation")
    step = manifest.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Prediction manifest must identify a nonnegative checkpoint step")
    checkpoint_hash = manifest.get("checkpoint_sha256", "")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_hash):
        raise ValueError("Prediction manifest must identify the checkpoint SHA256")
    for path, key, label in ((args.nbs_prob, "output_probability_sha256", "prediction probability"),
                             (args.input_manifest, "input_manifest_sha256", "prepared input")):
        if legacy._sha256(path) != manifest.get(key):
            raise ValueError(f"Prediction manifest does not match {label}; regenerate the affected export")
    declared_checkpoint = manifest.get("checkpoint_path")
    checkpoint_verified = False
    if declared_checkpoint:
        checkpoint = Path(declared_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = args.prediction_manifest.parent / checkpoint
        if checkpoint.is_file():
            if legacy._sha256(checkpoint) != checkpoint_hash:
                raise ValueError("Prediction checkpoint file differs from its manifest SHA256")
            checkpoint_verified = True
    return {
        "probability_path": str(args.nbs_prob.resolve()),
        "probability_sha256": manifest["output_probability_sha256"],
        "prediction_manifest_path": str(args.prediction_manifest.resolve()),
        "prediction_manifest_sha256": legacy._sha256(args.prediction_manifest),
        "input_manifest_path": str(args.input_manifest.resolve()),
        "input_manifest_sha256": manifest["input_manifest_sha256"],
        "backbone_probability_path": str(args.backbone_prob.resolve()),
        "backbone_probability_sha256": legacy._sha256(args.backbone_prob),
        "checkpoint_path": declared_checkpoint, "checkpoint_sha256": checkpoint_hash,
        "checkpoint_file_verified": checkpoint_verified,
        "step": step, "variant": variant, "branch": branch,
        **encoder_identity,
        **phase_identity,
        "source_flags": source_flags,
        "epoch": manifest.get("epoch"),
        "epoch_progress": manifest.get("epoch_progress"),
        "contract": manifest.get("contract", {}),
        "sampling_inference": manifest.get("sampling_inference"),
        "ablation": manifest.get("ablation"),
        "prediction_implementation": manifest.get("prediction_implementation", {}),
        "evaluator_sha256": legacy._sha256(Path(__file__)),
        "manifest": manifest,
    }

def _ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        if values.ndim != 1:
            raise ValueError(f"ID file must be one-dimensional: {path}")
        return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _order(source: list[str], target: list[str], label: str) -> np.ndarray:
    if len(source) != len(set(source)) or len(target) != len(set(target)):
        raise ValueError(f"{label} IDs must be unique; use original task IDs, not canonicalized aliases")
    missing, extra = set(target) - set(source), set(source) - set(target)
    if missing or extra:
        raise ValueError(f"{label} IDs differ: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    lookup = {key: i for i, key in enumerate(source)}
    return np.asarray([lookup[key] for key in target], dtype=np.int64)

def _registry(args: argparse.Namespace, legacy: Any) -> tuple[Path | None, list[str] | None]:
    manifest = {} if args.input_manifest is None else json.loads(args.input_manifest.read_text())
    registry = args.go_registry
    if registry is None:
        value = manifest.get("registries", {}).get("go_registry")
        if value:
            registry = Path(value)
            if not registry.is_absolute():
                registry = args.input_manifest.parent / registry
    if registry is None:
        return None, None
    registry = registry.resolve()
    expected_hash = (manifest.get("registries", {}).get("go_registry_sha256")
                     or manifest.get("cache_signature", {}).get("go_registry_sha256"))
    if expected_hash and legacy._sha256(registry) != expected_hash:
        raise ValueError("GO registry hash differs from prepared input")
    with registry.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        if not {"go_idx", "input_go_id"}.issubset(fields):
            raise ValueError("GO registry must provide go_idx and input_go_id (original classifier columns)")
        rows = sorted(reader, key=lambda row: int(row["go_idx"]))
    if [int(row["go_idx"]) for row in rows] != list(range(len(rows))):
        raise ValueError("GO registry indices must be exactly 0..num_classes-1")
    go_ids = [row["input_go_id"] for row in rows]
    expected_width = {"bp": 21312, "mf": 7038, "cc": 2903}[args.task]
    if len(go_ids) != expected_width:
        raise ValueError(f"GO registry has {len(go_ids)} columns, expected {expected_width}")
    _order(go_ids, go_ids, "registry GO")
    return registry, go_ids

def _write_compact_tsv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for name, entry in summary["methods"].items():
        row = {"comparison": name, "symbol": entry["symbol"], "status": "present"}
        for scope in ("primary", "micro", "standardized"):
            row.update({f"{scope}.{key}": value for key, value in entry.get(scope, {}).items() if isinstance(value, (int, float))})
        for k, values in entry["top_k"].items():
            row.update({f"{key}@{k}": value["mean"] for key, value in values.items()})
        rows.append(row)
    for name in summary["missing_methods"]:
        rows.append({"comparison": name, "status": "missing"})
    for name, entry in summary["deltas"].items():
        row = {"comparison": name, "status": entry["status"]}
        for scope in ("primary", "micro", "standardized"):
            row.update({f"{scope}.{key}": value for key, value in entry.get(scope, {}).items()})
        for k, values in entry.get("top_k", {}).items():
            row.update({f"{key}@{k}": value for key, value in values.items()})
        rows.append(row)
    columns = ["comparison", "symbol", "status"] + sorted(set().union(*(row.keys() for row in rows)) - {"comparison", "symbol", "status"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

def _labels(path: Path, task: str, legacy: Any) -> tuple[np.ndarray, list[str]]:
    import pickle
    with path.open("rb") as handle:
        data = legacy._task_dict(pickle.load(handle), "ind_test", task)
    ids = [str(x) for x in data["proteins"]]
    key = next((key for key in ("prop_annotations", "prop_annotation", "annotations", "labels") if key in data), None)
    if key is None:
        raise ValueError("Metadata has no independent-test annotation matrix")
    return legacy._dense_labels(data[key], {"bp": 21312, "mf": 7038, "cc": 2903}[task], len(ids)), ids

def compute_fmax_only(labels, probabilities):
    """Call original v085 kernels directly; no global ranking or micro metrics."""
    y, p = _validate(labels, probabilities)
    thresholds = _thresholds(.001)
    f1, precision, recall, eligible = _protein_curve(y, p, thresholds)
    best = int(np.argmax(f1))
    coarse, coarse_f1 = _thresholds(.01), f1[::10]
    coarse_best = int(np.argmax(coarse_f1))
    return {
        "standard_num_proteins": int(y.shape[0]), "standard_num_go": int(y.shape[1]),
        "standard_num_positive_labels": int(y.sum()), "standard_num_protein_fmax_eligible": eligible,
        "standard_num_empty_gold_proteins": int(y.shape[0]) - eligible,
        "standard_protein_fmax": 100 * float(f1[best]),
        "standard_protein_fmax_threshold": float(thresholds[best]),
        "standard_protein_precision_at_fmax": 100 * float(precision[best]),
        "standard_protein_recall_at_fmax": 100 * float(recall[best]),
        "standard_protein_fmax_threshold_step": .001,
        "standard_protein_fmax_grid_0p01": 100 * float(coarse_f1[coarse_best]),
        "standard_protein_fmax_grid_0p01_threshold": float(coarse[coarse_best]),
    }

def evaluate_fmax_only(args):
    if args.metric_backend != "stage1":
        raise ValueError("v090 uses the fixed standardized protein Fmax; local_micro is not an acceptance metric")
    if args.standard_threshold_step != .001:
        raise ValueError("v090 fixes the original standardized protein Fmax threshold step at 0.001")
    legacy = _legacy()
    prediction = _prediction_provenance(args, legacy)
    registry, go_ids = _registry(args, legacy)
    if registry is None or go_ids is None or args.protein_ids is None:
        raise ValueError("Fmax-only evaluation requires explicit protein IDs and the original GO registry")
    labels, metadata_ids = _labels(args.metadata_file, args.task, legacy)
    target_ids = _ids(args.protein_ids)
    _order(target_ids, metadata_ids, "input protein")
    inputs = json.loads(args.input_manifest.read_text())
    if inputs.get("protein_ids_file_sha256") and legacy._sha256(args.protein_ids) != inputs["protein_ids_file_sha256"]:
        raise ValueError("Protein ID file differs from prepared input")
    if inputs.get("protein_ids_sha256") and legacy._hash_ids(target_ids) != inputs["protein_ids_sha256"]:
        raise ValueError("Protein ID order differs from prepared input")
    declared_base = inputs.get("base_probability", {}).get("sha256")
    if declared_base and legacy._sha256(args.backbone_prob) != declared_base:
        raise ValueError("Backbone probability differs from prepared input")
    forbidden = ("ind_test_labels_in_representation", "ind_test_labels_in_candidate_selection",
                 "ind_test_labels_in_pp_retrieval", "expert_probability_in_forward_inputs", "test_to_test_edges")
    if any(inputs.get("leakage_contract", {}).get(key, False) for key in forbidden):
        raise ValueError("Prepared input violates the inductive contract")
    manifest = prediction["manifest"]
    if manifest.get("num_task_go") != labels.shape[1] or manifest.get("prediction_space") != "complete_task_classifier_columns":
        raise ValueError("Fmax evaluation must retain the complete original task GO columns")
    if (manifest.get("uses_expert_probability_in_nbs_forward") is not False
            or manifest.get("uses_modelout_probability_in_nbs_forward") is not False
            or manifest.get("uses_modelout_probability_as_target") is not False):
        raise ValueError("Prediction violates the reference-only probability contract")
    if manifest.get("step") != prediction["step"]:
        raise ValueError("Prediction step identity differs")
    prediction["protein_ids_path"] = str(args.protein_ids.resolve())
    prediction["protein_ids_file_sha256"] = legacy._sha256(args.protein_ids)
    prediction["input_go_registry_path"] = str(registry.resolve())
    prediction["input_go_registry_sha256"] = legacy._sha256(registry)
    methods = {}
    for name, symbol, path in (("NBS_final", "G", args.nbs_prob), ("backbone_base", "B", args.backbone_prob)):
        values = legacy._align_rows(legacy._load_prediction(path, labels.shape[1]), prediction_ids=target_ids, metadata_ids=metadata_ids)
        methods[name] = {"symbol": symbol, "primary": {}, "micro": {}, "top_k": {},
                         "primary_status": "not_computed", "micro_status": "not_computed",
                         "standardized": compute_fmax_only(labels, values)}
    summary = {
        "schema_version": 3, "evaluation_version": "0.9.0", "task": args.task,
        "num_samples": int(labels.shape[0]), "num_classes": int(labels.shape[1]),
        "evaluation_scope": "standard_protein_fmax_only", "methods": methods,
        "reference_comparison_complete": False, "missing_methods": [], "omitted_reference_methods": ["expert_prob", "stage1_modelout"],
        "evaluation_goal_complete": True, "goal_status": "model_baseline_fmax_comparison",
        "goal_definition": "Standalone NBS versus fixed B on standardized.standard_protein_fmax; acceptance is checked separately.",
        "metric_contract": {"primary_backend": "not_computed", "primary_scope": "not_computed",
                            "requested_legacy_backend": args.metric_backend,
                            "evaluation_scope": "standard_protein_fmax_only",
                            "compatibility_arguments_not_executed": ["metric-backend", "auprc-mode", "precision-k", "threshold-step", "bootstrap-replicates"],
                            "requested_auprc_mode": args.auprc_mode},
        "standardized_metric_contract": {"definitions": metric_definitions(),
            "implementation_sha256": metrics_implementation_sha256(), "threshold_step": .001,
            "score_unit": "percent except thresholds", "ontology_probability_propagation": False,
            "legacy_policy": "Historical primary fields remain unchanged; they are not standardized protein-centric Fmax or exact PR-AUC."},
        "prediction_provenance": prediction,
        "metadata_provenance": {"path": str(args.metadata_file.resolve()), "sha256": legacy._sha256(args.metadata_file),
                                "protein_ids_sha256": legacy._hash_ids(metadata_ids)},
        "reference_provenance": {}, "checkpoint_selection": "ind_test is descriptive, never an early-stop or hyperparameter-selection set",
        "deltas": {"G_minus_B": {"status": "ok", "standardized": {
            "standard_protein_fmax": methods["NBS_final"]["standardized"]["standard_protein_fmax"] - methods["backbone_base"]["standardized"]["standard_protein_fmax"]}}},
        "not_computed": ["historical_primary_fmax", "AP", "PR-AUC", "TopK", "bootstrap", "E", "M"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "nbs_w2s_comparison.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _write_compact_tsv(summary, args.output_dir / "nbs_w2s_comparison.tsv")
    for name, item in methods.items():
        print(f"[v090 Fmax-only] {name}={item['standardized']['standard_protein_fmax']:.10f}")
    print(f"[Fmax-only evaluation] {output}; no AP/PR/TopK/bootstrap/reference evaluation")
    return summary

def main(argv=None):
    return evaluate_fmax_only(parser().parse_args(argv))

if __name__ == "__main__":
    main()
