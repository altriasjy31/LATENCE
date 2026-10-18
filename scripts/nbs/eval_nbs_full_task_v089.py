#!/usr/bin/env python3
"""v089 phase-verified standalone Fmax, micro PR-AUC, and micro AP.

Default: retain native v089 checkpoint, phase/lifetime and input SHA checks;
score independent G/B and optional E/M with the unchanged native v085 kernel.
--fmax-only runs the original v089 evaluator exactly for regression.
"""
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
from nbs_pg.full_task_phase_v089 import PhasePlanV089

# Dark–Weak v0.4.4 shares the original v085 metric implementation verbatim.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.dark_weak.native_v085_metrics import native_imports, unified_v085_metrics, native_protocol



ABLATIONS = ("full",)
SOURCE_KEYS = ("query_candidate", "neighbor_candidate", "neighbor_pseudo", "core", "pp")


def expected_source_flags(ablation):
    if ablation != "full":
        raise ValueError("v0.8.9.1 supports full evaluation only; ablations are paused")
    return dict.fromkeys(SOURCE_KEYS, True)


def _validate_source_flags(manifest):
    flags = manifest.get("source_flags")
    if (flags != expected_source_flags(manifest.get("ablation"))
            or any(type(value) is not bool for value in flags.values())):
        raise ValueError("v0.8.9.1 requires all graph source flags enabled")
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
        raise ValueError("v089 prediction requires a phase_plan")
    try:
        plan = PhasePlanV089(**values)
    except TypeError as exc:
        raise ValueError("v089 phase_plan has an incomplete or unknown schema") from exc
    progress = plan.progress(manifest.get("step"))
    identity = {}
    for key in PHASE_IDENTITY_KEYS:
        expected = progress["cumulative_epoch"] if key == "epoch" else progress[key]
        value = manifest.get(key)
        if type(value) is not int or value != expected:
            raise ValueError(f"v089 prediction phase bookkeeping differs: {key}")
        identity[key] = value
    # Optional detailed counts also must be consistent if a runner writes them.
    for key, expected in progress.items():
        if key in identity or key not in manifest:
            continue
        value = manifest[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"v089 prediction phase bookkeeping differs: {key}")
        identity[key] = value
    origin = manifest.get("phase_origin")
    if not isinstance(origin, dict):
        raise ValueError("v089 prediction requires a phase_origin")
    for key in ("origin_epochs", "origin_weak_exposures", "origin_core_exposures", "origin_optimizer_updates"):
        if type(origin.get(key)) is not int or origin[key] != getattr(plan, key):
            raise ValueError(f"v089 phase_origin differs from phase_plan: {key}")
    if (origin.get("source_optimizer_state_reused") is not False
            or origin.get("source_rng_state_reused") is not False):
        raise ValueError("v089 phase origin must declare optimizer and RNG state reset")
    if type(origin.get("new_training_seed")) is not int or origin["new_training_seed"] < 0:
        raise ValueError("v089 phase_origin requires a nonnegative new_training_seed")
    mode = origin.get("initialization")
    if mode == "fresh":
        if plan.origin_epochs != 0:
            raise ValueError("v089 fresh phase cannot inherit completed epochs")
        if any(origin.get(key) is not None for key in ("source_checkpoint", "source_checkpoint_sha256", "source_runner_version")):
            raise ValueError("v089 fresh phase cannot identify a source checkpoint")
        for key in ("source_phase_step", "source_world_size"):
            if type(origin.get(key)) is not int or origin[key] != 0:
                raise ValueError(f"v089 fresh phase requires zero {key}")
    elif mode == "weights_only":
        source = origin.get("source_checkpoint")
        digest = origin.get("source_checkpoint_sha256")
        if plan.origin_epochs <= 0 or not isinstance(source, str) or not Path(source).is_absolute():
            raise ValueError("v089 initialized phase requires completed epochs and an absolute source checkpoint")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("v089 initialized phase requires a source checkpoint SHA256")
        if origin.get("source_runner_version") not in ("0.8.8", "0.8.9.1"):
            raise ValueError("v089 initialized phase requires source runner 0.8.8 or 0.8.9.1")
        for key in ("source_phase_step", "source_world_size"):
            if type(origin.get(key)) is not int or origin[key] <= 0:
                raise ValueError(f"v089 initialized phase requires positive {key}")
    else:
        raise ValueError("v089 phase_origin initialization must be fresh or weights_only")
    return {"phase_plan": dict(values), "phase_origin": dict(origin), **identity}


def _encoder_identity(manifest):
    for key in ("model_architecture_version", "data_architecture_version"):
        if manifest.get(key) != "0.8.8":
            raise ValueError(f"v0.8.9.1 prediction requires {key} 0.8.8")
    model, sampler = manifest.get("model_config"), manifest.get("sampler_config")
    if manifest.get("encoder_variant") != "hetero_tuned" or not isinstance(model, dict) or model.get("encoder_variant") != "hetero_tuned":
        raise ValueError("v0.8.9.1 requires explicit hetero_tuned encoder identity")
    if manifest.get("prediction_mode") != "direct" or model.get("prediction_mode") != "direct":
        raise ValueError("v0.8.9.1 requires direct prediction mode")
    if not isinstance(sampler, dict) or sampler.get("mode") != manifest.get("variant"):
        raise ValueError("v0.8.9.1 sampler_config differs from variant")
    if manifest.get("uses_dense_backbone_logits_in_forward") is not False or manifest.get("output_fusion") != "none":
        raise ValueError("v0.8.9.1 requires standalone model logits with no output fusion")
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
    for key in ("metadata-file", "nbs-prob", "backbone-prob", "output-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("input-manifest", "prediction-manifest"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("protein-ids", "candidate-go-index",
                "go-registry", "train-counts", "expert-prob", "modelout-prob",
                "expert-protein-ids", "modelout-protein-ids", "expert-go-ids", "modelout-go-ids",
                "stage1-reference-manifest"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--metric-backend", choices=("stage1", "local_micro"), default="stage1")
    p.add_argument("--auprc-mode", choices=("exact", "hist", "none"), default="exact")
    p.add_argument("--precision-k", default="10,50,100")
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--bootstrap-replicates", type=int, default=0)
    p.add_argument("--standard-threshold-step", type=float, default=0.001)
    p.add_argument("--references-aligned-to-input", action="store_true", help=(
        "Explicitly declare missing reference row/column ID files to have the exact "
        "prepared-input order. Supplied IDs are still verified and reordered."))
    p.add_argument("--allow-missing-references", action="store_true", default=True, help=(
        "Run diagnostics with absent expert/modelout references; report the final "
        "weak-to-strong goal as incomplete, never as achieved."))
    p.add_argument("--fmax-only", action="store_true", default=False, help="Explicit compatibility mode: unchanged v089 Fmax-only result")
    return p


def _prediction_provenance(args: argparse.Namespace, legacy: Any) -> dict[str, Any]:
    """Bind the evaluated matrix to the checkpoint and prepared input before evaluation.

    A manifest's existence alone does not associate a metrics JSON with its
    claimed checkpoint. Both array and input hashes are checked, then copied
    into the compact comparison used for analysis.
    """
    if args.prediction_manifest is None or args.input_manifest is None:
        raise ValueError("v0.8.9.1 evaluation requires prediction and input manifests")
    manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    if manifest.get("runner_version") != "0.8.9.1" or manifest.get("version") != "0.8.9.1":
        raise ValueError("Prediction manifest is not a v0.8.9.1 output; use its matching evaluator")
    variant, branch = manifest.get("variant"), manifest.get("branch")
    if variant not in ("fixed", "dynamic"):
        raise ValueError("Prediction manifest has an unknown v0.8.9.1 variant")
    encoder_identity = _encoder_identity(manifest)
    phase_identity = _phase_identity(manifest)
    source_flags = _validate_source_flags(manifest)
    if branch != "final":
        raise ValueError("v0.8.9.1 has a single final matching output")
    if manifest.get("ablation") not in ABLATIONS:
        raise ValueError("Prediction manifest has an unknown v0.8.9.1 ablation")
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


def _prepare_reference(*, name: str, probability: Path, protein_ids: Path | None,
                       go_ids: Path | None, target_ids: list[str], target_go: list[str] | None,
                       width: int, declared_aligned: bool, output_dir: Path,
                       registry: Path | None, legacy: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if protein_ids is None and not declared_aligned:
        raise ValueError(f"{name}: provide protein IDs or --references-aligned-to-input")
    if go_ids is None and not declared_aligned:
        raise ValueError(f"{name}: provide GO IDs or --references-aligned-to-input")
    if go_ids is not None and target_go is None:
        raise ValueError(f"{name}: GO IDs require --go-registry or an input-manifest registry")
    array = np.load(probability, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(target_ids), width) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name}: shape={array.shape}, expected {(len(target_ids), width)} numeric probabilities")
    source_rows = target_ids if protein_ids is None else _ids(protein_ids)
    source_cols = target_go if go_ids is None else _ids(go_ids)
    row_order = _order(source_rows, target_ids, f"{name} protein")
    col_order = np.arange(width) if go_ids is None else _order(source_cols, target_go, f"{name} GO")
    row_changed = not np.array_equal(row_order, np.arange(len(target_ids)))
    col_changed = not np.array_equal(col_order, np.arange(width))
    aligned_path = probability.resolve()
    if row_changed or col_changed:
        aligned_path = output_dir / f"{name}.aligned.f32.npy"
        aligned = np.lib.format.open_memmap(aligned_path, mode="w+", dtype=np.float32, shape=array.shape)
    else:
        aligned = None
    for start in range(0, array.shape[0], 64):
        block = np.asarray(array[row_order[start:start + 64]][:, col_order], dtype=np.float32)
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
            raise ValueError(f"{name}: values must be finite probabilities in [0,1]")
        if aligned is not None:
            aligned[start:start + len(block)] = block
    if aligned is not None:
        aligned.flush()
        del aligned
    target_ids_path = output_dir / "aligned_input_protein_ids.txt"
    target_ids_path.write_text("\n".join(target_ids) + "\n", encoding="utf-8")
    spec = {"probability": str(aligned_path), "protein_ids": str(target_ids_path),
            "role": "original_expert_reference" if name == "expert_prob" else "stage1_modelout_reference",
            "deployable": False, "description": "Evaluation reference only; not passed to NBS forward."}
    if registry is not None:
        spec["go_registry"] = str(registry)
    provenance = {
        "source_probability": str(probability.resolve()), "source_sha256": legacy._sha256(probability),
        "aligned_probability": str(aligned_path), "aligned_sha256": legacy._sha256(aligned_path),
        "source_protein_ids": None if protein_ids is None else str(protein_ids.resolve()),
        "source_protein_ids_sha256": None if protein_ids is None else legacy._sha256(protein_ids),
        "source_go_ids": None if go_ids is None else str(go_ids.resolve()),
        "source_go_ids_sha256": None if go_ids is None else legacy._sha256(go_ids),
        "rows_reordered": row_changed, "columns_reordered": col_changed,
        "row_alignment": "explicit_ids" if protein_ids is not None else "user_declared_input_order",
        "column_alignment": "explicit_original_task_go_ids" if go_ids is not None else "user_declared_input_order",
        "input_go_registry_sha256": None if registry is None else legacy._sha256(registry),
    }
    return spec, provenance


def _compact(metrics: dict[str, Any], provenance: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    symbol = "G"
    roles = {"expert_prob": "E", "stage1_modelout": "M", "backbone_base": "B", "NBS_final": symbol}
    methods = {}
    for method, role in roles.items():
        if method in metrics["metrics"]:
            methods[method] = {"symbol": role, "primary": metrics["metrics"][method],
                               "micro": metrics.get("micro_diagnostics", {}).get(method, {}),
                               "top_k": metrics.get("ranking_analysis", {}).get("models", {}).get(method, {})}
    missing = [name for name in roles if name not in methods]
    result: dict[str, Any] = {
        "schema_version": 3, "evaluation_version": "0.8.9.1", "task": metrics["task"],
        "num_samples": metrics["num_samples"], "num_classes": metrics["num_classes"],
        "reference_comparison_complete": not missing, "missing_methods": missing,
        "evaluation_goal_complete": ("NBS_final" in methods and "backbone_base" in methods
                                     and metrics["metric_contract"]["primary_backend"] == "stage1"),
        "goal_status": ("model_baseline_comparison" if metrics["metric_contract"]["primary_backend"] == "stage1"
                        else "diagnostic_local_micro"),
        "goal_definition": "Standalone NBS versus unchanged B. E/M are optional references. Acceptance is checked separately on standardized.standard_protein_fmax; completeness does not establish performance.",
        "metric_contract": metrics["metric_contract"], "methods": methods,
        "reference_provenance": provenance, "prediction_provenance": prediction, "deltas": {},
        "checkpoint_selection": "ind_test is descriptive, never an early-stop or hyperparameter-selection set",
    }
    final = methods["NBS_final"]
    for reference, reference_symbol in (("expert_prob", "E"), ("stage1_modelout", "M"), ("backbone_base", "B")):
        if reference not in methods:
            result["deltas"][f"{symbol}_minus_{reference_symbol}"] = {"status": "missing_reference"}
            continue
        other = methods[reference]
        entry = {"status": "ok", "primary": {}, "micro": {}, "top_k": {}}
        for scope in ("primary", "micro"):
            entry[scope] = {key: float(value) - float(other[scope][key])
                           for key, value in final[scope].items()
                           if key in other[scope] and isinstance(value, (int, float))
                           and (scope == "standardized" and _delta_key(key) or key.lower().startswith(("fmax", "auprc", "precision", "recall")))}
        for k, values in final["top_k"].items():
            if k in other["top_k"]:
                entry["top_k"][k] = {key: value["mean"] - other["top_k"][k][key]["mean"]
                                        for key, value in values.items() if key in other["top_k"][k]}
        result["deltas"][f"{symbol}_minus_{reference_symbol}"] = entry
    return result


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


def evaluate_predictions(args: argparse.Namespace) -> dict[str, Any]:
    legacy = _legacy()
    prediction = _prediction_provenance(args, legacy)
    generation = None
    reference_manifest = getattr(args, "stage1_reference_manifest", None)
    if reference_manifest is not None:
        generated = json.loads(reference_manifest.read_text())
        sources = generated.get("cache_signature", {}).get("sources", {})
        for path, key in ((args.input_manifest, "input_manifest"), (args.backbone_prob, "cached_backbone")):
            if path is None or legacy._sha256(path) != sources.get(key, {}).get("sha256"):
                raise ValueError(f"Stage1 reference manifest belongs to a different {key}")
        for argument, output_key in (("expert_prob", "expert_prob"), ("modelout_prob", "stage1_modelout"),
                                     ("expert_protein_ids", "protein_ids"), ("modelout_protein_ids", "protein_ids"),
                                     ("expert_go_ids", "go_ids"), ("modelout_go_ids", "go_ids")):
            path = getattr(args, argument)
            entry = generated.get("outputs", {}).get(output_key, {})
            if path is None or legacy._sha256(path) != entry.get("sha256"):
                raise ValueError(f"Stage1 reference manifest does not match {argument}")
        generation = {"manifest_path": str(reference_manifest.resolve()),
                      "manifest_sha256": legacy._sha256(reference_manifest), "manifest": generated}
    if args.expert_prob is not None and args.modelout_prob is not None and args.expert_prob.resolve() == args.modelout_prob.resolve():
        raise ValueError("expert_prob and stage1_modelout must be independently identified files, not the same path")
    if args.auprc_mode == "exact":
        try:
            from sklearn.metrics import average_precision_score  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Exact micro-AP requires scikit-learn; no histogram fallback is allowed.") from exc
    missing = [key for key in ("expert", "modelout") if getattr(args, key + "_prob") is None]
    if missing and not args.allow_missing_references:
        raise ValueError("Missing reference probability: " + ", ".join(missing)
                         + ". Supply both or use --allow-missing-references for explicitly incomplete diagnostics.")
    for key in ("expert", "modelout"):
        if getattr(args, key + "_prob") is None and any(getattr(args, key + suffix) is not None for suffix in ("_protein_ids", "_go_ids")):
            raise ValueError(f"{key}: ID files were supplied without a probability array")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.metadata_file.open("rb") as handle:
        import pickle
        metadata = pickle.load(handle)
    metadata_ids = [str(x) for x in legacy._task_dict(metadata, "ind_test", args.task)["proteins"]]
    target_ids = metadata_ids if args.protein_ids is None else _ids(args.protein_ids)
    _order(target_ids, metadata_ids, "input protein")
    registry, target_go = _registry(args, legacy)
    width = {"bp": 21312, "mf": 7038, "cc": 2903}[args.task]
    reference_dir = args.output_dir / "references"
    reference_dir.mkdir(exist_ok=True)
    predictions, provenance = {}, {}
    for source, name in (("expert", "expert_prob"), ("modelout", "stage1_modelout")):
        probability = getattr(args, source + "_prob")
        if probability is None:
            continue
        predictions[name], provenance[name] = _prepare_reference(
            name=name, probability=probability, protein_ids=getattr(args, source + "_protein_ids"),
            go_ids=getattr(args, source + "_go_ids"), target_ids=target_ids, target_go=target_go,
            width=width, declared_aligned=args.references_aligned_to_input,
            output_dir=reference_dir, registry=registry, legacy=legacy)
    comparison = reference_dir / "comparison_manifest.json"
    comparison.write_text(json.dumps({"predictions": predictions}, indent=2) + "\n", encoding="utf-8")
    command = [str(Path(legacy.__file__)), "--comparison-manifest", str(comparison), "--expected-comparisons", "expert_prob,stage1_modelout"]
    for key in ("task", "metadata_file", "nbs_prob", "backbone_prob", "protein_ids", "input_manifest",
                "prediction_manifest", "candidate_go_index", "train_counts", "output_dir", "metric_backend",
                "auprc_mode", "precision_k", "threshold_step", "bootstrap_replicates"):
        value = getattr(args, key)
        if value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    if registry is not None:
        command.extend(["--go-registry", str(registry)])
    old_argv = sys.argv
    try:
        sys.argv = command
        # The old evaluator prints its entire JSON; retain it on disk instead of flooding the terminal.
        with (args.output_dir / "evaluation_details.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            legacy.main()
    finally:
        sys.argv = old_argv
    metrics_path = args.output_dir / "nbs_ind_test_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    summary = _compact(metrics, provenance, prediction)
    source_arrays = {"NBS_final": args.nbs_prob, "backbone_base": args.backbone_prob}
    source_arrays.update({name: Path(item["aligned_probability"]) for name, item in provenance.items()})
    labels, metadata_ids = _labels(args.metadata_file, args.task, legacy)
    arrays = {name: legacy._align_rows(legacy._load_prediction(path, width),
               prediction_ids=target_ids, metadata_ids=metadata_ids) for name, path in source_arrays.items()}
    _add_standardized(summary, labels, arrays, args.standard_threshold_step)
    summary["metadata_provenance"] = {"path": str(args.metadata_file.resolve()),
        "sha256": legacy._sha256(args.metadata_file), "protein_ids_sha256": legacy._hash_ids(metadata_ids)}
    summary["prediction_provenance"]["protein_ids_path"] = None if args.protein_ids is None else str(args.protein_ids.resolve())
    summary["prediction_provenance"]["protein_ids_file_sha256"] = None if args.protein_ids is None else legacy._sha256(args.protein_ids)
    summary["prediction_provenance"]["input_go_registry_path"] = None if registry is None else str(registry.resolve())
    summary["prediction_provenance"]["input_go_registry_sha256"] = None if registry is None else legacy._sha256(registry)
    if generation is not None:
        summary["stage1_reference_generation"] = generation
    metrics["w2s_comparison"] = summary
    metrics["note"] = "expert_prob (E) and stage1_modelout (M) are independent evaluation references; neither is supplied by this evaluator to NBS forward."
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path = args.output_dir / "nbs_w2s_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_compact_tsv(summary, args.output_dir / "nbs_w2s_comparison.tsv")
    print(f"[W2S evaluation] {summary['goal_status']}; {summary_path}")
    for name, entry in summary["deltas"].items():
        print(f"[{name}] {json.dumps(entry.get('primary', entry), ensure_ascii=False)}")
    return summary


def _labels(path: Path, task: str, legacy: Any) -> tuple[np.ndarray, list[str]]:
    import pickle
    with path.open("rb") as handle:
        data = legacy._task_dict(pickle.load(handle), "ind_test", task)
    ids = [str(x) for x in data["proteins"]]
    key = next((key for key in ("prop_annotations", "prop_annotation", "annotations", "labels") if key in data), None)
    if key is None:
        raise ValueError("Metadata has no independent-test annotation matrix")
    return legacy._dense_labels(data[key], {"bp": 21312, "mf": 7038, "cc": 2903}[task], len(ids)), ids


def _delta_key(key: str) -> bool:
    return key.startswith("standard_") and not any(token in key for token in ("threshold", "step", "count", "num_", "min_included_score"))


def _add_standardized(summary: dict, labels: np.ndarray, arrays: dict, threshold_step: float) -> None:
    summary["standardized_metric_contract"] = {"definitions": metric_definitions(),
        "implementation_sha256": metrics_implementation_sha256(), "threshold_step": threshold_step,
        "score_unit": "percent except thresholds", "ontology_probability_propagation": False,
        "legacy_policy": "Historical primary fields remain unchanged; they are not standardized protein-centric Fmax or exact PR-AUC."}
    summary["legacy_metric_contract"] = summary["metric_contract"]
    summary["standardized_comparison_complete"] = all(name in arrays for name in ("NBS_final", "expert_prob", "stage1_modelout", "backbone_base"))
    summary["standardized_goal_definition"] = "Prospectively compare protein-centric Fmax and exact micro PR-AUC against Expert; report exact micro AP separately. Completeness does not imply improvement or statistical significance."

    for name, values in arrays.items():
        summary["methods"][name]["standardized"] = compute_standard_metrics(labels, values, threshold_step=threshold_step)
    final = summary["methods"]["NBS_final"]["standardized"]
    for reference, symbol in (("expert_prob", "E"), ("stage1_modelout", "M"), ("backbone_base", "B")):
        if reference not in summary["methods"]:
            continue
        other = summary["methods"][reference]["standardized"]
        summary["deltas"][f"G_minus_{symbol}"]["standardized"] = {
            key: float(value)-float(other[key]) for key,value in final.items()
            if _delta_key(key) and key in other and isinstance(value, (float, int)) and not isinstance(value, bool)}




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
        raise ValueError("v089 uses the fixed standardized protein Fmax; local_micro is not an acceptance metric")
    if args.standard_threshold_step != .001:
        raise ValueError("v089 fixes the original standardized protein Fmax threshold step at 0.001")
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
        "schema_version": 3, "evaluation_version": "0.8.9.1", "task": args.task,
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
        print(f"[v089 Fmax-only] {name}={item['standardized']['standard_protein_fmax']:.10f}")
    print(f"[Fmax-only evaluation] {output}; no AP/PR/TopK/bootstrap/reference evaluation")
    return summary


def evaluate_standard_only(args):
    """Keep the original v089 validation; upgrade scored outputs to v085 Fmax+PR-AUC.

    A deliberately explicit --fmax-only mode reproduces the historical path.
    Additional Expert/Modelout matrices, when supplied, are comparison-only.
    """
    if args.metric_backend != 'stage1':
        raise ValueError('v089 standardized metrics require --metric-backend stage1')
    if args.standard_threshold_step != .001:
        raise ValueError('v089 standardized protein Fmax requires threshold step 0.001')
    legacy = _legacy()
    prediction = _prediction_provenance(args, legacy)
    registry, go_ids = _registry(args, legacy)
    if registry is None or go_ids is None or args.protein_ids is None:
        raise ValueError('Standardized v089 evaluation requires explicit protein IDs and GO registry')
    labels, metadata_ids = _labels(args.metadata_file, args.task, legacy)
    target_ids = _ids(args.protein_ids)
    _order(target_ids, metadata_ids, 'input protein')
    inputs = json.loads(args.input_manifest.read_text())
    if inputs.get('protein_ids_file_sha256') and legacy._sha256(args.protein_ids) != inputs['protein_ids_file_sha256']:
        raise ValueError('Protein IDs differ from prepared input')
    if inputs.get('protein_ids_sha256') and legacy._hash_ids(target_ids) != inputs['protein_ids_sha256']:
        raise ValueError('Protein ordering differs from prepared input')
    declared_base = inputs.get('base_probability', {}).get('sha256')
    if declared_base and legacy._sha256(args.backbone_prob) != declared_base:
        raise ValueError('Backbone probability differs from prepared input')
    forbidden = ('ind_test_labels_in_representation', 'ind_test_labels_in_candidate_selection',
                 'ind_test_labels_in_pp_retrieval', 'expert_probability_in_forward_inputs', 'test_to_test_edges')
    if any(inputs.get('leakage_contract', {}).get(key, False) for key in forbidden):
        raise ValueError('Prepared input violates inductive leakage contract')
    manifest = prediction['manifest']
    if manifest.get('num_task_go') != labels.shape[1] or manifest.get('prediction_space') != 'complete_task_classifier_columns':
        raise ValueError('v089 evaluation requires complete original classifier GO columns')
    if (manifest.get('uses_expert_probability_in_nbs_forward') is not False
            or manifest.get('uses_modelout_probability_in_nbs_forward') is not False
            or manifest.get('uses_modelout_probability_as_target') is not False):
        raise ValueError('v089 references must never enter graph inference')
    if manifest.get('step') != prediction['step']:
        raise ValueError('Prediction step differs from verified manifest')
    # Restore optional Stage-1 reference provenance checks from the original
    # (inactive) evaluate_predictions() branch, without using references in forward.
    reference_generation = None
    if args.stage1_reference_manifest is not None:
        generated = json.loads(args.stage1_reference_manifest.read_text())
        sources = generated.get('cache_signature', {}).get('sources', {})
        for path, key in ((args.input_manifest, 'input_manifest'),
                          (args.backbone_prob, 'cached_backbone')):
            if legacy._sha256(path) != sources.get(key, {}).get('sha256'):
                raise ValueError(f'Stage-1 reference manifest differs from {key}')
        for attribute, output_key in (
            ('expert_prob','expert_prob'),('modelout_prob','stage1_modelout'),
            ('expert_protein_ids','protein_ids'),('modelout_protein_ids','protein_ids'),
            ('expert_go_ids','go_ids'),('modelout_go_ids','go_ids')):
            path = getattr(args, attribute)
            if path is None:
                raise ValueError('Stage-1 reference manifest requires ' + attribute)
            if legacy._sha256(path) != generated.get('outputs', {}).get(output_key, {}).get('sha256'):
                raise ValueError(f'Stage-1 reference manifest differs from {attribute}')
        reference_generation = {'path': str(args.stage1_reference_manifest.resolve()),
                                'sha256': legacy._sha256(args.stage1_reference_manifest)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction['protein_ids_path'] = str(args.protein_ids.resolve())
    prediction['protein_ids_file_sha256'] = legacy._sha256(args.protein_ids)
    prediction['input_go_registry_path'] = str(registry.resolve())
    prediction['input_go_registry_sha256'] = legacy._sha256(registry)
    kernels = native_imports(PROJECT_ROOT) if not args.fmax_only else None
    source_arrays = {'NBS_final': args.nbs_prob, 'backbone_base': args.backbone_prob}
    reference_provenance = {}
    reference_dir = args.output_dir / 'references'
    reference_dir.mkdir(exist_ok=True)
    for source, name in (('expert', 'expert_prob'), ('modelout', 'stage1_modelout')):
        probability = getattr(args, source + '_prob')
        row_ids = getattr(args, source + '_protein_ids')
        column_ids = getattr(args, source + '_go_ids')
        if probability is None:
            if row_ids is not None or column_ids is not None:
                raise ValueError(f'{source}: IDs supplied but probability array is missing')
            continue
        spec, provenance = _prepare_reference(
            name=name, probability=probability, protein_ids=row_ids,
            go_ids=column_ids, target_ids=target_ids, target_go=go_ids,
            width=labels.shape[1], declared_aligned=args.references_aligned_to_input,
            output_dir=reference_dir, registry=registry, legacy=legacy)
        source_arrays[name] = Path(provenance['aligned_probability'])
        reference_provenance[name] = provenance
    if (args.expert_prob is not None and args.modelout_prob is not None
            and args.expert_prob.resolve() == args.modelout_prob.resolve()):
        raise ValueError('Expert and Modelout references must be distinct independently identified files')
    names = [('NBS_final', 'G'), ('backbone_base', 'B'),
             ('expert_prob', 'E'), ('stage1_modelout', 'M')]
    methods = {}
    for name, symbol in names:
        if name not in source_arrays:
            continue
        values = legacy._align_rows(
            legacy._load_prediction(source_arrays[name], labels.shape[1]),
            prediction_ids=target_ids, metadata_ids=metadata_ids)
        native = compute_fmax_only(labels, values) if args.fmax_only else unified_v085_metrics(
            labels, values, kernels, verify_fmax=True)
        methods[name] = {'symbol': symbol, 'primary': {}, 'micro': {}, 'top_k': {},
                         'primary_status': 'not_computed', 'micro_status': 'not_computed',
                         'standardized': native}
    # Report B/G by default and E/M if they were explicitly available.
    fmax_only = bool(args.fmax_only)
    comparison_complete = all(x in methods for x in ('NBS_final', 'backbone_base', 'expert_prob', 'stage1_modelout'))
    scored_fields = (('standard_protein_fmax',) if fmax_only else
                     ('standard_protein_fmax', 'Fmax_percent', 'AuPRC_percent', 'AP_percent'))
    deltas = {}
    for reference, symbol in (('backbone_base', 'B'), ('expert_prob', 'E'), ('stage1_modelout', 'M')):
        if reference not in methods:
            deltas['G_minus_' + symbol] = {'status': 'missing_reference'}
        else:
            mine = methods['NBS_final']['standardized']
            theirs = methods[reference]['standardized']
            deltas['G_minus_' + symbol] = {'status': 'ok', 'standardized': {
                key: float(mine[key]) - float(theirs[key]) for key in scored_fields}}
    standard_protocol = ({'definitions': metric_definitions(),
                          'implementation_sha256': metrics_implementation_sha256(),
                          'threshold_step': .001, 'units': 'percent except thresholds',
                          'score_type': 'original Fmax-only compatibility'} if fmax_only
                         else native_protocol(kernels))
    summary = {
        'schema_version': 4, 'evaluation_version': '0.8.9.1', 'task': args.task,
        'num_samples': int(labels.shape[0]), 'num_classes': int(labels.shape[1]),
        'evaluation_scope': ('standard_protein_fmax_only' if fmax_only
                             else 'standard_protein_fmax_and_v085_pr_auc'),
        'methods': methods,
        'reference_comparison_complete': comparison_complete,
        'missing_methods': [name for name in ('expert_prob', 'stage1_modelout') if name not in methods],
        'evaluation_goal_complete': True,
        'goal_status': 'model_baseline_standardized_metrics',
        'goal_definition': 'Standalone G versus B; evaluation-only E/M where provided. No fusion.',
        'metric_contract': {
            'primary_backend': 'v085_original_standardized',
            'evaluation_scope': ('fmax_only' if fmax_only else 'fmax_and_pr_auc'),
            'not_executed_cli': ['precision-k', 'bootstrap-replicates', 'auprc-mode',
                                 'threshold-step', 'metric-backend'],
            'note': ('Original v085 standard_micro_pr_auc is trapezoidal micro PR-AUC; '
                     'standard_micro_ap is non-interpolated AP. Under all-equal scores '
                     'PR-AUC=(1+positive_prevalence)/2, whereas AP=positive_prevalence. '
                     'Both are reported; neither affects model forward.')},
        'standardized_metric_contract': standard_protocol,
        'prediction_provenance': prediction,
        'phase_identity': {k: prediction.get(k) for k in ('phase_plan', 'phase_origin', 'epoch', 'step',
            'phase_step', 'lifetime_optimizer_updates', 'lifetime_weak_exposures',
            'lifetime_core_exposures', 'cumulative_completed_epochs')},
        'metadata_provenance': {
            'path': str(args.metadata_file.resolve()), 'sha256': legacy._sha256(args.metadata_file),
            'protein_ids_sha256': legacy._hash_ids(metadata_ids)},
        'reference_provenance': reference_provenance,
        'stage1_reference_generation': reference_generation,
        'checkpoint_selection': 'ind_test is descriptive; no model selection on test scores',
        'deltas': deltas,
        'not_computed': (['AP', 'PR-AUC', 'TopK', 'bootstrap'] if fmax_only
                         else ['TopK', 'bootstrap', 'legacy micro Fmax']),
    }
    output = args.output_dir / 'nbs_w2s_comparison.json'
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    _write_compact_tsv(summary, args.output_dir / 'nbs_w2s_comparison.tsv')
    for name, item in methods.items():
        m = item['standardized']
        if fmax_only:
            print(f'[v089 compatible] {name} Fmax={m["standard_protein_fmax"]:.6f}')
        else:
            print(f'[v089 standardized] {name}: Fmax={m["Fmax_percent"]:.6f}% '
                  f'AuPRC={m["AuPRC_percent"]:.6f}% AP={m["AP_percent"]:.6f}% '
                  f'(native key={m["AuPRC_native_key"]})')
    print(f'[v089 standardized evaluation] {output}')
    return summary




def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--summarize-work-dir" in arguments or "--recompute-comparison" in arguments:
        raise ValueError("v089 standalone standardized evaluation supports no ablation sweeps or historical audits")
    args = parser().parse_args(arguments)
    return evaluate_fmax_only(args) if args.fmax_only else evaluate_standard_only(args)


if __name__ == "__main__":
    main()
