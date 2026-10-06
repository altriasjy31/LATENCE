#!/usr/bin/env python3
"""v088 standalone-model protein Fmax, using the unchanged v085 metric kernels.

The command computes only the original standardized protein Fmax for G and B.
Legacy metric/reference arguments are accepted for runner compatibility but are
not executed. No AP/PR ranking, TopK, bootstrap, E/M evaluation or score fusion.
The historical helper functions below are retained solely for input/provenance
compatibility; main dispatches only to evaluate_fmax_only.
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


ABLATIONS = ("full",)
SOURCE_KEYS = ("query_candidate", "neighbor_candidate", "neighbor_pseudo", "core", "pp")


def expected_source_flags(ablation):
    if ablation != "full":
        raise ValueError("v0.8.8 supports full evaluation only; ablations are paused")
    return dict.fromkeys(SOURCE_KEYS, True)


def _validate_source_flags(manifest):
    flags = manifest.get("source_flags")
    if (flags != expected_source_flags(manifest.get("ablation"))
            or any(type(value) is not bool for value in flags.values())):
        raise ValueError("v0.8.8 requires all graph source flags enabled")
    return dict(flags)


ENCODER_IDENTITY_KEYS = ("encoder_variant", "model_config", "sampler_config",
                         "model_architecture_version", "data_architecture_version", "prediction_mode",
                         "uses_dense_backbone_logits_in_forward", "output_fusion")


def _encoder_identity(manifest):
    for key in ("model_architecture_version", "data_architecture_version"):
        if manifest.get(key) != "0.8.8":
            raise ValueError(f"v0.8.8 prediction requires {key} 0.8.8")
    model, sampler = manifest.get("model_config"), manifest.get("sampler_config")
    if manifest.get("encoder_variant") != "hetero_tuned" or not isinstance(model, dict) or model.get("encoder_variant") != "hetero_tuned":
        raise ValueError("v0.8.8 requires explicit hetero_tuned encoder identity")
    if manifest.get("prediction_mode") != "direct" or model.get("prediction_mode") != "direct":
        raise ValueError("v0.8.8 requires direct prediction mode")
    if not isinstance(sampler, dict) or sampler.get("mode") != manifest.get("variant"):
        raise ValueError("v0.8.8 sampler_config differs from variant")
    if manifest.get("uses_dense_backbone_logits_in_forward") is not False or manifest.get("output_fusion") != "none":
        raise ValueError("v0.8.8 requires standalone model logits with no output fusion")
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
    p.add_argument("--fmax-only", action="store_true", default=True, help="v088 always evaluates only the unchanged standardized protein Fmax; no AP/PR/TopK/bootstrap")
    return p


def _prediction_provenance(args: argparse.Namespace, legacy: Any) -> dict[str, Any]:
    """Bind the evaluated matrix to the checkpoint and prepared input before evaluation.

    A manifest's existence alone does not associate a metrics JSON with its
    claimed checkpoint. Both array and input hashes are checked, then copied
    into the compact comparison used for analysis.
    """
    if args.prediction_manifest is None or args.input_manifest is None:
        raise ValueError("v0.8.8 evaluation requires prediction and input manifests")
    manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    if manifest.get("runner_version", manifest.get("version")) != "0.8.8":
        raise ValueError("Prediction manifest is not a v0.8.8 output; use its matching evaluator")
    variant, branch = manifest.get("variant"), manifest.get("branch")
    if variant not in ("fixed", "dynamic"):
        raise ValueError("Prediction manifest has an unknown v0.8.8 variant")
    encoder_identity = _encoder_identity(manifest)
    source_flags = _validate_source_flags(manifest)
    if branch != "final":
        raise ValueError("v0.8.8 has a single final matching output")
    if manifest.get("ablation") not in ABLATIONS:
        raise ValueError("Prediction manifest has an unknown v0.8.8 ablation")
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
        "schema_version": 3, "evaluation_version": "0.8.8", "task": metrics["task"],
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
        raise ValueError("v088 uses the fixed standardized protein Fmax; local_micro is not an acceptance metric")
    if args.standard_threshold_step != .001:
        raise ValueError("v088 fixes the original standardized protein Fmax threshold step at 0.001")
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
    if manifest.get("uses_expert_probability_in_nbs_forward", True) or manifest.get("uses_modelout_probability_as_target", False):
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
        "schema_version": 3, "evaluation_version": "0.8.8", "task": args.task,
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
        print(f"[v088 Fmax-only] {name}={item['standardized']['standard_protein_fmax']:.10f}")
    print(f"[Fmax-only evaluation] {output}; no AP/PR/TopK/bootstrap/reference evaluation")
    return summary


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--summarize-work-dir" in arguments or "--recompute-comparison" in arguments:
        raise ValueError("v088 exposes full model Fmax evaluation only; ablation sweeps and historical audits are paused")
    return evaluate_fmax_only(parser().parse_args(arguments))


if __name__ == "__main__":
    main()
