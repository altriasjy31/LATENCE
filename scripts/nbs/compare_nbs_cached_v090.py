#!/usr/bin/env python3
"""Compare fixed BP epochs using existing, source-bound probability caches.

No model load, GPU inference, export, calibration, or probability propagation.
P@R is the highest protein-mean precision on the original 0.001 grid whose
protein-mean recall reaches 45/50/55 percent; it uses no interpolation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRIC_FILE = ROOT / "nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py"
BASELINE_SHA = "ad34e5baff69e092974ed42aa4e236046c5af89a4a3d2ee81c4291be8a3fc04e"
VERSIONS = {"0.8.9.1": "v089", "0.9.0": "v090"}
FIELDS = ("standard_protein_fmax", "standard_protein_precision_at_fmax",
          "standard_protein_recall_at_fmax", "standard_protein_fmax_threshold")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_METRICS = load_module(METRIC_FILE, "_nbs_cached_original_v085_metrics")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path, expected, label):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(f"{label}: missing valid source SHA256")
    if sha256(path) != expected:
        raise ValueError(f"{label}: SHA256 mismatch: {path}")
    return expected


def curve_statistics(labels, probabilities, *, save_curve=False):
    """Use the unchanged native protein curve; scores are percentages."""
    y, p = _METRICS._validate(labels, probabilities)
    thresholds = _METRICS._thresholds(.001)
    fmax, precision, recall, _ = _METRICS._protein_curve(y, p, thresholds)
    best = int(np.argmax(fmax))
    result = dict(zip(FIELDS, (100 * float(fmax[best]),
                              100 * float(precision[best]),
                              100 * float(recall[best]), float(thresholds[best]))))
    result["precision_at_recall"] = {
        str(target): (100 * float(np.max(precision[recall >= target / 100]))
                      if np.any(recall >= target / 100) else None)
        for target in (45, 50, 55)}
    if save_curve:
        result["curve"] = {"threshold": thresholds.tolist(),
                           "precision_percent": (100 * precision).tolist(),
                           "recall_percent": (100 * recall).tolist()}
    return result


def completed_epoch(manifest, epoch):
    plan = manifest.get("phase_plan", {})
    return (type(manifest.get("epoch")) is int and manifest["epoch"] == epoch
            and manifest.get("cumulative_completed_epochs") == epoch
            and type(plan.get("steps_per_epoch")) is int
            and manifest.get("phase_step_in_epoch") == plan["steps_per_epoch"]
            and plan["steps_per_epoch"] > 0)


def find_cache(work_dir, epoch):
    matches = []
    for path in sorted(work_dir.glob("eval_step*/full/nbs_full_task_prediction_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not completed_epoch(manifest, epoch):
            continue
        if path.parent.parent.name != f"eval_step{manifest.get('step')}":
            raise ValueError(f"Cache directory differs from manifest step: {path}")
        checkpoint = Path(manifest.get("checkpoint_path", ""))
        if checkpoint.name != f"nbs_epoch{epoch}.pt":
            raise ValueError(f"Cache is not bound to fixed nbs_epoch{epoch}.pt: {path}")
        report_path = None
        for directory in ("metrics_fmax_only", "metrics"):
            candidate = path.parent / directory / "nbs_w2s_comparison.json"
            if candidate.is_file():
                report = json.loads(candidate.read_text(encoding="utf-8"))
                if report.get("evaluation_scope") == "standard_protein_fmax_only":
                    report_path = candidate
                    break
        if report_path is None:
            raise FileNotFoundError(f"Missing source-bound Fmax-only comparison for epoch {epoch}: {path.parent}")
        matches.append((path, manifest, report_path))
    if len(matches) != 1:
        raise ValueError(f"Expected one completed fixed cache for epoch {epoch}, found {len(matches)}")
    return matches[0]


def verify_cache(report, manifest_path, manifest, paths, baseline):
    """Recheck actual cached bytes, the original gate, phase, and input identity."""
    version = manifest.get("runner_version")
    if version not in VERSIONS or report.get("evaluation_version") != version:
        raise ValueError("Unsupported or mismatched cache/evaluation version")
    suffix = VERSIONS[version]
    checker = load_module(ROOT / f"scripts/nbs/check_nbs_{suffix}_acceptance.py", f"_cached_checker_{suffix}")
    checker.check_reports(baseline, report)  # Includes full original scope and phase checks.
    evaluator = load_module(ROOT / f"scripts/nbs/eval_nbs_full_task_{suffix}.py", f"_cached_evaluator_{suffix}")
    legacy = evaluator._legacy()
    provenance = report.get("prediction_provenance", {})
    if provenance.get("manifest") != manifest:
        raise ValueError("Comparison embeds a different prediction manifest")
    for path, expected, label in (
        (manifest_path, provenance.get("prediction_manifest_sha256"), "prediction manifest"),
        (paths["probability"], provenance.get("probability_sha256"), "probability"),
        (paths["input"], provenance.get("input_manifest_sha256"), "input manifest"),
        (paths["backbone"], provenance.get("backbone_probability_sha256"), "backbone"),
        (paths["protein_ids"], provenance.get("protein_ids_file_sha256"), "protein IDs"),
        (paths["metadata"], report.get("metadata_provenance", {}).get("sha256"), "metadata"),
        (METRIC_FILE, report.get("standardized_metric_contract", {}).get("implementation_sha256"), "metric source"),
        (Path(evaluator.__file__), provenance.get("evaluator_sha256"), "evaluator source"),
    ):
        verify_hash(path, expected, label)
    source = manifest.get("prediction_implementation")
    if not isinstance(source, dict) or not source or provenance.get("prediction_implementation") != source:
        raise ValueError("Missing or inconsistent prediction implementation provenance")
    for name, expected in source.items():
        if Path(name).name != name:
            raise ValueError("Unexpected prediction implementation filename")
        folder = ROOT / ("scripts/nbs" if name.startswith("train_nbs_") else "nbs_models/nbs_protein_go/nbs_pg")
        verify_hash(folder / name, expected, f"prediction source {name}")
    args = argparse.Namespace(task="bp", nbs_prob=paths["probability"], backbone_prob=paths["backbone"],
                              prediction_manifest=manifest_path, input_manifest=paths["input"], go_registry=None)
    evaluator._prediction_provenance(args, legacy)
    registry, go_ids = evaluator._registry(args, legacy)
    if registry is None or go_ids is None or len(go_ids) != 21312:
        raise ValueError("Original complete 21312-column GO registry is required")
    verify_hash(registry, provenance.get("input_go_registry_sha256"), "GO registry")
    inputs = json.loads(paths["input"].read_text(encoding="utf-8"))
    for key in ("ind_test_labels_in_representation", "ind_test_labels_in_candidate_selection",
                "ind_test_labels_in_pp_retrieval", "expert_probability_in_forward_inputs", "test_to_test_edges"):
        if inputs.get("leakage_contract", {}).get(key, False):
            raise ValueError(f"Prepared input violates inductive contract: {key}")
    return evaluator, legacy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--epochs", default="25,30")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2")
    parser.add_argument("--metadata-file", type=Path, default=Path("/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-curves", action="store_true", help="Include original grid curves; omitted by default")
    args = parser.parse_args(argv)
    try:
        epochs = [int(value.strip()) for value in args.epochs.split(",")]
    except ValueError:
        parser.error("epochs must be comma-separated positive integers")
    if not epochs or len(set(epochs)) != len(epochs) or any(epoch <= 0 for epoch in epochs):
        parser.error("epochs must be unique positive integers")
    work = args.work_dir.resolve()
    baseline_path = ROOT / "BASELINE_CONTRACT_v088.json"
    verify_hash(baseline_path, BASELINE_SHA, "fixed baseline contract")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    paths = {"input": args.input_dir.resolve() / "ind_test_input_manifest.json",
             "backbone": args.input_dir.resolve() / "backbone_ind_test_prob.f16.npy",
             "protein_ids": args.input_dir.resolve() / "protein_ids.txt",
             "metadata": args.metadata_file.resolve()}
    result = {"schema": "nbs-cached-protein-fmax-v090", "task": "bp", "num_proteins": 1800,
              "num_GO_columns": 21312, "score_unit": "percent; differences are percentage points",
              "threshold_step": .001, "strict_comparison": ">", "ontology_probability_propagation": False,
              "precision_at_recall_definition": "Maximum protein-mean precision at grid recall >= target; no interpolation; null if unreachable",
              "baseline_contract_sha256": BASELINE_SHA, "epochs": {},
              "checkpoint_selection": "Descriptive fixed checkpoints; no model selection on ind_test"}
    baseline_score = baseline["methods"]["backbone_base"]["standardized"][FIELDS[0]]
    for epoch in epochs:
        manifest_path, manifest, comparison_path = find_cache(work, epoch)
        report = json.loads(comparison_path.read_text(encoding="utf-8"))
        paths["probability"] = manifest_path.parent / "nbs_ind_test_prob.f32.npy"
        evaluator, legacy = verify_cache(report, manifest_path, manifest, paths, baseline)
        labels, metadata_ids = evaluator._labels(paths["metadata"], "bp", legacy)
        if labels.shape != (1800, 21312):
            raise ValueError("Expected original 1800 x 21312 gold matrix")
        target_ids = evaluator._ids(paths["protein_ids"])
        evaluator._order(target_ids, metadata_ids, "input protein")
        if legacy._hash_ids(metadata_ids) != report["metadata_provenance"].get("protein_ids_sha256"):
            raise ValueError("Metadata protein order differs from original comparison")
        for method, path in (("backbone_base", paths["backbone"]), ("NBS_final", paths["probability"])):
            if method == "backbone_base" and "B" in result:
                continue
            values = legacy._align_rows(legacy._load_prediction(path, 21312),
                                        prediction_ids=target_ids, metadata_ids=metadata_ids)
            statistics = curve_statistics(labels, values, save_curve=args.save_curves)
            expected = report["methods"][method]["standardized"]
            if any(not np.isclose(statistics[key], expected[key], atol=1e-8, rtol=0) for key in FIELDS):
                raise ValueError(f"Recomputed original Fmax/P/R/threshold differ: {method}, epoch {epoch}")
            statistics["delta_Fmax_vs_B"] = statistics[FIELDS[0]] - baseline_score
            if method == "backbone_base":
                result["B"] = statistics
            else:
                statistics.update(phase_step=manifest["step"], runner_version=manifest["runner_version"],
                                  comparison_path=str(comparison_path), comparison_sha256=sha256(comparison_path),
                                  probability_sha256=report["prediction_provenance"]["probability_sha256"])
                result["epochs"][str(epoch)] = statistics
    output = args.output or work / ("cached_fmax_" + "_".join(map(str, epochs)) + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise FileExistsError(f"Output already contains a different report: {output}")
    else:
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, item in [("B", result["B"]), *[(f"epoch {epoch}", result["epochs"][str(epoch)]) for epoch in epochs]]:
        print(f"{name}: Fmax={item[FIELDS[0]]:.6f}% P={item[FIELDS[1]]:.6f}% R={item[FIELDS[2]]:.6f}% "
              f"t={item[FIELDS[3]]:.3f} P@R={item['precision_at_recall']}")
    print(output)
    return result


if __name__ == "__main__":
    main()
