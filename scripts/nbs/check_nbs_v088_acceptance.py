#!/usr/bin/env python3
"""Check one full v088 model result against the fixed, identical-test B baseline.

Acceptance metric is EXACTLY methods.*.standardized.standard_protein_fmax.
It is NOT the historical primary.fmax field. No metric or prediction is fitted,
combined or recomputed here. Missing new evidence is explicitly unverified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

BASELINE_COMPARISON_SHA256 = "ad34e5baff69e092974ed42aa4e236046c5af89a4a3d2ee81c4291be8a3fc04e"
BASELINE_FMAX = 57.72314936447243
METRIC = "standardized.standard_protein_fmax"
METRIC_KEY = "standard_protein_fmax"
REQUIRED_GAIN = 3.0
ABSOLUTE_MINIMUM = 60.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def score(report, method):
    value = report["methods"][method]["standardized"][METRIC_KEY]
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100,
            f"{method}: invalid {METRIC}")
    return float(value)


def metric_contract(report):
    value = dict(report["metric_contract"])
    # Location may change; helper bytes and every semantic field must match.
    value.pop("stage1_helper", None)
    return value


def verify_baseline(baseline):
    require(baseline.get("evaluation_version") == "0.8.7", "Expected original v0.8.7 baseline comparison")
    require((baseline.get("task"), baseline.get("num_samples"), baseline.get("num_classes")) == ("bp", 1800, 21312),
            "Baseline must be BP, 1800 proteins, complete 21312 GO columns")
    require(baseline["prediction_provenance"].get("ablation") == "full", "Baseline comparison must be full")
    require(baseline["metric_contract"].get("primary_backend") == "stage1", "Baseline must preserve original Stage1 evaluator")
    value = score(baseline, "backbone_base")
    require(math.isclose(value, BASELINE_FMAX, rel_tol=0, abs_tol=1e-10),
            "B standardized protein Fmax differs from the fixed 57.72314936447243 baseline")
    return value


def check_reports(baseline, new=None):
    base = verify_baseline(baseline)
    threshold = max(ABSOLUTE_MINIMUM, base + REQUIRED_GAIN)
    result = {"schema": "v088-standalone-acceptance-v1", "status": "未验证",
              "metric": "methods.NBS_final." + METRIC,
              "baseline_metric": "methods.backbone_base." + METRIC,
              "score_unit": "percentage points on a 0..100 scale",
              "baseline_Fmax": base, "new_Fmax": None, "delta_Fmax": None,
              "required_gain": REQUIRED_GAIN, "absolute_minimum": ABSOLUTE_MINIMUM,
              "effective_threshold": threshold, "fixed_conditions_verified": False,
              "verification_scope": "comparison file hashes and recorded evaluator provenance; this checker does not rerun inference or metrics",
              "note": "Historical primary.fmax is retained in evaluation but is not mixed into this gate. CPU tests do not establish trained-model performance."}
    if new is None:
        result["reason"] = "No new source-bound full v088 comparison was supplied"
        return result
    require(new.get("evaluation_version") == "0.8.8", "New comparison must be v0.8.8")
    for key in ("task", "num_samples", "num_classes"):
        require(new.get(key) == baseline.get(key), f"Evaluation population differs: {key}")
    before, after = baseline["prediction_provenance"], new["prediction_provenance"]
    require(after.get("ablation") == "full" and after.get("branch") == "final", "Only the full final model is eligible")
    require(after.get("prediction_mode") == "direct" and after.get("encoder_variant") == "hetero_tuned",
            "Only standalone direct hetero_tuned output is eligible")
    require(after.get("output_fusion") == "none" and after.get("uses_dense_backbone_logits_in_forward") is False,
            "Output fusion or dense B logits are not allowed")
    flags = after.get("source_flags")
    require(isinstance(flags, dict) and set(flags) == {"query_candidate", "neighbor_candidate", "neighbor_pseudo", "core", "pp"}
            and all(value is True for value in flags.values()), "Every graph source must remain enabled")
    for key in ("input_manifest_sha256", "protein_ids_file_sha256", "input_go_registry_sha256", "backbone_probability_sha256"):
        require(bool(before.get(key)) and after.get(key) == before.get(key), f"Fixed input identity differs: {key}")
    for key in ("sha256", "protein_ids_sha256"):
        require(bool(baseline.get("metadata_provenance", {}).get(key)) and new.get("metadata_provenance", {}).get(key) == baseline["metadata_provenance"][key],
                f"Gold metadata identity differs: {key}")
    require(new.get("evaluation_scope") == "standard_protein_fmax_only", "New report must compute only the fixed standardized protein Fmax")
    require(new.get("metric_contract", {}).get("primary_backend") == "not_computed"
            and new["metric_contract"].get("evaluation_scope") == "standard_protein_fmax_only",
            "Historical primary metrics must be explicitly marked not computed")
    require(new.get("standardized_metric_contract") == baseline.get("standardized_metric_contract"), "Standardized Fmax metric contract differs")
    require(new["standardized_metric_contract"].get("ontology_probability_propagation") is False,
            "Acceptance does not permit added ontology score propagation")
    # B is evaluated from exactly the same matrix using exactly the same definitions.
    fmax_fields = ("standard_num_proteins", "standard_num_go", "standard_num_positive_labels",
                   "standard_num_protein_fmax_eligible", "standard_num_empty_gold_proteins",
                   "standard_protein_fmax", "standard_protein_fmax_threshold",
                   "standard_protein_precision_at_fmax", "standard_protein_recall_at_fmax",
                   "standard_protein_fmax_threshold_step", "standard_protein_fmax_grid_0p01",
                   "standard_protein_fmax_grid_0p01_threshold")
    for key in fmax_fields:
        old_value = baseline["methods"]["backbone_base"]["standardized"].get(key)
        require(old_value is not None and new["methods"]["backbone_base"]["standardized"].get(key) == old_value,
                f"Unchanged B baseline Fmax fields differ: {key}")
    for key in ("standard_num_proteins", "standard_num_go", "standard_num_positive_labels",
                "standard_num_protein_fmax_eligible", "standard_num_empty_gold_proteins"):
        require(new["methods"]["NBS_final"]["standardized"].get(key) == baseline["methods"]["backbone_base"]["standardized"].get(key),
                f"NBS evaluation scope differs: {key}")
    require(after.get("checkpoint_sha256") and after.get("probability_sha256") and after.get("prediction_manifest_sha256"),
            "New prediction must bind checkpoint/probability/manifest SHA256")
    manifest = after.get("manifest", {})
    for key in ("prediction_mode", "encoder_variant", "output_fusion", "uses_dense_backbone_logits_in_forward", "ablation", "source_flags"):
        require(manifest.get(key) == after.get(key), f"Prediction manifest identity differs: {key}")
    require(manifest.get("num_task_go") == 21312 and manifest.get("prediction_space") == "complete_task_classifier_columns",
            "Prediction manifest must declare the full original GO column space")
    for key, expected in (("output_probability_sha256", after["probability_sha256"]),
                          ("checkpoint_sha256", after["checkpoint_sha256"]),
                          ("input_manifest_sha256", after["input_manifest_sha256"])):
        require(manifest.get(key) == expected, f"Prediction manifest source binding differs: {key}")
    value = score(new, "NBS_final")
    result.update(status="达标" if value >= threshold else "未达标", new_Fmax=value, delta_Fmax=value - base,
                  fixed_conditions_verified=True, checkpoint_sha256=after["checkpoint_sha256"],
                  probability_sha256=after["probability_sha256"],
                  fixed_conditions={"task": "bp", "proteins": 1800, "GO_columns": 21312,
                                    "input_manifest_sha256": after["input_manifest_sha256"],
                                    "protein_ids_sha256": after["protein_ids_file_sha256"],
                                    "GO_registry_sha256": after["input_go_registry_sha256"],
                                    "metadata_sha256": new["metadata_provenance"]["sha256"],
                                    "ablation": "full", "output_fusion": "none"})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-comparison", type=Path, required=True,
                        help="Original v087 direct epoch1 comparison; SHA256 is pinned in this script")
    parser.add_argument("--comparison", type=Path, help="New full v088 comparison; omit to record 未验证")
    parser.add_argument("--output", type=Path, required=True, help="New check.json file; existing files are never overwritten")
    args = parser.parse_args(argv)
    require(sha256(args.baseline_comparison) == BASELINE_COMPARISON_SHA256, "Baseline comparison file SHA256 differs from the fixed reference")
    if args.output.exists():
        raise FileExistsError(f"Acceptance output already exists: {args.output}")
    baseline = json.loads(args.baseline_comparison.read_text())
    new = json.loads(args.comparison.read_text()) if args.comparison and args.comparison.is_file() else None
    result = check_reports(baseline, new)
    result["baseline_comparison"] = {"path": str(args.baseline_comparison.resolve()), "sha256": BASELINE_COMPARISON_SHA256}
    result["new_comparison"] = None if new is None else {"path": str(args.comparison.resolve()), "sha256": sha256(args.comparison)}
    if args.comparison and new is None:
        result["reason"] = "Requested new comparison does not yet exist; no performance claim can be made"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    status = main()["status"]
    raise SystemExit(0 if status == "达标" else (2 if status == "未验证" else 1))
