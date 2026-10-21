"""Synthetic contract tests only; these are not trained model results."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("accept_v088", ROOT / "scripts/nbs/check_nbs_v088_acceptance.py")
checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)


def reports(value=61.):
    standard = {"standard_protein_fmax": checker.BASELINE_FMAX, "standard_num_proteins": 1800,
                "standard_num_go": 21312, "standard_num_positive_labels": 63688,
                "standard_num_protein_fmax_eligible": 1800, "standard_num_empty_gold_proteins": 0,
                "standard_protein_fmax_threshold": .901, "standard_protein_precision_at_fmax": 71.1805163778,
                "standard_protein_recall_at_fmax": 48.5452071117, "standard_protein_fmax_threshold_step": .001,
                "standard_protein_fmax_grid_0p01": 57.6895108913, "standard_protein_fmax_grid_0p01_threshold": .9}
    baseline = {"evaluation_version": "0.8.7", "task": "bp", "num_samples": 1800, "num_classes": 21312,
                "metric_contract": {"primary_backend": "stage1", "stage1_helper_sha256": "a" * 64},
                "standardized_metric_contract": {"ontology_probability_propagation": False, "threshold_step": .001},
                "metadata_provenance": {"sha256": "b" * 64, "protein_ids_sha256": "c" * 64},
                "prediction_provenance": {"ablation": "full", "input_manifest_sha256": "d" * 64,
                                          "protein_ids_file_sha256": "c" * 64, "input_go_registry_sha256": "e" * 64,
                                          "backbone_probability_sha256": "f" * 64},
                "methods": {"backbone_base": {"primary": {"fmax": 55.6317152769}, "micro": {}, "standardized": standard}}}
    new = copy.deepcopy(baseline); new["evaluation_version"] = "0.8.8"
    new["evaluation_scope"] = "standard_protein_fmax_only"
    new["metric_contract"] = {"primary_backend": "not_computed", "evaluation_scope": "standard_protein_fmax_only"}
    p = new["prediction_provenance"]
    p.update(branch="final", prediction_mode="direct", encoder_variant="hetero_tuned", output_fusion="none",
             uses_dense_backbone_logits_in_forward=False, checkpoint_sha256="1" * 64,
             probability_sha256="2" * 64, prediction_manifest_sha256="3" * 64,
             source_flags=dict.fromkeys(("query_candidate", "neighbor_candidate", "neighbor_pseudo", "core", "pp"), True))
    p["manifest"] = {key: copy.deepcopy(p[key]) for key in ("prediction_mode", "encoder_variant", "output_fusion", "uses_dense_backbone_logits_in_forward", "ablation", "source_flags")}
    p["manifest"].update(num_task_go=21312, prediction_space="complete_task_classifier_columns",
                          output_probability_sha256=p["probability_sha256"], checkpoint_sha256=p["checkpoint_sha256"], input_manifest_sha256=p["input_manifest_sha256"])
    new["methods"]["NBS_final"] = {"primary": {"fmax": 99.}, "standardized": dict(standard, standard_protein_fmax=value)}
    return baseline, new


def test_sixty_is_insufficient_for_required_three_point_gain():
    baseline, new = reports(60.)
    result = checker.check_reports(baseline, new)
    assert result["status"] == "未达标"
    assert result["effective_threshold"] == pytest.approx(60.72314936447243)
    assert result["new_Fmax"] == 60.  # primary.fmax=99 must not affect acceptance
    assert result["delta_Fmax"] < 3


def test_exact_gate_boundary_and_absent_result():
    baseline, new = reports(checker.BASELINE_FMAX + 3)
    assert checker.check_reports(baseline, new)["status"] == "达标"
    pending = checker.check_reports(baseline)
    assert pending["status"] == "未验证" and pending["new_Fmax"] is None


@pytest.mark.parametrize("field", ["input_manifest_sha256", "protein_ids_file_sha256", "input_go_registry_sha256", "backbone_probability_sha256"])
def test_input_mismatch_is_not_treated_as_low_score(field):
    baseline, new = reports()
    new["prediction_provenance"][field] = "9" * 64
    with pytest.raises(ValueError, match="identity differs"):
        checker.check_reports(baseline, new)


@pytest.mark.parametrize("mutation", ["metadata", "metric", "baseline", "classes", "ablation", "fusion"])
def test_fixed_conditions_cannot_be_changed(mutation):
    baseline, new = reports()
    if mutation == "metadata": new["metadata_provenance"]["sha256"] = "0" * 64
    elif mutation == "metric": new["standardized_metric_contract"]["threshold_step"] = .01
    elif mutation == "baseline": new["methods"]["backbone_base"]["standardized"]["standard_protein_fmax"] = 56.
    elif mutation == "classes": new["num_classes"] = 100
    elif mutation == "ablation": new["prediction_provenance"]["ablation"] = "core_off"
    else: new["prediction_provenance"]["output_fusion"] = "B_plus_G"
    with pytest.raises(ValueError): checker.check_reports(baseline, new)
