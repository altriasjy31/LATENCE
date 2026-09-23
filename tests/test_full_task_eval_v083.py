"""Verify the exported checkpoint/branch actually generated the evaluated array."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
previous = load("v081_eval_test_fixture", Path(__file__).with_name("test_full_task_eval_v081.py"))
module = load("v083_eval", ROOT / "scripts/nbs/eval_nbs_full_task_v083.py")
files = previous.files


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(path, branch="final"):
    checkpoint = path / "nbs_step600.pt"
    checkpoint.write_bytes(b"test checkpoint identity")
    manifest = {
        "version": "0.8.3", "runner_version": "0.8.3", "variant": "graph",
        "branch": branch, "step": 600, "ablation": "full",
        "num_task_go": 2903, "prediction_space": "complete_task_classifier_columns",
        "uses_expert_probability_in_nbs_forward": False,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "output_probability_sha256": sha(path / "nbs.npy"),
        "input_manifest_sha256": sha(path / "input.json"),
        "prediction_implementation": {"model.py": "a" * 64},
    }
    (path / "prediction.json").write_text(json.dumps(manifest))
    return previous.arguments(path) + [
        "--prediction-manifest", str(path / "prediction.json"),
        "--expert-prob", str(path / "expert.npy"),
        "--modelout-prob", str(path / "modelout.npy"), "--references-aligned-to-input"]


def test_evaluation_binds_probabilities_manifest_checkpoint_and_branch(files):
    result = module.main(arguments(files))
    provenance = result["prediction_provenance"]
    assert result["evaluation_version"] == "0.8.3"
    assert provenance["probability_sha256"] == sha(files / "nbs.npy")
    assert provenance["prediction_manifest_sha256"] == sha(files / "prediction.json")
    assert provenance["input_manifest_sha256"] == sha(files / "input.json")
    assert provenance["checkpoint_sha256"] == sha(files / "nbs_step600.pt")
    assert provenance["checkpoint_file_verified"]
    assert (provenance["step"], provenance["variant"], provenance["branch"]) == (600, "graph", "final")
    assert set(result["deltas"]) == {"G_minus_E", "G_minus_M", "G_minus_B"}
    saved = json.loads((files / "out/nbs_ind_test_metrics.json").read_text())
    assert saved["w2s_comparison"]["prediction_provenance"] == provenance


@pytest.mark.parametrize("source,message", [
    ("nbs.npy", "prediction probability"),
    ("input.json", "prepared input"),
    ("nbs_step600.pt", "checkpoint file"),
])
def test_mismatched_source_rejected_before_metric_output(files, source, message):
    argv = arguments(files)
    path = files / source
    if path.suffix == ".npy":
        array = np.load(path)
        array[0, 0] = .25
        np.save(path, array)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=message):
        module.main(argv)
    assert not (files / "out/nbs_ind_test_metrics.json").exists()


def test_classification_branch_rejected(files):
    with pytest.raises(ValueError, match="single final matching"):
        module.main(arguments(files, branch="classification"))


def test_stale_v081_manifest_rejected(files):
    argv = arguments(files)
    path = files / "prediction.json"
    data = json.loads(path.read_text())
    data["runner_version"] = "0.8.1"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not a v0.8.3 output"):
        module.main(argv)


def intervention_results(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"same checkpoint for all interventions")
    method = {"symbol": "G", "primary": {"Fmax": 50.0, "AUPRC": 40.0},
              "micro": {"auprc_micro_exact": 45.0},
              "top_k": {"10": {"precision": {"mean": 0.6}}}}
    reference = {"source_sha256": "e" * 64, "aligned_sha256": "e" * 64}
    for ablation in module.ABLATIONS:
        item = {"evaluation_version": "0.8.3", "task": "bp", "num_samples": 1800,
                "num_classes": 21312, "metric_contract": {"primary_backend": "stage1"},
                "reference_comparison_complete": True,
                "prediction_provenance": {"checkpoint_sha256": sha(checkpoint),
                    "step": 600, "variant": "graph", "branch": "final", "ablation": ablation,
                    "input_manifest_sha256": "a" * 64, "backbone_probability_sha256": "b" * 64,
                    "prediction_implementation": {"model": "m" * 64}},
                "methods": {"NBS_final": json.loads(json.dumps(method)),
                            "expert_prob": method, "stage1_modelout": method, "backbone_base": method},
                "reference_provenance": {"expert_prob": reference, "stage1_modelout": reference},
                "deltas": {"G_minus_E": {"status": "ok"}}}
        if ablation != "full":
            item["methods"]["NBS_final"]["primary"]["Fmax"] -= 2
        directory = tmp_path / "eval_step600" / ablation / "metrics"
        directory.mkdir(parents=True)
        (directory / "nbs_w2s_comparison.json").write_text(json.dumps(item))
    return checkpoint


def test_interventions_same_sources_have_descriptive_deltas(tmp_path):
    checkpoint = intervention_results(tmp_path)
    result = module.main(["--summarize-work-dir", str(tmp_path), "--summary-checkpoint", str(checkpoint)])
    assert result["full_minus_intervention"]["full_minus_graph_off"]["primary"]["Fmax"] == 2
    assert len(result["full_minus_intervention"]) == 5
    assert "Inference dependence only" in result["interpretation"]
    assert (tmp_path / "eval_step600/nbs_graph_effects.json").is_file()


@pytest.mark.parametrize("mismatch", ["checkpoint", "input", "reference", "metric"])
def test_intervention_summary_rejects_mixed_sources(tmp_path, mismatch):
    checkpoint = intervention_results(tmp_path)
    path = tmp_path / "eval_step600/core_off/metrics/nbs_w2s_comparison.json"
    item = json.loads(path.read_text())
    if mismatch == "checkpoint":
        item["prediction_provenance"]["checkpoint_sha256"] = "c" * 64
    elif mismatch == "input":
        item["prediction_provenance"]["input_manifest_sha256"] = "c" * 64
    elif mismatch == "reference":
        item["reference_provenance"]["expert_prob"]["source_sha256"] = "c" * 64
    else:
        item["metric_contract"]["primary_backend"] = "local_micro"
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="differs"):
        module.summarize_ablations(tmp_path, checkpoint)
    assert not (tmp_path / "eval_step600/nbs_graph_effects.json").exists()
