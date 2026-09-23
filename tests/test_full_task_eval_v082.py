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
module = load("v082_eval", ROOT / "scripts/nbs/eval_nbs_full_task_v082.py")
files = previous.files


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(path, branch="final"):
    checkpoint = path / "nbs_step600.pt"
    checkpoint.write_bytes(b"test checkpoint identity")
    manifest = {
        "version": "0.8.2", "runner_version": "0.8.2", "variant": "dual",
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
    assert result["evaluation_version"] == "0.8.2"
    assert provenance["probability_sha256"] == sha(files / "nbs.npy")
    assert provenance["prediction_manifest_sha256"] == sha(files / "prediction.json")
    assert provenance["input_manifest_sha256"] == sha(files / "input.json")
    assert provenance["checkpoint_sha256"] == sha(files / "nbs_step600.pt")
    assert provenance["checkpoint_file_verified"]
    assert (provenance["step"], provenance["variant"], provenance["branch"]) == (600, "dual", "final")
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


def test_classification_branch_has_C_symbol_and_distinct_deltas(files):
    result = module.main(arguments(files, branch="classification"))
    assert result["methods"]["NBS_final"]["symbol"] == "C"
    assert set(result["deltas"]) == {"C_minus_E", "C_minus_M", "C_minus_B"}
    assert result["prediction_provenance"]["branch"] == "classification"


def test_stale_v081_manifest_rejected(files):
    argv = arguments(files)
    path = files / "prediction.json"
    data = json.loads(path.read_text())
    data["runner_version"] = "0.8.1"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not a v0.8.2 output"):
        module.main(argv)
