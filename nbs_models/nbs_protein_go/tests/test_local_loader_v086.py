"""The shipped v086 stage must pass the real binary-supervision provenance gate."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from nbs_pg import local_loader, local_loader_v086
from scripts.nbs import train_nbs_full_task_v086 as runner


@pytest.fixture
def sources(tmp_path):
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "gold_annotations_manifest.json").write_text(json.dumps({
        "task": "bp", "role": "core", "mode": "train", "source": {"label_key": "prop_annotations"}}))
    manifest = {
        "roles": [{"role": "core", "dataset_mode": "train"}, {"role": "weak", "dataset_mode": "exp_train"}],
        "rare_definition": {"training_annotation_key": "prop_annotations"},
        "model_semantics": {"modelout": {
            "prediction_key": "modelout::mix_expert_base_anchor::decoderprob::expert",
            "decoder_prob_source": "expert", "external_probability_used": True},
            "protein_go_selector": {"scope": "full_task"}},
        "weak_pseudo_targets": {"role": "weak", "comparison": ">", "threshold": .5,
                                "edge_attr_columns": ["modelout_probability"]}}
    inverted = {"indices": {"candidate": {"fixed_degree": 512},
        "pseudo": {"payloads": {"probability": "pseudo.npy"}},
        "gold": {"source_edge_index": str(gold_dir / "edge.npy")}}}
    return dict(weak_manifest=manifest, weak_manifest_path=tmp_path / "weak.json",
                inverted=inverted, inverted_manifest=tmp_path / "inverted.json")


def shipped(preset="legacy"):
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs" / f"bp_full_task_v0.8.6_{preset}.json").read_text())


@pytest.mark.parametrize("preset", ["legacy", "preln", "preln_dropedge"])
def test_new_stages_preserve_all_provenance_checks_and_old_gate(sources, preset):
    config = shipped(preset)
    before = copy.deepcopy(config)
    local_loader_v086._validate_supervision_provenance(config, **sources)
    assert config == before  # Never disguise the new stage as an older one.
    with pytest.raises(ValueError, match="outside explicit v083/v084"):
        local_loader._validate_supervision_provenance(config, **sources)
    for mutation, message in (("threshold", "threshold"), ("payload", "probability payload"),
                               ("gold_source", "source_edge_index")):
        changed = copy.deepcopy(sources)
        if mutation == "threshold":
            changed["weak_manifest"]["weak_pseudo_targets"]["threshold"] = .4
        elif mutation == "payload":
            changed["inverted"]["indices"]["pseudo"]["payloads"] = {}
        else:
            changed["inverted"]["indices"]["gold"].pop("source_edge_index")
        with pytest.raises(ValueError, match=message):
            local_loader_v086._validate_supervision_provenance(config, **changed)


@pytest.mark.parametrize("mutation,message", [("stage", "explicit nbs_v086"),
    ("release", "release_version"), ("mode", "binary_membership"),
    ("probability", "binary membership")])
def test_new_builder_rejects_wrong_consumer_before_opening_data(mutation, message):
    config = shipped()
    if mutation == "stage":
        config["stage"]["name"] = "nbs_v084_fixed"
    elif mutation == "release":
        config["release_version"] = "0.8.5"
    elif mutation == "mode":
        config["full_task"]["weak_target_mode"] = "soft"
    else:
        config["supervision_contract"]["weak_pseudo"]["use_probability_as_soft_target"] = True
    with pytest.raises(ValueError, match=message):
        local_loader_v086.build_latence_nbs_stores(config)


def test_main_prepare_uses_explicit_new_loader_without_mutating_stage(tmp_path, monkeypatch):
    config = shipped("preln")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    stores = object()
    calls = []

    def load(value):
        assert value["stage"]["name"] == "nbs_v086_preln"
        calls.append("new_loader")
        return stores

    def data(value, *, stores):
        assert stores is token
        calls.append("v084_data")
        return SimpleNamespace(prepare_neighbors=lambda **kwargs: "prepared")

    token = stores
    assert runner.build_latence_nbs_stores is local_loader_v086.build_latence_nbs_stores
    monkeypatch.setattr(runner, "build_latence_nbs_stores", load)
    monkeypatch.setattr(runner, "FullTaskData", data)
    monkeypatch.setattr(sys, "argv", ["runner", "--stage", "prepare", "--config", str(path), "--device", "cpu"])
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    runner.main()
    assert calls == ["new_loader", "v084_data"]
    assert "local_loader_v086.py" in runner.prediction_implementation("fixed")
    assert "local_loader.py" in runner.prediction_implementation("fixed")
