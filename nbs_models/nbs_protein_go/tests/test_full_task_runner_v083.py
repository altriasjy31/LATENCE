"""Real v083 mmap/graph optimization, exact resume and bound prediction export."""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.nbs import train_nbs_full_task_v083 as runner
from nbs_pg.full_task_evidence_v081 import StructuralSupportConfig
from nbs_pg.full_task_loss_v081 import FullTaskLossConfigV081
from nbs_pg.latence_stores import RoleAwareBaseLogitStore, RoleProbabilitySlice
from test_full_task_data_v080 import inference_fixture
from test_full_task_data_v083 import data_v083


def _data(tmp_path, *, enabled=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = data_v083(tmp_path, holdout=1, enabled=enabled)
    rng = np.random.default_rng(34)
    boxes = {"center": rng.normal(size=(8, 4)).astype(np.float32),
             "offset": rng.uniform(.1, 1, size=(8, 4)).astype(np.float32),
             "stats": np.zeros((8, 2), np.float32)}
    data.stores.full_boxes = SimpleNamespace(
        num_go=8, gather=lambda rows: {key: value[rows] for key, value in boxes.items()})
    # Task columns remain distinct even when two GO identifiers share a box.
    data.stores.task_to_ontology = np.array([0, 1, 0, 3, 4, 5], dtype=np.int64)
    edge = np.array([[0, 6], [1, 6], [2, 7], [3, 7], [4, 6], [5, 7]], np.int64)
    data.stores.go_relations = {
        "is_a": SimpleNamespace(edge=edge),
        "has_child": SimpleNamespace(edge=edge[:, ::-1]),
    }
    data.prepare_neighbors(device="cpu", query_batch_size=2)
    return data


def _config(data, variant="graph"):
    config = copy.deepcopy(data.config)
    config["full_task"].update(
        variant=variant, use_pp_context=variant == "graph",
        learning_rate=.01, weight_decay=0., seed=77, weak_batch=1, core_batch=1,
        warmup_steps=1, steps=4, checkpoints=[2, 4], checkpoint_interval=0,
        log_every=1, eval_batch=2,
        model=dict(hidden_dim=16, query_dim=8, decoder_hidden=8, ontology_layers=1,
                   go_chunk=3, dropout=.2, candidate_dropout=.15,
                   activation_checkpointing=True, attention_backend="math"),
        loss=asdict(FullTaskLossConfigV081(graph_weight=0, hard_pu_k=1,
                     background_pu_k=1, ranking_hard_k=1, ranking_random_k=1)),
    )
    return config


def _train(data, config, directory, steps, resume=None):
    args = SimpleNamespace(work_dir=Path(directory), steps=steps, resume=resume)
    torch.manual_seed(config["full_task"]["seed"])
    runner.train(args, config, data, torch.device("cpu"), rank=0, world=1)
    return torch.load(Path(directory) / "latest.pt", map_location="cpu", weights_only=False)


def test_graph_training_and_resume_preserve_updates_pu_dropout_and_rng(tmp_path):
    data = _data(tmp_path / "data")
    config = _config(data)
    complete = _train(data, config, tmp_path / "complete", 4)
    first = _train(data, config, tmp_path / "resumed", 2)
    resumed = _train(data, config, tmp_path / "resumed", 4, tmp_path / "resumed/latest.pt")
    assert first["step"] == 2 and resumed["step"] == complete["step"] == 4
    assert resumed["runner_version"] == "0.8.3"
    assert torch.count_nonzero(first["model"]["correction_decoder.2.weight"]) > 0
    assert len(resumed["history"]) == 4
    assert resumed["history"][-1]["absolute_logit_delta"] > 0
    assert all(row["full_go_per_protein"] == 6 for row in resumed["history"])
    assert all(row["binary_positive_targets"] == 1 for row in resumed["history"])
    assert all(row["contrib_graph_aux"] == 0 for row in resumed["history"])
    assert any(row["pp_edge_count"] > 0 for row in resumed["history"])
    for key, value in complete["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
    assert torch.equal(complete["rng"][0]["cpu"], resumed["rng"][0]["cpu"])
    assert complete["weak_cycle"]["cursor"] == resumed["weak_cycle"]["cursor"]
    assert [row["loss"] for row in complete["history"]] == [row["loss"] for row in resumed["history"]]
    validation = json.loads((tmp_path / "resumed/validation_history.json").read_text())
    assert [row["step"] for row in validation] == [0, 2, 4]
    assert all(name in validation[-1] for name in (
        "backbone", "full", "weak_off", "core_off", "pp_off", "graph_off", "go_shuffle"))
    assert (tmp_path / "resumed/best_core_holdout.pt").is_file()


@pytest.mark.parametrize("variant", ["local", "no_graph"])
def test_trained_control_remains_trainable_without_pp_context(tmp_path, variant):
    data = _data(tmp_path / "data", enabled=False)
    config = _config(data, variant)
    saved = _train(data, config, tmp_path / variant, 2)
    assert saved["variant"] == variant
    assert saved["history"][-1]["absolute_logit_delta"] > 0
    assert all(row["pp_edge_count"] == 0 for row in saved["history"])
    model, _ = runner.build_model(data.feature_dim, data.ontology("cpu"), saved["model_config"], variant, "cpu")
    model.load_state_dict(saved["model"])
    model.eval()
    batch = data.batch([4, 5])
    flags = runner.forward_flags(variant)
    assert flags["use_pp_context"] is False
    with torch.no_grad():
        output = model(batch, return_details=True, **flags)
    assert not torch.equal(output["logits"], batch["base_logits"])
    if variant == "no_graph":
        assert flags["use_weak_go"] is False and flags["use_core_go"] is False
        changed = {key: value.clone() for key, value in batch.items()}
        changed["candidate_go"].fill_(3)
        changed["candidate_attr"].fill_(.25)
        changed["anchor_go_edge"][1].fill_(0)
        changed["anchor_x"].mul_(7)
        with torch.no_grad():
            altered = model(changed, **flags)
        torch.testing.assert_close(altered, output["logits"], rtol=0, atol=0)


@pytest.mark.parametrize("old_version", ["0.8.1", "0.8.2"])
def test_resume_rejects_old_runner_versions(tmp_path, old_version):
    data = _data(tmp_path / "data")
    config = _config(data)
    saved = _train(data, config, tmp_path / "run", 2)
    saved["runner_version"] = old_version
    torch.save(saved, tmp_path / "run/latest.pt")
    with pytest.raises(ValueError, match="needs a fresh run"):
        _train(data, config, tmp_path / "run", 4, tmp_path / "run/latest.pt")


def test_resume_rejects_changed_code_and_base_probability_contract(tmp_path, monkeypatch):
    data = _data(tmp_path / "data")
    config = _config(data)
    _train(data, config, tmp_path / "run", 2)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "training_implementation", lambda variant: {"changed": "source"})
        with pytest.raises(ValueError, match="unchanged v0.8.3 training implementation"):
            _train(data, config, tmp_path / "run", 4, tmp_path / "run/latest.pt")
    data.stores.episode_sampler.base_logit_store.probability_clip = .002
    with pytest.raises(ValueError, match="backbone probability files or clipping"):
        _train(data, config, tmp_path / "run", 4, tmp_path / "run/latest.pt")


def test_nonbinary_supervision_is_rejected_and_graph_aux_cannot_hide_in_loss(tmp_path):
    data = _data(tmp_path / "data")
    config = _config(data)
    model, _ = runner.build_model(data.feature_dim, data.ontology("cpu"),
                                  config["full_task"]["model"], "graph", "cpu")
    batch = data.batch([4, 0])
    batch["targets"][batch["positive_mask"]] = .8
    with pytest.raises(ValueError, match="binary membership"):
        runner.training_objective(model, batch, "graph",
            FullTaskLossConfigV081(**config["full_task"]["loss"]), StructuralSupportConfig())
    config["full_task"]["loss"]["graph_weight"] = .1
    with pytest.raises(ValueError, match="graph_weight=0"):
        _train(data, config, tmp_path / "run", 2)


def test_real_cc_export_interventions_and_source_bound_metrics(tmp_path, monkeypatch):
    data = _data(tmp_path / "data")
    width = 2903  # The real CC task width is required by the production evaluator.
    data.num_task_go = data.stores.num_task_go = data.stores.gold_messages.num_go = width
    data.stores.task_to_ontology = np.arange(width, dtype=np.int64)
    data.stores.task_to_ontology[-1] = 0  # Alias must not collapse the output columns.
    rng = np.random.default_rng(51)
    boxes = {"center": rng.normal(size=(width, 4)).astype(np.float32),
             "offset": rng.uniform(.1, 1, size=(width, 4)).astype(np.float32),
             "stats": np.zeros((width, 2), np.float32)}
    data.stores.full_boxes = SimpleNamespace(
        num_go=width, gather=lambda rows: {key: value[rows] for key, value in boxes.items()})
    for role, count in (("core", 4), ("weak", 2)):
        np.save(tmp_path / "data" / f"{role}_full_p.npy", np.full((count, width), .2, np.float32))
    data.stores.episode_sampler.base_logit_store = RoleAwareBaseLogitStore([
        RoleProbabilitySlice("core", 0, 4, str(tmp_path / "data/core_full_p.npy")),
        RoleProbabilitySlice("weak", 4, 6, str(tmp_path / "data/weak_full_p.npy")),
    ], num_go=width)
    config = _config(data)
    config["task"] = "cc"
    config["full_task"]["model"]["go_chunk"] = 512
    _train(data, config, tmp_path / "run", 2)
    input_dir = inference_fixture(tmp_path / "data", data)
    probability_path = input_dir / "backbone_ind_test_prob.f16.npy"
    np.save(probability_path, np.full((2, width), .2, np.float16))
    manifest_path = input_dir / "ind_test_input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_probability"]["sha256"] = runner.sha256(probability_path)
    manifest_path.write_text(json.dumps(manifest))
    (input_dir / "protein_ids.txt").write_text("p0\np1\n")
    metadata_path = tmp_path / "metadata.pkl"
    with metadata_path.open("wb") as handle:
        pickle.dump({"ind_test": {"cc": {"proteins": ["p0", "p1"], "prop_annotations": [[0, 1], [2]]}},
                     "train": {"cc": {"proteins": ["c0", "c1"], "prop_annotations": [[0, 1], [2]]}}}, handle)
    args = SimpleNamespace(checkpoint=tmp_path / "run/latest.pt", input_dir=input_dir,
                           metadata_file=metadata_path, work_dir=tmp_path / "run",
                           ablation="full", metric_backend="local_micro", allow_missing_references=True)
    hashes = {}
    for mode in ("full", "pp_off", "go_shuffle"):
        args.ablation = mode
        runner.evaluate(args, config, data, torch.device("cpu"))
        output = tmp_path / f"run/eval_step2/{mode}"
        prediction_path = output / "nbs_ind_test_prob.f32.npy"
        predictions = np.load(prediction_path)
        assert predictions.shape == (2, width) and predictions.dtype == np.float32
        assert np.all(np.isfinite(predictions))
        metrics = json.loads((output / "metrics/nbs_ind_test_metrics.json").read_text())
        assert metrics["ranking_analysis"]["status"] == "ok"
        comparison = json.loads((output / "metrics/nbs_w2s_comparison.json").read_text())
        provenance = comparison["prediction_provenance"]
        assert provenance["ablation"] == mode and provenance["branch"] == "final"
        assert provenance["checkpoint_file_verified"] is True
        assert provenance["probability_sha256"] == runner.sha256(prediction_path)
        assert provenance["input_manifest_sha256"] == runner.sha256(manifest_path)
        assert provenance["manifest"]["uses_modelout_probability_as_target"] is False
        hashes[mode] = provenance["probability_sha256"]
    # These are real interventions on learned predictions, not only labels in a manifest.
    assert hashes["full"] != hashes["pp_off"]
    assert hashes["full"] != hashes["go_shuffle"]
    args.ablation = "full"
    output = tmp_path / "run/eval_step2/full"
    mtime = (output / "nbs_ind_test_prob.f32.npy").stat().st_mtime_ns
    def forbidden_reprediction(*args, **kwargs):
        raise AssertionError("verified cached predictions must not execute another model forward")
    monkeypatch.setattr(runner.FullTaskGraphModelV083, "forward", forbidden_reprediction)
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: commands.append(command))
    runner.evaluate(args, config, data, torch.device("cpu"))
    assert (output / "nbs_ind_test_prob.f32.npy").stat().st_mtime_ns == mtime
    assert len(commands) == 1 and "--prediction-manifest" in commands[0]
    monkeypatch.setattr(runner, "prediction_implementation", lambda variant: {"changed": "source"})
    with pytest.raises(FileExistsError, match="existing predictions differ"):
        runner.evaluate(args, config, data, torch.device("cpu"))
