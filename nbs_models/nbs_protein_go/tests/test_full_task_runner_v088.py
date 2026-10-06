"""Real CPU integration for the single v088 model, macro masking and resume.

These small mmap/graph fixtures exercise training and prediction. They do not
claim GPU throughput or a successful multi-process distributed launch.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.nbs import train_nbs_full_task_v088 as runner
from nbs_pg.full_task_data_v088 import FullTaskDataV088, RELATIONS
from nbs_pg.full_task_loss_v088 import FullTaskLossConfigV088
from nbs_pg.full_task_metrics_v085 import compute_standard_metrics
from nbs_pg.full_task_model_v088 import FullTaskModelConfigV088
from test_full_task_runner_v086 import _data as old_data, _assert_state_equal
from test_full_task_data_v080 import inference_fixture


def _data(tmp_path, variant="dynamic"):
    old = old_data(tmp_path, variant)
    config = copy.deepcopy(old.config)
    config.setdefault("stage", {})["name"] = "nbs_v088_hetero_tuned"
    config["full_task"]["sampler"].update(
        fanouts=[{name: 1 for name in RELATIONS}, {name: 1 for name in RELATIONS}, {"cosine": 2}],
        max_nodes=100)
    data = FullTaskDataV088(config, stores=old.stores)
    data.ensure_prepared()
    return data


def _config(data, weak_batch=1):
    config = copy.deepcopy(data.config)
    config["full_task"].update(
        learning_rate=.01, weight_decay=0., grad_clip=5., seed=77,
        weak_batch=weak_batch, core_batch=2, accumulation_steps=2,
        epochs=2, epoch_unit="weak", checkpoint_epochs=[1, 2], checkpoint_interval=0,
        scheduler=dict(name="epoch_warmup_hold_cosine", min_lr=.001,
                       warmup_epochs=.1, hold_until_epoch=1.),
        log_every=1, eval_batch=2,
        model=asdict(FullTaskModelConfigV088(hidden_dim=8, sage_layers=3,
            ontology_layers=1, input_dropout=.1, dropout=.2)),
        loss=asdict(FullTaskLossConfigV088()))
    return config


def _train(data, config, directory, *, stop_step=None, stop_epoch=None, epochs=None, resume=None):
    args = SimpleNamespace(work_dir=Path(directory), stop_step=stop_step, stop_epoch=stop_epoch,
                           epochs=epochs, resume=Path(resume) if resume else None)
    torch.manual_seed(config["full_task"]["seed"])
    runner.train(args, config, data, torch.device("cpu"), rank=0, world=1)
    return torch.load(Path(directory) / "latest.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("weak_batch,variant", [(1, "dynamic"), (2, "fixed"), (4, "dynamic")])
def test_exact_resume_with_dropout_and_weak_tail(tmp_path, weak_batch, variant):
    # wb=1 permits an inside-epoch stop and an empty-weak microbatch. wb=4
    # exceeds this fixture's weak population, exercising an unpadded tail.
    data = _data(tmp_path / "data", variant)
    config = _config(data, weak_batch)
    complete = _train(data, config, tmp_path / "complete")
    first = _train(data, config, tmp_path / "resume", stop_step=1)
    data.set_sampling_context(step=900, rank=7, training=True)
    resumed = _train(data, config, tmp_path / "resume", resume=tmp_path / "resume/latest.pt")
    steps_per_epoch = (len(data.weak_ids) + weak_batch - 1) // weak_batch
    assert first["step"] == 1
    assert complete["step"] == resumed["step"] == 2 * steps_per_epoch
    assert resumed["runner_version"] == "0.8.8"
    for key in ("model", "optimizer", "scheduler", "rng", "epoch_stream", "epoch_plan", "validations"):
        _assert_state_equal(complete[key], resumed[key], key)
    for key in ("loss", "grad_norm", "learning_rate", "weak_exposures", "core_exposures"):
        assert [row[key] for row in complete["history"]] == [row[key] for row in resumed["history"]]
    for epoch in (1, 2):
        saved = torch.load(tmp_path / f"resume/nbs_epoch{epoch}.pt", weights_only=False)
        assert saved["step"] == epoch * steps_per_epoch and saved["epoch"] == epoch
    assert resumed["history"][-1]["weak_equivalent_passes"] == 2
    assert all(row["full_go_per_protein"] == 6 for row in resumed["history"])
    assert all(np.isfinite(row["loss"]) and np.isfinite(row["grad_norm"]) for row in resumed["history"])
    assert [row["step"] for row in resumed["validations"]] == [steps_per_epoch, 2 * steps_per_epoch]
    assert all("full" in row and "graph_off" not in row for row in resumed["validations"])
    assert not list((tmp_path / "resume").glob("eval_step*"))
    assert "local_loader_v088.py" in resumed["training_implementation"]


def test_all_microbatches_exclude_entire_macro_and_have_finite_gradients(tmp_path, monkeypatch):
    data = _data(tmp_path / "data")
    config = _config(data, weak_batch=2)
    calls, clipped = [], []
    original_batch = data.batch
    original_clip = runner.torch.nn.utils.clip_grad_norm_

    def batch(ids, *args, **kwargs):
        result = original_batch(ids, *args, **kwargs)
        if data._sampling_training:
            global_ids = np.asarray(kwargs["supervision_seed_ids"])
            forbidden = set(global_ids.tolist()) | set(data.validation_ids.tolist())
            calls.append((data._sampling_step, list(ids), global_ids.tolist()))
            assert len(global_ids) == 4 and len(np.unique(global_ids)) == 4
            for block in result["blocks"]:
                assert not forbidden.intersection(block["src_global_ids"].tolist())
            sampled = result["sampled_global_ids"]
            for name in ("sampled_gold_edge", "sampled_pseudo_edge"):
                assert not forbidden.intersection(sampled[result[name][0]].tolist())
            assert torch.equal(result["targets"], result["positive_mask"].float())
        return result

    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        assert parameters and all(p.grad is not None for p in parameters)
        assert all(torch.isfinite(p.grad).all() for p in parameters)
        clipped.append(len(parameters))
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(data, "batch", batch)
    monkeypatch.setattr(runner.torch.nn.utils, "clip_grad_norm_", clip)
    saved = _train(data, config, tmp_path / "run")
    assert len(calls) == 4 and len(clipped) == 2
    for left, right in zip(calls[::2], calls[1::2]):
        assert left[0] == right[0] and left[2] == right[2]
        assert set(left[1]).isdisjoint(right[1])
        assert set(left[1] + right[1]) == set(left[2])
    assert saved["model"]["classifier.weight"].shape == (6, 8)


def test_partial_smoke_saves_without_initial_or_non_epoch_validation(tmp_path, monkeypatch):
    data = _data(tmp_path / "data")
    config = _config(data)

    def forbidden(*args, **kwargs):
        raise AssertionError("partial operational smoke must not invoke full validation")

    monkeypatch.setattr(runner, "validate", forbidden)
    monkeypatch.setattr(runner, "evaluate_development", forbidden)
    saved = _train(data, config, tmp_path / "smoke", stop_step=1)
    assert saved["step"] == 1 and saved["validations"] == []
    assert saved["development_history"] == []
    assert (tmp_path / "smoke/nbs_step1.pt").is_file()
    assert not (tmp_path / "smoke/validation_history.json").exists()
    assert not (tmp_path / "smoke/best_core_monitor.pt").exists()


def test_resume_rejects_old_versions_and_changed_contracts(tmp_path):
    data = _data(tmp_path / "data")
    config = _config(data)
    original = _train(data, config, tmp_path / "run", stop_step=1)
    path = tmp_path / "run/latest.pt"
    for version in ("0.8.6", "0.8.7"):
        changed = copy.deepcopy(original)
        changed["runner_version"] = version
        torch.save(changed, path)
        with pytest.raises(ValueError, match="fresh run"):
            _train(data, config, path.parent, resume=path)
    torch.save(original, path)
    for field, value, message in (("accumulation_steps", 1, "preserve batch"),
                                  ("epochs", 3, "preserve batch")):
        changed = copy.deepcopy(config)
        changed["full_task"][field] = value
        with pytest.raises(ValueError, match=message):
            _train(data, changed, path.parent, resume=path)
    changed = copy.deepcopy(config)
    changed["full_task"]["model"]["dropout"] = .3
    with pytest.raises(ValueError, match="identical model/loss"):
        _train(data, changed, path.parent, resume=path)
    changed = copy.deepcopy(config)
    changed["full_task"]["scheduler"]["hold_until_epoch"] = 1.5
    with pytest.raises(ValueError, match="identical epochs"):
        _train(data, changed, path.parent, resume=path)
    with pytest.raises(ValueError, match="must match config"):
        _train(data, config, tmp_path / "wrong_horizon", epochs=1)


def test_fmax_pack_is_identical_to_existing_metric_kernel():
    labels = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
    probabilities = np.array([[.5, .5, .501, 0], [1, .5, .499, 0], [.1, .9, 0, 1]], np.float32)
    full = compute_standard_metrics(labels, probabilities)
    packed = runner.metric_pack(labels, probabilities)
    assert packed == {key: full[key] for key in packed}
    assert not any("auprc" in key or "average_precision" in key or "micro_fmax" in key for key in packed)


def test_export_full_axis_standalone_and_reuse(tmp_path, monkeypatch):
    data = _data(tmp_path / "data")
    config = _config(data, weak_batch=2)
    saved = _train(data, config, tmp_path / "run", stop_epoch=1)
    input_dir = inference_fixture(tmp_path / "data", data)
    args = SimpleNamespace(stage="export", checkpoint=tmp_path / "run/latest.pt",
                           input_dir=input_dir, work_dir=tmp_path / "predictions",
                           ablation="full", metadata_file=None)

    def no_external_evaluator(*args, **kwargs):
        raise AssertionError("export must not launch external evaluation")

    monkeypatch.setattr(runner.subprocess, "run", no_external_evaluator)
    runner.evaluate(args, config, data, torch.device("cpu"))
    destination = args.work_dir / f"eval_step{saved['step']}" / "full"
    path = destination / "nbs_ind_test_prob.f32.npy"
    probability = np.load(path)
    assert probability.shape == (2, 6) and np.isfinite(probability).all()
    manifest = json.loads((destination / "nbs_full_task_prediction_manifest.json").read_text())
    assert manifest["output_fusion"] == "none" and manifest["prediction_space"] == "complete_task_classifier_columns"
    assert not manifest["uses_dense_backbone_logits_in_forward"]
    assert not manifest["uses_expert_probability_in_nbs_forward"]
    assert not manifest["uses_modelout_probability_in_nbs_forward"]
    assert all(manifest["source_flags"].values())
    model, _ = runner.build_model(data.feature_dim, data.ontology("cpu"), saved["model_config"], "dynamic", "cpu")
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        batch = data.inference_batch(input_dir, [0, 1], "cpu")
        assert "targets" not in batch and "positive_mask" not in batch
        expected = model(batch).sigmoid().numpy()
    np.testing.assert_array_equal(probability, expected)
    digest = runner.sha256(path)
    runner.evaluate(args, config, data, torch.device("cpu"))
    assert runner.sha256(path) == digest


def test_only_full_mode_and_real_default_entry_config(monkeypatch):
    for variant in ("fixed", "dynamic"):
        assert runner.forward_flags(variant) == {}
    assert all(runner.source_flags("full").values())
    with pytest.raises(ValueError, match="only the complete model"):
        runner.forward_flags("dynamic", "pp_off")
    assert runner.build_latence_nbs_stores.__module__ == "nbs_pg.local_loader_v088"
    captured = {}

    class ReachedLoader(Exception):
        pass

    def capture(config):
        captured.update(config)
        raise ReachedLoader

    monkeypatch.setattr(runner, "build_latence_nbs_stores", capture)
    monkeypatch.setattr(runner.sys, "argv", ["train_nbs_full_task_v088.py", "--stage", "prepare", "--device", "cpu"])
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ReachedLoader):
        runner.main()
    assert captured["stage"]["name"] == "nbs_v088_hetero_tuned"
    assert captured["full_task"]["model"]["encoder_variant"] == "hetero_tuned"
