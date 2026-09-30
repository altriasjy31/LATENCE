"""Real mmap/graph CPU training, exact resume, and fixed epoch contracts."""
from __future__ import annotations
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scripts.nbs import train_nbs_full_task_v087 as runner
from nbs_pg.full_task_loss_v087 import FullTaskLossConfigV087
from test_full_task_runner_v086 import _data as old_data, _config as old_config, _assert_state_equal


def _data(tmp_path, variant="fixed"):
    from nbs_pg.full_task_data_v087 import FullTaskDataV087
    original = old_data(tmp_path, variant)
    data = FullTaskDataV087(original.config, stores=original.stores)
    data.ensure_prepared()
    return data


def _config(data, mode="direct", dropout=0.):
    config = old_config(data, "fixed")
    cfg = config["full_task"]
    for key in ("steps", "checkpoints", "warmup_steps"):
        cfg.pop(key, None)
    cfg.update(epochs=2, epoch_unit="weak", checkpoint_epochs=[1, 2], weak_batch=1,
               scheduler=dict(name="epoch_warmup_hold_cosine", min_lr=.001,
                              warmup_epochs=.1, hold_until_epoch=1.),
               loss=asdict(FullTaskLossConfigV087(graph_weight=0, anchor_weight=0 if mode == "direct" else .05,
                          hard_pu_k=1, background_pu_k=1, ranking_hard_k=1, ranking_random_k=1)))
    cfg["model"].update(prediction_mode=mode, source_dropout_query_candidate=dropout,
                        source_dropout_neighbor_candidate=dropout, source_dropout_neighbor_pseudo=dropout,
                        source_dropout_seed=8087, direct_bias_init=-4.)
    return config


def _train(data, config, directory, stop_step=None, resume=None, stop_epoch=None, epochs=None):
    args = SimpleNamespace(work_dir=Path(directory), stop_step=stop_step, stop_epoch=stop_epoch,
                           epochs=epochs, resume=Path(resume) if resume else None)
    torch.manual_seed(config["full_task"]["seed"])
    runner.train(args, config, data, torch.device("cpu"), rank=0, world=1)
    return torch.load(Path(directory) / "latest.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("mode,dropout", [("residual", 0.), ("direct", 0.), ("direct", .4)])
def test_exact_resume_inside_epoch_and_epoch_checkpoints(tmp_path, mode, dropout):
    data = _data(tmp_path / "data", "fixed")
    config = _config(data, mode, dropout)
    complete = _train(data, config, tmp_path / "complete")
    first = _train(data, config, tmp_path / "resume", 1)
    resumed = _train(data, config, tmp_path / "resume", resume=tmp_path / "resume/latest.pt")
    total = 2 * len(data.weak_ids)
    assert first["step"] == 1 and complete["step"] == resumed["step"] == total
    assert resumed["runner_version"] == "0.8.7"
    for key in ("model", "optimizer", "scheduler", "rng", "epoch_stream", "epoch_plan"):
        _assert_state_equal(complete[key], resumed[key], key)
    for key in ("loss", "learning_rate", "weak_exposures", "core_exposures"):
        assert [r[key] for r in complete["history"]] == [r[key] for r in resumed["history"]]
    for epoch in (1, 2):
        checkpoint = torch.load(tmp_path / f"resume/nbs_epoch{epoch}.pt", weights_only=False)
        assert checkpoint["step"] == epoch * len(data.weak_ids)
        assert checkpoint["epoch"] == epoch
    assert complete["history"][-1]["weak_equivalent_passes"] == 2
    assert all(row["contrib_graph_aux"] == 0 for row in complete["history"])
    if mode == "direct":
        assert all(row["contrib_anchor"] == 0 for row in complete["history"])
    assert not list((tmp_path / "resume").glob("eval_step*"))


def test_epoch_stop_and_horizon_configuration_cannot_be_rewritten(tmp_path):
    data = _data(tmp_path / "data", "fixed")
    config = _config(data)
    first = _train(data, config, tmp_path / "run", stop_epoch=1, epochs=2)
    assert first["step"] == len(data.weak_ids)
    assert first["scheduler"]["contract"]["epochs"] == 2
    with pytest.raises(ValueError, match="must match config"):
        _train(data, config, tmp_path / "invalid", epochs=1)
    changed = copy.deepcopy(config)
    changed["full_task"]["epochs"] = 3
    with pytest.raises(ValueError, match="preserve batch"):
        _train(data, changed, tmp_path / "run", resume=tmp_path / "run/latest.pt")
    changed = copy.deepcopy(config)
    changed["full_task"]["scheduler"]["hold_until_epoch"] = 1.5
    with pytest.raises(ValueError, match="identical epochs"):
        _train(data, changed, tmp_path / "run", resume=tmp_path / "run/latest.pt")
    with pytest.raises(ValueError, match="exceed resumed"):
        _train(data, config, tmp_path / "run", stop_epoch=1, resume=tmp_path / "run/latest.pt")


@pytest.mark.parametrize("version", ["0.8.4", "0.8.5", "0.8.6"])
def test_cannot_resume_legacy_checkpoint(tmp_path, version):
    data = _data(tmp_path / "data", "fixed")
    config = _config(data)
    checkpoint = _train(data, config, tmp_path / "run", 1)
    checkpoint["runner_version"] = version
    torch.save(checkpoint, tmp_path / "run/latest.pt")
    with pytest.raises(ValueError, match="fresh run"):
        _train(data, config, tmp_path / "run", resume=tmp_path / "run/latest.pt")


def test_direct_rejects_backbone_anchor_and_old_auxiliary(tmp_path):
    data = _data(tmp_path / "data", "fixed")
    config = _config(data)
    config["full_task"]["loss"]["anchor_weight"] = .1
    with pytest.raises(ValueError, match="anchor_weight=0"):
        _train(data, config, tmp_path / "anchor")
    config["full_task"]["loss"].update(anchor_weight=0, graph_weight=.1)
    with pytest.raises(ValueError, match="graph_weight=0"):
        _train(data, config, tmp_path / "aux")


def test_source_flags_match_interventions():
    assert all(runner.source_flags("full").values())
    assert not any(runner.source_flags("graph_off").values())
    for source in ("query_candidate", "neighbor_candidate", "neighbor_pseudo"):
        flags = runner.source_flags(source + "_off")
        assert flags[source] is False and sum(flags.values()) == 4
    assert sum(runner.source_flags("weak_off").values()) == 2
