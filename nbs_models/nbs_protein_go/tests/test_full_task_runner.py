"""Exercise real mmap batches, optimizer/checkpoint/RNG state and CPU DDP."""
from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.nbs import train_nbs_full_task as runner
from test_full_task_data_v080 import fixture_data, inference_fixture
from nbs_pg.latence_stores import RoleAwareBaseLogitStore, RoleProbabilitySlice


def _data(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = fixture_data(tmp_path, holdout=1)
    rng = np.random.default_rng(34)
    boxes = {"center": rng.normal(size=(8, 4)).astype(np.float32),
             "offset": rng.uniform(.1, 1, size=(8, 4)).astype(np.float32),
             "stats": np.zeros((8, 2), np.float32)}
    data.stores.full_boxes = SimpleNamespace(
        num_go=8, gather=lambda rows: {key: value[rows] for key, value in boxes.items()}
    )
    # Task columns 0 and 2 are canonical/alt-ID aliases, as allowed by the
    # production alignment contract. Exercise training/resume with that mapping.
    data.stores.task_to_ontology = np.array([0, 1, 0, 3, 4, 5], dtype=np.int64)
    edge = np.array([[0, 6], [1, 6], [2, 7], [3, 7], [4, 6], [5, 7]], dtype=np.int64)
    data.stores.go_relations = {
        "is_a": SimpleNamespace(edge=edge), "has_child": SimpleNamespace(edge=edge[:, ::-1]),
    }
    data.prepare_neighbors(device="cpu", query_batch_size=2)
    return data


def _config(data):
    config = copy.deepcopy(data.config)
    config["full_task"].update(
        learning_rate=.01, weight_decay=0., seed=77, weak_batch=1, core_batch=1,
        warmup_steps=1, steps=4, checkpoints=[2, 4], log_every=1, eval_batch=2,
        model=dict(hidden_dim=16, query_dim=8, decoder_hidden=8, ontology_layers=1,
                   go_chunk=3, dropout=.2),
        loss=dict(hard_pu_k=1, background_pu_k=1, gamma_neg=2., negative_clip=.05),
    )
    return config


def _train(data, config, directory, steps, resume=None):
    args = SimpleNamespace(work_dir=Path(directory), steps=steps, resume=resume)
    torch.manual_seed(config["full_task"]["seed"])
    runner.train(args, config, data, torch.device("cpu"), rank=0, world=1)
    return torch.load(Path(directory) / "latest.pt", map_location="cpu", weights_only=False)


def test_training_and_resume_preserve_updates_random_pu_and_dropout(tmp_path):
    data = _data(tmp_path / "data")
    config = _config(data)
    complete = _train(data, config, tmp_path / "complete", 4)
    first = _train(data, config, tmp_path / "resumed", 2)
    resumed = _train(data, config, tmp_path / "resumed", 4, tmp_path / "resumed/latest.pt")
    assert first["step"] == 2 and resumed["step"] == complete["step"] == 4
    assert torch.count_nonzero(first["model"]["decoder.2.weight"]) > 0
    assert len(resumed["history"]) == 4
    assert resumed["history"][-1]["absolute_logit_delta"] > 0
    assert all(row["full_go_per_protein"] == 6 for row in resumed["history"])
    assert all(row["positive_pairs_per_protein"] >= 1 for row in resumed["history"])
    for key, value in complete["model"].items():
        assert torch.equal(value, resumed["model"][key]), key
    assert torch.equal(complete["rng"][0]["cpu"], resumed["rng"][0]["cpu"])
    assert complete["weak_cycle"]["cursor"] == resumed["weak_cycle"]["cursor"]
    assert [row["loss"] for row in complete["history"]] == [row["loss"] for row in resumed["history"]]
    validation = json.loads((tmp_path / "resumed/validation_history.json").read_text())
    assert [row["step"] for row in validation] == [0, 2, 4]
    assert all(name in validation[-1] for name in ("backbone", "full", "weak_off", "core_off"))
    assert (tmp_path / "resumed/best.pt").is_file()


def test_resume_refuses_changed_supervision_contract(tmp_path):
    data = _data(tmp_path / "data")
    config = _config(data)
    _train(data, config, tmp_path / "run", 2)
    data._contract = {**data.data_contract(), "pseudo_probability_sha256": "changed"}
    with pytest.raises(ValueError, match="same data, core split"):
        _train(data, config, tmp_path / "run", 4, tmp_path / "run/latest.pt")


def test_real_inductive_export_and_existing_metric_cli(tmp_path, monkeypatch):
    data = _data(tmp_path / "data")
    width = 2903  # Real CC vocabulary width accepted by the production evaluator.
    data.num_task_go = data.stores.num_task_go = data.stores.gold_messages.num_go = width
    data.stores.task_to_ontology = np.arange(width, dtype=np.int64)
    data.stores.task_to_ontology[-1] = 0  # Export must retain all 2903 columns.
    rng = np.random.default_rng(51)
    boxes = {"center": rng.normal(size=(width, 4)).astype(np.float32),
             "offset": rng.uniform(.1, 1, size=(width, 4)).astype(np.float32),
             "stats": np.zeros((width, 2), np.float32)}
    data.stores.full_boxes = SimpleNamespace(
        num_go=width, gather=lambda rows: {key: value[rows] for key, value in boxes.items()}
    )
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
                           ablation="full", metric_backend="local_micro")
    runner.evaluate(args, config, data, torch.device("cpu"))
    output = tmp_path / "run/eval_step2/full"
    predictions = np.load(output / "nbs_ind_test_prob.f32.npy")
    assert predictions.shape == (2, width) and predictions.dtype == np.float32
    assert np.all(np.isfinite(predictions))
    metrics = json.loads((output / "metrics/nbs_ind_test_metrics.json").read_text())
    assert metrics["ranking_analysis"]["status"] == "ok"
    assert (output / "metrics/nbs_primary_comparison.tsv").is_file()
    mtime = (output / "nbs_ind_test_prob.f32.npy").stat().st_mtime_ns
    def forbidden_reprediction(*args, **kwargs):
        raise AssertionError("validated existing predictions must be reused for metrics-only retry")
    monkeypatch.setattr(runner.FullTaskGraphModel, "forward", forbidden_reprediction)
    runner.evaluate(args, config, data, torch.device("cpu"))
    assert (output / "nbs_ind_test_prob.f32.npy").stat().st_mtime_ns == mtime


def _ddp_worker(rank, world, init_file, directory, data, config):
    torch.set_num_threads(1)
    torch.manual_seed(config["full_task"]["seed"])
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file://{init_file}")
    try:
        runner.train(SimpleNamespace(work_dir=Path(directory), steps=2, resume=None),
                     config, data, torch.device("cpu"), rank=rank, world=world)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU DDP requires Gloo")
def test_two_rank_training_saves_synchronized_checkpoint(tmp_path):
    data = _data(tmp_path / "data")
    config = _config(data)
    try:
        mp.start_processes(_ddp_worker, args=(2, str(tmp_path / "rendezvous"),
                                            str(tmp_path / "ddp"), data, config),
                           nprocs=2, join=True, start_method="fork")
    except mp.ProcessRaisedException as exc:
        if "gloo/transport/tcp/device.cc" in str(exc) and "Operation not permitted" in str(exc):
            pytest.skip("This execution environment denies Gloo TCP sockets; run CPU/CUDA DDP on the training host")
        raise
    saved = torch.load(tmp_path / "ddp/latest.pt", map_location="cpu", weights_only=False)
    assert saved["world_size"] == 2 and saved["step"] == 2
    assert len(saved["rng"]) == 2
    assert not torch.equal(saved["rng"][0]["cpu"], saved["rng"][1]["cpu"])
    assert torch.count_nonzero(saved["model"]["decoder.2.weight"]) > 0
    assert len(saved["history"]) == 2
