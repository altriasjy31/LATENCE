"""Two-rank CPU/gloo training and resume with an empty local weak tail.

This exercises actual DDP backward, activation checkpointing and checkpoint
collectives. It does not claim CUDA/NCCL or production-graph validation.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.nbs import train_nbs_full_task_v087 as runner
from nbs_pg.full_task_epochs_v087 import EpochPlanV087, EpochSeedStreamV087
from test_full_task_runner_v087 import _config, _data
from test_full_task_runner_v086 import _assert_state_equal


def _ddp_worker(rank, root_string, init_method):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(root_string)
    try:
        dist.init_process_group("gloo", init_method=init_method, rank=rank,
                                world_size=2, timeout=timedelta(seconds=60))
    except RuntimeError as exc:
        # Some managed runtimes deny gloo's network-device inspection even
        # when ordinary loopback sockets are allowed. This is a transport
        # prerequisite failure, not a successful DDP test or a runner failure.
        if "gloo/transport/tcp/device.cc" in str(exc) and "Operation not permitted" in str(exc):
            raise RuntimeError("GLOO_RUNTIME_PERMISSION_DENIED: " + str(exc)) from exc
        raise
    try:
        # The fixture has mmap files and a locally defined ontology accessor;
        # rebuild it in each spawned process, serially at identical paths.
        # No process reads the fixture while another writes its deterministic
        # contents, and both ranks then retain their own stores/accessors.
        for builder_rank in range(2):
            if rank == builder_rank:
                data = _data(root / "data", "fixed")
            dist.barrier()
        config = _config(data, "direct", .2)
        config["full_task"].update(weak_batch=2, core_batch=1)
        assert len(data.weak_ids) == 2 and len(data.core_ids) >= 2
        assert config["full_task"]["model"]["activation_checkpointing"] is True
        assert config["full_task"]["model"]["pp_edge_dropout"] > 0
        plan = EpochPlanV087(len(data.weak_ids), len(data.core_ids), 2, 1, 2, epochs=2)
        assert plan.steps_per_epoch == 1 and plan.total_steps == 2
        audit_stream = EpochSeedStreamV087(data.weak_ids, data.core_ids, plan,
                                          seed=config["full_task"]["seed"], rank=rank)
        expected = {step: audit_stream.next_batch() for step in (1, 2)}
        original_batch = data.batch
        records = []
        phase = ""

        def record_batch(*args, **kwargs):
            batch = original_batch(*args, **kwargs)
            if data._sampling_training:
                step = data._sampling_step
                seeds = expected[step]
                weak_count = int(batch["is_weak"].sum())
                core_count = int((~batch["is_weak"].bool()).sum())
                assert weak_count == (2 if rank == 0 else 0)
                assert core_count == 1
                forbidden = set(map(int, np.concatenate((
                    seeds["global_weak_ids"], seeds["global_core_ids"], data.validation_ids))))
                nodes = batch["sampled_global_ids"]
                for key in ("sampled_gold_edge", "sampled_pseudo_edge"):
                    assert not forbidden.intersection(nodes[batch[key][0]].tolist()), key
                anchors = nodes[batch["sampled_anchor_index"]]
                assert not forbidden.intersection(anchors[batch["anchor_go_edge"][0]].tolist())
                local_ids = np.asarray(args[0], np.int64)
                loss_anchors = data.anchor_core_ids[np.unique(data._neighbors[local_ids])]
                loss_sources = loss_anchors[batch["loss_anchor_go_edge"][0].cpu().numpy()]
                assert not forbidden.intersection(map(int, loss_sources)), "loss_anchor_go_edge"
                records.append(dict(phase=phase, step=step, weak=weak_count, core=core_count))
            return batch

        data.batch = record_batch
        for phase, directory, stop, resume in (
            ("complete", "complete", None, None),
            ("first", "resumed", 1, None),
            ("resume", "resumed", 2, root / "resumed/latest.pt"),
        ):
            # Reconstruct model/optimizer through the real runner, including
            # deliberately resetting the process RNG before a resume.
            torch.manual_seed(config["full_task"]["seed"])
            args = SimpleNamespace(work_dir=root / directory, stop_step=stop,
                                   stop_epoch=None, epochs=None, resume=resume)
            runner.train(args, copy.deepcopy(config), data, torch.device("cpu"),
                         rank=rank, world=2)
            dist.barrier()
        (root / f"rank{rank}_audit.json").write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(),
                    reason="CPU gloo distributed support is unavailable")
def test_two_rank_empty_weak_tail_checkpointing_and_exact_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    # The FileStore path must not already exist; tmp_path is shared by workers.
    rendezvous = (tmp_path / "gloo_rendezvous").resolve().as_uri()
    context = mp.spawn(_ddp_worker, args=(str(tmp_path), rendezvous), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("two-rank CPU/gloo training or checkpoint collective timed out")
    except mp.ProcessRaisedException as exc:
        if "GLOO_RUNTIME_PERMISSION_DENIED" in str(exc):
            pytest.skip("runtime denied gloo TCP device initialization; real DDP remains unverified")
        raise
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)

    complete = torch.load(tmp_path / "complete/latest.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/latest.pt", map_location="cpu", weights_only=False)
    first = torch.load(tmp_path / "resumed/nbs_epoch1.pt", map_location="cpu", weights_only=False)
    assert first["step"] == 1 and complete["step"] == resumed["step"] == 2
    assert complete["world_size"] == resumed["world_size"] == 2
    for key in ("model", "optimizer", "scheduler", "rng", "epoch_stream", "epoch_plan"):
        _assert_state_equal(complete[key], resumed[key], key)
    for key in ("loss", "learning_rate", "weak_exposures", "core_exposures"):
        assert [row[key] for row in complete["history"]] == [row[key] for row in resumed["history"]]
    assert complete["history"][-1]["weak_exposures"] == 4
    assert complete["history"][-1]["core_exposures"] == 4
    assert len(complete["rng"]) == 2
    assert not torch.equal(complete["rng"][0]["cpu"], complete["rng"][1]["cpu"])
    assert all(torch.isfinite(value).all() for value in complete["model"].values())
    for rank in (0, 1):
        records = json.loads((tmp_path / f"rank{rank}_audit.json").read_text())
        assert [(row["phase"], row["step"]) for row in records] == [
            ("complete", 1), ("complete", 2), ("first", 1), ("resume", 2)]
        assert all(row["weak"] == (2 if rank == 0 else 0) and row["core"] == 1 for row in records)
