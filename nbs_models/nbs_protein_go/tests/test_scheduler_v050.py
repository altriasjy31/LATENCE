from __future__ import annotations

import math

import pytest
import torch

from nbs_pg.training import (
    NBSFixedEpochTrainingConfig,
    build_nbs_scheduler,
    resolve_scheduler_step_plan,
    validate_scheduler_resume_contract,
)


class _Loader:
    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length


def _optimizer() -> torch.optim.Optimizer:
    main = torch.nn.Parameter(torch.tensor(1.0))
    scale = torch.nn.Parameter(torch.tensor(0.0))
    return torch.optim.AdamW(
        [
            {"params": [main], "lr": 1e-4, "weight_decay": 1e-4},
            {"params": [scale], "lr": 5e-4, "weight_decay": 0.0},
        ]
    )


def test_onecycle_q64_coverage_contract() -> None:
    cfg = NBSFixedEpochTrainingConfig(
        epochs=150,
        save_epochs=(100, 150),
        accumulation_steps=1,
        scheduler_step="batch",
        include_scheduler_state=True,
    )
    optimizer = _optimizer()
    scheduler, contract = build_nbs_scheduler(
        optimizer,
        {
            "name": "onecycle",
            "pct_start": 0.05,
            "anneal_strategy": "cos",
            "div_factor": 5.0,
            "final_div_factor": 20.0,
            "three_phase": False,
            "cycle_momentum": False,
        },
        cfg,
        _Loader(191),
        runtime={"world_size": 2},
        episode_config={"num_queries": 64},
        local_sampling_config={"coverage_cycles_per_epoch": 1.0},
    )
    assert scheduler is not None
    assert contract["optimizer_steps_per_epoch"] == 191
    assert contract["total_optimizer_steps"] == 28650
    assert contract["world_size"] == 2
    assert contract["num_queries"] == 64
    assert contract["max_lrs"] == [1e-4, 5e-4]
    # OneCycleLR initializes each group at max_lr / div_factor.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1e-4)


def test_scheduler_step_plan_respects_accumulation_and_smoke_cap() -> None:
    cfg = NBSFixedEpochTrainingConfig(
        epochs=2,
        save_epochs=(2,),
        accumulation_steps=4,
        max_steps_per_epoch=10,
    )
    plan = resolve_scheduler_step_plan(_Loader(191), cfg)
    assert plan["effective_loader_steps_per_epoch"] == 10
    assert plan["optimizer_steps_per_epoch"] == math.ceil(10 / 4)
    assert plan["total_optimizer_steps"] == 6


def test_scheduler_resume_contract_rejects_changed_world_size_or_q() -> None:
    saved = {
        "name": "onecycle",
        "scheduler_step": "batch",
        "epochs": 150,
        "accumulation_steps": 1,
        "world_size": 2,
        "num_queries": 64,
        "effective_loader_steps_per_epoch": 191,
        "optimizer_steps_per_epoch": 191,
        "total_optimizer_steps": 28650,
        "pct_start": 0.05,
        "anneal_strategy": "cos",
        "div_factor": 5.0,
        "final_div_factor": 20.0,
        "three_phase": False,
        "cycle_momentum": False,
        "max_lrs": [1e-4, 5e-4],
    }
    validate_scheduler_resume_contract(saved, dict(saved))
    changed = dict(saved)
    changed["world_size"] = 4
    changed["num_queries"] = 128
    with pytest.raises(ValueError, match="scheduler resume contract mismatch"):
        validate_scheduler_resume_contract(saved, changed)


def test_none_scheduler_keeps_optimizer_lrs() -> None:
    cfg = NBSFixedEpochTrainingConfig(epochs=1, save_epochs=(1,))
    optimizer = _optimizer()
    scheduler, contract = build_nbs_scheduler(
        optimizer,
        {"name": "none"},
        cfg,
        _Loader(5),
    )
    assert scheduler is None
    assert contract["name"] == "none"
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 5e-4]


def test_fixed_epoch_trainer_advances_onecycle_by_optimizer_step(tmp_path) -> None:
    from nbs_pg.training import NBSFixedEpochTrainer, NBSLocalBatch, NBSLossConfig, NBSRunComponents
    from nbs_pg.types import NBSMatchOutput, ProteinGOQueryBatch

    class _Matcher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.graph_delta_scale = torch.nn.Parameter(torch.tensor(0.0))

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))
            self.matcher = _Matcher()

        def forward(self, graph, query, global_go_cache=None, return_aux=False):
            del graph, global_go_cache
            logits = query.base_logits + self.bias + torch.tanh(self.matcher.graph_delta_scale)
            aux = None
            if return_aux:
                aux = {
                    "base_logits": query.base_logits,
                    "applied_graph_delta": logits - query.base_logits,
                    "source_weights": torch.ones(query.num_queries, 1),
                    "null_weight": torch.zeros(query.num_queries),
                }
            return NBSMatchOutput(
                logits=logits,
                labels=query.labels,
                mask=query.mask,
                confidence=query.confidence,
                pseudo_mask=query.pseudo_mask,
                supervision_weight=query.supervision_weight,
                auxiliary=aux,
            )

    def _batch():
        labels = torch.tensor([[1.0, 0.0]])
        query = ProteinGOQueryBatch(
            seed_protein_index=torch.tensor([0]),
            seed_query_index=torch.tensor([0]),
            num_queries=1,
            candidate_protein_index=torch.tensor([0, 1]),
            base_logits=torch.zeros_like(labels),
            labels=labels,
            mask=torch.ones_like(labels, dtype=torch.bool),
            supervision_weight=torch.ones_like(labels),
        )
        return NBSLocalBatch(graph=None, query=query)

    model = _Model()
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.bias], "lr": 1e-3},
            {"params": [model.matcher.graph_delta_scale], "lr": 5e-3},
        ]
    )
    loader = [_batch(), _batch()]
    cfg = NBSFixedEpochTrainingConfig(
        epochs=2,
        save_epochs=(2,),
        output_dir=str(tmp_path),
        amp=False,
        progress_bar=False,
        log_interval=100,
        scheduler_step="batch",
    )
    scheduler, contract = build_nbs_scheduler(
        optimizer,
        {
            "name": "onecycle",
            "pct_start": 0.5,
            "div_factor": 5.0,
            "final_div_factor": 10.0,
            "cycle_momentum": False,
        },
        cfg,
        loader,
        runtime={"world_size": 1},
        episode_config={"num_queries": 1},
    )
    components = NBSRunComponents(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        scheduler=scheduler,
        loss_config=NBSLossConfig(primary="bce"),
        metadata={"scheduler_contract": contract},
    )
    trainer = NBSFixedEpochTrainer(components, cfg, device="cpu", logger=lambda _: None)
    trainer.fit()
    assert scheduler.last_epoch == contract["total_optimizer_steps"]
    checkpoint = torch.load(tmp_path / "nbs_epoch2.pt", map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_contract"]["total_optimizer_steps"] == 4
    assert "scheduler_state_dict" in checkpoint


def test_short_smoke_onecycle_gets_non_degenerate_warmup() -> None:
    cfg = NBSFixedEpochTrainingConfig(
        epochs=1,
        save_epochs=(1,),
        max_steps_per_epoch=20,
        scheduler_step="batch",
    )
    optimizer = _optimizer()
    scheduler, contract = build_nbs_scheduler(
        optimizer,
        {
            "name": "onecycle",
            "pct_start": 0.05,
            "div_factor": 5.0,
            "final_div_factor": 20.0,
            "cycle_momentum": False,
        },
        cfg,
        _Loader(191),
    )
    assert scheduler is not None
    assert contract["requested_pct_start"] == pytest.approx(0.05)
    assert contract["pct_start"] == pytest.approx(0.10)
