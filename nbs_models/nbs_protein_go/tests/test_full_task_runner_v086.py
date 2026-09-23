"""End-to-end CPU training/resume and fixed-objective guarantees for v086.

Uses real mmap stores and graph batches. No GPU/distributed claims are made.
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

from scripts.nbs import train_nbs_full_task_v086 as runner
from nbs_pg.full_task_evidence_v081 import StructuralSupportConfig
from nbs_pg.full_task_loss_v081 import FullTaskLossConfigV081


def _data(tmp_path, variant="dynamic"):
    from test_full_task_data_v083 import data_v083
    from nbs_pg.full_task_data_v084 import FullTaskDataV084
    tmp_path.mkdir(parents=True, exist_ok=True)
    original = data_v083(tmp_path, holdout=1, enabled=False)
    config = copy.deepcopy(original.config)
    config["full_task"].update(variant=variant, sampler=dict(
        mode=variant, seed=8084, first_hop=2, second_hop=1,
        native_pool_per_relation=3, retrieval_pool_size=2, candidate_topk=1,
        stable_fraction=.5, native_dropout=.25, scan_block_size=3,
        cache_dir=str(tmp_path / "v084_pools")))
    data = FullTaskDataV084(config, stores=original.stores)
    rng = np.random.default_rng(34)
    boxes = {"center": rng.normal(size=(8, 4)).astype(np.float32),
             "offset": rng.uniform(.1, 1, size=(8, 4)).astype(np.float32),
             "stats": np.zeros((8, 2), np.float32)}
    data.stores.full_boxes = SimpleNamespace(
        num_go=8, gather=lambda rows: {key: value[rows] for key, value in boxes.items()})
    data.stores.task_to_ontology = np.array([0, 1, 0, 3, 4, 5], dtype=np.int64)
    edge = np.array([[0, 6], [1, 6], [2, 7], [3, 7], [4, 6], [5, 7]], np.int64)
    data.stores.go_relations = {
        "is_a": SimpleNamespace(edge=edge),
        "has_child": SimpleNamespace(edge=edge[:, ::-1]),
    }
    data.prepare_neighbors(device="cpu", query_batch_size=2)
    return data


def _config(data, variant="dynamic"):
    config = copy.deepcopy(data.config)
    config["full_task"].update(
        variant=variant,
        learning_rate=.01, weight_decay=0., seed=77, weak_batch=1, core_batch=1,
        warmup_steps=1, steps=4, checkpoints=[2, 4], checkpoint_interval=0,
        scheduler=dict(name="warmup_cosine", horizon_steps=4, min_lr=.001),
        log_every=1, eval_batch=2,
        model=dict(encoder_variant="preln", pp_edge_dropout=.1, pp_dropout_seed=8086, hidden_dim=16, query_dim=8, decoder_hidden=8, ontology_layers=1,
                   go_chunk=3, dropout=.2, candidate_dropout=.15,
                   activation_checkpointing=True, attention_backend="math"),
        loss=asdict(FullTaskLossConfigV081(graph_weight=0, hard_pu_k=1,
                     background_pu_k=1, ranking_hard_k=1, ranking_random_k=1)),
    )
    return config


def _train(data, config, directory, steps, resume=None):
    args = SimpleNamespace(work_dir=Path(directory), steps=steps,
                           resume=Path(resume) if resume else None)
    torch.manual_seed(config["full_task"]["seed"])
    runner.train(args, config, data, torch.device("cpu"), rank=0, world=1)
    return torch.load(Path(directory) / "latest.pt", map_location="cpu", weights_only=False)


def _assert_state_equal(left, right, path="state"):
    if torch.is_tensor(left):
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_state_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_state_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


@pytest.mark.parametrize("encoder_variant,edge_dropout", [("legacy", 0.), ("preln", 0.), ("preln", .1)])
def test_exact_resume_restores_optimization_and_explicit_sampling_step(tmp_path, monkeypatch, encoder_variant, edge_dropout):
    variant = "fixed"
    data = _data(tmp_path / "data", variant)
    config = _config(data, variant)
    config["full_task"]["model"].update(encoder_variant=encoder_variant, pp_edge_dropout=edge_dropout)
    contexts, training_batches = [], []
    context = {"step": None, "rank": None, "training": False}
    original_context = data.set_sampling_context
    original_batch = data.batch

    def record_context(step=0, rank=0, training=False):
        context.update(step=step, rank=rank, training=training)
        contexts.append(context.copy())
        return original_context(step=step, rank=rank, training=training)

    def record_batch(*args, **kwargs):
        batch = original_batch(*args, **kwargs)
        if context["training"]:
            training_batches.append(context.copy())
            # A current seed's labels must never enter another seed's ego graph.
            seeds = set(map(int, np.asarray(args[0]).reshape(-1)))
            forbidden = seeds | set(map(int, data.validation_ids))
            ids = batch["sampled_global_ids"]
            for key in ("sampled_gold_edge", "sampled_pseudo_edge"):
                sources = ids[batch[key][0]].tolist()
                assert not forbidden.intersection(sources), key
        return batch

    monkeypatch.setattr(data, "set_sampling_context", record_context)
    monkeypatch.setattr(data, "batch", record_batch)
    complete = _train(data, config, tmp_path / "complete", 4)
    complete_contexts = list(training_batches)
    training_batches.clear()
    first = _train(data, config, tmp_path / "resumed", 2)
    # This stale state must have no influence on the resumed step 3 neighborhood.
    data.set_sampling_context(step=900, rank=7, training=True)
    resumed = _train(data, config, tmp_path / "resumed", 4, tmp_path / "resumed/latest.pt")
    assert first["step"] == 2 and resumed["step"] == complete["step"] == 4
    assert resumed["runner_version"] == "0.8.6"
    assert complete_contexts == [{"step": step, "rank": 0, "training": True} for step in range(1, 5)]
    assert training_batches == complete_contexts
    assert any(not item["training"] for item in contexts)
    for field in ("model", "optimizer", "scheduler", "rng", "weak_cycle", "core_cycle"):
        _assert_state_equal(complete[field], resumed[field], field)
    assert len(resumed["history"]) == 4
    assert [row["loss"] for row in complete["history"]] == [row["loss"] for row in resumed["history"]]
    assert resumed["history"][-1]["absolute_logit_delta"] > 0
    assert all(row["binary_positive_targets"] == 1 for row in resumed["history"])
    assert all(row["contrib_graph_aux"] == 0 for row in resumed["history"])
    assert all(row["full_go_per_protein"] == 6 for row in resumed["history"])
    validation = json.loads((tmp_path / "resumed/validation_history.json").read_text())
    assert [row["step"] for row in validation] == [0, 2, 4]
    assert (tmp_path / "resumed/best_core_holdout.pt").is_file()


def test_fixed_dynamic_use_identical_forward_paths_and_only_sampler_changes():
    for ablation in ("full", "weak_off", "core_off", "pp_off", "graph_off", "go_shuffle"):
        assert runner.forward_flags("fixed", ablation) == runner.forward_flags("dynamic", ablation)
    assert runner.forward_flags("fixed")["use_weak_go"]
    assert runner.forward_flags("fixed")["use_core_go"]


@pytest.mark.parametrize("old_version", ["0.8.4", "0.8.5"])
def test_resume_rejects_old_runner_versions(tmp_path, old_version):
    data = _data(tmp_path / "data")
    config = _config(data)
    saved = _train(data, config, tmp_path / "run", 2)
    saved["runner_version"] = old_version
    torch.save(saved, tmp_path / "run/latest.pt")
    with pytest.raises(ValueError, match="fresh run"):
        _train(data, config, tmp_path / "run", 4, tmp_path / "run/latest.pt")


def test_resume_rejects_changed_code_and_backbone_contract(tmp_path, monkeypatch):
    data = _data(tmp_path / "data")
    config = _config(data)
    saved = _train(data, config, tmp_path / "run", 2)
    implementation = saved["training_implementation"]
    assert any("data_v084" in key for key in implementation)
    assert any("model_v084" in key for key in implementation)
    assert any("train_nbs_full_task_v086" in key for key in implementation)
    assert "full_task_loss_v081.py" in implementation
    with monkeypatch.context() as patch:
        patch.setattr(runner, "training_implementation", lambda variant: {"changed": "source"})
        with pytest.raises(ValueError, match="unchanged.*training implementation"):
            _train(data, config, tmp_path / "run", 4, tmp_path / "run/latest.pt")
    data.stores.episode_sampler.base_logit_store.probability_clip = .002
    with pytest.raises(ValueError, match="backbone probability files or clipping"):
        _train(data, config, tmp_path / "run", 4, tmp_path / "run/latest.pt")


def test_nonbinary_supervision_and_auxiliary_loss_are_rejected(tmp_path):
    data = _data(tmp_path / "data")
    config = _config(data)
    model, _ = runner.build_model(data.feature_dim, data.ontology("cpu"),
                                  config["full_task"]["model"], "dynamic", "cpu")
    data.set_sampling_context(step=1, rank=0, training=True)
    model.set_encoder_step(step=1, rank=0)
    batch = data.batch([4, 0])
    batch["targets"][batch["positive_mask"]] = .8
    with pytest.raises(ValueError, match="binary membership"):
        runner.training_objective(model, batch, "dynamic",
            FullTaskLossConfigV081(**config["full_task"]["loss"]), StructuralSupportConfig())
    config["full_task"]["loss"]["graph_weight"] = .1
    with pytest.raises(ValueError, match="graph_weight=0"):
        _train(data, config, tmp_path / "run", 2)


def test_sampling_does_not_change_the_structural_pu_objective():
    """Same logits + same fixed support must give the same loss and gradients."""
    class ConstantPrediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([
                [.2, -.6, -.7, -.8, 2., -.9],
                [-.3, .1, -.5, -.9, 1.7, -1.],
            ]))
            self.register_buffer("task_to_ontology", torch.arange(6))

        def forward(self, batch, **kwargs):
            return {"logits": self.logits}

    positive = torch.zeros(2, 6, dtype=torch.bool)
    positive[0, 0] = positive[1, 1] = True
    fixed_graph = {
        "neighbor_index": torch.tensor([[0, 1], [0, 1]]),
        "neighbor_attr": torch.full((2, 2, 3), .9),
        "anchor_go_edge": torch.tensor([[0, 1], [4, 4]]),
        "anchor_x": torch.ones(2, 3),
    }
    batch = {"targets": positive.float(), "positive_mask": positive,
             "base_logits": torch.full((2, 6), -.5),
             "is_weak": torch.tensor([True, False]), **fixed_graph}
    batch.update({"loss_" + key: value.clone() for key, value in fixed_graph.items()})
    config = FullTaskLossConfigV081(graph_weight=0, hard_pu_k=1, background_pu_k=0,
                                   ranking_hard_k=1, ranking_random_k=0)
    model = ConstantPrediction()
    _, original_loss, original_parts = runner.training_objective(
        model, batch, "fixed", config, StructuralSupportConfig())
    original_loss.backward()
    original_grad = model.logits.grad.clone()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["neighbor_index"].fill_(-1)
    changed["anchor_go_edge"] = torch.empty(2, 0, dtype=torch.long)
    model.zero_grad()
    _, sampled_loss, sampled_parts = runner.training_objective(
        model, changed, "dynamic", config, StructuralSupportConfig())
    sampled_loss.backward()
    assert torch.equal(original_loss, sampled_loss)
    assert torch.equal(original_grad, model.logits.grad)
    assert torch.equal(original_parts["loss"], sampled_parts["loss"])
    # Prove the support is actually used: removing FIXED support changes pressure.
    changed["loss_anchor_go_edge"] = torch.empty(2, 0, dtype=torch.long)
    _, unprotected_loss, _ = runner.training_objective(
        model, changed, "dynamic", config, StructuralSupportConfig())
    assert unprotected_loss > sampled_loss
    changed.pop("loss_anchor_x")
    with pytest.raises(ValueError, match="incomplete fixed loss-support"):
        runner.training_objective(model, changed, "dynamic", config, StructuralSupportConfig())


def test_protein_cycle_keeps_seed_batches_unique_across_pass_boundaries():
    cycle = runner.ProteinCycle(np.arange(7), seed=82)
    visited = []
    for _ in range(12):
        batch = cycle.take(5)
        assert len(np.unique(batch)) == len(batch)
        visited.extend(batch.tolist())
    assert set(visited) == set(range(7))
    saved = copy.deepcopy(cycle.state())
    uninterrupted = [cycle.take(5) for _ in range(3)]
    restored = runner.ProteinCycle(np.arange(7), seed=82)
    restored.restore(saved)
    for expected in uninterrupted:
        assert np.array_equal(expected, restored.take(5))


def test_count_diagnostics_sum_before_dividing():
    # Two rank-batches: 1/1 supported and 0/9 supported. Mean ratios is .5,
    # but the global count ratio is .1.
    row = dict(difficult_gold_pairs=5., difficult_gold_core_supported_pairs=.5,
               core_gold_positive_pairs=50., core_proteins=16.)
    runner.finalize_count_diagnostics(row, window_steps=1, world=2)
    assert row["difficult_gold_pairs_total"] == 10
    assert row["difficult_gold_core_supported_pairs_total"] == 1
    assert row["difficult_gold_core_coverage"] == .1
    assert row["difficult_gold_per_core_protein"] == 10 / 32
    assert row["difficult_gold_fraction_of_core_positive"] == .1
    zero = dict(difficult_gold_pairs=0., difficult_gold_core_supported_pairs=0.,
                core_gold_positive_pairs=2., core_proteins=1.)
    runner.finalize_count_diagnostics(zero, window_steps=2, world=2)
    assert zero["difficult_gold_core_coverage"] is None


@pytest.mark.parametrize("change", ["horizon", "min_lr", "name"])
def test_resume_rejects_changed_schedule_contract(tmp_path, change):
    data = _data(tmp_path / "data")
    config = _config(data)
    _train(data, config, tmp_path / "run", 2)
    changed = copy.deepcopy(config)
    if change == "horizon":
        changed["full_task"]["scheduler"]["horizon_steps"] = 5
    elif change == "min_lr":
        changed["full_task"]["scheduler"]["min_lr"] = .002
    else:
        changed["full_task"]["scheduler"]["name"] = "constant"
    with pytest.raises(ValueError, match="identical scheduler"):
        _train(data, changed, tmp_path / "run", 4, tmp_path / "run/latest.pt")


def test_stop_budget_cannot_silently_redefine_schedule(tmp_path):
    data = _data(tmp_path / "data")
    config = _config(data)
    with pytest.raises(ValueError, match="immutable scheduler.horizon_steps"):
        _train(data, config, tmp_path / "run", 5)


def test_external_development_selection_requires_both_metrics():
    result = {"eligible_for_selection": True, "methods": {
        "B": {"standard_protein_fmax": 60., "standard_micro_pr_auc": 50.},
        "G": {"standard_protein_fmax": 61., "standard_micro_pr_auc": 51.},
    }}
    assert runner.development_selection(result, -float("inf")) == (True, 51.)
    assert runner.development_selection(result, 52.) == (False, 52.)
    result["methods"]["G"]["standard_protein_fmax"] = 59.
    assert runner.development_selection(result, 49.) == (False, 49.)
    result["methods"]["G"] = {"standard_protein_fmax": 61., "standard_micro_pr_auc": 49.}
    assert runner.development_selection(result, -float("inf")) == (False, -float("inf"))


def test_development_hook_is_separate_bound_and_saves_only_eligible_checkpoint(tmp_path, monkeypatch):
    data = _data(tmp_path / "data")
    config = _config(data)
    config["full_task"]["development"] = {"manifest": str(tmp_path / "dev.json")}
    calls = []

    class Development:
        contract = {"manifest_sha256": "v1"}

        def evaluate(self, model, data, device, batch_size, forward_flags=None):
            calls.append((model.training_variant, forward_flags))
            return {"eligible_for_selection": True, "methods": {
                "B": {"standard_protein_fmax": 60., "standard_micro_pr_auc": 50.},
                "G": {"standard_protein_fmax": 61., "standard_micro_pr_auc": 51.},
            }}

    development = Development()
    monkeypatch.setattr(runner.DevelopmentSetV085, "load", lambda path, data: development)
    first = _train(data, config, tmp_path / "run", 2)
    assert len(calls) == 2
    assert all(flags == runner.forward_flags("dynamic") for _, flags in calls)
    assert [row["step"] for row in first["development_history"]] == [0, 2]
    assert first["development_contract"] == {"manifest_sha256": "v1"}
    assert first["best_development_score"] == 51.
    assert (tmp_path / "run/best_development.pt").is_file()
    resumed = _train(data, config, tmp_path / "run", 4, tmp_path / "run/latest.pt")
    assert [row["step"] for row in resumed["development_history"]] == [0, 2, 4]
    development.contract = {"manifest_sha256": "v2"}
    with pytest.raises(ValueError, match="identical development"):
        _train(data, config, tmp_path / "run", 4, tmp_path / "run/nbs_step2.pt")


def test_absent_development_never_claims_or_saves_validated_best(tmp_path):
    data = _data(tmp_path / "data")
    saved = _train(data, _config(data), tmp_path / "run", 2)
    assert saved["development_contract"] is None
    assert saved["development_history"] == []
    assert not (tmp_path / "run/best_development.pt").exists()
    monitor = saved["validations"][-1]
    assert monitor["selection_role"].startswith("core_monitor_only")
    assert "standard_micro_pr_auc" in monitor["full"]
    assert "standard_protein_fmax" in monitor["backbone"]
    row = saved["history"][-1]
    assert row["core_equivalent_passes"] == 2 / len(data.core_ids)
    assert row["scheduler_horizon_steps"] == 4
    assert row["difficult_gold_pairs_total"] >= 0


@pytest.mark.parametrize("field,value", [("encoder_variant", "legacy"), ("pp_edge_dropout", .2), ("pp_dropout_seed", 99)])
def test_resume_rejects_changed_encoder_contract(tmp_path, field, value):
    data = _data(tmp_path / "data", "fixed")
    config = _config(data, "fixed")
    _train(data, config, tmp_path / "run", 2)
    changed = copy.deepcopy(config)
    changed["full_task"]["model"][field] = value
    # Legacy is intentionally a no-DropEdge control.
    if field == "encoder_variant":
        changed["full_task"]["model"]["pp_edge_dropout"] = 0.
    with pytest.raises(ValueError, match="identical model/loss"):
        _train(data, changed, tmp_path / "run", 4, tmp_path / "run/latest.pt")


def test_matched_encoders_preserve_common_initialization_and_rng(tmp_path):
    data = _data(tmp_path / "data", "fixed")
    options = _config(data, "fixed")["full_task"]["model"]
    initializations, random_states = [], []
    for encoder, edge_dropout in (("legacy", 0.), ("preln", 0.), ("preln", .1)):
        torch.manual_seed(91)
        model, config = runner.build_model(data.feature_dim, data.ontology("cpu"),
            {**options, "encoder_variant": encoder, "pp_edge_dropout": edge_dropout}, "fixed", "cpu")
        initializations.append(model.state_dict())
        random_states.append(torch.get_rng_state())
    common = set.intersection(*(set(value) for value in initializations))
    assert common
    for name in common:
        assert torch.equal(initializations[0][name], initializations[1][name]), name
        assert torch.equal(initializations[0][name], initializations[2][name]), name
    assert torch.equal(random_states[0], random_states[1])
    assert torch.equal(random_states[0], random_states[2])
    hashes = runner.training_implementation("fixed")
    assert {"full_task_model_v086.py", "full_task_model_v084.py", "full_task_model_v083.py",
            "full_task_schedule_v085.py", "full_task_loss_v081.py", "train_nbs_full_task_v086.py"} <= hashes.keys()
