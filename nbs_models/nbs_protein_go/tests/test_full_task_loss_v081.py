from dataclasses import replace

import pytest
import torch

from nbs_pg.full_task_loss_v081 import FullTaskLossConfigV081, full_task_loss_v081


def isolated(**kwargs):
    config = FullTaskLossConfigV081(
        positive_weight=0, hard_pu_weight=0, background_pu_weight=0,
        medium_pu_weight=0, anchor_weight=0, ranking_weight=0, graph_weight=0,
    )
    return replace(config, **kwargs)


def evaluate(z, y, cfg, **kwargs):
    return full_task_loss_v081(
        z, kwargs.pop("base", z.detach()), y, y > 0,
        kwargs.pop("weak", torch.ones(z.shape[0], dtype=torch.bool)), cfg, **kwargs,
    )


@pytest.mark.parametrize("branch", ["hard", "background", "medium", "ranking", "anchor"])
def test_structural_support_attenuates_each_unknown_gradient_without_weight_cancellation(branch):
    options = {
        "hard": dict(hard_pu_weight=1, hard_pu_k=8),
        "background": dict(background_pu_weight=1, hard_pu_k=0, background_pu_k=8),
        "medium": dict(medium_pu_weight=1, hard_pu_k=0, background_pu_k=0),
        "ranking": dict(ranking_weight=1, ranking_hard_k=8, ranking_random_k=0),
        "anchor": dict(anchor_weight=1, hard_pu_k=0, background_pu_k=0),
    }
    gradients, losses = [], []
    for support_value in (0, 1):
        z = torch.tensor([[-1., 1., 1.]], requires_grad=True)
        y = torch.tensor([[1., 0., 0.]])
        support = torch.full_like(z, support_value, requires_grad=True)
        loss, metrics = evaluate(z, y, isolated(**options[branch]), structural_support=support, base=torch.zeros_like(z))
        loss.backward()
        assert support.grad is None
        assert (z.grad[:, 1:] > 0).all()
        gradients.append(z.grad[:, 1:].clone())
        losses.append(loss.detach())
        assert metrics["supported_unknown_pairs_per_protein"] == 2 * support_value
    # Both unknowns have support; dividing by sum(attenuated_weights) would
    # incorrectly make the two gradients equal instead of fourfold different.
    assert torch.allclose(gradients[1], gradients[0] * .25, atol=1e-7)
    assert torch.allclose(losses[1], losses[0] * .25, atol=1e-7)


def test_all_modelout_positives_are_preserved_and_pu_exclusions_are_respected():
    z = torch.ones(3, 6, requires_grad=True)
    y = torch.tensor([[1., 1., 1., 1., 1., 1.], [.8, 0., .9, 0., 0., 0.], [0., 0., 0., 0., 0., 0.]])
    excluded = torch.zeros_like(y, dtype=torch.bool)
    excluded[:, -1] = True
    cfg = isolated(hard_pu_weight=1, background_pu_weight=1, medium_pu_weight=1, hard_pu_k=2, background_pu_k=2)
    loss, metrics = evaluate(z, y, cfg, pu_exclusion_mask=excluded)
    loss.backward()
    assert (z.grad[y > 0] == 0).all()
    assert (z.grad[excluded] == 0).all()
    assert (z.grad[(y == 0) & ~excluded] > 0).all()
    assert metrics["positive_coverage"] == 1
    assert metrics["pu_excluded_pairs_per_protein"] == pytest.approx(2 / 3)


def test_multi_positive_multi_negative_ranking_has_gradients_for_every_pair_member():
    z = torch.tensor([[-2., -1., 1., 2., 3.]], requires_grad=True)
    y = torch.tensor([[.8, .9, 0., 0., 0.]])
    cfg = isolated(ranking_weight=1, ranking_hard_k=2, ranking_random_k=1)
    loss, metrics = evaluate(z, y, cfg)
    loss.backward()
    assert (z.grad[0, :2] < 0).all()
    assert (z.grad[0, 2:] > 0).all()
    assert metrics["ranking_pu_pairs_per_protein"] == 3
    assert metrics["objective_gain"].abs() < 1e-7


def test_graph_auxiliary_ranks_core_gold_and_weak_positives_without_base_mimic():
    z = torch.zeros(2, 5, requires_grad=True)
    graph = torch.zeros_like(z, requires_grad=True)
    y = torch.tensor([[1., 1., 0., 0., 0.], [.6, .9, 0., 0., 0.]])
    cfg = isolated(graph_weight=.3, ranking_hard_k=3, ranking_random_k=0)
    loss, metrics = evaluate(z, y, cfg, weak=torch.tensor([False, True]), graph_logits=graph)
    loss.backward()
    assert (graph.grad[y > 0] < 0).all()
    assert (graph.grad[y == 0] > 0).all()
    assert torch.all(z.grad == 0)
    assert metrics["loss"] > 0
    assert metrics["base_objective"] == 0
    assert metrics["objective_gain"] == 0
    assert metrics["comparable_objective"] == 0
    assert metrics["objective_gain_excludes_graph_aux"] == 1
    assert torch.allclose(metrics["loss"], metrics["contrib_graph_aux"])


def test_soft_targets_remain_soft_and_role_balance_ignores_batch_ratio():
    cfg = isolated(positive_weight=1)
    z = torch.tensor([[2., -1.], [2., -1.]], requires_grad=True)
    y = torch.tensor([[.6, 0.], [1., 0.]])
    loss, _ = evaluate(z, y, cfg, weak=torch.tensor([True, False]))
    loss.backward()
    assert z.grad[0, 0] > 0  # sigmoid(2) exceeds soft target .6.
    assert z.grad[1, 0] < 0
    repeat = torch.tensor([0, 0, 0, 0, 1])
    repeated, _ = evaluate(z.detach()[repeat], y[repeat], cfg, weak=torch.tensor([True] * 4 + [False]))
    assert torch.allclose(loss, repeated)


def test_full_positive_coverage_is_not_limited_by_candidate_prefix_or_pu_budget():
    z = torch.zeros(2, 21312, requires_grad=True)
    y = torch.zeros_like(z)
    y[0, [0, 17, 256, 21311]] = 1
    y[1, [4, 1900, 20000, 21000]] = .8
    loss, metrics = evaluate(z, y, isolated(positive_weight=1, ranking_weight=.1))
    loss.backward()
    assert (z.grad[y > 0] < 0).all()
    assert metrics["positive_pairs_per_protein"] == 4
    assert metrics["positive_coverage"] == 1


@pytest.mark.parametrize("positive_value", [0., 1.])
def test_empty_positive_and_all_positive_batches_remain_finite(positive_value):
    z = torch.tensor([[20., -20.], [1., -1.]], requires_grad=True)
    graph = torch.zeros_like(z, requires_grad=True)
    y = torch.full_like(z, positive_value)
    loss, metrics = evaluate(z, y, FullTaskLossConfigV081(), graph_logits=graph)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(graph.grad).all()
    assert all(value.ndim == 0 and torch.isfinite(value) for value in metrics.values())
    assert metrics["graph_aux"] == 0


def test_stratified_anchor_does_not_dilute_high_score_constraint_with_easy_background():
    gradients = []
    for width in (3, 21312):
        base = torch.full((1, width), -8.)
        base[0, 0] = 1.
        z = base.clone()
        z[0, 0] += 1.
        z.requires_grad_()
        loss, _ = evaluate(z, torch.zeros_like(z), isolated(anchor_weight=1, hard_pu_k=0, background_pu_k=0), base=base)
        loss.backward()
        gradients.append(z.grad[0, 0])
    assert gradients[0] > 0
    assert torch.allclose(gradients[0], gradients[1])


def test_bfloat16_inputs_produce_fp32_loss_and_finite_gradients():
    z = torch.zeros(2, 8, dtype=torch.bfloat16, requires_grad=True)
    graph = torch.zeros_like(z, requires_grad=True)
    y = torch.zeros_like(z)
    y[:, -1] = 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = evaluate(z, y, FullTaskLossConfigV081(), graph_logits=graph)
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(graph.grad).all()


def test_gain_compares_identical_selections_and_excludes_auxiliary():
    torch.manual_seed(21)
    z = torch.tensor([[1., -.5, .2, 2.]], requires_grad=True)
    y = torch.tensor([[1., .8, 0., 0.]])
    loss, metrics = evaluate(z, y, FullTaskLossConfigV081(), graph_logits=torch.zeros_like(z, requires_grad=True))
    assert loss > 0
    assert metrics["objective_gain"].abs() < 1e-7
    assert torch.allclose(metrics["loss"], metrics["comparable_objective"] + metrics["contrib_graph_aux"])
    assert torch.allclose(metrics["loss"], sum(v for k, v in metrics.items() if k.startswith("contrib_")))


def test_toy_optimization_recovers_multiple_positives_in_final_and_graph_rankings():
    torch.manual_seed(3)
    base = torch.tensor([[-2., -1., 1., 2., 3.], [-1., -2., 2., 3., 1.]])
    z = torch.nn.Parameter(base.clone())
    graph = torch.nn.Parameter(torch.zeros_like(z))
    y = torch.tensor([[1., 1., 0., 0., 0.], [.8, .9, 0., 0., 0.]])
    cfg = FullTaskLossConfigV081(hard_pu_k=3, background_pu_k=0, ranking_hard_k=3, ranking_random_k=0)
    optimizer = torch.optim.Adam([z, graph], lr=.12)
    initial, _ = evaluate(z, y, cfg, base=base, graph_logits=graph, weak=torch.tensor([False, True]))
    for _ in range(60):
        optimizer.zero_grad()
        loss, metrics = evaluate(z, y, cfg, base=base, graph_logits=graph, weak=torch.tensor([False, True]))
        loss.backward()
        optimizer.step()
    assert loss < initial * .25
    assert (z[:, :2].amin(1) > z[:, 2:].amax(1)).all()
    assert (graph[:, :2].amin(1) > graph[:, 2:].amax(1)).all()
    assert (z[:, 2:] < base[:, 2:]).all()
    assert metrics["objective_gain"] > 0


def test_configuration_and_optional_evidence_fail_clearly_when_invalid():
    with pytest.raises(ValueError, match="support_pu_floor"):
        FullTaskLossConfigV081(support_pu_floor=0)
    with pytest.raises(ValueError, match="Ranking objectives"):
        FullTaskLossConfigV081(ranking_hard_k=0, ranking_random_k=0)
    z = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="graph_logits are required"):
        evaluate(z, z, FullTaskLossConfigV081())
    with pytest.raises(ValueError, match="structural_support"):
        evaluate(z, z, isolated(), structural_support=torch.full_like(z, 1.1))
