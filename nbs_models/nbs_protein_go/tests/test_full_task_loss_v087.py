"""Loss parity and global-role DDP gradient contracts for v0.8.7."""
from dataclasses import asdict, replace

import pytest
import torch

from nbs_pg.full_task_loss_v081 import FullTaskLossConfigV081, full_task_loss_v081
from nbs_pg.full_task_loss_v087 import FullTaskLossConfigV087, full_task_loss_v087


def fixture():
    generator = torch.Generator().manual_seed(1901)
    x = torch.randn(5, 4, generator=generator)
    weights = torch.randn(4, 9, generator=generator) * .7
    graph_weights = torch.randn(4, 9, generator=generator) * .4
    base = torch.randn(5, 9, generator=generator) * 1.4 - .4
    positive = torch.zeros(5, 9, dtype=torch.bool)
    positive[torch.arange(5), torch.arange(5)] = True
    positive[torch.arange(5), (torch.arange(5) + 3) % 9] = True
    weak = torch.tensor([True, True, True, False, False])
    targets = positive.float()
    targets[:3] *= .73
    support = torch.rand(5, 9, generator=generator) * .8
    exclude = torch.zeros_like(positive)
    exclude[:, 8] = True
    return x, weights, graph_weights, base, targets, positive, weak, support, exclude


@pytest.mark.parametrize("roles", [None, [True] * 5, [False] * 5])
def test_single_rank_default_current_base_is_bitwise_legacy_loss_gradient_and_rng(roles):
    x, w, gw, base, targets, positive, weak, support, exclude = fixture()
    if roles is not None:
        weak = torch.tensor(roles)
    legacy_cfg = FullTaskLossConfigV081(hard_pu_k=2, background_pu_k=2,
                                       ranking_hard_k=2, ranking_random_k=2,
                                       weak_weight=.7, core_weight=1.3)
    current_cfg = FullTaskLossConfigV087(**asdict(legacy_cfg))
    assert current_cfg.mining_source == "current_base"
    results = []
    for loss_fn, config in [(full_task_loss_v081, legacy_cfg), (full_task_loss_v087, current_cfg)]:
        logits = (x @ w).detach().clone().requires_grad_()
        graph = (x @ gw).detach().clone().requires_grad_()
        torch.manual_seed(5517)
        loss, parts = loss_fn(logits, base, targets, positive, weak, config,
                              structural_support=support, graph_logits=graph,
                              pu_exclusion_mask=exclude)
        loss.backward()
        results.append((loss.detach(), parts, logits.grad, graph.grad, torch.get_rng_state().clone()))
    old, new = results
    assert torch.equal(old[0], new[0])
    for name in old[1]:
        assert torch.equal(old[1][name], new[1][name]), name
    for i in (2, 3, 4):
        assert torch.equal(old[i], new[i])


@pytest.mark.parametrize("partition", [((0, 1, 3), (2, 4)), ((0, 1, 2, 3), (4,)),
                                       ((0, 3), (1, 2), (4,))])
def test_ddp_average_gradient_equals_complete_batch_with_zero_weak_rank(partition):
    x, w, gw, base, targets, positive, weak, support, exclude = fixture()
    cfg = FullTaskLossConfigV087(hard_pu_k=2, background_pu_k=0,
                                ranking_hard_k=2, ranking_random_k=0,
                                weak_weight=.7, core_weight=1.3,
                                graph_weight=.13, anchor_weight=.07)
    full_w, full_gw = w.clone().requires_grad_(), gw.clone().requires_grad_()
    full_loss, full_parts = full_task_loss_v087(x @ full_w, base, targets, positive, weak, cfg,
                                               structural_support=support, graph_logits=x @ full_gw,
                                               pu_exclusion_mask=exclude)
    full_loss.backward()
    losses, gradients, graph_gradients, parts = [], [], [], []
    for indices in partition:
        ids = torch.tensor(indices)
        local_w, local_gw = w.clone().requires_grad_(), gw.clone().requires_grad_()
        local_loss, local_parts = full_task_loss_v087(
            x[ids] @ local_w, base[ids], targets[ids], positive[ids], weak[ids], cfg,
            structural_support=support[ids], graph_logits=x[ids] @ local_gw,
            pu_exclusion_mask=exclude[ids], global_role_counts=(3, 2), world_size=len(partition))
        local_loss.backward()
        losses.append(local_loss.detach())
        gradients.append(local_w.grad)
        graph_gradients.append(local_gw.grad)
        parts.append(local_parts)
    torch.testing.assert_close(torch.stack(losses).mean(), full_loss, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(torch.stack(gradients).mean(0), full_w.grad, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(torch.stack(graph_gradients).mean(0), full_gw.grad, rtol=2e-6, atol=1e-7)
    for name in ("contrib_positive", "contrib_hard_pu", "contrib_weak_positive", "contrib_core_positive",
                 "contrib_weak_hard_pu", "contrib_core_hard_pu", "contrib_anchor", "contrib_ranking"):
        torch.testing.assert_close(torch.stack([p[name] for p in parts]).mean(), full_parts[name],
                                   rtol=2e-6, atol=1e-7)
    for indices, local_parts in zip(partition, parts):
        if not weak[list(indices)].any():
            assert local_parts["contrib_weak_positive"].item() == 0
            assert local_parts["contrib_weak_hard_pu"].item() == 0


@pytest.mark.parametrize("global_counts,world,subset", [(None, 1, (0, 1, 2, 3, 4)),
                                                        ((3, 2), 2, (0, 3)),
                                                        ((3, 2), 2, (4,))])
def test_role_contributions_sum_to_their_actual_weighted_total(global_counts, world, subset):
    x, w, gw, base, targets, positive, weak, support, exclude = fixture()
    ids = torch.tensor(subset)
    cfg = FullTaskLossConfigV087(graph_weight=0, hard_pu_k=2, background_pu_k=0,
                                ranking_hard_k=2, ranking_random_k=0)
    _, parts = full_task_loss_v087(x[ids] @ w, base[ids], targets[ids], positive[ids], weak[ids], cfg,
                                  structural_support=support[ids], pu_exclusion_mask=exclude[ids],
                                  global_role_counts=global_counts, world_size=world)
    for term in ("positive", "hard_pu"):
        torch.testing.assert_close(parts[f"contrib_weak_{term}"] + parts[f"contrib_core_{term}"],
                                   parts[f"contrib_{term}"], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("counts,world", [(None, 2), (None, 0), (None, True), (None, 1.0),
                                          ((3, 2), 0), ((3, 2), True), ((3, 2), 2.0),
                                          ((3,), 2), ((3, 2, 1), 2), ((-1, 2), 2),
                                          ((3.0, 2), 2), ((True, 2), 2), ((0, 0), 2),
                                          ((2, 2), 2), (3, 2), ((4, 2), 1)])
def test_invalid_global_role_count_and_world_contract_rejected(counts, world):
    x, w, _, base, targets, positive, weak, _, _ = fixture()
    cfg = FullTaskLossConfigV087(graph_weight=0)
    with pytest.raises(ValueError):
        full_task_loss_v087(x @ w, base, targets, positive, weak, cfg,
                           global_role_counts=counts, world_size=world)


def test_current_only_mining_changes_selected_hard_pu_and_its_gradient():
    logits_value = torch.tensor([[0., -2., 2.]])
    base = torch.tensor([[0., 5., -5.]])
    positive = torch.tensor([[True, False, False]])
    target = positive.float()
    weak = torch.tensor([False])
    config = FullTaskLossConfigV087(positive_weight=0, hard_pu_weight=1,
                                   background_pu_weight=0, medium_pu_weight=0,
                                   anchor_weight=0, ranking_weight=0, graph_weight=0,
                                   hard_pu_k=1, background_pu_k=0,
                                   ranking_hard_k=0, ranking_random_k=0)
    gradients = {}
    for source in ("current_base", "current"):
        z = logits_value.clone().requires_grad_()
        loss, parts = full_task_loss_v087(z, base, target, positive, weak,
                                         replace(config, mining_source=source))
        loss.backward()
        gradients[source] = z.grad
        assert parts["hard_pu_pairs_per_protein"].item() == 1
    assert gradients["current_base"][0, 1] > 0
    assert gradients["current_base"][0, 2] == 0
    assert gradients["current"][0, 1] == 0
    assert gradients["current"][0, 2] > 0
