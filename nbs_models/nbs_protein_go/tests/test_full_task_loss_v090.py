"""Meaningful seen/alias/DDP supervision contracts for the shared GO head."""
import pytest
import torch

from nbs_pg.full_task_loss_v088 import full_task_loss_v088
from nbs_pg.full_task_loss_v090 import (full_task_loss_v090, validate_seen_task_mask,
                                      FullTaskLossConfigV090)
from test_full_task_loss_v088 import example


def test_closed_task_is_exact_existing_bce_with_role_contributions():
    x, weight, target, weak, excluded = example()
    logits = x @ weight
    old, old_parts = full_task_loss_v088(logits, target, weak, pu_exclusion_mask=excluded)
    new, new_parts = full_task_loss_v090(logits, target, weak, pu_exclusion_mask=excluded)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    for key, value in old_parts.items():
        torch.testing.assert_close(new_parts[key], value, rtol=0, atol=0)
    assert new_parts['seen_go_columns'] == 7
    assert new_parts['unseen_go_columns'] == 0


def test_unseen_has_no_negative_gradient_and_no_denominator_mass():
    logits = torch.tensor([[.1, 8., -.5, -12.], [-.7, 9., .2, 15.]], requires_grad=True)
    target = torch.tensor([[1., 0, 0, 0], [0., 0, 1, 0]])
    weak = torch.tensor([False, True])
    seen = torch.tensor([True, False, True, False])
    mapping = torch.arange(4)
    value, parts = full_task_loss_v090(logits, target, weak, seen_task_mask=seen,
                                      task_to_ontology=mapping)
    expected, _ = full_task_loss_v088(logits[:, seen], target[:, seen], weak)
    torch.testing.assert_close(value, expected)
    value.backward()
    assert torch.equal(logits.grad[:, ~seen], torch.zeros_like(logits.grad[:, ~seen]))
    assert parts['full_go_denominator'] == 2
    assert parts['output_go_columns'] == 4
    assert parts['seen_go_columns'] == parts['unseen_go_columns'] == 2
    contaminated = target.clone(); contaminated[0, 1] = 1
    with pytest.raises(ValueError, match='unseen positive'):
        full_task_loss_v090(logits, contaminated, weak, seen_task_mask=seen,
                           task_to_ontology=mapping)
    # Even arbitrary nonfinite untrained scores cannot alter the objective.
    changed = logits.detach().clone(); changed[:, ~seen] = float('nan')
    after, _ = full_task_loss_v090(changed, target, weak, seen_task_mask=seen,
                                  task_to_ontology=mapping)
    torch.testing.assert_close(after, value)


def test_alias_groups_cannot_cross_seen_unseen_and_exclusions_still_work():
    mapping = torch.tensor([0, 0, 2, 3])
    with pytest.raises(ValueError, match='alias'):
        validate_seen_task_mask(torch.tensor([True, False, True, False]), mapping, num_task_go=4)
    seen = torch.tensor([True, True, True, False])
    logits = torch.zeros(1, 4, requires_grad=True)
    value, parts = full_task_loss_v090(logits, torch.tensor([[1., 0, 0, 0]]),
        torch.tensor([False]), seen_task_mask=seen, task_to_ontology=mapping)
    value.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-1/6, 0, 1/6, 0]]))
    assert parts['alias_excluded_pairs_per_protein'] == 1
    assert parts['full_go_denominator'] == 3


@pytest.mark.parametrize('partitions', [((0, 1, 3), (2, 4)), ((0, 1, 2, 3), (4,))])
def test_seen_ddp_role_scaling_matches_global_loss_and_gradient(partitions):
    x, original, target, weak, _ = example()
    seen = torch.tensor([True, True, True, False, False, True, False])
    target = target.clone(); target[:, ~seen] = 0
    mapping = torch.tensor([0, 0, 2, 3, 4, 5, 6])
    weight = original.clone().requires_grad_()
    value, _ = full_task_loss_v090(x @ weight, target, weak,
                                 seen_task_mask=seen, task_to_ontology=mapping)
    value.backward()
    gradients, losses = [], []
    for ids in partitions:
        ids = torch.tensor(ids)
        local = original.clone().requires_grad_()
        partial, _ = full_task_loss_v090(x[ids] @ local, target[ids], weak[ids],
            seen_task_mask=seen, task_to_ontology=mapping,
            global_role_counts=(3, 2), world_size=len(partitions))
        partial.backward(); gradients.append(local.grad); losses.append(partial.detach())
    torch.testing.assert_close(torch.stack(gradients).mean(0), weight.grad, atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(torch.stack(losses).mean(), value)


@pytest.mark.parametrize('mask,mapping', [
    (torch.zeros(3, dtype=torch.bool), torch.arange(3)),
    (torch.ones(3), torch.arange(3)),
    (torch.ones(2, dtype=torch.bool), torch.arange(3)),
    (torch.tensor([True, True, False]), None),
    (torch.ones(3, dtype=torch.bool), torch.arange(3).float()),
])
def test_bad_seen_contract_is_rejected(mask, mapping):
    with pytest.raises(ValueError):
        validate_seen_task_mask(mask, mapping, num_task_go=3)


def test_detached_binary_targets_and_invalid_config():
    target = torch.tensor([[1., 0, 0]], requires_grad=True)
    value, _ = full_task_loss_v090(torch.zeros(1, 3, requires_grad=True), target,
                                 torch.tensor([False]))
    value.backward()
    assert target.grad is None
    with pytest.raises(ValueError):
        FullTaskLossConfigV090(core_weight=-1)
    with pytest.raises(ValueError, match='binary'):
        full_task_loss_v090(torch.zeros(1, 3), torch.tensor([[1., .5, 0]]), torch.tensor([False]))
