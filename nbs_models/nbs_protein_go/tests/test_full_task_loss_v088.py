from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nbs_pg.full_task_evidence_v081 import alias_pu_exclusions
from nbs_pg.full_task_loss_v088 import FullTaskLossConfigV088, compute_training_prior, full_task_loss_v088


def example():
    generator = torch.Generator().manual_seed(884)
    x = torch.randn((5, 3), generator=generator)
    weight = torch.randn((3, 7), generator=generator)
    y = torch.zeros((5, 7))
    y[torch.arange(5), torch.arange(5)] = 1
    y[:, 6] = 1
    weak = torch.tensor([True, True, True, False, False])
    mapping = torch.tensor([0, 0, 2, 3, 4, 5, 6])
    excluded = alias_pu_exclusions(y.bool(), mapping)
    return x, weight, y, weak, excluded


def test_default_role_mixture_and_full_column_mean_match_manual_bce():
    x, w, y, weak, excluded = example()
    z = (x @ w).requires_grad_()
    loss, parts = full_task_loss_v088(z, y, weak, pu_exclusion_mask=excluded)
    elementwise = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    rows = (elementwise * ~excluded).sum(1) / y.shape[1]
    expected = .2 * rows[weak].mean() + .8 * rows[~weak].mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(parts['contrib_positive'] + parts['contrib_negative'], loss)
    for component in ('positive', 'negative'):
        torch.testing.assert_close(parts[f'contrib_core_{component}'] + parts[f'contrib_weak_{component}'],
                                   parts[f'contrib_{component}'])
    assert parts['full_go_denominator'].item() == 7


@pytest.mark.parametrize('partition', [((0, 1, 3), (2, 4)), ((0, 1, 2, 3), (4,)), ((0, 3), (1, 2), (4,))])
def test_global_role_normalization_matches_ddp_average_with_empty_local_role(partition):
    x, weights, y, weak, excluded = example()
    weight = weights.clone().requires_grad_()
    full, full_parts = full_task_loss_v088(x @ weight, y, weak, pu_exclusion_mask=excluded)
    full.backward()
    gradients, losses, all_parts = [], [], []
    for index in partition:
        ids = torch.tensor(index)
        local = weights.clone().requires_grad_()
        loss, parts = full_task_loss_v088(x[ids] @ local, y[ids], weak[ids],
                                         pu_exclusion_mask=excluded[ids],
                                         global_role_counts=(3, 2), world_size=len(partition))
        loss.backward()
        gradients.append(local.grad)
        losses.append(loss.detach())
        all_parts.append(parts)
        if not weak[ids].any():
            assert parts['contrib_weak_positive'].item() == 0
            assert parts['contrib_weak_negative'].item() == 0
    torch.testing.assert_close(torch.stack(gradients).mean(0), weight.grad, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(torch.stack(losses).mean(), full)
    for key in ['contrib_core_positive', 'contrib_core_negative', 'contrib_weak_positive', 'contrib_weak_negative']:
        torch.testing.assert_close(torch.stack([p[key] for p in all_parts]).mean(), full_parts[key])


def test_alias_unknown_is_excluded_but_denominator_and_positive_are_preserved():
    z = torch.tensor([[.8, .4, -.3]], requires_grad=True)
    y = torch.tensor([[1., 0., 0.]])
    alias = alias_pu_exclusions(y.bool(), torch.tensor([0, 0, 1]))
    assert alias.tolist() == [[False, True, False]]
    # Even an overbroad mask cannot remove the known positive.
    alias[:, 0] = True
    loss, parts = full_task_loss_v088(z, y, torch.tensor([False]), pu_exclusion_mask=alias)
    loss.backward()
    expected = (F.softplus(-z[0, 0]) + F.softplus(z[0, 2])) / 3
    torch.testing.assert_close(loss, expected)
    assert z.grad[0, 0] < 0 and z.grad[0, 1] == 0 and z.grad[0, 2] > 0
    assert parts['negative_pairs_per_protein'].item() == 1
    assert parts['alias_excluded_pairs_per_protein'].item() == 1
    assert parts['full_go_denominator'].item() == 3


def test_common_bias_has_nonzero_negative_pressure_below_old_clip():
    bias = torch.tensor(-4., requires_grad=True)
    g, npositive = 21312, 32
    targets = torch.zeros((1, g)); targets[:, :npositive] = 1
    loss, parts = full_task_loss_v088(bias.expand_as(targets), targets, torch.tensor([False]))
    loss.backward()
    expected = torch.sigmoid(bias.detach()) - npositive / g
    torch.testing.assert_close(bias.grad, expected, rtol=1e-5, atol=1e-7)
    assert bias.grad > 0  # Gradient descent lowers the over-prevalent uniform prediction.
    assert parts['contrib_negative'] > 0
    # Independent labels have the exact p-y over full G derivative, without a dead zone.
    z = torch.tensor([[-4., -4.]], requires_grad=True)
    small, _ = full_task_loss_v088(z, torch.tensor([[1., 0.]]), torch.tensor([False]))
    small.backward()
    torch.testing.assert_close(z.grad, (z.detach().sigmoid() - torch.tensor([[1., 0.]])) / 2)
    assert z.grad[0, 1] > 0


def test_common_bias_gradient_with_aliases_uses_full_column_denominator():
    bias = torch.tensor(-4., requires_grad=True)
    y = torch.tensor([[1., 0., 0., 0.]])
    excluded = torch.tensor([[False, True, False, False]])
    loss, _ = full_task_loss_v088(bias.expand_as(y), y, torch.tensor([False]), pu_exclusion_mask=excluded)
    loss.backward()
    torch.testing.assert_close(bias.grad, (3 * bias.detach().sigmoid() - 1) / 4)


def test_loss_consumes_no_random_numbers_and_detaches_targets():
    x, w, y, weak, excluded = example()
    y.requires_grad_()
    z = (x @ w).requires_grad_()
    torch.manual_seed(332)
    before = torch.get_rng_state().clone()
    loss, _ = full_task_loss_v088(z, y, weak, pu_exclusion_mask=excluded)
    loss.backward()
    assert torch.equal(before, torch.get_rng_state())
    assert y.grad is None


@pytest.mark.parametrize('counts,world', [(None, 2), (None, True), (None, 1.), ((3, 2), 0),
                                          ((3, 2), True), ((3,), 2), ((3., 2), 2),
                                          ((2, 2), 2), ((2, 2), 1), (3, 2)])
def test_invalid_global_contract_rejected(counts, world):
    x, w, y, weak, _ = example()
    with pytest.raises(ValueError):
        full_task_loss_v088(x @ w, y, weak, global_role_counts=counts, world_size=world)


@pytest.mark.parametrize('target', [float('nan'), .7, -1., 2.])
def test_only_binary_targets_accepted(target):
    with pytest.raises(ValueError, match='binary'):
        full_task_loss_v088(torch.zeros((1, 2)), torch.tensor([[1., target]]), torch.tensor([False]))


class NoProbabilityCSR:
    def __init__(self, rows):
        self.indptr = np.cumsum([0] + [len(row) for row in rows]).astype(np.int64)
        self.go_idx = np.asarray([go for row in rows for go in row], dtype=np.int64)

    @property
    def probability(self):
        raise AssertionError('prior must not read any probability values')


def prior_data():
    # Core holdout global ID 1 and weak non-training ID 4 uniquely carry GO4.
    gold = NoProbabilityCSR([[0, 0, 1], [4], [1], [], [], []])
    pseudo = NoProbabilityCSR([[2], [4], [1, 2, 2]])
    return SimpleNamespace(num_task_go=5, core_ids=np.array([0, 2]), weak_ids=np.array([5, 3]),
                           validation_ids=np.array([1]), registry=SimpleNamespace(role_row=np.array([0, 1, 2, 0, 1, 2])),
                           stores=SimpleNamespace(gold_messages=gold, pseudo_messages=pseudo))


def test_prior_training_only_binary_unique_membership_ignores_holdout_and_probabilities():
    data = prior_data()
    prior, info = compute_training_prior(data)
    # Core means (.5,1,0,0,0), weak means (0,.5,1,0,0); .8/.2 role mixture.
    np.testing.assert_allclose(prior, np.array([.4, .9, .2, 1e-5, 1e-5]), rtol=1e-6)
    assert prior.dtype == np.float32
    assert info['roles']['core']['positive_memberships'] == 3
    assert info['roles']['weak']['positive_memberships'] == 3
    assert info['normalized_core_weight'] == .8 and info['normalized_weak_weight'] == .2
    assert info['alias_eligibility_corrected'] is False
    # Mutating both excluded rows cannot change the prior.
    data.stores.gold_messages.go_idx[data.stores.gold_messages.indptr[1]] = 3
    data.stores.pseudo_messages.go_idx[data.stores.pseudo_messages.indptr[1]] = 3
    after, _ = compute_training_prior(data)
    np.testing.assert_array_equal(after, prior)


def test_prior_rejects_holdout_accidentally_reintroduced_as_training():
    data = prior_data(); data.core_ids = np.array([0, 1, 2])
    with pytest.raises(ValueError, match='holdout'):
        compute_training_prior(data)


def test_optional_alias_corrected_prior_matches_bce_constant_optimum():
    data = prior_data()
    mapping = torch.tensor([0, 0, 2, 3, 4])
    prior, info = compute_training_prior(data, task_to_ontology=mapping)
    # GO0 is excluded in core protein2 and weak protein5 due to positive GO1:
    # eligible mass .8*.5 + .2*.5=.5; positive mass=.4; optimum .8.
    np.testing.assert_allclose(prior, np.array([.8, .9, .2, 1e-5, 1e-5]), rtol=1e-6)
    assert info['roles']['core']['excluded_alias_memberships'] == 1
    assert info['roles']['weak']['excluded_alias_memberships'] == 1
    y = torch.tensor([[1., 1., 0., 0., 0.], [0., 1., 0., 0., 0.],
                      [0., 1., 1., 0., 0.], [0., 0., 1., 0., 0.]])
    weak = torch.tensor([False, False, True, True])
    bias = torch.logit(torch.from_numpy(prior)).requires_grad_()
    loss, _ = full_task_loss_v088(bias[None].expand_as(y), y, weak,
                                 pu_exclusion_mask=alias_pu_exclusions(y.bool(), mapping))
    loss.backward()
    torch.testing.assert_close(bias.grad[:3], torch.zeros(3), atol=2e-8, rtol=0)


def test_prior_one_role_zero_weight_and_unidentified_alias_columns():
    data = prior_data()
    prior, info = compute_training_prior(data, FullTaskLossConfigV088(core_weight=1, weak_weight=0))
    np.testing.assert_allclose(prior, [.5, .99999, 1e-5, 1e-5, 1e-5], rtol=1e-6)
    assert info['normalized_weak_weight'] == 0
    data.core_ids = np.array([2]); data.weak_ids = np.array([], dtype=np.int64)
    prior, info = compute_training_prior(data, task_to_ontology=np.array([0, 0, 2, 3, 4]))
    assert prior[0] == .5  # Always excluded; objective cannot identify this column.
    assert info['unidentified_columns'] == 1


@pytest.mark.parametrize('kwargs', [{'core_weight': -1}, {'weak_weight': float('nan')},
                                    {'core_weight': 0, 'weak_weight': 0}, {'core_weight': True}])
def test_config_weight_validation(kwargs):
    with pytest.raises(ValueError):
        FullTaskLossConfigV088(**kwargs)


@pytest.mark.parametrize('rank_microbatches', [(((0,), (1,), (2, 3), (4,)),),
                                              (((0,), (1, 3)), ((2,), (4,)))])
def test_accumulation_uses_global_macro_counts_without_dividing_again(rank_microbatches):
    x, weights, y, weak, excluded = example()
    full_w = weights.clone().requires_grad_()
    full_loss, _ = full_task_loss_v088(x @ full_w, y, weak, pu_exclusion_mask=excluded)
    full_loss.backward()
    rank_gradients, rank_losses = [], []
    world = len(rank_microbatches)
    for microbatches in rank_microbatches:
        local_w = weights.clone().requires_grad_()
        summed_loss = 0.
        for indices in microbatches:
            ids = torch.tensor(indices)
            local_loss, _ = full_task_loss_v088(x[ids] @ local_w, y[ids], weak[ids],
                pu_exclusion_mask=excluded[ids], global_role_counts=(3, 2), world_size=world)
            local_loss.backward()
            summed_loss += local_loss.detach()
        rank_gradients.append(local_w.grad)
        rank_losses.append(summed_loss)
    torch.testing.assert_close(torch.stack(rank_gradients).mean(0), full_w.grad, atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(torch.stack(rank_losses).mean(), full_loss, atol=1e-7, rtol=2e-6)


def test_extreme_logits_stay_finite_without_probability_space_logs():
    logits = torch.tensor([[-1000., 1000.]], dtype=torch.float16, requires_grad=True)
    loss, _ = full_task_loss_v088(logits, torch.tensor([[1., 0.]]), torch.tensor([False]))
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    torch.testing.assert_close(logits.grad.float(), torch.tensor([[-.5, .5]]))


@pytest.mark.parametrize('target_value', [0., 1.])
def test_empty_positive_or_negative_group_keeps_full_go_gradient(target_value):
    logits = torch.zeros((1, 3), requires_grad=True)
    loss, _ = full_task_loss_v088(logits, torch.full((1, 3), target_value), torch.tensor([False]))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.full((1, 3), (.5 - target_value) / 3))
