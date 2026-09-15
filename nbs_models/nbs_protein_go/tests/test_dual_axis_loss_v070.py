from __future__ import annotations

import torch

from nbs_pg.losses import (
    NBSASLConfig,
    NBSLossWeights,
    masked_asl_protein_column_mean,
    nbs_training_loss,
)
from nbs_pg.types import NBSMatchOutput


def _zero_regularizers(**overrides):
    values = {
        "gold": 0.0,
        "pseudo": 0.0,
        "protein_column": 1.0,
        "base_anchor": 0.0,
        "hierarchy": 0.0,
        "delta_l2": 0.0,
        "routing_balance": 0.0,
        "null_collapse": 0.0,
    }
    values.update(overrides)
    return NBSLossWeights(**values)


def test_protein_column_asl_is_explicit_loss_contribution():
    logits = torch.tensor([[0.4, -0.2], [-0.1, 0.7], [0.3, -0.5]])
    labels = torch.tensor([[0.9, 0.0], [0.0, 0.8], [0.0, 0.0]])
    mask = torch.tensor([[True, True], [True, True], [False, True]])
    pseudo = labels > 0
    output = NBSMatchOutput(
        logits=logits,
        labels=labels,
        mask=mask,
        confidence=torch.ones_like(logits),
        pseudo_mask=pseudo,
        weak_primary_mask=mask,
        supervision_weight=torch.ones_like(logits),
        auxiliary={"base_logits": torch.zeros_like(logits)},
    )
    config = NBSASLConfig(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
    total, parts = nbs_training_loss(
        output,
        weights=_zero_regularizers(),
        protein_column_asl=config,
    )
    expected = masked_asl_protein_column_mean(
        logits,
        labels,
        mask,
        confidence=torch.ones_like(logits),
        gamma_neg=4.0,
        gamma_pos=0.0,
        clip=0.05,
    )
    assert torch.allclose(total, expected)
    assert torch.allclose(parts["protein_column_asl"], expected)
    assert int(parts["weak_primary_supervised_columns"]) == 2


def test_unknown_decoded_anchor_covers_only_unsupervised_pairs():
    logits = torch.zeros(2, 3)
    mask = torch.tensor([[True, False, False], [False, True, False]])
    output = NBSMatchOutput(
        logits=logits,
        labels=torch.zeros_like(logits),
        mask=mask,
        pseudo_mask=torch.zeros_like(mask),
        supervision_weight=torch.ones_like(logits),
        auxiliary={"base_logits": torch.zeros_like(logits)},
    )
    _, parts = nbs_training_loss(
        output,
        weights=_zero_regularizers(protein_column=0.0, base_anchor=1.0),
        base_anchor_scope="unknown_decoded",
    )
    assert int(parts["base_anchor_pairs"]) == 4


def test_column_mean_is_invariant_to_repeating_one_proteins_pairs():
    logits = torch.tensor([[2., -.5], [-1., .7]])
    labels = torch.tensor([[1., 1.], [0., 0.]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    original = masked_asl_protein_column_mean(logits, labels, mask)
    expanded_logits = torch.cat([logits, logits[:, :1].expand(-1, 2)], dim=0)
    expanded_labels = torch.cat([labels, labels[:, :1].expand(-1, 2)], dim=0)
    expanded_mask = torch.cat([mask, torch.tensor([[True, False], [True, False]])], dim=0)
    repeated = masked_asl_protein_column_mean(expanded_logits, expanded_labels, expanded_mask)
    torch.testing.assert_close(original, repeated)


def test_unknown_anchor_moves_toward_base_without_supervising_known_pairs():
    logits = torch.tensor([[1., -1.], [.3, -.3]], requires_grad=True)
    base = torch.tensor([[0., 0.], [.3, -.3]])
    mask = torch.tensor([[False, False], [True, True]])
    output = NBSMatchOutput(logits=logits, labels=torch.zeros_like(logits), mask=mask,
        pseudo_mask=torch.zeros_like(mask), auxiliary={'base_logits': base})
    loss, _ = nbs_training_loss(output, weights=_zero_regularizers(protein_column=0., base_anchor=1.), base_anchor_scope='unknown_decoded')
    loss.backward()
    assert logits.grad[0, 0] > 0
    assert logits.grad[0, 1] < 0
    assert torch.count_nonzero(logits.grad[1]) == 0
