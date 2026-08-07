import json
from pathlib import Path

import torch

from nbs_pg.losses import NBSASLConfig, NBSLossWeights, nbs_training_loss
from nbs_pg.training import NBSLossConfig
from nbs_pg.types import NBSMatchOutput


def _output():
    logits = torch.tensor([[0.2, -0.5, 0.7, -1.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0, 0.82, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    pseudo_mask = torch.tensor([[False, False, True, False]])
    confidence = torch.tensor([[1.0, 1.0, 0.64, 1.0]])
    supervision_weight = torch.tensor([[1.0, 0.2, 1.0, 0.0]])
    return NBSMatchOutput(
        logits=logits,
        labels=labels,
        mask=mask,
        pseudo_mask=pseudo_mask,
        confidence=confidence,
        supervision_weight=supervision_weight,
        auxiliary={
            "base_logits": torch.zeros_like(logits),
            "applied_graph_delta": torch.full_like(logits, 0.1),
        },
    )


def test_pseudo_asl_is_nonzero_and_contributes_to_total():
    output = _output()
    weights = NBSLossWeights(
        gold=1.0,
        pseudo=0.2,
        base_anchor=0.1,
        hierarchy=0.0,
        delta_l2=1e-4,
        routing_balance=0.0,
        null_collapse=0.0,
    )
    loss, parts = nbs_training_loss(
        output,
        primary="asl",
        weights=weights,
        gold_asl=NBSASLConfig(gamma_neg=4, gamma_pos=0, clip=0.05),
        pseudo_asl=NBSASLConfig(gamma_neg=4, gamma_pos=0, clip=0.05),
    )
    assert parts["pseudo_asl"].item() > 0
    assert parts["contrib_pseudo_asl"].item() > 0
    expected = sum(
        parts[name]
        for name in (
            "contrib_gold_asl",
            "contrib_pseudo_asl",
            "contrib_base_anchor",
            "contrib_hierarchy",
            "contrib_delta_l2",
            "contrib_routing_balance",
            "contrib_null_collapse",
        )
    )
    assert torch.allclose(parts["total"], expected, atol=1e-7)
    loss.backward()
    assert output.logits.grad is not None
    assert torch.isfinite(output.logits.grad).all()


def test_gold_and_pseudo_asl_configs_are_independent():
    cfg = NBSLossConfig.from_mapping(
        {
            "primary": "asl",
            "gold_asl": {"gamma_neg": 4, "gamma_pos": 0, "clip": 0.05},
            "pseudo_asl": {"gamma_neg": 2, "gamma_pos": 0, "clip": 0.03},
            "weights": {"gold": 1.0, "pseudo": 0.2, "base_anchor": 0.1},
        }
    )
    assert cfg.gold_asl.gamma_neg == 4
    assert cfg.pseudo_asl.gamma_neg == 2
    assert cfg.gold_asl.clip == 0.05
    assert cfg.pseudo_asl.clip == 0.03


def test_v04_legacy_loss_config_remains_loadable():
    cfg = NBSLossConfig.from_mapping(
        {
            "primary": "asl",
            "asl_gamma_neg": 4.0,
            "asl_gamma_pos": 0.0,
            "asl_clip": 0.05,
            "weights": {"pseudo": 0.0, "anchor": 0.1, "hierarchy": 0.0},
        }
    )
    assert cfg.gold_asl.gamma_neg == 4.0
    assert cfg.pseudo_asl.gamma_neg == 4.0
    assert cfg.weights.base_anchor == 0.1


def test_formal_bp_config_enables_pseudo_asl_and_messages():
    path = Path(__file__).resolve().parents[1] / "configs" / "bp_fixed_epoch_v0.4.5.json"
    cfg = json.loads(path.read_text())
    assert cfg["loss"]["primary"] == "asl"
    assert cfg["loss"]["weights"]["pseudo"] > 0
    assert cfg["loss"]["require_pseudo_signal_per_epoch"] is True
    assert cfg["episode"]["pseudo_positive_per_query"] > 0
    assert cfg["episode"]["pseudo_confidence_power"] == 0.5
    assert cfg["local_sampling"]["include_pseudo_messages"] is True
    assert cfg["stage"]["use_expert_probability_in_forward"] is False
