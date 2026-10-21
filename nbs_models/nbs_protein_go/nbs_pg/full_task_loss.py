"""Protein-wise supervision for the full task vocabulary.

The decoder must return every task GO, not a sampled query subset.  Every known
positive participates in the objective.  Other entries remain positive-unlabelled
(PU): a bounded hard subset and a disjoint random subset receive conservative
negative pressure; the rest are anchored to the first-stage prediction.

Normalisation is by protein, independently for the positive, hard-PU and
background branches.  Increasing the GO vocabulary therefore does not dilute
an individual protein's positive supervision or its hard-PU contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class FullTaskLossConfig:
    positive_weight: float = 1.0
    hard_pu_weight: float = 0.2
    background_pu_weight: float = 0.05
    anchor_weight: float = 0.05
    ranking_weight: float = 0.0
    ranking_margin: float = 1.0
    weak_weight: float = 1.0
    core_weight: float = 1.0
    hard_pu_k: int = 64
    background_pu_k: int = 64
    gamma_neg: float = 2.0
    negative_clip: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "positive_weight", "hard_pu_weight", "background_pu_weight",
            "anchor_weight", "ranking_weight", "ranking_margin", "weak_weight",
            "core_weight", "gamma_neg", "negative_clip",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.negative_clip >= 1:
            raise ValueError("negative_clip must be below 1")
        if self.weak_weight + self.core_weight <= 0:
            raise ValueError("At least one protein role must have nonzero weight")
        for name in ("hard_pu_k", "background_pu_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def _topk_mask(scores: Tensor, eligible: Tensor, k: int) -> Tensor:
    """Select up to k eligible entries per row, including short/empty rows."""
    selected = torch.zeros_like(eligible)
    if k == 0:
        return selected
    values, indices = scores.masked_fill(~eligible, -torch.inf).topk(
        min(k, scores.shape[1]), dim=1, sorted=False
    )
    # A row with fewer than k eligible entries contains -inf padding.  It must
    # not accidentally turn a known positive into an unlabelled negative.
    selected.scatter_(1, indices, torch.isfinite(values))
    return selected


def _masked_row_mean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)


def _negative_asl(logits: Tensor, config: FullTaskLossConfig) -> Tensor:
    probability = (logits.sigmoid() - config.negative_clip).clamp(
        min=0, max=1 - torch.finfo(logits.dtype).eps
    )
    return -torch.log1p(-probability) * probability.pow(config.gamma_neg)


def full_task_loss(
    logits: Tensor,
    base_logits: Tensor,
    targets: Tensor,
    positive_mask: Tensor,
    is_weak: Tensor,
    config: FullTaskLossConfig | None = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return the objective and detached scalar diagnostics.

    Args:
        logits: Trainable full task output, shape ``[protein, task GO]``.
        base_logits: First-stage output with the same ordered GO vocabulary.
        targets: Binary core targets or soft weak pseudo probabilities in
            ``[0, 1]``.  Only positions in ``positive_mask`` are used.  Weak
            probabilities remain soft targets, rather than becoming hard ones.
        positive_mask: The COMPLETE known positive set, independent of candidate
            K and irrespective of its first-stage probability.  A known positive
            can never be selected as hard or background PU.
        is_weak: Boolean role vector of shape ``[protein]``.
        config: Branch weights and PU budgets.  Weak/core means are balanced
            separately, so changing their batch ratio does not silently change
            their relative objective weight.

    Positive BCE is confidence-weighted and normalised within each protein.
    The negative terms are clipped asymmetric losses, not assertions that an
    unlabelled GO is a biological negative.  Hard mining uses the maximum of
    detached current and baseline probabilities.  Background mining is uniform
    without replacement over the remaining unknown entries.  PyTorch's current
    RNG controls background reproducibility.

    ``base_objective`` uses the SAME selected pairs and normalisation as the
    trainable objective.  ``objective_gain = base_objective - loss`` is therefore
    an interpretable within-batch fit diagnostic; it is not a held-out metric.
    All loss arithmetic is FP32 even when the graph encoder uses autocast.
    """
    cfg = config or FullTaskLossConfig()
    if logits.ndim != 2 or min(logits.shape) == 0:
        raise ValueError("logits must have non-empty shape [protein, task GO]")
    if any(value.shape != logits.shape for value in (base_logits, targets, positive_mask)):
        raise ValueError("base_logits, targets and positive_mask must match logits")
    if is_weak.shape != (logits.shape[0],):
        raise ValueError("is_weak must have shape [protein]")

    z = logits.float()
    base = base_logits.detach().float()
    y = targets.detach().float()
    positive = positive_mask.detach().bool()
    weak = is_weak.detach().bool()
    unknown = ~positive

    with torch.no_grad():
        mining_scores = torch.maximum(z.detach().sigmoid(), base.sigmoid())
        hard = _topk_mask(mining_scores, unknown, cfg.hard_pu_k)
        background = _topk_mask(
            torch.rand_like(mining_scores), unknown & ~hard, cfg.background_pu_k
        ) if cfg.background_pu_k else torch.zeros_like(positive)
        unselected = unknown & ~(hard | background)

        weak_count = weak.sum().float()
        core_count = (~weak).sum().float()
        active_role_weight = (
            cfg.weak_weight * (weak_count > 0).float()
            + cfg.core_weight * (core_count > 0).float()
        ).clamp_min(1e-8)
        row_weight = torch.where(
            weak,
            cfg.weak_weight / weak_count.clamp_min(1),
            cfg.core_weight / core_count.clamp_min(1),
        ) / active_role_weight
        positive_weight = positive.float() * y.clamp_min(1e-8)

    def reduce_rows(values: Tensor) -> Tensor:
        return (values * row_weight).sum()

    def branches(output: Tensor) -> Dict[str, Tensor]:
        positive_bce = F.binary_cross_entropy_with_logits(output, y, reduction="none")
        negative_asl = _negative_asl(output, cfg)
        result = {
            "positive": reduce_rows(_masked_row_mean(positive_bce, positive_weight)),
            "hard_pu": reduce_rows(_masked_row_mean(negative_asl, hard)),
            "background_pu": reduce_rows(_masked_row_mean(negative_asl, background)),
        }
        if cfg.ranking_weight:
            strongest_pu = output.masked_fill(~(hard | background), -torch.inf).amax(dim=1)
            ranking = F.softplus(cfg.ranking_margin + strongest_pu[:, None] - output)
            result["ranking"] = reduce_rows(_masked_row_mean(ranking, positive_weight))
        else:
            result["ranking"] = output.sum() * 0.0
        return result

    current = branches(z)
    # Bernoulli KL(base || output), with zero baseline value.  Unlike BCE's
    # nonzero target entropy, its magnitude measures deviation from the base.
    base_p = base.sigmoid()
    anchor_kl = (
        base_p * (F.logsigmoid(base) - F.logsigmoid(z))
        + (1 - base_p) * (F.logsigmoid(-base) - F.logsigmoid(-z))
    )
    current["anchor"] = reduce_rows(_masked_row_mean(anchor_kl, unselected))
    loss = (
        cfg.positive_weight * current["positive"]
        + cfg.hard_pu_weight * current["hard_pu"]
        + cfg.background_pu_weight * current["background_pu"]
        + cfg.anchor_weight * current["anchor"]
        + cfg.ranking_weight * current["ranking"]
    )

    with torch.no_grad():
        baseline = branches(base)
        base_objective = (
            cfg.positive_weight * baseline["positive"]
            + cfg.hard_pu_weight * baseline["hard_pu"]
            + cfg.background_pu_weight * baseline["background_pu"]
            + cfg.ranking_weight * baseline["ranking"]
        )
        hard_count = hard.sum().float()
        positive_count = positive.sum().float()
        probability_delta = z.sigmoid() - base_p
        diagnostics = {
            key: value.detach() for key, value in current.items()
        }
        diagnostics.update({
            "loss": loss.detach(),
            "base_objective": base_objective,
            "objective_gain": base_objective - loss.detach(),
            "positive_pairs_per_protein": positive.float().sum(1).mean(),
            "hard_pu_pairs_per_protein": hard.float().sum(1).mean(),
            "background_pu_pairs_per_protein": background.float().sum(1).mean(),
            "positive_coverage": torch.ones_like(positive_count),
            "hard_pu_active_fraction": (
                ((z.sigmoid() > cfg.negative_clip) & hard).sum().float()
                / hard_count.clamp_min(1)
            ),
            "supervised_pair_fraction": (positive | hard | background).float().mean(),
            "positive_delta_probability": (
                (probability_delta * positive).sum() / positive_count.clamp_min(1)
            ),
            "hard_pu_delta_probability": (
                (probability_delta * hard).sum() / hard_count.clamp_min(1)
            ),
        })
    return loss, diagnostics
