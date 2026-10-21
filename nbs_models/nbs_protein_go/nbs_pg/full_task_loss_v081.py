"""Evidence-aware PU supervision and full-positive ranking for NBS v0.8.1.

Structural support is fixed evidence computed outside the trainable model.  It
only attenuates pressure on unknown labels; it NEVER creates positive labels.
The nonzero attenuation floor is a conservative hyperparameter to validate on a
reliable development set, not a claim that supported unknowns are true positives.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class FullTaskLossConfigV081:
    positive_weight: float = 1.0
    hard_pu_weight: float = 0.2
    background_pu_weight: float = 0.05
    medium_pu_weight: float = 0.05
    anchor_weight: float = 0.05
    ranking_weight: float = 0.1
    graph_weight: float = 0.1
    ranking_margin: float = 1.0
    ranking_hard_k: int = 16
    ranking_random_k: int = 16
    weak_weight: float = 1.0
    core_weight: float = 1.0
    hard_pu_k: int = 64
    background_pu_k: int = 64
    gamma_neg: float = 2.0
    negative_clip: float = 0.05
    support_pu_floor: float = 0.25
    anchor_high_base_threshold: float = 0.1
    anchor_drift_threshold: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "positive_weight", "hard_pu_weight", "background_pu_weight",
            "medium_pu_weight", "anchor_weight", "ranking_weight", "graph_weight",
            "ranking_margin", "weak_weight", "core_weight", "gamma_neg",
            "negative_clip", "support_pu_floor", "anchor_high_base_threshold",
            "anchor_drift_threshold",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.negative_clip >= 1:
            raise ValueError("negative_clip must be below 1")
        if not 0 < self.support_pu_floor <= 1:
            raise ValueError("support_pu_floor must be in (0, 1]")
        if not 0 < self.anchor_high_base_threshold < 1:
            raise ValueError("anchor_high_base_threshold must be in (0, 1)")
        if not 0 < self.anchor_drift_threshold < 1:
            raise ValueError("anchor_drift_threshold must be in (0, 1)")
        if self.weak_weight + self.core_weight <= 0:
            raise ValueError("At least one protein role must have nonzero weight")
        for name in ("hard_pu_k", "background_pu_k", "ranking_hard_k", "ranking_random_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (self.ranking_weight or self.graph_weight) and not (self.ranking_hard_k or self.ranking_random_k):
            raise ValueError("Ranking objectives require at least one ranking PU")


def _topk_mask(scores: Tensor, eligible: Tensor, k: int) -> Tensor:
    selected = torch.zeros_like(eligible)
    if not k:
        return selected
    values, indices = scores.masked_fill(~eligible, -torch.inf).topk(
        min(k, scores.shape[1]), dim=1, sorted=False
    )
    selected.scatter_(1, indices, torch.isfinite(values))
    return selected


def _count_normalised(values: Tensor, eligible: Tensor, attenuation: Tensor) -> Tensor:
    """Do not divide by attenuated weights: that would cancel PU protection."""
    return (values * eligible * attenuation).sum(1) / eligible.sum(1).clamp_min(1)


def _negative_asl(logits: Tensor, cfg: FullTaskLossConfigV081) -> Tensor:
    probability = (logits.sigmoid() - cfg.negative_clip).clamp(
        min=0, max=1 - torch.finfo(logits.dtype).eps
    )
    return -torch.log1p(-probability) * probability.pow(cfg.gamma_neg)


def full_task_loss_v081(
    logits: Tensor,
    base_logits: Tensor,
    targets: Tensor,
    positive_mask: Tensor,
    is_weak: Tensor,
    config: FullTaskLossConfigV081 | None = None,
    *,
    structural_support: Tensor | None = None,
    graph_logits: Tensor | None = None,
    pu_exclusion_mask: Tensor | None = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return FP32 loss and detached scalar diagnostics.

    Every known positive participates in confidence-weighted soft BCE and, when
    any PU is available, ranking against multiple negatives.  Weak probabilities
    stay soft.  Graph auxiliary ranking uses the same targets/PU selections but
    independent graph logits; no backbone target is imposed on that branch.

    Unknowns may be excluded entirely using ``pu_exclusion_mask`` (e.g. known
    canonical aliases).  Structural support must have shape [B,G] and range
    [0,1]; it is detached and attenuates ASL, ranking and KL by a factor between
    ``support_pu_floor`` and one.  It never changes the complete positive mask.

    Hard mining uses detached max(current, base); background is uniform over
    remaining unknowns.  Medium PU covers remaining scores above ASL clip.  KL
    on unknowns outside hard/background is stratified into high-base,
    positive-drift, and background groups.  It intentionally overlaps medium PU
    so the high-score anchor stratum is not emptied by medium-PU selection.

    ``loss`` includes graph auxiliary ranking.  ``comparable_objective`` excludes
    that auxiliary, for which no backbone counterpart exists. ``base_objective``
    evaluates the SAME selected pairs/weights at backbone logits (zero KL).
    ``objective_gain`` = base_objective - comparable_objective.  This remains a
    within-batch fit diagnostic, never a claim of held-out improvement.
    """
    cfg = config or FullTaskLossConfigV081()
    if logits.ndim != 2 or min(logits.shape) == 0:
        raise ValueError("logits must have non-empty shape [protein, task GO]")
    for name, value in (("base_logits", base_logits), ("targets", targets), ("positive_mask", positive_mask)):
        if value.shape != logits.shape:
            raise ValueError(f"{name} must match logits")
    for name, value in (("structural_support", structural_support), ("graph_logits", graph_logits), ("pu_exclusion_mask", pu_exclusion_mask)):
        if value is not None and value.shape != logits.shape:
            raise ValueError(f"{name} must match logits")
    if is_weak.shape != (logits.shape[0],):
        raise ValueError("is_weak must have shape [protein]")
    if cfg.graph_weight and graph_logits is None:
        raise ValueError("graph_logits are required when graph_weight is nonzero")

    z, base = logits.float(), base_logits.detach().float()
    positive, weak = positive_mask.detach().bool(), is_weak.detach().bool()
    y = torch.where(positive, targets.detach().float(), torch.zeros_like(z))
    unknown = ~positive
    if pu_exclusion_mask is not None:
        unknown = unknown & ~pu_exclusion_mask.detach().bool()

    with torch.no_grad():
        support = torch.zeros_like(z) if structural_support is None else structural_support.detach().float()
        if not torch.isfinite(support).all() or ((support < 0) | (support > 1)).any():
            raise ValueError("structural_support must be finite and in [0, 1]")
        if not torch.isfinite(y).all() or ((y < 0) | (y > 1)).any():
            raise ValueError("positive targets must be finite and in [0, 1]")
        attenuation = 1 - (1 - cfg.support_pu_floor) * support
        base_p = base.sigmoid()
        current_p = z.detach().sigmoid()
        mining = torch.maximum(current_p, base_p)
        hard = _topk_mask(mining, unknown, cfg.hard_pu_k)
        background = _topk_mask(torch.rand_like(mining), unknown & ~hard, cfg.background_pu_k)
        remaining = unknown & ~(hard | background)
        medium = remaining & (mining > cfg.negative_clip)
        high_base = remaining & (base_p >= cfg.anchor_high_base_threshold)
        positive_drift = remaining & ~high_base & ((current_p - base_p) >= cfg.anchor_drift_threshold)
        anchor_background = remaining & ~(high_base | positive_drift)
        strata = (high_base, positive_drift, anchor_background)

        rank_hard = _topk_mask(mining, unknown, cfg.ranking_hard_k)
        rank_random = _topk_mask(torch.rand_like(mining), unknown & ~rank_hard, cfg.ranking_random_k)
        ranking_mask = rank_hard | rank_random
        rank_k = min(cfg.ranking_hard_k + cfg.ranking_random_k, z.shape[1])
        if rank_k:
            neg_valid, neg_index = ranking_mask.float().topk(rank_k, dim=1, sorted=False)
            neg_valid = neg_valid.bool()
            neg_attenuation = attenuation.gather(1, neg_index) * neg_valid
        else:
            neg_index = torch.empty((z.shape[0], 0), device=z.device, dtype=torch.long)
            neg_valid = torch.empty_like(neg_index, dtype=torch.bool)
            neg_attenuation = torch.empty_like(neg_index, dtype=z.dtype)
        positive_row, positive_column = positive.nonzero(as_tuple=True)
        confidence = positive.float() * y.clamp_min(1e-8)
        confidence_sum = confidence.sum(1).clamp_min(1e-8)
        weak_count, core_count = weak.sum().float(), (~weak).sum().float()
        role_total = (cfg.weak_weight * (weak_count > 0) + cfg.core_weight * (core_count > 0)).clamp_min(1e-8)
        row_weight = torch.where(weak, cfg.weak_weight / weak_count.clamp_min(1), cfg.core_weight / core_count.clamp_min(1)) / role_total

    def reduce_rows(values: Tensor) -> Tensor:
        return (values * row_weight).sum()

    def ranking_rows(output: Tensor) -> Tensor:
        # O(number_of_positive_pairs * bounded_negative_budget), not [B,G,R].
        # Empty rows and all-positive rows keep a differentiable finite zero.
        positive_logits = output[positive_row, positive_column]
        negatives = output.gather(1, neg_index)[positive_row]
        penalties = F.softplus(cfg.ranking_margin + negatives - positive_logits[:, None])
        penalties = (penalties * neg_attenuation[positive_row]).sum(1)
        penalties = penalties / neg_valid.sum(1).clamp_min(1)[positive_row]
        penalties = penalties * confidence[positive_row, positive_column]
        row_sum = output.sum(1) * 0
        row_sum = row_sum.index_add(0, positive_row, penalties)
        return row_sum / confidence_sum

    def branches(output: Tensor) -> Dict[str, Tensor]:
        bce = F.binary_cross_entropy_with_logits(output, y, reduction="none")
        negative = _negative_asl(output, cfg)
        return {
            "positive": reduce_rows((bce * confidence).sum(1) / confidence_sum),
            "hard_pu": reduce_rows(_count_normalised(negative, hard, attenuation)),
            "background_pu": reduce_rows(_count_normalised(negative, background, attenuation)),
            "medium_pu": reduce_rows(_count_normalised(negative, medium, attenuation)),
            "ranking": reduce_rows(ranking_rows(output)) if cfg.ranking_weight else output.sum() * 0,
        }

    current = branches(z)
    anchor_kl = (
        base_p * (F.logsigmoid(base) - F.logsigmoid(z))
        + (1 - base_p) * (F.logsigmoid(-base) - F.logsigmoid(-z))
    ).clamp_min(0)
    stratum_rows = torch.stack([_count_normalised(anchor_kl, mask, attenuation) for mask in strata], dim=1)
    stratum_active = torch.stack([mask.any(1) for mask in strata], dim=1)
    current["anchor"] = reduce_rows(stratum_rows.sum(1) / stratum_active.sum(1).clamp_min(1))
    for index, name in enumerate(("anchor_high_base", "anchor_positive_drift", "anchor_background")):
        current[name] = reduce_rows(stratum_rows[:, index])
    current["graph_aux"] = reduce_rows(ranking_rows(graph_logits.float())) if cfg.graph_weight else z.sum() * 0

    weights = {
        "positive": cfg.positive_weight, "hard_pu": cfg.hard_pu_weight,
        "background_pu": cfg.background_pu_weight, "medium_pu": cfg.medium_pu_weight,
        "anchor": cfg.anchor_weight, "ranking": cfg.ranking_weight,
    }
    comparable = sum(weights[name] * current[name] for name in weights)
    loss = comparable + cfg.graph_weight * current["graph_aux"]

    with torch.no_grad():
        baseline = branches(base)
        base_objective = sum(weights[name] * baseline[name] for name in weights if name != "anchor")
        positive_count, hard_count = positive.sum().float(), hard.sum().float()
        probability_delta = current_p - base_p
        target_entropy = -(y * y.clamp_min(1e-8).log() + (1 - y) * (1 - y).clamp_min(1e-8).log())
        positive_entropy = reduce_rows((target_entropy * confidence).sum(1) / confidence_sum)
        diagnostics = {name: value.detach() for name, value in current.items()}
        diagnostics.update({f"contrib_{name}": (weight * current[name]).detach() for name, weight in weights.items()})
        diagnostics["contrib_graph_aux"] = (cfg.graph_weight * current["graph_aux"]).detach()
        diagnostics.update({
            "loss": loss.detach(),
            "comparable_objective": comparable.detach(),
            "base_objective": base_objective,
            "objective_gain": base_objective - comparable.detach(),
            "supervised_gain": base_objective - comparable.detach(),
            "objective_gain_excludes_graph_aux": z.new_tensor(1.0),
            "positive_pairs_per_protein": positive.float().sum(1).mean(),
            "hard_pu_pairs_per_protein": hard.float().sum(1).mean(),
            "background_pu_pairs_per_protein": background.float().sum(1).mean(),
            "medium_pu_pairs_per_protein": medium.float().sum(1).mean(),
            "ranking_pu_pairs_per_protein": ranking_mask.float().sum(1).mean(),
            "supported_unknown_pairs_per_protein": (unknown & (support > 0)).float().sum(1).mean(),
            "pu_excluded_pairs_per_protein": ((~positive) & ~unknown).float().sum(1).mean(),
            "hard_pu_mean_attenuation": (attenuation * hard).sum() / hard_count.clamp_min(1),
            "positive_coverage": z.new_tensor(1.0),
            "hard_pu_active_fraction": ((current_p > cfg.negative_clip) & hard).sum().float() / hard_count.clamp_min(1),
            "supervised_pair_fraction": (positive | hard | background | medium | ranking_mask).float().mean(),
            "positive_delta_probability": (probability_delta * positive).sum() / positive_count.clamp_min(1),
            "hard_pu_delta_probability": (probability_delta * hard).sum() / hard_count.clamp_min(1),
            "positive_target_entropy": positive_entropy,
            "positive_target_kl": (current["positive"] - positive_entropy).clamp_min(0),
        })
    return loss, diagnostics
