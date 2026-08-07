from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import NBSMatchOutput


Reduction = Literal["weighted_mean", "batch_mean"]


@dataclass
class NBSASLConfig:
    """ASL parameters for one supervision source.

    ``weighted_mean`` is the NBS default because retrieval batches have shape
    ``[GO query, candidate protein]`` and a variable number of supervised
    positions.  ``batch_mean`` is retained for controlled comparisons with the
    first LATENCE stage and divides the weighted loss sum by the number of query
    rows instead of the number of supervised pairs.
    """

    gamma_neg: float = 4.0
    gamma_pos: float = 0.0
    clip: float = 0.05
    reduction: Reduction = "weighted_mean"

    def validate(self) -> None:
        if self.gamma_neg < 0 or self.gamma_pos < 0:
            raise ValueError("ASL gamma values cannot be negative")
        if self.clip < 0:
            raise ValueError("ASL clip cannot be negative")
        if self.reduction not in {"weighted_mean", "batch_mean"}:
            raise ValueError("ASL reduction must be 'weighted_mean' or 'batch_mean'")


@dataclass
class NBSLossWeights:
    gold: float = 1.0
    pseudo: float = 0.2
    base_anchor: float = 0.1
    hierarchy: float = 5e-4
    delta_l2: float = 1e-4
    routing_balance: float = 1e-3
    null_collapse: float = 1e-3
    # Constructor-level compatibility with v0.4.x tests/config helpers.
    anchor: Optional[float] = None

    def __post_init__(self) -> None:
        if self.anchor is not None:
            self.base_anchor = float(self.anchor)


def _masked_reduce(
    loss: Tensor,
    mask: Tensor,
    weight: Optional[Tensor],
    *,
    reduction: Reduction = "weighted_mean",
) -> Tensor:
    effective = mask.to(loss.dtype)
    if weight is not None:
        if weight.shape != loss.shape:
            raise ValueError("weight must align with loss")
        effective = effective * weight.to(loss.device, loss.dtype)
    weighted = loss * effective
    if reduction == "weighted_mean":
        denom = effective.sum().clamp_min(1.0)
    elif reduction == "batch_mean":
        # NBS logits are [GO query, candidate protein].  This is intentionally
        # query-row mean, not the first-stage protein-batch mean.
        denom = loss.new_tensor(max(int(loss.shape[0]), 1), dtype=loss.dtype)
    else:  # pragma: no cover - guarded by configuration validation
        raise ValueError("unsupported reduction")
    return weighted.sum() / denom


def masked_bce_logits(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    *,
    confidence: Optional[Tensor] = None,
    pos_weight: Optional[Tensor] = None,
    reduction: Reduction = "weighted_mean",
) -> Tensor:
    raw = F.binary_cross_entropy_with_logits(
        logits,
        labels.to(logits.dtype),
        pos_weight=pos_weight,
        reduction="none",
    )
    return _masked_reduce(raw, mask, confidence, reduction=reduction)


def masked_asl_logits(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    *,
    confidence: Optional[Tensor] = None,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    eps: float = 1e-8,
    reduction: Reduction = "weighted_mean",
) -> Tensor:
    """Masked ASL supporting hard gold and soft pseudo targets.

    Pseudo membership is defined upstream by first-stage ``modelout > 0.5``;
    the stored modelout probability itself remains the soft target.  Unknown
    positions never enter this objective because they are excluded by ``mask``.
    """
    labels = labels.to(logits.dtype)
    xs_pos = torch.sigmoid(logits)
    xs_neg = 1.0 - xs_pos
    if clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)
    loss = -labels * torch.log(xs_pos.clamp_min(eps))
    loss = loss - (1.0 - labels) * torch.log(xs_neg.clamp_min(eps))
    if gamma_neg > 0 or gamma_pos > 0:
        with torch.no_grad():
            pt = xs_pos * labels + xs_neg * (1.0 - labels)
            gamma = gamma_pos * labels + gamma_neg * (1.0 - labels)
            focal = (1.0 - pt).pow(gamma)
        loss = loss * focal
    return _masked_reduce(loss, mask, confidence, reduction=reduction)


def sigmoid_anchor_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    mask: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Residual-preservation regularizer against the first-stage student base.

    This is deliberately *not* expert knowledge distillation.  It only anchors
    NBS to the first-stage backbone logits while the graph branch learns a
    bounded correction.
    """
    t = max(float(temperature), 1e-6)
    with torch.no_grad():
        target = torch.sigmoid(base_logits.to(final_logits.dtype) / t)
    raw = (
        F.binary_cross_entropy_with_logits(final_logits / t, target, reduction="none")
        * (t * t)
    )
    return _masked_reduce(raw, mask, None, reduction="weighted_mean")


def hierarchy_violation_loss(
    probabilities: Tensor,
    child_parent_index: Tensor,
    *,
    margin: float = 0.0,
    weight: Optional[Tensor] = None,
    go_axis: Literal[0, 1] = 1,
) -> Tensor:
    """Penalize ``P(child) > P(parent)`` along a selected GO axis."""
    if probabilities.dim() != 2:
        raise ValueError("probabilities must be two-dimensional")
    if child_parent_index.dim() != 2 or child_parent_index.size(0) != 2:
        raise ValueError("child_parent_index must be [2, E]")
    child, parent = child_parent_index
    axis_size = probabilities.size(go_axis)
    if child.numel() and (
        int(child.min()) < 0
        or int(parent.min()) < 0
        or int(child.max()) >= axis_size
        or int(parent.max()) >= axis_size
    ):
        raise IndexError("child/parent indices exceed the selected GO axis")
    if go_axis == 1:
        violation = F.relu(probabilities[:, child] - probabilities[:, parent] + margin)
        weight_shape = (1, -1)
    elif go_axis == 0:
        violation = F.relu(probabilities[child, :] - probabilities[parent, :] + margin)
        weight_shape = (-1, 1)
    else:  # pragma: no cover
        raise ValueError("go_axis must be 0 or 1")
    if weight is not None:
        if weight.numel() != child.numel():
            raise ValueError("hierarchy weight must align with hierarchy edges")
        violation = violation * weight.reshape(*weight_shape).to(violation)
    return violation.mean() if violation.numel() else probabilities.new_tensor(0.0)


def query_hierarchy_violation_loss(
    probabilities: Tensor,
    child_parent_query_index: Tensor,
    *,
    margin: float = 0.0,
    weight: Optional[Tensor] = None,
) -> Tensor:
    return hierarchy_violation_loss(
        probabilities,
        child_parent_query_index,
        margin=margin,
        weight=weight,
        go_axis=0,
    )


def routing_balance_loss(source_weights: Tensor) -> Tensor:
    mean_load = source_weights.mean(dim=0)
    target = torch.full_like(mean_load, 1.0 / max(mean_load.numel(), 1))
    return F.kl_div(mean_load.clamp_min(1e-8).log(), target, reduction="sum")


def _combine_supervision_weights(
    base: Optional[Tensor],
    extra: Optional[Tensor],
    shape: torch.Size,
) -> Optional[Tensor]:
    if base is None and extra is None:
        return None
    result: Optional[Tensor] = None
    for value in (base, extra):
        if value is None:
            continue
        if value.shape != shape:
            raise ValueError("supervision/confidence weights must align with logits")
        result = value if result is None else result * value
    return result


def nbs_training_loss(
    output: NBSMatchOutput,
    *,
    primary: str = "asl",
    weights: Optional[NBSLossWeights] = None,
    gold_asl: Optional[NBSASLConfig] = None,
    pseudo_asl: Optional[NBSASLConfig] = None,
    pos_weight: Optional[Tensor] = None,
    # Legacy v0.4.x compatibility.  When supplied, these become defaults for
    # both source-specific ASL configs unless explicit configs are provided.
    asl_gamma_neg: Optional[float] = None,
    asl_gamma_pos: Optional[float] = None,
    asl_clip: Optional[float] = None,
    anchor_temperature: float = 1.0,
    hierarchy_probabilities: Optional[Tensor] = None,
    hierarchy_edges: Optional[Tensor] = None,
    hierarchy_edge_weight: Optional[Tensor] = None,
    hierarchy_go_axis: Literal[0, 1] = 0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Open-world NBS objective with source-specific gold/pseudo ASL.

    * Gold/core labels originate from ``train.prop_annotations``.
    * Weak pseudo membership originates from first-stage ``modelout > 0.5``.
    * Weak pseudo labels are the corresponding modelout probabilities.
    * No expert probability is consumed by the NBS forward pass or by a KD
      objective; expert information enters only through the frozen pseudo data.
    """
    if output.labels is None:
        raise ValueError("labels are required for training")
    mask = output.mask
    if mask is None:
        raise ValueError("mask is required: unknown GO labels must not become negatives")
    mask = mask.bool()
    pseudo = (
        torch.zeros_like(mask)
        if output.pseudo_mask is None
        else output.pseudo_mask.bool() & mask
    )
    gold = mask & ~pseudo
    weights = weights or NBSLossWeights()

    legacy_gold = NBSASLConfig(
        gamma_neg=4.0 if asl_gamma_neg is None else float(asl_gamma_neg),
        gamma_pos=0.0 if asl_gamma_pos is None else float(asl_gamma_pos),
        clip=0.05 if asl_clip is None else float(asl_clip),
    )
    gold_asl = gold_asl or legacy_gold
    pseudo_asl = pseudo_asl or NBSASLConfig(
        gamma_neg=legacy_gold.gamma_neg,
        gamma_pos=legacy_gold.gamma_pos,
        clip=legacy_gold.clip,
        reduction=legacy_gold.reduction,
    )
    gold_asl.validate()
    pseudo_asl.validate()

    zero = output.logits.new_tensor(0.0)
    gold_weight = _combine_supervision_weights(
        output.supervision_weight, None, output.logits.shape
    )
    pseudo_weight = _combine_supervision_weights(
        output.supervision_weight, output.confidence, output.logits.shape
    )

    if primary == "asl":
        gold_loss = (
            masked_asl_logits(
                output.logits,
                output.labels,
                gold,
                confidence=gold_weight,
                gamma_neg=gold_asl.gamma_neg,
                gamma_pos=gold_asl.gamma_pos,
                clip=gold_asl.clip,
                reduction=gold_asl.reduction,
            )
            if gold.any()
            else zero
        )
        pseudo_loss = (
            masked_asl_logits(
                output.logits,
                output.labels,
                pseudo,
                confidence=pseudo_weight,
                gamma_neg=pseudo_asl.gamma_neg,
                gamma_pos=pseudo_asl.gamma_pos,
                clip=pseudo_asl.clip,
                reduction=pseudo_asl.reduction,
            )
            if pseudo.any()
            else zero
        )
    elif primary == "bce":
        gold_loss = (
            masked_bce_logits(
                output.logits,
                output.labels,
                gold,
                confidence=gold_weight,
                pos_weight=pos_weight,
            )
            if gold.any()
            else zero
        )
        pseudo_loss = (
            masked_bce_logits(
                output.logits,
                output.labels,
                pseudo,
                confidence=pseudo_weight,
                pos_weight=pos_weight,
            )
            if pseudo.any()
            else zero
        )
    else:
        raise ValueError("primary must be 'asl' or 'bce'")

    aux = output.auxiliary or {}
    base_anchor_loss = zero
    if "base_logits" in aux and weights.base_anchor != 0:
        base_anchor_loss = sigmoid_anchor_loss(
            output.logits, aux["base_logits"], mask, temperature=anchor_temperature
        )
    delta = aux.get("applied_graph_delta", zero)
    delta_l2 = delta.square().mean() if isinstance(delta, Tensor) else zero
    balance = (
        routing_balance_loss(aux["source_weights"])
        if "source_weights" in aux
        else zero
    )
    null_collapse = (
        F.relu(aux["null_weight"].mean() - 0.95)
        if "null_weight" in aux
        else zero
    )
    hierarchy_loss = zero
    if hierarchy_probabilities is not None or hierarchy_edges is not None:
        if hierarchy_probabilities is None or hierarchy_edges is None:
            raise ValueError(
                "hierarchy_probabilities and hierarchy_edges must be supplied together"
            )
        hierarchy_loss = hierarchy_violation_loss(
            hierarchy_probabilities,
            hierarchy_edges,
            weight=hierarchy_edge_weight,
            go_axis=hierarchy_go_axis,
        )

    contributions = {
        "contrib_gold_asl": weights.gold * gold_loss,
        "contrib_pseudo_asl": weights.pseudo * pseudo_loss,
        "contrib_base_anchor": weights.base_anchor * base_anchor_loss,
        "contrib_hierarchy": weights.hierarchy * hierarchy_loss,
        "contrib_delta_l2": weights.delta_l2 * delta_l2,
        "contrib_routing_balance": weights.routing_balance * balance,
        "contrib_null_collapse": weights.null_collapse * null_collapse,
    }
    total = sum(contributions.values(), start=zero)
    parts = {
        "total": total.detach(),
        "gold_asl": gold_loss.detach(),
        "pseudo_asl": pseudo_loss.detach(),
        "base_anchor": base_anchor_loss.detach(),
        "hierarchy": hierarchy_loss.detach(),
        "delta_l2": delta_l2.detach(),
        "routing_balance": balance.detach(),
        "null_collapse": null_collapse.detach(),
        **{name: value.detach() for name, value in contributions.items()},
        "gold_supervised_pairs": gold.sum().to(output.logits.dtype).detach(),
        "pseudo_supervised_pairs": pseudo.sum().to(output.logits.dtype).detach(),
    }
    return total, parts


def masked_bce_with_logits(output: NBSMatchOutput, **kwargs) -> Tensor:
    if output.labels is None or output.mask is None:
        raise ValueError("labels and mask are required")
    weight = _combine_supervision_weights(
        output.supervision_weight, output.confidence, output.logits.shape
    )
    return masked_bce_logits(
        output.logits,
        output.labels,
        output.mask.bool(),
        confidence=weight,
        pos_weight=kwargs.get("pos_weight"),
    )
