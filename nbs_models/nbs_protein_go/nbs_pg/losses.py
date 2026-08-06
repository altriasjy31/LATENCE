from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import NBSMatchOutput


@dataclass
class NBSLossWeights:
    pseudo: float = 0.5
    anchor: float = 0.1
    hierarchy: float = 0.0
    delta_l2: float = 1e-4
    routing_balance: float = 1e-3
    null_collapse: float = 1e-3


def _masked_reduce(loss: Tensor, mask: Tensor, weight: Optional[Tensor]) -> Tensor:
    effective = mask.to(loss.dtype)
    if weight is not None:
        if weight.shape != loss.shape:
            raise ValueError("weight must align with loss")
        effective = effective * weight.to(loss.device, loss.dtype)
    denom = effective.sum().clamp_min(1.0)
    return (loss * effective).sum() / denom


def masked_bce_logits(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    *,
    confidence: Optional[Tensor] = None,
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    raw = F.binary_cross_entropy_with_logits(
        logits,
        labels.to(logits.dtype),
        pos_weight=pos_weight,
        reduction="none",
    )
    return _masked_reduce(raw, mask, confidence)


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
) -> Tensor:
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
    return _masked_reduce(loss, mask, confidence)


def sigmoid_anchor_loss(
    final_logits: Tensor,
    base_logits: Tensor,
    mask: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    t = max(float(temperature), 1e-6)
    with torch.no_grad():
        target = torch.sigmoid(base_logits.to(final_logits.dtype) / t)
    raw = (
        F.binary_cross_entropy_with_logits(final_logits / t, target, reduction="none")
        * (t * t)
    )
    return _masked_reduce(raw, mask, None)


def hierarchy_violation_loss(
    probabilities: Tensor,
    child_parent_index: Tensor,
    *,
    margin: float = 0.0,
    weight: Optional[Tensor] = None,
    go_axis: Literal[0, 1] = 1,
) -> Tensor:
    """Penalize ``P(child) > P(parent)`` along a selected GO axis.

    ``go_axis=1`` preserves the conventional ``[protein, GO]`` layout.  NBS
    retrieval outputs are ``[GO query, candidate protein]`` and therefore use
    ``go_axis=0`` with query-local child/parent row indices.
    """
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
    else:  # pragma: no cover - Literal protects typed callers
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
    """NBS hierarchy loss for ``[GO query, candidate protein]`` scores."""
    return hierarchy_violation_loss(
        probabilities,
        child_parent_query_index,
        margin=margin,
        weight=weight,
        go_axis=0,
    )


def routing_balance_loss(source_weights: Tensor) -> Tensor:
    """Batch-level load balancing without forcing each query to high entropy."""
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
    pos_weight: Optional[Tensor] = None,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 0.0,
    asl_clip: float = 0.05,
    anchor_temperature: float = 1.0,
    hierarchy_probabilities: Optional[Tensor] = None,
    hierarchy_edges: Optional[Tensor] = None,
    hierarchy_edge_weight: Optional[Tensor] = None,
    hierarchy_go_axis: Literal[0, 1] = 0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Open-world NBS objective with gold/pseudo supervision separation.

    Unknown positions are omitted through ``mask``.  ``supervision_weight`` can
    down-weight sampled-unlabelled negatives for both gold and pseudo branches;
    pseudo confidence is multiplied only on pseudo-supervised positions.
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
    true = mask & ~pseudo
    weights = weights or NBSLossWeights()

    loss_fn = masked_asl_logits if primary == "asl" else masked_bce_logits
    kwargs = {}
    if primary == "asl":
        kwargs.update(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip)
    elif primary == "bce":
        kwargs["pos_weight"] = pos_weight
    else:
        raise ValueError("primary must be 'asl' or 'bce'")

    zero = output.logits.new_tensor(0.0)
    true_weight = _combine_supervision_weights(
        output.supervision_weight, None, output.logits.shape
    )
    pseudo_weight = _combine_supervision_weights(
        output.supervision_weight, output.confidence, output.logits.shape
    )
    true_loss = (
        loss_fn(output.logits, output.labels, true, confidence=true_weight, **kwargs)
        if true.any()
        else zero
    )
    pseudo_loss = (
        loss_fn(output.logits, output.labels, pseudo, confidence=pseudo_weight, **kwargs)
        if pseudo.any()
        else zero
    )

    aux = output.auxiliary or {}
    anchor_loss = zero
    if "base_logits" in aux and weights.anchor != 0:
        anchor_loss = sigmoid_anchor_loss(
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

    total = (
        true_loss
        + weights.pseudo * pseudo_loss
        + weights.anchor * anchor_loss
        + weights.hierarchy * hierarchy_loss
        + weights.delta_l2 * delta_l2
        + weights.routing_balance * balance
        + weights.null_collapse * null_collapse
    )
    parts = {
        "total": total.detach(),
        "true": true_loss.detach(),
        "pseudo": pseudo_loss.detach(),
        "anchor": anchor_loss.detach(),
        "hierarchy": hierarchy_loss.detach(),
        "delta_l2": delta_l2.detach(),
        "routing_balance": balance.detach(),
        "null_collapse": null_collapse.detach(),
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
