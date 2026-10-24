"""Seen-column binary BCE for the v0.9.0 shared protein--GO scorer.

Unseen columns have neither positive nor negative supervision and do not enter
the denominator.  The role weighting and exact DDP macro-batch normalization
are inherited from v088.  This is a second-stage isolation contract, not proof
that Stage1 or label-side resources satisfy a strict zero-shot protocol.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .full_task_evidence_v081 import alias_pu_exclusions
from .full_task_loss_v088 import FullTaskLossConfigV088, full_task_loss_v088


@dataclass(frozen=True)
class FullTaskLossConfigV090(FullTaskLossConfigV088):
    pass


def validate_seen_task_mask(seen_task_mask, task_to_ontology=None, *, num_task_go: int,
                            device=None) -> Tensor:
    """Validate a nonempty seen set with whole canonical/alias groups.

    ``task_to_ontology`` contains canonical full-ontology row indices.  A
    partially seen task must supply it; splitting an alias group is rejected.
    """
    if seen_task_mask is None:
        seen = torch.ones(num_task_go, dtype=torch.bool, device=device)
    else:
        seen = torch.as_tensor(seen_task_mask, device=device)
        if seen.dtype != torch.bool or seen.shape != (num_task_go,):
            raise ValueError("seen_task_mask must be a boolean vector over all task columns")
    if not seen.any():
        raise ValueError("at least one seen task GO is required for training")
    if task_to_ontology is None:
        if not seen.all():
            raise ValueError("partial seen isolation requires canonical task_to_ontology mapping")
        return seen
    mapping = torch.as_tensor(task_to_ontology, device=seen.device)
    if (mapping.shape != seen.shape or mapping.dtype not in
            (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or (mapping < 0).any()):
        raise ValueError("task_to_ontology must contain valid integer canonical row indices")
    if not seen.all():
        # The same ontology row may back several immutable classifier columns.
        # A held-out alias must never remain supervised through another column.
        overlap = torch.isin(torch.unique(mapping[seen]), torch.unique(mapping[~seen]))
        if overlap.any():
            raise ValueError("seen/unseen split divides a canonical GO alias group")
    return seen


def full_task_loss_v090(logits: Tensor, targets: Tensor, is_weak: Tensor,
                       config: FullTaskLossConfigV090 | None = None, *,
                       seen_task_mask=None, task_to_ontology=None,
                       pu_exclusion_mask: Tensor | None = None,
                       global_role_counts: tuple[int, int] | None = None,
                       world_size: int = 1):
    """Return seen-only BCE, rejecting residual unseen positive supervision.

    Inputs retain the immutable closed-task column space.  The data adapter
    must remove unseen labels and evidence explicitly before calling this
    function.  In particular, replacing unseen positives by negative targets
    does not train those negatives: all unseen cells are excluded by indexing.
    """
    if logits.ndim != 2 or min(logits.shape) == 0 or not logits.is_floating_point():
        raise ValueError("logits must have nonempty floating shape [protein, task GO]")
    if targets.shape != logits.shape:
        raise ValueError("targets must match logits")
    if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("v090 targets must be finite binary membership")
    seen = validate_seen_task_mask(seen_task_mask, task_to_ontology,
                                  num_task_go=logits.shape[1], device=logits.device)
    if targets[:, ~seen].bool().any():
        raise ValueError("unseen positive supervision remains; filter targets before training")
    if pu_exclusion_mask is not None:
        if pu_exclusion_mask.shape != logits.shape or pu_exclusion_mask.dtype != torch.bool:
            raise ValueError("pu_exclusion_mask must be a matching boolean matrix")
    elif task_to_ontology is not None:
        pu_exclusion_mask = alias_pu_exclusions(targets.bool(),
            torch.as_tensor(task_to_ontology, device=targets.device).long())
    # The main all-seen run must not allocate full B*G copies merely to select
    # every column (6400*21312 is already a large training matrix).
    all_seen = bool(seen.all())
    selected = logits if all_seen else logits[:, seen]
    selected_targets = targets if all_seen else targets[:, seen]
    selected_exclusion = (pu_exclusion_mask if all_seen or pu_exclusion_mask is None
                          else pu_exclusion_mask[:, seen])
    if not torch.isfinite(selected).all():
        raise ValueError("seen logits must be finite")
    loss, parts = full_task_loss_v088(selected, selected_targets, is_weak,
        config or FullTaskLossConfigV090(),
        pu_exclusion_mask=selected_exclusion,
        global_role_counts=global_role_counts, world_size=world_size)
    parts.update(output_go_columns=loss.new_tensor(logits.shape[1]),
                 seen_go_columns=seen.sum().to(loss.dtype),
                 unseen_go_columns=(~seen).sum().to(loss.dtype))
    return loss, parts
