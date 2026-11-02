"""Weak-dominant hard membership plus full-column modelout distillation.

Hard targets are core gold or thresholded backbone labels. Their role means
are mixed by core_weight/weak_weight. Dense modelout probabilities supervise
ALL seen GO columns of weak queries in an independently normalized KL term,
including entries absent from the backbone hard-positive set. They never
enter model.forward. No rare-retention or second-head objective is present.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F

from .full_task_evidence_v081 import alias_pu_exclusions
from .full_task_loss_v090 import validate_seen_task_mask


@dataclass(frozen=True)
class FullTaskLossConfigV094:
    core_weight: float = 0.25
    weak_weight: float = 1.0
    modelout_soft_weight: float = 1.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.core_weight + self.weak_weight <= 0:
            raise ValueError('at least one hard-supervision role must have positive weight')


def _role_counts(is_weak, *, global_role_counts=None, world_size=1):
    if is_weak.ndim != 1 or is_weak.dtype != torch.bool:
        raise ValueError('is_weak must be a boolean protein vector')
    if isinstance(world_size, bool) or not isinstance(world_size, Integral) or world_size < 1:
        raise ValueError('world_size must be a positive integer')
    local = (int(is_weak.sum()), int((~is_weak).sum()))
    if global_role_counts is None:
        if world_size != 1:
            raise ValueError('distributed loss requires global_role_counts')
        totals = local
    else:
        if (not isinstance(global_role_counts, (tuple, list)) or len(global_role_counts) != 2
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0
                       for n in global_role_counts)):
            raise ValueError('global_role_counts must contain two nonnegative integers')
        totals = tuple(map(int, global_role_counts))
        if any(a > b for a, b in zip(local, totals)):
            raise ValueError('local role counts exceed global role counts')
    return local, totals


def role_row_weights(is_weak, config, *, global_role_counts=None, world_size=1):
    """Microbatch sums and the DDP rank mean equal the complete role mean."""
    _, totals = _role_counts(is_weak, global_role_counts=global_role_counts,
                             world_size=world_size)
    active = config.weak_weight * (totals[0] > 0) + config.core_weight * (totals[1] > 0)
    if active <= 0:
        raise ValueError('present roles have zero hard-supervision weight')
    return torch.where(is_weak, config.weak_weight / max(1, totals[0]),
                       config.core_weight / max(1, totals[1])).float() * (int(world_size) / active)


def full_task_loss_v094(logits: Tensor, targets: Tensor, is_weak: Tensor,
                       config: FullTaskLossConfigV094 | None = None, *,
                       weak_modelout_prob: Tensor | None = None,
                       seen_task_mask=None, task_to_ontology=None,
                       pu_exclusion_mask=None, global_role_counts=None, world_size=1):
    cfg = config or FullTaskLossConfigV094()
    if logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError('logits must have nonempty floating protein-by-GO shape')
    if targets.shape != logits.shape or is_weak.shape != (len(logits),):
        raise ValueError('targets/is_weak must match protein-by-GO logits')
    if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError('v094 hard targets must be binary memberships')
    seen = validate_seen_task_mask(seen_task_mask, task_to_ontology,
                                  num_task_go=logits.shape[1], device=logits.device)
    if targets[:, ~seen].bool().any():
        raise ValueError('unseen positive targets remain; filter the data view explicitly')
    if not torch.isfinite(logits[:, seen]).all():
        raise ValueError('supervised logits must be finite')
    local, totals = _role_counts(is_weak, global_role_counts=global_role_counts,
                                 world_size=world_size)
    weights = role_row_weights(is_weak, cfg, global_role_counts=global_role_counts,
                               world_size=world_size).to(logits.device)
    positive = targets.bool()
    if pu_exclusion_mask is None:
        excluded = (torch.zeros_like(positive) if task_to_ontology is None else
                    alias_pu_exclusions(positive, torch.as_tensor(task_to_ontology, device=logits.device)))
    else:
        if pu_exclusion_mask.shape != logits.shape or pu_exclusion_mask.dtype != torch.bool:
            raise ValueError('pu_exclusion_mask must be a matching boolean matrix')
        excluded = pu_exclusion_mask & ~positive
    eligible = seen[None, :] & ~excluded
    denominator = seen.sum().to(torch.float32)
    # Mask before arithmetic: 0 * NaN does not isolate an unseen task column.
    z = torch.where(eligible, logits.float(), 0.)
    hard = F.binary_cross_entropy_with_logits(z, targets.detach().float(), reduction='none')
    positive_rows = (hard * positive * eligible).sum(1) / denominator
    negative_rows = (hard * ~positive * eligible).sum(1) / denominator
    main = ((positive_rows + negative_rows) * weights).sum()

    # A zero with no dependency on possibly nonfinite unseen logits.
    soft = main * 0.
    soft_pairs = logits.new_tensor(0.)
    if cfg.modelout_soft_weight and local[0] > 0:
        if weak_modelout_prob is None:
            raise ValueError('modelout_soft_weight requires weak_modelout_prob')
        if (not torch.is_tensor(weak_modelout_prob) or weak_modelout_prob.shape != logits.shape
                or not weak_modelout_prob.is_floating_point()):
            raise ValueError('weak_modelout_prob must be floating and match the immutable task axis')
        if local[0] > 0:
            teacher = weak_modelout_prob.detach()[is_weak].float().to(logits.device)
            selected_teacher = teacher[:, seen]
            if (not torch.isfinite(selected_teacher).all() or (selected_teacher < 0).any()
                    or (selected_teacher > 1).any()):
                raise ValueError('seen weak modelout probabilities must be finite within [0,1]')
            teacher = torch.where(seen[None, :], teacher, 0.)
            student = torch.where(seen[None, :], logits[is_weak].float(), 0.)
            # KL(M || q) = BCE(q,M) - H(M), stable at M=0 and M=1.
            # Hard alias-PU exclusions deliberately do not mask this target:
            # M is an independent dense teacher, including B-negative cells.
            entropy = -torch.xlogy(teacher, teacher) - torch.xlogy(1. - teacher, 1. - teacher)
            divergence = F.binary_cross_entropy_with_logits(student, teacher, reduction='none') - entropy
            soft = ((divergence * seen[None, :]).sum() / denominator
                    * (int(world_size) / totals[0]))
            soft_pairs = logits.new_tensor(local[0]) * denominator
    loss = main + cfg.modelout_soft_weight * soft
    with torch.no_grad():
        parts = {'loss': loss.detach(), 'main_binary': main.detach(),
                 'modelout_soft_KL': soft.detach(),
                 'contrib_modelout_soft': (cfg.modelout_soft_weight * soft).detach(),
                 'positive': (positive_rows * weights).sum().detach(),
                 'negative': (negative_rows * weights).sum().detach(),
                 'contrib_positive': (positive_rows * weights).sum().detach(),
                 'contrib_negative': (negative_rows * weights).sum().detach(),
                 'full_go_denominator': denominator.detach(),
                 'seen_go_columns': seen.sum().to(logits.dtype),
                 'unseen_go_columns': (~seen).sum().to(logits.dtype),
                 'weak_modelout_supervised_pairs': soft_pairs,
                 'positive_pairs_per_protein': (positive & eligible).sum(1).float().mean(),
                 'negative_pairs_per_protein': (~positive & eligible).sum(1).float().mean(),
                 'alias_excluded_pairs_per_protein': (excluded & seen).sum(1).float().mean(),
                 'mean_probability': logits[:, seen].sigmoid().mean()}
        for role, rows in (('core', ~is_weak), ('weak', is_weak)):
            parts[f'{role}_proteins'] = rows.sum().to(logits.dtype)
            parts[f'contrib_{role}_positive'] = (positive_rows * weights * rows).sum().detach()
            parts[f'contrib_{role}_negative'] = (negative_rows * weights * rows).sum().detach()
    return loss, parts


def training_objective(model, batch, loss_config, *, global_role_counts, world_size=1):
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    supplied = batch.get('seen_task_mask')
    if supplied is None or supplied.dtype != torch.bool or not torch.equal(supplied, raw.seen_task_mask):
        raise ValueError('batch seen_task_mask differs from model supervision mask')
    if not torch.equal(batch['targets'], batch['positive_mask'].float()):
        raise ValueError('v094 hard targets must be explicit binary memberships')
    output = model(batch, return_details=True)
    loss, parts = full_task_loss_v094(output['logits'], batch['targets'], batch['is_weak'],
        loss_config, weak_modelout_prob=batch.get('weak_modelout_prob'),
        seen_task_mask=raw.seen_task_mask, task_to_ontology=raw.task_to_ontology,
        global_role_counts=global_role_counts, world_size=world_size)
    return output['logits'], loss, parts
