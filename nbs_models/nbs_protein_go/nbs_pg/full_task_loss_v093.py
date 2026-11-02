"""Role-normalized binary supervision with shared-head and rare-retention auxiliaries.

Every supervised term excludes natural-zero and held-out canonical GO groups.
Weak nonmembers retain the existing binary-membership objective; their zero
target is not an assertion of biological absence. Teacher probabilities only
supervise selected weak positives in the explicitly named soft auxiliary.
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
class FullTaskLossConfigV093:
    core_weight: float = 1.0
    weak_weight: float = 0.25
    shared_weight: float = 0.25
    rare_distill_weight: float = 0.1
    weak_soft_weight: float = 0.1

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.core_weight + self.weak_weight <= 0:
            raise ValueError('at least one role must have positive weight')


def role_row_weights(is_weak, cfg, *, global_role_counts=None, world_size=1):
    """Per-row weights whose microbatch sums equal one global role-mean loss."""
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
                or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in global_role_counts)):
            raise ValueError('global_role_counts must contain two nonnegative integers')
        totals = tuple(map(int, global_role_counts))
        if any(a > b for a, b in zip(local, totals)):
            raise ValueError('local role counts exceed global role counts')
    active = cfg.weak_weight * (totals[0] > 0) + cfg.core_weight * (totals[1] > 0)
    if active <= 0:
        raise ValueError('present roles have zero objective weight')
    return torch.where(is_weak, cfg.weak_weight / max(1, totals[0]),
                       cfg.core_weight / max(1, totals[1])).float() * (int(world_size) / active)


def full_task_loss_v093(logits: Tensor, shared_logits: Tensor, base_logits: Tensor,
                       targets: Tensor, is_weak: Tensor,
                       config: FullTaskLossConfigV093 | None = None, *,
                       core_counts: Tensor, weak_soft_targets: Tensor | None = None,
                       seen_task_mask=None, task_to_ontology=None,
                       pu_exclusion_mask=None, global_role_counts=None, world_size=1):
    cfg = config or FullTaskLossConfigV093()
    if logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError('logits must have nonempty floating protein-by-GO shape')
    if any(t.shape != logits.shape for t in (shared_logits, base_logits, targets)):
        raise ValueError('all logit/target matrices must share one protein-by-GO shape')
    if is_weak.shape != (len(logits),):
        raise ValueError('is_weak length differs from proteins')
    if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError('v093 main targets must be binary memberships')
    seen = validate_seen_task_mask(seen_task_mask, task_to_ontology,
                                  num_task_go=logits.shape[1], device=logits.device)
    if targets[:, ~seen].bool().any():
        raise ValueError('unseen positive targets remain; filter the data view explicitly')
    if any(not torch.isfinite(value).all() for value in (logits[:, seen], shared_logits[:, seen], base_logits[:, seen])):
        raise ValueError('supervised logit matrices must be finite')
    counts = torch.as_tensor(core_counts, device=logits.device)
    if counts.shape != seen.shape or counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) or (counts < 0).any():
        raise ValueError('core_counts must be one nonnegative integer per task GO')
    positive = targets.bool()
    if pu_exclusion_mask is None:
        excluded = (torch.zeros_like(positive) if task_to_ontology is None
                    else alias_pu_exclusions(positive, torch.as_tensor(task_to_ontology, device=logits.device)))
    else:
        if pu_exclusion_mask.shape != logits.shape or pu_exclusion_mask.dtype != torch.bool:
            raise ValueError('pu_exclusion_mask must be a matching boolean matrix')
        excluded = pu_exclusion_mask & ~positive
    eligible = seen[None, :] & ~excluded
    weights = role_row_weights(is_weak, cfg, global_role_counts=global_role_counts,
                               world_size=world_size).to(logits.device)
    denominator = seen.sum().to(torch.float32)
    # Avoid copying almost the entire Q*G matrix merely to drop a few zero-GO
    # columns. Multiplication by the explicit mask gives zero unseen gradient.
    # Select before arithmetic: multiplying an already-computed NaN by a zero
    # unseen mask would still poison the objective. Unseen cells have neither
    # a teacher dependency nor a gradient through any auxiliary.
    z = torch.where(eligible, logits.float(), 0.)
    shared_z = torch.where(eligible, shared_logits.float(), 0.)
    base_z = torch.where(eligible, base_logits.detach().float(), 0.)
    hard = F.binary_cross_entropy_with_logits(z, targets.detach().float(), reduction='none')
    shared = F.binary_cross_entropy_with_logits(shared_z, targets.detach().float(), reduction='none')
    positive_rows = (hard * positive * eligible).sum(1) / denominator
    negative_rows = (hard * ~positive * eligible).sum(1) / denominator
    main = ((positive_rows + negative_rows) * weights).sum()
    shared_loss = ((shared * eligible).sum(1) / denominator * weights).sum()
    rare = seen & (counts > 0) & (counts < 1000)
    teacher = base_z.sigmoid()
    # KL(B || final), written as cross-entropy minus detached entropy. This
    # auxiliary is zero at an unchanged backbone, including its gradients.
    rare_mask = eligible & rare[None, :]
    divergence = (F.binary_cross_entropy_with_logits(z, teacher, reduction='none')
                  - F.binary_cross_entropy_with_logits(base_z, teacher, reduction='none'))
    rare_loss = (((divergence * rare_mask).sum(1) / rare.sum().clamp_min(1)) * weights).sum()
    soft_loss = z.sum() * 0
    soft_count = logits.new_tensor(0.)
    if weak_soft_targets is not None:
        soft = weak_soft_targets.detach().float()
        if soft.shape != logits.shape:
            raise ValueError('weak_soft_targets must match logit axes')
        selected = eligible & positive & is_weak[:, None]
        if not torch.isfinite(soft[selected]).all() or ((soft[selected] < 0) | (soft[selected] > 1)).any():
            raise ValueError('weak_soft_targets must be matching finite probabilities')
        soft = torch.where(selected, soft, 0.)
        soft_terms = F.binary_cross_entropy_with_logits(z, soft, reduction='none')
        soft_loss = (((soft_terms * selected).sum(1) / denominator) * weights).sum()
        soft_count = selected.sum().detach().to(logits.dtype)
    elif cfg.weak_soft_weight:
        raise ValueError('weak_soft_weight requires explicit weak_soft_targets in the batch')
    loss = main + cfg.shared_weight * shared_loss + cfg.rare_distill_weight * rare_loss + cfg.weak_soft_weight * soft_loss
    with torch.no_grad():
        parts = {'loss': loss.detach(), 'main_binary': main.detach(),
                 'positive': (positive_rows * weights).sum().detach(),
                 'negative': (negative_rows * weights).sum().detach(),
                 'contrib_positive': (positive_rows * weights).sum().detach(),
                 'contrib_negative': (negative_rows * weights).sum().detach(),
                 'shared_binary': shared_loss.detach(), 'rare_backbone_KL': rare_loss.detach(),
                 'weak_selected_soft': soft_loss.detach(),
                 'contrib_shared': (cfg.shared_weight * shared_loss).detach(),
                 'contrib_rare_backbone': (cfg.rare_distill_weight * rare_loss).detach(),
                 'contrib_weak_soft': (cfg.weak_soft_weight * soft_loss).detach(),
                 'full_go_denominator': denominator.detach(),
                 'seen_go_columns': seen.sum().to(logits.dtype),
                 'unseen_go_columns': (~seen).sum().to(logits.dtype),
                 'positive_pairs_per_protein': (positive & eligible).sum(1).float().mean(),
                 'negative_pairs_per_protein': (~positive & eligible).sum(1).float().mean(),
                 'alias_excluded_pairs_per_protein': (excluded & seen).sum(1).float().mean(),
                 'weak_selected_soft_pairs': soft_count,
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
        raise ValueError('v093 main targets must be explicit binary membership')
    output = model(batch, return_details=True)
    loss, parts = full_task_loss_v093(output['logits'], output['shared_logits'], output['base_logits'],
        batch['targets'], batch['is_weak'], loss_config, core_counts=batch['core_counts'],
        weak_soft_targets=batch.get('weak_soft_targets'), seen_task_mask=raw.seen_task_mask,
        task_to_ontology=raw.task_to_ontology, global_role_counts=global_role_counts, world_size=world_size)
    return output['logits'], loss, parts
