"""ASL + positive/unlabelled weak supervision + selected teacher KL + hierarchy.

B-threshold memberships remain the hard positive set. Unselected weak cells
are unknown, not exhaustive negatives. Modelout is supervision only: confident
teacher candidates and bounded student-hard negatives are selected per protein.
Every term has exact macro-batch/DDP row normalization; top-k never crosses rows.
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
from .full_task_loss_v094 import _role_counts, role_row_weights


@dataclass(frozen=True)
class FullTaskLossConfigV095:
    core_weight: float = 0.25
    weak_weight: float = 1.0
    modelout_soft_weight: float = 1.0
    hierarchy_weight: float = 0.01
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.05
    teacher_topk: int = 2048
    teacher_min_probability: float = 0.05
    teacher_confidence_power: float = 0.5
    hard_negative_topk: int = 256
    hard_negative_max_teacher: float = 0.01
    hard_negative_weight: float = 0.25
    loss_scale: float = 1000.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.core_weight + self.weak_weight <= 0 or self.loss_scale <= 0:
            raise ValueError('positive role weight and loss_scale required')
        for name in ('teacher_topk', 'hard_negative_topk'):
            if not isinstance(getattr(self, name), Integral):
                raise ValueError(f'{name} must be a nonnegative integer')
        if not 0 <= self.asl_clip < 1:
            raise ValueError('asl_clip must be in [0,1)')
        if not 0 <= self.hard_negative_max_teacher < self.teacher_min_probability <= 1:
            raise ValueError('teacher thresholds must leave a nonempty unknown region')
        if not 0 <= self.hard_negative_weight <= 1:
            raise ValueError('hard_negative_weight must be in [0,1]')


def build_hierarchy_task_edges(ontology, seen_task_mask):
    """Return canonical [parent, child] task indices, one representative per GO.

    The dataset ontology schema stores is_a/part_of as child -> parent. Inverse
    relations are deliberately not included. Unseen endpoints never supervise.
    """
    mapping = torch.as_tensor(ontology['task_to_ontology']).detach().cpu().long()
    seen = validate_seen_task_mask(torch.as_tensor(seen_task_mask).cpu(), mapping,
                                  num_task_go=len(mapping))
    representative = {}
    for task, canonical in enumerate(mapping.tolist()):
        if bool(seen[task]):
            representative.setdefault(canonical, task)
    pairs = set()
    for name in ('is_a', 'part_of'):
        edges = ontology.get('edges', {}).get(name)
        if edges is None:
            continue
        edges = torch.as_tensor(edges).detach().cpu()
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype not in (torch.int32, torch.int64):
            raise ValueError(f'{name} must have integer shape [2,E]')
        for child, parent in edges.T.tolist():
            if child != parent and child in representative and parent in representative:
                pairs.add((representative[parent], representative[child]))
    return (torch.tensor(sorted(pairs), dtype=torch.long).T.contiguous()
            if pairs else torch.empty((2, 0), dtype=torch.long))


def _topk_mask(candidate, score, k):
    if k == 0:
        return torch.zeros_like(candidate)
    indices = score.detach().masked_fill(~candidate, -torch.inf).topk(min(k, score.shape[1]), dim=1).indices
    return torch.zeros_like(candidate).scatter_(1, indices, candidate.gather(1, indices))


def _asl(z, positive, config):
    q = z.sigmoid()
    q_negative = (1. - q + config.asl_clip).clamp(max=1.)
    # Match Stage1's masked ASL: focal factors are treated as weights.
    positive_factor = (1. - q).pow(config.asl_gamma_pos).detach()
    negative_factor = (1. - q_negative).pow(config.asl_gamma_neg).detach()
    positive_loss = -F.logsigmoid(z) * positive_factor
    negative_loss = -q_negative.clamp_min(1e-8).log() * negative_factor
    return torch.where(positive, positive_loss, negative_loss)


def full_task_loss_v095(logits: Tensor, targets: Tensor, is_weak: Tensor,
                       config: FullTaskLossConfigV095 | None = None, *,
                       weak_modelout_prob=None, seen_task_mask=None,
                       task_to_ontology=None, pu_exclusion_mask=None,
                       hierarchy_edges=None, global_role_counts=None, world_size=1):
    cfg = config or FullTaskLossConfigV095()
    if logits.ndim != 2 or min(logits.shape) < 1 or not logits.is_floating_point():
        raise ValueError('logits must have nonempty floating protein-by-GO shape')
    if targets.shape != logits.shape or is_weak.shape != (len(logits),):
        raise ValueError('targets/is_weak must match protein-by-GO logits')
    if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError('v095 hard targets must be binary memberships')
    seen = validate_seen_task_mask(seen_task_mask, task_to_ontology,
                                  num_task_go=logits.shape[1], device=logits.device)
    if targets[:, ~seen].bool().any():
        raise ValueError('unseen positive targets remain')
    if not torch.isfinite(logits[:, seen]).all():
        raise ValueError('supervised logits must be finite')
    local, totals = _role_counts(is_weak, global_role_counts=global_role_counts, world_size=world_size)
    weights = role_row_weights(is_weak, cfg, global_role_counts=global_role_counts, world_size=world_size).to(logits.device)
    positive = targets.bool()
    if pu_exclusion_mask is None:
        excluded = (torch.zeros_like(positive) if task_to_ontology is None else
                    alias_pu_exclusions(positive, torch.as_tensor(task_to_ontology, device=logits.device)))
    else:
        if pu_exclusion_mask.shape != logits.shape or pu_exclusion_mask.dtype != torch.bool:
            raise ValueError('pu_exclusion_mask must be a matching boolean matrix')
        excluded = pu_exclusion_mask & ~positive
    eligible = seen[None, :] & ~excluded
    denominator = seen.sum().float()
    z = torch.where(eligible, logits.float(), 0.)
    hard_eligible = eligible & ((~is_weak)[:, None] | positive)
    hard = _asl(z, positive, cfg)
    positive_rows = (hard * positive * hard_eligible).sum(1) / denominator
    negative_rows = (hard * ~positive * hard_eligible).sum(1) / denominator
    main = ((positive_rows + negative_rows) * weights).sum()
    soft = main * 0.
    teacher_soft = main * 0.
    negative_soft = main * 0.
    soft_pairs = logits.new_tensor(0.)
    teacher_pairs = logits.new_tensor(0.)
    negative_pairs = logits.new_tensor(0.)
    if cfg.modelout_soft_weight and local[0]:
        if (not torch.is_tensor(weak_modelout_prob) or weak_modelout_prob.shape != logits.shape
                or not weak_modelout_prob.is_floating_point()):
            raise ValueError('weak_modelout_prob must be floating and match logits')
        teacher = weak_modelout_prob.detach()[is_weak].float().to(logits.device)
        if not torch.isfinite(teacher[:, seen]).all() or not ((teacher[:, seen] >= 0) & (teacher[:, seen] <= 1)).all():
            raise ValueError('seen weak modelout probabilities must lie in [0,1]')
        allowed = eligible[is_weak]
        b_positive = positive[is_weak]
        teacher = torch.where(allowed, teacher, 0.)
        student = z[is_weak]
        # B-positive terms with low M remain hard positives, never teacher negatives.
        teacher_candidate = allowed & (teacher >= cfg.teacher_min_probability) & (~b_positive | (teacher >= .5))
        selected_teacher = _topk_mask(teacher_candidate, teacher, cfg.teacher_topk)
        selected_teacher |= allowed & b_positive & (teacher >= .5)
        # Only bounded student-confident, teacher-low nonmembers are negatives.
        negative_candidate = allowed & ~b_positive & (teacher <= cfg.hard_negative_max_teacher)
        selected_negative = _topk_mask(negative_candidate, student, cfg.hard_negative_topk)
        entropy = -torch.xlogy(teacher, teacher) - torch.xlogy(1.-teacher, 1.-teacher)
        divergence = F.binary_cross_entropy_with_logits(student, teacher, reduction='none') - entropy
        factor = int(world_size) / totals[0] / denominator
        teacher_soft = (divergence * selected_teacher * teacher.pow(cfg.teacher_confidence_power)).sum() * factor
        negative_soft = (divergence * selected_negative * (1.-teacher).pow(cfg.teacher_confidence_power)).sum() * factor * cfg.hard_negative_weight
        soft = teacher_soft + negative_soft
        teacher_pairs = selected_teacher.sum().to(logits.dtype)
        negative_pairs = selected_negative.sum().to(logits.dtype)
        soft_pairs = teacher_pairs + negative_pairs
    hierarchy = main * 0.
    if cfg.hierarchy_weight:
        if hierarchy_edges is None:
            raise ValueError('hierarchy_weight requires explicit canonical hierarchy_edges')
        edges = torch.as_tensor(hierarchy_edges, device=logits.device)
        if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError('hierarchy_edges must be long [2,E] with parent then child')
        if edges.numel():
            if (edges < 0).any() or (edges >= logits.shape[1]).any() or not seen[edges].all():
                raise ValueError('hierarchy endpoints must be valid seen task columns')
            parent, child = edges
            valid = eligible[:, parent] & eligible[:, child]
            violation = F.relu(z[:, child].sigmoid() - z[:, parent].sigmoid())
            rows = (violation * valid).sum(1) / edges.shape[1]
            hierarchy = (rows * weights).sum()
    unscaled = main + cfg.modelout_soft_weight * soft + cfg.hierarchy_weight * hierarchy
    loss = cfg.loss_scale * unscaled
    with torch.no_grad():
        scale = cfg.loss_scale
        parts = dict(loss=loss.detach(), unscaled_loss=unscaled.detach(),
            scaled_hard_ASL=(scale*main).detach(), main_binary=main.detach(), hard_ASL=main.detach(),
            modelout_soft_KL=soft.detach(), hierarchy=hierarchy.detach(),
            contrib_modelout_soft=(scale*cfg.modelout_soft_weight*soft).detach(),
            contrib_teacher_positive=(scale*cfg.modelout_soft_weight*teacher_soft).detach(),
            contrib_teacher_negative=(scale*cfg.modelout_soft_weight*negative_soft).detach(),
            contrib_hierarchy=(scale*cfg.hierarchy_weight*hierarchy).detach(),
            positive=(scale*positive_rows*weights).sum().detach(),
            negative=(scale*negative_rows*weights).sum().detach(),
            contrib_positive=(scale*positive_rows*weights).sum().detach(),
            contrib_negative=(scale*negative_rows*weights).sum().detach(),
            full_go_denominator=denominator.detach(), seen_go_columns=seen.sum().to(logits.dtype),
            unseen_go_columns=(~seen).sum().to(logits.dtype),
            weak_modelout_supervised_pairs=soft_pairs, teacher_selected_pairs=teacher_pairs,
            hard_negative_selected_pairs=negative_pairs,
            teacher_selected_pairs_per_weak=teacher_pairs * (int(world_size) / max(1, totals[0])),
            hard_negative_selected_pairs_per_weak=negative_pairs * (int(world_size) / max(1, totals[0])),
            positive_pairs_per_protein=(positive & hard_eligible).sum(1).float().mean(),
            negative_pairs_per_protein=(~positive & hard_eligible).sum(1).float().mean(),
            alias_excluded_pairs_per_protein=(excluded & seen).sum(1).float().mean(),
            mean_probability=logits[:, seen].sigmoid().mean(),
            loss_scale=logits.new_tensor(scale))
        for role, rows in (('core', ~is_weak), ('weak', is_weak)):
            parts[f'{role}_proteins'] = rows.sum().to(logits.dtype)
            parts[f'contrib_{role}_positive'] = (scale*positive_rows*weights*rows).sum().detach()
            parts[f'contrib_{role}_negative'] = (scale*negative_rows*weights*rows).sum().detach()
    return loss, parts


def training_objective(model, batch, loss_config, *, global_role_counts, world_size=1):
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    supplied = batch.get('seen_task_mask')
    if supplied is None or supplied.dtype != torch.bool or not torch.equal(supplied, raw.seen_task_mask):
        raise ValueError('batch seen_task_mask differs from model supervision mask')
    if not torch.equal(batch['targets'], batch['positive_mask'].float()):
        raise ValueError('v095 hard targets must be explicit binary memberships')
    output = model(batch, return_details=True)
    loss, parts = full_task_loss_v095(output['logits'], batch['targets'], batch['is_weak'],
        loss_config, weak_modelout_prob=batch.get('weak_modelout_prob'),
        seen_task_mask=raw.seen_task_mask, task_to_ontology=raw.task_to_ontology,
        hierarchy_edges=getattr(raw, 'hierarchy_task_edges', None),
        global_role_counts=global_role_counts, world_size=world_size)
    return output['logits'], loss, parts
