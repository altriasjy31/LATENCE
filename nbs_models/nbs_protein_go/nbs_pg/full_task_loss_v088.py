"""v0.8.8 full-column binary BCE and training-only output priors.

Every retained label uses the same denominator G. Known-positive aliases may
exclude otherwise-negative cells, but never remove positives, add labels, or
change G. This is binary membership supervision (including weak pseudo-label
membership), not a claim that all unannotated GO functions are biologically
absent. No clipping, focal factors, hard mining, ranking, or anchor loss occurs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class FullTaskLossConfigV088:
    core_weight: float = 1.0
    weak_weight: float = 0.25

    def __post_init__(self):
        for name in ("core_weight", "weak_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.core_weight + self.weak_weight <= 0:
            raise ValueError("at least one role must have positive weight")


def full_task_loss_v088(logits: Tensor, targets: Tensor, is_weak: Tensor,
                       config: FullTaskLossConfigV088 | None = None, *,
                       pu_exclusion_mask: Tensor | None = None,
                       global_role_counts: tuple[int, int] | None = None,
                       world_size: int = 1):
    """Return FP32 BCE and detached diagnostics with exact DDP role means.

    global_role_counts is (weak, core) for the complete global MACRO batch,
    including all accumulation microbatches. DDP gradient averaging is offset
    by world_size, so unequal local batches and a rank without one role agree
    with a single global macro batch. Sum microbatch losses without dividing
    again by the accumulation count. Denominator for every protein is logits.G,
    including excluded alias cells. The caller computes alias exclusions using
    alias_pu_exclusions(positive_mask, task_to_ontology), as in earlier versions.
    """
    cfg = config or FullTaskLossConfigV088()
    if isinstance(world_size, bool) or not isinstance(world_size, Integral) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if logits.ndim != 2 or min(logits.shape) == 0 or not logits.is_floating_point():
        raise ValueError("logits must have nonempty floating shape [protein, GO]")
    if targets.shape != logits.shape:
        raise ValueError("targets must match logits")
    if is_weak.shape != (logits.shape[0],) or is_weak.dtype != torch.bool:
        raise ValueError("is_weak must be a boolean protein vector")
    if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("v088 targets must be finite binary membership")
    if pu_exclusion_mask is not None and (pu_exclusion_mask.shape != logits.shape or pu_exclusion_mask.dtype != torch.bool):
        raise ValueError("pu_exclusion_mask must be a matching boolean matrix")
    if global_role_counts is not None and (
            not isinstance(global_role_counts, (tuple, list)) or len(global_role_counts) != 2
            or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in global_role_counts)):
        raise ValueError("global_role_counts must contain two nonnegative integers")
    z, target = logits.float(), targets.detach().float()
    weak, positive = is_weak.detach(), target.bool()
    excluded = torch.zeros_like(positive) if pu_exclusion_mask is None else pu_exclusion_mask.detach() & ~positive
    negative = ~positive & ~excluded
    local_weak, local_core = weak.sum(), (~weak).sum()
    if global_role_counts is None:
        if world_size != 1:
            raise ValueError("DDP requires global_role_counts")
        global_weak, global_core = local_weak, local_core
    else:
        global_weak, global_core = [z.new_tensor(int(n)) for n in global_role_counts]
        if local_weak > global_weak or local_core > global_core:
            raise ValueError("local role counts exceed global role counts")
    active_weight = cfg.weak_weight * (global_weak > 0) + cfg.core_weight * (global_core > 0)
    if active_weight <= 0:
        raise ValueError("present global roles have zero objective weight")
    row_weight = torch.where(weak, cfg.weak_weight / global_weak.clamp_min(1),
                             cfg.core_weight / global_core.clamp_min(1)) * int(world_size) / active_weight
    bce = F.binary_cross_entropy_with_logits(z, target, reduction="none")
    go_count = z.shape[1]
    positive_rows = (bce * positive).sum(1) / go_count
    negative_rows = (bce * negative).sum(1) / go_count
    positive_loss = (positive_rows * row_weight).sum()
    negative_loss = (negative_rows * row_weight).sum()
    loss = positive_loss + negative_loss
    with torch.no_grad():
        parts = {"loss": loss.detach(), "positive": positive_loss.detach(),
                 "negative": negative_loss.detach(),
                 "contrib_positive": positive_loss.detach(), "contrib_negative": negative_loss.detach(),
                 "full_go_denominator": z.new_tensor(go_count),
                 "positive_pairs_per_protein": positive.sum(1).float().mean(),
                 "negative_pairs_per_protein": negative.sum(1).float().mean(),
                 "alias_excluded_pairs_per_protein": excluded.sum(1).float().mean(),
                 "mean_probability": z.sigmoid().mean(),
                 "positive_probability": (z.sigmoid() * positive).sum() / positive.sum().clamp_min(1),
                 "negative_probability": (z.sigmoid() * negative).sum() / negative.sum().clamp_min(1)}
        for role, rows in (("core", ~weak), ("weak", weak)):
            parts[f"contrib_{role}_positive"] = (positive_rows * row_weight * rows).sum().detach()
            parts[f"contrib_{role}_negative"] = (negative_rows * row_weight * rows).sum().detach()
            parts[f"{role}_proteins"] = rows.sum().to(z.dtype)
    return loss, parts


def _training_ids(data, name):
    source = np.asarray(getattr(data, name))
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer ID array")
    ids = source.astype(np.int64, copy=False)
    if np.any(ids < 0) or np.unique(ids).size != len(ids):
        raise ValueError(f"{name} must contain unique nonnegative IDs")
    return ids


def compute_training_prior(data, lossconfig: FullTaskLossConfigV088 | None = None, *,
                           prior_clip: float = 1e-5, task_to_ontology=None):
    """Return (float32[G] prior, metadata), using only training CSR membership.

    Gold rows are indexed by global core ID; pseudo rows by registry.role_row.
    Neither probabilities nor holdout/inference labels are read. Each role's
    frequency is a population mean, mixed using the same role weights as loss.
    If task_to_ontology is supplied, exclude known-positive aliases in the
    frequency denominator too: prior = weighted positives / weighted eligible
    rows. This is the optimal per-GO constant prediction for this BCE before
    clipping. No alias label is promoted to a positive target.
    """
    cfg = lossconfig or FullTaskLossConfigV088()
    if isinstance(prior_clip, bool) or not isinstance(prior_clip, (int, float)) or not math.isfinite(prior_clip) or not 0 < prior_clip < .5:
        raise ValueError("prior_clip must be finite and in (0, .5)")
    go_count = int(data.num_task_go)
    if go_count <= 0:
        raise ValueError("num_task_go must be positive")
    core_ids, weak_ids = _training_ids(data, "core_ids"), _training_ids(data, "weak_ids")
    if np.intersect1d(core_ids, weak_ids).size:
        raise ValueError("training core and weak IDs must be disjoint")
    holdout = np.asarray(getattr(data, "validation_ids", []), dtype=np.int64)
    if np.intersect1d(core_ids, holdout).size or np.intersect1d(weak_ids, holdout).size:
        raise ValueError("training prior IDs must exclude validation/holdout IDs")
    alias_groups = {}
    mapping = None
    if task_to_ontology is not None:
        if isinstance(task_to_ontology, torch.Tensor):
            task_to_ontology = task_to_ontology.detach().cpu().numpy()
        raw_mapping = np.asarray(task_to_ontology)
        if raw_mapping.shape != (go_count,) or raw_mapping.dtype.kind not in "iu" or np.any(raw_mapping < 0):
            raise ValueError("task_to_ontology must be one nonnegative integer per GO")
        mapping = raw_mapping.astype(np.int64, copy=False)
        unique, multiplicity = np.unique(mapping, return_counts=True)
        alias_groups = {int(c): np.flatnonzero(mapping == c) for c in unique[multiplicity > 1]}
    roles = (("core", core_ids, cfg.core_weight, data.stores.gold_messages, core_ids),
             ("weak", weak_ids, cfg.weak_weight, data.stores.pseudo_messages,
              np.asarray(data.registry.role_row, dtype=np.int64)[weak_ids]))
    positive_rate = np.zeros(go_count, dtype=np.float64)
    eligible_rate = np.zeros(go_count, dtype=np.float64)
    role_info = {}
    active_weight = 0.0
    for role, ids, weight, store, rows in roles:
        info = {"training_proteins": len(ids), "weight": weight,
                "training_ids_sha256": hashlib.sha256(ids.astype("<i8", copy=False).tobytes()).hexdigest(),
                "positive_memberships": 0, "excluded_alias_memberships": 0}
        role_info[role] = info
        if not len(ids) or not weight:
            continue
        if store is None:
            raise ValueError(f"missing {role} membership CSR")
        if np.any(rows < 0) or np.any(rows + 1 >= len(store.indptr)):
            raise ValueError(f"{role} training rows outside CSR")
        positive_counts = np.zeros(go_count, dtype=np.int64)
        excluded_counts = np.zeros(go_count, dtype=np.int64)
        for row in rows:
            start, end = int(store.indptr[row]), int(store.indptr[row + 1])
            if start < 0 or end < start or end > len(store.go_idx):
                raise ValueError(f"invalid {role} CSR row boundaries")
            raw_go = np.asarray(store.go_idx[start:end])
            if raw_go.dtype.kind not in "iu" or np.any(raw_go < 0) or np.any(raw_go >= go_count):
                raise ValueError(f"{role} GO membership outside immutable classifier columns")
            go = np.unique(raw_go.astype(np.int64, copy=False))
            positive_counts[go] += 1
            if alias_groups and len(go):
                for canonical in np.unique(mapping[go]):
                    members = alias_groups.get(int(canonical))
                    if members is not None:
                        excluded = np.setdiff1d(members, go, assume_unique=True)
                        excluded_counts[excluded] += 1
        active_weight += weight
        positive_rate += weight * positive_counts / len(ids)
        eligible_rate += weight * (1 - excluded_counts / len(ids))
        info["positive_memberships"] = int(positive_counts.sum())
        info["excluded_alias_memberships"] = int(excluded_counts.sum())
    if not active_weight:
        raise ValueError("training prior has no populated role with positive weight")
    # A column excluded from every positively weighted training row is
    # unidentified by the loss. Set neutral initialization and report it.
    unidentified = eligible_rate <= 0
    prior = np.divide(positive_rate, eligible_rate, out=np.full(go_count, .5), where=~unidentified)
    result = np.clip(prior, prior_clip, 1 - prior_clip).astype(np.float32)
    metadata = {"scope": "training_core_gold_and_weak_binary_CSR_membership_only",
                "num_task_go": go_count, "roles": role_info,
                "normalized_core_weight": cfg.core_weight / active_weight if len(core_ids) else 0.,
                "normalized_weak_weight": cfg.weak_weight / active_weight if len(weak_ids) else 0.,
                "alias_eligibility_corrected": task_to_ontology is not None,
                "alias_groups": len(alias_groups), "unidentified_columns": int(unidentified.sum()),
                "prior_clip": prior_clip, "clipped_low_columns": int((prior < prior_clip).sum()),
                "clipped_high_columns": int((prior > 1 - prior_clip).sum()),
                "prior_min": float(result.min()), "prior_max": float(result.max()),
                "prior_mean": float(result.mean()),
                "prior_sha256": hashlib.sha256(result.astype("<f4", copy=False).tobytes()).hexdigest()}
    return result, metadata
