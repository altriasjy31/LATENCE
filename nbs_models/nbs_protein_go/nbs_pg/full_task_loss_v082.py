"""Dense Stage-1 modelout supervision and selected refinement for NBS v0.8.2.

Weak proteins use all task-GO teacher probabilities as soft targets.  Sparse
pseudo-label masks are diagnostic only for weak rows: an omitted probability
is never silently replaced by a negative label or a backbone target.  Core
proteins retain gold positives and conservative positive-unlabelled pressure.
Teacher values and all mining decisions are detached from the model graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .full_task_loss_v081 import _count_normalised, _negative_asl, _topk_mask


@dataclass(frozen=True)
class FullTaskLossConfigV082:
    teacher_weight: float = 1.0
    core_positive_weight: float = 1.0
    core_hard_pu_weight: float = 0.2
    core_background_pu_weight: float = 0.05
    core_medium_pu_weight: float = 0.05
    weak_weight: float = 1.0
    core_weight: float = 1.0
    refine_weight: float = 0.5
    teacher_low_threshold: float = 0.05
    teacher_high_threshold: float = 0.5
    hard_pu_k: int = 64
    background_pu_k: int = 64
    gamma_neg: float = 2.0
    negative_clip: float = 0.05
    support_pu_floor: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "teacher_weight", "core_positive_weight", "core_hard_pu_weight",
            "core_background_pu_weight", "core_medium_pu_weight", "weak_weight",
            "core_weight", "refine_weight", "gamma_neg", "negative_clip",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.negative_clip >= 1:
            raise ValueError("negative_clip must be below 1")
        if not 0 < self.support_pu_floor <= 1:
            raise ValueError("support_pu_floor must be in (0, 1]")
        if not 0 < self.teacher_low_threshold < self.teacher_high_threshold < 1:
            raise ValueError("Teacher stratum thresholds must satisfy 0 < low < high < 1")
        if self.weak_weight + self.core_weight <= 0:
            raise ValueError("At least one role weight must be nonzero")
        for name in ("hard_pu_k", "background_pu_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def full_task_loss_v082(
    logits: Tensor,
    base_logits: Tensor,
    targets: Tensor,
    positive_mask: Tensor,
    is_weak: Tensor,
    config: FullTaskLossConfigV082 | None = None,
    *,
    teacher_prob: Tensor,
    teacher_available: Tensor,
    structural_support: Tensor | None = None,
    pu_exclusion_mask: Tensor | None = None,
    classification_logits: Tensor | None = None,
    selected_mask: Tensor | None = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return FP32 loss plus detached scalar diagnostics.

    Main objective: dense teacher KL on weak proteins, gold/PU on core proteins.
    Teacher KL first averages each confidence stratum within each protein, then
    averages its nonempty strata.  Role means receive explicit weak/core weights
    independent of the counts of proteins in the minibatch.

    With a dual decoder, ``classification_logits`` is supervised over ALL GO.
    ``logits`` is the final output and receives an additional objective confined
    to ``selected_mask``.  Strata and core positive/PU means are recomputed in
    that subset, so a small matching budget is not divided by the full GO count.
    Outside that subset the refinement term has exactly zero gradient.

    ``fit_loss_comparable`` and ``gain_vs_backbone_objective`` describe the main
    objective (C for dual, final for a single head), excluding the refinement.
    Base is evaluated on the SAME masks as the current main output.  A separate
    ``final_fit_loss_comparable`` applies those same masks to final dual output.
    These are training fit diagnostics, not held-out performance gains.
    """
    cfg = config or FullTaskLossConfigV082()
    if logits.ndim != 2 or min(logits.shape) == 0:
        raise ValueError("logits must have nonempty shape [protein, task GO]")
    for name, value in (("base_logits", base_logits), ("targets", targets),
                        ("positive_mask", positive_mask), ("teacher_prob", teacher_prob)):
        if value.shape != logits.shape:
            raise ValueError(f"{name} must match logits")
    for name, value in (("structural_support", structural_support),
                        ("pu_exclusion_mask", pu_exclusion_mask),
                        ("classification_logits", classification_logits), ("selected_mask", selected_mask)):
        if value is not None and value.shape != logits.shape:
            raise ValueError(f"{name} must match logits")
    if is_weak.shape != (logits.shape[0],) or teacher_available.shape != is_weak.shape:
        raise ValueError("is_weak and teacher_available must have shape [protein]")
    if (classification_logits is None) != (selected_mask is None):
        raise ValueError("classification_logits and selected_mask must be supplied together")

    final = logits.float()
    main = final if classification_logits is None else classification_logits.float()
    base = base_logits.detach().float()
    weak = is_weak.detach().bool()
    core = ~weak
    positive = positive_mask.detach().bool()
    if (weak & ~teacher_available.detach().bool()).any():
        raise ValueError("Every weak protein requires a complete, aligned modelout teacher row")
    teacher = torch.where(weak[:, None], teacher_prob.detach().float(), torch.zeros_like(main))
    gold = torch.where(positive & core[:, None], targets.detach().float(), torch.zeros_like(main))
    support = torch.zeros_like(main) if structural_support is None else structural_support.detach().float()
    for name, value in (("teacher_prob on weak proteins", teacher), ("core gold targets", gold), ("structural_support", support)):
        if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError(f"{name} must be finite and in [0, 1]")

    with torch.no_grad():
        core_positive = positive & core[:, None]
        core_unknown = ~positive & core[:, None]
        if pu_exclusion_mask is not None:
            core_unknown &= ~pu_exclusion_mask.detach().bool()
        attenuation = 1 - (1 - cfg.support_pu_floor) * support
        entropy = -(teacher * teacher.clamp_min(1e-30).log()
                    + (1 - teacher) * (1 - teacher).clamp_min(1e-30).log())
        weak_count, core_count = weak.sum().float(), core.sum().float()
        role_total = (cfg.weak_weight * (weak_count > 0) + cfg.core_weight * (core_count > 0)).clamp_min(1e-8)
        row_weight = torch.where(weak, cfg.weak_weight / weak_count.clamp_min(1),
                                 cfg.core_weight / core_count.clamp_min(1)) / role_total
        ones = torch.ones_like(main)
        base_p = base.sigmoid()

    def weighted_rows(values: Tensor) -> Tensor:
        return (values * row_weight).sum()

    def masks_for(output: Tensor, scope: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad():
            eligible = core_unknown & scope
            mining = torch.maximum(output.detach().sigmoid(), base_p)
            hard = _topk_mask(mining, eligible, cfg.hard_pu_k)
            background = _topk_mask(torch.rand_like(mining), eligible & ~hard, cfg.background_pu_k)
            medium = eligible & ~(hard | background) & (mining > cfg.negative_clip)
            weak_scope = scope & weak[:, None]
            return {
                "low": weak_scope & (teacher < cfg.teacher_low_threshold),
                "mid": weak_scope & (teacher >= cfg.teacher_low_threshold) & (teacher < cfg.teacher_high_threshold),
                "high": weak_scope & (teacher >= cfg.teacher_high_threshold),
                "positive": core_positive & scope,
                "hard": hard, "background": background, "medium": medium,
            }

    def objective(output: Tensor, masks: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # BCE minus target entropy is KL(Bernoulli(M) || Bernoulli(output)).
        # There is no term pulling weak probabilities to zero or back to B.
        kl = (F.binary_cross_entropy_with_logits(output, teacher, reduction="none") - entropy).clamp_min(0)
        stratum_rows = torch.stack([_count_normalised(kl, masks[name], ones) for name in ("low", "mid", "high")], dim=1)
        active = torch.stack([masks[name].any(1) for name in ("low", "mid", "high")], dim=1)
        teacher_rows = stratum_rows.sum(1) / active.sum(1).clamp_min(1)
        positive_bce = F.binary_cross_entropy_with_logits(output, gold, reduction="none")
        negative = _negative_asl(output, cfg)
        components = {
            "core_positive": _count_normalised(positive_bce, masks["positive"], ones),
            "core_hard": _count_normalised(negative, masks["hard"], attenuation),
            "core_background": _count_normalised(negative, masks["background"], attenuation),
            "core_medium": _count_normalised(negative, masks["medium"], attenuation),
        }
        core_rows = (cfg.core_positive_weight * components["core_positive"]
                     + cfg.core_hard_pu_weight * components["core_hard"]
                     + cfg.core_background_pu_weight * components["core_background"]
                     + cfg.core_medium_pu_weight * components["core_medium"])
        result = {
            "teacher": weighted_rows(cfg.teacher_weight * teacher_rows),
            "core": weighted_rows(core_rows),
            "teacher_kl": teacher_rows.sum() / weak_count.clamp_min(1),
            "core_objective": core_rows.sum() / core_count.clamp_min(1),
        }
        result.update({f"teacher_kl_{name}": stratum_rows[:, i].sum() / weak_count.clamp_min(1)
                       for i, name in enumerate(("low", "mid", "high"))})
        result.update({key: value.sum() / core_count.clamp_min(1) for key, value in components.items()})
        result["total"] = result["teacher"] + result["core"]
        return result

    full_scope = torch.ones_like(positive)
    main_masks = masks_for(main, full_scope)
    current = objective(main, main_masks)
    zero = final.sum() * 0
    refine = zero
    selection = torch.zeros_like(positive) if selected_mask is None else selected_mask.detach().bool()
    if classification_logits is not None and cfg.refine_weight:
        refine_parts = objective(final, masks_for(final, selection))
        refine = cfg.refine_weight * refine_parts["total"]
    comparable = current["total"]
    loss = comparable + refine

    with torch.no_grad():
        baseline = objective(base, main_masks)["total"]
        final_comparable = comparable.detach() if classification_logits is None else objective(final.detach(), main_masks)["total"]
        delta = final.detach().sigmoid() - base_p
        positive_count = positive.sum().float()
        core_hard = main_masks["hard"]
        hard_count = core_hard.sum().float()
        high_teacher = weak[:, None] & (teacher >= cfg.teacher_high_threshold)
        diagnostics = {key: value.detach() for key, value in current.items() if key not in ("total", "teacher", "core")}
        diagnostics.update({
            "loss": loss.detach(),
            "fit_loss_comparable": comparable.detach(),
            "base_reference_loss": baseline,
            "gain_vs_backbone_objective": baseline - comparable.detach(),
            "final_fit_loss_comparable": final_comparable,
            "final_gain_vs_backbone_objective": baseline - final_comparable,
            "gain_excludes_refine": main.new_tensor(1.),
            "contrib_teacher": current["teacher"].detach(),
            "contrib_core": current["core"].detach(),
            "contrib_refine": refine.detach(),
            "comparable_objective": comparable.detach(),
            "base_objective": baseline,
            "objective_gain": baseline - comparable.detach(),
            "weak_proteins": weak_count,
            "core_proteins": core_count,
            "weak_role_mass": row_weight[weak].sum(),
            "core_role_mass": row_weight[core].sum(),
            "teacher_pairs_per_weak_protein": weak_count.gt(0).float() * main.shape[1],
            "teacher_high_pairs_per_weak_protein": high_teacher.sum().float() / weak_count.clamp_min(1),
            "positive_pairs_per_protein": positive.float().sum(1).mean(),
            "hard_pu_pairs_per_protein": core_hard.float().sum(1).mean(),
            "hard_pu_pairs_per_core_protein": hard_count / core_count.clamp_min(1),
            "background_pu_pairs_per_protein": main_masks["background"].float().sum(1).mean(),
            "medium_pu_pairs_per_protein": main_masks["medium"].float().sum(1).mean(),
            "core_pu_excluded_pairs_per_protein": ((~positive & core[:, None]) & ~core_unknown).float().sum(1).mean(),
            "positive_mean_probability_delta": (delta * positive).sum() / positive_count.clamp_min(1),
            "hard_mean_probability_delta": (delta * core_hard).sum() / hard_count.clamp_min(1),
            "abs_logit_delta": (final.detach() - base).abs().mean(),
            "classification_abs_logit_delta": (main.detach() - base).abs().mean(),
            "selected_pairs_per_protein": selection.float().sum(1).mean(),
            "selected_fraction": selection.float().mean(),
            "selected_high_teacher_coverage": (selection & high_teacher).sum().float() / high_teacher.sum().clamp_min(1),
            "selected_core_positive_coverage": (selection & core_positive).sum().float() / core_positive.sum().clamp_min(1),
        })
    return loss, diagnostics
