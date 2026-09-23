"""Fixed label-neighbour support for conservative PU weighting, never targets.

This module reads OTHER core annotations and fixed retrieval similarities.
It does not consume current predictions or learned gates. Agreement is a
structural heuristic, not a calibrated probability of biological correctness.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class StructuralSupportConfig:
    min_similarity: float = 0.5
    min_neighbors: int = 2
    min_vote_fraction: float = 0.5

    def __post_init__(self):
        for key in ("min_similarity", "min_vote_fraction"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and in [0,1]")
        if not isinstance(self.min_neighbors, int) or self.min_neighbors < 1:
            raise ValueError("min_neighbors must be a positive integer")


@torch.no_grad()
def structural_core_support(batch, num_go, config=None):
    cfg = config or StructuralSupportConfig()
    neighbor = batch["neighbor_index"].long()
    similarity = batch["neighbor_attr"][..., 0].detach().float()
    edge = batch["anchor_go_edge"].long()
    b, c = len(neighbor), len(batch["anchor_x"])
    result = similarity.new_zeros(b, num_go)
    if not c or not edge.numel():
        return result
    valid = (neighbor >= 0) & (neighbor < c) & torch.isfinite(similarity)
    valid &= (similarity >= cfg.min_similarity) & (similarity > 0)
    # Repeated neighbours/duplicate annotation rows must not manufacture votes.
    incidence = similarity.new_zeros(b, c)
    incidence.scatter_reduce_(1, neighbor.clamp(0, c - 1),
                              similarity.clamp(0, 1).masked_fill(~valid, 0),
                              reduce="amax", include_self=True)
    annotation = torch.sparse_coo_tensor(
        edge, torch.ones(edge.shape[1], device=edge.device), (c, num_go)
    ).coalesce()
    annotation = torch.sparse_coo_tensor(annotation.indices(),
                                         annotation.values().clamp_max(1),
                                         annotation.shape).coalesce()
    with torch.autocast(device_type=similarity.device.type, enabled=False):
        vote = torch.sparse.mm(annotation.t(), incidence.t()).t()
        count = torch.sparse.mm(annotation.t(), (incidence > 0).float().t()).t()
    fraction = vote / incidence.sum(1, keepdim=True).clamp_min(1e-8)
    supported = (count >= cfg.min_neighbors) & (fraction >= cfg.min_vote_fraction)
    return fraction.clamp(0, 1).masked_fill(~supported, 0)


@torch.no_grad()
def alias_pu_exclusions(positive_mask, task_to_ontology):
    """Keep all task outputs, but avoid PU pressure on known-positive aliases.

    This only affects loss eligibility. It does not add pseudo labels or alter
    the model input, output vocabulary, reference matrices or evaluation labels.
    """
    mapping = torch.as_tensor(task_to_ontology, device=positive_mask.device).long()
    if mapping.shape != (positive_mask.shape[1],) or (mapping < 0).any():
        raise ValueError("task-to-ontology mapping must match the full GO columns")
    if mapping.unique().numel() == mapping.numel():
        return torch.zeros_like(positive_mask, dtype=torch.bool)
    counts = torch.zeros(len(positive_mask), int(mapping.max()) + 1,
                         device=positive_mask.device, dtype=torch.int32)
    index = mapping[None].expand(len(positive_mask), -1)
    counts.scatter_add_(1, index, positive_mask.to(torch.int32))
    return (counts.gather(1, index) > 0) & ~positive_mask.bool()
