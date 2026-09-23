from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

GO_BOX_EDGE_FEATURE_DIM = 8
GO_TOPOLOGY_EDGE_FEATURE_DIM = 2
GO_ONTOLOGY_EDGE_FEATURE_DIM = GO_BOX_EDGE_FEATURE_DIM + GO_TOPOLOGY_EDGE_FEATURE_DIM
ANNOTATION_EDGE_FEATURE_DIM = 3
CANDIDATE_EDGE_FEATURE_DIM = 3


@dataclass
class BoxPairMetrics:
    worst_margin: Tensor
    mean_margin: Tensor
    normalized_violation: Tensor
    center_distance: Tensor
    mean_log_offset_ratio: Tensor
    calibrated_probability: Optional[Tensor] = None


def _validate_boxes(center: Tensor, offset: Tensor, name: str) -> None:
    if center.dim() != 2 or offset.dim() != 2:
        raise ValueError(f"{name} center/offset must be [N, D]")
    if center.shape != offset.shape:
        raise ValueError(f"{name} center and offset must have identical shapes")


def box_pair_metrics(
    child_center: Tensor,
    child_offset: Tensor,
    parent_center: Tensor,
    parent_offset: Tensor,
    *,
    eps: float = 1e-8,
) -> BoxPairMetrics:
    """Directed child -> parent BoxSquaredEL measurements.

    ``worst_margin >= 0`` is exact axis-aligned box inclusion. The normalized
    violation is smoother and should be preferred for ranking and edge weights.
    """
    _validate_boxes(child_center, child_offset, "child")
    _validate_boxes(parent_center, parent_offset, "parent")
    if child_center.shape != parent_center.shape:
        raise ValueError("child and parent boxes must be pairwise aligned")

    child_offset = child_offset.abs()
    parent_offset = parent_offset.abs()
    gap = parent_offset - child_offset - (child_center - parent_center).abs()
    violation = F.relu(-gap)
    d = child_center.size(-1)
    return BoxPairMetrics(
        worst_margin=gap.amin(dim=-1),
        mean_margin=gap.mean(dim=-1),
        normalized_violation=torch.linalg.vector_norm(violation, dim=-1) / max(d ** 0.5, 1.0),
        center_distance=torch.linalg.vector_norm(child_center - parent_center, dim=-1) / max(d ** 0.5, 1.0),
        mean_log_offset_ratio=(
            torch.log(parent_offset.clamp_min(eps))
            - torch.log(child_offset.clamp_min(eps))
        ).mean(dim=-1),
    )


class BoxHierarchyCalibrator(nn.Module):
    """Calibrates soft BoxSquaredEL inclusion without redefining exact inclusion."""

    def __init__(
        self,
        threshold_init: float = -0.15,
        scale_init: float = 10.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        threshold = torch.tensor(float(threshold_init))
        raw_scale = torch.log(torch.expm1(torch.tensor(max(float(scale_init), 1e-4))))
        if trainable:
            self.threshold = nn.Parameter(threshold)
            self.raw_scale = nn.Parameter(raw_scale)
        else:
            self.register_buffer("threshold", threshold)
            self.register_buffer("raw_scale", raw_scale)

    @property
    def scale(self) -> Tensor:
        return F.softplus(self.raw_scale).clamp_min(1e-4)

    def forward(self, worst_margin: Tensor) -> Tensor:
        return torch.sigmoid(self.scale * (worst_margin - self.threshold))


def build_go_box_edge_features(
    center: Tensor,
    offset: Tensor,
    edge_index: Tensor,
    *,
    calibrator: Optional[BoxHierarchyCalibrator] = None,
    direction: float = 1.0,
    depth: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Create 8-dimensional directed GO edge features.

    Columns:
      calibrated probability, worst margin, mean margin,
      negative normalized violation, negative centre distance,
      mean log offset ratio, depth difference, direction indicator.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be [2, E]")
    child, parent = edge_index
    metrics = box_pair_metrics(
        center[child], offset[child], center[parent], offset[parent], eps=eps
    )
    if calibrator is None:
        probability = torch.sigmoid(10.0 * (metrics.worst_margin + 0.15))
    else:
        probability = calibrator(metrics.worst_margin)
    if depth is None:
        depth_diff = torch.zeros_like(metrics.worst_margin)
    else:
        depth_diff = depth[parent].to(metrics.worst_margin.dtype) - depth[child].to(
            metrics.worst_margin.dtype
        )
    direction_col = torch.full_like(metrics.worst_margin, float(direction))
    return torch.stack(
        [
            probability,
            metrics.worst_margin,
            metrics.mean_margin,
            -metrics.normalized_violation,
            -metrics.center_distance,
            metrics.mean_log_offset_ratio,
            depth_diff,
            direction_col,
        ],
        dim=-1,
    )


def build_go_topology_edge_features(
    num_edges: int,
    *,
    topology: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build [inverse_hop_weight, is_direct] features.

    ``topology`` is the raw LATENCE relation contract
    ``[hop_distance, is_direct]``.  The inverse-hop value is bounded in (0, 1]
    and can therefore serve as an absolute message weight for part-of edges.
    """
    if topology is None:
        return torch.ones(num_edges, 2, device=device, dtype=dtype)
    if topology.dim() != 2 or topology.size(0) != num_edges or topology.size(1) != 2:
        raise ValueError("topology must be [E, 2] with [hop_distance, is_direct]")
    topology = topology.to(device=device, dtype=dtype)
    hop = topology[:, 0].clamp_min(1.0)
    inverse_hop = hop.reciprocal()
    is_direct = topology[:, 1].clamp(0.0, 1.0)
    return torch.stack([inverse_hop, is_direct], dim=-1)


def append_go_topology_features(box_features: Tensor, topology_features: Tensor) -> Tensor:
    if box_features.dim() != 2 or box_features.size(1) != GO_BOX_EDGE_FEATURE_DIM:
        raise ValueError("box_features must be [E, 8]")
    if topology_features.shape != (box_features.size(0), GO_TOPOLOGY_EDGE_FEATURE_DIM):
        raise ValueError("topology_features must be [E, 2]")
    out = torch.cat([box_features, topology_features.to(box_features.dtype)], dim=-1)
    # Attenuate the calibrated inclusion confidence for transitive closure hops.
    out[:, 0] = out[:, 0] * topology_features[:, 0].to(out.dtype)
    return out


def default_candidate_edge_features(
    num_edges: int,
    *,
    backbone_probability: Optional[Tensor] = None,
    selector_score: Optional[Tensor] = None,
    reciprocal_rank: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create the LATENCE backbone-candidate three-column edge contract."""
    def _column(value: Optional[Tensor], default: float) -> Tensor:
        if value is None:
            return torch.full((num_edges,), default, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype).reshape(-1)
        if value.numel() != num_edges:
            raise ValueError("candidate feature columns must align with edge count")
        return value

    return torch.stack(
        [
            _column(backbone_probability, 1.0).clamp(0.0, 1.0),
            _column(selector_score, 0.0),
            _column(reciprocal_rank, 0.0).clamp_min(0.0),
        ],
        dim=-1,
    )


def default_annotation_edge_features(
    num_edges: int,
    *,
    confidence: Optional[Tensor] = None,
    is_pseudo: bool,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if confidence is None:
        confidence = torch.ones(num_edges, device=device, dtype=dtype)
    else:
        confidence = confidence.to(device=device, dtype=dtype).reshape(-1)
        if confidence.numel() != num_edges:
            raise ValueError("confidence must contain one value per annotation edge")
    true_flag = torch.zeros_like(confidence) if is_pseudo else torch.ones_like(confidence)
    pseudo_flag = torch.ones_like(confidence) if is_pseudo else torch.zeros_like(confidence)
    return torch.stack([confidence, true_flag, pseudo_flag], dim=-1)
