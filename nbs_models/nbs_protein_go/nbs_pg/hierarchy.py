from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .types import (
    NBSHeteroBackboneOutput,
    NBSHomogeneousBackboneOutput,
    NBSNeighborhoodHierarchy,
)


class NBSHierarchyAdapter(nn.Module):
    """Maps homogeneous or heterogeneous residual streams to the NBS-GDAR API."""

    def __init__(self, hidden_dim: int, use_projection: bool = True) -> None:
        super().__init__()
        if use_projection:
            self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.eye_(self.projection.weight)
        else:
            self.projection = nn.Identity()

    def from_homogeneous(
        self,
        output: NBSHomogeneousBackboneOutput,
        node_ids: Optional[Tensor] = None,
    ) -> NBSNeighborhoodHierarchy:
        hierarchy = NBSNeighborhoodHierarchy(
            final_context=self.projection(output.final_state),
            source_contexts=torch.stack(
                [self.projection(delta) for delta in output.layer_deltas], dim=0
            ),
            source_names=output.source_names,
            layer_contexts=torch.stack(
                [self.projection(state) for state in output.layer_states], dim=0
            ),
            target_global_ids=node_ids,
        )
        hierarchy.validate()
        return hierarchy

    def from_heterogeneous(
        self,
        output: NBSHeteroBackboneOutput,
        target_node_type: str,
        node_ids: Optional[Tensor] = None,
    ) -> NBSNeighborhoodHierarchy:
        hierarchy = NBSNeighborhoodHierarchy(
            final_context=self.projection(output.final_states[target_node_type]),
            source_contexts=torch.stack(
                [self.projection(source) for source in output.target_sources], dim=0
            ),
            source_names=output.target_source_names,
            layer_contexts=torch.stack(
                [self.projection(state[target_node_type]) for state in output.layer_states],
                dim=0,
            ),
            target_global_ids=node_ids,
        )
        hierarchy.validate()
        return hierarchy
