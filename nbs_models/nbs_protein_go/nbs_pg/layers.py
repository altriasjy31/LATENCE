from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import ActivationType, BackboneType, NormType

try:
    from torch_geometric.nn import GATConv, TransformerConv
    from torch_geometric.utils import scatter
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in PyG environments
    if exc.name != "torch_geometric":
        raise
    GATConv = TransformerConv = None
    scatter = None


def build_activation(name: ActivationType) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def build_norm(name: NormType, hidden_dim: int) -> nn.Module:
    if name == "layer":
        return nn.LayerNorm(hidden_dim)
    if name == "batch":
        return nn.BatchNorm1d(hidden_dim)
    if name == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported normalization: {name}")


class NBSMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        activation: ActivationType = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class WeightedSAGEConv(nn.Module):
    """Edge-aware bipartite mean aggregation with no target root transform."""

    def __init__(self, hidden_dim: int, edge_dim: Optional[int] = None) -> None:
        super().__init__()
        self.source_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_gate = (
            nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            if edge_dim is not None
            else None
        )
        if self.edge_gate is not None:
            nn.init.zeros_(self.edge_gate[-1].weight)
            nn.init.zeros_(self.edge_gate[-1].bias)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        if scatter is None:
            raise ModuleNotFoundError("torch_geometric is required for WeightedSAGEConv")
        x_src, x_dst = x
        src, dst = edge_index
        # ``source_proj`` is edge independent and bias free.  Project each
        # source node once, then gather by edge.  The previous order projected
        # the same 256-D source once per outgoing edge, which dominates the
        # large fixed-degree Protein-GO relations while producing the same
        # mathematical result.
        if src.numel() >= x_src.size(0):
            message = self.source_proj(x_src)[src]
        else:
            # Sparse relations can touch fewer source occurrences than there
            # are local nodes, so retain the edge-first path for those cases.
            message = self.source_proj(x_src[src])
        if self.edge_gate is None:
            weight = message.new_ones(src.numel(), 1)
        else:
            if edge_attr is None:
                raise ValueError("edge_attr is required for this weighted relation")
            edge_attr = edge_attr.to(message.dtype)
            # NBS reserves column 0 for confidence/calibrated probability.
            # The learned factor is initialized to 1, so evidence confidence is
            # active from the first forward pass rather than waiting to be learned.
            base_weight = edge_attr[:, :1].clamp(0.0, 1.0)
            learned_factor = 2.0 * torch.sigmoid(self.edge_gate(edge_attr))
            weight = base_weight * learned_factor
        out = scatter(message * weight, dst, dim=0, dim_size=x_dst.size(0), reduce="sum")
        count = scatter(
            torch.ones_like(weight), dst, dim=0, dim_size=x_dst.size(0), reduce="sum"
        )
        return out / count.clamp_min(1.0)


class NBSRelationConv(nn.Module):
    """Relation operator preserving an explicit external residual stream."""

    def __init__(
        self,
        kind: BackboneType,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        edge_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.edge_dim = edge_dim
        if kind == "sage":
            self.conv = WeightedSAGEConv(hidden_dim, edge_dim=edge_dim)
        elif kind == "gat":
            if GATConv is None:
                raise ModuleNotFoundError("torch_geometric is required for GAT")
            self.conv = GATConv(
                (hidden_dim, hidden_dim),
                hidden_dim,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=edge_dim,
            )
        elif kind == "transformer":
            if TransformerConv is None:
                raise ModuleNotFoundError("torch_geometric is required for TransformerConv")
            self.conv = TransformerConv(
                (hidden_dim, hidden_dim),
                hidden_dim,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                edge_dim=edge_dim,
                root_weight=False,
            )
        else:
            raise ValueError(f"Unsupported relation operator: {kind}")

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        if self.kind == "sage":
            return self.conv(x, edge_index, edge_attr=edge_attr)
        if self.edge_dim is not None:
            if edge_attr is None:
                raise ValueError("This relation was configured with edge_dim but edge_attr is absent")
            return self.conv(x, edge_index, edge_attr=edge_attr)
        return self.conv(x, edge_index)


class NBSRelationGate(nn.Module):
    """Dimension-wise relation gate; gated contributions remain exactly additive."""

    def __init__(self, hidden_dim: int, bias_init: float = -1.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(bias_init))

    def forward(self, target_state: Tensor, relation_delta: Tensor) -> Tensor:
        return torch.sigmoid(self.net(torch.cat([target_state, relation_delta], dim=-1)))


class NBSIdentityModulation(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.beta = nn.Linear(hidden_dim, hidden_dim)
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, x: Tensor) -> Tensor:
        transformed = self.gamma(x) * x + self.beta(x)
        gate = torch.sigmoid(self.gate(x))
        return gate * transformed + (1.0 - gate) * x
