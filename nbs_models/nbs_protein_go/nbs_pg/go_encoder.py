from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .config import NBSConfig
from .layers import NBSMLP, NBSRelationConv, build_activation, build_norm
from .schema import GO_HAS_CHILD, GO_HAS_PART, GO_IS_A, GO_PART_OF, EdgeType
from .types import BoxGOEncoding, NBSGOBoxCache


class BoxSquaredGOEncoder(nn.Module):
    """Encodes BoxSquaredEL centre and offset through separate channels."""

    def __init__(self, config: NBSConfig, box_dim: int) -> None:
        super().__init__()
        config.validate()
        if box_dim <= 0:
            raise ValueError("box_dim must be positive")
        self.config = config
        self.box_dim = box_dim
        d = config.hidden_dim
        self.center_norm = nn.LayerNorm(box_dim)
        self.offset_norm = nn.LayerNorm(box_dim)
        self.center_proj = NBSMLP(
            box_dim, max(d, box_dim // 2), d, config.activation, config.input_dropout
        )
        self.offset_proj = NBSMLP(
            box_dim, max(d, box_dim // 2), d, config.activation, config.input_dropout
        )
        self.stats_proj = (
            NBSMLP(config.go_stat_dim, d, d, config.activation, config.input_dropout)
            if config.go_stat_dim > 0
            else None
        )
        gate_in = 2 * d + (d if self.stats_proj is not None else 0)
        self.hierarchy_gate = nn.Sequential(
            nn.Linear(gate_in, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        nn.init.zeros_(self.hierarchy_gate[-1].weight)
        nn.init.constant_(
            self.hierarchy_gate[-1].bias,
            float(config.go_hierarchy_gate_bias_init),
        )
        self.output_norm = nn.LayerNorm(d)

    def forward(
        self,
        center: Tensor,
        offset: Tensor,
        stats: Optional[Tensor] = None,
    ) -> BoxGOEncoding:
        if center.dim() != 2 or offset.dim() != 2:
            raise ValueError("GO center and offset must be [G, box_dim]")
        if center.shape != offset.shape or center.size(-1) != self.box_dim:
            raise ValueError("GO center/offset dimensions do not match box_dim")
        offset = offset.abs()
        log_offset = torch.log(offset.clamp_min(self.config.box_log_eps))
        semantic = self.center_proj(self.center_norm(center))
        hierarchy = self.offset_proj(self.offset_norm(log_offset))

        gate_parts = [semantic, hierarchy]
        encoded_stats: Optional[Tensor] = None
        if self.stats_proj is not None:
            if stats is None:
                stats = center.new_zeros(center.size(0), self.config.go_stat_dim)
            if stats.shape != (center.size(0), self.config.go_stat_dim):
                raise ValueError(
                    f"stats must be [{center.size(0)}, {self.config.go_stat_dim}]"
                )
            encoded_stats = self.stats_proj(stats.to(center.dtype))
            hierarchy = hierarchy + encoded_stats
            gate_parts.append(encoded_stats)
        elif stats is not None and stats.size(-1) != 0:
            raise ValueError("GO stats were provided but go_stat_dim=0")

        gate = torch.sigmoid(self.hierarchy_gate(torch.cat(gate_parts, dim=-1)))
        static = self.output_norm(semantic + gate * hierarchy)
        return BoxGOEncoding(
            semantic=semantic,
            hierarchy=hierarchy,
            static=static,
            hierarchy_gate=gate,
            center=center,
            offset=offset,
            log_offset=log_offset,
            stats=stats,
        )


class NBSGOOntologyTower(nn.Module):
    """Small full-GO tower that preserves the static box anchor.

    It can be evaluated on all ~43k GO nodes and cached before protein-rooted
    neighbour sampling. The heterogeneous backbone may then continue to update
    the cached GO state with annotation evidence.
    """

    def __init__(
        self,
        config: NBSConfig,
        edge_input_dims: Optional[Mapping[EdgeType, int]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        edge_input_dims = edge_input_dims or {}
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.scales = nn.Parameter(torch.full(
            (config.go_tower_layers,), float(config.go_tower_scale_init)
        ))
        for _ in range(config.go_tower_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "is_a": NBSRelationConv(
                            config.backbone_type,
                            d,
                            config.num_heads,
                            config.relation_dropout,
                            edge_dim=edge_input_dims.get(GO_IS_A),
                        ),
                        "has_child": NBSRelationConv(
                            config.backbone_type,
                            d,
                            config.num_heads,
                            config.relation_dropout,
                            edge_dim=edge_input_dims.get(GO_HAS_CHILD),
                        ),
                        "part_of": NBSRelationConv(
                            config.backbone_type,
                            d,
                            config.num_heads,
                            config.relation_dropout,
                            edge_dim=edge_input_dims.get(GO_PART_OF),
                        ),
                        "has_part": NBSRelationConv(
                            config.backbone_type,
                            d,
                            config.num_heads,
                            config.relation_dropout,
                            edge_dim=edge_input_dims.get(GO_HAS_PART),
                        ),
                    }
                )
            )
            self.norms.append(build_norm(config.norm, d))
        self.activation = build_activation(config.activation)
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        static: Tensor,
        edge_index_dict: Mapping[EdgeType, Tensor],
        edge_attr_dict: Optional[Mapping[EdgeType, Tensor]] = None,
    ) -> tuple[Tensor, list[Tensor]]:
        if self.config.go_tower_layers == 0 or not self.config.use_go_tower:
            return static, []
        edge_attr_dict = edge_attr_dict or {}
        h = static
        states: list[Tensor] = []
        for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = norm(h)
            messages = []
            for name, edge_type in (
                ("is_a", GO_IS_A),
                ("has_child", GO_HAS_CHILD),
                ("part_of", GO_PART_OF),
                ("has_part", GO_HAS_PART),
            ):
                edge_index = edge_index_dict.get(edge_type)
                if edge_index is None or edge_index.numel() == 0:
                    continue
                messages.append(
                    layer[name]((x, x), edge_index, edge_attr_dict.get(edge_type))
                )
            if messages:
                delta = torch.stack(messages, dim=0).mean(dim=0)
                delta = self.dropout(self.activation(delta))
                h = h + self.scales[layer_idx] * delta
            states.append(h)
        return h, states

    @torch.no_grad()
    def make_cache(
        self,
        box: BoxGOEncoding,
        edge_index_dict: Mapping[EdgeType, Tensor],
        edge_attr_dict: Optional[Mapping[EdgeType, Tensor]] = None,
    ) -> NBSGOBoxCache:
        was_training = self.training
        self.eval()
        context, _ = self(box.static, edge_index_dict, edge_attr_dict)
        if was_training:
            self.train()
        return NBSGOBoxCache(
            semantic=box.semantic.detach(),
            hierarchy=box.hierarchy.detach(),
            static=box.static.detach(),
            context=context.detach(),
            center=box.center.detach(),
            offset=box.offset.detach(),
            stats=None if box.stats is None else box.stats.detach(),
        )
