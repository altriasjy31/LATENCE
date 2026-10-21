from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import NBSConfig
from .layers import NBSRelationConv, NBSRelationGate, WeightedSAGEConv, build_activation, build_norm
from .schema import EdgeType
from .types import NBSHeteroBackboneOutput, NBSHomogeneousBackboneOutput

Metadata = Tuple[Sequence[str], Sequence[EdgeType]]


def edge_type_key(edge_type: EdgeType) -> str:
    return "___".join(edge_type)


class NBSHomogeneousResidualBackbone(nn.Module):
    """Homogeneous reference backbone using the same additive hierarchy semantics."""

    def __init__(self, config: NBSConfig, input_dim: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.hidden_dim
        self.input_proj = nn.Linear(input_dim, d)
        self.input_drop = nn.Dropout(config.input_dropout)
        self.activation = build_activation(config.activation)
        self.pre_norms = nn.ModuleList([build_norm(config.norm, d) for _ in range(config.num_layers)])
        self.delta_norms = nn.ModuleList([build_norm(config.norm, d) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.residual_dropout)
        self.convs = nn.ModuleList([WeightedSAGEConv(d) for _ in range(config.num_layers)])
        self.residual_scales = nn.Parameter(
            torch.full((config.num_layers,), float(config.residual_scale_init))
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> NBSHomogeneousBackboneOutput:
        h = self.input_drop(self.activation(self.input_proj(x)))
        states: List[Tensor] = []
        deltas: List[Tensor] = []
        for layer, (conv, pre_norm, delta_norm) in enumerate(
            zip(self.convs, self.pre_norms, self.delta_norms)
        ):
            message = conv((pre_norm(h), pre_norm(h)), edge_index)
            delta = self.dropout(self.activation(delta_norm(message)))
            delta = self.residual_scales[layer] * delta
            h = h + delta
            deltas.append(delta)
            states.append(h)
        names = tuple(f"layer:{i + 1}" for i in range(self.config.num_layers))
        return NBSHomogeneousBackboneOutput(h, states, deltas, names)


class NBSHeterogeneousResidualBackbone(nn.Module):
    """Relation-aware additive residual backbone for NBS.

    Every relation produces a contribution. With ``gated_sum``, a dimension-wise
    gate controls its strength per destination node, and the exposed relation
    sources still add exactly to the destination residual.
    """

    def __init__(
        self,
        config: NBSConfig,
        metadata: Metadata,
        node_input_dims: Mapping[str, int],
        edge_input_dims: Optional[Mapping[EdgeType, int]] = None,
        preencoded_node_types: Optional[set[str]] = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.node_types = tuple(metadata[0])
        self.edge_types = tuple(tuple(e) for e in metadata[1])
        self.target_type = config.target_node_type
        if self.target_type not in self.node_types:
            raise ValueError(f"target type {self.target_type!r} is absent from metadata")
        missing = set(self.node_types) - set(node_input_dims)
        if missing:
            raise ValueError(f"Missing node feature dimensions: {sorted(missing)}")

        preencoded_node_types = preencoded_node_types or set()
        d = config.hidden_dim
        self.input_proj = nn.ModuleDict()
        for nt in self.node_types:
            if nt in preencoded_node_types:
                if node_input_dims[nt] != d:
                    raise ValueError(f"preencoded node type {nt!r} must have hidden_dim features")
                self.input_proj[nt] = nn.Identity()
            else:
                self.input_proj[nt] = nn.Linear(node_input_dims[nt], d)
        self.type_embedding = nn.ParameterDict(
            {nt: nn.Parameter(torch.zeros(d)) for nt in self.node_types}
        )
        self.input_drop = nn.Dropout(config.input_dropout)
        self.activation = build_activation(config.activation)
        self.relation_dropout = nn.Dropout(config.relation_dropout)

        edge_input_dims = edge_input_dims or {}
        self.relation_layers = nn.ModuleList()
        self.relation_norms = nn.ModuleList()
        self.relation_gates = nn.ModuleList()
        self.pre_norms = nn.ModuleList()
        for _ in range(config.num_layers):
            convs = nn.ModuleDict()
            norms = nn.ModuleDict()
            gates = nn.ModuleDict()
            pre = nn.ModuleDict({nt: build_norm(config.norm, d) for nt in self.node_types})
            for edge_type in self.edge_types:
                key = edge_type_key(edge_type)
                convs[key] = NBSRelationConv(
                    config.backbone_type,
                    d,
                    config.num_heads,
                    config.relation_dropout,
                    edge_dim=edge_input_dims.get(edge_type),
                )
                norms[key] = build_norm(config.norm, d)
                gates[key] = NBSRelationGate(d, config.relation_gate_bias_init)
            self.relation_layers.append(convs)
            self.relation_norms.append(norms)
            self.relation_gates.append(gates)
            self.pre_norms.append(pre)

        self.relation_scales = nn.ParameterDict(
            {
                edge_type_key(e): nn.Parameter(
                    torch.full((config.num_layers,), float(config.relation_scale_init))
                )
                for e in self.edge_types
            }
        )
        self.residual_scales = nn.ParameterDict(
            {
                nt: nn.Parameter(
                    torch.full((config.num_layers,), float(config.residual_scale_init))
                )
                for nt in self.node_types
            }
        )
        self.incoming_target_relations = tuple(e for e in self.edge_types if e[2] == self.target_type)
        if not self.incoming_target_relations:
            raise ValueError(f"No relations point to target type {self.target_type!r}")

    @property
    def num_target_sources(self) -> int:
        if self.config.source_mode == "layer":
            return self.config.num_layers
        return self.config.num_layers * len(self.incoming_target_relations)

    def forward(
        self,
        x_dict: Mapping[str, Tensor],
        edge_index_dict: Mapping[EdgeType, Tensor],
        edge_attr_dict: Optional[Mapping[EdgeType, Tensor]] = None,
    ) -> NBSHeteroBackboneOutput:
        edge_attr_dict = edge_attr_dict or {}
        h: Dict[str, Tensor] = {
            nt: self.input_drop(
                self.activation(self.input_proj[nt](x_dict[nt]) + self.type_embedding[nt])
            )
            for nt in self.node_types
        }
        layer_states: List[Dict[str, Tensor]] = []
        layer_deltas: List[Dict[str, Tensor]] = []
        target_relation_sources: List[Tensor] = []
        target_relation_names: List[str] = []
        target_layer_sources: List[Tensor] = []
        relation_gate_means: Dict[str, Tensor] = {}

        for layer_idx, (convs, rel_norms, rel_gates, pre_norms) in enumerate(
            zip(self.relation_layers, self.relation_norms, self.relation_gates, self.pre_norms)
        ):
            normalized = {nt: pre_norms[nt](h[nt]) for nt in self.node_types}
            raw_relation_delta: Dict[EdgeType, Tensor] = {}
            relation_contribution: Dict[EdgeType, Tensor] = {}
            relation_active: Dict[EdgeType, bool] = {}
            incoming: Dict[str, List[Tuple[EdgeType, Tensor]]] = defaultdict(list)

            for edge_type in self.edge_types:
                src, _, dst = edge_type
                key = edge_type_key(edge_type)
                edge_index = edge_index_dict.get(edge_type)
                active = edge_index is not None and edge_index.numel() > 0
                relation_active[edge_type] = active
                if not active:
                    raw = torch.zeros_like(h[dst])
                else:
                    message = convs[key](
                        (normalized[src], normalized[dst]),
                        edge_index,
                        edge_attr_dict.get(edge_type),
                    )
                    raw = self.relation_dropout(self.activation(rel_norms[key](message)))
                    raw = self.relation_scales[key][layer_idx] * raw
                raw_relation_delta[edge_type] = raw

                if self.config.relation_aggr == "gated_sum":
                    gate = rel_gates[key](normalized[dst], raw) if active else torch.zeros_like(raw)
                    contribution = gate * raw
                    relation_gate_means[f"layer:{layer_idx + 1}|{key}"] = gate.mean().detach()
                else:
                    contribution = raw
                    relation_gate_means[f"layer:{layer_idx + 1}|{key}"] = raw.new_tensor(
                        1.0 if active else 0.0
                    )
                relation_contribution[edge_type] = contribution
                if active:
                    incoming[dst].append((edge_type, contribution))

            next_h: Dict[str, Tensor] = {}
            current_layer_delta: Dict[str, Tensor] = {}
            target_contributions: Dict[EdgeType, Tensor] = {}
            for node_type in self.node_types:
                messages = incoming.get(node_type, [])
                if not messages:
                    aggregate = torch.zeros_like(h[node_type])
                    denom = 1
                else:
                    aggregate = torch.stack([value for _, value in messages], dim=0).sum(dim=0)
                    denom = len(messages) if self.config.relation_aggr == "mean" else 1
                    aggregate = aggregate / denom

                target_scale = self.residual_scales[node_type][layer_idx]
                # Shared dropout mask preserves exact source additivity.
                mask = F.dropout(
                    torch.ones_like(aggregate),
                    p=self.config.residual_dropout,
                    training=self.training,
                )
                delta = target_scale * aggregate * mask
                next_h[node_type] = h[node_type] + delta
                current_layer_delta[node_type] = delta

                if node_type == self.target_type:
                    for edge_type in self.incoming_target_relations:
                        if relation_active[edge_type]:
                            source = target_scale * relation_contribution[edge_type] / denom * mask
                        else:
                            source = torch.zeros_like(delta)
                        target_contributions[edge_type] = source

            target_layer_sources.append(current_layer_delta[self.target_type])
            for edge_type in self.incoming_target_relations:
                target_relation_sources.append(target_contributions[edge_type])
                target_relation_names.append(
                    f"layer:{layer_idx + 1}|relation:{'-'.join(edge_type)}"
                )

            h = next_h
            layer_states.append(dict(h))
            layer_deltas.append(current_layer_delta)

        if self.config.source_mode == "layer":
            target_sources = target_layer_sources
            names = tuple(
                f"layer:{i + 1}|target:{self.target_type}"
                for i in range(self.config.num_layers)
            )
        else:
            target_sources = target_relation_sources
            names = tuple(target_relation_names)

        return NBSHeteroBackboneOutput(
            final_states=h,
            layer_states=layer_states,
            layer_deltas=layer_deltas,
            target_sources=target_sources,
            target_source_names=names,
            relation_gate_means=relation_gate_means,
        )
