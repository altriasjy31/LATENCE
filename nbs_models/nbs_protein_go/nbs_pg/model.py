from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.data import HeteroData
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "torch_geometric":
        raise
    HeteroData = object  # type: ignore[misc,assignment]

from .backbones import (
    Metadata,
    NBSHeterogeneousResidualBackbone,
    NBSHomogeneousResidualBackbone,
    edge_type_key,
)
from .box_geometry import (
    BoxHierarchyCalibrator,
    append_go_topology_features,
    build_go_box_edge_features,
)
from .config import NBSConfig
from .go_encoder import BoxSquaredGOEncoder, NBSGOOntologyTower
from .hierarchy import NBSHierarchyAdapter
from .matcher import NBSGatedDeltaAttnRes
from .query import NBSProteinGOQueryEncoder
from .schema import (
    BACKBONE_CANDIDATE_GO_TO_PROTEIN,
    BACKBONE_CANDIDATE_PROTEIN_TO_GO,
    GO,
    GO_HAS_CHILD,
    GO_HAS_PART,
    GO_IS_A,
    GO_PART_OF,
    GOLD_GO_TO_PROTEIN,
    GOLD_PROTEIN_TO_GO,
    NBS_PROTEIN_GO_METADATA,
    PPI,
    PROTEIN,
    PSEUDO_GO_TO_PROTEIN,
    PSEUDO_PROTEIN_TO_GO,
    SIMILAR_TO,
    WEAK_TO_CORE,
    EdgeType,
)
from .types import (
    NBSGOBoxCache,
    NBSMatchOutput,
    NBSNeighborhoodHierarchy,
    NBSQueryCondition,
    ProteinGOEncodedGraph,
    ProteinGOQueryBatch,
)


def default_nbs_edge_input_dims(config: NBSConfig) -> dict[EdgeType, int]:
    """Edge dimensions matching the actual LATENCE manifests."""
    return {
        PPI: config.pp_edge_dim,
        SIMILAR_TO: config.pp_edge_dim,
        WEAK_TO_CORE: config.pp_edge_dim,
        GOLD_PROTEIN_TO_GO: config.annotation_edge_dim,
        GOLD_GO_TO_PROTEIN: config.annotation_edge_dim,
        BACKBONE_CANDIDATE_PROTEIN_TO_GO: config.candidate_edge_dim,
        BACKBONE_CANDIDATE_GO_TO_PROTEIN: config.candidate_edge_dim,
        PSEUDO_PROTEIN_TO_GO: config.annotation_edge_dim,
        PSEUDO_GO_TO_PROTEIN: config.annotation_edge_dim,
        GO_IS_A: config.go_edge_dim,
        GO_HAS_CHILD: config.go_edge_dim,
        GO_PART_OF: config.go_part_edge_dim,
        GO_HAS_PART: config.go_part_edge_dim,
    }


class ProteinGONBSModel(nn.Module):
    """Neighborhood--BoxSquare model using BoxSquaredEL GO geometry."""

    # Bump this only when the public isolated-inductive scoring contract
    # changes.  The exporter checks it before reading data or a checkpoint so
    # that a new exporter cannot silently run against an older model module.
    INDUCTIVE_INFERENCE_API_VERSION = 5

    def __init__(
        self,
        config: NBSConfig,
        protein_input_dim: int,
        go_box_dim: int,
        *,
        metadata: Metadata = NBS_PROTEIN_GO_METADATA,
        edge_input_dims: Optional[Mapping[EdgeType, int]] = None,
    ) -> None:
        super().__init__()
        config.validate()
        if config.target_node_type != PROTEIN:
            raise ValueError("ProteinGONBSModel requires target_node_type='protein'")
        self.config = config
        self.metadata = metadata
        self.box_calibrator = BoxHierarchyCalibrator(
            config.box_margin_threshold_init,
            config.box_calibration_scale_init,
            trainable=config.train_box_calibrator,
        )
        self.go_box_encoder = BoxSquaredGOEncoder(config, go_box_dim)
        edge_input_dims = dict(default_nbs_edge_input_dims(config)) | dict(edge_input_dims or {})
        self.go_tower = NBSGOOntologyTower(config, edge_input_dims=edge_input_dims)
        self.go_context_scale = nn.Parameter(torch.tensor(float(config.go_tower_scale_init)))
        self.backbone = NBSHeterogeneousResidualBackbone(
            config,
            metadata,
            node_input_dims={PROTEIN: protein_input_dim, GO: config.hidden_dim},
            edge_input_dims=edge_input_dims,
            preencoded_node_types={GO},
        )
        self.hierarchy_adapter = NBSHierarchyAdapter(config.hidden_dim, use_projection=True)
        self.query_encoder = NBSProteinGOQueryEncoder(config)
        self.matcher = NBSGatedDeltaAttnRes(config, self.backbone.num_target_sources)

    def _edge_dicts(self, graph: HeteroData) -> tuple[dict, dict]:
        edge_index_dict = dict(graph.edge_index_dict)
        edge_attr_dict = {
            edge_type: graph[edge_type].edge_attr
            for edge_type in graph.edge_types
            if hasattr(graph[edge_type], "edge_attr")
        }
        # Recompute GO edge features through the trainable calibrator. The exact
        # ontology edges stay fixed; only their soft message weights are fitted.
        depth = None
        if hasattr(graph[GO], "stats") and graph[GO].stats.size(-1) >= 5:
            depth = graph[GO].stats[:, 4]
        if GO_IS_A in edge_index_dict and edge_index_dict[GO_IS_A].numel() > 0:
            topology = getattr(graph[GO_IS_A], "topology_attr", None)
            if topology is None:
                topology = graph[GO_IS_A].edge_attr[:, -self.config.go_topology_edge_dim :]
            edge_attr_dict[GO_IS_A] = append_go_topology_features(
                build_go_box_edge_features(
                    graph[GO].center,
                    graph[GO].offset,
                    edge_index_dict[GO_IS_A],
                    calibrator=self.box_calibrator,
                    direction=1.0,
                    depth=depth,
                    eps=self.config.box_log_eps,
                ),
                topology,
            )
        if GO_HAS_CHILD in edge_index_dict and edge_index_dict[GO_HAS_CHILD].numel() > 0:
            # Convert parent->child back to child->parent for asymmetric box
            # measurements.  The topology columns retain the original hop/direct
            # evidence and the feature direction is explicitly -1.
            child_parent = edge_index_dict[GO_HAS_CHILD].flip(0)
            topology = getattr(graph[GO_HAS_CHILD], "topology_attr", None)
            if topology is None:
                topology = graph[GO_HAS_CHILD].edge_attr[:, -self.config.go_topology_edge_dim :]
            edge_attr_dict[GO_HAS_CHILD] = append_go_topology_features(
                build_go_box_edge_features(
                    graph[GO].center,
                    graph[GO].offset,
                    child_parent,
                    calibrator=self.box_calibrator,
                    direction=-1.0,
                    depth=depth,
                    eps=self.config.box_log_eps,
                ),
                topology,
            )
        return edge_index_dict, edge_attr_dict

    def _encode_local_box(self, graph: HeteroData):
        if not hasattr(graph[GO], "center") or not hasattr(graph[GO], "offset"):
            raise ValueError("GO store must contain center and offset tensors")
        stats = getattr(graph[GO], "stats", None)
        return self.go_box_encoder(graph[GO].center, graph[GO].offset, stats)

    def encode_go_tower(self, graph: HeteroData) -> NBSGOBoxCache:
        """Encode the complete GO ontology for caching before protein sampling."""
        box = self._encode_local_box(graph)
        edge_index_dict, edge_attr_dict = self._edge_dicts(graph)
        context, _ = self.go_tower(box.static, edge_index_dict, edge_attr_dict)
        return NBSGOBoxCache(
            semantic=box.semantic,
            hierarchy=box.hierarchy,
            static=box.static,
            context=context,
            center=box.center,
            offset=box.offset,
            stats=box.stats,
        )

    @torch.no_grad()
    def make_go_cache(self, graph: HeteroData) -> NBSGOBoxCache:
        was_training = self.training
        self.eval()
        cache = self.encode_go_tower(graph)
        cache = NBSGOBoxCache(
            semantic=cache.semantic.detach(),
            hierarchy=cache.hierarchy.detach(),
            static=cache.static.detach(),
            context=cache.context.detach(),
            center=cache.center.detach(),
            offset=cache.offset.detach(),
            stats=None if cache.stats is None else cache.stats.detach(),
        )
        if was_training:
            self.train()
        return cache

    def encode_graph(
        self,
        graph: HeteroData,
        *,
        global_go_cache: Optional[NBSGOBoxCache] = None,
    ) -> ProteinGOEncodedGraph:
        local_box = self._encode_local_box(graph)
        edge_index_dict, edge_attr_dict = self._edge_dicts(graph)

        if global_go_cache is not None:
            global_ids = getattr(graph[GO], "n_id", None)
            if global_ids is None:
                global_ids = getattr(graph[GO], "node_id", None)
            if global_ids is None:
                raise ValueError("sampled GO nodes need n_id/node_id to index the global cache")
            cached_context = global_go_cache.context[global_ids]
        else:
            cached_context, _ = self.go_tower(
                local_box.static, edge_index_dict, edge_attr_dict
            )

        if self.config.use_static_go_anchor:
            go_initial = local_box.static + torch.tanh(self.go_context_scale) * (
                cached_context - local_box.static
            )
        else:
            go_initial = cached_context

        x_dict = {PROTEIN: graph[PROTEIN].x, GO: go_initial}
        output = self.backbone(x_dict, edge_index_dict, edge_attr_dict)
        protein_ids = getattr(graph[PROTEIN], "n_id", None)
        hierarchy = self.hierarchy_adapter.from_heterogeneous(
            output, target_node_type=PROTEIN, node_ids=protein_ids
        )
        return ProteinGOEncodedGraph(
            hierarchy=hierarchy,
            final_states=output.final_states,
            layer_states=output.layer_states,
            local_go_box=local_box,
            global_go_cache=global_go_cache,
            backbone_aux={
                key: value for key, value in output.relation_gate_means.items()
            },
        )

    def encode_external_protein_candidates(
        self,
        protein_x: Tensor,
        *,
        neighbor_x: Optional[Tensor] = None,
        neighbor_edge_attr: Optional[Tensor] = None,
        neighbor_relation: EdgeType = SIMILAR_TO,
        neighbor_fanouts: Optional[Sequence[int]] = None,
        source_names: Optional[tuple[str, ...]] = None,
    ) -> NBSNeighborhoodHierarchy:
        """Encode inductive proteins without inserting them into the train graph.

        This path is intended for independent-test inference.  The external
        protein uses the same protein input projection/type embedding as the
        heterogeneous backbone.  When ``neighbor_x`` is supplied, its
        ``[N,K,D_in]`` core-neighbour representations are aggregated as
        ``core -> external`` messages through the already-trained P--P relation
        operator (``similar_to`` by default).  No external protein is inserted
        into the frozen training graph and no test--test message is permitted.

        With no neighbours this remains the exact v0.5.3 feature-only fallback:
        all graph-relation residual sources are zero.
        """
        if protein_x.dim() != 2:
            raise ValueError("protein_x must be [N,D]")
        proj = self.backbone.input_proj[PROTEIN]
        h = self.backbone.input_drop(
            self.backbone.activation(
                proj(protein_x) + self.backbone.type_embedding[PROTEIN]
            )
        )
        source_count = int(self.backbone.num_target_sources)

        if neighbor_x is None:
            if neighbor_edge_attr is not None:
                raise ValueError("neighbor_edge_attr requires neighbor_x")
            if source_names is None:
                source_names = tuple(f"external_source:{i}" for i in range(source_count))
            if len(source_names) != source_count:
                raise ValueError("source_names do not match NBS target-source count")
            projected_h = self.hierarchy_adapter.projection(h)
            hierarchy = NBSNeighborhoodHierarchy(
                final_context=projected_h,
                source_contexts=projected_h.new_zeros(
                    source_count, projected_h.size(0), projected_h.size(1)
                ),
                source_names=tuple(source_names),
            )
            hierarchy.validate()
            return hierarchy

        if neighbor_x.dim() != 3 or neighbor_x.size(0) != protein_x.size(0):
            raise ValueError("neighbor_x must be [N,K,D_in] and align with protein_x")
        if neighbor_x.size(2) != protein_x.size(1):
            raise ValueError("neighbor and external protein input dimensions disagree")
        if neighbor_x.size(1) <= 0:
            raise ValueError("inductive P-P inference needs at least one neighbour")
        if neighbor_fanouts is None:
            resolved_fanouts = (int(neighbor_x.size(1)),) * self.config.num_layers
        else:
            resolved_fanouts = tuple(int(value) for value in neighbor_fanouts)
            if len(resolved_fanouts) != self.config.num_layers:
                raise ValueError("neighbor_fanouts must contain one value per NBS layer")
            if any(value <= 0 or value > neighbor_x.size(1) for value in resolved_fanouts):
                raise ValueError(
                    "every external P-P fanout must lie in [1, stored neighbour count]"
                )
        if neighbor_relation not in self.backbone.incoming_target_relations:
            raise ValueError(f"relation {neighbor_relation!r} does not point to protein")

        relation_key = edge_type_key(neighbor_relation)
        relation_dim = default_nbs_edge_input_dims(self.config).get(neighbor_relation)
        if relation_dim is not None:
            if neighbor_edge_attr is None:
                raise ValueError("the selected P-P relation requires neighbor_edge_attr")
            if neighbor_edge_attr.shape != (
                neighbor_x.size(0),
                neighbor_x.size(1),
                int(relation_dim),
            ):
                raise ValueError(
                    "neighbor_edge_attr must be [N,K,F] with the trained relation feature width"
                )
        elif neighbor_edge_attr is not None and neighbor_edge_attr.shape[:2] != neighbor_x.shape[:2]:
            raise ValueError("neighbor_edge_attr rows do not align with neighbor_x")

        # Neighbours are immutable core anchors.  The external target evolves
        # across NBS layers, while the same retrieved core evidence is available
        # at every layer.  This is deliberately isolated per test protein.
        n_external, k_neighbors, _ = neighbor_x.shape
        flat_neighbor_x = neighbor_x.reshape(n_external * k_neighbors, -1)
        neighbor_h = self.backbone.input_drop(
            self.backbone.activation(
                proj(flat_neighbor_x) + self.backbone.type_embedding[PROTEIN]
            )
        ).reshape(n_external, k_neighbors, -1)

        relation_sources: list[Tensor] = []
        relation_names: list[str] = []
        layer_sources: list[Tensor] = []
        for layer_idx in range(self.config.num_layers):
            layer_fanout = resolved_fanouts[layer_idx]
            layer_neighbor_h = neighbor_h[:, :layer_fanout].reshape(
                n_external * layer_fanout, -1
            )
            src = torch.arange(
                n_external * layer_fanout, device=h.device, dtype=torch.long
            )
            dst = torch.arange(
                n_external, device=h.device, dtype=torch.long
            ).repeat_interleave(layer_fanout)
            edge_index = torch.stack([src, dst], dim=0)
            flat_edge_attr = (
                None
                if neighbor_edge_attr is None
                else neighbor_edge_attr[:, :layer_fanout].reshape(
                    n_external * layer_fanout, -1
                )
            )
            pre_norm = self.backbone.pre_norms[layer_idx][PROTEIN]
            normalized_neighbor = pre_norm(layer_neighbor_h)
            normalized_target = pre_norm(h)
            raw = self.backbone.relation_layers[layer_idx][relation_key](
                (normalized_neighbor, normalized_target),
                edge_index,
                flat_edge_attr,
            )
            raw = self.backbone.relation_dropout(
                self.backbone.activation(
                    self.backbone.relation_norms[layer_idx][relation_key](raw)
                )
            )
            raw = self.backbone.relation_scales[relation_key][layer_idx] * raw
            if self.config.relation_aggr == "gated_sum":
                gate = self.backbone.relation_gates[layer_idx][relation_key](
                    normalized_target, raw
                )
                contribution = gate * raw
            else:
                contribution = raw
            delta = self.backbone.residual_scales[PROTEIN][layer_idx] * contribution
            h = h + delta
            layer_sources.append(delta)

            for edge_type in self.backbone.incoming_target_relations:
                relation_sources.append(delta if edge_type == neighbor_relation else torch.zeros_like(delta))
                relation_names.append(
                    f"layer:{layer_idx + 1}|relation:{'-'.join(edge_type)}"
                )

        if self.config.source_mode == "layer":
            contexts = torch.stack(
                [self.hierarchy_adapter.projection(value) for value in layer_sources],
                dim=0,
            )
            generated_names = tuple(
                f"layer:{i + 1}|target:{PROTEIN}" for i in range(self.config.num_layers)
            )
        else:
            contexts = torch.stack(
                [self.hierarchy_adapter.projection(value) for value in relation_sources],
                dim=0,
            )
            generated_names = tuple(relation_names)
        if source_names is not None and tuple(source_names) != generated_names:
            raise ValueError(
                "external P-P source order does not match the encoded support graph"
            )
        source_names = generated_names
        hierarchy = NBSNeighborhoodHierarchy(
            final_context=self.hierarchy_adapter.projection(h),
            source_contexts=contexts,
            source_names=tuple(source_names),
        )
        if hierarchy.source_contexts.size(0) != source_count:
            raise RuntimeError("external P-P hierarchy has the wrong source count")
        hierarchy.validate()
        return hierarchy

    def score_external_candidates(
        self,
        encoded_support: ProteinGOEncodedGraph,
        query: ProteinGOQueryBatch,
        candidate_x: Tensor,
        *,
        neighbor_x: Optional[Tensor] = None,
        neighbor_edge_attr: Optional[Tensor] = None,
        neighbor_relation: EdgeType = SIMILAR_TO,
        neighbor_fanouts: Optional[Sequence[int]] = None,
        return_aux: bool = False,
    ) -> NBSMatchOutput:
        """Score external proteins against GO queries using full NBS queries.

        ``query.base_logits`` and optional ``query.candidate_evidence`` must be
        [Q,C] / [Q,C,F], where C equals ``candidate_x.size(0)``.  No expert
        probability is required or consumed in the default student mode.
        """
        if query.candidate_protein_index is not None:
            raise ValueError(
                "external candidate scoring expects candidate_protein_index=None"
            )
        condition = self.query_encoder(
            encoded_support.hierarchy.final_context,
            encoded_support.final_states[GO],
            encoded_support.local_go_box,
            query,
            global_cache=encoded_support.global_go_cache,
        )
        external_hierarchy = self.encode_external_protein_candidates(
            candidate_x,
            neighbor_x=neighbor_x,
            neighbor_edge_attr=neighbor_edge_attr,
            neighbor_relation=neighbor_relation,
            neighbor_fanouts=neighbor_fanouts,
            source_names=encoded_support.hierarchy.source_names,
        )
        output = self.matcher(
            external_hierarchy,
            condition,
            return_aux=return_aux,
            routing_hierarchy=encoded_support.hierarchy,
        )
        if return_aux and output.auxiliary is not None:
            output.auxiliary["inference/external_candidate_mode"] = output.logits.new_tensor(1.0)
            output.auxiliary["inference/separate_support_routing"] = output.logits.new_tensor(1.0)
            output.auxiliary["inference/external_pp_enabled"] = output.logits.new_tensor(
                float(neighbor_x is not None)
            )
            output.auxiliary["inference/external_source_norm"] = (
                external_hierarchy.source_contexts.float().norm(dim=-1)
            )
            output.auxiliary["inference/external_final_context_norm"] = (
                external_hierarchy.final_context.float().norm(dim=-1)
            )
        return output

    def score_encoded(
        self,
        encoded: ProteinGOEncodedGraph,
        query: ProteinGOQueryBatch,
        *,
        return_aux: bool = False,
    ) -> NBSMatchOutput:
        condition = self.query_encoder(
            encoded.hierarchy.final_context,
            encoded.final_states[GO],
            encoded.local_go_box,
            query,
            global_cache=encoded.global_go_cache,
        )
        output = self.matcher(encoded.hierarchy, condition, return_aux=return_aux)
        if return_aux and output.auxiliary is not None:
            if encoded.backbone_aux:
                output.auxiliary.update(
                    {f"backbone/{k}": v for k, v in encoded.backbone_aux.items()}
                )
            output.auxiliary.update(
                {
                    "box/calibration_threshold": self.box_calibrator.threshold,
                    "box/calibration_scale": self.box_calibrator.scale,
                    "box/local_hierarchy_gate_mean": encoded.local_go_box.hierarchy_gate.mean(),
                    "box/go_context_scale": torch.tanh(self.go_context_scale),
                }
            )
        return output

    def forward(
        self,
        graph: HeteroData,
        query: ProteinGOQueryBatch,
        *,
        global_go_cache: Optional[NBSGOBoxCache] = None,
        return_aux: bool = False,
    ) -> NBSMatchOutput:
        encoded = self.encode_graph(graph, global_go_cache=global_go_cache)
        return self.score_encoded(encoded, query, return_aux=return_aux)


class HomogeneousNBSModel(nn.Module):
    """Graph-type agnostic reference wrapper for the NBS-GDAR skeleton."""

    def __init__(self, config: NBSConfig, input_dim: int) -> None:
        super().__init__()
        self.backbone = NBSHomogeneousResidualBackbone(config, input_dim)
        self.adapter = NBSHierarchyAdapter(config.hidden_dim, use_projection=True)
        self.matcher = NBSGatedDeltaAttnRes(config, config.num_layers)

    def encode_graph(self, x: Tensor, edge_index: Tensor) -> NBSNeighborhoodHierarchy:
        return self.adapter.from_homogeneous(self.backbone(x, edge_index))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        condition: NBSQueryCondition,
        *,
        return_aux: bool = False,
    ) -> NBSMatchOutput:
        return self.matcher(self.encode_graph(x, edge_index), condition, return_aux=return_aux)
