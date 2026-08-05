from __future__ import annotations

from typing import Mapping, Optional

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
