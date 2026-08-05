from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

BackboneType = Literal["sage", "gat", "transformer"]
NormType = Literal["layer", "batch", "none"]
ActivationType = Literal["relu", "gelu", "leaky_relu", "silu", "tanh"]
RelationAggregation = Literal["sum", "mean", "gated_sum"]
SourceMode = Literal["layer", "layer_relation"]
GOPooling = Literal["attention", "mean"]
PrimaryLoss = Literal["asl", "bce"]
DeltaGateFeatureMode = Literal["student_candidate", "student_only", "legacy_expert"]


@dataclass
class NBSConfig:
    """Configuration for Neighborhood--BoxSquare (NBS).

    BoxSquaredEL supplies the nested GO boxes.  NBS places protein functional
    neighbourhoods inside those boxes and learns a controlled weak-to-strong
    graph residual over the frozen first-stage classifier.
    """

    hidden_dim: int = 256
    num_layers: int = 3
    backbone_type: BackboneType = "sage"
    num_heads: int = 4
    activation: ActivationType = "gelu"
    norm: NormType = "layer"
    input_dropout: float = 0.1
    relation_dropout: float = 0.1
    residual_dropout: float = 0.1

    target_node_type: str = "protein"
    relation_aggr: RelationAggregation = "gated_sum"
    source_mode: SourceMode = "layer_relation"
    residual_scale_init: float = 0.1
    relation_scale_init: float = 0.1
    relation_gate_bias_init: float = -1.0

    # BoxSquaredEL input encoding.
    box_log_eps: float = 1e-8
    go_stat_dim: int = 6
    go_hierarchy_gate_bias_init: float = -1.0
    go_tower_layers: int = 2
    go_tower_scale_init: float = 0.1
    use_go_tower: bool = True
    use_static_go_anchor: bool = True
    go_query_pool: GOPooling = "attention"

    # Calibrated box inclusion.  The threshold is an initialization, not a
    # hard ontology rule and should be fitted only on training/validation data.
    box_margin_threshold_init: float = -0.15
    box_calibration_scale_init: float = 10.0
    train_box_calibrator: bool = True

    # NBS-GDAR source routing.
    use_null_source: bool = True
    null_logit_init: float = 2.0
    route_temperature: float = 1.0
    use_context_routing: bool = True
    query_residual_init: float = 0.0
    context_residual_init: float = 0.0

    # Protein--GO query construction.
    external_query_dim: int = 0
    query_semantic_gate_bias_init: float = -1.0
    query_hierarchy_gate_bias_init: float = -2.0

    # Base-logit residual refinement.  The default gate is student-only: it may
    # use first-stage backbone candidate evidence, but not expert probabilities.
    use_base_logit_residual: bool = True
    graph_delta_scale_init: float = 0.0
    use_candidate_delta_gate: bool = True
    delta_gate_hidden_dim: int = 16
    delta_gate_bias_init: float = -2.0
    delta_gate_feature_mode: DeltaGateFeatureMode = "student_candidate"

    # Candidate scoring. None scores all candidates in one operation.
    score_chunk_size: Optional[int] = 65536

    # Actual LATENCE edge contracts.
    pp_edge_dim: int = 3
    annotation_edge_dim: int = 3
    candidate_edge_dim: int = 3
    go_box_edge_dim: int = 8
    go_topology_edge_dim: int = 2
    go_edge_dim: int = 10
    go_part_edge_dim: int = 2

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.route_temperature <= 0:
            raise ValueError("route_temperature must be positive")
        if self.external_query_dim < 0:
            raise ValueError("external_query_dim cannot be negative")
        if self.go_stat_dim < 0:
            raise ValueError("go_stat_dim cannot be negative")
        if self.go_tower_layers < 0:
            raise ValueError("go_tower_layers cannot be negative")
        if self.delta_gate_hidden_dim <= 0:
            raise ValueError("delta_gate_hidden_dim must be positive")
        if self.score_chunk_size is not None and self.score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive or None")
        if self.pp_edge_dim != 3:
            raise ValueError("pp_edge_dim must be 3 for [confidence,source_score,reciprocal_rank]")
        if self.annotation_edge_dim != 3:
            raise ValueError("annotation_edge_dim must be 3 for [confidence,gold,pseudo]")
        if self.candidate_edge_dim != 3:
            raise ValueError(
                "candidate_edge_dim must be 3 for "
                "[backbone_probability,selector_score,reciprocal_rank]"
            )
        if self.go_box_edge_dim != 8:
            raise ValueError("go_box_edge_dim must be 8 for BoxSquaredEL geometry")
        if self.go_topology_edge_dim != 2:
            raise ValueError("go_topology_edge_dim must be 2 for [inverse_hop,is_direct]")
        if self.go_edge_dim != self.go_box_edge_dim + self.go_topology_edge_dim:
            raise ValueError("go_edge_dim must equal go_box_edge_dim + go_topology_edge_dim")
        if self.go_part_edge_dim != self.go_topology_edge_dim:
            raise ValueError("go_part_edge_dim must equal go_topology_edge_dim")
        if self.delta_gate_feature_mode not in {"student_candidate", "student_only", "legacy_expert"}:
            raise ValueError("unsupported delta_gate_feature_mode")
