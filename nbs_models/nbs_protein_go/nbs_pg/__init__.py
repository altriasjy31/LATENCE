"""Neighborhood--BoxSquare (NBS) with BoxSquaredEL protein--GO geometry."""

__version__ = "0.3.1"

from .boxsqel_manifest import (
    BoxSquaredELTrainingContract,
    align_boxsqel_to_go_registry,
    load_boxsqel_training_contract,
    normalize_go_identifier,
)
from .box_geometry import (
    ANNOTATION_EDGE_FEATURE_DIM,
    CANDIDATE_EDGE_FEATURE_DIM,
    GO_BOX_EDGE_FEATURE_DIM,
    GO_ONTOLOGY_EDGE_FEATURE_DIM,
    GO_TOPOLOGY_EDGE_FEATURE_DIM,
    BoxHierarchyCalibrator,
    box_pair_metrics,
    build_go_box_edge_features,
)
from .config import NBSConfig
from .go_encoder import BoxSquaredGOEncoder
from .inverted_index import (
    InvertedIndexFiles,
    build_go_inverted_from_edge_index,
    build_go_inverted_from_protein_csr,
    build_latence_go_protein_indices,
    load_role_global_indices,
)
from .losses import (
    NBSLossWeights,
    hierarchy_violation_loss,
    query_hierarchy_violation_loss,
    masked_asl_logits,
    masked_bce_with_logits,
    nbs_training_loss,
)
from .matcher import NBSGatedDeltaAttnRes
from .episode import GOQueryEpisodeSampler, NBSGlobalEpisode, NBSQueryEpisodeConfig
from .latence_stores import (
    FixedDegreeCandidateAttributeStore,
    GOBoxStore,
    GOProteinCSRStore,
    RoleAwareBaseLogitStore,
    RoleProbabilitySlice,
    load_go_protein_stores,
)
from .training import (
    NBSFixedEpochTrainer,
    NBSFixedEpochTrainingConfig,
    NBSLocalBatch,
    NBSLossConfig,
    NBSRunComponents,
    freeze_go_geometry,
)
from .schema import *
from .types import (
    BoxGOEncoding,
    NBSGOBoxCache,
    NBSMatchOutput,
    NBSNeighborhoodHierarchy,
    NBSQueryCondition,
    ProteinGOEncodedGraph,
    ProteinGOQueryBatch,
)

# PyG-dependent imports remain optional so geometry and matcher tests can run in
# lightweight environments. Production training requires torch-geometric.
try:  # pragma: no cover
    from .data import (
        build_go_box_statistics,
        build_nbs_protein_go_heterodata,
        build_protein_go_heterodata,
        filter_annotation_edges_by_protein_mask,
        mask_candidate_annotation_edges,
        mask_candidate_evidence_edges,
    )
    from .model import HomogeneousNBSModel, ProteinGONBSModel
    from .sampling import build_protein_neighbor_loader, cache_to_device, remap_query_to_sampled_graph
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "torch_geometric":
        raise

__all__ = [
    "NBSConfig",
    "BoxSquaredELTrainingContract",
    "load_boxsqel_training_contract",
    "align_boxsqel_to_go_registry",
    "normalize_go_identifier",
    "NBSFixedEpochTrainer",
    "NBSFixedEpochTrainingConfig",
    "NBSLocalBatch",
    "NBSLossConfig",
    "NBSRunComponents",
    "freeze_go_geometry",
    "GOProteinCSRStore",
    "FixedDegreeCandidateAttributeStore",
    "RoleProbabilitySlice",
    "RoleAwareBaseLogitStore",
    "GOBoxStore",
    "load_go_protein_stores",
    "NBSQueryEpisodeConfig",
    "NBSGlobalEpisode",
    "GOQueryEpisodeSampler",
    "InvertedIndexFiles",
    "build_go_inverted_from_edge_index",
    "build_go_inverted_from_protein_csr",
    "build_latence_go_protein_indices",
    "load_role_global_indices",
    "NBSLossWeights",
    "NBSGatedDeltaAttnRes",
    "NBSNeighborhoodHierarchy",
    "NBSQueryCondition",
    "NBSMatchOutput",
    "NBSGOBoxCache",
    "BoxGOEncoding",
    "ProteinGOQueryBatch",
    "ProteinGOEncodedGraph",
    "BoxSquaredGOEncoder",
    "BoxHierarchyCalibrator",
    "box_pair_metrics",
    "build_go_box_edge_features",
    "nbs_training_loss",
    "masked_asl_logits",
    "masked_bce_with_logits",
    "hierarchy_violation_loss",
    "query_hierarchy_violation_loss",
    "GO_BOX_EDGE_FEATURE_DIM",
    "GO_TOPOLOGY_EDGE_FEATURE_DIM",
    "GO_ONTOLOGY_EDGE_FEATURE_DIM",
    "ANNOTATION_EDGE_FEATURE_DIM",
    "CANDIDATE_EDGE_FEATURE_DIM",
    "PROTEIN",
    "GO",
    "PPI",
    "SIMILAR_TO",
    "WEAK_TO_CORE",
    "GOLD_PROTEIN_TO_GO",
    "GOLD_GO_TO_PROTEIN",
    "BACKBONE_CANDIDATE_PROTEIN_TO_GO",
    "BACKBONE_CANDIDATE_GO_TO_PROTEIN",
    "TRUE_PROTEIN_TO_GO",
    "TRUE_GO_TO_PROTEIN",
    "PSEUDO_PROTEIN_TO_GO",
    "PSEUDO_GO_TO_PROTEIN",
    "GO_IS_A",
    "GO_HAS_CHILD",
    "GO_PART_OF",
    "GO_HAS_PART",
    "NBS_PROTEIN_GO_METADATA",
]

for _name in (
    "ProteinGONBSModel",
    "HomogeneousNBSModel",
    "build_nbs_protein_go_heterodata",
    "build_protein_go_heterodata",
    "filter_annotation_edges_by_protein_mask",
    "mask_candidate_annotation_edges",
    "mask_candidate_evidence_edges",
    "build_go_box_statistics",
    "build_protein_neighbor_loader",
    "remap_query_to_sampled_graph",
    "cache_to_device",
):
    if _name in globals():
        __all__.append(_name)
