"""Template for connecting LATENCE tensors to NBS."""

import torch

from nbs_pg import (
    NBSConfig,
    ProteinGONBSModel,
    build_nbs_protein_go_heterodata,
)

protein_embeddings = torch.load("protein_embeddings.pt", map_location="cpu")
go_center = torch.load("boxsqel_go_center.pt", map_location="cpu")
go_offset = torch.load("boxsqel_go_offset.pt", map_location="cpu")
go_is_a_edge_index = torch.load("go_is_a_edge_index.pt", map_location="cpu")

ppi_edge_index = torch.load("ppi_edge_index.pt", map_location="cpu")
similarity_edge_index = torch.load("protein_similarity_edge_index.pt", map_location="cpu")
weak_to_core_edge_index = torch.load("weak_to_core_edge_index.pt", map_location="cpu")
true_protein_go = torch.load("true_protein_go_edge_index.pt", map_location="cpu")
pseudo_protein_go = torch.load("pseudo_protein_go_edge_index.pt", map_location="cpu")
pseudo_confidence = torch.load("pseudo_protein_go_confidence.pt", map_location="cpu")

allowed_annotation_mask = torch.load(
    "annotation_allowed_protein_mask.pt", map_location="cpu"
)
blocked_target_mask = torch.load(
    "annotation_blocked_target_protein_mask.pt", map_location="cpu"
)

graph = build_nbs_protein_go_heterodata(
    protein_embeddings,
    go_center,
    go_offset,
    go_is_a_edge_index,
    ppi_edge_index=ppi_edge_index,
    similarity_edge_index=similarity_edge_index,
    weak_to_core_edge_index=weak_to_core_edge_index,
    true_protein_go_edge_index=true_protein_go,
    pseudo_protein_go_edge_index=pseudo_protein_go,
    pseudo_annotation_confidence=pseudo_confidence,
    allowed_annotation_protein_mask=allowed_annotation_mask,
    blocked_annotation_protein_mask=blocked_target_mask,
)

config = NBSConfig(
    hidden_dim=256,
    num_layers=3,
    go_tower_layers=2,
    source_mode="layer_relation",
    relation_aggr="gated_sum",
    score_chunk_size=32768,
)
model = ProteinGONBSModel(
    config,
    protein_input_dim=protein_embeddings.size(1),
    go_box_dim=go_center.size(1),
)

# Optional full-GO cache before protein-rooted NeighborLoader sampling:
# go_cache = model.make_go_cache(graph)
# output = model(sampled_graph, query, global_go_cache=go_cache, return_aux=True)
