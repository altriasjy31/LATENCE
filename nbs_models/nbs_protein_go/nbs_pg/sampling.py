from __future__ import annotations

from typing import Literal, Mapping, Optional

import torch
from torch import Tensor

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import NeighborLoader
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "torch_geometric":
        raise
    HeteroData = NeighborLoader = object  # type: ignore[misc,assignment]

from .schema import EdgeType, GO, PROTEIN
from .types import NBSGOBoxCache, ProteinGOQueryBatch


def build_protein_neighbor_loader(
    graph: HeteroData,
    input_proteins: Tensor,
    *,
    num_neighbors: Mapping[EdgeType, list[int]] | list[int],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    subgraph_type: Literal["directional", "bidirectional", "induced"] = "directional",
) -> NeighborLoader:
    """Protein-rooted loader with direction-safe sampling by default.

    ``weak_to_core`` is a directed evidence relation.  Therefore the default is
    ``directional``; callers must opt in explicitly before creating a
    bidirectional sampled subgraph.
    """
    return NeighborLoader(
        graph,
        input_nodes=(PROTEIN, input_proteins),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        subgraph_type=subgraph_type,
    )


def global_to_local(global_ids: Tensor, local_n_id: Tensor) -> Tensor:
    if local_n_id.numel() == 0:
        raise KeyError("Cannot map IDs into an empty sampled-node store")
    order = torch.argsort(local_n_id)
    sorted_ids = local_n_id[order]
    pos = torch.searchsorted(sorted_ids, global_ids)
    valid = (pos < sorted_ids.numel()) & (
        sorted_ids[pos.clamp_max(sorted_ids.numel() - 1)] == global_ids
    )
    if not bool(valid.all()):
        missing = global_ids[~valid]
        raise KeyError(f"Sampled subgraph is missing global IDs: {missing[:10].tolist()}")
    return order[pos]


def remap_query_to_sampled_graph(
    query: ProteinGOQueryBatch,
    sampled_graph: HeteroData,
    *,
    global_seed_proteins: Tensor,
    global_query_go: Optional[Tensor] = None,
    global_candidates: Optional[Tensor] = None,
    allow_global_go_fallback: bool = True,
) -> ProteinGOQueryBatch:
    """Remap an episode after PyG relabeling.

    If a protein-rooted sample omits query GO nodes, the global IDs are retained
    for ``NBSGOBoxCache`` lookup instead of failing the episode.
    """
    protein_n_id = sampled_graph[PROTEIN].n_id
    seed_local = global_to_local(global_seed_proteins, protein_n_id)
    candidate_local = (
        None
        if global_candidates is None
        else global_to_local(global_candidates, protein_n_id)
    )

    go_local = None
    go_global = query.query_go_global_index
    if global_query_go is not None:
        go_global = global_query_go
        go_n_id = sampled_graph[GO].n_id
        try:
            go_local = global_to_local(global_query_go, go_n_id)
            go_global = None
        except KeyError:
            if not allow_global_go_fallback:
                raise
            go_local = None

    return ProteinGOQueryBatch(
        seed_protein_index=seed_local,
        seed_query_index=query.seed_query_index,
        num_queries=query.num_queries,
        query_go_index=go_local,
        query_go_global_index=go_global,
        go_query_index=query.go_query_index,
        external_query_features=query.external_query_features,
        candidate_protein_index=candidate_local,
        base_logits=query.base_logits,
        candidate_evidence=query.candidate_evidence,
        expert_prob=query.expert_prob,
        query_go_frequency=query.query_go_frequency,
        labels=query.labels,
        mask=query.mask,
        confidence=query.confidence,
        pseudo_mask=query.pseudo_mask,
    )


def cache_to_device(cache: NBSGOBoxCache, device: torch.device | str) -> NBSGOBoxCache:
    return NBSGOBoxCache(
        semantic=cache.semantic.to(device),
        hierarchy=cache.hierarchy.to(device),
        static=cache.static.to(device),
        context=cache.context.to(device),
        center=cache.center.to(device),
        offset=cache.offset.to(device),
        stats=None if cache.stats is None else cache.stats.to(device),
    )
