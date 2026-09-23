from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import coalesce
    PYG_AVAILABLE = True
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "torch_geometric":
        raise
    HeteroData = object  # type: ignore[misc,assignment]
    coalesce = None
    PYG_AVAILABLE = False

from .box_geometry import (
    append_go_topology_features,
    build_go_box_edge_features,
    build_go_topology_edge_features,
    default_annotation_edge_features,
    default_candidate_edge_features,
)
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
    PPI,
    PROTEIN,
    PSEUDO_GO_TO_PROTEIN,
    PSEUDO_PROTEIN_TO_GO,
    SIMILAR_TO,
    WEAK_TO_CORE,
    EdgeType,
)

MaskMode = Literal["none", "all", "query_only"]


def empty_edge_index(device: Optional[torch.device] = None) -> Tensor:
    return torch.empty(2, 0, dtype=torch.long, device=device)


def reverse_edge_index(edge_index: Tensor) -> Tensor:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be [2, E]")
    return edge_index.flip(0)


def make_undirected(edge_index: Tensor, num_nodes: int) -> Tensor:
    if coalesce is None:
        raise ModuleNotFoundError("torch_geometric is required for graph construction")
    both = torch.cat([edge_index, reverse_edge_index(edge_index)], dim=1)
    return coalesce(both, num_nodes=num_nodes)


def _filter_edges(
    edge_index: Tensor,
    keep: Tensor,
    edge_attr: Optional[Tensor],
) -> tuple[Tensor, Optional[Tensor]]:
    edge_index = edge_index[:, keep]
    if edge_attr is not None:
        edge_attr = edge_attr[keep]
    return edge_index, edge_attr


def filter_annotation_edges_by_protein_mask(
    protein_to_go: Tensor,
    allowed_protein_mask: Tensor,
    *,
    blocked_protein_mask: Optional[Tensor] = None,
    edge_attr: Optional[Tensor] = None,
) -> tuple[Tensor, Optional[Tensor]]:
    """Retain annotation evidence only for allowed, non-target proteins."""
    if allowed_protein_mask.dtype != torch.bool:
        raise TypeError("allowed_protein_mask must be boolean")
    keep = allowed_protein_mask[protein_to_go[0]]
    if blocked_protein_mask is not None:
        if blocked_protein_mask.dtype != torch.bool:
            raise TypeError("blocked_protein_mask must be boolean")
        keep = keep & ~blocked_protein_mask[protein_to_go[0]]
    return _filter_edges(protein_to_go, keep, edge_attr)


def build_go_box_statistics(
    offset: Tensor,
    *,
    depth: Optional[Tensor] = None,
    descendant_count: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Six stable hierarchy features derived from BoxSquaredEL and GO topology."""
    if offset.dim() != 2:
        raise ValueError("offset must be [G, D]")
    log_offset = torch.log(offset.abs().clamp_min(eps))
    mean_log = log_offset.mean(dim=-1)
    std_log = log_offset.std(dim=-1, unbiased=False)
    mean_log_volume = torch.log((2.0 * offset.abs()).clamp_min(eps)).mean(dim=-1)
    max_offset = offset.abs().amax(dim=-1)
    if depth is None:
        depth_feature = torch.zeros_like(mean_log)
    else:
        depth_feature = depth.to(mean_log.dtype)
        depth_feature = depth_feature / depth_feature.max().clamp_min(1.0)
    if descendant_count is None:
        descendant_feature = torch.zeros_like(mean_log)
    else:
        descendant_feature = torch.log1p(descendant_count.to(mean_log.dtype))
        descendant_feature = descendant_feature / descendant_feature.max().clamp_min(1.0)
    return torch.stack(
        [
            mean_log,
            std_log,
            mean_log_volume,
            max_offset,
            depth_feature,
            descendant_feature,
        ],
        dim=-1,
    )


def _set_edge_store(
    data: HeteroData,
    edge_type: EdgeType,
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    *,
    topology_attr: Optional[Tensor] = None,
) -> None:
    data[edge_type].edge_index = edge_index
    if edge_attr is not None:
        if edge_attr.size(0) != edge_index.size(1):
            raise ValueError(f"edge_attr for {edge_type} does not align with edge_index")
        data[edge_type].edge_attr = edge_attr
    if topology_attr is not None:
        if topology_attr.size(0) != edge_index.size(1):
            raise ValueError(f"topology_attr for {edge_type} does not align with edge_index")
        data[edge_type].topology_attr = topology_attr


def _default_topology(
    edge_index: Tensor,
    raw_topology: Optional[Tensor],
    *,
    dtype: torch.dtype,
) -> Tensor:
    return build_go_topology_edge_features(
        edge_index.size(1),
        topology=raw_topology,
        device=edge_index.device,
        dtype=dtype,
    )


def build_nbs_protein_go_heterodata(
    protein_x: Tensor,
    go_center: Tensor,
    go_offset: Tensor,
    go_is_a_edge_index: Tensor,
    *,
    go_part_of_edge_index: Optional[Tensor] = None,
    ppi_edge_index: Optional[Tensor] = None,
    similarity_edge_index: Optional[Tensor] = None,
    weak_to_core_edge_index: Optional[Tensor] = None,
    gold_protein_go_edge_index: Optional[Tensor] = None,
    backbone_candidate_protein_go_edge_index: Optional[Tensor] = None,
    pseudo_protein_go_edge_index: Optional[Tensor] = None,
    # Backwards-compatible argument name.
    true_protein_go_edge_index: Optional[Tensor] = None,
    allowed_annotation_protein_mask: Optional[Tensor] = None,
    blocked_annotation_protein_mask: Optional[Tensor] = None,
    make_ppi_symmetric: bool = False,
    make_similarity_symmetric: bool = False,
    ppi_edge_attr: Optional[Tensor] = None,
    similarity_edge_attr: Optional[Tensor] = None,
    weak_to_core_edge_attr: Optional[Tensor] = None,
    gold_annotation_edge_attr: Optional[Tensor] = None,
    backbone_candidate_edge_attr: Optional[Tensor] = None,
    pseudo_annotation_edge_attr: Optional[Tensor] = None,
    # Backwards-compatible argument name.
    true_annotation_edge_attr: Optional[Tensor] = None,
    pseudo_annotation_confidence: Optional[Tensor] = None,
    go_is_a_edge_attr: Optional[Tensor] = None,
    go_has_child_edge_attr: Optional[Tensor] = None,
    go_part_of_edge_attr: Optional[Tensor] = None,
    go_has_part_edge_attr: Optional[Tensor] = None,
    # Raw LATENCE topology columns: [hop_distance, is_direct].
    go_is_a_topology: Optional[Tensor] = None,
    go_part_of_topology: Optional[Tensor] = None,
    go_stats: Optional[Tensor] = None,
    go_depth: Optional[Tensor] = None,
    go_descendant_count: Optional[Tensor] = None,
    protein_ids: Optional[Tensor] = None,
    go_ids: Optional[Tensor] = None,
) -> HeteroData:
    """Build a local, evidence-separated NBS protein--GO graph.

    This constructor is intended for toy graphs and sampled subgraphs.  The
    full LATENCE graph, especially the 281,457,664 backbone-candidate edges,
    must remain in mmap/CSR stores and be materialized only for a local batch.
    """
    if go_center.shape != go_offset.shape or go_center.dim() != 2:
        raise ValueError("go_center and go_offset must be aligned [G, box_dim]")
    if gold_protein_go_edge_index is not None and true_protein_go_edge_index is not None:
        raise ValueError("provide gold_protein_go_edge_index or true_protein_go_edge_index, not both")
    if gold_annotation_edge_attr is not None and true_annotation_edge_attr is not None:
        raise ValueError("provide gold_annotation_edge_attr or true_annotation_edge_attr, not both")
    gold_protein_go_edge_index = (
        true_protein_go_edge_index
        if gold_protein_go_edge_index is None
        else gold_protein_go_edge_index
    )
    gold_annotation_edge_attr = (
        true_annotation_edge_attr
        if gold_annotation_edge_attr is None
        else gold_annotation_edge_attr
    )

    device = protein_x.device
    ppi_edge_index = empty_edge_index(device) if ppi_edge_index is None else ppi_edge_index
    similarity_edge_index = (
        empty_edge_index(device) if similarity_edge_index is None else similarity_edge_index
    )
    weak_to_core_edge_index = (
        empty_edge_index(device) if weak_to_core_edge_index is None else weak_to_core_edge_index
    )
    gold_protein_go_edge_index = (
        empty_edge_index(device)
        if gold_protein_go_edge_index is None
        else gold_protein_go_edge_index
    )
    backbone_candidate_protein_go_edge_index = (
        empty_edge_index(device)
        if backbone_candidate_protein_go_edge_index is None
        else backbone_candidate_protein_go_edge_index
    )
    pseudo_protein_go_edge_index = (
        empty_edge_index(device)
        if pseudo_protein_go_edge_index is None
        else pseudo_protein_go_edge_index
    )
    go_part_of_edge_index = (
        empty_edge_index(go_center.device)
        if go_part_of_edge_index is None
        else go_part_of_edge_index
    )

    if ppi_edge_attr is None:
        ppi_edge_attr = default_candidate_edge_features(
            ppi_edge_index.size(1), device=device, dtype=protein_x.dtype
        )
    if similarity_edge_attr is None:
        similarity_edge_attr = default_candidate_edge_features(
            similarity_edge_index.size(1), device=device, dtype=protein_x.dtype
        )
    if weak_to_core_edge_attr is None:
        weak_to_core_edge_attr = default_candidate_edge_features(
            weak_to_core_edge_index.size(1), device=device, dtype=protein_x.dtype
        )

    if gold_annotation_edge_attr is None:
        gold_annotation_edge_attr = default_annotation_edge_features(
            gold_protein_go_edge_index.size(1),
            is_pseudo=False,
            device=device,
            dtype=protein_x.dtype,
        )
    if pseudo_annotation_edge_attr is None:
        pseudo_annotation_edge_attr = default_annotation_edge_features(
            pseudo_protein_go_edge_index.size(1),
            confidence=pseudo_annotation_confidence,
            is_pseudo=True,
            device=device,
            dtype=protein_x.dtype,
        )
    if backbone_candidate_edge_attr is None:
        backbone_candidate_edge_attr = default_candidate_edge_features(
            backbone_candidate_protein_go_edge_index.size(1),
            device=device,
            dtype=protein_x.dtype,
        )

    # Gold and pseudo edges are supervision-derived annotation evidence and are
    # subject to split protection.  Backbone candidate edges are first-stage
    # student evidence and are masked episode-wise instead.
    if allowed_annotation_protein_mask is not None:
        gold_protein_go_edge_index, gold_annotation_edge_attr = (
            filter_annotation_edges_by_protein_mask(
                gold_protein_go_edge_index,
                allowed_annotation_protein_mask,
                blocked_protein_mask=blocked_annotation_protein_mask,
                edge_attr=gold_annotation_edge_attr,
            )
        )
        pseudo_protein_go_edge_index, pseudo_annotation_edge_attr = (
            filter_annotation_edges_by_protein_mask(
                pseudo_protein_go_edge_index,
                allowed_annotation_protein_mask,
                blocked_protein_mask=blocked_annotation_protein_mask,
                edge_attr=pseudo_annotation_edge_attr,
            )
        )
    elif blocked_annotation_protein_mask is not None:
        allowed = torch.ones(
            protein_x.size(0), dtype=torch.bool, device=blocked_annotation_protein_mask.device
        )
        gold_protein_go_edge_index, gold_annotation_edge_attr = (
            filter_annotation_edges_by_protein_mask(
                gold_protein_go_edge_index,
                allowed,
                blocked_protein_mask=blocked_annotation_protein_mask,
                edge_attr=gold_annotation_edge_attr,
            )
        )
        pseudo_protein_go_edge_index, pseudo_annotation_edge_attr = (
            filter_annotation_edges_by_protein_mask(
                pseudo_protein_go_edge_index,
                allowed,
                blocked_protein_mask=blocked_annotation_protein_mask,
                edge_attr=pseudo_annotation_edge_attr,
            )
        )

    # The upstream PPI builder already emits bidirectional message edges.  The
    # defaults are therefore False to avoid silently duplicating 5.4M edges.
    if make_ppi_symmetric and ppi_edge_index.numel() > 0:
        if ppi_edge_attr is None:
            ppi_edge_index = make_undirected(ppi_edge_index, protein_x.size(0))
        else:
            ppi_edge_index = torch.cat([ppi_edge_index, reverse_edge_index(ppi_edge_index)], dim=1)
            ppi_edge_attr = torch.cat([ppi_edge_attr, ppi_edge_attr], dim=0)
    if make_similarity_symmetric and similarity_edge_index.numel() > 0:
        if similarity_edge_attr is None:
            similarity_edge_index = make_undirected(similarity_edge_index, protein_x.size(0))
        else:
            similarity_edge_index = torch.cat(
                [similarity_edge_index, reverse_edge_index(similarity_edge_index)], dim=1
            )
            similarity_edge_attr = torch.cat([similarity_edge_attr, similarity_edge_attr], dim=0)

    if go_stats is None:
        go_stats = build_go_box_statistics(
            go_offset, depth=go_depth, descendant_count=go_descendant_count
        )

    is_a_topology_attr = _default_topology(
        go_is_a_edge_index, go_is_a_topology, dtype=go_center.dtype
    )
    part_of_topology_attr = _default_topology(
        go_part_of_edge_index, go_part_of_topology, dtype=go_center.dtype
    )
    if go_is_a_edge_attr is None:
        go_is_a_edge_attr = append_go_topology_features(
            build_go_box_edge_features(
                go_center, go_offset, go_is_a_edge_index, direction=1.0, depth=go_depth
            ),
            is_a_topology_attr,
        )
    if go_has_child_edge_attr is None:
        go_has_child_edge_attr = append_go_topology_features(
            build_go_box_edge_features(
                go_center, go_offset, go_is_a_edge_index, direction=-1.0, depth=go_depth
            ),
            is_a_topology_attr,
        )
    if go_part_of_edge_attr is None:
        go_part_of_edge_attr = part_of_topology_attr
    if go_has_part_edge_attr is None:
        go_has_part_edge_attr = part_of_topology_attr

    data = HeteroData()
    data[PROTEIN].x = protein_x
    data[PROTEIN].num_nodes = protein_x.size(0)
    data[GO].center = go_center
    data[GO].offset = go_offset.abs()
    data[GO].stats = go_stats
    data[GO].num_nodes = go_center.size(0)
    if protein_ids is not None:
        data[PROTEIN].node_id = protein_ids
    if go_ids is not None:
        data[GO].node_id = go_ids

    _set_edge_store(data, PPI, ppi_edge_index, ppi_edge_attr)
    _set_edge_store(data, SIMILAR_TO, similarity_edge_index, similarity_edge_attr)
    _set_edge_store(data, WEAK_TO_CORE, weak_to_core_edge_index, weak_to_core_edge_attr)
    _set_edge_store(
        data, GOLD_PROTEIN_TO_GO, gold_protein_go_edge_index, gold_annotation_edge_attr
    )
    _set_edge_store(
        data,
        GOLD_GO_TO_PROTEIN,
        reverse_edge_index(gold_protein_go_edge_index),
        gold_annotation_edge_attr,
    )
    _set_edge_store(
        data,
        BACKBONE_CANDIDATE_PROTEIN_TO_GO,
        backbone_candidate_protein_go_edge_index,
        backbone_candidate_edge_attr,
    )
    _set_edge_store(
        data,
        BACKBONE_CANDIDATE_GO_TO_PROTEIN,
        reverse_edge_index(backbone_candidate_protein_go_edge_index),
        backbone_candidate_edge_attr,
    )
    _set_edge_store(
        data,
        PSEUDO_PROTEIN_TO_GO,
        pseudo_protein_go_edge_index,
        pseudo_annotation_edge_attr,
    )
    _set_edge_store(
        data,
        PSEUDO_GO_TO_PROTEIN,
        reverse_edge_index(pseudo_protein_go_edge_index),
        pseudo_annotation_edge_attr,
    )
    _set_edge_store(
        data,
        GO_IS_A,
        go_is_a_edge_index,
        go_is_a_edge_attr,
        topology_attr=is_a_topology_attr,
    )
    _set_edge_store(
        data,
        GO_HAS_CHILD,
        reverse_edge_index(go_is_a_edge_index),
        go_has_child_edge_attr,
        topology_attr=is_a_topology_attr,
    )
    _set_edge_store(
        data,
        GO_PART_OF,
        go_part_of_edge_index,
        go_part_of_edge_attr,
        topology_attr=part_of_topology_attr,
    )
    _set_edge_store(
        data,
        GO_HAS_PART,
        reverse_edge_index(go_part_of_edge_index),
        go_has_part_edge_attr,
        topology_attr=part_of_topology_attr,
    )
    data.validate(raise_on_error=True)
    return data


def _mask_evidence_relation(
    data: HeteroData,
    forward: EdgeType,
    reverse: EdgeType,
    candidate_mask: Tensor,
    query_mask: Optional[Tensor],
    mode: MaskMode,
) -> None:
    if mode == "none":
        return
    edge = data[forward].edge_index
    if mode == "all":
        keep = ~candidate_mask[edge[0]]
    elif mode == "query_only":
        if query_mask is None:
            raise ValueError("query_go_index is required for query_only masking")
        keep = ~(candidate_mask[edge[0]] & query_mask[edge[1]])
    else:  # pragma: no cover - type checker plus caller validation
        raise ValueError(f"unsupported mask mode: {mode}")
    attr = getattr(data[forward], "edge_attr", None)
    edge, attr = _filter_edges(edge, keep, attr)
    data[forward].edge_index = edge
    data[reverse].edge_index = reverse_edge_index(edge)
    if attr is not None:
        data[forward].edge_attr = attr
        data[reverse].edge_attr = attr


def mask_candidate_evidence_edges(
    data: HeteroData,
    candidate_protein_index: Tensor,
    *,
    query_go_index: Optional[Tensor] = None,
    gold_mode: MaskMode = "all",
    pseudo_mode: MaskMode = "all",
    candidate_mode: MaskMode = "query_only",
    inplace: bool = False,
) -> HeteroData:
    """Remove target leakage with relation-specific policies.

    Gold and pseudo annotations are removed for candidate proteins by default.
    The backbone relation removes only candidate--query-GO pairs, preserving
    other first-stage candidate evidence for the same protein.
    """
    valid_modes = {"none", "all", "query_only"}
    for name, value in (
        ("gold_mode", gold_mode),
        ("pseudo_mode", pseudo_mode),
        ("candidate_mode", candidate_mode),
    ):
        if value not in valid_modes:
            raise ValueError(f"{name} must be one of {sorted(valid_modes)}")
    out = data if inplace else data.clone()
    candidate_mask = torch.zeros(
        out[PROTEIN].num_nodes,
        dtype=torch.bool,
        device=candidate_protein_index.device,
    )
    candidate_mask[candidate_protein_index] = True
    query_mask = None
    if "query_only" in {gold_mode, pseudo_mode, candidate_mode}:
        if query_go_index is None:
            raise ValueError("query_go_index is required by a query_only mask")
        query_mask = torch.zeros(
            out[GO].num_nodes, dtype=torch.bool, device=query_go_index.device
        )
        query_mask[query_go_index] = True

    _mask_evidence_relation(
        out,
        GOLD_PROTEIN_TO_GO,
        GOLD_GO_TO_PROTEIN,
        candidate_mask,
        query_mask,
        gold_mode,
    )
    _mask_evidence_relation(
        out,
        PSEUDO_PROTEIN_TO_GO,
        PSEUDO_GO_TO_PROTEIN,
        candidate_mask,
        query_mask,
        pseudo_mode,
    )
    _mask_evidence_relation(
        out,
        BACKBONE_CANDIDATE_PROTEIN_TO_GO,
        BACKBONE_CANDIDATE_GO_TO_PROTEIN,
        candidate_mask,
        query_mask,
        candidate_mode,
    )
    out.validate(raise_on_error=True)
    return out


def mask_candidate_annotation_edges(
    data: HeteroData,
    candidate_protein_index: Tensor,
    *,
    query_go_index: Optional[Tensor] = None,
    mode: str = "all",
    include_pseudo: bool = True,
    inplace: bool = False,
) -> HeteroData:
    """Compatibility wrapper for the original gold/pseudo-only API."""
    if mode not in {"all", "query_only"}:
        raise ValueError("mode must be 'all' or 'query_only'")
    return mask_candidate_evidence_edges(
        data,
        candidate_protein_index,
        query_go_index=query_go_index,
        gold_mode=mode,  # type: ignore[arg-type]
        pseudo_mode=mode if include_pseudo else "none",  # type: ignore[arg-type]
        candidate_mode="none",
        inplace=inplace,
    )


# Compatibility name for callers migrating from NGH. The signature is now NBS.
build_protein_go_heterodata = build_nbs_protein_go_heterodata
