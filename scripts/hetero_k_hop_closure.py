#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HGAT/NBS-style k-hop node closure for a DGL heterograph.

This is the distinctly named v2 helper paired with
``build_pp_edge_types_v2.py``.

This module does not materialize transitive edges.  It only finds the parent
graph node IDs required by a seed batch, preserving the original evidence
relations for ``source_mode="layer_relation"`` routing.

For the intended LATENCE route

    weak --weak_to_core--> core --annotated_with--> GO

start from GO seed nodes, use ``k=2`` and ``edge_dir="in"``.  The first hop
collects annotated proteins and the second hop can collect weak proteins that
send messages to those core proteins.

Candidate protein--GO edges that would leak evaluation labels must be removed
or masked in ``full_g`` before calling this function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, Tuple


CanonicalEType = Tuple[str, str, str]
__version__ = "2.1.0-weak-to-core-closure"


def _load_dgl_torch():
    try:
        import dgl
        import torch
    except ImportError as exc:  # pragma: no cover - depends on training env
        raise ImportError(
            "hetero_k_hop_closure_v2 requires the training environment with "
            "PyTorch and DGL installed"
        ) from exc
    return dgl, torch


def _canonical_etype_set(
    full_g: Any,
    allowed_etypes: Iterable[str | CanonicalEType] | None,
) -> set[CanonicalEType] | None:
    if allowed_etypes is None:
        return None

    canonical = list(full_g.canonical_etypes)
    by_relation: Dict[str, list[CanonicalEType]] = {}
    for etype in canonical:
        by_relation.setdefault(etype[1], []).append(etype)

    selected: set[CanonicalEType] = set()
    unknown: list[str] = []
    for requested in allowed_etypes:
        if isinstance(requested, str):
            matches = by_relation.get(requested, [])
            if not matches:
                unknown.append(requested)
            selected.update(matches)
        else:
            normalized = tuple(requested)
            if len(normalized) != 3 or normalized not in canonical:
                unknown.append(str(requested))
            else:
                selected.add(normalized)
    if unknown:
        raise KeyError(
            f"Unknown allowed_etypes={unknown}; "
            f"available={list(full_g.canonical_etypes)}"
        )
    return selected


def _resolve_fanout(
    full_g: Any,
    fanout: int | Mapping[str | CanonicalEType, int],
    allowed_etypes: Iterable[str | CanonicalEType] | None,
) -> int | Dict[CanonicalEType, int]:
    selected = _canonical_etype_set(full_g, allowed_etypes)
    if isinstance(fanout, int):
        if fanout < -1:
            raise ValueError("fanout must be -1 or a non-negative integer")
        if selected is None:
            return fanout
        return {
            etype: fanout if etype in selected else 0
            for etype in full_g.canonical_etypes
        }

    canonical = list(full_g.canonical_etypes)
    by_relation: Dict[str, list[CanonicalEType]] = {}
    for etype in canonical:
        by_relation.setdefault(etype[1], []).append(etype)
    resolved: Dict[CanonicalEType, int] = {etype: 0 for etype in canonical}
    unknown: list[str] = []
    for key, value in fanout.items():
        value = int(value)
        if value < -1:
            raise ValueError(
                f"fanout for {key!r} must be -1 or non-negative, got {value}"
            )
        if isinstance(key, str):
            matches = by_relation.get(key, [])
            if not matches:
                unknown.append(key)
                continue
        else:
            normalized = tuple(key)
            matches = [normalized] if normalized in canonical else []
            if not matches:
                unknown.append(str(key))
                continue
        for etype in matches:
            if selected is None or etype in selected:
                resolved[etype] = value
    if unknown:
        raise KeyError(
            f"Unknown fanout relation keys={unknown}; available={canonical}"
        )
    return resolved


def _normalize_seed_nodes(
    full_g: Any,
    seed_nodes_parent: Mapping[str, Any],
    torch: Any,
) -> Dict[str, Any]:
    unknown = sorted(set(seed_nodes_parent) - set(full_g.ntypes))
    if unknown:
        raise KeyError(f"Unknown seed node types={unknown}; available={full_g.ntypes}")

    normalized: Dict[str, Any] = {}
    for ntype, ids in seed_nodes_parent.items():
        tensor = torch.as_tensor(
            ids, dtype=full_g.idtype, device=full_g.device
        ).reshape(-1)
        if tensor.numel() == 0:
            continue
        if bool(torch.any(tensor < 0)) or bool(
            torch.any(tensor >= full_g.num_nodes(ntype))
        ):
            raise IndexError(
                f"Seed parent IDs for {ntype!r} fall outside "
                f"[0, {full_g.num_nodes(ntype)})"
            )
        normalized[ntype] = torch.unique(tensor)
    return normalized


def _sample_parent_node_ids(
    full_g: Any,
    frontier: Mapping[str, Any],
    fanout: int | Mapping[CanonicalEType, int],
    edge_dir: str,
    dgl: Any,
) -> Dict[str, Any]:
    sampled = dgl.sampling.sample_neighbors(
        full_g,
        dict(frontier),
        fanout=fanout,
        edge_dir=edge_dir,
        replace=False,
    )
    compacted = dgl.compact_graphs(
        sampled,
        always_preserve=dict(frontier),
        copy_ndata=False,
        copy_edata=False,
    )

    result: Dict[str, Any] = {}
    for ntype in full_g.ntypes:
        if compacted.num_nodes(ntype) == 0:
            continue
        if dgl.NID not in compacted.nodes[ntype].data:
            raise RuntimeError(
                "dgl.compact_graphs did not attach dgl.NID parent IDs"
            )
        result[ntype] = compacted.nodes[ntype].data[dgl.NID]
    return result


def k_hop_closure_hetero(
    full_g: Any,
    seed_nodes_parent: Mapping[str, Any],
    k: int = 2,
    fanout: int | Mapping[str | CanonicalEType, int] = -1,
    *,
    edge_dir: str = "in",
    allowed_etypes: Sequence[str | CanonicalEType] | None = None,
) -> Dict[str, Any]:
    """Collect related parent node IDs through k-hop heterograph expansion.

    Parameters
    ----------
    full_g:
        Parent ``dgl.DGLHeteroGraph``.  Returned IDs always address this graph.
    seed_nodes_parent:
        Mapping from node type to parent node IDs.
    k:
        Number of expansion hops.  ``k=0`` returns unique seeds.
    fanout:
        ``-1`` for all neighbors, a shared non-negative fanout, or a mapping
        from canonical edge type / relation name to per-relation fanout.
    edge_dir:
        ``"in"`` follows message sources into the current frontier;
        ``"out"`` follows outgoing destinations; ``"both"`` takes their union.
    allowed_etypes:
        Optional relation allow-list.  Relation names such as
        ``"weak_to_core"`` and canonical triples are both accepted.

    Returns
    -------
    dict
        ``{ntype: unique_parent_node_ids}`` on the parent graph's device.
    """
    dgl, torch = _load_dgl_torch()
    if k < 0:
        raise ValueError("k must be non-negative")
    if edge_dir not in {"in", "out", "both"}:
        raise ValueError("edge_dir must be one of: in, out, both")

    resolved_fanout = _resolve_fanout(full_g, fanout, allowed_etypes)
    all_nodes = _normalize_seed_nodes(full_g, seed_nodes_parent, torch)
    frontier = dict(all_nodes)

    for _ in range(k):
        if not frontier:
            break
        directions = ("in", "out") if edge_dir == "both" else (edge_dir,)
        sampled_by_type: Dict[str, list[Any]] = {}
        for direction in directions:
            sampled = _sample_parent_node_ids(
                full_g,
                frontier,
                resolved_fanout,
                direction,
                dgl,
            )
            for ntype, parent_ids in sampled.items():
                sampled_by_type.setdefault(ntype, []).append(parent_ids)

        new_frontier: Dict[str, Any] = {}
        for ntype, pieces in sampled_by_type.items():
            parent_ids = torch.unique(torch.cat(pieces))
            previous = all_nodes.get(ntype)
            if previous is None:
                added = parent_ids
                merged = parent_ids
            else:
                added = parent_ids[~torch.isin(parent_ids, previous)]
                merged = torch.unique(torch.cat((previous, parent_ids)))
            all_nodes[ntype] = merged
            if added.numel() > 0:
                new_frontier[ntype] = added
        frontier = new_frontier

    return all_nodes


__all__ = ["k_hop_closure_hetero", "__version__"]