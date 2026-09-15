"""Label-independent alignment of first-stage candidate features (NumPy only)."""
from __future__ import annotations

import numpy as np


def align_candidate_evidence(
    protein_idx: np.ndarray,
    query_go_idx: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> np.ndarray:
    """Scatter sparse (protein, task GO) features into [Q,C,F].

    Neither labels nor supervision masks are inputs. Axis order is preserved;
    missing pairs stay zero. Conflicting duplicate edges are rejected rather
    than allowing train/inference to choose different records silently.
    """
    proteins = np.asarray(protein_idx, dtype=np.int64)
    queries = np.asarray(query_go_idx, dtype=np.int64)
    edges = np.asarray(edge_index, dtype=np.int64)
    attrs = np.asarray(edge_attr, dtype=np.float32)
    if proteins.ndim != 1 or queries.ndim != 1:
        raise ValueError("protein/query axes must be one-dimensional")
    if len(np.unique(proteins)) != len(proteins) or len(np.unique(queries)) != len(queries):
        raise ValueError("protein/query axes must be unique")
    if edges.ndim != 2 or edges.shape[0] != 2 or attrs.ndim != 2 or attrs.shape[0] != edges.shape[1]:
        raise ValueError("candidate edge indices/attributes must align")
    out = np.zeros((queries.size, proteins.size, attrs.shape[1]), dtype=np.float32)
    if not proteins.size or not queries.size or not edges.shape[1]:
        return out
    p_order, q_order = np.argsort(proteins), np.argsort(queries)
    p_sorted, q_sorted = proteins[p_order], queries[q_order]
    pi = np.searchsorted(p_sorted, edges[0]).clip(max=proteins.size - 1)
    qi = np.searchsorted(q_sorted, edges[1]).clip(max=queries.size - 1)
    keep = (p_sorted[pi] == edges[0]) & (q_sorted[qi] == edges[1])
    p, q, selected = p_order[pi[keep]], q_order[qi[keep]], attrs[keep]
    if not np.isfinite(selected).all():
        raise ValueError("decoded candidate evidence contains NaN/Inf")
    pair = q * proteins.size + p
    _, first, inverse = np.unique(pair, return_index=True, return_inverse=True)
    if first.size != pair.size and not np.array_equal(selected, selected[first][inverse]):
        raise ValueError("conflicting duplicate candidate pair attributes")
    out[q[first], p[first]] = selected[first]
    return out
