from __future__ import annotations

import numpy as np
import pytest
import torch

from nbs_pg.latence_graph_stores import FixedDegreeProteinGOStore
from nbs_pg.local_loader import (
    _dedupe_edges,
    _dedupe_grouped_fixed_degree_edges,
)


def test_fixed_degree_candidate_gather_keeps_legacy_order(tmp_path):
    sources = np.repeat(np.asarray([10, 11, 12], np.int32), 4)
    destinations = np.asarray(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32
    )
    edge = np.stack([sources, destinations], axis=0)
    attr = np.arange(36, dtype=np.float32).reshape(12, 3)
    edge_path = tmp_path / "edge.npy"
    attr_path = tmp_path / "attr.npy"
    np.save(edge_path, edge)
    np.save(attr_path, attr)
    store = FixedDegreeProteinGOStore(
        edge_path,
        attr_path,
        fixed_degree=4,
        source_protein_start=10,
    )

    gathered_edge, gathered_attr = store.gather([12, 99, 10, 12], topk=2)

    expected_offsets = np.asarray([0, 1, 8, 9])
    np.testing.assert_array_equal(gathered_edge, edge[:, expected_offsets])
    np.testing.assert_array_equal(gathered_attr, attr[expected_offsets])


def test_grouped_candidate_dedupe_matches_generic_path():
    edge = np.asarray(
        [
            [1, 1, 1, 1, 4, 4, 4, 4],
            [5, 3, 5, 4, 2, 2, 7, 6],
        ],
        dtype=np.int64,
    )
    attr = np.asarray(
        [
            [0.7, 0.0, 1.0],
            [0.8, 0.0, 1.0],
            [0.9, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
            [0.6, 0.0, 1.0],
            [0.4, 0.0, 1.0],
            [0.5, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    expected_edge, expected_attr = _dedupe_edges(edge, attr)
    actual_edge, actual_attr = _dedupe_grouped_fixed_degree_edges(
        edge, attr, block_size=4
    )

    np.testing.assert_array_equal(actual_edge, expected_edge)
    np.testing.assert_array_equal(actual_attr, expected_attr)


def test_weighted_sage_node_projection_matches_edge_projection():
    pytest.importorskip("torch_geometric")
    from torch_geometric.utils import scatter

    from nbs_pg.layers import WeightedSAGEConv

    torch.manual_seed(7)
    layer = WeightedSAGEConv(hidden_dim=5, edge_dim=3)
    x_src = torch.randn(4, 5)
    x_dst = torch.randn(3, 5)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 1, 2, 2, 3], [0, 1, 0, 1, 2, 0, 2, 1]],
        dtype=torch.long,
    )
    edge_attr = torch.rand(edge_index.shape[1], 3)

    actual = layer((x_src, x_dst), edge_index, edge_attr)
    src, dst = edge_index
    legacy_message = layer.source_proj(x_src[src])
    base_weight = edge_attr[:, :1].clamp(0.0, 1.0)
    learned_factor = 2.0 * torch.sigmoid(layer.edge_gate(edge_attr))
    weight = base_weight * learned_factor
    expected = scatter(
        legacy_message * weight,
        dst,
        dim=0,
        dim_size=x_dst.size(0),
        reduce="sum",
    )
    count = scatter(
        torch.ones_like(weight),
        dst,
        dim=0,
        dim_size=x_dst.size(0),
        reduce="sum",
    )
    expected = expected / count.clamp_min(1.0)

    torch.testing.assert_close(actual, expected)
