from __future__ import annotations

import numpy as np

from nbs_pg.latence_graph_stores import EdgeOffsetCSRStore
from nbs_pg.sampling_indices import build_edge_offset_csr


def test_edge_offset_csr_preserves_message_direction(tmp_path):
    edge = np.asarray([[4, 2, 3, 1], [1, 1, 0, 0]], dtype=np.int32)
    attr = np.arange(12, dtype=np.float32).reshape(4, 3)
    edge_path = tmp_path / "edge.npy"
    attr_path = tmp_path / "attr.npy"
    np.save(edge_path, edge)
    np.save(attr_path, attr)
    result = build_edge_offset_csr(
        edge_path,
        tmp_path / "index",
        prefix="similar_to",
        num_nodes=5,
        key_axis=1,
    )
    store = EdgeOffsetCSRStore(
        indptr_path=result["indptr"],
        edge_offset_path=result["edge_offset"],
        edge_index_path=edge_path,
        edge_attr_path=attr_path,
        key_axis=1,
    )
    sampled, sampled_attr = store.sample([1], 8, rng=np.random.default_rng(1))
    assert set(map(tuple, sampled.T.tolist())) == {(4, 1), (2, 1)}
    assert sampled_attr.shape == (2, 3)
