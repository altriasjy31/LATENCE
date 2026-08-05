from pathlib import Path

import numpy as np

from nbs_pg.inverted_index import (
    build_go_inverted_from_edge_index,
    build_go_inverted_from_protein_csr,
)


def test_edge_index_two_pass_inversion(tmp_path: Path):
    # Three proteins, fixed degree two, protein-major source order.
    edge = np.array([[0, 0, 1, 1, 2, 2], [2, 0, 1, 2, 0, 2]], dtype=np.int32)
    source = tmp_path / "edge.npy"
    np.save(source, edge)
    result = build_go_inverted_from_edge_index(
        source,
        tmp_path / "out",
        prefix="candidate",
        num_go=3,
        chunk_edges=2,
        expected_fixed_degree=2,
        require_protein_major_fixed_degree=True,
    )
    indptr = np.load(result.indptr)
    protein = np.load(result.protein_idx)
    rank = np.load(result.payloads["source_rank"])
    np.testing.assert_array_equal(indptr, [0, 2, 3, 6])
    np.testing.assert_array_equal(protein[indptr[2] : indptr[3]], [0, 1, 2])
    np.testing.assert_array_equal(rank[indptr[2] : indptr[3]], [0, 1, 1])


def test_protein_csr_inversion_preserves_probabilities(tmp_path: Path):
    indptr = np.array([0, 2, 2, 4], dtype=np.int64)
    indices = np.array([1, 2, 0, 2], dtype=np.int32)
    prob = np.array([0.6, 0.7, 0.8, 0.9], dtype=np.float16)
    np.save(tmp_path / "indptr.npy", indptr)
    np.save(tmp_path / "indices.npy", indices)
    np.save(tmp_path / "prob.npy", prob)
    result = build_go_inverted_from_protein_csr(
        tmp_path / "indptr.npy",
        tmp_path / "indices.npy",
        tmp_path / "out",
        prefix="pseudo",
        num_go=3,
        global_protein_index=np.array([10, 11, 12], dtype=np.int64),
        probability_path=tmp_path / "prob.npy",
        chunk_edges=2,
    )
    out_indptr = np.load(result.indptr)
    out_protein = np.load(result.protein_idx)
    out_prob = np.load(result.payloads["probability"])
    np.testing.assert_array_equal(out_indptr, [0, 1, 2, 4])
    np.testing.assert_array_equal(out_protein, [12, 10, 10, 12])
    np.testing.assert_allclose(out_prob, [0.8, 0.6, 0.7, 0.9], rtol=0, atol=1e-3)
