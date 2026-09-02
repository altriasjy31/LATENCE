from pathlib import Path

import numpy as np

from nbs_pg.inverted_index import (
    build_go_inverted_from_edge_index,
    build_go_inverted_from_protein_csr,
    build_protein_major_annotation_csr,
)


def test_v060_export_regression_test_uses_full_task_top512_contract():
    """Prevent deployment of the historical rare-first/top-20 assertion."""
    import importlib.util

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "test_export_weak_graph_predictions.py"
    )
    source = script.read_text(encoding="utf-8")
    assert "test_parser_defaults_to_full_task_candidates" in source
    assert 'self.assertEqual(action.default, "full_task")' in source
    assert "self.assertEqual(topk.default, 512)" in source
    assert "test_parser_defaults_to_rare_first" not in source
    assert importlib.util.spec_from_file_location("export_regression", script) is not None


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


def test_gold_protein_major_csr_preserves_weak_core_go_message_path(tmp_path: Path):
    edge = np.array([[2, 0, 2, 1], [3, 1, 0, 2]], dtype=np.int32)
    source = tmp_path / "gold.npy"
    np.save(source, edge)
    result = build_protein_major_annotation_csr(
        source, tmp_path / "gold_out", prefix="gold", num_proteins=4, num_go=4, chunk_edges=2
    )
    indptr = np.load(result.indptr)
    go = np.load(result.go_idx)
    np.testing.assert_array_equal(indptr, [0, 1, 2, 4, 4])
    np.testing.assert_array_equal(go[indptr[2]:indptr[3]], [3, 0])


def test_incremental_gold_build_preserves_existing_candidate_manifest(tmp_path: Path):
    import csv
    import json
    from nbs_pg.inverted_index import build_latence_go_protein_indices

    source = tmp_path / "source"
    source.mkdir()
    candidate_edge = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int32)
    np.save(source / "candidate_edge.npy", candidate_edge)
    np.save(source / "candidate_attr.npy", np.ones((4, 3), dtype=np.float32))
    weak_manifest = {
        "go_registry": {"num_terms": 2},
        "backbone_candidate_edges": {
            "edge_index_file": "candidate_edge.npy",
            "edge_attr_file": "candidate_attr.npy",
            "edge_attr_columns": ["backbone_probability", "selector_score", "reciprocal_rank"],
        },
        "roles": [
            {"role": "core", "candidate_degree_mean": 2},
        ],
    }
    manifest_path = source / "weak.json"
    manifest_path.write_text(json.dumps(weak_manifest), encoding="utf-8")
    registry = tmp_path / "registry.csv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_idx", "role", "role_row_idx"])
        writer.writeheader()
        writer.writerow({"protein_idx": 0, "role": "core", "role_row_idx": 0})
        writer.writerow({"protein_idx": 1, "role": "core", "role_row_idx": 1})
    output = tmp_path / "out"
    build_latence_go_protein_indices(
        manifest_path,
        registry,
        output,
        build_candidate=True,
        build_pseudo=False,
    )
    gold_edge = np.array([[0, 1], [1, 0]], dtype=np.int32)
    np.save(tmp_path / "gold.npy", gold_edge)
    result = build_latence_go_protein_indices(
        manifest_path,
        registry,
        output,
        build_candidate=False,
        build_pseudo=False,
        gold_edge_index_path=tmp_path / "gold.npy",
        merge_existing_manifest=True,
    )
    assert set(result["indices"]) == {"candidate", "gold"}
    assert "protein_major" in result["indices"]["gold"]
    assert Path(result["indices"]["candidate"]["indptr"]).exists()
