"""GO identity/hash checks are independent of PyTorch and protein labels."""
import json

import numpy as np
import pytest

from nbs_pg.go_registry_v090 import GORegistryV090, normalize_go_id, sha256_file


def _records():
    return [{"go_id": "GO:0000001", "ontology_row": 2, "alt_ids": ["GO:0000011"]},
            {"go_id": "GO:0000002", "ontology_row": 4},
            {"go_id": "GO:0000003", "ontology_row": 5}]


def test_task_alias_columns_and_dynamic_target_order_are_preserved():
    registry = GORegistryV090(_records(), ["GO:0000011", "GO:0000002", "GO:0000001"],
                             [2, 4, 2], num_ontology_rows=6)
    assert registry.task_to_ontology.tolist() == [2, 4, 2]
    assert registry.resolve_target_go(["GO:0000003", "GO:0000011"]).tolist() == [5, 2]
    assert registry.resolve_target_go(["GO:0000001", "GO:0000001"]).tolist() == [2, 2]
    assert registry.canonical_go_id("GO:0000011") == "GO:0000001"
    with pytest.raises(ValueError, match="no supplied ontology geometry"):
        registry.resolve_target_go(["GO:0000099"])
    assert registry.contract()["geometry_policy"] == "existing_hash_bound_ontology_rows_only"


@pytest.mark.parametrize("go", ["GO:123", "GO:00000010", "prefixGO:0000001suffix", None])
def test_malformed_go_is_rejected(go):
    with pytest.raises(ValueError, match="GO"):
        normalize_go_id(go)


def test_go_iri_normalization():
    assert normalize_go_id("http://purl.obolibrary.org/obo/GO_0000001") == "GO:0000001"


def test_conflicting_duplicates_aliases_and_mapping_fail():
    with pytest.raises(ValueError, match="conflicting duplicate"):
        GORegistryV090(_records() + [{"go_id": "GO:0000001", "ontology_row": 3}])
    with pytest.raises(ValueError, match="conflicting GO alias"):
        GORegistryV090(_records() + [{"go_id": "GO:0000004", "ontology_row": 3, "alt_ids": ["GO:0000011"]}])
    with pytest.raises(ValueError, match="mapping"):
        GORegistryV090(_records(), ["GO:0000001"], [1])


def test_obsolete_requires_explicit_unambiguous_replacement():
    records = _records() + [{"go_id": "GO:0000004", "ontology_row": 3,
                             "is_obsolete": True, "replaced_by": ["GO:0000001"]}]
    registry = GORegistryV090(records)
    with pytest.raises(ValueError, match="obsolete GO"):
        registry.resolve_target_go(["GO:0000004"])
    assert registry.resolve_target_go(["GO:0000004"], allow_obsolete_replacement=True).tolist() == [2]
    records[-1]["replaced_by"] = ["GO:0000001", "GO:0000002"]
    with pytest.raises(ValueError, match="ambiguous"):
        GORegistryV090(records).resolve_target_go(["GO:0000004"], allow_obsolete_replacement=True)


def _manifest_fixture(tmp_path):
    full_registry = tmp_path / "ontology.tsv"
    full_registry.write_text("ontology_go_idx\tnormalized_go_id\n0\tGO:0000001\n1\tGO:0000002\n2\t\n")
    task_registry = tmp_path / "task.tsv"
    task_registry.write_text("go_idx\tinput_go_id\tgo_id\n0\tGO:0000011\tGO:0000001\n1\tGO:0000002\tGO:0000002\n2\tGO:0000001\tGO:0000001\n")
    full = tmp_path / "full.json"
    full.write_text(json.dumps({"num_ontology_classes": 3,
        "registry": {"file": full_registry.name, "sha256": sha256_file(full_registry)},
        "source": {"checkpoint_sha256": "a" * 64}}))
    mapping = tmp_path / "mapping.npy"
    np.save(mapping, np.array([0, 1, 0], np.int32))
    alignment = tmp_path / "alignment.json"
    alignment.write_text(json.dumps({"source": {"checkpoint_sha256": "a" * 64,
        "go_registry_sha256": sha256_file(task_registry)},
        "arrays": {"source_row": {"file": mapping.name}}}))
    return full, task_registry, alignment, full_registry


def test_real_manifest_formats_are_hash_bound(tmp_path):
    full, tasks, alignment, _ = _manifest_fixture(tmp_path)
    registry = GORegistryV090.from_manifests(full, tasks, alignment)
    assert registry.resolve_target_go(["GO:0000011", "GO:0000002"]).tolist() == [0, 1]
    assert registry.num_ontology_rows == 3
    assert registry.contract()["artifact_hashes"]["task_registry_sha256"] == sha256_file(tasks)


def test_registry_tamper_and_alignment_checkpoint_mismatch_rejected(tmp_path):
    full, tasks, alignment, ontology = _manifest_fixture(tmp_path)
    info = json.loads(alignment.read_text())
    info["source"]["checkpoint_sha256"] = "b" * 64
    alignment.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="checkpoint hash"):
        GORegistryV090.from_manifests(full, tasks, alignment)
    ontology.write_text(ontology.read_text().replace("GO:0000001", "GO:0000009"))
    with pytest.raises(ValueError, match="registry hash"):
        GORegistryV090.from_manifests(full, tasks)
