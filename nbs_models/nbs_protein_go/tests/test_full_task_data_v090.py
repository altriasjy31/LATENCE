"""Whole-canonical label isolation over real v088 mmap/sampling fixtures."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from nbs_pg.full_task_data_v090 import FullTaskDataV090, LABEL_SPLIT_SCHEMA
from nbs_pg.go_registry_v090 import GORegistryV090, mapping_sha256, sha256_file
from test_full_task_data_v088 import _data, _equal
from test_full_task_data_v084 import inference_fixture


def _split(tmp_path, old, columns, scope="stage2_label_holdout", **extra):
    path = tmp_path / "label_split.json"
    path.write_text(json.dumps({"schema": LABEL_SPLIT_SCHEMA, "scope": scope,
        "num_task_go": old.num_task_go,
        "task_to_ontology_sha256": mapping_sha256(old.stores.task_to_ontology),
        "seen_task_columns": columns, **extra}))
    cfg = deepcopy(old.config)
    cfg["full_task"]["label_split"] = {"manifest": str(path)}
    return cfg


def test_default_closed_set_keeps_every_v088_array_and_contract(tmp_path):
    old = _data(tmp_path)
    new = FullTaskDataV090(old.config, stores=old.stores)
    new.ensure_prepared()
    previous, current = old.batch([4]), new.batch([4])
    assert current["seen_task_mask"].dtype == torch.bool
    assert current["seen_task_mask"].tolist() == [True] * 6
    _equal(previous, {k: v for k, v in current.items() if k != "seen_task_mask"})
    contract = new.data_contract()
    assert contract["v088_sampling"] == old.data_contract()["v088_sampling"]
    assert contract["v090_label_split"]["scope"] == "closed_set_all_seen"


def test_holdout_filters_every_active_label_view_without_mutating_source(tmp_path):
    old = _data(tmp_path)
    old.stores.task_to_ontology = np.array([0, 1, 2, 3, 4, 4])
    cfg = _split(tmp_path, old, [0, 1, 2, 3])
    new = FullTaskDataV090(cfg, stores=old.stores)
    new.ensure_prepared()
    expected = old.batch([4])
    batch = new.batch([4])
    assert new.seen_task_mask.dtype == np.bool_
    assert batch["positive_mask"][0].tolist() == [True, True, False, False, False, False]
    assert batch["targets"].shape == (1, 6)
    for key in ("candidate_go", "sampled_candidate_go"):
        valid = batch[key] >= 0
        assert (batch[key][valid] < 4).all()
        attrs = batch[key.replace("_go", "_attr")]
        assert (attrs[~valid] == 0).all()
    for key in ("sampled_gold_edge", "sampled_pseudo_edge"):
        assert (batch[key][1] < 4).all()
    after = old.batch([4])
    _equal(expected, after)
    assert new.label_split_contract["strict_pipeline_zero_shot"] is False
    assert "upstream" in new.label_split_contract["claim"]


def test_external_queries_receive_same_sparse_filter_without_gold(tmp_path):
    old = _data(tmp_path)
    cfg = _split(tmp_path, old, [0, 1, 2, 3])
    new = FullTaskDataV090(cfg, stores=old.stores)
    new.ensure_prepared()
    directory = inference_fixture(tmp_path, old)
    batch = new.inference_batch(directory, [0, 1])
    assert "targets" not in batch and "positive_mask" not in batch
    assert (batch["candidate_go"] < 4).all()
    assert (batch["sampled_gold_edge"][1] < 4).all()


def test_alias_split_and_wrong_mapping_are_rejected(tmp_path):
    old = _data(tmp_path)
    old.stores.task_to_ontology = np.array([0, 1, 2, 3, 4, 4])
    cfg = _split(tmp_path, old, [0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="alias groups"):
        FullTaskDataV090(cfg, stores=old.stores)
    cfg = _split(tmp_path, old, [0, 1, 2, 3])
    old.stores.task_to_ontology = np.arange(6)
    with pytest.raises(ValueError, match="mapping hash"):
        FullTaskDataV090(cfg, stores=old.stores)


@pytest.mark.parametrize("columns", [[], [True], [6], [1, 1], [1.0]])
def test_invalid_seen_column_lists_fail_closed(tmp_path, columns):
    old = _data(tmp_path)
    cfg = _split(tmp_path, old, columns)
    with pytest.raises(ValueError, match="seen_task_columns"):
        FullTaskDataV090(cfg, stores=old.stores)


def test_strict_claim_rejects_missing_upstream_provenance(tmp_path):
    old = _data(tmp_path)
    cfg = _split(tmp_path, old, [0, 1, 2], scope="strict_pipeline_zero_shot")
    with pytest.raises(ValueError, match="strict_provenance"):
        FullTaskDataV090(cfg, stores=old.stores)


def test_strict_claim_rejects_missing_registry_even_with_sources(tmp_path):
    old = _data(tmp_path)
    cfg = _split(tmp_path, old, [0, 1, 2], scope="strict_pipeline_zero_shot",
                 strict_provenance={"training_cutoff": "2022-04-01", "sources": {}})
    with pytest.raises(ValueError, match="go_registry_v090"):
        FullTaskDataV090(cfg, stores=old.stores)


def _strict_fixture(tmp_path):
    old = _data(tmp_path)
    task = tmp_path / "task.tsv"
    task.write_text("go_idx\tinput_go_id\tgo_id\n" + "".join(
        f"{i}\tGO:{i + 1:07d}\tGO:{i + 1:07d}\n" for i in range(6)))
    ontology = tmp_path / "ontology.tsv"
    ontology.write_text("ontology_go_idx\tnormalized_go_id\n" + "".join(
        f"{i}\tGO:{i + 1:07d}\n" for i in range(6)))
    full = tmp_path / "full.json"
    full.write_text(json.dumps({"num_ontology_classes": 6,
        "registry": {"file": ontology.name, "sha256": sha256_file(ontology)}}))
    registry = GORegistryV090.from_manifests(full, task)
    old.stores.go_registry_v090 = registry
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"go_registry": {"path": task.name}}))
    old.config["data"]["weak_graph_predictions_manifest"] = candidate.name
    sources = {}
    for role in ("stage1", "teacher", "features", "candidate"):
        if role == "features":
            artifact = old.stores.features.manifest_path
        elif role == "candidate":
            artifact = candidate
        else:
            artifact = tmp_path / f"{role}.bin"
            artifact.write_bytes(role.encode())
        artifact_hash = sha256_file(artifact)
        provenance = tmp_path / f"{role}_provenance.json"
        provenance.write_text(json.dumps({"schema": "nbs-v090-source-supervision-v1",
            "source_role": role, "artifact_sha256": artifact_hash,
            "go_registry_identity_sha256": registry.identity_sha256,
            "supervision_cutoff": "2022-03-31", "supervised_canonical_go_ids": ["GO:0000001"],
            "supervision_scope_complete": True, "includes_all_upstream_label_dependencies": True}))
        sources[role] = {"artifact_path": str(artifact), "artifact_sha256": artifact_hash,
            "provenance_path": str(provenance), "provenance_sha256": sha256_file(provenance)}
    cfg = _split(tmp_path, old, [0, 1, 2], scope="strict_pipeline_zero_shot",
        go_registry_sha256=sha256_file(task), strict_provenance={"training_cutoff": "2022-04-01", "sources": sources})
    return old, cfg


def test_strict_hash_bound_complete_disjoint_provenance_passes(tmp_path):
    old, cfg = _strict_fixture(tmp_path)
    data = FullTaskDataV090(cfg, stores=old.stores)
    assert data.label_split_contract["strict_pipeline_zero_shot"] is True
    assert set(data.label_split_contract["strict_provenance"]["sources"]) == {
        "stage1", "teacher", "features", "candidate"}


@pytest.mark.parametrize("change,match", [
    ({"supervised_canonical_go_ids": ["GO:0000006"]}, "overlaps unseen"),
    ({"supervision_cutoff": "2025-03-01"}, "newer than"),
    ({"includes_all_upstream_label_dependencies": False}, "dependency attestation"),
    ({"supervision_scope_complete": False}, "complete upstream supervision"),
])
def test_strict_provenance_rejects_leakage_and_incomplete_scope(tmp_path, change, match):
    old, cfg = _strict_fixture(tmp_path)
    manifest = tmp_path / "label_split.json"
    split = json.loads(manifest.read_text())
    source = split["strict_provenance"]["sources"]["features"]
    provenance = tmp_path / "features_provenance.json"
    info = json.loads(provenance.read_text())
    info.update(change)
    provenance.write_text(json.dumps(info))
    source["provenance_sha256"] = sha256_file(provenance)
    manifest.write_text(json.dumps(split))
    with pytest.raises(ValueError, match=match):
        FullTaskDataV090(cfg, stores=old.stores)


def test_strict_provenance_tampered_file_rejected(tmp_path):
    old, cfg = _strict_fixture(tmp_path)
    (tmp_path / "stage1.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        FullTaskDataV090(cfg, stores=old.stores)
