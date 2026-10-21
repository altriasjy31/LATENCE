"""Alignment and no-leakage tests for the v0.8.2 dense weak teacher."""
import hashlib
import json

import numpy as np
import pytest
import torch

from nbs_pg.full_task_data import _sha256
from nbs_pg.full_task_data_v082 import FullTaskDataV082
from test_full_task_data_v080 import fixture_data, inference_fixture


def fixture_teacher_data(tmp_path, *, no_positive=False, construct=True, holdout=0):
    legacy = fixture_data(tmp_path, holdout=holdout)
    registry = legacy.registry.path
    registry.write_text("protein_idx,protein_id,role,role_row_idx\n" + "".join(
        f"{i},P{i},{'core' if i < 4 else 'weak'},{i if i < 4 else i - 4}\n"
        for i in range(6)
    ))
    legacy.stores.task_to_ontology = np.arange(6, dtype=np.int64)
    go_dir = tmp_path / "gg_relations"
    go_dir.mkdir()
    go_path = go_dir / "go_registry.tsv"
    go_path.write_text("go_idx\tgo_id\n" + "".join(f"{i}\tGO:{i:07d}\n" for i in range(6)))
    teacher = np.array([[.5, .8, .05, .1, .99, .3], [.02, .1, .9, .2, .3, .4]], np.float32)
    if no_positive:
        legacy.stores.pseudo_messages.indptr = np.array([0, 3, 3])
        legacy.stores.pseudo_messages.go_idx = np.array([0, 1, 4])
        legacy.stores.pseudo_messages.probability = np.array([.5, .8, .99], np.float32)
        teacher[1, 2] = .09
    teacher_path = tmp_path / "modelout_weak_prob.f32.npy"
    np.save(teacher_path, teacher)
    manifest = {
        "checkpoint": {"sha256": "a" * 64},
        "protein_registry": {"path": str(registry), "sha256": _sha256(registry), "num_proteins": 6},
        "go_registry": {"path": str(go_path), "sha256": _sha256(go_path), "num_terms": 6},
        "model_semantics": {"modelout": {"prediction_key": "modelout::expert_prob::decoderprob::expert",
                                            "label_hint_used": False, "external_probability_used": True}},
        "roles": [{"role": "weak", "rows": 2,
                   "protein_ids_sha256": hashlib.sha256(b"P4\nP5\n").hexdigest(),
                   "global_protein_idx_min": 4, "global_protein_idx_max": 5,
                   "modelout_dense_file": teacher_path.name}],
    }
    manifest_path = tmp_path / "weak_graph_predictions_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = dict(legacy.config)
    config["data"] = dict(config["data"], weak_graph_predictions_manifest=manifest_path.name)
    if not construct:
        return config, legacy.stores, manifest_path, teacher_path
    return FullTaskDataV082(config, stores=legacy.stores)


def test_dense_all_go_teacher_and_core_unavailable_mask(tmp_path):
    data = fixture_teacher_data(tmp_path)
    assert isinstance(data.teacher_probability, np.memmap)
    data.prepare_neighbors(device="cpu")
    batch = data.batch([5, 0, 4, 5])
    assert batch["teacher_prob"].shape == (4, 6)
    assert batch["teacher_available"].tolist() == [True, False, True, True]
    assert torch.equal(batch["teacher_prob"][0], batch["teacher_prob"][3])
    assert batch["teacher_prob"][2, 5] == pytest.approx(.3)
    assert not batch["positive_mask"][2, 5]  # Retained below-threshold information.
    assert torch.count_nonzero(batch["teacher_prob"][1]) == 0
    contract = data.data_contract()
    assert contract["teacher_v082"]["probability_sha256"] == _sha256(data.teacher_path)
    assert contract["weak_eligibility"] == "all_weak_registry_rows"


def test_weak_without_thresholded_positive_is_still_trained(tmp_path):
    data = fixture_teacher_data(tmp_path, no_positive=True)
    assert data.weak_ids.tolist() == [4, 5]
    data.prepare_neighbors(device="cpu")
    batch = data.batch([5])
    assert not batch["positive_mask"].any()
    assert batch["teacher_available"].item()
    assert torch.all(batch["teacher_prob"] > 0)


def test_teacher_change_cannot_change_graph_or_inductive_features(tmp_path):
    data = fixture_teacher_data(tmp_path, holdout=1)
    data.prepare_neighbors(device="cpu")
    original = data.batch([4, 5])
    mutable = np.load(data.teacher_path, mmap_mode="r+")
    mutable[:, 5] += .1
    mutable.flush()
    changed = data.batch([4, 5])
    assert not torch.equal(original["teacher_prob"], changed["teacher_prob"])
    for key in original:
        if key != "teacher_prob":
            assert torch.equal(original[key], changed[key]), key
    # Fixture cache gets the newly required immutable GO-column provenance.
    input_dir = inference_fixture(tmp_path, data)
    path = input_dir / "ind_test_input_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["cache_signature"]["go_registry_sha256"] = data._go_registry_sha256
    path.write_text(json.dumps(manifest))
    batch = data.inference_batch(input_dir, [0, 1])
    assert "teacher_prob" not in batch and "teacher_available" not in batch
    assert "targets" not in batch and "positive_mask" not in batch


@pytest.mark.parametrize("change,message", [
    ("row_count", "row count"), ("role_order", "protein order hash"),
    ("registry", "protein registry hash"), ("go_order", "GO registry hash"),
    ("go_count", "GO column count"), ("label_hint", "label_hint_used=false"),
    ("checkpoint", "checkpoint SHA256"), ("missing_file", "Missing exporter-declared"),
    ("missing_dense", "save-dense-modelout true"),
])
def test_teacher_provenance_mismatch_rejected(tmp_path, change, message):
    config, stores, manifest_path, _ = fixture_teacher_data(tmp_path, construct=False)
    manifest = json.loads(manifest_path.read_text())
    if change == "row_count":
        manifest["roles"][0]["rows"] = 1
    elif change == "role_order":
        manifest["roles"][0]["protein_ids_sha256"] = hashlib.sha256(b"P5\nP4\n").hexdigest()
    elif change == "registry":
        manifest["protein_registry"]["sha256"] = "0" * 64
    elif change == "go_order":
        manifest["go_registry"]["sha256"] = "0" * 64
    elif change == "go_count":
        manifest["go_registry"]["num_terms"] = 5
    elif change == "label_hint":
        manifest["model_semantics"]["modelout"]["label_hint_used"] = True
    elif change == "checkpoint":
        manifest["checkpoint"]["sha256"] = None
    elif change == "missing_file":
        manifest["roles"][0]["modelout_dense_file"] = "absent.npy"
    else:
        manifest["roles"][0]["modelout_dense_file"] = None
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        FullTaskDataV082(config, stores=stores)


def test_teacher_invalid_values_and_swapped_dense_file_rejected(tmp_path):
    data = fixture_teacher_data(tmp_path)
    data.prepare_neighbors(device="cpu")
    mutable = np.load(data.teacher_path, mmap_mode="r+")
    mutable[0, 3] = np.nan
    mutable.flush()
    with pytest.raises(ValueError, match="non-finite"):
        data.batch([4])
    mutable[0, 3] = .1
    mutable[0, 1] = .7
    mutable.flush()
    with pytest.raises(ValueError, match="disagrees.*pseudo CSR"):
        data.batch([4])


def test_no_teacher_permission_only_allows_inductive_or_core_use(tmp_path):
    legacy = fixture_data(tmp_path)
    data = FullTaskDataV082(legacy.config, stores=legacy.stores, require_teacher=False)
    data.prepare_neighbors(device="cpu")
    assert not data.batch([0])["teacher_available"].any()
    with pytest.raises(RuntimeError, match="require_teacher=True"):
        data.batch([4])


def test_old_neighbor_cache_rebuilt_when_zero_positive_weak_was_absent(tmp_path):
    config, stores, _, _ = fixture_teacher_data(tmp_path, no_positive=True, construct=False)
    from nbs_pg.full_task_data import FullTaskData
    legacy = FullTaskData(config, stores=stores)
    assert legacy.weak_ids.tolist() == [4]
    legacy.prepare_neighbors(device="cpu")
    assert (legacy._neighbors[5] == -1).all()
    data = FullTaskDataV082(config, stores=stores)
    assert not data._load_neighbors()
    data.prepare_neighbors(device="cpu")
    assert np.all(data._neighbors[5] >= 0)
