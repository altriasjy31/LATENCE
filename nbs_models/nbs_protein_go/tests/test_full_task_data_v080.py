import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nbs_pg.full_task_data import FullTaskData, _CoreSearch, _sha256
from nbs_pg.latence_graph_stores import (
    FixedDegreeProteinGOStore, GlobalProteinGOCSRStore, ProteinRegistryStore,
    RoleAwareProteinFeatureStore, RoleLocalProteinGOCSRStore,
)
from nbs_pg.latence_stores import RoleAwareBaseLogitStore, RoleProbabilitySlice


def fixture_data(tmp_path, holdout=0):
    def save(name, value):
        path = tmp_path / name
        np.save(path, value)
        return path

    registry_path = tmp_path / "protein_registry.csv"
    registry_path.write_text("protein_idx,role,role_row_idx\n" + "".join(
        f"{i},{'core' if i < 4 else 'weak'},{i if i < 4 else i - 4}\n" for i in range(6)
    ))
    registry = ProteinRegistryStore(registry_path)
    core_x = np.array([[1, 0], [.8, .2], [0, 1], [-1, 0]], dtype=np.float32)
    weak_x = np.array([[.9, .1], [.1, .9]], dtype=np.float32)
    save("core_x.npy", core_x)
    save("weak_x.npy", weak_x)
    representation = tmp_path / "representation_manifest.json"
    representation.write_text(json.dumps({"feature_dim": 2, "roles": [
        {"role": "core", "count": 4, "feature_file": "core_x.npy"},
        {"role": "weak", "count": 2, "feature_file": "weak_x.npy"},
    ]}))
    features = RoleAwareProteinFeatureStore(representation, registry)
    base = RoleAwareBaseLogitStore([
        RoleProbabilitySlice("core", 0, 4, str(save("core_p.npy", np.full((4, 6), .2, np.float32)))),
        RoleProbabilitySlice("weak", 4, 6, str(save("weak_p.npy", np.full((2, 6), .2, np.float32)))),
    ], num_go=6)
    gold = GlobalProteinGOCSRStore(
        save("gold_ptr.npy", np.array([0, 2, 3, 5, 6, 6, 6])),
        save("gold_go.npy", np.array([1, 2, 2, 3, 4, 5])), num_go=6,
    )
    pseudo = RoleLocalProteinGOCSRStore(
        save("pseudo_ptr.npy", np.array([0, 3, 4])),
        save("pseudo_go.npy", np.array([0, 1, 4, 2])), role="weak", registry=registry,
        probability_path=save("pseudo_p.npy", np.array([.5, .8, .99, .9], dtype=np.float32)),
    )
    edges = np.stack((np.repeat(np.arange(6), 2), np.tile([0, 5], 6)))
    candidate = FixedDegreeProteinGOStore(
        save("candidate_edge.npy", edges), save("candidate_attr.npy", np.ones((12, 3), np.float32)),
        fixed_degree=2, source_protein_start=0,
    )
    stores = SimpleNamespace(
        registry=registry, features=features, feature_dim=2, num_task_go=6,
        gold_messages=gold, pseudo_messages=pseudo, candidate_messages=candidate,
        episode_sampler=SimpleNamespace(base_logit_store=base),
    )
    config = {"data": {"root": str(tmp_path)}, "full_task": {
        "core_neighbors": 2, "holdout_core_count": holdout, "holdout_seed": 8080,
        "neighbor_cache": str(tmp_path / "neighbors"),
    }}
    return FullTaskData(config, stores=stores)


def test_all_positive_go_supervision_and_no_pseudo_graph_input(tmp_path):
    data = fixture_data(tmp_path)
    data.prepare_neighbors(device="cpu", query_batch_size=2)
    batch = data.batch([4, 0, 4])
    assert batch["base_logits"].shape == (3, 6)
    assert batch["positive_mask"].sum(1).tolist() == [3, 2, 3]
    assert batch["targets"][0, 0] == .5  # Original >.5 rounded to FP16 .5 is retained.
    assert batch["positive_mask"][0, 4]  # Not present in its candidate top-k.
    assert batch["is_weak"].tolist() == [True, False, True]
    original_graph = {key: value.clone() for key, value in batch.items()
                      if key not in {"targets", "positive_mask", "is_weak"}}
    # Changing weak targets must have NO effect on any forward input.
    data.stores.pseudo_messages.probability = np.array([.9, .7, .8, .95], np.float32)
    changed = data.batch([4, 0, 4])
    assert not torch.equal(changed["targets"], batch["targets"])
    for key, value in original_graph.items():
        assert torch.equal(value, changed[key]), key
    # Core target zero is excluded even though it is its own nearest feature.
    core_target_neighbors = batch["anchor_x"][batch["neighbor_index"][1]]
    assert not torch.any(torch.all(core_target_neighbors == torch.tensor([1., 0.]), dim=1))


def test_holdout_is_absent_from_every_annotation_bearing_anchor(tmp_path):
    data = fixture_data(tmp_path, holdout=1)
    assert not np.intersect1d(data.validation_ids, data.anchor_core_ids).size
    assert not np.intersect1d(data.validation_ids, data.core_ids).size
    data.prepare_neighbors(device="cpu")
    heldout_x = data.stores.features.gather(data.validation_ids)
    batch = data.batch([*data.core_ids, *data.weak_ids, *data.validation_ids])
    for x in heldout_x:
        assert not torch.any(torch.all(batch["anchor_x"] == torch.from_numpy(x), dim=1))
    assert batch["positive_mask"][-1].sum() > 0


def test_exact_retrieval_matches_cosine_and_rejects_self(tmp_path):
    core = np.array([[1, 0], [.9, .1], [0, 1], [-1, 0]], np.float32)
    search = _CoreSearch(core, backend="torch", device="cpu")
    neighbors, attrs = search.search(core[:2], 2, np.array([0, 1]))
    assert neighbors.tolist() == [[1, 2], [0, 2]]
    assert np.all(attrs[..., 0] >= 0)
    assert attrs[0, :, 2].tolist() == [1., .5]
    assert attrs[0, 0, 1] == pytest.approx(.9 / np.sqrt(.82), rel=1e-6)


def test_cache_reuse_checks_split_identity(tmp_path):
    data = fixture_data(tmp_path)
    path = data.prepare_neighbors(device="cpu")
    modified = json.loads(path.read_text())
    modified["signature"]["anchor_ids_sha256"] = "different"
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="differs from this split/data"):
        data.prepare_neighbors(device="cpu")


def inference_fixture(tmp_path, data):
    input_dir = tmp_path / "inductive"
    input_dir.mkdir()
    manifest = {
        "representation": {}, "base_probability": {},
        "candidate_evidence": {"selector_scope": "full_task", "expert_probability_used": False,
                               "label_hint_used": False},
        "cache_signature": {"representation_manifest_sha256": _sha256(data.stores.features.manifest_path)},
        "external_pp": {"core_representation": str(tmp_path / "core_x.npy")},
    }
    specs = [
        ("ind_test_repr.f16.npy", np.array([[.9, .1], [.1, .9]], np.float16), "representation", "sha256"),
        ("backbone_ind_test_prob.f16.npy", np.full((2, 6), .2, np.float16), "base_probability", "sha256"),
        ("candidate_go_index.i32.npy", np.array([[0, 5], [0, 5]], np.int32), "candidate_evidence", "go_index_sha256"),
        ("candidate_edge_attr.f32.npy", np.ones((2, 2, 3), np.float32), "candidate_evidence", "edge_attr_sha256"),
    ]
    for name, array, section, key in specs:
        path = input_dir / name
        np.save(path, array)
        manifest[section][key] = _sha256(path)
    (input_dir / "ind_test_input_manifest.json").write_text(json.dumps(manifest))
    return input_dir


def test_inductive_and_training_use_same_heldout_free_anchor_pool(tmp_path):
    data = fixture_data(tmp_path, holdout=1)
    data.prepare_neighbors(device="cpu")
    input_dir = inference_fixture(tmp_path, data)
    training = data.batch([4, 5])
    inference = data.inference_batch(input_dir, [0, 1])
    assert "targets" not in inference and "positive_mask" not in inference
    assert torch.equal(training["neighbor_index"], inference["neighbor_index"])
    assert torch.equal(training["anchor_x"], inference["anchor_x"])
    assert torch.equal(training["anchor_go_edge"], inference["anchor_go_edge"])
    assert inference["base_logits"].dtype == torch.float32
    # Data corruption is rejected even if the dimensions still match.
    data._inference.clear()
    np.save(input_dir / "candidate_go_index.i32.npy", np.array([[1, 5], [0, 5]], np.int32))
    with pytest.raises(ValueError, match="hash mismatch"):
        data.inference_batch(input_dir, [0])
