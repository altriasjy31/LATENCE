import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nbs_pg.full_task_data_v083 import (
    FullTaskDataV083, PP_RELATIONS, PP_INDEX_FILE, PP_MANIFEST_FILE,
    _merge_incoming_topk,
)
from test_full_task_data_v080 import fixture_data, inference_fixture


def data_v083(tmp_path, *, holdout=0, fanout=2, enabled=True, block=3):
    old = fixture_data(tmp_path, holdout=holdout)
    # Duplicate source 4->0 occurs across blocks, with distinct confidence.
    edges = np.array([[4, 5, 4, 0, 1, 0, 2, 1],
                      [0, 0, 0, 0, 0, 2, 1, 2]], dtype=np.int64)
    values = np.array([[.6, .1, 1], [.8, .2, .5], [.9, .3, .3],
                       [1, 1, 1], [.7, .4, .25], [.8, .8, 1],
                       [.8, .8, 1], [.8, .8, 1]], dtype=np.float32)
    old.stores.pp = {
        # key_axis intentionally varies; it must not reverse direction.
        "ppi": SimpleNamespace(edge_index=edges, edge_attr=values, key_axis=0),
        "similar_to": SimpleNamespace(edge_index=np.array([[3], [2]]),
                                      edge_attr=np.array([[.3, .4, .5]], np.float32), key_axis=1),
        "weak_to_core": SimpleNamespace(edge_index=np.array([[5], [1]]),
                                        edge_attr=np.array([[.4, .5, .6]], np.float32), key_axis=0),
    }
    old.stores.task_to_ontology = np.arange(old.num_task_go)
    config = old.config
    config["full_task"].update(use_pp_context=enabled, pp_context_fanout=fanout,
                               pp_candidate_topk=1, pp_scan_block_size=block,
                               pp_context_cache=str(tmp_path / "pp_context"))
    return FullTaskDataV083(config, stores=old.stores)


def test_native_direction_dedup_topk_ties_padding_and_coverage(tmp_path):
    data = data_v083(tmp_path)
    data.prepare_neighbors(device="cpu")
    # Self-loop 0->0 removed; the stronger duplicate 4->0 retained exactly once.
    assert data._pp_neighbors[0, 0].tolist() == [4, 5]
    assert data._pp_attrs[0, 0, 0, 0] == pytest.approx(.9)
    # Equal scores use deterministic ascending source ID.
    assert data._pp_neighbors[2, 0].tolist() == [0, 1]
    # Directed 5->1 is NOT reversed into a center 5 adjacency.
    assert data._pp_neighbors[1, 2].tolist() == [5, -1]
    assert data._pp_neighbors[2, 1].tolist() == [3, -1]
    assert data._pp_neighbors[3].tolist() == [[-1, -1]] * 3
    report = json.loads((data.pp_cache_dir / PP_MANIFEST_FILE).read_text())
    assert report["signature"]["relations"] == list(PP_RELATIONS)
    assert report["coverage"]["weak_to_core"]["centers_with_context"] == 1
    assert report["coverage"]["ppi"]["retained_unique_edges"] == 5


def test_streaming_merge_matches_global_sort_independent_of_blocking():
    rng = np.random.default_rng(43)
    dst = rng.integers(0, 11, size=700)
    src = rng.integers(0, 21, size=700)
    attr = rng.integers(0, 5, size=(700, 3)).astype(np.float32) / 4
    expected = []
    for center in range(11):
        pairs = {}
        for source, value in zip(src[dst == center], attr[dst == center]):
            score = tuple(value.tolist())
            pairs[int(source)] = max(pairs.get(int(source), (-1, -1, -1)), score)
        expected.append(sorted(pairs, key=lambda item: (*[-v for v in pairs[item]], item))[:4])
    for chunk in (1, 3, 29, 700):
        idx = np.full((11, 4), -1, np.int32)
        values = np.zeros((11, 4, 3), np.float32)
        permutation = rng.permutation(len(src))
        for start in range(0, len(src), chunk):
            rows = permutation[start:start + chunk]
            _merge_incoming_topk(idx, values, dst[rows], src[rows], attr[rows])
        assert idx.tolist() == expected


def test_cache_reuse_and_corruption_rejection(tmp_path):
    data = data_v083(tmp_path)
    data.prepare_neighbors(device="cpu")
    original = np.array(data._pp_neighbors)
    data._neighbors = data._pp_neighbors = data._pp_attrs = None
    assert data._load_neighbors()
    assert np.array_equal(data._pp_neighbors, original)
    modified = np.array(original)
    modified[0, 0, 0] = 1
    np.save(data.pp_cache_dir / PP_INDEX_FILE, modified)
    with pytest.raises(ValueError, match="PP-context cache hash mismatch"):
        data._load_neighbors()
    data.prepare_pp_context(force=True)
    assert np.array_equal(data._pp_neighbors, original)
    manifest = data.pp_cache_dir / PP_MANIFEST_FILE
    payload = json.loads(manifest.read_text())
    payload["signature"]["fanout"] += 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs from native graph"):
        data.prepare_pp_context()


def test_missing_pp_cache_reuses_valid_cosine_cache(tmp_path, monkeypatch):
    data = data_v083(tmp_path)
    data._preparing_core_cache = True
    from nbs_pg.full_task_data import FullTaskData
    FullTaskData.prepare_neighbors(data, device="cpu")
    data._preparing_core_cache = False
    monkeypatch.setattr(data, "_search_index", lambda *args: pytest.fail("cosine cache must be reused"))
    data.prepare_neighbors(device="cpu")
    assert data._pp_neighbors is not None


def test_binarized_membership_and_context_never_read_neighbor_labels(tmp_path):
    data = data_v083(tmp_path)
    data.prepare_neighbors(device="cpu")
    batch = data.batch([4, 0, 5])
    assert torch.equal(batch["targets"], batch["positive_mask"].float())
    assert batch["targets"][0, 0] == 1  # Original rounded .5 still a positive.
    pp_keys = [key for key in batch if key.startswith("pp_")]
    assert set(pp_keys) == {"pp_protein_x", "pp_candidate_go", "pp_candidate_attr",
                            "pp_neighbor_index", "pp_neighbor_attr"}
    data.stores.pseudo_messages.probability = np.array([.99, .51, .9, .6], np.float32)
    changed = data.batch([4, 0, 5])
    assert torch.equal(changed["targets"], batch["targets"])
    for key in pp_keys:
        assert torch.equal(changed[key], batch[key])
    # Gold is allowed in direct OTHER-core anchor_go_edge, but never in P-P
    # context proteins, where it could return a target's label via two hops.
    data.stores.gold_messages.go_idx = np.full_like(data.stores.gold_messages.go_idx, 0)
    changed_gold = data.batch([4, 0, 5])
    for key in pp_keys:
        assert torch.equal(changed_gold[key], batch[key])
    assert data.data_contract()["v083_supervision"].startswith("binary_original_pseudo")
    assert "teacher_targets" not in batch


def test_train_inductive_context_is_identical_and_holdout_excluded(tmp_path):
    data = data_v083(tmp_path, holdout=1)
    data.prepare_neighbors(device="cpu")
    assert not np.isin(data._pp_neighbors[data._pp_neighbors >= 0], data.validation_ids).any()
    input_dir = inference_fixture(tmp_path, data)
    train = data.batch([4, 5])
    independent = data.inference_batch(input_dir, [0, 1])
    for key in train:
        if key.startswith("pp_") or key in ("anchor_x", "anchor_go_edge", "neighbor_index"):
            assert torch.equal(train[key], independent[key]), key
    assert "targets" not in independent


def test_local_ablation_does_not_require_native_graph_or_teacher(tmp_path):
    data = data_v083(tmp_path, enabled=False)
    del data.stores.pp
    data.prepare_neighbors(device="cpu")
    batch = data.batch([4])
    assert not any(key.startswith("pp_") for key in batch)
    assert batch["targets"].sum() == 3
    assert data.data_contract()["pp_context_signature"] is None


def test_original_candidate_rank_not_array_position_selects_context_go(tmp_path):
    data = data_v083(tmp_path)
    attrs = np.array(data.stores.candidate_messages.edge_attr)
    attrs[:, 2] = np.tile([.1, .9], 6)
    data.stores.candidate_messages.edge_attr = attrs
    go, values = data._pp_candidate_rows(np.array([4, 5]))
    assert go.tolist() == [[5], [5]]
    assert np.all(values[..., 2] == .9)


def test_empty_native_context_is_valid_zero_padding(tmp_path):
    data = data_v083(tmp_path)
    for store in data.stores.pp.values():
        store.edge_index = np.empty((2, 0), np.int64)
        store.edge_attr = np.empty((0, 3), np.float32)
    data._pp_signature = data._pp_source_signature()
    data.prepare_neighbors(device="cpu")
    batch = data.batch([4])
    assert batch["pp_protein_x"].shape == (0, 2)
    assert batch["pp_candidate_go"].shape == (0, 1)
    assert (batch["pp_neighbor_index"] == -1).all()


def test_runtime_provenance_accepts_only_explicit_v083_binary_consumer(tmp_path):
    from nbs_pg.local_loader import _validate_supervision_provenance
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "gold_annotations_manifest.json").write_text(json.dumps({
        "task": "bp", "role": "core", "mode": "train", "source": {"label_key": "prop_annotations"}}))
    config = {"task": "bp", "stage": {"name": "nbs_v083_graph"},
              "full_task": {"weak_target_mode": "binary_membership"},
              "supervision_contract": {
                  "core_gold": {"role": "core", "dataset_mode": "train", "metadata_label_key": "prop_annotations"},
                  "weak_pseudo": {"use_probability_as_soft_target": False}}}
    manifest = {
        "roles": [{"role": "core", "dataset_mode": "train"}, {"role": "weak", "dataset_mode": "exp_train"}],
        "rare_definition": {"training_annotation_key": "prop_annotations"},
        "model_semantics": {"modelout": {
            "prediction_key": "modelout::mix_expert_base_anchor::decoderprob::expert",
            "decoder_prob_source": "expert", "external_probability_used": True}},
        "weak_pseudo_targets": {"role": "weak", "comparison": ">", "threshold": .5,
                                "edge_attr_columns": ["modelout_probability"]}}
    inverted = {"indices": {"candidate": {}, "pseudo": {"payloads": {"probability": "pseudo.npy"}},
                            "gold": {"source_edge_index": str(gold_dir / "edge.npy")}}}

    def validate():
        _validate_supervision_provenance(config, weak_manifest=manifest,
            weak_manifest_path=tmp_path / "weak.json", inverted=inverted,
            inverted_manifest=tmp_path / "inverted.json")

    validate()
    # Exercise the shipped defaults, not just a fixture-only mode switch.
    fixture_config = config
    manifest["model_semantics"]["protein_go_selector"] = {"scope": "full_task"}
    inverted["indices"]["candidate"]["fixed_degree"] = 512
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for variant in ("no_graph", "local", "graph"):
        config = json.loads((root / "configs" / f"bp_full_task_v0.8.3_{variant}.json").read_text())
        validate()
    config = fixture_config
    config["stage"]["name"] = "nbs_v081_graph"
    with pytest.raises(ValueError, match="outside explicit v083"):
        validate()
    config["stage"]["name"] = "nbs_v083_graph"
    config["full_task"].pop("weak_target_mode")
    with pytest.raises(ValueError, match="outside explicit v083"):
        validate()
    # Existing soft-target consumers retain their old default without a mode.
    config["supervision_contract"]["weak_pseudo"]["use_probability_as_soft_target"] = True
    config["stage"]["name"] = "nbs_v081_graph"
    validate()
    manifest["weak_pseudo_targets"]["threshold"] = .2
    with pytest.raises(ValueError, match="threshold"):
        validate()
