import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nbs_pg.full_task_data_v084 import FullTaskDataV084, RELATIONS, POOL_MANIFEST
from test_full_task_data_v080 import fixture_data, inference_fixture


def data_v084(tmp_path, *, holdout=0, variant="fixed", block=3, dropout=0.0):
    old = fixture_data(tmp_path, holdout=holdout)
    def relation(edges, key_axis):
        edge = np.asarray(edges, np.int64)
        values = np.ones((edge.shape[1], 3), np.float32)
        values[:, 2] = 1 / (np.arange(edge.shape[1]) + 1)
        return SimpleNamespace(edge_index=edge, edge_attr=values, key_axis=key_axis)
    old.stores.pp = {
        "ppi": relation([[5, 0, 1, 2, 3, 5], [0, 1, 2, 3, 0, 0]], 0),
        "similar_to": relation([[2, 3, 0], [0, 1, 2]], 1),
        "weak_to_core": relation([[4, 5, 4, 5], [0, 1, 2, 3]], 0),
    }
    old.stores.task_to_ontology = np.arange(old.num_task_go)
    old.config["full_task"]["sampler"] = {
        "mode": variant, "seed": 8084, "first_hop": 2, "second_hop": 2,
        "native_pool_per_relation": 3, "retrieval_pool_size": 2,
        "candidate_topk": 1, "stable_fraction": .5, "native_dropout": dropout,
        "scan_block_size": block, "cache_dir": str(tmp_path / "pools"),
    }
    return FullTaskDataV084(old.config, stores=old.stores)


def global_edges(batch):
    ids = batch["sampled_global_ids"].numpy()
    edges = batch["sampled_edge_index"].numpy()
    return {(int(ids[s]), int(ids[t]), int(r)) for (s, t), r in
            zip(edges.T, batch["sampled_edge_type"].tolist())}


def test_native_direction_explicit_inverse_and_dedup(tmp_path):
    data = data_v084(tmp_path)
    data.prepare_neighbors()
    for code, src, dst in ((0, 5, 0), (1, 2, 0), (2, 4, 0), (3, 0, 4)):
        lookup, neighbors, attr = data._native[code]
        values = neighbors[lookup[dst]].tolist()
        assert src in values
        assert values.count(src) == 1
    assert RELATIONS == ("ppi", "similar_to", "weak_to_core", "core_to_weak", "cosine")
    report = json.loads((data.pool_dir / POOL_MANIFEST).read_text())
    assert report["coverage"]["ppi"]["retained_edges"] == 5
    assert report["coverage"]["core_to_weak"]["centers"] == 2


def test_fixed_is_stable_dynamic_changes_and_resume_is_stateless(tmp_path):
    data = data_v084(tmp_path, variant="dynamic")
    data.prepare_neighbors()
    samples = []
    for step in range(12):
        data.set_sampling_context(step, 1, True)
        samples.append(global_edges(data.batch([4])))
    assert len({frozenset(x) for x in samples}) > 1
    data.set_sampling_context(3, 1, True)
    assert global_edges(data.batch([4])) == samples[3]
    data.mode = "fixed"
    fixed = []
    for step in (1, 2, 6):
        data.set_sampling_context(step, 1, True)
        fixed.append(global_edges(data.batch([4])))
    assert fixed[0] == fixed[1] == fixed[2]


def test_native_dropout_matches_controls_and_external_fallback_budget(tmp_path):
    data = data_v084(tmp_path, dropout=1)
    data.prepare_neighbors()
    data.set_sampling_context(12, 0, True)
    fixed = data.batch([4])
    data.mode = "dynamic"
    dynamic = data.batch([4])
    assert fixed["sampled_native_dropout_fraction"] == dynamic["sampled_native_dropout_fraction"] == 1
    assert fixed["sampled_fallback_fraction"] == 1
    assert fixed["sampled_first_neighbors"] == 2
    for _, receiver, relation in global_edges(fixed):
        if receiver == 4:
            assert relation == 4
    assert set(fixed["sampler_diagnostics"]) == set(dynamic["sampler_diagnostics"])


def test_binary_targets_and_seed_labels_excluded_from_every_forward_incidence(tmp_path):
    data = data_v084(tmp_path)
    data.prepare_neighbors()
    batch = data.batch([4, 0])
    assert torch.equal(batch["targets"], batch["positive_mask"].float())
    assert batch["targets"][0, 0] == 1  # .5 rounding preserves original membership
    nodes = batch["sampled_global_ids"]
    for key in ("sampled_gold_edge", "sampled_pseudo_edge"):
        labeled_global = nodes[batch[key][0]]
        assert not torch.isin(labeled_global, torch.tensor([4, 0])).any()
    anchor_global = nodes[batch["sampled_anchor_index"]]
    assert not torch.isin(anchor_global[batch["anchor_go_edge"][0]], torch.tensor([4, 0])).any()
    # Other weak membership participates in graph evidence, never its magnitude.
    first = data.batch([4])
    assert first["sampled_pseudo_edge"].shape[1] > 0
    data.stores.pseudo_messages.probability = np.array([.99, .51, .9, .6], np.float32)
    changed = data.batch([4])
    for key in ("sampled_pseudo_edge", "sampled_candidate_attr", "targets"):
        assert torch.equal(first[key], changed[key])


def test_holdout_labels_never_enter_context_even_via_two_hops(tmp_path):
    data = data_v084(tmp_path, holdout=1)
    data.prepare_neighbors()
    batch = data.batch([4, *data.validation_ids])
    nodes = batch["sampled_global_ids"]
    for key in ("sampled_gold_edge", "sampled_pseudo_edge"):
        assert not torch.isin(nodes[batch[key][0]], torch.from_numpy(data.validation_ids)).any()
    for lookup, neighbors, _ in data._native:
        assert not np.isin(neighbors, data.validation_ids).any()


def test_original_cosine_loss_support_fixed_across_sampler_and_step(tmp_path):
    data = data_v084(tmp_path, variant="dynamic")
    data.prepare_neighbors()
    reference = data.batch([4])
    for step in (1, 5, 12):
        data.set_sampling_context(step, 0, True)
        changed = data.batch([4])
        for key in ("loss_anchor_x", "loss_anchor_go_edge", "loss_neighbor_index", "loss_neighbor_attr"):
            assert torch.equal(reference[key], changed[key])
    assert reference["loss_neighbor_index"].shape[1] == data.k


def _receiver_edges(batch, receiver):
    return {edge for edge in global_edges(batch) if edge[1] == receiver}


def test_inductive_graph_is_deterministic_and_batch_partition_invariant(tmp_path):
    data = data_v084(tmp_path, holdout=1, variant="dynamic", dropout=.75)
    data.prepare_neighbors()
    input_dir = inference_fixture(tmp_path, data)
    data.set_sampling_context(37, 1, True)
    together = data.inference_batch(input_dir, [0, 1])
    assert (data._sampling_step, data._sampling_rank, data._sampling_training) == (37, 1, True)
    assert "targets" not in together
    for row in (0, 1):
        alone = data.inference_batch(input_dir, [row])
        seed = -row - 1
        assert _receiver_edges(together, seed) == _receiver_edges(alone, seed)
        first = {edge[0] for edge in _receiver_edges(alone, seed)}
        for node in first:
            assert _receiver_edges(together, node) == _receiver_edges(alone, node)
    assert together["sampled_fallback_fraction"] == 1
    assert len(data._inference[str(input_dir)]["v084_neighbors"][0]) == data.retrieval_pool_size


def test_native_cache_reuse_hash_corruption_and_missing_hash_rejected(tmp_path, monkeypatch):
    data = data_v084(tmp_path)
    data.prepare_neighbors()
    monkeypatch.setattr(data, "_search_index", lambda *args: pytest.fail("must reuse pools"))
    data._native = None
    data.ensure_prepared()
    data.prepare_pools()
    name = data.pool_dir / "ppi_neighbors.i32.npy"
    changed = np.array(np.load(name))
    changed[0, 0] = 0
    np.save(name, changed)
    data._native = None
    with pytest.raises(ValueError, match="hash mismatch"):
        data.ensure_prepared()
    manifest = data.pool_dir / POOL_MANIFEST
    payload = json.loads(manifest.read_text())
    del payload["arrays"]["ppi_neighbors.i32.npy"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="every candidate array hash"):
        data.ensure_prepared()


def test_relation_pool_results_do_not_depend_on_scan_block_size(tmp_path):
    data = data_v084(tmp_path, block=1)
    data.prepare_neighbors()
    expected = [np.array(x[1]) for x in data._native]
    data.block_size = 1000
    data.prepare_pools(force=True)
    for original, (_, changed, _) in zip(expected, data._native):
        assert np.array_equal(original, changed)


def test_missing_native_relations_have_complete_diagnostics_and_cosine_context(tmp_path):
    data = data_v084(tmp_path)
    data.stores.pp = {}
    data._pool_signature = data._build_pool_signature()
    data.prepare_neighbors()
    batch = data.batch([4])
    assert batch["sampled_native_missing_fraction"] == 1
    assert batch["sampled_fallback_fraction"] == 1
    for relation in RELATIONS:
        assert f"sampled_relation_{relation}_edges" in batch["sampler_diagnostics"]


def test_modifying_supervision_seed_membership_changes_no_forward_evidence(tmp_path):
    data = data_v084(tmp_path)
    data.prepare_neighbors()
    before = data.batch([4, 0])
    gold = np.array(data.stores.gold_messages.go_idx)
    gold[:2] = [0, 5]
    data.stores.gold_messages.go_idx = gold
    pseudo = np.array(data.stores.pseudo_messages.go_idx)
    pseudo[:3] = [2, 3, 5]
    data.stores.pseudo_messages.go_idx = pseudo
    after = data.batch([4, 0])
    assert not torch.equal(before["positive_mask"], after["positive_mask"])
    for key, value in before.items():
        if key in {"targets", "positive_mask", "sampler_diagnostics"}:
            continue
        assert torch.equal(value, after[key]), key


def test_shipped_binary_configs_pass_actual_supervision_validator(tmp_path):
    from pathlib import Path
    from nbs_pg.local_loader import _validate_supervision_provenance
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "gold_annotations_manifest.json").write_text(json.dumps({
        "task": "bp", "role": "core", "mode": "train", "source": {"label_key": "prop_annotations"}}))
    manifest = {
        "roles": [{"role": "core", "dataset_mode": "train"}, {"role": "weak", "dataset_mode": "exp_train"}],
        "rare_definition": {"training_annotation_key": "prop_annotations"},
        "model_semantics": {"modelout": {
            "prediction_key": "modelout::mix_expert_base_anchor::decoderprob::expert",
            "decoder_prob_source": "expert", "external_probability_used": True},
            "protein_go_selector": {"scope": "full_task"}},
        "weak_pseudo_targets": {"role": "weak", "comparison": ">", "threshold": .5,
                                "edge_attr_columns": ["modelout_probability"]}}
    inverted = {"indices": {"candidate": {"fixed_degree": 512},
        "pseudo": {"payloads": {"probability": "pseudo.npy"}},
        "gold": {"source_edge_index": str(gold_dir / "edge.npy")}}}
    root = Path(__file__).resolve().parents[1]
    for variant in ("fixed", "dynamic"):
        config = json.loads((root / "configs" / f"bp_full_task_v0.8.4_{variant}.json").read_text())
        assert config["supervision_contract"]["weak_pseudo"]["use_probability_as_soft_target"] is False
        _validate_supervision_provenance(config, weak_manifest=manifest,
            weak_manifest_path=tmp_path / "weak.json", inverted=inverted,
            inverted_manifest=tmp_path / "inverted.json")
