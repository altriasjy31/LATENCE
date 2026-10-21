"""Real mmap/pool tests for inductive three-layer relation-specific blocks."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nbs_pg.full_task_data import FullTaskData, _sha256
from nbs_pg.full_task_data_v088 import FullTaskDataV088, RELATIONS, NATIVE_RELATIONS
from test_full_task_data_v084 import data_v084, inference_fixture


def _data(tmp_path, *, holdout=1, fanouts=None, max_nodes=100_000, duplicate_relation=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    old = data_v084(tmp_path, holdout=holdout, variant="dynamic")
    if duplicate_relation:
        store = old.stores.pp["similar_to"]
        old.stores.pp["similar_to"] = SimpleNamespace(
            edge_index=np.concatenate((store.edge_index, np.array([[5], [0]])), axis=1),
            edge_attr=np.concatenate((store.edge_attr, np.ones((1, 3), np.float32))), key_axis=1)
        old._pool_signature = old._build_pool_signature()
    old.prepare_neighbors()
    config = deepcopy(old.config)
    config["full_task"]["sampler"].update(
        fanouts=fanouts or [{name: 1 for name in RELATIONS},
                           {name: 1 for name in RELATIONS}, {"cosine": 2}],
        max_nodes=max_nodes)
    data = FullTaskDataV088(config, stores=old.stores)
    data.ensure_prepared()
    return data


def _equal(left, right, path="batch"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _equal(left[key], right[key], path + "." + key)
    elif isinstance(left, list):
        assert len(left) == len(right)
        for i, (a, b) in enumerate(zip(left, right)):
            _equal(a, b, path + f"[{i}]")
    else:
        assert torch.equal(left, right), path


def _edges(block):
    return {(int(block["src_global_ids"][s]), int(block["dst_global_ids"][t]), int(r))
            for (s, t), r in zip(block["edge_index"].T.tolist(), block["edge_type"].tolist())}


def _assert_blocks(batch, queries):
    assert len(batch["blocks"]) == 3
    assert torch.equal(batch["sampled_global_ids"], batch["blocks"][0]["src_global_ids"])
    assert batch["blocks"][-1]["dst_global_ids"].tolist() == queries
    for left, right in zip(batch["blocks"], batch["blocks"][1:]):
        assert torch.equal(left["dst_global_ids"], right["src_global_ids"])
    for block in batch["blocks"]:
        assert torch.equal(block["src_global_ids"][block["dst_in_src"]], block["dst_global_ids"])
        assert block["edge_index"].shape == (2, len(block["edge_type"]))
        assert block["edge_attr"].shape == (len(block["edge_type"]), 3)
        assert len(block["src_global_ids"].unique()) == len(block["src_global_ids"])
        assert len(_edges(block)) == len(block["edge_type"])
        for _, target, relation in _edges(block):
            if target < 0:
                assert relation == 4


def test_global_macro_queries_are_absent_from_every_support_layer(tmp_path):
    data = _data(tmp_path, holdout=0)
    batch = data.batch([4], supervision_seed_ids=[4, 0, 5])
    _assert_blocks(batch, [-5])
    nodes = batch["sampled_global_ids"]
    assert not torch.isin(nodes[nodes >= 0], torch.tensor([4, 0, 5])).any()
    for block in batch["blocks"]:
        assert not torch.isin(block["src_global_ids"], torch.tensor([4, 0, 5])).any()
    for key in ("sampled_gold_edge", "sampled_pseudo_edge"):
        source = nodes[batch[key][0]]
        assert (source >= 0).all()
        assert not torch.isin(source, torch.tensor([4, 0, 5])).any()
    assert torch.equal(batch["targets"], batch["positive_mask"].float())
    assert batch["targets"][0, 0] == 1  # stored .5 belongs to the original >.5 CSR
    assert not {"anchor_go_edge", "neighbor_index", "loss_anchor_go_edge"}.intersection(batch)
    assert torch.equal(batch["sampled_node_type"] == 2, nodes < 0)


def test_real_three_hop_dependency_and_block_indices(tmp_path):
    fanouts = [{"ppi": 1, "cosine": 1}, {"ppi": 1, "cosine": 1}, {"cosine": 1}]
    data = _data(tmp_path, holdout=0, fanouts=fanouts)
    n = data.registry.num_proteins
    native = []
    for relation in NATIVE_RELATIONS:
        neighbors = np.full((n, data.pool_size), -1, np.int32)
        attrs = np.zeros((n, data.pool_size, 3), np.float32)
        if relation == "ppi":
            neighbors[0, 0], neighbors[1, 0] = 1, 2
            attrs[0, 0] = attrs[1, 0] = 1
        native.append((np.arange(n), neighbors, attrs))
    data._native = native
    data._retrieval_neighbors = np.full((n, 2), -1, np.int32)
    data._retrieval_attrs = np.zeros((n, 2, 3), np.float32)
    data._retrieval_neighbors[4, 0] = 0
    data._retrieval_attrs[4, 0] = 1
    batch = data.batch([4])
    _assert_blocks(batch, [-5])
    assert (2, 1, 0) in _edges(batch["blocks"][0])
    assert (1, 0, 0) in _edges(batch["blocks"][1])
    assert (0, -5, 4) in _edges(batch["blocks"][2])
    assert 2 not in batch["blocks"][1]["src_global_ids"].tolist()
    assert 1 not in batch["blocks"][2]["src_global_ids"].tolist()


def test_same_pair_retains_independent_native_relation_identity(tmp_path):
    fanouts = [{name: 2 for name in RELATIONS}, {name: 2 for name in RELATIONS}, {"cosine": 2}]
    data = _data(tmp_path, holdout=0, fanouts=fanouts, duplicate_relation=True)
    batch = data.batch([4])
    all_edges = set().union(*map(_edges, batch["blocks"]))
    assert (5, 0, 0) in all_edges and (5, 0, 1) in all_edges
    _assert_blocks(batch, [-5])


def test_dynamic_streams_resume_without_changing_global_rng(tmp_path):
    fanouts = [{"cosine": 1}, {"cosine": 1}, {"cosine": 1}]
    data = _data(tmp_path, holdout=0, fanouts=fanouts)
    torch.manual_seed(711)
    np.random.seed(711)
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    samples = []
    for step in range(20):
        data.set_sampling_context(step, 1, True)
        samples.append(data.batch([4]))
    assert len({frozenset(_edges(b["blocks"][-1])) for b in samples}) > 1
    data.set_sampling_context(7, 1, True)
    _equal(samples[7], data.batch([4]))
    assert torch.equal(torch_state, torch.get_rng_state())
    changed = np.random.get_state()
    assert numpy_state[0] == changed[0] and np.array_equal(numpy_state[1], changed[1])
    assert numpy_state[2:] == changed[2:]
    data.set_sampling_context(100, 99, False)
    fixed = data.batch([4])
    data.set_sampling_context(22, 2, False)
    _equal(fixed, data.batch([4]))


def test_external_blocks_are_deterministic_and_partition_invariant(tmp_path):
    data = _data(tmp_path)
    input_dir = inference_fixture(tmp_path, data)
    data.set_sampling_context(71, 1, True)
    together = data.inference_batch(input_dir, [1, 0])
    _assert_blocks(together, [-2, -1])
    assert (data._sampling_step, data._sampling_rank, data._sampling_training) == (71, 1, True)
    assert "targets" not in together and "positive_mask" not in together
    for row in (0, 1):
        alone = data.inference_batch(input_dir, [row])
        for all_block, one_block in zip(together["blocks"], alone["blocks"]):
            destinations = set(one_block["dst_global_ids"].tolist())
            assert {e for e in _edges(all_block) if e[1] in destinations} == _edges(one_block)
    assert not torch.isin(together["sampled_global_ids"], torch.from_numpy(data.validation_ids)).any()


def test_training_and_external_views_use_the_same_ghost_engine(tmp_path):
    data = _data(tmp_path)
    input_dir = inference_fixture(tmp_path, data)
    # The synthetic external fixture copies weak 4/5's features. Remove those
    # originals from the shared reference graph in BOTH views, as required for
    # a fair identity-relabeling comparison of a genuinely absent query.
    data._permitted[data.weak_ids] = False
    train = data.batch([4, 5])
    external = data.inference_batch(input_dir, [0, 1])
    for left, right in zip(train["blocks"], external["blocks"]):
        for key in ("src_global_ids", "dst_global_ids"):
            renamed = left[key].clone()
            renamed[left[key] == -5], renamed[left[key] == -6] = -1, -2
            assert torch.equal(renamed, right[key])
        for key in ("edge_index", "edge_type", "dst_in_src"):
            assert torch.equal(left[key], right[key])
        torch.testing.assert_close(left["edge_attr"], right["edge_attr"], atol=3e-4, rtol=3e-4)
    for key in ("sampled_gold_edge", "sampled_pseudo_edge", "sampled_node_type",
                "sampled_candidate_go", "sampled_candidate_attr"):
        assert torch.equal(train[key], external[key]), key
    torch.testing.assert_close(train["sampled_protein_x"], external["sampled_protein_x"], atol=3e-4, rtol=3e-4)


def test_labels_of_global_queries_and_holdout_never_enter_forward_inputs(tmp_path):
    data = _data(tmp_path)
    heldout = data.validation_ids.tolist()
    global_ids = np.unique([4, 5, 0, *heldout])
    before = data.batch([4], supervision_seed_ids=global_ids)
    data.stores.gold_messages.go_idx = np.zeros_like(data.stores.gold_messages.go_idx)
    # Only change excluded core labels, retaining all allowed support labels.
    gold_original = np.load(tmp_path / "gold_go.npy")
    for protein in data.anchor_core_ids:
        if protein not in global_ids:
            a, b = data.stores.gold_messages.indptr[protein:protein + 2]
            data.stores.gold_messages.go_idx[a:b] = gold_original[a:b]
    data.stores.pseudo_messages.go_idx = np.full_like(data.stores.pseudo_messages.go_idx, 3)
    data.stores.pseudo_messages.probability = np.full_like(data.stores.pseudo_messages.probability, .75)
    after = data.batch([4], supervision_seed_ids=global_ids)
    assert not torch.equal(before["positive_mask"], after["positive_mask"])
    _equal({k: v for k, v in before.items() if k not in {"positive_mask", "targets"}},
           {k: v for k, v in after.items() if k not in {"positive_mask", "targets"}})


def test_resource_guard_fails_without_graph_truncation_or_large_feature_gather(tmp_path, monkeypatch):
    data = _data(tmp_path, holdout=0, max_nodes=1)
    original = data.stores.features.gather
    sizes = []
    def gather(ids):
        sizes.append(len(ids))
        return original(ids)
    monkeypatch.setattr(data.stores.features, "gather", gather)
    with pytest.raises(RuntimeError, match="max_nodes exceeded"):
        data.batch([4])
    assert sizes == [1]  # query array only; support features were never gathered


def test_v084_pool_files_reused_without_mutation(tmp_path):
    data = _data(tmp_path)
    before = {p.name: _sha256(p) for p in data.pool_dir.iterdir() if p.is_file()}
    data.batch([4])
    data._native = None
    data.ensure_prepared()
    assert before == {p.name: _sha256(p) for p in data.pool_dir.iterdir() if p.is_file()}
    contract = data.data_contract()
    assert "v084_sampling" not in contract
    assert contract["v088_sampling"]["fanouts_outer_to_inner"] == list(data.fanouts)


@pytest.mark.parametrize("seeds", [[4, 4], [0, 5], [-1, 4], [4, 6], [4., 0.], [[4, 0]], []])
def test_invalid_global_supervision_ids_are_rejected(tmp_path, seeds):
    data = _data(tmp_path)
    with pytest.raises(ValueError, match="global supervision"):
        data.batch([4], supervision_seed_ids=seeds)


@pytest.mark.parametrize("fanouts", [[], [{"cosine": 1}], [{"bad": 1}] * 3,
    [{"cosine": -1}] * 3, [{"cosine": True}] * 3, [{"cosine": 3}] * 3, [{}] * 3])
def test_invalid_layer_fanouts_are_rejected(tmp_path, fanouts):
    data = _data(tmp_path)
    config = deepcopy(data.config)
    config["full_task"]["sampler"]["fanouts"] = fanouts
    with pytest.raises(ValueError):
        FullTaskDataV088(config, stores=data.stores)
