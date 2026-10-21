from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import nbs_pg.local_loader as module
from nbs_pg.episode import NBSGlobalEpisode
from nbs_pg.local_loader import LatenceNBSLocalGraphMaterializer, LatenceNBSStores, NBSLocalGraphSamplingConfig

pytestmark = pytest.mark.skipif(
    not module.PYG_AVAILABLE,
    reason="torch_geometric is required",
)


class PPStore:
    def __init__(self, edge): self.edge = np.asarray(edge, np.int64)
    def sample(self, keys, fanout, *, rng):
        keep = np.isin(self.edge[0], list(keys)) | np.isin(self.edge[1], list(keys))
        edge = self.edge[:, keep]
        return edge, np.ones((edge.shape[1], 3), np.float32)


class CandidateStore:
    def gather(self, proteins, *, topk=None):
        p = np.asarray(list(proteins), np.int64)
        return np.stack([p, np.zeros(len(p), np.int64)]), np.ones((len(p), 3), np.float32)


class GoldStore:
    def gather(self, proteins, *, topk=None):
        p = np.asarray(list(proteins), dtype=np.int64)
        if p.size == 0:
            return np.empty((2, 0), dtype=np.int64)
        return np.stack([p, np.zeros(p.size, dtype=np.int64)], axis=0)


class Features:
    feature_dim = 4
    def gather(self, p): return np.zeros((len(p), 4), np.float32)


class Boxes:
    num_go = 3
    def gather(self, g):
        return {"center": np.zeros((len(g), 2), np.float32), "offset": np.ones((len(g), 2), np.float32), "stats": np.zeros((len(g), 6), np.float32)}


class GORel:
    def __init__(self, edge): self.edge = np.asarray(edge, np.int64)
    def sample(self, sources, fanout, *, rng):
        return self.edge[np.isin(self.edge[:, 0], list(sources))][:fanout]


def test_real_pyg_local_batch_is_valid_and_masked():
    empty_pp = PPStore(np.empty((2, 0), np.int64))
    stores = LatenceNBSStores(
        registry=SimpleNamespace(), features=Features(), episode_sampler=SimpleNamespace(),
        candidate_messages=CandidateStore(), gold_messages=GoldStore(), pseudo_messages=None,
        pp={"ppi": empty_pp, "similar_to": empty_pp, "weak_to_core": PPStore([[2], [1]])},
        full_boxes=Boxes(),
        go_relations={
            "is_a": GORel([[0, 1]]), "has_child": GORel([[1, 0]]),
            "part_of": GORel(np.empty((0, 2), np.int64)), "has_part": GORel(np.empty((0, 2), np.int64)),
        },
        task_to_ontology=np.asarray([0, 1]), feature_dim=4, num_task_go=2,
    )
    episode = NBSGlobalEpisode(
        query_go_idx=np.asarray([0]), query_ontology_go_idx=np.asarray([0]),
        seed_protein_idx=np.asarray([1]), seed_query_idx=np.asarray([0]),
        candidate_protein_idx=np.asarray([2]), base_logits=np.zeros((1, 1), np.float32),
        candidate_evidence=np.zeros((1, 1, 3), np.float32), query_go_frequency=np.ones(1, np.float32),
        labels=np.ones((1, 1), np.float32), mask=np.ones((1, 1), bool),
        confidence=np.ones((1, 1), np.float32), pseudo_mask=np.zeros((1, 1), bool),
        supervision_weight=np.ones((1, 1), np.float32),
    )
    batch = LatenceNBSLocalGraphMaterializer(
        stores,
        NBSLocalGraphSamplingConfig(pp_hops=1, ppi_fanouts=(0,), similar_to_fanouts=(0,), weak_to_core_fanouts=(4,), steps_per_epoch_per_rank=1),
    ).materialize(episode, seed=11)
    assert batch.graph.validate(raise_on_error=True)
    # Current candidate/query direct edge is removed in both directions.
    forward = batch.graph[("protein", "backbone_rare_candidate", "go")].edge_index
    reverse = batch.graph[("go", "candidate_of", "protein")].edge_index
    assert not ((forward[0] == batch.query.candidate_protein_index[0]) & (forward[1] == batch.query.query_go_index[0])).any()
    assert not ((reverse[1] == batch.query.candidate_protein_index[0]) & (reverse[0] == batch.query.query_go_index[0])).any()
