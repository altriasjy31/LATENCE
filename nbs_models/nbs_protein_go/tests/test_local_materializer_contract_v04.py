from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import nbs_pg.local_loader as module
from nbs_pg.episode import NBSGlobalEpisode
from nbs_pg.local_loader import (
    LatenceNBSLocalGraphMaterializer,
    LatenceNBSStores,
    NBSLocalGraphSamplingConfig,
)


class PPStore:
    def __init__(self, edge):
        self.edge = np.asarray(edge, dtype=np.int64)

    def sample(self, keys, fanout, *, rng):
        if fanout <= 0:
            return np.empty((2, 0), np.int64), np.empty((0, 3), np.float32)
        keep = np.isin(self.edge[0], np.asarray(list(keys))) | np.isin(self.edge[1], np.asarray(list(keys)))
        edge = self.edge[:, keep]
        return edge, np.ones((edge.shape[1], 3), np.float32)


class CandidateStore:
    def gather(self, proteins, *, topk=None):
        p = np.asarray(list(proteins), dtype=np.int64)
        edge = np.stack([p, np.zeros(p.size, np.int64)], axis=0)
        return edge, np.tile(np.asarray([[0.8, 0.7, 1.0]], np.float32), (p.size, 1))


class GoldStore:
    def gather(self, proteins, *, topk=None):
        p = np.asarray(list(proteins), dtype=np.int64)
        if p.size == 0:
            return np.empty((2, 0), dtype=np.int64)
        return np.stack([p, np.zeros(p.size, dtype=np.int64)], axis=0)


class Features:
    feature_dim = 4

    def gather(self, proteins):
        return np.asarray(proteins, np.float32)[:, None].repeat(4, axis=1)


class Boxes:
    num_go = 3

    def gather(self, go):
        n = len(go)
        return {
            "center": np.zeros((n, 2), np.float32),
            "offset": np.ones((n, 2), np.float32),
            "stats": np.zeros((n, 6), np.float32),
        }


class GORel:
    def __init__(self, edge):
        self.edge = np.asarray(edge, np.int64)

    def sample(self, sources, fanout, *, rng):
        keep = np.isin(self.edge[:, 0], np.asarray(list(sources)))
        return self.edge[keep][:fanout]


class FakeGraph:
    pass


def test_materializer_preserves_weak_to_core_direction_and_localizes_candidate(monkeypatch):
    captured = {}

    def fake_builder(*args, **kwargs):
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(module, "PYG_AVAILABLE", True)
    monkeypatch.setattr(module, "build_nbs_protein_go_heterodata", fake_builder)
    monkeypatch.setattr(module, "mask_candidate_evidence_edges", lambda graph, *a, **k: graph)
    stores = LatenceNBSStores(
        registry=SimpleNamespace(),
        features=Features(),
        episode_sampler=SimpleNamespace(),
        candidate_messages=CandidateStore(),
        gold_messages=GoldStore(),
        pseudo_messages=None,
        pp={
            "ppi": PPStore(np.empty((2, 0), np.int64)),
            "similar_to": PPStore(np.empty((2, 0), np.int64)),
            "weak_to_core": PPStore(np.asarray([[2], [1]], np.int64)),
        },
        full_boxes=Boxes(),
        go_relations={
            "is_a": GORel(np.asarray([[0, 1]], np.int64)),
            "has_child": GORel(np.asarray([[1, 0]], np.int64)),
            "part_of": GORel(np.empty((0, 2), np.int64)),
            "has_part": GORel(np.empty((0, 2), np.int64)),
        },
        task_to_ontology=np.asarray([0, 1], np.int64),
        feature_dim=4,
        num_task_go=2,
    )
    episode = NBSGlobalEpisode(
        query_go_idx=np.asarray([0]),
        seed_protein_idx=np.asarray([1]),
        seed_query_idx=np.asarray([0]),
        candidate_protein_idx=np.asarray([2]),
        base_logits=np.zeros((1, 1), np.float32),
        candidate_evidence=np.zeros((1, 1, 3), np.float32),
        query_go_frequency=np.ones(1, np.float32),
        labels=np.ones((1, 1), np.float32),
        mask=np.ones((1, 1), bool),
        confidence=np.ones((1, 1), np.float32),
        pseudo_mask=np.zeros((1, 1), bool),
        supervision_weight=np.ones((1, 1), np.float32),
        query_ontology_go_idx=np.asarray([0]),
    )
    materializer = LatenceNBSLocalGraphMaterializer(
        stores,
        NBSLocalGraphSamplingConfig(
            steps_per_epoch_per_rank=1,
            pp_hops=1,
            ppi_fanouts=(0,),
            similar_to_fanouts=(0,),
            weak_to_core_fanouts=(4,),
            candidate_message_topk=1,
            go_hops=1,
        ),
    )
    batch = materializer.materialize(episode, seed=7)
    weak = captured["weak_to_core_edge_index"]
    assert tuple(weak[:, 0].tolist()) == (1, 0)  # global 2->1 remapped to local 1->0
    assert captured["backbone_candidate_protein_go_edge_index"].shape[1] == 2
    assert batch.query.query_go_index.tolist() == [0]


def test_materializer_adds_gold_messages_for_sampled_core_neighbors(monkeypatch):
    captured = {}

    def fake_builder(*args, **kwargs):
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(module, "PYG_AVAILABLE", True)
    monkeypatch.setattr(module, "build_nbs_protein_go_heterodata", fake_builder)
    monkeypatch.setattr(module, "mask_candidate_evidence_edges", lambda graph, *a, **k: graph)
    stores = LatenceNBSStores(
        registry=SimpleNamespace(),
        features=Features(),
        episode_sampler=SimpleNamespace(),
        candidate_messages=CandidateStore(),
        gold_messages=GoldStore(),
        pseudo_messages=None,
        pp={
            "ppi": PPStore(np.empty((2, 0), np.int64)),
            "similar_to": PPStore(np.empty((2, 0), np.int64)),
            "weak_to_core": PPStore(np.asarray([[2], [1]], np.int64)),
        },
        full_boxes=Boxes(),
        go_relations={
            "is_a": GORel(np.asarray([[0, 1]], np.int64)),
            "has_child": GORel(np.asarray([[1, 0]], np.int64)),
            "part_of": GORel(np.empty((0, 2), np.int64)),
            "has_part": GORel(np.empty((0, 2), np.int64)),
        },
        task_to_ontology=np.asarray([0, 1], np.int64),
        feature_dim=4,
        num_task_go=2,
    )
    episode = NBSGlobalEpisode(
        query_go_idx=np.asarray([0]),
        seed_protein_idx=np.asarray([0]),
        seed_query_idx=np.asarray([0]),
        candidate_protein_idx=np.asarray([2]),
        base_logits=np.zeros((1, 1), np.float32),
        candidate_evidence=np.zeros((1, 1, 3), np.float32),
        query_go_frequency=np.ones(1, np.float32),
        labels=np.ones((1, 1), np.float32),
        mask=np.ones((1, 1), bool),
        confidence=np.ones((1, 1), np.float32),
        pseudo_mask=np.zeros((1, 1), bool),
        supervision_weight=np.ones((1, 1), np.float32),
        query_ontology_go_idx=np.asarray([0]),
    )
    LatenceNBSLocalGraphMaterializer(
        stores,
        NBSLocalGraphSamplingConfig(
            steps_per_epoch_per_rank=1,
            pp_hops=1,
            ppi_fanouts=(0,),
            similar_to_fanouts=(0,),
            weak_to_core_fanouts=(4,),
            candidate_message_topk=1,
            go_hops=1,
        ),
    ).materialize(episode, seed=7)
    gold = captured["gold_protein_go_edge_index"]
    # Local proteins are [0,1,2].  The sampled core neighbour (global/local 1)
    # carries a gold edge even though it is not an explicit query support seed.
    assert bool((gold[0] == 1).any())
