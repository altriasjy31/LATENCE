from __future__ import annotations

import numpy as np

from nbs_pg.episode import (
    GOQueryEpisodeSampler,
    NBSQueryEpisodeConfig,
    select_background_unlabelled_columns,
)


class _Store:
    def __init__(self, rows: list[np.ndarray]):
        self.rows = [np.asarray(row, dtype=np.int64) for row in rows]
        self.num_go = len(rows)
        counts = np.asarray([row.size for row in self.rows], dtype=np.int64)
        self.indptr = np.concatenate([np.asarray([0], np.int64), np.cumsum(counts)])

    def get(self, go_idx: int):
        return {"protein_idx": self.rows[int(go_idx)]}


class _Base:
    def gather_matrix(self, proteins: np.ndarray, go_idx: np.ndarray) -> np.ndarray:
        # Very low cross-query base confidence.  Positive gold labels are still
        # supplied explicitly by the supervision store and are not inferred
        # from this synthetic base model.
        return np.full((len(go_idx), len(proteins)), -5.0, dtype=np.float32)


def test_background_selector_respects_probability_occupied_and_excluded():
    candidate = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    # sigmoid approx [0.0067, 0.018, 0.119, 0.0025, 0.029]
    logits = np.asarray([-5.0, -4.0, -2.0, -6.0, -3.5], dtype=np.float32)
    occupied = np.asarray([False, True, False, False, False])
    excluded = np.asarray([14], dtype=np.int64)
    cols = select_background_unlabelled_columns(
        candidate_protein_idx=candidate,
        base_logits_row=logits,
        occupied_mask=occupied,
        excluded_protein_idx=excluded,
        count=16,
        max_probability=0.05,
        rng=np.random.default_rng(7),
    )
    assert set(cols.tolist()) == {0, 3}


def test_background_config_is_lower_weight_than_hard_pu():
    cfg = NBSQueryEpisodeConfig(
        background_unlabelled_per_query=16,
        sampled_unlabelled_weight=0.2,
        background_unlabelled_weight=0.05,
        background_base_probability_max=0.05,
    )
    cfg.validate()
    bad = NBSQueryEpisodeConfig(
        sampled_unlabelled_weight=0.1,
        background_unlabelled_weight=0.2,
    )
    try:
        bad.validate()
    except ValueError as exc:
        assert "should not exceed" in str(exc)
    else:
        raise AssertionError("expected invalid background PU weight to be rejected")


def test_episode_adds_background_without_adding_proteins_or_graph_candidates():
    # Two GO queries with disjoint core gold sets.  The candidate store is
    # empty, so without background PU both rows would have positive-only
    # supervision.  The held-out positive from the other query already belongs
    # to the shared union and becomes a safe low-base PU contrast.
    gold = _Store([
        np.asarray([0, 1], np.int64),
        np.asarray([2, 3], np.int64),
    ])
    candidate = _Store([
        np.empty(0, np.int64),
        np.empty(0, np.int64),
    ])
    cfg = NBSQueryEpisodeConfig(
        num_queries=2,
        support_per_query=1,
        gold_positive_per_query=1,
        hard_candidate_per_query=1,
        pseudo_positive_per_query=0,
        max_candidates=8,
        hierarchy_pairs_per_episode=0,
        query_sampling_mode="random",
        gold_support_policy="fixed",
        background_unlabelled_per_query=1,
        background_unlabelled_weight=0.05,
        background_base_probability_max=0.05,
    )
    sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=candidate,
        base_logits=_Base(),
        train_go_counts=np.asarray([2.0, 2.0]),
        config=cfg,
        seed=11,
    )
    episode = sampler.sample(seed=13)
    assert episode.candidate_protein_idx.size == 2
    assert episode.metadata["background_pairs_retained"] == 2
    assert episode.metadata["negative_query_rows"] == 2
    assert episode.metadata["positive_only_query_rows"] == 0
    assert episode.metadata["mixed_query_rows"] == 2
    assert np.all(episode.supervision_weight[episode.labels == 0] == 0.05)
