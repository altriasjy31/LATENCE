from pathlib import Path

import numpy as np

from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.latence_stores import (
    FixedDegreeCandidateAttributeStore,
    GOProteinCSRStore,
    RoleAwareBaseLogitStore,
    RoleProbabilitySlice,
)


def _save(path: Path, value):
    np.save(path, value)
    return path


def test_episode_sampler_keeps_support_and_candidates_disjoint(tmp_path: Path):
    # Two GO terms, four gold proteins each.
    gold_indptr = _save(tmp_path / "gold_indptr.npy", np.array([0, 4, 8], np.int64))
    gold_protein = _save(
        tmp_path / "gold_protein.npy",
        np.array([0, 1, 2, 3, 4, 5, 6, 7], np.int32),
    )
    candidate_indptr = _save(
        tmp_path / "candidate_indptr.npy", np.array([0, 3, 6], np.int64)
    )
    candidate_protein = _save(
        tmp_path / "candidate_protein.npy",
        np.array([8, 9, 10, 11, 12, 13], np.int32),
    )
    source_rank = _save(
        tmp_path / "candidate_rank.npy",
        np.array([0, 1, 0, 1, 0, 1], np.uint16),
    )
    edge_attr = np.zeros((14 * 2, 3), dtype=np.float32)
    edge_attr[:, 2] = np.tile([1.0, 0.5], 14)
    edge_attr_path = _save(tmp_path / "edge_attr.npy", edge_attr)
    probability = np.full((14, 2), 0.2, dtype=np.float16)
    probability_path = _save(tmp_path / "prob.npy", probability)

    gold = GOProteinCSRStore(gold_indptr, gold_protein)
    candidate = GOProteinCSRStore(
        candidate_indptr,
        candidate_protein,
        payload_paths={"source_rank": source_rank},
    )
    base = RoleAwareBaseLogitStore(
        [RoleProbabilitySlice("all", 0, 14, str(probability_path))], num_go=2
    )
    attrs = FixedDegreeCandidateAttributeStore(
        edge_attr_path, fixed_degree=2, source_protein_start=0
    )
    sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=candidate,
        base_logits=base,
        candidate_attributes=attrs,
        train_go_counts=np.array([4, 4]),
        config=NBSQueryEpisodeConfig(
            num_queries=2,
            support_per_query=1,
            gold_positive_per_query=1,
            hard_candidate_per_query=2,
            max_candidates=8,
        ),
        seed=1,
    )
    episode = sampler.sample()
    episode.validate()
    assert np.intersect1d(
        episode.seed_protein_idx, episode.candidate_protein_idx
    ).size == 0
    assert episode.base_logits.shape == episode.labels.shape
    assert episode.candidate_evidence.shape[-1] == 3
    assert episode.mask.any()
