from pathlib import Path

import numpy as np

from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.latence_stores import (
    GOProteinCSRStore,
    RoleAwareBaseLogitStore,
    RoleProbabilitySlice,
)


def _save(path: Path, value):
    np.save(path, value)
    return path


def test_episode_emits_exact_query_gold_counts(tmp_path: Path):
    gold_indptr = _save(tmp_path / "gold_indptr.npy", np.array([0, 1, 3, 7, 12], np.int64))
    gold_protein = _save(tmp_path / "gold_protein.npy", np.arange(12, dtype=np.int32))
    cand_indptr = _save(tmp_path / "cand_indptr.npy", np.array([0, 2, 4, 6, 8], np.int64))
    cand_protein = _save(tmp_path / "cand_protein.npy", np.arange(20, 28, dtype=np.int32))
    pseudo_indptr = _save(tmp_path / "pseudo_indptr.npy", np.array([0, 1, 2, 3, 4], np.int64))
    pseudo_protein = _save(tmp_path / "pseudo_protein.npy", np.arange(40, 44, dtype=np.int32))
    pseudo_prob = _save(tmp_path / "pseudo_prob.npy", np.array([0.9, 0.8, 0.7, 0.6], np.float16))
    prob = _save(tmp_path / "prob.npy", np.full((64, 4), 0.2, dtype=np.float16))

    sampler = GOQueryEpisodeSampler(
        gold=GOProteinCSRStore(gold_indptr, gold_protein),
        candidate=GOProteinCSRStore(cand_indptr, cand_protein),
        pseudo=GOProteinCSRStore(
            pseudo_indptr, pseudo_protein, payload_paths={"probability": pseudo_prob}
        ),
        base_logits=RoleAwareBaseLogitStore(
            [RoleProbabilitySlice("all", 0, 64, str(prob))], num_go=4
        ),
        train_go_counts=np.array([1, 2, 4, 5], np.float64),
        config=NBSQueryEpisodeConfig(
            num_queries=4,
            support_per_query=2,
            gold_positive_per_query=1,
            hard_candidate_per_query=1,
            pseudo_positive_per_query=1,
            max_candidates=32,
            gold_support_policy="adaptive_rare",
            singleton_requires_pseudo=True,
        ),
        seed=3,
    )
    episode = sampler.sample(seed=9)
    by_go = dict(zip(episode.metadata["query_go_idx"], episode.metadata["query_gold_counts"]))
    assert by_go == {0: 1, 1: 2, 2: 4, 3: 5}
    assert episode.metadata["query_with_pseudo_pool"] == 4
    assert episode.metadata["query_with_backbone_candidate_pool"] == 4
