from pathlib import Path

import numpy as np

from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.latence_stores import GOProteinCSRStore, RoleAwareBaseLogitStore, RoleProbabilitySlice
from nbs_pg.training import NBSFixedEpochTrainingConfig


def _save(path: Path, value):
    np.save(path, value)
    return path


def test_hierarchy_aware_episode_contains_direct_pair(tmp_path: Path):
    # Four GO terms with enough gold support; two task-level hierarchy edges.
    gold_indptr = _save(tmp_path / "gold_indptr.npy", np.array([0, 4, 8, 12, 16], np.int64))
    gold_protein = _save(tmp_path / "gold_protein.npy", np.arange(16, dtype=np.int32))
    candidate_indptr = _save(tmp_path / "cand_indptr.npy", np.array([0, 2, 4, 6, 8], np.int64))
    candidate_protein = _save(tmp_path / "cand_protein.npy", np.arange(16, 24, dtype=np.int32))
    prob = _save(tmp_path / "prob.npy", np.full((24, 4), 0.2, dtype=np.float16))

    gold = GOProteinCSRStore(gold_indptr, gold_protein)
    candidate = GOProteinCSRStore(candidate_indptr, candidate_protein)
    base = RoleAwareBaseLogitStore([RoleProbabilitySlice("all", 0, 24, str(prob))], num_go=4)
    hierarchy = np.array([[0, 2], [1, 3]], dtype=np.int64)
    sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=candidate,
        base_logits=base,
        train_go_counts=np.ones(4, dtype=np.float64) * 4,
        hierarchy_pairs=hierarchy,
        config=NBSQueryEpisodeConfig(
            num_queries=4,
            support_per_query=1,
            gold_positive_per_query=1,
            hard_candidate_per_query=1,
            max_candidates=16,
            hierarchy_pairs_per_episode=1,
        ),
        seed=7,
    )
    episode = sampler.sample(seed=11)
    selected = set(episode.query_go_idx.tolist())
    assert ({0, 1} <= selected) or ({2, 3} <= selected)


def test_progress_config_round_trip():
    cfg = NBSFixedEpochTrainingConfig.from_mapping(
        {
            "epochs": 2,
            "save_epochs": [2],
            "progress_bar": True,
            "progress_mininterval": 0.25,
        }
    )
    cfg.validate()
    assert cfg.progress_bar is True
    assert cfg.progress_mininterval == 0.25
