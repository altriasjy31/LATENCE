from pathlib import Path

import numpy as np

from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.latence_stores import GOProteinCSRStore, RoleAwareBaseLogitStore, RoleProbabilitySlice


def _save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    return path


def _sampler(tmp_path: Path, rank_partition: bool = True):
    # Two GO terms share several weak pseudo proteins. Unique mode should avoid
    # reusing them across GO/episodes while unseen alternatives remain.
    gold_indptr = _save(tmp_path / "gold_indptr.npy", np.array([0, 3, 6], np.int64))
    gold_protein = _save(tmp_path / "gold_protein.npy", np.array([0, 1, 2, 3, 4, 5], np.int32))
    cand_indptr = _save(tmp_path / "cand_indptr.npy", np.array([0, 1, 2], np.int64))
    cand_protein = _save(tmp_path / "cand_protein.npy", np.array([30, 31], np.int32))
    pseudo_indptr = _save(tmp_path / "pseudo_indptr.npy", np.array([0, 6, 12], np.int64))
    pseudo_protein = _save(
        tmp_path / "pseudo_protein.npy",
        np.array([10, 11, 12, 13, 14, 15, 10, 11, 16, 17, 18, 19], np.int32),
    )
    pseudo_prob = _save(tmp_path / "pseudo_prob.npy", np.full(12, 0.8, np.float16))
    prob = _save(tmp_path / "prob.npy", np.full((40, 2), 0.2, np.float16))
    return GOQueryEpisodeSampler(
        gold=GOProteinCSRStore(gold_indptr, gold_protein),
        candidate=GOProteinCSRStore(cand_indptr, cand_protein),
        pseudo=GOProteinCSRStore(pseudo_indptr, pseudo_protein, payload_paths={"probability": pseudo_prob}),
        base_logits=RoleAwareBaseLogitStore([RoleProbabilitySlice("all", 0, 40, str(prob))], num_go=2),
        train_go_counts=np.array([3, 3], np.float64),
        config=NBSQueryEpisodeConfig(
            num_queries=2,
            support_per_query=1,
            gold_positive_per_query=1,
            hard_candidate_per_query=1,
            pseudo_positive_per_query=2,
            pseudo_sampling_mode="go_cyclic_unique",
            pseudo_rank_partition=rank_partition,
            max_candidates=16,
            query_sampling_mode="shuffled_cycle",
        ),
        seed=7,
        weak_global_start=10,
        weak_global_end=20,
    )


def test_unique_pseudo_mode_reduces_cross_go_reuse(tmp_path: Path):
    sampler = _sampler(tmp_path)
    first = sampler.sample(seed=100, epoch=1, global_episode=0, rank=0, world_size=1)
    second = sampler.sample(seed=101, epoch=1, global_episode=1, rank=0, world_size=1)
    p1 = set(np.asarray(first.metadata["pseudo_target_protein_idx"]).tolist())
    p2 = set(np.asarray(second.metadata["pseudo_target_protein_idx"]).tolist())
    assert p1
    assert p2
    assert p1.isdisjoint(p2)
    assert second.metadata["pseudo_seen_rank"] >= len(p1 | p2)


def test_unique_pseudo_mode_prefers_ddp_owned_partition(tmp_path: Path):
    sampler0 = _sampler(tmp_path / "r0")
    sampler1 = _sampler(tmp_path / "r1")
    first0 = sampler0.sample(seed=200, epoch=1, global_episode=0, rank=0, world_size=2)
    first1 = sampler1.sample(seed=201, epoch=1, global_episode=1, rank=1, world_size=2)
    p0 = np.asarray(first0.metadata["pseudo_target_protein_idx"], dtype=np.int64)
    p1 = np.asarray(first1.metadata["pseudo_target_protein_idx"], dtype=np.int64)
    assert p0.size and p1.size
    # The preferred owner is based on weak-role local row parity.
    assert np.mean(((p0 - 10) % 2) == 0) >= 0.5
    assert np.mean(((p1 - 10) % 2) == 1) >= 0.5
