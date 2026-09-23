from pathlib import Path

import numpy as np

from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.latence_stores import GOProteinCSRStore, RoleAwareBaseLogitStore, RoleProbabilitySlice
from nbs_pg.local_loader import NBSLocalGraphSamplingConfig


def _save(path: Path, value):
    np.save(path, value)
    return path


def _stores(tmp_path: Path):
    # Gold degrees [1, 2, 4, 4].
    gold_indptr = _save(tmp_path / "gold_indptr.npy", np.array([0, 1, 3, 7, 11], np.int64))
    gold_protein = _save(tmp_path / "gold_protein.npy", np.arange(11, dtype=np.int32))
    # Candidate edges use disjoint proteins.
    cand_indptr = _save(tmp_path / "cand_indptr.npy", np.array([0, 4, 8, 12, 16], np.int64))
    cand_protein = _save(tmp_path / "cand_protein.npy", np.arange(20, 36, dtype=np.int32))
    # Singleton GO0 has a modelout-positive weak protein, making it eligible in adaptive mode.
    pseudo_indptr = _save(tmp_path / "pseudo_indptr.npy", np.array([0, 1, 2, 3, 4], np.int64))
    pseudo_protein = _save(tmp_path / "pseudo_protein.npy", np.arange(40, 44, dtype=np.int32))
    pseudo_prob = _save(tmp_path / "pseudo_prob.npy", np.array([0.9, 0.8, 0.7, 0.6], np.float16))
    probs = _save(tmp_path / "prob.npy", np.full((64, 4), 0.2, dtype=np.float16))
    gold = GOProteinCSRStore(gold_indptr, gold_protein)
    cand = GOProteinCSRStore(cand_indptr, cand_protein)
    pseudo = GOProteinCSRStore(pseudo_indptr, pseudo_protein, payload_paths={"probability": pseudo_prob})
    base = RoleAwareBaseLogitStore([RoleProbabilitySlice("all", 0, 64, str(probs))], num_go=4)
    return gold, cand, pseudo, base


def test_adaptive_rare_support_includes_count1_and_count2(tmp_path: Path):
    gold, cand, pseudo, base = _stores(tmp_path)
    cfg = NBSQueryEpisodeConfig(
        num_queries=4,
        support_per_query=2,
        gold_positive_per_query=1,
        hard_candidate_per_query=1,
        pseudo_positive_per_query=1,
        max_candidates=32,
        gold_support_policy="adaptive_rare",
        singleton_requires_pseudo=True,
    )
    sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=cand,
        pseudo=pseudo,
        base_logits=base,
        train_go_counts=np.array([1, 2, 4, 4], np.float64),
        config=cfg,
        seed=3,
    )
    assert set(sampler.eligible_go.tolist()) == {0, 1, 2, 3}
    episode = sampler.sample(seed=9)
    assert episode.metadata["query_gold_count_eq1"] == 1
    assert episode.metadata["query_gold_count_eq2"] == 1
    # Count-1 query contributes a support protein but no impossible held-out gold target.
    row = int(np.flatnonzero(episode.query_go_idx == 0)[0])
    assert int(np.sum(episode.mask[row] & ~episode.pseudo_mask[row] & (episode.labels[row] == 1.0))) == 0
    assert int(np.sum(episode.pseudo_mask[row])) >= 1


def test_shuffled_cycle_covers_eligible_go_across_global_episodes(tmp_path: Path):
    gold, cand, pseudo, base = _stores(tmp_path)
    cfg = NBSQueryEpisodeConfig(
        num_queries=2,
        support_per_query=1,
        gold_positive_per_query=1,
        hard_candidate_per_query=1,
        pseudo_positive_per_query=1,
        max_candidates=16,
        gold_support_policy="adaptive_rare",
        singleton_requires_pseudo=True,
        query_sampling_mode="shuffled_cycle",
    )
    sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=cand,
        pseudo=pseudo,
        base_logits=base,
        train_go_counts=np.array([1, 2, 4, 4], np.float64),
        config=cfg,
        seed=100,
    )
    seen = set()
    # Two global episodes x two queries must traverse all four eligible terms.
    for global_episode in range(2):
        episode = sampler.sample(
            seed=1000 + global_episode,
            epoch=1,
            global_episode=global_episode,
        )
        seen.update(episode.query_go_idx.tolist())
    assert seen == {0, 1, 2, 3}


def test_candidate_budget_diagnostics_report_truncation(tmp_path: Path):
    gold, cand, pseudo, base = _stores(tmp_path)
    cfg = NBSQueryEpisodeConfig(
        num_queries=2,
        support_per_query=1,
        gold_positive_per_query=1,
        hard_candidate_per_query=4,
        pseudo_positive_per_query=1,
        max_candidates=4,
        gold_support_policy="adaptive_rare",
        singleton_requires_pseudo=True,
    )
    sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=cand,
        pseudo=pseudo,
        base_logits=base,
        train_go_counts=np.array([1, 2, 4, 4], np.float64),
        config=cfg,
        seed=5,
    )
    episode = sampler.sample(seed=12)
    assert episode.metadata["candidate_union_after_cap"] <= 4
    assert episode.metadata["candidate_union_before_cap"] >= episode.metadata["candidate_union_after_cap"]
    assert 0.0 <= episode.metadata["candidate_truncation_fraction"] <= 1.0
    assert episode.metadata["hard_pairs_retained"] <= episode.metadata["hard_pairs_requested"]


def test_coverage_based_epoch_config_accepts_auto_steps():
    cfg = NBSLocalGraphSamplingConfig.from_mapping(
        {"steps_per_epoch_per_rank": None, "coverage_cycles_per_epoch": 1.25}
    )
    assert cfg.steps_per_epoch_per_rank is None
    assert cfg.coverage_cycles_per_epoch == 1.25


def test_eligibility_summary_explains_singleton_and_candidate_pools(tmp_path: Path):
    gold, cand, pseudo, base = _stores(tmp_path)
    cfg = NBSQueryEpisodeConfig(
        num_queries=4,
        support_per_query=2,
        gold_positive_per_query=1,
        hard_candidate_per_query=1,
        pseudo_positive_per_query=1,
        max_candidates=32,
        gold_support_policy="adaptive_rare",
        singleton_requires_pseudo=True,
    )
    sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=cand,
        pseudo=pseudo,
        base_logits=base,
        train_go_counts=np.array([1, 2, 4, 4], np.float64),
        config=cfg,
        seed=3,
    )
    summary = sampler.eligibility_summary
    assert summary["gold_count_eq1_all"] == 1
    assert summary["gold_count_eq1_with_pseudo"] == 1
    assert summary["eligible_gold_count_eq1"] == 1
    assert summary["eligible_gold_count_eq2"] == 1
    assert summary["eligible_gold_count_3_4"] == 2
    assert summary["eligible_with_pseudo"] == 4
    assert summary["eligible_with_backbone_candidate"] == 4
    episode = sampler.sample(seed=9)
    assert episode.metadata["query_with_pseudo_pool"] == 4
    assert episode.metadata["query_with_backbone_candidate_pool"] == 4
