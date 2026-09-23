from pathlib import Path

import numpy as np

from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.latence_stores import GOProteinCSRStore, RoleAwareBaseLogitStore, RoleProbabilitySlice
from nbs_pg.local_loader import NBSLocalGraphSamplingConfig, resolve_epoch_cycle_requirements


def _save(path: Path, value):
    np.save(path, value)
    return path


def test_protein_major_cycle_plan_uses_weak_as_primary_constraint():
    plan = resolve_epoch_cycle_requirements(
        epoch_unit="protein_major_with_go_floor",
        go_cycles_floor=1.0,
        weak_pseudo_pass_target=1.0,
        core_gold_pass_target=1.0,
        pseudo_active_weak=1000,
        predicted_pseudo_pairs_per_go_cycle=250,
        core_count=100,
        predicted_core_occurrences_per_go_cycle=200,
    )
    assert plan["go_cycles_required"] == 1.0
    assert plan["weak_pseudo_cycles_required"] == 4.0
    assert plan["core_gold_cycles_required"] == 0.5
    assert plan["resolved_cycles"] == 4.0


def test_go_only_epoch_keeps_old_coverage_semantics():
    plan = resolve_epoch_cycle_requirements(
        epoch_unit="eligible_go_coverage_cycle",
        go_cycles_floor=1.5,
        weak_pseudo_pass_target=0.0,
        core_gold_pass_target=0.0,
        pseudo_active_weak=0,
        predicted_pseudo_pairs_per_go_cycle=0,
        core_count=0,
        predicted_core_occurrences_per_go_cycle=0,
    )
    assert plan["resolved_cycles"] == 1.5


def test_protein_major_config_validation():
    cfg = NBSLocalGraphSamplingConfig.from_mapping({
        "steps_per_epoch_per_rank": None,
        "coverage_cycles_per_epoch": 1.0,
        "epoch_unit": "protein_major_with_go_floor",
        "weak_pseudo_equivalent_passes_per_epoch": 1.0,
        "core_gold_equivalent_passes_per_epoch": 1.0,
    })
    assert cfg.epoch_unit == "protein_major_with_go_floor"


def test_go_cyclic_pseudo_sampling_walks_different_weak_targets(tmp_path: Path):
    # One GO is repeated in successive global episodes.  Its pseudo pool has
    # eight weak proteins; count=2 must move to a different deterministic chunk.
    gold_indptr = _save(tmp_path / "gold_indptr.npy", np.array([0, 3], np.int64))
    gold_protein = _save(tmp_path / "gold_protein.npy", np.array([0, 1, 2], np.int32))
    cand_indptr = _save(tmp_path / "cand_indptr.npy", np.array([0, 2], np.int64))
    cand_protein = _save(tmp_path / "cand_protein.npy", np.array([20, 21], np.int32))
    pseudo_indptr = _save(tmp_path / "pseudo_indptr.npy", np.array([0, 8], np.int64))
    pseudo_protein = _save(tmp_path / "pseudo_protein.npy", np.arange(40, 48, dtype=np.int32))
    pseudo_prob = _save(tmp_path / "pseudo_prob.npy", np.linspace(0.6, 0.95, 8).astype(np.float16))
    prob = _save(tmp_path / "prob.npy", np.full((64, 1), 0.2, dtype=np.float16))

    sampler = GOQueryEpisodeSampler(
        gold=GOProteinCSRStore(gold_indptr, gold_protein),
        candidate=GOProteinCSRStore(cand_indptr, cand_protein),
        pseudo=GOProteinCSRStore(
            pseudo_indptr, pseudo_protein, payload_paths={"probability": pseudo_prob}
        ),
        base_logits=RoleAwareBaseLogitStore(
            [RoleProbabilitySlice("all", 0, 64, str(prob))], num_go=1
        ),
        train_go_counts=np.array([3], np.float64),
        config=NBSQueryEpisodeConfig(
            num_queries=1,
            support_per_query=1,
            gold_positive_per_query=1,
            hard_candidate_per_query=1,
            pseudo_positive_per_query=2,
            pseudo_sampling_mode="go_cyclic",
            max_candidates=16,
            query_sampling_mode="shuffled_cycle",
        ),
        seed=7,
    )
    first = sampler.sample(seed=100, epoch=1, global_episode=0)
    second = sampler.sample(seed=101, epoch=1, global_episode=1)
    p1 = set(np.asarray(first.metadata["pseudo_target_protein_idx"]).tolist())
    p2 = set(np.asarray(second.metadata["pseudo_target_protein_idx"]).tolist())
    assert len(p1) == 2 and len(p2) == 2
    assert p1.isdisjoint(p2)
