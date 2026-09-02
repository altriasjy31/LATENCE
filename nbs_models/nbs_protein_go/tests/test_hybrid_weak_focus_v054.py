from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.latence_graph_stores import ProteinRegistryStore, RoleLocalProteinGOCSRStore
from nbs_pg.latence_stores import GOProteinCSRStore, RoleAwareBaseLogitStore, RoleProbabilitySlice
from nbs_pg.local_loader import (
    resolve_hybrid_epoch_requirements,
    resolve_weak_primary_epoch_requirements,
)


def _save(path: Path, value: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    return path


def _registry(path: Path, *, core: int, weak: int) -> ProteinRegistryStore:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_idx", "protein_id", "role", "role_row_idx"],
        )
        writer.writeheader()
        for idx in range(core):
            writer.writerow(
                {
                    "protein_idx": idx,
                    "protein_id": f"C{idx}",
                    "role": "core",
                    "role_row_idx": idx,
                }
            )
        for row in range(weak):
            writer.writerow(
                {
                    "protein_idx": core + row,
                    "protein_id": f"W{row}",
                    "role": "weak",
                    "role_row_idx": row,
                }
            )
    return ProteinRegistryStore(path)


def _go_major_from_protein_rows(rows: list[list[tuple[int, float]]], num_go: int):
    proteins_by_go: list[list[int]] = [[] for _ in range(num_go)]
    probs_by_go: list[list[float]] = [[] for _ in range(num_go)]
    for role_row, annotations in enumerate(rows):
        for go_idx, probability in annotations:
            proteins_by_go[go_idx].append(12 + role_row)
            probs_by_go[go_idx].append(probability)
    indptr = [0]
    protein: list[int] = []
    probability: list[float] = []
    for go_idx in range(num_go):
        protein.extend(proteins_by_go[go_idx])
        probability.extend(probs_by_go[go_idx])
        indptr.append(len(protein))
    return (
        np.asarray(indptr, dtype=np.int64),
        np.asarray(protein, dtype=np.int32),
        np.asarray(probability, dtype=np.float16),
    )


def _sampler(tmp_path: Path) -> GOQueryEpisodeSampler:
    registry = _registry(tmp_path / "registry.csv", core=12, weak=20)

    gold_indptr = _save(
        tmp_path / "gold_indptr.npy", np.array([0, 3, 6, 9, 12], np.int64)
    )
    gold_protein = _save(
        tmp_path / "gold_protein.npy", np.arange(12, dtype=np.int32)
    )
    candidate_indptr = _save(
        tmp_path / "candidate_indptr.npy", np.array([0, 2, 4, 6, 8], np.int64)
    )
    candidate_protein = _save(
        tmp_path / "candidate_protein.npy", np.arange(12, 20, dtype=np.int32)
    )

    protein_rows: list[list[tuple[int, float]]] = []
    for row in range(20):
        group = row // 5
        first = group % 4
        second = (group + 1) % 4
        protein_rows.append([(first, 0.90 - row * 0.001), (second, 0.80)])

    protein_indptr = [0]
    protein_go: list[int] = []
    protein_prob: list[float] = []
    for annotations in protein_rows:
        for go_idx, probability in annotations:
            protein_go.append(go_idx)
            protein_prob.append(probability)
        protein_indptr.append(len(protein_go))
    pseudo_by_protein = RoleLocalProteinGOCSRStore(
        _save(tmp_path / "protein_indptr.npy", np.asarray(protein_indptr, np.int64)),
        _save(tmp_path / "protein_go.npy", np.asarray(protein_go, np.int32)),
        role="weak",
        registry=registry,
        probability_path=_save(
            tmp_path / "protein_prob.npy", np.asarray(protein_prob, np.float16)
        ),
    )
    active_rows = pseudo_by_protein.active_role_rows_for_go_mask(
        np.ones(4, dtype=np.bool_)
    )
    assert active_rows.size == 20

    pseudo_indptr, pseudo_protein, pseudo_prob = _go_major_from_protein_rows(
        protein_rows, 4
    )
    base_prob = _save(
        tmp_path / "base_prob.npy", np.full((32, 4), 0.2, dtype=np.float16)
    )
    return GOQueryEpisodeSampler(
        gold=GOProteinCSRStore(gold_indptr, gold_protein),
        candidate=GOProteinCSRStore(candidate_indptr, candidate_protein),
        pseudo=GOProteinCSRStore(
            _save(tmp_path / "pseudo_indptr.npy", pseudo_indptr),
            _save(tmp_path / "pseudo_protein.npy", pseudo_protein),
            payload_paths={"probability": _save(tmp_path / "pseudo_prob.npy", pseudo_prob)},
        ),
        pseudo_by_protein=pseudo_by_protein,
        pseudo_active_role_rows=active_rows,
        base_logits=RoleAwareBaseLogitStore(
            [RoleProbabilitySlice("all", 0, 32, str(base_prob))], num_go=4
        ),
        train_go_counts=np.array([3, 3, 3, 3], np.float64),
        config=NBSQueryEpisodeConfig(
            num_queries=4,
            support_per_query=1,
            gold_positive_per_query=1,
            hard_candidate_per_query=1,
            pseudo_positive_per_query=2,
            pseudo_sampling_mode="go_cyclic_unique",
            pseudo_rank_partition=True,
            weak_focus_queries_per_episode=2,
            weak_focus_targets_per_query=2,
            weak_focus_scan_limit=64,
            max_candidates=24,
            query_sampling_mode="shuffled_cycle",
        ),
        seed=13,
        weak_global_start=12,
        weak_global_end=32,
    )


def test_weak_focus_reserves_queries_and_forces_unseen_targets(tmp_path: Path):
    sampler = _sampler(tmp_path)
    assert sampler.coverage_slots_per_episode == 2
    first = sampler.sample(
        seed=101, epoch=1, global_episode=0, rank=0, world_size=1
    )
    second = sampler.sample(
        seed=102, epoch=1, global_episode=1, rank=0, world_size=1
    )
    assert first.metadata["weak_focus_queries_requested"] == 2
    assert first.metadata["weak_focus_queries_realized"] == 2
    assert first.metadata["weak_focus_targets_retained"] == 4
    assert first.metadata["weak_focus_anchor_new_count"] == 2
    assert first.metadata["weak_focus_target_capacity_requested"] == 4
    p1 = set(np.asarray(first.metadata["pseudo_target_protein_idx"]).tolist())
    p2 = set(np.asarray(second.metadata["pseudo_target_protein_idx"]).tolist())
    assert p1 and p2
    # There are enough unseen weak proteins for two complete episodes.
    assert p1.isdisjoint(p2)


def test_hybrid_epoch_planner_uses_unique_weak_target():
    plan = resolve_hybrid_epoch_requirements(
        go_cycles_floor=1.0,
        eligible_go_count=100,
        coverage_slots_per_episode=2,
        world_size=2,
        weak_unique_coverage_target=0.70,
        pseudo_eligible_active_weak=1000,
        weak_focus_queries_per_episode=2,
        weak_focus_targets_per_query=4,
        weak_focus_planning_efficiency=0.875,
        core_gold_pass_target=1.0,
        core_count=100,
        predicted_core_occurrences_per_go_cycle=200,
    )
    assert plan["go_steps_required"] == 25
    assert plan["weak_unique_steps_required"] == 50
    assert plan["core_steps_required"] == 13
    assert plan["resolved_steps"] == 50
    assert plan["weak_unique_target_count"] == 700


def test_weak_primary_queue_exhausts_only_on_forced_anchors(tmp_path: Path):
    sampler = _sampler(tmp_path)
    sampler.config.weak_primary_proteins_per_episode = 2
    remaining = []
    for episode_index in range(10):
        episode = sampler.sample(
            seed=200 + episode_index,
            epoch=1,
            global_episode=episode_index,
            rank=0,
            world_size=1,
        )
        assert episode.metadata["weak_primary_anchor_count"] == 2
        assert episode.metadata["weak_primary_anchor_retained"] == 2
        remaining.append(int(episode.metadata["weak_primary_remaining"]))
    assert remaining == list(range(18, -1, -2))
    progress = sampler.weak_primary_progress(epoch=1, rank=0, world_size=1)
    assert progress == {"owned": 20, "selected": 20, "remaining": 0}


def test_weak_primary_epoch_plan_uses_largest_ddp_shard():
    plan = resolve_weak_primary_epoch_requirements(
        pseudo_eligible_active_weak=1001,
        weak_primary_proteins_per_episode=32,
        world_size=2,
    )
    assert plan["largest_owned_weak_shard"] == 501
    assert plan["resolved_steps"] == 16
    assert plan["weak_primary_global_capacity_per_step"] == 64


def test_active_rows_only_count_eligible_pseudo_go(tmp_path: Path):
    registry = _registry(tmp_path / "registry.csv", core=1, weak=4)
    indptr = _save(tmp_path / "indptr.npy", np.array([0, 1, 2, 3, 3], np.int64))
    go = _save(tmp_path / "go.npy", np.array([0, 1, 2], np.int32))
    prob = _save(tmp_path / "prob.npy", np.array([0.8, 0.8, 0.8], np.float16))
    store = RoleLocalProteinGOCSRStore(
        indptr, go, role="weak", registry=registry, probability_path=prob
    )
    rows = store.active_role_rows_for_go_mask(
        np.array([False, True, True], dtype=np.bool_)
    )
    assert np.array_equal(rows, np.array([1, 2], dtype=np.int64))
