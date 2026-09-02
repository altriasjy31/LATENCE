from __future__ import annotations

import numpy as np

from nbs_pg.local_loader import (
    build_ontology_space_summary,
    resolve_hybrid_epoch_requirements,
)


def test_bp_like_hybrid_planner_uses_all_pseudo_query_capacity():
    plan = resolve_hybrid_epoch_requirements(
        go_cycles_floor=1.0,
        eligible_go_count=18300,
        coverage_slots_per_episode=32,
        world_size=2,
        weak_unique_coverage_target=0.70,
        pseudo_eligible_active_weak=489222,
        weak_focus_queries_per_episode=16,
        weak_focus_targets_per_query=32,
        weak_focus_planning_efficiency=0.60,
        core_gold_pass_target=1.0,
        core_count=60500,
        predicted_core_occurrences_per_go_cycle=52997,
        num_queries_per_episode=64,
        predicted_pseudo_pairs_per_go_cycle=142906,
    )
    assert plan["go_steps_required"] == 286
    assert 315 <= plan["weak_unique_steps_required"] <= 325
    assert plan["core_steps_required"] == 327
    assert plan["resolved_steps"] == 327
    assert plan["estimated_pseudo_pairs_per_global_step"] > 1700
    assert plan["effective_unique_weak_capacity_per_global_step"] > 1000


def test_ontology_space_summary_separates_task_query_and_full_context():
    # Five classifier columns, with one canonical duplicate, mapped into an
    # eight-row full ontology.  Only three classifier columns are supervised
    # queries; the remaining task ontology row still exists as task context.
    mapping = np.asarray([0, 1, 1, 3, 5], dtype=np.int64)
    eligible = np.asarray([0, 1, 4], dtype=np.int64)
    task_mask, eligible_mask, context_only_mask, summary = build_ontology_space_summary(
        mapping,
        eligible,
        full_ontology_rows=8,
        relation_edge_counts={"is_a": 7, "part_of": 2},
    )
    assert summary["task_label_columns"] == 5
    assert summary["unique_task_ontology_rows"] == 4
    assert summary["task_to_ontology_duplicate_columns"] == 1
    assert summary["eligible_task_query_columns"] == 3
    assert summary["ineligible_task_query_columns"] == 2
    assert summary["unique_eligible_query_ontology_rows"] == 3
    assert summary["unique_context_only_task_ontology_rows"] == 1
    assert summary["full_ontology_rows"] == 8
    assert summary["non_task_ontology_rows"] == 4
    assert summary["global_go_cache_rows"] == 8
    assert summary["global_go_cache_uses_full_ontology"] is True
    assert summary["direct_relation_edges"] == {"is_a": 7, "part_of": 2}
    assert np.flatnonzero(task_mask).tolist() == [0, 1, 3, 5]
    assert np.flatnonzero(eligible_mask).tolist() == [0, 1, 5]
    assert np.flatnonzero(context_only_mask).tolist() == [3]
