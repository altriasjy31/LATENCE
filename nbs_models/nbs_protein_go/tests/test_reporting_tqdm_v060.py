from __future__ import annotations

from nbs_pg.reporting import (
    build_epoch_summary_lines,
    build_progress_detail_lines,
    build_progress_live_lines,
    build_progress_postfix,
)
from nbs_pg.training import NBSFixedEpochTrainingConfig


def _snapshot():
    return {
        "epoch": 3,
        "step": 20,
        "global_step": 60,
        "loss": 0.0417,
        "gold_asl": 0.000551,
        "pseudo_asl": 0.118,
        "contrib_gold_asl": 0.000551,
        "contrib_pseudo_asl": 0.0236,
        "contrib_base_anchor": 0.0174,
        "hierarchy": 0.00236,
        "contrib_hierarchy": 0.0,
        "hierarchy_pairs": 33,
        "graph_delta_scale": 0.01,
        "lr": 8.2e-5,
        "scale_lr": 4.1e-4,
        "memory_allocated_gb": 39.0,
        "memory_reserved_gb": 42.0,
        "query_seen": 1280,
        "meta": {
            "gold_pairs_retained": 64,
            "hard_pairs_retained": 159,
            "pseudo_pairs_retained": 1014,
            "background_pairs_requested": 1024,
            "background_pairs_retained": 1003,
            "positive_query_rows": 64,
            "negative_query_rows": 63,
            "positive_only_query_rows": 1,
            "mixed_query_rows": 63,
            "weak_focus_queries_requested": 16,
            "weak_focus_queries_realized": 16,
            "weak_focus_anchor_new_count": 16,
            "weak_primary_anchor_retained": 96,
            "weak_primary_candidate_query_anchor_count": 90,
            "weak_primary_candidate_query_hit_rate": 0.9375,
            "weak_primary_remaining": 12,
            "weak_focus_targets_retained": 512,
            "query_with_pseudo_pool": 62,
            "query_with_backbone_candidate_pool": 64,
            "candidate_truncation_fraction": 0.0,
        },
    }


def test_compact_postfix_is_deliberately_small():
    status = build_progress_postfix(_snapshot(), mode="compact")
    assert set(status) == {"L", "G", "P", "gp", "hd", "pp", "bg%", "lr", "M"}
    assert status["bg%"] == "98%"
    assert status["M"] == "39/42G"


def test_full_postfix_preserves_legacy_diagnostics():
    status = build_progress_postfix(_snapshot(), mode="full")
    for key in ("loss", "gold", "pseudo", "gpos", "hard", "ppos", "wfq", "wpa", "qcan", "mem"):
        assert key in status


def test_detail_lines_are_semantically_grouped():
    lines = build_progress_detail_lines(_snapshot())
    assert len(lines) == 4
    assert "objective" in lines[0]
    assert "sampling" in lines[1]
    assert "weak" in lines[2]
    assert "runtime" in lines[3]


def test_live_multiline_rows_are_short_and_semantically_grouped():
    lines = build_progress_live_lines(_snapshot())
    assert len(lines) == 4
    assert lines[0].startswith(" objective |")
    assert lines[1].startswith(" sampling  |")
    assert lines[2].startswith(" weak-flow |")
    assert lines[3].startswith(" runtime   |")
    assert all(len(line) < 150 for line in lines)


def test_reporting_config_is_backward_compatible_and_validates_new_modes():
    legacy = NBSFixedEpochTrainingConfig(epochs=1, save_epochs=(1,))
    assert legacy.progress_display_mode == "singleline"
    assert legacy.progress_postfix_mode == "full"
    assert legacy.progress_update_interval is None
    assert legacy.progress_detail_interval is None
    assert legacy.epoch_summary_mode == "legacy"
    legacy.validate()

    optimized = NBSFixedEpochTrainingConfig(
        epochs=1,
        save_epochs=(1,),
        progress_display_mode="multiline",
        progress_postfix_mode="compact",
        progress_update_interval=1,
        progress_detail_interval=20,
        epoch_summary_mode="multiline",
    )
    optimized.validate()


def test_multiline_epoch_summary_has_stable_sections():
    lines = build_epoch_summary_lines(
        3,
        {
            "total": 0.0417,
            "elapsed_seconds": 120.0,
            "gold_asl": 0.000551,
            "pseudo_asl": 0.118,
            "query_row_partition_ok": True,
            "learning_rates": [8.2e-5, 4.1e-4],
            "protein_universe": {"total": 100, "role_counts": {"core": 20, "weak": 80}},
            "ontology_universe": {"full_ontology_rows": 44919},
        },
    )
    labels = "\n".join(lines)
    for section in ("objective", "sampling", "query", "weak", "protein", "ontology", "runtime"):
        assert section in labels
