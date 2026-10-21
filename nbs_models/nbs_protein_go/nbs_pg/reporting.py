from __future__ import annotations

from typing import Any, Mapping


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _ratio(numerator: Any, denominator: Any, *, empty: float = 1.0) -> float:
    den = _as_float(denominator, 0.0)
    if den <= 0.0:
        return float(empty)
    return _as_float(numerator, 0.0) / den


def _memory_label(snapshot: Mapping[str, Any]) -> str:
    allocated = _as_float(snapshot.get("memory_allocated_gb"), 0.0)
    reserved = _as_float(snapshot.get("memory_reserved_gb"), 0.0)
    if allocated <= 0.0 and reserved <= 0.0:
        return "-"
    return f"{allocated:.0f}/{reserved:.0f}G"


def build_progress_postfix(
    snapshot: Mapping[str, Any],
    *,
    mode: str = "compact",
) -> dict[str, Any]:
    """Build a terminal-friendly tqdm postfix without changing training metrics.

    ``minimal`` is intended for narrow SSH terminals, ``compact`` is the
    recommended training view, and ``full`` preserves the legacy v0.6.0 field
    set for users who explicitly want every live diagnostic on one line.
    """

    mode = str(mode).strip().lower()
    if mode not in {"minimal", "compact", "full"}:
        raise ValueError(f"unsupported progress postfix mode: {mode}")

    meta = snapshot.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    loss = _as_float(snapshot.get("loss"), float("nan"))
    gold = _as_float(snapshot.get("gold_asl"), float("nan"))
    pseudo = _as_float(snapshot.get("pseudo_asl"), float("nan"))
    lr = _as_float(snapshot.get("lr"), float("nan"))

    if mode == "minimal":
        result: dict[str, Any] = {
            "L": f"{loss:.4f}",
            "G": f"{gold:.3g}",
            "P": f"{pseudo:.3g}",
            "lr": f"{lr:.1e}",
        }
        if _memory_label(snapshot) != "-":
            result["M"] = _memory_label(snapshot)
        return result

    if mode == "compact":
        bg_retained = _as_int(meta.get("background_pairs_retained"), 0)
        bg_requested = _as_int(meta.get("background_pairs_requested"), 0)
        result = {
            "L": f"{loss:.4f}",
            "G": f"{gold:.3g}",
            "P": f"{pseudo:.3g}",
            "gp": _as_int(meta.get("gold_pairs_retained"), 0),
            "hd": _as_int(meta.get("hard_pairs_retained"), 0),
            "pp": _as_int(meta.get("pseudo_pairs_retained"), 0),
            "bg%": f"{_ratio(bg_retained, bg_requested):.0%}",
            "lr": f"{lr:.1e}",
        }
        if _memory_label(snapshot) != "-":
            result["M"] = _memory_label(snapshot)
        return result

    # Full mode intentionally mirrors the original v0.6.0 postfix names.
    result = {
        "loss": f"{loss:.4f}",
        "gold": f"{gold:.3g}",
        "pseudo": f"{pseudo:.3g}",
        "gpos": _as_int(meta.get("gold_pairs_retained"), 0),
        "hard": _as_int(meta.get("hard_pairs_retained"), 0),
        "ppos": _as_int(meta.get("pseudo_pairs_retained"), 0),
        "bg": _as_int(meta.get("background_pairs_retained"), 0),
        "posonly": _as_int(meta.get("positive_only_query_rows"), 0),
        "negrows": _as_int(meta.get("negative_query_rows"), 0),
        "wfq": _as_int(meta.get("weak_focus_queries_realized"), 0),
        "wfa": _as_int(meta.get("weak_focus_anchor_new_count"), 0),
        "wpa": _as_int(meta.get("weak_primary_anchor_retained"), 0),
        "wpq": _as_int(meta.get("weak_primary_candidate_query_anchor_count"), 0),
        "wpr": _as_int(meta.get("weak_primary_remaining"), 0),
        "wft": _as_int(meta.get("weak_focus_targets_retained"), 0),
        "qps": _as_int(meta.get("query_with_pseudo_pool"), 0),
        "qcan": _as_int(meta.get("query_with_backbone_candidate_pool"), 0),
        "h_pairs": _as_int(snapshot.get("hierarchy_pairs"), 0),
        "qseen_r0": _as_int(snapshot.get("query_seen"), 0),
        "capdrop": f"{_as_float(meta.get('candidate_truncation_fraction'), 0.0):.1%}",
        "gscale": f"{_as_float(snapshot.get('graph_delta_scale'), float('nan')):.4f}",
        "lr": f"{lr:.2e}",
        "slr": f"{_as_float(snapshot.get('scale_lr'), float('nan')):.2e}",
    }
    if _memory_label(snapshot) != "-":
        allocated = _as_float(snapshot.get("memory_allocated_gb"), 0.0)
        reserved = _as_float(snapshot.get("memory_reserved_gb"), 0.0)
        result["mem"] = f"{allocated:.1f}A/{reserved:.1f}R"
    return result



def build_progress_live_lines(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    """Build four short rows for a persistent multi-line tqdm display.

    These lines deliberately contain fewer fields than the periodic detailed
    diagnostics. Their job is to remain readable without terminal wrapping
    while still exposing optimization, sampling, weak-flow, and runtime state.
    """

    meta = snapshot.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}

    bg_retained = _as_int(meta.get("background_pairs_retained"), 0)
    bg_requested = _as_int(meta.get("background_pairs_requested"), 0)

    objective = (
        " objective | "
        f"L={_as_float(snapshot.get('loss'), float('nan')):.4f} "
        f"G={_as_float(snapshot.get('gold_asl'), float('nan')):.4g} "
        f"P={_as_float(snapshot.get('pseudo_asl'), float('nan')):.4g} "
        f"cA={_as_float(snapshot.get('contrib_base_anchor'), float('nan')):.4g} "
        f"H={_as_float(snapshot.get('hierarchy'), float('nan')):.4g}"
    )
    sampling = (
        " sampling  | "
        f"gp={_as_int(meta.get('gold_pairs_retained'), 0)} "
        f"hd={_as_int(meta.get('hard_pairs_retained'), 0)} "
        f"pp={_as_int(meta.get('pseudo_pairs_retained'), 0)} "
        f"bg={bg_retained}/{bg_requested}({_ratio(bg_retained, bg_requested):.0%}) "
        f"posonly={_as_int(meta.get('positive_only_query_rows'), 0)} "
        f"neg={_as_int(meta.get('negative_query_rows'), 0)}"
    )
    weak = (
        " weak-flow | "
        f"wfq={_as_int(meta.get('weak_focus_queries_realized'), 0)}/"
        f"{_as_int(meta.get('weak_focus_queries_requested'), 0)} "
        f"wfa={_as_int(meta.get('weak_focus_anchor_new_count'), 0)} "
        f"wpa={_as_int(meta.get('weak_primary_anchor_retained'), 0)} "
        f"wpq={_as_int(meta.get('weak_primary_candidate_query_anchor_count'), 0)} "
        f"wpr={_as_int(meta.get('weak_primary_remaining'), 0)} "
        f"qps={_as_int(meta.get('query_with_pseudo_pool'), 0)} "
        f"qcan={_as_int(meta.get('query_with_backbone_candidate_pool'), 0)}"
    )
    runtime = (
        " runtime   | "
        f"step={_as_int(snapshot.get('step'), 0)} "
        f"global={_as_int(snapshot.get('global_step'), 0)} "
        f"qseen={_as_int(snapshot.get('query_seen'), 0)} "
        f"lr={_as_float(snapshot.get('lr'), float('nan')):.2e} "
        f"slr={_as_float(snapshot.get('scale_lr'), float('nan')):.2e} "
        f"M={_memory_label(snapshot)}"
    )
    return objective, sampling, weak, runtime

def build_progress_detail_lines(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    """Return grouped step diagnostics suitable for ``tqdm.write``."""

    meta = snapshot.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    epoch = _as_int(snapshot.get("epoch"), 0)
    step = _as_int(snapshot.get("step"), 0)
    global_step = _as_int(snapshot.get("global_step"), 0)

    bg_retained = _as_int(meta.get("background_pairs_retained"), 0)
    bg_requested = _as_int(meta.get("background_pairs_requested"), 0)
    positive_rows = _as_int(meta.get("positive_query_rows"), 0)
    negative_rows = _as_int(meta.get("negative_query_rows"), 0)
    posonly_rows = _as_int(meta.get("positive_only_query_rows"), 0)
    mixed_rows = _as_int(meta.get("mixed_query_rows"), 0)

    objective = (
        f"[E{epoch:03d} S{step:04d} G{global_step:07d}] objective "
        f"L={_as_float(snapshot.get('loss'), float('nan')):.5f} "
        f"G={_as_float(snapshot.get('gold_asl'), float('nan')):.6f} "
        f"P={_as_float(snapshot.get('pseudo_asl'), float('nan')):.5f} "
        f"cG={_as_float(snapshot.get('contrib_gold_asl'), float('nan')):.6f} "
        f"cP={_as_float(snapshot.get('contrib_pseudo_asl'), float('nan')):.6f} "
        f"cA={_as_float(snapshot.get('contrib_base_anchor'), float('nan')):.6f} "
        f"H={_as_float(snapshot.get('hierarchy'), float('nan')):.6f} "
        f"cH={_as_float(snapshot.get('contrib_hierarchy'), float('nan')):.6f}"
    )
    sampling = (
        "    sampling  "
        f"gp={_as_int(meta.get('gold_pairs_retained'), 0)} "
        f"hd={_as_int(meta.get('hard_pairs_retained'), 0)} "
        f"pp={_as_int(meta.get('pseudo_pairs_retained'), 0)} "
        f"bg={bg_retained}/{bg_requested}({_ratio(bg_retained, bg_requested):.1%}) "
        f"rows=pos:{positive_rows} neg:{negative_rows} posonly:{posonly_rows} mixed:{mixed_rows} "
        f"capdrop={_as_float(meta.get('candidate_truncation_fraction'), 0.0):.1%}"
    )
    weak = (
        "    weak      "
        f"wfq={_as_int(meta.get('weak_focus_queries_realized'), 0)}/"
        f"{_as_int(meta.get('weak_focus_queries_requested'), 0)} "
        f"wfa={_as_int(meta.get('weak_focus_anchor_new_count'), 0)} "
        f"wpa={_as_int(meta.get('weak_primary_anchor_retained'), 0)} "
        f"wpq={_as_int(meta.get('weak_primary_candidate_query_anchor_count'), 0)} "
        f"wphit={_as_float(meta.get('weak_primary_candidate_query_hit_rate'), 0.0):.1%} "
        f"wpr={_as_int(meta.get('weak_primary_remaining'), 0)} "
        f"wft={_as_int(meta.get('weak_focus_targets_retained'), 0)} "
        f"qps={_as_int(meta.get('query_with_pseudo_pool'), 0)} "
        f"qcan={_as_int(meta.get('query_with_backbone_candidate_pool'), 0)}"
    )
    runtime = (
        "    runtime   "
        f"hpair={_as_int(snapshot.get('hierarchy_pairs'), 0)} "
        f"qseen={_as_int(snapshot.get('query_seen'), 0)} "
        f"gscale={_as_float(snapshot.get('graph_delta_scale'), float('nan')):.5f} "
        f"lr={_as_float(snapshot.get('lr'), float('nan')):.2e} "
        f"slr={_as_float(snapshot.get('scale_lr'), float('nan')):.2e} "
        f"mem={_memory_label(snapshot)}"
    )
    return objective, sampling, weak, runtime


def build_epoch_summary_lines(epoch: int, metrics: Mapping[str, Any]) -> tuple[str, ...]:
    """Format the full epoch diagnostics as readable semantic groups.

    All raw values remain in ``training_history.json``; this function only
    changes terminal presentation.
    """

    protein_universe = metrics.get("protein_universe")
    protein_universe = protein_universe if isinstance(protein_universe, Mapping) else {}
    role_counts = protein_universe.get("role_counts")
    role_counts = role_counts if isinstance(role_counts, Mapping) else {}
    ontology_universe = metrics.get("ontology_universe")
    ontology_universe = ontology_universe if isinstance(ontology_universe, Mapping) else {}
    learning_rates = list(metrics.get("learning_rates") or [])
    lr = _as_float(learning_rates[0], float("nan")) if learning_rates else float("nan")
    slr = _as_float(learning_rates[1], float("nan")) if len(learning_rates) > 1 else float("nan")

    header = (
        f"epoch={epoch} complete "
        f"loss={_as_float(metrics.get('total'), float('nan')):.6f} "
        f"elapsed={_as_float(metrics.get('elapsed_seconds'), 0.0):.3f}s"
    )
    objective = (
        "  objective  "
        f"gold={_as_float(metrics.get('gold_asl'), float('nan')):.6f} "
        f"pseudo={_as_float(metrics.get('pseudo_asl'), float('nan')):.6f} "
        f"c_gold={_as_float(metrics.get('contrib_gold_asl'), float('nan')):.6f} "
        f"c_pseudo={_as_float(metrics.get('contrib_pseudo_asl'), float('nan')):.6f} "
        f"c_anchor={_as_float(metrics.get('contrib_base_anchor'), float('nan')):.6f} "
        f"hier={_as_float(metrics.get('hierarchy'), float('nan')):.6f} "
        f"c_hier={_as_float(metrics.get('contrib_hierarchy'), float('nan')):.6f} "
        f"hier_pairs={_as_float(metrics.get('hierarchy_pairs'), 0.0):.1f}"
    )
    sampling = (
        "  sampling   "
        f"gpos={_as_float(metrics.get('avg_gold_pairs_retained'), 0.0):.1f} "
        f"hard={_as_float(metrics.get('avg_hard_pairs_retained'), 0.0):.1f} "
        f"ppos={_as_float(metrics.get('avg_pseudo_pairs_retained'), 0.0):.1f} "
        f"bg={_as_float(metrics.get('avg_background_pairs_retained'), 0.0):.1f} "
        f"bg_fill={_as_float(metrics.get('background_pair_fill_rate'), 1.0):.1%} "
        f"negrows={_as_float(metrics.get('negative_query_row_rate'), 0.0):.1%}"
        f"[{_as_int(metrics.get('negative_query_row_occurrences'), 0)}/{_as_int(metrics.get('query_occurrences_total'), 0)}] "
        f"posonly={_as_float(metrics.get('positive_only_query_row_rate'), 0.0):.1%}"
        f"[{_as_int(metrics.get('positive_only_query_row_occurrences'), 0)}/{_as_int(metrics.get('query_occurrences_total'), 0)}] "
        f"mixed={_as_float(metrics.get('mixed_query_row_rate'), 0.0):.1%} "
        f"np={_as_float(metrics.get('negative_positive_pair_ratio_global'), 0.0):.2f} "
        f"rowpart={'ok' if metrics.get('query_row_partition_ok', False) else 'FAIL'}"
    )
    query = (
        "  query      "
        f"qcov={_as_int(metrics.get('unique_query_go_count'), 0)}/"
        f"{_as_int(metrics.get('eligible_go_count'), 0)}({_as_float(metrics.get('query_coverage_rate'), 0.0):.2%}) "
        f"q1={_as_int(metrics.get('unique_query_go_eq1'), 0)}/{_as_int(metrics.get('eligible_query_go_eq1'), 0)} "
        f"q2={_as_int(metrics.get('unique_query_go_eq2'), 0)}/{_as_int(metrics.get('eligible_query_go_eq2'), 0)} "
        f"q3_4={_as_int(metrics.get('unique_query_go_3_4'), 0)}/{_as_int(metrics.get('eligible_query_go_3_4'), 0)} "
        f"qgt4={_as_int(metrics.get('unique_query_go_gt4'), 0)}/{_as_int(metrics.get('eligible_query_go_gt4'), 0)} "
        f"qcan={_as_float(metrics.get('query_with_backbone_candidate_pool_rate'), 0.0):.1%} "
        f"qps={_as_float(metrics.get('query_with_pseudo_pool_rate'), 0.0):.1%}"
    )
    weak = (
        "  weak       "
        f"wfq={_as_float(metrics.get('avg_weak_focus_queries_realized'), 0.0):.1f}/"
        f"{_as_float(metrics.get('avg_weak_focus_queries_requested'), 0.0):.1f} "
        f"wfa={_as_float(metrics.get('avg_weak_focus_anchor_new_count'), 0.0):.1f} "
        f"wpa={_as_float(metrics.get('avg_weak_primary_anchor_retained'), 0.0):.1f} "
        f"wpq={_as_float(metrics.get('weak_primary_candidate_query_hit_rate_global'), 0.0):.1%} "
        f"wft={_as_float(metrics.get('avg_weak_focus_targets_retained'), 0.0):.1f} "
        f"wft_fill={_as_float(metrics.get('weak_focus_target_fill_rate'), 1.0):.1%} "
        f"wft_keep={_as_float(metrics.get('weak_focus_target_retention_rate'), 1.0):.1%} "
        f"hard_keep={_as_float(metrics.get('hard_pair_retention_rate'), 1.0):.1%} "
        f"cand_keep={_as_float(metrics.get('candidate_capacity_retention_rate'), 1.0):.1%}"
    )
    protein = (
        "  protein    "
        f"root={_as_int(metrics.get('unique_root_protein_count'), 0)}/{_as_int(protein_universe.get('total'), 0)}"
        f"({_as_float(metrics.get('root_protein_coverage_rate'), 0.0):.1%}) "
        f"wroot={_as_int(metrics.get('unique_root_weak_protein_count'), 0)}/{_as_int(role_counts.get('weak'), 0)}"
        f"({_as_float(metrics.get('root_weak_protein_coverage_rate'), 0.0):.1%}) "
        f"local={_as_int(metrics.get('unique_local_protein_count'), 0)}/{_as_int(protein_universe.get('total'), 0)}"
        f"({_as_float(metrics.get('local_protein_coverage_rate'), 0.0):.1%}) "
        f"coreG={_as_int(metrics.get('unique_gold_target_protein_count'), 0)}/{_as_int(role_counts.get('core'), 0)}"
        f"({_as_float(metrics.get('gold_target_protein_coverage_rate'), 0.0):.1%}) "
        f"pweak={_as_int(metrics.get('unique_pseudo_target_protein_count'), 0)}/"
        f"{_as_int(protein_universe.get('pseudo_eligible_active_weak', protein_universe.get('pseudo_active_weak')), 0)}"
        f"({_as_float(metrics.get('pseudo_target_eligible_weak_coverage_rate'), 0.0):.1%}) "
        f"puniq={_as_float(metrics.get('pseudo_target_unique_efficiency'), 0.0):.1%} "
        f"wprim={_as_int(metrics.get('unique_weak_primary_anchor_protein_count'), 0)}/"
        f"{_as_int(protein_universe.get('pseudo_eligible_active_weak'), 0)}"
        f"({_as_float(metrics.get('weak_primary_anchor_eligible_coverage_rate'), 0.0):.1%})"
    )
    ontology = (
        "  ontology   "
        f"local={_as_int(metrics.get('unique_local_ontology_go_count'), 0)}/{_as_int(ontology_universe.get('full_ontology_rows'), 0)}"
        f"({_as_float(metrics.get('local_ontology_go_coverage_rate'), 0.0):.1%}) "
        f"non_task={_as_int(metrics.get('unique_non_task_ontology_go_count'), 0)}/{_as_int(ontology_universe.get('non_task_ontology_rows'), 0)}"
        f"({_as_float(metrics.get('non_task_ontology_go_coverage_rate'), 0.0):.1%}) "
        f"task_ctx={_as_int(metrics.get('unique_context_only_task_ontology_go_count'), 0)}/{_as_int(ontology_universe.get('unique_context_only_task_ontology_rows'), 0)}"
        f"({_as_float(metrics.get('context_only_task_ontology_go_coverage_rate'), 0.0):.1%})"
    )
    runtime = (
        "  runtime    "
        f"cpu_mat={_as_float(metrics.get('avg_loader_materialize_seconds'), 0.0):.3f}s "
        f"cand={_as_float(metrics.get('avg_candidate_gather_seconds'), 0.0):.3f}s "
        f"localize={_as_float(metrics.get('avg_edge_localize_seconds'), 0.0):.3f}s "
        f"graph={_as_float(metrics.get('avg_graph_build_seconds'), 0.0):.3f}s "
        f"lr={lr:.2e} slr={slr:.2e} "
        f"peak={_as_float(metrics.get('gpu_peak_allocated_gb'), float('nan')):.2f}/"
        f"{_as_float(metrics.get('gpu_peak_reserved_gb'), float('nan')):.2f}GB "
        f"end_resv={_as_float(metrics.get('gpu_epoch_end_reserved_before_empty_cache_gb'), float('nan')):.2f}->"
        f"{_as_float(metrics.get('gpu_epoch_end_reserved_after_empty_cache_gb'), float('nan')):.2f}GB "
        f"cache_rel={_as_float(metrics.get('gpu_epoch_end_cache_released_gb'), float('nan')):.2f}GB"
    )
    return header, objective, sampling, query, weak, protein, ontology, runtime
