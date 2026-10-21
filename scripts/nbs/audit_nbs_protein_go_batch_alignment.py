#!/usr/bin/env python3
"""Audit protein--GO query/loss alignment without running the NBS model.

The script loads the production mmap/CSR stores, iterates only the episode
sampler, and reports alignment from the sampled weak protein's point of view.
It is intentionally CPU-only and does not materialize PyG graphs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
MODEL_ROOT = ROOT / "nbs_models" / "nbs_protein_go"
for value in (ROOT, MODEL_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from nbs_pg.local_loader import build_latence_nbs_stores  # noqa: E402


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in ("mean", "p05", "p25", "p50", "p75", "p95")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _eligible_pseudo(
    stores: Any, protein: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampler = stores.episode_sampler
    role_row = int(stores.registry.role_row[int(protein)])
    values = sampler.pseudo_by_protein.get_role_row(role_row)
    go = np.asarray(values["go_idx"], dtype=np.int64)
    probability = np.asarray(values["probability"], dtype=np.float32)
    keep = sampler.eligible_go_mask[go]
    keep &= probability >= float(sampler.config.weak_focus_min_probability)
    go = go[keep]
    probability = probability[keep]
    if sampler.candidate_by_protein is None:
        candidate_edge = np.empty((2, 0), dtype=np.int64)
    else:
        candidate_edge, _ = sampler.candidate_by_protein.gather([int(protein)])
    candidate_go = (
        np.asarray(candidate_edge[1], dtype=np.int64)
        if candidate_edge.size
        else np.empty(0, dtype=np.int64)
    )
    intersection = go[np.isin(go, candidate_go, assume_unique=False)]
    return go, probability, intersection


def _row_for_anchor(
    stores: Any,
    episode: Any,
    *,
    epoch: int,
    episode_index: int,
    protein: int,
) -> dict[str, Any]:
    eligible_go, _, intersection_go = _eligible_pseudo(stores, protein)
    query_go = np.asarray(episode.query_go_idx, dtype=np.int64)
    column_values = np.flatnonzero(
        np.asarray(episode.candidate_protein_idx, dtype=np.int64) == int(protein)
    )
    if column_values.size != 1:
        raise RuntimeError(
            f"weak-primary protein {protein} occurs {column_values.size} times in candidate union"
        )
    column = int(column_values[0])
    pseudo_loss_rows = np.flatnonzero(
        np.asarray(episode.pseudo_mask[:, column], dtype=np.bool_)
    )
    supervised_rows = np.flatnonzero(
        np.asarray(episode.mask[:, column], dtype=np.bool_)
    )
    positive_rows = np.flatnonzero(
        np.asarray(episode.mask[:, column], dtype=np.bool_)
        & (np.asarray(episode.labels[:, column]) > 0)
    )
    negative_rows = np.flatnonzero(
        np.asarray(episode.mask[:, column], dtype=np.bool_)
        & ~np.asarray(episode.pseudo_mask[:, column], dtype=np.bool_)
        & (np.asarray(episode.labels[:, column]) <= 0)
    )
    weak_primary_mask = getattr(episode, "weak_primary_mask", None)
    primary_rows = (
        np.empty(0, dtype=np.int64)
        if weak_primary_mask is None
        else np.flatnonzero(
            np.asarray(weak_primary_mask[:, column], dtype=np.bool_)
        )
    )
    query_hits = np.intersect1d(eligible_go, query_go, assume_unique=False)
    loss_go = query_go[pseudo_loss_rows]
    candidate_go = set(stores.episode_sampler._weak_primary_anchor_candidate_attr_current.get(int(protein), {}))
    # Legacy audit also resolves membership independently of evidence/masks.
    if stores.episode_sampler.candidate_by_protein is not None:
        edges, _ = stores.episode_sampler.candidate_by_protein.gather([int(protein)])
        candidate_go = set(np.asarray(edges[1], dtype=np.int64).tolist())
    hard_rows = np.asarray([r for r in negative_rows if int(query_go[r]) in candidate_go], dtype=np.int64)
    background_rows = np.asarray([r for r in negative_rows if int(query_go[r]) not in candidate_go], dtype=np.int64)
    expected_primary_pos = min(int(stores.episode_sampler.config.weak_primary_positive_go_per_protein), int(eligible_go.size))
    denominator = max(1, int(eligible_go.size))
    confidence = np.asarray(episode.confidence[:, column], dtype=np.float64)
    supervision_weight = np.asarray(
        episode.supervision_weight[:, column], dtype=np.float64
    )
    positive_effective_weight = float(
        np.sum(confidence[pseudo_loss_rows] * supervision_weight[pseudo_loss_rows])
    )
    negative_effective_weight = float(np.sum(supervision_weight[negative_rows]))
    return {
        "epoch": int(epoch),
        "episode": int(episode_index),
        "protein_idx": int(protein),
        "eligible_pseudo_go": int(eligible_go.size),
        "pseudo_candidate_intersection_go": int(intersection_go.size),
        "query_pseudo_hits": int(query_hits.size),
        "loss_pseudo_hits": int(loss_go.size),
        "positive_supervised_go": int(positive_rows.size),
        "negative_supervised_go": int(negative_rows.size),
        "hard_pu_go": int(hard_rows.size),
        "background_pu_go": int(background_rows.size),
        "positive_quota_met": int(loss_go.size >= expected_primary_pos),
        "hard_quota_met": int(hard_rows.size >= stores.episode_sampler.config.weak_primary_hard_negative_go_per_protein),
        "query_to_loss_positive_gap": int(query_hits.size - loss_go.size),
        "primary_column_go": int(primary_rows.size),
        "unknown_query_go": int(query_go.size - supervised_rows.size),
        "query_recall": float(query_hits.size / denominator),
        "loss_recall": float(loss_go.size / denominator),
        "candidate_intersection_fraction": float(
            intersection_go.size / denominator
        ),
        "positive_effective_weight": positive_effective_weight,
        "negative_effective_weight": negative_effective_weight,
        "effective_negative_positive_weight_ratio": float(
            negative_effective_weight / max(positive_effective_weight, 1e-12)
        ),
        "query_hit_go_idx": ";".join(str(int(value)) for value in query_hits),
        "loss_positive_go_idx": ";".join(str(int(value)) for value in loss_go),
        "loss_hard_go_idx": ";".join(str(int(query_go[r])) for r in hard_rows),
        "loss_background_go_idx": ";".join(str(int(query_go[r])) for r in background_rows),
    }


def _summary(
    rows: list[dict[str, Any]],
    batches: list[dict[str, Any]],
    *,
    config_path: Path,
) -> dict[str, Any]:
    numeric = (
        "eligible_pseudo_go",
        "pseudo_candidate_intersection_go",
        "query_pseudo_hits",
        "loss_pseudo_hits",
        "positive_supervised_go",
        "negative_supervised_go",
        "hard_pu_go", "background_pu_go", "query_to_loss_positive_gap",
        "primary_column_go",
        "unknown_query_go",
        "query_recall",
        "loss_recall",
        "candidate_intersection_fraction",
        "positive_effective_weight",
        "negative_effective_weight",
        "effective_negative_positive_weight_ratio",
    )
    per_protein_go: dict[int, set[int]] = defaultdict(set)
    per_protein_epochs: dict[int, dict[int, set[int]]] = defaultdict(dict)
    per_protein_hard: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        protein = int(row["protein_idx"])
        epoch = int(row["epoch"])
        per_protein_epochs[protein].setdefault(epoch, set()).update(int(x) for x in str(row["loss_positive_go_idx"]).split(";") if x)
        per_protein_hard[protein].update(int(x) for x in str(row["loss_hard_go_idx"]).split(";") if x)
        for token in str(row["loss_positive_go_idx"]).split(";"):
            if token:
                per_protein_go[int(row["protein_idx"])].add(int(token))
    repeated = {p: values for p, values in per_protein_epochs.items() if len(values) >= 2}
    gains, jaccard = [], []
    for values in repeated.values():
        ordered = [values[e] for e in sorted(values)]
        gains.append(len(set.union(*ordered) - ordered[0]))
        for a, b in zip(ordered, ordered[1:]):
            jaccard.append(len(a & b) / max(1, len(a | b)))
    return {
        "schema_version": 2,
        "weight_note": "effective weights are pre-ASL confidence x supervision_weight; neither branch coefficient nor actual gradient",
        "cross_epoch_repeated_proteins": len(repeated),
        "cross_epoch_new_positive_go_after_first": _quantiles(gains),
        "cross_epoch_adjacent_positive_jaccard": _quantiles(jaccard),
        "cross_epoch_unique_hard_go": _quantiles([len(v) for v in per_protein_hard.values()]),
        "config": str(config_path.resolve()),
        "anchors_audited": int(len(rows)),
        "unique_proteins_audited": int(len(per_protein_go)),
        "batches_audited": int(len(batches)),
        "anchor_metrics": {
            name: _quantiles([float(row[name]) for row in rows])
            for name in numeric
        },
        "anchor_rates": {
            "positive_quota_met": float(np.mean([r["positive_quota_met"] for r in rows])) if rows else 0.0,
            "hard_quota_met": float(np.mean([r["hard_quota_met"] for r in rows])) if rows else 0.0,
            "all_query_positives_supervised": float(np.mean([r["query_to_loss_positive_gap"] == 0 for r in rows])) if rows else 0.0,
            "query_hit_at_least_1": float(
                np.mean([row["query_pseudo_hits"] >= 1 for row in rows])
            ) if rows else 0.0,
            "loss_positive_at_least_1": float(
                np.mean([row["loss_pseudo_hits"] >= 1 for row in rows])
            ) if rows else 0.0,
            "loss_positive_at_least_2": float(
                np.mean([row["loss_pseudo_hits"] >= 2 for row in rows])
            ) if rows else 0.0,
            "loss_positive_at_least_4": float(
                np.mean([row["loss_pseudo_hits"] >= 4 for row in rows])
            ) if rows else 0.0,
            "has_controlled_negative": float(
                np.mean([row["negative_supervised_go"] >= 1 for row in rows])
            ) if rows else 0.0,
        },
        "cross_epoch_unique_loss_go": _quantiles(
            [float(len(values)) for values in per_protein_go.values()]
        ),
        "batch_metrics": batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fixed-cohort", action="store_true", help="Replay the same ordered primary cohort across epochs to isolate GO rotation; not a coverage estimate")
    parser.add_argument("--episodes-per-epoch", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must lie in [0, world_size)")
    if args.epochs <= 0 or args.episodes_per_epoch <= 0:
        raise ValueError("epochs and episodes-per-epoch must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stores = build_latence_nbs_stores(config)
    sampler = stores.episode_sampler
    if int(sampler.config.weak_primary_proteins_per_episode) <= 0:
        raise ValueError("alignment audit requires weak-primary sampling")

    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    cohort = None
    for epoch_offset in range(args.epochs):
        epoch = epoch_offset + 1
        if args.fixed_cohort:
            sampler._ensure_weak_primary_epoch(epoch=epoch, rank=args.rank, world_size=args.world_size)
            if cohort is None:
                count = args.episodes_per_epoch * int(sampler.config.weak_primary_proteins_per_episode)
                cohort = list(sampler._weak_primary_pending)[:count]
            # Audit-only controlled replay. Production queues are unchanged.
            sampler._weak_primary_pending = deque(cohort)
            sampler._weak_primary_total_owned = len(cohort)
            sampler._weak_primary_selected = 0
        for episode_index in range(args.episodes_per_epoch):
            global_episode = episode_index * args.world_size + args.rank
            episode = sampler.sample(
                seed=args.seed + epoch * 1_000_003 + global_episode * 97,
                epoch=epoch,
                global_episode=global_episode,
                rank=args.rank,
                world_size=args.world_size,
            )
            anchors = np.asarray(
                episode.metadata.get(
                    "weak_primary_anchor_protein_idx",
                    np.empty(0, dtype=np.int64),
                ),
                dtype=np.int64,
            )
            batch_rows = [
                _row_for_anchor(
                    stores,
                    episode,
                    epoch=epoch,
                    episode_index=episode_index,
                    protein=int(protein),
                )
                for protein in anchors.tolist()
            ]
            if sampler.config.weak_primary_complete_query_positives and any(row["query_to_loss_positive_gap"] != 0 for row in batch_rows):
                raise RuntimeError("known in-query primary pseudo labels are missing from loss")
            rows.extend(batch_rows)
            evidence_store = sampler.candidate_by_protein
            evidence_mismatches = None
            if evidence_store is not None:
                expected = evidence_store.gather_matrix(episode.candidate_protein_idx, episode.query_go_idx)
                evidence_mismatches = int(np.count_nonzero(np.any(np.abs(expected - episode.candidate_evidence) > 1e-6, axis=2)))
                if sampler.config.candidate_evidence_scope == "decoded_all" and evidence_mismatches:
                    raise RuntimeError(f"decoded evidence mismatch: {evidence_mismatches} pairs")
            batches.append(
                {
                    "evidence_mismatched_pairs": evidence_mismatches,
                    "decoded_pairs": int(episode.mask.size),
                    "supervised_pairs": int(np.count_nonzero(episode.mask)),
                    "hard_query_realized": episode.metadata.get("weak_primary_hard_query_realized", 0),
                    "decoded_cross_pseudo_pairs": episode.metadata.get("decoded_cross_pseudo_pairs", 0),
                    "epoch": epoch,
                    "episode": episode_index,
                    "query_go": int(episode.query_go_idx.size),
                    "weak_primary_anchors": int(anchors.size),
                    "unique_query_go": int(np.unique(episode.query_go_idx).size),
                    "mean_query_recall": float(
                        np.mean([row["query_recall"] for row in batch_rows])
                    ) if batch_rows else 0.0,
                    "mean_loss_recall": float(
                        np.mean([row["loss_recall"] for row in batch_rows])
                    ) if batch_rows else 0.0,
                    "multi_positive_anchor_rate": float(
                        np.mean([row["loss_pseudo_hits"] >= 2 for row in batch_rows])
                    ) if batch_rows else 0.0,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "anchor_alignment.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = _summary(rows, batches, config_path=args.config)
    summary["cohort_mode"] = "fixed_ordered_replay" if args.fixed_cohort else "production_epoch_queue"
    summary_path = args.output_dir / "alignment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["anchor_rates"], indent=2, ensure_ascii=False))
    print(f"wrote: {summary_path}")
    print(f"wrote: {csv_path}")


if __name__ == "__main__":
    main()
