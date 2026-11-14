#!/usr/bin/env python3
"""Rank proteins for each unseen GO over a complete v095 prediction cohort."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile

import numpy as np

from compare_unseen_go_controls_v095 import anchor_plan, load_anchor_scores
from eval_unseen_go_v095 import load_inputs, sha256, write_tsv


def term_ranks(scores, positive_rows, top_k):
    """Rank the positives and compute observed AP, grouping exact score ties."""
    ordered = np.sort(scores)
    positive_scores = scores[positive_rows]
    left = np.searchsorted(ordered, positive_scores, side="left")
    right = np.searchsorted(ordered, positive_scores, side="right")
    higher = len(scores) - right
    tied = right - left
    _, groups, counts = np.unique(positive_scores, return_inverse=True, return_counts=True)
    positives_at_threshold = np.cumsum(counts[::-1])[::-1][groups]
    midrank = higher + (tied + 1) / 2
    metrics = {
        "observed_ap_proxy": float(np.mean(positives_at_threshold / (higher + tied))),
        "mean_positive_midrank": float(midrank.mean()),
        "mean_positive_fraction_ranked_ahead": float(((midrank - 1) / max(len(scores) - 1, 1)).mean()),
    }
    hits = {}
    for k in top_k:
        effective_k = min(k, len(scores))
        credit = np.clip(effective_k - higher, 0, tied) / tied
        hits[k] = credit
        metrics[f"observed_recall@{k}"] = float(credit.mean())
        metrics[f"observed_precision_proxy@{k}"] = float(credit.sum() / effective_k)
    return metrics, (positive_scores, higher, tied, midrank, hits)


def evaluate_retrieval(candidates, pool_ids, positives_by_go, column_scores, top_k):
    go_results, pair_results = [], []
    pool_size = len(pool_ids)
    pool_index = {protein: row for row, protein in enumerate(pool_ids)}
    for col, go in enumerate(candidates):
        hits = positives_by_go[go]
        record = {"go_id": go, "n_proteins_ranked": pool_size, "n_observed_positives": len(hits)}
        if not hits:
            record.update(observed_ap_proxy=None, mean_positive_midrank=None,
                          mean_positive_fraction_ranked_ahead=None)
            for k in top_k:
                record[f"observed_recall@{k}"] = None
                record[f"observed_precision_proxy@{k}"] = None
            go_results.append(record)
            continue

        rows = np.asarray([pool_index[protein] for protein in hits], dtype=np.intp)
        scores, values = term_ranks(column_scores(col, go), rows, top_k)
        go_results.append({**record, **scores})
        positive_scores, higher, tied, midrank, credits = values
        for i, protein in enumerate(hits):
            pair_results.append({
                "go_id": go,
                "protein_id": protein,
                "score": float(positive_scores[i]),
                "rank_min": int(higher[i]) + 1,
                "rank_max": int(higher[i] + tied[i]),
                "midrank": float(midrank[i]),
                "tie_count": int(tied[i]),
                **{f"top@{k}_credit": float(credits[k][i]) for k in top_k},
            })

    observed = [record for record in go_results if record["n_observed_positives"]]
    result = {
        "candidate_go_count": len(candidates),
        "observed_positive_go_count": len(observed),
        "n_proteins_ranked_per_go": pool_size,
        "n_observed_positive_pairs": len(pair_results),
        "n_observed_positive_proteins": len({row["protein_id"] for row in pair_results}),
        "go_macro_observed_ap_proxy": float(np.mean([row["observed_ap_proxy"] for row in observed]))
        if observed else None,
        "go_macro_mean_positive_fraction_ranked_ahead": float(np.mean(
            [row["mean_positive_fraction_ranked_ahead"] for row in observed]
        )) if observed else None,
    }
    for k in top_k:
        effective_k = min(k, pool_size)
        result[f"effective_k@{k}"] = effective_k
        result[f"uniform_random_recall@{k}"] = effective_k / pool_size
        for suffix in ("observed_recall", "observed_precision_proxy"):
            key = f"{suffix}@{k}"
            result[f"go_macro_{key}"] = float(np.mean([row[key] for row in observed])) if observed else None
        result[f"pair_observed_recall@{k}"] = float(np.mean(
            [row[f"top@{k}_credit"] for row in pair_results]
        )) if pair_results else None
    return result, go_results, pair_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--cohort", default="ind_test_exp")
    parser.add_argument("--protein-pool", choices=("full", "evaluable"), default="full",
                        help="Rank the full predicted cohort or its identity-evaluable subset")
    parser.add_argument("--top-k", type=int, nargs="+", default=(10, 50, 100, 500, 1000))
    parser.add_argument("--extension-dir", type=Path)
    parser.add_argument("--anchor-predictions-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.top_k) < 1 or len(set(args.top_k)) != len(args.top_k):
        parser.error("--top-k must contain distinct positive integers")
    if (args.extension_dir is None) != (args.anchor_predictions_dir is None):
        parser.error("--extension-dir and --anchor-predictions-dir must be provided together")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"Choose a new retrieval output directory: {args.output_dir}")
    (test, cohort, pred, pred_path, candidates, positives, _, _, _, _, ids, probability) = load_inputs(
        args.test_manifest, args.predictions_dir, args.cohort, with_full_scores=True,
        materialize_positive_scores=False,
    )
    if args.protein_pool == "full":
        ids_to_rank = ids
        score_rows = slice(None)
    else:
        eligible = set(cohort["evaluable_protein_ids"])
        score_rows = np.asarray([i for i, protein in enumerate(ids) if protein in eligible], dtype=np.intp)
        ids_to_rank = [ids[i] for i in score_rows]
    if not ids_to_rank:
        raise ValueError("The selected protein pool is empty")

    by_go = defaultdict(list)
    for protein, observed in positives.items():
        for go in observed:
            by_go[go].append(protein)
    expanded, expanded_go, expanded_pairs = evaluate_retrieval(
        candidates, ids_to_rank, by_go, lambda col, _: probability[score_rows, col], args.top_k,
    )
    result = {
        "schema": "latence-unseen-go-protein-retrieval-v095-v1",
        "complete": True,
        "status": "ok" if expanded_pairs else "no_observed_positive_metrics_undefined",
        "namespace": test["namespace"],
        "cohort": args.cohort,
        "cohort_role": cohort.get("role"),
        "evidence_type": cohort.get("evidence_type"),
        "protein_pool": args.protein_pool,
        "n_proteins_predicted": len(ids),
        "n_identity_evaluable": len(cohort["evaluable_protein_ids"]),
        "test_manifest_sha256": sha256(args.test_manifest),
        "expanded_prediction_manifest_sha256": sha256(pred_path),
        "checkpoint_epoch": pred.get("checkpoint_epoch"),
        "checkpoint_step": pred.get("checkpoint_step"),
        "definitions": {
            "scope": "For each unseen GO, rank the chosen complete protein pool. Protein-centric evaluation ranks GO within protein.",
            "unobserved": "Proteins without an observed positive for the GO remain in the ranking pool. They are unlabeled; AP and precision are observed-label proxies.",
            "macro": "Average over GO terms with at least one observed positive; other terms remain in the coverage table.",
            "ties": "For top-K and midrank use the expectation under uniform order inside exact ties. AP groups exact ties into thresholds.",
            "uniform_random": "The expected recall under a uniform protein ordering is effective K divided by the protein pool size.",
        },
        "expanded": expanded,
        "anchor_transfer": None,
    }

    if args.anchor_predictions_dir is not None:
        _, extension_hash, mapping = anchor_plan(test, args.extension_dir)
        _, anchor_path, anchor_ids, anchor_probs, _, columns = load_anchor_scores(
            args.anchor_predictions_dir, pred, cohort["protein_ids"], mapping,
        )
        index = {protein: row for row, protein in enumerate(anchor_ids)}
        anchor_rows = np.asarray([index[protein] for protein in ids_to_rank], dtype=np.intp)

        def anchor_column(_, go):
            return np.maximum.reduce([anchor_probs[anchor_rows, columns[row]] for row in mapping[go]])

        anchor, anchor_go, anchor_pairs = evaluate_retrieval(
            candidates, ids_to_rank, by_go, anchor_column, args.top_k,
        )
        result["anchor_transfer"] = anchor
        result["anchor_prediction_manifest_sha256"] = sha256(anchor_path)
        result["extension_manifest_sha256"] = extension_hash
        result["expanded_minus_anchor"] = {
            key: value - anchor[key] if value is not None else None
            for key, value in expanded.items() if key.startswith(("go_macro_", "pair_observed_"))
        }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unseen_retrieval.", dir=args.output_dir.parent) as directory:
        stage = Path(directory) / "output"
        stage.mkdir()
        (stage / "retrieval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                              encoding="utf-8")
        write_tsv(stage / "expanded_per_go.tsv", expanded_go, ["go_id"])
        write_tsv(stage / "expanded_positive_ranks.tsv", expanded_pairs, ["go_id", "protein_id"])
        if args.anchor_predictions_dir is not None:
            write_tsv(stage / "anchor_transfer_per_go.tsv", anchor_go, ["go_id"])
            write_tsv(stage / "anchor_transfer_positive_ranks.tsv", anchor_pairs, ["go_id", "protein_id"])
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(f"Choose a new retrieval output directory: {args.output_dir}")
        stage.rename(args.output_dir)
    print(args.output_dir / "retrieval.json")


if __name__ == "__main__":
    main()
