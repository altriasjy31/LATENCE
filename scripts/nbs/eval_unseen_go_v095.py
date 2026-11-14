#!/usr/bin/env python3
"""Evaluate anchored temporal unseen-GO predictions on observed direct labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np


TEST_SCHEMA = "latence-unseen-go-test-v1"
PREDICTION_SCHEMA = "nbs-v095-requested-go-prediction-v1"
GO_ID = re.compile(r"GO:\d{7}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path, schema):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != schema or data.get("complete") is not True:
        raise ValueError(f"Expected completed {schema} manifest at {path}")
    return data


def checked_ids(values, label, *, allow_empty=False):
    if (not isinstance(values, list) or (not values and not allow_empty)
            or any(not isinstance(value, str) or not value or value != value.strip()
                   or "\r" in value or "\n" in value for value in values)
            or len(values) != len(set(values))):
        raise ValueError(f"{label} must contain unique, nonempty, canonical strings")
    return values


def prediction_file(folder, manifest, key):
    filename = manifest[key]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"Invalid prediction artifact filename for {key}")
    return folder / filename


def axis_hashes(values):
    # v095 hashes the source bytes, then writes the exported ID list with LF
    # and a terminal newline. Account for the equivalent source spellings.
    return {
        hashlib.sha256((newline.join(values) + ending).encode("utf-8")).hexdigest()
        for newline in ("\n", "\r\n", "\r")
        for ending in ("", newline)
    }


def load_score_matrix(prediction_dir, pred, cohort_ids):
    """Read the hashed prediction matrix and check its full protein axis."""
    protein_path = prediction_file(prediction_dir, pred, "protein_ids_file")
    pred_ids = checked_ids(protein_path.read_text(encoding="utf-8").splitlines(), "predicted protein axis")
    if pred["identity"]["protein_ids_sha256"] not in axis_hashes(pred_ids):
        raise ValueError("Predicted protein ID order does not match the source ID hash")
    if pred["num_proteins"] != len(pred_ids):
        raise ValueError("Prediction manifest protein count differs from the protein ID axis")
    if set(pred_ids) != set(cohort_ids):
        missing = sorted(set(cohort_ids) - set(pred_ids))
        extra = sorted(set(pred_ids) - set(cohort_ids))
        raise ValueError(
            f"Prediction protein set must equal the full cohort: "
            f"{len(missing)} missing ({missing[:5]}), {len(extra)} extra ({extra[:5]})"
        )
    probability_path = prediction_file(prediction_dir, pred, "probability_file")
    if pred["output_probability_sha256"] != sha256(probability_path):
        raise ValueError("Prediction probability SHA256 differs from the manifest")
    probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if probability.dtype != np.float32 or probability.shape != (len(pred_ids), pred["num_target_GO"]):
        raise ValueError("Expected float32 predictions shaped (proteins, candidate GO terms)")
    for start in range(0, len(pred_ids), 4096):
        block = probability[start:start + 4096]
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
            raise ValueError("Prediction contains nonfinite values or scores outside [0, 1]")
    return pred_ids, probability


def load_inputs(test_path, prediction_dir, cohort_name, *, with_full_scores=False,
                materialize_positive_scores=True):
    test = read_manifest(test_path, TEST_SCHEMA)
    candidates = checked_ids(test["candidate_go_ids"], "candidate GO axis")
    if any(GO_ID.fullmatch(go) is None for go in candidates):
        raise ValueError("Candidate axis must use canonical GO:NNNNNNN identifiers")
    rows = test["candidate_ontology_rows"]
    if (not isinstance(rows, list) or len(rows) != len(candidates)
            or any(type(row) is not int or row < 0 for row in rows)
            or len(rows) != len(set(rows))):
        raise ValueError("Invalid candidate ontology rows in the test manifest")
    cohort = test["cohorts"].get(cohort_name)
    if cohort is None:
        raise ValueError(f"Unknown cohort {cohort_name!r}; choose from {list(test['cohorts'])}")
    cohort_ids = checked_ids(cohort["protein_ids"], "cohort protein axis", allow_empty=True)
    eligible = checked_ids(cohort["evaluable_protein_ids"], "evaluable protein axis", allow_empty=True)
    eligible_set = set(eligible)
    if eligible != [protein for protein in cohort_ids if protein in eligible_set]:
        raise ValueError("Evaluable proteins must be an ordered subset of the cohort")
    go_columns = {go: column for column, go in enumerate(candidates)}
    positives = defaultdict(dict)
    for pair in cohort["positives"]:
        protein = pair["protein_id"]
        go = pair["go_id"]
        if protein not in eligible_set or go not in go_columns:
            raise ValueError(f"Observed positive is off the eligible axes: {protein}/{go}")
        if go in positives[protein]:
            raise ValueError(f"Duplicate positive pair: {protein}/{go}")
        positives[protein][go] = pair

    pred_path = prediction_dir / "prediction_manifest.json"
    pred = read_manifest(pred_path, PREDICTION_SCHEMA)
    identity = pred["identity"]
    bundle_hash = test["ontology_bundle_sha256"]
    extension = pred.get("ontology_extension")
    if (not isinstance(bundle_hash, str) or SHA256.fullmatch(bundle_hash) is None
            or identity["ontology_bundle_sha256"] != bundle_hash
            or not isinstance(extension, dict)):
        raise ValueError("Prediction ontology bundle does not match the test manifest")
    if (extension.get("schema") != "nbs-v095-anchored-ontology-replacement-v1"
            or extension.get("old_coordinate_anchors_verified") is not True
            or extension.get("new_trainable_parameter_rows_added") is not False
            or extension.get("num_ontology_go") != max(rows) + 1):
        raise ValueError("Prediction does not attest anchored v095 ontology extension over the declared rows")
    config_hash = test.get("v095_resolved_config_sha256")
    if config_hash is not None and config_hash != identity["resolved_config_sha256"]:
        raise ValueError("Prediction v095 resolved configuration does not match the test manifest")
    input_hash = cohort.get("input_manifest_sha256")
    if input_hash is not None and input_hash != identity["input_manifest_sha256"]:
        raise ValueError("Prediction feature input does not match the bound cohort manifest")
    if (identity["requested_go_ids"] != candidates or identity["target_ontology_rows"] != rows
            or pred["num_target_GO"] != len(candidates) or len(pred["targets"]) != len(candidates)):
        raise ValueError("Prediction must score the full ordered candidate GO axis from the test manifest")
    target_path = prediction_file(prediction_dir, pred, "target_go_file")
    with target_path.open(encoding="utf-8", newline="") as stream:
        target_table = list(csv.DictReader(stream, delimiter="\t"))
    if len(target_table) != len(candidates):
        raise ValueError("Prediction target table length differs from the candidate GO axis")
    for column, (go, row, target, record) in enumerate(
        zip(candidates, rows, pred["targets"], target_table)
    ):
        expected = {
            "output_column": column,
            "requested_go_id": go,
            "canonical_go_id": go,
            "ontology_row": row,
            "task_columns": [],
            "seen_task_columns": [],
            "stage2_supervised": False,
            "readout": "single_shared_GO_scorer",
            "zero_shot_scope": "Stage2_unsupervised_target_only",
        }
        if (any(target.get(key) != value for key, value in expected.items())
                or target.get("stage2_supervised") is not False
                or any(record.get(key) != str(value) for key, value in expected.items())):
            raise ValueError(f"Prediction target column {column} is not the declared new GO row for {go}")
    pred_ids, probability = load_score_matrix(prediction_dir, pred, cohort_ids)
    positive_ids = [protein for protein in eligible if protein in positives]
    scores = truth = None
    if materialize_positive_scores:
        row_index = {protein: row for row, protein in enumerate(pred_ids)}
        scores = probability[np.asarray([row_index[protein] for protein in positive_ids], dtype=np.int64)]
        truth = np.zeros(scores.shape, dtype=np.bool_)
        for row, protein in enumerate(positive_ids):
            truth[row, [go_columns[go] for go in positives[protein]]] = True
    loaded = (test, cohort, pred, pred_path, candidates, positives, positive_ids, scores, truth,
              pred_ids != cohort_ids)
    return (*loaded, pred_ids, probability) if with_full_scores else loaded


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
                     where=np.asarray(denominator) != 0)


def exact_threshold_metrics(scores, truth):
    """Evaluate every distinct threshold, grouping equal scores before scoring."""
    order = np.argsort(-scores, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    sorted_truth = np.take_along_axis(truth, order, axis=1)
    row_precision = np.cumsum(sorted_truth, axis=1) / np.arange(1, scores.shape[1] + 1)
    delta_precision = np.diff(row_precision, axis=1, prepend=0.0)
    delta_recall = sorted_truth / truth.sum(axis=1, keepdims=True)
    row_activations = np.zeros_like(sorted_truth)
    row_activations[:, 0] = True
    events = np.argsort(-sorted_scores.ravel(), kind="stable")
    thresholds = sorted_scores.ravel()[events]
    starts = np.r_[0, np.flatnonzero(thresholds[1:] != thresholds[:-1]) + 1]
    ends = np.r_[starts[1:], len(events)]
    tp_per_group = np.add.reduceat(sorted_truth.ravel()[events].astype(np.int64), starts)
    tp = np.cumsum(tp_per_group)
    total_tp = int(truth.sum())
    micro_recall = tp / total_tp
    micro_precision = tp / ends
    activated = np.cumsum(np.add.reduceat(row_activations.ravel()[events].astype(np.int64), starts))
    precision_sums = np.cumsum(np.add.reduceat(delta_precision.ravel()[events], starts))
    precision = np.clip(precision_sums / activated, 0, 1)
    recall_sums = np.cumsum(np.add.reduceat(delta_recall.ravel()[events], starts))
    recall = np.clip(recall_sums / len(scores), 0, 1)
    f1 = divide(2 * precision * recall, precision + recall)
    best = int(np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[0])
    return {
        "protein_fmax_observed_proxy": float(f1[best]),
        "protein_fmax_threshold": float(thresholds[starts[best]]),
        "protein_precision_at_fmax_observed_proxy": float(precision[best]),
        "protein_recall_at_fmax_observed": float(recall[best]),
        "protein_coverage_at_fmax": float(activated[best] / len(scores)),
        "micro_ap_observed_proxy": float(np.sum(tp_per_group / total_tp * micro_precision)),
        "micro_auprc_trapezoid_observed_proxy": float(np.sum(
            np.diff(np.r_[0.0, micro_recall]) * (np.r_[1.0, micro_precision[:-1]] + micro_precision) / 2
        )),
        "protein_auprc_trapezoid_observed_proxy": float(np.sum(
            np.diff(np.r_[0.0, recall]) * (np.r_[1.0, precision[:-1]] + precision) / 2
        )),
        "distinct_probability_thresholds": len(starts),
    }


def first_positive_tie_distribution(better_count, tie_count, positive_count):
    """Exact first relevant position under uniform permutation of a tied block."""
    survival = 1.0
    for offset in range(tie_count - positive_count + 1):
        hazard = positive_count / (tie_count - offset)
        yield better_count + offset + 1, survival * hazard
        survival *= 1 - hazard


def ranking_metrics(candidates, positive_ids, scores, truth, positives, top_k, eligible_count):
    pair_rows = []
    protein_rows = []
    per_go_rows = []
    summary = {
        "mrr_first_positive_expected": None, "mean_positive_pair_midrank": None,
        "tied_positive_pair_count": 0, "positive_proteins_with_tied_positive": 0,
        "max_positive_score_tie_count": None,
    }
    k_eff = {k: min(k, len(candidates)) for k in top_k}
    for k, effective in k_eff.items():
        for metric in ("protein_recall", "pair_recall", "go_macro_recall", "protein_hit_rate", "precision",
                       "precision_all_evaluable_rows_proxy"):
            summary[f"{metric}@{k}"] = None
        summary[f"effective_k@{k}"] = effective
        summary[f"random_pair_recall@{k}"] = effective / len(candidates)
    if not len(positive_ids):
        return summary, pair_rows, protein_rows, per_go_rows

    harmonic = np.r_[0.0, np.cumsum(1.0 / np.arange(1, len(candidates) + 1))]
    go_pairs = defaultdict(list)
    for protein, values, mask in zip(positive_ids, scores, truth):
        best_score = values[mask].max()
        best_better = int(np.count_nonzero(values > best_score))
        best_ties = int(np.count_nonzero(values == best_score))
        best_positives = int(np.count_nonzero(values[mask] == best_score))
        first = list(first_positive_tie_distribution(best_better, best_ties, best_positives))
        entry = {
            "protein_id": protein,
            "observed_positives": int(mask.sum()),
            "first_positive_rank_best": first[0][0],
            "first_positive_rank_worst": first[-1][0],
            "reciprocal_first_rank_expected": sum(chance / rank for rank, chance in first),
        }
        for k in top_k:
            entry[f"expected_hits@{k}"] = 0.0
            entry[f"probability_hit@{k}"] = min(1.0, sum(chance for rank, chance in first if rank <= k_eff[k]))
        for column in np.flatnonzero(mask):
            go = candidates[column]
            score = values[column]
            better = int(np.count_nonzero(values > score))
            tied = int(np.count_nonzero(values == score))
            pair = {
                "protein_id": protein, "go_id": go, "score": float(score),
                "rank_min": better + 1, "rank_max": better + tied,
                "midrank": float(better + (tied + 1) / 2),
                "tie_count": tied,
                "reciprocal_rank_expected": float((harmonic[better + tied] - harmonic[better]) / tied),
                "evidence_codes": positives[protein][go].get("evidence_codes", []),
            }
            for k in top_k:
                chance = min(max(k_eff[k] - better, 0), tied) / tied
                pair[f"probability_in_top@{k}"] = chance
                entry[f"expected_hits@{k}"] += chance
            pair_rows.append(pair)
            go_pairs[go].append(pair)
        for k in top_k:
            entry[f"recall@{k}"] = entry[f"expected_hits@{k}"] / entry["observed_positives"]
            entry[f"precision@{k}"] = entry[f"expected_hits@{k}"] / k_eff[k]
        protein_rows.append(entry)

    for go in candidates:
        observed = go_pairs[go]
        if observed:
            result = {
                "go_id": go, "observed_positive_proteins": len(observed),
                "mean_midrank_among_candidates": float(np.mean([row["midrank"] for row in observed])),
            }
            for k in top_k:
                result[f"pair_recall@{k}"] = float(np.mean([row[f"probability_in_top@{k}"] for row in observed]))
            per_go_rows.append(result)
    summary["mrr_first_positive_expected"] = float(np.mean([p["reciprocal_first_rank_expected"] for p in protein_rows]))
    summary["mean_positive_pair_midrank"] = float(np.mean([p["midrank"] for p in pair_rows]))
    tied_pairs = [row for row in pair_rows if row["tie_count"] > 1]
    summary["tied_positive_pair_count"] = len(tied_pairs)
    summary["positive_proteins_with_tied_positive"] = len({row["protein_id"] for row in tied_pairs})
    summary["max_positive_score_tie_count"] = max(row["tie_count"] for row in pair_rows)
    for k, effective in k_eff.items():
        hits = sum(p[f"expected_hits@{k}"] for p in protein_rows)
        summary[f"protein_recall@{k}"] = float(np.mean([p[f"recall@{k}"] for p in protein_rows]))
        summary[f"pair_recall@{k}"] = hits / len(pair_rows)
        summary[f"go_macro_recall@{k}"] = float(np.mean([go[f"pair_recall@{k}"] for go in per_go_rows]))
        summary[f"protein_hit_rate@{k}"] = float(np.mean([p[f"probability_hit@{k}"] for p in protein_rows]))
        summary[f"precision@{k}"] = hits / (len(positive_ids) * effective)
        summary[f"precision_all_evaluable_rows_proxy@{k}"] = hits / (eligible_count * effective)
    return summary, pair_rows, protein_rows, per_go_rows


def write_tsv(path, rows, empty_columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else empty_columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list)
                             else value for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--cohort", default="ind_test_exp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=(1, 3, 5, 10, 20, 50))
    args = parser.parse_args(argv)
    if min(args.top_k) < 1 or len(set(args.top_k)) != len(args.top_k):
        parser.error("--top-k must contain distinct positive integers")
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new evaluation output directory: {args.output_dir}")
    (test, cohort, pred, pred_path, candidates, positives, positive_ids, scores,
     truth, reordered) = load_inputs(args.test_manifest, args.predictions_dir, args.cohort)
    ranking, pairs, proteins, per_go = ranking_metrics(
        candidates, positive_ids, scores, truth, positives, args.top_k, len(cohort["evaluable_protein_ids"]),
    )
    classification = exact_threshold_metrics(scores, truth) if len(positive_ids) else {
        "protein_fmax_observed_proxy": None, "protein_fmax_threshold": None,
        "protein_precision_at_fmax_observed_proxy": None, "protein_recall_at_fmax_observed": None,
        "protein_coverage_at_fmax": None, "micro_ap_observed_proxy": None,
        "micro_auprc_trapezoid_observed_proxy": None,
        "protein_auprc_trapezoid_observed_proxy": None, "distinct_probability_thresholds": None,
    }
    summary = {
        "schema": "latence-unseen-go-evaluation-v1",
        "complete": True,
        "status": "ok" if pairs else "no_observed_positive_metrics_undefined",
        "namespace": test["namespace"],
        "cohort": args.cohort,
        "cohort_role": cohort.get("role"),
        "identity_policy": cohort.get("identity_policy"),
        "protein_rows_reordered": reordered,
        "candidate_go_count": len(candidates),
        "cohort_protein_count": len(cohort["protein_ids"]),
        "evaluable_protein_count": len(cohort["evaluable_protein_ids"]),
        "positive_protein_count": len(positive_ids),
        "evaluable_proteins_without_observed_positive": len(cohort["evaluable_protein_ids"]) - len(positive_ids),
        "positive_go_count": len(per_go),
        "positive_pair_count": len(pairs),
        "checkpoint_epoch": pred.get("checkpoint_epoch"),
        "checkpoint_step": pred.get("checkpoint_step"),
        "bound_input_manifest_sha256": cohort.get("input_manifest_sha256"),
        "test_manifest_sha256": sha256(args.test_manifest),
        "prediction_manifest_sha256": sha256(pred_path),
        "probability_sha256": pred["output_probability_sha256"],
        "ontology_bundle_sha256": test["ontology_bundle_sha256"],
        "strict_pipeline_zero_shot_claimed": False,
        "stage1_exposure_audited": pred.get("stage1_exposure_audited", False),
        "definitions": {
            "scale": "unit interval except scores reported as ranks or counts",
            "scope": "Eligible proteins with observed positives, scored over the complete unseen candidate GO axis.",
            "negative_policy": "Unannotated pairs are unlabeled. Precision and PR summaries use observed positives as a proxy.",
            "threshold_policy": "All distinct pooled scores with tied scores entering together. The Fmax threshold is descriptive.",
            "protein_fmax": "Mean precision over proteins predicted at a threshold and mean recall over proteins with observed positives.",
            "top_k": "Uniform ordering within exactly tied score groups. k is capped at candidate count.",
            "mrr": "Mean exact expected reciprocal rank of the first observed positive in each protein.",
            "go_macro_recall": "Mean per-GO observed-pair recall across GO terms with observed positives.",
            "random_pair_recall": "Expected hit probability under a uniform ranking of all candidate GO terms.",
            "precision_all_evaluable_rows_proxy": "Top-k observed hits divided by effective k times all eligible proteins.",
        },
        "classification": classification,
        "ranking": ranking,
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unseen_eval.", dir=args.output_dir.parent) as temporary:
        staged = Path(temporary) / "output"
        staged.mkdir()
        (staged / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        write_tsv(staged / "positive_pairs.tsv", pairs, ["protein_id", "go_id"])
        write_tsv(staged / "positive_proteins.tsv", proteins, ["protein_id", "observed_positives"])
        write_tsv(staged / "positive_go.tsv", per_go, ["go_id", "observed_positive_proteins"])
        if args.output_dir.exists():
            raise FileExistsError(f"Choose a new evaluation output directory: {args.output_dir}")
        staged.rename(args.output_dir)
    print(f"{args.cohort}: {len(positive_ids)} positive proteins, {len(per_go)} GO, {len(pairs)} pairs")
    print(args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
