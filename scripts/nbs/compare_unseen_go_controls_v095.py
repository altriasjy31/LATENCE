#!/usr/bin/env python3
"""Compare unseen GO inference with historical anchors and protein-row permutation."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np

from eval_unseen_go_v095 import (
    GO_ID,
    PREDICTION_SCHEMA,
    SHA256,
    TEST_SCHEMA,
    checked_ids,
    exact_threshold_metrics,
    load_inputs,
    load_score_matrix,
    prediction_file,
    ranking_metrics,
    read_manifest,
    sha256,
    write_tsv,
)
from unseen_go_row_permutation import protein_row_permutation


EXTENSION_SCHEMA = "latence-v095-anchored-new-go-v1"
PROVENANCE_FIELDS = (
    "checkpoint_sha256", "resolved_config_sha256", "input_manifest_sha256",
    "contract", "prediction_implementation",
)
CLASSIFICATION_METRICS = (
    "protein_fmax_observed_proxy", "micro_ap_observed_proxy",
    "micro_auprc_trapezoid_observed_proxy", "protein_auprc_trapezoid_observed_proxy",
)


def anchor_plan(test, extension_dir):
    """Bind the exact historical rows selected by the fitted ontology extension."""
    manifest_path = extension_dir / "extension_manifest.json"
    extension = read_manifest(manifest_path, EXTENSION_SCHEMA)
    manifest_hash = sha256(manifest_path)
    expected = test.get("sources", {}).get("extension_manifest", {}).get("sha256")
    if manifest_hash != expected:
        raise ValueError("The ontology extension manifest differs from the test manifest source")
    if extension.get("ontology_only") is not True:
        raise ValueError("Expected an ontology-only extension")
    if extension["artifacts"]["ontology_bundle.pt"]["sha256"] != test["ontology_bundle_sha256"]:
        raise ValueError("The anchor plan belongs to a different ontology bundle")
    binding = extension["sources"]["v095_bindings"]
    config_hash = test.get("v095_resolved_config_sha256")
    if config_hash is not None and binding["v095_config"]["sha256"] != config_hash:
        raise ValueError("The anchor plan belongs to a different v095 configuration")
    namespace = test["namespace"]
    if extension["task"] != namespace or extension["namespaces"] != [namespace]:
        raise ValueError("The anchor plan namespace differs from the test manifest")
    go_ids = checked_ids(test["candidate_go_ids"], "candidate GO axis")
    if any(GO_ID.fullmatch(go) is None for go in go_ids):
        raise ValueError("Candidate GO axis must use canonical GO:NNNNNNN identifiers")
    old_rows, new_rows = extension["old_rows"], extension["new_rows"]
    if (type(old_rows) is not int or old_rows < 1 or type(new_rows) is not int
            or new_rows != len(go_ids)):
        raise ValueError("Invalid old or new ontology row count")
    candidate_rows = test["candidate_ontology_rows"]
    if candidate_rows != list(range(old_rows, old_rows + new_rows)):
        raise ValueError("The test candidates do not match the appended ontology rows")

    by_go = {}
    for term in extension["fit"]["per_term"]:
        go = term["go_id"]
        if go in by_go:
            raise ValueError(f"Duplicate anchor record for {go}")
        by_go[go] = term
    if set(by_go) != set(go_ids):
        raise ValueError("The extension does not cover the entire test candidate axis")
    mapping = {}
    for go, row in zip(go_ids, candidate_rows):
        record = by_go[go]
        anchors = record["anchor_rows"]
        if (record["row"] != row or not isinstance(anchors, list) or not anchors
                or any(type(x) is not int or x < 0 or x >= old_rows for x in anchors)
                or len(anchors) != len(set(anchors))):
            raise ValueError(f"Invalid or nonhistorical anchor rows for {go}")
        mapping[go] = tuple(sorted(anchors))
    return extension, manifest_hash, mapping


def original_anchor_ids(extension, mapping, args):
    """Resolve historical canonical IDs through the original registry policy."""
    bindings = extension["sources"]["v095_bindings"]
    paths = []
    for name in ("full_ontology_manifest", "task_registry", "alignment_manifest"):
        binding = bindings[name]
        path = getattr(args, name) or Path(binding["path"])
        if sha256(path) != binding["sha256"]:
            raise ValueError(f"{name} differs from the v095 artifact bound to the extension: {path}")
        paths.append(path)
    root = Path(__file__).resolve().parents[2]
    registry_source = root / "nbs_models/nbs_protein_go/nbs_pg/go_registry_v093.py"
    spec = importlib.util.spec_from_file_location("_unseen_controls_go_registry", registry_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original GO registry implementation: {registry_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = module.GORegistryV093.from_manifests(*paths)
    if registry.num_ontology_rows != extension["old_rows"]:
        raise ValueError("Original registry row count differs from the ontology extension")
    needed = {row for rows in mapping.values() for row in rows}
    by_row = {}
    for go, row in sorted(registry._row.items()):
        if row not in needed or row in by_row:
            continue
        try:
            if registry.canonical_go_id(go) == go and registry.resolve_one(go) == row:
                by_row[row] = go
        except ValueError:
            continue
    if missing := sorted(needed - set(by_row)):
        raise ValueError(f"No active canonical IDs for historical ontology rows: {missing[:10]}")
    return [by_row[row] for row in sorted(needed)]


def prepare_anchors(args):
    test = read_manifest(args.test_manifest, TEST_SCHEMA)
    extension, _, mapping = anchor_plan(test, args.extension_dir)
    ids = original_anchor_ids(extension, mapping, args)
    content = "".join(f"{go}\n" for go in ids)
    args.output_go_ids.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output_go_ids.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except FileExistsError:
        if args.output_go_ids.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing anchor GO list differs: {args.output_go_ids}") from None
    print(f"{len(ids)} historical anchor rows for {len(mapping)} candidate GO terms")
    print(args.output_go_ids)


def load_anchor_scores(folder, original_pred, cohort_ids, mapping):
    path = folder / "prediction_manifest.json"
    pred = read_manifest(path, PREDICTION_SCHEMA)
    identity = pred["identity"]
    reference = original_pred["identity"]
    if identity["ontology_bundle_sha256"] is not None or pred["ontology_extension"] is not None:
        raise ValueError("Historical anchor predictions must use the original ontology without --ontology-file")
    for key in PROVENANCE_FIELDS:
        if key not in identity or key not in reference or identity[key] != reference[key]:
            raise ValueError(f"Anchor predictions differ in {key}")
        if key.endswith("_sha256") and (not isinstance(identity[key], str)
                                         or SHA256.fullmatch(identity[key]) is None):
            raise ValueError(f"Invalid prediction provenance hash: {key}")
    for key in ("checkpoint_epoch", "checkpoint_step"):
        if type(pred.get(key)) is not int or pred[key] < 1 or pred[key] != original_pred.get(key):
            raise ValueError(f"Anchor predictions differ in {key}")

    needed = {row for rows in mapping.values() for row in rows}
    rows = identity["target_ontology_rows"]
    ids = checked_ids(identity["requested_go_ids"], "anchor prediction GO axis")
    if (not isinstance(rows, list) or any(type(row) is not int for row in rows)
            or len(rows) != len(needed) or set(rows) != needed
            or len(ids) != len(rows) or pred["num_target_GO"] != len(rows)
            or any(GO_ID.fullmatch(go) is None for go in ids)):
        raise ValueError("Anchor prediction columns must cover each required historical row once")
    target_path = prediction_file(folder, pred, "target_go_file")
    with target_path.open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream, delimiter="\t"))
    if len(table) != len(rows) or len(pred["targets"]) != len(rows):
        raise ValueError("Anchor target metadata length differs from the prediction axis")
    for col, (row, go, target, record) in enumerate(zip(rows, ids, pred["targets"], table)):
        fields = {"output_column": col, "requested_go_id": go, "ontology_row": row,
                  "canonical_go_id": go, "readout": "single_shared_GO_scorer"}
        if (any(target.get(key) != value for key, value in fields.items())
                or any(record.get(key) != str(value) for key, value in fields.items())):
            raise ValueError(f"Anchor prediction column {col} does not identify the expected row and GO")
    protein_ids, probabilities = load_score_matrix(folder, pred, cohort_ids)
    return pred, path, protein_ids, probabilities, dict(zip(rows, ids)), {
        row: col for col, row in enumerate(rows)
    }


def transferred_scores(probabilities, protein_ids, eval_protein_ids, mapping, col_for_row):
    """Project nearest-anchor scores into the fixed candidate GO columns."""
    protein_rows = {protein: i for i, protein in enumerate(protein_ids)}
    aligned = probabilities[[protein_rows[protein] for protein in eval_protein_ids]]
    output = np.empty((len(eval_protein_ids), len(mapping)), dtype=aligned.dtype)
    for col, anchors in enumerate(mapping.values()):
        output[:, col] = aligned[:, [col_for_row[row] for row in anchors]].max(axis=1)
    return output


def pair_comparison(expanded_pairs, anchor_pairs, top_k):
    by_pair = {(row["protein_id"], row["go_id"]): row for row in anchor_pairs}
    records = []
    for row in expanded_pairs:
        anchor = by_pair[row["protein_id"], row["go_id"]]
        record = {"protein_id": row["protein_id"], "go_id": row["go_id"],
                  "evidence_codes": row["evidence_codes"]}
        for field in ("score", "rank_min", "rank_max", "midrank", "tie_count"):
            record[f"expanded_{field}"] = row[field]
            record[f"anchor_{field}"] = anchor[field]
        record["midrank_improvement_over_anchor"] = anchor["midrank"] - row["midrank"]
        for k in top_k:
            record[f"expanded_top@{k}_credit"] = row[f"probability_in_top@{k}"]
            record[f"anchor_top@{k}_credit"] = anchor[f"probability_in_top@{k}"]
        records.append(record)
    return records


def summarize(expanded, anchor, permutation, top_k):
    ranking_keys = ["mrr_first_positive_expected", "mean_positive_pair_midrank"]
    for k in top_k:
        ranking_keys.extend(f"{metric}@{k}" for metric in (
            "pair_recall", "protein_recall", "go_macro_recall", "protein_hit_rate", "precision",
        ))
    metric_groups = {"ranking": ranking_keys, "classification": CLASSIFICATION_METRICS}
    result = []
    for group, keys in metric_groups.items():
        for key in keys:
            full = expanded[group].get(key)
            baseline = anchor[group].get(key) if anchor else None
            null = permutation.get(key, {})
            result.append({
                "metric": key,
                "expanded": full,
                "anchor_transfer": baseline,
                "expanded_minus_anchor": full - baseline if full is not None and baseline is not None else None,
                "permutation_mean": null.get("null_mean"),
                "expanded_minus_permutation_mean": full - null["null_mean"] if null else None,
                "permutation_q025": null.get("null_q025"),
                "permutation_q975": null.get("null_q975"),
                "permutation_p_ge": null.get("empirical_p_ge"),
                "permutation_p_le": null.get("empirical_p_le"),
            })
    return result


def compare(args):
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"Choose a new control output directory: {args.output_dir}")
    (test, cohort, pred, pred_path, candidates, positives, positive_ids, scores, truth,
     reordered, pred_ids, probability) = load_inputs(
        args.test_manifest, args.predictions_dir, args.cohort, with_full_scores=True,
    )
    eligible_count = len(cohort["evaluable_protein_ids"])

    def metrics(matrix):
        ranking, pairs, _, _ = ranking_metrics(
            candidates, positive_ids, matrix, truth, positives, args.top_k, eligible_count,
        )
        classification = exact_threshold_metrics(matrix, truth) if positive_ids else {}
        return {"ranking": ranking, "classification": classification}, pairs

    expanded, pairs = metrics(scores)
    row_by_id = {protein: row for row, protein in enumerate(pred_ids)}
    positive_rows = np.asarray([row_by_id[protein] for protein in positive_ids], dtype=np.int64)
    permutation, draws = protein_row_permutation(
        probability, truth, positive_rows, args.top_k,
        n_permutations=args.n_permutations, seed=args.seed,
    )
    for key, values in permutation.items():
        if abs(expanded["ranking"][key] - values["observed"]) > 1e-12:
            raise RuntimeError(f"Permutation implementation differs from the reference metric: {key}")

    result = {
        "schema": "latence-unseen-go-controls-v095-v1",
        "complete": True,
        "status": "ok" if positive_ids else "no_observed_positive_metrics_undefined",
        "namespace": test["namespace"],
        "cohort": args.cohort,
        "candidate_go_count": len(candidates),
        "cohort_protein_count": len(pred_ids),
        "evaluable_protein_count": eligible_count,
        "positive_protein_count": len(positive_ids),
        "positive_pair_count": int(truth.sum()),
        "positive_go_count": int(truth.any(axis=0).sum()),
        "checkpoint_epoch": pred["checkpoint_epoch"],
        "checkpoint_step": pred["checkpoint_step"],
        "checkpoint_sha256": pred["identity"]["checkpoint_sha256"],
        "test_manifest_sha256": sha256(args.test_manifest),
        "expanded_prediction_manifest_sha256": sha256(pred_path),
        "expanded_probability_sha256": pred["output_probability_sha256"],
        "expanded_ontology_bundle_sha256": test["ontology_bundle_sha256"],
        "expanded_protein_rows_reordered": reordered,
        "expanded_exporter_sha256": pred["identity"].get("exporter_sha256"),
        "expanded_inference_batch_size": pred["identity"].get("inference_batch_size"),
        "stage1_exposure_audited": pred.get("stage1_exposure_audited", False),
        "strict_pipeline_zero_shot_claimed": False,
        "definitions": {
            "anchor_transfer": "Maximum original-ontology score over the fitted nearest historical rows for each candidate GO.",
            "permutation": "Jointly permute full protein score vectors across the complete prediction cohort with labels and GO columns fixed.",
            "permutation_p_ge": "One-sided Monte Carlo tail (1 + count(null >= observed)) / (1 + number of draws). Per-metric diagnostics, unadjusted.",
            "permutation_p_le": "Lower-tail counterpart for mean positive pair midrank, where lower is better.",
            "permutation_interval": "2.5th and 97.5th quantiles of the permutation null, not a confidence interval for model performance.",
            "labels": "All candidates are retained. Ranking and threshold summaries use proteins with observed positive annotations. Unannotated pairs are unlabeled.",
        },
        "expanded": expanded,
        "anchor_transfer": None,
        "protein_row_permutation": {
            "seed": args.seed, "requested_draw_count": args.n_permutations,
            "actual_draw_count": len(next(iter(draws.values()))) if draws else 0,
            "donor_protein_count": len(pred_ids), "metrics": permutation,
        },
    }
    baseline = None
    mapping_records, pair_records = [], []
    if args.anchor_predictions_dir is not None:
        _, extension_hash, mapping = anchor_plan(test, args.extension_dir)
        anchor_pred, anchor_path, anchor_ids, anchor_probs, go_for_row, col_for_row = load_anchor_scores(
            args.anchor_predictions_dir, pred, cohort["protein_ids"], mapping,
        )
        transfer = transferred_scores(anchor_probs, anchor_ids, pred_ids, mapping, col_for_row)
        baseline, anchor_pairs = metrics(transfer[positive_rows])
        pair_records = pair_comparison(pairs, anchor_pairs, args.top_k)
        mapping_records = [
            {"go_id": go, "historical_ontology_rows": list(rows),
             "historical_go_ids": [go_for_row[row] for row in rows]}
            for go, rows in mapping.items()
        ]
        hit_changes = {}
        for k in args.top_k:
            differences = [row[f"expanded_top@{k}_credit"] - row[f"anchor_top@{k}_credit"] for row in pair_records]
            hit_changes[str(k)] = {
                "increased_hit_credit": sum(max(value, 0.0) for value in differences),
                "decreased_hit_credit": sum(max(-value, 0.0) for value in differences),
                "net_hit_credit": sum(differences),
            } if differences else None
        result["anchor_transfer"] = {
            **baseline, "historical_go_count": len(go_for_row),
            "extension_manifest_sha256": extension_hash,
            "prediction_manifest_sha256": sha256(anchor_path),
            "probability_sha256": anchor_pred["output_probability_sha256"],
            "protein_rows_reordered": anchor_ids != cohort["protein_ids"],
            "exporter_sha256": anchor_pred["identity"].get("exporter_sha256"),
            "inference_batch_size": anchor_pred["identity"].get("inference_batch_size"),
            "paired_hit_credit_changes": hit_changes,
        }

    summary = summarize(expanded, baseline, permutation, args.top_k)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unseen_controls.", dir=args.output_dir.parent) as temp:
        staged = Path(temp) / "output"
        staged.mkdir()
        (staged / "comparison.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        write_tsv(staged / "summary.tsv", summary, ["metric"])
        with (staged / "permutation_draws.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["draw", *draws])
            writer.writerows((i, *row) for i, row in enumerate(zip(*draws.values()), 1))
        if baseline is not None:
            write_tsv(staged / "anchor_mapping.tsv", mapping_records, ["go_id"])
            write_tsv(staged / "positive_pairs_comparison.tsv", pair_records, ["protein_id", "go_id"])
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(f"Choose a new control output directory: {args.output_dir}")
        staged.rename(args.output_dir)
    print(args.output_dir / "comparison.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare-anchors", help="Write the original-ontology GO query list")
    prepare.add_argument("--test-manifest", type=Path, required=True)
    prepare.add_argument("--extension-dir", type=Path, required=True)
    prepare.add_argument("--output-go-ids", type=Path, required=True)
    for key in ("full-ontology-manifest", "task-registry", "alignment-manifest"):
        prepare.add_argument(f"--{key}", type=Path, help="Override a relocated file with the same SHA256")
    run = sub.add_parser("compare", help="Evaluate the controls on the fixed unseen GO test axes")
    run.add_argument("--test-manifest", type=Path, required=True)
    run.add_argument("--predictions-dir", type=Path, required=True)
    run.add_argument("--anchor-predictions-dir", type=Path)
    run.add_argument("--extension-dir", type=Path)
    run.add_argument("--cohort", default="ind_test_exp")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--top-k", type=int, nargs="+", default=(1, 3, 5, 10, 20, 50))
    run.add_argument("--n-permutations", type=int, default=1000)
    run.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.action == "prepare-anchors":
        prepare_anchors(args)
        return
    if (args.anchor_predictions_dir is None) != (args.extension_dir is None):
        run.error("--anchor-predictions-dir and --extension-dir must be supplied together")
    if min(args.top_k) < 1 or len(set(args.top_k)) != len(args.top_k):
        run.error("--top-k must contain distinct positive integers")
    if args.n_permutations < 1 or args.seed < 0:
        run.error("--n-permutations must be positive and --seed must be nonnegative")
    compare(args)


if __name__ == "__main__":
    main()
