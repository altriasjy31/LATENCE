#!/usr/bin/env python3
"""CPU diagnostics for existing v088 epoch-5/10 probabilities, without inference.

Run from the LATENCE repository; --project-root defaults to the current directory.
Canonical Fmax is recomputed with the existing v085 kernel and must reproduce
both saved comparisons. The extra diagnostics are descriptive, not new metrics
for acceptance, forecasts, output fusion, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import resource
import sys
import time

import numpy as np

BASELINE = 57.72314936447243
IDENTITY_KEYS = (
    "input_manifest_sha256", "backbone_probability_sha256",
    "protein_ids_file_sha256", "input_go_registry_sha256", "variant", "branch",
    "encoder_variant", "model_config", "sampler_config", "model_architecture_version",
    "data_architecture_version", "prediction_mode", "output_fusion", "source_flags",
    "prediction_implementation",
)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path, expected, cache):
    path = Path(path).resolve()
    if not expected or len(expected) != 64:
        raise ValueError(f"Missing recorded SHA256 for {path}")
    if path not in cache:
        cache[path] = sha256(path)
    if cache[path] != expected:
        raise ValueError(f"Recorded SHA256 does not match {path}")
    return path


def point(curve, index):
    return {"threshold": float(curve["thresholds"][index]),
            "protein_f1": float(curve["f1"][index] * 100),
            "protein_precision": float(curve["precision"][index] * 100),
            "protein_recall": float(curve["recall"][index] * 100)}


def make_curve(kernel, labels, scores):
    labels, scores = kernel._validate(labels, scores)
    thresholds = kernel._thresholds(.001)
    f1, precision, recall, eligible = kernel._protein_curve(labels, scores, thresholds)
    return dict(thresholds=thresholds, f1=f1, precision=precision, recall=recall,
                eligible=eligible, best=int(np.argmax(f1)))


def constrained_point(curve, *, recall_floor=None, precision_floor=None):
    if recall_floor is not None:
        feasible = np.flatnonzero(curve["recall"] * 100 >= recall_floor - 1e-10)
        objective = curve["precision"]
    else:
        feasible = np.flatnonzero(curve["precision"] * 100 >= precision_floor - 1e-10)
        objective = curve["recall"]
    if not len(feasible):
        return None
    return point(curve, int(feasible[np.argmax(objective[feasible])]))


def group_end_rank(scores, positive_indices):
    """Rank at the END of each tie group; invariant to monotone recalibration."""
    sorted_scores = np.sort(scores)
    return len(scores) - np.searchsorted(sorted_scores, scores[positive_indices], side="left")


def error_diagnostics(labels, arrays, curves):
    thresholds = {k: curves[k]["thresholds"][curves[k]["best"]] for k in arrays}
    transitions = dict(gained_true_positive_pairs=0, lost_true_positive_pairs=0,
                       removed_false_positive_pairs=0, added_false_positive_pairs=0,
                       proteins_with_more_true_positives=0, proteins_with_fewer_true_positives=0,
                       proteins_with_same_true_positives=0)
    overlap = dict(G10_missed_B_correct_positive_pairs=0,
                   G10_correct_B_missed_positive_pairs=0,
                   both_correct_positive_pairs=0, both_missed_positive_pairs=0)
    shifts = {"all_G10_false_negatives": [], "G10_false_negatives_detected_by_B": []}
    for i, gold in enumerate(labels):
        gold = gold != 0
        pred = {k: np.asarray(v[i], dtype=np.float64) > thresholds[k] for k, v in arrays.items()}
        a, b, base = pred["G5"], pred["G10"], pred["B"]
        transitions["gained_true_positive_pairs"] += int((gold & ~a & b).sum())
        transitions["lost_true_positive_pairs"] += int((gold & a & ~b).sum())
        transitions["removed_false_positive_pairs"] += int((~gold & a & ~b).sum())
        transitions["added_false_positive_pairs"] += int((~gold & ~a & b).sum())
        if gold.any():
            delta = int((gold & b).sum()) - int((gold & a).sum())
            key = ("proteins_with_more_true_positives" if delta > 0 else
                   "proteins_with_fewer_true_positives" if delta < 0 else
                   "proteins_with_same_true_positives")
            transitions[key] += 1
        overlap["G10_missed_B_correct_positive_pairs"] += int((gold & ~b & base).sum())
        overlap["G10_correct_B_missed_positive_pairs"] += int((gold & b & ~base).sum())
        overlap["both_correct_positive_pairs"] += int((gold & b & base).sum())
        overlap["both_missed_positive_pairs"] += int((gold & ~b & ~base).sum())
        positive_indices = np.flatnonzero(gold)
        if not len(positive_indices):
            continue
        change = (group_end_rank(arrays["G5"][i], positive_indices)
                  - group_end_rank(arrays["G10"][i], positive_indices))
        missing = ~b[positive_indices]
        shifts["all_G10_false_negatives"].extend(change[missing].tolist())
        shifts["G10_false_negatives_detected_by_B"].extend(change[missing & base[positive_indices]].tolist())
    rank_summary = {}
    for name, values in shifts.items():
        v = np.asarray(values, dtype=np.int64)
        rank_summary[name] = {
            "positive_pairs": len(values), "rank_moved_up": int((v > 0).sum()),
            "rank_moved_down": int((v < 0).sum()), "rank_unchanged": int((v == 0).sum()),
            "median_rank_change_positive_means_up": float(np.median(v)) if len(v) else None,
        }
    return dict(
        thresholds={k: float(v) for k, v in thresholds.items()},
        epoch5_to10_decision_changes=transitions, baseline_comparison=overlap,
        epoch5_to10_rank_change_of_remaining_missed_gold=rank_summary,
        interpretation=("Decision counts use each model's own canonical Fmax threshold. "
                        "Pair counts are descriptive and do not replace protein Fmax. "
                        "Rank changes use full GO columns and complete tie groups. "
                        "Baseline overlap is comparison only, never a fused prediction. "
                        "Neither nearby scores nor improving ranks prove future recovery."))


def self_test(kernel):
    # Compare the original efficient kernel with explicit strict-threshold counting,
    # including empty gold, empty predictions, float16 ties, and earliest F ties.
    y = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
    p = np.array([[.5, .5, 1., 0.], [0., .5, .5, 0.], [1., 0., 0., 0.]], dtype=np.float16)
    curve = make_curve(kernel, y, p)
    expected = []
    for t in curve["thresholds"]:
        pred = p.astype(np.float64) > t
        positives = y.sum(1)
        counts = pred.sum(1)
        tp = (pred & y).sum(1)
        active = (positives > 0) & (counts > 0)
        prec = float((tp[active] / counts[active]).mean()) if active.any() else 0.
        eligible = positives > 0
        rec = float((tp[eligible] / positives[eligible]).mean())
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.
        expected.append((f1, prec, rec))
    np.testing.assert_allclose(np.stack([curve[k] for k in ("f1", "precision", "recall")], 1), expected,
                               atol=1e-14, rtol=0)
    assert curve["best"] == int(np.argmax(np.asarray(expected)[:, 0]))
    q = np.array([.1, .4, .4, .9])
    np.testing.assert_array_equal(group_end_rank(q, np.arange(4)), [4, 3, 3, 1])
    np.testing.assert_array_equal(group_end_rank(q, np.arange(4)), group_end_rank(q ** 2, np.arange(4)))
    label = np.array([[1, 0, 1, 0]], dtype=bool)
    arrays = {"G5": np.array([[.9, .8, .3, .1]]), "G10": np.array([[.9, .2, .8, .1]]),
              "B": np.array([[.9, .1, .8, .1]])}
    curves = {k: make_curve(kernel, label, v) for k, v in arrays.items()}
    result = error_diagnostics(label, arrays, curves)
    assert result["epoch5_to10_decision_changes"]["removed_false_positive_pairs"] == 1
    assert result["epoch5_to10_decision_changes"]["lost_true_positive_pairs"] == 0
    assert constrained_point(curves["G10"], recall_floor=101.) is None
    try:
        make_curve(kernel, label, np.array([[np.nan, 0, 0, 0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite probabilities were accepted")
    print("[self-test] strict protein Fmax, tie handling, rank invariance, transitions, validation: passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--backbone-prob", type=Path)
    parser.add_argument("--protein-ids", type=Path)
    parser.add_argument("--go-registry", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    kernel = module_at("_growth_v085_metric_kernel", root / "nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py")
    if args.self_test:
        self_test(kernel)
        return
    if args.work_dir is None:
        parser.error("--work-dir is required unless --self-test is used")
    started = time.perf_counter()
    work = args.work_dir.resolve()
    output = args.output or work / "analysis_epoch10_growth.json"
    if output.exists():
        raise FileExistsError(f"Output already exists; choose another --output: {output}")
    comparisons = {name: json.loads((work / f"eval_step{step}/full/metrics/nbs_w2s_comparison.json").read_text())
                   for name, step in (("G5", 480), ("G10", 960))}
    c5, c10 = comparisons["G5"], comparisons["G10"]
    for c, step in ((c5, 480), (c10, 960)):
        if c["task"] != "bp" or (c["num_samples"], c["num_classes"]) != (1800, 21312):
            raise ValueError("Expected unchanged full BP task: 1800 proteins x 21312 GO")
        prov = c["prediction_provenance"]
        if prov["step"] != step or prov["ablation"] != "full" or prov["output_fusion"] != "none":
            raise ValueError("Expected standalone full epoch-5/10 predictions")
        if prov.get("uses_dense_backbone_logits_in_forward") is not False:
            raise ValueError("Unexpected dense baseline logits in model forward")
        if c["standardized_metric_contract"]["implementation_sha256"] != kernel.metrics_implementation_sha256():
            raise ValueError("Local metric implementation differs from original evaluation")
    if c5["standardized_metric_contract"] != c10["standardized_metric_contract"]:
        raise ValueError("Evaluation contracts differ")
    if c5["metadata_provenance"] != c10["metadata_provenance"]:
        raise ValueError("Metadata identities differ")
    for key in IDENTITY_KEYS:
        if key not in c5["prediction_provenance"] or c5["prediction_provenance"][key] != c10["prediction_provenance"][key]:
            raise ValueError(f"Missing/different input or model identity: {key}")
    prov = c10["prediction_provenance"]
    cache = {}
    metadata = verify_file(args.metadata_file or c10["metadata_provenance"]["path"],
                           c10["metadata_provenance"]["sha256"], cache)
    protein_file = verify_file(args.protein_ids or prov["protein_ids_path"], prov["protein_ids_file_sha256"], cache)
    verify_file(args.go_registry or prov["input_go_registry_path"], prov["input_go_registry_sha256"], cache)
    legacy = module_at("_growth_existing_label_loader", root / "scripts/nbs/eval_nbs_ind_test_predictions.py")
    with metadata.open("rb") as handle:
        payload = pickle.load(handle)
    task = legacy._task_dict(payload, "ind_test", "bp")
    ids = [str(x) for x in task["proteins"]]
    annotation_key = next((k for k in ("prop_annotations", "prop_annotation", "annotations", "labels") if k in task), None)
    if annotation_key is None:
        raise ValueError("Independent-test gold annotations are absent")
    labels = legacy._dense_labels(task[annotation_key], 21312, len(ids))
    del task, payload
    if legacy._hash_ids(ids) != c10["metadata_provenance"]["protein_ids_sha256"]:
        raise ValueError("Independent-test metadata protein order differs")
    source_ids = [s.strip() for s in protein_file.read_text().splitlines() if s.strip()]
    arrays = {}
    for name, c in comparisons.items():
        step = c["prediction_provenance"]["step"]
        path = verify_file(work / f"eval_step{step}/full/nbs_ind_test_prob.f32.npy",
                           c["prediction_provenance"]["probability_sha256"], cache)
        arrays[name] = legacy._align_rows(np.load(path, mmap_mode="r", allow_pickle=False),
                                          prediction_ids=source_ids, metadata_ids=ids)
    baseline_path = verify_file(args.backbone_prob or prov["backbone_probability_path"],
                                prov["backbone_probability_sha256"], cache)
    arrays["B"] = legacy._align_rows(np.load(baseline_path, mmap_mode="r", allow_pickle=False),
                                     prediction_ids=source_ids, metadata_ids=ids)
    curves = {}
    canonical = {}
    for name, scores in arrays.items():
        curves[name] = make_curve(kernel, labels, scores)
        canonical[name] = point(curves[name], curves[name]["best"])
        method = "backbone_base" if name == "B" else "NBS_final"
        records = [c["methods"][method]["standardized"] for c in comparisons.values()] if name == "B" else [comparisons[name]["methods"][method]["standardized"]]
        for recorded in records:
            checks = {"protein_f1": "standard_protein_fmax", "threshold": "standard_protein_fmax_threshold",
                      "protein_precision": "standard_protein_precision_at_fmax", "protein_recall": "standard_protein_recall_at_fmax"}
            for actual, expected in checks.items():
                if abs(canonical[name][actual] - recorded[expected]) > 1e-8:
                    raise ValueError(f"{name} recomputation differs from original {expected}")
            if int(labels.sum()) != recorded["standard_num_positive_labels"]:
                raise ValueError("Positive label population differs")
        print(f"[verified] {name}: Fmax={canonical[name]['protein_f1']:.10f}", flush=True)
    if abs(canonical["B"]["protein_f1"] - BASELINE) > 1e-8:
        raise ValueError("Baseline differs from the pinned 57.72314936447243")
    recall_floors = sorted(set([canonical["B"]["protein_recall"], canonical["G10"]["protein_recall"], 52.]))
    matched_recall = [{"recall_floor": floor, "methods": {
        name: constrained_point(curve, recall_floor=floor) for name, curve in curves.items()}}
        for floor in recall_floors]
    matched_precision = {name: constrained_point(curve, precision_floor=canonical["G10"]["protein_precision"])
                         for name, curve in curves.items()}
    diagnostics = error_diagnostics(labels, arrays, curves)
    report = dict(
        schema="nbs-epoch10-growth-diagnostics-v1", state="measured_from_verified_saved_probabilities",
        purpose="Descriptive evidence about remaining errors; not a forecast or checkpoint/hyperparameter selection",
        evaluation=dict(proteins=1800, GO_columns=21312, positive_pairs=int(labels.sum()),
                        strict_threshold=True, threshold_step=.001, ontology_probability_propagation=False,
                        metric_implementation_sha256=kernel.metrics_implementation_sha256()),
        canonical_Fmax=canonical,
        current_G10_minus_B=canonical["G10"]["protein_f1"] - BASELINE,
        baseline_exceeded=canonical["G10"]["protein_f1"] > BASELINE,
        maximum_precision_with_recall_at_least_floor=matched_recall,
        maximum_recall_with_precision_at_least_G10_best=matched_precision,
        constrained_curve_note="All points are actual original 0.001-grid points; achieved recall/precision is reported, with no interpolation.",
        diagnostics=diagnostics, verified_files={str(k): v for k, v in cache.items()},
        runtime=dict(seconds=time.perf_counter() - started, process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                     inference_runs=0, GPU_training_runs=0),
        limitations=["No later checkpoint is observed, so no future Fmax or crossing probability is inferred.",
                     "Conditioning rank analysis on remaining epoch-10 misses is descriptive, not evidence of causality.",
                     "Existing Fmax evaluation and the fixed training schedule remain unchanged."])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"[done] {output}; descriptive diagnostics only")


if __name__ == "__main__":
    main()
