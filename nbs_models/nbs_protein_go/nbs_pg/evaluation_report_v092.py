"""Exact AP and core-frequency diagnostics over an existing v092 prediction.

This additive report never runs the model, changes probabilities, propagates
GO scores, or selects a checkpoint. Its AP is non-interpolated micro average
precision, distinct from the legacy evaluator's trapezoidal PR-AUC.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FREQUENCY_BINS = ((0, 0), (1, 5), (1, 9), (10, 99), (100, 999), (1000, None))
SCHEMA = "nbs-v092-extended-evaluation-v1"


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_helpers(root=ROOT):
    path = Path(root) / "scripts/nbs/diagnose_nbs_knn_v090.py"
    if not path.is_file():
        raise FileNotFoundError("extended evaluation requires " + str(path))
    name = "_nbs_v092_eval_helpers_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def core_counts(data):
    """Count unique task-column memberships in training-core rows only.

    Gold CSR rows use GLOBAL protein IDs. Holdout IDs are rejected rather than
    silently counted; weak pseudo-labels are never used to define these bins.
    Alias columns remain separate immutable classifier columns.
    """
    ids = np.asarray(data.core_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("training core IDs must be a nonnegative integer vector")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("training core IDs must be unique")
    holdout = np.asarray(getattr(data, "validation_ids", []), dtype=np.int64)
    if np.intersect1d(ids, holdout).size:
        raise ValueError("training core frequency must exclude holdout proteins")
    num_go = data.num_task_go
    if isinstance(num_go, (bool, np.bool_)) or not isinstance(num_go, (int, np.integer)) or num_go < 1:
        raise ValueError("task GO count must be positive")
    store = data.stores.gold_messages
    ptr, columns = np.asarray(store.indptr), store.go_idx
    if ptr.ndim != 1 or ptr.dtype.kind not in "iu" or not len(ptr):
        raise ValueError("gold CSR indptr must be an integer vector")
    if len(ids) and np.any(ids >= len(ptr) - 1):
        raise ValueError("training core IDs are outside global gold CSR rows")
    counts = np.zeros(int(num_go), dtype=np.int64)
    for protein in ids:
        start, stop = int(ptr[int(protein)]), int(ptr[int(protein) + 1])
        if not 0 <= start <= stop <= len(columns):
            raise ValueError("invalid global gold CSR row boundaries")
        go = np.asarray(columns[start:stop])
        if go.dtype.kind not in "iu" or np.any(go < 0) or np.any(go >= num_go):
            raise ValueError("gold memberships are outside immutable task GO columns")
        counts[np.unique(go)] += 1
    return counts


def _validate_matrix(labels, values):
    y, scores = np.asarray(labels), np.asarray(values)
    if y.ndim != 2 or min(y.shape) < 1 or scores.shape != y.shape:
        raise ValueError("labels/probabilities need matching nonempty protein-by-GO axes")
    if y.dtype.kind not in "biuf" or not np.isfinite(y).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError("test labels must be finite binary memberships")
    if scores.dtype.kind not in "biuf" or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("predicted probabilities must be finite and within [0,1]")
    return y.astype(bool, copy=False), scores


def summarize_probabilities(labels, probs, counts, helpers):
    """Summarize one matrix or a method-name -> matrix mapping, without torch."""
    methods = dict(probs) if isinstance(probs, Mapping) else {"NBS_final": probs}
    if not methods:
        raise ValueError("at least one method is required")
    y, _ = _validate_matrix(labels, next(iter(methods.values())))
    frequencies = np.asarray(counts)
    if (frequencies.shape != (y.shape[1],) or frequencies.dtype.kind not in "iu"
            or np.any(frequencies < 0)):
        raise ValueError("training-core frequencies must match immutable GO columns")
    results = {}
    for name, values in methods.items():
        if not isinstance(name, str) or not name:
            raise ValueError("method names must be nonempty strings")
        _, matrix = _validate_matrix(y, values)
        results[name] = {**helpers.protein_fmax(y, matrix), "micro_ap": helpers.grouped_ap(y, matrix)}
    strata = helpers.frequency_strata(y, {name: (values, results[name]["threshold"])
                                          for name, values in methods.items()}, frequencies,
                                      bins=FREQUENCY_BINS)
    report = {
        "num_proteins": int(y.shape[0]), "num_GO_columns": int(y.shape[1]),
        "test_positive_pairs": int(y.sum()), "metrics": results, "frequency_strata": strata,
        "frequency_definition": "unique raw task-column memberships among data.core_ids; holdout excluded; no weak counts",
        "frequency_bins_overlap": True,
        "frequency_strata_note": "1-5 is nested in 1-9; these two rows overlap and must not be summed",
        "score_unit": "percent; differences are percentage points; thresholds use probability units",
        "protein_fmax_threshold_step": 0.001, "strict_threshold_comparison": ">",
        "AP_definition": "exact non-interpolated micro average precision; equal scores grouped",
        "PR_AUC_computed_here": False,
        "all_go_columns_retained": True, "ontology_probability_propagation": False,
        "zero_shot_performance_claimed": False,
        "checkpoint_selection": "fixed checkpoint diagnostic; no automatic checkpoint selection",
    }
    if "NBS_final" in results and "backbone_base" in results:
        report["delta_NBS_minus_backbone"] = {
            key: (results["NBS_final"][key] - results["backbone_base"][key]
                  if results["NBS_final"][key] is not None and results["backbone_base"][key] is not None else None)
            for key in ("fmax", "micro_ap")}
    return report


def _write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".v092_report_", delete=False,
                                     encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_extended_report(args, config, data):
    """Call after successful original evaluate(args,config,data,device).

    Probability and source-checkpoint identities are checked against the
    original prediction manifest. The test helper checks feature provenance,
    GO count, unique IDs and metadata ordering. No additional GNN forward runs.
    """
    import torch
    if getattr(args, "stage", None) != "evaluate":
        raise ValueError("extended report is for stage=evaluate only")
    checkpoint = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step, epoch = saved.get("step"), saved.get("epoch")
    version = saved.get("runner_version", saved.get("version"))
    del saved
    if type(step) is not int or step < 0:
        raise ValueError("checkpoint has an invalid optimizer step")
    ablation = getattr(args, "ablation", "full")
    if ablation != "full":
        raise ValueError("v092 extended report requires the complete prediction path")
    folder = Path(args.work_dir).resolve() / f"eval_step{step}" / ablation
    probability_path = folder / "nbs_ind_test_prob.f32.npy"
    prediction_manifest = folder / "nbs_full_task_prediction_manifest.json"
    provenance = json.loads(prediction_manifest.read_text(encoding="utf-8"))
    probability_sha, checkpoint_sha = _sha256(probability_path), _sha256(checkpoint)
    if (provenance.get("checkpoint_sha256") != checkpoint_sha or
            provenance.get("output_probability_sha256") != probability_sha or provenance.get("step") != step):
        raise ValueError("prediction/checkpoint identity differs from the completed evaluation")
    input_dir = Path(args.input_dir).resolve()
    input_sha = _sha256(input_dir / "ind_test_input_manifest.json")
    if provenance.get("input_manifest_sha256") != input_sha:
        raise ValueError("prediction inputs differ from the extended report request")
    representation_sha = data._signature.get("representation_manifest_sha256")
    if not isinstance(representation_sha, str) or not representation_sha:
        raise ValueError("training feature provenance is missing")
    helpers = _load_helpers()
    test = helpers.load_test_inputs(input_dir, Path(args.metadata_file).resolve(), config["task"], representation_sha)
    probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    if probabilities.shape[1:] != (data.num_task_go,):
        raise ValueError("prediction GO axis differs from the training task")
    counts = core_counts(data)
    report = summarize_probabilities(test["labels"],
        {"NBS_final": probabilities, "backbone_base": test["backbone"]}, counts, helpers)
    report.update(
        schema=SCHEMA, complete=True, task=config["task"], checkpoint_step=step, checkpoint_epoch=epoch,
        runner_version=version, checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        probability_path=str(probability_path), probability_sha256=probability_sha,
        input_manifest_sha256=input_sha, prediction_manifest_sha256=_sha256(prediction_manifest),
        num_training_core_proteins=len(data.core_ids),
        training_core_ids_sha256=_array_sha256(np.asarray(data.core_ids, dtype="<i8")),
        core_frequency_sha256=_array_sha256(counts.astype("<i8", copy=False)),
        metric_helper_sha256=_sha256(ROOT / "scripts/nbs/diagnose_nbs_knn_v090.py"),
        scope="existing full-column independent-test probabilities; exact AP and frequency diagnostics; no model forward",
    )
    output = folder / "metrics_v092" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    print(f"[v092 extended evaluation] {output}", flush=True)
    return output
