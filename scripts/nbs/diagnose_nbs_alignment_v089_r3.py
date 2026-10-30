#!/usr/bin/env python3
"""Read-only: how much of a trained v089 per-GO head lives in the GO-box space?

v089 scores with independent rows ``logit = h . W_g + b_g``. This diagnostic
measures how well its W_g/b_g can be linearly reconstructed
from the supplied GO representation. This is a parameter-predictability probe,
not a proof of semantic alignment. For a fixed checkpoint:

1. GO side  X_g: (a) the checkpoint's own ontology encoding enc(box_g) [H];
   (b) raw geometry [center, log|offset|]; controls: (c) training-frequency
   features only, (d) enc rows permuted across canonical GO groups within
   core-frequency bins, and
   (e) enc + frequency.
2. Cross-fitted ridge  [W_g, b_g] ~ A X_g + c  with K folds over canonical GO
   groups (aliases never split), stratified by training-core GO frequency.
   Every GO's row is predicted without fitting or selecting lambda on that GO
   group. Lambda is selected by an inner GO-group cross-validation using only
   the outer training GO groups (classifier rows only, never test labels).
3. Test proteins: the checkpoint's own hidden h (classifier input, same fixed
   deterministic inference path) scored by the original head and by each
   box-regressed head. Fmax / micro AP overall and per frequency bin, plus
   per-GO logit correlation with the original head.

These are closed-set head-reconstruction diagnostics: h and the ontology
encoder were trained with the GO labels being evaluated. No result establishes
zero-shot performance or acceptance; nothing is selected on test proteins.

R3 scores every regressed head with the original device, float32 precision and
inference batch size. Historical export agreement includes actual Fmax/AP
changes, probability-error statistics and threshold flips. A user-requested
larger probability tolerance is recorded explicitly, not called exact equality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nbs_models/nbs_protein_go")]

SCHEMA = "nbs-head-box-alignment-v3"
FREQUENCY_BINS = ((0, 0), (1, 9), (10, 99), (100, 999), (1000, None))
SUPPORTED_RUNNERS = ("0.8.9.1",)
DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


KNN = None


def knn_helpers():
    """Lazy import keeps the numerical regression helpers usable without torch."""
    global KNN
    if KNN is None:
        candidates = (Path(__file__).resolve().parent / "diagnose_nbs_knn_v090.py",
                      ROOT / "scripts/nbs/diagnose_nbs_knn_v090.py")
        source = next((path for path in candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError("diagnose_nbs_knn_v090.py helper is required next to this script or in scripts/nbs")
        KNN = _load(source, "_nbs_alignment_knn_helpers")
    return KNN


def log(message):
    print(f"[alignment {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- folds / ridge

def frequency_bin(counts, bins=FREQUENCY_BINS):
    out = np.full(len(counts), -1, np.int64)
    for index, (low, high) in enumerate(bins):
        out[(counts >= low) & (True if high is None else counts <= high)] = index
    return out


def canonical_folds(task_to_ontology, column_counts, folds, seed):
    """Whole canonical groups, frequency stratification, balanced sparse bins."""
    mapping = np.asarray(task_to_ontology, np.int64)
    counts = np.asarray(column_counts, np.int64)
    if mapping.ndim != 1 or mapping.shape != counts.shape or not len(mapping):
        raise ValueError("GO mapping and counts must be matching nonempty vectors")
    groups, inverse = np.unique(mapping, return_inverse=True)
    if not isinstance(folds, (int, np.integer)) or not 3 <= folds <= len(groups):
        raise ValueError("nested GO cross-validation needs 3 <= folds <= canonical GO groups")
    if np.any(counts < 0):
        raise ValueError("GO frequencies must be nonnegative")
    group_count = np.zeros(len(groups), np.int64)
    np.maximum.at(group_count, inverse, counts)
    group_bin = frequency_bin(group_count)
    rng = np.random.default_rng(seed)
    group_fold = np.empty(len(groups), np.int64)
    offset = 0
    for value in np.unique(group_bin):
        members = rng.permutation(np.flatnonzero(group_bin == value))
        group_fold[members] = (np.arange(len(members)) + offset) % folds
        offset += len(members)
    return group_fold[inverse]


def _validate_regression(features, targets, fold, alphas):
    x, y, fold = np.asarray(features, np.float64), np.asarray(targets, np.float64), np.asarray(fold)
    alphas = tuple(dict.fromkeys(float(value) for value in alphas))
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or fold.shape != (len(x),):
        raise ValueError("ridge needs matching feature/target matrices and fold vector")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs must be finite")
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("ridge alphas must be finite positive values")
    if len(np.unique(fold)) < 2:
        raise ValueError("ridge cross-validation needs at least two nonempty folds")
    return x, y, fold, alphas


def _ridge_predict_path(train_x, train_y, query_x, alphas):
    """Yield one alpha at a time; avoid holding G x target_dim x alphas arrays."""
    mx, my = train_x.mean(0), train_y.mean(0)
    xc = train_x - mx
    values, vectors = np.linalg.eigh(xc.T @ xc)
    values = np.clip(values, 0., None)
    projected = vectors.T @ (xc.T @ (train_y - my))
    scale = max(float(values.mean()), 1e-12)
    query = (query_x - mx) @ vectors
    for alpha in alphas:
        yield alpha, query @ (projected / (values + alpha * scale)[:, None]) + my


def ridge_crossfit(features, targets, fold, alphas):
    """Unselected out-of-fold paths, primarily for numerical diagnostics/tests."""
    x, y, fold, alphas = _validate_regression(features, targets, fold, alphas)
    out = {alpha: np.empty_like(y) for alpha in alphas}
    for k in np.unique(fold):
        train, test = fold != k, fold == k
        for alpha, predicted in _ridge_predict_path(x[train], y[train], x[test], alphas):
            out[alpha][test] = predicted
    return out


def nested_ridge_crossfit(features, targets, fold, alphas):
    """Select alpha inside each outer training split; no held-out-GO tuning."""
    x, y, fold, alphas = _validate_regression(features, targets, fold, alphas)
    if len(np.unique(fold)) < 3 or y.shape[1] < 2:
        raise ValueError("nested ridge needs >=3 outer folds and weight+bias targets")
    predicted = np.empty_like(y)
    selections = []
    for k in np.unique(fold):
        train, test = fold != k, fold == k
        tx, ty, inner_fold = x[train], y[train], fold[train]
        denominator = float(((ty[:, :-1] - ty[:, :-1].mean(0)) ** 2).sum())
        errors = {alpha: 0. for alpha in alphas}
        for inner in np.unique(inner_fold):
            inner_train, inner_test = inner_fold != inner, inner_fold == inner
            for alpha, value in _ridge_predict_path(tx[inner_train], ty[inner_train], tx[inner_test], alphas):
                errors[alpha] += float(((ty[inner_test, :-1] - value[:, :-1]) ** 2).sum())
        grid = [{"alpha": a, "inner_weight_r2": None if denominator <= 0 else 1. - errors[a] / denominator}
                for a in alphas]
        # If weights are constant, select a fixed predeclared alpha rather than
        # inventing an informative R2. Ties preserve the command-line order.
        best = min(alphas, key=lambda a: errors[a]) if denominator > 0 else alphas[0]
        predicted[test] = next(_ridge_predict_path(tx, ty, x[test], (best,)))[1]
        selections.append({"outer_fold": int(k), "selected_alpha": best, "inner_alpha_grid": grid})
    return predicted, selections


def r2(truth, predicted, rows=None):
    truth, predicted = np.asarray(truth, np.float64), np.asarray(predicted, np.float64)
    if rows is not None:
        truth, predicted = truth[rows], predicted[rows]
    if not len(truth):
        return None
    total = float(((truth - truth.mean(0)) ** 2).sum())
    return None if total <= 0 else 1. - float(((truth - predicted) ** 2).sum()) / total


def row_cosine(truth, predicted):
    t, p = np.asarray(truth, np.float64), np.asarray(predicted, np.float64)
    norm = np.linalg.norm(t, axis=1) * np.linalg.norm(p, axis=1)
    return np.divide((t * p).sum(1), norm, out=np.zeros(len(t)), where=norm > 0)


def fit_report(name, features, weight, bias, fold, bins, alphas):
    """Nested GO-group cross-fitting of both weights and per-GO biases."""
    targets = np.concatenate((weight, bias[:, None]), axis=1)
    predicted, selections = nested_ridge_crossfit(features, targets, fold, alphas)
    cosine = row_cosine(weight, predicted[:, :-1])
    per_bin = []
    for index, (low, high) in enumerate(FREQUENCY_BINS):
        rows = np.flatnonzero(bins == index)
        per_bin.append({"train_core_count_min": low, "train_core_count_max": high, "columns": int(len(rows)),
                        "weight_r2_within_bin": r2(weight, predicted[:, :-1], rows),
                        "bias_r2_within_bin": r2(bias, predicted[:, -1], rows),
                        "mean_row_cosine": float(cosine[rows].mean()) if len(rows) else None})
    return {"input": name, "input_dim": int(np.asarray(features).shape[1]),
            "outer_fold_alpha_selection": selections,
            "selection": "inner GO-group weight-R2; each outer test group excluded from fitting AND selection",
            "weight_r2": r2(weight, predicted[:, :-1]), "bias_r2": r2(bias, predicted[:, -1]),
            "mean_row_cosine": float(cosine.mean()), "per_frequency_bin": per_bin}, predicted


# ---------------------------------------------------------------- test scoring

def sigmoid(z):
    return (1. / (1. + np.exp(-np.clip(z, -60., 60.)))).astype(np.float32)


def head_sanity(hidden, weight, bias, logits, exported=None, *, compare_atol=2e-5,
                native_reconstruction_max_abs_logit=None):
    """Validate reconstructed original head; absence of an export is explicit."""
    reconstructed = np.asarray(hidden, np.float32) @ np.asarray(weight, np.float32).T + np.asarray(bias, np.float32)
    logits = np.asarray(logits, np.float32)
    if reconstructed.shape != logits.shape or not np.isfinite(logits).all():
        raise ValueError("invalid original logits or classifier reconstruction shape")
    native = native_reconstruction_max_abs_logit
    result = {"numpy_head_max_abs_logit_vs_model": float(np.abs(reconstructed - logits).max()),
              "head_reconstruction_max_abs_logit": float(np.abs(reconstructed - logits).max()) if native is None else float(native),
              "head_reconstruction_backend": "numpy_cpu" if native is None else "same_device_same_batch_torch_linear",
              "head_reconstruction_passed": (bool(np.allclose(reconstructed, logits, atol=1e-4, rtol=1e-5))
                                             if native is None else bool(np.isfinite(native) and native <= 1e-5)),
              "export_comparison": "not_requested", "export_probability_atol": float(compare_atol)}
    if exported is not None:
        exported = np.asarray(exported, np.float32)
        if exported.shape != logits.shape or not np.isfinite(exported).all():
            raise ValueError("exported probability shape/finite check failed")
        if np.any((exported < 0) | (exported > 1)):
            raise ValueError("exported probabilities must lie in [0,1]")
        difference = float(np.abs(sigmoid(logits) - exported).max())
        result.update(max_abs_prob_vs_exported=difference,
                      export_comparison="passed" if difference <= compare_atol else "failed")
    return result


def probability_comparison(labels, logits, exported, *, compare_atol, metric_atol, counts=None):
    """Report numerical drift AND actual metric changes; units are explicit.

    A relaxed max-probability tolerance alone never establishes metric agreement.
    This is an engineering check for a diagnostic, not a statistical test.
    """
    helpers = knn_helpers()
    current = sigmoid(np.asarray(logits, np.float32))
    exported = np.asarray(exported, np.float32)
    if (current.shape != exported.shape or not current.size or
            not np.isfinite(current).all() or not np.isfinite(exported).all()):
        raise ValueError("probability comparison needs matching nonempty finite matrices")
    if np.any((exported < 0) | (exported > 1)):
        raise ValueError("exported probabilities must lie in [0,1]")
    if any(not np.isfinite(x) or x < 0 for x in (compare_atol, metric_atol)):
        raise ValueError("comparison tolerances must be finite and nonnegative")
    difference = np.abs(current - exported)
    quantiles = np.quantile(difference, [.5, .95, .99, .999])
    index = np.unravel_index(int(difference.argmax()), difference.shape)
    metrics = {}
    for name, values in (("recomputed", current), ("exported", exported)):
        metrics[name] = {**helpers.protein_fmax(labels, values),
                         "micro_ap": helpers.grouped_ap(labels, values)}
    delta = {key: abs(metrics["recomputed"][key] - metrics["exported"][key])
             for key in ("fmax", "micro_ap")}
    flips = {}
    for name in ("recomputed", "exported"):
        threshold = metrics[name]["threshold"]
        changed = (current > threshold) != (exported > threshold)
        flips[name + "_fmax_threshold"] = {
            "threshold": threshold, "changed_pairs": int(changed.sum()),
            "proteins_affected": int(changed.any(axis=1).sum())}
    maximum = float(difference[index])
    result = {
        "mean_abs_probability_difference": float(difference.mean(dtype=np.float64)),
        "abs_probability_difference_quantiles": dict(zip(("p50", "p95", "p99", "p99_9"), map(float, quantiles))),
        "max_abs_probability_difference": maximum,
        "max_difference_location": {"protein_row": int(index[0]), "task_go_column": int(index[1]),
                                    "recomputed_probability": float(current[index]),
                                    "exported_probability": float(exported[index])},
        "pairs_exceeding": {str(t): int(np.count_nonzero(difference > t)) for t in (2e-5, 1e-4, 1e-3)},
        "probability_atol": compare_atol, "probability_passed": maximum <= compare_atol,
        "strict_r2_probability_atol": 2e-5, "strict_r2_probability_passed": maximum <= 2e-5,
        "metrics_percent_units": metrics, "absolute_metric_delta_percentage_points": delta,
        "metric_atol_percentage_points": metric_atol,
        "metrics_passed": all(value <= metric_atol for value in delta.values()),
        "threshold_prediction_flips": flips,
        "scope": "diagnostic engineering tolerances; not bitwise or statistical equivalence"}
    if counts is not None:
        methods = {name: (values, metrics[name]["threshold"])
                   for name, values in (("recomputed", current), ("exported", exported))}
        strata = helpers.frequency_strata(labels, methods, counts)
        for row in strata:
            first = row["methods"]["recomputed"]["micro_ap"]
            second = row["methods"]["exported"]["micro_ap"]
            row["absolute_ap_delta_percentage_points"] = None if first is None or second is None else abs(first - second)
        result["frequency_strata_drift"] = strata
        result["frequency_strata_scope"] = "reported separately; metrics_passed gates overall Fmax and micro AP only"
    return result


def make_head_scorer(device, batch_size):
    """Use the original classifier backend, precision and inference batch size."""
    import torch
    if batch_size < 1:
        raise ValueError("head scoring batch size must be positive")

    def score(hidden, weight, bias):
        hidden = np.asarray(hidden, np.float32)
        weight = np.asarray(weight, np.float32)
        bias = np.asarray(bias, np.float32)
        if (hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]
                or bias.shape != (len(weight),) or not len(hidden)):
            raise ValueError("invalid fixed-hidden classifier dimensions")
        parts = []
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            w = torch.from_numpy(np.ascontiguousarray(weight)).to(device)
            b = torch.from_numpy(np.ascontiguousarray(bias)).to(device)
            for start in range(0, len(hidden), batch_size):
                h = torch.from_numpy(np.ascontiguousarray(hidden[start:start + batch_size])).to(device)
                parts.append(torch.nn.functional.linear(h, w, b).float().cpu().numpy())
        return np.concatenate(parts, axis=0)
    return score


def column_correlation(left, right, bins):
    a = left - left.mean(0)
    b = right - right.mean(0)
    denominator = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    corr = np.divide((a * b).sum(0), denominator, out=np.full(left.shape[1], np.nan), where=denominator > 0)
    result = []
    for index, (low, high) in enumerate(FREQUENCY_BINS):
        values = corr[bins == index]
        values = values[np.isfinite(values)]
        result.append({"train_core_count_min": low, "train_core_count_max": high,
                       "mean_logit_pearson": float(values.mean()) if len(values) else None})
    return result


def score_head(name, hidden, weight, bias, labels, counts, reference_logits=None, bins=None, *,
               actual_logits=None, head_scorer=None):
    if actual_logits is not None:
        logits = np.asarray(actual_logits, np.float32)
    elif head_scorer is not None:
        logits = head_scorer(hidden, weight, bias)
    else:
        logits = hidden @ np.asarray(weight, np.float32).T + np.asarray(bias, np.float32)[None, :]
    probability = sigmoid(logits)
    helpers = knn_helpers()
    result = {"head": name, **helpers.protein_fmax(labels, probability),
              "micro_ap": helpers.grouped_ap(labels, probability)}
    strata = helpers.frequency_strata(labels, {name: (probability, result["threshold"])}, counts)
    result["per_frequency_bin"] = [{key: value for key, value in row.items() if key != "methods"}
                                   | row["methods"][name] for row in strata]
    if reference_logits is not None:
        result["logit_correlation_with_original"] = column_correlation(reference_logits, logits, bins)
    return result, logits


# ---------------------------------------------------------------- model / data

def ontology_contract(ontology):
    tensors = {k: v for k, v in ontology.items() if k != "edges"}
    tensors.update({"edge_" + k: v for k, v in ontology["edges"].items()})
    return {key: hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
            for key, value in tensors.items()}


def load_model_and_data(config, checkpoint, device, sampler):
    import torch
    from nbs_pg.full_task_model_v088 import FullTaskGraphModelV088, FullTaskModelConfigV088
    from nbs_pg.local_loader_v089 import build_latence_nbs_stores
    if sampler == "fast":
        from nbs_pg.full_task_sampler_fast import FullTaskDataV088Fast as Data
    else:
        from nbs_pg.full_task_data_v088 import FullTaskDataV088 as Data
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("runner_version") not in SUPPORTED_RUNNERS:
        raise ValueError(f"alignment diagnostic expects a v089 checkpoint, got runner {saved.get('runner_version')!r}")
    if saved.get("model_config", {}).get("prediction_mode") != "direct":
        raise ValueError("checkpoint has no independent per-GO classifier rows")
    from scripts.nbs import train_nbs_full_task_v089 as runner
    variant = saved.get("variant", saved["config"]["full_task"].get("variant", "fixed"))
    trained = saved.get("training_implementation", {})
    if any(trained.get(key) != value for key, value in runner.prediction_implementation(variant).items()):
        raise ValueError("original prediction implementation differs from this checkpoint")
    if config.get("task") != saved["config"].get("task"):
        raise ValueError("config task differs from checkpoint")
    if config["full_task"].get("model", {}) != saved["config"]["full_task"].get("model", {}):
        raise ValueError("config model differs from checkpoint")
    if config["full_task"].get("sampler", {}) != saved["config"]["full_task"].get("sampler", {}):
        raise ValueError("config sampler differs from the checkpoint; use the run's resolved_config.json")
    data = Data(config, stores=build_latence_nbs_stores(config))
    data.ensure_prepared()
    ontology = data.ontology("cpu")
    contract = {"data": data.data_contract(), "ontology": ontology_contract(ontology)}
    if saved["contract"] != contract:
        raise ValueError("checkpoint data/ontology contract differs from this config")
    if saved.get("training_baseline_identity") != runner.training_baseline_identity(data):
        raise ValueError("checkpoint backbone inputs/clipping differ from this config")
    model = FullTaskGraphModelV088(data.feature_dim, ontology, FullTaskModelConfigV088(**saved["model_config"]))
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    info = {"checkpoint": str(Path(checkpoint).resolve()), "runner_version": saved["runner_version"],
            "epoch": saved.get("epoch"), "step": saved.get("step"), "model_config": saved["model_config"]}
    return model, data, ontology, info


def test_hidden(model, data, input_dir, batch_size, device):
    """Classifier inputs h and logits on the unchanged deterministic inference path."""
    import torch
    rows = len(np.load(Path(input_dir) / "ind_test_repr.f16.npy", mmap_mode="r"))
    captured = []
    reconstruction_error = []

    def capture(module, inputs, output):
        captured.append(inputs[0].detach().float().cpu())
        # Same backend/precision/batch shape as the original classifier. A CPU
        # numpy product need not reproduce CUDA's configured TF32 numerics.
        rebuilt = torch.nn.functional.linear(inputs[0], module.weight, module.bias)
        reconstruction_error.append(float((rebuilt - output).abs().max()))

    handle = model.classifier.register_forward_hook(capture)
    logits = []
    data.set_sampling_context(step=0, rank=0, training=False)
    try:
        with torch.no_grad():
            go = model.encode_go()
            for start in range(0, rows, batch_size):
                batch = data.inference_batch(input_dir, np.arange(start, min(start + batch_size, rows)), device)
                logits.append(model(batch, go_encoding=go).float().cpu())
                if start // batch_size % 20 == 0:
                    log(f"inference {min(start + batch_size, rows)}/{rows}")
    finally:
        handle.remove()
    if not captured or len(captured) != len(logits):
        raise ValueError("classifier hook did not capture exactly one hidden batch per inference batch")
    return torch.cat(captured).numpy(), torch.cat(logits).numpy(), max(reconstruction_error)


# ---------------------------------------------------------------- report

def go_side_inputs(encoding, ontology, task_to_ontology, core_counts, weak_counts, seed):
    center = ontology["center"].cpu().numpy()[task_to_ontology]
    offset = ontology["offset"].cpu().numpy()[task_to_ontology]
    raw = np.concatenate((center, np.log(np.clip(np.abs(offset), 1e-8, None))), axis=1)
    frequency = np.stack((np.log1p(core_counts), np.log1p(weak_counts)), axis=1).astype(np.float64)
    groups, first, inverse = np.unique(task_to_ontology, return_index=True, return_inverse=True)
    group_count = np.zeros(len(groups), np.int64)
    np.maximum.at(group_count, inverse, core_counts)
    group_bin = frequency_bin(group_count)
    permutation = np.arange(len(groups))
    rng = np.random.default_rng(seed + 1)
    for value in np.unique(group_bin):
        members = np.flatnonzero(group_bin == value)
        permutation[members] = rng.permutation(members)
    permuted = encoding[first[permutation][inverse]]
    return {"ontology_encoding": encoding, "raw_box_geometry": raw, "frequency_only": frequency,
            "permuted_encoding_control": permuted,
            "ontology_encoding_plus_frequency": np.concatenate((encoding, frequency), axis=1)}


def analyse(weight, bias, inputs, hidden, labels, core_counts, task_to_ontology, *, folds=5, seed=20261024,
            alphas=DEFAULT_ALPHAS, scored=("ontology_encoding", "raw_box_geometry", "frequency_only",
                                           "ontology_encoding_plus_frequency", "permuted_encoding_control"),
            original_logits=None, head_scorer=None):
    bins = frequency_bin(core_counts)
    fold = canonical_folds(task_to_ontology, core_counts, folds, seed)
    report = {"regression": [], "test_heads": []}
    original, reference = score_head("original_head", hidden, weight, bias, labels, core_counts,
                                     actual_logits=original_logits)
    report["test_heads"].append(original)
    for name, features in inputs.items():
        fitted, predicted = fit_report(name, features, weight, bias, fold, bins, alphas)
        report["regression"].append(fitted)
        log(f"{name}: weight R2 {fitted['weight_r2']}, bias R2 {fitted['bias_r2']}, "
            f"cos {fitted['mean_row_cosine']:.4f} (nested alpha selection)")
        if name in scored:
            head, _ = score_head("box_regressed:" + name, hidden, predicted[:, :-1], predicted[:, -1],
                                 labels, core_counts, reference, bins, head_scorer=head_scorer)
            report["test_heads"].append(head)
            log(f"  test Fmax {head['fmax']:.3f}, micro AP {head['micro_ap']:.3f}")
    report["folds"] = {"count": folds, "seed": seed, "unit": "canonical GO group (aliases kept together)",
                       "stratified_by": "training-core GO frequency bin", "alpha_selection": "nested; no outer group targets used"}
    return report


def summary_markdown(report):
    lines = ["# v089 head ↔ GO-box alignment", "",
             "Closed-set classifier reconstruction only. The encoders saw these GO labels; no zero-shot or acceptance claim.", "",
             f"Checkpoint: `{report['model']['checkpoint']}` (epoch {report['model']['epoch']})", "",
             "## Cross-fitted ridge [W_g, b_g] ~ A·X_g (held-out GO groups)", "",
             "| GO-side input | dim | alpha | W R² | b R² | mean cos(Ŵ,W) |", "|---|---|---|---|---|---|"]
    def fmt(value):
        return "n/a" if value is None else f"{value:.4f}"
    for row in report["regression"]:
        alphas = ",".join(f"{a['selected_alpha']:g}" for a in row["outer_fold_alpha_selection"])
        lines.append(f"| {row['input']} | {row['input_dim']} | {alphas} | {fmt(row['weight_r2'])} | "
                     f"{fmt(row['bias_r2'])} | {fmt(row['mean_row_cosine'])} |")
    lines += ["", "Within-bin W R² / mean cosine by training-core GO count:", "",
              "| input | " + " | ".join(f"{b['train_core_count_min']}-{b['train_core_count_max'] if b['train_core_count_max'] is not None else 'inf'}"
                                       for b in report["regression"][0]["per_frequency_bin"]) + " |",
              "|---|" + "---|" * len(report["regression"][0]["per_frequency_bin"])]
    for row in report["regression"]:
        cells = []
        for b in row["per_frequency_bin"]:
            value = b["weight_r2_within_bin"]
            cells.append("" if value is None else f"{value:.3f} / {b['mean_row_cosine']:.3f}")
        lines.append(f"| {row['input']} | " + " | ".join(cells) + " |")
    lines += ["", "## Test proteins: original head vs box-regressed heads", "",
              "| head | Fmax | thr | P | R | micro AP |", "|---|---|---|---|---|---|"]
    for head in report["test_heads"]:
        lines.append(f"| {head['head']} | {head['fmax']:.3f} | {head['threshold']:.3f} | {head['precision']:.2f} | "
                     f"{head['recall']:.2f} | {head['micro_ap']:.3f} |")
    lines += ["", "Per-bin micro AP (and mean per-GO logit Pearson with the original head):", "",
              "| head | " + " | ".join(f"{b['train_core_count_min']}-{b['train_core_count_max'] if b['train_core_count_max'] is not None else 'inf'}"
                                      for b in report["test_heads"][0]["per_frequency_bin"]) + " |",
              "|---|" + "---|" * len(report["test_heads"][0]["per_frequency_bin"])]
    for head in report["test_heads"]:
        corr = head.get("logit_correlation_with_original")
        cells = []
        for index, b in enumerate(head["per_frequency_bin"]):
            ap = "" if b.get("micro_ap") is None else f"{b['micro_ap']:.2f}"
            if corr and corr[index]["mean_logit_pearson"] is not None:
                ap += f" (r={corr[index]['mean_logit_pearson']:.2f})"
            cells.append(ap)
        lines.append(f"| {head['head']} | " + " | ".join(cells) + " |")
    if "sanity" in report:
        lines += ["", f"Sanity: {json.dumps(report['sanity'])}"]
    if "export_comparison_details" in report:
        lines += ["", "## Agreement with historical export", "",
                  json.dumps(report["export_comparison_details"], indent=2)]
    return "\n".join(lines) + "\n"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="resolved_config.json of the v089 run")
    p.add_argument("--checkpoint", type=Path, required=True, help="fixed nbs_epoch<N>.pt")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--metadata-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task", default="bp", choices=("bp", "mf", "cc"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sampler", choices=("original", "fast"), default="original",
                   help="inference block sampler; 'fast' is output-identical (see bench_nbs_sampler_fast.py)")
    p.add_argument("--compare-prob", type=Path, help="existing exported probabilities of this checkpoint")
    p.add_argument("--compare-atol", type=float, default=2e-5,
                   help="fail before regression if exported probabilities differ by more than this absolute tolerance")
    p.add_argument("--compare-metric-atol", type=float, default=.02,
                   help="maximum absolute Fmax AND micro AP difference in percentage points (diagnostic only)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    import torch
    for name in ("config", "checkpoint", "input_dir", "metadata_file", "output_dir", "compare_prob"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    helpers = knn_helpers()
    if any(not np.isfinite(x) or x < 0 for x in (args.compare_atol, args.compare_metric_atol)):
        raise ValueError("comparison tolerances must be finite and nonnegative")
    alphas = tuple(float(a) for a in args.alphas.split(",") if a.strip())
    if args.folds < 3 or not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("use folds >= 3 and finite positive alphas")
    out = helpers.prepare_output_dir(args.output_dir)
    os.chdir(ROOT)  # data contracts resolve manifest paths relative to the project root
    config = json.loads(args.config.read_text())
    if config.get("task") != args.task:
        raise ValueError("--task differs from config task")
    # Match the original v089 runner's export/training numerical setting.
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    model, data, ontology, info = load_model_and_data(config, args.checkpoint, device, args.sampler)
    test = helpers.load_test_inputs(args.input_dir, args.metadata_file, args.task,
                                data._signature["representation_manifest_sha256"])
    log("computing test hidden states on the fixed inference path")
    batch_size = config["full_task"].get("eval_batch", 80)
    hidden, logits, native_reconstruction = test_hidden(model, data, args.input_dir.resolve(), batch_size, device)
    with torch.no_grad():
        encoding = model.encode_go().float().cpu().numpy()
    weight = model.classifier.weight.detach().float().cpu().numpy()
    bias = model.classifier.bias.detach().float().cpu().numpy()
    head_scorer = make_head_scorer(device, batch_size)
    replay_logits = head_scorer(hidden, weight, bias)
    replay_error = float(np.abs(replay_logits - logits).max())
    replay_passed = bool(np.isfinite(replay_error) and replay_error <= 1e-5)
    del replay_logits
    mapping = model.task_to_ontology.cpu().numpy().astype(np.int64)
    sources = helpers.TrainingSources(data)
    core_counts = sources.core_column_counts(data.core_ids)
    weak_counts = np.bincount(np.asarray(data.stores.pseudo_messages.go_idx, np.int64), minlength=data.num_task_go)
    sanity = head_sanity(hidden, weight, bias, logits,
                         np.load(args.compare_prob, mmap_mode="r") if args.compare_prob is not None else None,
                         compare_atol=args.compare_atol,
                         native_reconstruction_max_abs_logit=native_reconstruction)
    sanity.update(fixed_hidden_head_replay_max_abs_logit=replay_error,
                  fixed_hidden_head_replay_passed=replay_passed)
    comparison = None
    if args.compare_prob is not None:
        comparison = probability_comparison(test["labels"], logits, np.load(args.compare_prob, mmap_mode="r"),
                                             compare_atol=args.compare_atol, metric_atol=args.compare_metric_atol,
                                             counts=core_counts)
        (out / "export_comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        log("export comparison: " + json.dumps(comparison, ensure_ascii=False))
    if (not sanity["head_reconstruction_passed"] or not replay_passed or
            sanity["export_comparison"] == "failed" or (comparison is not None and not comparison["metrics_passed"])):
        failure = dict(sanity)
        if comparison is not None:
            failure["export_comparison_details"] = comparison
        (out / "sanity_failed.json").write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise ValueError("sanity failed; see printed diagnostics and sanity_failed.json (probability, metric and head checks are separate)")
    log(f"sanity {sanity}")
    inputs = go_side_inputs(encoding, ontology, mapping, core_counts, weak_counts, args.seed)
    report = analyse(weight, bias, inputs, hidden, test["labels"], core_counts, mapping, folds=args.folds,
                     seed=args.seed, alphas=alphas, original_logits=logits, head_scorer=head_scorer)
    report.update(schema=SCHEMA, model=info, sanity=sanity, sampler=args.sampler,
                  scope="closed-set classifier reconstruction; encoders trained with evaluated GO; not zero-shot or acceptance",
                  provenance={"matmul_precision": "high (matches v089 runner)",
                              "regressed_head_scoring": "torch F.linear on original device, same precision and inference batch size; unchanged head replay verified",
                              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "config": str(args.config.resolve()), "input_dir": str(args.input_dir.resolve()),
                              "metadata_file": str(args.metadata_file.resolve())})
    if comparison is not None:
        report["export_comparison_details"] = comparison
    (out / "alignment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    log(f"done: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
