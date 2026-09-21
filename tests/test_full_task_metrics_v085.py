"""Independent checks of the supplementary metric contract and legacy artifact."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, auc, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py"
SPEC = importlib.util.spec_from_file_location("v085_metrics", PATH)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def brute_protein(y, p, step):
    eligible = y.sum(1) > 0
    y, p = y[eligible], p[eligible].astype(np.float64)
    best = (0., 0., 0., 0.)
    if not len(y):
        return best
    for threshold in np.arange(round(1 / step) + 1, dtype=float) / round(1 / step):
        prediction = p > threshold
        tp = (prediction & y.astype(bool)).sum(1)
        predicted = prediction.sum(1)
        precision = np.mean(tp[predicted > 0] / predicted[predicted > 0]) if np.any(predicted > 0) else 0.
        recall = np.mean(tp / y.sum(1))
        f = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        if f > best[0]:
            best = (f, threshold, precision, recall)
    return best


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_curves_match_sklearn_with_ties_across_chunks(dtype):
    rng = np.random.default_rng(713)
    y = (rng.random((11, 17)) < .17).astype(np.int8)
    p = (rng.integers(0, 11, y.shape) / 10).astype(dtype)
    result = metrics._exact_micro(y, p, rank_chunk=7)
    precision, recall, _ = precision_recall_curve(y.ravel(), p.ravel())
    assert result["standard_micro_ap"] == pytest.approx(100 * average_precision_score(y.ravel(), p.ravel()), abs=1e-10)
    assert result["standard_micro_pr_auc"] == pytest.approx(100 * auc(recall, precision), abs=1e-10)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=precision + recall > 0)
    assert result["standard_micro_fmax_exact"] == pytest.approx(100 * f1.max())
    chosen = p > result["standard_micro_fmax_exact_threshold"]
    tp = np.logical_and(chosen, y).sum()
    assert 200 * tp / (chosen.sum() + y.sum()) == pytest.approx(result["standard_micro_fmax_exact"])


@pytest.mark.parametrize("step", [.001, .01, .02])
def test_protein_fmax_matches_independent_brute_force(step):
    rng = np.random.default_rng(92)
    y = (rng.random((8, 31)) < .16).astype(np.int8)
    y[2] = 0
    p = rng.random(y.shape).astype(np.float32)
    p[0, :4] = [0, .25, .5, 1]
    result = metrics.compute_standard_metrics(y, p, step)
    expected = brute_protein(y, p, step)
    assert result["standard_protein_fmax"] == pytest.approx(100 * expected[0])
    assert result["standard_protein_fmax_threshold"] == expected[1]
    assert result["standard_protein_precision_at_fmax"] == pytest.approx(100 * expected[2])
    assert result["standard_protein_recall_at_fmax"] == pytest.approx(100 * expected[3])
    coarse = brute_protein(y, p, .01)
    assert result["standard_protein_fmax_grid_0p01"] == pytest.approx(100 * coarse[0])
    assert result["standard_protein_fmax_grid_0p01_threshold"] == coarse[1]


def test_empty_gold_excluded_for_protein_but_retained_for_micro():
    y = np.array([[1, 0], [0, 0]], dtype=np.int8)
    p = np.array([[.8, .1], [.9, .9]])
    result = metrics.compute_standard_metrics(y, p)
    assert result["standard_protein_fmax"] == 100
    assert result["standard_num_protein_fmax_eligible"] == 1
    assert result["standard_num_empty_gold_proteins"] == 1
    assert result["standard_micro_ap"] == pytest.approx(100 / 3)


def test_recall_counts_eligible_proteins_with_no_predictions():
    y = np.array([[1, 0], [1, 0]])
    p = np.array([[.9, .1], [0., 0.]])
    result = metrics.compute_standard_metrics(y, p)
    assert result["standard_protein_precision_at_fmax"] == 100
    assert result["standard_protein_recall_at_fmax"] == 50
    assert result["standard_protein_fmax"] == pytest.approx(200 / 3)


def test_strict_greater_than_excludes_zero_and_equal_threshold_scores():
    y = np.array([[1, 0, 1, 0]])
    p = np.array([[1., .5, 0., .25]])
    thresholds = np.array([0., .25, .5, 1.])
    f, precision, recall, eligible = metrics._protein_curve(y, p, thresholds)
    assert eligible == 1
    np.testing.assert_allclose(precision, [1 / 3, 1 / 2, 1, 0])
    np.testing.assert_allclose(recall, [.5, .5, .5, 0])
    np.testing.assert_allclose(f, [.4, .5, 2 / 3, 0])


def test_exact_micro_can_include_zero_scores_without_splitting_ties():
    y = np.array([[1, 1, 0]])
    p = np.zeros((1, 3), dtype=np.float32)
    result = metrics.compute_standard_metrics(y, p)
    assert result["standard_micro_ap"] == pytest.approx(200 / 3)
    assert result["standard_micro_fmax_exact"] == 80
    assert result["standard_micro_fmax_exact_threshold"] < 0
    assert np.all(p > result["standard_micro_fmax_exact_threshold"])
    assert result["standard_protein_fmax"] == 0


def test_no_gold_has_explicit_finite_zero_scores():
    result = metrics.compute_standard_metrics(np.zeros((2, 3)), np.ones((2, 3)))
    assert result["standard_num_protein_fmax_eligible"] == 0
    for key in ("standard_micro_ap", "standard_micro_pr_auc", "standard_protein_fmax", "standard_micro_fmax_exact"):
        assert result[key] == 0
    assert all(np.isfinite(value) for value in result.values())


@pytest.mark.parametrize("y,p,step,error", [
    ([[1]], [[np.nan]], .001, "finite"),
    ([[1]], [[1.1]], .001, "finite"),
    ([[.5]], [[.1]], .001, "binary"),
    ([[1]], [[.1]], .003, "divide 1"),
    ([[1]], [[.1]], 0, "positive"),
    ([1], [.1], .001, "2-D"),
])
def test_invalid_inputs_rejected(y, p, step, error):
    with pytest.raises(ValueError, match=error):
        metrics.compute_standard_metrics(np.array(y), np.array(p), step)


def test_user_legacy_grid_changes_under_monotonic_rescaling_but_exact_ap_does_not():
    import torch
    source = Path(__file__).with_name("fixtures") / "stage1_evalperf_torch_original.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evalperf_torch")
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function]
    isolated = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {"torch": torch, "th": torch, "math": math}
    exec(compile(isolated, str(source), "exec"), namespace)
    y = np.array([[1., 0., 1., 0.]])
    p = np.array([[.9, .8, .7, .6]])
    compressed = .01 * p + .795
    legacy = namespace["evalperf_torch"]
    before = legacy(torch.tensor(y), torch.tensor(p), threshold=True, auprc=True)
    after = legacy(torch.tensor(y), torch.tensor(compressed), threshold=True, auprc=True)
    assert before["auprc"] == pytest.approx(50)
    assert after["auprc"] == pytest.approx(25)
    assert before["fmax"] == pytest.approx(80)
    assert after["fmax"] == pytest.approx(200 / 3)
    original_standard = metrics.compute_standard_metrics(y, p)
    changed_standard = metrics.compute_standard_metrics(y, compressed)
    for field in ("standard_micro_ap", "standard_micro_pr_auc", "standard_micro_fmax_exact"):
        assert original_standard[field] == pytest.approx(changed_standard[field])
    assert original_standard["standard_micro_ap"] == pytest.approx(250 / 3)


def test_metric_source_is_bound_and_legacy_names_are_not_reused():
    result = metrics.compute_standard_metrics(np.array([[1, 0]]), np.array([[.8, .2]]))
    assert not ({"fmax", "auprc", "threshold"} & result.keys())
    assert metrics.metrics_implementation_sha256() == hashlib.sha256(PATH.read_bytes()).hexdigest()
    assert metrics.metric_definitions()["schema"] == metrics.METRIC_SCHEMA
