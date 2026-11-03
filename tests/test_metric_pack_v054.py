"""Parity against frozen Stage-1 metric functions and independent scalar counts."""
from __future__ import annotations

import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(os.environ.get('LATENCE_TEST_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'scripts'))
from dark_weak.complete_metric_pack_v054 import (  # noqa: E402
    comprehensive_metrics, load_kernels, metric_protocol,
)


def direct_source_functions(path, names, namespace):
    """Independent extraction in the test; no inference imports executed."""
    parsed = ast.parse(Path(path).read_text())
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[]))
    exec(compile(code, str(path), 'exec'), namespace)
    return namespace


class MetricPackParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernels = load_kernels(PROJECT, official=False)

    def compute(self, y, p, **kwargs):
        return comprehensive_metrics(np.asarray(y), np.asarray(p), **self.kernels_args(), **kwargs)

    def kernels_args(self):
        return {key: self.kernels[key] for key in ('native', 'legacy', 'official')}

    def test_legacy_exact_source_parity(self):
        names = ('fmax_from_histograms', 'exact_micro_auprc', 'filter_metric_arrays', 'compute_metric_pack')
        ref = direct_source_functions(PROJECT/'experiments/eval_weak_ind_test_detr_diagnostics.py',
                                     names, dict(np=np))['compute_metric_pack']
        rng = np.random.default_rng(453)
        y = (rng.random((7, 11)) < .17).astype(np.uint8)
        p = rng.choice(np.array([0, .01, .18, .5, .77, .99, 1], dtype=np.float32), size=y.shape)
        got = self.compute(y, p)
        expected = ref(y, p, .01, 'exact', True, False, False, True)
        for key, value in expected.items():
            self.assertEqual(got['stage1_hist__'+key], value, key)

    def test_scalar_histogram_reference_boundary_and_ties(self):
        y = np.array([[1, 0, 0, 1], [0, 0, 1, 0]], dtype=np.uint8)
        p = np.array([[0, .01, .1, .31], [1, .99, .31, .009999999]], dtype=np.float32)
        got = self.compute(y, p)
        indices = np.clip(np.floor(p/.01).astype(np.int32), 0, 100)
        best, best_i, precision_best, recall_best = -1., None, None, None
        ap, prev_recall = 0., 0.
        for threshold_bin in range(101):
            prediction = indices >= threshold_bin
            tp = int(np.sum(prediction & (y > 0)))
            precision = tp/max(int(prediction.sum()), 1)
            recall = tp/max(int(y.sum()), 1)
            score = 2*precision*recall/max(precision+recall, 1e-12)
            if score > best:
                best, best_i, precision_best, recall_best = score, threshold_bin, precision, recall
        for threshold_bin in range(100, -1, -1):
            prediction = indices >= threshold_bin
            tp = int(np.sum(prediction & (y > 0)))
            precision = tp/max(int(prediction.sum()), 1)
            recall = tp/max(int(y.sum()), 1)
            if recall > prev_recall:
                ap += (recall-prev_recall)*precision
                prev_recall = recall
        self.assertAlmostEqual(got['stage1_hist__fmax_micro'], 100*best)
        self.assertAlmostEqual(got['stage1_hist__threshold_micro'], float(np.float32(best_i)*.01), places=6)
        self.assertAlmostEqual(got['stage1_hist__precision_micro'], 100*precision_best)
        self.assertAlmostEqual(got['stage1_hist__recall_micro'], 100*recall_best)
        self.assertAlmostEqual(got['stage1_hist__auprc_micro_hist'], 100*ap)

    def test_native_retains_every_original_field(self):
        y = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
        p = np.array([[.9, .8, .1], [.5, .7, .6]], dtype=np.float32)
        expected = self.kernels['native']['compute'](y, p, threshold_step=.001)
        got = self.compute(y, p)
        for key, value in expected.items():
            self.assertEqual(got[key], value, key)
        self.assertEqual(got['Fmax_percent'], expected['standard_protein_fmax'])
        self.assertEqual(got['AP_percent'], expected['standard_micro_ap'])

    def test_empty_gold_rows_affect_micro_but_not_native_protein_fmax(self):
        got = self.compute([[1, 0], [0, 0]], [[.9, .1], [1, 1]])
        self.assertEqual(got['standard_protein_fmax'], 100.)
        self.assertAlmostEqual(got['stage1_hist__fmax_micro'], 50.)
        filtered = self.compute([[1, 0], [0, 0]], [[.9, .1], [1, 1]], no_empty_labels=True)
        self.assertEqual(filtered['standard_protein_fmax'], got['standard_protein_fmax'])
        self.assertEqual(filtered['stage1_hist__fmax_micro'], 100.)
        self.assertEqual(filtered['stage1_hist__num_proteins'], 1)

    def test_zero_score_tie_exposes_different_threshold_semantics(self):
        got = self.compute([[1, 0], [0, 0]], np.zeros((2, 2), dtype=np.float32))
        self.assertEqual(got['standard_protein_fmax'], 0.)
        self.assertEqual(got['stage1_hist__fmax_micro'], 40.)
        self.assertEqual(got['stage1_hist__auprc_micro_hist'], 25.)
        self.assertEqual(got['standard_micro_ap'], 25.)
        self.assertEqual(got['standard_micro_pr_auc'], 62.5)
        self.assertEqual(got['score_zero_fraction'], 1.)
        self.assertTrue(got['score_all_scores_equal'])
        self.assertEqual(got['score_positive_zero_fraction'], 1.)

    def test_no_positive_labels_and_empty_axis(self):
        got = self.compute(np.zeros((2, 3)), np.ones((2, 3))*.3)
        self.assertFalse(got['metric_has_positive_labels'])
        self.assertEqual(got['standard_micro_ap'], 0.)
        self.assertEqual(got['standard_protein_fmax'], 0.)
        self.assertIsNone(got['score_positive_zero_fraction'])
        for shape in ((2, 0), (0, 3)):
            empty = self.compute(np.zeros(shape), np.zeros(shape))
            self.assertTrue(empty['metric_empty_axis'])
            self.assertIsNone(empty['Fmax_percent'])
            self.assertIsNone(empty['stage1_hist__fmax_micro'])
            self.assertEqual(empty['standard_num_go'], shape[1])

    def test_degenerate_filtered_axes_and_invalid_inputs(self):
        got = self.compute(np.zeros((2, 3)), np.zeros((2, 3)), no_empty_labels=True)
        self.assertIsNone(got['stage1_hist__fmax_micro'])
        got = self.compute(np.zeros((2, 3)), np.zeros((2, 3)), no_zero_classes=True)
        self.assertIsNone(got['stage1_hist__fmax_micro'])
        for y, p in (([[2]], [[.1]]), ([[1]], [[np.nan]]), ([[1]], [[1.1]])):
            with self.assertRaises(ValueError):
                self.compute(y, p)
        with self.assertRaises(ValueError):
            self.compute([[1]], [[.3]], threshold_step=.03)
        with self.assertRaises(ValueError):
            self.compute([[1]], [[.3]], ic=[-1])

    def test_protocol_provenance_and_disabled_official(self):
        protocol = metric_protocol(self.kernels)
        self.assertEqual(protocol['stage1_hist']['threshold_step'], .01)
        self.assertFalse(protocol['stage1_official']['enabled'])
        source = protocol['sources']['stage1_hist']
        self.assertEqual(len(source['sha256']), 64)
        self.assertEqual(len(source['function_sha256']['fmax_from_histograms']), 64)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable; original official AST requires torch')
    def test_original_official_ast_parity_including_smin(self):
        import torch
        kernels = load_kernels(PROJECT, official=True)
        source = kernels['sources']['stage1_official']['path']
        fn = direct_source_functions(source, ('evalperf_torch',), dict(torch=torch, th=torch, math=math))['evalperf_torch']
        y = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float32)
        p = np.array([[.9, .1, .9], [1, 0, .3]], dtype=np.float32)
        ic = np.array([1, 3, 2], dtype=np.float32)
        got = comprehensive_metrics(y, p, native=kernels['native'], legacy=kernels['legacy'], official=kernels['official'], ic=ic)
        ref = fn(torch.from_numpy(y), torch.from_numpy(p), threshold=True, auprc=True,
                 smin=True, icary=torch.from_numpy(ic))
        for key, value in ref.items():
            self.assertEqual(got['stage1_official__'+key], value, key)
        # The helper's strict > excludes all zero probabilities at t=0.
        zero = comprehensive_metrics(y, np.zeros_like(p), native=kernels['native'], legacy=kernels['legacy'], official=kernels['official'])
        self.assertEqual(zero['stage1_official__fmax'], 0.)


if __name__ == '__main__':
    unittest.main()
