"""Numerical equivalence to the original dense complete metric pack."""
from __future__ import annotations

import importlib.util
import ast
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
_reference_candidates = [Path(os.environ['LATENCE_REFERENCE_PROJECT'])] if os.environ.get('LATENCE_REFERENCE_PROJECT') else []
_reference_candidates += [ROOT, ROOT.parent / 'validation_v060']
REFERENCE_PROJECT = next((path for path in _reference_candidates
    if (path / 'nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py').is_file()),
    _reference_candidates[0])


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pack = _load('reference_pack_v062', 'scripts/dark_weak/complete_metric_pack_v054.py')
stream = _load('stream_pack_v062', 'scripts/dark_weak/streaming_metric_pack_v062.py')
TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(REFERENCE_PROJECT.is_dir(), 'Original source fixture required')
class StreamingMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernels = pack.load_kernels(REFERENCE_PROJECT, official=False)

    def compare(self, labels, scores, *, chunk_rows=2, threshold_step=.01, official=False):
        kernels = pack.load_kernels(REFERENCE_PROJECT, official=True) if official else self.kernels
        expected = pack.comprehensive_metrics(labels, scores,
            native=kernels['native'], legacy=kernels['legacy'], official=kernels['official'],
            threshold_step=threshold_step)
        actual = stream.comprehensive_metrics_streaming(scores, np.arange(len(scores)),
            [np.flatnonzero(row) for row in labels], kernels=kernels,
            chunk_rows=chunk_rows, threshold_step=threshold_step)
        for key, value in expected.items():
            if key == 'evaluation_backend':
                continue
            self.assertIn(key, actual)
            if value is None or isinstance(value, (bool, str)):
                self.assertEqual(actual[key], value, key)
            else:
                self.assertAlmostEqual(float(actual[key]), float(value), delta=1e-8, msg=key)
        return actual

    def test_negative_only_groups_before_between_and_after_positive_groups(self):
        scores = np.asarray([[.99, .95, .9, .85, .7, .7, .69, .6, .3, .1]], np.float32)
        labels = np.asarray([[0, 1, 0, 0, 1, 0, 0, 0, 1, 0]], np.uint8)
        actual = self.compare(labels, scores, chunk_rows=1)
        # This exact trapezoidal area differs from connecting positive groups
        # without retaining their intervening negative-only precision changes.
        self.assertGreater(actual['standard_micro_pr_auc'], 0)

    def test_exact_score_ties_cross_chunks_and_all_zero_score(self):
        labels = np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, 0]], np.uint8)
        scores = np.zeros(labels.shape, np.float32)
        actual = self.compare(labels, scores, chunk_rows=1)
        q = labels.mean()
        self.assertAlmostEqual(actual['standard_micro_ap'], 100 * q)
        self.assertAlmostEqual(actual['standard_micro_pr_auc'], 100 * (1 + q) / 2)
        self.assertLess(actual['standard_micro_fmax_exact_threshold'], 0)
        self.assertEqual(actual['standard_protein_fmax'], 0)

    def test_zero_positive_labels_preserve_missing_legacy_exact_ap(self):
        self.compare(np.zeros((3, 5), np.uint8),
                     np.random.default_rng(60).random((3, 5), dtype=np.float32))

    def test_empty_gold_rows_have_distinct_native_and_legacy_averaging(self):
        labels = np.asarray([[1, 0, 0], [0, 0, 0], [0, 1, 1]], np.uint8)
        scores = np.asarray([[.9, .1, 0], [.99, .98, .97], [.2, .9, .9]], np.float32)
        actual = self.compare(labels, scores)
        self.assertEqual(actual['standard_num_empty_gold_proteins'], 1)

    def test_float32_grid_boundaries_and_adjacent_values(self):
        anchors = np.asarray([0, .001, .01, .1, .29, .3, .5, .99, 1], np.float32)
        values = np.r_[np.nextafter(anchors[1:], np.float32(-np.inf)),
                       anchors, np.nextafter(anchors[:-1], np.float32(np.inf))]
        scores = np.tile(values, (4, 1))
        labels = np.zeros_like(scores, np.uint8)
        labels[0, ::2] = 1
        labels[2, 1::3] = 1
        labels[3, -1] = 1
        self.compare(labels, scores)

    def test_random_continuous_and_rounded_ties_multiple_chunks(self):
        rng = np.random.default_rng(6062)
        for seed in range(12):
            scores = rng.random((7 + seed % 3, 17 + seed), dtype=np.float32)
            if seed % 2:
                scores = np.round(scores, 1)
            labels = (rng.random(scores.shape) < .15).astype(np.uint8)
            with self.subTest(seed=seed):
                self.compare(labels, scores, chunk_rows=1 + seed % 4)

    def test_nondefault_histogram_grid(self):
        rng = np.random.default_rng(99)
        scores = rng.random((5, 23), dtype=np.float32)
        labels = (rng.random(scores.shape) < .2).astype(np.uint8)
        self.compare(labels, scores, threshold_step=.02)

    def test_mmap_selected_reordered_rows_and_no_global_argsort(self):
        rng = np.random.default_rng(33)
        scores = rng.random((9, 37), dtype=np.float32)
        labels = (rng.random((4, 37)) < .1).astype(np.uint8)
        selected = np.asarray([8, 1, 5, 0])
        expected = self.compare(labels, scores[selected])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'scores.npy'
            np.save(path, scores)
            mmap = np.load(path, mmap_mode='r')
            with mock.patch.object(stream.np, 'argsort', side_effect=AssertionError('Full argsort forbidden')):
                actual = stream.comprehensive_metrics_streaming(mmap, selected,
                    [np.flatnonzero(row) for row in labels], kernels=self.kernels, chunk_rows=2)
            for key in ['standard_micro_ap', 'standard_micro_pr_auc', 'standard_micro_fmax_exact']:
                self.assertAlmostEqual(actual[key], expected[key], delta=1e-8)

    @unittest.skipUnless(TORCH_AVAILABLE, 'Torch is required for original official helper parity')
    def test_original_official_torch_threshold_arithmetic_and_trapezoid(self):
        anchors = np.asarray([0, .01, .29, .3, .5, .99, 1], np.float32)
        scores = np.asarray([anchors, np.nextafter(anchors, np.float32(-np.inf))], np.float32)
        scores = np.clip(scores, 0, 1)
        labels = np.asarray([[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0]], np.uint8)
        self.compare(labels, scores, official=True)

    def test_invalid_rows_and_gold_indices_fail(self):
        scores = np.ones((2, 3), np.float32)
        for rows, gold in [(np.array([0, 0]), [[0], [1]]),
                           (np.array([3]), [[0]]), (np.array([0]), [[4]])]:
            with self.subTest(rows=rows, gold=gold), self.assertRaises(ValueError):
                stream.comprehensive_metrics_streaming(scores, rows, gold, kernels=self.kernels)

    def test_reference_contract_ignores_comments_but_detects_numerical_change(self):
        verified = stream.verify_reference_contract(self.kernels)
        self.assertTrue(verified['checked'])
        with tempfile.TemporaryDirectory() as td:
            source = Path(self.kernels['sources']['native_kernel']['path']).read_text()
            path = Path(td) / 'native.py'
            path.write_text(source + '\n# Unrelated formatting or comments are harmless.\n')
            kernels = copy.copy(self.kernels)
            kernels['sources'] = copy.deepcopy(self.kernels['sources'])
            kernels['sources']['native_kernel']['path'] = str(path)
            stream.verify_reference_contract(kernels)
            tree = ast.parse(source)
            function = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == 'compute_standard_metrics')
            function.args.defaults[-1] = ast.Constant(value=.002)
            path.write_text(ast.unparse(tree))
            with self.assertRaisesRegex(RuntimeError, 'reference contract changed'):
                stream.verify_reference_contract(kernels)


if __name__ == '__main__':
    unittest.main()
