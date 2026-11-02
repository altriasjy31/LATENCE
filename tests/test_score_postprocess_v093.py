"""Math/metric regressions for the offline postprocessing module."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load('postprocess_v093_math_test', ROOT/'scripts/nbs/nbs_score_postprocess_v093.py')
METRIC_RELATIVE = Path('nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py')
METRIC_PATHS = (ROOT/METRIC_RELATIVE,
                ROOT.parent/'source/latence_nbs_v090_single12800'/METRIC_RELATIVE)
metric_path = next((p for p in METRIC_PATHS if p.is_file()), None)
if metric_path is None:
    raise FileNotFoundError('Tests require the existing project full_task_metrics_v085.py')
metric = load('postprocess_v093_production_metric_test', metric_path)


class TransformsAndFusion(unittest.TestCase):
    def setUp(self):
        self.b = np.array([[0., .2, .8, 1., .4, .9],
                           [.5, .3, .7, .6, .8, .2],
                           [1., .4, .6, .2, .1, 0.]], dtype=np.float32)
        self.g = np.array([[.7, .1, .3, 0., .8, .2],
                           [.6, .5, .2, .9, .7, .3],
                           [.1, .4, 1., .8, 0., 1.]], dtype=np.float32)
        self.counts = np.array([0, 5, 9, 99, 100, 1000])

    def test_transform_names_shapes_finiteness_and_no_mutation(self):
        before_b, before_g = self.b.copy(), self.g.copy()
        found = []
        for name, values, metadata in core.build_methods('transform', self.b, self.g, self.counts):
            found.append(name)
            self.assertEqual(values.shape, self.b.shape)
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(((values >= 0) & (values <= 1)).all())
            self.assertFalse(metadata['calibration_labels_used'])
            if name == 'G_sigmoid_again':
                self.assertEqual(values.dtype, np.float64)
                np.testing.assert_allclose(values, 1/(1+np.exp(-self.g.astype(np.float64))), rtol=1e-15)
        self.assertEqual(found, ['B', 'G', 'G_sigmoid_again', 'G_global_minmax',
                                 'G_row_max', 'G_go_logit_zscore'])
        np.testing.assert_array_equal(self.b, before_b)
        np.testing.assert_array_equal(self.g, before_g)

    def test_probability_fusion_alpha_is_graph_weight(self):
        values = {n: (v.copy(), m) for n, v, m in core.build_methods('mix', self.b, self.g, self.counts)}
        self.assertEqual(len(values), 9)
        for alpha in (.25, .5, .75):
            name = f'prob_mix_g{round(alpha*100):03d}'
            np.testing.assert_array_equal(values[name][0],
                (1-alpha)*self.b.astype(np.float64)+alpha*self.g.astype(np.float64))
            self.assertEqual(values[name][1]['graph_weight'], alpha)
        np.testing.assert_array_equal(values['max_BG'][0], np.maximum(self.b, self.g))

    def test_frequency_gate_preserves_protected_stored_values_and_dtype(self):
        for name, values, metadata in core.build_methods('mix', self.b, self.g, self.counts):
            if not name.startswith('freq_protect'):
                continue
            mask = self.counts <= metadata['core_count_cutoff']
            self.assertEqual(values.dtype, self.b.dtype)
            self.assertEqual(values[:, mask].tobytes(), self.b[:, mask].tobytes())
            np.testing.assert_allclose(values[:, ~mask], .5*self.b[:, ~mask]+.5*self.g[:, ~mask], rtol=1e-7)

    def test_logit_mixture_endpoints_are_finite_and_interior_is_exact_formula(self):
        values = next(v for n, v, _ in core.build_methods('mix', self.b, self.g, self.counts)
                      if n == 'logit_mix_g050')
        self.assertTrue(np.isfinite(values).all())
        mask = (self.b > 0) & (self.b < 1) & (self.g > 0) & (self.g < 1)
        b, g = self.b[mask].astype(float), self.g[mask].astype(float)
        expected = 1/(1+np.exp(-.5*(np.log(b/(1-b))+np.log(g/(1-g)))))
        np.testing.assert_allclose(values[mask], expected, atol=1e-15)
        self.assertEqual(core._stable_logit(np.array([1e-30]))[0],
                         np.log(1e-30)-np.log1p(-1e-30))

    def test_constant_columns_and_zero_rows_have_explicit_policies(self):
        constant = np.full((4, 6), .3)
        data = {n: (v.copy(), m) for n, v, m in core.build_methods('transform', constant, constant, self.counts)}
        np.testing.assert_array_equal(data['G_global_minmax'][0], np.zeros((4, 6)))
        np.testing.assert_allclose(data['G_go_logit_zscore'][0], .5, atol=1e-12)
        self.assertEqual(data['G_go_logit_zscore'][1]['columns_below_std_floor'], 6)
        self.assertTrue(data['G_go_logit_zscore'][1]['transductive_cohort_diagnostic'])
        zero = np.zeros((4, 6))
        row = next(v for n, v, _ in core.build_methods('transform', zero, zero, self.counts) if n == 'G_row_max')
        np.testing.assert_array_equal(row, zero)

    def test_logit_mixture_opposite_endpoints_are_symmetric_and_report_clipping(self):
        b = np.array([[0., 1., 1e-10, 1-1e-10]], dtype=np.float64)
        g = 1-b
        counts = np.array([1, 9, 100, 1000])
        first, description = next((v, m) for n, v, m in core.build_methods('mix', b, g, counts)
                                  if n == 'logit_mix_g050')
        second = next(v for n, v, _ in core.build_methods('mix', g, b, counts)
                      if n == 'logit_mix_g050')
        np.testing.assert_allclose(first, .5, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(description['logit_clipping'], 'symmetric_fixed_epsilon')
        self.assertEqual(description['logit_clip_epsilon'], 1e-6)
        for name in ('B', 'G'):
            self.assertEqual(description['input_clipping_counts'][name],
                             {'below_epsilon': 2, 'above_one_minus_epsilon': 2, 'total_clipped': 4})

    def test_nonfinite_out_of_range_shape_and_count_errors(self):
        invalid = self.g.copy(); invalid[0, 0] = np.nan
        for b, g, counts in [(self.b, invalid, self.counts),
                              (self.b, self.g-1, self.counts),
                              (self.b[:, :-1], self.g, self.counts),
                              (self.b, self.g, self.counts.astype(float)),
                              (self.b, self.g, -self.counts)]:
            with self.assertRaises(ValueError):
                list(core.build_methods('mix', b, g, counts))
        with self.assertRaises(ValueError):
            list(core.build_methods('other', self.b, self.g, self.counts))


class ExactMetrics(unittest.TestCase):
    def setUp(self):
        self.y = np.array([[1, 0, 0, 0, 1, 0],
                           [0, 1, 0, 1, 0, 0],
                           [1, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        self.p = np.array([[.8, .5, .2, .4, .9, .1],
                           [.8, .8, .2, .8, .2, .1],
                           [.2, .5, .9, .1, .2, .1],
                           [.8, .5, .2, .4, .2, .1]])
        self.counts = np.array([1, 5, 9, 99, 100, 1000])

    def test_summarize_reuses_original_protein_grid_and_exact_micro(self):
        summary = core.summarize_method(self.y, self.p, self.counts, metric, backbone_threshold=.5)
        production = metric.compute_standard_metrics(self.y, self.p)
        for output, source in [('fmax', 'standard_protein_fmax'),
                               ('threshold', 'standard_protein_fmax_threshold'),
                               ('precision', 'standard_protein_precision_at_fmax'),
                               ('recall', 'standard_protein_recall_at_fmax'),
                               ('micro_ap', 'standard_micro_ap')]:
            self.assertEqual(summary[output], production[source])
        self.assertEqual(summary['eligible_proteins'], 3)
        self.assertEqual(summary['macro_ap_eligible_go'], 5)
        fixed = summary['at_backbone_threshold']
        active = self.p > .5
        self.assertEqual(fixed['predicted_pairs'], int(active.sum()))
        self.assertEqual(fixed['true_positive_pairs'], int((active & (self.y != 0)).sum()))

    def test_macro_ap_ties_match_sklearn(self):
        from sklearn.metrics import average_precision_score
        values = core.per_go_average_precision(self.y, self.p)
        for column in range(5):
            self.assertAlmostEqual(values[column],
                100*average_precision_score(self.y[:, column], self.p[:, column]), places=12)
        self.assertTrue(np.isnan(values[5]))

    def test_macro_and_micro_ties_match_sklearn_random_fixtures(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(903)
        y = rng.integers(0, 2, (39, 271), dtype=np.uint8)
        p = rng.integers(0, 11, y.shape)/10
        actual = core.per_go_average_precision(y, p)
        expected = np.array([100*average_precision_score(y[:, i], p[:, i]) for i in range(y.shape[1])])
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        self.assertAlmostEqual(metric._exact_micro(y, p, rank_chunk=37)['standard_micro_ap'],
                               100*average_precision_score(y.ravel(), p.ravel()), places=12)

    def test_group_counts_ap_macro_and_overlapping_bins(self):
        summary = core.summarize_method(self.y, self.p, self.counts, metric, backbone_threshold=.5)
        groups = summary['frequency_strata']
        rare = next(g for g in groups if g['train_core_count_max'] == 5)
        mask = self.counts <= 5
        self.assertEqual(rare['columns'], 2)
        self.assertEqual(rare['test_positive_pairs'], 3)
        self.assertEqual(rare['micro_ap'], metric._exact_micro(self.y[:, mask], self.p[:, mask])['standard_micro_ap'])
        self.assertEqual(rare['macro_ap'], float(summary['_per_go_ap'][mask].mean()))
        zero = groups[0]
        self.assertEqual(zero['columns'], 0)
        self.assertIsNone(zero['micro_ap'])
        self.assertIsNone(zero['macro_ap'])
        self.assertEqual(zero['at_method_fmax']['predicted_pairs'], 0)

    def test_protected_rare_ap_and_fixed_threshold_detections_equal_backbone(self):
        g = self.p[::-1].copy()
        bsummary = core.summarize_method(self.y, self.p, self.counts, metric)
        gate = next(v for n, v, _ in core.build_methods('mix', self.p, g, self.counts) if n == 'freq_protect_9')
        mixed = core.summarize_method(self.y, gate, self.counts, metric, backbone_threshold=bsummary['threshold'])
        for b, m in zip(bsummary['frequency_strata'], mixed['frequency_strata']):
            if b['train_core_count_max'] not in (5, 9):
                continue
            self.assertEqual(b['micro_ap'], m['micro_ap'])
            self.assertEqual(b['macro_ap'], m['macro_ap'])
            self.assertEqual(b['at_method_fmax'], m['at_backbone_threshold'])

    def test_go_zscore_preserves_per_go_ap_away_from_clipping_and_tie_collapse(self):
        rng = np.random.default_rng(8)
        y = rng.integers(0, 2, (40, 6))
        p = rng.uniform(.02, .98, y.shape)
        z = next(v for n, v, _ in core.build_methods('transform', p, p, self.counts) if n == 'G_go_logit_zscore')
        np.testing.assert_allclose(core.per_go_average_precision(y, p),
                                   core.per_go_average_precision(y, z), atol=1e-12)

    def test_empty_gold_keeps_all_columns_and_returns_undefined_ap(self):
        y = np.zeros_like(self.y)
        summary = core.summarize_method(y, self.p, self.counts, metric)
        self.assertEqual(summary['fmax'], 0)
        self.assertEqual(summary['num_GO_columns'], 6)
        self.assertEqual(summary['eligible_proteins'], 0)
        self.assertIsNone(summary['micro_ap'])
        self.assertIsNone(summary['macro_ap'])
        self.assertTrue(np.isnan(summary['_per_go_ap']).all())
        self.assertTrue(all(item['micro_ap'] is None for item in summary['frequency_strata']))

    def test_invalid_labels_and_threshold_rejected(self):
        invalid = self.y.astype(float); invalid[0, 0] = .5
        with self.assertRaises(ValueError):
            core.summarize_method(invalid, self.p, self.counts, metric)
        with self.assertRaises(ValueError):
            core.summarize_method(self.y, self.p, self.counts, metric, backbone_threshold=np.nan)


if __name__ == '__main__':
    unittest.main()
