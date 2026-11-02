from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = load(PACKAGE/'scripts/nbs/predict_nbs_go_v093.py', 'v093_predict_test_cli')
report = load(PACKAGE/'nbs_models/nbs_protein_go/nbs_pg/evaluation_report_v093.py', 'v093_report_test_module')
OLD = PACKAGE/'nbs_models/nbs_protein_go/nbs_pg'
if not (OLD/'go_registry_v090.py').is_file():
    OLD = WORKSPACE/'fusion_v093/source/latence_nbs_v090_single12800/nbs_models/nbs_protein_go/nbs_pg'
registry_module = load(OLD/'go_registry_v090.py', 'v093_predict_test_registry')
metrics = load(OLD/'full_task_metrics_v085.py', 'v093_predict_test_metrics')
Registry = registry_module.GORegistryV090


def registry():
    return Registry([{'go_id': 'GO:0000001', 'ontology_row': 0, 'alt_ids': ['GO:0000011']},
                     {'go_id': 'GO:0000002', 'ontology_row': 1},
                     {'go_id': 'GO:0000003', 'ontology_row': 2}],
                    ['GO:0000001', 'GO:0000011', 'GO:0000002'],
                    np.array([0, 0, 1]), num_ontology_rows=3)


@dataclass
class ModelConfig:
    hidden_dim: int = 4


@dataclass
class Phase:
    steps_per_epoch: int


def validation_fixture():
    config = {'full_task': {'experiment_revision': '0.9.3-unified-r1',
                           'model': {'hidden_dim': 4}, 'sampler': {'topk': 16}}}
    data = SimpleNamespace(data_contract=lambda: {'dataset': 'new_weak_hash'})
    base = SimpleNamespace(RELEASE_VERSION='0.9.0', MODEL_VERSION='0.9.0', DATA_VERSION='0.8.8',
                           PhasePlanV090=Phase, _phase_fields=lambda phase, step: {'epoch': 2, 'cumulative_completed_epochs': 2},
                           training_baseline_identity=lambda d: {'base': 'verified'},
                           ontology_contract=lambda ontology: {'coordinates': 'verified'},
                           prediction_implementation=lambda variant: {'model.py': 'trained_hash'},
                           FullTaskModelConfigV090=ModelConfig)
    saved = {'runner_version': '0.9.0', 'model_architecture_version': '0.9.0',
             'data_architecture_version': '0.8.8', 'phase_plan': {'steps_per_epoch': 3},
             'step': 6, 'epoch': 2, 'cumulative_completed_epochs': 2,
             'training_baseline_identity': {'base': 'verified'},
             'contract': {'data': data.data_contract(), 'ontology': {'coordinates': 'verified'}},
             'variant': 'dynamic', 'config': deepcopy(config),
             'training_implementation': {'model.py': 'trained_hash'}, 'model_config': {'hidden_dim': 4}}
    return base, saved, config, data


class RequestedGOTests(unittest.TestCase):
    def test_GO_list_order_alias_notation_and_comments(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'targets.txt'
            path.write_text('# requested\nGO_0000003\n\nGO:0000001\n')
            self.assertEqual(cli.read_go_ids(path), ['GO:0000003', 'GO:0000001'])
            path.write_text('GO:0000001\nGO_0000001\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                cli.read_go_ids(path)
            path.write_text('arbitrary protein\n')
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                cli.read_go_ids(path)

    def test_seen_alias_unseen_task_and_ontology_only_status(self):
        rows, records = cli.resolve_targets(registry(), ['GO:0000011', 'GO:0000002', 'GO:0000003'],
                                            np.array([True, True, False]))
        np.testing.assert_array_equal(rows, [0, 1, 2])
        self.assertEqual(records[0]['seen_task_columns'], [0, 1])
        self.assertEqual(records[0]['readout'], 'backbone_plus_shared_residual')
        self.assertEqual(records[1]['readout'], 'shared_protein_GO_scorer')
        self.assertEqual(records[2]['task_columns'], [])
        self.assertFalse(records[2]['stage2_supervised'])

    def test_unknown_GO_requires_supplied_geometry(self):
        with self.assertRaisesRegex(ValueError, 'no supplied ontology geometry'):
            cli.resolve_targets(registry(), ['GO:9999999'], np.array([True, True, False]))

    def test_extension_preserves_original_GO_and_alias_rows(self):
        records = [{'go_id': 'GO:0000001', 'ontology_row': 2, 'alt_ids': ['GO:0000011']},
                   {'go_id': 'GO:0000002', 'ontology_row': 0},
                   {'go_id': 'GO:0000003', 'ontology_row': 1},
                   {'go_id': 'GO:0000004', 'ontology_row': 3}]
        new = cli.updated_registry(registry(), {'registry_records': records}, np.array([2, 0, 1]), 4, Registry)
        np.testing.assert_array_equal(new.task_to_ontology, [2, 2, 0])
        self.assertEqual(new.resolve_one('GO:0000004'), 3)
        wrong = deepcopy(records)
        wrong[2]['ontology_row'] = 3
        with self.assertRaisesRegex(ValueError, 'anchored row'):
            cli.updated_registry(registry(), {'registry_records': wrong}, np.array([2, 0, 1]), 4, Registry)
        with self.assertRaisesRegex(ValueError, 'injectively'):
            cli.updated_registry(registry(), {'registry_records': records}, np.array([2., 0., 1.]), 4, Registry)

    def test_checkpoint_validation_uses_original_phase_data_code_contracts(self):
        base, saved, config, data = validation_fixture()
        checked = cli.validate_checkpoint_identity(base, saved, Path('nbs_epoch2.pt'), config, data, {})
        self.assertEqual(checked['variant'], 'dynamic')
        for field, value in [('training_baseline_identity', {'base': 'changed'}),
                             ('contract', {}), ('training_implementation', {}),
                             ('model_config', {'hidden_dim': 8}), ('epoch', 3)]:
            altered = deepcopy(saved)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                cli.validate_checkpoint_identity(base, altered, Path('nbs_epoch2.pt'), config, data, {})
        with self.assertRaisesRegex(ValueError, 'fixed completed-epoch'):
            cli.validate_checkpoint_identity(base, saved, Path('latest.pt'), config, data, {})
        changed = deepcopy(config)
        changed['full_task']['sampler']['topk'] = 32
        with self.assertRaisesRegex(ValueError, 'sampler'):
            cli.validate_checkpoint_identity(base, saved, Path('nbs_epoch2.pt'), changed, data, {})
        changed = deepcopy(config)
        changed['full_task']['experiment_revision'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'revision'):
            cli.validate_checkpoint_identity(base, saved, Path('nbs_epoch2.pt'), changed, data, {})


class ReportTests(unittest.TestCase):
    def test_fixed_working_points_use_overall_thresholds_and_strict_comparison(self):
        y = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], bool)
        b = np.array([[.9, .2, .1], [.3, .8, .1], [.8, .7, .1]])
        g = np.array([[.6, .3, .8], [.4, .6, .8], [.7, .5, .8]])
        result = report.summarize_predictions(y, {'backbone_base': b, 'NBS_final': g},
                                              np.array([1, 20, 0]), np.array([True, True, False]), metrics)
        rare = result['frequency_strata'][1]['methods']['NBS_final']
        self.assertEqual(rare['at_overall_fmax']['threshold'], result['methods']['NBS_final']['threshold'])
        self.assertEqual(rare['at_backbone_overall_threshold']['threshold'], result['methods']['backbone_base']['threshold'])
        expected = report.decision_counts(y, g, np.array([True, False, False]), result['methods']['backbone_base']['threshold'])
        self.assertEqual(rare['at_backbone_overall_threshold'], expected)
        tied = report.decision_counts(y, b, np.array([True, False, False]), .8)
        self.assertEqual(tied['predicted_pairs'], 1)
        self.assertEqual(tied['tp'], 1)
        zero = result['frequency_strata'][0]['methods']['NBS_final']
        self.assertIsNone(zero['micro_ap'])
        self.assertEqual(zero['at_backbone_overall_threshold']['test_positive_pairs'], 0)
        self.assertIsNone(zero['at_backbone_overall_threshold']['pair_recall'])

    def test_loaded_backbone_reference_preserves_original_and_clips_endpoints(self):
        original = np.array([[0., .2, 1.]], np.float16)
        reference = report.clipped_backbone_reference(original, 1e-4)
        self.assertEqual(reference.dtype, np.float32)
        self.assertAlmostEqual(float(reference[0, 0]), 1e-4, places=10)
        self.assertAlmostEqual(float(reference[0, 2]), 1-1e-4, places=7)
        np.testing.assert_array_equal(original, [[0., np.float16(.2), 1.]])
        with self.assertRaises(ValueError):
            report.clipped_backbone_reference(original, 0)

    def test_AP_ties_match_sklearn_and_exclude_no_positive_columns(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(93)
        y = rng.random((50, 9)) < .2
        y[:, -1] = False
        p = rng.integers(0, 5, size=y.shape)/4
        result = report.per_go_average_precision(y, p, block_columns=3)
        for column in range(8):
            self.assertAlmostEqual(result[column], 100*average_precision_score(y[:, column], p[:, column]), places=12)
        self.assertTrue(np.isnan(result[-1]))

    def test_seen_unseen_groups_are_explicit_and_AP_percent_correct(self):
        y = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], bool)
        p = np.array([[.9, .1, .7], [.2, .8, .7], [.8, .7, .7]])
        result = report.summarize_predictions(y, {'NBS_final': p}, np.array([1, 20, 0]),
                                              np.array([True, False, False]), metrics)
        self.assertAlmostEqual(result['methods']['NBS_final']['micro_ap'], metrics._exact_micro(y.astype(np.int8), p)['standard_micro_ap'])
        self.assertEqual(result['supervision_subsets']['seen']['GO_columns'], 1)
        self.assertEqual(result['supervision_subsets']['unseen']['GO_columns'], 2)
        self.assertEqual(result['supervision_subsets']['unseen']['methods']['NBS_final']['test_positive_pairs'], 2)
        self.assertFalse(result['strict_pipeline_zero_shot_claimed'])
        zero = result['frequency_strata'][0]['methods']['NBS_final']
        self.assertIsNone(zero['micro_ap'])
        self.assertTrue(result['frequency_bins_overlap'])


if __name__ == '__main__':
    unittest.main()
