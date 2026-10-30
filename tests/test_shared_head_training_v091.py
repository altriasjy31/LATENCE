"""Stage identity, binary targets and sampling tests; optional real torch checks."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / 'scripts/nbs/train_nbs_shared_head_v091.py'
spec = importlib.util.spec_from_file_location('shared_head_training_v091', SCRIPT)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def config():
    return {'source_epoch': 60, 'source_checkpoint': 'source/nbs_epoch60.pt',
            'source_config': 'source/resolved_config.json', 'cache_dir': 'cache',
            'head': {'hidden_dim': 320, 'mlp_width': 512, 'initial_bias': -6.},
            'train': {'seed': 1091, 'weak_batch': 8, 'core_batch': 2, 'steps': 500,
                      'learning_rate': .001, 'warmup_steps': 20, 'weight_decay': .0001,
                      'grad_clip': 5., 'save_every': 100}, 'loss': None}


def manifest():
    return {'complete': True, 'source_epoch': 60,
            'source_config': {'full_task': {'loss': {'core_weight': 1., 'weak_weight': .25}}}}


class SharedHeadStageTests(unittest.TestCase):
    def test_inherited_loss_is_resolved_without_mutation(self):
        original, cache = config(), manifest()
        result = training.validated_config(original, cache)
        self.assertIsNone(original['loss'])
        self.assertEqual(result['loss'], {'core_weight': 1., 'weak_weight': .25})
        result['loss']['weak_weight'] = 1.
        self.assertEqual(cache['source_config']['full_task']['loss']['weak_weight'], .25)

    def test_explicit_research_loss_remains_explicit(self):
        supplied = config()
        supplied['loss'] = {'core_weight': .65, 'weak_weight': 1.}
        self.assertEqual(training.validated_config(supplied, manifest())['loss'], supplied['loss'])

    def test_source_epoch_and_complete_cache_are_bound(self):
        for value in (5, 65, 60., True):
            supplied = config(); supplied['source_epoch'] = value
            with self.assertRaises(ValueError):
                training.validated_config(supplied, manifest())
        cache = manifest(); cache['source_epoch'] = 70
        with self.assertRaises(ValueError):
            training.validated_config(config(), cache)
        cache = manifest(); cache['complete'] = False
        with self.assertRaises(ValueError):
            training.validated_config(config(), cache)

    def test_invalid_stage_hyperparameters_are_rejected(self):
        for name, value in (('steps', True), ('core_batch', 0), ('learning_rate', float('nan')),
                            ('learning_rate', 0), ('warmup_steps', 501), ('weight_decay', -1)):
            supplied = config(); supplied['train'][name] = value
            with self.assertRaises(ValueError, msg=name):
                training.validated_config(supplied, manifest())

    def test_new_warmup_counter_is_not_old_gnn_epoch_schedule(self):
        cfg = config()['train']
        self.assertAlmostEqual(training.stage_learning_rate(1, cfg), .00005)
        self.assertAlmostEqual(training.stage_learning_rate(20, cfg), .001)
        self.assertEqual(training.stage_learning_rate(500, cfg), .001)
        for invalid in (0, 501):
            with self.assertRaises(ValueError):
                training.stage_learning_rate(invalid, cfg)

    def test_csr_membership_duplicates_empty_rows_and_order(self):
        indptr = np.array([0, 3, 3, 5])
        indices = np.array([2, 2, 0, 3, 1])
        value = training.dense_membership(indptr, indices, [2, 0, 1], 4)
        np.testing.assert_array_equal(value, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0]])
        self.assertEqual(value.dtype, np.float32)
        with self.assertRaises(ValueError):
            training.dense_membership([0, 1], [4], [0], 4)

    def test_role_sampling_is_unique_and_rng_resume_is_exact(self):
        rng = np.random.default_rng(19)
        weak, core = np.arange(100), np.arange(100, 120)
        first = training.sample_rows(rng, weak, core, 8, 2)
        state = deepcopy(rng.bit_generator.state)
        expected = training.sample_rows(rng, weak, core, 8, 2)
        other = np.random.default_rng(999); other.bit_generator.state = state
        actual = training.sample_rows(other, weak, core, 8, 2)
        self.assertEqual(len(np.unique(first)), 10)
        self.assertTrue((first[:8] < 100).all())
        self.assertTrue((first[8:] >= 100).all())
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ValueError):
            training.sample_rows(rng, weak, core, 101, 2)

    def test_resume_rejects_source_or_cache_or_directory_changes(self):
        identity = {'config': config(), 'cache_manifest_sha256': 'cache-a',
                    'source_checkpoint_sha256': 'source-a'}
        with tempfile.TemporaryDirectory() as path:
            saved = {'version': training.VERSION, 'identity': deepcopy(identity),
                     'step': 100, 'work_dir': str(Path(path).resolve())}
            training.assert_resume_contract(saved, identity, Path(path))
            for key in ('cache_manifest_sha256', 'source_checkpoint_sha256'):
                changed = deepcopy(identity); changed[key] += '-changed'
                with self.assertRaises(ValueError):
                    training.assert_resume_contract(saved, changed, Path(path))
            with self.assertRaises(ValueError):
                training.assert_resume_contract(saved, identity, Path(path) / 'other')
            saved['step'] = 501
            with self.assertRaises(ValueError):
                training.assert_resume_contract(saved, identity, Path(path))

    def test_identity_hash_is_canonical_and_changes_with_loss(self):
        self.assertEqual(training.json_hash({'a': 1, 'b': 2}), training.json_hash({'b': 2, 'a': 1}))
        value = config(); changed = deepcopy(value); changed['loss'] = {'core_weight': .65, 'weak_weight': 1.}
        self.assertNotEqual(training.json_hash(value), training.json_hash(changed))


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires real PyTorch; not available in build environment')
class SharedHeadTorchTests(unittest.TestCase):
    def test_real_forward_backward_new_geometry_and_nonpersistent_rows(self):
        import torch
        path = PACKAGE / 'nbs_models/nbs_protein_go/nbs_pg/shared_head_v091.py'
        spec = importlib.util.spec_from_file_location('real_shared_head_v091', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        geometry = torch.arange(60).reshape(10, 6).float()
        model = module.SharedGeometryHeadV091(geometry, hidden_dim=4, mlp_width=8)
        hidden = torch.randn(3, 4)
        output = model(hidden)
        self.assertEqual(output.shape, (3, 10))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
        self.assertNotIn('geometry', model.state_dict())
        self.assertIn('geometry_mean', model.state_dict())
        torch.testing.assert_close(model.score_geometry(hidden, geometry[:2]), model(hidden)[:, :2])
        self.assertFalse(any('geometry' in name for name, _ in model.named_parameters()))


if __name__ == '__main__':
    unittest.main()
