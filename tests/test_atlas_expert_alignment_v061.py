"""Meaningful ID/precision tests for larger-cohort Expert slicing."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('atlas_expert_align_v061', ROOT/'scripts/dark_weak/align_atlas_expert_v061.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class ExpertAlignmentTests(unittest.TestCase):
    def test_verified_id_reorder_preserves_tiny_float32_signal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = np.full((3, 21312), .2, dtype=np.float32)
            source[0, 4] = np.float32(1e-8)
            source[2, 7] = np.float32(.0030021667)
            np.save(root/'source.npy', source)
            (root/'source.ids').write_text('A\nB\nC\n')
            (root/'target.ids').write_text('C\nA\n')
            report = mod.align(probabilities=root/'source.npy', source_ids=root/'source.ids',
                target_ids=root/'target.ids', output_probabilities=root/'aligned.npy',
                output_ids=root/'aligned.ids', task='bp')
            actual = np.load(root/'aligned.npy')
            self.assertEqual(actual.dtype, np.dtype('float32'))
            np.testing.assert_array_equal(actual, source[[2, 0]])
            self.assertGreater(actual[1, 4], 0)
            self.assertEqual((root/'aligned.ids').read_text(), 'C\nA\n')
            self.assertEqual(report['output_dtype'], 'float32')
            with self.assertRaises(FileExistsError):
                mod.align(probabilities=root/'source.npy', source_ids=root/'source.ids',
                    target_ids=root/'target.ids', output_probabilities=root/'aligned.npy',
                    output_ids=root/'aligned.ids', task='bp')

    def test_small_old_export_cannot_fill_missing_targets(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            np.save(root/'source.npy', np.zeros((1, 21312), np.float32))
            (root/'source.ids').write_text('A\n')
            (root/'target.ids').write_text('A\nD\n')
            with self.assertRaisesRegex(ValueError, 'targets missing'):
                mod.align(probabilities=root/'source.npy', source_ids=root/'source.ids',
                    target_ids=root/'target.ids', output_probabilities=root/'aligned.npy',
                    output_ids=root/'aligned.ids', task='bp')
            self.assertFalse((root/'aligned.npy').exists())

    def test_original_metadata_order_is_explicit_source_axis(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = np.array([[.2]*7038, [.7]*7038], np.float32)
            np.save(root/'source.npy', values)
            metadata = {'exp_train': {'molecular_function': {'proteins': ['B', 'A']}}}
            (root/'source.pkl').write_bytes(pickle.dumps(metadata))
            (root/'target.ids').write_text('A\nB\n')
            report = mod.align(probabilities=root/'source.npy', source_metadata=root/'source.pkl',
                source_split='exp_train', target_ids=root/'target.ids',
                output_probabilities=root/'aligned.npy', output_ids=root/'aligned.ids', task='mf')
            np.testing.assert_array_equal(np.load(root/'aligned.npy'), values[[1, 0]])
            self.assertFalse(report['historical_source_time_cutoff_verified'])


if __name__ == '__main__':
    unittest.main()
