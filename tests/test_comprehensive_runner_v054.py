"""End-to-end synthetic evaluation: identity alignment, raw bins and ranks."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture_module = load_module('frequency_fixture_v054', ROOT/'tests/test_dark_weak_stage1_frequency_v053.py')
mod = load_module('comprehensive_runner_v054', ROOT/'scripts/dark_weak/eval_stage1_metrics_v054.py')


def options(manifest, train, unified, out, *extra):
    return mod.parser().parse_args(['--complete-manifest', str(manifest),
                '--train-metadata', str(train), '--unified-root', str(unified),
                '--skip-stage1-official', '--bootstrap', '4', '--ks', '1', '3', '5',
                '--out', str(out), *extra])


def quiet_run(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.run(args)


class ComprehensiveRunnerTests(unittest.TestCase):
    def test_full_suite_original_intact_raw_groups_native_and_rank_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, train, unified, inp, gos, original = fixture_module.fixture(root)
            before = mod.common.sha256(manifest)
            result, audit = quiet_run(options(manifest, train, unified, root/'output'))
            self.assertEqual(mod.common.sha256(manifest), before)
            self.assertFalse(audit['original_reports_modified'])
            self.assertEqual(len(result), 3 * (1 + 6 + 6 + 6) * 3)
            self.assertEqual(set(result.family), {'overall', 'stage1_raw_classic', 'historical_classic', 'historical_fixed'})
            for key in ('stage1_hist__fmax_micro', 'stage1_hist__precision_micro',
                        'stage1_hist__auprc_micro_hist', 'standard_protein_fmax',
                        'standard_micro_ap', 'standard_micro_fmax_exact'):
                self.assertIn(key, result)
            raw = result[result.family == 'stage1_raw_classic']
            self.assertTrue(raw.rank_status.str.startswith('not_defined').all())
            # Raw masks exactly follow the original Stage-1 quantile function,
            # not canonical masks expanded onto alias-equivalent columns.
            lut = pd.read_csv(root/'output/go_frequency_lookup.tsv', sep='\t')
            masks = mod.common.classic_bins(lut.historical_raw_count.to_numpy())[0]
            for name, mask in masks.items():
                np.testing.assert_array_equal(lut['stage1_raw_classic__'+name], mask)
            ranks = pd.read_csv(root/'output/positive_global_ranks.tsv', sep='\t')
            self.assertTrue((ranks['rank_optimistic'] <= ranks['rank']).all())
            self.assertTrue((ranks['rank'] <= ranks['rank_pessimistic']).all())
            self.assertNotIn('stage1_raw_classic', set(ranks.family))
            self.assertEqual(set(ranks.rank_vocabulary_size), {6}) # 7 canonical GO minus root
            self.assertEqual(len(pd.read_csv(root/'output/overall_metrics.tsv', sep='\t')), 9)
            stored = json.loads((root/'output/metrics_manifest.json').read_text())
            self.assertEqual(stored['schema'], mod.SCHEMA)
            self.assertFalse(stored['metric_protocol']['stage1_official']['enabled'])
            self.assertTrue((result[result.n_metric_positive_columns == 0].status.isin(['no_positives', 'empty_group'])).all())
            with self.assertRaises(FileExistsError):
                quiet_run(options(manifest, train, unified, root/'output'))

    def test_explicit_indtest_and_reordered_prediction_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, train, unified, inp, gos, original = fixture_module.fixture(root)
            # Independent metadata has propagated labels only, as original
            # ind_test can; never silently describe these ranks as future-direct.
            data = pickle.loads(Path(original['metadata']).read_bytes())
            data['ind_test'] = data['dark_weak_test_direct']
            data['ind_test']['biological_process'].pop('annotations')
            metadata = root/'ind.pkl'
            metadata.write_bytes(pickle.dumps(data))
            pp = inp/'backbone.npy'
            ids = inp/'protein_ids.txt'
            command = ['--metadata', str(metadata), '--namespace', 'bp', '--splits', 'ind_test',
                '--train-metadata', str(train), '--unified-root', str(unified),
                '--prediction', f'backbone={pp}', '--prediction-ids', f'backbone={ids}',
                '--prediction-go-ids', f'backbone={inp / "go_ids.txt"}',
                '--skip-stage1-official', '--bootstrap', '0', '--out', str(root/'one')]
            first, audit = quiet_run(mod.parser().parse_args(command))
            self.assertEqual(audit['label_policy']['ind_test']['ranking_label_field'], 'prop_annotations')
            values = np.load(pp)
            np.save(pp, values[[2, 0, 1]])
            ids.write_text('C\nA\nB\n')
            command[-1] = str(root/'two')
            second, _ = quiet_run(mod.parser().parse_args(command))
            fields = ['family', 'bin', 'standard_protein_fmax', 'stage1_hist__fmax_micro',
                      'standard_micro_ap', 'micro_recall_at_100']
            pd.testing.assert_frame_equal(first[fields], second[fields])

    def test_true_global_ties_report_bounds(self):
        p = np.zeros((1, 4), dtype=np.float32)
        canon = ['GO:0000001', 'GO:0000002', 'GO:0000003', 'GO:0008150']
        families = {'overall': {'rank_masks': {'all': np.ones(4, bool)}, 'counts': np.ones(4)}}
        rows = mod.positive_records(p, [[2]], ['A'], canon, np.arange(4), np.ones(4),
                                    'GO:0008150', families, 'expert', 'x', {}, [1, 3])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['rank'], 3)
        self.assertEqual(rows[0]['rank_optimistic'], 1)
        self.assertEqual(rows[0]['rank_pessimistic'], 3)
        self.assertEqual(rows[0]['hit_at_1_lower'], 0)
        self.assertEqual(rows[0]['hit_at_1_upper'], 1)

    def test_invalid_manifest_prediction_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, train, unified, inp, gos, original = fixture_module.fixture(root)
            np.save(inp/'expert.npy', np.zeros((3, 8), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'prediction differs'):
                quiet_run(options(manifest, train, unified, root/'output'))

    def test_no_rank_positives_keeps_readable_output_headers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, train, unified, inp, gos, original = fixture_module.fixture(root)
            metadata_path = Path(original['metadata'])
            metadata = pickle.loads(metadata_path.read_bytes())
            for case in metadata.values():
                case['biological_process']['annotations'] = [[], [], []]
                case['biological_process']['prop_annotations'] = [[], [], []]
            metadata_path.write_bytes(pickle.dumps(metadata))
            original['metadata_sha256'] = mod.common.sha256(metadata_path)
            manifest.write_text(json.dumps(original))
            result, _ = quiet_run(options(manifest, train, unified, root/'output'))
            self.assertTrue((result.n_metric_positive_columns == 0).all())
            self.assertTrue((result[result.family != 'stage1_raw_classic'].n_rank_positive_pairs == 0).all())
            for filename in ['positive_global_ranks.tsv', 'paired_rank_bootstrap_deltas.tsv']:
                frame = pd.read_csv(root/'output'/filename, sep='\t')
                self.assertTrue(frame.empty)
                self.assertIn('model', frame.columns)


if __name__ == '__main__':
    unittest.main()
