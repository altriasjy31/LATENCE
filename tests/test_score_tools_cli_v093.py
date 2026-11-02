"""End-to-end CPU fixtures using the real input reader and metric kernels.

The disposable project reduces NUM_CLASSES['bp'] to six only in its copied
reader, while retaining the production ID reordering, gold parser, metrics,
CSR frequency reconstruction, hashes and command-line control flow.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
REL_METRIC = Path('nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py')
REL_READER = Path('scripts/nbs/diagnose_nbs_knn_v090.py')
REL_LEGACY = Path('scripts/nbs/eval_nbs_ind_test_predictions.py')
RUNTIMES = (PACKAGE, PACKAGE.parent/'source/latence_nbs_v090_single12800')
RUNTIME = next((p for p in RUNTIMES if all((p/name).is_file()
                    for name in (REL_METRIC, REL_READER, REL_LEGACY))), None)
if RUNTIME is None:
    raise FileNotFoundError('CLI tests require the installed LATENCE metric and diagnostic input-reader files')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ScoreToolsCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nbs_score_cli_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('eval_nbs_score_tools_v093.py', 'nbs_score_postprocess_v093.py'):
            dest = self.root/'scripts/nbs'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PACKAGE/'scripts/nbs'/name, dest)
        for relative in (REL_METRIC, REL_READER, REL_LEGACY):
            target = self.root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RUNTIME/relative, target)
        with (self.root/REL_READER).open('a', encoding='utf-8') as stream:
            stream.write("\n# Synthetic fixture width only; production reader logic unchanged.\nNUM_CLASSES['bp'] = 6\n")
        self.script = self.root/'scripts/nbs/eval_nbs_score_tools_v093.py'
        self.inputs, self.work = self.root/'inputs', self.root/'work'
        self.inputs.mkdir(); self.work.mkdir()
        write_json(self.inputs/'ind_test_input_manifest.json', {
            'cache_signature': {'representation_manifest_sha256': 'fixture-representation'}})
        (self.inputs/'protein_ids.txt').write_text('proteinA\nproteinB\nproteinC\n')
        np.save(self.inputs/'ind_test_repr.f16.npy', np.array([[1, 0], [0, 1], [1, 1]], np.float16))
        self.b = np.array([[.9, .2, .3, .1, .8, .1],
                           [.3, .85, .1, .75, .2, .4],
                           [.25, .4, .8, .2, .3, .9]], np.float16)
        np.save(self.inputs/'backbone_ind_test_prob.f16.npy', self.b)
        self.g = np.array([[.7, .95, .2, .3, .55, .2],
                           [.5, .6, .1, .35, .3, .2],
                           [.1, .3, .4, .5, .8, .4]], np.float32)
        self.y = np.array([[1, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0, 0]], bool)
        self.metadata = self.root/'metadata.pkl'
        with self.metadata.open('wb') as stream:
            pickle.dump({'ind_test': {'bp': {
                'proteins': ['proteinC', 'proteinA', 'proteinB'],
                'prop_annotations': [[2], [0, 4], [1, 3]]}}}, stream)

        data = self.root/'training_data'
        registry = data/'features/protein_registry.csv'
        registry.parent.mkdir(parents=True)
        # Registry global IDs are intentionally different from core role rows.
        with registry.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['protein_idx', 'role', 'role_row_idx'])
            writer.writeheader()
            for index, role, local in [(0, 'weak', 0), (1, 'core', 2), (2, 'core', 0),
                                       (3, 'weak', 1), (4, 'core', 1), (5, 'core', 3), (6, 'core', 4)]:
                writer.writerow({'protein_idx': index, 'role': role, 'role_row_idx': local})
        memberships = [[], [0, 1, 1], [2], [], [0, 4, 4], [1, 3, 3], []]
        ptr = np.r_[0, np.cumsum([len(r) for r in memberships])].astype(np.int64)
        go = np.concatenate([np.asarray(r, np.int32) for r in memberships])
        indices = data/'indices'; indices.mkdir()
        ptr_path, go_path = indices/'gold.indptr.npy', indices/'gold.go_idx.npy'
        np.save(ptr_path, ptr); np.save(go_path, go)
        inverted = indices/'manifest.json'
        write_json(inverted, {'indices': {'gold': {'protein_major': {
            'num_go': 6, 'indptr': ptr_path.name, 'go_idx': go_path.name}}}})
        # Core role order before filtering is [2,4,1,5,6]. Empty core 6 is
        # removed; seed 8080 holds out global ID 1, leaving [2,4,5]. Duplicate
        # memberships in rows 4/5 must each count only once.
        self.training = np.array([2, 4, 5], np.int64)
        self.counts = np.array([1, 1, 1, 1, 1, 0], np.int64)
        self.config = self.work/'resolved_config.json'
        write_json(self.config, {'task': 'bp', 'full_task': {
            'label_split': 'all_seen', 'holdout_core_count': 1, 'holdout_seed': 8080},
            'data': {'root': 'training_data',
                     'go_protein_inverted_index_manifest': 'indices/manifest.json'}})
        self.contract = {'representation_manifest_sha256': 'fixture-representation',
            'registry_sha256': sha(registry), 'gold_indptr_sha256': sha(ptr_path),
            'gold_go_idx_sha256': sha(go_path),
            'go_protein_inverted_index_manifest_sha256': sha(inverted)}
        self.folder = self.work/'eval_step2880/full'; self.folder.mkdir(parents=True)
        self.probability = self.folder/'nbs_ind_test_prob.f32.npy'
        np.save(self.probability, self.g)
        self.manifest = self.folder/'nbs_full_task_prediction_manifest.json'
        write_json(self.manifest, {'checkpoint_sha256': 'fixture-fixed-checkpoint',
            'output_probability_sha256': sha(self.probability),
            'input_manifest_sha256': sha(self.inputs/'ind_test_input_manifest.json'),
            'contract': {'data': self.contract}})
        helper = load_module('_score_cli_fixture_helper', self.root/REL_READER)
        self.expected = {'backbone_base': {
                **helper.protein_fmax(self.y, self.b.astype(np.float32)),
                'micro_ap': helper.grouped_ap(self.y, self.b.astype(np.float32))},
            'NBS_final': {**helper.protein_fmax(self.y, self.g),
                'micro_ap': helper.grouped_ap(self.y, self.g)}}
        self.summary = self.folder/'metrics_v092/summary.json'
        write_json(self.summary, {'schema': 'nbs-v092-extended-evaluation-v1',
            'complete': True, 'task': 'bp', 'checkpoint_epoch': 60, 'checkpoint_step': 2880,
            'num_proteins': 3, 'num_GO_columns': 6, 'test_positive_pairs': int(self.y.sum()),
            'checkpoint_sha256': 'fixture-fixed-checkpoint',
            'probability_sha256': sha(self.probability),
            'prediction_manifest_sha256': sha(self.manifest),
            'input_manifest_sha256': sha(self.inputs/'ind_test_input_manifest.json'),
            'num_training_core_proteins': len(self.training),
            'training_core_ids_sha256': array_sha(self.training),
            'core_frequency_sha256': array_sha(self.counts), 'metrics': self.expected})
        self.old_files = {p: p.read_bytes() for p in (
            self.probability, self.config, self.summary, self.manifest, registry, ptr_path,
            go_path, inverted, self.metadata, self.inputs/'backbone_ind_test_prob.f16.npy')}

    def command(self, mode='all'):
        return [sys.executable, str(self.script), mode, '--epoch', '60',
                '--work-dir', str(self.work), '--input-dir', str(self.inputs),
                '--metadata-file', str(self.metadata)]

    def run_cli(self, mode='all', success=True):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run(self.command(mode), cwd=self.root, env=env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def output(self, mode='all'):
        return self.work/'score_postprocess_epoch60'/mode

    def test_reader_reorders_metadata_ids_and_reconstructs_role_local_holdout_counts(self):
        cli = load_module('_score_cli_fixture_module', self.script)
        args = cli.parser().parse_args(self.command()[2:])
        y, b, g, counts, summary, _, _ = cli.load_inputs(args)
        np.testing.assert_array_equal(y, self.y)
        np.testing.assert_array_equal(b, self.b.astype(np.float32))
        np.testing.assert_array_equal(g, self.g)
        np.testing.assert_array_equal(counts, self.counts)
        self.assertEqual(summary['training_core_ids_sha256'], array_sha(self.training))

    def test_all_modes_run_cpu_reproduce_raw_scores_preserve_inputs_and_resume(self):
        result = self.run_cli('all')
        out = self.output('all')
        report = json.loads((out/'report.json').read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(len(report['methods']), 13)
        self.assertFalse(report['acceptance_protocol_changed'])
        for name, original in [('B', 'backbone_base'), ('G', 'NBS_final')]:
            for key in ('fmax', 'micro_ap'):
                self.assertAlmostEqual(report['methods'][name][key], self.expected[original][key], places=12)
        self.assertTrue((out/'methods.csv').is_file())
        self.assertTrue((out/'frequency_strata.csv').is_file())
        ap_files = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in out.glob('*_per_go_ap.npy')}
        self.assertEqual(len(ap_files), 13)
        for p in ap_files:
            self.assertEqual(np.load(p).shape, (6,))
        self.run_cli('all')
        for p, (stamp, content) in ap_files.items():
            self.assertEqual(p.stat().st_mtime_ns, stamp)
            self.assertEqual(p.read_bytes(), content)
        for p, content in self.old_files.items():
            self.assertEqual(p.read_bytes(), content)
        self.assertIn('[score-tools] completed:', result.stdout)

    def test_changed_exported_probability_is_rejected(self):
        np.save(self.probability, self.g*.5)
        result = self.run_cli('mix', success=False)
        self.assertIn('Prediction matrix/manifest differs', result.stderr)

    def test_existing_report_identity_rejected_even_without_inputs_json(self):
        self.run_cli('mix')
        out = self.output('mix'); report_path = out/'report.json'
        report = json.loads(report_path.read_text())
        report['identity']['epoch'] = 5
        write_json(report_path, report)
        (out/'inputs.json').unlink()
        before = report_path.read_bytes()
        result = self.run_cli('mix', success=False)
        self.assertIn('Existing report belongs to different inputs/code', result.stderr)
        self.assertEqual(report_path.read_bytes(), before)

    def test_changed_metadata_prevents_reusing_old_results(self):
        self.run_cli('mix')
        with self.metadata.open('wb') as stream:
            pickle.dump({'ind_test': {'bp': {'proteins': ['proteinC', 'proteinA', 'proteinB'],
                'prop_annotations': [[0], [0, 4], [1, 3]]}}}, stream)
        report = self.output('mix')/'report.json'; before = report.read_bytes()
        result = self.run_cli('mix', success=False)
        self.assertIn('Existing diagnostic belongs to different inputs/code', result.stderr)
        self.assertEqual(report.read_bytes(), before)

    def test_missing_per_go_output_is_rejected_on_resume(self):
        self.run_cli('mix')
        (self.output('mix')/'B_per_go_ap.npy').unlink()
        result = self.run_cli('mix', success=False)
        self.assertIn('missing per-GO AP for B', result.stderr)

    def test_metric_mismatch_rejected_before_accepting_transformed_methods(self):
        value = json.loads(self.summary.read_text())
        value['metrics']['NBS_final']['micro_ap'] = 0
        write_json(self.summary, value)
        result = self.run_cli('all', success=False)
        self.assertIn('fails source-metric reproduction', result.stderr)
        report = json.loads((self.output('all')/'report.json').read_text())
        self.assertFalse(report['complete'])
        self.assertEqual(list(report['methods']), ['B'])


if __name__ == '__main__':
    unittest.main()
