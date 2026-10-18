import csv
import hashlib
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ORIGINAL_NATIVE = HERE / 'nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py'
if not ORIGINAL_NATIVE.is_file():
    ORIGINAL_NATIVE = HERE.parent / 'full_task_metrics_v085.py'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class CompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        d = self.root / 'scripts/nbs'
        d.mkdir(parents=True)
        (self.root / 'nbs_models/nbs_protein_go/nbs_pg').mkdir(parents=True)
        shutil.copy2(HERE / 'scripts/nbs/eval_nbs_v085_supplementary.py', d / 'eval_nbs_v085_supplementary.py')
        native = self.root / 'nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py'
        shutil.copy2(ORIGINAL_NATIVE, native)
        (native.parent / '__init__.py').touch()
        spec = importlib.util.spec_from_file_location('native_metric', native)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.mod = mod
        width = 2903
        self.meta_ids = ['P_A', 'P_B', 'P_C']
        self.src_ids = ['P_C', 'P_A', 'P_B']
        ys = [[1, 5], [13], [20, 25]]
        y = np.zeros((3, width), np.uint8)
        for i, terms in enumerate(ys):
            y[i, terms] = 1
        self.meta_file = self.root / 'meta.pkl'
        with self.meta_file.open('wb') as f:
            pickle.dump({'ind_test': {'cellular_component': {'proteins': self.meta_ids, 'prop_annotations': ys}}}, f)
        rng = np.random.default_rng(23)
        pG = rng.random((3, width), dtype=np.float32) * 0.4
        pB = rng.random((3, width), dtype=np.float32) * 0.3
        for idx, term in enumerate((1, 13, 20)):
            pG[idx, term] = 0.95
        for idx, term in enumerate((5, 13, 25)):
            pB[idx, term] = 0.75
        reord = [self.meta_ids.index(pid) for pid in self.src_ids]
        self.nbs_file = self.root / 'nbs.npy'
        self.base_file = self.root / 'base.npy'
        np.save(self.nbs_file, pG[reord]); np.save(self.base_file, pB[reord])
        self.base_fmax = self.mod.compute_standard_metrics(y, pB)['standard_protein_fmax']
        self.g_fmax = self.mod.compute_standard_metrics(y, pG)['standard_protein_fmax']
        self.ids_file = self.root / 'protein_ids.txt'
        self.ids_file.write_text('\n'.join(self.src_ids) + '\n')
        self.go_registry = self.root / 'go_registry.tsv'
        with self.go_registry.open('w', encoding='utf-8') as f:
            f.write('go_idx\tinput_go_id\n')
            for i in range(width):
                f.write(f'{i}\tGO:{i:07d}\n')
        self.input_file = self.root / 'ind_test_input_manifest.json'
        self.input_file.write_text('{}')
        self.ckpt_file = self.root / 'ckpt.pt'
        self.ckpt_file.write_bytes(b'fixture-only; no checkpoint loading')
        self.pred_file = self.root / 'nbs_full_task_prediction_manifest.json'
        self.pred_file.write_text(json.dumps({
            'runner_version': '0.8.8', 'prediction_space': 'complete_task_classifier_columns',
            'num_task_go': width, 'uses_expert_probability_in_nbs_forward': False,
            'output_fusion': 'none', 'input_manifest_sha256': digest(self.input_file),
            'output_probability_sha256': digest(self.nbs_file),
            'checkpoint_sha256': digest(self.ckpt_file),
        }))
        self.report_file = self.root / 'nbs_w2s_comparison.json'
        self.report_file.write_text(json.dumps({
            'schema_version': 3, 'evaluation_version': '0.8.8',
            'evaluation_scope': 'standard_protein_fmax_only', 'task': 'cc',
            'num_samples': 3, 'num_classes': width,
            'standardized_metric_contract': {'implementation_sha256': digest(native), 'threshold_step': 0.001},
            'methods': {
                'NBS_final': {'standardized': {'standard_protein_fmax': self.g_fmax}},
                'backbone_base': {'standardized': {'standard_protein_fmax': self.base_fmax}},
            },
            'prediction_provenance': {
                'probability_path': str(self.nbs_file), 'probability_sha256': digest(self.nbs_file),
                'backbone_probability_path': str(self.base_file), 'backbone_probability_sha256': digest(self.base_file),
                'prediction_manifest_path': str(self.pred_file), 'prediction_manifest_sha256': digest(self.pred_file),
                'input_manifest_path': str(self.input_file), 'input_manifest_sha256': digest(self.input_file),
                'protein_ids_path': str(self.ids_file), 'protein_ids_file_sha256': digest(self.ids_file),
                'input_go_registry_path': str(self.go_registry),
                'input_go_registry_sha256': digest(self.go_registry),
                'checkpoint_path': str(self.ckpt_file), 'checkpoint_sha256': digest(self.ckpt_file),
            },
            'metadata_provenance': {'path': str(self.meta_file), 'sha256': digest(self.meta_file)},
        }))

    def run_script(self):
        return subprocess.run([sys.executable, str(self.root / 'scripts/nbs/eval_nbs_v085_supplementary.py'),
                               '--comparison', str(self.report_file)], capture_output=True, text=True)

    def test_01_original_native_v088_byte_identical(self):
        self.assertEqual(digest(HERE / 'scripts/nbs/eval_nbs_full_task_v088.py'), '524dff8942070a4f615b743059e84cc5e62859da6c04fe886abae8d9d0748145')

    def test_02_no_acceptance_mutation(self):
        before = digest(self.report_file)
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(digest(self.report_file), before)
        supplement = json.loads((self.root / 'nbs_supplementary_v085.json').read_text())
        self.assertIn('standard_micro_pr_auc', supplement['methods']['NBS_final'])
        self.assertIn('standard_micro_ap', supplement['methods']['NBS_final'])
        self.assertEqual(supplement['methods']['NBS_final']['standard_protein_fmax'], self.g_fmax)

    def test_03_incompatible_schema_fails(self):
        obj = json.loads(self.report_file.read_text())
        obj['schema_version'] = 4
        obj['evaluation_scope'] = 'standard_protein_fmax_and_v085_pr_auc'
        self.report_file.write_text(json.dumps(obj))
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ORIGINAL', result.stderr)

    def test_04_input_mutation_fails(self):
        self.ids_file.write_text('P_C\nP_B\nP_A\n')
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256 mismatch', result.stderr)

    def test_05_native_fmax_mismatch_fails(self):
        obj = json.loads(self.report_file.read_text())
        obj['methods']['NBS_final']['standardized']['standard_protein_fmax'] += 1
        self.report_file.write_text(json.dumps(obj))
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Fmax disagrees', result.stderr)

    def test_06_prediction_manifest_mutation_fails(self):
        self.pred_file.write_text('{}')
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256 mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main()
