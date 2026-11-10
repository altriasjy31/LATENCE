"""Four-view, all-metric integration tests on frozen synthetic exports.

Run after overlaying this additive package onto the existing LATENCE project.
No model, GPU or large dense fixture is required.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load_file('atlas_full_runner_v062_under_test', ROOT / 'scripts/dark_weak/eval_atlas_v062.py')
fixture_module = load_file('atlas_full_runner_v062_fixture', ROOT / 'tests/test_atlas_runner_v060.py')


def stage1_fixture(root):
    """Three distinct branches, including abstention and a large tied block."""
    metadata_path, config_path, unified, metadata, config = fixture_module.fixture(root)
    for namespace, records in config.items():
        records.pop('nbs')
        for model, record in records.items():
            values = np.full((4, 8), .1 if model != 'modelout' else 0, np.float32)
            values[:, 6] = .99  # Root must never occupy a direct Top-K slot.
            if model != 'expert':
                values[0, 2] = .9
                values[1, [0, 7]] = .9  # Two classifier aliases, one candidate.
                values[2, 5] = .9
                values[3, 4] = .9
            np.save(record['probability'], values)
    config_path.write_text(json.dumps(config), encoding='utf-8')
    return metadata_path, config_path, unified, metadata, config


def arguments(metadata, unified, out):
    return ['--metadata', str(metadata), '--unified-root', str(unified),
            '--namespaces', 'bp', '--classification', 'all',
            '--max-metric-cells', '1', '--metric-chunk-rows', '1',
            '--skip-stage1-official', '--bootstrap', '0', '--ks', '1', '3',
            '--out', str(out)]


def quiet_run(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.run(args)


def native_manifest(directory, records):
    """Bind native manifests to the same authoritative fixture axes/probabilities."""
    directory.mkdir()
    ids = directory / 'protein_ids.txt'
    gos = directory / 'go_ids.txt'
    shutil.copyfile(records['backbone']['protein_ids'], ids)
    shutil.copyfile(records['backbone']['go_ids'], gos)
    channels = {'backbone_base': 'backbone', 'expert': 'expert', 'modelout': 'modelout'}
    outputs = {}
    for channel, model in channels.items():
        target = directory / f'{channel}.prob.f32.npy'
        shutil.copyfile(records[model]['probability'], target)
        outputs[channel] = {'path': str(target.resolve()), 'sha256': mod.common.sha256(target),
                            'storage_dtype': 'float32'}
    record = {'model_family': 'LATENCE Stage1 DETR', 'task': 'bp', 'mode': 'atlas_inference',
              'num_proteins': 4, 'num_go': 8, 'channels': list(channels),
              'forward_uses_eval_labels': False, 'output_storage_dtype': 'float32',
              'expected_protein_ids_sha256': mod.common.sha256(ids),
              'go_ids_sha256': mod.common.sha256(gos), 'output_files': outputs,
              'modelout_prediction_key': 'fixture::modelout'}
    path = directory / 'prediction_manifest.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    return path


class FullAtlasRunnerTests(unittest.TestCase):
    def test_parser_defaults_to_all_four_classifications(self):
        args = mod.parser().parse_args(['--out', 'new-result'])
        self.assertEqual(args.classification, 'all')
        self.assertGreater(args.metric_chunk_rows, 0)

    def test_all_views_complete_despite_one_cell_legacy_limit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata, config_path, unified, _, config = stage1_fixture(root)
            source_hashes = {record['probability']: mod.common.sha256(record['probability'])
                             for record in config['bp'].values()}
            out = root / 'complete'
            args = mod.parser().parse_args(arguments(metadata, unified, out) +
                ['--prediction-config', str(config_path), '--train-metadata', str(metadata)])
            inventory, manifest = quiet_run(args)
            self.assertEqual(len(inventory), 4)
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['classification'], 'all')
            classification = pd.read_csv(out / 'classification_metrics.tsv', sep='\t')
            self.assertEqual(len(classification), 12)
            self.assertEqual(set(classification.evidence_layer), set(mod.SPLITS))
            self.assertEqual(set(classification.model), {'backbone', 'expert', 'modelout'})
            self.assertEqual(set(classification.label_field), {'prop_annotations'})
            self.assertTrue({'standard_protein_fmax', 'standard_micro_ap',
                'standard_micro_pr_auc', 'standard_micro_fmax_exact',
                'stage1_hist__fmax_micro', 'stage1_hist__fmax_sample'}.issubset(classification.columns))
            self.assertFalse(classification[['standard_protein_fmax', 'standard_micro_ap',
                'standard_micro_pr_auc', 'stage1_hist__fmax_micro']].isna().any().any())
            views = mod.read_views(mod.pickle.loads(metadata.read_bytes()), 'bp', 8)
            for row in classification.itertuples():
                self.assertEqual(row.standard_num_proteins, len(views[row.evidence_layer]['ids']))
                self.assertEqual(row.standard_num_go, 8)
                self.assertEqual(row.standard_num_positive_labels,
                    sum(len(cols) for cols in views[row.evidence_layer]['propagated'].values()))
            direct = pd.read_csv(out / 'direct_topk_metrics.tsv', sep='\t')
            self.assertEqual(set(direct.evidence_layer), set(mod.SPLITS))
            all_rows = direct[direct.frequency_bin.eq('all')]
            self.assertEqual(len(all_rows), 24)  # Four views x three models x two K.
            self.assertEqual(set(all_rows.model), {'backbone', 'expert', 'modelout'})
            self.check_direct_rank_parity(direct, out)
            self.check_actual_denominators(all_rows)
            self.assertEqual(source_hashes,
                {path: mod.common.sha256(path) for path in source_hashes})
            result_hash = mod.common.sha256(out / 'classification_metrics.tsv')
            with self.assertRaises(FileExistsError):
                quiet_run(args)
            self.assertEqual(mod.common.sha256(out / 'classification_metrics.tsv'), result_hash)

    @staticmethod
    def check_direct_rank_parity(direct, out):
        ranks = pd.read_csv(out / 'atlas_metrics.tsv', sep='\t')
        index = ranks.set_index(['evidence_layer', 'model', 'frequency_bin'])
        for row in direct.to_dict('records'):
            old = index.loc[(row['evidence_layer'], row['model'], row['frequency_bin'])]
            k = row['k']
            if row['n_pairs']:
                np.testing.assert_allclose(row['observed_micro_recall'], old[f'micro_recall_at_{k}_conservative'],
                                           rtol=0, atol=1e-12)
            if row['n_hits'] != old[f'n_hits_at_{k}_conservative']:
                raise AssertionError(f'Direct hit accounting changed: {row}')

    @staticmethod
    def check_actual_denominators(all_rows):
        modelout = all_rows[all_rows.model.eq('modelout') & all_rows.evidence_layer.eq('future_union')]
        # This model has exactly one positive-score nonroot candidate per protein,
        # so increasing K from one to three does not create six zero-score outputs.
        for row in modelout.to_dict('records'):
            if row['n_predictions'] != 3 or row['n_hits'] != 3:
                raise AssertionError(f'Zero scores or alias columns entered direct candidates: {row}')
            np.testing.assert_allclose(row['observed_micro_precision'], 1, rtol=0, atol=1e-12)
        expert = all_rows[all_rows.model.eq('expert')]
        if not (expert.n_predictions == 0).all():
            raise AssertionError('A tied block crossing K must be withheld in full')

    def test_config_free_native_input_runs_all_four_views(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata, _, unified, _, config = stage1_fixture(root)
            directory = root / 'export'
            source_manifest = native_manifest(directory, config['bp'])
            before = {str(path): mod.common.sha256(path) for path in directory.iterdir()}
            out = root / 'native-complete'
            command = arguments(metadata, unified, out) + ['--stage1-export-dir', f'bp={directory}']
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main(command)
            classification = pd.read_csv(out / 'classification_metrics.tsv', sep='\t')
            self.assertEqual(len(classification), 12)
            self.assertEqual(set(classification.evidence_layer), set(mod.SPLITS))
            sidecar = root / 'native-complete_inputs/prediction_inputs.json'
            records = json.loads(sidecar.read_text())
            self.assertEqual(records['bp']['modelout']['branch'], 'fixture::modelout')
            self.assertEqual(records['bp']['expert']['source_export_manifest_sha256'],
                             before[str(source_manifest)])
            self.assertEqual(before, {str(path): mod.common.sha256(path) for path in directory.iterdir()})
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                mod.main(command)


if __name__ == '__main__':
    unittest.main()
