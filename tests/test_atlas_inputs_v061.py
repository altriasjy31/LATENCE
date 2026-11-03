"""Native export manifest bridge and real v060 roundtrip tests; no GPU."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get('LATENCE_TEST_PROJECT_ROOT', ROOT))
spec = importlib.util.spec_from_file_location('atlas_inputs_v061', ROOT/'scripts/dark_weak/eval_atlas_v061.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def native_export(directory):
    directory.mkdir(parents=True)
    (directory/'protein_ids.txt').write_text('A\nB\nC\nD\n')
    (directory/'go_ids.txt').write_text('GO:0000001\nGO:0000002\nGO:0008150\n')
    outputs = {}
    values = np.array([[.9, .1, .99], [.1, .9, .99], [.9, .1, .99], [.9, .1, .99]], np.float32)
    for channel in mod.CHANNELS:
        p = directory/f'{channel}.prob.f32.npy'
        np.save(p, values)
        outputs[channel] = {'path': str(p.resolve()), 'sha256': mod.sha256(p), 'storage_dtype': 'float32'}
    manifest = {'model_family': 'LATENCE Stage1 DETR', 'task': 'bp', 'mode': 'atlas_inference',
        'num_proteins': 4, 'num_go': 3, 'channels': list(mod.CHANNELS),
        'checkpoint': '/source/frozen.pt', 'checkpoint_sha256': 'a'*64,
        'input_metadata': '/source/label_free.pkl',
        'expected_protein_ids_sha256': mod.sha256(directory/'protein_ids.txt'),
        'go_ids_sha256': mod.sha256(directory/'go_ids.txt'),
        'modelout_prediction_key': 'modelout::mix_expert_base_anchor::decoderprob::expert',
        'output_storage_dtype': 'float32', 'forward_uses_eval_labels': False, 'output_files': outputs}
    save_manifest(directory, manifest)
    return manifest


def save_manifest(directory, manifest):
    (directory/'prediction_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def atlas_data(root):
    unified = root/'unified'
    folder = unified/'labels/bp'
    folder.mkdir(parents=True)
    gos = ['GO:0000001', 'GO:0000002', 'GO:0008150']
    (folder/'go_ids.txt').write_text('\n'.join(gos)+'\n')
    (folder/'vocabulary_mapping.tsv').write_text(
        'model_column\tmodel_go_id\tcanonical_go_id_2022\n' +
        ''.join(f'{i}\t{go}\t{go}\n' for i, go in enumerate(gos)))
    direct = {'dark_weak_test_exp_direct': {'A': [0]},
        'dark_weak_test_ipr2go_direct': {'B': [1], 'C': [0]},
        'dark_weak_test_direct': {'A': [0], 'B': [1], 'C': [0]},
        'dark_weak_test_2022_ipr2go_direct': {'A': [1], 'D': [0]}}
    metadata = {}
    for split, entries in direct.items():
        metadata[split] = {'biological_process': {
            'proteins': list(entries), 'annotations': list(entries.values()),
            'prop_annotations': [list(x) + [2] for x in entries.values()]}}
    mp = root/'metadata.pkl'
    mp.write_bytes(pickle.dumps(metadata))
    return mp, unified


class NativeBridgeTests(unittest.TestCase):
    def test_valid_records_keep_native_source_binding(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)/'export'; original = native_export(directory)
            before = mod.sha256(directory/'prediction_manifest.json')
            records, provenance = mod.build_stage1_records('bp', directory)
            self.assertEqual(set(records), {'backbone', 'expert', 'modelout'})
            self.assertEqual(records['modelout']['branch'], original['modelout_prediction_key'])
            self.assertEqual(provenance['native_export_manifest_sha256'], before)
            self.assertEqual(mod.sha256(directory/'prediction_manifest.json'), before)
            self.assertEqual(provenance['num_proteins'], 4)

    def test_shape_namespace_and_id_hash_validation(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)/'export'; manifest = native_export(directory)
            with self.assertRaisesRegex(ValueError, 'namespace'):
                mod.build_stage1_records('mf', directory)
            manifest['num_go'] = 4; save_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'num_proteins/num_go'):
                mod.build_stage1_records('bp', directory)
            manifest['num_go'] = 3; save_manifest(directory, manifest)
            (directory/'protein_ids.txt').write_text('B\nA\nC\nD\n')
            with self.assertRaisesRegex(ValueError, 'expected_protein_ids_sha256'):
                mod.build_stage1_records('bp', directory)

    def test_probability_hash_shape_and_range_validation(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)/'export'; manifest = native_export(directory)
            p = Path(manifest['output_files']['expert']['path'])
            np.save(p, np.zeros((2, 3), np.float32))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                mod.build_stage1_records('bp', directory)
            manifest['output_files']['expert']['sha256'] = mod.sha256(p); save_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'shape/dtype'):
                mod.build_stage1_records('bp', directory)
            np.save(p, np.full((4, 3), np.nan, np.float32))
            manifest['output_files']['expert']['sha256'] = mod.sha256(p); save_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'finite'):
                mod.build_stage1_records('bp', directory)

    def test_manifest_contract_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)/'export'; manifest = native_export(directory)
            manifest['forward_uses_eval_labels'] = True; save_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'forward_uses_eval_labels'):
                mod.build_stage1_records('bp', directory)
            manifest['forward_uses_eval_labels'] = False
            manifest['channels'].remove('expert'); save_manifest(directory, manifest)
            with self.assertRaisesRegex(ValueError, 'three|channels'):
                mod.build_stage1_records('bp', directory)
            manifest['channels'] = list(mod.CHANNELS); save_manifest(directory, manifest)
            (directory/'protein_ids.txt').write_text('A\nA\nC\nD\n')
            with self.assertRaisesRegex(ValueError, 'unique'):
                mod.build_stage1_records('bp', directory)

    def test_relocated_native_paths_still_require_probability_hash(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)/'export'; manifest = native_export(directory)
            for channel, info in manifest['output_files'].items():
                info['path'] = '/missing/original/' + Path(info['path']).name
            save_manifest(directory, manifest)
            records, _ = mod.build_stage1_records('bp', directory)
            self.assertEqual(records['backbone']['source_path_resolution'], 'relocated_basename_with_required_hash')
            self.assertEqual(Path(records['backbone']['probability']).parent, directory)

    def test_generated_inputs_protected_and_all_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); directory = root/'export'; native_export(directory)
            before = {p.name: mod.sha256(p) for p in directory.iterdir()}
            path = mod.prepare_generated_inputs([f'bp={directory}'], root/'result')
            self.assertEqual(path, root/'result_inputs/prediction_inputs.json')
            saved = json.loads(path.read_text())
            self.assertEqual(saved['bp']['expert']['protein_ids_sha256'], mod.sha256(directory/'protein_ids.txt'))
            self.assertEqual(before, {p.name: mod.sha256(p) for p in directory.iterdir()})
            self.assertTrue((path.parent/'source_provenance.json').is_file())
            with self.assertRaises(FileExistsError):
                mod.prepare_generated_inputs([f'bp={directory}'], root/'result')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                mod.prepare_generated_inputs([f'bp={directory}', f'bp={directory}'], root/'other')
            self.assertFalse((root/'other_inputs').exists())


@unittest.skipUnless((PROJECT/'scripts/dark_weak/eval_atlas_v060.py').is_file(),
                     'v060 project unavailable; set LATENCE_TEST_PROJECT_ROOT for end-to-end tests')
class RealV060RoundtripTests(unittest.TestCase):
    def command(self, metadata, unified, out):
        return ['--metadata', str(metadata), '--unified-root', str(unified), '--namespaces', 'bp',
                '--classification', 'none', '--bootstrap', '0', '--ks', '1', '2', '--out', str(out)]

    def test_native_export_runs_original_v060_without_handmade_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); directory = root/'export'; native_export(directory)
            metadata, unified = atlas_data(root)
            before = mod.sha256(metadata)
            with contextlib.redirect_stdout(io.StringIO()):
                inventory, audit = mod.main(self.command(metadata, unified, root/'result') +
                    ['--stage1-export-dir', f'bp={directory}'], project_root=PROJECT)
            self.assertEqual(len(inventory), 4)
            self.assertEqual(audit['schema'], 'latence-functional-observability-atlas-evaluation-v060')
            self.assertEqual(mod.sha256(metadata), before)
            config = json.loads((root/'result_inputs/prediction_inputs.json').read_text())
            self.assertIn('source_export_manifest_sha256', config['bp']['backbone'])
            self.assertTrue((root/'result/atlas_metrics.tsv').is_file())

    def test_complete_manifest_still_delegates_without_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); directory = root/'export'; native_export(directory)
            metadata, unified = atlas_data(root)
            records, _ = mod.build_stage1_records('bp', directory)
            complete = root/'complete.json'
            complete.write_text(json.dumps({'namespace': 'bp', 'input_files': records}))
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main(self.command(metadata, unified, root/'result') +
                         ['--complete-manifest', f'bp={complete}'], project_root=PROJECT)
            self.assertFalse((root/'result_inputs').exists())

    def test_inventory_prints_missing_and_evaluation_never_pads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); directory = root/'export'; manifest = native_export(directory)
            metadata, unified = atlas_data(root)
            (directory/'protein_ids.txt').write_text('A\nB\nC\n')
            manifest['num_proteins'] = 3
            manifest['expected_protein_ids_sha256'] = mod.sha256(directory/'protein_ids.txt')
            for info in manifest['output_files'].values():
                p = Path(info['path']); values = np.load(p); np.save(p, values[:3])
                info['sha256'] = mod.sha256(p)
            save_manifest(directory, manifest)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                mod.main(self.command(metadata, unified, root/'inventory') +
                    ['--stage1-export-dir', f'bp={directory}', '--inventory-only'], project_root=PROJECT)
            self.assertIn('Coverage is incomplete', output.getvalue())
            self.assertIn('required_proteins', output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'cannot fill missing rows'):
                mod.main(self.command(metadata, unified, root/'evaluation') +
                    ['--stage1-export-dir', f'bp={directory}'], project_root=PROJECT)
            self.assertFalse((root/'evaluation').exists())


if __name__ == '__main__':
    unittest.main()
