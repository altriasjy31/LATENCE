"""Regression for task aliases whose original IDs retain physical OWL rows."""
from __future__ import annotations

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
NEW = PACKAGE / 'nbs_models/nbs_protein_go/nbs_pg/go_registry_v093.py'
OLD = NEW.with_name('go_registry_v090.py')
if not OLD.is_file():
    OLD = (PACKAGE.parents[1] / 'fusion_v093/source/latence_nbs_v090_single12800/'
           'nbs_models/nbs_protein_go/nbs_pg/go_registry_v090.py')
ALIAS, CANONICAL, OTHER = 'GO:0039528', 'GO:0000001', 'GO:0000002'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding='utf-8')


def manifest_fixture(root):
    folder = root / 'modules'
    folder.mkdir()
    shutil.copyfile(OLD, folder / OLD.name)
    shutil.copyfile(NEW, folder / NEW.name)
    module = load(folder / NEW.name, 'alias_regression_registry')
    registry_path = root / 'ontology.tsv'
    with registry_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['go_id', 'ontology_row'])
        writer.writerows([(ALIAS, 0), (CANONICAL, 1), (OTHER, 2)])
    task_path = root / 'task.tsv'
    task_path.write_text('go_idx\tinput_go_id\tgo_id\n'
                         f'0\t{ALIAS}\t{CANONICAL}\n'
                         f'1\t{CANONICAL}\t{CANONICAL}\n', encoding='utf-8')
    full_path = root / 'full.json'
    write_json(full_path, {'num_ontology_classes': 3,
                          'registry': {'file': registry_path.name,
                                       'sha256': module.sha256_file(registry_path)},
                          'source': {'checkpoint_sha256': 'a' * 64}})
    mapping_path = root / 'mapping.npy'
    np.save(mapping_path, np.array([1, 1], np.int64))
    alignment_path = root / 'alignment.json'
    write_json(alignment_path, {
        'source': {'go_registry_sha256': module.sha256_file(task_path),
                   'checkpoint_sha256': 'a' * 64},
        'arrays': {'source_row': {'file': mapping_path.name,
                                  'sha256': module.sha256_file(mapping_path)}}})
    return module, full_path, task_path, alignment_path


class AuditedAliasTests(unittest.TestCase):
    def test_real_collision_resolves_by_task_alignment_without_losing_physical_row(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            module, full, task, alignment = manifest_fixture(root)
            before = {path: module.sha256_file(path) for path in root.iterdir() if path.is_file()}
            with self.assertRaisesRegex(ValueError, 'conflicting GO alias'):
                module._base.GORegistryV090.from_manifests(full, task, alignment)
            registry = module.GORegistryV093.from_manifests(full, task, alignment)
            self.assertEqual(registry.task_go_ids, (ALIAS, CANONICAL))
            self.assertEqual(registry.task_canonical_go_ids, (CANONICAL, CANONICAL))
            np.testing.assert_array_equal(registry.task_to_ontology, [1, 1])
            np.testing.assert_array_equal(registry.resolve_target_go([ALIAS, CANONICAL, OTHER]), [1, 1, 2])
            self.assertEqual(registry._row[ALIAS], 0)
            self.assertEqual(registry.canonical_go_id(ALIAS), CANONICAL)
            self.assertEqual(registry.num_ontology_rows, 3)
            self.assertFalse(registry.task_to_ontology.flags.writeable)
            self.assertTrue(registry.contract()['physical_ontology_rows_preserved'])
            self.assertEqual(registry.contract()['audited_task_aliases'], {ALIAS: CANONICAL})
            self.assertEqual(before, {path: module.sha256_file(path) for path in before})

    def test_unrelated_multi_target_alias_conflict_is_not_suppressed(self):
        with tempfile.TemporaryDirectory() as name:
            module, *_ = manifest_fixture(Path(name))
            records = [{'go_id': CANONICAL, 'ontology_row': 0, 'alt_ids': ['GO:7777777']},
                       {'go_id': OTHER, 'ontology_row': 1, 'alt_ids': ['GO:7777777']}]
            with self.assertRaisesRegex(ValueError, 'conflicting GO alias'):
                module.GORegistryV093(records, num_ontology_rows=2)
            records[0]['alt_ids'] = [ALIAS]
            records[1]['alt_ids'] = [ALIAS]
            records.append({'go_id': ALIAS, 'ontology_row': 2})
            with self.assertRaisesRegex(ValueError, 'conflicting GO alias'):
                module.GORegistryV093(records, [ALIAS], [0], num_ontology_rows=3,
                                      task_canonical_go_ids=[CANONICAL])

    def test_same_original_cannot_have_two_audited_canonical_targets(self):
        with tempfile.TemporaryDirectory() as name:
            module, *_ = manifest_fixture(Path(name))
            records = [{'go_id': CANONICAL, 'ontology_row': 0},
                       {'go_id': OTHER, 'ontology_row': 1}]
            with self.assertRaisesRegex(ValueError, '[Cc]onflicting audited task alias'):
                module.GORegistryV093(records, [ALIAS, ALIAS], [0, 1],
                                      task_canonical_go_ids=[CANONICAL, OTHER])

    def test_audited_alias_chains_and_cycles_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            module, *_ = manifest_fixture(Path(name))
            records = [{'go_id': go, 'ontology_row': i} for i, go in enumerate([ALIAS, CANONICAL, OTHER])]
            for targets in ([CANONICAL, OTHER], [CANONICAL, ALIAS]):
                with self.subTest(targets=targets), self.assertRaisesRegex(ValueError, 'chain or cycle'):
                    module.GORegistryV093(records, [ALIAS, CANONICAL], [1, 2],
                                          task_canonical_go_ids=targets)

    def test_redirect_requires_mapping_and_native_aliases_keep_working(self):
        with tempfile.TemporaryDirectory() as name:
            module, *_ = manifest_fixture(Path(name))
            records = [{'go_id': CANONICAL, 'ontology_row': 0, 'alt_ids': [ALIAS]}]
            with self.assertRaisesRegex(ValueError, 'explicit aligned task mapping'):
                module.GORegistryV093(records, [ALIAS], task_canonical_go_ids=[CANONICAL])
            registry = module.GORegistryV093(records, [ALIAS], [0])
            self.assertEqual(registry.canonical_go_id(ALIAS), CANONICAL)
            self.assertEqual(registry.task_canonical_go_ids, (CANONICAL,))

    def test_wrong_aligned_row_and_float_mapping_are_rejected(self):
        for value in (np.array([0, 1], np.int64), np.array([1., 1.], np.float64),
                      np.array([1], np.int64)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                module, full, task, alignment = manifest_fixture(root)
                np.save(root / 'mapping.npy', value)
                payload = json.loads(alignment.read_text())
                payload['arrays']['source_row']['sha256'] = module.sha256_file(root / 'mapping.npy')
                write_json(alignment, payload)
                with self.assertRaisesRegex(ValueError, 'mapping|task columns'):
                    module.GORegistryV093.from_manifests(full, task, alignment)

    def test_manifest_identity_and_payload_hashes_remain_enforced(self):
        for kind in ('registry', 'tasks', 'mapping', 'checkpoint'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                module, full, task, alignment = manifest_fixture(root)
                if kind == 'registry':
                    path = root / 'ontology.tsv'
                    path.write_text(path.read_text().replace(ALIAS, 'GO:1234567'))
                elif kind == 'tasks':
                    task.write_text(task.read_text().replace(ALIAS, 'GO:1234567'))
                elif kind == 'mapping':
                    np.save(root / 'mapping.npy', np.array([0, 1], np.int64))
                else:
                    payload = json.loads(alignment.read_text())
                    payload['source']['checkpoint_sha256'] = 'b' * 64
                    write_json(alignment, payload)
                with self.assertRaisesRegex(ValueError, '[Hh]ash mismatch'):
                    module.GORegistryV093.from_manifests(full, task, alignment)

    def test_obsolete_canonical_target_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            module, *_ = manifest_fixture(Path(name))
            records = [{'go_id': ALIAS, 'ontology_row': 0},
                       {'go_id': CANONICAL, 'ontology_row': 1, 'is_obsolete': True,
                        'replaced_by': [OTHER]}, {'go_id': OTHER, 'ontology_row': 2}]
            with self.assertRaisesRegex(ValueError, 'obsolete GO'):
                module.GORegistryV093(records, [ALIAS], [1],
                                      task_canonical_go_ids=[CANONICAL])


class IntegrationTests(unittest.TestCase):
    def test_calibration_then_build_with_retained_original_geometry_row(self):
        calibration_tests = load(Path(__file__).with_name('test_unified_calibration_v093.py'),
                                 'alias_regression_calibration_tests')
        calibration, dataset = calibration_tests.calibration, calibration_tests.dataset
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            calibration_args, build_args, _ = calibration_tests.fixture(root)
            registry_path = root / 'ontology/registry.tsv'
            with registry_path.open('a') as stream:
                stream.write('GO:0000011\t5\n')
            full_path = root / 'ontology/full.json'
            full = dataset.read_json(full_path)
            full['num_ontology_classes'] = 6
            full['registry']['sha256'] = dataset.sha256(registry_path)
            write_json(full_path, full)
            before = {path: dataset.sha256(path) for folder in ('data', 'ontology', 'work')
                      for path in (root / folder).rglob('*') if path.is_file()}
            result = calibration.calibrate(calibration_args, root)
            self.assertFalse(result['independently_calibrated'])
            build_args.thresholds_json = root / 'thresholds.json'
            manifest = dataset.build_dataset(build_args, root)
            self.assertTrue(manifest['complete'])
            mapping = np.load(root / 'new' / manifest['files']['task_to_ontology'])
            np.testing.assert_array_equal(mapping, [0, 1, 2, 3, 4, 0])
            config = dataset.read_json(root / 'work/resolved_config.json')
            weak_path = root / 'data/weak/manifest.json'
            loaded_mapping, registry, _ = dataset._load_go_mapping(
                root, config, dataset.read_json(weak_path), weak_path, 6)
            np.testing.assert_array_equal(loaded_mapping, mapping)
            self.assertEqual(registry._row['GO:0000011'], 5)
            self.assertEqual(registry.resolve_one('GO:0000011'), 0)
            self.assertEqual(before, {path: dataset.sha256(path) for path in before})

    def test_updated_go_bundle_preserves_redirected_physical_anchor(self):
        cli = load(PACKAGE / 'scripts/nbs/predict_nbs_go_v093.py', 'alias_regression_predict_cli')
        with tempfile.TemporaryDirectory() as name:
            module, full, task, alignment = manifest_fixture(Path(name))
            registry = module.GORegistryV093.from_manifests(full, task, alignment)
            old_rows = np.array([2, 0, 1], np.int64)
            records = [{'go_id': ALIAS, 'ontology_row': 2},
                       {'go_id': CANONICAL, 'ontology_row': 0},
                       {'go_id': OTHER, 'ontology_row': 1},
                       {'go_id': 'GO:9999999', 'ontology_row': 3}]
            new = cli.updated_registry(registry, {'registry_records': records}, old_rows, 4,
                                       module.GORegistryV093)
            np.testing.assert_array_equal(new.task_to_ontology, [0, 0])
            self.assertEqual(new._row[ALIAS], 2)
            self.assertEqual(new.resolve_one(ALIAS), 0)
            bad = deepcopy(records)
            bad[0]['ontology_row'] = 3
            with self.assertRaisesRegex(ValueError, 'anchored row'):
                cli.updated_registry(registry, {'registry_records': bad}, old_rows, 4,
                                     module.GORegistryV093)


if __name__ == '__main__':
    unittest.main()
