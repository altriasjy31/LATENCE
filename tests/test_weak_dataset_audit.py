from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/nbs/audit_nbs_weak_dataset.py'
spec = importlib.util.spec_from_file_location('weak_audit_test_module', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root):
    data = root / 'data'
    (data/'features').mkdir(parents=True)
    registry = [('weak', 1), ('core', 1), ('weak', 0), ('core', 0),
                ('weak', 2), ('core', 2), ('core', 3)]
    with (data/'features/protein_registry.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['protein_idx', 'role', 'role_row_idx'])
        writer.writerows((index, role, row) for index, (role, row) in enumerate(registry))
    (data/'indices').mkdir()
    gold = [[], [1], [], [0, 1], [], [2], [0, 3]]
    ptr = np.concatenate([[0], np.cumsum([len(x) for x in gold])]).astype(np.int64)
    np.save(data/'indices/gold_ptr.npy', ptr)
    np.save(data/'indices/gold_go.npy', np.asarray(sum(gold, []), np.int32))
    inverted = {'num_go': 4, 'indices': {
        'gold': {'protein_major': {'indptr': 'gold_ptr.npy', 'go_idx': 'gold_go.npy', 'num_go': 4}},
        'candidate': {'fixed_degree': 4, 'source_protein_start': 0}}}
    save_json(data/'indices/inverted.json', inverted)
    (data/'weak').mkdir()
    np.save(data/'weak/pseudo_ptr.npy', np.array([0, 1, 1, 3], np.int64))
    np.save(data/'weak/pseudo_go.npy', np.array([0, 1, 2], np.int32))
    np.save(data/'weak/pseudo_prob.npy', np.array([.5, .9, .7], np.float16))
    candidate_go = np.tile(np.array([0, 1, 2, 3], np.int32), (7, 1))
    edge = np.stack([np.repeat(np.arange(7, dtype=np.int32), 4), candidate_go.ravel()])
    attr = np.zeros((7, 4, 3), np.float32)
    attr[:, :, 0] = [.7, .8, .9, .1]
    attr[:, :, 1] = [1, 2, 3, 4]
    attr[:, :, 2] = [.5, 1, .25, 1/3]
    np.save(data/'weak/candidate.npy', edge)
    np.save(data/'weak/attr.npy', attr.reshape(-1, 3))
    weak = {'task': 'bp', 'go_registry': {'num_terms': 4},
            'model_semantics': {'modelout': {'prediction_key': 'fixture_modelout'}},
            'weak_pseudo_targets': {'role': 'weak', 'comparison': '>', 'threshold': .5,
                 'csr_indptr_file': 'pseudo_ptr.npy', 'csr_indices_file': 'pseudo_go.npy',
                 'csr_probability_file': 'pseudo_prob.npy'},
            'backbone_candidate_edges': {'edge_index_file': 'candidate.npy',
                                        'edge_attr_file': 'attr.npy'}}
    save_json(data/'weak/manifest.json', weak)
    config = {'task': 'bp', 'data': {'root': 'data', 'weak_graph_predictions_manifest': 'weak/manifest.json',
                  'go_protein_inverted_index_manifest': 'indices/inverted.json'},
              'full_task': {'label_split': 'all_seen', 'holdout_core_count': 1,
                            'holdout_seed': 11, 'sampler': {'candidate_topk': 2}}}
    save_json(root/'work/resolved_config.json', config)
    args = argparse.Namespace(work_dir=Path('work'), source_summary=None, output_dir=None,
                              skip_candidates=False, chunk_edges=8)
    return args, ptr


class MembershipTests(unittest.TestCase):
    def test_csr_gather_scattered_rows_and_deduplicate(self):
        ptr = np.array([0, 2, 2, 5], np.int64)
        go = np.array([2, 2, 1, 0, 1])
        result = audit.membership_counts(ptr, go, [2, 0, 1], 3,
                                         groups=audit.frequency_groups(np.array([1, 6, 10])), chunk_edges=2)
        np.testing.assert_array_equal(result['counts'], [1, 1, 1])
        np.testing.assert_array_equal(result['degree'], [2, 1, 0])
        self.assertEqual(result['duplicate_edges'], 2)

    def test_probability_change_does_not_change_membership(self):
        ptr, go = np.array([0, 1, 1, 3]), np.array([0, 1, 2])
        first = audit.membership_counts(ptr, go, [0, 1, 2], 4, probability=np.array([.5, .7, .9]))
        second = audit.membership_counts(ptr, go, [0, 1, 2], 4, probability=np.array([.2, .9, .6]))
        np.testing.assert_array_equal(first['counts'], second['counts'])
        np.testing.assert_array_equal(first['degree'], second['degree'])
        self.assertEqual(first['probability']['stored_exactly_0_5'], 1)
        self.assertEqual(second['probability']['below_0_5'], 1)

    def test_exact_frequency_boundaries(self):
        counts = np.array([0, 1, 5, 6, 9, 10, 99, 100, 999, 1000])
        np.testing.assert_array_equal(audit.frequency_groups(counts), [0, 1, 1, 2, 2, 3, 3, 4, 4, 5])

    def test_holdout_uses_role_order_and_not_global_sort(self):
        core = np.array([3, 1, 5, 6])
        ptr = np.array([0, 0, 1, 1, 3, 3, 4, 6])
        options = {'holdout_core_count': 1, 'holdout_seed': 11}
        train, val = audit.reconstruct_split(core, ptr, options)
        expected = np.sort(np.random.default_rng(11).permutation(core)[:1])
        np.testing.assert_array_equal(val, expected)
        self.assertFalse(np.isin(train, val).any())

    def test_order_matches_reciprocal_rank_probability_then_GO(self):
        go = np.array([[3, 1, 2, 0]])
        attr = np.array([[[.7, 0, .5], [.7, 0, .5], [.9, 0, .5], [.1, 0, 1]]])
        np.testing.assert_array_equal(audit.candidate_order(go, attr, 4, 4), [[0, 2, 1, 3]])


class AuditIntegrationTests(unittest.TestCase):
    def test_interleaved_roles_and_empty_weak_rows(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, ptr = fixture(root)
            before = {str(p): audit.sha256(p) for p in (root/'data').rglob('*') if p.is_file()}
            report = audit.audit(args, root)
            self.assertEqual(report['num_training_core_proteins'], 3)
            self.assertEqual(report['num_holdout_core_proteins'], 1)
            self.assertEqual(report['num_training_weak_proteins'], 2)
            self.assertEqual(report['num_weak_registry_proteins'], 3)
            self.assertEqual(report['pseudo_probability']['stored_exactly_0_5'], 1)
            self.assertFalse(report['weak_precision_or_recall_estimated'])
            self.assertFalse(report['dense_teacher_probabilities_read'])
            self.assertEqual(report['candidate']['retained_topk'], 2)
            self.assertEqual(report['candidate']['weak_source_proteins'], 3)
            self.assertEqual(report['candidate']['active_weak_source_proteins'], 2)
            strata = report['frequency_strata']
            self.assertEqual(sum(x['weak_positive_memberships'] for x in strata), 3)
            self.assertEqual(sum(x['potential_negative_pairs_before_alias_exclusion'] for x in strata), 5)
            self.assertEqual(sum(x['candidate_pseudo_in_export_pairs'] for x in strata), 3)
            self.assertEqual(sum(x['candidate_pseudo_in_retained_pairs'] for x in strata), 2)
            with (root/'work/weak_dataset_audit_r1/go_coverage.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertAlmostEqual(float(rows[0]['weak_prevalence_among_training_weak']), .5)
            self.assertAlmostEqual(float(rows[0]['weak_prevalence_among_all_weak']), 1/3)
            self.assertEqual(before, {str(p): audit.sha256(p) for p in (root/'data').rglob('*') if p.is_file()})

    def test_summary_only_checks_training_identity_not_epoch(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, _ = fixture(root)
            first = audit.audit(args, root)
            reference = {'checkpoint_epoch': 5, 'num_GO_columns': 4, 'num_training_core_proteins': 3,
                         'training_core_ids_sha256': first['identity']['training_core_ids_sha256'],
                         'core_frequency_sha256': first['identity']['core_frequency_sha256']}
            save_json(root/'other_epoch.json', reference)
            args.source_summary = Path('other_epoch.json')
            args.output_dir = Path('second')
            report = audit.audit(args, root)
            self.assertTrue(report['identity']['reference']['epoch_not_used_as_training_input'])
            self.assertFalse(report['identity']['source_prediction_binding']['weak_source_identity_bound'])
            reference['core_frequency_sha256'] = 'wrong'
            save_json(root/'other_epoch.json', reference)
            with self.assertRaisesRegex(ValueError, 'frequencies differ'):
                audit.audit(args, root)

    def test_missing_optional_summary_and_candidates_are_explicit(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, _ = fixture(root)
            (root/'data/weak/candidate.npy').unlink()
            report = audit.audit(args, root)
            self.assertFalse(report['identity']['reference']['available'])
            self.assertFalse(report['candidate']['available'])
            self.assertNotIn('candidate_export_weak_pairs', report['frequency_strata'][0])

    def test_skip_candidates_reads_membership_only(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, _ = fixture(root)
            (root/'data/weak/candidate.npy').write_text('not a npy')
            args.skip_candidates = True
            report = audit.audit(args, root)
            self.assertEqual(report['candidate']['reason'], '--skip-candidates')

    def test_bad_fixed_degree_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, _ = fixture(root)
            edge = np.load(root/'data/weak/candidate.npy')
            edge[0, 0] = 1
            np.save(root/'data/weak/candidate.npy', edge)
            with self.assertRaisesRegex(ValueError, 'not contiguous'):
                audit.audit(args, root)

    def test_changed_input_cannot_overwrite_old_report(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, _ = fixture(root)
            audit.audit(args, root)
            np.save(root/'data/weak/pseudo_prob.npy', np.array([.5, .8, .7], np.float16))
            with self.assertRaisesRegex(ValueError, 'different data/code/options'):
                audit.audit(args, root)

    def test_invalid_CSR_probabilities_and_GO_axis_fail(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, _ = fixture(root)
            np.save(root/'data/weak/pseudo_prob.npy', np.array([.5, np.nan, .7], np.float16))
            with self.assertRaisesRegex(ValueError, 'finite'):
                audit.audit(args, root)
            np.save(root/'data/weak/pseudo_prob.npy', np.array([.5, .8, .7], np.float16))
            np.save(root/'data/weak/pseudo_go.npy', np.array([0, 1, 8], np.int32))
            with self.assertRaisesRegex(ValueError, 'outside immutable'):
                audit.audit(args, root)

    def test_prediction_manifest_binds_current_weak_sources(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args, _ = fixture(root)
            first = audit.audit(args, root)
            files = first['identity']['files']
            keys = {'protein_registry': 'registry_sha256',
                    'weak_manifest': 'weak_graph_predictions_manifest_sha256',
                    'inverted_manifest': 'go_protein_inverted_index_manifest_sha256',
                    'gold_indptr': 'gold_indptr_sha256', 'gold_go_idx': 'gold_go_idx_sha256',
                    'pseudo_indptr': 'pseudo_indptr_sha256', 'pseudo_go_idx': 'pseudo_go_idx_sha256',
                    'pseudo_probability': 'pseudo_probability_sha256'}
            pred_path = root/'reference/full/nbs_full_task_prediction_manifest.json'
            summary_path = root/'reference/full/metrics_v092/summary.json'
            save_json(pred_path, {'contract': {'data': {key: files[name]['sha256'] for name, key in keys.items()}}})
            source = {'num_GO_columns': 4, 'num_training_core_proteins': 3,
                      'training_core_ids_sha256': first['identity']['training_core_ids_sha256'],
                      'core_frequency_sha256': first['identity']['core_frequency_sha256'],
                      'prediction_manifest_sha256': audit.sha256(pred_path)}
            save_json(summary_path, source)
            args.source_summary = summary_path
            args.output_dir = Path('bound')
            result = audit.audit(args, root)
            self.assertTrue(result['identity']['source_prediction_binding']['weak_source_identity_bound'])
            np.save(root/'data/weak/pseudo_prob.npy', np.array([.5, .8, .7], np.float16))
            with self.assertRaisesRegex(ValueError, 'pseudo_probability_sha256'):
                audit.audit(args, root)

    def test_prediction_manifest_must_match_summary_hash(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root/'full/nbs_full_task_prediction_manifest.json'
            summary = root/'full/metrics_v092/summary.json'
            save_json(path, {'contract': {'data': {}}})
            save_json(summary, {'prediction_manifest_sha256': 'wrong'})
            with self.assertRaisesRegex(ValueError, 'manifest hash differs'):
                audit.bind_prediction_contract(summary, {})


if __name__ == '__main__':
    unittest.main()
