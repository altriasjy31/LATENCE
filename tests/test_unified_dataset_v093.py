from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
MODULE = PACKAGE/'nbs_models/nbs_protein_go/nbs_pg/unified_weak_dataset_v093.py'
spec = importlib.util.spec_from_file_location('unified_builder_test_module', MODULE)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
SOURCE_GO_MODULE = Path(__file__).resolve().parents[3]/'fusion_v093/source/latence_nbs_v090_single12800/nbs_models/nbs_protein_go/nbs_pg/go_registry_v090.py'


def js(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root, *, source_go_module=SOURCE_GO_MODULE):
    folder = root/'nbs_models/nbs_protein_go/nbs_pg'
    folder.mkdir(parents=True)
    if not source_go_module.exists():
        source_go_module = MODULE.parent/'go_registry_v090.py'
    shutil.copyfile(source_go_module, folder/'go_registry_v090.py')
    shutil.copyfile(MODULE.with_name('go_registry_v093.py'), folder/'go_registry_v093.py')
    data = root/'data'
    (data/'features').mkdir(parents=True)
    registry = [('weak', 1), ('core', 1), ('weak', 0), ('core', 0), ('weak', 2), ('core', 2), ('core', 3)]
    with (data/'features/protein_registry.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['protein_idx', 'protein_id', 'role', 'role_row_idx'])
        w.writerows((i, 'p'+str(i), role, row) for i, (role, row) in enumerate(registry))
    (data/'indices').mkdir()
    gold = [[], [1], [], [0, 1], [], [2], [0, 3]]
    ptr = np.concatenate([[0], np.cumsum([len(r) for r in gold])]).astype(np.int64)
    np.save(data/'indices/gold_ptr.npy', ptr)
    np.save(data/'indices/gold_go.npy', np.array(sum(gold, []), np.int32))
    inv = {'num_go': 6, 'indices': {'gold': {'protein_major': {'indptr': 'gold_ptr.npy', 'go_idx': 'gold_go.npy', 'num_go': 6}},
                                  'candidate': {'fixed_degree': 6, 'source_protein_start': 0}}}
    js(data/'indices/inverted.json', inv)
    (data/'weak').mkdir()
    b = np.array([[.6, .8, .2, .6, .99, .7], [.7, .1, .8, .4, .9, .2], [.9, .5, .7, .55, .7, .9]], np.float32)
    m = np.array([[.2, .9, .8, .6, .99, .1], [.1, .9, .1, .8, .9, .2], [.9, .2, .7, .55, .7, .9]], np.float32)
    old_y = m > .5
    row, labels = np.nonzero(old_y)
    np.save(data/'weak/old_ptr.npy', np.concatenate([[0], np.cumsum(old_y.sum(1))]).astype(np.int64))
    np.save(data/'weak/old_go.npy', labels.astype(np.int32))
    np.save(data/'weak/old_prob.npy', m[row, labels])
    np.save(data/'weak/b.npy', b); np.save(data/'weak/m.npy', m)
    go = np.tile(np.arange(6, dtype=np.int32), (7, 1))
    edge = np.stack([np.repeat(np.arange(7, dtype=np.int32), 6), go.ravel()])
    attr = np.zeros((7, 6, 3), np.float32)
    attr[:, :, 0] = [.7, .8, .9, .1, .2, .3]
    attr[:, :, 1] = np.arange(6)
    attr[:, :, 2] = 1/np.arange(1, 7)
    np.save(data/'weak/edge.npy', edge)
    np.save(data/'weak/attr.npy', attr.reshape(-1, 3))
    ids = ['p2', 'p0', 'p4']
    ih = hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
    weak = {'task': 'bp', 'go_registry': {'num_terms': 6, 'path': '../gg_relations/go_registry.tsv'},
            'roles': [{'role': 'weak', 'rows': 3, 'protein_ids_sha256': ih,
                       'backbone_dense_file': 'b.npy', 'modelout_dense_file': 'm.npy'}],
            'weak_pseudo_targets': {'role': 'weak', 'comparison': '>', 'threshold': .5,
                                   'csr_indptr_file': 'old_ptr.npy', 'csr_indices_file': 'old_go.npy',
                                   'csr_probability_file': 'old_prob.npy'},
            'backbone_candidate_edges': {'edge_index_file': 'edge.npy', 'edge_attr_file': 'attr.npy'}}
    js(data/'weak/manifest.json', weak)
    (data/'gg_relations').mkdir()
    with (data/'gg_relations/go_registry.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t'); w.writerow(['go_idx', 'input_go_id', 'go_id'])
        for i in range(6):
            w.writerow([i, 'GO:'+str(i+1 if i < 5 else 11).zfill(7), 'GO:'+str(i+1 if i < 5 else 1).zfill(7)])
    ont = root/'ontology';ont.mkdir()
    with (ont/'registry.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t');w.writerow(['go_id', 'ontology_row'])
        w.writerows(('GO:'+str(i+1).zfill(7),i) for i in range(5))
    source_hash = 'a'*64
    js(ont/'full.json', {'registry': {'file': 'registry.tsv', 'sha256': builder.sha256(ont/'registry.tsv')},
                         'source': {'checkpoint_sha256': source_hash}, 'num_ontology_classes': 5})
    mapping = np.array([0, 1, 2, 3, 4, 0], np.int64)
    np.save(ont/'mapping.npy', mapping)
    js(ont/'alignment.json', {'source': {'go_registry_sha256': builder.sha256(data/'gg_relations/go_registry.tsv'),
                                         'checkpoint_sha256': source_hash},
                              'arrays': {'source_row': {'file': 'mapping.npy', 'sha256': builder.sha256(ont/'mapping.npy')}}})
    config = {'task': 'bp', 'data': {'root': 'data', 'weak_graph_predictions_manifest': 'weak/manifest.json',
                 'go_protein_inverted_index_manifest': 'indices/inverted.json', 'full_go_box_manifest': 'ontology/full.json'},
              'go_boxsqel': {'alignment_manifest': 'ontology/alignment.json'},
              'full_task': {'label_split': 'all_seen', 'holdout_core_count': 1, 'holdout_seed': 11}}
    js(root/'work/resolved_config.json', config)
    args = argparse.Namespace(source_work_dir=Path('work'), output_dir=Path('new'), backbone_prob=None,
          modelout_prob=None, thresholds_json=None, low_frequency_max=999, candidate_mode='frequency',
          label_holdout_fraction=0., seed=1337, chunk_rows=2)
    return args, b, m


class PolicyTests(unittest.TestCase):
    def test_alias_groups_not_split_and_natural_zero_excluded(self):
        counts = np.array([2, 1, 0, 1, 0])
        mapping = np.array([0, 1, 2, 3, 0])
        seen, heldout = builder.make_seen_mask(counts, mapping, .5, 7)
        self.assertEqual(seen[0], seen[4])
        self.assertFalse(seen[2])
        self.assertEqual(len(heldout), 1)
        self.assertTrue(seen.any())

    def test_candidate_quota_dedup_filter_and_padding(self):
        counts = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
        go = np.arange(len(counts))[None]
        attrs = np.zeros((1, len(counts), 3), np.float32)
        attrs[0, :, 2] = np.arange(len(counts))
        seen = np.ones(len(counts), bool);seen[4] = False
        got, attr = builder.choose_candidate_rows(go, attrs, counts, seen)
        valid = got[0][got[0] >= 0]
        self.assertEqual(len(valid), 16)
        self.assertEqual(len(np.unique(valid)), 16)
        self.assertNotIn(4, valid)
        self.assertEqual(int(((counts[valid] >= 1)&(counts[valid] <= 9)).sum()), 4)
        self.assertEqual(int(((counts[valid] >= 10)&(counts[valid] <= 999)).sum()), 4)
        tiny, _ = builder.choose_candidate_rows(np.array([[0, 0, 1]]), np.ones((1, 3, 3), np.float32), counts, seen)
        self.assertEqual(int((tiny >= 0).sum()), 2)

    def test_candidate_global_order_matches_original(self):
        go = np.array([[3, 1, 2, 0]])
        attrs = np.array([[[.7, 0, .5], [.7, 0, .5], [.9, 0, .5], [.1, 0, 1]]])
        chosen, _ = builder.choose_candidate_rows(go, attrs, np.ones(4, np.int64), np.ones(4, bool), mode='global')
        np.testing.assert_array_equal(chosen[0, :4], [0, 2, 1, 3])

    def test_candidate_padding_is_ignored_before_quota_and_refilled(self):
        counts=np.array([1,10,1000])
        go=np.array([[-1,2,0,-1,1],[-1,-1,-1,-1,-1]])
        attrs=np.zeros((2,5,3),np.float32)
        attrs[:,:,2]=np.array([10,4,3,2,1])
        got,used=builder.choose_candidate_rows(go,attrs,counts,np.ones(3,bool))
        np.testing.assert_array_equal(got[0,:3],[0,1,2])
        self.assertTrue((got[0,3:]==-1).all())
        self.assertTrue((got[1]==-1).all())
        self.assertTrue((used[1]==0).all())
        with self.assertRaises(ValueError):
            builder.choose_candidate_rows(np.array([[-2]]),np.zeros((1,1,3),np.float32),counts,np.ones(3,bool))

    def test_threshold_policy_rejects_nan_unknown_and_boundary(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'thresholds.json'
            js(path, {'thresholds': {'1-5': .8}, 'provenance': {'scope': 'fixed'}})
            policy, metadata = builder.threshold_policy(path)
            self.assertEqual(policy['1-5'], .8)
            self.assertFalse(metadata['independently_calibrated'])
            js(path, {'1-5': float('nan')})
            with self.assertRaises(ValueError):builder.threshold_policy(path)
            js(path, {'rare': .5})
            with self.assertRaises(ValueError):builder.threshold_policy(path)


class BuildTests(unittest.TestCase):
    def test_installed_cpu_cli_and_manifest_payload_tamper(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name);fixture(root)
            destination=root/'nbs_models/nbs_protein_go/nbs_pg/unified_weak_dataset_v093.py'
            shutil.copyfile(MODULE,destination)
            cli=root/'scripts/nbs/build_nbs_unified_dataset_v093.py';cli.parent.mkdir(parents=True)
            shutil.copyfile(PACKAGE/'scripts/nbs/build_nbs_unified_dataset_v093.py',cli)
            result=subprocess.run([sys.executable,str(cli),'--source-work-dir',str(root/'work'),
                                   '--output-dir',str(root/'new'),'--chunk-rows','2'],
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=builder.load_unified_manifest(root/'new/unified_dataset_manifest.json')
            payload=root/'new'/manifest['pseudo']['source']
            values=np.load(payload);values[0]=77;np.save(payload,values)
            with self.assertRaisesRegex(ValueError,'file hash mismatch'):
                builder.load_unified_manifest(root/'new/unified_dataset_manifest.json')

    def test_full_build_real_schema_read_only_and_reuse(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name);args, b, _ = fixture(root)
            before = {str(p):builder.sha256(p) for p in root.rglob('*') if p.is_file()}
            result = builder.build_dataset(args, root)
            folder = root/'new'
            self.assertEqual(result['num_task_go'], 6)
            mask = np.load(folder/result['files']['seen_task_mask'])
            self.assertFalse(mask[4])
            self.assertEqual(mask[0], mask[5])
            ptr = np.load(folder/result['pseudo']['indptr'])
            go = np.load(folder/result['pseudo']['go_idx'])
            prob = np.load(folder/result['pseudo']['probability'])
            source = np.load(folder/result['pseudo']['source'])
            for row in range(3):
                expected = np.flatnonzero((b[row] > .5) & mask)
                actual = go[ptr[row]:ptr[row+1]]
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(prob[ptr[row]:ptr[row+1]], b[row, expected])
            self.assertTrue((source == 1).all())
            self.assertNotIn(1, go[ptr[2]:ptr[3]])  # exact 0.5 is not newly promoted
            for p, digest in before.items():self.assertEqual(builder.sha256(p),digest)
            self.assertEqual(builder.load_unified_manifest(folder/'unified_dataset_manifest.json')['identity'], result['identity'])
            reused = builder.build_dataset(args, root)
            self.assertEqual(reused['identity'], result['identity'])
            args.low_frequency_max = 0
            with self.assertRaisesRegex(ValueError, 'different source/policy'):
                builder.build_dataset(args, root)

    def test_holdout_filters_supervision_and_candidates(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name);args, _, _ = fixture(root)
            args.label_holdout_fraction = .5
            result = builder.build_dataset(args, root)
            folder = root/'new';mask = np.load(folder/result['files']['seen_task_mask'])
            labels = np.load(folder/result['pseudo']['go_idx'])
            candidates = np.load(folder/result['candidates']['go'])
            self.assertTrue(mask[labels].all())
            self.assertTrue(mask[candidates[candidates >= 0]].all())
            mapping = np.load(folder/result['files']['task_to_ontology'])
            for row in np.unique(mapping):self.assertIn(int(mask[mapping==row].sum()), [0, int((mapping==row).sum())])

    def test_original_quantized_membership_restored_for_unchanged_modelout(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name);args, _, _ = fixture(root)
            args.low_frequency_max = 0
            m = np.load(root/'data/weak/m.npy')
            m[0, 1] = .5
            np.save(root/'data/weak/m.npy', m)
            old_go = np.load(root/'data/weak/old_go.npy')
            old_ptr = np.load(root/'data/weak/old_ptr.npy')
            old_p = np.load(root/'data/weak/old_prob.npy')
            old_p[:old_ptr[1]][old_go[:old_ptr[1]] == 1] = .5
            np.save(root/'data/weak/old_prob.npy', old_p)
            result = builder.build_dataset(args, root)
            folder=root/'new';ptr=np.load(folder/result['pseudo']['indptr']);go=np.load(folder/result['pseudo']['go_idx'])
            self.assertIn(1, go[ptr[0]:ptr[1]])
            self.assertEqual(result['restored_quantized_boundary_memberships'],1)

    def test_teacher_source_and_unaltered_probability_with_custom_threshold(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name);args, b, m = fixture(root)
            args.low_frequency_max = 1
            js(root/'thresholds.json', {'thresholds': {'1-5': .1, 'zero': .1}})
            args.thresholds_json = root/'thresholds.json'
            result=builder.build_dataset(args,root)
            folder=root/'new';counts=np.load(folder/result['files']['core_counts']);mask=np.load(folder/result['files']['seen_task_mask'])
            ptr=np.load(folder/result['pseudo']['indptr']);go=np.load(folder/result['pseudo']['go_idx'])
            prob=np.load(folder/result['pseudo']['probability']);source=np.load(folder/result['pseudo']['source'])
            for row in range(3):
                labels=go[ptr[row]:ptr[row+1]]
                np.testing.assert_array_equal(source[ptr[row]:ptr[row+1]],np.where(counts[labels]<=1,1,2))
                np.testing.assert_array_equal(prob[ptr[row]:ptr[row+1]],np.where(counts[labels]<=1,b[row,labels],m[row,labels]))
            self.assertTrue((prob < .5).any())

    def test_protein_order_and_source_probability_corruption_fail(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);args,_,_=fixture(root)
            weak=builder.read_json(root/'data/weak/manifest.json')
            weak['roles'][0]['protein_ids_sha256']='wrong';js(root/'data/weak/manifest.json',weak)
            with self.assertRaisesRegex(ValueError,'protein-ID order'):builder.build_dataset(args,root)


if __name__ == '__main__':
    unittest.main()
