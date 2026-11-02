"""CPU integration of all-frequency B membership and independent M targets."""
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
MODULE_DIR = PACKAGE/'nbs_models/nbs_protein_go/nbs_pg'
OLD_PACKAGE = PACKAGE.parents[1]/'unified_v093/package'
OLD_PROJECT = PACKAGE.parents[1]/'unified_v093/project'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def js(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root):
    folder = root/'nbs_models/nbs_protein_go/nbs_pg'
    folder.mkdir(parents=True)
    for name in ['weak_dataset_v094.py', 'unified_weak_dataset_v093.py',
                 'go_registry_v093.py', 'go_registry_v090.py']:
        source = MODULE_DIR/name
        if not source.is_file():
            source = OLD_PACKAGE/'nbs_models/nbs_protein_go/nbs_pg'/name
        if not source.is_file():
            source = OLD_PROJECT/'nbs_models/nbs_protein_go/nbs_pg'/name
        shutil.copyfile(source, folder/name)
    script_folder = root/'scripts/nbs'
    script_folder.mkdir(parents=True)
    for name in ['calibrate_nbs_backbone_v094.py', 'build_nbs_dataset_v094.py']:
        shutil.copyfile(PACKAGE/'scripts/nbs'/name, script_folder/name)
    dataset = load(folder/'weak_dataset_v094.py', 'v094_fixture_dataset')
    calibration = load(script_folder/'calibrate_nbs_backbone_v094.py', 'v094_fixture_calibration')
    data = root/'data'
    (data/'features').mkdir(parents=True)
    n_core, n_weak, n_go = 1204, 3, 7
    registry = [('weak', 1), ('weak', 0), ('weak', 2)] + [('core', i) for i in range(n_core)]
    with (data/'features/protein_registry.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['protein_idx', 'protein_id', 'role', 'role_row_idx'])
        writer.writerows((i, 'p'+str(i), role, local) for i, (role, local) in enumerate(registry))
    (data/'indices').mkdir()
    gold = [[], [], []] + [[g for g, cutoff in enumerate([4, 8, 50, 110, 1104]) if i < cutoff]
                         for i in range(n_core)]
    for i in range(1104, n_core):
        gold[n_weak+i] = [3]  # Keep high-frequency negatives in the labeled core population.
    ptr = np.concatenate([[0], np.cumsum([len(row) for row in gold])]).astype(np.int64)
    np.save(data/'indices/gold_ptr.npy', ptr)
    np.save(data/'indices/gold_go.npy', np.array(sum(gold, []), np.int32))
    js(data/'indices/inverted.json', {'indices': {'gold': {'protein_major': {
        'indptr': 'gold_ptr.npy', 'go_idx': 'gold_go.npy', 'num_go': n_go}},
        'candidate': {'fixed_degree': n_go, 'source_protein_start': 0}}})
    (data/'weak').mkdir()
    b = np.array([[.05]*n_go, [.95,.05,.05,.05,.95,.95,.7], [.05,.95,.95,.95,.05,.99,.05]], np.float32)
    m = np.array([[.99]*n_go, [0.,.99,.99,.99,.02,.9,.04], [.99,.1,.2,.3,.99,.9,.99]], np.float32)
    core_prob = np.full((n_core, n_go), .1, np.float32)
    for row in range(n_core):
        core_prob[row, gold[n_weak+row]] = .9
    np.save(data/'weak/b.npy', b)
    np.save(data/'weak/m.npy', m)
    np.save(data/'weak/core.npy', core_prob)
    old_member = m > .5
    row, col = np.nonzero(old_member)
    np.save(data/'weak/old_ptr.npy', np.concatenate([[0],np.cumsum(old_member.sum(1))]).astype(np.int64))
    np.save(data/'weak/old_go.npy', col.astype(np.int32))
    np.save(data/'weak/old_prob.npy', m[row,col])
    edges = np.stack([np.repeat(np.arange(len(registry)), n_go), np.tile(np.arange(n_go),len(registry))]).astype(np.int32)
    attrs = np.zeros((len(registry)*n_go,3),np.float32)
    attrs[:,0] = .7
    attrs[:,2] = np.tile(1/np.arange(1,n_go+1),len(registry))
    np.save(data/'weak/edge.npy',edges);np.save(data/'weak/attr.npy',attrs)
    def id_hash(ids):
        return hashlib.sha256(('\n'.join('p'+str(i) for i in ids)+'\n').encode()).hexdigest()
    weak = {'task':'bp','go_registry':{'num_terms':n_go,'path':'../gg_relations/go_registry.tsv'},
        'roles':[{'role':'weak','rows':n_weak,'protein_ids_sha256':id_hash([1,0,2]),
                  'backbone_dense_file':'b.npy','modelout_dense_file':'m.npy'},
                 {'role':'core','rows':n_core,'protein_ids_sha256':id_hash(range(3,len(registry))),
                  'backbone_dense_file':'core.npy'}],
        'weak_pseudo_targets':{'role':'weak','comparison':'>','threshold':.5,
                              'csr_indptr_file':'old_ptr.npy','csr_indices_file':'old_go.npy',
                              'csr_probability_file':'old_prob.npy'},
        'backbone_candidate_edges':{'edge_index_file':'edge.npy','edge_attr_file':'attr.npy'}}
    js(data/'weak/manifest.json',weak)
    (data/'gg_relations').mkdir()
    with (data/'gg_relations/go_registry.tsv').open('w',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['go_idx','input_go_id','go_id'])
        writer.writerows((i,f'GO:{i+1 if i<6 else 11:07d}',f'GO:{i+1 if i<6 else 1:07d}') for i in range(n_go))
    ontology=root/'ontology';ontology.mkdir()
    (ontology/'registry.tsv').write_text('go_id\tontology_row\n'+''.join(f'GO:{i+1:07d}\t{i}\n' for i in range(6)))
    np.save(ontology/'mapping.npy',np.array([0,1,2,3,4,5,0],np.int64))
    js(ontology/'full.json',{'registry':{'file':'registry.tsv','sha256':dataset.sha256(ontology/'registry.tsv')},
                           'source':{'checkpoint_sha256':'a'*64},'num_ontology_classes':6})
    js(ontology/'alignment.json',{'source':{'go_registry_sha256':dataset.sha256(data/'gg_relations/go_registry.tsv'),
                                           'checkpoint_sha256':'a'*64},
                                 'arrays':{'source_row':{'file':'mapping.npy','sha256':dataset.sha256(ontology/'mapping.npy')}}})
    config={'task':'bp','data':{'root':'data','weak_graph_predictions_manifest':'weak/manifest.json',
                              'go_protein_inverted_index_manifest':'indices/inverted.json',
                              'full_go_box_manifest':'ontology/full.json'},
            'go_boxsqel':{'alignment_manifest':'ontology/alignment.json'},
            'full_task':{'label_split':'all_seen','holdout_core_count':1,'holdout_seed':11}}
    js(root/'work/resolved_config.json',config)
    calibration_args=argparse.Namespace(source_work_dir=Path('work'),output=Path('thresholds.json'),
        label_holdout_fraction=0.,seed=1337,chunk_rows=128)
    build_args=argparse.Namespace(source_work_dir=Path('work'),output_dir=Path('new'),
        backbone_prob=None,modelout_prob=None,thresholds_json=root/'thresholds.json',
        label_holdout_fraction=0.,seed=1337,chunk_rows=128)
    return dataset,calibration,calibration_args,build_args,b,m


class DatasetV094Tests(unittest.TestCase):
    def test_calibration_fits_high_frequency_group_and_excludes_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d,c,args,_,_,_=fixture(root)
            report=c.calibrate(args,root)
            self.assertFalse(report['independently_calibrated'])
            self.assertEqual(report['provenance']['num_excluded_validation_core_proteins'],1)
            self.assertEqual(report['identity']['membership_teacher'],'backbone_all_frequencies')
            groups={g['frequency_bin']:g for g in report['groups']}
            for name in ['1-5','6-9','10-99','100-999','1000+']:
                self.assertTrue(groups[name]['threshold_fitted'])
                self.assertAlmostEqual(report['thresholds'][name],.101)
            self.assertFalse(groups['zero']['threshold_fitted'])
            before=(root/'thresholds.json').read_bytes()
            c.calibrate(args,root)
            self.assertEqual(before,(root/'thresholds.json').read_bytes())

    def test_backbone_membership_modelout_payload_and_empty_rows_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d,c,ca,ba,b,m=fixture(root)
            old={p:d.sha256(p) for folder in ['data','ontology','work'] for p in (root/folder).rglob('*') if p.is_file()}
            c.calibrate(ca,root);report=d.build_dataset(ba,root)
            folder=root/'new';ptr=np.load(folder/report['pseudo']['indptr'])
            go=np.load(folder/report['pseudo']['go_idx']);prob=np.load(folder/report['pseudo']['probability'])
            source=np.load(folder/report['pseudo']['source']);seen=np.load(folder/report['files']['seen_task_mask'])
            counts=np.load(folder/report['files']['core_counts'])
            group=d.frequency_groups(counts);t=np.array([report['policies']['thresholds'][d.GROUPS[i][0]] for i in group])
            for row in range(len(b)):
                expected=np.flatnonzero((b[row].astype(np.float64)>t)&seen)
                got=go[ptr[row]:ptr[row+1]]
                np.testing.assert_array_equal(got,expected)
                np.testing.assert_array_equal(prob[ptr[row]:ptr[row+1]],m[row,expected])
            self.assertEqual(ptr[0],ptr[1])
            self.assertTrue((prob<.5).any());self.assertIn(0.,prob)
            self.assertTrue((source==2).all());self.assertFalse(seen[5])
            np.testing.assert_array_equal(np.load(folder/report['files']['training_weak_ids']),[1,0,2])
            self.assertEqual(report['empty_hard_membership_weak_proteins'],1)
            self.assertEqual(report['soft_teacher']['sha256'],d.sha256(root/'data/weak/m.npy'))
            self.assertEqual(report['soft_teacher']['scope'],'all_seen_GO')
            self.assertIsNone(report['policies']['candidate_quotas'])
            hard=(b.astype(np.float64)>t[None])&seen[None]
            diagnostics=report['teacher_disagreement']
            self.assertEqual(diagnostics['backbone_hard_positive_modelout_below_0_5_pairs'],int((hard&(m<.5)).sum()))
            self.assertEqual(diagnostics['backbone_hard_negative_modelout_above_0_5_pairs'],int(((~hard)&seen[None]&(m>.5)).sum()))
            self.assertAlmostEqual(diagnostics['mean_modelout_on_backbone_hard_positives'],float(m[hard].astype(np.float64).mean()))
            self.assertEqual(old,{p:d.sha256(p) for p in old})
            self.assertEqual(d.build_dataset(ba,root)['identity'],report['identity'])

    def test_label_holdout_filters_hard_labels_candidates_and_calibration_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d,c,ca,ba,_,_=fixture(root)
            ca.label_holdout_fraction=.5;c.calibrate(ca,root)
            with self.assertRaisesRegex(ValueError,'supervision split differs'):
                d.build_dataset(ba,root)
            ba.label_holdout_fraction=.5;report=d.build_dataset(ba,root);folder=root/'new'
            seen=np.load(folder/report['files']['seen_task_mask'])
            go=np.load(folder/report['pseudo']['go_idx']);candidate=np.load(folder/report['candidates']['go'])
            self.assertTrue(seen[go].all());self.assertTrue(seen[candidate[candidate>=0]].all())
            mapping=np.load(folder/report['files']['task_to_ontology'])
            for row in np.unique(mapping):
                values=seen[mapping==row];self.assertTrue(values.all() or not values.any())

    def test_global_candidate_order_does_not_apply_frequency_quotas(self):
        with tempfile.TemporaryDirectory() as tmp:
            d,*_=fixture(Path(tmp))
            go=np.arange(20)[None]
            attrs=np.zeros((1,20,3),np.float32);attrs[0,:,2]=1/np.arange(1,21)
            counts=np.array([1000]*16+[1,2,10,20])
            chosen,_=d.choose_candidate_rows(go,attrs,counts,np.ones(20,bool))
            np.testing.assert_array_equal(chosen[0],np.arange(16))
            with self.assertRaisesRegex(ValueError,'quotas are removed'):
                d.choose_candidate_rows(go,attrs,counts,np.ones(20,bool),mode='frequency')

    def test_installed_cli_calibrate_build_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d,_,_,_,_,_=fixture(root)
            common=['--source-work-dir',str(root/'work'),'--chunk-rows','128']
            for script,extra in [('calibrate_nbs_backbone_v094.py',['--output',str(root/'thresholds.json')]),
                                 ('build_nbs_dataset_v094.py',['--output-dir',str(root/'new'),
                                                            '--thresholds-json',str(root/'thresholds.json')])]:
                done=subprocess.run([sys.executable,str(root/'scripts/nbs'/script),*common,*extra],capture_output=True,text=True)
                self.assertEqual(done.returncode,0,done.stderr)
            path=root/'new/unified_dataset_manifest.json';manifest=d.load_unified_manifest(path)
            payload=root/'new'/manifest['pseudo']['source']
            a=np.load(payload);a[0]=99;np.save(payload,a)
            with self.assertRaisesRegex(ValueError,'file hash mismatch'):
                d.load_unified_manifest(path)


if __name__=='__main__':
    unittest.main()
