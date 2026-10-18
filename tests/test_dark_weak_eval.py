import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
# The production package already includes build_dark_weak_test.py. During
# isolated unit testing, the source is mirrored into scripts/ by a test fixture
# (see below) without touching the production installation.
import build_dark_weak_eval_metadata as bm
import eval_dark_weak_predictions as ev
import export_dark_weak_stage1_arrays as exporter

OBO='''format-version: 1.2
[Term]
id: GO:0008150
name: biological_process
namespace: biological_process
[Term]
id: GO:0000001
name: biological process A
namespace: biological_process
alt_id: GO:0000009
is_a: GO:0008150 ! biological process
[Term]
id: GO:0000002
name: biological process B
namespace: biological_process
is_a: GO:0000001 ! BP A
[Term]
id: GO:0000005
name: off-model BP child
namespace: biological_process
is_a: GO:0000001 ! BP A
[Term]
id: GO:0003674
name: molecular function
namespace: molecular_function
[Term]
id: GO:0000003
name: molecular function A
namespace: molecular_function
is_a: GO:0003674 ! MF
[Term]
id: GO:0005575
name: cellular_component
namespace: cellular_component
[Term]
id: GO:0000004
name: cellular component A
namespace: cellular_component
is_a: GO:0005575 ! CC
'''

class TestDarkWeakEval(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory()
        self.tmp=Path(self.dir.name)
        self.root=self.tmp/'unified'
        self.obo=self.tmp/'go.obo'
        self.obo.write_text(OBO)
        self.root.mkdir()
        self.ids=['P1','P2','P3','P4','P5']
        (self.root/'protein_ids.txt').write_text('\n'.join(self.ids)+'\n')
        (self.root/'unified_manifest.json').write_text(json.dumps({'propagation':'deepgozero','ontology_2022':'data/go.obo'}))
        (self.root/'tables').mkdir()
        pd.DataFrame({'protein_22':self.ids,'valid_future':[True,True,True,True,False],
                      'sequence_sha256':['h1','h2','h3','h4','h5']}).to_csv(self.root/'tables'/'full_dark_weak_cohort.tsv',sep='\t',index=False)
        self.gos={'bp':['GO:0000001','GO:0000009','GO:0000002','GO:0008150'],
                  'mf':['GO:0000003','GO:0003674'],
                  'cc':['GO:0000004','GO:0005575']}
        self.canon={'bp':['GO:0000001','GO:0000001','GO:0000002','GO:0008150'],
                    'mf':self.gos['mf'],'cc':self.gos['cc']}
        for ns in bm.NS:
            fold=self.root/'labels'/ns
            fold.mkdir(parents=True)
            (fold/'go_ids.txt').write_text('\n'.join(self.gos[ns])+'\n')
            pd.DataFrame({'model_column':list(range(len(self.gos[ns]))),
                          'model_go_id':self.gos[ns],
                          'canonical_go_id_2022':self.canon[ns]}).to_csv(fold/'vocabulary_mapping.tsv',sep='\t',index=False)
        def mat(ns,positives):
            rr=[];cc=[]
            for row, cols in positives.items():
                for col in cols:
                    rr.append(row);cc.append(col)
            return sparse.csr_matrix((np.ones(len(rr),dtype=np.uint8),(rr,cc)),shape=(5,len(self.gos[ns])))
        # Future union: P2 and P3 BP, P4 MF; P5 must NOT get a future label.
        fake={
          'future_union_novel_direct':{'bp':{1:[0,1],2:[2]},'mf':{3:[0]},'cc':{}},
          'future_experimental_crosssource_novel_direct':{'bp':{1:[0,1]},'mf':{},'cc':{}},
          'future_ipr2go_novel_direct':{'bp':{2:[2]},'mf':{3:[0]},'cc':{}},
          'historical_ipr2go_direct':{'bp':{0:[0,1]},'mf':{0:[0]},'cc':{}},
          'historical_ipr2go_matched_direct':{'bp':{0:[0,1]},'mf':{0:[0]},'cc':{}},
          'future_ipr2go_all_direct':{'bp':{2:[2]},'mf':{3:[0]},'cc':{}},
          'future_ipr2go_delta_direct':{'bp':{2:[2]},'mf':{3:[0]},'cc':{}},
          'future_ipr2go_retained_direct':{'bp':{0:[0,1]},'mf':{},'cc':{}},
          'future_ipr2go_removed_direct':{'bp':{0:[0,1]},'mf':{},'cc':{}},
        }
        # One future GO outside the trained model's direct vocabulary can
        # still produce in-vocabulary ancestors on propagation.
        pd.DataFrame([
            {'rid22':1,'canonical_go_2022':'GO:0000001',
             'evaluation_union_label':True,'experimental_crosssource_novel':True, 'interpro2go_crosssource_novel':False},
            {'rid22':2,'canonical_go_2022':'GO:0000002',
             'evaluation_union_label':True,'experimental_crosssource_novel':False,'interpro2go_crosssource_novel':True},
            {'rid22':3,'canonical_go_2022':'GO:0000003',
             'evaluation_union_label':True,'experimental_crosssource_novel':False,'interpro2go_crosssource_novel':True},
            {'rid22':3,'canonical_go_2022':'GO:0000005',
             'evaluation_union_label':True,'experimental_crosssource_novel':False,'interpro2go_crosssource_novel':True},
        ]).to_csv(self.root/'tables'/'unified_direct_positive_pairs.tsv',sep='\t',index=False)
        pd.DataFrame([
            {'rid22':0,'go_id_2022':'GO:0000001'},
            {'rid22':0,'go_id_2022':'GO:0000003'},
        ]).to_csv(self.root/'tables'/'ipr2go_2022_baseline_pairs.tsv',sep='\t',index=False)
        pd.DataFrame([
            {'rid22':0,'go_id_2022':'GO:0000001','temporal_status':'retained'},
            {'rid22':2,'go_id_2022':'GO:0000002','temporal_status':'added'},
            {'rid22':3,'go_id_2022':'GO:0000003','temporal_status':'added'},
            {'rid22':0,'go_id_2022':'GO:0000001','temporal_status':'removed'},
        ]).to_csv(self.root/'tables'/'ipr2go_temporal_status_pairs.tsv',sep='\t',index=False)
        # All required propagated matrices exist; this is a synthetic case.
        for view, key in bm.VIEW_TO_SOURCE.items():
            for ns in bm.NS:
                sparse.save_npz(self.root/'labels'/ns/f'{key}.npz',mat(ns,fake[key][ns]))
        for v,src in bm.PROPAGATED_SOURCES.items():
            key=bm.VIEW_TO_SOURCE[v]
            for ns in bm.NS:
                x=mat(ns,fake[key][ns])
                sparse.save_npz(self.root/'labels'/ns/f'{src}.npz',x)
        self.orig={'train':{'biological_process':{'proteins':['TRAIN'],'prop_annotations':[[1]]}},
                   'ind_test':{'biological_process':{'proteins':['OLD'],'prop_annotations':[[2]]}}}
        self.original=self.tmp/'original.pkl'
        self.output=self.tmp/'augmented.pkl'
        self.manifest=self.tmp/'manifest.json'
        with self.original.open('wb') as f:pickle.dump(self.orig,f)

    def tearDown(self):
        self.dir.cleanup()

    def build(self):
        from argparse import Namespace
        args=Namespace(unified_root=self.root,metadata=self.original,output=self.output,
                       manifest=self.manifest,go_obo_2022=self.obo,
                       bp_terms=self.root/'labels'/'bp'/'go_ids.txt',
                       mf_terms=self.root/'labels'/'mf'/'go_ids.txt',
                       cc_terms=self.root/'labels'/'cc'/'go_ids.txt',
                       groups=['future','history','diagnostic'],create_empty=False,
                       overwrite=False,replace_existing_splits=False)
        return bm.build(args)

    def test_all_metadata_splits_and_propagated_contract(self):
        report=self.build()
        with self.output.open('rb') as f:data=pickle.load(f)
        self.assertEqual(data['train'],self.orig['train'])
        self.assertEqual(data['ind_test'],self.orig['ind_test'])
        d=data['dark_weak_test_direct']['biological_process']
        self.assertEqual(d['proteins'],['P2','P3'])
        self.assertEqual(d['annotations'],[[0,1],[2]])
        # *_direct DOES NOT change the meaning of prop_annotations!
        self.assertEqual(d['prop_annotations'],[[0,1,3],[0,1,2,3]])
        p=data['dark_weak_test']['biological_process']
        self.assertEqual(p['proteins'],['P2','P3','P4'])
        self.assertEqual(p['prop_annotations'],[[0,1,3],[0,1,2,3],[0,1,3]])
        self.assertEqual(p['annotations'],[[0,1],[2],[]])
        self.assertEqual(data['dark_weak_test']['molecular_function']['proteins'],['P4'])
        self.assertEqual(data['dark_weak_test_2022_ipr2go_direct']['biological_process']['proteins'],['P1'])
        self.assertEqual(report['splits']['dark_weak_test']['row_selection'],'prop')
        self.assertEqual(len(data['dark_weak_test']['cellular_component']['proteins']),0)

    def test_prop_annotations_have_full_ancestor_closure_for_every_split(self):
        self.build()
        from build_dark_weak_test import Ontology
        ont=Ontology(self.obo,mode='deepgozero')
        with self.output.open('rb') as f: metadata=pickle.load(f)
        for split,data in metadata.items():
            if not split.startswith('dark_weak_test'):
                continue
            for ns,namespace in bm.NS.items():
                records=data[namespace]
                direct=records['annotations']
                propagated=records['prop_annotations']
                self.assertEqual(len(direct),len(propagated),split)
                go_ids=self.gos[ns]
                canonical=self.canon[ns]
                for orig,prop in zip(direct,propagated):
                    actual={canonical[i] for i in prop}
                    # For each direct term, every ancestor of that term which
                    # exists in this namespace's model vocabulary must be kept.
                    for j in orig:
                        need=ont.propagate({canonical[j]}) & set(canonical)
                        self.assertLessEqual(need,actual,(split,ns,j))
                    self.assertEqual(len(prop),len(set(prop)))

    def test_future_row_must_be_sequence_identical(self):
        path=self.root/'labels'/'bp'/'future_union_novel_direct.npz'
        x=sparse.load_npz(path).tolil();x[4,2]=1;sparse.save_npz(path,x.tocsr())
        with self.assertRaisesRegex(ValueError,'non-identical'):
            self.build()

    def test_bad_vocab_order(self):
        file=self.root/'labels'/'bp'/'go_ids.txt'
        bad=self.tmp/'bad.txt';bad.write_text('GO:0000002\nGO:0000009\nGO:0000001\nGO:0008150\n')
        orig=bm.load_matrices
        with self.assertRaisesRegex(ValueError,'GO order differs'):
            bm.load_matrices(self.root,{'future_union_novel_direct'},{'bp':bad})

    def test_canonical_ranking_and_id_alignment(self):
        self.build()
        from argparse import Namespace
        # Prediction row order is intentionally reversed.
        names=list(reversed(self.ids))
        path=self.tmp/'scores.npy';idfile=self.tmp/'protein_ids.txt';gofile=self.tmp/'go.txt'
        idfile.write_text('\n'.join(names)+'\n')
        gofile.write_text('\n'.join(self.gos['bp'])+'\n')
        m=np.zeros((5,4),np.float32)
        for i,name in enumerate(names):
            if name=='P2':m[i]=[.9,.8,.1,.05]
            elif name=='P3':m[i]=[.1,.2,.95,.05]
        np.save(path,m)
        args=Namespace(metadata=self.output,unified_root=self.root,split='dark_weak_test_direct',
             namespace='bp',prediction=[f'model={path}'],prediction_ids=[f'model={idfile}'],
             prediction_go_ids=[f'model={gofile}'],assume_go_order=False,reference=None,
             ks=[1],bootstrap=20,seed=1,out=self.tmp/'metrics',save_pair_ranks=True,
             allow_descriptive_labels=False,exclude_roots=True)
        metric=ev.run(args)
        self.assertEqual(metric['model_metrics']['model']['n_proteins'],2)
        self.assertEqual(metric['model_metrics']['model']['n_confirmed_positive_pairs'],2)
        self.assertEqual(metric['model_metrics']['model']['macro_recall_at_1'],1)
        wrong=self.tmp/'reversedgo.txt';wrong.write_text('GO:0000002\nGO:0000009\nGO:0000001\nGO:0008150\n')
        args.prediction_go_ids=[f'model={wrong}']
        with self.assertRaisesRegex(ValueError,'GO column order'):
            ev.run(args)

    def test_dense_export_with_explicit_order_and_probabilities(self):
        self.build()
        from argparse import Namespace
        ids=list(reversed(self.ids))
        idpath=self.tmp/'reversed_ids.txt'
        idpath.write_text('\n'.join(ids)+'\n')
        gopath=self.tmp/'bp_vocab.txt'
        gopath.write_text('\n'.join(self.gos['bp'])+'\n')
        predpath=self.tmp/'raw.npy'
        scores=np.zeros((5,4),dtype=np.float32)
        scores[ids.index('P2')]=[0.,0.,-2.,-1.]
        scores[ids.index('P3')]=[-2.,-2.,3.,-1.]
        np.save(predpath,scores)
        args=Namespace(metadata=self.output,unified_root=self.root,
                       split='dark_weak_test_direct',namespace='bp',
                       prediction=[f'model={predpath}'],
                       prediction_ids=[f'model={idpath}'],
                       prediction_go_ids=[f'model={gopath}'],
                       logit_models=['model'],batch_size=2,out=self.tmp/'export',overwrite=False)
        info=exporter.run(args)
        y=np.load(info['labels_file'])
        probs=np.load(info['probability_files']['model'])
        self.assertEqual(y.shape,(2,4))
        self.assertEqual(y.tolist(),[[1,1,0,0],[0,0,1,0]])
        self.assertAlmostEqual(float(probs[0,0]),0.5,places=3)
        self.assertGreater(float(probs[1,2]),float(probs[1,0]))
        self.assertEqual((args.out/'protein_ids.txt').read_text().splitlines(),['P2','P3'])

    def test_paired_bootstrap(self):
        a=pd.DataFrame({'protein':['a','b','c'],'sequence_sha256':['hash1','hash1','hash2'],
                        'n_positive':[1,1,1],'recall_at_1':[1.,1.,1.],
                        'best_rr':[1.,1.,1.]})
        b=a.copy();b['recall_at_1']=[0.,0.,1.];b['best_rr']=[.5,.5,1.]
        result=ev.paired_bootstrap(a,b,[1],reps=20,seed=1)
        self.assertGreater(result['delta_recall_at_1']['estimate'],0)
        self.assertEqual(result['delta_best_rr']['bootstrap_unit'],'2022 sequence SHA256')

if __name__=='__main__':
    unittest.main()
