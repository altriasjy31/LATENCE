import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dark_weak.prepare_inference_split import build as prepare
from dark_weak.export_nbs_v088_predictions import preflight
from dark_weak.run_complete_evaluation import standardized_fmax


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.r=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_inference_metadata_is_label_free(self):
        d={'train':{'biological_process':{'proteins':['T'],'prop_annotations':[[2]]}},
           'dark_weak_test_2022_ipr2go_matched_direct':{},
           'dark_weak_test_direct':{}}
        for ns in ('biological_process','molecular_function','cellular_component'):
            d['dark_weak_test_2022_ipr2go_matched_direct'][ns]={'proteins':['A'],'annotations':[[4]],'prop_annotations':[[0,4]]}
            d['dark_weak_test_direct'][ns]={'proteins':['C'],'annotations':[[3]],'prop_annotations':[[1,3]]}
        source=self.r/'meta.pkl'
        with source.open('wb') as f:pickle.dump(d,f)
        unified=self.r/'unified';(unified/'tables').mkdir(parents=True)
        pd.DataFrame({'protein_22':['C','A','B']}).to_csv(unified/'tables'/'full_dark_weak_cohort.tsv',sep='\t',index=False)
        args=SimpleNamespace(metadata=source,output=self.r/'new.pkl',unified_root=unified,
            id_dir=self.r/'ids',split='dark_weak_inference',source_splits=[
               'dark_weak_test_2022_ipr2go_matched_direct','dark_weak_test_direct'],
            full_cohort=False,overwrite=False,replace_split=False)
        prepare(args)
        with args.output.open('rb') as f:aug=pickle.load(f)
        self.assertEqual(aug['train'],d['train'])
        for ns in ('biological_process','molecular_function','cellular_component'):
            test=aug['dark_weak_inference'][ns]
            self.assertEqual(test['proteins'],['C','A'])
            self.assertEqual(test['annotations'],[[],[]])
            self.assertEqual(test['prop_annotations'],[[],[]])

    def make_v088_input(self):
        x=self.r/'input';x.mkdir()
        (x/'protein_ids.txt').write_text('A\nB\n')
        (x/'ind_test_input_manifest.json').write_text(json.dumps({'registries':{'go_registry':'go_registry.tsv'},'leakage_contract':{}}))
        num=21312
        ids=[f'GO:{i:07d}' for i in range(num)]
        reg=pd.DataFrame({'go_idx':list(range(num)),'input_go_id':ids})
        reg.to_csv(x/'go_registry.tsv',sep='\t',index=False)
        np.save(x/'ind_test_repr.f16.npy',np.zeros((2,16),dtype=np.float16))
        np.save(x/'backbone_ind_test_prob.f16.npy',np.zeros((2,num),dtype=np.float16))
        np.save(x/'candidate_go_index.i32.npy',np.zeros((2,4),dtype=np.int32))
        (self.r/'desired.txt').write_text('A\nB\n')
        (self.r/'go.txt').write_text('\n'.join(ids)+'\n')
        return SimpleNamespace(input_dir=x,expected_protein_ids=self.r/'desired.txt',
                               go_ids=self.r/'go.txt',task='bp')

    def test_nbs_preflight_protein_go_order_and_leakage(self):
        args=self.make_v088_input()
        indir,_=preflight(args)
        self.assertEqual(indir,args.input_dir)
        args.expected_protein_ids.write_text('B\nA\n')
        with self.assertRaisesRegex(ValueError,'order'):
            preflight(args)
        args.expected_protein_ids.write_text('A\nB\n')
        manifest=json.loads((args.input_dir/'ind_test_input_manifest.json').read_text())
        manifest['leakage_contract']={'test_to_test_edges':True}
        (args.input_dir/'ind_test_input_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'leaked'):
            preflight(args)

    def test_native_protein_fmax_output_contract(self):
        true=np.array([[1,0],[0,1]],np.uint8)
        scores=np.array([[.8,.1],[.2,.9]],np.float32)
        def validate(y,p):return y,p
        def grid(step):return np.array([0,.5,1],dtype=np.float32)
        def curve(y,p,t):
            prec=[];rec=[];f=[]
            for threshold in t:
                active=p>=threshold
                tp=float(np.sum(active & (y>0)))
                px=float(active.sum());positives=float(y.sum())
                p0=tp/px if px else 0.;r0=tp/positives
                prec.append(p0);rec.append(r0);f.append(2*p0*r0/max(p0+r0,1e-10))
            return np.array(f),np.array(prec),np.array(rec),len(y)
        d=standardized_fmax(true,scores,(validate,grid,curve,lambda:{},lambda:'abc'))
        self.assertAlmostEqual(d['standard_protein_fmax'],100)
        self.assertEqual(d['standard_num_positive_labels'],2)

if __name__=='__main__':unittest.main()
