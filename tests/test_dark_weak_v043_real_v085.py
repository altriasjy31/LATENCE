"""Contract tests against the REAL nbs_pg/full_task_metrics_v085.py.

Never mock its AP/PR-AUC values: that masked the field names and the 50% tie
artifact in v0.4.2. These tests must run from a LATENCE project with v085.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'nbs_models' / 'nbs_protein_go'))
from nbs_pg import full_task_metrics_v085 as real
from scripts.dark_weak.native_v085_metrics import (
    native_imports, native_protocol, unified_v085_metrics,
    validate_metric_contract, REQUIRED_SCHEMA,
)
from scripts.dark_weak import run_complete_evaluation as unified
from scripts.nbs import eval_nbs_full_task_v088 as v088


class RealMetricKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernels = native_imports(ROOT)

    def test_schema_and_keys_are_exact(self):
        self.assertEqual(native_protocol(self.kernels)['metric_schema'], REQUIRED_SCHEMA)
        y=np.asarray([[1,0,1],[0,1,0]], np.uint8)
        p=np.asarray([[.9,.3,.8],[.2,.85,.1]],np.float32)
        m=unified_v085_metrics(y,p,self.kernels)
        self.assertEqual(m['AuPRC_native_key'],'standard_micro_pr_auc')
        self.assertEqual(m['AP_native_key'],'standard_micro_ap')
        self.assertEqual(m['Fmax_percent'],m['standard_protein_fmax'])
        self.assertEqual(m['AuPRC_percent'],m['standard_micro_pr_auc'])
        self.assertEqual(m['AP_percent'],m['standard_micro_ap'])
        self.assertEqual(m['standard_threshold_step_used'],.001)
        for key in ('Fmax_percent','AuPRC_percent','AP_percent'):
            self.assertAlmostEqual(m[key],100.)

    def test_inverted_positive_and_uniform_prevalence(self):
        y=np.asarray([[1,0]],np.uint8);p=np.asarray([[.1,.9]],np.float32)
        m=unified_v085_metrics(y,p,self.kernels)
        self.assertAlmostEqual(m['AuPRC_percent'],25.)
        self.assertAlmostEqual(m['AP_percent'],50.)
        y=np.asarray([[1]+[0]*99],np.uint8)
        p=np.full((1,100),.5,np.float32)
        m=unified_v085_metrics(y,p,self.kernels)
        self.assertAlmostEqual(m['AuPRC_percent'],50.5)
        self.assertAlmostEqual(m['AP_percent'],1.)
        self.assertAlmostEqual(m['observed_positive_prevalence_percent'],1.)

    def test_real_chunk_boundary_ties_match_sklearn(self):
        rng=np.random.default_rng(132)
        for j in range(60):
            y=rng.integers(0,2,size=(2,13),dtype=np.uint8)
            p=(rng.integers(0,4,size=(2,13))/4).astype(np.float32)
            if y.sum()==0:continue
            precision,recall,_=precision_recall_curve(y.ravel(),p.ravel())
            target_auc=100*auc(recall,precision)
            target_ap=100*average_precision_score(y.ravel(),p.ravel())
            for chunk in (1,2,3,5,7):
                actual=real._exact_micro(y,p,rank_chunk=chunk)
                self.assertAlmostEqual(actual['standard_micro_pr_auc'],target_auc,places=9)
                self.assertAlmostEqual(actual['standard_micro_ap'],target_ap,places=9)

    def test_bad_schema_or_missing_fields_fail_closed(self):
        bad=dict(self.kernels,definitions=lambda:dict(schema='future_version',micro_pr_auc='trapezoidal'))
        with self.assertRaisesRegex(RuntimeError,'schema'):
            validate_metric_contract(bad)
        incomplete=dict(self.kernels,compute=lambda *a,**kw: {'standard_protein_fmax':50.})
        with self.assertRaisesRegex(KeyError,'required fields'):
            unified_v085_metrics(np.array([[1,0]],np.uint8),np.array([[.8,.1]],np.float32),incomplete)

    def test_mismatched_fmax_parity_fails(self):
        def wrong(y,p,threshold_step=.001):
            ans=real.compute_standard_metrics(y,p,threshold_step)
            ans['standard_protein_fmax']-=.1
            return ans
        bad=dict(self.kernels,compute=wrong)
        with self.assertRaisesRegex(AssertionError,'parity'):
            unified_v085_metrics(np.array([[1,0]],np.uint8),np.array([[.8,.1]],np.float32),bad)


class RealNativeEntrypointTests(unittest.TestCase):
    def test_native_v088_fmax_auprc_ap_four_models(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            y=np.array([[1,0,1],[0,1,0]],dtype=np.uint8)
            files={}
            for key, scores in {'nbs':[[.9,.1,.8],[.2,.8,.3]],
                                'backbone':[[.5,.1,.8],[.3,.5,.2]],
                                'expert':[[.8,.2,.8],[.2,.5,.3]],
                                'modelout':[[.8,.1,.9],[.1,.5,.4]]}.items():
                f=root/f'{key}.npy';np.save(f,np.asarray(scores,np.float32));files[key]=f
            for name,text in {'protein_ids.txt':'A\nB\n',
                              'go.txt':'GO:0000001\nGO:0000002\nGO:0000003\n',
                              'meta.pkl':'unused','pred.json':'{}',
                              'input.json':json.dumps({'leakage_contract':{}})}.items():
                (root/name).write_text(text)
            def make_args(extra=[]):
                return v088.parser().parse_args([
                    '--task','bp','--metadata-file',str(root/'meta.pkl'),
                    '--nbs-prob',str(files['nbs']), '--backbone-prob',str(files['backbone']),
                    '--output-dir',str(root/'out'), '--input-manifest',str(root/'input.json'),
                    '--prediction-manifest',str(root/'pred.json'),
                    '--protein-ids',str(root/'protein_ids.txt'),
                    '--expert-prob',str(files['expert']), '--modelout-prob',str(files['modelout']),
                    '--references-aligned-to-input']+extra)
            class Legacy:
                @staticmethod
                def _sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
                @staticmethod
                def _hash_ids(ids):return 'test-ids-hash'
                @staticmethod
                def _align_rows(mat,prediction_ids,metadata_ids):
                    assert prediction_ids==metadata_ids
                    return mat
                @staticmethod
                def _load_prediction(path,width):
                    ans=np.load(path)
                    assert ans.shape[1]==width
                    return ans
            manifest=dict(num_task_go=3,prediction_space='complete_task_classifier_columns',
                          uses_expert_probability_in_nbs_forward=False,
                          uses_modelout_probability_as_target=False,step=20)
            context=(patch.object(v088,'_legacy',return_value=Legacy()),
                     patch.object(v088,'_prediction_provenance',return_value={'manifest':manifest,'step':20}),
                     patch.object(v088,'_registry',return_value=(root/'go.txt',['GO:0000001','GO:0000002','GO:0000003'])),
                     patch.object(v088,'_labels',return_value=(y,['A','B'])))
            with context[0],context[1],context[2],context[3]:
                scored=v088.evaluate_standard_only(make_args())
                old=v088.evaluate_standard_only(make_args(['--fmax-only']))
            self.assertEqual(set(scored['methods']),{'NBS_final','backbone_base','expert_prob','stage1_modelout'})
            for method in scored['methods'].values():
                metrics=method['standardized']
                self.assertIn('AP_percent',metrics)
                self.assertEqual(metrics['AuPRC_native_key'],'standard_micro_pr_auc')
            self.assertAlmostEqual(scored['methods']['NBS_final']['standardized']['Fmax_percent'],
                                   old['methods']['NBS_final']['standardized']['standard_protein_fmax'])
            self.assertIn('AP_percent',scored['deltas']['G_minus_B']['standardized'])

    def test_unified_suite_reports_ap_and_dtype_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            vocab=root/'unified'/'labels'/'bp';vocab.mkdir(parents=True)
            gos=['GO:0000001','GO:0000002','GO:0000003']
            (vocab/'go_ids.txt').write_text('\n'.join(gos)+'\n')
            pd.DataFrame({'model_column':[0,1,2], 'model_go_id':gos,
                          'canonical_go_id_2022':gos}).to_csv(vocab/'vocabulary_mapping.tsv',sep='\t',index=False)
            cohort=root/'unified'/'tables';cohort.mkdir(parents=True)
            pd.DataFrame({'protein_22':['A','B'],
                          'sequence_sha256':['aaa','bbb']}).to_csv(
                          cohort/'full_dark_weak_cohort.tsv',sep='\t',index=False)
            split='dark_weak_test_exp_direct'
            metadata={split:{'biological_process':{'proteins':['A','B'],
                       'annotations':[[1],[2]],'prop_annotations':[[0,1],[0,2]]}}}
            mpath=root/'metadata.pkl'
            with mpath.open('wb') as f:pickle.dump(metadata,f)
            inp=root/'inputs';inp.mkdir()
            np.save(inp/'B.npy',np.array([[.6,.85,.1],[.5,.1,.9]],np.float16))
            np.save(inp/'G.npy',np.array([[.5,.9,.1],[.4,.1,.95]],np.float32))
            (inp/'ids.txt').write_text('A\nB\n')
            (inp/'go.txt').write_text('\n'.join(gos)+'\n')
            args=argparse.Namespace(namespace='bp',metadata=mpath,unified_root=root/'unified',
                panel='future',split=split,label_view='auto',metric_label_view=None,
                ranking_label_view=None,
                prediction=[f'B={inp/"B.npy"}',f'G={inp/"G.npy"}'],
                prediction_ids=[f'B={inp/"ids.txt"}',f'G={inp/"ids.txt"}'],
                prediction_go_ids=[f'B={inp/"go.txt"}',f'G={inp/"go.txt"}'],
                reference='B',ks=[1,2],bootstrap=3,seed=4,save_pair_ranks=False,
                allow_descriptive_labels=False,include_roots=True,
                metric_scope='unified',auprc_mode='exact',stage1_threshold_step=.01,
                max_metric_cells=10000,allow_large_metrics=False,skip_fmax_parity=False,
                strict_prediction_dtype=False,out=root/'out')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                unified.run(args)
            self.assertTrue(any('mixed storage dtypes' in str(w.message) for w in caught))
            report=pd.read_csv(root/'out'/'bp_complete_metrics.tsv',sep='\t')
            self.assertTrue({'Fmax_percent','AuPRC_percent','AP_percent','observed_positive_prevalence_percent'}
                            <=set(report.columns))
            scored=json.loads((root/'out'/'bp'/split/'complete_metrics.json').read_text())
            self.assertEqual(scored['metric_label_field'],'prop_annotations')
            self.assertEqual(scored['ranking_label_field'],'annotations')
            self.assertEqual(scored['num_positive_model_columns'],4)
            self.assertEqual(scored['metrics']['B']['v085_standardized']['AP_native_key'],'standard_micro_ap')
            self.assertEqual(scored['metrics']['G']['delta_vs_reference']['v085_standardized']['AP_percent'],
                             scored['metrics']['G']['v085_standardized']['AP_percent'] -
                             scored['metrics']['B']['v085_standardized']['AP_percent'])
            args.strict_prediction_dtype=True
            with self.assertRaisesRegex(ValueError,'mixed storage dtypes'):
                unified.run(args)

if __name__=='__main__':unittest.main()
