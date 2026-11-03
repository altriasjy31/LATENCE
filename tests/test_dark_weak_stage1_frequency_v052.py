"""True native-v085 + original Stage-1 utilities regression, synthetic data only.

Run on original project: python -m unittest discover -s tests -p 'test_dark_weak_stage1_frequency_v052.py' -v
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('frequency_v052',ROOT/'scripts/dark_weak/eval_stage1_frequency_v052.py')
mod=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def store_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def fixture(root):
    rr=root/'unified'/'labels'/'bp';rr.mkdir(parents=True)
    gos=['GO:0000001','GO:0000002','GO:0000003','GO:0000004',
         'GO:0000005','GO:0000006','GO:0008150','GO:0000008']
    can=gos.copy();can[-1]=gos[0]
    (rr/'go_ids.txt').write_text('\n'.join(gos)+'\n')
    pd.DataFrame({'model_column':range(len(gos)),'model_go_id':gos,'canonical_go_id_2022':can}).to_csv(
        rr/'vocabulary_mapping.tsv',sep='\t',index=False)
    table=root/'unified'/'tables';table.mkdir(parents=True)
    pd.DataFrame({'protein_22':['A','B','C'],'sequence_sha256':['hA','hB','hB']}).to_csv(
        table/'full_dark_weak_cohort.tsv',sep='\t',index=False)
    training={'train':{'biological_process':{'proteins':[f'T{i}' for i in range(1,8)],
        'annotations':[[0,7,1],[0,2],[7,3],[4],[5],[6],[1,4]],
        'prop_annotations':[[0,7,1],[0,2],[7,3],[4],[5],[6],[1,4]]}}}
    train=root/'train.pkl';train.write_bytes(pickle.dumps(training))
    meta={}
    for name in ('dark_weak_test_exp_direct','dark_weak_test_ipr2go_direct','dark_weak_test_direct'):
        meta[name]={'biological_process':{'proteins':['A','B','C'],
            'annotations':[[2],[0,7],[5]],
            'prop_annotations':[[2,6],[0,7,6],[5,6]]}}
    mp=root/'eval.pkl';mp.write_bytes(pickle.dumps(meta))
    inp=root/'predictions';inp.mkdir()
    p0=np.array([[.8,.01,.11,.9,.7,.35,.04,.1],
                 [.9,.02,.06,.2,.15,.5,.5,.1],
                 [.5,.01,.1,.09,.8,.7,.2,.1]],np.float32)
    p1=p0.copy();p1[:,2]=.96;p1[:,5]=.99;p1[:,0]=.99;p1[:,7]=.99
    p2=(p1*.82+.05).astype(np.float32)
    input_files={}
    ids=inp/'protein_ids.txt';ids.write_text('A\nB\nC\n')
    go=inp/'go_ids.txt';go.write_text('\n'.join(gos)+'\n')
    for name, matrix in [('backbone',p0),('expert',p1),('modelout',p2)]:
        arrpath=inp/(name+'.npy');np.save(arrpath,matrix)
        input_files[name]={'probability':str(arrpath),'probability_sha256':mod.sha256(arrpath),
                          'protein_ids':str(ids),'go_ids':str(go)}
    suite=root/'original';suite.mkdir()
    summary={'namespace':'bp','metric_scope':'unified','metric_label_view':'propagated',
             'ranking_label_view':'direct', 'protocol':{'model_fusion':'none'},
             'cases':list(meta),'reference':'backbone','metadata':str(mp),
             'metadata_sha256':mod.sha256(mp),
             'vocabulary_sha256':mod.sha256(rr/'go_ids.txt'),
             'native_kernels':{'v085_native_standardized':{'metric_schema':'0.8.5-standard-metrics-v1'}},
             'input_files':input_files}
    manifest=suite/'bp_complete_manifest.json';store_json(manifest,summary)
    pd.DataFrame([{'split':k,'model':m,'Fmax_percent':99.0,'AuPRC_percent':98.0,'AP_percent':97.0}
       for k in meta for m in input_files]).to_csv(suite/'bp_complete_metrics.tsv',sep='\t',index=False)
    return manifest,train,root/'unified',inp,gos,summary


def config(manifest,train,unified,out,**extras):
    base={'complete_manifest':manifest,'train_metadata':train,'unified_root':unified,
          'train_split':'train','train_label_field':'annotations',
          'v094_unified_manifest':None,'families':['historical_classic','historical_fixed'],
          'splits':None,'metric_scope':'all','ks':[1,3,5],
          'bootstrap':12,'seed':6061,'skip_fmax_parity':False,'skip_hash_verify':False,
          'max_metric_cells':80_000_000,'allow_large_metrics':False,'out':out}
    base.update(extras)
    return SimpleNamespace(**base)


class FrequencyV052(unittest.TestCase):
    def test_frozen_native_metrics_and_rank_full_vocab_and_original_intact(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,train,unified,inp,gos,summary=fixture(root)
            before=mod.sha256(manifest)
            result,check=mod.run(config(manifest,train,unified,root/'strata'))
            self.assertEqual(before,mod.sha256(manifest))
            self.assertFalse(check['original_reports_modified'])
            self.assertEqual(len(result),3*2*12*3//2) # 3 splits, 2 families, 6 groups, 3 models
            self.assertEqual(set(result.model),{'backbone','expert','modelout'})
            self.assertEqual(check['num_GO_alias_extra_columns'],1)
            # Alias GO1 and GO8 are counted once per historical TRAIN protein.
            lut=pd.read_csv(root/'strata'/'go_frequency_lookup.tsv',sep='\t')
            self.assertEqual(int(lut.iloc[0].historical_train_canonical_unique_protein_count),3)
            self.assertEqual(int(lut.iloc[7].historical_train_canonical_unique_protein_count),3)
            self.assertEqual(int(lut.iloc[7].historical_train_raw_column_count),2)
            self.assertTrue((result[(result.bin=='rare_le_5')&(result.Fmax_percent.notna())].Fmax_percent>=0).all())
            rank=pd.read_csv(root/'strata'/'direct_positive_global_ranks.tsv',sep='\t')
            # GO3 (A) has score .11; with other high-scoring GO, cannot be rank 1.
            row=rank[(rank.protein=='A')&(rank.go_id=='GO:0000003')&(rank.model=='backbone')]
            self.assertGreater(int(row.iloc[0]['rank']),1)
            self.assertEqual(int(row.iloc[0]['hit_at_1']),0)
            # GO1/GO8 are a single rankable positive for B, not two.
            row=rank[(rank.protein=='B')&(rank.model=='backbone')&(rank.family=='historical_classic')]
            self.assertEqual(set(row.go_id),{'GO:0000001'})
            self.assertTrue((root/'strata'/'original_unstratified_metrics_copy.tsv').is_file())
            self.assertTrue((root/'strata'/'paired_rank_bootstrap_deltas.tsv').is_file())

    def test_prediction_hash_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,train,unified,inp,gos,s=fixture(root)
            old=np.load(inp/'expert.npy');old[0,0]=0.001;np.save(inp/'expert.npy',old)
            with self.assertRaisesRegex(ValueError,'prediction differs'):
                mod.run(config(manifest,train,unified,root/'output'))

    def test_go_column_order_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,train,unified,inp,gos,s=fixture(root)
            (inp/'go_ids.txt').write_text('\n'.join(gos[::-1])+'\n')
            with self.assertRaisesRegex(ValueError,'GO vocabulary order'):
                mod.run(config(manifest,train,unified,root/'output'))

    def test_wrong_output_exists_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,train,unified,inp,gos,s=fixture(root)
            out=root/'output';out.mkdir()
            with self.assertRaises(FileExistsError):
                mod.run(config(manifest,train,unified,out))

    def test_v094_native_frequency_integration(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,train,unified,inp,gos,s=fixture(root)
            ud=root/'native';ud.mkdir()
            counts=np.array([10,5,4,3,2,1,50,6],np.int64)
            np.save(ud/'core_counts.i64.npy',counts)
            reg=ud/'go_registry.tsv'
            pd.DataFrame({'go_idx':range(len(gos)), 'input_go_id':gos}).to_csv(reg,sep='\t',index=False)
            native={'schema':'nbs-weak-dataset-v094-r1','complete':True,'num_task_go':len(gos),
                    'files':{'core_counts':'core_counts.i64.npy'},
                    'hashes':{'core_counts.i64.npy':mod.sha256(ud/'core_counts.i64.npy')},
                    'sources':{'task_registry':{'path':str(reg),'sha256':mod.sha256(reg)}}}
            nativep=ud/'unified_dataset_manifest.json';store_json(nativep,native)
            summary, audit=mod.run(config(manifest,train,unified,root/'out',
                    v094_unified_manifest=nativep,
                    families=['v094_native_core_fixed'], metric_scope='ranking_only',bootstrap=0))
            self.assertEqual(set(summary.family),{'v094_native_core_fixed'})
            self.assertEqual(audit['v094_native_frequency_provenance']['canonical_alias_policy'].split(';')[0],
                             'maximum raw core count among canonical alias-equivalent columns')

    def test_no_future_label_used_to_construct_bins(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,train,unified,inp,gos,s=fixture(root)
            _,can,inv,sizes=mod.read_vocabulary(unified,'bp')
            a,b,n=mod.get_train_counts(train,'bp','train','annotations',len(gos),inv,len(can))
            self.assertEqual(int(b.sum()),11) # train proteins contribute 11 distinct (protein, canonical GO) pairs
            # Flip a future label; canonical historical training frequency does not change.
            p=Path(s['metadata']);m=pickle.loads(p.read_bytes())
            m['dark_weak_test_direct']['biological_process']['annotations'][0]=[4]
            p.write_bytes(pickle.dumps(m))
            a2,b2,n2=mod.get_train_counts(train,'bp','train','annotations',len(gos),inv,len(can))
            np.testing.assert_array_equal(b,b2)


if __name__=='__main__':unittest.main()
