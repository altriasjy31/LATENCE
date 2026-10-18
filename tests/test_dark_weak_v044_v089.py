"""Synthetic *native v089* provenance/Fmax/PR-AUC regression without GPU."""
from __future__ import annotations
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'nbs_models'/'nbs_protein_go'/'nbs_pg'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

LEGACY='''
import hashlib
from pathlib import Path
import numpy as np

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _hash_ids(ids):
    return hashlib.sha256(('\\n'.join(map(str,ids))+'\\n').encode()).hexdigest()
def _task_dict(meta,split,task):
    return meta[split][{'bp':'biological_process','mf':'molecular_function','cc':'cellular_component'}[task]]
def _dense_labels(ann,width,rows):
    y=np.zeros((rows,width),dtype=np.uint8)
    for i, cols in enumerate(ann):y[i,cols]=1
    return y
def _load_prediction(path,width):
    p=np.load(path,mmap_mode='r')
    if p.ndim!=2 or p.shape[1]!=width: raise ValueError('prediction shape')
    return p
def _align_rows(values,prediction_ids,metadata_ids):
    ix={name:i for i,name in enumerate(prediction_ids)}
    return np.asarray(values[[ix[name] for name in metadata_ids]],dtype=np.float32)
'''


class NativeV089Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'scripts/nbs').mkdir(parents=True)
        (self.root/'scripts/dark_weak').mkdir(parents=True)
        (self.root/'nbs_models/nbs_protein_go/nbs_pg').mkdir(parents=True)
        for folder in ('scripts','scripts/nbs','scripts/dark_weak','nbs_models','nbs_models/nbs_protein_go','nbs_models/nbs_protein_go/nbs_pg'):
            (self.root/folder/'__init__.py').write_text('')
        shutil.copyfile(ROOT/'scripts/nbs/eval_nbs_full_task_v089.py',self.root/'scripts/nbs/eval_nbs_full_task_v089.py')
        shutil.copyfile(ROOT/'scripts/dark_weak/native_v085_metrics.py',self.root/'scripts/dark_weak/native_v085_metrics.py')
        shutil.copyfile(ROOT/'scripts/dark_weak/run_complete_evaluation.py',self.root/'scripts/dark_weak/run_complete_evaluation.py')
        shutil.copyfile(ROOT/'scripts/eval_dark_weak_predictions.py',self.root/'scripts/eval_dark_weak_predictions.py')
        shutil.copyfile(SRC/'full_task_metrics_v085.py',self.root/'nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py')
        shutil.copyfile(SRC/'full_task_phase_v089.py',self.root/'nbs_models/nbs_protein_go/nbs_pg/full_task_phase_v089.py')
        (self.root/'scripts/nbs/eval_nbs_ind_test_predictions.py').write_text(LEGACY)
        self.ids=['A_001','B_002']
        self.go=[f'GO:{i:07d}' for i in range(1,21313)]
        (self.root/'ids.txt').write_text('\n'.join(self.ids)+'\n')
        import csv
        with (self.root/'go_registry.tsv').open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t'); w.writerow(['go_idx','input_go_id'])
            for i,g in enumerate(self.go): w.writerow([i,g])
        with (self.root/'labels.pkl').open('wb') as f:
            pickle.dump({'ind_test':{'biological_process':{'proteins':self.ids,'prop_annotations':[[0,1],[0,3]]}}},f)
        self.b=np.zeros((2,21312),np.float32);self.g=np.zeros((2,21312),np.float32)
        self.b[0,:2]=[.5,.6];self.b[1,[0,3]]=[.5,.8]
        self.g[0,:2]=[.9,.9];self.g[1,[0,3]]=[.9,.9]
        np.save(self.root/'B.npy',self.b);np.save(self.root/'G.npy',self.g)
        self.checkpoint=self.root/'model.pt';self.checkpoint.write_bytes(b'checkpoint')
        self.inp=self.root/'ind_test_input_manifest.json'
        # tiny no-leakage input; SHA bound to the output manifest
        self.inp.write_text(json.dumps({'registries':{'go_registry':str(self.root/'go_registry.tsv'),
            'go_registry_sha256':digest(self.root/'go_registry.tsv')},
            'protein_ids_file_sha256':digest(self.root/'ids.txt'),
            'base_probability':{'sha256':digest(self.root/'B.npy')},
            'leakage_contract':{k:False for k in ('ind_test_labels_in_representation','ind_test_labels_in_candidate_selection',
                   'ind_test_labels_in_pp_retrieval','expert_probability_in_forward_inputs','test_to_test_edges')}}))
        self.manifest=self.root/'prediction_manifest.json'
        phase=dict(origin_epochs=0,target_epochs=2,origin_weak_exposures=0,
                   origin_core_exposures=0,origin_optimizer_updates=0,steps_per_epoch=3,
                   n_weak=5,n_core=5,global_core_batch=2,global_weak_batch=2)
        self.manifest_data={
            'runner_version':'0.8.9.1','version':'0.8.9.1','branch':'final',
            'variant':'dynamic','step':3,'epoch':1,'ablation':'full',
            'model_architecture_version':'0.8.8','data_architecture_version':'0.8.8',
            'encoder_variant':'hetero_tuned','model_config':{'encoder_variant':'hetero_tuned','prediction_mode':'direct'},
            'sampler_config':{'mode':'dynamic'},'prediction_mode':'direct',
            'uses_dense_backbone_logits_in_forward':False,'output_fusion':'none',
            'uses_expert_probability_in_nbs_forward':False,
            'uses_modelout_probability_in_nbs_forward':False,
            'uses_modelout_probability_as_target':False,
            'source_flags':{x:True for x in ('query_candidate','neighbor_candidate','neighbor_pseudo','core','pp')},
            'num_task_go':21312,'prediction_space':'complete_task_classifier_columns',
            'checkpoint_path':str(self.checkpoint),'checkpoint_sha256':digest(self.checkpoint),
            'input_manifest_sha256':digest(self.inp),'output_probability_sha256':digest(self.root/'G.npy'),
            'phase_plan':phase,'phase_origin':{'initialization':'fresh','source_checkpoint':None,
                'source_checkpoint_sha256':None,'source_runner_version':None,
                'origin_epochs':0,'origin_weak_exposures':0,'origin_core_exposures':0,'origin_optimizer_updates':0,
                'source_phase_step':0,'source_world_size':0,'source_optimizer_state_reused':False,
                'source_rng_state_reused':False,'new_training_seed':5},
            'phase_step':3,'phase_epoch':1,'phase_step_in_epoch':3,'cumulative_completed_epochs':1,
            'lifetime_weak_exposures':5,'lifetime_core_exposures':6,'lifetime_optimizer_updates':3,
        }
        self.manifest.write_text(json.dumps(self.manifest_data))

    def tearDown(self):self.temp.cleanup()

    def execute(self,fmax_only=False):
        out=self.root/('fmax_only' if fmax_only else 'standard')
        args=[sys.executable,str(self.root/'scripts/nbs/eval_nbs_full_task_v089.py'),
            '--task','bp','--metadata-file',str(self.root/'labels.pkl'),
            '--nbs-prob',str(self.root/'G.npy'),'--backbone-prob',str(self.root/'B.npy'),
            '--output-dir',str(out),'--input-manifest',str(self.inp),
            '--prediction-manifest',str(self.manifest),'--protein-ids',str(self.root/'ids.txt')]
        if fmax_only:args+=['--fmax-only']
        return subprocess.run(args,cwd=self.root,text=True,capture_output=True)

    def test_fmax_auprc_ap_consistent_with_old_fmax(self):
        r=self.execute()
        self.assertEqual(r.returncode,0,r.stderr)
        summary=json.loads((self.root/'standard/nbs_w2s_comparison.json').read_text())
        self.assertEqual(summary['evaluation_version'],'0.8.9.1')
        self.assertEqual(summary['phase_identity']['lifetime_optimizer_updates'],3)
        self.assertEqual(summary['phase_identity']['epoch'],1)
        g=summary['methods']['NBS_final']['standardized']
        self.assertIn('standard_micro_pr_auc',g)
        self.assertIn('standard_micro_ap',g)
        self.assertIn('Fmax_percent',g)
        self.assertIn('AuPRC_percent',g)
        self.assertIn('AP_percent',g)
        self.assertAlmostEqual(g['standard_protein_fmax'],100.,places=6)
        r0=self.execute(fmax_only=True)
        self.assertEqual(r0.returncode,0,r0.stderr)
        old=json.loads((self.root/'fmax_only/nbs_w2s_comparison.json').read_text())
        self.assertAlmostEqual(g['standard_protein_fmax'],old['methods']['NBS_final']['standardized']['standard_protein_fmax'],places=10)
        self.assertNotIn('standard_micro_pr_auc',old['methods']['NBS_final']['standardized'])
        self.assertIn('AuPRC_percent',summary['deltas']['G_minus_B']['standardized'])

    def test_phase_lineage_tampering_is_rejected(self):
        self.manifest_data['lifetime_optimizer_updates']=99
        self.manifest.write_text(json.dumps(self.manifest_data))
        r=self.execute()
        self.assertNotEqual(r.returncode,0)
        self.assertIn('phase bookkeeping differs',r.stderr)

    def test_native_nbs_adapter_provenance_scoring_guard(self):
        keys=['phase_step','phase_epoch','phase_step_in_epoch','epoch',
            'lifetime_weak_exposures','lifetime_core_exposures',
            'lifetime_optimizer_updates','cumulative_completed_epochs']
        phase={'phase_plan':self.manifest_data['phase_plan'],
               'phase_origin':self.manifest_data['phase_origin']}
        phase.update({key:self.manifest_data[key] for key in keys})
        adapter={'runner_version':'0.8.9.1','probability':str(self.root/'G.npy'),
            'prediction_sha256':digest(self.root/'G.npy'),
            'protein_ids':str(self.root/'ids.txt'),
            'go_ids':str(self.root/'go_registry.tsv'),
            'native_prediction_manifest':str(self.manifest),
            'native_input_manifest':str(self.inp),
            'checkpoint_sha256':digest(self.checkpoint),
            'checkpoint_path':str(self.checkpoint),
            'prepared_input_manifest_sha256':digest(self.inp),
            'phase_identity':phase}
        adapter_path=self.root/'adapter.json';adapter_path.write_text(json.dumps(adapter))
        code='''
import json
from pathlib import Path
from scripts.dark_weak.run_complete_evaluation import verify_native_nbs_adapters
root=Path(__import__('sys').argv[1])
go=[f'GO:{i:07d}' for i in range(1,21313)]
r=verify_native_nbs_adapters({'G':root/'adapter.json'},
    {'G':root/'G.npy'},{'G':root/'ids.txt'},{'G':root/'go_ids.txt'},go)
print(json.dumps(r))
'''
        (self.root/'go_ids.txt').write_text('\n'.join(self.go)+'\n')
        def run_guard():
            env=dict(os.environ,PYTHONPATH=str(self.root))
            return subprocess.run([sys.executable,'-c',code,str(self.root)],cwd=self.root,env=env,
                                  text=True,capture_output=True)
        success=run_guard()
        self.assertEqual(success.returncode,0,success.stderr)
        self.assertEqual(json.loads(success.stdout)['G']['runner_version'],'0.8.9.1')
        adapter['phase_identity']['lifetime_optimizer_updates']=42
        adapter_path.write_text(json.dumps(adapter))
        failed=run_guard()
        self.assertNotEqual(failed.returncode,0)
        self.assertIn('phase identity differs',failed.stderr)

    def test_wrong_config_version_refused_before_gpu_export(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from dark_weak.export_nbs_v089_predictions import export
        cfg=self.root/'config.json';cfg.write_text(json.dumps({'release_version':'0.8.8',
            'model_version':'0.8.8','stage':{'name':'nbs_v088_hetero_tuned'},'task':'bp'}))
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError,'release_version'):
            export(SimpleNamespace(config=cfg,task='bp'))

    def test_checkpoint_hash_tampering_is_rejected(self):
        self.checkpoint.write_bytes(b'mutated')
        r=self.execute()
        self.assertNotEqual(r.returncode,0)
        self.assertIn('checkpoint file differs',r.stderr)


if __name__=='__main__':unittest.main()
