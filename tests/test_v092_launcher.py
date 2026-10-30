"""Explicit launcher routing and matched-preset invariants; no production data."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/nbs/run_nbs_v092.py'
spec=importlib.util.spec_from_file_location('v092_worker_test',ROOT/'scripts/nbs/train_nbs_full_task_v092.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
spec=importlib.util.spec_from_file_location('v092_launcher_module_test',SCRIPT)
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


class V092LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);scripts=self.root/'scripts/nbs';scripts.mkdir(parents=True)
        shutil.copy2(SCRIPT,scripts/SCRIPT.name)
        self.script=scripts/SCRIPT.name
        source=ROOT/'nbs_models/nbs_protein_go/configs';dest=self.root/'nbs_models/nbs_protein_go/configs'
        shutil.copytree(source,dest)
        (scripts/'train_nbs_full_task_v092.py').write_text('''import argparse,json,os,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--stage');p.add_argument('--work-dir');p.add_argument('--preset');p.add_argument('--stop-step');p.add_argument('--stop-epoch');p.add_argument('--checkpoint');p.add_argument('--resume');a,_=p.parse_known_args()
if a.stage=='train':
 d=pathlib.Path(a.work_dir);d.mkdir(parents=True,exist_ok=True);(d/'latest.pt').write_text('stub');(d/'nbs_epoch5.pt').write_text('stub')
print(json.dumps({'stage':a.stage,'args':sys.argv[1:],'world':os.environ.get('WORLD_SIZE'),'rank':os.environ.get('RANK'),'config_env':os.environ.get('CONFIG'),'resume_env':os.environ.get('RESUME'),'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'),'threads':os.environ.get('OMP_NUM_THREADS')}))
''')
        self.env=dict(os.environ,CONFIG='WRONG.json',WORK_DIR='/wrong',RESUME='wrong.pt',WORLD_SIZE='8',RANK='7',LOCAL_RANK='7',PRESET='wrong')
    def run_action(self,*args,ok=True):
        r=subprocess.run([sys.executable,str(self.script),*args],cwd=self.root,env=self.env,text=True,capture_output=True)
        self.assertEqual(r.returncode==0,ok,r.stdout+r.stderr)
        rows=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')]
        return r,rows
    def test_training_ignores_old_environment_and_defaults_to_epoch5(self):
        _,rows=self.run_action('train','--preset','geometry','--gpu','1')
        self.assertEqual([r['stage'] for r in rows],['prepare','train'])
        for r in rows:
            self.assertIsNone(r['world']);self.assertIsNone(r['rank']);self.assertIsNone(r['config_env']);self.assertIsNone(r['resume_env'])
            self.assertEqual(r['cuda'],'1');self.assertEqual(r['threads'],'4')
        self.assertIn('--stop-epoch',rows[-1]['args']);self.assertEqual(rows[-1]['args'][-1],'5')
    def test_smoke_uses_separate_directory_and_three_updates(self):
        _,rows=self.run_action('smoke')
        self.assertEqual(rows[-1]['args'][-2:],['--stop-step','3'])
        directory=Path(rows[-1]['args'][rows[-1]['args'].index('--work-dir')+1])
        self.assertTrue(directory.name.endswith('_smoke'))
        self.assertTrue(directory.with_name(directory.name+'.v092.lock').is_file())
    def test_fresh_train_rejects_existing_outputs_and_resume_is_explicit(self):
        self.run_action('train')
        self.run_action('train',ok=False)
        _,rows=self.run_action('resume','--stop-epoch','20')
        self.assertIn('--resume',rows[-1]['args']);self.assertEqual(rows[-1]['args'][-2:],['--stop-epoch','20'])
    def test_evaluate_uses_fixed_epoch_single_worker_only(self):
        self.run_action('train','--preset','baseline')
        _,rows=self.run_action('evaluate','--preset','baseline','--epoch','5')
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['stage'],'evaluate')
        self.assertIn('--allow-missing-references',rows[0]['args'])
        self.assertNotIn('--resume',rows[0]['args'])
        self.run_action('evaluate','--preset','baseline',ok=False)
        self.run_action('evaluate','--preset','baseline','--epoch','5','--stop-step','3',ok=False)
    def test_shared_evaluate_keeps_exact_metric_and_baseline_contract(self):
        self.run_action('train')
        _,rows=self.run_action('evaluate','--epoch','5')
        self.assertIn('--baseline-comparison',rows[0]['args'])
        self.assertIn('stage1',rows[0]['args']);self.assertIn('exact',rows[0]['args'])
    def test_invalid_gpu_and_endpoints_are_rejected(self):
        for args in [('train','--gpu','0,1'),('train','--threads','0'),('train','--stop-epoch','101'),('smoke','--stop-epoch','2'),('prepare','--stop-step','1')]:
            self.run_action(*args,ok=False)


class V092SmokeTimingTests(unittest.TestCase):
    def test_three_update_smoke_has_no_post_warmup_estimate(self):
        result=launcher.summarize_smoke_timing([{'step':3,'window_updates':3,'window_training_seconds':30.,'peak_allocated_gb':2.}])
        self.assertEqual(result['status'],'insufficient_post_warmup_updates')
        self.assertEqual(result['measured_updates'],0)
        self.assertIsNone(result['seconds_per_update'])
        self.assertIsNone(result['mean_queries_per_update'])
        self.assertEqual(result['peak_allocated_gb'],2.)
    def test_short_final_window_uses_actual_update_and_query_weights(self):
        result=launcher.summarize_smoke_timing([
            {'step':5,'window_updates':5,'window_training_seconds':100.,'weak_exposures':50000,'core_exposures':10000},
            {'step':10,'window_updates':5,'window_training_seconds':50.,'weak_exposures':100000,'core_exposures':20000},
            {'step':13,'window_updates':3,'window_training_seconds':60.,'weak_exposures':125000,'core_exposures':26000}])
        self.assertEqual(result['measured_updates'],8)
        self.assertAlmostEqual(result['seconds_per_update'],110/8)
        self.assertAlmostEqual(result['mean_queries_per_update'],91000/8)
        self.assertEqual(result['status'],'measured')


class V092ConfigTests(unittest.TestCase):
    def configs(self):
        return {preset:json.loads((ROOT/'nbs_models/nbs_protein_go/configs'/f'bp_full_task_v092_{preset}_single12800.json').read_text()) for preset in worker.PRESETS}
    def test_all_presets_have_matched_optimization_and_three_layer_graphs(self):
        configs=self.configs()
        keys=('seed','weak_batch','core_batch','accumulation_steps','expected_world_size','learning_rate','scheduler','loss','epochs','sampler')
        for preset,cfg in configs.items():
            worker.validate_experiment(cfg,preset)
            self.assertEqual(cfg['full_task']['model']['sage_layers'],3)
            for key in keys:self.assertEqual(cfg['full_task'][key],configs['baseline']['full_task'][key],key)
    def test_wrong_model_switch_and_wrong_rank_configuration_are_rejected(self):
        cfg=self.configs()['geometry'];cfg['full_task']['model']['go_input_mode']='ontology'
        with self.assertRaises(ValueError):worker.validate_experiment(cfg,'geometry')
        cfg=self.configs()['baseline'];cfg['full_task']['expected_world_size']=2
        with self.assertRaises(ValueError):worker.validate_experiment(cfg,'baseline')
    def test_warmstarts_are_rejected_before_runtime_imports(self):
        for arg in ['--init-from=old.pt','--encoder-from=old.pt']:
            with self.assertRaisesRegex(ValueError,'scratch'):worker.main([arg])
    def test_adapter_is_idempotent_and_tracks_implementation_without_mutating_original(self):
        scripts=types.ModuleType('scripts');scripts.__path__=[]
        nbs=types.ModuleType('scripts.nbs');nbs.__path__=[]
        original=lambda variant:{'original.py':'original-hash'}
        base=types.ModuleType('scripts.nbs.train_nbs_full_task_v089');base.prediction_implementation=original;base.evaluate=lambda *a:None
        nbs.train_nbs_full_task_v089=base;scripts.nbs=nbs
        nbs_pg=types.ModuleType('nbs_pg');nbs_pg.__path__=[]
        fast=types.ModuleType('nbs_pg.full_task_sampler_fast');fast.implementation_sha256=lambda:'fast-hash';fast.FullTaskDataV088Fast=object
        nbs_pg.full_task_sampler_fast=fast
        with patch.dict(sys.modules,{'scripts':scripts,'scripts.nbs':nbs,'scripts.nbs.train_nbs_full_task_v089':base,'nbs_pg':nbs_pg,'nbs_pg.full_task_sampler_fast':fast}):
            self.assertIs(worker.install('baseline'),base)
            result=base.prediction_implementation('dynamic')
            self.assertEqual(result['original.py'],'original-hash')
            self.assertIn('train_nbs_full_task_v092.py',result)
            self.assertIn('full_task_sampler_fast.py',result)
            self.assertIs(worker.install('baseline'),base)
            self.assertEqual(original('dynamic'),{'original.py':'original-hash'})


if __name__=='__main__':unittest.main()
