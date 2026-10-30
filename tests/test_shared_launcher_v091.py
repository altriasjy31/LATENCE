import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
LAUNCHER = PACKAGE / 'scripts/nbs/run_nbs_shared_head_v091.sh'
CONFIG = PACKAGE / 'nbs_models/nbs_protein_go/configs/bp_shared_head_v091_from_v089_ep60.json'


class SharedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scripts = self.root / 'scripts/nbs'
        self.scripts.mkdir(parents=True)
        shutil.copyfile(LAUNCHER, self.scripts / LAUNCHER.name)
        self.config = self.root / 'research.json'
        shutil.copyfile(CONFIG, self.config)
        stub = '''import argparse,json,os,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--work-dir');a,_=p.parse_known_args()
if a.work_dir:
 d=pathlib.Path(a.work_dir)
 if d.exists() and any(d.iterdir()):raise RuntimeError('new work directory contaminated by launcher')
 d.mkdir(parents=True,exist_ok=True)
print(json.dumps({'script':pathlib.Path(__file__).name,'args':sys.argv[1:],'rank':os.environ.get('RANK'),'world':os.environ.get('WORLD_SIZE'),'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')}))
'''
        for name in ('cache_nbs_shared_head_v091.py', 'train_nbs_shared_head_v091.py', 'eval_nbs_shared_head_v091.py', 'convert_nbs_shared_head_v091.py', 'train_nbs_hybrid_v091.py'):
            (self.scripts / name).write_text(stub)
        self.env = {k: v for k, v in os.environ.items() if k not in
                    ('INIT_FROM', 'ENCODER_FROM', 'RESUME', 'STEPS', 'STOP_STEP', 'STOP_EPOCH', 'CHECKPOINT', 'HEAD_CHECKPOINT', 'PRESET', 'HYBRID_INIT', 'JOINT_CHECKPOINT', 'JOINT_WORK_DIR', 'CONVERT_DIR', 'EVAL_DIR')}
        self.env.update(PYTHON_BIN=sys.executable, CONFIG=str(self.config), DEVICE='cuda:0', NUM_GPUS='1',
                        CUDA_VISIBLE_DEVICES='1', RANK='2', WORLD_SIZE='4', WORK_DIR=str(self.root / 'outputs/head'))

    def run_stage(self, stage):
        return subprocess.run(['bash', str(self.scripts / LAUNCHER.name), stage],
                              cwd=self.root, env=self.env, text=True, capture_output=True)

    def test_fresh_training_directory_is_not_polluted_by_lock(self):
        result = self.run_stage('train')
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(result.stdout.splitlines()[-1])
        self.assertIsNone(row['rank'])
        self.assertIsNone(row['world'])
        self.assertEqual(row['cuda'], '1')
        self.assertTrue(Path(self.env['WORK_DIR'] + '.launcher.lock').is_file())
        self.assertEqual(list(Path(self.env['WORK_DIR']).iterdir()), [])

    def test_smoke_is_only_new_head_three_updates(self):
        result = self.run_stage('smoke')
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(result.stdout.splitlines()[-1])
        self.assertIn('--stop-step', row['args'])
        self.assertEqual(row['args'][-1], '3')
        self.assertEqual(row['script'], 'train_nbs_shared_head_v091.py')

    def test_cache_and_evaluate_use_distinct_entrypoints(self):
        cache = self.run_stage('cache')
        self.assertEqual(cache.returncode, 0, cache.stderr)
        self.assertEqual(json.loads(cache.stdout.splitlines()[-1])['script'], 'cache_nbs_shared_head_v091.py')
        checkpoint = self.root / 'shared_head_step500.pt'
        checkpoint.touch()
        self.env['HEAD_CHECKPOINT'] = str(checkpoint)
        result = self.run_stage('evaluate')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])['script'], 'eval_nbs_shared_head_v091.py')

    def test_convert_joint_and_joint_evaluation_routes(self):
        head=self.root/'shared_head_step500.pt';head.touch()
        self.env['HEAD_CHECKPOINT']=str(head)
        result=self.run_stage('convert')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])['script'],'convert_nbs_shared_head_v091.py')
        self.env.pop('HEAD_CHECKPOINT')
        initial=self.root/'hybrid_init.pt';initial.touch()
        self.env.update(HYBRID_INIT=str(initial),JOINT_WORK_DIR=str(self.root/'joint'),STOP_STEP='3')
        result=self.run_stage('joint')
        self.assertEqual(result.returncode,0,result.stderr)
        row=json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(row['script'],'train_nbs_hybrid_v091.py')
        self.assertIn('--initialization',row['args'])
        self.env.pop('STOP_STEP');self.env.pop('HYBRID_INIT')
        self.env['JOINT_CHECKPOINT']=str(initial)
        result=self.run_stage('evaluate-joint')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--hybrid-checkpoint',json.loads(result.stdout.splitlines()[-1])['args'])

    def test_old_six_layer_config_and_resume_flags_are_rejected(self):
        self.config.write_text(json.dumps({'release_version': '0.9.0', 'full_task': {'model': {'sage_layers': 6}}}))
        self.assertNotEqual(self.run_stage('train').returncode, 0)
        shutil.copyfile(CONFIG, self.config)
        self.env['ENCODER_FROM'] = 'nbs_epoch60.pt'
        self.assertNotEqual(self.run_stage('train').returncode, 0)


if __name__ == '__main__':
    unittest.main()
