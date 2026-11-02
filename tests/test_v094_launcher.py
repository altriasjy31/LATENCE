import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('launcher_v094_test', ROOT/'scripts/nbs/run_nbs_v094.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_config_freezes_the_requested_new_supervision_and_preserves_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'source';old.mkdir();data=root/'new';data.mkdir()
            source={'task':'bp','data':{'root':'data'},'stage':{'protein_go_candidate_topk':512},
                    'supervision_contract':{'weak_pseudo':{}},'full_task':{'model':{},'sampler':{},
                    'scheduler':{'name':'epoch_warmup_hold_cosine','warmup_epochs':2.,'hold_until_epoch':10.,'min_lr':.0001}},
                    'runtime':{'obsolete':True}}
            p=old/'resolved_config.json';p.write_text(json.dumps(source));before=p.read_bytes()
            m=data/'unified_dataset_manifest.json';m.write_text('{}')
            args=launcher.parser().parse_args(['build','--source-work-dir',str(old),'--dataset-dir',str(data)])
            result=json.loads(launcher.configure(args,m).read_text())
            self.assertEqual(result['full_task']['loss'],{'core_weight':.25,'weak_weight':1.,'modelout_soft_weight':1.})
            self.assertEqual(result['full_task']['model']['output_strategy'],'shared_direct')
            self.assertEqual(result['full_task']['sampler']['candidate_topk'],16)
            self.assertEqual(result['full_task']['epochs'],200)
            self.assertNotIn('runtime',result)
            self.assertEqual(p.read_bytes(),before)
            launcher.configure(args,m)
            args.weak_weight=2
            with self.assertRaisesRegex(ValueError,'different horizon/policy'):
                launcher.configure(args,m)

    def test_build_uses_all_frequency_policy_without_quota_arguments(self):
        args=launcher.parser().parse_args(['build','--thresholds-json','/tmp/threshold.json'])
        command=launcher.build_command(args)
        self.assertNotIn('--candidate-mode',command)
        self.assertNotIn('--low-frequency-max',command)
        self.assertEqual(command[command.index('--thresholds-json')+1],'/tmp/threshold.json')

    def test_frozen_loss_flags_cannot_silently_change_resume(self):
        with self.assertRaisesRegex(ValueError,'Loss weights are frozen'):
            launcher.main(['resume','--weak-weight','2'])

    def test_env_and_evaluation_lock_are_separate_from_training(self):
        args=launcher.parser().parse_args(['train','--gpu','1'])
        with patch.dict(os.environ,{'WORLD_SIZE':'4','CONFIG':'old'}):
            env=launcher.environment(args)
        self.assertNotIn('WORLD_SIZE',env);self.assertNotIn('CONFIG',env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
        e=launcher.parser().parse_args(['evaluate','--epoch','5'])
        self.assertEqual(launcher.operation_lock(e,Path('/tmp/new')).name,'v092_evaluate.lock')
        self.assertNotEqual(launcher.operation_lock(e,Path('/tmp/new')),launcher.operation_lock(args,Path('/tmp/new')))


if __name__=='__main__': unittest.main()
