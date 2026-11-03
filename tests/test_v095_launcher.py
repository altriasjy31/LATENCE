import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.nbs import run_nbs_v095 as launcher
from scripts.nbs import train_nbs_full_task_v095 as worker


def fixture(root):
    source=root/'source';source.mkdir();data=root/'dataset';data.mkdir()
    manifest=data/'unified_dataset_manifest.json';manifest.write_text('{}')
    config=dict(release_version='0.9.4',model_version='0.9.4',task='bp',
                data={'unified_dataset_manifest':str(manifest)},stage={},
                supervision_contract={'weak_pseudo':{}},
                full_task={'model':{},'sampler':{'seed':8084}},runtime={'old':True})
    p=source/'resolved_config.json';p.write_text(json.dumps(config))
    return source,manifest


class LauncherTests(unittest.TestCase):
    def test_config_reuses_dataset_and_freezes_all_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,manifest=fixture(root)
            before=(source/'resolved_config.json').read_bytes();before_m=manifest.read_bytes()
            args=launcher.parser().parse_args(['train','--source-work-dir',str(source),'--work-dir',str(root/'new')])
            path=launcher.configure(args);cfg=json.loads(path.read_text());ft=cfg['full_task']
            self.assertEqual(ft['weak_batch']+ft['core_batch'],12800)
            self.assertEqual(ft['accumulation_steps'],2)
            self.assertEqual(ft['model']['jk_mode'],'sum')
            self.assertEqual(ft['loss']['loss_scale'],1000.)
            self.assertEqual(ft['loss']['teacher_topk'],2048)
            self.assertEqual(ft['scheduler']['min_lr'],.00003)
            self.assertEqual(ft['optimizer']['decay_policy'],'matrices_only')
            self.assertEqual(ft['precision'],'bf16')
            self.assertEqual(ft['initialization'],{'mode':'fresh'})
            self.assertNotIn('runtime',cfg)
            self.assertEqual(before,(source/'resolved_config.json').read_bytes())
            self.assertEqual(before_m,manifest.read_bytes())
            self.assertFalse((root/'new').exists())
            launcher.configure(args)
            args.epochs=300
            with self.assertRaisesRegex(ValueError,'differs'):
                launcher.configure(args)

    def test_optional_init_is_fixed_file_hash_and_not_reused_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,_=fixture(root);checkpoint=root/'nbs_epoch20.pt';checkpoint.write_bytes(b'fake-fixture')
            args=launcher.parser().parse_args(['build','--source-work-dir',str(source),'--work-dir',str(root/'new'),
                                              '--init-from',str(checkpoint)])
            cfg=json.loads(launcher.configure(args).read_text())
            initial=cfg['full_task']['initialization']
            self.assertEqual(initial['sha256'],launcher.digest(checkpoint))
            self.assertFalse(initial['optimizer_reused']);self.assertFalse(initial['scheduler_reused'])
            args.init_from=root/'latest.pt';args.init_from.write_bytes(b'fake')
            with self.assertRaisesRegex(ValueError,'fixed'):
                launcher.configure(args)

    def test_resume_cannot_redefine_init_or_precision(self):
        with self.assertRaisesRegex(ValueError,'only for a new'):
            launcher.main(['resume','--init-from','x'])
        with self.assertRaisesRegex(ValueError,'Precision is frozen'):
            launcher.main(['resume','--precision','fp32'])

    def test_smoke_and_training_use_separate_paths_and_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,_=fixture(root)
            with patch.object(launcher.subprocess,'run') as run, patch.dict(os.environ,{'CONFIG':'bad','WORLD_SIZE':'4'}):
                launcher.main(['smoke','--source-work-dir',str(source),'--work-dir',str(root/'new'),'--gpu','1','--stop-step','1'])
            self.assertEqual(run.call_count,2)
            command=run.call_args.args[0];env=run.call_args.kwargs['env']
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1');self.assertNotIn('WORLD_SIZE',env);self.assertNotIn('CONFIG',env)
            self.assertEqual(command[command.index('--work-dir')+1],str(root/'new_smoke'))
            self.assertEqual(command[-2:],['--stop-step','1'])
            self.assertTrue((root/'new_smoke_config/training_config.json').is_file())
            self.assertFalse((root/'new_config').exists())

    def test_eval_has_separate_lock(self):
        train=launcher.parser().parse_args(['train']);evaluation=launcher.parser().parse_args(['evaluate','--epoch','5'])
        self.assertNotEqual(launcher.operation_lock(train,Path('/tmp/new')),launcher.operation_lock(evaluation,Path('/tmp/new')))


@dataclass
class ModelConfig:
    hidden_dim:int=4
    jk_mode:str='sum'

@dataclass
class Phase:
    steps_per_epoch:int=3


class InitializationTests(unittest.TestCase):
    def test_initialization_verifies_data_model_manifest_and_fixed_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'nbs_epoch20.pt';path.write_bytes(b'checkpoint');manifest=root/'manifest.json';manifest.write_text('{}')
            data=SimpleNamespace(data_contract=lambda:{'data':'same'},ontology=lambda device:{})
            base=SimpleNamespace(PhasePlanV090=Phase,_phase_fields=lambda phase,step:{'epoch':20,'cumulative_completed_epochs':20},
                                 ontology_contract=lambda ontology:{'ontology':'same'},training_baseline_identity=lambda d:{'base':'same'},
                                 FullTaskModelConfigV090=ModelConfig)
            config={'full_task':{'model':{'hidden_dim':4,'jk_mode':'sum'}},'data':{'unified_dataset_manifest':str(manifest)}}
            saved=dict(runner_version='0.9.4',model_architecture_version='0.9.4',data_architecture_version='0.9.4',
                       phase_plan={'steps_per_epoch':3},step=60,epoch=20,cumulative_completed_epochs=20,
                       contract={'data':data.data_contract(),'ontology':{'ontology':'same'}},training_baseline_identity={'base':'same'},
                       config={'v094_experiment':{'unified_dataset_manifest_sha256':worker.sha256(manifest)}},
                       model_config={'hidden_dim':4,'jk_mode':'last'})
            initial={'checkpoint':str(path),'sha256':worker.sha256(path)}
            fake=SimpleNamespace(load=lambda *a,**kw:saved)
            with patch.dict(sys.modules,{'torch':fake}):
                self.assertIs(worker.validate_initialization(base,config,data,initial),saved)
                saved['model_config']['hidden_dim']=5
                with self.assertRaisesRegex(ValueError,'only in jk_mode'):
                    worker.validate_initialization(base,config,data,initial)
                saved['model_config']['hidden_dim']=4
                saved['contract']['data']={'data':'different'}
                with self.assertRaisesRegex(ValueError,'data/ontology'):
                    worker.validate_initialization(base,config,data,initial)
                saved['contract']['data']=data.data_contract()
                saved['config']['v094_experiment']['unified_dataset_manifest_sha256']='wrong'
                with self.assertRaisesRegex(ValueError,'identical v094 dataset'):
                    worker.validate_initialization(base,config,data,initial)
                initial['sha256']='changed'
                with self.assertRaisesRegex(ValueError,'changed'):
                    worker.validate_initialization(base,config,data,initial)


if __name__=='__main__':unittest.main()
