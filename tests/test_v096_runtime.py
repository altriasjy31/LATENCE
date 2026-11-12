"""v096 configuration contracts and strict source checkpoint lineage."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'nbs_models/nbs_protein_go')]
from scripts.nbs import run_nbs_v096 as launcher
from scripts.nbs import train_nbs_full_task_v096 as worker


def fixture(root):
    source = root / 'source'; source.mkdir()
    manifest = root / 'unified_dataset_manifest.json'; manifest.write_text('{}')
    esm = root / 'manifest.json'; esm.write_text(json.dumps({'complete': True, 'feature_dim': 1024}))
    ft = dict(expected_world_size=1, weak_batch=10240, core_batch=2560,
              accumulation_steps=2, epochs=200, precision='bf16', learning_rate=.003,
              weight_decay=.0001, grad_clip=5.,
              model={'encoder_variant':'hetero_tuned', 'jk_mode':'sum', 'output_strategy':'shared_direct'},
              sampler={'mode':'dynamic','candidate_topk':16},
              optimizer={'name':'adamw','eps':1e-8,'betas':[.9,.999],'decay_policy':'matrices_only'},
              scheduler={'name':'epoch_warmup_hold_cosine','warmup_epochs':5.,'hold_until_epoch':10.,'min_lr':.00003},
              loss={'loss_scale':1000.,'hierarchy_weight':.01,'asl_gamma_neg':4.,'teacher_topk':2048},
              initialization={'mode':'fresh'})
    config = dict(release_version='0.9.5', model_version='0.9.5', task='bp', stage={},
                  data={'unified_dataset_manifest':str(manifest)}, full_task=ft,
                  v095_experiment={'unified_dataset_manifest_sha256':worker.sha256(manifest)},
                  runtime={'old':True})
    (source/'resolved_config.json').write_text(json.dumps(config))
    return source, manifest, esm, config


class LauncherTests(unittest.TestCase):
    def test_preserves_v095_protocol_and_frozen_feature_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,manifest,esm,original=fixture(root)
            before=(source/'resolved_config.json').read_bytes()
            args=launcher.parser().parse_args(['build','--source-work-dir',str(source),
                    '--esm-manifest',str(esm),'--work-dir',str(root/'new')])
            out=launcher.configure(args);config=json.loads(out.read_text());ft=config['full_task']
            for key in ('loss','sampler','optimizer','scheduler','learning_rate','weak_batch','core_batch','accumulation_steps'):
                self.assertEqual(ft[key],original['full_task'][key],key)
            self.assertEqual(ft['model']['esm_dim'],1024)
            self.assertEqual(ft['model']['esm_gate_init'],0.)
            self.assertEqual(config['data']['esm_features_manifest_sha256'],worker.sha256(esm))
            self.assertEqual(ft['initialization'],{'mode':'fresh'})
            self.assertEqual(config['release_version'],'0.9.6')
            self.assertNotIn('runtime',config)
            self.assertEqual(before,(source/'resolved_config.json').read_bytes())
            launcher.configure(args)
            esm.write_text(json.dumps({'complete':True,'feature_dim':512}))
            with self.assertRaisesRegex(ValueError,'differs'):
                launcher.configure(args)

    def test_fixed_weights_only_lineage_and_no_implicit_checkpoint_choice(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,_,esm,_=fixture(root)
            cp=root/'nbs_epoch125.pt';cp.write_bytes(b'fake-fixture')
            args=launcher.parser().parse_args(['build','--source-work-dir',str(source),
                    '--esm-manifest',str(esm),'--work-dir',str(root/'new'),'--init-from',str(cp)])
            ft=json.loads(launcher.configure(args).read_text())['full_task']
            self.assertEqual(ft['learning_rate'],.001)
            init=ft['initialization']
            self.assertEqual(init['mode'],'v095_weights_only')
            self.assertFalse(init['optimizer_reused']);self.assertFalse(init['scheduler_reused'])
            self.assertEqual(init['sha256'],worker.sha256(cp))
            bad=root/'latest.pt';bad.write_bytes(b'fake');args.init_from=bad
            with self.assertRaisesRegex(ValueError,'fixed'):
                launcher.configure(args)

    def test_resume_cannot_replace_frozen_features_precision_or_initialization(self):
        for args,message in [(['--esm-manifest','x'],'ESM cache is frozen'),
                             (['--precision','fp32'],'Precision is frozen'),
                             (['--learning-rate','.002'],'Learning rate is frozen'),
                             (['--init-from','x'],'only for a new')]:
            with self.assertRaisesRegex(ValueError,message):
                launcher.main(['resume',*args])

    def test_missing_cache_fails_before_config_write(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,_,_,_=fixture(root)
            args=launcher.parser().parse_args(['build','--source-work-dir',str(source),'--work-dir',str(root/'new')])
            with self.assertRaisesRegex(ValueError,'requires --esm-manifest'):
                launcher.configure(args)
            self.assertFalse((root/'new_config').exists())

    def test_smoke_is_isolated_and_environment_cleaned(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,_,esm,_=fixture(root)
            with patch.object(launcher.subprocess,'run') as run, patch.dict(os.environ,{'CONFIG':'bad','WORLD_SIZE':'4'}):
                launcher.main(['smoke','--source-work-dir',str(source),'--esm-manifest',str(esm),
                    '--work-dir',str(root/'new'),'--gpu','0','--stop-step','1'])
            self.assertEqual(run.call_count,2)
            cmd=run.call_args.args[0];env=run.call_args.kwargs['env']
            self.assertEqual(cmd[cmd.index('--work-dir')+1],str(root/'new_smoke'))
            self.assertEqual(cmd[-2:],['--stop-step','1'])
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
            self.assertNotIn('CONFIG',env);self.assertNotIn('WORLD_SIZE',env)


@dataclass
class TinyModelConfig:
    hidden_dim:int=4
    jk_mode:str='sum'
    esm_dim:int=5
    esm_dropout:float=.1
    esm_gate_init:float=0.

@dataclass
class Phase:
    steps_per_epoch:int=3


class InitializationTests(unittest.TestCase):
    def test_only_esm_extension_may_differ_from_v095(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cp=root/'nbs_epoch125.pt';cp.write_bytes(b'checkpoint')
            manifest=root/'dataset.json';manifest.write_text('{}')
            data=SimpleNamespace(data_contract=lambda:{'old':'same','v096_esm_features':{'cache':'new'}},
                ontology=lambda device:{})
            base=SimpleNamespace(PhasePlanV090=Phase,
                _phase_fields=lambda p,s:{'epoch':125,'cumulative_completed_epochs':125},
                ontology_contract=lambda o:{'ontology':'same'},training_baseline_identity=lambda d:{'base':'same'},
                FullTaskModelConfigV090=TinyModelConfig)
            ft={'model':{'hidden_dim':4,'jk_mode':'sum','esm_dim':5},'loss':{'same':True},'sampler':{'mode':'dynamic'},
                'weak_batch':10240,'core_batch':2560,'accumulation_steps':2}
            cfg={'full_task':ft,'data':{'unified_dataset_manifest':str(manifest)}}
            source_cfg={'full_task':deepcopy(ft),'v095_experiment':{'unified_dataset_manifest_sha256':worker.sha256(manifest)}}
            saved=dict(runner_version='0.9.5',model_architecture_version='0.9.5',data_architecture_version='0.9.4',
                phase_plan={'steps_per_epoch':3},step=375,epoch=125,cumulative_completed_epochs=125,
                contract={'data':{'old':'same'},'ontology':{'ontology':'same'}},training_baseline_identity={'base':'same'},
                config=source_cfg,model_config={'hidden_dim':4,'jk_mode':'sum'},
                training_implementation={'train_nbs_full_task_v095.py':worker.sha256(ROOT/'scripts/nbs/train_nbs_full_task_v095.py')})
            init={'checkpoint':str(cp),'sha256':worker.sha256(cp)}
            fake=SimpleNamespace(load=lambda *args,**kwargs:saved)
            with patch.dict(sys.modules,{'torch':fake}):
                self.assertIs(worker.validate_initialization(base,cfg,data,init),saved)
                saved['model_config']['jk_mode']='last'
                with self.assertRaisesRegex(ValueError,'only in new ESM'):
                    worker.validate_initialization(base,cfg,data,init)
                saved['model_config']['jk_mode']='sum'
                saved['config']['full_task']['loss']={'changed':True}
                with self.assertRaisesRegex(ValueError,'training protocol: loss'):
                    worker.validate_initialization(base,cfg,data,init)
                saved['config']['full_task']['loss']={'same':True}
                saved['contract']['data']['old']='different'
                with self.assertRaisesRegex(ValueError,'data/ontology'):
                    worker.validate_initialization(base,cfg,data,init)


if __name__=='__main__':unittest.main()
