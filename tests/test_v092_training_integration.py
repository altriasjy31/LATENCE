"""Real v090 phase-engine train/resume/evaluate with the actual V092 model.

Only production stores and the external metric/checker subprocess boundary are
substituted. Model, objective, optimizer, RNG, counters and scheduler are real.
Run in a child process to isolate the intentional runtime adapter mutations.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'nbs_models/nbs_protein_go')]
RUNTIME=(ROOT/'scripts/nbs/train_nbs_full_task_v090.py').is_file() and importlib.util.find_spec('torch') is not None


def run_fixture():
    import torch
    from scripts.nbs.train_nbs_full_task_v092 import install
    from nbs_pg.full_task_model_v092 import FullTaskModelConfigV092
    from nbs_pg.full_task_metrics_v085 import compute_standard_metrics
    base=install('geometry')
    torch.set_num_threads(1)
    device=torch.device('cpu')
    torch.manual_seed(66)
    ontology={'center':torch.randn(8,3),'offset':torch.ones(8,3)*.2,
              'task_to_ontology':torch.arange(8),'edges':{}}
    options=vars(FullTaskModelConfigV092(hidden_dim=8,ontology_layers=0,raw_geometry_width=12,
        random_feature_dim=8,input_dropout=.1,dropout=.25,go_input_mode='ontology_raw'))
    config={'task':'bp','release_version':'0.9.0','model_version':'0.9.0',
        'full_task':{'seed':66,'weak_batch':2,'core_batch':2,'accumulation_steps':2,'expected_world_size':1,
            'epochs':100,'epoch_unit':'weak','variant':'dynamic','sampler':{'mode':'dynamic'},
            'learning_rate':.001,'weight_decay':.0001,'grad_clip':5.,'log_every':1,'eval_batch':2,
            'checkpoint_every_epochs':5,'monitor_core':False,'model':options,
            'loss':{'core_weight':1.,'weak_weight':.25},
            'scheduler':{'name':'epoch_warmup_hold_cosine','warmup_epochs':2.,'hold_until_epoch':10.,'min_lr':.0001}}}
    calls=[]
    class Data:
        feature_dim=6
        num_task_go=8
        weak_ids=np.arange(6,dtype=np.int64)
        core_ids=np.arange(6,10,dtype=np.int64)
        validation_ids=np.array([],np.int64)
        seen_task_mask=np.ones(8,bool)
        _sampling_step=0
        _sampling_rank=0
        _sampling_training=False
        def ensure_prepared(self):pass
        def ontology(self,device):return ontology
        def data_contract(self):return {'synthetic_fixture':'v092-integration','seen':self.seen_task_mask.tolist()}
        def set_sampling_context(self,step,rank,training):
            self._sampling_step=step;self._sampling_rank=rank;self._sampling_training=training
        def batch(self,ids,device='cpu',supervision_seed_ids=None):
            ids=np.asarray(ids,np.int64)
            forbidden=ids if supervision_seed_ids is None else np.asarray(supervision_seed_ids)
            assert np.isin(ids,forbidden).all()
            calls.append((len(ids),len(forbidden)))
            n=len(ids);source=torch.tensor(list(-ids-1)+[100,101]);blocks=[]
            for layer in range(3):
                dest=source if layer<2 else source[:n]
                edge=torch.tensor([[n+i%2 for i in range(len(dest))],list(range(len(dest)))])
                blocks.append(dict(src_global_ids=source.clone(),dst_global_ids=dest.clone(),dst_in_src=torch.arange(len(dest)),
                    edge_index=edge,edge_type=torch.zeros(len(dest),dtype=torch.long),edge_attr=torch.ones(len(dest),3)))
            feat=torch.sin(torch.tensor(list(ids)+[100,101]).float()[:,None]*torch.arange(1,7).float()[None])
            candidate=torch.tensor([[int(i)%8,(int(i)+2)%8] for i in list(ids)+[100,101]])
            target=torch.zeros(n,8)
            for i,protein in enumerate(ids):target[i,[protein%8,(protein+1)%8]]=1
            return dict(sampled_protein_x=feat,sampled_node_type=torch.tensor([2]*n+[0,1]),
                sampled_candidate_go=candidate,sampled_candidate_attr=torch.ones(n+2,2,3)*.8,
                sampled_gold_edge=torch.tensor([[n],[0]]),sampled_pseudo_edge=torch.tensor([[n+1],[2]]),
                sampled_seed_index=torch.arange(n),blocks=blocks,targets=target,positive_mask=target.bool(),
                is_weak=torch.tensor(ids<6),protein_x=feat[:n],candidate_go=candidate[:n],candidate_attr=torch.ones(n,2,3)*.8,
                seen_task_mask=torch.ones(8,dtype=torch.bool))
        def inference_batch(self,folder,ids,device='cpu'):
            batch=self.batch(ids,device)
            for key in ('targets','positive_mask','is_weak'):batch.pop(key)
            return batch
    data=Data()
    def train(folder,stop,resume=None):
        args=types.SimpleNamespace(work_dir=folder,encoder_from=None,resume=resume,epochs=None,stop_epoch=None,stop_step=stop)
        if resume is None:torch.manual_seed(66)
        base.train(args,deepcopy(config),data,device,0,1)
    with tempfile.TemporaryDirectory() as temp,patch.object(base,'training_baseline_identity',return_value={'fixture':'unchanged'}):
        root=Path(temp);full=root/'continuous';split=root/'resumed'
        train(full,3)
        train(split,1)
        train(split,3,split/'latest.pt')
        first=torch.load(full/'latest.pt',weights_only=False);second=torch.load(split/'latest.pt',weights_only=False)
        assert first['step']==second['step']==3
        assert first['cumulative_completed_epochs']==second['cumulative_completed_epochs']==1
        assert first['scheduler']['contract']['epochs']==100
        assert first['scheduler']==second['scheduler']
        assert first['phase_training_contract']==second['phase_training_contract']
        for key,value in first['model'].items():torch.testing.assert_close(value,second['model'][key],rtol=0,atol=0)
        assert first['model_config']['go_input_mode']=='ontology_raw'
        assert first['model_config']['protein_random_features']=='none'
        assert 'full_task_model_v092.py' in first['training_implementation']
        assert all(local==2 and forbidden==4 for local,forbidden in calls)
        # Genuine gradients reached the raw geometry encoder and scorer.
        state_at1=torch.load(split/'nbs_step1.pt',weights_only=False)['model']
        for prefix in ('raw_geometry_encoder.','scorer.','node_encoder.'):
            assert any(not torch.equal(value,state_at1[key]) for key,value in first['model'].items() if key.startswith(prefix)),prefix
        inp=root/'inputs';inp.mkdir();np.save(inp/'ind_test_repr.f16.npy',np.zeros((3,6),np.float16))
        np.save(inp/'backbone_ind_test_prob.f16.npy',np.full((3,8),.5,np.float16))
        (inp/'ind_test_input_manifest.json').write_text('{}')
        (inp/'protein_ids.txt').write_text('p0\np1\np2\n')
        metadata=root/'metadata.pkl';metadata.write_bytes(b'fixture metadata')
        baseline=root/'baseline.json';baseline.write_text('{}')
        args=types.SimpleNamespace(checkpoint=full/'nbs_epoch1.pt',input_dir=inp,metadata_file=metadata,work_dir=full,
            ablation='full',stage='evaluate',baseline_comparison=baseline)
        external=[]
        def metric_subprocess(command,**kwargs):
            external.append(Path(command[1]).name)
            def argument(name):return Path(command[command.index(name)+1])
            if Path(command[1]).name=='eval_nbs_full_task_v090.py':
                probability=argument('--nbs-prob')
                assert probability.is_file()
                p=np.load(probability);assert p.shape==(3,8) and np.isfinite(p).all()
                manifest=json.loads(argument('--prediction-manifest').read_text())
                assert manifest['model_config']['go_input_mode']=='ontology_raw'
                assert manifest['prediction_implementation']['full_task_model_v092.py']==first['training_implementation']['full_task_model_v092.py']
                labels=np.zeros((3,8),bool)
                for i in range(3):labels[i,[i,i+1]]=True
                metrics=compute_standard_metrics(labels,p)
                out=argument('--output-dir');out.mkdir(parents=True,exist_ok=True)
                (out/'nbs_w2s_comparison.json').write_text(json.dumps({'methods':{'NBS_final':{'standardized':metrics}}}))
                return subprocess.CompletedProcess(command,0)
            assert Path(command[1]).name=='check_nbs_v090_acceptance.py'
            artifact={'status':'未达标','fixed_conditions_verified':True,
                      'new_comparison':{'sha256':base.sha256(argument('--comparison'))},
                      'baseline_comparison':{'sha256':base.sha256(argument('--baseline-comparison'))}}
            argument('--output').write_text(json.dumps(artifact))
            return subprocess.CompletedProcess(command,1)
        report_stub=types.ModuleType('nbs_pg.evaluation_report_v092')
        report_calls=[]
        report_stub.write_extended_report=lambda *a:report_calls.append(a) or full/'fixture_summary.json'
        with patch.object(base.subprocess,'run',side_effect=metric_subprocess),patch.dict(sys.modules,{'nbs_pg.evaluation_report_v092':report_stub}):
            base.evaluate(args,deepcopy(config),data,device)
            probability=full/'eval_step3/full/nbs_ind_test_prob.f32.npy'
            before=base.sha256(probability)
            base.evaluate(args,deepcopy(config),data,device)
            assert base.sha256(probability)==before
        assert len(report_calls)==2
        assert len(external)==4
        assert json.loads((full/'acceptance_epoch1.json').read_text())['status']=='未达标'
        # Wrong raw-input architecture must fail instead of reusing predictions.
        wrong=deepcopy(config);wrong['full_task']['model']['go_input_mode']='ontology'
        try:base.evaluate(args,wrong,data,device)
        except ValueError as error:assert 'configuration' in str(error)
        else:raise AssertionError('mismatched architecture was accepted')
    print('V092 real model/loss/scheduler/AdamW train-resume-evaluate integration passed',flush=True)


@unittest.skipUnless(RUNTIME,'requires installed LATENCE sources and real PyTorch')
class V092TrainingIntegrationTests(unittest.TestCase):
    def test_actual_train_resume_and_evaluation_in_isolated_process(self):
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--run-fixture'],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('V092 real model/loss/scheduler/AdamW',result.stdout)


if __name__=='__main__':
    if '--run-fixture' in sys.argv:run_fixture()
    else:unittest.main()
