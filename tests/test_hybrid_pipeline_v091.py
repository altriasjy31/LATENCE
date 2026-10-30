"""Actual PyTorch 2.4 model/loss/optimizer integration on a synthetic graph.

Production stores are substituted; original three-layer model forward and loss
are used. This does not certify production data or CUDA behavior.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'nbs_models/nbs_protein_go')]

HAS_RUNTIME=importlib.util.find_spec('torch') is not None and (ROOT/'nbs_models/nbs_protein_go/nbs_pg/full_task_model_v088.py').exists()


@unittest.skipUnless(HAS_RUNTIME,'requires PyTorch and installation in the existing LATENCE project')
class HybridPipelineTests(unittest.TestCase):
    def setUp(self):
        import torch
        from nbs_pg.full_task_model_v088 import FullTaskGraphModelV088,FullTaskModelConfigV088
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.dir=Path(self.tmp.name)
        self.oldcwd=Path.cwd();self.addCleanup(os.chdir,self.oldcwd)
        torch.manual_seed(14)
        self.ontology={'center':torch.randn(8,512),'offset':torch.ones(8,512)*.2,
                       'task_to_ontology':torch.arange(8),'edges':{}}
        self.source=FullTaskGraphModelV088(6,self.ontology,FullTaskModelConfigV088(hidden_dim=8,ontology_layers=0)).eval()
        self.source.requires_grad_(False)
        self.base_cfg={'task':'bp','full_task':{'seed':14,'weak_batch':4,'core_batch':2,'eval_batch':3,
                                             'loss':{'core_weight':1.,'weak_weight':.25}}}
        self.checkpoint=self.dir/'nbs_epoch60.pt';torch.save({'model':self.source.state_dict()},self.checkpoint)
        self.source_config=self.dir/'resolved_source.json';self.source_config.write_text(json.dumps(self.base_cfg))
        self.cfg={'schema':'nbs-shared-head-v091','source_epoch':60,'source_config':str(self.source_config),
                  'source_checkpoint':str(self.checkpoint),'cache_dir':str(self.dir/'cache'),
                  'cache':{'seed':31,'query_batch':3,'global_weak_batch':4,'global_core_batch':2,'sampler':'fast'},
                  'head':{'hidden_dim':8,'mlp_width':12,'initial_bias':-2.},
                  'train':{'seed':31,'weak_batch':4,'core_batch':2,'steps':4,'warmup_steps':1,'save_every':2,
                           'learning_rate':.001,'weight_decay':.0001,'grad_clip':5.},'loss':None,
                  'joint':{'seed':32,'steps':4,'weak_batch':4,'core_batch':2,'accumulation_steps':2,
                           'encoder_lr':.0003,'shared_lr':.001,'residual_lr':.0001,'weight_decay':.0001,
                           'shared_aux_weight':1.,'detach_shared_hidden':True,'grad_clip':5.,'warmup_steps':1,'save_every':2}}
        self.config=self.dir/'research.json';self.config.write_text(json.dumps(self.cfg))
        self.data=self.make_data()
        from nbs_pg.shared_head_io_v091 import sha256
        self.metadata={'source_epoch':60,'source_step':2880,'source_world_size':1,'source_config':self.base_cfg,
                       'source_checkpoint_sha256':sha256(self.checkpoint),'source_checkpoint':str(self.checkpoint)}
        def loader(*args,**kwargs):return deepcopy(self.source),self.data,self.ontology,deepcopy(self.metadata)
        self.mock=patch('nbs_pg.shared_head_io_v091.load_source',side_effect=loader)
        self.mock.start();self.addCleanup(self.mock.stop)

    def make_data(self):
        import torch
        data=types.SimpleNamespace(weak_ids=np.arange(4,12),core_ids=np.arange(4),validation_ids=np.array([],np.int64),
                                   num_task_go=8,_signature={'representation_manifest_sha256':'fixture'})
        data.registry=types.SimpleNamespace(role_to_code={'core':0,'weak':1},role_code=np.r_[np.zeros(4,int),np.ones(8,int)],
                                            role_row=np.r_[np.arange(4),np.arange(8)])
        def labels(ids):return [np.unique([int(i)%8,(int(i)+1)%8]).astype(np.int32) for i in ids]
        gold=types.SimpleNamespace(indptr=np.arange(13)*2,go_idx=np.concatenate(labels(range(12))))
        pseudo=types.SimpleNamespace(get_role_row=lambda row:{'go_idx':labels([row+4])[0]})
        data.stores=types.SimpleNamespace(gold_messages=gold,pseudo_messages=pseudo)
        self.calls=[];self.fail_call=None
        def batch(ids,device='cpu',supervision_seed_ids=None):
            ids=np.asarray(ids);forbidden=ids if supervision_seed_ids is None else supervision_seed_ids
            self.calls.append((ids.copy(),np.array(forbidden)))
            if self.fail_call==len(self.calls):raise RuntimeError('injected interruption')
            assert np.isin(ids,forbidden).all()
            n=len(ids);src=torch.tensor(list(-ids-1)+[100,101]);blocks=[]
            for layer in range(3):
                dst=src if layer<2 else src[:n]
                mapping=torch.arange(len(dst));edge=torch.tensor([[n+i%2 for i in range(len(dst))],list(range(len(dst)))])
                blocks.append(dict(src_global_ids=src.clone(),dst_global_ids=dst.clone(),dst_in_src=mapping,
                    edge_index=edge,edge_type=torch.zeros(len(dst),dtype=torch.long),edge_attr=torch.ones(len(dst),3)))
            features=torch.sin(torch.arange((n+2)*6).reshape(n+2,6).float())
            candidate=torch.tensor([[int(i)%8,(int(i)+2)%8] for i in list(ids)+[100,101]])
            target=torch.zeros(n,8)
            for r,cols in enumerate(labels(ids)):target[r,cols]=1
            return dict(sampled_protein_x=features,sampled_node_type=torch.tensor([2]*n+[0,1]),
                sampled_candidate_go=candidate,sampled_candidate_attr=torch.ones(n+2,2,3)*.8,
                sampled_gold_edge=torch.tensor([[n],[0]]),sampled_pseudo_edge=torch.tensor([[n+1],[2]]),
                sampled_seed_index=torch.arange(n),blocks=blocks,targets=target,positive_mask=target.bool(),
                is_weak=torch.tensor(ids>=4),candidate_go=candidate[:n],candidate_attr=torch.ones(n,2,3)*.8)
        data.batch=batch;data.set_sampling_context=lambda *a,**k:None
        data.inference_batch=lambda path,ids,device:batch(ids,device)
        return data

    def cache(self):
        from scripts.nbs.cache_nbs_shared_head_v091 import main
        main(['--config',str(self.config),'--device','cpu'])

    def head_train(self,path,stop=None,resume=None):
        from scripts.nbs.train_nbs_shared_head_v091 import main
        args=['--config',str(self.config),'--work-dir',str(path),'--device','cpu']
        if stop:args+=['--stop-step',str(stop)]
        if resume:args+=['--resume',str(resume)]
        main(args)

    def convert(self,head):
        from scripts.nbs.convert_nbs_shared_head_v091 import main
        main(['--config',str(self.config),'--head-checkpoint',str(head),'--device','cpu','--output-dir',str(self.dir/'converted')])
        return self.dir/'converted/hybrid_init.pt'

    def joint(self,path,init=None,stop=None,resume=None):
        from scripts.nbs.train_nbs_hybrid_v091 import main
        args=['--config',str(self.config),'--work-dir',str(path),'--device','cpu']
        args+=['--initialization',str(init)] if init else ['--resume',str(resume)]
        if stop:args+=['--stop-step',str(stop)]
        main(args)

    def test_cache_resume_and_global_exclusion(self):
        from nbs_pg.shared_head_io_v091 import load_cache
        self.fail_call=3
        with self.assertRaisesRegex(RuntimeError,'injected'):self.cache()
        partial=json.loads((self.dir/'cache/cache_manifest.json').read_text())
        self.assertEqual(partial['completed_macros'],1)
        self.fail_call=None;self.cache()
        manifest,a=load_cache(self.dir/'cache')
        self.assertEqual(manifest['completed_macros'],2)
        self.assertEqual(manifest['candidate_stats']['core']['proteins'],4)
        self.assertEqual(manifest['candidate_stats']['weak']['proteins'],8)
        self.assertTrue(all(len(forbidden)==6 and len(ids)<=3 for ids,forbidden in self.calls))
        committed=np.array(a['hidden.f32.npy']);self.cache()
        np.testing.assert_array_equal(committed,load_cache(self.dir/'cache')[1]['hidden.f32.npy'])

    def test_actual_pipeline_conversion_gradients_and_exact_resume(self):
        import torch
        from scripts.nbs.convert_nbs_shared_head_v091 import load_hybrid
        self.cache()
        one,two=self.dir/'head_a',self.dir/'head_b'
        self.head_train(one)
        self.head_train(two,stop=2);self.head_train(two,resume=two/'latest.pt')
        a=torch.load(one/'latest.pt',weights_only=False);b=torch.load(two/'latest.pt',weights_only=False)
        for key,value in a['head_state'].items():torch.testing.assert_close(value,b['head_state'][key],atol=0,rtol=0)
        init=self.convert(one/'shared_head_step4.pt')
        state=torch.load(init,weights_only=False)
        self.assertTrue(state['conversion_check']['passed'])
        self.assertEqual(state['conversion_check']['real_batch_max_abs_logit'],0.)
        _,_,_,_,model,_=load_hybrid(self.config,init,'cpu')
        model.train()
        self.assertFalse(model.backbone.training)
        batch=self.data.batch([4,5,0]);result=model(batch,go_encoding=model.encode_go())
        result['shared'].sum().backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.shared.parameters()))
        ja,jb=self.dir/'joint_a',self.dir/'joint_b'
        self.joint(ja,init=init)
        self.joint(jb,init=init,stop=2);self.joint(jb,resume=jb/'latest.pt')
        a=torch.load(ja/'latest.pt',weights_only=False);b=torch.load(jb/'latest.pt',weights_only=False)
        for key,value in a['model'].items():torch.testing.assert_close(value,b['model'][key],atol=0,rtol=0)
        self.assertFalse(torch.equal(a['model']['backbone.node_encoder.weight'],state['model']['backbone.node_encoder.weight']))
        self.assertFalse(torch.equal(a['model']['residual_delta_weight'],state['model']['residual_delta_weight']))
        for key,value in state['model'].items():
            if key.startswith('backbone.ontology_encoder.') or 'running_' in key or 'num_batches_tracked' in key:
                torch.testing.assert_close(value,a['model'][key],atol=0,rtol=0)
        # Real metric kernels, substituting only external-file input decoding.
        from scripts.nbs import eval_nbs_shared_head_v091 as ev
        helpers=ev.metric_helpers()
        y=np.zeros((4,8),bool)
        for i in range(4):y[i,[i,i+1]]=True
        inp=self.dir/'inputs';inp.mkdir();(inp/'ind_test_input_manifest.json').write_text('{}')
        with patch.object(helpers,'load_test_inputs',return_value={'labels':y}),patch.object(ev,'metric_helpers',return_value=helpers):
            ev.main(['--config',str(self.config),'--head-checkpoint',str(one/'shared_head_step4.pt'),
                     '--input-dir',str(inp),'--metadata-file','unused','--device','cpu','--output-dir',str(self.dir/'eval_head')])
            ev.main(['--config',str(self.config),'--hybrid-checkpoint',str(ja/'hybrid_step4.pt'),
                     '--input-dir',str(inp),'--metadata-file','unused','--device','cpu','--output-dir',str(self.dir/'eval_joint')])
        self.assertTrue(json.loads((self.dir/'eval_head/shared_head_metrics.json').read_text())['composition_check']['passed'])
        self.assertTrue(json.loads((self.dir/'eval_joint/evaluation_manifest.json').read_text())['complete'])

    def test_global_role_loss_matches_summed_microbatch_gradients(self):
        import torch
        from nbs_pg.full_task_loss_v088 import full_task_loss_v088
        z=torch.randn(6,8,requires_grad=True);y=torch.zeros(6,8);y[:,0]=1
        weak=torch.tensor([True]*4+[False]*2)
        loss,_=full_task_loss_v088(z,y,weak,global_role_counts=(4,2));loss.backward();expected=z.grad.clone();z.grad=None
        # Each microbatch has both roles, like the real launcher.
        for rows in ([0,1,4],[2,3,5]):
            piece,_=full_task_loss_v088(z[rows],y[rows],weak[rows],global_role_counts=(4,2));piece.backward()
        torch.testing.assert_close(z.grad,expected,atol=1e-8,rtol=1e-7)


if __name__=='__main__':unittest.main()
