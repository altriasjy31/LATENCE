"""Real CPU forward/backward and novel-protein/GO feature contracts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'nbs_models/nbs_protein_go')]
from nbs_pg.full_task_model_v092 import FullTaskGraphModelV092,FullTaskModelConfigV092,FixedRandomFeatureProjectionV092
from nbs_pg.full_task_loss_v090 import full_task_loss_v090
from nbs_pg.full_task_loss_v088 import full_task_loss_v088


def ontology():
    generator=torch.Generator().manual_seed(210)
    return {'center':torch.randn(10,6,generator=generator),'offset':torch.rand(10,6,generator=generator)+.1,
            'task_to_ontology':torch.arange(8),'edges':{'is_a':torch.tensor([[0,1,2,3],[1,2,8,9]])}}


def batch():
    generator=torch.Generator().manual_seed(22)
    blocks=[]
    stages=[([-1,-2,0,1,4,5],[1,2,0,4]),([-2,0,-1,4],[1,0,2]),([0,-2,-1],[2,1])]
    for src,mapping in stages:
        src=torch.tensor(src);mapping=torch.tensor(mapping)
        supports=torch.where(src>=0)[0].tolist()
        edges=[(supports[k%len(supports)],r) for r in range(len(mapping)) for k in range(5)]
        blocks.append(dict(src_global_ids=src,dst_global_ids=src[mapping],dst_in_src=mapping,
            edge_index=torch.tensor(edges).T,edge_type=torch.arange(len(edges))%5,edge_attr=torch.ones(len(edges),3)))
    return dict(sampled_protein_x=torch.randn(6,6,generator=generator),sampled_node_type=torch.tensor([2,2,0,0,1,1]),
        sampled_candidate_go=torch.randint(8,(6,2),generator=generator),sampled_candidate_attr=torch.rand(6,2,3,generator=generator),
        sampled_gold_edge=torch.tensor([[2,2,3],[0,1,4]]),sampled_pseudo_edge=torch.tensor([[4,5],[2,6]]),blocks=blocks)


def model(**kwargs):
    options=dict(hidden_dim=16,ontology_layers=1,input_dropout=0.,dropout=0.,raw_geometry_width=12,
                 random_feature_dim=8)
    options.update(kwargs)
    return FullTaskGraphModelV092(6,ontology(),FullTaskModelConfigV092(**options))


class V092ModelTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(920)
    def test_real_training_updates_both_go_paths_and_graph(self):
        m=model().train();x=batch();y=torch.zeros(2,8);y[0,[0,2]]=1;y[1,[1,4]]=1
        logits=m(x)
        loss,_=full_task_loss_v090(logits,y,torch.tensor([True,False]),task_to_ontology=m.task_to_ontology)
        loss.backward()
        for prefix in ('raw_geometry_encoder','ontology_encoder','go_fusion_norm','node_encoder','scorer','sage_layers'):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                for name,p in m.named_parameters() if name.startswith(prefix)),prefix)
        optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
        old=m.raw_geometry_encoder[0].weight.detach().clone();optimizer.step()
        self.assertFalse(torch.equal(old,m.raw_geometry_encoder[0].weight))
    def test_ontology_control_uses_same_final_fusion_norm(self):
        m=model(go_input_mode='ontology').eval()
        self.assertFalse(hasattr(m,'raw_geometry_encoder'))
        from nbs_pg.full_task_model_v090 import FullTaskGraphModelV090
        expected=m.go_fusion_norm(FullTaskGraphModelV090.encode_ontology(m))
        torch.testing.assert_close(m.encode_ontology(),expected,atol=0,rtol=0)
    def test_raw_path_really_reaches_go_encoding(self):
        m=model().eval();before=m.encode_ontology().detach()
        with torch.no_grad():m.raw_geometry_encoder[-1].bias[0].add_(1)
        self.assertGreater(float((before-m.encode_ontology()).abs().max()),.01)
    def test_raw_statistics_use_only_supplied_geometry(self):
        m=model();o=ontology();raw=torch.cat([o['center'],o['offset'].abs().log()],-1)
        torch.testing.assert_close(m.raw_geometry_mean,raw.mean(0),rtol=0,atol=0)
        torch.testing.assert_close(m.raw_geometry_scale,raw.std(0,unbiased=False).clamp_min(1e-6),rtol=0,atol=0)
    def test_new_ontology_rows_can_be_scored_without_parameter_rows(self):
        m=model().eval();x=batch()
        with torch.no_grad():
            go=m.encode_ontology();task=m(x,ontology_encoding=go)
            dynamic=m(x,target_ontology_ids=torch.tensor([9,0,8]),ontology_encoding=go)
        self.assertEqual(dynamic.shape,(2,3))
        torch.testing.assert_close(dynamic[:,1],task[:,0],atol=1e-6,rtol=1e-6)
        self.assertFalse(hasattr(m,'classifier'))
        self.assertFalse(any(isinstance(mod,nn.Embedding) for mod in m.modules()))
    def test_no_forward_dependency_on_gold_targets_or_dense_teacher(self):
        m=model().eval();x=batch()
        with torch.no_grad():
            expected=m(x)
            x.update(targets=torch.ones(2,8),positive_mask=torch.ones(2,8,dtype=torch.bool),
                     expert_prob=torch.full((2,8),float('nan')),base_logits=torch.full((2,8),float('nan')))
            torch.testing.assert_close(expected,m(x),atol=0,rtol=0)
    def test_rff_is_repeatable_and_available_for_new_input_rows(self):
        m=model(protein_random_features='rff').eval();projection=m.node_encoder
        x=torch.randn(5,6)
        expected=projection.random_features(x)
        actual=torch.cat([projection.random_features(x[i:i+1]) for i in range(len(x))])
        torch.testing.assert_close(expected,actual,atol=1e-6,rtol=1e-6)
        self.assertEqual(projection(torch.randn(7,6)).shape,(7,16))
        self.assertFalse(projection.random_matrix.requires_grad)
        self.assertFalse(projection.random_phase.requires_grad)
        self.assertIn('node_encoder.random_matrix',m.state_dict())
    def test_rff_branch_has_trainable_shared_gradients(self):
        m=model(protein_random_features='rff').train();z=m(batch());z.square().mean().backward()
        self.assertGreater(float(m.node_encoder.random_projection.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.node_encoder.stage1.weight.grad.abs().sum()),0)
        self.assertIsNone(m.node_encoder.random_matrix.grad)
    def test_state_reload_preserves_random_features_and_outputs(self):
        m=model(protein_random_features='rff').eval();x=batch();state=deepcopy(m.state_dict())
        other=model(protein_random_features='rff').eval();other.load_state_dict(state,strict=True)
        with torch.no_grad():torch.testing.assert_close(m(x),other(x),atol=0,rtol=0)
    def test_all_seen_objective_preserves_original_bce(self):
        z=torch.randn(3,8);y=torch.zeros(3,8);y[0,0]=1;y[1,1]=1;y[2,2]=1;role=torch.tensor([True,True,False])
        a,_=full_task_loss_v090(z,y,role,task_to_ontology=torch.arange(8))
        b,_=full_task_loss_v088(z,y,role)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    def test_invalid_controls_fail_before_training(self):
        for kwargs in ({'go_input_mode':'bad'},{'raw_geometry_width':0},{'protein_random_features':'trainable_id'},
                       {'random_feature_dim':3},{'random_feature_seed':-1},{'random_feature_scale':float('nan')}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):model(**kwargs)


if __name__=='__main__':unittest.main()
