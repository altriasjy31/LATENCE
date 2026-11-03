"""CPU integration: real sampled hetero model, objective, and optimizer restart."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nbs_models/nbs_protein_go'))
import torch
from nbs_pg.full_task_model_v095 import FullTaskGraphModelV095, FullTaskModelConfigV095
from nbs_pg.full_task_loss_v095 import FullTaskLossConfigV095, training_objective
from nbs_pg.optimizer_v095 import configure_training_runtime


def ontology():
    g = torch.Generator().manual_seed(210)
    return dict(center=torch.randn(10, 6, generator=g),
                offset=torch.rand(10, 6, generator=g) + .1,
                task_to_ontology=torch.arange(8),
                edges={'is_a': torch.tensor([[0, 1, 2, 3], [1, 2, 8, 9]])})


def batch():
    g = torch.Generator().manual_seed(22)
    blocks = []
    stages = [([-1, -2, 0, 1, 4, 5], [1, 2, 0, 4]),
              ([-2, 0, -1, 4], [1, 0, 2]), ([0, -2, -1], [2, 1])]
    for src, mapping in stages:
        src, mapping = torch.tensor(src), torch.tensor(mapping)
        supports = torch.where(src >= 0)[0].tolist()
        edges = [(supports[k % len(supports)], r)
                 for r in range(len(mapping)) for k in range(5)]
        blocks.append(dict(src_global_ids=src, dst_global_ids=src[mapping], dst_in_src=mapping,
                           edge_index=torch.tensor(edges).T, edge_type=torch.arange(len(edges)) % 5,
                           edge_attr=torch.ones(len(edges), 3)))
    target = torch.zeros(2, 8)
    target[0, [1, 3]], target[1, [2, 5]] = 1., 1.
    return dict(sampled_protein_x=torch.randn(6, 6, generator=g),
                sampled_node_type=torch.tensor([2, 2, 0, 0, 1, 1]),
                sampled_candidate_go=torch.randint(8, (6, 2), generator=g),
                sampled_candidate_attr=torch.rand(6, 2, 3, generator=g),
                sampled_gold_edge=torch.tensor([[2, 2, 3], [0, 1, 4]]),
                sampled_pseudo_edge=torch.tensor([[4, 5], [2, 6]]), blocks=blocks,
                targets=target, positive_mask=target.bool(), is_weak=torch.tensor([False, True]),
                seen_task_mask=torch.ones(8, dtype=torch.bool),
                weak_modelout_prob=torch.tensor([[float('nan')] * 8,
                                                [.001, .2, .8, .3, .001, .9, .1, .001]]))


def model():
    cfg = FullTaskModelConfigV095(hidden_dim=16, ontology_layers=1,
                                 input_dropout=0., dropout=0., raw_geometry_width=12,
                                 random_feature_dim=8)
    return FullTaskGraphModelV095(6, ontology(), cfg)


def optimizer(m):
    base = SimpleNamespace(torch=torch)
    configure_training_runtime(base, {'full_task': {'optimizer': dict(
        name='adamw', eps=1e-8, betas=[.9, .999], decay_policy='matrices_only')}})
    return base.torch.optim.AdamW(m.parameters(), lr=.003, weight_decay=.0001)


class V095IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(95)
        torch.set_num_threads(1)

    def step(self, m, opt, x):
        opt.zero_grad(set_to_none=True)
        _, loss, parts = training_objective(m, x, FullTaskLossConfigV095(),
                                            global_role_counts=(1, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        return loss.detach(), parts

    def test_real_graph_objective_trains_shared_head_geometry_and_relations(self):
        m, x = model(), batch()
        before = deepcopy(m.state_dict())
        loss, parts = self.step(m, optimizer(m), x)
        self.assertTrue(torch.isfinite(loss))
        for prefix in ('scorer.', 'raw_geometry_encoder.', 'sage_layers.'):
            self.assertTrue(any(name.startswith(prefix) and not torch.equal(value, before[name])
                                for name, value in m.state_dict().items()), prefix)
        self.assertEqual(m.config.jk_mode, 'sum')
        self.assertTrue(any(isinstance(layer, torch.nn.BatchNorm1d) for layer in m.modules()))
        self.assertGreater(parts['teacher_selected_pairs'].item(), 0)
        self.assertGreater(parts['hard_negative_selected_pairs'].item(), 0)
        self.assertEqual(tuple(m.hierarchy_task_edges.shape), (2, 2))

    def test_optimizer_resume_matches_uninterrupted_next_update(self):
        m, x = model(), batch()
        opt = optimizer(m)
        self.step(m, opt, x)
        restored = model()
        restored.load_state_dict(deepcopy(m.state_dict()), strict=True)
        resumed_opt = optimizer(restored)
        resumed_opt.load_state_dict(deepcopy(opt.state_dict()))
        first, _ = self.step(m, opt, x)
        second, _ = self.step(restored, resumed_opt, x)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name, value in m.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        for group in opt.param_groups:
            self.assertTrue(all((p.ndim >= 2) == (group['weight_decay'] > 0) for p in group['params']))

    def test_checkpoint_roundtrip_inference_needs_no_teacher_or_targets(self):
        m, x = model(), batch()
        self.step(m, optimizer(m), x)
        m.eval()
        saved = dict(model_config=asdict(m.config), model=deepcopy(m.state_dict()))
        restored = FullTaskGraphModelV095(6, ontology(), FullTaskModelConfigV095(**saved['model_config'])).eval()
        restored.load_state_dict(saved['model'], strict=True)
        inference = {k: v for k, v in x.items() if k not in
                     ('targets', 'positive_mask', 'weak_modelout_prob', 'is_weak')}
        with torch.no_grad():
            expected = m(inference)
            actual = restored(inference)
            extended = restored(inference, target_ontology_ids=[8, 1, 9])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(extended[:, 1], actual[:, 1])
        self.assertTrue(torch.isfinite(extended).all())

    def test_runtime_optimizer_proxy_does_not_mutate_global_torch(self):
        original = torch.optim.AdamW
        m = model()
        opt = optimizer(m)
        self.assertIs(torch.optim.AdamW, original)
        self.assertIsInstance(opt, original)

    def test_cpu_bfloat16_autocast_has_finite_graph_forward_and_backward(self):
        m, x = model(), batch()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output, loss, _ = training_objective(m, x, FullTaskLossConfigV095(),
                                                 global_role_counts=(1, 1))
        loss.backward()
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))


if __name__ == '__main__':
    unittest.main()
