"""Numerical regression tests for v095 masking, ASL and macro-batch gradients."""
from pathlib import Path
import importlib.util
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT/'nbs_models/nbs_protein_go'))
    from nbs_pg.full_task_loss_v095 import (FullTaskLossConfigV095, full_task_loss_v095,
                                          build_hierarchy_task_edges)


@unittest.skipUnless(HAS_TORCH, 'PyTorch required for numerical regression checks')
class V095LossTests(unittest.TestCase):
    def evaluate(self, z, y, weak, p=None, cfg=None, **kwargs):
        return full_task_loss_v095(z, y, weak, cfg or FullTaskLossConfigV095(hierarchy_weight=0),
            weak_modelout_prob=p, seen_task_mask=kwargs.pop('seen_task_mask', torch.ones(z.shape[1], dtype=torch.bool)),
            task_to_ontology=kwargs.pop('task_to_ontology', torch.arange(z.shape[1])), **kwargs)

    def test_easy_core_negatives_have_zero_gradient_but_positives_are_retained(self):
        z = torch.tensor([[-5., -5., 0.]], requires_grad=True)
        y = torch.tensor([[0., 1., 0.]])
        loss, _ = self.evaluate(z, y, torch.tensor([False]),
            cfg=FullTaskLossConfigV095(modelout_soft_weight=0, hierarchy_weight=0, loss_scale=1))
        loss.backward()
        self.assertEqual(float(z.grad[0, 0]), 0.)
        self.assertLess(float(z.grad[0, 1]), -.3)
        self.assertGreater(float(z.grad[0, 2]), 0.)
        q = z.detach().sigmoid()
        neg = (1-q+.05).clamp(max=1)
        expected = (-F.logsigmoid(z.detach())[0,1] - neg[0,2].log()*(1-neg[0,2])**4)/3
        torch.testing.assert_close(loss, expected)

    def test_weak_unknown_is_not_hard_negative(self):
        z = torch.tensor([[0., 2., -1.]], requires_grad=True)
        y = torch.tensor([[1., 0., 0.]])
        loss, _ = self.evaluate(z, y, torch.tensor([True]),
            cfg=FullTaskLossConfigV095(modelout_soft_weight=0, hierarchy_weight=0))
        loss.backward()
        self.assertLess(float(z.grad[0, 0]), 0.)
        torch.testing.assert_close(z.grad[0, 1:], torch.zeros(2))

    def test_low_teacher_cannot_override_B_positive(self):
        z = torch.tensor([[.1, .2, .3]], requires_grad=True)
        y = torch.tensor([[1., 0., 0.]])
        p = torch.tensor([[0., .2, .01]])
        cfg = FullTaskLossConfigV095(hierarchy_weight=0, teacher_topk=1, hard_negative_topk=1, loss_scale=1)
        loss, parts = self.evaluate(z, y, torch.tensor([True]), p, cfg)
        loss.backward()
        expected_positive_gradient = (z.detach().sigmoid()[0,0]-1)/3
        torch.testing.assert_close(z.grad[0,0], expected_positive_gradient)
        self.assertEqual(float(parts['teacher_selected_pairs']), 1.)
        self.assertEqual(float(parts['hard_negative_selected_pairs']), 1.)

    def test_selected_KL_confidence_and_negative_weight(self):
        z = torch.tensor([[0., 1., -.2, -.3]], requires_grad=True)
        y = torch.zeros_like(z)
        p = torch.tensor([[.8, .005, .02, .2]])
        cfg = FullTaskLossConfigV095(hierarchy_weight=0, teacher_topk=1, hard_negative_topk=1, loss_scale=1)
        loss, parts = self.evaluate(z, y, torch.tensor([True]), p, cfg)
        loss.backward()
        expected = torch.zeros_like(z)
        expected[0,0] = (z.detach().sigmoid()[0,0]-.8)*(.8**.5)/4
        expected[0,1] = (z.detach().sigmoid()[0,1]-.005)*(.995**.5)*.25/4
        torch.testing.assert_close(z.grad, expected)
        self.assertEqual(float(parts['weak_modelout_supervised_pairs']), 2.)

    def test_unseen_nan_and_alias_cells_have_no_gradient(self):
        z = torch.tensor([[0., 2., .3, float('nan')]], requires_grad=True)
        y = torch.tensor([[1., 0., 0., 0.]])
        p = torch.tensor([[.9, 0., .2, float('nan')]])
        loss, _ = self.evaluate(z, y, torch.tensor([True]), p,
            seen_task_mask=torch.tensor([True, True, True, False]),
            task_to_ontology=torch.tensor([0,0,1,2]))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(float(z.grad[0,1]), 0.)
        self.assertEqual(float(z.grad[0,3]), 0.)

    def test_microbatch_and_DDP_loss_and_gradients_match_global(self):
        generator = torch.Generator().manual_seed(74)
        z = torch.randn(7, 12, generator=generator)
        y = (torch.rand(7, 12, generator=generator) > .85).float()
        weak = torch.tensor([False, True, True, False, True, True, False])
        p = torch.rand(7, 12, generator=generator)
        p[:, 0] = .005
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        cfg = FullTaskLossConfigV095(teacher_topk=3, hard_negative_topk=1)
        complete = z.clone().requires_grad_()
        full, _ = self.evaluate(complete, y, weak, p, cfg, hierarchy_edges=edges)
        full.backward()
        split = z.clone().requires_grad_()
        partial = []
        for sl in (slice(0,2), slice(2,3), slice(3,7)):
            term, _ = self.evaluate(split[sl], y[sl], weak[sl], p[sl], cfg,
                hierarchy_edges=edges, global_role_counts=(4,3))
            partial.append(term)
        summed = sum(partial)
        summed.backward()
        torch.testing.assert_close(summed, full)
        torch.testing.assert_close(split.grad, complete.grad)
        ddp_z = z.clone().requires_grad_()
        rank_losses = []
        for sl in (slice(0,3), slice(3,7)):
            term, _ = self.evaluate(ddp_z[sl], y[sl], weak[sl], p[sl], cfg,
                hierarchy_edges=edges, global_role_counts=(4,3), world_size=2)
            rank_losses.append(term)
        average = sum(rank_losses)/2
        average.backward()
        torch.testing.assert_close(average, full)
        torch.testing.assert_close(ddp_z.grad, complete.grad)

    def test_hierarchy_orientation_canonical_dedup_and_unseen_exclusion(self):
        ontology = dict(task_to_ontology=torch.tensor([0,0,1,2,3]),
            edges={'is_a': torch.tensor([[0,0,1,2], [1,1,2,3]]),
                   'part_of': torch.tensor([[0],[1]]),
                   'has_child': torch.tensor([[3],[2]])})
        seen = torch.tensor([True,True,True,True,False])
        edges = build_hierarchy_task_edges(ontology, seen)
        torch.testing.assert_close(edges, torch.tensor([[2,3],[0,2]]))
        z = torch.logit(torch.tensor([[.2,.8]]))
        y = torch.zeros_like(z)
        weak = torch.tensor([False])
        cfg = FullTaskLossConfigV095(modelout_soft_weight=0, hierarchy_weight=.01, loss_scale=1)
        _, parts = self.evaluate(z, y, weak, cfg=cfg, hierarchy_edges=torch.tensor([[0],[1]]))
        torch.testing.assert_close(parts['hierarchy'], torch.tensor(.6))
        torch.testing.assert_close(parts['contrib_hierarchy'], torch.tensor(.006))
        _, good = self.evaluate(z.flip(1), y, weak, cfg=cfg, hierarchy_edges=torch.tensor([[0],[1]]))
        self.assertEqual(float(good['hierarchy']), 0.)

    def test_config_rejects_invalid_thresholds_and_topk(self):
        for kwargs in ({'teacher_min_probability':0}, {'teacher_topk':1.5},
                       {'loss_scale':0}, {'hard_negative_weight':2}, {'asl_clip':1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FullTaskLossConfigV095(**kwargs)


if __name__ == '__main__':
    unittest.main()
