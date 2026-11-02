"""Numerical contracts for direct shared scoring and weak-primary supervision."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F
    # A standalone overlay can be checked against its assembled project;
    # installed copies need only their own project tree.
    for candidate in (ROOT.parent/'project/nbs_models/nbs_protein_go',
                      ROOT/'nbs_models/nbs_protein_go'):
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
    import nbs_pg
    nbs_pg.__path__.insert(0, str(ROOT/'nbs_models/nbs_protein_go/nbs_pg'))
    from nbs_pg.full_task_model_v094 import FullTaskGraphModelV094, FullTaskModelConfigV094
    from nbs_pg.full_task_loss_v094 import FullTaskLossConfigV094, full_task_loss_v094


def ontology():
    generator = torch.Generator().manual_seed(210)
    return {'center': torch.randn(10, 6, generator=generator),
            'offset': torch.rand(10, 6, generator=generator) + .1,
            'task_to_ontology': torch.arange(8),
            'edges': {'is_a': torch.tensor([[0, 1, 2, 3], [1, 2, 8, 9]])}}


def batch():
    generator = torch.Generator().manual_seed(22)
    blocks = []
    stages = [([-1, -2, 0, 1, 4, 5], [1, 2, 0, 4]),
              ([-2, 0, -1, 4], [1, 0, 2]), ([0, -2, -1], [2, 1])]
    for src, mapping in stages:
        src, mapping = torch.tensor(src), torch.tensor(mapping)
        supports = torch.where(src >= 0)[0].tolist()
        edges = [(supports[k % len(supports)], r)
                 for r in range(len(mapping)) for k in range(5)]
        blocks.append(dict(src_global_ids=src, dst_global_ids=src[mapping], dst_in_src=mapping,
                           edge_index=torch.tensor(edges).T,
                           edge_type=torch.arange(len(edges)) % 5,
                           edge_attr=torch.ones(len(edges), 3)))
    return dict(sampled_protein_x=torch.randn(6, 6, generator=generator),
                sampled_node_type=torch.tensor([2, 2, 0, 0, 1, 1]),
                sampled_candidate_go=torch.randint(8, (6, 2), generator=generator),
                sampled_candidate_attr=torch.rand(6, 2, 3, generator=generator),
                sampled_gold_edge=torch.tensor([[2, 2, 3], [0, 1, 4]]),
                sampled_pseudo_edge=torch.tensor([[4, 5], [2, 6]]), blocks=blocks)


def model(seen=None):
    options = FullTaskModelConfigV094(hidden_dim=16, ontology_layers=1,
                                     input_dropout=0., dropout=0.,
                                     raw_geometry_width=12, random_feature_dim=8)
    return FullTaskGraphModelV094(6, ontology(), options, seen_task_mask=seen)


def loss_inputs():
    z = torch.tensor([[-1., .4, .7], [.8, -.2, 1.1], [-.7, .5, .1], [.2, -.9, -.4]])
    y = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 0., 1.], [1., 0., 0.]])
    weak = torch.tensor([False, True, True, False])
    probability = torch.tensor([[float('nan')]*3, [.8, .2, .6], [.4, .7, .9], [float('nan')]*3])
    return z, y, weak, probability


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for direct model/loss numerical checks')
class V094LossTests(unittest.TestCase):
    def evaluate(self, z, y, weak, probability, config=None, **kwargs):
        return full_task_loss_v094(z, y, weak, config,
                                  weak_modelout_prob=probability,
                                  seen_task_mask=kwargs.pop('seen_task_mask',
                                      torch.ones(z.shape[1], dtype=torch.bool)),
                                  task_to_ontology=kwargs.pop('task_to_ontology',
                                      torch.arange(z.shape[1])), **kwargs)

    def test_role_means_and_dense_KL_match_independent_distribution_formula(self):
        z, y, weak, p = loss_inputs()
        total, parts = self.evaluate(z, y, weak, p)
        element = F.binary_cross_entropy_with_logits(z, y, reduction='none')
        hard = .2*element[~weak].mean() + .8*element[weak].mean()
        teacher = torch.distributions.Bernoulli(probs=p[weak])
        predicted = torch.distributions.Bernoulli(logits=z[weak])
        soft = torch.distributions.kl_divergence(teacher, predicted).mean()
        torch.testing.assert_close(total, hard+soft)
        torch.testing.assert_close(parts['main_binary'], hard)
        torch.testing.assert_close(parts['modelout_soft_KL'], soft)
        self.assertNotIn('rare_backbone_KL', parts)

    def test_microbatch_loss_and_gradients_sum_to_global_batch(self):
        z, y, weak, p = loss_inputs()
        full_z = z.clone().requires_grad_()
        full, _ = self.evaluate(full_z, y, weak, p)
        full.backward()
        micro_z = z.clone().requires_grad_()
        total = 0.
        # First microbatch contains only core; second contains only weak.
        for indices in ([0, 3], [1, 2]):
            indices = torch.tensor(indices)
            term, _ = self.evaluate(micro_z[indices], y[indices], weak[indices], p[indices],
                                    global_role_counts=(2, 2))
            total = total + term
        total.backward()
        torch.testing.assert_close(total, full)
        torch.testing.assert_close(micro_z.grad, full_z.grad)

    def test_DDP_rank_mean_has_same_loss_and_parameter_gradient(self):
        z, y, weak, p = loss_inputs()
        parameter = torch.tensor(.3, requires_grad=True)
        expected, _ = self.evaluate(z+parameter, y, weak, p)
        expected.backward()
        expected_grad = parameter.grad.clone()
        parameter.grad = None
        ranks = []
        for indices in ([0, 1, 3], [2]):
            indices = torch.tensor(indices)
            value, _ = self.evaluate(z[indices]+parameter, y[indices], weak[indices], p[indices],
                                     global_role_counts=(2, 2), world_size=2)
            ranks.append(value)
        averaged = sum(ranks)/2
        averaged.backward()
        torch.testing.assert_close(averaged, expected)
        torch.testing.assert_close(parameter.grad, expected_grad)

    def test_dense_modelout_teaches_a_backbone_negative(self):
        z = torch.zeros((1, 1), requires_grad=True)
        total, _ = self.evaluate(z, torch.zeros_like(z), torch.tensor([True]),
                                 torch.tensor([[.9]]),
                                 FullTaskLossConfigV094(core_weight=.25, weak_weight=1.,
                                                       modelout_soft_weight=2.))
        total.backward()
        self.assertLess(z.grad.item(), 0.)  # Gradient descent raises this B-negative score.

    def test_soft_KL_is_zero_at_teacher_and_teacher_is_detached(self):
        p = torch.tensor([[.2, .7]], requires_grad=True)
        z = torch.logit(p.detach()).requires_grad_()
        total, parts = self.evaluate(z, torch.tensor([[0., 1.]]), torch.tensor([True]), p)
        self.assertAlmostEqual(parts['modelout_soft_KL'].item(), 0., places=6)
        total.backward()
        self.assertIsNone(p.grad)

    def test_core_teacher_rows_are_ignored_including_NaN(self):
        z, y, weak, p = loss_inputs()
        expected, _ = self.evaluate(z, y, weak, p)
        changed = p.clone(); changed[~weak] = -1000.
        actual, _ = self.evaluate(z, y, weak, changed)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_unseen_NaNs_are_excluded_from_every_term_and_gradient(self):
        z = torch.tensor([[.2, float('nan'), -.3]], requires_grad=True)
        target = torch.tensor([[1., 0., 0.]])
        p = torch.tensor([[.8, float('nan'), .1]])
        total, _ = self.evaluate(z, target, torch.tensor([True]), p,
                                 seen_task_mask=torch.tensor([True, False, True]))
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(z.grad[0, 1].item(), 0.)
        target[0, 1] = 1.
        with self.assertRaises(ValueError):
            self.evaluate(z.detach(), target, torch.tensor([True]), p,
                          seen_task_mask=torch.tensor([True, False, True]))

    def test_hard_PU_exclusion_does_not_remove_dense_teacher_signal(self):
        z = torch.zeros((1, 2), requires_grad=True)
        target = torch.tensor([[1., 0.]])
        total, _ = self.evaluate(z, target, torch.tensor([True]), torch.tensor([[.8, .9]]),
                                 pu_exclusion_mask=torch.tensor([[False, True]]))
        total.backward()
        self.assertLess(z.grad[0, 1].item(), 0.)

    def test_invalid_weights_and_supervised_probabilities_fail(self):
        for field in ('core_weight', 'weak_weight', 'modelout_soft_weight'):
            for value in (-1., float('nan'), float('inf'), True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    FullTaskLossConfigV094(**{field: value})
        with self.assertRaises(ValueError):
            FullTaskLossConfigV094(core_weight=0., weak_weight=0.)
        z, y, weak, p = loss_inputs()
        p[weak] = 1.1
        with self.assertRaises(ValueError):
            self.evaluate(z, y, weak, p)


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for direct model/loss numerical checks')
class V094ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(940)

    def test_single_head_forward_ignores_backbone_and_teacher_inputs(self):
        m, x = model().eval(), batch()
        with torch.no_grad():
            expected = m(x, return_details=True)
            x['base_logits'] = torch.full((2, 8), float('nan'))
            x['weak_modelout_prob'] = torch.full((2, 8), float('nan'))
            x['targets'] = torch.full((2, 8), float('nan'))
            actual = m(x, return_details=True)
        torch.testing.assert_close(actual['logits'], expected['logits'], atol=0, rtol=0)
        torch.testing.assert_close(actual['shared_logits'], actual['logits'], atol=0, rtol=0)
        self.assertFalse(hasattr(m, 'residual_scorer'))
        self.assertTrue(torch.isfinite(actual['logits']).all())

    def test_requested_GO_order_and_seen_unseen_share_readout(self):
        seen = torch.tensor([True]*6+[False]*2)
        m = model(seen).eval()
        x = m.sanitize_evidence(batch())
        with torch.no_grad():
            closed = m(x)
            dynamic = m(x, target_ontology_ids=[9, 0, 7, 8, 3])
        self.assertEqual(tuple(dynamic.shape), (2, 5))
        torch.testing.assert_close(dynamic[:, 1], closed[:, 0])
        torch.testing.assert_close(dynamic[:, 2], closed[:, 7])
        torch.testing.assert_close(dynamic[:, 4], closed[:, 3])

    def test_shared_head_and_encoder_receive_training_gradients(self):
        m, x = model().train(), batch()
        output = m(x)
        target = torch.zeros_like(output); target[0, [1, 3]] = 1.; target[1, [2, 5]] = 1.
        F.binary_cross_entropy_with_logits(output, target).backward()
        for prefix in ('scorer', 'raw_geometry_encoder', 'sage_layers'):
            self.assertTrue(any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                and parameter.grad.abs().sum() > 0
                                for name, parameter in m.named_parameters() if name.startswith(prefix)), prefix)

    def test_reload_is_exact_and_anchored_extension_adds_no_parameters(self):
        original = ontology()
        m, other, x = model().eval(), model().eval(), batch()
        other.load_state_dict(deepcopy(m.state_dict()), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(m(x), other(x), atol=0, rtol=0)
        count = sum(p.numel() for p in m.parameters())
        updated = deepcopy(original)
        updated['center'] = torch.cat((original['center'], original['center'][:1]+.2))
        updated['offset'] = torch.cat((original['offset'], original['offset'][:1]))
        m.replace_ontology_for_inference(updated, old_ontology_rows=torch.arange(10))
        self.assertEqual(sum(p.numel() for p in m.parameters()), count)
        with torch.no_grad():
            scored = m(x, target_ontology_ids=[10, 8])
        self.assertEqual(tuple(scored.shape), (2, 2))
        self.assertTrue(torch.isfinite(scored).all())


if __name__ == '__main__':
    unittest.main()
