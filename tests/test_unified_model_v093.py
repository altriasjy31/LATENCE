"""CPU contracts for backbone preservation and supplied novel-GO geometry."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'nbs_models/nbs_protein_go')]
TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from nbs_pg.full_task_model_v093 import FullTaskGraphModelV093, FullTaskModelConfigV093


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
        edges = [(supports[k % len(supports)], r) for r in range(len(mapping)) for k in range(5)]
        blocks.append(dict(src_global_ids=src, dst_global_ids=src[mapping], dst_in_src=mapping,
                           edge_index=torch.tensor(edges).T,
                           edge_type=torch.arange(len(edges)) % 5, edge_attr=torch.ones(len(edges), 3)))
    return dict(sampled_protein_x=torch.randn(6, 6, generator=generator),
                sampled_node_type=torch.tensor([2, 2, 0, 0, 1, 1]),
                sampled_candidate_go=torch.randint(8, (6, 2), generator=generator),
                sampled_candidate_attr=torch.rand(6, 2, 3, generator=generator),
                sampled_gold_edge=torch.tensor([[2, 2, 3], [0, 1, 4]]),
                sampled_pseudo_edge=torch.tensor([[4, 5], [2, 6]]), blocks=blocks,
                base_logits=torch.randn(2, 8, generator=generator))


def model(seen=None, supplied_ontology=None, **kwargs):
    options = dict(hidden_dim=16, ontology_layers=1, input_dropout=0., dropout=0.,
                   raw_geometry_width=12, random_feature_dim=8)
    options.update(kwargs)
    return FullTaskGraphModelV093(6, ontology() if supplied_ontology is None else supplied_ontology,
                                 FullTaskModelConfigV093(**options), seen_task_mask=seen)


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is required for real CPU model tests')
class V093ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(930)

    def test_initial_seen_output_exactly_preserves_backbone(self):
        m, x = model().eval(), batch()
        with torch.no_grad():
            result = m(x, return_details=True)
        torch.testing.assert_close(result['logits'], x['base_logits'], atol=0, rtol=0)
        self.assertEqual(torch.count_nonzero(result['residual_logits']).item(), 0)
        self.assertEqual(result['shared_logits'].shape, (2, 8))

    def test_unseen_output_does_not_read_its_backbone_values(self):
        seen = torch.tensor([True, True, True, True, True, True, False, False])
        m = model(seen).eval()
        x = m.sanitize_evidence(batch())
        with torch.no_grad():
            expected = m(x, return_details=True)
            x['base_logits'][:, ~seen] = float('nan')
            actual = m(x, return_details=True)
        torch.testing.assert_close(actual['logits'], expected['logits'], atol=0, rtol=0)
        torch.testing.assert_close(actual['logits'][:, ~seen], actual['shared_logits'][:, ~seen], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(actual['base_logits']).all())
        self.assertTrue(torch.equal(actual['base_available_mask'], seen))

    def test_ontology_only_targets_need_no_backbone_input(self):
        m = model().eval()
        x = batch()
        del x['base_logits']
        with torch.no_grad():
            result = m(x, target_ontology_ids=[8, 9], return_details=True)
        torch.testing.assert_close(result['logits'], result['shared_logits'], atol=0, rtol=0)
        self.assertFalse(result['base_available_mask'].any())
        self.assertEqual(result['logits'].shape, (2, 2))
        with self.assertRaisesRegex(ValueError, 'requires aligned base_logits'):
            m(x)

    def test_dynamic_targets_mix_seen_and_general_in_requested_order(self):
        m, x = model().eval(), batch()
        with torch.no_grad():
            full = m(x)
            result = m(x, target_ontology_ids=[9, 0, 8, 3], return_details=True)
        torch.testing.assert_close(result['logits'][:, 1], full[:, 0], atol=0, rtol=0)
        torch.testing.assert_close(result['logits'][:, 3], full[:, 3], atol=0, rtol=0)
        self.assertEqual(result['base_available_mask'].tolist(), [False, True, False, True])

    def test_dynamic_canonical_alias_uses_first_seen_task_column(self):
        o = ontology()
        o['task_to_ontology'][3] = 0
        m, x = model(supplied_ontology=o).eval(), batch()
        x['base_logits'][:, 3] = 100.
        with torch.no_grad():
            result = m(x, target_ontology_ids=[0])
        torch.testing.assert_close(result[:, 0], x['base_logits'][:, 0], atol=0, rtol=0)

    def test_residual_and_general_head_receive_gradients(self):
        m, x = model().train(), batch()
        result = m(x, return_details=True)
        target = torch.zeros_like(result['logits'])
        target[0, [0, 2]] = 1
        target[1, [1, 4]] = 1
        loss = nn.functional.binary_cross_entropy_with_logits(result['logits'], target)
        loss += .25 * nn.functional.binary_cross_entropy_with_logits(result['shared_logits'], target)
        loss.backward()
        for prefix in ('residual_scorer', 'scorer', 'raw_geometry_encoder', 'sage_layers'):
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                                for name, p in m.named_parameters() if name.startswith(prefix)), prefix)

    def test_reload_preserves_outputs_and_parameter_shapes(self):
        m, x = model().eval(), batch()
        other = model().eval()
        other.load_state_dict(deepcopy(m.state_dict()), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(m(x), other(x), atol=0, rtol=0)
        self.assertFalse(any(isinstance(module, nn.Embedding) for module in m.modules()))
        self.assertFalse(hasattr(m, 'classifier'))

    def test_anchored_new_node_scoring_preserves_parameters_and_statistics(self):
        m, x, o = model().eval(), batch(), ontology()
        old_state = deepcopy(m.state_dict())
        old_parameters = {name: p.detach().clone() for name, p in m.named_parameters()}
        mean, scale = m.raw_geometry_mean.clone(), m.raw_geometry_scale.clone()
        o['center'] = torch.cat([o['center'], torch.full((1, 6), .25)])
        o['offset'] = torch.cat([o['offset'], torch.full((1, 6), .5)])
        o['edges']['is_a'] = torch.cat([o['edges']['is_a'], torch.tensor([[10], [2]])], dim=1)
        result = m.replace_ontology_for_inference(o, old_ontology_rows=torch.arange(10))
        self.assertEqual(m.num_ontology_go, 11)
        self.assertFalse(result['raw_geometry_statistics_refitted'])
        for name, p in m.named_parameters():
            torch.testing.assert_close(p, old_parameters[name], atol=0, rtol=0)
        self.assertEqual(set(old_state), set(m.state_dict()))
        torch.testing.assert_close(m.raw_geometry_mean, mean, atol=0, rtol=0)
        torch.testing.assert_close(m.raw_geometry_scale, scale, atol=0, rtol=0)
        with torch.no_grad():
            logits = m(x, target_ontology_ids=[10, 0])
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits[:, 1], x['base_logits'][:, 0], atol=0, rtol=0)

    def test_reordered_old_nodes_use_explicit_identity_anchors(self):
        m, x, o = model().eval(), batch(), ontology()
        old = m.encode_ontology().detach()
        permutation = torch.tensor([2, 5, 1, 9, 0, 6, 3, 8, 4, 7])
        old_rows = torch.argsort(permutation)
        updated = {'center': o['center'][permutation], 'offset': o['offset'][permutation],
                   'task_to_ontology': old_rows[o['task_to_ontology']],
                   'edges': {name: old_rows[edge] for name, edge in o['edges'].items()}}
        m.replace_ontology_for_inference(updated, old_ontology_rows=old_rows)
        with torch.no_grad():
            torch.testing.assert_close(m.encode_ontology()[old_rows], old, atol=1e-6, rtol=1e-6)

    def test_invalid_extension_fails_without_mutating_model(self):
        m, o = model().eval(), ontology()
        original = m.ontology_encoder.box_center.clone()
        with self.assertRaisesRegex(ValueError, 'old_ontology_rows is required'):
            m.replace_ontology_for_inference(o)
        changed = deepcopy(o)
        changed['center'][0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'coordinates changed'):
            m.replace_ontology_for_inference(changed, old_ontology_rows=torch.arange(10))
        changed = deepcopy(o)
        changed['task_to_ontology'][0] = 1
        with self.assertRaisesRegex(ValueError, 'immutable task GO identity'):
            m.replace_ontology_for_inference(changed, old_ontology_rows=torch.arange(10))
        changed = deepcopy(o)
        changed['edges']['part_of'] = torch.empty(2, 0, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, 'relation names'):
            m.replace_ontology_for_inference(changed, old_ontology_rows=torch.arange(10))
        torch.testing.assert_close(m.ontology_encoder.box_center, original, atol=0, rtol=0)
        self.assertEqual(m._ontology_revision, 0)

    def test_extension_during_training_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'inference-only'):
            model().train().replace_ontology_for_inference(ontology(), old_ontology_rows=torch.arange(10))

    @unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), 'CUDA anchor compatibility check')
    def test_cuda_replacement_reuses_the_original_cpu_log_transform(self):
        m = model().eval().to('cuda:0')
        result = m.replace_ontology_for_inference(ontology(), old_ontology_rows=torch.arange(10))
        self.assertTrue(result['old_coordinate_anchors_verified'])

    def test_targets_and_expert_probabilities_are_not_forward_inputs(self):
        m, x = model().eval(), batch()
        with torch.no_grad():
            expected = m(x)
            x.update(targets=torch.full((2, 8), float('nan')),
                     positive_mask=torch.ones(2, 8, dtype=torch.bool),
                     expert_prob=torch.full((2, 8), float('nan')))
            torch.testing.assert_close(m(x), expected, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
