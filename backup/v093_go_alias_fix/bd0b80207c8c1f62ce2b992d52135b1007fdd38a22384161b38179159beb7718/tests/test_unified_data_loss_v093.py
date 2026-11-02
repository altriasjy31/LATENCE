from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    ROOT = Path(__file__).resolve().parents[1]
    candidates = [ROOT/'nbs_models/nbs_protein_go',
                  ROOT.parent/'project/nbs_models/nbs_protein_go']
    for candidate in reversed(candidates):
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
    import nbs_pg
    nbs_pg.__path__.insert(0, str(ROOT/'nbs_models/nbs_protein_go/nbs_pg'))
    from nbs_pg.full_task_data_v093 import FullTaskDataV093, build_latence_nbs_stores_v093
    from nbs_pg.full_task_loss_v093 import FullTaskLossConfigV093, full_task_loss_v093, training_objective


def make_data():
    data = object.__new__(FullTaskDataV093)
    data.num_task_go = 4
    data.feature_dim = 2
    data.seen_task_mask = np.array([True, True, False, True])
    data.core_counts = np.array([2, 1, 0, 1000], np.int64)
    data.registry = SimpleNamespace(num_proteins=4, role_code=np.array([1, 0, 1, 0]),
        role_to_code={'core': 0, 'weak': 1}, role_row=np.array([1, 1, 0, 0]))
    gold = SimpleNamespace(indptr=np.array([0, 0, 2, 2, 3]), go_idx=np.array([2, 3, 0]))
    pseudo = SimpleNamespace(indptr=np.array([0, 1, 3]), go_idx=np.array([1, 0, 2]),
                             probability=np.array([.3, .5, .9], np.float32))
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    base_store = SimpleNamespace(probability_clip=1e-5,
        gather_matrix=lambda ids, go: np.tile(np.array([-.1, .2, .3, .4], np.float32)[:, None], (1, len(ids))))
    data.stores = SimpleNamespace(gold_messages=gold, pseudo_messages=pseudo,
        features=SimpleNamespace(gather=lambda ids: features[ids]),
        episode_sampler=SimpleNamespace(base_logit_store=base_store))
    data.candidate_topk = 16
    data._cached_candidate_go = np.full((4, 16), -1, np.int32)
    data._cached_candidate_go[:, :3] = [0, 1, 3]
    data._cached_candidate_attr = np.zeros((4, 16, 3), np.float32)
    data._cached_candidate_attr[:, :3, 0] = [.7, .8, .9]
    data._cached_candidate_attr[:, :3, 2] = [1, .5, 1/3]
    data.unified_manifest = {'candidates': {'mode': 'frequency'}}
    data._neighbors = np.array([[0], [1], [1], [0]], np.int32)
    data._neighbor_attrs = np.ones((4, 1, 3), np.float32)
    data.anchor_core_ids = np.array([1, 3], np.int64)
    data.validation_ids = np.empty(0, np.int64)
    data._permitted = np.ones(4, bool)
    data._retrieval_neighbors = np.array([[1], [3], [3], [1]], np.int32)
    data._retrieval_attrs = np.ones((4, 1, 3), np.float32)
    data._sampling_step, data._sampling_rank, data._sampling_training = 5, 0, True
    data._require_pools = lambda: None

    def blocks(query_ids, retrieval, forbidden):
        known = [i for i in (1, 3) if i not in forbidden]
        sources = np.concatenate((query_ids, np.array(known, np.int64)))
        edge = np.array([[len(query_ids) + (i % len(known)), i] for i in range(len(query_ids))], np.int64).T
        return [dict(src_global_ids=sources, dst_global_ids=query_ids,
                     dst_in_src=np.arange(len(query_ids)), edge_index=edge,
                     edge_type=np.full(len(query_ids), 4, np.int64),
                     edge_attr=np.ones((len(query_ids), 3), np.float32))], 0
    data._build_blocks = blocks
    return data


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for runtime/loss checks')
class RuntimeTests(unittest.TestCase):
    def test_bootstrap_uses_snapshot_then_replaces_weak_CSR_explicitly(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            source = {'task': 'bp', 'supervision_contract': {'weak_pseudo': {'source': 'old_modelout'}}}
            snapshot = folder/'source_config_snapshot.json'
            snapshot.write_text(json.dumps(source))
            arrays = {'ptr.npy': np.array([0, 1, 2]), 'go.npy': np.array([0, 1], np.int32),
                      'prob.npy': np.array([.3, .5], np.float32)}
            for filename, values in arrays.items():
                np.save(folder/filename, values)
            files = [snapshot.name, *arrays]
            manifest = {'schema': 'nbs-unified-dataset-v093-r1', 'complete': True,
                'source_config': {'path': str(folder/'old_config_no_longer_needed.json'),
                    'sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest()},
                'files': {'source_config_snapshot': snapshot.name}, 'sources': {}, 'num_task_go': 4,
                'pseudo': {'indptr': 'ptr.npy', 'go_idx': 'go.npy', 'probability': 'prob.npy'},
                'hashes': {f: hashlib.sha256((folder/f).read_bytes()).hexdigest() for f in files}}
            path = folder/'unified_dataset_manifest.json'; path.write_text(json.dumps(manifest))
            registry = SimpleNamespace(role_global_indices={'weak': np.array([2, 0])})
            stores = SimpleNamespace(num_task_go=4, registry=registry,
                                     episode_sampler=SimpleNamespace(), pseudo_messages='old_CSR')
            with mock.patch('nbs_pg.full_task_data_v093._build_bootstrap_stores', return_value=stores) as bootstrap:
                result = build_latence_nbs_stores_v093({'task': 'bp',
                    'data': {'unified_dataset_manifest': str(path)},
                    'supervision_contract': {'weak_pseudo': {'source': 'unified_frequency_routing'}}})
            self.assertEqual(bootstrap.call_args.args[0], source)
            self.assertAlmostEqual(float(result.pseudo_messages.get_role_row(0)['probability'][0]), .3, places=6)
            self.assertIs(result.episode_sampler.pseudo_by_protein, result.pseudo_messages)

    def test_global_core_and_role_local_weak_rows_no_probability_floor(self):
        data = make_data()
        weak, positive, soft = data._target_arrays(np.arange(4))
        np.testing.assert_array_equal(weak, [True, False, True, False])
        np.testing.assert_array_equal(positive, [[1, 0, 0, 0], [0, 0, 0, 1],
                                                 [0, 1, 0, 0], [1, 0, 0, 0]])
        self.assertAlmostEqual(float(soft[2, 1]), .3, places=6)
        self.assertEqual(soft[0, 0], .5)
        self.assertTrue((soft[:, 2] == 0).all())

    def test_real_batch_dispatch_through_graph_and_attach_blocks(self):
        data = make_data()
        batch = data.batch(np.array([0, 2]), 'cpu', supervision_seed_ids=np.array([0, 2]))
        self.assertEqual(tuple(batch['base_logits'].shape), (2, 4))
        self.assertEqual(tuple(batch['sampled_candidate_go'].shape), (4, 16))
        self.assertEqual(len(batch['blocks']), 1)
        torch.testing.assert_close(batch['targets'], batch['positive_mask'].float())
        self.assertAlmostEqual(float(batch['weak_soft_targets'][1, 1]), .3, places=6)
        self.assertFalse((batch['sampled_gold_edge'][1] == 2).any())
        self.assertFalse((batch['sampled_pseudo_edge'][1] == 2).any())
        self.assertTrue(torch.equal(batch['seen_task_mask'], torch.tensor([True, True, False, True])))

    def test_real_inference_graph_and_attach_apply_quota_before_ghost_graph(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as name:
            inp = Path(name).resolve()
            go = np.tile(np.arange(4, dtype=np.int32), (2, 1))
            attrs = np.ones((2, 4, 3), np.float32)
            attrs[..., 2] = [1, .5, 1/3, .25]
            data._inference = {str(inp): {'protein_x': np.ones((2, 2), np.float32),
                'probability': np.full((2, 4), .3, np.float32), 'candidate_go': go, 'candidate_attr': attrs,
                'neighbors': np.array([[0], [1]], np.int32), 'neighbor_attrs': np.ones((2, 1, 3), np.float32),
                'v084_neighbors': np.array([[1], [3]], np.int32), 'v084_attrs': np.ones((2, 1, 3), np.float32)}}
            result = data.inference_batch(inp, np.array([0, 1]), 'cpu')
        self.assertNotIn('targets', result)
        self.assertEqual(tuple(result['sampled_candidate_go'].shape), (4, 16))
        self.assertFalse((result['sampled_candidate_go'] == 2).any())
        self.assertFalse((result['sampled_gold_edge'][1] == 2).any())
        self.assertEqual((data._sampling_step, data._sampling_rank, data._sampling_training), (5, 0, True))

    def test_cached_quota_not_globally_retruncated(self):
        data = make_data()
        go, attrs = data._candidate_rows(np.array([0]))
        original = go.copy()
        selected, _ = data._truncate_candidates(go, attrs)
        np.testing.assert_array_equal(selected, original)

    def test_legacy_label_edges_are_filtered(self):
        data = make_data()
        batch = {'protein_x': torch.ones(1, 2),
                 'anchor_go_edge': torch.tensor([[0, 0], [1, 2]]),
                 'loss_anchor_go_edge': torch.tensor([[0, 0], [2, 3]])}
        result = data._filter_label_view(batch)
        self.assertEqual(result['anchor_go_edge'][1].tolist(), [1])
        self.assertEqual(result['loss_anchor_go_edge'][1].tolist(), [3])
        self.assertEqual(batch['anchor_go_edge'].shape[1], 2)

    def test_alias_group_cannot_be_split(self):
        data = make_data()
        data.task_to_ontology_rows = np.array([0, 0, 1, 2])
        data._unified_seen = np.array([True, False, False, True])
        with self.assertRaisesRegex(ValueError, 'canonical alias'):
            data._load_label_split('unified_dataset')


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for runtime/loss checks')
class LossTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(12)
        final = torch.randn(4, 4, requires_grad=True)
        shared = torch.randn(4, 4, requires_grad=True)
        base = torch.randn(4, 4)
        target = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1], [1, 1, 0, 0]])
        weak = torch.tensor([True, False, True, False])
        soft = target * .7
        options = {'core_counts': torch.tensor([2, 3, 0, 1000]), 'weak_soft_targets': soft,
                   'seen_task_mask': torch.tensor([True, True, False, True]),
                   'task_to_ontology': torch.arange(4)}
        return final, shared, base, target, weak, options

    def test_unseen_zero_gradient_in_all_objectives_even_nonfinite_unseen(self):
        z, s, b, y, w, opts = self.inputs()
        with torch.no_grad():
            z[:, 2] = float('nan'); s[:, 2] = float('nan'); b[:, 2] = float('nan')
            opts['weak_soft_targets'][:, 2] = float('nan')
        loss, _ = full_task_loss_v093(z, s, b, y, w, **opts)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        torch.testing.assert_close(z.grad[:, 2], torch.zeros(4))
        torch.testing.assert_close(s.grad[:, 2], torch.zeros(4))
        self.assertGreater(float(z.grad[:, [0, 1, 3]].abs().sum()), 0)

    def test_microbatch_sums_match_global_role_means(self):
        z, s, b, y, w, opts = self.inputs()
        full, _ = full_task_loss_v093(z, s, b, y, w, global_role_counts=(2, 2), **opts)
        pieces = []
        for rows in ([0, 2], [1, 3]):
            subopts = {**opts, 'weak_soft_targets': opts['weak_soft_targets'][rows]}
            part, _ = full_task_loss_v093(z[rows], s[rows], b[rows], y[rows], w[rows],
                                         global_role_counts=(2, 2), **subopts)
            pieces.append(part)
        torch.testing.assert_close(sum(pieces), full)

    def test_distributed_rank_losses_average_to_full_macro_loss(self):
        z, s, b, y, w, opts = self.inputs()
        full, _ = full_task_loss_v093(z, s, b, y, w, global_role_counts=(2, 2), **opts)
        parts = []
        for rows in ([0, 2], [1, 3]):
            subopts = {**opts, 'weak_soft_targets': opts['weak_soft_targets'][rows]}
            value, _ = full_task_loss_v093(z[rows], s[rows], b[rows], y[rows], w[rows],
                                          global_role_counts=(2, 2), world_size=2, **subopts)
            parts.append(value)
        torch.testing.assert_close(sum(parts) / 2, full)

    def test_backbone_KL_zero_when_final_equals_backbone(self):
        _, s, b, y, w, opts = self.inputs()
        z = b.detach().clone().requires_grad_(True)
        _, stats = full_task_loss_v093(z, s, b, y, w, **opts)
        self.assertAlmostEqual(float(stats['rare_backbone_KL']), 0., places=7)

    def test_unseen_positive_is_rejected(self):
        z, s, b, y, w, opts = self.inputs(); y[0, 2] = 1
        with self.assertRaisesRegex(ValueError, 'unseen positive'):
            full_task_loss_v093(z, s, b, y, w, **opts)

    def test_invalid_weights_and_soft_target_contract(self):
        for kwargs in ({'shared_weight': -1}, {'weak_weight': float('nan')}, {'rare_distill_weight': True}):
            with self.assertRaises(ValueError):
                FullTaskLossConfigV093(**kwargs)
        z, s, b, y, w, opts = self.inputs(); opts['weak_soft_targets'][0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, 'finite probabilities'):
            full_task_loss_v093(z, s, b, y, w, **opts)

    def test_objective_requires_matching_seen_mask(self):
        z, s, b, y, w, opts = self.inputs()
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('seen_task_mask', opts['seen_task_mask'])
                self.register_buffer('task_to_ontology', opts['task_to_ontology'])
            def forward(self, batch, return_details=False):
                return {'logits': z, 'shared_logits': s, 'base_logits': b}
        batch = {'targets': y, 'positive_mask': y.bool(), 'is_weak': w,
                 'weak_soft_targets': opts['weak_soft_targets'], 'core_counts': opts['core_counts'],
                 'seen_task_mask': opts['seen_task_mask']}
        logits, loss, stats = training_objective(FakeModel(), batch, FullTaskLossConfigV093(), global_role_counts=(2, 2))
        self.assertIs(logits, z); self.assertTrue(torch.isfinite(loss)); self.assertIn('shared_binary', stats)
        batch['seen_task_mask'] = torch.ones(4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, 'seen_task_mask differs'):
            training_objective(FakeModel(), batch, FullTaskLossConfigV093(), global_role_counts=(2, 2))


if __name__ == '__main__':
    unittest.main()
