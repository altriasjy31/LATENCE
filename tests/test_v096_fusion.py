"""v096 invariants: exact warm start, new-path learning and protein identity."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'nbs_models/nbs_protein_go'))
from nbs_pg.esm_feature_store_v096 import (ESMPenultimateStoreV096, SCHEMA,
    read_registry_protein_ids, sha256_file)
from nbs_pg.full_task_data_v096 import FullTaskDataV096
from nbs_pg.full_task_model_v095 import FullTaskGraphModelV095, FullTaskModelConfigV095
from nbs_pg.full_task_model_v096 import FullTaskGraphModelV096, FullTaskModelConfigV096
from nbs_pg.full_task_loss_v095 import FullTaskLossConfigV095, training_objective


def ontology():
    g = torch.Generator().manual_seed(111)
    return dict(center=torch.randn(7, 4, generator=g),
                offset=torch.rand(7, 4, generator=g) + .1,
                task_to_ontology=torch.arange(5),
                edges={'is_a': torch.tensor([[0, 1, 2], [1, 2, 6]])})


def graph_batch():
    g = torch.Generator().manual_seed(222)
    src = torch.tensor([-1, -2, 0, 1])
    mapping = torch.tensor([0, 1])
    blocks = [dict(src_global_ids=src, dst_global_ids=src[mapping], dst_in_src=mapping,
                   edge_index=torch.tensor([[2, 3, 2, 3], [0, 0, 1, 1]]),
                   edge_type=torch.tensor([0, 1, 2, 3]), edge_attr=torch.ones(4, 3))]
    targets = torch.tensor([[0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.]])
    return dict(sampled_global_ids=src, sampled_protein_x=torch.randn(4, 6, generator=g),
                sampled_esm_x=torch.randn(4, 9, generator=g),
                sampled_node_type=torch.tensor([2, 2, 0, 1]),
                sampled_candidate_go=torch.randint(5, (4, 2), generator=g),
                sampled_candidate_attr=torch.rand(4, 2, 3, generator=g),
                sampled_gold_edge=torch.tensor([[2, 2], [0, 1]]),
                sampled_pseudo_edge=torch.tensor([[3], [2]]), blocks=blocks,
                targets=targets, positive_mask=targets.bool(), is_weak=torch.tensor([False, True]),
                seen_task_mask=torch.ones(5, dtype=torch.bool),
                weak_modelout_prob=torch.tensor([[0.] * 5, [.001, .2, .8, .001, .1]]))


def options():
    return dict(hidden_dim=12, ontology_layers=1, sage_layers=1, input_dropout=0.,
                dropout=0., ontology_dropout=0., raw_geometry_width=8, random_feature_dim=8)


def cache(directory, names=('C0', 'W1', 'C2'), values=None, identity_extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    values = np.arange(len(names) * 3, dtype=np.float32).reshape(len(names), 3) if values is None else values
    np.save(directory / 'features.npy', values)
    np.save(directory / 'protein_ids.npy', np.asarray(names, dtype=str))
    representation = dict(checkpoints_sha256=['a' * 64], repr_layers=[34, 35, 36], ensemble='single',
                          feature_layer='output_layer.0.input', dimensions=[values.shape[1]],
                          model_source_sha256='b' * 64)
    if identity_extra:
        representation.update(identity_extra)
    manifest = dict(schema=SCHEMA, complete=True, num_proteins=len(names), feature_dim=values.shape[1],
        files={name: {'path': filename, 'sha256': sha256_file(directory / filename)}
               for name, filename in [('features', 'features.npy'), ('protein_ids', 'protein_ids.npy')]},
        representation_identity=representation)
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path


class FusionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(96)
        torch.set_num_threads(1)

    def models(self):
        old = FullTaskGraphModelV095(6, ontology(), FullTaskModelConfigV095(**options()))
        new = FullTaskGraphModelV096(6, ontology(), FullTaskModelConfigV096(**options(), esm_dim=9, esm_dropout=0.))
        new.load_v095_weights(old.state_dict())
        return old, new

    def test_zero_gate_preserves_real_graph_logits_exactly(self):
        old, new = self.models()
        old.eval(); new.eval()
        batch = graph_batch()
        with torch.no_grad():
            self.assertTrue(torch.equal(old(batch), new(batch)))
            batch['sampled_esm_x'] *= 100
            self.assertTrue(torch.equal(old(batch), new(batch)))
        self.assertEqual(set(old.state_dict()), {k for k in new.state_dict() if not new.is_esm_state_key(k)})

    def test_gate_learns_first_then_projection_and_changes_predictions(self):
        _, new = self.models()
        new.train()
        batch = graph_batch()
        # A deliberately differentiable frozen cache must still never get grad.
        batch['sampled_esm_x'].requires_grad_(True)
        optimizer = torch.optim.AdamW(new.parameters(), lr=.01, weight_decay=0.)
        before = new.esm_projection.weight.detach().clone()
        optimizer.zero_grad()
        _, loss, _ = training_objective(new, batch, FullTaskLossConfigV095(), global_role_counts=(1, 1))
        loss.backward()
        self.assertIsNone(batch['sampled_esm_x'].grad)
        self.assertGreater(abs(float(new.esm_gate.grad)), 0.)
        self.assertTrue(torch.equal(new.esm_projection.weight.grad, torch.zeros_like(before)))
        optimizer.step()
        optimizer.zero_grad()
        _, loss, _ = training_objective(new, batch, FullTaskLossConfigV095(), global_role_counts=(1, 1))
        loss.backward()
        self.assertGreater(float(new.esm_projection.weight.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(before, new.esm_projection.weight))
        new.eval()
        batch2 = deepcopy({key: value.detach().clone() if torch.is_tensor(value) else value for key, value in batch.items()})
        batch2['sampled_esm_x'] = batch2['sampled_esm_x'].roll(1, 0)
        with torch.no_grad():
            self.assertFalse(torch.equal(new(batch), new(batch2)))

    def test_migration_rejects_missing_extra_axis_and_nonzero_gate(self):
        old, new = self.models()
        state = deepcopy(old.state_dict())
        state.pop('node_encoder.weight')
        with self.assertRaises(ValueError): new.load_v095_weights(state)
        state = deepcopy(old.state_dict()); state['unknown'] = torch.ones(1)
        with self.assertRaises(ValueError): new.load_v095_weights(state)
        state = deepcopy(old.state_dict()); state['seen_task_mask'][0] = False
        with self.assertRaises(ValueError): new.load_v095_weights(state)
        new.esm_gate.data.fill_(.1)
        with self.assertRaises(ValueError): new.load_v095_weights(old.state_dict())

    def test_missing_or_misaligned_features_fail_even_with_zero_gate(self):
        _, new = self.models()
        new.eval(); batch = graph_batch()
        batch.pop('sampled_esm_x')
        with self.assertRaises(ValueError): new(batch)
        batch['sampled_esm_x'] = torch.zeros(2, 9)
        with self.assertRaises(ValueError): new(batch)
        batch['sampled_esm_x'] = torch.full((4, 9), float('nan'))
        with self.assertRaises(ValueError): new(batch)

    def test_cpu_bf16_has_finite_backward(self):
        _, new = self.models()
        new.train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            _, loss, _ = training_objective(new, graph_batch(), FullTaskLossConfigV095(), global_role_counts=(1, 1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None))


class FeatureIdentityTests(unittest.TestCase):
    def test_id_alignment_missing_and_modified_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            path = cache(temp, names=('C2', 'C0', 'W1'))
            store = ESMPenultimateStoreV096(path)
            self.assertTrue(np.array_equal(store.gather(['C0', 'C2']), [[3, 4, 5], [0, 1, 2]]))
            with self.assertRaises(ValueError): store.gather(['unknown'])
            np.save(Path(temp) / 'features.npy', np.ones((3, 3), dtype=np.float32))
            with self.assertRaises(ValueError): store.gather(['C0'])
            with self.assertRaises(ValueError): ESMPenultimateStoreV096(path)

    def test_registry_identity_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'registry.csv'
            path.write_text('protein_idx,protein_id,role,role_row_idx\n0,C0,core,0\n1,W1,weak,0\n')
            self.assertEqual(read_registry_protein_ids(path, 2).tolist(), ['C0', 'W1'])
            path.write_text('protein_idx,protein_id\n0,C0\n1,C0\n')
            with self.assertRaises(ValueError): read_registry_protein_ids(path)
            manifest = cache(Path(temp) / 'esm')
            with self.assertRaises(ValueError): ESMPenultimateStoreV096(manifest, expected_sha256='0' * 64)

    def make_data_stub(self, temp):
        data = FullTaskDataV096.__new__(FullTaskDataV096)
        data.esm = ESMPenultimateStoreV096(cache(Path(temp) / 'train', names=('W1', 'C2', 'C0')))
        data.registry = SimpleNamespace(num_proteins=3)
        data.registry_protein_ids = np.asarray(['C0', 'W1', 'C2'])
        data._registry_esm_rows = data.esm.rows_for(data.registry_protein_ids)
        data._esm_inference = {}
        return data

    def test_all_sampled_support_and_training_ghost_nodes_are_aligned(self):
        with tempfile.TemporaryDirectory() as temp:
            data = self.make_data_stub(temp)
            batch = dict(sampled_global_ids=torch.tensor([-3, 1, -1, 2]),
                         sampled_protein_x=torch.zeros(4, 6))
            original = batch['sampled_protein_x'].clone()
            result = data._attach_esm(batch)
            expected = data.esm.gather(['C2', 'W1', 'C0', 'C2'])
            self.assertTrue(np.array_equal(result['sampled_esm_x'].numpy(), expected))
            self.assertTrue(torch.equal(original, result['sampled_protein_x']))
            batch['sampled_global_ids'] = torch.tensor([-4, 1, -1, 2])
            with self.assertRaises(ValueError): data._attach_esm(batch)

    def test_external_ghost_axis_is_not_registry_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            data = self.make_data_stub(temp)
            directory = Path(temp) / 'input'; directory.mkdir()
            (directory / 'protein_ids.txt').write_text('NEW2\nNEW0\n')
            np.save(directory / 'ind_test_repr.f16.npy', np.zeros((2, 6), dtype=np.float16))
            cache(directory / 'esm_penultimate', names=('NEW0', 'NEW2'),
                  values=np.asarray([[90, 91, 92], [20, 21, 22]], dtype=np.float32))
            identity = data.inference_esm_identity(directory)
            self.assertTrue(identity['uses_supplemental_cache'])
            batch = dict(sampled_global_ids=torch.tensor([-2, 1, -1]), sampled_protein_x=torch.zeros(3, 6))
            result = data._attach_esm(batch, external=data._inference_esm(directory))
            self.assertTrue(np.array_equal(result['sampled_esm_x'].numpy(),
                [[90, 91, 92], data.esm.gather(['W1'])[0].tolist(), [20, 21, 22]]))
            (directory / 'protein_ids.txt').write_text('NEW0\nNEW2\n')
            with self.assertRaises(ValueError): data.inference_esm_identity(directory)

    def test_new_protein_requires_cache_and_same_representation_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            data = self.make_data_stub(temp)
            directory = Path(temp) / 'input'; directory.mkdir()
            (directory / 'protein_ids.txt').write_text('NEW\n')
            np.save(directory / 'ind_test_repr.f16.npy', np.zeros((1, 6), dtype=np.float16))
            with self.assertRaises(ValueError): data.inference_esm_identity(directory)
            cache(directory / 'esm_penultimate', names=('NEW',),
                  identity_extra={'checkpoints_sha256': ['c' * 64]})
            with self.assertRaises(ValueError): data.inference_esm_identity(directory)


if __name__ == '__main__':
    unittest.main()
