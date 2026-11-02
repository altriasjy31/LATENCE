"""Runtime contracts for separate hard membership and dense soft teachers."""
from __future__ import annotations

import hashlib
import importlib.util
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
    for path in reversed([ROOT/'nbs_models/nbs_protein_go',
                          ROOT.parents[1]/'unified_v093/project/nbs_models/nbs_protein_go']):
        if path.is_dir():
            sys.path.insert(0, str(path))
    import nbs_pg
    nbs_pg.__path__.insert(0, str(ROOT/'nbs_models/nbs_protein_go/nbs_pg'))
    from nbs_pg import full_task_data_v094 as runtime


def make_data():
    data = object.__new__(runtime.FullTaskDataV094)
    data.num_task_go, data.feature_dim = 4, 2
    data.seen_task_mask = np.array([True, True, False, True])
    data.core_counts = np.array([2, 1, 0, 1000], np.int64)
    data.registry = SimpleNamespace(num_proteins=4, role_code=np.array([1, 0, 1, 0]),
        role_to_code={'core': 0, 'weak': 1}, role_row=np.array([1, 1, 0, 0]),
        role_global_indices={'weak': np.array([2, 0]), 'core': np.array([3, 1])})
    gold = SimpleNamespace(indptr=np.array([0, 0, 2, 2, 3]), go_idx=np.array([2, 3, 0]))
    pseudo = SimpleNamespace(indptr=np.array([0, 0, 1]), go_idx=np.array([0]),
                             probability=np.array([0.], np.float32))
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    forbidden_base = SimpleNamespace(gather_matrix=mock.Mock(side_effect=AssertionError('B logits must not be read')))
    data.stores = SimpleNamespace(gold_messages=gold, pseudo_messages=pseudo,
        features=SimpleNamespace(gather=lambda ids: features[ids]),
        episode_sampler=SimpleNamespace(base_logit_store=forbidden_base))
    data._modelout_probability = np.array([[.05, .2, np.nan, .99], [.8, .03, np.inf, .2]], np.float32)
    data.candidate_topk = 16
    data._cached_candidate_go = np.full((4, 16), -1, np.int32)
    data._cached_candidate_go[:, :3] = [0, 1, 3]
    data._cached_candidate_attr = np.zeros((4, 16, 3), np.float32)
    data._cached_candidate_attr[:, :3, 0] = [.7, .8, .9]
    data._cached_candidate_attr[:, :3, 2] = [1, .5, 1/3]
    data.unified_manifest = {'candidates': {'mode': 'global'}}
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
        edge = np.array([[len(query_ids)+(i % len(known)), i]
                         for i in range(len(query_ids))], np.int64).T
        return [dict(src_global_ids=sources, dst_global_ids=query_ids,
                     dst_in_src=np.arange(len(query_ids)), edge_index=edge,
                     edge_type=np.full(len(query_ids), 4, np.int64),
                     edge_attr=np.ones((len(query_ids), 3), np.float32))], 0
    data._build_blocks = blocks
    return data


@unittest.skipUnless(HAS_TORCH, 'PyTorch is required for runtime data tests')
class RuntimeTests(unittest.TestCase):
    def test_role_local_dense_soft_targets_include_hard_negative_pairs(self):
        data = make_data()
        weak, hard, soft = data._target_arrays(np.arange(4))
        np.testing.assert_array_equal(weak, [True, False, True, False])
        np.testing.assert_array_equal(hard, [[1,0,0,0], [0,0,0,1], [0,0,0,0], [1,0,0,0]])
        np.testing.assert_allclose(soft[0], [.8,.03,0,.2])
        np.testing.assert_allclose(soft[2], [.05,.2,0,.99])
        self.assertTrue((soft[~weak] == 0).all())
        self.assertFalse(hard[2,3])
        self.assertGreater(float(soft[2,3]), .9)

    def test_training_does_not_gather_backbone_and_teacher_is_not_graph_evidence(self):
        data = make_data()
        first = data.batch(np.array([0,2]), 'cpu')
        self.assertNotIn('base_logits', first)
        self.assertNotIn('weak_soft_targets', first)
        self.assertIn('weak_modelout_prob', first)
        self.assertEqual(first['weak_modelout_prob'].shape, (2,4))
        self.assertFalse(first['positive_mask'][1].any())
        data._modelout_probability[:,[0,1,3]] = .123
        second = data.batch(np.array([0,2]), 'cpu')
        for key in ('targets','sampled_protein_x','sampled_candidate_go',
                    'sampled_candidate_attr','sampled_gold_edge','sampled_pseudo_edge'):
            torch.testing.assert_close(first[key], second[key])
        self.assertFalse(torch.equal(first['weak_modelout_prob'], second['weak_modelout_prob']))
        data.stores.episode_sampler.base_logit_store.gather_matrix.assert_not_called()

    def test_seen_modelout_values_are_validated_but_unseen_values_are_excluded(self):
        data = make_data()
        _, _, soft = data._target_arrays(np.array([0,2]))
        self.assertTrue(np.isfinite(soft).all())
        for value in (-.1, 1.1, np.nan):
            data._modelout_probability[0,0] = value
            with self.assertRaisesRegex(ValueError, 'Dense modelout teacher probabilities'):
                data._target_arrays(np.array([2]))

    def test_inference_uses_no_modelout_teacher_and_removes_backbone_logits(self):
        data = make_data()
        data._modelout_probability = None
        with tempfile.TemporaryDirectory() as name:
            inp = Path(name).resolve()
            go = np.tile(np.arange(4, dtype=np.int32), (2,1))
            attrs = np.ones((2,4,3), np.float32)
            attrs[...,2] = [1,.5,1/3,.25]
            data._inference = {str(inp): {'protein_x': np.ones((2,2), np.float32),
                'probability': np.full((2,4), .3, np.float32), 'candidate_go': go,
                'candidate_attr': attrs, 'neighbors': np.array([[0],[1]], np.int32),
                'neighbor_attrs': np.ones((2,1,3), np.float32),
                'v084_neighbors': np.array([[1],[3]], np.int32),
                'v084_attrs': np.ones((2,1,3), np.float32)}}
            result = data.inference_batch(inp, np.array([0,1]), 'cpu')
        self.assertNotIn('base_logits', result)
        self.assertNotIn('weak_modelout_prob', result)
        self.assertNotIn('targets', result)
        self.assertFalse((result['sampled_candidate_go'] == 2).any())
        self.assertEqual((data._sampling_step,data._sampling_rank,data._sampling_training),(5,0,True))

    def test_dense_teacher_hash_shape_and_identity_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path = root/'modelout.npy'
            np.save(path,np.full((2,4),.7,np.float16))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            stores = make_data().stores
            stores.registry = make_data().registry; stores.num_task_go = 4
            manifest = {'soft_teacher': {'role':'weak','path':path.name,'sha256':digest,
                'source':'modelout','row_order':'registry_role_local','scope':'all_seen_GO'},
                'sources': {'modelout_weak': {'path':path.name,'sha256':digest}}}
            teacher = runtime._load_dense_teacher(stores,manifest,root/'manifest.json')
            self.assertIsInstance(teacher,np.memmap)
            manifest['sources']['modelout_weak']['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError,'source identity'):
                runtime._load_dense_teacher(stores,manifest,root/'manifest.json')
            manifest['sources']['modelout_weak']['sha256'] = digest
            np.save(path,np.full((3,4),.7,np.float16))
            with self.assertRaisesRegex(ValueError,'hash differs'):
                runtime._load_dense_teacher(stores,manifest,root/'manifest.json')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest['soft_teacher']['sha256']=digest
            manifest['sources']['modelout_weak']['sha256']=digest
            with self.assertRaisesRegex(ValueError,'role-local weak rows'):
                runtime._load_dense_teacher(stores,manifest,root/'manifest.json')

    def test_constructor_keeps_hard_empty_weak_proteins_before_epoch_planning(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            arrays={'counts.npy':np.array([2,1,0,1000]),
                'seen.npy':np.array([True,True,False,True]),
                'mapping.npy':np.arange(4,dtype=np.int64),
                'candidates.npy':np.full((4,16),-1,np.int32),
                'attr.npy':np.zeros((4,16,3),np.float32),
                'core.npy':np.array([3,1]),'val.npy':np.array([],np.int64),
                'weak.npy':np.array([2,0])}
            for filename,value in arrays.items():np.save(root/filename,value)
            manifest={'schema':'nbs-weak-dataset-v094-r1','complete':True,
                'files':{'core_counts':'counts.npy','seen_task_mask':'seen.npy',
                    'task_to_ontology':'mapping.npy','training_core_ids':'core.npy',
                    'validation_ids':'val.npy','training_weak_ids':'weak.npy'},
                'candidates':{'go':'candidates.npy','attr':'attr.npy','topk':16,'mode':'global'},
                'hashes':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in arrays}}
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            registry=make_data().registry
            stores=SimpleNamespace(registry=registry,_unified_manifest_path=path,
                                   _v094_modelout_probability=np.full((2,4),.7,np.float32))
            def legacy_init(instance,config,stores=None):
                instance.registry=registry;instance.num_task_go=4
                instance.task_to_ontology_rows=np.arange(4)
                instance.candidate_topk=16;instance.core_ids=np.array([3,1])
                instance.validation_ids=np.array([],np.int64)
                instance.weak_ids=np.array([0])  # Legacy CSR nonempty subset.
            with mock.patch.object(runtime.FullTaskDataV090,'__init__',legacy_init):
                data=runtime.FullTaskDataV094({'data':{'unified_dataset_manifest':str(path)}},stores=stores)
            np.testing.assert_array_equal(data.weak_ids,[2,0])
            self.assertEqual(len(data.weak_ids),2)

    def test_canonical_holdout_must_not_split_aliases(self):
        data=make_data();data.task_to_ontology_rows=np.array([0,0,1,2])
        data._unified_seen=np.array([True,False,False,True])
        with self.assertRaisesRegex(ValueError,'canonical alias'):
            data._load_label_split('unified_dataset')


if __name__=='__main__':unittest.main()
