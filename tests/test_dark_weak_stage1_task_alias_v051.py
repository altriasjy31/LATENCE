"""Offline regression tests of Stage-1 full-/short-task resolution and row alignment.

Imports only selected functions from exporter via AST, avoiding a real checkpoint and
heavy Stage-1 model imports. No server files, training state, or GPU are touched.
"""
from __future__ import annotations

import ast
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
EXPORTER = PROJECT / 'scripts/dark_weak/export_stage1_predictions.py'


def _real_stage1_task_mapping():
    """Read actual LATENCE task names, independently of our exporter implementation."""
    source = (PROJECT / 'experiments/exp_train.py').read_text(encoding='utf-8')
    parsed = ast.parse(source)
    names = next(n for n in parsed.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'TASKS' for t in n.targets))
    return ast.literal_eval(names.value)


def _extract_methods():
    """Load executable real helper/export functions, without importing models."""
    code = ast.parse(EXPORTER.read_text(encoding='utf-8'))
    desired = {'read_ids', 'hash_file', 'resolve_task_aliases', 'export'}
    functions = [n for n in code.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name in desired]
    found = {n.name for n in functions}
    assert found == desired, (found, desired)
    return compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(EXPORTER), 'exec')


class TaskAliasContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = _real_stage1_task_mapping()
        cls.source = EXPORTER.read_text(encoding='utf-8')
        cls.functions = _extract_methods()

    def test_all_three_task_names_match_actual_exp_train(self):
        env = {'stage1': SimpleNamespace(normalize_task=lambda task: self.mapping[task])}
        exec(self.functions, env)
        for short, full in [('bp','biological_process'), ('mf','molecular_function'), ('cc','cellular_component')]:
            self.assertEqual(env['resolve_task_aliases'](short), (short, full))
            self.assertEqual(env['resolve_task_aliases'](full), (short, full))
        with self.assertRaises((ValueError, KeyError)):
            env['resolve_task_aliases']('unsupported_namespace')

    def test_export_always_uses_full_namespace_for_metadata_and_msa(self):
        parsed = ast.parse(self.source)
        fn = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'export')
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
        msa = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'build_msa_dataset']
        self.assertEqual(len(msa), 1)
        self.assertEqual(next(k.value.id for k in msa[0].keywords if k.arg == 'task'), 'task')
        self.assertIn("meta[args.mode][task]['proteins']", self.source)
        self.assertIn("'task':task_short", self.source)

    def test_export_bp_expert_modelout_scatter_order(self):
        # Exercise the actual export() body up to output hashing with a fake
        # Stage-1 model; no real GPU or checkpoint is necessary.
        try:
            import torch
            from torch.amp import autocast
            from tqdm import tqdm
            from scripts.dark_weak.stage1_alignment import ProteinRowPlacement
        except ImportError as e:
            self.skipTest(f'missing test-only runtime dependency: {e}')
        class FakeModel:
            def __init__(self, opt, args):
                self.classifier = SimpleNamespace(weight=torch.ones((21312,1)))
            def to(self, device): return self
            def eval(self): return self
            def query_decoder(self, **kwargs):
                # Test only interface wiring: the dummy decoder returns logits
                # mapping back to the supplied Expert probabilities.
                return {'logits': torch.logit(kwargs['external_prob'].clamp(1e-5,1-1e-5))}
        class FakeBase:
            proteins = ['A','B']
            def __len__(self):return 2
        class FakeWithProb(FakeBase):
            def __init__(self, **kwargs):
                self.proteins=kwargs['base_dataset'].proteins
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            metadata = root/'meta.pkl'
            with metadata.open('wb') as f:
                pickle.dump({'dark_weak_inference':{'biological_process':{
                    'proteins':['A','B'], 'annotations':[[],[]], 'prop_annotations':[[],[]]
                }}},f)
            ids=root/'ids.txt'; ids.write_text('A\nB\n',encoding='utf-8')
            go_ids=root/'go.txt'; go_ids.write_text(''.join(f'GO:{i:07d}\n' for i in range(21312)))
            ckpt=root/'checkpoint.pt'; ckpt.write_bytes(b'test-no-model')
            external=np.full((2,21312),.1,np.float32)
            external[0,0]=.8; external[1,0]=.6
            external_path=root/'expert.npy';np.save(external_path, external)
            opts=SimpleNamespace(
                task='bp',num_classes=21312, allow_eval_label_boost=False,
                expected_protein_ids=ids, file_address=metadata, mode='dark_weak_inference',
                device='cpu', eval_batch_size=2,dataloader_num_workers=0,pin_memory=False,
                prefetch_factor=2,persistent_workers=False,seed=1,no_amp=True,
                go_ids=go_ids,checkpoint=ckpt,train_args_json='auto',
                external_prob_path=external_path,external_protein_ids=ids,
                export_expert=True,export_modelout=True,
                modelout_contract='v088_weak_supervision',
                query_decoder_logit_base_mode='mix_expert_base_anchor',
                decoder_prob_source='expert', decoder_mix_alpha=.8,
                original_prob_path=None,original_protein_ids=None,
                output_dtype='float32',output_dir=root/'out',overwrite=False,
                permute_dims=(0,3,2,1),
            )
            def iterator(*args,**kwargs):
                # Batch order is deliberately B, A rather than metadata A, B.
                proteins=['B','A']
                x=torch.tensor([[2],[1]],dtype=torch.long)
                y=torch.zeros((2,21312),dtype=torch.float32)
                prob=torch.from_numpy(external[[1,0]])
                yield ((proteins,x),{'hard_y':y,'prob':prob})
            stage1=SimpleNamespace(
                normalize_task=lambda s:self.mapping[s],
                infer_num_classes_from_checkpoint=lambda p:21312,
                build_weak_opt_from_config=lambda args:SimpleNamespace(),
                WeakMSAGOWithDETRDecoder=FakeModel,
                load_detr_checkpoint=lambda *a,**k:{'checkpoint_type':'dummy'},
                build_msa_dataset=lambda opt,mode,task,need_proteins: (
                    FakeBase() if task=='biological_process' else (_ for _ in ()).throw(AssertionError('non-canonical task'))
                ),
                clean_optional_path=lambda x:str(x) if x else None,
                PseudoProbDataset=FakeWithProb,
                auto_train_args_path=lambda p:None,
                load_json_if_exists=lambda p:{},
                make_loader=iterator,
                unpack_pseudo_batch=lambda batch: (batch[0][0],batch[0][1],batch[1]['hard_y'],batch[1]['prob']),
                set_model_proteins=lambda model,proteins:None,
                model_forward_once=lambda model,x,permute_dims:(
                    x.float().repeat(1,21312) * torch.ones(1,21312),torch.ones((len(x),1))
                ),
            )
            import hashlib
            ns={'stage1':stage1,'Path':Path,'np':np,'torch':torch,'autocast':autocast,
                'tqdm':lambda xs,**kw: xs, 'json':json,'hashlib':hashlib,
                'ProteinRowPlacement':ProteinRowPlacement}
            exec(self.functions,ns)
            result=ns['export'](opts)
            self.assertEqual(result['task'],'bp')
            self.assertEqual(result['task_namespace'],'biological_process')
            prob=np.load(root/'out'/'expert.prob.f32.npy')
            self.assertEqual(prob.shape,(2,21312))
            np.testing.assert_allclose(prob[:,0],[.8,.6],atol=1e-7)
            modelout=np.load(root/'out'/'modelout.prob.f32.npy')
            np.testing.assert_allclose(modelout[:,0],[.8,.6],atol=1e-6)
            self.assertEqual((root/'out'/'protein_ids.txt').read_text(),'A\nB\n')


if __name__ == '__main__':
    unittest.main()
