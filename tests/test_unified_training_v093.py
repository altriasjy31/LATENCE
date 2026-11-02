"""Real CPU phase-engine optimizer/resume/export probe, isolated in a child."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HAS_RUNTIME = importlib.util.find_spec('torch') is not None


def run_fixture():
    sys.path[:0] = [str(ROOT), str(ROOT/'nbs_models/nbs_protein_go')]
    import torch
    from scripts.nbs.train_nbs_full_task_v093 import install, REVISION
    from nbs_pg.full_task_model_v093 import FullTaskModelConfigV093
    base = install()
    torch.set_num_threads(1)
    torch.manual_seed(66)
    device = torch.device('cpu')
    ontology = {'center': torch.randn(8, 3), 'offset': torch.ones(8, 3)*.2,
                'task_to_ontology': torch.arange(8), 'edges': {}}
    options = vars(FullTaskModelConfigV093(hidden_dim=8, ontology_layers=0,
        raw_geometry_width=12, random_feature_dim=8, input_dropout=.1, dropout=.25))
    config = {'task': 'bp', 'release_version': '0.9.3', 'model_version': '0.9.3',
        'full_task': {'seed': 66, 'weak_batch': 2, 'core_batch': 2, 'accumulation_steps': 2,
            'expected_world_size': 1, 'epochs': 200, 'epoch_unit': 'weak', 'variant': 'dynamic',
            'sampler': {'mode': 'dynamic'}, 'learning_rate': .001, 'weight_decay': .0001,
            'grad_clip': 5., 'log_every': 1, 'eval_batch': 2, 'checkpoint_every_epochs': 5,
            'monitor_core': False, 'experiment_revision': REVISION, 'model': options,
            'loss': {'core_weight': 1., 'weak_weight': .25, 'shared_weight': .25,
                     'rare_distill_weight': .1, 'weak_soft_weight': .1},
            'scheduler': {'name': 'epoch_warmup_hold_cosine', 'warmup_epochs': 2.,
                          'hold_until_epoch': 10., 'min_lr': .0001}}}

    class Data:
        feature_dim, num_task_go = 6, 8
        weak_ids, core_ids = np.arange(6, dtype=np.int64), np.arange(6, 10, dtype=np.int64)
        validation_ids = np.array([], np.int64)
        seen_task_mask = np.array([True]*7 + [False])
        def ensure_prepared(self): pass
        def ontology(self, device): return ontology
        def data_contract(self): return {'fixture': 'v093-unified', 'seen': self.seen_task_mask.tolist()}
        def set_sampling_context(self, step, rank, training): pass
        def batch(self, ids, device='cpu', supervision_seed_ids=None):
            ids = np.asarray(ids, np.int64)
            n = len(ids)
            source = torch.tensor(list(-ids-1) + [100, 101])
            blocks = []
            for layer in range(3):
                dest = source if layer < 2 else source[:n]
                edge = torch.tensor([[n+i%2 for i in range(len(dest))], list(range(len(dest)))])
                blocks.append(dict(src_global_ids=source.clone(), dst_global_ids=dest.clone(),
                    dst_in_src=torch.arange(len(dest)), edge_index=edge,
                    edge_type=torch.zeros(len(dest), dtype=torch.long), edge_attr=torch.ones(len(dest), 3)))
            features = torch.sin(torch.tensor(list(ids)+[100, 101]).float()[:, None]*torch.arange(1, 7).float()[None])
            candidate = torch.tensor([[int(i)%7, (int(i)+2)%7] for i in list(ids)+[100, 101]])
            target = torch.zeros(n, 8)
            for i, protein in enumerate(ids): target[i, [protein%7, (protein+1)%7]] = 1
            return dict(sampled_protein_x=features, sampled_node_type=torch.tensor([2]*n+[0, 1]),
                sampled_candidate_go=candidate, sampled_candidate_attr=torch.ones(n+2, 2, 3)*.8,
                sampled_gold_edge=torch.tensor([[n], [0]]), sampled_pseudo_edge=torch.tensor([[n+1], [2]]),
                sampled_seed_index=torch.arange(n), blocks=blocks, targets=target, positive_mask=target.bool(),
                weak_soft_targets=target*.8, is_weak=torch.tensor(ids < 6), protein_x=features[:n],
                candidate_go=candidate[:n], candidate_attr=torch.ones(n, 2, 3)*.8,
                base_logits=torch.cos(torch.tensor(ids).float()[:, None]+torch.arange(8).float()[None]),
                seen_task_mask=torch.tensor(self.seen_task_mask), core_counts=torch.tensor([3]*7+[0]))
        def inference_batch(self, folder, ids, device='cpu'):
            batch = self.batch(ids, device)
            for key in ('targets', 'positive_mask', 'weak_soft_targets', 'is_weak'): batch.pop(key)
            return batch

    data = Data()
    def train(folder, stop, resume=None):
        args = SimpleNamespace(work_dir=folder, encoder_from=None, resume=resume,
                               epochs=None, stop_epoch=None, stop_step=stop)
        if resume is None: torch.manual_seed(66)
        base.train(args, deepcopy(config), data, device, 0, 1)

    with tempfile.TemporaryDirectory() as name, patch.object(base, 'training_baseline_identity', return_value={'fixture': True}):
        root = Path(name)
        continuous, resumed = root/'continuous', root/'resumed'
        train(continuous, 3)
        train(resumed, 1)
        train(resumed, 3, resumed/'latest.pt')
        first = torch.load(continuous/'latest.pt', weights_only=False)
        second = torch.load(resumed/'latest.pt', weights_only=False)
        for key, value in first['model'].items():
            torch.testing.assert_close(value, second['model'][key], rtol=0, atol=0)
        assert first['scheduler'] == second['scheduler']
        assert first['scheduler']['contract']['epochs'] == 200
        assert 'full_task_model_v093.py' in first['training_implementation']
        for key in ('shared_binary', 'rare_backbone_KL', 'weak_selected_soft', 'contrib_shared'):
            assert key in first['history'][-1]
        inp = root/'inputs'; inp.mkdir()
        np.save(inp/'ind_test_repr.f16.npy', np.zeros((3, 6), np.float16))
        (inp/'ind_test_input_manifest.json').write_text('{}')
        args = SimpleNamespace(checkpoint=continuous/'nbs_epoch1.pt', input_dir=inp,
                               metadata_file=root/'metadata.pkl', work_dir=continuous, stage='evaluate')
        from nbs_pg import evaluation_report_v093 as report
        with patch.object(report, 'write_extended_report', return_value=root/'fixture_summary.json'):
            base.evaluate(args, deepcopy(config), data, device)
            path = continuous/'eval_step3/full/nbs_ind_test_prob.f32.npy'
            before = base.sha256(path)
            assert np.load(path).shape == (3, 8)
            base.evaluate(args, deepcopy(config), data, device)
            assert base.sha256(path) == before
        manifest = json.loads((path.parent/'nbs_full_task_prediction_manifest.json').read_text())
        assert manifest['uses_dense_backbone_logits_in_forward'] is True
        assert manifest['experiment_revision'] == REVISION
        changed = deepcopy(config); changed['full_task']['model']['dropout'] = .5
        try: base.evaluate(args, changed, data, device)
        except ValueError: pass
        else: raise AssertionError('changed architecture accepted')
    print('V093 actual model/objective/optimizer/resume/export fixture passed')


@unittest.skipUnless(HAS_RUNTIME, 'Real PyTorch is required for the integrated phase-engine probe')
class UnifiedTrainingTests(unittest.TestCase):
    def test_actual_optimizer_resume_and_export(self):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--run-fixture'],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('V093 actual model/objective/optimizer/resume/export', result.stdout)


if __name__ == '__main__':
    if '--run-fixture' in sys.argv: run_fixture()
    else: unittest.main()
