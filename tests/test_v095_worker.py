"""Run the installed v095 worker loop against a tiny, real sampled graph.

The fixture supplies tensors in memory, so this checks worker/checkpoint/resume
wiring, not production cache loading, CUDA throughput, or model quality.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'nbs_models/nbs_protein_go'), str(ROOT/'tests')]


def exercise():
    import numpy as np
    import torch
    from test_v095_integration import batch, ontology
    from scripts.nbs.train_nbs_full_task_v095 import install
    from nbs_pg.full_task_loss_v095 import FullTaskLossConfigV095
    from nbs_pg.full_task_model_v095 import FullTaskModelConfigV095
    from nbs_pg.optimizer_v095 import configure_training_runtime
    torch.set_num_threads(1)
    base = install()

    class TinyData:
        feature_dim, num_task_go = 6, 8
        weak_ids, core_ids = np.array([1]), np.array([0])
        seen_task_mask = np.ones(8, dtype=bool)
        stores = SimpleNamespace(episode_sampler=SimpleNamespace(base_logit_store=
            SimpleNamespace(arrays={}, slices=[], probability_clip=.0001)))

        def ensure_prepared(self): pass
        def set_sampling_context(self, **kwargs): pass
        def ontology(self, device): return ontology()
        def data_contract(self): return {'synthetic_fixture': 'v095-worker-integration'}
        def batch(self, ids, device, **kwargs):
            assert len(ids) == 2
            x = batch()
            x['is_weak'] = torch.tensor(ids == 1)
            x['weak_modelout_prob'] = torch.full((2, 8), .2)
            x['weak_modelout_prob'][x['positive_mask']] = .8
            return x

    options = FullTaskModelConfigV095(hidden_dim=16, ontology_layers=1,
        input_dropout=.1, dropout=.1, raw_geometry_width=12, random_feature_dim=8)
    config = dict(task='bp', release_version='0.9.5', model_version='0.9.5', full_task=dict(
        seed=95, epochs=2, expected_world_size=1, weak_batch=1, core_batch=1,
        accumulation_steps=1, learning_rate=.003, weight_decay=.0001, grad_clip=5.,
        epoch_unit='weak', variant='dynamic', sampler={'mode':'dynamic'},
        monitor_core=False, checkpoint_every_epochs=1, log_every=1,
        precision='fp32', initialization={'mode':'fresh'},
        model=asdict(options), loss=asdict(FullTaskLossConfigV095()),
        optimizer=dict(name='adamw',eps=1e-8,betas=[.9,.999],decay_policy='matrices_only'),
        scheduler=dict(name='epoch_warmup_hold_cosine',warmup_epochs=.1,hold_until_epoch=1.,min_lr=.00003)))
    configure_training_runtime(base, config)
    base._v095_precision = 'fp32'
    data = TinyData()

    def train(directory, stop=None, resume=None):
        args = SimpleNamespace(work_dir=directory,stop_step=stop,stop_epoch=None,
                               epochs=None,resume=resume,encoder_from=None)
        torch.manual_seed(95)
        base.train(args, deepcopy(config), data, torch.device('cpu'), 0, 1)
        return torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)

    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        complete=train(root/'complete')
        first=train(root/'resume',stop=1)
        resumed=train(root/'resume',resume=root/'resume/latest.pt')
        assert first['step']==1 and resumed['step']==2
        assert resumed['runner_version']==resumed['model_architecture_version']=='0.9.5'
        assert resumed['data_architecture_version']=='0.9.4'
        for name,value in complete['model'].items():
            torch.testing.assert_close(value,resumed['model'][name],rtol=0,atol=0)
        for old,new in zip(complete['history'],resumed['history']):
            for key in ('loss','grad_norm','hard_ASL','modelout_soft_KL','hierarchy','contrib_hierarchy'):
                assert old[key]==new[key], key
        for state,new_state in zip(complete['optimizer']['state'].values(),resumed['optimizer']['state'].values()):
            for key,value in state.items():
                torch.testing.assert_close(value,new_state[key],rtol=0,atol=0)
        assert complete['scheduler']==resumed['scheduler']
        assert (root/'resume/nbs_epoch2.pt').is_file()


class V095WorkerTests(unittest.TestCase):
    def test_installed_training_worker_checkpoint_and_resume(self):
        result=subprocess.run([sys.executable,__file__,'--exercise'],cwd=ROOT,
                              capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)


if __name__=='__main__':
    if '--exercise' in sys.argv:
        exercise()
    else:
        unittest.main()
