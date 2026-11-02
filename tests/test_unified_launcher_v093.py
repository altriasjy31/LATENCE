from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('unified_launcher_test', ROOT/'scripts/nbs/run_nbs_v093.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def source_config():
    return {'task': 'bp', 'data': {'root': 'immutable_data'},
            'release_version': '0.9.0', 'model_version': '0.9.0',
            'stage': {'protein_go_candidate_topk': 512},
            'supervision_contract': {'weak_pseudo': {'source': 'modelout'}},
            'runtime': {'old_training': True},
            'full_task': {'model': {}, 'sampler': {}, 'v092_preset': 'geometry',
                          'scheduler': {'name': 'epoch_warmup_hold_cosine', 'warmup_epochs': 2.,
                                        'hold_until_epoch': 10., 'min_lr': .0001}}}


class LauncherTests(unittest.TestCase):
    def test_configuration_new_horizon_and_sources_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            old = root/'old'; old.mkdir()
            original = source_config()
            path = old/'resolved_config.json'; path.write_text(json.dumps(original))
            dataset = root/'dataset'; dataset.mkdir()
            manifest = dataset/'unified_dataset_manifest.json'; manifest.write_text('{}')
            args = launcher.parser().parse_args(['build', '--source-work-dir', str(old), '--dataset-dir', str(dataset)])
            config_path = launcher.configure(args, manifest)
            config = json.loads(config_path.read_text())
            self.assertEqual(json.loads(path.read_text()), original)
            self.assertEqual(config['full_task']['epochs'], 200)
            self.assertEqual(config['full_task']['label_split'], 'unified_dataset')
            self.assertEqual(config['full_task']['model']['output_strategy'], 'backbone_shared_residual')
            self.assertEqual(config['stage']['protein_go_candidate_topk'], 512)
            self.assertEqual(config['stage']['effective_candidate_topk'], 16)
            self.assertEqual(config['full_task']['sampler']['candidate_topk'], 16)
            self.assertNotIn('runtime', config)
            self.assertEqual(config['supervision_contract']['weak_pseudo']['source'], 'unified_frequency_routing')
            self.assertEqual(launcher.configure(args, manifest), config_path)
            args.epochs = 100
            with self.assertRaisesRegex(ValueError, 'different horizon'):
                launcher.configure(args, manifest)

    def test_clean_child_environment_selects_one_gpu(self):
        args = launcher.parser().parse_args(['train', '--gpu', '1', '--threads', '3'])
        with patch.dict(os.environ, {'CONFIG': 'stale', 'RESUME': 'stale', 'WORLD_SIZE': '4',
                                    'CUDA_VISIBLE_DEVICES': '0,1', 'UNRELATED': 'keep'}):
            child = launcher.environment(args)
        self.assertNotIn('CONFIG', child)
        self.assertNotIn('RESUME', child)
        self.assertNotIn('WORLD_SIZE', child)
        self.assertEqual(child['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(child['OMP_NUM_THREADS'], '3')
        self.assertEqual(child['UNRELATED'], 'keep')

    def test_build_passes_explicit_holdout_and_threshold_policy(self):
        args = launcher.parser().parse_args(['build', '--label-holdout-fraction', '.1',
                                            '--thresholds-json', '/tmp/policy.json', '--seed', '99'])
        cmd = launcher.build_command(args)
        self.assertEqual(cmd[cmd.index('--label-holdout-fraction') + 1], '0.1')
        self.assertEqual(cmd[cmd.index('--thresholds-json') + 1], '/tmp/policy.json')
        self.assertEqual(cmd[cmd.index('--seed') + 1], '99')

    def test_evaluation_shares_v092_cache_lock_but_training_is_independent(self):
        evaluate = launcher.parser().parse_args(['evaluate', '--epoch', '5'])
        predict = launcher.parser().parse_args(['predict-go', '--epoch', '5'])
        train = launcher.parser().parse_args(['train'])
        work = Path('/tmp/one_run')
        self.assertEqual(launcher.operation_lock(evaluate, work), launcher.operation_lock(predict, work))
        self.assertEqual(launcher.operation_lock(evaluate, work).name, 'v092_evaluate.lock')
        self.assertNotEqual(launcher.operation_lock(evaluate, work), launcher.operation_lock(train, work))


if __name__ == '__main__':
    unittest.main()
