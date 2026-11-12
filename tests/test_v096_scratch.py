"""Explicit scratch profile, source-only configuration and frozen resume."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nbs import run_nbs_v096_scratch as scratch


def fixture(root):
    profile = scratch.load_profile(scratch.PROFILE)
    source = root / 'v095'; source.mkdir()
    dataset = root / 'unified_dataset_manifest.json'; dataset.write_text('{}')
    esm = root / 'esm_manifest.json'; esm.write_text(json.dumps({'complete': True, 'feature_dim': 1024}))
    ft = deepcopy(profile['full_task'])
    ft.update(seed=1095, learning_rate=.001, precision='fp32',
              initialization={'mode': 'v094_weights_only', 'checkpoint': '/nonexistent/weights.pt'})
    for key in ('esm_dropout', 'esm_gate_init'):
        ft['model'].pop(key)
    config = {'release_version': '0.9.5', 'model_version': '0.9.5', 'stage': {}, 'task': 'bp',
              'data': {'unified_dataset_manifest': str(dataset)}, 'full_task': ft,
              'runtime': {'old': 'source-run-only'}}
    source_config = source / 'resolved_config.json'; source_config.write_text(json.dumps(config))
    output = root / 'scratch'
    args = ['--source-work-dir', str(source), '--esm-manifest', str(esm), '--work-dir', str(output)]
    return profile, source, source_config, output, args


class ScratchProfileTests(unittest.TestCase):
    def test_stale_source_lr_and_weight_lineage_replaced_by_explicit_fresh_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            profile, source, source_config, output, args = fixture(Path(temp))
            before = source_config.read_bytes()
            scratch.main(['build', *args])
            path = output.with_name(output.name + '_config') / 'training_config.json'
            config = json.loads(path.read_text())
            self.assertEqual(config['full_task']['initialization'], {'mode': 'fresh'})
            self.assertEqual(config['full_task']['learning_rate'], .003)
            self.assertEqual(config['full_task']['precision'], 'bf16')
            self.assertEqual(config['full_task']['epochs'], 200)
            self.assertEqual(config['full_task']['seed'], 1096)
            self.assertEqual(config['full_task']['model']['esm_dim'], 1024)
            self.assertEqual(config['full_task']['model']['esm_gate_init'], 0.)
            self.assertEqual(config['full_task']['scheduler'], profile['full_task']['scheduler'])
            self.assertNotIn('runtime', config)
            self.assertEqual(source_config.read_bytes(), before)
            self.assertEqual(json.loads((path.parent / 'scratch_profile.json').read_text())['profile'], profile)
            self.assertFalse(output.exists())
            scratch.main(['build', *args])

    def test_rejects_weights_or_conflicting_lr_and_horizon_before_source_access(self):
        for extra, message in [(['--init-from', 'missing.pt'], 'forbids --init-from'),
                                (['--learning-rate', '.001'], 'learning_rate differs'),
                                (['--epochs', '20'], 'fixed scratch profile'),
                                (['--precision', 'fp32'], 'precision differs')]:
            with self.assertRaisesRegex(ValueError, message):
                scratch.main(['train', '--source-work-dir', '/missing/source', *extra])

    def test_protocol_mismatch_fails_before_writing_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, source_config, output, args = fixture(Path(temp))
            config = json.loads(source_config.read_text())
            config['full_task']['scheduler']['warmup_epochs'] = 1.
            source_config.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'source.full_task.scheduler.warmup_epochs'):
                scratch.main(['build', *args])
            self.assertFalse(output.with_name(output.name + '_config').exists())

    def test_training_and_smoke_are_delegated_with_fixed_profile_and_isolated_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, _, output, args = fixture(Path(temp))
            for action, expected_work in [('smoke', output.with_name(output.name + '_smoke')), ('train', output)]:
                with patch.object(scratch.base.subprocess, 'run') as run:
                    scratch.main([action, *args, '--gpu', '0'])
                self.assertEqual(run.call_count, 2)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index('--work-dir') + 1], str(expected_work))
                self.assertNotIn('--init-from', command)
                self.assertNotIn('--resume', command)
                self.assertEqual(command[-2:], ['--stop-step', '3'] if action == 'smoke' else ['--stop-epoch', '200'])
            output.mkdir()
            (output / 'latest.pt').write_bytes(b'previous checkpoint')
            with self.assertRaisesRegex(FileExistsError, 'empty work directory'):
                scratch.main(['train', *args])

    def test_resume_uses_saved_fresh_config_without_v095_access_or_lr_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            _, source, _, output, args = fixture(Path(temp))
            scratch.main(['build', *args])
            generated = json.loads((output.with_name(output.name + '_config') / 'training_config.json').read_text())
            output.mkdir()
            (output / 'resolved_config.json').write_text(json.dumps(generated))
            (output / 'latest.pt').write_bytes(b'checkpoint-fixture')
            for p in source.iterdir(): p.unlink()
            source.rmdir()
            with patch.object(scratch.base.subprocess, 'run') as run:
                scratch.main(['resume', '--work-dir', str(output), '--source-work-dir', str(source)])
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--resume') + 1], str(output / 'latest.pt'))
            self.assertNotIn('--learning-rate', command)
            self.assertEqual(command[-2:], ['--stop-epoch', '200'])
            generated['full_task']['initialization'] = {'mode': 'v095_weights_only'}
            (output / 'resolved_config.json').write_text(json.dumps(generated))
            with self.assertRaisesRegex(ValueError, 'only resumes/evaluates a scratch run'):
                scratch.main(['resume', '--work-dir', str(output)])

    def test_evaluation_keeps_fixed_config_and_profile_values_cannot_change(self):
        with tempfile.TemporaryDirectory() as temp:
            _, source, _, output, args = fixture(Path(temp))
            scratch.main(['build', *args])
            generated = json.loads((output.with_name(output.name + '_config') / 'training_config.json').read_text())
            output.mkdir(); saved = output / 'resolved_config.json'
            saved.write_text(json.dumps(generated))
            (output / 'nbs_epoch5.pt').write_bytes(b'checkpoint-fixture')
            with patch.object(scratch.base.subprocess, 'run') as run:
                scratch.main(['evaluate', '--work-dir', str(output), '--source-work-dir', '/missing', '--epoch', '5'])
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--stage') + 1], 'evaluate')
            with self.assertRaisesRegex(ValueError, 'omit new overrides'):
                scratch.main(['resume', '--work-dir', str(output), '--learning-rate', '.003'])
            generated['full_task']['learning_rate'] = .001
            saved.write_text(json.dumps(generated))
            with self.assertRaisesRegex(ValueError, 'full_task.learning_rate'):
                scratch.main(['resume', '--work-dir', str(output)])

    def test_help_requires_no_source_data_and_describes_only_scratch_defaults(self):
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            scratch.main(['--help', '--source-work-dir', '/missing/v095'])
        self.assertEqual(stopped.exception.code, 0)
        text = output.getvalue()
        self.assertIn('profile peak LR (0.003)', text)
        self.assertIn('profile precision (bf16)', text)
        self.assertNotIn('--init-from', text)
        self.assertNotIn('E125', text)
        self.assertNotIn('default .001', text)

    def test_default_workdir_is_scratch_and_original_configure_restored(self):
        original = scratch.base.configure
        with patch.object(scratch.base, 'main') as delegated:
            scratch.main(['check'])
        args = scratch.base.parser().parse_args(delegated.call_args.args[0])
        self.assertEqual(args.work_dir, ROOT / 'outputs/latence_nbs_experiments/v096/esm_penultimate_single12800_scratch')
        self.assertEqual(args.epochs, 200)
        self.assertIsNone(args.init_from)
        self.assertIs(scratch.base.configure, original)


if __name__ == '__main__':
    unittest.main()
