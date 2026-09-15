#!/usr/bin/env python3
"""Ordered v0.7.1 experiments; each invocation runs one explicitly named stage."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('correctness_control', 'sampling_loss', 'inductive_aligned')
EVAL_ENV = {'NBS_EVAL_DEVICE', 'NBS_EVAL_MIN_FREE_GPU_GB', 'NBS_EVAL_PROTEIN_BATCH',
            'NBS_METRIC_BACKEND', 'NBS_NUM_GPUS'}


def clean_environment(source):
    """Prevent old smoke/epoch/Q overrides from silently changing this experiment."""
    return {k: v for k, v in source.items() if not k.startswith('NBS_') or k in EVAL_ENV}


def make_training_config(root: Path, variant: str, phase: str, output: Path, epochs: int):
    path = root / f'nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.7.1_{variant}_probe.json'
    config = json.loads(path.read_text())
    training = config['training']
    training.update(output_dir=str(output), epochs=1 if phase == 'smoke' else epochs,
                    save_epochs=[1] if phase == 'smoke' else list(range(1, epochs + 1)),
                    max_steps_per_epoch=20 if phase == 'smoke' else None,
                    gradient_probe_steps=20)
    if phase == 'smoke':
        # A 20-step OneCycle with pct_start=.05 has a zero-length first phase.
        # This is a semantic/gradient smoke, not an optimization comparison.
        config['scheduler'] = {'name': 'none'}
        training['scheduler_step'] = 'none'
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('baseline-ablation', 'audit', 'smoke', 'probe', 'evaluate', 'analyze'))
    parser.add_argument('--variant', choices=VARIANTS, default='sampling_loss')
    parser.add_argument('--work-root', type=Path, default=Path('outputs/latence_nbs_experiments/v071'))
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('outputs/latence_nbs_train/bp_nbs_v060_perfopt_tailfix_fullrestore_formal'))
    parser.add_argument('--baseline-config', type=Path)
    parser.add_argument('--input-cache', type=Path, default=None)
    parser.add_argument('--num-gpus', type=int, default=2)
    parser.add_argument('--probe-epochs', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--audit-episodes', type=int, default=32)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.num_gpus <= 0 or args.audit_episodes <= 0:
        parser.error('num-gpus and audit-episodes must be positive')
    root = Path(os.environ.get('LATENCE_PROJECT_ROOT', ROOT)).resolve()
    def resolve(path):
        return (path if path.is_absolute() else root / path).resolve()
    work = resolve(args.work_root)
    checkpoint_dir = resolve(args.checkpoint_dir)
    baseline_config = resolve(args.baseline_config) if args.baseline_config else checkpoint_dir / 'resolved_config.json'
    input_cache = resolve(args.input_cache) if args.input_cache else resolve(Path(os.environ.get('NBS_IND_TEST_WORK_DIR', work / 'shared_inductive_inputs')))
    env = clean_environment(os.environ)
    env.update(LATENCE_PROJECT_ROOT=str(root), TASK='bp', NBS_NUM_GPUS=str(args.num_gpus))

    def run(script, arguments=(), updates=None):
        child = {**env, **(updates or {})}
        command = [sys.executable, str(root / 'scripts/nbs' / script), *map(str, arguments)]
        print(shlex.join(command), flush=True)
        if updates:
            print(json.dumps({'experiment_environment': updates}, indent=2), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, env=child, check=True)

    def analyze():
        run('analyze_nbs_v071_experiments.py', ['--work-root', work])

    def evaluation(checkpoints, config, output, epochs, *, evidence=True, candidate_only=False):
        if not args.dry_run:
            if not config.is_file():
                raise FileNotFoundError(f'{config}; specify the actual saved config with --baseline-config')
            if not env.get('STAGE1_CHECKPOINT'):
                raise ValueError('STAGE1_CHECKPOINT must identify the same frozen Stage-1 artifact as training')
            manifest = input_cache / 'ind_test_input_manifest.json'
            if manifest.is_file():
                metadata = json.loads(manifest.read_text())
                candidate = metadata.get('candidate_evidence', metadata.get('artifacts', {}).get('candidate_evidence', {}))
                if candidate.get('selector_scope') != 'full_task' or int(candidate.get('fixed_k', 0)) != 512:
                    raise ValueError('The selected cache is not documented as full_task top-512. Use the corrected epoch16-20 cache or rebuild a separate cache.')
        run('run_eval_nbs_checkpoint_series.py', updates={
            'NBS_CHECKPOINT_DIR': str(checkpoints), 'NBS_TRAIN_CONFIG': str(config),
            'NBS_EVAL_CANDIDATE_SELECTOR_SCOPE': 'full_task',
            'NBS_EVAL_PRECISION_K': '10,50,100',
            'NBS_EVAL_EPOCHS': epochs, 'NBS_EVAL_SERIES_ROOT': str(output),
            'NBS_IND_TEST_WORK_DIR': str(input_cache), 'NBS_EVAL_USE_EXTERNAL_PP': '1',
            'NBS_EVAL_USE_CANDIDATE_EVIDENCE': '1' if evidence else '0',
            'NBS_EVAL_PRESERVE_BASE_OUTSIDE_CANDIDATES': '1' if candidate_only else '0',
            'NBS_EVAL_SUPPORT_PER_QUERY': '2', 'NBS_EVAL_GO_CHUNK': '256',
            'NBS_EVAL_SKIP_COMPLETED': '1', 'NBS_INPUT_CACHE_POLICY': 'reuse',
            'NBS_SAVE_INFERENCE_DIAGNOSTICS': '1', 'NBS_USE_TMP_WORKSPACE': '0',
        })

    if args.stage == 'baseline-ablation':
        for name, evidence, candidate_only in [('full', True, False), ('evidence_off', False, False), ('candidate_only', True, True)]:
            evaluation(checkpoint_dir, baseline_config, work / 'baseline' / name, '20', evidence=evidence, candidate_only=candidate_only)
        analyze()
    elif args.stage == 'audit':
        if not args.dry_run and not baseline_config.is_file():
            raise FileNotFoundError(f'{baseline_config}; audit the actual legacy configuration')
        configs = {'legacy': baseline_config, **{v: root / f'nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.7.1_{v}_probe.json' for v in ('correctness_control', 'sampling_loss')}}
        for name, config in configs.items():
            for rank in range(args.num_gpus):
                run('audit_nbs_protein_go_batch_alignment.py', ['--config', config, '--output-dir', work / 'audit' / f'{name}_rank{rank}', '--epochs', '1', '--episodes-per-epoch', args.audit_episodes, '--world-size', args.num_gpus, '--rank', rank])
            run('audit_nbs_protein_go_batch_alignment.py', ['--config', config, '--output-dir', work / 'audit' / f'{name}_fixed_cohort', '--epochs', '3', '--episodes-per-epoch', min(8, args.audit_episodes), '--world-size', args.num_gpus, '--rank', '0', '--fixed-cohort'])
        analyze()
    elif args.stage in ('smoke', 'probe'):
        phase = args.stage
        name = f'{args.variant}_' + ('smoke20' if phase == 'smoke' else f'probe{args.probe_epochs}')
        output = work / 'train' / name
        config_path = work / 'configs' / f'{name}.json'
        config = make_training_config(root, args.variant, phase, output, args.probe_epochs)
        if not args.dry_run:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f'{output} is not empty; use a new --work-root for a fresh experiment')
            if phase == 'probe':
                from analyze_nbs_v071_experiments import analyze_smoke
                status = analyze_smoke(work / 'train' / f'{args.variant}_smoke20')
                if status['status'] != 'passed':
                    raise RuntimeError(f'Run and resolve the {args.variant} 20-step smoke first: {status}')
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2) + '\n')
        else:
            print(json.dumps({'generated_config': str(config_path), 'training': config['training'], 'scheduler': config['scheduler']}, indent=2))
        updates = {'NBS_TRAIN_CONFIG': str(config_path), 'NBS_FRESH_START': '1'}
        run('run_train_nbs_fixed_epochs.py', updates={**updates, 'NBS_NUM_GPUS': '1', 'NBS_VALIDATE_CONFIG_ONLY': '1'})
        run('run_train_nbs_fixed_epochs.py', updates=updates)
        analyze()
        if phase == 'smoke' and not args.dry_run:
            from analyze_nbs_v071_experiments import analyze_smoke
            result = analyze_smoke(output)
            if result['status'] != 'passed':
                raise RuntimeError(f"Smoke requires attention; inspect {work / 'analysis/experiment_analysis.json'}")
    elif args.stage == 'evaluate':
        train_dir = work / 'train' / f'{args.variant}_probe{args.probe_epochs}'
        evaluation(train_dir, train_dir / 'resolved_config.json', work / 'eval' / args.variant,
                   ','.join(str(e) for e in range(1, args.probe_epochs + 1)))
        analyze()
    else:
        analyze()


if __name__ == '__main__':
    main()
