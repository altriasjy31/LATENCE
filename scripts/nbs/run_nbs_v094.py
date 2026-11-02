#!/usr/bin/env python3
"""One entry point for Backbone hard labels and dense modelout soft supervision."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REVISION = '0.9.4-weak-teacher-r1'
OLD_ENV = ('CONFIG', 'WORK_DIR', 'WORK_ROOT', 'RESUME', 'INIT_FROM', 'ENCODER_FROM',
           'STEPS', 'STOP_STEP', 'STOP_EPOCH', 'CHECKPOINT', 'HEAD_CHECKPOINT', 'HYBRID_INIT',
           'JOINT_CHECKPOINT', 'JOINT_WORK_DIR', 'PRESET', 'NUM_GPUS', 'DEVICE', 'INPUT_DIR',
           'METADATA_FILE', 'DEV_MANIFEST', 'ABLATION', 'METRIC_BACKEND', 'AUPRC_MODE', 'SCRIPTS',
           'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK',
           'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'TORCHELASTIC_RUN_ID',
           'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS', 'TORCHELASTIC_ERROR_FILE',
           'TORCHELASTIC_USE_AGENT_STORE')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check', 'build', 'prepare', 'smoke', 'train', 'resume', 'evaluate', 'predict-go'))
    p.add_argument('--gpu', default='0')
    p.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--source-work-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v092/geometry_single12800_scratch')
    p.add_argument('--dataset-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v094/backbone_hard_modelout_soft')
    p.add_argument('--work-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v094/weak_teacher_single12800')
    p.add_argument('--epochs', type=int, default=200, help='Fixed fresh-run horizon; same value is required for resume')
    endpoints = p.add_mutually_exclusive_group()
    endpoints.add_argument('--stop-epoch', type=int)
    endpoints.add_argument('--stop-step', type=int)
    p.add_argument('--epoch', type=int, help='Completed fixed epoch for evaluation/prediction')
    p.add_argument('--input-dir', type=Path, default=ROOT/'outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2')
    p.add_argument('--metadata-file', type=Path, default=Path('/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl'))
    p.add_argument('--go-ids-file', type=Path)
    p.add_argument('--ontology-file', type=Path)
    p.add_argument('--output-dir', type=Path, help='Optional custom GO prediction output')
    p.add_argument('--thresholds-json', type=Path)
    p.add_argument('--calibrate-backbone', action='store_true', help='Accepted explicitly; build calibrates automatically unless --thresholds-json is supplied')
    p.add_argument('--backbone-prob', type=Path)
    p.add_argument('--modelout-prob', type=Path)
    p.add_argument('--core-weight', type=float, default=.25, help='Build only: hard BCE core role weight')
    p.add_argument('--weak-weight', type=float, default=1., help='Build only: hard BCE weak role weight')
    p.add_argument('--modelout-soft-weight', type=float, default=1., help='Build only: independent weak-mean dense modelout KL weight')
    p.add_argument('--label-holdout-fraction', type=float, default=0.)
    p.add_argument('--seed', type=int, default=1337, help='Dataset canonical-GO split seed')
    return p


def environment(args):
    env = {k: v for k, v in os.environ.items() if k not in OLD_ENV}
    env['CUDA_VISIBLE_DEVICES'] = args.gpu if args.device == 'cuda:0' else ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[name] = str(args.threads)
    return env


def configure(args, dataset_manifest):
    """Freeze a separate new config; never write into the old v092 run."""
    source_path = args.source_work_dir.resolve()/'resolved_config.json'
    config = deepcopy(json.loads(source_path.read_text()))
    config.pop('runtime', None)
    config['data']['unified_dataset_manifest'] = str(dataset_manifest.resolve())
    config['release_version'], config['model_version'] = '0.9.4', '0.9.4'
    config['stage'].update(name='nbs_v094_weak_teacher', freeze_first_stage_backbone=True,
                           use_expert_probability_in_forward=False,
                           protein_go_candidate_topk=int(config['stage'].get('protein_go_candidate_topk', 512)),
                           effective_candidate_topk=16)
    config['supervision_contract']['weak_pseudo'] = dict(
        role='weak', dataset_mode='exp_train', source='backbone_membership_modelout_dense_soft',
        prediction_key='v094_all_frequency_backbone_hard_plus_modelout_soft',
        threshold_policy='manifest_group_thresholds', use_probability_as_soft_target='independent_dense_modelout_KL_all_seen_GO',
        negative_policy='binary_membership_full_columns_except_unseen_and_alias_exclusions',
        forbid_exp_train_prop_annotations=True)
    ft = config['full_task']
    ft.update(seed=1094, epochs=args.epochs, expected_world_size=1,
              experiment_revision=REVISION, experiment_version='0.9.4',
              experiment_preset='weak_teacher_shared_direct', label_split='unified_dataset',
              weak_target_mode='backbone_binary_plus_modelout_dense_soft',
              monitor_core=False, checkpoint_every_epochs=5,
              weak_batch=10240, core_batch=2560, accumulation_steps=2,
              learning_rate=.001, epoch_unit='weak', variant='dynamic')
    ft.pop('v092_preset', None)
    ft['architecture_extension'] = 'v094_single_shared_head_raw_geometry_dense_teacher'
    ft['sampler'].update(mode='dynamic', candidate_topk=16)
    # The immutable native graph pools can be shared: v093 does not alter them.
    # New candidate evidence is read directly from its own dataset cache.
    ft['model'].update(sage_layers=3, hidden_dim=320, prediction_mode='shared_bilinear',
                        output_strategy='shared_direct', go_input_mode='ontology_raw',
                        protein_random_features='none')
    ft['loss'] = dict(core_weight=args.core_weight, weak_weight=args.weak_weight,
                      modelout_soft_weight=args.modelout_soft_weight)
    config.pop('v092_experiment', None)
    config.pop('v093_experiment', None)
    config['v094_experiment'] = dict(
        revision=REVISION, initialization='fresh_single_shared_scorer_without_backbone_logit_skip',
        source_config_sha256=digest(source_path),
        unified_dataset_manifest_sha256=digest(dataset_manifest),
        strict_pipeline_zero_shot=False)
    path = dataset_manifest.parent/'training_config.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Existing training config has a different horizon/policy; use a new dataset directory')
    else:
        atomic_json(path, config)
    return path


def build_command(args):
    command = [args.python, str(ROOT/'scripts/nbs/build_nbs_dataset_v094.py'),
               '--source-work-dir', str(args.source_work_dir.resolve()),
               '--output-dir', str(args.dataset_dir.resolve()),
               '--label-holdout-fraction', str(args.label_holdout_fraction), '--seed', str(args.seed)]
    for name in ('thresholds_json', 'backbone_prob', 'modelout_prob'):
        if getattr(args, name) is not None:
            command += ['--' + name.replace('_', '-'), str(getattr(args, name).resolve())]
    return command


def operation_lock(args, work):
    if args.action in ('evaluate', 'predict-go'):
        # Match the installed parallel-evaluation fix for v092, because both
        # versions may build the same inherited inference cache.
        return ROOT/'outputs/latence_nbs_experiments/v092/v092_evaluate.lock'
    return work.with_name(work.name + '.v094.lock')


def main(argv=None):
    args = parser().parse_args(argv)
    supplied = list(sys.argv[1:] if argv is None else argv)
    loss_flags = ('--core-weight', '--weak-weight', '--modelout-soft-weight')
    if args.action != 'build' and any(token.split('=', 1)[0] in loss_flags for token in supplied):
        raise ValueError('Loss weights are frozen by build; use a separate dataset/run for a changed objective')
    os.chdir(ROOT)
    import math
    if any(not math.isfinite(v) or v < 0 for v in (args.core_weight,args.weak_weight,args.modelout_soft_weight)) or args.core_weight+args.weak_weight <= 0:
        raise ValueError('Loss weights must be finite/nonnegative and core+weak positive')
    if args.threads < 1 or args.epochs < 11:
        raise ValueError('threads must be positive and epochs >=11')
    if not re.fullmatch(r'[^,\s]+', args.gpu) or args.gpu == '-1':
        raise ValueError('--gpu must select one physical GPU or UUID')
    if args.stop_epoch is not None and not 1 <= args.stop_epoch <= args.epochs:
        raise ValueError('stop-epoch must lie within the fixed horizon')
    if args.stop_step is not None and args.stop_step < 1:
        raise ValueError('stop-step must be positive')
    inference = args.action in ('evaluate', 'predict-go')
    if inference:
        if args.epoch is None or not 1 <= args.epoch <= args.epochs:
            raise ValueError('Evaluation/prediction requires a completed --epoch')
        if args.stop_epoch is not None or args.stop_step is not None:
            raise ValueError('Evaluation/prediction has no stop endpoint')
    elif args.epoch is not None:
        raise ValueError('--epoch is only for evaluate/predict-go')
    env = environment(args)
    if args.action == 'check':
        subprocess.run([args.python, '-c', 'import torch; print("PyTorch", torch.__version__)'],
                       cwd=ROOT, env=env, check=True)
        for pattern in ('test_v094*.py',):
            subprocess.run([args.python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v'],
                           cwd=ROOT, env=env, check=True)
        return
    dataset_manifest = args.dataset_dir.resolve()/'unified_dataset_manifest.json'
    if args.action == 'build':
        if args.stop_epoch is not None or args.stop_step is not None:
            raise ValueError('Build has no training endpoint')
        if args.calibrate_backbone and args.thresholds_json is not None:
            raise ValueError('Choose automatic calibration or --thresholds-json')
        if args.thresholds_json is None:
            args.thresholds_json = args.dataset_dir.resolve().with_name(args.dataset_dir.name + '_backbone_thresholds.json')
            calibration = [args.python, str(ROOT/'scripts/nbs/calibrate_nbs_backbone_v094.py'),
                           '--source-work-dir', str(args.source_work_dir.resolve()),
                           '--output', str(args.thresholds_json),
                           '--label-holdout-fraction', str(args.label_holdout_fraction), '--seed', str(args.seed)]
            subprocess.run(calibration, cwd=ROOT, env=env, check=True)
        subprocess.run(build_command(args), cwd=ROOT, env=env, check=True)
        config_path = configure(args, dataset_manifest)
        summary = json.loads(dataset_manifest.read_text())
        print('[weak coverage] bin: old covered -> new covered; old positives -> new positives', flush=True)
        for group in summary['frequency_strata']:
            print(f"  {group['frequency_bin']}: {group['old_weak_covered_GO_columns']} -> "
                  f"{group['weak_covered_GO_columns']} / {group['GO_columns']} GO columns; "
                  f"{group['old_weak_positive_memberships']} -> {group['weak_positive_memberships']} memberships", flush=True)
        print('[teacher disagreement] ' + json.dumps(summary['teacher_disagreement'], ensure_ascii=False), flush=True)
        print(f'[v094 ready] dataset={dataset_manifest}\nconfig={config_path}', flush=True)
        return
    work = args.work_dir.resolve()
    if inference or args.action == 'resume':
        config_path = work/'resolved_config.json'
    else:
        if not dataset_manifest.is_file():
            raise FileNotFoundError('Build the new dataset first: python scripts/nbs/run_nbs_v094.py build')
        config_path = args.dataset_dir.resolve()/'training_config.json'
    if not config_path.is_file():
        raise FileNotFoundError('Run configuration is missing: ' + str(config_path))
    config = json.loads(config_path.read_text())
    if config['full_task']['epochs'] != args.epochs:
        raise ValueError('Requested horizon differs from the frozen training config')
    if args.action == 'smoke':
        work = work.with_name(work.name + '_smoke')
    worker = ROOT/'scripts/nbs/train_nbs_full_task_v094.py'
    common = [args.python, str(worker), '--config', str(config_path), '--work-dir', str(work), '--device', args.device]
    if args.action == 'predict-go':
        if args.go_ids_file is None:
            raise ValueError('predict-go requires --go-ids-file')
        command = [args.python, str(ROOT/'scripts/nbs/predict_nbs_go_v094.py'),
                   '--work-dir', str(work), '--epoch', str(args.epoch), '--device', args.device,
                   '--input-dir', str(args.input_dir.resolve()), '--go-ids-file', str(args.go_ids_file.resolve())]
        if args.ontology_file:
            command += ['--ontology-file', str(args.ontology_file.resolve())]
        if args.output_dir:
            command += ['--output-dir', str(args.output_dir.resolve())]
        commands = [command]
    elif args.action == 'evaluate':
        checkpoint = work/f'nbs_epoch{args.epoch}.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        commands = [common + ['--stage', 'evaluate', '--checkpoint', str(checkpoint),
                    '--input-dir', str(args.input_dir.resolve()), '--metadata-file', str(args.metadata_file.resolve())]]
    elif args.action == 'prepare':
        commands = [common + ['--stage', 'prepare', '--backend', 'torch']]
    else:
        if args.action in ('train', 'smoke') and work.exists() and any(work.iterdir()):
            raise FileExistsError('Fresh training requires an empty work directory; use resume for the existing run')
        extra = []
        if args.action == 'resume':
            checkpoint = work/'latest.pt'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            extra += ['--resume', str(checkpoint)]
        if args.action == 'smoke':
            if args.stop_epoch is not None:
                raise ValueError('Smoke uses --stop-step, default3')
            extra += ['--stop-step', str(args.stop_step or 3)]
        elif args.stop_step is not None:
            extra += ['--stop-step', str(args.stop_step)]
        else:
            extra += ['--stop-epoch', str(args.stop_epoch or args.epochs)]
        commands = [common + ['--stage', 'prepare', '--backend', 'torch'],
                    common + ['--stage', 'train', *extra]]
    work.parent.mkdir(parents=True, exist_ok=True)
    # All evaluations serialize because inherited input-cache builders use a
    # shared temporary filename. Training has a separate, per-run lock.
    lock_path = operation_lock(args, work)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'The requested operation is already running: {lock_path}') from None
        for command in commands:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    print(f'[v094 complete] {args.action}: {work}', flush=True)


if __name__ == '__main__':
    main()
