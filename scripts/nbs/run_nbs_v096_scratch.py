#!/usr/bin/env python3
"""Fixed v096 scratch profile: random Stage2 initialization, peak LR .003.

The v095 source directory supplies dataset paths and architecture/optimization
protocol only. Checkpoint initialization is forbidden. Resume/evaluation use the
v096 run's own frozen configuration without reopening the v095 source directory.
The original v096 r1 worker and checkpoint implementation remain unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.nbs import run_nbs_v096 as base

PROFILE = ROOT / 'configs/nbs/v096_scratch_single12800.json'
FRESH_ACTIONS = {'build', 'prepare', 'smoke', 'train'}
FROZEN_ACTIONS = {'resume', 'evaluate', 'predict-go'}


def assert_subset(actual, expected, label='full_task'):
    """Validate profile-owned fields; keep source paths and sampler details."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f'Scratch profile mismatch: {label}')
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f'Scratch profile missing field: {label}.{key}')
            assert_subset(actual[key], value, label + '.' + key)
    elif actual != expected:
        raise ValueError(f'Scratch profile mismatch: {label}: {actual!r} != {expected!r}')


def load_profile(path):
    path = Path(path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') != 'nbs-v096-scratch-profile-v1':
        raise ValueError('Unknown scratch profile schema')
    ft = payload.get('full_task', {})
    # The additive wrapper deliberately controls only these fresh-run CLI
    # fields. All other profile fields must match the inherited protocol.
    if ft.get('initialization') != {'mode': 'fresh'}:
        raise ValueError('Scratch profile requires initialization.mode=fresh only')
    lr = ft.get('learning_rate')
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Scratch profile peak learning rate must be positive and finite')
    if type(ft.get('epochs')) is not int or ft['epochs'] < 11:
        raise ValueError('Scratch profile requires a fixed horizon >=11')
    if ft.get('precision') not in ('bf16', 'fp32') or not payload.get('work_dir'):
        raise ValueError('Scratch profile needs precision and work_dir')
    if ft.get('seed') != 1096 or ft.get('model', {}).get('esm_gate_init') != 0.:
        raise ValueError('This v096 runtime uses seed1096 and esm_gate_init=0')
    if ft.get('scheduler', {}).get('min_lr', 0) > lr:
        raise ValueError('Scratch profile minimum LR exceeds peak LR')
    return payload


def validate_source(args, profile):
    source = args.source_work_dir.resolve() / 'resolved_config.json'
    config = json.loads(source.read_text(encoding='utf-8'))
    if config.get('release_version') != '0.9.5' or config.get('model_version') != '0.9.5':
        raise ValueError('Scratch source must be a v095 resolved_config.json')
    expected = deepcopy(profile['full_task'])
    for key in ('initialization', 'seed', 'epochs', 'learning_rate', 'precision', 'checkpoint_every_epochs'):
        expected.pop(key, None)
    for key in ('esm_dim', 'esm_dropout', 'esm_gate_init'):
        expected.get('model', {}).pop(key, None)
    assert_subset(config.get('full_task', {}), expected, 'source.full_task')
    return config


def validate_frozen(config, profile):
    if config.get('release_version') != '0.9.6' or config.get('model_version') != '0.9.6':
        raise ValueError('This scratch entrypoint requires its own v096 run')
    ft = config.get('full_task', {})
    if ft.get('initialization') != {'mode': 'fresh'}:
        raise ValueError('This entrypoint only resumes/evaluates a scratch run, not weight initialization')
    assert_subset(ft, profile['full_task'])


def _argv(args):
    result = [args.action]
    for name, value in vars(args).items():
        if name != 'action' and value is not None:
            result.extend(['--' + name.replace('_', '-'), str(value)])
    return result


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    profile_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    profile_parser.add_argument('--profile', type=Path, default=PROFILE)
    selected, remaining = profile_parser.parse_known_args(raw)
    profile = load_profile(selected.profile)
    p = base.parser()
    p.description = __doc__
    help_text = {
        'init_from': argparse.SUPPRESS,
        'learning_rate': f"Optional assertion of profile peak LR ({profile['full_task']['learning_rate']}); conflicting values fail",
        'precision': f"Optional assertion of profile precision ({profile['full_task']['precision']}); conflicting values fail",
        'epochs': f"Fixed profile horizon ({profile['full_task']['epochs']}); use --stop-epoch for an earlier stop",
        'source_work_dir': 'v095 directory supplying data paths and protocol only; no model checkpoint is loaded',
        'work_dir': 'Independent v096 scratch run directory; default comes from the explicit JSON profile',
    }
    for action in p._actions:
        if action.dest in help_text:
            action.help = help_text[action.dest]
    p.epilog = 'Optional --profile PATH selects the explicit JSON profile; esm_dim is inferred from its cache.'
    work = Path(profile['work_dir'])
    p.set_defaults(work_dir=work if work.is_absolute() else ROOT / work,
                   epochs=profile['full_task']['epochs'])
    args = p.parse_args(remaining)
    if args.init_from is not None:
        raise ValueError('Scratch training forbids --init-from; the source directory supplies configuration only')
    if args.epochs != profile['full_task']['epochs']:
        raise ValueError('epochs differs from the fixed scratch profile; use --stop-epoch for an earlier stop')
    if args.action in FRESH_ACTIONS:
        for option in ('learning_rate', 'precision'):
            value = getattr(args, option)
            if value is not None and value != profile['full_task'][option]:
                raise ValueError(f'{option} differs from the explicit scratch profile')
            setattr(args, option, profile['full_task'][option])
        validate_source(args, profile)
    elif args.action in FROZEN_ACTIONS:
        if args.learning_rate is not None or args.precision is not None:
            raise ValueError('Resume/evaluation keep the saved learning rate and precision; omit new overrides')
        # No v095 file is read here. Same-run resume restores its optimizer,
        # scheduler step and RNG; the wrapper never resets those states.
        path = args.work_dir.resolve() / 'resolved_config.json'
        validate_frozen(json.loads(path.read_text(encoding='utf-8')), profile)
    original_configure = base.configure

    def configure_scratch(parsed):
        if parsed.init_from is not None:
            raise ValueError('Scratch training forbids checkpoint initialization')
        path = original_configure(parsed)
        validate_frozen(json.loads(path.read_text(encoding='utf-8')), profile)
        snapshot = path.parent / 'scratch_profile.json'
        record = {'profile': profile, 'profile_sha256': hashlib.sha256(
            json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
        if snapshot.exists():
            if json.loads(snapshot.read_text(encoding='utf-8')) != record:
                raise ValueError('Scratch profile snapshot differs; use a new work directory')
        else:
            base.atomic_json(snapshot, record)
        return path

    try:
        base.configure = configure_scratch
        return base.main(_argv(args))
    finally:
        base.configure = original_configure


if __name__ == '__main__':
    main()
