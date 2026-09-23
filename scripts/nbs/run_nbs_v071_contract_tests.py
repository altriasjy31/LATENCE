#!/usr/bin/env python3
"""Run bounded NBS regression contracts using PyTorch and the standard library.

No pytest dependency; invokes the same test functions pytest would collect.
PyG graph materialization and production CUDA/DDP remain server smoke gates.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import platform
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'nbs_models' / 'nbs_protein_go'
for path in (ROOT, MODEL, MODEL / 'tests', ROOT / 'scripts/nbs'):
    sys.path.insert(0, str(path))

MODULES = (
    'test_alignment_v071', 'test_candidate_evidence_encoder',
    'test_dual_axis_loss_v070', 'test_isolated_candidate_encoder_v070',
    'test_go_residual_query_v070', 'test_residual_refinement',
    'test_matcher_equivalence', 'test_source_additivity',
    'test_hybrid_weak_focus_v054', 'test_background_unlabelled_v056',
    'test_supervision_weight', 'test_loss_v045',
    'test_fixed_epoch_training', 'test_episode_sampler',
    'test_eval_checkpoint_series_summary', 'test_experiment_runner_v071',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/latence_nbs_tests/v071_contracts.json')
    parser.add_argument('--require-pyg', action='store_true', help='Server gate: reject missing PyG rather than skip its graph contracts')
    args = parser.parse_args()
    if args.require_pyg and importlib.util.find_spec('torch_geometric') is None:
        parser.error('torch_geometric is required for the server graph contracts')
    import torch
    torch.set_num_threads(1)
    results = []
    for name in MODULES:
        if name in {'test_source_additivity', 'test_isolated_candidate_encoder_v070'} and importlib.util.find_spec('torch_geometric') is None:
            results.append({'test': name, 'status': 'skipped', 'reason': 'PyG graph/scatter dependency unavailable; run --require-pyg on the training server'})
            continue
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            results.append({'test': name, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            traceback.print_exc()
            continue
        for key, function in vars(module).items():
            if not key.startswith('test_') or not inspect.isfunction(function):
                continue
            entry = {'test': f'{name}.{key}'}
            try:
                with tempfile.TemporaryDirectory(prefix='nbs_v071_contract_') as temp:
                    kwargs = {}
                    for parameter in inspect.signature(function).parameters:
                        if parameter != 'tmp_path':
                            raise ValueError(f'unsupported test fixture {parameter}; use pytest for this test')
                        kwargs[parameter] = Path(temp)
                    function(**kwargs)
                entry['status'] = 'passed'
            except Exception as exc:
                entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
            results.append(entry)
            print(f"{entry['status'].upper()} {entry['test']}", flush=True)
    payload = {'python': platform.python_version(), 'torch': torch.__version__,
               'cuda_available': torch.cuda.is_available(),
               'scope': 'CPU tensor/sampler/trainer/experiment contracts; no production data or PyG graph materialization',
               'passed': sum(r['status'] == 'passed' for r in results),
               'skipped': sum(r['status'] == 'skipped' for r in results),
               'failed': sum(r['status'] == 'failed' for r in results), 'tests': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: payload[k] for k in ('passed', 'failed', 'skipped', 'torch')}))
    raise SystemExit(bool(payload['failed']))


if __name__ == '__main__':
    main()
