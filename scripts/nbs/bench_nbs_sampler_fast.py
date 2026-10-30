#!/usr/bin/env python3
"""Read-only CPU batch equivalence and timing; excludes model/H2D/DDP costs.

Global supervision includes every simulated rank and accumulation microbatch.
Only explicitly compared cases are covered; synthetic seeds are not stream replay.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'nbs_models/nbs_protein_go')]
RELEASES = {
    '0.8.9.1': ('nbs_pg.local_loader_v089', 'nbs_pg.full_task_data_v088.FullTaskDataV088', 'FullTaskDataV088Fast'),
    '0.9.0': ('nbs_pg.local_loader_v090', 'nbs_pg.full_task_data_v090.FullTaskDataV090', 'FullTaskDataV090Fast'),
}

def log(message):
    print(f"[sampler-bench {time.strftime('%H:%M:%S')}] {message}", flush=True)

def differences(left, right, path='batch'):
    """Compare every key, sequence type, dtype, shape and array/tensor byte."""
    if isinstance(left, dict) and isinstance(right, dict):
        missing = set(left) ^ set(right)
        result = [f'{path}: unequal keys {sorted(missing, key=str)}'] if missing else []
        for key in left:
            if key in right:
                result.extend(differences(left[key], right[key], f'{path}.{key}'))
        return result
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        result = []
        if type(left) is not type(right):
            result.append(f'{path}: unequal sequence types')
        if len(left) != len(right):
            result.append(f'{path}: length {len(left)} != {len(right)}')
        for i, (a, b) in enumerate(zip(left, right)):
            result.extend(differences(a, b, f'{path}[{i}]'))
        return result
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape:
            return [f'{path}: unequal array dtype or shape']
        same = np.array_equal(left, right) if left.dtype.hasobject else left.tobytes() == right.tobytes()
        return [] if same else [f'{path}: unequal array bytes/values']
    torch = sys.modules.get('torch')
    if torch is not None and torch.is_tensor(left) and torch.is_tensor(right):
        if left.dtype != right.dtype or left.shape != right.shape or left.layout != right.layout:
            return [f'{path}: unequal tensor dtype, shape or layout']
        if left.layout != torch.strided:
            raise TypeError(f'{path}: expected dense batch tensor')
        a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return [] if torch.equal(a, b) else [f'{path}: unequal tensor bytes']
    if type(left) is not type(right):
        return [f'{path}: unequal types {type(left).__name__}/{type(right).__name__}']
    return [] if left == right else [f'{path}: {left!r} != {right!r}']

def timed(function, *args, **kwargs):
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - start

def build(config):
    release = config.get('release_version')
    if release not in RELEASES:
        raise ValueError(f'unsupported release {release!r}')
    loader, class_path, fast_name = RELEASES[release]
    module, name = class_path.rsplit('.', 1)
    fast = importlib.import_module('nbs_pg.full_task_sampler_fast')
    stores = importlib.import_module(loader).build_latence_nbs_stores(config)
    original = getattr(importlib.import_module(module), name)(config, stores=stores)
    log('verifying existing caches with original ensure_prepared')
    original.ensure_prepared()
    accelerated = copy.copy(original)
    accelerated.__class__ = getattr(fast, fast_name)
    return original, accelerated, fast

def macro_seeds(data, weak, core, world_size, seed):
    """Unique global role samples and rank-local partitions, before accumulation."""
    if any(type(v) is not int or v < 1 for v in (weak, core, world_size)):
        raise ValueError('macro role counts and world_size must be positive integers')
    rng = np.random.default_rng(seed)
    w = rng.choice(data.weak_ids, weak * world_size, replace=False)
    c = rng.choice(data.core_ids, core * world_size, replace=False)
    ranks = [(w[r*weak:(r+1)*weak], c[r*core:(r+1)*core]) for r in range(world_size)]
    return np.concatenate((w, c)).astype(np.int64), ranks

def microbatches(weak, core, accumulation):
    if type(accumulation) is not int or accumulation < 1 or len(core) < accumulation:
        raise ValueError('accumulation must be positive and <= local core count')
    return [np.concatenate((w, c)) for w, c in zip(np.array_split(weak, accumulation),
                                                 np.array_split(core, accumulation))]

def compare_training(original, accelerated, batches, global_ids, steps, ranks, label, contexts=(True, False)):
    results = []
    for rank in ranks:
        for training in contexts:
            for step in steps:
                for micro, ids in enumerate(batches[rank]):
                    row = dict(case=label, compared=True, training=training, step=step, rank=rank,
                               microbatch=micro, queries=len(ids), global_supervision_queries=len(global_ids))
                    values = []
                    for name, data in (('original', original), ('fast', accelerated)):
                        data.set_sampling_context(step=step, rank=rank, training=training)
                        batch, seconds = timed(data.batch, ids, 'cpu', supervision_seed_ids=global_ids)
                        values.append(batch)
                        row[name + '_comparison_seconds'] = seconds
                    row['differences'] = differences(*values)
                    row['identical'] = not row['differences']
                    results.append(row)
                    log(f"{label} rank={rank} train={training} step={step} micro={micro}: identical={row['identical']}")
    return results

def timing_series(data, batches, global_ids, step, rank, warmup, repeats):
    """One timing sample includes ALL rank-local microbatches of one update."""
    def one():
        data.set_sampling_context(step=step, rank=rank, training=True)
        start = time.perf_counter()
        for ids in batches:
            batch = data.batch(ids, 'cpu', supervision_seed_ids=global_ids)
            del batch
        return time.perf_counter() - start
    for _ in range(warmup):
        one()
    seconds = [one() for _ in range(repeats)]
    return dict(seconds=seconds, median_seconds=float(np.median(seconds)),
                min_seconds=float(min(seconds)), max_seconds=float(max(seconds)),
                warmup_builds=warmup, measured_builds=repeats, rank=rank, sampling_step=step,
                microbatches_per_build=len(batches), queries_per_build=sum(map(len, batches)),
                global_supervision_queries=len(global_ids), scope='CPU batch construction only')

def compare_inference(original, accelerated, input_dir, rows, batch_size):
    rows = min(rows, len(np.load(input_dir / 'ind_test_repr.f16.npy', mmap_mode='r')))
    if rows < 1:
        raise ValueError('inference input has no rows to compare')
    result = dict(compared=True, rows=rows, batch_size=batch_size, differences=[])
    for start in range(0, rows, batch_size):
        ids = np.arange(start, min(start + batch_size, rows))
        a = original.inference_batch(input_dir, ids, 'cpu')
        b = accelerated.inference_batch(input_dir, ids, 'cpu')
        result['differences'].extend(differences(a, b, f'inference[{start}]'))
    result['identical'] = not result['differences']
    return result

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='new JSON report path')
    p.add_argument('--compare-weak', type=int, default=1280)
    p.add_argument('--compare-core', type=int, default=320)
    p.add_argument('--full-weak', type=int, help='default: config per-rank macro weak_batch')
    p.add_argument('--full-core', type=int, help='default: config per-rank macro core_batch')
    p.add_argument('--steps', default='1,2,4097', help='sampling contexts; not seed-stream replay')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--all-ranks', action='store_true')
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--full-original', action='store_true', help='compare full-size original/fast, first step, training context')
    p.add_argument('--full-eval', action='store_true', help='with --full-original, also compare full-size eval context')
    p.add_argument('--time-full-original', action='store_true', help='with --full-original, also repeat/time original full builds; expensive')
    p.add_argument('--input-dir', type=Path)
    p.add_argument('--inference-rows', type=int, default=64)
    p.add_argument('--inference-batch', type=int, default=16)
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'report exists: {args.output}')
    if (args.full_eval or args.time_full_original) and not args.full_original:
        raise ValueError('--full-eval and --time-full-original require --full-original')
    config = json.loads(args.config.read_text())
    cfg = config['full_task']
    world, accumulation = cfg.get('expected_world_size', 1), cfg.get('accumulation_steps', 1)
    weak = cfg['weak_batch'] if args.full_weak is None else args.full_weak
    core = cfg['core_batch'] if args.full_core is None else args.full_core
    if type(world) is not int or world < 1 or not 0 <= args.rank < world:
        raise ValueError('rank must lie in config expected_world_size')
    if args.warmup < 0 or args.repeats < 1 or args.inference_rows < 1 or args.inference_batch < 1:
        raise ValueError('warmup>=0, repeats>=1, inference counts>=1 required')
    if not 1 <= args.compare_weak <= weak or not 1 <= args.compare_core <= core:
        raise ValueError('comparison counts must be positive and <= full local macro role counts')
    steps = [int(x) for x in args.steps.split(',') if x.strip()]
    if not steps or any(x < 0 for x in steps):
        raise ValueError('steps must be a nonempty list of nonnegative integers')
    original, accelerated, fast = build(config)
    global_ids, per_rank = macro_seeds(original, weak, core, world, args.seed)
    full = [microbatches(w, c, accumulation) for w, c in per_rank]
    small = [microbatches(w[:args.compare_weak], c[:args.compare_core], accumulation) for w, c in per_rank]
    ranks = list(range(world)) if args.all_ranks else [args.rank]
    import torch
    report = dict(schema='nbs-fast-sampler-bench-v2', config=str(args.config.resolve()),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  release_version=config['release_version'], sampler_layers=int(original.NUM_LAYERS),
                  fast_sampler_version=fast.FAST_SAMPLER_VERSION, fast_sampler_sha256=fast.implementation_sha256(),
                  threads=torch.get_num_threads(), expected_world_size=world, compared_ranks=ranks,
                  accumulation_steps=accumulation, global_supervision_queries=len(global_ids), seed=args.seed,
                  global_supervision_sha256=hashlib.sha256(global_ids.tobytes()).hexdigest(),
                  seeds='synthetic unique role samples; runner-shaped, not stream replay',
                  full_size_matches_config=(weak == cfg['weak_batch'] and core == cfg['core_batch']))
    report['equivalence'] = compare_training(original, accelerated, small, global_ids, steps, ranks, 'subset')
    report['full_size'] = dict(compared=args.full_original, identical=None)
    if args.full_original:
        contexts = (True, False) if args.full_eval else (True,)
        rows = compare_training(original, accelerated, full, global_ids, steps[:1], ranks, 'full', contexts)
        report['full_size'].update(cases=rows, identical=all(row['identical'] for row in rows))
    batches = full if args.time_full_original else small
    report['paired_timing'] = {name: timing_series(data, batches[args.rank], global_ids,
                                                   steps[0], args.rank, args.warmup, args.repeats)
                              for name, data in (('original', original), ('fast', accelerated))}
    times = report['paired_timing']
    times['median_build_speedup'] = times['original']['median_seconds'] / times['fast']['median_seconds']
    if not args.time_full_original:
        report['full_size']['fast_only_timing'] = timing_series(accelerated, full[args.rank], global_ids,
                                                               steps[0], args.rank, args.warmup, args.repeats)
    if args.input_dir is not None:
        report['inference'] = compare_inference(original, accelerated, args.input_dir.resolve(),
                                                args.inference_rows, args.inference_batch)
    errors = [d for row in report['equivalence'] + report['full_size'].get('cases', []) for d in row['differences']]
    errors += report.get('inference', {}).get('differences', [])
    report['identical'] = not errors
    report['identical_scope'] = 'explicitly_compared_cases_only; untested configurations are not certified'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    log(f"compared identical={report['identical']}; full compared={args.full_original}; {args.output}")
    return 0 if report['identical'] else 1

if __name__ == '__main__':
    sys.exit(main())
