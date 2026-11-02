#!/usr/bin/env python3
"""Offline v092 probability transformations and fixed B/G mixtures; CPU only."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'nbs-offline-score-tools-v093-r1'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.', delete=False,
                                     encoding='utf-8') as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def resolve(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base/path).resolve()


def core_frequencies(config, summary, prediction_contract, project_root=ROOT):
    """Read only registry + core-gold CSR; reproduce the original holdout split.

    No training Data object, torch, protein features, graph pools or neighbour
    caches are constructed. The final arrays must match the v092 report hashes.
    """
    ft, data = config['full_task'], config['data']
    if ft.get('label_split', 'all_seen') not in ('all_seen', None):
        raise ValueError('This offline r1 reader requires the current all_seen v092 experiment')
    root = resolve(project_root, data['root'])
    registry_path = root/'features/protein_registry.csv'
    ids, local = [], []
    with registry_path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'protein_idx', 'role', 'role_row_idx'} <= set(reader.fieldnames or []):
            raise ValueError('Unexpected protein registry columns')
        for expected, row in enumerate(reader):
            index = int(row['protein_idx'])
            if index != expected:
                raise ValueError('Protein registry global IDs are not row-aligned')
            if row['role'] == 'core':
                ids.append(index)
                local.append(int(row['role_row_idx']))
        num_proteins = expected+1 if 'expected' in locals() else 0
    order = np.argsort(local, kind='stable')
    role_rows = np.asarray(local, dtype=np.int64)[order]
    if not np.array_equal(role_rows, np.arange(len(ids))):
        raise ValueError('Core role-local registry rows must be contiguous')
    core = np.asarray(ids, dtype=np.int64)[order]
    inverted_path = resolve(root, data['go_protein_inverted_index_manifest'])
    spec = read_json(inverted_path)['indices']['gold']['protein_major']
    if int(spec['num_go']) != int(summary['num_GO_columns']):
        raise ValueError('Core gold CSR GO axis differs from the evaluation')
    ptr_path = resolve(inverted_path.parent, spec['indptr'])
    go_path = resolve(inverted_path.parent, spec['go_idx'])
    for path, key in [(registry_path, 'registry_sha256'), (ptr_path, 'gold_indptr_sha256'),
                      (go_path, 'gold_go_idx_sha256'),
                      (inverted_path, 'go_protein_inverted_index_manifest_sha256')]:
        expected_hash = prediction_contract.get(key)
        if not isinstance(expected_hash, str) or sha256(path) != expected_hash:
            raise ValueError(f'Training data identity differs: {key}')
    ptr = np.load(ptr_path, mmap_mode='r', allow_pickle=False)
    go = np.load(go_path, mmap_mode='r', allow_pickle=False)
    if (ptr.ndim != 1 or go.ndim != 1 or ptr.dtype.kind not in 'iu' or go.dtype.kind not in 'iu'
            or len(ptr) != num_proteins+1 or ptr[0] != 0 or ptr[-1] != len(go)
            or np.any(ptr[1:] < ptr[:-1])):
        raise ValueError('Invalid row-aligned core gold CSR')
    core = core[np.diff(ptr)[core] > 0]
    holdout = ft.get('holdout_core_count', 0)
    if type(holdout) is not int or not 0 <= holdout < len(core):
        raise ValueError('Invalid core holdout size')
    rng = np.random.default_rng(int(ft.get('holdout_seed', 8080)))
    validation = np.sort(rng.permutation(core)[:holdout])
    training = np.setdiff1d(core, validation)
    if (len(training) != summary['num_training_core_proteins']
            or array_sha256(training) != summary['training_core_ids_sha256']):
        raise ValueError('Reconstructed training core IDs differ from the source summary')
    count = np.zeros(int(spec['num_go']), dtype=np.int64)
    for protein in training:
        labels = np.unique(go[int(ptr[protein]):int(ptr[protein+1])])
        if len(labels) and (labels.min() < 0 or labels.max() >= len(count)):
            raise ValueError('Core annotation outside the task GO axis')
        count[labels] += 1
    if array_sha256(count) != summary['core_frequency_sha256']:
        raise ValueError('Core frequencies differ from the original v092 strata')
    return count


def load_inputs(args):
    work = args.work_dir.resolve()
    folder = work/f'eval_step{args.epoch*48}'/'full'
    summary_path = folder/'metrics_v092/summary.json'
    summary = read_json(summary_path)
    if (summary.get('schema') != 'nbs-v092-extended-evaluation-v1' or not summary.get('complete')
            or summary.get('checkpoint_epoch') != args.epoch
            or summary.get('checkpoint_step') != args.epoch*48):
        raise ValueError('Use a completed fixed-epoch v092 extended evaluation')
    probability_path = folder/'nbs_ind_test_prob.f32.npy'
    manifest_path = folder/'nbs_full_task_prediction_manifest.json'
    manifest = read_json(manifest_path)
    digest = sha256(probability_path)
    if (digest != summary['probability_sha256'] or digest != manifest.get('output_probability_sha256')
            or sha256(manifest_path) != summary['prediction_manifest_sha256']
            or manifest.get('checkpoint_sha256') != summary['checkpoint_sha256']):
        raise ValueError('Prediction matrix/manifest differs from the source evaluation')
    input_dir = args.input_dir.resolve()
    input_sha = sha256(input_dir/'ind_test_input_manifest.json')
    if input_sha != summary['input_manifest_sha256'] or input_sha != manifest.get('input_manifest_sha256'):
        raise ValueError('Independent test inputs differ from the source evaluation')
    config_path = work/'resolved_config.json'
    config = read_json(config_path)
    if config.get('task') != summary.get('task'):
        raise ValueError('Resolved task differs from the evaluation')
    contract = manifest['contract']['data']
    representation_sha = contract.get('representation_manifest_sha256')
    if not isinstance(representation_sha, str) or not representation_sha:
        raise ValueError('Source prediction lacks training representation provenance')
    helper_path = ROOT/'scripts/nbs/diagnose_nbs_knn_v090.py'
    metric_path = ROOT/'nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py'
    helpers = load_module(helper_path, '_v093_existing_input_helpers')
    metrics = load_module(metric_path, '_v093_original_protein_metrics')
    test = helpers.load_test_inputs(input_dir, args.metadata_file.resolve(), config['task'], representation_sha)
    y = np.asarray(test['labels'], dtype=bool)
    b = np.asarray(test['backbone'])
    del test['features']
    g = np.load(probability_path, mmap_mode='r', allow_pickle=False)
    if y.shape != (summary['num_proteins'], summary['num_GO_columns']) or g.shape != y.shape or b.shape != y.shape:
        raise ValueError('Protein/GO dimensions differ from the source summary')
    counts = core_frequencies(config, summary, contract)
    identity = {
        'schema': SCHEMA, 'epoch': args.epoch, 'task': config['task'],
        'source_summary_sha256': sha256(summary_path), 'source_probability_sha256': digest,
        'source_prediction_manifest_sha256': sha256(manifest_path), 'input_manifest_sha256': input_sha,
        'metadata_sha256': sha256(args.metadata_file.resolve()),
        'backbone_probability_sha256': sha256(input_dir/'backbone_ind_test_prob.f16.npy'),
        'core_frequency_sha256': array_sha256(counts), 'resolved_config_sha256': sha256(config_path),
        'input_helper_sha256': sha256(helper_path), 'metric_module_sha256': sha256(metric_path),
        'tool_sha256': sha256(__file__),
        'math_module_sha256': sha256(Path(__file__).with_name('nbs_score_postprocess_v093.py')),
    }
    return y, b, g, counts, summary, metrics, identity


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('transform', 'mix', 'all'), nargs='?', default='all')
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--work-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v092/geometry_single12800_scratch')
    p.add_argument('--input-dir', type=Path, default=ROOT/'outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2')
    p.add_argument('--metadata-file', type=Path, default=Path('/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl'))
    p.add_argument('--output-dir', type=Path)
    return p


def write_table(out, methods):
    with (out/'methods.csv').open('w', encoding='utf-8', newline='') as stream:
        names = ['method', 'fmax', 'threshold', 'precision', 'recall', 'micro_ap', 'macro_ap', 'macro_ap_eligible_go']
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for name, row in methods.items():
            writer.writerow({'method': name, **{k: row.get(k) for k in names if k != 'method'}})
    with (out/'frequency_strata.csv').open('w', encoding='utf-8', newline='') as stream:
        names = ['method', 'core_min', 'core_max', 'micro_ap', 'macro_ap', 'macro_ap_eligible_go',
                 'test_positive_pairs', 'tp', 'predicted_pairs', 'pair_precision', 'pair_recall']
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for name, method in methods.items():
            for row in method['frequency_strata']:
                point = row['at_method_fmax']
                writer.writerow({'method': name, 'core_min': row['train_core_count_min'],
                    'core_max': row['train_core_count_max'],
                    **{key: row.get(key) for key in ('micro_ap', 'macro_ap', 'macro_ap_eligible_go', 'test_positive_pairs')},
                    **{key: point.get(key) for key in ('tp', 'predicted_pairs', 'pair_precision', 'pair_recall')}})


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.epoch <= 100:
        raise SystemExit('--epoch must identify an existing v092 epoch 1..100 evaluation')
    os.chdir(ROOT)
    out = (args.output_dir or args.work_dir/f'score_postprocess_epoch{args.epoch}'/args.mode).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'analysis.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This offline analysis output is already in use') from None
        y, b, g, counts, source, metrics, identity = load_inputs(args)
        identity['mode'] = args.mode
        identity_path = out/'inputs.json'
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError('Existing diagnostic belongs to different inputs/code; use a new --output-dir')
        core = load_module(Path(__file__).with_name('nbs_score_postprocess_v093.py'), '_v093_score_math')
        report_path = out/'report.json'
        if report_path.exists():
            existing = read_json(report_path)
            if existing.get('schema') != SCHEMA or existing.get('identity') != identity:
                raise ValueError('Existing report belongs to different inputs/code; use a new --output-dir')
            for name, row in existing.get('methods', {}).items():
                ap_path = out/row.get('per_go_ap_file', '')
                if not ap_path.is_file() or ap_path.parent != out:
                    raise ValueError(f'Existing report is incomplete: missing per-GO AP for {name}')
        atomic_json(identity_path, identity)
        report = read_json(report_path) if report_path.exists() else {
            'schema': SCHEMA, 'complete': False, 'identity': identity, 'methods': {},
            'source_epoch': args.epoch, 'source_metrics': source['metrics'],
            'score_unit': 'percent; thresholds use probability units',
            'protocol': 'offline fixed descriptive methods; no training and no parameter selection on labels',
            'acceptance_protocol_changed': False, 'zero_shot_claimed': False,
            'normalization_note': 'GO-wise statistics from this unlabeled evaluation cohort are transductive diagnostics, not independently fitted calibration',
            'frequency_note': 'core training counts only, original holdout excluded; 1-5 is nested in 1-9',
        }
        baseline_threshold = float(source['metrics']['backbone_base']['threshold'])
        seen = set()
        modes = ('transform', 'mix') if args.mode == 'all' else (args.mode,)
        for mode in modes:
            for name, scores, description in core.build_methods(mode, b, g, counts):
                if name in seen:
                    continue
                seen.add(name)
                if name in report['methods']:
                    continue
                print(f'[score-tools] evaluating {name}', flush=True)
                start = time.monotonic()
                result = core.summarize_method(y, scores, counts, metrics, backbone_threshold=baseline_threshold)
                ap = result.pop('_per_go_ap')
                if name in ('B', 'G'):
                    old = source['metrics']['backbone_base' if name == 'B' else 'NBS_final']
                    for key in ('fmax', 'micro_ap'):
                        equal = (result[key] is None and old[key] is None) or (
                            result[key] is not None and old[key] is not None
                            and math.isclose(result[key], old[key], rel_tol=0., abs_tol=1e-6))
                        if not equal:
                            raise ValueError(f'{name} {key} fails source-metric reproduction; no transformed result is accepted')
                ap_path = out/f'{name}_per_go_ap.npy'
                with tempfile.NamedTemporaryFile('wb', dir=out, prefix=ap_path.name+'.', delete=False) as stream:
                    temporary = Path(stream.name)
                    np.save(stream, ap, allow_pickle=False)
                os.replace(temporary, ap_path)
                result.update(description=description, elapsed_seconds=time.monotonic()-start,
                              per_go_ap_file=ap_path.name,
                              delta_vs_B={key: (None if result[key] is None or source['metrics']['backbone_base'][key] is None
                                               else result[key]-source['metrics']['backbone_base'][key])
                                          for key in ('fmax', 'micro_ap')})
                report['methods'][name] = result
                report['complete'] = False
                atomic_json(report_path, report)
                write_table(out, report['methods'])
                formatted = {key: ('NA' if result[key] is None else f'{result[key]:.6f}')
                             for key in ('fmax', 'micro_ap', 'macro_ap')}
                print(f"[score-tools] {name}: Fmax={formatted['fmax']} AP={formatted['micro_ap']} macroAP={formatted['macro_ap']}", flush=True)
                del scores, ap
        report['complete'] = True
        atomic_json(report_path, report)
        write_table(out, report['methods'])
        print(f'[score-tools] completed: {report_path}', flush=True)


if __name__ == '__main__':
    main()
