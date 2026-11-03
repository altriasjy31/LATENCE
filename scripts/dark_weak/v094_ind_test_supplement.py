#!/usr/bin/env python3
"""Independent ind_test standardized v085 Fmax/AuPRC/AP for v094 B/G (+ E/M).

Reads a COMPLETED native v094 prediction. Never changes metrics_v094/summary.json
or the original model acceptance report. Historical ind_test labels are taken
from the requested metadata, never passed into a forward model.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts.dark_weak import v094_native_adapter as native
from scripts.dark_weak.native_v085_metrics import native_imports, unified_v085_metrics, native_protocol

NS = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}


def parse_optional_sources(values):
    sources = {}
    for value in values:
        if '=' not in value:
            raise ValueError('Reference requires NAME=path')
        name, path = value.split('=',1)
        if not name or name in ('NBS_final','backbone_base') or name in sources:
            raise ValueError(f'Invalid/duplicate reference name {name!r}')
        sources[name] = Path(path).resolve(strict=True)
    return sources


def dense_labels(meta, split, task, width):
    if split not in meta or NS[task] not in meta[split]:
        raise KeyError(f'No {split}/{NS[task]} in supplied metadata')
    block = meta[split][NS[task]]
    ids = [str(i) for i in block['proteins']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Metadata test protein IDs must be unique and nonempty')
    labels = block.get('prop_annotations')
    if labels is None or len(labels) != len(ids):
        raise ValueError('Standardized ind_test requires full ancestor prop_annotations')
    y = np.zeros((len(ids), width), np.uint8)
    for row, cols in enumerate(labels):
        arr = np.asarray(cols, np.int64)
        if len(arr) and (arr.min() < 0 or arr.max() >= width):
            raise ValueError('Metadata GO label outside immutable output axis')
        y[row, arr] = 1
    return ids, y


def align_reference(prob, row_file, go_file, target_ids, target_go, name):
    rids = native.read_ids(row_file)
    gids = native.read_ids(go_file)
    if len(rids) != len(target_ids) or set(rids) != set(target_ids):
        raise ValueError(f'{name}: protein identities differ from prepared targets')
    if gids != target_go:
        raise ValueError(f'{name}: GO order must be the same original model classifier order')
    arr = np.load(prob, mmap_mode='r', allow_pickle=False)
    if arr.shape != (len(target_ids), len(target_go)):
        raise ValueError(f'{name}: probability matrix shape mismatch')
    lookup = {p:i for i,p in enumerate(rids)}
    return np.asarray(arr[[lookup[p] for p in target_ids]], np.float32)


def run(args):
    cfg = native.json_file(args.config)
    native.validate_config(cfg, args.task)
    input_dir = args.input_dir.resolve(strict=True)
    ids = native.read_ids(input_dir/'protein_ids.txt')
    a = SimpleNamespace(input_dir=input_dir, expected_protein_ids=input_dir/'protein_ids.txt',
                        expected_mode='ind_test', task=args.task, go_ids=args.go_ids)
    prepared = native.validate_input(cfg, a)
    checkpoint_sha = native.sha256(args.checkpoint)
    pred_manifest = native.json_file(args.prediction_manifest)
    prob = Path(pred_manifest.get('probability_path','')).resolve(strict=True)
    pred = native.verify_native_prediction(pred_manifest, prob,
        SimpleNamespace(task=args.task, expected_epoch=args.expected_epoch), cfg,
        prepared, checkpoint_sha)
    if args.prediction_manifest.resolve() != (prob.parent/'nbs_full_task_prediction_manifest.json').resolve():
        raise ValueError('Prediction manifest is not adjacent to its verified probability')
    with args.metadata.open('rb') as handle:
        metadata = pickle.load(handle)
    metadata_ids, all_y = dense_labels(metadata, args.split, args.task, prepared['num_task_go'])
    if set(metadata_ids) != set(ids) or len(metadata_ids) != len(ids):
        raise ValueError('Gold metadata protein IDs differ from the native prediction population')
    idx = {name:i for i,name in enumerate(metadata_ids)}
    y = all_y[np.array([idx[name] for name in ids], np.int64)]
    methods = {
        'NBS_final': np.asarray(np.load(prob, mmap_mode='r', allow_pickle=False), np.float32),
        'backbone_base': np.asarray(np.load(input_dir/'backbone_ind_test_prob.f16.npy',mmap_mode='r'), np.float32),
    }
    target_go = native.read_ids(args.go_ids)
    ref = parse_optional_sources(args.reference)
    ref_ids = parse_optional_sources(args.reference_ids)
    ref_go = parse_optional_sources(args.reference_go_ids)
    if set(ref) != set(ref_ids) or set(ref) != set(ref_go):
        raise ValueError('Each E/M reference requires probability, explicit protein IDs and GO IDs')
    for name, path in ref.items():
        methods[name] = align_reference(path, ref_ids[name], ref_go[name], ids, target_go, name)
    kernel = native_imports(ROOT)
    result = {'schema':'latence-v094-indtest-supplement-v1',
        'primary_metric_contract': native_protocol(kernel),
        'task': args.task, 'split': args.split, 'rows':len(ids), 'num_task_go':y.shape[1],
        'positive_labels':int(y.sum()), 'gold_field':'prop_annotations',
        'native_prediction_manifest':str(args.prediction_manifest.resolve()),
        'native_prediction_manifest_sha256':native.sha256(args.prediction_manifest),
        'model_checkpoint':str(args.checkpoint.resolve()), 'model_checkpoint_sha256':checkpoint_sha,
        'prepared_input_manifest_sha256':prepared['input_manifest_sha256'],
        'test_metadata_sha256':native.sha256(args.metadata),
        'models':{}, 'deltas_vs_backbone':{}, 'prediction_storage_dtypes':{}}
    for name,p in methods.items():
        result['models'][name] = unified_v085_metrics(y, p, kernel)
        result['prediction_storage_dtypes'][name] = str(p.dtype)
    for name, item in result['models'].items():
        if name == 'backbone_base':
            continue
        result['deltas_vs_backbone'][name] = {k: item[k] - result['models']['backbone_base'][k]
            for k in ('Fmax_percent','AuPRC_percent','AP_percent')}
    original_path = prob.parent/'metrics_v094/summary.json'
    if original_path.is_file():
        original = native.json_file(original_path)
        if original.get('checkpoint_sha256') != checkpoint_sha or original.get('probability_sha256') != native.sha256(prob):
            raise ValueError('Existing native v094 metrics do not belong to the same checkpoint/prediction')
        for name in ('NBS_final', 'backbone_base'):
            entry = original.get('methods', {}).get(name)
            if not isinstance(entry, dict):
                raise ValueError(f'Original native v094 summary misses {name}')
            if not np.isclose(float(entry['fmax']), result['models'][name]['Fmax_percent'], rtol=0, atol=1e-8):
                raise ValueError(f'{name} Fmax differs from original native v094 summary')
            if not np.isclose(float(entry['micro_ap']), result['models'][name]['AP_percent'], rtol=0, atol=1e-8):
                raise ValueError(f'{name} AP differs from original native v094 summary')
        result['native_metrics_parity']='Fmax_and_micro_AP_exact'
    else:
        result['native_metrics_parity']='not_available_native_export_only'
    if args.out.is_file() and not args.overwrite:
        raise FileExistsError(f'Supplementary report already exists; do not overwrite: {args.out}')
    native.atomic_json(args.out, result, overwrite=args.overwrite)
    print('[v094 v085 supplementary] '+str(args.out), flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--task', default='bp', choices=tuple(NS))
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--split', default='ind_test')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prediction-manifest', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--go-ids', type=Path, required=True)
    p.add_argument('--expected-epoch', type=int)
    p.add_argument('--reference', action='append', default=[])
    p.add_argument('--reference-ids', action='append', default=[])
    p.add_argument('--reference-go-ids', action='append', default=[])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true')
    return run(p.parse_args(argv))


if __name__=='__main__':
    main()
