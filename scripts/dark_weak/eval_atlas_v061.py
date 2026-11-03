#!/usr/bin/env python3
"""Config-free Stage-1 export input adapter for the unchanged v0.6.0 evaluator.

This frontend reads native Stage-1 prediction manifests and writes a protected
sidecar input record. It never runs a model or fabricates missing prediction
rows. Existing --complete-manifest and --prediction-config inputs still work.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CHANNELS = {'backbone_base': 'backbone', 'expert': 'expert', 'modelout': 'modelout'}
NAMESPACES = {'bp', 'mf', 'cc'}
SCHEMA = 'latence-stage1-export-input-bridge-v061'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_ids(path, kind):
    values = [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f'{kind} file must contain nonempty, unique identifiers: {path}')
    return values


def source_path(export_dir, text):
    """Use the recorded path, permitting a hash-checked relocated basename."""
    path = Path(text)
    recorded = path if path.is_absolute() else export_dir / path
    if recorded.is_file():
        return recorded.resolve(), 'recorded_path'
    relocated = export_dir / path.name
    if relocated.is_file():
        return relocated.resolve(), 'relocated_basename_with_required_hash'
    raise FileNotFoundError(f'Native manifest output missing: {recorded}')


def build_stage1_records(namespace, export_dir):
    """Validate a native three-channel export and return v060 input records.

    Width comes from the native manifest; the v060 evaluator independently
    checks the full ordered vocabulary against the unified dataset.
    """
    if namespace not in NAMESPACES:
        raise ValueError(f'Unknown namespace: {namespace}')
    directory = Path(export_dir).resolve(strict=True)
    manifest_path = directory / 'prediction_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('model_family') != 'LATENCE Stage1 DETR' or manifest.get('task') != namespace:
        raise ValueError(f'{manifest_path}: not the native Stage-1 export for namespace {namespace}')
    if manifest.get('forward_uses_eval_labels') is not False:
        raise ValueError('Native Stage-1 manifest must explicitly declare forward_uses_eval_labels=false')
    if set(manifest.get('channels', ())) != set(CHANNELS):
        raise ValueError('Stage-1 export directory must contain backbone_base, expert and modelout channels; '
                         'export with --export-expert --export-modelout, or use --complete-manifest for other inputs')
    outputs = manifest.get('output_files')
    if not isinstance(outputs, dict) or set(outputs) != set(CHANNELS):
        raise ValueError('Native output_files and channels do not describe the same three Stage-1 outputs')
    protein_file, go_file = directory / 'protein_ids.txt', directory / 'go_ids.txt'
    proteins, gos = read_ids(protein_file, 'Protein ID'), read_ids(go_file, 'GO ID')
    if len(proteins) != manifest.get('num_proteins') or len(gos) != manifest.get('num_go'):
        raise ValueError('Native manifest num_proteins/num_go differs from exported ID files')
    id_hash, go_hash = sha256(protein_file), sha256(go_file)
    # Native exporter records hashes of the SOURCE expected-ID/GO files, then
    # writes normalized ID copies. Matching bytes bind these output copies to
    # the recorded sources. Do not silently bypass a mismatch due to formatting.
    for key, actual in [('expected_protein_ids_sha256', id_hash), ('go_ids_sha256', go_hash)]:
        if not manifest.get(key) or manifest[key] != actual:
            raise ValueError(f'{key} does not bind the exported ID file bytes. Native hashes refer to source '
                             'ID files; if those used different formatting, audit the original ID order before use.')
    provenance = {'schema': SCHEMA, 'native_export_manifest': str(manifest_path),
        'native_export_manifest_sha256': sha256(manifest_path), 'namespace': namespace,
        'num_proteins': len(proteins), 'num_go': len(gos),
        'exported_protein_ids_sha256': id_hash, 'exported_go_ids_sha256': go_hash,
        'id_hash_scope': 'exported copies byte-match native source expected-ID/GO hashes',
        'checkpoint': manifest.get('checkpoint'), 'checkpoint_sha256': manifest.get('checkpoint_sha256'),
        'input_metadata': manifest.get('input_metadata'), 'mode': manifest.get('mode'),
        'modelout_prediction_key': manifest.get('modelout_prediction_key'),
        'historical_input_cutoff_audited_by_this_frontend': False}
    records = {}
    for channel, model in CHANNELS.items():
        info = outputs[channel]
        path, resolution = source_path(directory, info['path'])
        digest = sha256(path)
        if not info.get('sha256') or info['sha256'] != digest:
            raise ValueError(f'{namespace}/{channel}: probability SHA256 differs from native export manifest')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.shape != (len(proteins), len(gos)) or array.dtype.kind != 'f':
            raise ValueError(f'{namespace}/{channel}: probability matrix shape/dtype differs from native ID axes')
        for expected_dtype in (info.get('storage_dtype'), manifest.get('output_storage_dtype')):
            if expected_dtype is not None and str(array.dtype) != expected_dtype:
                raise ValueError(f'{namespace}/{channel}: storage dtype differs from native manifest')
        for start in range(0, len(proteins), 64):
            block = array[start:start + 64]
            if not np.isfinite(block).all() or np.any((block < 0) | (block > 1)):
                raise ValueError(f'{namespace}/{channel}: probabilities must be finite in [0,1]')
        branch = channel
        if model == 'modelout':
            branch = manifest.get('modelout_prediction_key') or manifest.get('modelout_semantics') or 'not_supplied'
        records[model] = {'probability': str(path), 'protein_ids': str(protein_file), 'go_ids': str(go_file),
            'probability_sha256': digest, 'protein_ids_sha256': id_hash, 'go_ids_sha256': go_hash,
            'branch': branch, 'source_export_manifest': str(manifest_path),
            'source_export_manifest_sha256': provenance['native_export_manifest_sha256'],
            'source_path_resolution': resolution}
    return records, provenance


def prepare_generated_inputs(specifications, out, input_dir=None):
    """Create one new persistent input sidecar; original exports stay untouched."""
    out = Path(out).resolve()
    directory = Path(input_dir).resolve() if input_dir else out.with_name(out.name + '_inputs')
    if out.exists() or directory.exists():
        raise FileExistsError(f'Output or generated-input directory exists; choose new paths: {out}, {directory}')
    if directory == out or out in directory.parents:
        raise ValueError('--input-dir must be separate from the evaluator output directory')
    config, provenance = {}, {}
    for item in specifications:
        if '=' not in item:
            raise ValueError('--stage1-export-dir requires NAMESPACE=DIRECTORY')
        namespace, path = item.split('=', 1)
        if namespace in config:
            raise ValueError(f'Duplicate Stage-1 namespace: {namespace}')
        config[namespace], provenance[namespace] = build_stage1_records(namespace, path)
    if not config:
        raise ValueError('At least one Stage-1 export directory is required')
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / 'prediction_inputs.json'
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    record = {'schema': SCHEMA, 'generated_config': str(config_path),
              'generated_config_sha256': sha256(config_path), 'native_exports': provenance,
              'no_missing_predictions_synthesized': True}
    (directory / 'source_provenance.json').write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return config_path


def load_runner(project_root=None):
    path = Path(project_root or PROJECT) / 'scripts/dark_weak/eval_atlas_v060.py'
    if not path.is_file():
        raise FileNotFoundError(f'Install the v0.6.0 Atlas evaluator first: {path}')
    spec = importlib.util.spec_from_file_location('_latence_atlas_v060_frontend', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def parser(project_root=None):
    runner = load_runner(project_root)
    ap = runner.parser()
    ap.description = __doc__
    ap.add_argument('--stage1-export-dir', action='append', default=[], metavar='NS=DIRECTORY')
    ap.add_argument('--input-dir', type=Path, help='New auto-generated input directory; default is OUT_inputs')
    return ap, runner


def run(args, runner):
    if args.stage1_export_dir:
        if args.prediction_config or args.complete_manifest:
            raise ValueError('--stage1-export-dir cannot be combined with --prediction-config/--complete-manifest')
        args.prediction_config = prepare_generated_inputs(args.stage1_export_dir, args.out, args.input_dir)
    elif args.input_dir:
        raise ValueError('--input-dir is only used with --stage1-export-dir')
    try:
        result = runner.run(args)
    except ValueError as exc:
        if 'required proteins missing' in str(exc):
            raise ValueError(str(exc) + '\nThe input format was recognized. A configuration file cannot fill missing rows; '
                             'prepare the four-view target union and export its missing predictions first.') from exc
        raise
    coverage_path = Path(args.out) / 'prediction_coverage.tsv'
    if args.inventory_only and coverage_path.is_file():
        print('\n[Prediction coverage: no model inference performed]')
        print(coverage_path.read_text(encoding='utf-8').rstrip())
        with coverage_path.open(encoding='utf-8', newline='') as handle:
            missing = any(int(row['missing']) > 0 for row in csv.DictReader(handle, delimiter='\t'))
        if missing:
            print('Coverage is incomplete. Existing predictions cannot be extended by a config change; '
                  'export the missing four-view targets before full evaluation.')
    return result


def main(argv=None, *, project_root=None):
    ap, runner = parser(project_root)
    args = ap.parse_args(argv)
    if len(set(args.namespaces)) != len(args.namespaces) or len(set(args.ks)) != len(args.ks) or any(k < 1 for k in args.ks):
        ap.error('Namespaces and K values must be distinct; K must be positive')
    if args.bootstrap < 0 or args.max_metric_cells < 1:
        ap.error('Bootstrap must be >=0 and memory cell limit must be positive')
    if len(set(args.stage1_models)) != len(args.stage1_models) or args.target_model in args.stage1_models:
        ap.error('Stage-1 models must be distinct and cannot include target model')
    return run(args, runner)


if __name__ == '__main__':
    main()
