#!/usr/bin/env python3
"""Additive, read-only provenance adapter for NBS v0.9.4 exports.

The only native code invoked is the UNMODIFIED train_nbs_full_task_v094.py
--stage export; this file never imports/patches that worker or changes its
training/metrics/checkpoint files. The adapted output is an independent JSON.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

WIDTH = {'bp': 21312, 'mf': 7038, 'cc': 2903}
SCHEMA = 'latence-dark-weak-v094-native-adapter-v1'
NATIVE_SCHEMA = 'nbs-v094-unified-prediction-v1'
REVISION = '0.9.4-weak-teacher-r1'
FORBIDDEN_INPUTS = (
    'ind_test_labels_in_representation',
    'ind_test_labels_in_candidate_selection',
    'ind_test_labels_in_pp_retrieval',
    'expert_probability_in_forward_inputs',
    'test_to_test_edges',
)
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_ids(path: Path) -> list[str]:
    values = [v.strip() for v in Path(path).read_text(encoding='utf-8').splitlines() if v.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f'IDs must be nonempty and distinct: {path}')
    return values


def json_file(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise TypeError(f'Expected JSON object: {path}')
    return value


def atomic_json(path: Path, payload: dict, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Existing adapter not overwritten: {path}; use --overwrite-adapter')
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', encoding='utf-8', delete=False) as f:
        tmp = Path(f.name)
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def resolve_registry(path_string: str, input_dir: Path) -> Path:
    p = Path(path_string)
    return p if p.is_absolute() else (input_dir / p)


def validate_config(config: dict, namespace: str) -> None:
    if (config.get('release_version') != '0.9.4'
            or config.get('model_version') != '0.9.4'
            or config.get('stage', {}).get('name') != 'nbs_v094_weak_teacher'
            or config.get('task') != namespace):
        raise ValueError('Requires the native v094 weak-teacher config for the requested task')
    ft = config.get('full_task', {})
    model = ft.get('model', {})
    if (ft.get('experiment_revision') != REVISION
            or ft.get('weak_target_mode') != 'backbone_binary_plus_modelout_dense_soft'
            or ft.get('architecture_extension') != 'v094_single_shared_head_raw_geometry_dense_teacher'
            or model.get('output_strategy') != 'shared_direct'
            or model.get('prediction_mode') != 'shared_bilinear'):
        raise ValueError('Wrong v094 supervision/scorer identity; do not run the v088/v089 adapter')
    if config.get('v094_experiment', {}).get('strict_pipeline_zero_shot') is not False:
        raise ValueError('v094 zero-shot scope must not be asserted as strict pipeline zero-shot')


def _array(path: Path, shape, kind=None) -> np.ndarray:
    arr = np.load(path, mmap_mode='r', allow_pickle=False)
    if arr.ndim != len(shape) or any(s is not None and int(got) != s for got, s in zip(arr.shape, shape)):
        raise ValueError(f'{path}: shape {arr.shape}, expected {shape}')
    if kind is not None and arr.dtype.kind not in kind:
        raise TypeError(f'{path}: dtype {arr.dtype} incompatible with {kind}')
    return arr


def validate_input(config: dict, args) -> dict:
    directory = args.input_dir.resolve(strict=True)
    expected = read_ids(args.expected_protein_ids)
    observed = read_ids(directory / 'protein_ids.txt')
    if expected != observed:
        raise ValueError('Prepared input protein IDs/order do not equal the frozen Dark–Weak target list')
    names = ('ind_test_repr.f16.npy', 'backbone_ind_test_prob.f16.npy',
             'candidate_go_index.i32.npy', 'candidate_edge_attr.f32.npy',
             'test_core_neighbors.i32.npy', 'test_core_edge_attr.f32.npy',
             'ind_test_input_manifest.json')
    for name in names:
        if not (directory / name).is_file():
            raise FileNotFoundError(f'Native v094 input is missing {directory/name}')
    manifest_path = directory / 'ind_test_input_manifest.json'
    manifest = json_file(manifest_path)
    if manifest.get('task') != args.task or manifest.get('num_proteins') != len(expected):
        raise ValueError('Prepared input task/sample-count differs')
    if manifest.get('mode') != args.expected_mode:
        raise ValueError(f'Input mode must be {args.expected_mode!r}, got {manifest.get("mode")!r}')
    leakage = manifest.get('leakage_contract')
    if not isinstance(leakage, dict) or any(leakage.get(k) is not False for k in FORBIDDEN_INPUTS):
        raise ValueError('Input manifest lacks required explicit negative leakage declarations')
    if manifest.get('protein_ids_file_sha256') != sha256(directory / 'protein_ids.txt'):
        raise ValueError('Input manifest protein-ID file hash mismatch')
    n, g = len(expected), WIDTH[args.task]
    repr_values = _array(directory/'ind_test_repr.f16.npy', (n, None), 'f')
    _array(directory/'backbone_ind_test_prob.f16.npy', (n, g), 'f')
    selector = manifest.get('candidate_evidence', {})
    k = int(selector.get('fixed_k', -1))
    required_pool = min(g, int(config['stage'].get('protein_go_candidate_topk', 512)))
    if selector.get('selector_scope') != 'full_task' or k < required_pool:
        raise ValueError('v094 requires an untruncated full-task candidate pool (typically top512)')
    if (selector.get('expert_probability_used') is not False
            or selector.get('label_hint_used') is not False):
        raise ValueError('Candidate evidence must explicitly exclude Expert probabilities and GO label hints')
    _array(directory/'candidate_go_index.i32.npy', (n, k), 'iu')
    _array(directory/'candidate_edge_attr.f32.npy', (n, k, 3), 'f')
    pp = manifest.get('external_pp', {})
    pk = int(manifest.get('cache_signature', {}).get('pp_topk', -1))
    if pk < 1:
        raise ValueError('Invalid external-core neighbor budget')
    _array(directory/'test_core_neighbors.i32.npy', (n, pk), 'iu')
    _array(directory/'test_core_edge_attr.f32.npy', (n, pk, 3), 'f')
    evidence = {
        'representation': ('ind_test_repr.f16.npy', manifest.get('representation', {}).get('sha256')),
        'base': ('backbone_ind_test_prob.f16.npy', manifest.get('base_probability', {}).get('sha256')),
        'go': ('candidate_go_index.i32.npy', selector.get('go_index_sha256')),
        'attr': ('candidate_edge_attr.f32.npy', selector.get('edge_attr_sha256')),
        'pp': ('test_core_neighbors.i32.npy', pp.get('neighbor_index_sha256')),
        'pp_attr': ('test_core_edge_attr.f32.npy', pp.get('edge_attr_sha256')),
    }
    for kind, (filename, declared_sha) in evidence.items():
        if not isinstance(declared_sha, str) or sha256(directory/filename) != declared_sha:
            raise ValueError(f'Native input {kind} hash missing or differs from manifest')
    regval = manifest.get('registries', {}).get('go_registry')
    if not regval:
        raise ValueError('Prepared input missing GO registry')
    registry = resolve_registry(regval, directory).resolve(strict=True)
    with registry.open(encoding='utf-8', newline='') as f:
        rows = sorted(csv.DictReader(f, delimiter='\t'), key=lambda r: int(r['go_idx']))
    if [int(r['go_idx']) for r in rows] != list(range(g)):
        raise ValueError('GO registry must preserve 0..G-1 original column positions')
    if [r['input_go_id'] for r in rows] != read_ids(args.go_ids):
        raise ValueError('Frozen model GO ID order differs from supplied --go-ids')
    expected_registry_sha = manifest.get('registries', {}).get('go_registry_sha256')
    if expected_registry_sha != sha256(registry):
        raise ValueError('GO registry hash differs from prepared manifest')
    return {'num_proteins': n, 'num_task_go': g, 'feature_dim': repr_values.shape[1],
            'candidate_pool_size': k, 'external_core_neighbors': pk,
            'input_dir': str(directory), 'input_manifest': str(manifest_path.resolve()),
            'input_manifest_sha256': sha256(manifest_path),
            'protein_ids': str((directory/'protein_ids.txt').resolve()),
            'protein_ids_sha256': sha256(directory/'protein_ids.txt'),
            'go_registry': str(registry), 'go_registry_sha256': sha256(registry),
            'inference_mode': manifest.get('mode'),
            'leakage_contract': leakage,
            'msa_source_cutoff_verified': False}


def verify_graph_audit(path: Path | None, prepared: dict, allow_self_support=False) -> dict:
    if path is None:
        return {'status': 'not_run', 'strict_inductive_certified': False}
    p = Path(path).resolve(strict=True)
    report = json_file(p)
    hashes = report.get('inputs_sha256', {})
    if hashes.get('manifest') != prepared['input_manifest_sha256'] or hashes.get('protein_ids') != prepared['protein_ids_sha256']:
        raise ValueError('Graph audit belongs to different prepared inputs')
    if report.get('model_architecture_version') != '0.9.4' or report.get('runner_version') != '0.9.4':
        raise ValueError('Graph audit is not from the native v094 model/data')
    if report.get('leaked_label_fields') or report.get('query_edges_noncosine'):
        raise ValueError('Graph audit detected label or query graph contract failure')
    if report.get('batch_self_support_hits', 0) and not allow_self_support:
        raise ValueError('The original target protein appears in its ghost-query batch support; use --allow-self-support only for explicitly transductive/descriptive analysis')
    return {'status': report.get('self_support_status', 'unknown'),
            'strict_inductive_certified': False,
            'audited_rows': report.get('audited'),
            'original_global_ids_provided': report.get('original_global_ids_provided'),
            'batch_self_support_hits': report.get('batch_self_support_hits'),
            'audit_path': str(p), 'audit_sha256': sha256(p)}


def locate_native_prediction(work_dir: Path, checkpoint_sha: str, input_sha: str) -> tuple[Path, Path, dict]:
    found = []
    for manifest_path in work_dir.glob('eval_step*/full/nbs_full_task_prediction_manifest.json'):
        manifest = json_file(manifest_path)
        if (manifest.get('checkpoint_sha256') == checkpoint_sha
                and manifest.get('input_manifest_sha256') == input_sha):
            found.append((manifest_path, manifest))
    if len(found) != 1:
        raise ValueError(f'Expected exactly one native v094 prediction matching checkpoint/input hashes, found {len(found)} under {work_dir}')
    manifest_path, manifest = found[0]
    prob_path = manifest_path.parent/'nbs_ind_test_prob.f32.npy'
    return manifest_path, prob_path, manifest


def verify_native_prediction(manifest: dict, probability_path: Path, args, config: dict,
                             prepared: dict, checkpoint_sha: str) -> dict:
    if (manifest.get('schema') != NATIVE_SCHEMA or manifest.get('complete') is not True
            or manifest.get('experiment_revision') != REVISION
            or manifest.get('task') != args.task
            or manifest.get('checkpoint_sha256') != checkpoint_sha
            or manifest.get('input_manifest_sha256') != prepared['input_manifest_sha256']
            or manifest.get('num_task_go') != prepared['num_task_go']):
        raise ValueError('Native v094 prediction schema/task/checkpoint/input/GO identity mismatch')
    if (manifest.get('output_fusion') != 'single_shared_GO_scorer'
            or manifest.get('uses_dense_backbone_logits_in_forward') is not False
            or manifest.get('uses_expert_probability_in_nbs_forward') is not False
            or manifest.get('zero_shot_scope') != 'Stage2-only; reused upstream representations may encode held-out GO'):
        raise ValueError('Native v094 scorer or zero-shot semantics changed; review the new version')
    if manifest.get('model_config') != config['full_task']['model']:
        # Model state may serialize normalized dataclass defaults; its validation
        # belongs to the original native exporter, not this overlay.
        if not isinstance(manifest.get('model_config'), dict):
            raise ValueError('Native prediction is missing its trained model config')
    if not isinstance(manifest.get('prediction_implementation'), dict) or not manifest['prediction_implementation']:
        raise ValueError('Missing trained-code digests')
    if not Path(probability_path).is_file() or sha256(probability_path) != manifest.get('output_probability_sha256'):
        raise ValueError('Native v094 probability bytes changed or missing')
    a = _array(probability_path, (prepared['num_proteins'], prepared['num_task_go']), 'f')
    if a.dtype != np.float32:
        raise ValueError('Expected original float32 native v094 predictions')
    for start in range(0, len(a), 64):
        block = np.asarray(a[start:start+64])
        if not np.isfinite(block).all() or ((block < 0) | (block > 1)).any():
            raise ValueError('Native output must be finite probabilities in [0,1]')
    if type(manifest.get('epoch')) is not int or manifest['epoch'] < 1 or type(manifest.get('step')) is not int or manifest['step'] < 1:
        raise ValueError('Native v094 epoch/step not recorded')
    if args.expected_epoch is not None and manifest['epoch'] != args.expected_epoch:
        raise ValueError('Native prediction epoch does not match requested cumulative epoch')
    return {'epoch': manifest['epoch'], 'step': manifest['step'],
            'prediction_sha256': manifest['output_probability_sha256']}


def make_adapter(args, execute_native: bool) -> dict:
    config_path, checkpoint, work_dir = (p.resolve(strict=True) if name != 'work_dir' else p.resolve()
                                         for p, name in ((args.config, 'config'), (args.checkpoint, 'checkpoint'), (args.work_dir, 'work_dir')))
    config = json_file(config_path)
    validate_config(config, args.task)
    prepared = validate_input(config, args)
    if any((work_dir/n).exists() for n in ('training_history.json', 'latest.pt')) or list(work_dir.glob('nbs_epoch*.pt')):
        raise ValueError('Refusing to place Dark–Weak evaluation artifacts in a model training directory')
    graph = verify_graph_audit(args.graph_audit_json, prepared, args.allow_self_support)
    checkpoint_sha = sha256(checkpoint)
    runner = args.train_runner.resolve(strict=True)
    if execute_native:
        command = [sys.executable, str(runner), '--stage', 'export', '--config', str(config_path),
                   '--checkpoint', str(checkpoint), '--work-dir', str(work_dir),
                   '--input-dir', str(Path(args.input_dir).resolve()), '--device', args.device]
        print('[v094 native export]', ' '.join(command), flush=True)
        lockfile = ROOT / 'outputs/latence_nbs_experiments/v092/v092_evaluate.lock'
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        with lockfile.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'A v092/v094 evaluation is already using the shared inference cache: {lockfile}') from None
            subprocess.run(command, cwd=ROOT, check=True)
    manifest_path, prob_path, manifest = locate_native_prediction(work_dir, checkpoint_sha, prepared['input_manifest_sha256'])
    result = verify_native_prediction(manifest, prob_path, args, config, prepared, checkpoint_sha)
    adapter = {
        'schema': SCHEMA, 'model': 'NBS_v094', 'runner_version': '0.9.4',
        'model_architecture_version': '0.9.4', 'data_architecture_version': '0.9.4',
        'experiment_revision': REVISION, 'task': args.task,
        'probability': str(prob_path.resolve()), 'probability_sha256': result['prediction_sha256'],
        'protein_ids': prepared['protein_ids'], 'protein_ids_sha256': prepared['protein_ids_sha256'],
        'go_ids': str(args.go_ids.resolve()), 'go_ids_sha256': sha256(args.go_ids),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
        'config': str(config_path), 'config_sha256': sha256(config_path),
        'native_prediction_manifest': str(manifest_path.resolve()),
        'native_prediction_manifest_sha256': sha256(manifest_path),
        'prepared_input_manifest': prepared['input_manifest'],
        'prepared_input_manifest_sha256': prepared['input_manifest_sha256'],
        'prepared': prepared, 'graph_audit': graph,
        'epoch': result['epoch'], 'step': result['step'],
        'prediction_space': 'complete_task_classifier_columns',
        'output_fusion': 'single_shared_GO_scorer',
        'modelout_use': 'dense_teacher_training_loss_only_not_model_forward',
        'zero_shot_scope': 'stage2_only; no full-pipeline zero-shot claim',
        'evaluation_protocol': 'historical_seen_weak_ghost_or_external_as_audited',
    }
    atomic_json(args.out, adapter, overwrite=args.overwrite_adapter)
    print(f'[v094 adapter] {args.out}; N={prepared["num_proteins"]}, G={prepared["num_task_go"]}; '
          f'epoch={result["epoch"]}, step={result["step"]}', flush=True)
    return adapter


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('export', 'attach'), help='attach validates existing native export without running a GPU forward')
    p.add_argument('--task', required=True, choices=tuple(WIDTH))
    p.add_argument('--config', type=Path, required=True, help='A copy of v094 resolved_config.json or derived training_config.json')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--expected-protein-ids', type=Path, required=True)
    p.add_argument('--go-ids', type=Path, required=True, help='Frozen original task classifier GO IDs in column order')
    p.add_argument('--expected-mode', default='dark_weak_inference')
    p.add_argument('--expected-epoch', type=int)
    p.add_argument('--work-dir', type=Path, required=True, help='Separate NEW directory; NEVER the actual training work dir')
    p.add_argument('--train-runner', type=Path, default=ROOT/'scripts/nbs/train_nbs_full_task_v094.py')
    p.add_argument('--graph-audit-json', type=Path)
    p.add_argument('--allow-self-support', action='store_true', help='Descriptive transductive audit only; no inductive claim')
    p.add_argument('--overwrite-adapter', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    make_adapter(a, execute_native=a.action == 'export')


if __name__ == '__main__':
    main()
