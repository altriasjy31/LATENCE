#!/usr/bin/env python3
"""Score explicitly requested GO identities using a verified v093 checkpoint.

This interface does not create new GO geometry. Previously absent ontology
nodes require an anchored, supplied ontology bundle and explicit GO registry.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'nbs-v093-requested-go-prediction-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.',
                                     delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_go_ids(path):
    """Preserve explicit target order; blank/comment lines carry no targets."""
    targets = []
    for line_number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'(?:https?://purl\.obolibrary\.org/obo/)?GO[_:](\d{7})', line, re.I)
        if not match:
            raise ValueError(f'Invalid GO identity at {path}:{line_number}: {line!r}')
        targets.append('GO:' + match.group(1))
    if not targets:
        raise ValueError('The GO list must contain at least one explicit GO identity')
    if len(set(targets)) != len(targets):
        raise ValueError('Duplicate requested GO identities; keep each target once')
    return targets


def resolve_targets(registry, targets, seen_task_mask):
    """Describe target supervision status without changing the model mapping."""
    rows = registry.resolve_target_go(targets)
    mapping = np.asarray(registry.task_to_ontology, np.int64)
    seen = np.asarray(seen_task_mask, bool)
    if seen.shape != mapping.shape:
        raise ValueError('Seen task mask does not match the immutable GO columns')
    records = []
    for index, (identity, row) in enumerate(zip(targets, rows)):
        columns = np.flatnonzero(mapping == row)
        seen_columns = columns[seen[columns]]
        records.append({'output_column': index, 'requested_go_id': identity,
                        'canonical_go_id': registry.canonical_go_id(identity),
                        'ontology_row': int(row), 'task_columns': columns.tolist(),
                        'seen_task_columns': seen_columns.tolist(),
                        'stage2_supervised': bool(len(seen_columns)),
                        'readout': ('backbone_plus_shared_residual' if len(seen_columns)
                                    else 'shared_protein_GO_scorer'),
                        'zero_shot_scope': ('not_zero_shot' if len(seen_columns)
                                           else 'Stage2_unsupervised_target_only')})
    return np.asarray(rows, np.int64), records


def updated_registry(original, payload, old_rows, num_rows, registry_class):
    """Require every original GO identity to follow its anchored geometry row."""
    records = payload.get('registry_records')
    if not isinstance(records, list) or not records:
        raise ValueError('An updated ontology bundle needs explicit registry_records')
    old_rows = np.asarray(old_rows)
    if (old_rows.shape != (original.num_ontology_rows,) or old_rows.dtype.kind not in 'iu'
            or (old_rows < 0).any() or (old_rows >= num_rows).any()
            or len(np.unique(old_rows)) != len(old_rows)):
        raise ValueError('old_ontology_rows must injectively map all training ontology rows')
    new = registry_class(records, original.task_go_ids,
                         old_rows[np.asarray(original.task_to_ontology, np.int64)],
                         num_ontology_rows=num_rows)
    # GORegistryV090 has no public iterator. Its immutable canonical-row table
    # is checked in full, including obsolete GO records and non-task GO IDs.
    for identity, old_row in original._row.items():
        if new._row.get(identity) != int(old_rows[old_row]):
            raise ValueError(f'Updated GO identity does not follow its anchored row: {identity}')
        if new._obsolete.get(identity) != original._obsolete.get(identity):
            raise ValueError(f'Updated registry changes obsolete/replacement policy: {identity}')
    for alias, canonical in original._canonical.items():
        if new._canonical.get(alias) != canonical:
            raise ValueError(f'Updated registry changes an existing alias: {alias}')
    return new


def validate_checkpoint_identity(base, saved, checkpoint, config, data, ontology):
    """Use the established phase/data/code contracts, without exporting twice.

    This is the validation prefix of v090.evaluate, factored for the v093
    closed-task and requested-GO exporters. It calls the same phase, baseline,
    ontology and implementation identity helpers used by the training engine.
    """
    if saved.get('runner_version') != base.RELEASE_VERSION:
        raise ValueError('Checkpoint runner envelope differs from the installed phase engine')
    if (saved.get('model_architecture_version') != base.MODEL_VERSION
            or saved.get('data_architecture_version') != base.DATA_VERSION):
        raise ValueError('Checkpoint model/data architecture envelope differs')
    phase = base.PhasePlanV090(**saved['phase_plan'])
    fields = base._phase_fields(phase, saved['step'])
    if (saved['step'] % phase.steps_per_epoch
            or Path(checkpoint).name != f"nbs_epoch{fields['cumulative_completed_epochs']}.pt"):
        raise ValueError('Prediction requires the matching fixed completed-epoch nbs_epoch<number>.pt')
    if any(saved.get(key) != value for key, value in fields.items()):
        raise ValueError('Checkpoint phase/lifetime counters differ')
    if saved.get('training_baseline_identity') != base.training_baseline_identity(data):
        raise ValueError('Backbone probability files/clipping differ from the checkpoint')
    contract = {'data': data.data_contract(), 'ontology': base.ontology_contract(ontology)}
    if saved.get('contract') != contract:
        raise ValueError('Checkpoint data/ontology/split differs from current inputs')
    variant = saved.get('variant', saved['config']['full_task'].get('variant', 'fixed'))
    implementation = base.prediction_implementation(variant)
    trained = saved.get('training_implementation', {})
    if any(trained.get(key) != value for key, value in implementation.items()):
        raise ValueError('Prediction implementation differs from checkpoint training code')
    requested_model = asdict(base.FullTaskModelConfigV090(**config['full_task'].get('model', {})))
    if requested_model != saved.get('model_config'):
        raise ValueError('Requested model configuration differs from the checkpoint')
    if config['full_task'].get('sampler', {}) != saved['config']['full_task'].get('sampler', {}):
        raise ValueError('Requested sampler configuration differs from the checkpoint')
    revision = config['full_task'].get('experiment_revision')
    if not revision or revision != saved['config']['full_task'].get('experiment_revision'):
        raise ValueError('Experiment revision differs from checkpoint training configuration')
    return {'contract': contract, 'implementation': implementation,
            'variant': variant, 'phase_fields': fields}


def load_validated_model(base, args, config, data, device):
    import torch
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    data.ensure_prepared()
    ontology = data.ontology('cpu')
    checked = validate_checkpoint_identity(base, saved, checkpoint, config, data, ontology)
    model, _ = base.build_model(data.feature_dim, ontology, saved['model_config'],
                               checked['variant'], device, data.seen_task_mask)
    model.load_state_dict(saved['model'], strict=True)
    data.set_sampling_context(step=0, rank=0, training=False)
    model.eval()
    return {'saved': saved, 'model': model, **checked}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', type=Path,
                   default=ROOT/'outputs/latence_nbs_experiments/v093/unified_single12800')
    checkpoint = p.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument('--checkpoint', type=Path)
    checkpoint.add_argument('--epoch', type=int)
    p.add_argument('--input-dir', type=Path,
                   default=ROOT/'outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2')
    p.add_argument('--go-ids-file', type=Path, required=True)
    p.add_argument('--ontology-file', type=Path,
                   help='Optional anchored torch ontology bundle with registry_records; no automatic embedding creation')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.epoch is not None and args.epoch < 1:
        raise ValueError('--epoch must be positive')
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    work = args.work_dir.resolve(strict=True)
    args.input_dir = args.input_dir.resolve(strict=True)
    args.go_ids_file = args.go_ids_file.resolve(strict=True)
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve(strict=True)
    if args.ontology_file is not None:
        args.ontology_file = args.ontology_file.resolve(strict=True)
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    os.chdir(ROOT)
    config_path = work/'resolved_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    args.checkpoint = args.checkpoint or work/f'nbs_epoch{args.epoch}.pt'
    targets = read_go_ids(args.go_ids_file)
    sys.path[:0] = [str(ROOT), str(ROOT/'nbs_models/nbs_protein_go')]
    import torch
    from scripts.nbs.train_nbs_full_task_v093 import install
    base = install()
    device = torch.device(args.device)
    data = base.FullTaskData(config, stores=base.build_latence_nbs_stores(config))
    checked = load_validated_model(base, args, config, data, device)
    model, saved = checked['model'], checked['saved']
    registry = data.stores.go_registry_v090
    extension = None
    if args.ontology_file is not None:
        from nbs_pg.go_registry_v090 import GORegistryV090
        payload = torch.load(args.ontology_file.resolve(strict=True), map_location='cpu', weights_only=False)
        if not isinstance(payload, dict) or 'ontology' not in payload or 'old_ontology_rows' not in payload:
            raise ValueError('Ontology bundle must contain ontology and old_ontology_rows')
        update = payload['ontology']
        if not isinstance(update, dict) or 'center' not in update:
            raise ValueError('Ontology bundle ontology requires supplied center/offset/edges/task_to_ontology')
        old_rows = np.asarray(payload['old_ontology_rows'])
        registry = updated_registry(registry, payload, old_rows, len(update['center']), GORegistryV090)
        extension = model.replace_ontology_for_inference(update, old_ontology_rows=old_rows)
    target_rows, target_records = resolve_targets(registry, targets, data.seen_task_mask)
    inp = args.input_dir.resolve(strict=True)
    protein_ids = (inp/'protein_ids.txt').read_text(encoding='utf-8').splitlines()
    if not protein_ids or any(not identity.strip() for identity in protein_ids) or len(set(protein_ids)) != len(protein_ids):
        raise ValueError('Prediction protein IDs must be nonempty and unique')
    rows = len(np.load(inp/'ind_test_repr.f16.npy', mmap_mode='r', allow_pickle=False))
    if len(protein_ids) != rows:
        raise ValueError('Protein IDs and inference feature rows differ')
    # This existing reader validates source arrays against their input manifest.
    data.inference_batch(inp, np.arange(min(1, rows)), device)
    batch_size = args.batch_size or config['full_task'].get('eval_batch', 16)
    identity = {'checkpoint_sha256': sha256(args.checkpoint),
                'resolved_config_sha256': sha256(config_path),
                'input_manifest_sha256': sha256(inp/'ind_test_input_manifest.json'),
                'protein_ids_sha256': sha256(inp/'protein_ids.txt'),
                'requested_go_ids': targets, 'target_ontology_rows': target_rows.tolist(),
                'GO_registry_identity_sha256': registry.identity_sha256,
                'inference_batch_size': batch_size,
                'ontology_bundle_sha256': sha256(args.ontology_file) if args.ontology_file else None,
                'prediction_implementation': checked['implementation'],
                'exporter_sha256': sha256(__file__), 'contract': checked['contract']}
    tag = hashlib.sha256(json.dumps(targets, separators=(',', ':')).encode()).hexdigest()[:12]
    out = args.output_dir.resolve() if args.output_dir else work/f"predict_go_epoch{saved['epoch']}"/tag
    out.mkdir(parents=True, exist_ok=True)
    probability_path, manifest_path = out/'probabilities.f32.npy', out/'prediction_manifest.json'
    with (out/'prediction.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Requested-GO output is in use by another predictor') from None
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if (previous.get('identity') != identity or not probability_path.is_file()
                    or previous.get('output_probability_sha256') != sha256(probability_path)):
                raise FileExistsError('Existing requested-GO output differs; use a new --output-dir')
            print(f'[predict GO] verified existing output: {manifest_path}', flush=True)
            return manifest_path
        if probability_path.exists():
            raise FileExistsError('Incomplete output without manifest; use a new --output-dir')
        descriptor, temporary_name = tempfile.mkstemp(prefix='.probabilities.', suffix='.npy', dir=out)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            probability = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                                   shape=(rows, len(targets)))
            with torch.no_grad():
                full_encoding = model.encode_ontology()
                for start in range(0, rows, batch_size):
                    batch = data.inference_batch(inp, np.arange(start, min(rows, start+batch_size)), device)
                    values = model(batch, target_ontology_ids=target_rows,
                                   ontology_encoding=full_encoding).sigmoid().cpu().numpy()
                    if values.shape != (min(rows-start, batch_size), len(targets)) or not np.isfinite(values).all():
                        raise FloatingPointError('Invalid requested-GO probability output')
                    probability[start:start+len(values)] = values
                    print(f'[predict GO] {start+len(values)}/{rows}', flush=True)
            probability.flush()
            del probability
            os.replace(temporary, probability_path)
        finally:
            temporary.unlink(missing_ok=True)
        (out/'protein_ids.txt').write_text('\n'.join(protein_ids)+'\n', encoding='utf-8')
        with (out/'target_go.tsv').open('w', encoding='utf-8', newline='') as stream:
            keys = ['output_column', 'requested_go_id', 'canonical_go_id', 'ontology_row',
                    'task_columns', 'seen_task_columns', 'stage2_supervised', 'readout', 'zero_shot_scope']
            writer = csv.DictWriter(stream, fieldnames=keys, delimiter='\t')
            writer.writeheader()
            writer.writerows(target_records)
        manifest = {'schema': SCHEMA, 'complete': True, 'identity': identity,
                    'checkpoint_epoch': saved['epoch'], 'checkpoint_step': saved['step'],
                    'num_proteins': rows, 'num_target_GO': len(targets), 'targets': target_records,
                    'probability_file': probability_path.name,
                    'output_probability_sha256': sha256(probability_path),
                    'protein_ids_file': 'protein_ids.txt', 'target_go_file': 'target_go.tsv',
                    'ontology_extension': extension,
                    'stage1_exposure_audited': False, 'strict_pipeline_zero_shot_claimed': False,
                    'unseen_target_readout': 'shared_protein_GO_scorer_without_backbone_class_row',
                    'new_GO_geometry_created': False,
                    'limitations': ['Target scoring capability is not proof of biological generalization.',
                                    'Previously absent GO nodes require supplied anchored geometry and an identity registry.',
                                    'Updated ontology relations can change existing node encodings even when old coordinates are frozen.']}
        write_json(manifest_path, manifest)
    print(f'[predict GO] {manifest_path}', flush=True)
    return manifest_path


if __name__ == '__main__':
    main()
