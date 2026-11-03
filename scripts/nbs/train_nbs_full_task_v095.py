#!/usr/bin/env python3
"""ASL, selected teacher supervision and hierarchy on the heterogeneous encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'nbs_models/nbs_protein_go')]
REVISION = '0.9.5-asl-hierarchy-r1'


def validate_initialization(base, config, data, source):
    """Verify a fixed v094 checkpoint before any new-run output is written."""
    import torch
    path = Path(source['checkpoint']).resolve(strict=True)
    if sha256(path) != source['sha256']:
        raise ValueError('Initialization checkpoint changed after configuration was frozen')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if sha256(path) != source['sha256']:
        raise ValueError('Initialization checkpoint changed while reading')
    if (saved.get('runner_version') != '0.9.4'
            or saved.get('model_architecture_version') != '0.9.4'
            or saved.get('data_architecture_version') != '0.9.4'):
        raise ValueError('--init-from requires a fixed v094 checkpoint')
    phase = base.PhasePlanV090(**saved['phase_plan'])
    fields = base._phase_fields(phase, saved['step'])
    if (saved['step'] % phase.steps_per_epoch
            or path.name != f"nbs_epoch{fields['cumulative_completed_epochs']}.pt"
            or any(saved.get(k) != v for k, v in fields.items())):
        raise ValueError('Initialization requires matching fixed completed-epoch counters')
    ontology = data.ontology('cpu')
    contract = {'data': data.data_contract(), 'ontology': base.ontology_contract(ontology)}
    if saved.get('contract') != contract:
        raise ValueError('Initialization data/ontology/seen split differs from this run')
    if saved.get('training_baseline_identity') != base.training_baseline_identity(data):
        raise ValueError('Initialization backbone input identity differs')
    manifest_sha = sha256(config['data']['unified_dataset_manifest'])
    source_experiment = saved['config'].get('v094_experiment', {})
    if source_experiment.get('unified_dataset_manifest_sha256') != manifest_sha:
        raise ValueError('Initialization requires the identical v094 dataset manifest')
    requested = asdict(base.FullTaskModelConfigV090(**config['full_task']['model']))
    previous = saved.get('model_config', {})
    if {k: v for k, v in requested.items() if k != 'jk_mode'} != {
            k: v for k, v in previous.items() if k != 'jk_mode'}:
        raise ValueError('Initialization model options may differ only in jk_mode')
    return saved


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_experiment(config):
    ft = config.get('full_task', {})
    if ft.get('experiment_revision') != REVISION:
        raise ValueError('Use a generated v095 unified training_config.json')
    if config.get('release_version') != '0.9.5' or config.get('model_version') != '0.9.5':
        raise ValueError('Unified checkpoints use their own v095 model/data identity')
    if ft.get('expected_world_size') != 1 or type(ft.get('epochs')) is not int or ft['epochs'] < 5:
        raise ValueError('v095 requires one GPU and a declared fixed horizon >=5')
    if not config.get('data', {}).get('unified_dataset_manifest'):
        raise ValueError('Unified weak dataset is required before training')
    if ft.get('model', {}).get('output_strategy') != 'shared_direct':
        raise ValueError('v095 uses a single shared GO scorer without a backbone logit skip')
    return config


def install():
    from scripts.nbs import train_nbs_full_task_v090 as base
    identity = str(Path(__file__).resolve())
    previous = getattr(base, '_nbs_v095_install_identity', None)
    if previous is not None:
        if previous != identity:
            raise RuntimeError('A different unified adapter is already installed')
        return base
    if any(getattr(base, key, None) is not None for key in ('_nbs_v092_install_identity', '_nbs_v093_install_identity', '_nbs_v094_install_identity')):
        raise RuntimeError('Run v092, v093 and v095 workers in separate processes')
    from nbs_pg.full_task_data_v094 import FullTaskDataV094, build_latence_nbs_stores_v094
    from nbs_pg.full_task_loss_v095 import FullTaskLossConfigV095, training_objective
    from nbs_pg.full_task_model_v095 import FullTaskGraphModelV095, FullTaskModelConfigV095
    from nbs_pg import full_task_sampler_fast as fast

    # Keep the original counter/scheduler implementation, while assigning an
    # honest new checkpoint identity to the new model and derived data.
    base.RELEASE_VERSION = '0.9.5'
    base.MODEL_VERSION = '0.9.5'
    base.DATA_VERSION = '0.9.4'
    original_resume_contract = base.resume_contract

    def resume_contract(*args, **kwargs):
        result = original_resume_contract(*args, **kwargs)
        result.update(runner_version='0.9.5', model_version='0.9.5', experiment_revision=REVISION)
        cfg = args[0] if args else kwargs['full_task']
        result['v095_training_runtime'] = {
            key: cfg.get(key) for key in ('optimizer', 'precision', 'initialization')}
        return result

    base.resume_contract = resume_contract

    base.FullTaskData = FullTaskDataV094
    base.build_latence_nbs_stores = build_latence_nbs_stores_v094
    base.FullTaskModelConfigV090 = FullTaskModelConfigV095
    base.FullTaskLossConfigV090 = FullTaskLossConfigV095
    diagnostic_window = {}
    initialization_context = {}

    def objective_with_diagnostics(*args, **kwargs):
        import torch
        batch = args[1]
        device = batch['targets'].device
        enabled = device.type == 'cuda' and getattr(base, '_v095_precision', 'bf16') == 'bf16'
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
            logits, loss, parts = training_objective(*args, **kwargs)
        for name in ('main_binary', 'hard_ASL', 'modelout_soft_KL', 'hierarchy',
                     'contrib_modelout_soft', 'contrib_hierarchy', 'unscaled_loss',
                     'scaled_hard_ASL', 'contrib_teacher_positive', 'contrib_teacher_negative',
                     'teacher_selected_pairs_per_weak', 'hard_negative_selected_pairs_per_weak'):
            value = parts[name].detach()
            diagnostic_window[name] = diagnostic_window.get(name, 0) + value
        return logits, loss, parts

    base.training_objective = objective_with_diagnostics
    original_implementation = base.prediction_implementation

    def prediction_implementation(variant):
        result = dict(original_implementation(variant))
        result['full_task_sampler_fast.py'] = fast.implementation_sha256()
        for filename in ('go_registry_v093.py', 'unified_weak_dataset_v093.py',
                         'weak_dataset_v094.py', 'full_task_data_v094.py', 'optimizer_v095.py',
                         'full_task_model_v092.py', 'full_task_model_v093.py',
                         'full_task_model_v094.py', 'full_task_model_v095.py', 'full_task_loss_v095.py'):
            result[filename] = sha256(ROOT / 'nbs_models/nbs_protein_go/nbs_pg' / filename)
        result['train_nbs_full_task_v095.py'] = sha256(__file__)
        return result

    def build_model(feature_dim, ontology, model_options, variant, device, seen_task_mask=None):
        if variant not in ('fixed', 'dynamic'):
            raise ValueError('Unknown sampling variant')
        import numpy as np
        cfg = FullTaskModelConfigV095(**model_options)
        seen = None if seen_task_mask is None else np.array(seen_task_mask, dtype=bool, copy=True)
        model = FullTaskGraphModelV095(feature_dim, ontology, cfg, seen_task_mask=seen).to(device)
        model.training_variant = variant
        saved = initialization_context.get('saved')
        if saved is not None:
            import torch
            current = model.state_dict()
            for key in ('seen_task_mask',):
                if key not in saved['model'] or not torch.equal(current[key].cpu(), saved['model'][key].cpu()):
                    raise ValueError('Initialization changes immutable model axis: ' + key)
            # task_to_ontology is intentionally a nonpersistent buffer. Bind it
            # through the source data contract rather than expecting a state key.
            mapping_hash = hashlib.sha256(model.task_to_ontology.detach().cpu().numpy().astype('<i8').tobytes()).hexdigest()
            if saved['contract']['data'].get('task_to_ontology_sha256') != mapping_hash:
                raise ValueError('Initialization task-to-ontology mapping differs')
            model.load_state_dict(saved['model'], strict=True)
            initialization_context['loaded'] = True
        return model, cfg

    base.prediction_implementation = prediction_implementation
    base.build_model = build_model
    original_write = base.write_json

    original_train = base.train

    def train(args, config, data, device, rank, world):
        source = config['full_task'].get('initialization', {})
        try:
            if not args.resume and source.get('mode') == 'v094_weights_only':
                initialization_context['saved'] = validate_initialization(base, config, data, source)
                initialization_context['source'] = source
            return original_train(args, config, data, device, rank, world)
        finally:
            initialization_context.clear()

    base.train = train

    def write_json(path, value):
        if initialization_context.get('loaded'):
            lineage = (value.get('runtime', {}).get('phase_origin')
                       if Path(path).name == 'resolved_config.json' else
                       value.get('phase_origin') if Path(path).name == 'phase_manifest.json' else None)
            if lineage is not None:
                source, saved = initialization_context['source'], initialization_context['saved']
                lineage.update(initialization='v094_weights_only_new_objective',
                               source_checkpoint=source['checkpoint'],
                               source_checkpoint_sha256=source['sha256'], source_runner_version='0.9.4',
                               source_completed_epochs=saved['epoch'], source_phase_step=saved['step'],
                               source_world_size=saved['world_size'],
                               source_optimizer_state_reused=False, source_rng_state_reused=False)
        if Path(path).name == 'training_history.json' and value and diagnostic_window:
            updates = int(value[-1]['window_updates'])
            value[-1].update({name: float(total.cpu()) / updates
                              for name, total in diagnostic_window.items()})
            value[-1]['loss_definition'] = 'scaled_ASL + selected_confidence_modelout_KL + seen_GO_hierarchy'
            diagnostic_window.clear()
        if Path(path).name == 'resolved_config.json':
            value.setdefault('runtime', {}).update(
                experiment_revision=REVISION,
                training_control='core full ASL; weak B-positive ASL; selected confidence teacher KL; seen GO hierarchy; single shared GO scorer',
                data_contract_supervision_fields='inherited v094 source provenance; optimization is defined by the v095 loss and frozen full_task configuration',
                backbone_preservation='none; backbone probability is not a logit skip or rare-retention target',
                zero_shot_scope='Stage2 supervision exclusion and shared target scoring; upstream Stage1 provenance not erased')
        return original_write(path, value)

    base.write_json = write_json
    base.evaluate = evaluate
    base._nbs_v095_install_identity = identity
    return base


def evaluate(args, config, data, device):
    """Export the single shared GO scorer with unchanged exact metrics."""
    import numpy as np
    import torch
    from scripts.nbs.predict_nbs_go_v095 import load_validated_model
    base = install()
    checked = load_validated_model(base, args, config, data, device)
    saved, model = checked['saved'], checked['model']
    inp = Path(args.input_dir).resolve()
    input_manifest = inp / 'ind_test_input_manifest.json'
    rows = len(np.load(inp / 'ind_test_repr.f16.npy', mmap_mode='r', allow_pickle=False))
    destination = Path(args.work_dir).resolve() / f"eval_step{saved['step']}" / 'full'
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'nbs_ind_test_prob.f32.npy'
    manifest_path = destination / 'nbs_full_task_prediction_manifest.json'
    request = dict(checkpoint_sha256=sha256(args.checkpoint),
                   input_manifest_sha256=sha256(input_manifest),
                   prediction_implementation=checked['implementation'], contract=checked['contract'],
                   experiment_revision=REVISION, model_config=saved['model_config'])
    reuse = False
    if path.exists() or manifest_path.exists():
        if not path.is_file() or not manifest_path.is_file():
            raise FileExistsError('Incomplete evaluation output; use another --output-dir or remove only this incomplete evaluation')
        recorded = json.loads(manifest_path.read_text())
        if any(recorded.get(k) != v for k, v in request.items()) or recorded.get('output_probability_sha256') != sha256(path):
            raise FileExistsError('Existing evaluation differs from this checkpoint/data/code')
        reuse = True
    data.set_sampling_context(0, 0, False)
    model.eval()
    with torch.no_grad():
        go = model.encode_go()
        if not reuse:
            temp = path.with_name(path.name + '.partial.npy')
            values = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32,
                                              shape=(rows, data.num_task_go))
            try:
                size = int(config['full_task'].get('eval_batch', 16))
                for start in range(0, rows, size):
                    batch = data.inference_batch(inp, np.arange(start, min(start + size, rows)), device)
                    p = model(batch, go_encoding=go).sigmoid().cpu().numpy()
                    if not np.isfinite(p).all():
                        raise FloatingPointError('Nonfinite unified predictions')
                    values[start:start + len(p)] = p
                    print(f'[v095 predict] {start + len(p)}/{rows}', flush=True)
                values.flush()
                del values
                os.replace(temp, path)
            except BaseException:
                if 'values' in locals():
                    del values
                temp.unlink(missing_ok=True)
                raise
        else:
            data.inference_batch(inp, np.arange(min(1, rows)), device)
    manifest = dict(request, schema='nbs-v095-unified-prediction-v1', complete=True,
                    checkpoint_path=str(Path(args.checkpoint).resolve()), probability_path=str(path),
                    output_probability_sha256=sha256(path), step=saved['step'], epoch=saved['epoch'],
                    num_task_go=data.num_task_go, task=config['task'],
                    uses_dense_backbone_logits_in_forward=False,
                    uses_expert_probability_in_nbs_forward=False,
                    output_fusion='single_shared_GO_scorer',
                    zero_shot_scope='Stage2-only; reused upstream representations may encode held-out GO',
                    acceptance_protocol='same exact protein Fmax and AP; backbone is a comparison reference only')
    base.write_json(manifest_path, manifest)
    if args.stage == 'evaluate':
        from nbs_pg.evaluation_report_v095 import write_extended_report
        output = write_extended_report(args, config, data)
        print(f'[v095 evaluation] {output}', flush=True)
    return manifest_path


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if any(x.split('=', 1)[0] in ('--encoder-from', '--init-from') for x in raw):
        raise ValueError('The unified experiment starts fresh; only same-run resume is supported')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--stage', required=True, choices=('prepare', 'train', 'evaluate', 'export'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--metadata-file', type=Path)
    known, _ = parser.parse_known_args(raw)
    os.chdir(ROOT)
    config = validate_experiment(json.loads(known.config.read_text()))
    base = install()
    from nbs_pg.optimizer_v095 import configure_training_runtime
    configure_training_runtime(base, config)
    base._v095_precision = config['full_task'].get('precision', 'bf16')
    if known.device.startswith('cuda') and base._v095_precision == 'bf16':
        import torch
        torch.cuda.set_device(torch.device(known.device))
        if not torch.cuda.is_bf16_supported():
            raise ValueError('This GPU does not support BF16; create a new run with --precision fp32')
    if known.stage in ('evaluate', 'export'):
        if not known.checkpoint or not known.input_dir:
            raise ValueError('Evaluation needs --checkpoint and --input-dir')
        if known.stage == 'evaluate' and not known.metadata_file:
            raise ValueError('Evaluation needs --metadata-file')
        import torch
        device = torch.device(known.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        torch.manual_seed(config['full_task']['seed'])
        data = base.FullTaskData(config, stores=base.build_latence_nbs_stores(config))
        known.ablation = 'full'
        evaluate(known, config, data, device)
    else:
        sys.argv = [str(Path(__file__).resolve()), *raw]
        base.main()


if __name__ == '__main__':
    main()
