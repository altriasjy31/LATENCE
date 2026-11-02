#!/usr/bin/env python3
"""Unified weak/rare/GO-unseen training over the established phase engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'nbs_models/nbs_protein_go')]
REVISION = '0.9.3-unified-r1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_experiment(config):
    ft = config.get('full_task', {})
    if ft.get('experiment_revision') != REVISION:
        raise ValueError('Use a generated v093 unified training_config.json')
    if config.get('release_version') != '0.9.3' or config.get('model_version') != '0.9.3':
        raise ValueError('Unified checkpoints use their own v093 model/data identity')
    if ft.get('expected_world_size') != 1 or type(ft.get('epochs')) is not int or ft['epochs'] < 5:
        raise ValueError('v093 requires one GPU and a declared fixed horizon >=5')
    if not config.get('data', {}).get('unified_dataset_manifest'):
        raise ValueError('Unified weak dataset is required before training')
    if ft.get('model', {}).get('output_strategy') != 'backbone_shared_residual':
        raise ValueError('The unified model must preserve the loaded Backbone at initialization')
    return config


def install():
    from scripts.nbs import train_nbs_full_task_v090 as base
    identity = str(Path(__file__).resolve())
    previous = getattr(base, '_nbs_v093_install_identity', None)
    if previous is not None:
        if previous != identity:
            raise RuntimeError('A different unified adapter is already installed')
        return base
    if getattr(base, '_nbs_v092_install_identity', None) is not None:
        raise RuntimeError('Run v092 and v093 workers in separate processes')
    from nbs_pg.full_task_data_v093 import FullTaskDataV093, build_latence_nbs_stores_v093
    from nbs_pg.full_task_loss_v093 import FullTaskLossConfigV093, training_objective
    from nbs_pg.full_task_model_v093 import FullTaskGraphModelV093, FullTaskModelConfigV093
    from nbs_pg import full_task_sampler_fast as fast

    # Keep the original counter/scheduler implementation, while assigning an
    # honest new checkpoint identity to the new model and derived data.
    base.RELEASE_VERSION = '0.9.3'
    base.MODEL_VERSION = '0.9.3'
    base.DATA_VERSION = '0.9.3'
    original_resume_contract = base.resume_contract

    def resume_contract(*args, **kwargs):
        result = original_resume_contract(*args, **kwargs)
        result.update(runner_version='0.9.3', model_version='0.9.3', experiment_revision=REVISION)
        return result

    base.resume_contract = resume_contract

    base.FullTaskData = FullTaskDataV093
    base.build_latence_nbs_stores = build_latence_nbs_stores_v093
    base.FullTaskModelConfigV090 = FullTaskModelConfigV093
    base.FullTaskLossConfigV090 = FullTaskLossConfigV093
    diagnostic_window = {}

    def objective_with_diagnostics(*args, **kwargs):
        logits, loss, parts = training_objective(*args, **kwargs)
        for name in ('main_binary', 'shared_binary', 'rare_backbone_KL', 'weak_selected_soft',
                     'contrib_shared', 'contrib_rare_backbone', 'contrib_weak_soft'):
            value = parts[name].detach()
            diagnostic_window[name] = diagnostic_window.get(name, 0) + value
        return logits, loss, parts

    base.training_objective = objective_with_diagnostics
    original_implementation = base.prediction_implementation

    def prediction_implementation(variant):
        result = dict(original_implementation(variant))
        result['full_task_sampler_fast.py'] = fast.implementation_sha256()
        for filename in ('go_registry_v093.py', 'unified_weak_dataset_v093.py', 'full_task_data_v093.py',
                         'full_task_model_v092.py', 'full_task_model_v093.py',
                         'full_task_loss_v093.py'):
            result[filename] = sha256(ROOT / 'nbs_models/nbs_protein_go/nbs_pg' / filename)
        result['train_nbs_full_task_v093.py'] = sha256(__file__)
        return result

    def build_model(feature_dim, ontology, model_options, variant, device, seen_task_mask=None):
        if variant not in ('fixed', 'dynamic'):
            raise ValueError('Unknown sampling variant')
        import numpy as np
        cfg = FullTaskModelConfigV093(**model_options)
        seen = None if seen_task_mask is None else np.array(seen_task_mask, dtype=bool, copy=True)
        model = FullTaskGraphModelV093(feature_dim, ontology, cfg, seen_task_mask=seen).to(device)
        model.training_variant = variant
        return model, cfg

    base.prediction_implementation = prediction_implementation
    base.build_model = build_model
    original_write = base.write_json

    def write_json(path, value):
        if Path(path).name == 'training_history.json' and value and diagnostic_window:
            updates = int(value[-1]['window_updates'])
            value[-1].update({name: float(total.cpu()) / updates
                              for name, total in diagnostic_window.items()})
            value[-1]['loss_definition'] = 'main_binary + weighted_shared + weighted_rare_KL + weighted_weak_selected_soft'
            diagnostic_window.clear()
        if Path(path).name == 'resolved_config.json':
            value.setdefault('runtime', {}).update(
                experiment_revision=REVISION,
                training_control='frequency-routed weak membership; shared residual + general GO head; rare retention and weak confidence auxiliaries',
                backbone_preservation='zero initialized shared correction equals loaded clipped Backbone logits at initialization',
                zero_shot_scope='Stage2 supervision exclusion and shared target scoring; upstream Stage1 provenance not erased')
        return original_write(path, value)

    base.write_json = write_json
    base.evaluate = evaluate
    base._nbs_v093_install_identity = identity
    return base


def evaluate(args, config, data, device):
    """Export honestly labelled residual predictions and unchanged exact metrics."""
    import numpy as np
    import torch
    from scripts.nbs.predict_nbs_go_v093 import load_validated_model
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
                    print(f'[v093 predict] {start + len(p)}/{rows}', flush=True)
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
    manifest = dict(request, schema='nbs-v093-unified-prediction-v1', complete=True,
                    checkpoint_path=str(Path(args.checkpoint).resolve()), probability_path=str(path),
                    output_probability_sha256=sha256(path), step=saved['step'], epoch=saved['epoch'],
                    num_task_go=data.num_task_go, task=config['task'],
                    uses_dense_backbone_logits_in_forward=True,
                    uses_expert_probability_in_nbs_forward=False,
                    output_fusion='backbone_plus_shared_residual_for_seen; general_shared_for_unseen',
                    zero_shot_scope='Stage2-only; reused upstream representations may encode held-out GO',
                    acceptance_protocol='same exact protein Fmax and AP; old no-backbone-forward acceptance gate is not applicable')
    base.write_json(manifest_path, manifest)
    if args.stage == 'evaluate':
        from nbs_pg.evaluation_report_v093 import write_extended_report
        output = write_extended_report(args, config, data)
        print(f'[v093 evaluation] {output}', flush=True)
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
