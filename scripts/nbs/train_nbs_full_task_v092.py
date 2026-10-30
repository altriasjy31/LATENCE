#!/usr/bin/env python3
"""Additive single-GPU scratch adapter over the strict v089/v090 phase engines.

Underlying checkpoint envelope/metric contracts remain v089/v090. The v092
experiment revision, complete model config and this implementation's hashes
identify the actual experiment. No original source file is rewritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'nbs_models/nbs_protein_go')]
PRESETS = ('baseline', 'ontology', 'geometry', 'geometry_random')
REVISION = '0.9.2-scratch-r1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_experiment(config, preset):
    ft = config.get('full_task', {})
    if ft.get('experiment_revision') != REVISION or ft.get('v092_preset') != preset:
        raise ValueError('Use the matching dedicated v092 preset config')
    if ft.get('expected_world_size') != 1 or ft.get('epochs') != 100:
        raise ValueError('v092 scratch presets require one GPU and a fixed100-epoch horizon')
    if ft.get('weak_batch') != 10240 or ft.get('core_batch') != 2560 or ft.get('accumulation_steps') != 2:
        raise ValueError('v092 matched presets require global10240/2560 and two microbatches')
    model = ft.get('model', {})
    if model.get('sage_layers') != 3 or model.get('hidden_dim') != 320:
        raise ValueError('v092 pilot fixes the original three-layer,320-dimensional backbone')
    if preset == 'baseline':
        if config.get('release_version') != '0.8.9.1' or model.get('prediction_mode') != 'direct':
            raise ValueError('baseline requires the original direct classifier envelope')
    else:
        if config.get('release_version') != '0.9.0' or model.get('prediction_mode') != 'shared_bilinear':
            raise ValueError('shared presets require the v090 scorer-phase envelope')
        expected_mode = 'ontology' if preset == 'ontology' else 'ontology_raw'
        expected_random = 'rff' if preset == 'geometry_random' else 'none'
        if model.get('go_input_mode') != expected_mode or model.get('protein_random_features') != expected_random:
            raise ValueError('raw geometry/random-feature switches do not match preset identity')
    return config


def install(preset):
    if preset not in PRESETS:
        raise ValueError('unknown v092 preset')
    if preset == 'baseline':
        from scripts.nbs import train_nbs_full_task_v089 as base
    else:
        from scripts.nbs import train_nbs_full_task_v090 as base
    from nbs_pg import full_task_sampler_fast as fast
    identity = (str(Path(__file__).resolve()), preset)
    installed = getattr(base, '_nbs_v092_install_identity', None)
    if installed is not None:
        if installed != identity:
            raise RuntimeError('different v092 adapter is already installed in this process')
        return base
    if getattr(base, '_nbs_p0_fast_wrapper_path', None) is not None:
        raise RuntimeError('do not combine independent P0 and v092 entrypoints in one process')
    original_prediction = base.prediction_implementation

    def prediction_implementation(variant):
        result = dict(original_prediction(variant))
        result['full_task_sampler_fast.py'] = fast.implementation_sha256()
        result['train_nbs_full_task_v092.py'] = sha256(__file__)
        result['v092_experiment_preset'] = hashlib.sha256(preset.encode()).hexdigest()
        if preset != 'baseline':
            result['full_task_model_v092.py'] = sha256(ROOT / 'nbs_models/nbs_protein_go/nbs_pg/full_task_model_v092.py')
        return result

    base.prediction_implementation = prediction_implementation
    if preset == 'baseline':
        base.FullTaskData = fast.FullTaskDataV088Fast
    else:
        from nbs_pg.full_task_model_v092 import FullTaskGraphModelV092, FullTaskModelConfigV092
        base.FullTaskData = fast.FullTaskDataV090Fast
        # The established evaluator constructs this config again to check all
        # architecture fields. Rebind that class in this worker only.
        base.FullTaskModelConfigV090 = FullTaskModelConfigV092

        def build_model(feature_dim, ontology, model_options, variant, device, seen_task_mask=None):
            if variant not in ('fixed', 'dynamic'):
                raise ValueError('unknown sampler variant')
            import numpy as np
            config = FullTaskModelConfigV092(**model_options)
            seen = None if seen_task_mask is None else np.array(seen_task_mask,dtype=bool,copy=True)
            model = FullTaskGraphModelV092(feature_dim, ontology, config,
                                          seen_task_mask=seen).to(device)
            model.training_variant = variant
            return model, config
        base.build_model = build_model
    original_evaluate = base.evaluate
    def evaluate(args, config, data, device):
        original_evaluate(args, config, data, device)
        if args.stage == 'evaluate':
            from nbs_pg.evaluation_report_v092 import write_extended_report
            report = write_extended_report(args, config, data)
            print(f'[v092] full Fmax/AP and frequency report: {report}', flush=True)
    base.evaluate = evaluate
    setattr(base, '_nbs_v092_install_identity', identity)
    return base


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if any(arg.split('=',1)[0] in ('--init-from', '--encoder-from') for arg in raw):
        raise ValueError('v092 experiments start from scratch; old encoder/head warmstarts are not exposed')
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--preset', choices=PRESETS, required=True)
    bootstrap.add_argument('--config', type=Path, required=True)
    known, rest = bootstrap.parse_known_args(raw)
    path = known.config.resolve()
    validate_experiment(json.loads(path.read_text(encoding='utf-8')), known.preset)
    base = install(known.preset)
    # The base parser receives explicit ordinary CLI arguments; shell variables
    # CONFIG/WORK_DIR/RESUME/NUM_GPUS are never consulted by this adapter.
    sys.argv = [str(Path(__file__).resolve()), '--config', str(path), *rest]
    base.main()


if __name__ == '__main__':
    main()
