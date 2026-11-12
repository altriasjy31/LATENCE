#!/usr/bin/env python3
"""Single-GPU v095 training with frozen ESM pre-classifier protein features."""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.nbs.run_nbs_v094 import atomic_json, digest, environment

REVISION = '0.9.6-esm-penultimate-r1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check','build','prepare','smoke','train','resume','evaluate','predict-go'))
    p.add_argument('--gpu', default='0')
    p.add_argument('--device', choices=('cuda:0','cpu'), default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--source-work-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v095/asl_hierarchy_single12800_scratch')
    p.add_argument('--dataset-dir', type=Path, help='Optional existing v094 dataset; default taken from source config')
    p.add_argument('--work-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v096/esm_penultimate_single12800')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--learning-rate', type=float, help='New run peak LR; default .001 for v095 initialization, source LR for scratch')
    p.add_argument('--precision', choices=('bf16','fp32'), default=None, help='Fresh configuration only; otherwise inherit source precision (normally bf16)')
    p.add_argument('--init-from', type=Path, help='Fresh run only: fixed completed v095 epoch (recommended E125), weights only; new optimizer and schedule')
    p.add_argument('--esm-manifest', type=Path, help='Frozen exporter manifest; required for new build/train/smoke')
    endpoints=p.add_mutually_exclusive_group()
    endpoints.add_argument('--stop-epoch',type=int)
    endpoints.add_argument('--stop-step',type=int)
    p.add_argument('--epoch',type=int)
    p.add_argument('--input-dir',type=Path,default=ROOT/'outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2')
    p.add_argument('--metadata-file',type=Path,default=Path('/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl'))
    p.add_argument('--go-ids-file',type=Path)
    p.add_argument('--ontology-file',type=Path)
    p.add_argument('--output-dir',type=Path)
    return p


def configure(args):
    """Clone the v095 optimization protocol and add an immutable feature stream."""
    source_path = args.source_work_dir.resolve() / 'resolved_config.json'
    config = deepcopy(json.loads(source_path.read_text()))
    if config.get('release_version') != '0.9.5' or config.get('model_version') != '0.9.5':
        raise ValueError('--source-work-dir must contain a v095 resolved_config.json')
    config.pop('runtime', None)
    manifest = (args.dataset_dir.resolve() / 'unified_dataset_manifest.json' if args.dataset_dir is not None
                else Path(config['data']['unified_dataset_manifest']).resolve())
    if not manifest.is_file():
        raise FileNotFoundError('Existing v094 unified dataset is required: ' + str(manifest))
    if args.esm_manifest is None:
        raise ValueError('A new v096 run requires --esm-manifest from the ESM exporter')
    esm_path = args.esm_manifest.resolve(strict=True)
    esm = json.loads(esm_path.read_text())
    # Full byte/hash/ID coverage checks run in prepare before a training directory
    # is written. The model width is inferred, never guessed as 2560 or 1024.
    dim = esm.get('feature_dim')
    if type(dim) is not int or dim < 1 or esm.get('complete') is not True:
        raise ValueError('ESM manifest must be complete with a positive integer feature_dim')
    ft = config['full_task']
    if ft.get('expected_world_size') != 1 or ft.get('weak_batch', 0) + ft.get('core_batch', 0) != 12800:
        raise ValueError('v096 source must use the v095 single-GPU effective batch 12800 protocol')
    if ft.get('accumulation_steps') != 2:
        raise ValueError('v096 preserves the v095 two-step gradient accumulation protocol')
    if ft.get('model', {}).get('encoder_variant') != 'hetero_tuned' or ft['model'].get('jk_mode') != 'sum':
        raise ValueError('v096 requires the v095 hetero_tuned encoder with JK sum')
    config['data']['unified_dataset_manifest'] = str(manifest)
    config['data']['esm_features_manifest'] = str(esm_path)
    config['data']['esm_features_manifest_sha256'] = digest(esm_path)
    config['release_version'] = config['model_version'] = '0.9.6'
    config['stage'].update(name='nbs_v096_esm_penultimate', freeze_first_stage_backbone=True,
                           freeze_esm_classifier=True, use_expert_probability_in_forward=False)
    ft.update(seed=1096, epochs=args.epochs, experiment_revision=REVISION,
              experiment_version='0.9.6', experiment_preset='v095_plus_esm_penultimate',
              checkpoint_every_epochs=5)
    if args.precision is not None:
        ft['precision'] = args.precision
    if args.init_from is not None:
        ft['learning_rate'] = .001
    if args.learning_rate is not None:
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError('--learning-rate must be finite and positive')
        ft['learning_rate'] = args.learning_rate
    if ft.get('scheduler', {}).get('min_lr', 0.) > ft['learning_rate']:
        raise ValueError('Peak learning rate must not be smaller than the inherited minimum LR')
    ft['architecture_extension'] = 'v095_hetero_tuned_plus_frozen_ESM_penultimate_gated_input'
    ft['model'].update(esm_dim=dim, esm_dropout=.1, esm_gate_init=0.)
    # Reuse v095 loss, sampler, optimizer, batch, schedule shape and all old model
    # fields. Warm initialization uses peak LR .001; scratch keeps the source LR.
    # A new schedule starts at epoch zero; no optimizer reuse.
    ft['initialization'] = {'mode': 'fresh'}
    if args.init_from is not None:
        path = args.init_from.resolve(strict=True)
        if not re.fullmatch(r'nbs_epoch[1-9][0-9]*\.pt', path.name):
            raise ValueError('--init-from requires a fixed nbs_epoch<number>.pt, never latest.pt')
        ft['initialization'] = dict(mode='v095_weights_only', checkpoint=str(path), sha256=digest(path),
                                    optimizer_reused=False, scheduler_reused=False)
    config.pop('v094_experiment', None)
    config.pop('v095_experiment', None)
    config['v096_experiment'] = dict(revision=REVISION, source_config_sha256=digest(source_path),
                                    unified_dataset_manifest_sha256=digest(manifest),
                                    esm_manifest_sha256=digest(esm_path),
                                    reused_dataset_version='0.9.4', strict_pipeline_zero_shot=False,
                                    feature_source='ESM2ResNet.output_layer[0] input, frozen eval')
    directory = args.work_dir.resolve().with_name(args.work_dir.name + '_config')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'training_config.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Existing v096 configuration differs; use a separate --work-dir')
    else:
        atomic_json(path, config)
    return path


def operation_lock(args,work):
    if args.action in ('evaluate','predict-go'):
        return ROOT/'outputs/latence_nbs_experiments/v092/v092_evaluate.lock'
    return work.with_name(work.name+'.v096.lock')


def main(argv=None):
    args=parser().parse_args(argv)
    os.chdir(ROOT)
    if args.threads<1 or args.epochs<11:
        raise ValueError('threads must be positive and epochs >=11')
    if not re.fullmatch(r'[^,\s]+',args.gpu) or args.gpu=='-1':
        raise ValueError('--gpu must select one physical GPU or UUID')
    inference=args.action in ('evaluate','predict-go')
    if args.init_from is not None and args.action not in ('build','prepare','train','smoke'):
        raise ValueError('--init-from is only for a new run; resume/evaluate use saved lineage')
    if args.learning_rate is not None and args.action in ('resume','evaluate','predict-go'):
        raise ValueError('Learning rate is frozen; resume/evaluate use the saved configuration')
    if args.esm_manifest is not None and args.action in ('resume','evaluate','predict-go'):
        raise ValueError('ESM cache is frozen; resume/evaluate use the saved configuration')
    if args.precision is not None and args.action in ('resume','evaluate','predict-go'):
        raise ValueError('Precision is frozen; resume/evaluate use saved configuration')
    if args.stop_epoch is not None and not 1<=args.stop_epoch<=args.epochs:
        raise ValueError('stop-epoch must lie within the fixed horizon')
    if args.stop_step is not None and args.stop_step<1:
        raise ValueError('stop-step must be positive')
    if inference:
        if args.epoch is None or not 1<=args.epoch<=args.epochs:
            raise ValueError('Evaluation/prediction requires a completed --epoch')
        if args.stop_epoch is not None or args.stop_step is not None:
            raise ValueError('Evaluation/prediction has no stop endpoint')
    elif args.epoch is not None:
        raise ValueError('--epoch is only for evaluate/predict-go')
    env=environment(args)
    if args.action=='check':
        subprocess.run([args.python,'-c','import torch; print("PyTorch",torch.__version__)'],cwd=ROOT,env=env,check=True)
        subprocess.run([args.python,'-m','unittest','discover','-s','tests','-p','test_v096*.py','-v'],cwd=ROOT,env=env,check=True)
        return
    work=args.work_dir.resolve()
    if args.action=='smoke':
        work=work.with_name(work.name+'_smoke')
        args.work_dir=work
    lock_path=operation_lock(args,work)
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'The requested operation is already running: {lock_path}') from None
        config_path=work/'resolved_config.json' if inference or args.action=='resume' else configure(args)
        if not config_path.is_file():
            raise FileNotFoundError('Run configuration is missing: '+str(config_path))
        config=json.loads(config_path.read_text())
        if config['full_task']['epochs']!=args.epochs:
            raise ValueError('Requested horizon differs from the frozen configuration')
        if args.action=='build':
            if args.stop_epoch is not None or args.stop_step is not None:
                raise ValueError('Build has no training endpoint')
            print(f'[v096 configured; existing data reused] {config_path}',flush=True)
            return
        worker=ROOT/'scripts/nbs/train_nbs_full_task_v096.py'
        common=[args.python,str(worker),'--config',str(config_path),'--work-dir',str(work),'--device',args.device]
        if args.action=='predict-go':
            if args.go_ids_file is None:
                raise ValueError('predict-go requires --go-ids-file')
            command=[args.python,str(ROOT/'scripts/nbs/predict_nbs_go_v096.py'),'--work-dir',str(work),
                     '--epoch',str(args.epoch),'--device',args.device,'--input-dir',str(args.input_dir.resolve()),
                     '--go-ids-file',str(args.go_ids_file.resolve())]
            for name in ('ontology_file','output_dir'):
                if getattr(args,name) is not None:
                    command += ['--'+name.replace('_','-'),str(getattr(args,name).resolve())]
            commands=[command]
        elif args.action=='evaluate':
            checkpoint=work/f'nbs_epoch{args.epoch}.pt'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            commands=[common+['--stage','evaluate','--checkpoint',str(checkpoint),'--input-dir',str(args.input_dir.resolve()),
                             '--metadata-file',str(args.metadata_file.resolve())]]
        elif args.action=='prepare':
            commands=[common+['--stage','prepare','--backend','torch']]
        else:
            if args.action in ('train','smoke') and work.exists() and any(work.iterdir()):
                raise FileExistsError('Fresh training requires an empty work directory; use resume for existing runs')
            extra=[]
            if args.action=='resume':
                checkpoint=work/'latest.pt'
                if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
                extra += ['--resume',str(checkpoint)]
            if args.action=='smoke':
                if args.stop_epoch is not None: raise ValueError('Smoke uses --stop-step, default 3')
                extra += ['--stop-step',str(args.stop_step or 3)]
            elif args.stop_step is not None:
                extra += ['--stop-step',str(args.stop_step)]
            else:
                extra += ['--stop-epoch',str(args.stop_epoch or args.epochs)]
            commands=[common+['--stage','prepare','--backend','torch'],common+['--stage','train',*extra]]
        for command in commands:
            subprocess.run(command,cwd=ROOT,env=env,check=True)
    print(f'[v096 complete] {args.action}: {work}',flush=True)


if __name__=='__main__': main()
