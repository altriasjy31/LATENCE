#!/usr/bin/env python3
"""Single-GPU ASL + selected weak teacher + hierarchy training on the v094 dataset."""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.nbs.run_nbs_v094 import atomic_json, digest, environment

REVISION = '0.9.5-asl-hierarchy-r1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check','build','prepare','smoke','train','resume','evaluate','predict-go'))
    p.add_argument('--gpu', default='1')
    p.add_argument('--device', choices=('cuda:0','cpu'), default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--source-work-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v094/weak_teacher_single12800')
    p.add_argument('--dataset-dir', type=Path, help='Optional existing v094 dataset; default taken from source config')
    p.add_argument('--work-dir', type=Path, default=ROOT/'outputs/latence_nbs_experiments/v095/asl_hierarchy_single12800')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--precision', choices=('bf16','fp32'), default=None, help='Fresh configuration only; default bf16')
    p.add_argument('--init-from', type=Path, help='Fresh run only: fixed completed v094 epoch, weights only; new optimizer and schedule')
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
    """Reuse immutable data and freeze a separate v095 training configuration."""
    source_path=args.source_work_dir.resolve()/'resolved_config.json'
    config=deepcopy(json.loads(source_path.read_text()))
    if config.get('release_version') != '0.9.4':
        raise ValueError('--source-work-dir must contain a v094 resolved_config.json')
    config.pop('runtime',None)
    manifest=(args.dataset_dir.resolve()/'unified_dataset_manifest.json' if args.dataset_dir is not None
              else Path(config['data']['unified_dataset_manifest']).resolve())
    if not manifest.is_file():
        raise FileNotFoundError('Existing v094 unified dataset is required: '+str(manifest))
    config['data']['unified_dataset_manifest']=str(manifest)
    config['release_version']=config['model_version']='0.9.5'
    config['stage'].update(name='nbs_v095_asl_hierarchy',freeze_first_stage_backbone=True,
                           use_expert_probability_in_forward=False,effective_candidate_topk=16)
    config['supervision_contract']['weak_pseudo'].update(
        source='backbone_membership_and_selected_modelout',
        use_probability_as_soft_target='selected_confidence_weighted_modelout_KL',
        negative_policy='unknown_masked_except_selected_trustworthy_teacher_negatives')
    ft=config['full_task']
    ft.update(seed=1095,epochs=args.epochs,expected_world_size=1,
              experiment_revision=REVISION,experiment_version='0.9.5',
              experiment_preset='asl_selected_teacher_hierarchy',
              weak_target_mode='backbone_positive_ASL_selected_modelout',
              monitor_core=False,checkpoint_every_epochs=5,
              weak_batch=10240,core_batch=2560,accumulation_steps=2,
              learning_rate=.003,weight_decay=.0001,grad_clip=5.,
              epoch_unit='weak',variant='dynamic',precision=args.precision or 'bf16')
    ft['architecture_extension']='hetero_tuned_typed_BN_residual_JK_sum_raw_GO_geometry'
    ft['sampler'].update(mode='dynamic',candidate_topk=16)
    ft['model'].update(encoder_variant='hetero_tuned',sage_layers=3,hidden_dim=320,
                       prediction_mode='shared_bilinear',output_strategy='shared_direct',
                       go_input_mode='ontology_raw',protein_random_features='none',jk_mode='sum')
    ft['scheduler']=dict(name='epoch_warmup_hold_cosine',warmup_epochs=5.,hold_until_epoch=10.,min_lr=.00003)
    ft['optimizer']=dict(name='adamw',eps=1e-8,betas=[.9,.999],decay_policy='matrices_only')
    ft['loss']=dict(core_weight=.25,weak_weight=1.,modelout_soft_weight=1.,
                    loss_scale=1000.,hierarchy_weight=.01,asl_gamma_neg=4.,asl_gamma_pos=0.,asl_clip=.05,
                    teacher_topk=2048,teacher_min_probability=.05,teacher_confidence_power=.5,
                    hard_negative_topk=256,hard_negative_max_teacher=.01,hard_negative_weight=.25)
    ft['initialization']={'mode':'fresh'}
    if args.init_from is not None:
        path=args.init_from.resolve(strict=True)
        if not re.fullmatch(r'nbs_epoch[1-9][0-9]*\.pt',path.name):
            raise ValueError('--init-from requires a fixed nbs_epoch<number>.pt, never latest.pt')
        ft['initialization']=dict(mode='v094_weights_only',checkpoint=str(path),sha256=digest(path),
                                  optimizer_reused=False,scheduler_reused=False)
    config.pop('v094_experiment',None)
    config['v095_experiment']=dict(revision=REVISION,source_config_sha256=digest(source_path),
                                  unified_dataset_manifest_sha256=digest(manifest),
                                  reused_dataset_version='0.9.4',strict_pipeline_zero_shot=False)
    # A sibling configuration directory leaves the training output genuinely
    # empty for the inherited trainer's strict fresh-run check.
    directory=args.work_dir.resolve().with_name(args.work_dir.name+'_config')
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'training_config.json'
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError('Existing v095 configuration differs; use a separate --work-dir')
    else:
        atomic_json(path,config)
    return path


def operation_lock(args,work):
    if args.action in ('evaluate','predict-go'):
        return ROOT/'outputs/latence_nbs_experiments/v092/v092_evaluate.lock'
    return work.with_name(work.name+'.v095.lock')


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
        subprocess.run([args.python,'-m','unittest','discover','-s','tests','-p','test_v095*.py','-v'],cwd=ROOT,env=env,check=True)
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
            print(f'[v095 configured; existing data reused] {config_path}',flush=True)
            return
        worker=ROOT/'scripts/nbs/train_nbs_full_task_v095.py'
        common=[args.python,str(worker),'--config',str(config_path),'--work-dir',str(work),'--device',args.device]
        if args.action=='predict-go':
            if args.go_ids_file is None:
                raise ValueError('predict-go requires --go-ids-file')
            command=[args.python,str(ROOT/'scripts/nbs/predict_nbs_go_v095.py'),'--work-dir',str(work),
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
    print(f'[v095 complete] {args.action}: {work}',flush=True)


if __name__=='__main__': main()
