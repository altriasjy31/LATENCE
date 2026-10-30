#!/usr/bin/env python3
"""Single-GPU scratch experiments with explicit arguments and isolated outputs.

Examples:
  python scripts/nbs/run_nbs_v092.py smoke --preset geometry --gpu 0
  python scripts/nbs/run_nbs_v092.py train --preset geometry --stop-epoch 5
  python scripts/nbs/run_nbs_v092.py resume --preset geometry --stop-epoch 20
  python scripts/nbs/run_nbs_v092.py evaluate --preset geometry --epoch 5
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PRESETS = ('baseline', 'ontology', 'geometry', 'geometry_random')
DDP_ENV = ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE',
           'MASTER_ADDR','MASTER_PORT','TORCHELASTIC_RUN_ID','TORCHELASTIC_RESTART_COUNT',
           'TORCHELASTIC_MAX_RESTARTS','TORCHELASTIC_ERROR_FILE','TORCHELASTIC_USE_AGENT_STORE')
OLD_ENV = ('CONFIG','WORK_DIR','WORK_ROOT','RESUME','INIT_FROM','ENCODER_FROM','STEPS','STOP_STEP','STOP_EPOCH',
           'CHECKPOINT','HEAD_CHECKPOINT','HYBRID_INIT','JOINT_CHECKPOINT','JOINT_WORK_DIR','PRESET','NUM_GPUS',
           'DEVICE','INPUT_DIR','METADATA_FILE','DEV_MANIFEST','ABLATION','METRIC_BACKEND','AUPRC_MODE','SCRIPTS')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','smoke','train','resume','evaluate'))
    p.add_argument('--preset',choices=PRESETS,default='geometry')
    p.add_argument('--gpu',default='0',help='One physical GPU index or UUID; child device is cuda:0')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--python',default=sys.executable,help='Interpreter for the existing LATENCE environment')
    p.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0')
    p.add_argument('--work-dir',type=Path,help='Optional isolated directory override')
    endpoint=p.add_mutually_exclusive_group()
    endpoint.add_argument('--stop-epoch',type=int)
    endpoint.add_argument('--stop-step',type=int)
    p.add_argument('--epoch',type=int,help='Fixed completed epoch to evaluate')
    p.add_argument('--input-dir',type=Path,default=Path('outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2'))
    p.add_argument('--metadata-file',type=Path,default=Path('/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl'))
    p.add_argument('--baseline-comparison',type=Path,default=Path('BASELINE_CONTRACT_v088.json'))
    return p


def child_environment(args):
    env={key:value for key,value in os.environ.items() if key not in DDP_ENV+OLD_ENV}
    env['CUDA_VISIBLE_DEVICES']=args.gpu if args.device=='cuda:0' else ''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        env[key]=str(args.threads)
    return env


def resolve_work_dir(args):
    default=ROOT/'outputs/latence_nbs_experiments/v092'/f'{args.preset}_single12800_scratch'
    if args.action=='smoke':default=default.with_name(default.name+'_smoke')
    return args.work_dir.resolve() if args.work_dir is not None else default


def build_commands(args):
    if args.threads<1:raise ValueError('--threads must be positive')
    if not re.fullmatch(r'[^,\s]+',args.gpu) or args.gpu=='-1':
        raise ValueError('--gpu must identify exactly one GPU')
    if args.action=='evaluate':
        if args.epoch is None or not 1<=args.epoch<=100:
            raise ValueError('evaluate requires --epoch1..100')
        if args.stop_epoch is not None or args.stop_step is not None:
            raise ValueError('evaluate does not use stop endpoints')
    elif args.epoch is not None:
        raise ValueError('--epoch is only for evaluate')
    if args.stop_epoch is not None and not 1<=args.stop_epoch<=100:
        raise ValueError('--stop-epoch must lie within the fixed100-epoch horizon')
    if args.stop_step is not None and args.stop_step<1:raise ValueError('--stop-step must be positive')
    if args.action=='prepare' and (args.stop_epoch is not None or args.stop_step is not None):
        raise ValueError('prepare does not use stop endpoints')
    if args.action=='smoke' and args.stop_epoch is not None:
        raise ValueError('smoke uses --stop-step, default3')
    config=ROOT/'nbs_models/nbs_protein_go/configs'/f'bp_full_task_v092_{args.preset}_single12800.json'
    worker=Path(__file__).with_name('train_nbs_full_task_v092.py')
    if not config.is_file() or not worker.is_file():raise FileNotFoundError('Install the complete v092 package first')
    cfg=json.loads(config.read_text())
    if cfg.get('full_task',{}).get('v092_preset')!=args.preset:raise ValueError('preset config identity differs')
    work=resolve_work_dir(args)
    common=[args.python,str(worker),'--preset',args.preset,'--config',str(config),'--work-dir',str(work),'--device',args.device]
    if args.action in ('train','smoke') and work.exists() and any(work.iterdir()):
        raise FileExistsError('Fresh training needs an empty work directory; use resume for an existing phase')
    if args.action=='prepare':return work,[common+['--stage','prepare','--backend','torch']]
    if args.action=='evaluate':
        checkpoint=work/f'nbs_epoch{args.epoch}.pt'
        if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
        command=common+['--stage','evaluate','--checkpoint',str(checkpoint),'--input-dir',str(args.input_dir.resolve()),
                        '--metadata-file',str(args.metadata_file.resolve()),'--ablation','full','--metric-backend','stage1','--auprc-mode','exact']
        if args.preset=='baseline':command+=['--allow-missing-references']
        else:command+=['--baseline-comparison',str(args.baseline_comparison.resolve())]
        return work,[command]
    extra=[]
    if args.action=='resume':
        checkpoint=work/'latest.pt'
        if not checkpoint.is_file():raise FileNotFoundError('No v092 checkpoint to resume: '+str(checkpoint))
        extra+=['--resume',str(checkpoint)]
    if args.action=='smoke':extra+=['--stop-step',str(args.stop_step or 3)]
    elif args.stop_epoch is not None:extra+=['--stop-epoch',str(args.stop_epoch)]
    elif args.stop_step is not None:extra+=['--stop-step',str(args.stop_step)]
    elif args.action=='train':extra+=['--stop-epoch','5']
    return work,[common+['--stage','prepare','--backend','torch'],common+['--stage','train',*extra]]


def summarize_smoke_timing(history):
    """Weight actual log windows, excluding updates1..5 and boundary windows."""
    counts={0:(0,0)}
    selected=[]
    peak=0.
    for row in history:
        step=row.get('step')
        if type(step) is not int:continue
        weak,core=row.get('weak_exposures'),row.get('core_exposures')
        if all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (weak,core)):
            counts[step]=(weak,core)
        value=row.get('peak_allocated_gb',0.)
        if isinstance(value,(int,float)) and math.isfinite(value):peak=max(peak,float(value))
        updates=row.get('window_updates')
        if type(updates) is not int or updates<1 or step-updates<5:continue
        seconds=row.get('window_training_seconds')
        if seconds is None and isinstance(row.get('seconds_per_step'),(int,float)):
            seconds=row['seconds_per_step']*updates
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:continue
        selected.append((step,updates,float(seconds)))
    updates=sum(row[1] for row in selected)
    seconds=sum(row[2] for row in selected)
    known_queries=True;queries=0.
    for step,count,_ in selected:
        if step not in counts or step-count not in counts:
            known_queries=False;continue
        delta=sum(counts[step])-sum(counts[step-count])
        if delta<0:known_queries=False
        else:queries+=delta
    return {'schema':'nbs-v092-smoke-timing-v1',
            'status':'measured' if updates else 'insufficient_post_warmup_updates',
            'excluded_initial_updates':5,'measured_updates':updates,'measured_training_seconds':seconds,
            'seconds_per_update':seconds/updates if updates else None,
            'mean_queries_per_update':queries/updates if updates and known_queries else None,
            'nominal_queries_per_update':12800,'peak_allocated_gb':peak,
            'scope':'single-GPU full-model training windows, including forward/backward/optimizer; excludes prepare/evaluation; not joint training',
            'query_count_scope':'actual weak+core exposure deltas across measured windows; includes short weak tails',
            'extrapolation':'none; short smoke timing is not proof of stable throughput or P0 speedup'}


def command_lock_path(args, work):
    """Keep legacy training exclusion; serialize evaluations separately.

    Evaluation reads a completed checkpoint and writes its eval_step outputs.
    It must not take the training lock. A project-wide evaluation lock also
    protects shared inference caches whose first build uses fixed temp names.
    The GPU is deliberately absent from the key: outputs and caches can overlap
    even when two evaluators use different GPUs.
    """
    if args.action == 'evaluate':
        return ROOT/'outputs/latence_nbs_experiments/v092/v092_evaluate.lock'
    return work.with_name(work.name+'.v092.lock')


def main(argv=None):
    args=parser().parse_args(argv)
    os.chdir(ROOT)
    work,commands=build_commands(args)
    env=child_environment(args)
    work.parent.mkdir(parents=True,exist_ok=True)
    # Preserve the old training lock for compatibility with a running launcher.
    lock_path=command_lock_path(args, work)
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            if args.action == 'evaluate':
                message='another v092 evaluation is running; wait for it to finish (training may continue)'
            else:
                message='another training or preparation command is running for this work directory'
            raise SystemExit(f'[v092] {message}. Lock: {lock_path}') from None
        print(f'[v092] action={args.action} preset={args.preset} work_dir={work}',flush=True)
        for command in commands:
            subprocess.run(command,cwd=ROOT,env=env,check=True)
    if args.action=='smoke':
        history_path=work/'training_history.json'
        history=json.loads(history_path.read_text()) if history_path.is_file() else []
        timing=summarize_smoke_timing(history)
        if not history_path.is_file():timing['note']='training history not available; no measured throughput reported'
        report=work/'smoke_timing.json'
        report.write_text(json.dumps(timing,indent=2,allow_nan=False)+'\n')
        print(f'[v092] smoke timing: {report}',flush=True)
        if timing['seconds_per_update'] is not None:
            print(f"[v092] measured {timing['seconds_per_update']:.3f}s/update over {timing['measured_updates']} post-warmup updates",flush=True)
        else:
            print('[v092] insufficient post-warmup updates for throughput; use smoke --stop-step 10 or more in a new smoke directory',flush=True)
    if args.action=='evaluate':
        print(f'[v092] evaluation: {work}/eval_step*/full/',flush=True)
    else:
        print(f'[v092] completed: {work}',flush=True)


if __name__=='__main__':main()
