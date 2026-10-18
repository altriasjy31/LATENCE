#!/usr/bin/env python3
"""Invoke unchanged NBS v0.8.9.1 --stage export on independently prepared targets.

This script does not build P–P neighborhoods, candidate evidence, or GNN inputs:
those MUST originate from the project's verified v089 input preparation path.
It checks target identity, native provenance, original GO registry and the
checkpoint/input-linked prediction manifest, then writes a small adapter JSON.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

WIDTH={'bp':21312,'mf':7038,'cc':2903}
FORBIDDEN=('ind_test_labels_in_representation','ind_test_labels_in_candidate_selection',
           'ind_test_labels_in_pp_retrieval','expert_probability_in_forward_inputs','test_to_test_edges')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for piece in iter(lambda:f.read(8<<20),b''):h.update(piece)
    return h.hexdigest()


def read_ids(path):
    ids=[x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if not ids or len(set(ids))!=len(ids):raise ValueError(f'Duplicate or empty IDs: {path}')
    return ids


def preflight(args):
    input_dir=Path(args.input_dir)
    required=['protein_ids.txt','ind_test_repr.f16.npy','backbone_ind_test_prob.f16.npy',
              'ind_test_input_manifest.json','candidate_go_index.i32.npy']
    missing=[name for name in required if not (input_dir/name).is_file()]
    if missing:
        raise FileNotFoundError('v089 input directory lacks '+', '.join(missing)+
            '. Generate the complete *dark-weak* representation/neighbors/candidates '
            'using the existing v089 input-preparation code; an ind_test input directory '
            'cannot be reused for a different protein population.')
    protein_ids=read_ids(input_dir/'protein_ids.txt')
    desired=read_ids(args.expected_protein_ids)
    if protein_ids!=desired:
        raise ValueError(f'Prepared v089 input protein rows/order do not match dark-weak inference target; '
                         f'input {len(protein_ids)}, expected {len(desired)}, first={protein_ids[:3]}/{desired[:3]}')
    manifest=json.loads((input_dir/'ind_test_input_manifest.json').read_text())
    if any(manifest.get('leakage_contract',{}).get(k) is True for k in FORBIDDEN):
        raise ValueError('v089 input manifest reports leaked target labels, expert probability, or test-to-test edges')
    goids=read_ids(args.go_ids)
    if len(goids)!=WIDTH[args.task]:raise ValueError('Original GO model vocabulary width mismatch')
    registry=manifest.get('registries',{}).get('go_registry')
    if not registry:raise ValueError('Input manifest has no original GO registry to verify output column identity')
    registry=Path(registry)
    if not registry.is_absolute():registry=input_dir/registry
    if not registry.is_file():raise FileNotFoundError(f'GO registry does not exist: {registry}')
    with registry.open(newline='',encoding='utf-8') as f:
        rows=sorted(csv.DictReader(f,delimiter='\t'),key=lambda r:int(r['go_idx']))
    ordered=[r['input_go_id'] for r in rows]
    if ordered!=goids:raise ValueError('v089 GO registry differs from the original frozen GO output columns')
    for file_name, shape in [('ind_test_repr.f16.npy',None),('backbone_ind_test_prob.f16.npy',WIDTH[args.task])]:
        array=np.load(input_dir/file_name,mmap_mode='r')
        if array.ndim!=2 or len(array)!=len(protein_ids) or (shape is not None and array.shape[1]!=shape):
            raise ValueError(f'v089 {file_name} shape is not aligned to requested protein identities: {array.shape}')
    # An input manifest without explicit anti-leakage statements is not an audit.
    leakage=manifest.get('leakage_contract',{})
    unreported=[k for k in FORBIDDEN if k not in leakage]
    if unreported:
        print('[v089 provenance WARNING] input manifest lacks explicit leakage fields: '
              + ', '.join(unreported), flush=True)
    if getattr(args,'graph_audit_json',None):
        audited=json.loads(Path(args.graph_audit_json).read_text())
        if audited.get('runner_version')!='0.8.9.1' or audited.get('model_architecture_version')!='0.8.8':
            raise ValueError('Graph audit was not generated for the native v089/v088-model protocol')
        if audited.get('audited',0)<len(protein_ids):
            print('[v089 graph audit WARNING] only a subset of target graph batches was audited',flush=True)
        if not audited.get('original_global_ids_provided',False):
            print('[v089 graph audit WARNING] original registry IDs were not provided: cannot certify no self-support',flush=True)
        if audited.get('inputs_sha256',{}).get('manifest')!=sha256(input_dir/'ind_test_input_manifest.json'):
            raise ValueError('Graph audit belongs to another v089 input manifest')
        if audited.get('inputs_sha256',{}).get('protein_ids')!=sha256(input_dir/'protein_ids.txt'):
            raise ValueError('Graph audit belongs to another target protein list')
        if audited.get('self_support_hits'):
            raise ValueError('Graph audit detected the original target protein reentering sampled graph support')
        if audited.get('ghost_edges_noncosine') or audited.get('leaked_label_fields'):
            raise ValueError('Graph audit reports a v089 inference contract violation')
    return input_dir,manifest


def export(args):
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if (config.get('release_version')!='0.8.9.1'
            or config.get('model_version')!='0.8.8'
            or config.get('stage',{}).get('name')!='nbs_v089_long_phase'):
        raise ValueError('v089 export requires release_version=0.8.9.1, model_version=0.8.8 and stage=nbs_v089_long_phase')
    if str(config.get('task'))!=str(args.task):
        raise ValueError('NBS config task differs from target namespace')
    input_dir,prepared=preflight(args)
    runner=Path(args.train_runner)
    if not runner.is_file():raise FileNotFoundError(f'Missing installed native v089 train runner {runner}')
    command=[sys.executable,str(runner.resolve()),'--stage','export',
        '--config',str(Path(args.config).resolve()),'--work-dir',str(Path(args.work_dir).resolve()),
        '--checkpoint',str(Path(args.checkpoint).resolve()),'--input-dir',str(input_dir.resolve()),
        '--device',args.device,'--ablation','full']
    print('[Native v089 export]', ' '.join(command),flush=True)
    subprocess.run(command,check=True)
    checkpoint_sha=sha256(args.checkpoint)
    prepared_sha=sha256(input_dir/'ind_test_input_manifest.json')
    matches=[]
    for p in Path(args.work_dir).glob('eval_step*/full/nbs_full_task_prediction_manifest.json'):
        manifest=json.loads(p.read_text())
        if (manifest.get('checkpoint_sha256')==checkpoint_sha
            and manifest.get('input_manifest_sha256')==prepared_sha
            and manifest.get('runner_version')=='0.8.9.1'
            and manifest.get('num_task_go')==WIDTH[args.task]):
            matches.append((p,manifest))
    if len(matches)!=1:
        raise RuntimeError(f'Expected one native v089 export with matching checkpoint/input SHA, got {len(matches)}')
    path,manifest=matches[0]
    # Native v089 evaluator performs strict phase/lifetime/origin validation.
    project_root=Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path: sys.path.insert(0,str(project_root))
    from scripts.nbs.eval_nbs_full_task_v089 import _phase_identity, _encoder_identity, _validate_source_flags
    phase_identity=_phase_identity(manifest)
    encoder_identity=_encoder_identity(manifest)
    _validate_source_flags(manifest)
    if manifest.get('model_architecture_version')!='0.8.8' or manifest.get('data_architecture_version')!='0.8.8':
        raise ValueError('v089 must use v088 model/data architecture 0.8.8')
    if (manifest.get('uses_expert_probability_in_nbs_forward') is not False
        or manifest.get('uses_modelout_probability_in_nbs_forward') is not False
        or manifest.get('output_fusion')!='none'):
        raise ValueError('v089 standalone inference must not fuse Expert/Modelout')
    prob=Path(manifest['probability_path'])
    if not prob.is_file():raise FileNotFoundError(f'Native probability file missing: {prob}')
    if sha256(prob)!=manifest.get('output_probability_sha256'):
        raise ValueError('v089 exported probability hash differs from native manifest')
    arr=np.load(prob,mmap_mode='r')
    if arr.shape!=(len(read_ids(args.expected_protein_ids)),WIDTH[args.task]):
        raise ValueError(f'v089 output matrix shape mismatch: {arr.shape}')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    adapter={
        'model':'NBS_v089','runner_version':'0.8.9.1','model_architecture_version':'0.8.8','task':args.task,'probability':str(prob.resolve()),
        'protein_ids':str((input_dir/'protein_ids.txt').resolve()),
        'go_ids':str(Path(args.go_ids).resolve()),
        'native_prediction_manifest':str(path.resolve()),
        'native_input_manifest':str((input_dir/'ind_test_input_manifest.json').resolve()),
        'checkpoint_sha256':checkpoint_sha,'checkpoint_path':str(Path(args.checkpoint).resolve()),
        'prepared_input_manifest_sha256':prepared_sha,
        'prediction_sha256':manifest['output_probability_sha256'],
        'phase_identity':phase_identity,'encoder_identity':encoder_identity,
        'cumulative_epoch':manifest.get('epoch'), 'phase_step':manifest.get('step'),
        'forward_uses_expert_or_modelout':False,
        'evaluation_protocol':'historical_seen_weak_ghost',
        'graph_audit':str(Path(args.graph_audit_json).resolve()) if args.graph_audit_json else None,
        'graph_protocol_warning':('The prepared target graph may contain proteins already present '
             'as 2022 weak training nodes. This adapter does NOT certify absence of identity/self-neighbor '
             'information leakage; inspect data.inference_batch and neighbor construction before '
             'claiming independent inductive generalization.'),
    }
    out.write_text(json.dumps(adapter,indent=2,ensure_ascii=False)+'\n')
    print('[Dark–Weak v089 export]',out)
    return adapter


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=WIDTH)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--input-dir',required=True,type=Path)
    p.add_argument('--expected-protein-ids',required=True,type=Path)
    p.add_argument('--go-ids',required=True,type=Path)
    p.add_argument('--train-runner',type=Path,default=Path('scripts/nbs/train_nbs_full_task_v089.py'))
    p.add_argument('--work-dir',type=Path,default=Path('outputs/dark_weak_v089_export'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--graph-audit-json',type=Path,help='Optional native v089 actual-batch graph audit')
    p.add_argument('--out',type=Path,default=Path('data/dark_weak_test/evaluations/nbs_v089_adapter.json'))
    export(p.parse_args(argv))

if __name__=='__main__':main()
