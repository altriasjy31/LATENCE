#!/usr/bin/env python3
"""Audit actual NBS v0.8.9.1 inference batches without changing native model code.

Checks native ghost-query feature/graph construction and reported supervision
absence, and optionally compares sampled graph support nodes against ORIGINAL
2022 role-registry IDs of the same target proteins. This is a graph-inference
preflight, NOT an inductive-generalization certificate: some training information
may already be encoded in frozen representations and checkpoint parameters.

Supply --target-original-global-ids for a defensible seen-weak identity audit:
 .npy int64 [N] aligned to prepared protein_ids.txt; -1 means unknown/not in
 the training registry. These IDs must come from the ORIGINAL 2022 registry,
 not recomputed from a different protein sorting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'nbs_models'/'nbs_protein_go'))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def audit(args):
    from nbs_pg.full_task_data_v088 import FullTaskDataV088
    from nbs_pg.local_loader_v089 import build_latence_nbs_stores
    from nbs_pg.full_task_data_v084 import RELATIONS

    cfg=json.loads(Path(args.config).read_text())
    if (cfg.get('release_version')!='0.8.9.1'
            or cfg.get('model_version')!='0.8.8'
            or cfg.get('stage',{}).get('name')!='nbs_v089_long_phase'):
        raise ValueError('v089 graph audit requires stage=nbs_v089_long_phase and release/model version 0.8.9.1/0.8.8')
    ids=[s.strip() for s in (args.input_dir/'protein_ids.txt').read_text().splitlines() if s.strip()]
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Prepared protein_ids.txt is empty or duplicated')
    originals=None
    if args.target_original_global_ids:
        originals=np.load(args.target_original_global_ids,allow_pickle=False)
        if originals.ndim!=1 or len(originals)!=len(ids) or originals.dtype.kind!='i':
            raise ValueError('Original global IDs must be int64 [N], aligned to input protein_ids')
        if np.any(originals < -1):
            raise ValueError('Unknown original global IDs must use -1')
    data=FullTaskDataV088(cfg,stores=build_latence_nbs_stores(cfg))
    data.ensure_prepared()
    data.set_sampling_context(step=0,rank=0,training=False)
    device=torch.device(args.device)
    n=len(ids) if args.all_proteins else min(args.limit,len(ids))
    if n<=0: raise ValueError('No protein IDs to audit')
    summary=dict(input_dir=str(args.input_dir.resolve()),config=str(args.config.resolve()),
        num_targets=len(ids), audited=n, original_global_ids_provided=originals is not None,
        targets_with_known_original_global_id=0, self_support_hits=0,
        batches_checked=0, ghost_edges_noncosine=0, leaked_label_fields=0,
        sampled_edges=0, sampled_pseudo_edges=0, sampled_gold_edges=0,
        model_architecture_version="0.8.8",runner_version="0.8.9.1",
        rows_with_self_support=[], support_scope='actual_v089_inference_batch')
    cosine=RELATIONS.index('cosine')
    for start in range(0,n,args.batch_size):
        rows=np.arange(start,min(n,start+args.batch_size),dtype=np.int64)
        batch=data.inference_batch(args.input_dir,rows,device=device)
        # The no-label contract is a requirement, not an inference assumption.
        forbidden_labels={'targets','positive_mask','hard_y','prop_annotations','future_annotations'}
        present=forbidden_labels.intersection(batch)
        if present:
            summary['leaked_label_fields']+=len(present)
            raise RuntimeError(f'v089 target batch unexpectedly contains label fields: {sorted(present)}')
        global_ids=batch['sampled_global_ids'].detach().cpu().numpy().astype(np.int64)
        seeds=batch['sampled_seed_index'].detach().cpu().numpy().astype(np.int64)
        expected_ghost=-rows-1
        if not np.array_equal(global_ids[seeds],expected_ghost):
            raise RuntimeError('v089 inference batch violates negative-id ghost-query invariant')
        known=global_ids[global_ids>=0]
        if originals is not None:
            for row in rows:
                original=int(originals[row])
                if original<0:
                    continue
                summary['targets_with_known_original_global_id']+=1
                if original in known:
                    summary['self_support_hits']+=1
                    summary['rows_with_self_support'].append(dict(row=int(row),protein=ids[int(row)],original_id=original))
        for block in batch['blocks']:
            dst=block['dst_global_ids'].detach().cpu().numpy().astype(np.int64)
            edge_index=block['edge_index'].detach().cpu().numpy().astype(np.int64)
            edge_type=block['edge_type'].detach().cpu().numpy().astype(np.int64)
            summary['sampled_edges']+=edge_type.size
            if edge_type.size:
                if edge_index.ndim!=2 or edge_index.shape[0]!=2:
                    raise RuntimeError('v089 block has invalid edge_index shape')
                ghost_destination=dst[edge_index[1]]<0
                summary['ghost_edges_noncosine']+=int(np.count_nonzero(ghost_destination & (edge_type!=cosine)))
        for name, target in [('sampled_gold_edge','sampled_gold_edges'),
                             ('sampled_pseudo_edge','sampled_pseudo_edges')]:
            edges=batch.get(name)
            if edges is not None:
                summary[target]+=int(edges.shape[1] if edges.ndim==2 else edges.numel())
        summary['batches_checked']+=1
    if summary['ghost_edges_noncosine']:
        raise RuntimeError('A ghost query received a non-cosine edge; invalid v089 protocol')
    summary['seen_weak_inference_contract']=(
        'Graph ghost queries may represent training-seen 2022 weak sequences. '
        'Do not label these as strictly inductive or protein-zero-shot.')
    summary['self_support_status']= ('not_audited_without_original_registry_ids'
        if originals is None else ('detected_REQUIRES_GRAPH_INPUT_FIX' if summary['self_support_hits']
                                   else 'not_detected_in_audited_batches'))
    summary['inputs_sha256']={'manifest':sha256(args.input_dir/'ind_test_input_manifest.json'),
                             'protein_ids':sha256(args.input_dir/'protein_ids.txt')}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows_with_self_support'},indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--target-original-global-ids',type=Path)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--limit',type=int,default=32)
    p.add_argument('--all-proteins',action='store_true')
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',type=Path,default=Path('data/dark_weak_test/evaluations/v089_graph_audit.json'))
    a=p.parse_args()
    if a.batch_size<1 or a.limit<1:p.error('batch-size and limit must be positive')
    audit(a)

if __name__=='__main__':main()
