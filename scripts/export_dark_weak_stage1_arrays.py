#!/usr/bin/env python3
"""Export aligned dense (Y, probability) arrays for existing LATENCE evaluators.

This is an I/O ADAPTER, not a replacement for the project's existing metric
implementation. Zeros in Y are unknown labels, not experimentally established
negatives; any Fmax/AUPRC produced by treating them as negatives is a proxy.

The exporter streams .npy predictions and writes arrays in the exact trained
GO order, with row IDs verified independently for each model.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from eval_dark_weak_predictions import (
    NS,read_go_list,parse_kv,load_prediction,load_protein_order)
import pickle


def run(args):
    with args.metadata.open('rb') as handle:
        original=pickle.load(handle) # trusted project metadata pickle
    src=original[args.split][NS[args.namespace]]
    names=list(src['proteins'])
    label_field='annotations' if args.split.endswith('_direct') else 'prop_annotations'
    if label_field not in src:
        raise KeyError(
            f'Missing {label_field} for split {args.split}. '
            'Rebuild metadata with build_dark_weak_eval_metadata.py v0.3.1.'
        )
    labels=list(src[label_field])
    if not names or len(names)!=len(labels) or len(set(names))!=len(names):
        raise ValueError('Invalid or empty benchmark subset')
    go_ids=read_go_list(args.unified_root/'labels'/args.namespace/'go_ids.txt')
    predictions=parse_kv(args.prediction,'--prediction')
    protein_ids=parse_kv(args.prediction_ids,'--prediction-ids')
    go_id_files=parse_kv(args.prediction_go_ids,'--prediction-go-ids')
    if set(predictions)!=set(protein_ids) or set(predictions)!=set(go_id_files):
        raise ValueError('Every prediction requires explicit MODEL=prediction.npy, MODEL=protein_ids.txt, and MODEL=go_ids.txt')
    arrs={}
    rowindices={}
    for model,file in predictions.items():
        arrs[model]=load_prediction(file,len(go_ids))
        rows=load_protein_order(protein_ids[model],arrs[model].shape[0])
        if read_go_list(go_id_files[model])!=go_ids:
            raise ValueError(f'{model} GO vocabulary differs from trained model order')
        missing=[v for v in names if v not in rows]
        if missing:
            raise KeyError(f'{model} predictions missing {len(missing)} test proteins: {missing[:5]}')
        rowindices[model]=[rows[name] for name in names]
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise FileExistsError(f'Output nonempty: {args.out}; use --overwrite')
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'protein_ids.txt').write_text('\n'.join(names)+'\n')
    (args.out/'go_ids.txt').write_text('\n'.join(go_ids)+'\n')
    shape=(len(names),len(go_ids))
    targetfile=args.out/'y_true.u8.npy'
    y=np.lib.format.open_memmap(targetfile,mode='w+',dtype='uint8',shape=shape)
    y[:]=0
    for i,ints in enumerate(labels):
        ix=np.asarray(ints,dtype=np.int64)
        if len(ix) and (ix.min()<0 or ix.max()>=len(go_ids)):
            raise IndexError(f'Invalid GO index for {names[i]}')
        y[i,ix]=1
    y.flush();del y
    preds={}
    for model,source in arrs.items():
        file=args.out/f'{model}.prob.f16.npy'
        dst=np.lib.format.open_memmap(file,mode='w+',dtype='float16',shape=shape)
        idx=rowindices[model]
        for i0 in range(0,len(names),args.batch_size):
            i1=min(i0+args.batch_size,len(names))
            block=np.asarray(source[idx[i0:i1]],dtype=np.float32)
            if not np.isfinite(block).all():
                raise ValueError(f'{model} scores include NaN or infinity')
            if model in args.logit_models:
                block=expit(block)
            elif (block.min()<0 or block.max()>1):
                raise ValueError(f'{model} scores are not probabilities [0,1]; use --logit-model {model}')
            dst[i0:i1]=block.astype('float16')
        dst.flush();del dst
        preds[model]=str(file)
    note={'split':args.split,'namespace':args.namespace,'label_field':label_field,'shape':list(shape),
          'protein_order_file':'protein_ids.txt','go_order_file':'go_ids.txt',
          'labels_file':str(targetfile), 'probability_files':preds,
          'go_column_aliases':'preserved exactly, model output order',
          'labels_warning':'Zeros in dense Y are unobserved GO labels, not verified negatives.',
          'metrics_note':'Existing LATENCE Fmax/AUPRC helper may score zeros as negatives; label as open-world proxy.',
          'row_alignment':'verified protein ID for each model; scores reindexed into metadata row order',
          'independent_time_cutoff':'must be independently audited for model/teacher/graph input sources'}
    (args.out/'export_manifest.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(note,ensure_ascii=False,indent=2))
    return note


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,default=Path('data/unidata_with_dark_weak_eval.pkl'))
    p.add_argument('--unified-root',type=Path,default=Path('data/dark_weak_test/unified'))
    p.add_argument('--split',required=True)
    p.add_argument('--namespace',choices=NS,required=True)
    p.add_argument('--prediction',action='append',required=True,help='MODEL=path.npy')
    p.add_argument('--prediction-ids',action='append',required=True,help='MODEL=protein_ids.txt')
    p.add_argument('--prediction-go-ids',action='append',required=True,help='MODEL=go_ids.txt')
    p.add_argument('--logit-model',dest='logit_models',action='append',default=[],help='Model using logits rather than probabilities')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args(argv)
    if args.batch_size<=0:
        p.error('--batch-size must be positive')
    if set(args.logit_models)-set(parse_kv(args.prediction,'--prediction')):
        p.error('Unknown --logit-model')
    run(args)

if __name__=='__main__':
    main()
