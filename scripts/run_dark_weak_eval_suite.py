#!/usr/bin/env python3
"""Run grouped dark-weak ranking evaluations on already-exported model predictions.

This runner does NOT execute Backbone/NBS forward passes; it calls the robust
model-ID/GO-order-aligned scorer for each selected evidence tier.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from argparse import Namespace
import pandas as pd
from eval_dark_weak_predictions import run, NS, parse_kv

PANELS={
    'future':[
        'dark_weak_test_exp_direct',
        'dark_weak_test_ipr2go_direct',
        'dark_weak_test_direct',
    ],
    'history':[
        'dark_weak_test_2022_ipr2go_matched_direct',
        'dark_weak_test_2025_ipr2go_all_direct',
        'dark_weak_test_2025_ipr2go_added_direct',
        'dark_weak_test_2025_ipr2go_retained_direct',
    ],
}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',choices=['future','history','all'],default='future')
    p.add_argument('--namespace',choices=NS,required=True)
    p.add_argument('--metadata',type=Path,default=Path('data/unidata_with_dark_weak_eval.pkl'))
    p.add_argument('--unified-root',type=Path,default=Path('data/dark_weak_test/unified'))
    p.add_argument('--prediction',action='append',required=True,help='MODEL=pred.npy')
    p.add_argument('--prediction-ids',action='append',required=True)
    p.add_argument('--prediction-go-ids',action='append',required=True)
    p.add_argument('--reference')
    p.add_argument('--ks',nargs='+',type=int,default=[10,50,100])
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--seed',type=int,default=6061)
    p.add_argument('--out',type=Path,default=Path('data/dark_weak_test/evaluations/suite'))
    p.add_argument('--save-pair-ranks',action='store_true')
    p.add_argument('--include-ontology-roots',dest='exclude_roots',action='store_false')
    p.set_defaults(exclude_roots=True)
    a=p.parse_args(argv)
    if a.panel=='all':
        cases=PANELS['future']+PANELS['history']
    else:
        cases=PANELS[a.panel]
    a.out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    completed=[]
    for split in cases:
        print(f'\n### {a.namespace}: {split} ###',flush=True)
        local=Namespace(**vars(a))
        local.split=split
        local.out=a.out/a.namespace/split
        local.assume_go_order=False
        local.allow_descriptive_labels=False
        result=run(local)
        summaries.extend({'split':split,'model':model,'namespace':a.namespace,**data}
                         for model,data in result['model_metrics'].items())
        completed.append(split)
    pd.DataFrame(summaries).to_csv(a.out/f'{a.namespace}_suite_metrics.tsv',sep='\t',index=False)
    (a.out/f'{a.namespace}_suite_manifest.json').write_text(json.dumps({
        'panel':a.panel,'namespace':a.namespace,'completed_splits':completed,
        'model_predictions':str(a.prediction),
        'note':'Open-world positive GO ranking; absent GO is unknown, not a biological negative.'
    },ensure_ascii=False,indent=2)+'\n')
    print('Done:',a.out/f'{a.namespace}_suite_metrics.tsv')

if __name__=='__main__':
    main()
