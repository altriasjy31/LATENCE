#!/usr/bin/env python3
"""Make label-free Dark–Weak inference split without changing original LATENCE metadata.

Creates a NEW pickle containing `dark_weak_inference` with 2022 protein IDs
and EMPTY GO labels. Evaluation labels remain in their original benchmark
splits and cannot be read by the forward pass through this new split.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import pandas as pd

NS={'bp':'biological_process','mf':'molecular_function','cc':'cellular_component'}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for piece in iter(lambda:f.read(8<<20),b''):h.update(piece)
    return h.hexdigest()


def build(args):
    if Path(args.output).resolve() == Path(args.metadata).resolve():
        raise ValueError('Cannot overwrite the original metadata pickle')
    with open(args.metadata,'rb') as f:metadata=pickle.load(f)
    if not isinstance(metadata,dict):raise TypeError('Expected LATENCE nested dict metadata')
    if args.split in metadata and not args.replace_split:
        raise FileExistsError(f'Split {args.split} already exists. Refusing to replace without --replace-split')
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(f'{args.output} exists; specify --overwrite')
    cohort=pd.read_csv(Path(args.unified_root)/'tables'/'full_dark_weak_cohort.tsv',sep='\t')
    if 'protein_22' not in cohort:raise ValueError('Missing protein_22 in full_dark_weak_cohort.tsv')
    cohort_ids=cohort.protein_22.astype(str).tolist()
    if len(set(cohort_ids))!=len(cohort_ids):raise ValueError('Duplicate cohort protein names')
    if args.full_cohort:
        if 'sequence_identical' not in cohort and 'sequence_status' not in cohort:
            raise ValueError('--full-cohort requires per-protein sequence status to exclude revised sequences')
        if 'sequence_identical' in cohort:
            good=cohort.sequence_identical.astype(str).str.lower().isin(['true','1'])
        else:
            good=cohort.sequence_status.eq('identical')
        valid=set(cohort.loc[good,'protein_22'].astype(str))
    else:
        valid=set(cohort_ids)
    outputs={}
    for ns,nsk in NS.items():
        selections=set()
        for spl in args.source_splits:
            if spl not in metadata:raise KeyError(f'Missing benchmark split {spl}')
            if nsk not in metadata[spl]:raise KeyError(f'No {nsk} in {spl}')
            selections.update(map(str,metadata[spl][nsk]['proteins']))
        if args.full_cohort:selections=valid
        extraneous=selections-set(cohort_ids)
        if extraneous:raise ValueError(f'{ns} requested {len(extraneous)} proteins outside historical cohort')
        ids=[item for item in cohort_ids if item in selections and item in valid]
        if not ids:raise ValueError(f'No proteins selected for {ns}')
        outputs[nsk]={'proteins':ids,'annotations':[[] for _ in ids],
                      'prop_annotations':[[] for _ in ids]}
    # Inference-only pickle MUST NOT contain 2025 benchmark GO labels at all.
    # The model requires the original 2022 train/test/exp_train/ind_test entries,
    # but does not need any dark_weak_test* label-bearing split.
    retain_benchmarks = bool(getattr(args, 'retain_benchmark_splits', False))
    new={key: value for key,value in metadata.items()
         if retain_benchmarks or not str(key).startswith('dark_weak_test')}
    new[args.split]=outputs
    stripped_splits = [key for key in metadata if key not in new]
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as f:pickle.dump(new,f,pickle.HIGHEST_PROTOCOL)
    iddir=Path(args.id_dir);iddir.mkdir(parents=True,exist_ok=True)
    files={}
    for ns,nsk in NS.items():
        path=iddir/f'{ns}_protein_ids.txt';path.write_text('\n'.join(outputs[nsk]['proteins'])+'\n')
        files[ns]={'count':len(outputs[nsk]['proteins']),'path':str(path.resolve()),'sha256':sha256(path)}
    manifest={'split':args.split,'metadata_source':str(Path(args.metadata).resolve()),
              'metadata_source_sha256':sha256(args.metadata),'new_metadata':str(target.resolve()),
              'new_metadata_sha256':sha256(target), 'historical_only_proteins':True,
              'all_inference_annotation_lists_empty':True,'future_benchmark_splits_removed':not retain_benchmarks,
              'removed_benchmark_splits':stripped_splits,'full_cohort':args.full_cohort,
              'source_splits':args.source_splits,'namespaces':files,
              'note':'This is label-free inference metadata. Evaluate against the separate benchmark annotation splits.'}
    dest=iddir/'inference_split_manifest.json'
    dest.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--unified-root',type=Path,default=Path('data/dark_weak_test/unified'))
    p.add_argument('--output',type=Path,default=Path('data/unidata_with_dark_weak_inference.pkl'))
    p.add_argument('--id-dir',type=Path,default=Path('data/dark_weak_test/eval_inference_ids'))
    p.add_argument('--split',default='dark_weak_inference')
    p.add_argument('--source-splits',nargs='+',default=[
      'dark_weak_test_2022_ipr2go_matched_direct','dark_weak_test_direct'])
    p.add_argument('--full-cohort',action='store_true')
    p.add_argument('--retain-benchmark-splits',action='store_true',help='UNSAFE for strict input isolation; retain labels only for diagnostics')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--replace-split',action='store_true')
    build(p.parse_args(argv))

if __name__=='__main__':main()
