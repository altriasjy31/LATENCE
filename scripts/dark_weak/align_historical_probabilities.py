#!/usr/bin/env python3
"""Align existing 2022 Expert/teacher probability rows to Dark–Weak target IDs.

Never use a 2025 label-derived probability array as historical evidence.
Provide either explicit source protein IDs, or an original 2022 LATENCE
metadata split whose protein order is the exact row order of the source .npy.
This script checks IDs and shapes but cannot prove a source model's time cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

NS={'bp':'biological_process','mf':'molecular_function','cc':'cellular_component'}
WIDTH={'bp':21312,'mf':7038,'cc':2903}


def sha256(path):
    hash=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda:handle.read(8<<20),b''):hash.update(part)
    return hash.hexdigest()


def read_ids(path):
    names=[line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not names or len(names)!=len(set(names)):
        raise ValueError(f'Empty or duplicate protein IDs: {path}')
    return names


def align(*, probabilities, target_ids, output_probabilities, output_ids,
          task, source_ids=None, source_metadata=None, source_split=None, block_size=256):
    if task not in NS:raise ValueError(f'Unknown task: {task}')
    if (source_ids is None)==(source_metadata is None):
        raise ValueError('Provide exactly one of source_ids or source_metadata')
    if source_metadata is not None and not source_split:
        raise ValueError('source_split is required with source_metadata')
    requested=read_ids(target_ids)
    if source_ids is not None:
        original=read_ids(source_ids)
    else:
        with Path(source_metadata).open('rb') as handle:
            meta=pickle.load(handle)
        original=[str(s) for s in meta[source_split][NS[task]]['proteins']]
        if not original or len(original)!=len(set(original)):
            raise ValueError(f'Source metadata {source_split}/{NS[task]} contains duplicated/empty proteins')
    source=np.load(probabilities,mmap_mode='r',allow_pickle=False)
    if source.ndim!=2 or source.shape!=(len(original),WIDTH[task]):
        raise ValueError(f'Source probability matrix {source.shape} differs from '
                         f'({len(original)}, {WIDTH[task]}), refusing alignment')
    indices={name:i for i,name in enumerate(original)}
    missing=[name for name in requested if name not in indices]
    if missing:
        raise ValueError(f'{len(missing)} targets missing from original 2022 probability rows: {missing[:8]}')
    dest=Path(output_probabilities)
    ids_dest=Path(output_ids)
    if dest.exists() or ids_dest.exists():
        raise FileExistsError('An output exists; write to a new versioned directory')
    dest.parent.mkdir(parents=True,exist_ok=True)
    ids_dest.parent.mkdir(parents=True,exist_ok=True)
    values=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float16,
                                     shape=(len(requested),WIDTH[task]))
    try:
        for start in range(0,len(requested),block_size):
            selected=requested[start:start+block_size]
            subset=np.asarray(source[[indices[name] for name in selected]],dtype=np.float32)
            if not np.isfinite(subset).all() or np.any((subset<0)|(subset>1)):
                raise ValueError('Expert input must contain finite probabilities in [0,1]')
            values[start:start+len(subset)]=subset.astype(np.float16)
        values.flush()
    except Exception:
        del values
        dest.unlink(missing_ok=True)
        raise
    del values
    ids_dest.write_text('\n'.join(requested)+'\n',encoding='utf-8')
    manifest={'task':task,'source_probabilities':str(Path(probabilities).resolve()),
              'source_probabilities_sha256':sha256(probabilities),
              'source_metadata':str(Path(source_metadata).resolve()) if source_metadata else None,
              'source_split':source_split, 'source_ids':str(Path(source_ids).resolve()) if source_ids else None,
              'source_count':len(original),'target_count':len(requested),
              'target_ids_sha256':sha256(target_ids),
              'aligned_probability_path':str(dest.resolve()),
              'aligned_probability_sha256':sha256(dest),
              'aligned_ids_sha256':sha256(ids_dest),
              'historical_source_time_cutoff_verified':False,
              'warning':'Check source probability provenance and data cutoff separately; ID alignment cannot establish a historic teacher.'}
    dest.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-prob',type=Path,required=True)
    sources=p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source-ids',type=Path)
    sources.add_argument('--source-metadata',type=Path)
    p.add_argument('--source-split',help='Original 2022 metadata split, e.g. exp_train')
    p.add_argument('--target-ids',type=Path,required=True)
    p.add_argument('--task',choices=NS,required=True)
    p.add_argument('--out-prob',type=Path,required=True)
    p.add_argument('--out-ids',type=Path,required=True)
    p.add_argument('--block-size',type=int,default=256)
    args=p.parse_args()
    if args.block_size<1:p.error('block-size must be >=1')
    print(json.dumps(align(probabilities=args.source_prob,target_ids=args.target_ids,
                           output_probabilities=args.out_prob,output_ids=args.out_ids,
                           task=args.task,source_ids=args.source_ids,
                           source_metadata=args.source_metadata,source_split=args.source_split,
                           block_size=args.block_size),indent=2))

if __name__=='__main__':main()
