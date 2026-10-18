#!/usr/bin/env python3
"""Build a lossless, ordered Dark–Weak MSA index VIEW over existing binary shards.

No MSA is recomputed or copied; only original index records are selected.
A historical cutoff/provenance audit of the source MSA database is still
required before claiming 2022-frozen inputs.

Example:
 python scripts/dark_weak/prepare_msa_index_view.py \\
   --target-ids data/dark_weak_test/eval_inference_ids/bp_protein_ids.txt \\
   --source-index data/exp_MSA_bin/index.pkl \\
   --source-index data/train_MSA_bin/index.pkl \\
   --out-index data/dark_weak_test/msa_view/index.pkl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

REQUIRED = ('shards','proteins','shard_ids','offsets','nseqs','seqlens')


def hash_file(path):
    dig = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            dig.update(block)
    return dig.hexdigest()


def read_ids(path):
    items = [s.strip() for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not items or len(items) != len(set(items)):
        raise ValueError(f'Target protein IDs must be nonempty and unique: {path}')
    return items


def make_view(target_ids, source_indices, output_index, duplicate_policy='error', verify_shards=True):
    target_ids = Path(target_ids)
    dest = Path(output_index)
    desired = read_ids(target_ids)
    if duplicate_policy not in ('error', 'first'):
        raise ValueError('duplicate_policy must be error or first')
    if dest.exists():
        raise FileExistsError(f'{dest} exists; output must be a new versioned index')
    lookup = {}
    sources = []
    collisions = []
    for path in source_indices:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open('rb') as handle:
            index = pickle.load(handle)
        missing = [k for k in REQUIRED if k not in index]
        if missing:
            raise ValueError(f'Index {path} is missing keys: {missing}')
        records = list(index['proteins'])
        if len(records) != len(set(map(str, records))):
            raise ValueError(f'Index {path} contains duplicate protein IDs')
        for k in ('shard_ids','offsets','nseqs','seqlens'):
            if len(index[k]) != len(records):
                raise ValueError(f'Index {path}: inconsistent {k} length')
        shards = [(path.parent / Path(s)).resolve() for s in index['shards']]
        shard_ids = np.asarray(index['shard_ids'], dtype=np.int64)
        if np.any(shard_ids < 0) or np.any(shard_ids >= len(shards)):
            raise ValueError(f'Index {path} has out-of-range shard IDs')
        if verify_shards:
            absent = [str(x) for x in shards if not x.is_file()]
            if absent:
                raise FileNotFoundError(f'{path}: missing binary MSA shard(s): {absent[:5]}')
        relevant = set(desired)
        for i, protein in enumerate(records):
            protein = str(protein)
            if protein not in relevant:
                continue
            record = {
                'shard': str(shards[int(shard_ids[i])]),
                'offset': int(index['offsets'][i]),
                'nseqs': int(index['nseqs'][i]),
                'seqlens': int(index['seqlens'][i]),
                'source': str(path),
            }
            if record['offset'] < 0 or record['nseqs'] <= 0 or record['seqlens'] <= 0:
                raise ValueError(f'Invalid MSA binary record {protein} in {path}')
            if protein in lookup:
                collisions.append({'protein': protein, 'first_source': lookup[protein]['source'],
                                   'other_source': str(path)})
                if duplicate_policy == 'error':
                    raise ValueError(f'Protein {protein} occurs in multiple MSA indices; '
                                     'use --duplicate-policy first only after verifying source priority')
                continue
            lookup[protein] = record
        sources.append({'index_path': str(path), 'index_sha256': hash_file(path),
                        'num_records': len(records), 'num_shards': len(shards)})
    absent = [name for name in desired if name not in lookup]
    if absent:
        raise ValueError(f'Missing {len(absent)}/{len(desired)} requested proteins from MSA indices: {absent[:10]}')
    shard_to_idx = {}
    shards = []
    ids, offsets, nseqs, seqlens = [], [], [], []
    for protein in desired:
        record = lookup[protein]
        shard = record['shard']
        if shard not in shard_to_idx:
            shard_to_idx[shard] = len(shards)
            shards.append(shard)
        ids.append(shard_to_idx[shard]); offsets.append(record['offset'])
        nseqs.append(record['nseqs']); seqlens.append(record['seqlens'])
    view = dict(shards=shards, proteins=desired,
                shard_ids=np.asarray(ids, np.int32), offsets=np.asarray(offsets,np.int64),
                nseqs=np.asarray(nseqs,np.int32), seqlens=np.asarray(seqlens,np.int32))
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary=dest.with_suffix(dest.suffix+'.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(view,handle,protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(dest)
    manifest = dict(schema='latence_msa_index_view_v1', view_index=str(dest.resolve()),
        view_sha256=hash_file(dest), target_ids=str(target_ids.resolve()),
        target_ids_sha256=hash_file(target_ids), num_proteins=len(desired),
        num_shards=len(shards), source_indices=sources,
        duplicate_policy=duplicate_policy, collisions=collisions[:200],
        n_collisions=len(collisions), source_cutoff_verified=False,
        note=('Reuses unmodified MSA shards. Source MSA alignment/retrieval databases MUST '
              'be audited for temporal leakage before historical claims.'))
    dest.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-ids',type=Path,required=True)
    p.add_argument('--source-index',action='append',type=Path,required=True)
    p.add_argument('--out-index',type=Path,required=True)
    p.add_argument('--duplicate-policy',choices=('first','error'),default='error')
    p.add_argument('--no-verify-shards',action='store_true')
    a=p.parse_args()
    print(json.dumps(make_view(a.target_ids,a.source_index,a.out_index,a.duplicate_policy,
                               verify_shards=not a.no_verify_shards),indent=2))

if __name__=='__main__':main()
