#!/usr/bin/env python3
"""Audit actual v094 inference graph batches without changing model/training code.

Query ghosts must remain label-free and contain NO dense modelout/B logits.
If original global IDs are supplied, report conservative *batch-union* self
support risk; this is not proof of a direct edge to the same target. Neither a
clean audit nor a trained-seen weak protein is an independent zero-shot claim.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_ids(path):
    values = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError('Protein IDs must be nonempty and unique')
    return values


def discover_original_ids(registry_path, target_ids):
    """Use recorded protein_id column ONLY; never guess role-local name order."""
    with Path(registry_path).open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {'protein_id', 'protein_idx'}.issubset(reader.fieldnames or []):
            return None
        named = {}
        for row in reader:
            name = row['protein_id']
            if name in named:
                raise ValueError('Duplicate protein IDs in original NBS registry')
            named[name] = int(row['protein_idx'])
    return np.array([named.get(i, -1) for i in target_ids], dtype=np.int64)


def graph_schema_checks(query_rows, batch, originals=None):
    """Pure-array check for one sampled batch; reused by unit tests."""
    rows = np.asarray(query_rows, np.int64)
    if not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError('Query rows must be nonempty and unique')
    bad_labels = {'targets', 'positive_mask', 'hard_y', 'prop_annotations',
                  'future_annotations', 'weak_modelout_prob', 'base_logits'} & set(batch)
    if bad_labels:
        raise ValueError(f'Future/teacher/backbone information entered inference batch: {sorted(bad_labels)}')
    ids = np.asarray(batch['sampled_global_ids'], np.int64)
    seeds = np.asarray(batch['sampled_seed_index'], np.int64)
    if not np.array_equal(ids[seeds], -rows-1):
        raise ValueError('v094 external queries are not isolated negative-ID ghosts')
    known = ids[ids >= 0]
    self_hits = []
    if originals is not None:
        for row in rows:
            id_ = int(originals[int(row)])
            if id_ >= 0 and id_ in known:
                self_hits.append(int(row))
    # Validate typed graph relations without inferring a relation type index from text.
    forbidden_ghost_edges = 0
    edges = 0
    for block in batch['blocks']:
        dst = np.asarray(block['dst_global_ids'], np.int64)
        ei = np.asarray(block['edge_index'], np.int64)
        types = np.asarray(block['edge_type'], np.int64)
        if ei.ndim != 2 or ei.shape[0] != 2 or ei.shape[1] != len(types):
            raise ValueError('Typed layer edge indexing has changed')
        if np.any(ei < 0) or (ei.size and int(ei[1].max()) >= len(dst)):
            raise ValueError('Invalid typed block destination index')
        edges += len(types)
        # In v094 target ghosts are cosine-only; relation code must be obtained
        # from the native RELATIONS ordering, not hardcoded as 4.
        cosine_type = batch.get('_audit_cosine_relation_index')
        if cosine_type is None:
            raise ValueError('Missing native cosine relation index in audit context')
        forbidden_ghost_edges += int(np.count_nonzero((dst[ei[1]] < 0) & (types != int(cosine_type))))
    if forbidden_ghost_edges:
        raise ValueError('Ghost received noncosine graph messages')
    return {'batch_self_support_rows': self_hits,
            'batch_self_support_hits': len(self_hits),
            'sampled_edges': edges, 'support_nodes': len(known),
            'query_edges_noncosine': forbidden_ghost_edges}


def _audit_unlocked(args):
    import torch
    sys.path[:0] = [str(ROOT), str(ROOT/'nbs_models/nbs_protein_go')]
    from nbs_pg.full_task_data_v094 import FullTaskDataV094, build_latence_nbs_stores_v094
    from nbs_pg.full_task_data_v084 import RELATIONS
    from scripts.dark_weak.v094_native_adapter import validate_config, json_file

    config = json_file(args.config)
    validate_config(config, args.task)
    directory = args.input_dir.resolve(strict=True)
    names = read_ids(directory/'protein_ids.txt')
    if args.original_global_ids is not None:
        originals = np.load(args.original_global_ids, allow_pickle=False)
        if originals.ndim != 1 or len(originals) != len(names) or originals.dtype.kind != 'i':
            raise ValueError('--original-global-ids must contain exactly one int ID per target row')
        originals = originals.astype(np.int64, copy=False)
    else:
        originals = discover_original_ids(Path(config['data']['root'])/'features/protein_registry.csv', names)
    if originals is not None and np.any(originals < -1):
        raise ValueError('Original global ID must be >=-1 (-1: not in source registry)')
    # ensure_prepared() in v084 explicitly loads immutable pools read-only; it
    # refuses to build missing neighbor caches. The native _open_inference can
    # still create a derived core-neighbor cache if a core holdout exists.
    data = FullTaskDataV094(config, stores=build_latence_nbs_stores_v094(config))
    data.ensure_prepared()
    data.set_sampling_context(step=0, rank=0, training=False)
    n = len(names) if args.all_proteins else min(args.limit, len(names))
    if not n or args.batch_size < 1:
        raise ValueError('No targets or invalid audit batch size')
    cosine = RELATIONS.index('cosine')
    observed = []
    known_originals = 0
    accum = {'batch_self_support_hits': 0, 'sampled_edges': 0, 'support_nodes': 0}
    for start in range(0, n, args.batch_size):
        rows = np.arange(start, min(start+args.batch_size, n), dtype=np.int64)
        batch = data.inference_batch(directory, rows, device=args.device)
        arrays = {}
        for key in ('sampled_global_ids','sampled_seed_index'):
            arrays[key] = batch[key].detach().cpu().numpy()
        arrays['blocks'] = [{k: v.detach().cpu().numpy() for k,v in b.items()}
                            for b in batch['blocks']]
        arrays['_audit_cosine_relation_index'] = cosine
        for flag in ('targets','positive_mask','hard_y','prop_annotations',
                     'future_annotations','weak_modelout_prob','base_logits'):
            if flag in batch:arrays[flag] = None
        b = graph_schema_checks(rows, arrays, originals)
        observed.extend(b['batch_self_support_rows'])
        for k in accum:accum[k] += b[k]
        if originals is not None:
            known_originals += int(np.count_nonzero(originals[rows]>=0))
    report = dict(schema='latence-v094-graph-audit-v1', model_architecture_version='0.9.4',
                  runner_version='0.9.4', task=args.task, audited=n, num_targets=len(names),
                  original_global_ids_provided=originals is not None,
                  targets_with_known_original_id=known_originals,
                  batch_self_support_rows=observed[:500],
                  batch_self_support_hits=accum['batch_self_support_hits'],
                  self_support_status=('unresolved_no_original_global_ids' if originals is None
                    else ('detected_batch_union_support_risk' if observed else 'not_detected_in_audited_batch_unions')),
                  query_edges_noncosine=0, leaked_label_fields=0,
                  sampled_edges=accum['sampled_edges'], sampled_support_nodes=accum['support_nodes'],
                  inputs_sha256={'manifest':sha256(directory/'ind_test_input_manifest.json'),
                                 'protein_ids':sha256(directory/'protein_ids.txt')},
                  notes=('No future GO labels, weak teacher probabilities or B logits were supplied to the ghost query. '
                         'This audit does not remove historical weak training exposure and is NOT a strict '
                         'inductive/zero-shot certificate; self hits reflect batch-union support risk.'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f'Audit report exists: {args.out}; use --overwrite')
    from scripts.dark_weak.v094_native_adapter import atomic_json
    atomic_json(args.out, report, overwrite=args.overwrite)
    print(json.dumps({k:v for k,v in report.items() if k != 'batch_self_support_rows'}, indent=2, ensure_ascii=False))
    return report


def audit(args):
    lock_path = ROOT/'outputs/latence_nbs_experiments/v092/v092_evaluate.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A v092/v094 evaluation is already using the shared inference cache') from None
        return _audit_unlocked(args)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--task', default='bp', choices=('bp','mf','cc'))
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--original-global-ids', type=Path, help='Original registry global IDs aligned to prepared targets; -1 for unknown')
    p.add_argument('--limit', type=int, default=32)
    p.add_argument('--all-proteins', action='store_true')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args(argv)


if __name__ == '__main__':
    audit(main())
