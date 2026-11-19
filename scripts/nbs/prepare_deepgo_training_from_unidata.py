#!/usr/bin/env python3
"""Preserve old unidata train/validation splits and labels for an external baseline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

NAMESPACES = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
GO = re.compile(r'GO:\d{7}')
IPR = re.compile(r'IPR\d{6}')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def axis(values, name):
    values = list(values)
    if not values or any(not isinstance(x, str) or not x or x != x.strip() for x in values):
        raise ValueError(f'{name}: expected nonempty string IDs')
    if len(values) != len(set(values)):
        raise ValueError(f'{name}: duplicate IDs')
    return values


def sequence(value):
    if not isinstance(value, str):
        raise ValueError('Missing/nonstring sequence')
    value = ''.join(value.split()).upper().rstrip('*')
    if not re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWYBXZJUO]+', value):
        raise ValueError('Invalid protein sequence')
    return value


def items(value, name):
    if isinstance(value, (str, bytes, dict)) or value is None:
        raise ValueError(f'{name}: expected a list/set/tuple/1D array, not {type(value).__name__}')
    try:
        result = list(value)
    except TypeError as e:
        raise ValueError(f'{name}: expected iterable values') from e
    return result


def block(metadata, split, namespace):
    splits = metadata.get(split)
    if not isinstance(splits, dict):
        raise ValueError(f'Expected metadata[{split!r}] mapping; available splits: {list(metadata)}')
    keys = [k for k in (namespace, NAMESPACES[namespace], namespace.upper()) if k in splits]
    if len(keys) != 1 or not isinstance(splits[keys[0]], (dict, pd.DataFrame)):
        raise ValueError(f'Expected one {namespace} block in {split}; available keys: {list(splits)}')
    return splits[keys[0]]


def aligned(values, proteins, name):
    if isinstance(values, dict):
        if set(values) != set(proteins):
            raise ValueError(f'{name}: ID keys differ from protein axis')
        return [values[p] for p in proteins]
    if len(values) != len(proteins):
        raise ValueError(f'{name}: protein/value length mismatch')
    return list(values)


def make_frame(data, proteins, table, terms, args):
    if args.label_field not in data:
        raise ValueError(f'Missing label field {args.label_field!r}; available fields: {list(data)}')
    raw = data[args.label_field]
    if hasattr(raw, 'tocsr') and args.label_representation != 'multihot':
        raise ValueError('Sparse label matrices require --label-representation multihot')
    if args.label_representation == 'multihot':
        if args.label_format != 'indices' or getattr(raw, 'shape', None) != (len(proteins), len(terms)):
            raise ValueError('multihot requires --label-format indices and shape (n_proteins, n_terms)')
        if hasattr(raw, 'tocsr'):
            matrix = raw.tocsr(copy=True)
            matrix.sum_duplicates()
            if not np.isin(matrix.data, [0, 1]).all():
                raise ValueError('Multihot sparse values must be binary')
            matrix.eliminate_zeros()
            raw = [matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]] for i in range(len(proteins))]
        else:
            matrix = np.asarray(raw)
            if not np.isin(matrix, [0, 1]).all():
                raise ValueError('Multihot values must be binary')
            raw = [np.flatnonzero(row) for row in matrix]
    labels = aligned(raw, proteins, args.label_field)
    sequences = [sequence(table.loc[p, 'sequences']) for p in proteins]
    for field in ('sequences', 'sequence'):
        if field in data:
            original = aligned(data[field], proteins, field)
            if [sequence(s) for s in original] != sequences:
                raise ValueError(f'{field}: metadata and old protein table sequence mismatch')
    term_set, records, excluded = set(terms), [], 0
    for protein, seq, values in zip(proteins, sequences, labels):
        values = items(values, f'{protein}/{args.label_field}')
        if args.label_format == 'indices':
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values):
                raise ValueError(f'{protein}: indices must be integers (not bool/float/string)')
            if any(v < 0 or v >= len(terms) for v in values):
                raise ValueError(f'{protein}: label index outside supplied terms axis')
            if (args.label_representation == 'lists' and len(values) == len(terms)
                    and all(v in (0, 1) for v in values)):
                raise ValueError(f'{protein}: ambiguous binary vector; use --label-representation multihot')
            values = [terms[int(v)] for v in values]
        elif any(not isinstance(v, str) or GO.fullmatch(v) is None for v in values):
            raise ValueError(f'{protein}: expected GO:NNNNNNN IDs; use --label-format indices for indices')
        seen = set(values)
        excluded += len(seen - term_set)
        kept = [g for g in dict.fromkeys(values) if g in term_set]
        iprs = items(table.loc[protein, 'interpros'], f'{protein}/interpros')
        if any(not isinstance(v, str) or IPR.fullmatch(v) is None for v in iprs):
            raise ValueError(f'{protein}: invalid InterPro IDs')
        records.append(dict(proteins=protein, sequences=seq, interpros=sorted(set(iprs)), prop_annotations=kept))
    return pd.DataFrame(records), excluded


def run(args):
    out = Path(args.output_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Choose a new output directory: {out}')
    terms = axis(Path(args.terms_file).read_text().splitlines(), 'terms')
    if any(GO.fullmatch(g) is None for g in terms):
        raise ValueError('terms-file must contain one canonical GO:NNNNNNN per line, no header')
    metadata = pd.read_pickle(args.metadata)
    if not isinstance(metadata, dict):
        raise ValueError('Expected nested unidata mapping')
    valid = args.valid_split
    if valid is None:
        matches = [s for s in ('valid', 'val', 'validation') if s in metadata]
        if len(matches) != 1:
            raise ValueError(f'Specify --valid-split; auto candidates={matches}, available={list(metadata)}')
        valid = matches[0]
    if valid == args.train_split:
        raise ValueError('Train and validation must be distinct existing splits')
    table = pd.read_pickle(args.protein_table)
    if not isinstance(table, pd.DataFrame) or any(c not in table for c in ('proteins', 'sequences', 'interpros')):
        raise ValueError('Old protein table needs DataFrame columns proteins, sequences, interpros')
    axis(table.proteins, 'old protein table')
    table = table.set_index('proteins', drop=False)
    frames, counts = {}, {}
    for target, split in [('train', args.train_split), ('valid', valid)]:
        data = block(metadata, split, args.namespace)
        if 'proteins' not in data:
            raise ValueError(f'{split}: missing proteins; available fields: {list(data)}')
        proteins = axis(data['proteins'], split)
        missing = set(proteins) - set(table.index)
        if missing:
            raise ValueError(f'{split}: {len(missing)} proteins absent from old table, e.g. {sorted(missing)[:5]}')
        frames[target], excluded = make_frame(data, proteins, table, terms, args)
        counts[target] = dict(split=split, n_proteins=len(proteins), excluded_nonvocabulary_pairs=excluded,
                              kept_pairs=int(frames[target].prop_annotations.map(len).sum()),
                              sequence_verified_against_metadata=any(k in data for k in ('sequences', 'sequence')))
    for field in ('proteins', 'sequences'):
        if set(frames['train'][field]).intersection(frames['valid'][field]):
            raise ValueError(f'Train and validation overlap in {field}; existing splits are not modified')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare_deepgo.', dir=out.parent) as temp:
        stage = Path(temp) / 'output'; stage.mkdir()
        for name, frame in frames.items():
            frame.to_pickle(stage / f'{name}.pkl')
        pd.DataFrame({'gos': terms}).to_pickle(stage / 'terms.pkl')
        manifest = dict(schema='latence-external-training-preparation-v1', complete=True, namespace=args.namespace,
                        label_source='Only the selected old unidata field; no protein-table labels are read',
                        sequence_normalization='Remove whitespace, uppercase, trim terminal stop characters',
                        args=vars(args), split_inventory=counts,
                        sources={k: {'path': str(Path(getattr(args, k)).resolve()), 'sha256': sha(getattr(args, k))}
                                 for k in ('metadata', 'protein_table', 'terms_file')},
                        outputs={p.name: sha(p) for p in stage.iterdir()})
        (stage / 'prepare_training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(out)
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'protein-table', 'terms-file', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--namespace', choices=NAMESPACES, default='bp')
    p.add_argument('--label-field', default='prop_annotations')
    p.add_argument('--label-format', choices=['go-ids', 'indices'], required=True)
    p.add_argument('--label-representation', choices=['lists', 'multihot'], default='lists')
    p.add_argument('--train-split', default='train')
    p.add_argument('--valid-split')
    run(p.parse_args())
