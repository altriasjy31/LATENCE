#!/usr/bin/env python3
"""Prepare exact protein identities for frozen ESM penultimate extraction.

All registry nodes are included, not just training seeds: sampled neighbors also
need ESM features. Inference IDs are included without reading any test labels.
Optionally collect missing raw-embedding sequences from user-supplied FASTA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checked_ids(ids, source):
    ids = list(ids)
    if not ids or any(not x or x != x.strip() or any(c.isspace() for c in x)
                      or '/' in x or '\\' in x or x in ('.', '..') for x in ids):
        raise ValueError(f'Empty or unsafe protein ID in {source}')
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate protein ID in {source}')
    return ids


def registry_ids(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'protein_idx', 'protein_id', 'role', 'role_row_idx'} <= set(reader.fieldnames or []):
            raise ValueError('Registry needs protein_idx, protein_id, role, role_row_idx')
        ids = []
        for expected, row in enumerate(reader):
            if int(row['protein_idx']) != expected:
                raise ValueError('Registry indices must be contiguous and row-aligned')
            ids.append(row['protein_id'])
    return checked_ids(ids, path)


def fasta_records(path):
    name, seq = None, []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(seq)
                parts = line[1:].split()
                if not parts:
                    raise ValueError(f'Empty FASTA header: {path}')
                name, seq = parts[0], []
            else:
                if name is None:
                    raise ValueError(f'Sequence before FASTA header: {path}')
                seq.append(line)
    if name is not None:
        yield name, ''.join(seq)


def resolve_registry(args):
    if args.registry:
        return args.registry.resolve(strict=True)
    path = args.source_work_dir.resolve(strict=True) / 'resolved_config.json'
    config = json.loads(path.read_text())
    if config.get('release_version') != '0.9.5':
        raise ValueError('Expected a v095 resolved_config.json')
    manifest_path = Path(config['data']['unified_dataset_manifest']).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    spec = manifest.get('sources', {}).get('protein_registry')
    if not spec:
        raise ValueError('Derived dataset lacks a hash-bound protein_registry source')
    registry = Path(spec['path'])
    if not registry.is_absolute():
        registry = manifest_path.parent / registry
    registry = registry.resolve(strict=True)
    if digest(registry) != spec['sha256']:
        raise ValueError('Registry changed since v094/v095 dataset construction')
    return registry


def prepare(args):
    registry = resolve_registry(args)
    ids = registry_ids(registry)
    sources = [dict(kind='registry', path=str(registry), sha256=digest(registry), rows=len(ids))]
    seen = set(ids)
    for inp in args.input_dir:
        path = inp.resolve(strict=True) / 'protein_ids.txt'
        more = checked_ids(path.read_text(encoding='utf-8').splitlines(), path)
        sources.append(dict(kind='inference', path=str(path), sha256=digest(path), rows=len(more)))
        for name in more:
            if name not in seen:
                ids.append(name)
                seen.add(name)
    dirs = [p.resolve(strict=True) for p in args.features_dir]
    missing = [name for name in ids if not any((d / (name + '.npy')).is_file() for d in dirs)] if dirs else ids[:]
    wanted, sequences = set(missing), {}
    fasta_sources = []
    for path in args.fasta:
        path = path.resolve(strict=True)
        fasta_sources.append(dict(path=str(path), sha256=digest(path)))
        for name, sequence in fasta_records(path):
            if name not in wanted:
                continue
            if not sequence or any(c.isspace() for c in sequence):
                raise ValueError(f'Invalid FASTA sequence: {name}')
            if name in sequences and sequences[name] != sequence:
                raise ValueError(f'Conflicting sequences for protein ID: {name}')
            sequences[name] = sequence
    if args.fasta and wanted - set(sequences):
        absent = sorted(wanted - set(sequences))
        raise ValueError(f'{len(absent)} missing proteins absent from supplied FASTA; examples={absent[:10]}')
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.' + out.name + '-', dir=out.parent))
    try:
        (temp / 'protein_ids.txt').write_text('\n'.join(ids) + '\n', encoding='utf-8')
        (temp / 'missing_embedding_ids.txt').write_text(''.join(x + '\n' for x in missing), encoding='utf-8')
        if args.fasta:
            with (temp / 'missing_embeddings.fasta').open('w', encoding='utf-8') as stream:
                for name in missing:
                    stream.write(f'>{name}\n{sequences[name]}\n')
        report = dict(schema='nbs-v096-esm-request-v1', num_proteins=len(ids), sources=sources,
                      num_missing_embeddings=len(missing), features_dirs=[str(d) for d in dirs],
                      fasta_sources=fasta_sources, identity_rule='exact protein_id; no accession normalization',
                      raw_cache_check='existence only; exporter validates dimensions, finiteness and content',
                      test_labels_read=False, protein_ids_sha256=digest(temp / 'protein_ids.txt'))
        (temp / 'request_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        os.replace(temp, out)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print(json.dumps(dict(output_dir=str(out), num_proteins=len(ids), num_missing_embeddings=len(missing))))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--registry', type=Path)
    source.add_argument('--source-work-dir', type=Path)
    p.add_argument('--input-dir', type=Path, action='append', default=[], help='Repeat for each inference set')
    p.add_argument('--features-dir', type=Path, action='append', default=[], help='Existing raw ESM caches; repeat as needed')
    p.add_argument('--fasta', type=Path, action='append', default=[], help='Optional source FASTAs for missing embedding sequences')
    p.add_argument('--output-dir', type=Path, required=True)
    return p


if __name__ == '__main__':
    prepare(parser().parse_args())
