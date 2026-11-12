"""Hash-bound, ID-addressed frozen ESM2GO penultimate feature cache.

This reader has no torch dependency. Numeric registry rows never substitute
for protein identities, and absent features are errors rather than zeros.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'latence-esm-penultimate-v096-v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_ids(values, *, name='protein IDs'):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'US' or len(values) == 0:
        raise ValueError(name + ' must be a nonempty one-dimensional string array')
    result = [value.decode('utf-8') if isinstance(value, bytes) else str(value)
              for value in values.tolist()]
    if any(not value or value != value.strip() or '\n' in value or '\r' in value
           for value in result):
        raise ValueError(name + ' contain empty identities or whitespace')
    if len(set(result)) != len(result):
        raise ValueError(name + ' contain duplicate identities')
    return np.asarray(result, dtype=str)


def read_registry_protein_ids(path, expected_count=None):
    result = []
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'protein_idx', 'protein_id'}.issubset(reader.fieldnames or []):
            raise ValueError('Protein registry requires protein_idx and protein_id columns')
        for index, row in enumerate(reader):
            if int(row['protein_idx']) != index:
                raise ValueError('Registry protein_idx must be contiguous and row-aligned')
            result.append(row['protein_id'])
    identities = checked_ids(np.asarray(result, dtype=str), name='registry protein IDs')
    if expected_count is not None and len(identities) != expected_count:
        raise ValueError('Registry protein identity count differs from the graph registry')
    return identities


def _signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


class ESMPenultimateStoreV096:
    def __init__(self, manifest_path, *, expected_sha256=None):
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        actual = sha256_file(self.manifest_path)
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError('ESM feature manifest changed after configuration was frozen')
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('schema') != SCHEMA or self.manifest.get('complete') is not True:
            raise ValueError('A complete v096 penultimate feature manifest is required')
        identity = self.manifest.get('representation_identity')
        required = {'checkpoints_sha256', 'repr_layers', 'ensemble', 'feature_layer',
                    'dimensions', 'model_source_sha256'}
        if (not isinstance(identity, dict) or not required.issubset(identity)
                or identity['feature_layer'] != 'output_layer.0.input'
                or identity['ensemble'] not in ('single', 'concat')
                or not identity['checkpoints_sha256']):
            raise ValueError('ESM manifest lacks an explicit pre-classifier representation identity')
        self.paths = {}
        for key in ('features', 'protein_ids'):
            spec = self.manifest.get('files', {}).get(key, {})
            if not isinstance(spec.get('path'), str) or not spec.get('sha256'):
                raise ValueError('ESM manifest is missing a hash-bound file: ' + key)
            path = Path(spec['path'])
            path = path.resolve() if path.is_absolute() else (self.manifest_path.parent / path).resolve()
            if sha256_file(path) != spec['sha256']:
                raise ValueError('ESM cache file hash differs: ' + key)
            self.paths[key] = path
        self.features = np.load(self.paths['features'], mmap_mode='r', allow_pickle=False)
        self.protein_ids = checked_ids(np.load(self.paths['protein_ids'], allow_pickle=False),
                                       name='ESM cache protein IDs')
        self.feature_dim = self.manifest.get('feature_dim')
        if (type(self.feature_dim) is not int or self.feature_dim < 1
                or self.features.shape != (len(self.protein_ids), self.feature_dim)
                or self.features.dtype not in (np.dtype('float16'), np.dtype('float32'))
                or self.manifest.get('num_proteins') != len(self.protein_ids)
                or sum(identity['dimensions']) != self.feature_dim
                or len(identity['dimensions']) != len(identity['checkpoints_sha256'])):
            raise ValueError('ESM feature dimensions, dtype or identity segments differ')
        for start in range(0, len(self.features), 8192):
            if not np.isfinite(self.features[start:start + 8192]).all():
                raise ValueError('ESM cache contains nonfinite features')
        self._row_by_id = {name: row for row, name in enumerate(self.protein_ids.tolist())}
        self._signatures = {p: _signature(p) for p in (*self.paths.values(), self.manifest_path)}
        self._identity = {'manifest_sha256': actual,
                          'features_sha256': self.manifest['files']['features']['sha256'],
                          'protein_ids_sha256': self.manifest['files']['protein_ids']['sha256'],
                          'num_proteins': len(self.protein_ids), 'feature_dim': self.feature_dim,
                          'storage_dtype': str(self.features.dtype),
                          'representation_identity': deepcopy(identity)}

    def assert_unchanged(self):
        if any(_signature(path) != signature for path, signature in self._signatures.items()):
            raise ValueError('Immutable ESM cache changed while in use')

    def identity(self):
        self.assert_unchanged()
        return deepcopy(self._identity)

    def rows_for(self, protein_ids):
        self.assert_unchanged()
        names = [str(value) for value in protein_ids]
        missing = [name for name in names if name not in self._row_by_id]
        if missing:
            raise ValueError(f'ESM cache is missing {len(missing)} requested protein identities; '
                             f'first: {missing[:8]}')
        return np.asarray([self._row_by_id[name] for name in names], dtype=np.int64)

    def gather_rows(self, rows):
        self.assert_unchanged()
        rows = np.asarray(rows)
        if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or (rows < 0).any()
                or (rows >= len(self.features)).any()):
            raise ValueError('ESM cache row indices must be valid one-dimensional integers')
        values = np.asarray(self.features[rows], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite ESM features gathered')
        return values

    def gather(self, protein_ids):
        return self.gather_rows(self.rows_for(protein_ids))
