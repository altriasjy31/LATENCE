"""Task-audited GO aliases over unchanged full-ontology geometry rows.

An old OWL export can retain a physical row for a GO ID which the task
registry already canonicalizes to another ID. The hash-bound task alignment
decides task resolution; physical rows are retained for geometry anchoring.
This module is independently loadable without importing torch or nbs_pg.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


_spec = importlib.util.spec_from_file_location(
    '_v093_go_registry_base', Path(__file__).with_name('go_registry_v090.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
normalize_go_id = _base.normalize_go_id
sha256_file = _base.sha256_file
mapping_sha256 = _base.mapping_sha256
ALIAS_POLICY = 'audited_task_canonical_over_physical_self_identity_v1'


class GORegistryV093(_base.GORegistryV090):
    SCHEMA = 'nbs-v093-go-registry-v1'

    def __init__(self, records, task_go_ids=(), task_to_ontology=None, *,
                 num_ontology_rows=None, artifact_hashes=None,
                 task_canonical_go_ids=None):
        originals = tuple(normalize_go_id(go) for go in task_go_ids)
        canonical = (originals if task_canonical_go_ids is None else
                     tuple(normalize_go_id(go) for go in task_canonical_go_ids))
        if len(originals) != len(canonical):
            raise ValueError('Task canonical identities must preserve every immutable task column')
        aliases = {}
        for original, target in zip(originals, canonical):
            if original in aliases and aliases[original] != target:
                raise ValueError('Conflicting audited task alias: ' + original)
            aliases[original] = target
        redirects = {key: value for key, value in aliases.items() if key != value}
        if any(target in redirects for target in redirects.values()):
            raise ValueError('Audited task aliases must not form a chain or cycle')
        if redirects and task_to_ontology is None:
            raise ValueError('Audited task aliases require an explicit aligned task mapping')
        self.task_canonical_go_ids = canonical
        self._audited_task_aliases = redirects
        records = [dict(record) for record in records]
        by_id = {}
        for record in records:
            identity = record.get('go_id') or record.get('normalized_go_id')
            if identity:
                by_id.setdefault(normalize_go_id(identity), []).append(record)
        for alias, target in redirects.items():
            if target not in by_id:
                raise ValueError('Task canonical GO is absent from full ontology geometry: ' + target)
            for record in by_id[target]:
                record['alt_ids'] = _base._ids(record.get('alt_ids', record.get('alt_id'))) + [alias]
        super().__init__(records, originals, task_to_ontology,
                         num_ontology_rows=num_ontology_rows,
                         artifact_hashes=artifact_hashes)
        if task_canonical_go_ids is None:
            self.task_canonical_go_ids = tuple(self.canonical_go_id(go) for go in originals)
        # Each canonical task ID must itself remain the final active identity.
        for target in set(self.task_canonical_go_ids):
            if self.canonical_go_id(target) != target:
                raise ValueError('Audited task canonical GO resolves to another identity: ' + target)

    def _add_alias(self, alias, canonical):
        previous = self._canonical.get(alias)
        if (previous == alias and alias != canonical and
                self._audited_task_aliases.get(alias) == canonical):
            # Keep self._row[alias] unchanged. Only string resolution follows
            # the task table, and the constructor checks every aligned column.
            self._canonical[alias] = canonical
            return
        super()._add_alias(alias, canonical)

    def contract(self):
        result = super().contract()
        result.update(alias_policy=ALIAS_POLICY,
                      task_canonical_go_ids=list(self.task_canonical_go_ids),
                      audited_task_aliases=dict(sorted(self._audited_task_aliases.items())),
                      physical_ontology_rows_preserved=True)
        return result

    @classmethod
    def from_manifests(cls, full_manifest, task_registry_path, alignment_manifest):
        full_path, task_path = Path(full_manifest).resolve(), Path(task_registry_path).resolve()
        payload = json.loads(full_path.read_text(encoding='utf-8'))
        registry_spec = payload.get('registry', {})
        registry_path = Path(registry_spec.get('file', ''))
        if not registry_path.is_absolute():
            registry_path = full_path.parent / registry_path
        if not registry_spec.get('sha256') or sha256_file(registry_path) != registry_spec['sha256']:
            raise ValueError('Full ontology registry hash mismatch or missing hash')
        with registry_path.open(encoding='utf-8', newline='') as stream:
            records = list(csv.DictReader(stream, delimiter='\t'))
        with task_path.open(encoding='utf-8', newline='') as stream:
            tasks = list(csv.DictReader(stream, delimiter='\t'))
        if not tasks or any(int(item.get('go_idx', -1)) != i for i, item in enumerate(tasks)):
            raise ValueError('Task registry columns must be contiguous and preserve order')
        hashes = {'full_manifest_sha256': sha256_file(full_path),
                  'ontology_registry_sha256': sha256_file(registry_path),
                  'task_registry_sha256': sha256_file(task_path)}
        alignment_path = Path(alignment_manifest).resolve()
        alignment = json.loads(alignment_path.read_text(encoding='utf-8'))
        if alignment.get('source', {}).get('go_registry_sha256') != hashes['task_registry_sha256']:
            raise ValueError('Alignment task registry hash mismatch')
        source = alignment.get('source', {}).get('checkpoint_sha256')
        if not source or source != payload.get('source', {}).get('checkpoint_sha256'):
            raise ValueError('Alignment/full ontology checkpoint hash mismatch')
        array = alignment['arrays']['source_row']
        array_path = alignment_path.parent / array['file']
        array_hash = sha256_file(array_path)
        if array.get('sha256') and array_hash != array['sha256']:
            raise ValueError('Alignment mapping hash mismatch')
        hashes.update(alignment_manifest_sha256=sha256_file(alignment_path),
                      alignment_mapping_file_sha256=array_hash)
        mapping = np.load(array_path, allow_pickle=False)
        return cls(records, [item.get('input_go_id') or item['go_id'] for item in tasks],
                   mapping, num_ontology_rows=int(payload['num_ontology_classes']),
                   artifact_hashes=hashes,
                   task_canonical_go_ids=[item['go_id'] for item in tasks])
