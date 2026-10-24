"""Hash-bound GO identities and target rows; this module never creates geometry.

The immutable classifier column order may contain several aliases of one GO.
Dynamic targets resolve through an explicit ontology registry into its existing
rows. Replacing an ontology embedding with independently fitted coordinates is
not made valid by this identity registry.
"""
from __future__ import annotations

import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(mapping):
    return hashlib.sha256(np.asarray(mapping, dtype="<i8").tobytes()).hexdigest()


def normalize_go_id(value):
    if not isinstance(value, str):
        raise ValueError("GO identity must be a string")
    text = value.strip()
    match = re.fullmatch(r"(?:https?://purl\.obolibrary\.org/obo/)?GO[_:](\d{7})", text, re.I)
    if not match:
        raise ValueError(f"invalid GO identifier: {value!r}")
    return "GO:" + match.group(1)


def _ids(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = re.split(r"[;,|\s]+", value.strip())
    if not isinstance(value, (list, tuple)):
        raise ValueError("GO alias/replacement values must be a list or delimited string")
    return [normalize_go_id(item) for item in value if item]


def _boolean(value):
    if value in (False, 0, "", "0", "false", "False", None):
        return False
    if value in (True, 1, "1", "true", "True"):
        return True
    raise ValueError("is_obsolete must be an explicit boolean")


class GORegistryV090:
    """Canonical/alt-ID resolution with immutable task columns and row hashes."""
    SCHEMA = "nbs-v090-go-registry-v1"

    def __init__(self, records, task_go_ids=(), task_to_ontology=None, *,
                 num_ontology_rows=None, artifact_hashes=None):
        if (num_ontology_rows is not None and (isinstance(num_ontology_rows, bool)
                or not isinstance(num_ontology_rows, Integral) or num_ontology_rows < 1)):
            raise ValueError("num_ontology_rows must be a positive integer")
        self._row = {}
        self._canonical = {}
        self._obsolete = {}
        records = list(records)
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("GO registry records must be mappings")
            identity = record.get("go_id") or record.get("normalized_go_id")
            if not identity:
                continue  # Explicit non-GO ontology classes remain legal rows.
            go = normalize_go_id(identity)
            row = record.get("ontology_row", record.get("ontology_go_idx"))
            if (isinstance(row, (bool, np.bool_)) or
                    not (isinstance(row, Integral) or isinstance(row, str) and re.fullmatch(r"\d+", row))):
                raise ValueError("ontology row must be a nonnegative integer")
            try:
                row = int(row)
            except (ValueError, TypeError) as exc:
                raise ValueError("ontology row must be a nonnegative integer") from exc
            if row < 0 or (num_ontology_rows is not None and row >= num_ontology_rows):
                raise ValueError("GO row outside supplied ontology geometry")
            obsolete = _boolean(record.get("is_obsolete", False))
            replacements = tuple(_ids(record.get("replaced_by")))
            if go in self._row and (self._row[go] != row or self._obsolete[go] != (obsolete, replacements)):
                raise ValueError(f"conflicting duplicate GO record: {go}")
            self._row[go] = row
            self._obsolete[go] = obsolete, replacements
        if not self._row:
            raise ValueError("ontology registry has no GO identities")
        self.num_ontology_rows = (max(self._row.values()) + 1 if num_ontology_rows is None
                                  else int(num_ontology_rows))
        for go in self._row:
            self._canonical[go] = go
        for record in records:
            identity = record.get("go_id") or record.get("normalized_go_id")
            if not identity:
                continue
            go = normalize_go_id(identity)
            for alt in _ids(record.get("alt_ids", record.get("alt_id"))):
                self._add_alias(alt, go)
        self.task_go_ids = tuple(normalize_go_id(go) for go in task_go_ids)
        derived = np.asarray([self.resolve_one(go) for go in self.task_go_ids], dtype=np.int64)
        if task_to_ontology is not None:
            raw = np.asarray(task_to_ontology)
            if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.shape != derived.shape:
                raise ValueError("task_to_ontology must preserve all immutable task columns")
            if not np.array_equal(raw, derived):
                raise ValueError("task GO registry disagrees with task_to_ontology mapping")
        self.task_to_ontology = derived
        self.task_to_ontology.setflags(write=False)
        self.artifact_hashes = dict(artifact_hashes or {})
        identity_payload = [(go, self._row[go], self._obsolete[go]) for go in sorted(self._row)]
        self.identity_sha256 = hashlib.sha256(json.dumps(
            {"rows": identity_payload, "aliases": sorted(self._canonical.items()),
             "task_go_ids": self.task_go_ids, "num_rows": self.num_ontology_rows},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _add_alias(self, alias, canonical):
        if alias in self._canonical and self._canonical[alias] != canonical:
            raise ValueError(f"conflicting GO alias: {alias}")
        self._canonical[alias] = canonical

    def canonical_go_id(self, identity, *, allow_obsolete_replacement=False):
        go = normalize_go_id(identity)
        if go not in self._canonical:
            raise ValueError(f"GO has no supplied ontology geometry: {go}")
        go = self._canonical[go]
        visited = set()
        while self._obsolete[go][0]:
            if not allow_obsolete_replacement:
                raise ValueError(f"obsolete GO requires explicit replacement policy: {go}")
            if go in visited:
                raise ValueError("obsolete GO replacement cycle")
            visited.add(go)
            replacements = self._obsolete[go][1]
            if len(replacements) != 1 or replacements[0] not in self._canonical:
                raise ValueError(f"ambiguous or missing obsolete replacement: {go}")
            go = self._canonical[replacements[0]]
        return go

    def resolve_one(self, identity, *, allow_obsolete_replacement=False):
        return self._row[self.canonical_go_id(identity,
            allow_obsolete_replacement=allow_obsolete_replacement)]

    def resolve_target_go(self, go_ids: Sequence[str], *, allow_obsolete_replacement=False):
        if isinstance(go_ids, str) or not isinstance(go_ids, (list, tuple, np.ndarray)):
            raise ValueError("target GO identities must be a nonempty ordered sequence")
        if not len(go_ids):
            raise ValueError("target GO identities must be nonempty")
        return np.asarray([self.resolve_one(go, allow_obsolete_replacement=allow_obsolete_replacement)
                           for go in go_ids], dtype=np.int64)

    def contract(self):
        return {"schema": self.SCHEMA, "identity_sha256": self.identity_sha256,
                "artifact_hashes": dict(self.artifact_hashes),
                "num_ontology_rows": self.num_ontology_rows,
                "num_task_columns": len(self.task_go_ids),
                "task_to_ontology_sha256": mapping_sha256(self.task_to_ontology),
                "geometry_policy": "existing_hash_bound_ontology_rows_only"}

    @classmethod
    def from_manifests(cls, full_manifest, task_registry_path, alignment_manifest=None):
        full_path = Path(full_manifest).resolve()
        payload = json.loads(full_path.read_text(encoding="utf-8"))
        spec = payload.get("registry", {})
        registry_path = Path(spec.get("file", ""))
        if not registry_path.is_absolute():
            registry_path = full_path.parent / registry_path
        if not spec.get("sha256") or sha256_file(registry_path) != spec["sha256"]:
            raise ValueError("full ontology registry hash mismatch or missing hash")
        with registry_path.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream, delimiter="\t"))
        task_path = Path(task_registry_path).resolve()
        with task_path.open(encoding="utf-8", newline="") as stream:
            tasks = list(csv.DictReader(stream, delimiter="\t"))
        if not tasks or any(int(item.get("go_idx", -1)) != i for i, item in enumerate(tasks)):
            raise ValueError("task registry columns must be contiguous and preserve order")
        # Task registry's audited canonical identity supplies explicit alt-ID links.
        for item in tasks:
            canonical = normalize_go_id(item["go_id"])
            original = normalize_go_id(item.get("input_go_id") or item["go_id"])
            if original != canonical:
                matching = next((r for r in records if r.get("normalized_go_id", r.get("go_id")) == canonical), None)
                if matching is None:
                    raise ValueError("task canonical GO is absent from full ontology registry")
                matching["alt_ids"] = _ids(matching.get("alt_ids")) + [original]
        hashes = {"full_manifest_sha256": sha256_file(full_path),
                  "ontology_registry_sha256": sha256_file(registry_path),
                  "task_registry_sha256": sha256_file(task_path)}
        mapping = None
        if alignment_manifest is not None:
            alignment_path = Path(alignment_manifest).resolve()
            alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
            if alignment.get("source", {}).get("go_registry_sha256") != hashes["task_registry_sha256"]:
                raise ValueError("alignment task registry hash mismatch")
            source = alignment.get("source", {}).get("checkpoint_sha256")
            full_source = payload.get("source", {}).get("checkpoint_sha256")
            if not source or source != full_source:
                raise ValueError("alignment/full ontology checkpoint hash mismatch")
            array = alignment["arrays"]["source_row"]
            array_path = alignment_path.parent / array["file"]
            if array.get("sha256") and sha256_file(array_path) != array["sha256"]:
                raise ValueError("alignment mapping hash mismatch")
            mapping = np.load(array_path, allow_pickle=False)
            hashes["alignment_manifest_sha256"] = sha256_file(alignment_path)
            hashes["alignment_mapping_file_sha256"] = sha256_file(array_path)
        return cls(records, [item.get("input_go_id") or item["go_id"] for item in tasks],
                   mapping, num_ontology_rows=int(payload["num_ontology_classes"]),
                   artifact_hashes=hashes)
