#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build relation-specific GO--GO direct and transitive-closure edges.

The model GO vocabulary is supplied in classifier-index order.  Edges are
computed on the full OBO graph and are then projected back to that vocabulary,
so an unregistered intermediate GO term does not break an ancestor path.

Different classifier columns may resolve to the same current GO term (for
example, an old primary ID and a current ``alt_id``).  Those columns are never
dropped, because their row positions must stay aligned with the first-stage
classifier.  ``--duplicate-policy`` controls whether ontology edges are
replicated to every such row, routed only through a representative row, or
rejected for strict auditing.

Four independently routable edge sources are written:

    child --is_a-----> parent/ancestor
    parent --has_child-> child/descendant
    part --part_of---> whole/transitive whole
    whole --has_part-> part/transitive part

Each ``.npy`` file is an int32 matrix with columns
``[src_go_idx, dst_go_idx, hop_distance, is_direct]``.  Direct and closure
edges coexist in the same relation file; ``is_direct == 1`` selects strict
one-hop ontology edges.  For inverse files, rows with ``hop_distance > 1`` are
the inverse transport relation to descendants/transitive parts rather than a
claim that the destination is an immediate child/part.

``is_a`` and ``part_of`` closures are deliberately computed separately.  Mixed
paths are not assigned a misleading single ontology relation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


GO_ID_RE = re.compile(r"^GO:\d{7}$")
GO_ID_SEARCH_RE = re.compile(r"GO:\d{7}")

TASK_NAMESPACES = {
    "bp": "biological_process",
    "biological_process": "biological_process",
    "mf": "molecular_function",
    "molecular_function": "molecular_function",
    "cc": "cellular_component",
    "cellular_component": "cellular_component",
}


@dataclass
class GOTerm:
    go_id: str
    name: str = ""
    namespace: str = ""
    alt_ids: List[str] = field(default_factory=list)
    is_a: List[str] = field(default_factory=list)
    part_of: List[str] = field(default_factory=list)
    is_obsolete: bool = False
    replaced_by: List[str] = field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build relation-specific GO--GO closure edges in classifier index space"
    )
    p.add_argument("--obo", type=Path, required=True, help="GO OBO file used by the experiment.")
    p.add_argument(
        "--go-terms",
        type=Path,
        required=True,
        help="Ordered model GO vocabulary (.txt/.tsv/.csv/.json/.npy/.pkl/.pt).",
    )
    p.add_argument(
        "--go-terms-key",
        type=str,
        default=None,
        help="Optional dotted key inside JSON/pickle/torch input.",
    )
    p.add_argument("--task", choices=sorted(TASK_NAMESPACES), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--expected-num-terms",
        type=int,
        default=None,
        help="Fail unless vocabulary length matches the first-stage classifier.",
    )
    p.add_argument(
        "--include-part-of",
        dest="include_part_of",
        action="store_true",
        help="Build part_of/has_part as separate sources (default).",
    )
    p.add_argument(
        "--no-part-of",
        dest="include_part_of",
        action="store_false",
        help="Build only is_a/has_child.",
    )
    p.set_defaults(include_part_of=True)
    p.add_argument(
        "--max-hops",
        type=int,
        default=0,
        help="0 computes the complete closure; a positive value truncates traversal.",
    )
    p.add_argument(
        "--obsolete-policy",
        choices=["error", "keep", "replace"],
        default="error",
        help="How to handle obsolete IDs in the model vocabulary.",
    )
    p.add_argument(
        "--duplicate-policy",
        choices=["error", "representative", "replicate", "preserve"],
        default="error",
        help=(
            "How to handle multiple classifier indices that canonicalize to the same "
            "GO term. 'error' fails safely; 'representative' routes edges only through "
            "the first classifier index in each group; 'replicate' copies the same "
            "ontology context to every classifier index. The deprecated 'preserve' "
            "name is accepted as an alias of 'replicate'. Vocabulary length is never changed."
        ),
    )
    p.add_argument(
        "--allow-missing-terms",
        action="store_true",
        help="Keep vocabulary IDs absent from the OBO as zero-degree nodes.",
    )
    p.add_argument(
        "--allow-cross-namespace-terms",
        action="store_true",
        help="Allow vocabulary terms outside the requested task namespace.",
    )
    p.add_argument(
        "--allow-pickle-input",
        action="store_true",
        help="Allow trusted .pkl/.pickle/.pt vocabulary inputs.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_ordered_ids(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for go_id in ids:
        h.update(go_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def ensure_writable(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        preview = "\n  ".join(existing[:12])
        raise FileExistsError(
            f"Output already exists; use --overwrite to replace it:\n  {preview}"
        )


def parse_obo(path: Path) -> Dict[str, GOTerm]:
    """Parse fields needed for GO graph construction without extra packages."""
    terms: Dict[str, GOTerm] = {}
    stanza: Dict[str, List[str]] | None = None
    stanza_type: str | None = None

    def commit() -> None:
        nonlocal stanza, stanza_type
        if stanza_type != "Term" or stanza is None:
            stanza = None
            stanza_type = None
            return
        ids = stanza.get("id", [])
        if len(ids) != 1 or not GO_ID_RE.fullmatch(ids[0]):
            stanza = None
            stanza_type = None
            return
        go_id = ids[0]
        if go_id in terms:
            raise ValueError(f"Duplicate [Term] stanza for {go_id}")

        is_a: List[str] = []
        for value in stanza.get("is_a", []):
            match = GO_ID_SEARCH_RE.search(value)
            if match:
                is_a.append(match.group(0))

        part_of: List[str] = []
        for value in stanza.get("relationship", []):
            fields = value.split()
            if len(fields) >= 2 and fields[0] == "part_of" and GO_ID_RE.fullmatch(fields[1]):
                part_of.append(fields[1])

        replaced_by: List[str] = []
        for value in stanza.get("replaced_by", []):
            match = GO_ID_SEARCH_RE.search(value)
            if match:
                replaced_by.append(match.group(0))

        terms[go_id] = GOTerm(
            go_id=go_id,
            name=(stanza.get("name") or [""])[0],
            namespace=(stanza.get("namespace") or [""])[0],
            alt_ids=[x for x in stanza.get("alt_id", []) if GO_ID_RE.fullmatch(x)],
            is_a=sorted(set(is_a)),
            part_of=sorted(set(part_of)),
            is_obsolete=(stanza.get("is_obsolete") or ["false"])[0].lower() == "true",
            replaced_by=sorted(set(replaced_by)),
        )
        stanza = None
        stanza_type = None

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            if line.startswith("[") and line.endswith("]"):
                commit()
                stanza_type = line[1:-1]
                stanza = defaultdict(list)
                continue
            if stanza is None or not line or line.startswith("!"):
                continue
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            stanza[key].append(value.strip())
    commit()
    if not terms:
        raise ValueError(f"No GO [Term] stanzas found in {path}")
    return terms


def _resolve_dotted_key(obj: Any, dotted_key: str) -> Any:
    value = obj
    for component in dotted_key.split("."):
        if isinstance(value, Mapping):
            if component not in value:
                raise KeyError(f"Missing component {component!r} in --go-terms-key={dotted_key!r}")
            value = value[component]
        else:
            if not hasattr(value, component):
                raise AttributeError(
                    f"Missing attribute {component!r} in --go-terms-key={dotted_key!r}"
                )
            value = getattr(value, component)
    return value


def _sequence_of_go_ids(value: Any) -> List[str] | None:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            return None
        value = value.tolist()
    if hasattr(value, "classes_"):
        return _sequence_of_go_ids(getattr(value, "classes_"))
    if not isinstance(value, (list, tuple)):
        return None
    ids = [str(x).strip() for x in value]
    if ids and all(GO_ID_RE.fullmatch(x) for x in ids):
        return ids
    return None


def _mapping_as_go_ids(value: Any) -> List[str] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    if all(GO_ID_RE.fullmatch(str(k)) for k in value):
        try:
            pairs = sorted((int(v), str(k)) for k, v in value.items())
        except (TypeError, ValueError):
            return None
        if [idx for idx, _ in pairs] != list(range(len(pairs))):
            return None
        return [go_id for _, go_id in pairs]
    if all(GO_ID_RE.fullmatch(str(v)) for v in value.values()):
        try:
            pairs = sorted((int(k), str(v)) for k, v in value.items())
        except (TypeError, ValueError):
            return None
        if [idx for idx, _ in pairs] != list(range(len(pairs))):
            return None
        return [go_id for _, go_id in pairs]
    return None


def _find_go_sequences(
    obj: Any,
    *,
    path: str = "$",
    depth: int = 0,
    max_depth: int = 6,
    seen: set[int] | None = None,
) -> List[Tuple[str, List[str]]]:
    if seen is None:
        seen = set()
    if depth > max_depth:
        return []
    oid = id(obj)
    if oid in seen:
        return []
    seen.add(oid)

    direct = _sequence_of_go_ids(obj)
    if direct is not None:
        return [(path, direct)]
    mapped = _mapping_as_go_ids(obj)
    if mapped is not None:
        return [(path, mapped)]

    found: List[Tuple[str, List[str]]] = []
    if isinstance(obj, Mapping):
        priority = {
            "go_terms",
            "terms",
            "classes",
            "classes_",
            "labels",
            "idx_to_go",
            "go_to_idx",
            *TASK_NAMESPACES.keys(),
        }
        items = sorted(obj.items(), key=lambda kv: (str(kv[0]) not in priority, str(kv[0])))
        for key, value in items:
            found.extend(
                _find_go_sequences(
                    value,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                    max_depth=max_depth,
                    seen=seen,
                )
            )
    elif hasattr(obj, "classes_"):
        found.extend(
            _find_go_sequences(
                getattr(obj, "classes_"),
                path=f"{path}.classes_",
                depth=depth + 1,
                max_depth=max_depth,
                seen=seen,
            )
        )
    return found


def load_text_go_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = [
            line.rstrip("\n\r")
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not raw_rows:
        return []

    delimiter = "\t" if "\t" in raw_rows[0] else "," if "," in raw_rows[0] else None
    if delimiter is not None:
        parsed = list(csv.reader(raw_rows, delimiter=delimiter))
        header = [x.strip().lower() for x in parsed[0]]
        go_names = {"go_id", "go_term", "term_id", "go"}
        idx_names = {"go_idx", "class_idx", "index", "idx"}
        go_columns = [i for i, name in enumerate(header) if name in go_names]
        if len(go_columns) == 1:
            go_col = go_columns[0]
            idx_columns = [i for i, name in enumerate(header) if name in idx_names]
            indexed: List[Tuple[int, str]] = []
            ids: List[str] = []
            for row_number, row in enumerate(parsed[1:], start=2):
                if go_col >= len(row):
                    raise ValueError(f"Missing GO column in {path}:{row_number}")
                value = row[go_col].strip()
                if not GO_ID_RE.fullmatch(value):
                    raise ValueError(f"Invalid GO ID in {path}:{row_number}: {value!r}")
                if idx_columns:
                    idx_col = idx_columns[0]
                    if idx_col >= len(row):
                        raise ValueError(f"Missing index column in {path}:{row_number}")
                    indexed.append((int(row[idx_col]), value))
                else:
                    ids.append(value)
            if indexed:
                indexed.sort()
                indices = [idx for idx, _ in indexed]
                if indices != list(range(len(indices))):
                    raise ValueError(
                        f"GO index column in {path} must be exactly 0..N-1, got "
                        f"{indices[:10]}...{indices[-10:]}"
                    )
                return [go_id for _, go_id in indexed]
            return ids

    ids = []
    for line_number, line in enumerate(raw_rows, start=1):
        matches = GO_ID_SEARCH_RE.findall(line)
        if not matches:
            if line_number == 1 or "go" in line.lower():
                continue
            raise ValueError(f"No GO ID found in {path}:{line_number}: {line!r}")
        if len(matches) != 1:
            raise ValueError(f"Expected one GO ID in {path}:{line_number}, found {matches}")
        ids.append(matches[0])
    return ids


def load_ordered_go_ids(
    path: Path,
    *,
    dotted_key: str | None,
    allow_pickle_input: bool,
) -> Tuple[List[str], str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".tsv", ".csv"}:
        return load_text_go_ids(path), "text"
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        ids = _sequence_of_go_ids(arr)
        if ids is None:
            raise ValueError(f"Expected a 1-D string GO array in {path}, got {arr.shape}/{arr.dtype}")
        return ids, "numpy"
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        loader_name = "json"
    elif suffix in {".pkl", ".pickle"}:
        if not allow_pickle_input:
            raise ValueError(
                "Refusing pickle input by default. Use --allow-pickle-input only for a trusted file."
            )
        with path.open("rb") as f:
            obj = pickle.load(f)
        loader_name = "pickle"
    elif suffix in {".pt", ".pth"}:
        if not allow_pickle_input:
            raise ValueError(
                "Refusing torch/pickle input by default. Use --allow-pickle-input only for a trusted file."
            )
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required to load a .pt/.pth vocabulary") from exc
        obj = torch.load(path, map_location="cpu")
        loader_name = "torch"
    else:
        raise ValueError(
            f"Unsupported GO vocabulary extension {suffix!r}; "
            "use .txt/.tsv/.csv/.json/.npy/.pkl/.pt"
        )

    if dotted_key:
        value = _resolve_dotted_key(obj, dotted_key)
        ids = _sequence_of_go_ids(value) or _mapping_as_go_ids(value)
        if ids is None:
            raise ValueError(f"Object selected by --go-terms-key is not an ordered GO vocabulary")
        return ids, f"{loader_name}:{dotted_key}"

    candidates = _find_go_sequences(obj)
    unique: Dict[Tuple[str, ...], Tuple[str, List[str]]] = {}
    for candidate_path, ids in candidates:
        unique.setdefault(tuple(ids), (candidate_path, ids))
    candidates = list(unique.values())
    if len(candidates) != 1:
        preview = ", ".join(f"{p} (n={len(v)})" for p, v in candidates[:10]) or "none"
        raise ValueError(
            "Could not identify one unambiguous ordered GO vocabulary. "
            f"Candidates: {preview}. Supply --go-terms-key."
        )
    candidate_path, ids = candidates[0]
    return ids, f"{loader_name}:{candidate_path}"


def build_alias_map(terms: Mapping[str, GOTerm]) -> Dict[str, str]:
    alias: Dict[str, str] = {}
    for go_id, term in terms.items():
        for alt_id in term.alt_ids:
            previous = alias.setdefault(alt_id, go_id)
            if previous != go_id:
                raise ValueError(f"Alternative GO ID {alt_id} maps to both {previous} and {go_id}")
    return alias


def canonicalize_vocabulary(
    input_ids: Sequence[str],
    *,
    terms: Mapping[str, GOTerm],
    aliases: Mapping[str, str],
    obsolete_policy: str,
    allow_missing: bool,
    duplicate_policy: str,
) -> Tuple[
    List[str],
    List[str],
    Dict[str, int],
    Dict[str, List[int]],
    List[Dict[str, Any]],
]:
    """Canonicalize classifier-ordered GO IDs without dropping any row."""
    canonical: List[str] = []
    statuses: List[str] = []
    counts: MutableMapping[str, int] = defaultdict(int)

    for input_id in input_ids:
        go_id = aliases.get(input_id, input_id)
        status = "alt_id" if go_id != input_id else "canonical"
        term = terms.get(go_id)
        if term is None:
            if not allow_missing:
                raise KeyError(
                    f"Vocabulary GO ID {input_id} is absent from the OBO. "
                    "Use the ontology version paired with first-stage training, or explicitly "
                    "use --allow-missing-terms."
                )
            status = "missing"
        elif term.is_obsolete:
            if obsolete_policy == "error":
                raise ValueError(
                    f"Vocabulary GO ID {input_id} resolves to obsolete term {go_id}. "
                    "Use the training-time OBO, or choose --obsolete-policy keep/replace deliberately."
                )
            if obsolete_policy == "replace":
                replacements = [aliases.get(x, x) for x in term.replaced_by]
                replacements = [x for x in replacements if x in terms and not terms[x].is_obsolete]
                replacements = sorted(set(replacements))
                if len(replacements) != 1:
                    raise ValueError(
                        f"Obsolete term {go_id} has {len(replacements)} usable replacements: "
                        f"{replacements}; cannot preserve one classifier index unambiguously."
                    )
                go_id = replacements[0]
                status = "obsolete_replaced"
            else:
                status = "obsolete_kept"
        canonical.append(go_id)
        statuses.append(status)
        counts[status] += 1

    go_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, go_id in enumerate(canonical):
        go_to_indices[go_id].append(idx)

    duplicate_groups: List[Dict[str, Any]] = []
    for go_id, indices in go_to_indices.items():
        if len(indices) < 2:
            continue
        group_inputs = [input_ids[idx] for idx in indices]
        duplicate_groups.append(
            {
                "canonical_go_id": go_id,
                "classifier_indices": list(indices),
                "input_go_ids": group_inputs,
                "statuses": [statuses[idx] for idx in indices],
                "representative_idx": indices[0],
                "literal_input_duplicate": len(set(group_inputs)) != len(group_inputs),
            }
        )
    duplicate_groups.sort(key=lambda group: group["classifier_indices"][0])

    if duplicate_groups:
        redundant_count = sum(
            len(group["classifier_indices"]) - 1 for group in duplicate_groups
        )
        preview = "; ".join(
            (
                f"{group['canonical_go_id']}:"
                f"indices={group['classifier_indices']} "
                f"inputs={list(zip(group['input_go_ids'], group['statuses']))}"
            )
            for group in duplicate_groups[:20]
        )
        if len(duplicate_groups) > 20:
            preview += f"; ... (+{len(duplicate_groups) - 20} groups)"
        summary = (
            f"{len(duplicate_groups)} canonical GO terms are shared by multiple "
            f"classifier indices ({redundant_count} redundant indices). {preview}"
        )
        if duplicate_policy == "error":
            raise ValueError(
                "GO vocabulary is not one-to-one after alt/obsolete canonicalization. "
                + summary
                + ". Use the training-time OBO snapshot or explicitly choose "
                "--duplicate-policy replicate/representative."
            )
        print(
            f"[duplicate-policy={duplicate_policy}] {summary}\n"
            f"[duplicate-policy={duplicate_policy}] preserving all "
            f"{len(canonical)} classifier indices; inspect go_registry.tsv and "
            "gg_relations_manifest.json for the projection audit.",
            flush=True,
        )

    return (
        canonical,
        statuses,
        dict(counts),
        dict(go_to_indices),
        duplicate_groups,
    )


def build_adjacency(
    terms: Mapping[str, GOTerm],
    aliases: Mapping[str, str],
    relation: str,
    *,
    keep_obsolete: bool,
) -> Dict[str, Tuple[str, ...]]:
    adjacency: Dict[str, Tuple[str, ...]] = {}
    for go_id, term in terms.items():
        if term.is_obsolete and not keep_obsolete:
            continue
        raw_parents = term.is_a if relation == "is_a" else term.part_of
        parents: List[str] = []
        for raw_parent in raw_parents:
            parent = aliases.get(raw_parent, raw_parent)
            parent_term = terms.get(parent)
            if parent_term is None:
                continue
            if parent_term.is_obsolete and not keep_obsolete:
                continue
            if parent != go_id:
                parents.append(parent)
        adjacency[go_id] = tuple(sorted(set(parents)))
    return adjacency


def validate_acyclic(
    adjacency: Mapping[str, Sequence[str]],
    *,
    relation: str,
) -> None:
    nodes = set(adjacency)
    for parents in adjacency.values():
        nodes.update(parents)
    indegree = {node: 0 for node in nodes}
    children: Dict[str, List[str]] = defaultdict(list)
    for child, parents in adjacency.items():
        for parent in parents:
            indegree[parent] += 1
            children[child].append(parent)
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for parent in children.get(node, ()):
            indegree[parent] -= 1
            if indegree[parent] == 0:
                queue.append(parent)
    if visited != len(nodes):
        cycle_nodes = sorted(node for node, degree in indegree.items() if degree > 0)
        raise ValueError(
            f"Cycle detected in OBO relation {relation}; examples={cycle_nodes[:10]}"
        )


def compute_closure_rows(
    vocabulary: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
    *,
    relation: str,
    max_hops: int,
    go_to_indices: Mapping[str, Sequence[int]],
    duplicate_policy: str,
) -> np.ndarray:
    """Return shortest-path closure rows projected to classifier indices.

    ``replicate`` gives every classifier row in a canonical collision group the
    same ontology context.  ``representative`` routes both source and
    destination edges only through the group's first classifier row.  Pairs
    within one canonical group are never emitted, because they would be
    canonicalization artifacts rather than ontology relations.
    """
    representative_idx = {
        go_id: indices[0] for go_id, indices in go_to_indices.items()
    }
    rows: List[Tuple[int, int, int, int]] = []
    closure_cache: Dict[str, Dict[str, int]] = {}

    for src_idx, src_go in enumerate(vocabulary):
        if (
            duplicate_policy == "representative"
            and src_idx != representative_idx[src_go]
        ):
            continue

        distance = closure_cache.get(src_go)
        if distance is None:
            distance = {}
            queue: deque[Tuple[str, int]] = deque(
                (parent, 1) for parent in adjacency.get(src_go, ())
            )
            while queue:
                node, hops = queue.popleft()
                if node == src_go:
                    raise ValueError(f"Cycle detected in {relation}: {src_go} reaches itself")
                previous = distance.get(node)
                if previous is not None and previous <= hops:
                    continue
                distance[node] = hops
                if max_hops > 0 and hops >= max_hops:
                    continue
                for parent in adjacency.get(node, ()):
                    queue.append((parent, hops + 1))
            closure_cache[src_go] = distance

        for dst_go, hops in distance.items():
            destination_indices = go_to_indices.get(dst_go, ())
            if duplicate_policy == "representative":
                destination_indices = destination_indices[:1]
            for dst_idx in destination_indices:
                if dst_go == src_go or dst_idx == src_idx:
                    continue
                rows.append((src_idx, dst_idx, hops, int(hops == 1)))

        if src_idx == 0 or src_idx + 1 == len(vocabulary) or (src_idx + 1) % 5000 == 0:
            print(
                f"[{relation} closure] terms={src_idx + 1}/{len(vocabulary)} "
                f"edges={len(rows)}",
                flush=True,
            )

    rows.sort(key=lambda x: (x[0], x[2], x[1]))
    if not rows:
        return np.empty((0, 4), dtype=np.int32)
    return np.asarray(rows, dtype=np.int32)


def reverse_relation(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows.copy()
    reverse = rows[:, [1, 0, 2, 3]].copy()
    order = np.lexsort((reverse[:, 1], reverse[:, 2], reverse[:, 0]))
    return reverse[order]


def degree_gini(degrees: np.ndarray) -> float:
    x = np.asarray(degrees, dtype=np.float64)
    if x.size == 0 or float(x.sum()) == 0.0:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2.0 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def relation_stats(rows: np.ndarray, num_terms: int) -> Dict[str, Any]:
    if rows.shape[0] == 0:
        return {
            "edge_count": 0,
            "direct_edge_count": 0,
            "closure_only_edge_count": 0,
            "max_hop_distance": 0,
            "mean_hop_distance": 0.0,
            "zero_out_degree_terms": int(num_terms),
            "zero_in_degree_terms": int(num_terms),
            "max_out_degree": 0,
            "max_in_degree": 0,
            "out_degree_gini": 0.0,
            "in_degree_gini": 0.0,
        }
    out_degree = np.bincount(rows[:, 0], minlength=num_terms)
    in_degree = np.bincount(rows[:, 1], minlength=num_terms)
    direct = int(rows[:, 3].sum())
    return {
        "edge_count": int(rows.shape[0]),
        "direct_edge_count": direct,
        "closure_only_edge_count": int(rows.shape[0] - direct),
        "max_hop_distance": int(rows[:, 2].max()),
        "mean_hop_distance": float(rows[:, 2].mean()),
        "zero_out_degree_terms": int((out_degree == 0).sum()),
        "zero_in_degree_terms": int((in_degree == 0).sum()),
        "max_out_degree": int(out_degree.max(initial=0)),
        "max_in_degree": int(in_degree.max(initial=0)),
        "out_degree_gini": degree_gini(out_degree),
        "in_degree_gini": degree_gini(in_degree),
    }


def validate_relation(rows: np.ndarray, num_terms: int, relation: str) -> None:
    if rows.ndim != 2 or rows.shape[1] != 4 or rows.dtype != np.int32:
        raise ValueError(f"{relation}: expected int32 [E,4], got {rows.shape}/{rows.dtype}")
    if rows.size == 0:
        return
    src, dst, hops, direct = rows.T
    if np.any(src < 0) or np.any(src >= num_terms) or np.any(dst < 0) or np.any(dst >= num_terms):
        raise ValueError(f"{relation}: node index outside [0, {num_terms})")
    if np.any(src == dst):
        raise ValueError(f"{relation}: self edge found")
    if np.any(hops < 1):
        raise ValueError(f"{relation}: hop distance below one")
    if np.any((direct != 0) & (direct != 1)):
        raise ValueError(f"{relation}: is_direct is not binary")
    if np.any((direct == 1) & (hops != 1)):
        raise ValueError(f"{relation}: a direct edge has hop_distance != 1")
    pairs = (src.astype(np.int64) << 32) | dst.astype(np.uint32).astype(np.int64)
    if np.unique(pairs).size != pairs.size:
        raise ValueError(f"{relation}: duplicate src/dst pair found")


def validate_inverse(forward: np.ndarray, reverse: np.ndarray, label: str) -> None:
    expected = reverse_relation(forward)
    if not np.array_equal(expected, reverse):
        raise ValueError(f"{label}: inverse relation does not exactly match forward relation")


def registry_tsv(
    input_ids: Sequence[str],
    canonical_ids: Sequence[str],
    statuses: Sequence[str],
    terms: Mapping[str, GOTerm],
    go_to_indices: Mapping[str, Sequence[int]],
) -> str:
    lines = [
        "go_idx\tinput_go_id\tgo_id\tname\tnamespace\tstatus\t"
        "dup_representative_idx\tdup_group_size"
    ]
    for idx, (input_id, go_id, status) in enumerate(zip(input_ids, canonical_ids, statuses)):
        term = terms.get(go_id)
        name = "" if term is None else term.name.replace("\t", " ").replace("\n", " ")
        namespace = "" if term is None else term.namespace
        group = go_to_indices[go_id]
        lines.append(
            f"{idx}\t{input_id}\t{go_id}\t{name}\t{namespace}\t{status}\t"
            f"{group[0]}\t{len(group)}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.duplicate_policy == "preserve":
        print(
            "[deprecated] --duplicate-policy preserve is an alias of replicate; "
            "use replicate in new runs.",
            flush=True,
        )
        args.duplicate_policy = "replicate"
    args.obo = args.obo.expanduser().resolve()
    args.go_terms = args.go_terms.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    task_namespace = TASK_NAMESPACES[args.task]

    if not args.obo.is_file():
        raise FileNotFoundError(f"OBO file not found: {args.obo}")
    if not args.go_terms.is_file():
        raise FileNotFoundError(f"GO vocabulary not found: {args.go_terms}")
    if args.expected_num_terms is not None and args.expected_num_terms <= 0:
        raise ValueError("--expected-num-terms must be positive")
    if args.max_hops < 0:
        raise ValueError("--max-hops must be zero or positive")

    started = time.time()
    terms = parse_obo(args.obo)
    aliases = build_alias_map(terms)
    input_ids, vocabulary_loader = load_ordered_go_ids(
        args.go_terms,
        dotted_key=args.go_terms_key,
        allow_pickle_input=args.allow_pickle_input,
    )
    if not input_ids:
        raise ValueError("GO vocabulary is empty")
    if args.expected_num_terms is not None and len(input_ids) != args.expected_num_terms:
        raise ValueError(
            f"GO vocabulary length {len(input_ids)} != --expected-num-terms "
            f"{args.expected_num_terms}. This would misalign graph indices and classifier outputs."
        )

    (
        vocabulary,
        statuses,
        status_counts,
        go_to_indices,
        duplicate_groups,
    ) = canonicalize_vocabulary(
        input_ids,
        terms=terms,
        aliases=aliases,
        obsolete_policy=args.obsolete_policy,
        allow_missing=args.allow_missing_terms,
        duplicate_policy=args.duplicate_policy,
    )
    if duplicate_groups:
        duplicate_index_count = sum(
            len(group["classifier_indices"]) for group in duplicate_groups
        )
        preview = ", ".join(
            f"{group['canonical_go_id']}:{group['classifier_indices'][:5]}"
            for group in duplicate_groups[:10]
        )
        print(
            "[GO vocabulary canonical collisions] "
            f"groups={len(duplicate_groups)} classifier_indices={duplicate_index_count}; "
            f"preserving classifier index space. Examples: {preview}",
            flush=True,
        )
    wrong_namespace = [
        (idx, go_id, terms[go_id].namespace)
        for idx, go_id in enumerate(vocabulary)
        if go_id in terms and terms[go_id].namespace != task_namespace
    ]
    if wrong_namespace and not args.allow_cross_namespace_terms:
        raise ValueError(
            f"{len(wrong_namespace)} vocabulary terms are outside namespace={task_namespace}; "
            f"examples={wrong_namespace[:10]}"
        )

    relation_names = ["is_a", "has_child"]
    if args.include_part_of:
        relation_names.extend(["part_of", "has_part"])
    output_paths = {
        relation: args.output_dir / f"gg_{relation}.i32.npy" for relation in relation_names
    }
    registry_path = args.output_dir / "go_registry.tsv"
    manifest_path = args.output_dir / "gg_relations_manifest.json"
    ensure_writable([*output_paths.values(), registry_path, manifest_path], args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    keep_obsolete = args.obsolete_policy == "keep"
    is_a_adjacency = build_adjacency(
        terms, aliases, "is_a", keep_obsolete=keep_obsolete
    )
    validate_acyclic(is_a_adjacency, relation="is_a")
    is_a_rows = compute_closure_rows(
        vocabulary,
        is_a_adjacency,
        relation="is_a",
        max_hops=args.max_hops,
        go_to_indices=go_to_indices,
        duplicate_policy=args.duplicate_policy,
    )
    has_child_rows = reverse_relation(is_a_rows)
    relations: Dict[str, np.ndarray] = {
        "is_a": is_a_rows,
        "has_child": has_child_rows,
    }

    if args.include_part_of:
        part_of_adjacency = build_adjacency(
            terms, aliases, "part_of", keep_obsolete=keep_obsolete
        )
        validate_acyclic(part_of_adjacency, relation="part_of")
        part_of_rows = compute_closure_rows(
            vocabulary,
            part_of_adjacency,
            relation="part_of",
            max_hops=args.max_hops,
            go_to_indices=go_to_indices,
            duplicate_policy=args.duplicate_policy,
        )
        relations["part_of"] = part_of_rows
        relations["has_part"] = reverse_relation(part_of_rows)

    for relation, rows in relations.items():
        validate_relation(rows, len(vocabulary), relation)
    validate_inverse(relations["is_a"], relations["has_child"], "is_a/has_child")
    if args.include_part_of:
        validate_inverse(relations["part_of"], relations["has_part"], "part_of/has_part")

    for relation, rows in relations.items():
        atomic_save_npy(output_paths[relation], rows)
    atomic_write_text(
        registry_path,
        registry_tsv(input_ids, vocabulary, statuses, terms, go_to_indices),
    )

    relation_records: List[Dict[str, Any]] = []
    semantics = {
        "is_a": "child->parent_or_is_a_ancestor",
        "has_child": "parent_or_ancestor->child_or_descendant; is_direct=1 means immediate child",
        "part_of": "part->whole_or_transitive_whole",
        "has_part": "whole_or_transitive_whole->part; is_direct=1 means immediate part",
    }
    inverse_of = {
        "is_a": "has_child",
        "has_child": "is_a",
        "part_of": "has_part",
        "has_part": "part_of",
    }
    for relation, rows in relations.items():
        relation_records.append(
            {
                "relation": relation,
                "inverse_relation": inverse_of[relation],
                "semantics": semantics[relation],
                "file": output_paths[relation].name,
                "dtype": "int32",
                "shape": [int(rows.shape[0]), 4],
                "columns": [
                    "src_go_idx",
                    "dst_go_idx",
                    "hop_distance",
                    "is_direct",
                ],
                **relation_stats(rows, len(vocabulary)),
            }
        )

    manifest = {
        "schema_version": 2,
        "task": args.task,
        "namespace": task_namespace,
        "num_go_terms": len(vocabulary),
        "num_unique_canonical_go_terms": len(set(vocabulary)),
        "node_index_space": "zero-based row index in go_registry.tsv and first-stage classifier output",
        "go_registry_file": registry_path.name,
        "go_ids_sha256": hash_ordered_ids(vocabulary),
        "input_go_ids_sha256": hash_ordered_ids(input_ids),
        "vocabulary_source": str(args.go_terms),
        "vocabulary_source_sha256": sha256_file(args.go_terms),
        "vocabulary_loader": vocabulary_loader,
        "ontology_source": str(args.obo),
        "ontology_sha256": sha256_file(args.obo),
        "obo_term_count": len(terms),
        "obo_alt_id_count": len(aliases),
        "vocabulary_status_counts": status_counts,
        "duplicate_policy": args.duplicate_policy,
        "unique_go_term_count": len(go_to_indices),
        "unique_go_ids_sha256": hash_ordered_ids(sorted(go_to_indices)),
        "duplicate_term_count": len(duplicate_groups),
        "duplicate_extra_index_count": sum(
            len(group["classifier_indices"]) - 1 for group in duplicate_groups
        ),
        "duplicate_groups": duplicate_groups[:200],
        "duplicate_groups_truncated": len(duplicate_groups) > 200,
        "wrong_namespace_term_count": len(wrong_namespace),
        "canonical_projection": {
            "classifier_indices_preserved": True,
            "ontology_traversal_space": "canonical GO IDs",
            "ancestor_projection": (
                "replicate: every classifier index; representative: first index only"
            ),
            "edges_between_same_canonical_group_members": False,
        },
        "closure": {
            "complete": args.max_hops == 0,
            "max_hops": None if args.max_hops == 0 else args.max_hops,
            "path_policy": "relation-specific shortest path",
            "intermediate_nodes": "full OBO graph, including terms outside classifier vocabulary",
            "mixed_is_a_part_of_paths": False,
        },
        "include_part_of": bool(args.include_part_of),
        "obsolete_policy": args.obsolete_policy,
        "relations": relation_records,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
