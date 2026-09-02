#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align ESM2GO numpy outputs to namespace_terms.pkl, with optional GO ancestor propagation.

Typical dense inputs produced by ESM2GO_pred_large_multigpu.py:
    <working_dir>/names.npy
    <working_dir>/ESM2GO_numpy/BPO.terms.npy
    <working_dir>/ESM2GO_numpy/BPO.scores.float16.npy
    <working_dir>/ESM2GO_numpy/CCO.terms.npy
    <working_dir>/ESM2GO_numpy/CCO.scores.float16.npy
    <working_dir>/ESM2GO_numpy/MFO.terms.npy
    <working_dir>/ESM2GO_numpy/MFO.scores.float16.npy

namespace_terms.pkl:
    dict with keys:
        cellular_component
        molecular_function
        biological_process
    each value is an ordered list of GO IDs.

When --propagate_ancestors is used, each source GO score is assigned to:
    1) the same GO term if it exists in namespace_terms[namespace]
    2) every ancestor GO term reachable through the selected OBO relations
       if that ancestor exists in namespace_terms[namespace]
Scores are combined by max, which is the standard true-path-consistent operation
for probabilistic scores.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
from collections import OrderedDict, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


NAMESPACE_TO_ASPECT = OrderedDict(
    [
        ("cellular_component", "CCO"),
        ("molecular_function", "MFO"),
        ("biological_process", "BPO"),
    ]
)
ASPECT_TO_NAMESPACE = {v: k for k, v in NAMESPACE_TO_ASPECT.items()}
GO_ID_RE = re.compile(r"GO:\d{7}")


def _load_npy(path: Path, mmap_mode: Optional[str] = None, allow_pickle: bool = False):
    """Load npy robustly. Falls back to allow_pickle=True for old object arrays."""
    try:
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)
    except ValueError:
        if allow_pickle:
            raise
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=True)


def normalize_go_id(x) -> str:
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return str(x).strip()


def normalize_terms(xs: Iterable) -> List[str]:
    return [normalize_go_id(x) for x in xs]


def check_unique(values: Sequence[str], label: str) -> None:
    seen = set()
    duplicated = []
    for v in values:
        if v in seen:
            duplicated.append(v)
        seen.add(v)
    if duplicated:
        examples = ", ".join(duplicated[:10])
        raise ValueError(f"{label} contains duplicated GO IDs, e.g. {examples}")


def parse_float_fill_value(text: str) -> float:
    lower = str(text).strip().lower()
    if lower in {"nan", "+nan", "-nan"}:
        return float("nan")
    return float(text)


def load_namespace_terms(path: Path) -> Dict[str, List[str]]:
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, Mapping):
        raise TypeError(f"{path} must contain a dict-like object, got {type(obj)!r}")

    out: Dict[str, List[str]] = {}
    for namespace, terms in obj.items():
        namespace = str(namespace)
        out[namespace] = normalize_terms(terms)
    return out


def resolve_namespaces(requested: Optional[Sequence[str]], namespace_terms: Mapping[str, Sequence[str]]) -> List[str]:
    if requested:
        namespaces = list(requested)
    else:
        namespaces = [ns for ns in NAMESPACE_TO_ASPECT if ns in namespace_terms]
        namespaces.extend([ns for ns in namespace_terms if ns not in namespaces])

    missing = [ns for ns in namespaces if ns not in namespace_terms]
    if missing:
        raise KeyError(f"namespace_terms.pkl is missing namespace(s): {missing}")

    unsupported = [ns for ns in namespaces if ns not in NAMESPACE_TO_ASPECT]
    if unsupported:
        raise KeyError(
            "Unsupported namespace(s): "
            f"{unsupported}. Supported keys are {list(NAMESPACE_TO_ASPECT.keys())}."
        )
    return namespaces


def make_row_indices(source_names_npy: Path, target_names_npy: Optional[Path]) -> Tuple[Optional[np.ndarray], Path, int, str]:
    """Build optional source-row indices so outputs can follow target label row order."""
    source_names = normalize_terms(_load_npy(source_names_npy, allow_pickle=False))
    check_unique(source_names, f"{source_names_npy}")

    if target_names_npy is None:
        return None, source_names_npy, len(source_names), "source_names_npy order"

    target_names = normalize_terms(_load_npy(target_names_npy, allow_pickle=False))
    check_unique(target_names, f"{target_names_npy}")
    source_pos = {name: i for i, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_pos]
    if missing:
        examples = ", ".join(missing[:20])
        raise ValueError(
            f"target_names_npy contains {len(missing)} protein IDs absent from source names.npy. "
            f"Examples: {examples}"
        )

    row_indices = np.asarray([source_pos[name] for name in target_names], dtype=np.int64)
    return row_indices, target_names_npy, len(target_names), "target_names_npy order"


def save_json(path: Path, obj: Mapping) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def open_output_memmap(path: Path, dtype: np.dtype, shape: Tuple[int, int]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape,
        version=(2, 0),
    )


def find_topk_files(numpy_dir: Path, aspect: str, top_k: Optional[int]) -> Tuple[Optional[Path], Optional[Path], Optional[int]]:
    if top_k is not None:
        scores = numpy_dir / f"{aspect}.top{top_k}.scores.float16.npy"
        indices = numpy_dir / f"{aspect}.top{top_k}.indices.int32.npy"
        return (scores if scores.exists() else None, indices if indices.exists() else None, top_k)

    candidates = sorted(numpy_dir.glob(f"{aspect}.top*.scores.float16.npy"))
    for scores in candidates:
        prefix = scores.name.replace(".scores.float16.npy", "")
        if not prefix.startswith(f"{aspect}.top"):
            continue
        k_text = prefix[len(f"{aspect}.top") :]
        try:
            k = int(k_text)
        except ValueError:
            continue
        indices = numpy_dir / f"{aspect}.top{k}.indices.int32.npy"
        if indices.exists():
            return scores, indices, k
    return None, None, None


def make_alignment_index_exact(
    source_terms: Sequence[str],
    target_terms: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return source column indices and target column indices in exact target order."""
    source_pos = {term: i for i, term in enumerate(source_terms)}
    source_indices: List[int] = []
    target_indices: List[int] = []
    missing_terms: List[str] = []

    for target_j, term in enumerate(target_terms):
        source_j = source_pos.get(term)
        if source_j is None:
            missing_terms.append(term)
        else:
            source_indices.append(source_j)
            target_indices.append(target_j)

    return (
        np.asarray(source_indices, dtype=np.int64),
        np.asarray(target_indices, dtype=np.int64),
        missing_terms,
    )


class GoOntology:
    """Minimal GO OBO parser for ancestor propagation.

    The graph direction is child -> parents. get_ancestors() returns the transitive
    closure over the selected parent relation types.
    """

    def __init__(
        self,
        parents: Mapping[str, Sequence[str]],
        namespaces: Mapping[str, str],
        alt_id_to_id: Mapping[str, str],
        obsolete: Iterable[str],
        relationships: Sequence[str],
    ) -> None:
        self.parents = {k: tuple(v) for k, v in parents.items()}
        self.namespaces = dict(namespaces)
        self.alt_id_to_id = dict(alt_id_to_id)
        self.obsolete = set(obsolete)
        self.relationships = tuple(relationships)

    @classmethod
    def from_obo(cls, obo_path: Path, relationships: Sequence[str]) -> "GoOntology":
        if not obo_path.exists():
            raise FileNotFoundError(f"GO OBO file not found: {obo_path}")

        relationships = tuple(dict.fromkeys([r.strip() for r in relationships if r.strip()]))
        if not relationships:
            raise ValueError("At least one GO relation must be selected for propagation")

        raw_parents: Dict[str, List[str]] = defaultdict(list)
        namespaces: Dict[str, str] = {}
        alt_id_to_id: Dict[str, str] = {}
        obsolete = set()

        current: Dict[str, object] = {}
        in_term = False

        def flush_term() -> None:
            nonlocal current, in_term
            if not in_term:
                current = {}
                return
            term_id = current.get("id")
            if not term_id:
                current = {}
                in_term = False
                return
            term_id = str(term_id)
            namespaces[term_id] = str(current.get("namespace", ""))
            if bool(current.get("is_obsolete", False)):
                obsolete.add(term_id)

            for alt in current.get("alt_id", []):
                alt_id_to_id[str(alt)] = term_id

            raw_parent_ids: List[str] = []
            if "is_a" in relationships:
                raw_parent_ids.extend(current.get("is_a", []))
            for rel_type, parent_id in current.get("relationship", []):
                if rel_type in relationships:
                    raw_parent_ids.append(parent_id)
            raw_parents[term_id].extend(raw_parent_ids)

            current = {}
            in_term = False

        with open(obo_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line == "[Term]":
                    flush_term()
                    current = {"alt_id": [], "is_a": [], "relationship": []}
                    in_term = True
                    continue
                if line.startswith("["):
                    flush_term()
                    continue
                if not in_term:
                    continue

                if line.startswith("id: "):
                    m = GO_ID_RE.search(line)
                    if m:
                        current["id"] = m.group(0)
                elif line.startswith("namespace: "):
                    current["namespace"] = line.split("namespace: ", 1)[1].strip()
                elif line.startswith("alt_id: "):
                    m = GO_ID_RE.search(line)
                    if m:
                        current.setdefault("alt_id", []).append(m.group(0))
                elif line.startswith("is_a: "):
                    m = GO_ID_RE.search(line)
                    if m:
                        current.setdefault("is_a", []).append(m.group(0))
                elif line.startswith("relationship: "):
                    # Example: relationship: part_of GO:0008150 ! biological_process
                    rest = line.split("relationship: ", 1)[1].strip()
                    parts = rest.split()
                    if len(parts) >= 2:
                        rel_type = parts[0]
                        m = GO_ID_RE.search(rest)
                        if m:
                            current.setdefault("relationship", []).append((rel_type, m.group(0)))
                elif line.startswith("is_obsolete: "):
                    val = line.split("is_obsolete: ", 1)[1].strip().lower()
                    current["is_obsolete"] = val == "true"
        flush_term()

        # Canonicalize parent IDs after all alt_id lines have been read.
        parents: Dict[str, Tuple[str, ...]] = {}
        for child, par_list in raw_parents.items():
            child_c = alt_id_to_id.get(child, child)
            canonical_parents = []
            for parent in par_list:
                parent_c = alt_id_to_id.get(parent, parent)
                if parent_c != child_c:
                    canonical_parents.append(parent_c)
            # stable unique
            parents[child_c] = tuple(dict.fromkeys(canonical_parents))

        return cls(
            parents=parents,
            namespaces=namespaces,
            alt_id_to_id=alt_id_to_id,
            obsolete=obsolete,
            relationships=relationships,
        )

    def canonicalize(self, go_id: str) -> str:
        return self.alt_id_to_id.get(go_id, go_id)

    def namespace(self, go_id: str) -> str:
        go_id = self.canonicalize(go_id)
        return self.namespaces.get(go_id, "")

    @lru_cache(maxsize=None)
    def get_ancestors(self, go_id: str) -> Tuple[str, ...]:
        go_id = self.canonicalize(go_id)
        seen = set()
        stack = list(self.parents.get(go_id, ()))
        out = []
        while stack:
            parent = self.canonicalize(stack.pop())
            if parent in seen:
                continue
            seen.add(parent)
            out.append(parent)
            stack.extend(self.parents.get(parent, ()))
        return tuple(out)


def parse_relationships(text: str) -> List[str]:
    rels = [x.strip() for x in str(text).split(",") if x.strip()]
    if not rels:
        raise ValueError("--relationships cannot be empty")
    return rels


def resolve_go_obo_path(go_obo_arg: Optional[Path]) -> Path:
    if go_obo_arg is not None:
        return go_obo_arg.resolve()
    # Optional convenience for running inside the ESM2GO project root.
    try:
        from settings import settings_dict as settings  # type: ignore

        value = settings.get("obo_file")
        if value:
            return Path(value).resolve()
    except Exception:
        pass
    raise ValueError("--go_obo is required when --propagate_ancestors, unless settings['obo_file'] is importable")


def build_source_to_target_indices(
    source_terms: Sequence[str],
    target_terms: Sequence[str],
    namespace: str,
    ontology: Optional[GoOntology],
    propagate_ancestors: bool,
    canonicalize_alt_ids: bool,
    include_self: bool = True,
) -> Tuple[List[np.ndarray], Dict]:
    """For each source column, list target columns to receive max(source_score).

    In propagation mode this includes target ancestors. In direct-only mode this
    includes only the same GO term.
    """
    if propagate_ancestors and ontology is None:
        raise ValueError("ontology is required when propagate_ancestors=True")

    def key(term: str) -> str:
        if ontology is not None and canonicalize_alt_ids:
            return ontology.canonicalize(term)
        return term

    target_key_to_indices: Dict[str, List[int]] = defaultdict(list)
    for target_j, term in enumerate(target_terms):
        target_key_to_indices[key(term)].append(target_j)

    source_to_target: List[np.ndarray] = []
    directly_reached_target_indices = set()
    propagated_reached_target_indices = set()
    source_terms_with_any_target = 0
    source_terms_with_ancestor_target = 0
    source_terms_missing_in_obo = []

    for source_j, term in enumerate(source_terms):
        term_key = key(term)
        target_indices = []

        if include_self and term_key in target_key_to_indices:
            direct_indices = target_key_to_indices[term_key]
            target_indices.extend(direct_indices)
            directly_reached_target_indices.update(direct_indices)

        if propagate_ancestors:
            assert ontology is not None
            canonical = ontology.canonicalize(term) if canonicalize_alt_ids else term
            if canonical not in ontology.parents and canonical not in ontology.namespaces:
                source_terms_missing_in_obo.append(term)
            ancestor_hits = []
            for anc in ontology.get_ancestors(canonical):
                # target_key_to_indices already restricts updates to the target namespace term list.
                if anc in target_key_to_indices:
                    ancestor_hits.extend(target_key_to_indices[anc])
            if ancestor_hits:
                target_indices.extend(ancestor_hits)
                propagated_reached_target_indices.update(ancestor_hits)
                source_terms_with_ancestor_target += 1

        if target_indices:
            # Stable unique target columns. Sorting improves locality and deterministic summaries.
            arr = np.asarray(sorted(set(target_indices)), dtype=np.int64)
            source_terms_with_any_target += 1
        else:
            arr = np.asarray([], dtype=np.int64)
        source_to_target.append(arr)

    all_reached = directly_reached_target_indices | propagated_reached_target_indices
    missing_after = [term for j, term in enumerate(target_terms) if j not in all_reached]

    # For reporting duplicated canonical target IDs, usually caused by alt_id/canonical ID duplicates.
    duplicated_target_keys = {
        k: v for k, v in target_key_to_indices.items() if len(v) > 1
    }

    meta = {
        "namespace": namespace,
        "propagate_ancestors": bool(propagate_ancestors),
        "include_self": bool(include_self),
        "canonicalize_alt_ids": bool(canonicalize_alt_ids and ontology is not None),
        "source_columns": int(len(source_terms)),
        "target_columns": int(len(target_terms)),
        "directly_reachable_target_columns": int(len(directly_reached_target_indices)),
        "propagated_reachable_target_columns": int(len(propagated_reached_target_indices)),
        "reachable_target_columns_total": int(len(all_reached)),
        "unreachable_target_columns": int(len(missing_after)),
        "unreachable_target_terms_preview": missing_after[:50],
        "source_terms_contributing_to_any_target_column": int(source_terms_with_any_target),
        "source_terms_contributing_to_ancestor_target_column": int(source_terms_with_ancestor_target),
        "source_terms_missing_in_obo": int(len(source_terms_missing_in_obo)),
        "source_terms_missing_in_obo_preview": source_terms_missing_in_obo[:50],
        "duplicated_canonical_target_ids": int(len(duplicated_target_keys)),
        "duplicated_canonical_target_ids_preview": {
            k: v for k, v in list(duplicated_target_keys.items())[:20]
        },
    }
    return source_to_target, meta


def _threshold_values(values: np.ndarray, min_score: float) -> np.ndarray:
    """Emulate the user's original mask = preds > min_score; others become zero."""
    if min_score >= 0.0:
        # Use strict >, matching the original implementation.
        return np.where(values > min_score, values, 0.0)
    return values


def align_dense_direct(
    source_scores_path: Path,
    source_terms: Sequence[str],
    target_terms: Sequence[str],
    output_scores_path: Path,
    output_dtype: np.dtype,
    fill_value: float,
    row_chunk_size: int,
    flush_every: int,
    row_indices: Optional[np.ndarray] = None,
) -> Dict:
    src = _load_npy(source_scores_path, mmap_mode="r", allow_pickle=False)
    if src.ndim != 2:
        raise ValueError(f"{source_scores_path} must be a 2-D array, got shape {src.shape}")
    if src.shape[1] != len(source_terms):
        raise ValueError(
            f"Column count mismatch for {source_scores_path}: "
            f"scores has {src.shape[1]} columns but terms has {len(source_terms)} entries."
        )

    source_n_rows = int(src.shape[0])
    if row_indices is None:
        n_rows = source_n_rows
    else:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be 1-D")
        if row_indices.size and (int(row_indices.min()) < 0 or int(row_indices.max()) >= source_n_rows):
            raise ValueError(f"row_indices contains values outside source row range 0..{source_n_rows - 1}")
        n_rows = int(row_indices.shape[0])

    source_indices, target_indices, missing_terms = make_alignment_index_exact(source_terms, target_terms)
    n_target = len(target_terms)
    out = open_output_memmap(output_scores_path, output_dtype, (n_rows, n_target))

    for chunk_no, start in enumerate(
        tqdm(range(0, n_rows, row_chunk_size), desc=f"align dense {source_scores_path.name}", ascii=" >=")
    ):
        end = min(start + row_chunk_size, n_rows)
        block = np.empty((end - start, n_target), dtype=output_dtype)
        block[...] = fill_value
        if len(source_indices):
            if row_indices is None:
                values = src[start:end, source_indices]
            else:
                values = src[row_indices[start:end]][:, source_indices]
            block[:, target_indices] = values.astype(output_dtype, copy=False)
        out[start:end] = block
        if flush_every > 0 and (chunk_no + 1) % flush_every == 0:
            out.flush()

    out.flush()
    del out, src
    return {
        "source_mode": "dense",
        "operation": "direct_alignment_only",
        "source_scores_npy": str(source_scores_path),
        "source_rows": int(source_n_rows),
        "output_rows": int(n_rows),
        "source_columns": int(len(source_terms)),
        "target_columns": int(n_target),
        "directly_matched_columns": int(len(source_indices)),
        "missing_columns": int(len(missing_terms)),
        "missing_terms_preview": missing_terms[:50],
    }


def align_dense_with_ancestor_propagation(
    source_scores_path: Path,
    source_terms: Sequence[str],
    target_terms: Sequence[str],
    output_scores_path: Path,
    output_dtype: np.dtype,
    row_chunk_size: int,
    flush_every: int,
    source_to_target_indices: Sequence[np.ndarray],
    min_score: float,
    compute_dtype: np.dtype,
    row_indices: Optional[np.ndarray] = None,
) -> Dict:
    src = _load_npy(source_scores_path, mmap_mode="r", allow_pickle=False)
    if src.ndim != 2:
        raise ValueError(f"{source_scores_path} must be a 2-D array, got shape {src.shape}")
    if src.shape[1] != len(source_terms):
        raise ValueError(
            f"Column count mismatch for {source_scores_path}: "
            f"scores has {src.shape[1]} columns but terms has {len(source_terms)} entries."
        )
    if len(source_to_target_indices) != len(source_terms):
        raise ValueError("source_to_target_indices length does not match source_terms")

    source_n_rows = int(src.shape[0])
    if row_indices is None:
        n_rows = source_n_rows
    else:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be 1-D")
        if row_indices.size and (int(row_indices.min()) < 0 or int(row_indices.max()) >= source_n_rows):
            raise ValueError(f"row_indices contains values outside source row range 0..{source_n_rows - 1}")
        n_rows = int(row_indices.shape[0])

    n_target = len(target_terms)
    out = open_output_memmap(output_scores_path, output_dtype, (n_rows, n_target))
    contributing_sources = [(j, idx) for j, idx in enumerate(source_to_target_indices) if len(idx) > 0]

    for chunk_no, start in enumerate(
        tqdm(range(0, n_rows, row_chunk_size), desc=f"propagate dense {source_scores_path.name}", ascii=" >=")
    ):
        end = min(start + row_chunk_size, n_rows)
        if row_indices is None:
            src_block = np.asarray(src[start:end], dtype=compute_dtype)
        else:
            src_block = np.asarray(src[row_indices[start:end]], dtype=compute_dtype)

        # In propagation mode, zero is the neutral score. This matches the user's original
        # implementation, where scores <= min_score become 0.
        block = np.zeros((end - start, n_target), dtype=compute_dtype)

        for source_j, target_idx in contributing_sources:
            values = _threshold_values(src_block[:, source_j], min_score)
            # Skip all-zero source columns in this chunk, useful when min_score is > 0.
            if not np.any(values):
                continue
            if target_idx.size == 1:
                col = int(target_idx[0])
                np.maximum(block[:, col], values, out=block[:, col])
            else:
                # Fancy indexing returns a copy, so assignment back is required.
                block[:, target_idx] = np.maximum(block[:, target_idx], values[:, None])

        out[start:end] = block.astype(output_dtype, copy=False)
        if flush_every > 0 and (chunk_no + 1) % flush_every == 0:
            out.flush()

    out.flush()
    del out, src
    return {
        "source_mode": "dense",
        "operation": "direct_alignment_plus_ancestor_max_propagation",
        "source_scores_npy": str(source_scores_path),
        "source_rows": int(source_n_rows),
        "output_rows": int(n_rows),
        "source_columns": int(len(source_terms)),
        "target_columns": int(n_target),
        "min_score": float(min_score),
        "compute_dtype": np.dtype(compute_dtype).name,
        "propagation_rule": "output[target_ancestor] = max(output[target_ancestor], source_score), scores <= min_score become 0",
    }


def align_topk_direct(
    top_scores_path: Path,
    top_indices_path: Path,
    source_terms: Sequence[str],
    target_terms: Sequence[str],
    output_scores_path: Path,
    output_dtype: np.dtype,
    fill_value: float,
    row_chunk_size: int,
    flush_every: int,
    row_indices: Optional[np.ndarray] = None,
) -> Dict:
    top_scores = _load_npy(top_scores_path, mmap_mode="r", allow_pickle=False)
    top_indices = _load_npy(top_indices_path, mmap_mode="r", allow_pickle=False)

    if top_scores.ndim != 2 or top_indices.ndim != 2:
        raise ValueError(f"Top-k arrays must be 2-D, got {top_scores.shape} and {top_indices.shape}")
    if top_scores.shape != top_indices.shape:
        raise ValueError(f"Top-k score/index shape mismatch: {top_scores.shape} vs {top_indices.shape}")

    source_n_rows, top_k = map(int, top_scores.shape)
    n_rows = source_n_rows if row_indices is None else int(np.asarray(row_indices).shape[0])
    n_target = len(target_terms)

    source_pos = {term: i for i, term in enumerate(source_terms)}
    src_index_to_target_index = np.full(len(source_terms), -1, dtype=np.int64)
    missing_terms = []
    matched = 0
    for target_j, term in enumerate(target_terms):
        source_j = source_pos.get(term)
        if source_j is None:
            missing_terms.append(term)
        else:
            src_index_to_target_index[source_j] = target_j
            matched += 1

    out = open_output_memmap(output_scores_path, output_dtype, (n_rows, n_target))
    for chunk_no, start in enumerate(
        tqdm(range(0, n_rows, row_chunk_size), desc=f"align top-k {top_scores_path.name}", ascii=" >=")
    ):
        end = min(start + row_chunk_size, n_rows)
        block = np.empty((end - start, n_target), dtype=output_dtype)
        block[...] = fill_value
        if row_indices is None:
            idx_block = top_indices[start:end]
            score_block = top_scores[start:end]
        else:
            rows = row_indices[start:end]
            idx_block = top_indices[rows]
            score_block = top_scores[rows]

        for local_i in range(end - start):
            src_idx = idx_block[local_i]
            valid = src_idx >= 0
            if not np.any(valid):
                continue
            src_idx_valid = src_idx[valid].astype(np.int64, copy=False)
            in_range = src_idx_valid < len(src_index_to_target_index)
            if not np.any(in_range):
                continue
            src_idx_valid = src_idx_valid[in_range]
            target_idx = src_index_to_target_index[src_idx_valid]
            keep = target_idx >= 0
            if not np.any(keep):
                continue
            values = score_block[local_i][valid][in_range][keep]
            block[local_i, target_idx[keep]] = values.astype(output_dtype, copy=False)

        out[start:end] = block
        if flush_every > 0 and (chunk_no + 1) % flush_every == 0:
            out.flush()

    out.flush()
    del out, top_scores, top_indices
    return {
        "source_mode": "topk",
        "operation": "direct_alignment_only",
        "top_scores_npy": str(top_scores_path),
        "top_indices_npy": str(top_indices_path),
        "top_k": int(top_k),
        "source_rows": int(source_n_rows),
        "output_rows": int(n_rows),
        "source_columns": int(len(source_terms)),
        "target_columns": int(n_target),
        "directly_matched_columns": int(matched),
        "missing_columns": int(len(missing_terms)),
        "missing_terms_preview": missing_terms[:50],
        "important_note": "This output was reconstructed from compact top-k predictions only.",
    }


def align_topk_with_ancestor_propagation(
    top_scores_path: Path,
    top_indices_path: Path,
    source_terms: Sequence[str],
    target_terms: Sequence[str],
    output_scores_path: Path,
    output_dtype: np.dtype,
    row_chunk_size: int,
    flush_every: int,
    source_to_target_indices: Sequence[np.ndarray],
    min_score: float,
    compute_dtype: np.dtype,
    row_indices: Optional[np.ndarray] = None,
) -> Dict:
    top_scores = _load_npy(top_scores_path, mmap_mode="r", allow_pickle=False)
    top_indices = _load_npy(top_indices_path, mmap_mode="r", allow_pickle=False)

    if top_scores.ndim != 2 or top_indices.ndim != 2:
        raise ValueError(f"Top-k arrays must be 2-D, got {top_scores.shape} and {top_indices.shape}")
    if top_scores.shape != top_indices.shape:
        raise ValueError(f"Top-k score/index shape mismatch: {top_scores.shape} vs {top_indices.shape}")
    if len(source_to_target_indices) != len(source_terms):
        raise ValueError("source_to_target_indices length does not match source_terms")

    source_n_rows, top_k = map(int, top_scores.shape)
    n_rows = source_n_rows if row_indices is None else int(np.asarray(row_indices).shape[0])
    n_target = len(target_terms)
    out = open_output_memmap(output_scores_path, output_dtype, (n_rows, n_target))

    for chunk_no, start in enumerate(
        tqdm(range(0, n_rows, row_chunk_size), desc=f"propagate top-k {top_scores_path.name}", ascii=" >=")
    ):
        end = min(start + row_chunk_size, n_rows)
        block = np.zeros((end - start, n_target), dtype=compute_dtype)

        if row_indices is None:
            idx_block = top_indices[start:end]
            score_block = top_scores[start:end]
        else:
            rows = row_indices[start:end]
            idx_block = top_indices[rows]
            score_block = top_scores[rows]

        # Group by source term inside the chunk. This is much faster than a pure row-by-row loop
        # when many rows share the same top-k source terms.
        flat_idx = idx_block.reshape(-1).astype(np.int64, copy=False)
        flat_scores = np.asarray(score_block.reshape(-1), dtype=compute_dtype)
        flat_rows = np.repeat(np.arange(end - start, dtype=np.int64), idx_block.shape[1])
        valid = (flat_idx >= 0) & (flat_idx < len(source_to_target_indices)) & (flat_scores > min_score)
        if np.any(valid):
            flat_idx = flat_idx[valid]
            flat_scores = flat_scores[valid]
            flat_rows = flat_rows[valid]
            order = np.argsort(flat_idx, kind="mergesort")
            flat_idx = flat_idx[order]
            flat_scores = flat_scores[order]
            flat_rows = flat_rows[order]

            unique_src, starts = np.unique(flat_idx, return_index=True)
            starts = list(starts) + [len(flat_idx)]
            for group_no, source_j in enumerate(unique_src):
                source_j_int = int(source_j)
                target_idx = source_to_target_indices[source_j_int]
                if target_idx.size == 0:
                    continue
                left = starts[group_no]
                right = starts[group_no + 1]
                rows_g = flat_rows[left:right]
                values_g = flat_scores[left:right]
                if target_idx.size == 1:
                    col = int(target_idx[0])
                    np.maximum.at(block[:, col], rows_g, values_g)
                else:
                    # Rows are sparse; update per target column to avoid allocating rows x targets.
                    for col in target_idx:
                        np.maximum.at(block[:, int(col)], rows_g, values_g)

        out[start:end] = block.astype(output_dtype, copy=False)
        if flush_every > 0 and (chunk_no + 1) % flush_every == 0:
            out.flush()

    out.flush()
    del out, top_scores, top_indices
    return {
        "source_mode": "topk",
        "operation": "direct_alignment_plus_ancestor_max_propagation",
        "top_scores_npy": str(top_scores_path),
        "top_indices_npy": str(top_indices_path),
        "top_k": int(top_k),
        "source_rows": int(source_n_rows),
        "output_rows": int(n_rows),
        "source_columns": int(len(source_terms)),
        "target_columns": int(n_target),
        "min_score": float(min_score),
        "compute_dtype": np.dtype(compute_dtype).name,
        "important_note": "Propagation was reconstructed from compact top-k predictions only; scores outside top-k are unavailable and treated as 0.",
    }


def write_term_mapping_rows(summary_tsv: Path, rows: List[Tuple[str, str, int, str, str]], append: bool) -> None:
    mode = "a" if append and summary_tsv.exists() else "w"
    with open(summary_tsv, mode, encoding="utf-8", newline="") as handle:
        if mode == "w":
            handle.write("namespace\taspect\ttarget_index\tgo_id\tstatus\n")
        for namespace, aspect, target_index, go_id, status in rows:
            handle.write(f"{namespace}\t{aspect}\t{target_index}\t{go_id}\t{status}\n")


def align_one_namespace(
    namespace: str,
    target_terms: Sequence[str],
    numpy_dir: Path,
    output_dir: Path,
    source_mode: str,
    output_dtype: np.dtype,
    fill_value: float,
    row_chunk_size: int,
    top_k: Optional[int],
    strict: bool,
    flush_every: int,
    row_indices: Optional[np.ndarray],
    propagate_ancestors: bool,
    ontology: Optional[GoOntology],
    canonicalize_alt_ids: bool,
    min_score: float,
    compute_dtype: np.dtype,
) -> Dict:
    aspect = NAMESPACE_TO_ASPECT[namespace]
    terms_path = numpy_dir / f"{aspect}.terms.npy"
    if not terms_path.exists():
        raise FileNotFoundError(f"Missing model term file: {terms_path}")

    source_terms = normalize_terms(_load_npy(terms_path, allow_pickle=False))
    check_unique(source_terms, f"{terms_path}")
    check_unique(target_terms, f"namespace_terms[{namespace!r}]")

    output_terms_path = output_dir / f"{namespace}.terms.npy"
    output_scores_path = output_dir / f"{namespace}.scores.{np.dtype(output_dtype).name}.npy"
    np.save(output_terms_path, np.asarray(target_terms, dtype=str))

    dense_path = numpy_dir / f"{aspect}.scores.float16.npy"
    top_scores_path, top_indices_path, inferred_top_k = find_topk_files(numpy_dir, aspect, top_k)

    if source_mode == "auto":
        if dense_path.exists():
            chosen_mode = "dense"
        elif top_scores_path is not None and top_indices_path is not None:
            chosen_mode = "topk"
        else:
            raise FileNotFoundError(
                f"No dense or top-k source found for {aspect} in {numpy_dir}. "
                f"Expected {dense_path.name} or {aspect}.top*.scores.float16.npy + indices."
            )
    else:
        chosen_mode = source_mode

    if propagate_ancestors:
        if not np.isclose(fill_value, 0.0, equal_nan=False):
            raise ValueError("Ancestor propagation requires --fill_value 0.0; zero is the max-propagation neutral value.")
        source_to_target, reachability_meta = build_source_to_target_indices(
            source_terms=source_terms,
            target_terms=target_terms,
            namespace=namespace,
            ontology=ontology,
            propagate_ancestors=True,
            canonicalize_alt_ids=canonicalize_alt_ids,
            include_self=True,
        )
        if strict and int(reachability_meta["unreachable_target_columns"]) > 0:
            examples = ", ".join(reachability_meta["unreachable_target_terms_preview"][:20])
            raise ValueError(
                f"{namespace} has {reachability_meta['unreachable_target_columns']} target GO terms that are neither "
                f"direct model terms nor ancestors of model terms. Examples: {examples}"
            )
    else:
        source_to_target = []
        reachability_meta = {}
        source_pos = {term: i for i, term in enumerate(source_terms)}
        missing_terms = [term for term in target_terms if term not in source_pos]
        if strict and missing_terms:
            examples = ", ".join(missing_terms[:20])
            raise ValueError(
                f"{namespace} has {len(missing_terms)} target GO terms absent from {terms_path}. Examples: {examples}"
            )

    if chosen_mode == "dense":
        if not dense_path.exists():
            raise FileNotFoundError(f"Dense source requested but not found: {dense_path}")
        if propagate_ancestors:
            source_meta = align_dense_with_ancestor_propagation(
                source_scores_path=dense_path,
                source_terms=source_terms,
                target_terms=target_terms,
                output_scores_path=output_scores_path,
                output_dtype=output_dtype,
                row_chunk_size=row_chunk_size,
                flush_every=flush_every,
                source_to_target_indices=source_to_target,
                min_score=min_score,
                compute_dtype=compute_dtype,
                row_indices=row_indices,
            )
        else:
            source_meta = align_dense_direct(
                source_scores_path=dense_path,
                source_terms=source_terms,
                target_terms=target_terms,
                output_scores_path=output_scores_path,
                output_dtype=output_dtype,
                fill_value=fill_value,
                row_chunk_size=row_chunk_size,
                flush_every=flush_every,
                row_indices=row_indices,
            )
    elif chosen_mode == "topk":
        if top_scores_path is None or top_indices_path is None:
            raise FileNotFoundError(f"Top-k source requested but not found for {aspect} in {numpy_dir}")
        if propagate_ancestors:
            source_meta = align_topk_with_ancestor_propagation(
                top_scores_path=top_scores_path,
                top_indices_path=top_indices_path,
                source_terms=source_terms,
                target_terms=target_terms,
                output_scores_path=output_scores_path,
                output_dtype=output_dtype,
                row_chunk_size=row_chunk_size,
                flush_every=flush_every,
                source_to_target_indices=source_to_target,
                min_score=min_score,
                compute_dtype=compute_dtype,
                row_indices=row_indices,
            )
        else:
            source_meta = align_topk_direct(
                top_scores_path=top_scores_path,
                top_indices_path=top_indices_path,
                source_terms=source_terms,
                target_terms=target_terms,
                output_scores_path=output_scores_path,
                output_dtype=output_dtype,
                fill_value=fill_value,
                row_chunk_size=row_chunk_size,
                flush_every=flush_every,
                row_indices=row_indices,
            )
    else:
        raise ValueError(f"Unsupported source mode: {chosen_mode}")

    # Build a compact target-column status table.
    mapping_rows = []
    if propagate_ancestors:
        unreachable_preview_set = set(reachability_meta.get("unreachable_target_terms_preview", []))
        # For exact status of all target columns, derive target columns covered by source_to_target.
        covered_cols = set()
        direct_cols = set()
        source_pos = {term: i for i, term in enumerate(source_terms)}
        if ontology is not None and canonicalize_alt_ids:
            source_keys = {ontology.canonicalize(term) for term in source_terms}
        else:
            source_keys = set(source_terms)
        for arr in source_to_target:
            covered_cols.update(map(int, arr))
        for j, term in enumerate(target_terms):
            term_key = ontology.canonicalize(term) if (ontology is not None and canonicalize_alt_ids) else term
            if term_key in source_keys:
                direct_cols.add(j)
        for target_j, term in enumerate(target_terms):
            if target_j in direct_cols:
                status = "direct_model_term"
            elif target_j in covered_cols:
                status = "filled_by_ancestor_propagation"
            else:
                status = "unreachable_from_model_terms"
            mapping_rows.append((namespace, aspect, target_j, term, status))
    else:
        source_pos = {term: i for i, term in enumerate(source_terms)}
        for target_j, term in enumerate(target_terms):
            status = "matched" if term in source_pos else "missing_in_model"
            mapping_rows.append((namespace, aspect, target_j, term, status))

    meta = {
        "namespace": namespace,
        "aspect": aspect,
        "source_terms_npy": str(terms_path),
        "output_terms_npy": str(output_terms_path),
        "output_scores_npy": str(output_scores_path),
        "output_dtype": np.dtype(output_dtype).name,
        "fill_value_for_unavailable_terms": fill_value if not np.isnan(fill_value) else "nan",
        "row_order": "Rows are identical to output_dir/names.npy. If --target_names_npy was used, rows follow that target label order.",
        "column_order": "Columns are exactly namespace_terms.pkl[namespace] order.",
        **reachability_meta,
        **source_meta,
    }
    save_json(output_dir / f"{namespace}.alignment.json", meta)

    return {"meta": meta, "mapping_rows": mapping_rows}


def combine_outputs(
    output_dir: Path,
    namespaces: Sequence[str],
    output_dtype: np.dtype,
    row_chunk_size: int,
    flush_every: int,
) -> None:
    term_lists = []
    arrays = []
    n_rows = None
    for ns in namespaces:
        terms = normalize_terms(_load_npy(output_dir / f"{ns}.terms.npy", allow_pickle=False))
        arr_path = output_dir / f"{ns}.scores.{np.dtype(output_dtype).name}.npy"
        arr = _load_npy(arr_path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"{arr_path} must be 2-D, got {arr.shape}")
        if n_rows is None:
            n_rows = int(arr.shape[0])
        elif n_rows != int(arr.shape[0]):
            raise ValueError(f"Row count mismatch in {arr_path}: {arr.shape[0]} vs {n_rows}")
        if len(terms) != int(arr.shape[1]):
            raise ValueError(f"Term count mismatch in {arr_path}: {len(terms)} vs {arr.shape[1]}")
        term_lists.append(terms)
        arrays.append(arr)

    if n_rows is None:
        return

    combined_terms = [term for terms in term_lists for term in terms]
    combined_cols = sum(len(terms) for terms in term_lists)
    combined_path = output_dir / f"all_namespaces.scores.{np.dtype(output_dtype).name}.npy"
    combined_terms_path = output_dir / "all_namespaces.terms.npy"
    combined_namespace_path = output_dir / "all_namespaces.namespaces.npy"

    np.save(combined_terms_path, np.asarray(combined_terms, dtype=str))
    np.save(
        combined_namespace_path,
        np.asarray([ns for ns, terms in zip(namespaces, term_lists) for _ in terms], dtype=str),
    )

    out = open_output_memmap(combined_path, output_dtype, (n_rows, combined_cols))
    offsets = np.cumsum([0] + [len(terms) for terms in term_lists])

    for chunk_no, start in enumerate(
        tqdm(range(0, n_rows, row_chunk_size), desc="combine aligned namespaces", ascii=" >=")
    ):
        end = min(start + row_chunk_size, n_rows)
        for arr, left, right in zip(arrays, offsets[:-1], offsets[1:]):
            out[start:end, left:right] = arr[start:end]
        if flush_every > 0 and (chunk_no + 1) % flush_every == 0:
            out.flush()
    out.flush()
    del out
    for arr in arrays:
        del arr

    save_json(
        output_dir / "all_namespaces.alignment.json",
        {
            "output_scores_npy": str(combined_path),
            "output_terms_npy": str(combined_terms_path),
            "output_namespace_npy": str(combined_namespace_path),
            "shape": [int(n_rows), int(combined_cols)],
            "output_dtype": np.dtype(output_dtype).name,
            "namespace_order": list(namespaces),
            "column_order": "Concatenation of namespace-specific terms in namespace_order.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align ESM2GO numpy prediction matrices to namespace_terms.pkl order, optionally with GO ancestor propagation."
    )
    parser.add_argument("--working_dir", type=Path, required=True, help="ESM2GO working directory containing names.npy")
    parser.add_argument(
        "--numpy_dir",
        type=Path,
        default=None,
        help="Directory containing ESM2GO_numpy outputs. Default: <working_dir>/ESM2GO_numpy",
    )
    parser.add_argument("--namespace_terms", type=Path, required=True, help="Path to namespace_terms.pkl")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <working_dir>/ESM2GO_numpy_aligned, or "
            "<working_dir>/ESM2GO_numpy_aligned_propagated when --propagate_ancestors is used."
        ),
    )
    parser.add_argument(
        "--names_npy",
        type=Path,
        default=None,
        help="Source protein row-order npy. Default: <working_dir>/names.npy",
    )
    parser.add_argument(
        "--target_names_npy",
        type=Path,
        default=None,
        help="Optional target protein ID order, e.g. your label matrix row-order file.",
    )
    parser.add_argument(
        "--namespaces",
        nargs="+",
        default=None,
        choices=list(NAMESPACE_TO_ASPECT.keys()),
        help="Namespaces to align. Default: all supported namespaces present in namespace_terms.pkl",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "dense", "topk"],
        default="auto",
        help="Use dense scores if available, or reconstruct from compact top-k arrays.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k value in file names when --source topk. Default: auto-detect aspect.top*.scores.float16.npy",
    )
    parser.add_argument(
        "--output_dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Aligned score array dtype.",
    )
    parser.add_argument(
        "--fill_value",
        type=str,
        default="0.0",
        help="Value for target GO terms unavailable in direct alignment. Propagation requires 0.0.",
    )
    parser.add_argument(
        "--row_chunk_size",
        type=int,
        default=1024,
        help="Rows processed per chunk. Lower this if RAM pressure is high.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=16,
        help="Flush memmap every N chunks. 0 disables periodic flushing.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Direct mode: fail if namespace_terms contains GO IDs absent from model terms. "
            "Propagation mode: fail if target GO IDs are unreachable from model terms."
        ),
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also write all_namespaces.scores.<dtype>.npy by concatenating namespace outputs.",
    )
    parser.add_argument(
        "--propagate_ancestors",
        action="store_true",
        help="Propagate each source GO score to all selected GO ancestors present in namespace_terms.",
    )
    parser.add_argument(
        "--go_obo",
        type=Path,
        default=None,
        help="GO OBO file used for ancestor propagation. Required with --propagate_ancestors unless settings['obo_file'] is importable.",
    )
    parser.add_argument(
        "--relationships",
        type=str,
        default="is_a,part_of",
        help="Comma-separated OBO relations used as parent edges. Default: is_a,part_of",
    )
    parser.add_argument(
        "--propagate_min_score",
        type=float,
        default=0.001,
        help=(
            "Only scores strictly greater than this value are propagated; others become 0. "
            "This emulates mask = preds > min_score. Use 0.0 to propagate all positive scores."
        ),
    )
    parser.add_argument(
        "--compute_dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Temporary chunk compute dtype used during max propagation. float32 is safer; float16 uses less RAM.",
    )
    parser.add_argument(
        "--no_canonicalize_alt_ids",
        action="store_true",
        help="Do not map GO alt_id to canonical IDs when matching source/target/ancestor terms.",
    )
    args = parser.parse_args()

    working_dir = args.working_dir.resolve()
    numpy_dir = (args.numpy_dir or (working_dir / "ESM2GO_numpy")).resolve()
    if args.output_dir is None:
        default_name = "ESM2GO_numpy_aligned_propagated" if args.propagate_ancestors else "ESM2GO_numpy_aligned"
        output_dir = (working_dir / default_name).resolve()
    else:
        output_dir = args.output_dir.resolve()
    names_npy = (args.names_npy or (working_dir / "names.npy")).resolve()

    if args.row_chunk_size <= 0:
        raise ValueError("--row_chunk_size must be positive")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if not names_npy.exists():
        raise FileNotFoundError(f"Missing names.npy: {names_npy}")
    if not numpy_dir.exists():
        raise FileNotFoundError(f"Missing numpy_dir: {numpy_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    namespace_terms = load_namespace_terms(args.namespace_terms.resolve())
    namespaces = resolve_namespaces(args.namespaces, namespace_terms)
    output_dtype = np.dtype(args.output_dtype)
    compute_dtype = np.dtype(args.compute_dtype)
    fill_value = parse_float_fill_value(args.fill_value)

    ontology: Optional[GoOntology] = None
    rels: List[str] = []
    go_obo_path: Optional[Path] = None
    if args.propagate_ancestors:
        rels = parse_relationships(args.relationships)
        go_obo_path = resolve_go_obo_path(args.go_obo)
        print(f"Loading GO OBO for propagation: {go_obo_path}")
        ontology = GoOntology.from_obo(go_obo_path, rels)
        print(
            f"Loaded GO OBO: {len(ontology.namespaces)} terms, "
            f"{len(ontology.alt_id_to_id)} alt_id mappings, relations={rels}"
        )

    target_names_npy = args.target_names_npy.resolve() if args.target_names_npy is not None else None
    row_indices, effective_names_npy, n_rows_from_names, row_order_note = make_row_indices(names_npy, target_names_npy)

    names_out = output_dir / "names.npy"
    if effective_names_npy != names_out:
        shutil.copy2(effective_names_npy, names_out)

    summary_tsv = output_dir / "term_alignment_summary.tsv"
    all_mapping_rows = []
    all_meta = {
        "working_dir": str(working_dir),
        "source_numpy_dir": str(numpy_dir),
        "namespace_terms_pkl": str(args.namespace_terms.resolve()),
        "output_dir": str(output_dir),
        "source_names_npy": str(names_npy),
        "target_names_npy": str(target_names_npy) if target_names_npy is not None else None,
        "names_npy": str(names_out),
        "row_order": row_order_note,
        "n_proteins": n_rows_from_names,
        "namespaces": list(namespaces),
        "source": args.source,
        "output_dtype": args.output_dtype,
        "fill_value": fill_value if not np.isnan(fill_value) else "nan",
        "propagate_ancestors": bool(args.propagate_ancestors),
        "go_obo": str(go_obo_path) if go_obo_path is not None else None,
        "relationships": rels,
        "propagate_min_score": float(args.propagate_min_score),
        "compute_dtype": args.compute_dtype,
        "canonicalize_alt_ids": bool(not args.no_canonicalize_alt_ids and ontology is not None),
    }

    for namespace in namespaces:
        print(f"Processing {namespace} ({NAMESPACE_TO_ASPECT[namespace]})")
        result = align_one_namespace(
            namespace=namespace,
            target_terms=namespace_terms[namespace],
            numpy_dir=numpy_dir,
            output_dir=output_dir,
            source_mode=args.source,
            output_dtype=output_dtype,
            fill_value=fill_value,
            row_chunk_size=args.row_chunk_size,
            top_k=args.top_k,
            strict=args.strict,
            flush_every=args.flush_every,
            row_indices=row_indices,
            propagate_ancestors=args.propagate_ancestors,
            ontology=ontology,
            canonicalize_alt_ids=(not args.no_canonicalize_alt_ids),
            min_score=args.propagate_min_score,
            compute_dtype=compute_dtype,
        )
        meta = result["meta"]
        out_scores = _load_npy(Path(meta["output_scores_npy"]), mmap_mode="r", allow_pickle=False)
        if int(out_scores.shape[0]) != n_rows_from_names:
            raise ValueError(
                f"Row count mismatch for {meta['output_scores_npy']}: "
                f"{out_scores.shape[0]} rows vs {n_rows_from_names} names."
            )
        del out_scores

        all_meta[namespace] = meta
        all_mapping_rows.extend(result["mapping_rows"])

    write_term_mapping_rows(summary_tsv, all_mapping_rows, append=False)

    if args.combine:
        combine_outputs(
            output_dir=output_dir,
            namespaces=namespaces,
            output_dtype=output_dtype,
            row_chunk_size=args.row_chunk_size,
            flush_every=args.flush_every,
        )

    save_json(output_dir / "alignment_summary.json", all_meta)
    print(f"Saved aligned outputs to: {output_dir}")
    print(f"Saved summary to: {output_dir / 'alignment_summary.json'}")
    print(f"Saved term mapping to: {summary_tsv}")


if __name__ == "__main__":
    main()
