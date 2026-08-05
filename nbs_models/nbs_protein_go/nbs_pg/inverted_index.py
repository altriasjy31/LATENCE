from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


@dataclass
class InvertedIndexFiles:
    prefix: str
    indptr: str
    protein_idx: str
    payloads: dict[str, str]
    num_go: int
    num_edges: int
    max_go_degree: int
    mean_go_degree: float
    source_layout: str
    lookup_mode: str
    source_protein_start: Optional[int] = None


def _resolve_path(base: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _check_output_paths(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "output files already exist; pass overwrite=True: " + ", ".join(existing[:8])
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and path.exists():
            path.unlink()


def _open_npy_memmap(path: Path, *, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _validate_go_indices(go_idx: np.ndarray, num_go: int, start: int) -> None:
    if go_idx.size == 0:
        return
    minimum = int(go_idx.min())
    maximum = int(go_idx.max())
    if minimum < 0 or maximum >= num_go:
        raise IndexError(
            f"GO index out of range in source edges [{start}, {start + go_idx.size}): "
            f"min={minimum}, max={maximum}, num_go={num_go}"
        )


def _scatter_sorted_chunk(
    *,
    go_idx: np.ndarray,
    values: dict[str, np.ndarray],
    cursor: np.ndarray,
    outputs: dict[str, np.memmap],
) -> None:
    """Scatter one source chunk into GO-major CSR without a global edge sort."""
    if go_idx.size == 0:
        return
    order = np.argsort(go_idx, kind="stable")
    sorted_go = go_idx[order]
    unique_go, first, counts = np.unique(
        sorted_go, return_index=True, return_counts=True
    )
    repeated_group = np.repeat(np.arange(unique_go.size, dtype=np.int64), counts)
    within_group = np.arange(sorted_go.size, dtype=np.int64) - np.repeat(first, counts)
    destination = np.repeat(cursor[unique_go], counts) + within_group
    for name, value in values.items():
        outputs[name][destination] = value[order]
    cursor[unique_go] += counts.astype(np.int64, copy=False)


def _write_indptr(counts: np.ndarray, path: Path) -> np.memmap:
    indptr = _open_npy_memmap(path, dtype=np.int64, shape=(counts.size + 1,))
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indptr.flush()
    return indptr


def build_go_inverted_from_edge_index(
    edge_index_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    prefix: str,
    num_go: int,
    chunk_edges: int = 2_000_000,
    expected_fixed_degree: Optional[int] = None,
    require_protein_major_fixed_degree: bool = False,
    overwrite: bool = False,
) -> InvertedIndexFiles:
    """Invert a global ``protein -> GO`` COO array into GO-major CSR.

    The algorithm performs two streaming passes and never sorts all edges in
    memory.  When the source is verified as protein-major fixed-K, the output
    stores a compact rank payload rather than an 8-byte source offset.  This is
    the preferred mode for the 281,457,664 LATENCE backbone-candidate edges.
    """
    if num_go <= 0:
        raise ValueError("num_go must be positive")
    if chunk_edges <= 0:
        raise ValueError("chunk_edges must be positive")
    if expected_fixed_degree is not None and expected_fixed_degree <= 0:
        raise ValueError("expected_fixed_degree must be positive")

    source_path = Path(edge_index_path)
    edge_index = np.load(source_path, mmap_mode="r")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be an npy array with shape [2, E]")
    num_edges = int(edge_index.shape[1])
    counts = np.zeros(num_go, dtype=np.int64)

    fixed_ok = expected_fixed_degree is not None and num_edges % expected_fixed_degree == 0
    first_protein: Optional[int] = None
    for start in range(0, num_edges, chunk_edges):
        end = min(start + chunk_edges, num_edges)
        protein = np.asarray(edge_index[0, start:end], dtype=np.int64)
        go_idx = np.asarray(edge_index[1, start:end], dtype=np.int64)
        _validate_go_indices(go_idx, num_go, start)
        counts += np.bincount(go_idx, minlength=num_go).astype(np.int64, copy=False)
        if expected_fixed_degree is not None and fixed_ok:
            if first_protein is None and protein.size:
                first_protein = int(protein[0])
            expected = (
                int(first_protein or 0)
                + np.arange(start, end, dtype=np.int64) // expected_fixed_degree
            )
            if not np.array_equal(protein, expected):
                fixed_ok = False

    if require_protein_major_fixed_degree and not fixed_ok:
        raise ValueError(
            "source edge_index is not protein-major fixed-degree under the requested contract"
        )

    output = Path(output_dir)
    indptr_path = output / f"{prefix}_go_indptr.i64.npy"
    protein_path = output / f"{prefix}_protein_idx.i32.npy"
    if fixed_ok:
        rank_dtype = np.uint16 if int(expected_fixed_degree or 0) <= np.iinfo(np.uint16).max else np.uint32
        rank_suffix = "u16" if rank_dtype == np.uint16 else "u32"
        lookup_path = output / f"{prefix}_source_rank.{rank_suffix}.npy"
        lookup_name = "source_rank"
        lookup_mode = "protein_major_fixed_degree_rank"
    else:
        rank_dtype = np.int64
        lookup_path = output / f"{prefix}_source_edge_offset.i64.npy"
        lookup_name = "source_edge_offset"
        lookup_mode = "source_edge_offset"
    _check_output_paths([indptr_path, protein_path, lookup_path], overwrite)

    indptr = _write_indptr(counts, indptr_path)
    protein_out = _open_npy_memmap(protein_path, dtype=np.int32, shape=(num_edges,))
    lookup_out = _open_npy_memmap(lookup_path, dtype=rank_dtype, shape=(num_edges,))
    cursor = np.asarray(indptr[:-1], dtype=np.int64).copy()

    for start in range(0, num_edges, chunk_edges):
        end = min(start + chunk_edges, num_edges)
        protein = np.asarray(edge_index[0, start:end], dtype=np.int32)
        go_idx = np.asarray(edge_index[1, start:end], dtype=np.int64)
        source_offset = np.arange(start, end, dtype=np.int64)
        if fixed_ok:
            lookup = (source_offset % int(expected_fixed_degree)).astype(rank_dtype, copy=False)
        else:
            lookup = source_offset
        _scatter_sorted_chunk(
            go_idx=go_idx,
            values={"protein": protein, "lookup": lookup},
            cursor=cursor,
            outputs={"protein": protein_out, "lookup": lookup_out},
        )

    expected_cursor = np.asarray(indptr[1:], dtype=np.int64)
    if not np.array_equal(cursor, expected_cursor):
        raise RuntimeError("GO CSR cursor did not reach the expected indptr endpoints")
    protein_out.flush()
    lookup_out.flush()

    return InvertedIndexFiles(
        prefix=prefix,
        indptr=str(indptr_path),
        protein_idx=str(protein_path),
        payloads={lookup_name: str(lookup_path)},
        num_go=num_go,
        num_edges=num_edges,
        max_go_degree=int(counts.max(initial=0)),
        mean_go_degree=float(counts.mean()),
        source_layout="protein_go_edge_index_[2,E]",
        lookup_mode=lookup_mode,
        source_protein_start=first_protein if fixed_ok else None,
    )


def load_role_global_indices(
    protein_registry_path: str | os.PathLike[str],
    *,
    role: str,
) -> np.ndarray:
    """Load ``role_row_idx -> global protein_idx`` without pandas."""
    records: list[tuple[int, int]] = []
    with Path(protein_registry_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_idx", "role", "role_row_idx"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"protein registry must contain {sorted(required)}")
        for row in reader:
            if row["role"] == role:
                records.append((int(row["role_row_idx"]), int(row["protein_idx"])))
    if not records:
        raise ValueError(f"no proteins with role={role!r} in registry")
    records.sort()
    local = np.fromiter((item[0] for item in records), dtype=np.int64)
    expected = np.arange(local.size, dtype=np.int64)
    if not np.array_equal(local, expected):
        raise ValueError(f"role_row_idx for role={role!r} is not contiguous from zero")
    return np.fromiter((item[1] for item in records), dtype=np.int64)


def build_go_inverted_from_protein_csr(
    indptr_path: str | os.PathLike[str],
    indices_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    prefix: str,
    num_go: int,
    global_protein_index: np.ndarray,
    probability_path: Optional[str | os.PathLike[str]] = None,
    chunk_edges: int = 2_000_000,
    overwrite: bool = False,
) -> InvertedIndexFiles:
    """Invert role-local protein-major CSR (e.g. weak pseudo labels) to GO CSR."""
    indptr = np.load(Path(indptr_path), mmap_mode="r")
    indices = np.load(Path(indices_path), mmap_mode="r")
    if indptr.ndim != 1 or indices.ndim != 1:
        raise ValueError("CSR indptr and indices must be one-dimensional")
    if indptr.dtype.kind not in "iu" or indices.dtype.kind not in "iu":
        raise TypeError("CSR indptr and indices must be integer arrays")
    if int(indptr[0]) != 0 or int(indptr[-1]) != int(indices.size):
        raise ValueError("invalid CSR endpoints")
    if np.any(np.diff(np.asarray(indptr)) < 0):
        raise ValueError("CSR indptr must be nondecreasing")
    if global_protein_index.shape != (indptr.size - 1,):
        raise ValueError("global_protein_index must align with CSR protein rows")
    probabilities = None
    if probability_path is not None:
        probabilities = np.load(Path(probability_path), mmap_mode="r")
        if probabilities.shape != indices.shape:
            raise ValueError("probability payload must align with CSR indices")

    num_edges = int(indices.size)
    counts = np.zeros(num_go, dtype=np.int64)
    for start in range(0, num_edges, chunk_edges):
        end = min(start + chunk_edges, num_edges)
        go_idx = np.asarray(indices[start:end], dtype=np.int64)
        _validate_go_indices(go_idx, num_go, start)
        counts += np.bincount(go_idx, minlength=num_go).astype(np.int64, copy=False)

    output = Path(output_dir)
    out_indptr_path = output / f"{prefix}_go_indptr.i64.npy"
    protein_path = output / f"{prefix}_protein_idx.i32.npy"
    paths = [out_indptr_path, protein_path]
    probability_out_path = output / f"{prefix}_probability.f16.npy"
    if probabilities is not None:
        paths.append(probability_out_path)
    _check_output_paths(paths, overwrite)

    out_indptr = _write_indptr(counts, out_indptr_path)
    protein_out = _open_npy_memmap(protein_path, dtype=np.int32, shape=(num_edges,))
    outputs: dict[str, np.memmap] = {"protein": protein_out}
    payload_paths: dict[str, str] = {}
    if probabilities is not None:
        probability_out = _open_npy_memmap(
            probability_out_path, dtype=np.float16, shape=(num_edges,)
        )
        outputs["probability"] = probability_out
        payload_paths["probability"] = str(probability_out_path)

    cursor = np.asarray(out_indptr[:-1], dtype=np.int64).copy()
    for start in range(0, num_edges, chunk_edges):
        end = min(start + chunk_edges, num_edges)
        source_offset = np.arange(start, end, dtype=np.int64)
        protein_row = np.searchsorted(indptr, source_offset, side="right") - 1
        protein = global_protein_index[protein_row].astype(np.int32, copy=False)
        go_idx = np.asarray(indices[start:end], dtype=np.int64)
        values: dict[str, np.ndarray] = {"protein": protein}
        if probabilities is not None:
            values["probability"] = np.asarray(probabilities[start:end], dtype=np.float16)
        _scatter_sorted_chunk(
            go_idx=go_idx,
            values=values,
            cursor=cursor,
            outputs=outputs,
        )

    if not np.array_equal(cursor, np.asarray(out_indptr[1:], dtype=np.int64)):
        raise RuntimeError("GO CSR cursor did not reach the expected indptr endpoints")
    for output_array in outputs.values():
        output_array.flush()

    return InvertedIndexFiles(
        prefix=prefix,
        indptr=str(out_indptr_path),
        protein_idx=str(protein_path),
        payloads=payload_paths,
        num_go=num_go,
        num_edges=num_edges,
        max_go_degree=int(counts.max(initial=0)),
        mean_go_degree=float(counts.mean()),
        source_layout="protein_major_csr",
        lookup_mode="payload_copied" if probabilities is not None else "none",
        source_protein_start=None,
    )


def build_latence_go_protein_indices(
    weak_graph_manifest_path: str | os.PathLike[str],
    protein_registry_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    build_candidate: bool = True,
    build_pseudo: bool = True,
    gold_edge_index_path: Optional[str | os.PathLike[str]] = None,
    chunk_edges: int = 2_000_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the production GO->Protein indices from LATENCE manifests."""
    started = time.time()
    manifest_path = Path(weak_graph_manifest_path)
    source_dir = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    num_go = int(source_manifest["go_registry"]["num_terms"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "builder": "nbs_pg.inverted_index.build_latence_go_protein_indices",
        "source_manifest": str(manifest_path),
        "protein_registry": str(Path(protein_registry_path)),
        "num_go": num_go,
        "indices": {},
    }

    if build_candidate:
        candidate = source_manifest["backbone_rare_edges"]
        degree_values = {
            int(role["rare_degree_mean"])
            for role in source_manifest.get("roles", [])
            if "rare_degree_mean" in role
        }
        expected_degree = degree_values.pop() if len(degree_values) == 1 else None
        candidate_result = build_go_inverted_from_edge_index(
            _resolve_path(source_dir, candidate["edge_index_file"]),
            output_dir,
            prefix="candidate",
            num_go=num_go,
            chunk_edges=chunk_edges,
            expected_fixed_degree=expected_degree,
            require_protein_major_fixed_degree=expected_degree is not None,
            overwrite=overwrite,
        )
        result["indices"]["candidate"] = asdict(candidate_result)
        result["indices"]["candidate"]["source_edge_attr"] = str(
            _resolve_path(source_dir, candidate["edge_attr_file"])
        )
        result["indices"]["candidate"]["source_edge_attr_columns"] = candidate[
            "edge_attr_columns"
        ]
        result["indices"]["candidate"]["fixed_degree"] = expected_degree
        result["indices"]["candidate"]["edge_attr_lookup"] = (
            "source_row = (global_protein_idx - source_protein_start) * "
            "fixed_degree + source_rank"
        )

    if build_pseudo:
        pseudo = source_manifest["weak_pseudo_targets"]
        role = str(pseudo["role"])
        global_idx = load_role_global_indices(protein_registry_path, role=role)
        pseudo_result = build_go_inverted_from_protein_csr(
            _resolve_path(source_dir, pseudo["csr_indptr_file"]),
            _resolve_path(source_dir, pseudo["csr_indices_file"]),
            output_dir,
            prefix="pseudo",
            num_go=num_go,
            global_protein_index=global_idx,
            probability_path=_resolve_path(source_dir, pseudo["csr_probability_file"]),
            chunk_edges=chunk_edges,
            overwrite=overwrite,
        )
        result["indices"]["pseudo"] = asdict(pseudo_result)
        result["indices"]["pseudo"]["source_role"] = role
        result["indices"]["pseudo"]["threshold"] = pseudo["threshold"]
        result["indices"]["pseudo"]["comparison"] = pseudo["comparison"]

    if gold_edge_index_path is not None:
        gold_result = build_go_inverted_from_edge_index(
            gold_edge_index_path,
            output_dir,
            prefix="gold",
            num_go=num_go,
            chunk_edges=chunk_edges,
            overwrite=overwrite,
        )
        result["indices"]["gold"] = asdict(gold_result)

    result["elapsed_seconds"] = round(time.time() - started, 3)
    output_manifest = Path(output_dir) / "go_protein_inverted_index_manifest.json"
    if output_manifest.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {output_manifest}")
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    result["manifest"] = str(output_manifest)
    return result
