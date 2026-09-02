from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _resolve(base: Path, value: str | os.PathLike[str]) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else []
    candidates.extend([base / raw, base / raw.name])
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def build_edge_offset_csr(
    edge_index_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    prefix: str,
    num_nodes: int,
    key_axis: int,
    chunk_edges: int = 2_000_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    if key_axis not in (0, 1):
        raise ValueError("key_axis must be 0 or 1")
    edge_path = Path(edge_index_path).resolve()
    edge = np.load(edge_path, mmap_mode="r")
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("edge_index must be [2,E]")
    counts = np.zeros(num_nodes, dtype=np.int64)
    total = int(edge.shape[1])
    for start in range(0, total, chunk_edges):
        end = min(start + chunk_edges, total)
        keys = np.asarray(edge[key_axis, start:end], dtype=np.int64)
        if keys.size and (keys.min() < 0 or keys.max() >= num_nodes):
            raise IndexError("sampling key outside protein index space")
        counts += np.bincount(keys, minlength=num_nodes)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    indptr_path = output / f"{prefix}_indptr.i64.npy"
    offset_path = output / f"{prefix}_edge_offset.i64.npy"
    if not overwrite and (indptr_path.exists() or offset_path.exists()):
        raise FileExistsError(f"sampling index exists for {prefix}")
    indptr = np.lib.format.open_memmap(indptr_path, mode="w+", dtype=np.int64, shape=(num_nodes + 1,))
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    offsets = np.lib.format.open_memmap(offset_path, mode="w+", dtype=np.int64, shape=(total,))
    cursor = np.asarray(indptr[:-1], dtype=np.int64).copy()
    for start in range(0, total, chunk_edges):
        end = min(start + chunk_edges, total)
        keys = np.asarray(edge[key_axis, start:end], dtype=np.int64)
        source_offset = np.arange(start, end, dtype=np.int64)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        unique, first, group_counts = np.unique(sorted_keys, return_index=True, return_counts=True)
        within = np.arange(sorted_keys.size, dtype=np.int64) - np.repeat(first, group_counts)
        destination = np.repeat(cursor[unique], group_counts) + within
        offsets[destination] = source_offset[order]
        cursor[unique] += group_counts
    if not np.array_equal(cursor, np.asarray(indptr[1:], dtype=np.int64)):
        raise RuntimeError("sampling CSR cursor mismatch")
    indptr.flush(); offsets.flush()
    return {
        "prefix": prefix,
        "key_axis": key_axis,
        "num_nodes": num_nodes,
        "num_edges": total,
        "indptr": str(indptr_path),
        "edge_offset": str(offset_path),
        "edge_index": str(edge_path),
    }


def build_pp_sampling_indices(
    pp_manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    chunk_edges: int = 2_000_000,
    overwrite: bool = False,
) -> Path:
    manifest_path = Path(pp_manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_nodes = int(payload["num_proteins"])
    relation_policy = {
        "ppi": 0,
        # Similarity edges are neighbour -> core query; sample by destination
        # so every core query sees its retained top-k incoming evidence.
        "similar_to": 1,
        # weak_to_core is explicitly weak -> core; sample by weak source.
        "weak_to_core": 0,
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "builder": "build_pp_sampling_indices_v0.4",
        "num_proteins": num_nodes,
        "source_manifest": str(manifest_path),
        "relations": {},
    }
    relation_by_name = {str(item["relation"]): item for item in payload["relations"]}
    out = Path(output_dir).resolve()
    for name, key_axis in relation_policy.items():
        spec = relation_by_name[name]
        edge_index = _resolve(manifest_path.parent, spec["edge_index_file"])
        edge_attr = _resolve(manifest_path.parent, spec["edge_attr_file"])
        built = build_edge_offset_csr(
            edge_index,
            out,
            prefix=name,
            num_nodes=num_nodes,
            key_axis=key_axis,
            chunk_edges=chunk_edges,
            overwrite=overwrite,
        )
        built["edge_attr"] = str(edge_attr)
        built["message_direction"] = "source_to_destination"
        built["sampling_key"] = "source" if key_axis == 0 else "destination"
        result["relations"][name] = built
    output_manifest = out / "pp_sampling_indices_manifest.json"
    if output_manifest.exists() and not overwrite:
        raise FileExistsError(f"sampling manifest exists: {output_manifest}")
    output_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output_manifest
