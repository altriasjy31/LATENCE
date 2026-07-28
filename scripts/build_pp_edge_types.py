#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile evidence-separated protein--protein edges for ``nbs_protein_go``.

This is the distinctly named v2 implementation.  It defaults to directed
top-100 core similarity, ``neighbor -> query`` similarity messages, and the
``weak -> core -> GO`` weak-to-strong route.

Inputs
------
``feature_dir`` must contain the classifier/backbone-aligned
``protein_registry.csv`` produced by ``export_protein_universe_repr.py``.
``pp_relations_dir`` must contain the core--core and weak--core kNN arrays
produced by ``build_pp_relations_faiss.py``.

The compiler emits three independent PyG-style relation tensors:

``protein --ppi-----------> protein``
    STRING/experimental PPI evidence.  The default output is explicitly
    bidirectional because the source relation is biologically undirected.

``protein --similar_to----> protein``
    Core--core representation similarity.  The default keeps all directed
    top-k evidence and stores ``neighbor -> query`` message edges so every
    core query can receive its retrieved neighborhood.  Union-kNN and
    mutual-kNN remain available as ablations without rerunning FAISS.

``protein --weak_to_core--> protein``
    Weak-to-core retrieval evidence.  The default message direction is the
    literal ``weak -> core`` direction, enabling the two-layer route
    ``weak -> core -> GO``.  ``core -> weak`` and bidirectional variants
    remain available as explicit ablations.

Output contract
---------------
For relation ``R``:

``pp_R_edge_index.i32.npy``
    Shape ``[2, E]``; row 0 is source protein index and row 1 is destination
    protein index in ``protein_registry.csv``.

``pp_R_edge_attr.f32.npy``
    Shape ``[E, 3]`` with columns
    ``[confidence, source_score, reciprocal_rank]``.  Column 0 is always in
    ``[0, 1]`` and is compatible with the NBS convention that edge-attribute
    column 0 is the absolute message weight.

PPI parsing and deduplication are streaming/disk-backed.  Directed top-k
relations are counted and written in row chunks, so using k=100 does not
materialize every candidate and its temporary sort keys in memory at once.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


RELATION_NAMES = ("ppi", "similar_to", "weak_to_core")
EDGE_ATTR_COLUMNS = ("confidence", "source_score", "reciprocal_rank")
BUILDER_ID = "build_pp_edge_types_v2"
BUILDER_VERSION = "2.1.0-top100-weak-to-core"


def csv_items(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile PPI, core similarity and weak/core transfer edge types"
    )
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--pp-relations-dir", type=Path, required=True)
    parser.add_argument("--ppi-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--relations",
        type=csv_items,
        default=list(RELATION_NAMES),
        help="Subset of ppi,similar_to,weak_to_core.",
    )

    parser.add_argument("--ppi-min-score", type=float, default=700.0)
    parser.add_argument("--ppi-score-scale", type=float, default=1000.0)
    parser.add_argument(
        "--ppi-direction",
        choices=["bidirectional", "as-is"],
        default="bidirectional",
        help="Bidirectional canonicalizes an undirected pair and emits both message directions.",
    )
    parser.add_argument("--ppi-insert-batch-size", type=int, default=50000)
    parser.add_argument("--ppi-fetch-batch-size", type=int, default=100000)
    parser.add_argument("--skip-malformed-ppi-lines", action="store_true")
    parser.add_argument(
        "--min-ppi-mapped-edge-fraction",
        type=float,
        default=0.0,
        help="Optional audit guard over score-qualified, non-self input rows.",
    )

    parser.add_argument("--similar-k", type=int, default=100)
    parser.add_argument("--similar-min-score", type=float, default=0.0)
    parser.add_argument(
        "--similar-mode",
        choices=["directed", "union", "mutual"],
        default="directed",
        help=(
            "directed preserves all top-k candidates; union/mutual convert "
            "candidate pairs to explicit bidirectional edges."
        ),
    )
    parser.add_argument(
        "--similar-message-direction",
        choices=["neighbor-to-query", "query-to-neighbor"],
        default="neighbor-to-query",
        help=(
            "Message direction for directed similarity. neighbor-to-query "
            "gives every FAISS query incoming evidence from its top-k neighbors."
        ),
    )
    parser.add_argument(
        "--similar-score-reduce",
        choices=["min", "mean", "max"],
        default="min",
        help="How reciprocal core-core scores are reduced for union/mutual pairs.",
    )

    parser.add_argument("--weak-k", type=int, default=100)
    parser.add_argument("--weak-min-score", type=float, default=0.0)
    parser.add_argument(
        "--weak-message-direction",
        choices=["weak-to-core", "core-to-weak", "bidirectional"],
        default="weak-to-core",
        help=(
            "Graph message direction for weak/core retrieval. weak-to-core "
            "supports the intended weak -> core -> GO two-hop route."
        ),
    )
    parser.add_argument(
        "--knn-write-chunk-rows",
        type=int,
        default=65536,
        help="Number of kNN query rows processed per count/write chunk.",
    )
    parser.add_argument(
        "--large-input-sha256",
        action="store_true",
        help="Hash the full PPI TSV in addition to recording size and mtime.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path, include_sha256: bool = False) -> Dict[str, Any]:
    stat = path.stat()
    result: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        result["sha256"] = sha256_file(path)
    return result


def atomic_write_text(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def ensure_publishable(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        preview = "\n  ".join(existing[:10])
        raise FileExistsError(
            f"Output already exists; pass --overwrite to replace it:\n  {preview}"
        )


@dataclass(frozen=True)
class ProteinRegistry:
    path: Path
    protein_ids: Tuple[str, ...]
    protein_to_global: Mapping[str, int]
    role_to_global: Mapping[str, np.ndarray]
    role_counts: Mapping[str, int]

    @property
    def num_proteins(self) -> int:
        return len(self.protein_ids)


def load_protein_registry(path: Path) -> ProteinRegistry:
    required = {"protein_idx", "protein_id", "role", "role_row_idx"}
    rows: List[Tuple[int, str, str, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Protein registry has no header: {path}")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Protein registry is missing columns {missing}: {path}")
        for line_no, row in enumerate(reader, start=2):
            try:
                protein_idx = int(row["protein_idx"])
                role_row_idx = int(row["role_row_idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid integer at {path}:{line_no}") from exc
            protein_id = str(row["protein_id"]).strip()
            role = str(row["role"]).strip()
            if not protein_id or not role or protein_idx < 0 or role_row_idx < 0:
                raise ValueError(f"Invalid registry row at {path}:{line_no}: {row}")
            rows.append((protein_idx, protein_id, role, role_row_idx))

    if not rows:
        raise ValueError(f"Protein registry is empty: {path}")
    rows.sort(key=lambda item: item[0])
    observed_global = [row[0] for row in rows]
    if observed_global != list(range(len(rows))):
        raise ValueError(
            "protein_idx must be a contiguous zero-based global index in protein_registry.csv"
        )

    protein_ids = tuple(row[1] for row in rows)
    if len(set(protein_ids)) != len(protein_ids):
        duplicates = [pid for pid, count in Counter(protein_ids).items() if count > 1]
        raise ValueError(f"Duplicate protein IDs in registry: {duplicates[:10]}")

    role_rows: MutableMapping[str, List[Tuple[int, int]]] = defaultdict(list)
    for global_idx, _, role, role_row_idx in rows:
        role_rows[role].append((role_row_idx, global_idx))

    role_to_global: Dict[str, np.ndarray] = {}
    role_counts: Dict[str, int] = {}
    for role, pairs in role_rows.items():
        pairs.sort()
        local_rows = [pair[0] for pair in pairs]
        if local_rows != list(range(len(pairs))):
            raise ValueError(
                f"role_row_idx for role={role!r} is not contiguous zero-based"
            )
        role_to_global[role] = np.asarray(
            [pair[1] for pair in pairs], dtype=np.int64
        )
        role_counts[role] = len(pairs)

    return ProteinRegistry(
        path=path,
        protein_ids=protein_ids,
        protein_to_global={pid: idx for idx, pid in enumerate(protein_ids)},
        role_to_global=role_to_global,
        role_counts=role_counts,
    )


def relation_record(
    pp_manifest: Mapping[str, Any], query_role: str
) -> Mapping[str, Any]:
    matches = [
        record
        for record in pp_manifest.get("relations", [])
        if record.get("query_role") == query_role
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one pp_relations entry for query_role={query_role!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


@dataclass(frozen=True)
class KNNRelation:
    query_role: str
    neighbors: np.ndarray
    scores: np.ndarray
    manifest: Mapping[str, Any]
    manifest_path: Path
    neighbors_path: Path
    scores_path: Path


def load_knn_relation(
    directory: Path,
    pp_manifest: Mapping[str, Any],
    query_role: str,
) -> KNNRelation:
    record = relation_record(pp_manifest, query_role)
    if query_role == "core":
        default_prefix = "pp_core_core"
    else:
        default_prefix = f"pp_{query_role}_core"
    manifest_path = directory / f"{default_prefix}_manifest.json"
    relation_manifest = read_json(manifest_path) if manifest_path.is_file() else dict(record)

    neighbor_name = relation_manifest.get("neighbors_file") or record.get("neighbors_file")
    score_name = relation_manifest.get("scores_file") or record.get("scores_file")
    if not neighbor_name or not score_name:
        raise KeyError(f"Missing neighbor/score filename for query_role={query_role}")
    neighbors_path = directory / str(neighbor_name)
    scores_path = directory / str(score_name)
    if not neighbors_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError(
            f"Missing kNN arrays for role={query_role}: {neighbors_path}, {scores_path}"
        )

    neighbors = np.load(neighbors_path, mmap_mode="r")
    scores = np.load(scores_path, mmap_mode="r")
    if neighbors.ndim != 2 or scores.ndim != 2 or neighbors.shape != scores.shape:
        raise ValueError(
            f"kNN shape mismatch for {query_role}: "
            f"neighbors={neighbors.shape}, scores={scores.shape}"
        )
    expected_query = int(
        relation_manifest.get("query_count", record.get("query_count", -1))
    )
    expected_core = int(
        relation_manifest.get("core_count", record.get("core_count", -1))
    )
    if expected_query >= 0 and neighbors.shape[0] != expected_query:
        raise ValueError(
            f"{query_role} query count differs from manifest: "
            f"array={neighbors.shape[0]}, manifest={expected_query}"
        )
    if neighbors.size:
        min_neighbor = int(neighbors.min())
        max_neighbor = int(neighbors.max())
        if min_neighbor < 0 or (expected_core >= 0 and max_neighbor >= expected_core):
            raise ValueError(
                f"Neighbor index outside core row space for {query_role}: "
                f"min={min_neighbor}, max={max_neighbor}, core_count={expected_core}"
            )
    return KNNRelation(
        query_role=query_role,
        neighbors=neighbors,
        scores=scores,
        manifest=relation_manifest,
        manifest_path=manifest_path,
        neighbors_path=neighbors_path,
        scores_path=scores_path,
    )


def edge_statistics(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    num_nodes: int,
) -> Dict[str, Any]:
    if edge_index.shape[1] == 0:
        return {
            "edge_count": 0,
            "unique_source_nodes": 0,
            "unique_destination_nodes": 0,
        }
    source = edge_index[0]
    destination = edge_index[1]
    out_degree = np.bincount(source, minlength=num_nodes)
    in_degree = np.bincount(destination, minlength=num_nodes)
    confidence = edge_attr[:, 0]
    return {
        "edge_count": int(edge_index.shape[1]),
        "self_edge_count": int(np.count_nonzero(source == destination)),
        "unique_source_nodes": int(np.count_nonzero(out_degree)),
        "unique_destination_nodes": int(np.count_nonzero(in_degree)),
        "zero_out_degree_nodes": int(np.count_nonzero(out_degree == 0)),
        "zero_in_degree_nodes": int(np.count_nonzero(in_degree == 0)),
        "out_degree_mean": float(out_degree.mean()),
        "out_degree_max": int(out_degree.max(initial=0)),
        "in_degree_mean": float(in_degree.mean()),
        "in_degree_max": int(in_degree.max(initial=0)),
        "confidence_min": float(confidence.min()),
        "confidence_mean": float(confidence.mean()),
        "confidence_max": float(confidence.max()),
    }


def describe_relation_arrays(
    relation: str,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    num_nodes: int,
    index_name: str,
    attr_name: str,
) -> Dict[str, Any]:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{relation} edge_index must be [2, E], got {edge_index.shape}")
    if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], 3):
        raise ValueError(
            f"{relation} edge_attr must be [E, 3], got {edge_attr.shape}"
        )
    if edge_index.size:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise ValueError(f"{relation} contains an out-of-range global protein index")
        if np.any(edge_index[0] == edge_index[1]):
            raise ValueError(f"{relation} contains self edges")
    if not np.isfinite(edge_attr).all():
        raise FloatingPointError(f"{relation} contains non-finite edge attributes")
    if np.any(edge_attr[:, 0] < 0.0) or np.any(edge_attr[:, 0] > 1.0):
        raise ValueError(f"{relation} confidence must remain within [0, 1]")

    stats = edge_statistics(edge_index, edge_attr, num_nodes)
    return {
        "relation": relation,
        "edge_type": ["protein", relation, "protein"],
        "edge_index_file": index_name,
        "edge_index_dtype": "int32",
        "edge_index_shape": [2, int(edge_index.shape[1])],
        "edge_attr_file": attr_name,
        "edge_attr_dtype": "float32",
        "edge_attr_shape": [int(edge_index.shape[1]), 3],
        "edge_attr_columns": list(EDGE_ATTR_COLUMNS),
        "statistics": stats,
    }


def save_relation_arrays(
    stage_dir: Path,
    relation: str,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    num_nodes: int,
) -> Dict[str, Any]:
    index_name = f"pp_{relation}_edge_index.i32.npy"
    attr_name = f"pp_{relation}_edge_attr.f32.npy"
    index_path = stage_dir / index_name
    attr_path = stage_dir / attr_name
    index_mmap = np.lib.format.open_memmap(
        index_path, mode="w+", dtype=np.int32, shape=edge_index.shape
    )
    index_mmap[:] = edge_index.astype(np.int32, copy=False)
    index_mmap.flush()
    del index_mmap
    attr_mmap = np.lib.format.open_memmap(
        attr_path, mode="w+", dtype=np.float32, shape=edge_attr.shape
    )
    attr_mmap[:] = edge_attr.astype(np.float32, copy=False)
    attr_mmap.flush()
    del attr_mmap

    return describe_relation_arrays(
        relation,
        edge_index,
        edge_attr,
        num_nodes,
        index_name,
        attr_name,
    )


def split_text_line(line: str, delimiter: str | None) -> List[str]:
    if delimiter == "\t":
        return line.rstrip("\n\r").split("\t")
    return line.split()


def open_text_auto(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def ppi_rows(
    path: Path,
    skip_malformed: bool,
) -> Iterator[Tuple[int, str, str, float]]:
    with open_text_auto(path) as handle:
        header_line = handle.readline()
        if not header_line:
            raise ValueError(f"PPI file is empty: {path}")
        delimiter = "\t" if "\t" in header_line else None
        header = [item.strip() for item in split_text_line(header_line, delimiter)]
        required = ("source", "target", "combined_score")
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(
                f"PPI header is missing {missing}; observed columns={header}"
            )
        source_col = header.index("source")
        target_col = header.index("target")
        score_col = header.index("combined_score")
        required_width = max(source_col, target_col, score_col) + 1

        for line_no, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            fields = split_text_line(line, delimiter)
            if len(fields) < required_width:
                if skip_malformed:
                    yield line_no, "", "", math.nan
                    continue
                raise ValueError(f"Malformed PPI row at {path}:{line_no}")
            try:
                score = float(fields[score_col])
            except ValueError as exc:
                if skip_malformed:
                    yield line_no, "", "", math.nan
                    continue
                raise ValueError(
                    f"Invalid combined_score at {path}:{line_no}: {fields[score_col]!r}"
                ) from exc
            yield line_no, fields[source_col].strip(), fields[target_col].strip(), score


def sqlite_upsert_batch(
    connection: sqlite3.Connection,
    records: List[Tuple[int, float]],
) -> None:
    if not records:
        return
    connection.executemany(
        """
        INSERT INTO edges(edge_key, score)
        VALUES (?, ?)
        ON CONFLICT(edge_key) DO UPDATE SET score = MAX(score, excluded.score)
        """,
        records,
    )
    connection.commit()
    records.clear()


def compile_ppi(
    stage_dir: Path,
    registry: ProteinRegistry,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    database_path = stage_dir / ".ppi_dedup.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute(
        "CREATE TABLE edges(edge_key INTEGER PRIMARY KEY, score REAL NOT NULL) WITHOUT ROWID"
    )

    n = registry.num_proteins
    batch: List[Tuple[int, float]] = []
    counts: Counter[str] = Counter()
    missing_examples: List[str] = []
    started = time.time()
    try:
        for line_no, source_id, target_id, score in ppi_rows(
            args.ppi_tsv, args.skip_malformed_ppi_lines
        ):
            counts["input_rows"] += 1
            if not source_id or not target_id or not math.isfinite(score):
                counts["malformed_rows"] += 1
                continue
            if score < 0.0 or score > args.ppi_score_scale:
                raise ValueError(
                    f"PPI score outside [0, {args.ppi_score_scale}] at "
                    f"{args.ppi_tsv}:{line_no}: {score}"
                )
            if score < args.ppi_min_score:
                counts["below_threshold_rows"] += 1
                continue
            if source_id == target_id:
                counts["self_rows"] += 1
                continue
            counts["score_qualified_nonself_rows"] += 1
            source = registry.protein_to_global.get(source_id)
            destination = registry.protein_to_global.get(target_id)
            if source is None or destination is None:
                counts["unmapped_endpoint_rows"] += 1
                if len(missing_examples) < 20:
                    missing = source_id if source is None else target_id
                    if missing not in missing_examples:
                        missing_examples.append(missing)
                continue
            counts["mapped_rows_before_dedup"] += 1
            if args.ppi_direction == "bidirectional":
                left, right = sorted((source, destination))
                edge_key = left * n + right
            else:
                edge_key = source * n + destination
            if edge_key > np.iinfo(np.int64).max:
                raise OverflowError("Packed PPI edge key exceeds signed int64")
            batch.append((int(edge_key), float(score)))
            if len(batch) >= args.ppi_insert_batch_size:
                sqlite_upsert_batch(connection, batch)
                if counts["mapped_rows_before_dedup"] % (
                    args.ppi_insert_batch_size * 20
                ) == 0:
                    print(
                        f"[ppi] mapped rows={counts['mapped_rows_before_dedup']:,}",
                        flush=True,
                    )
        sqlite_upsert_batch(connection, batch)

        qualified = counts["score_qualified_nonself_rows"]
        mapped_fraction = (
            counts["mapped_rows_before_dedup"] / qualified if qualified else 0.0
        )
        if qualified and mapped_fraction < args.min_ppi_mapped_edge_fraction:
            raise RuntimeError(
                "PPI mapped-edge fraction is below the requested audit guard: "
                f"{mapped_fraction:.6f} < {args.min_ppi_mapped_edge_fraction:.6f}"
            )

        unique_pairs = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        if unique_pairs <= 0:
            raise RuntimeError(
                "No PPI edges survived thresholding and registry mapping. "
                f"Missing endpoint examples={missing_examples[:10]}"
            )
        edge_count = unique_pairs * (2 if args.ppi_direction == "bidirectional" else 1)
        index_name = "pp_ppi_edge_index.i32.npy"
        attr_name = "pp_ppi_edge_attr.f32.npy"
        edge_index = np.lib.format.open_memmap(
            stage_dir / index_name,
            mode="w+",
            dtype=np.int32,
            shape=(2, edge_count),
        )
        edge_attr = np.lib.format.open_memmap(
            stage_dir / attr_name,
            mode="w+",
            dtype=np.float32,
            shape=(edge_count, 3),
        )
        cursor = connection.execute("SELECT edge_key, score FROM edges ORDER BY edge_key")
        offset = 0
        while True:
            rows = cursor.fetchmany(args.ppi_fetch_batch_size)
            if not rows:
                break
            keys = np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
            raw_scores = np.fromiter(
                (row[1] for row in rows), dtype=np.float32, count=len(rows)
            )
            source = keys // n
            destination = keys % n
            confidence = np.clip(
                raw_scores / float(args.ppi_score_scale), 0.0, 1.0
            ).astype(np.float32, copy=False)
            if args.ppi_direction == "bidirectional":
                size = len(rows)
                positions = np.arange(offset, offset + 2 * size, 2)
                edge_index[0, positions] = source
                edge_index[1, positions] = destination
                edge_index[0, positions + 1] = destination
                edge_index[1, positions + 1] = source
                edge_attr[positions, 0] = confidence
                edge_attr[positions + 1, 0] = confidence
                edge_attr[positions, 1] = confidence
                edge_attr[positions + 1, 1] = confidence
                edge_attr[offset : offset + 2 * size, 2] = 1.0
                offset += 2 * size
            else:
                size = len(rows)
                edge_index[0, offset : offset + size] = source
                edge_index[1, offset : offset + size] = destination
                edge_attr[offset : offset + size, 0] = confidence
                edge_attr[offset : offset + size, 1] = confidence
                edge_attr[offset : offset + size, 2] = 1.0
                offset += size
        if offset != edge_count:
            raise RuntimeError(f"PPI output count mismatch: wrote={offset}, expected={edge_count}")
        edge_index.flush()
        edge_attr.flush()
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

    record = describe_relation_arrays(
        "ppi",
        edge_index,
        edge_attr,
        registry.num_proteins,
        index_name,
        attr_name,
    )
    del edge_index, edge_attr
    source_stats = {
        "input": file_provenance(
            args.ppi_tsv, include_sha256=args.large_input_sha256
        ),
        "source_columns": ["source", "target", "combined_score"],
        "minimum_combined_score": float(args.ppi_min_score),
        "score_scale": float(args.ppi_score_scale),
        "direction_policy": args.ppi_direction,
        "duplicate_score_reduce": "max",
        "input_counts": dict(counts),
        "unique_database_pairs_or_directed_edges": unique_pairs,
        "mapped_edge_fraction": mapped_fraction,
        "unmapped_protein_examples": missing_examples,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    return record, source_stats


def reduce_pair_scores(
    values: np.ndarray,
    starts: np.ndarray,
    reduce: str,
) -> np.ndarray:
    if reduce == "min":
        return np.minimum.reduceat(values, starts)
    if reduce == "max":
        return np.maximum.reduceat(values, starts)
    if reduce == "mean":
        counts = np.diff(np.r_[starts, values.size])
        return np.add.reduceat(values, starts) / counts
    raise ValueError(f"Unknown score reduction: {reduce}")


def iter_valid_knn_chunks(
    relation: KNNRelation,
    k: int,
    minimum_score: float,
    chunk_rows: int,
    *,
    drop_query_equals_neighbor: bool,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield thresholded kNN records without expanding the full matrix.

    The returned arrays are ``query_local, neighbor_local, raw_score, rank``.
    Query IDs address the relation's query-role row space; neighbor IDs always
    address the core row space.
    """
    num_queries = relation.neighbors.shape[0]
    for row_start in range(0, num_queries, chunk_rows):
        row_stop = min(row_start + chunk_rows, num_queries)
        neighbors = np.asarray(
            relation.neighbors[row_start:row_stop, :k], dtype=np.int64
        )
        scores = np.asarray(
            relation.scores[row_start:row_stop, :k], dtype=np.float32
        )
        keep = np.isfinite(scores) & (scores >= minimum_score)
        if drop_query_equals_neighbor:
            query_rows = np.arange(
                row_start, row_stop, dtype=np.int64
            )[:, None]
            keep &= neighbors != query_rows
        local_rows, ranks = np.nonzero(keep)
        if local_rows.size == 0:
            continue
        yield (
            local_rows.astype(np.int64, copy=False) + row_start,
            neighbors[local_rows, ranks],
            scores[local_rows, ranks],
            ranks.astype(np.int32, copy=False),
        )


def count_valid_knn_edges(
    relation: KNNRelation,
    k: int,
    minimum_score: float,
    chunk_rows: int,
    *,
    drop_query_equals_neighbor: bool,
) -> int:
    count = 0
    for query, _, _, _ in iter_valid_knn_chunks(
        relation,
        k,
        minimum_score,
        chunk_rows,
        drop_query_equals_neighbor=drop_query_equals_neighbor,
    ):
        count += int(query.size)
    return count


def compile_directed_knn(
    stage_dir: Path,
    relation_name: str,
    relation: KNNRelation,
    query_to_global: np.ndarray,
    neighbor_to_global: np.ndarray,
    k: int,
    minimum_score: float,
    chunk_rows: int,
    num_nodes: int,
    *,
    message_direction: str,
    drop_query_equals_neighbor: bool,
) -> Tuple[Dict[str, Any], int]:
    """Write a directed kNN relation in two streaming passes.

    ``message_direction`` is expressed relative to FAISS retrieval:
    ``query-to-neighbor``, ``neighbor-to-query`` or ``bidirectional``.
    """
    if message_direction not in {
        "query-to-neighbor",
        "neighbor-to-query",
        "bidirectional",
    }:
        raise ValueError(f"Unknown kNN message direction: {message_direction}")

    candidate_count = count_valid_knn_edges(
        relation,
        k,
        minimum_score,
        chunk_rows,
        drop_query_equals_neighbor=drop_query_equals_neighbor,
    )
    direction_factor = 2 if message_direction == "bidirectional" else 1
    edge_count = candidate_count * direction_factor
    index_name = f"pp_{relation_name}_edge_index.i32.npy"
    attr_name = f"pp_{relation_name}_edge_attr.f32.npy"
    edge_index = np.lib.format.open_memmap(
        stage_dir / index_name,
        mode="w+",
        dtype=np.int32,
        shape=(2, edge_count),
    )
    edge_attr = np.lib.format.open_memmap(
        stage_dir / attr_name,
        mode="w+",
        dtype=np.float32,
        shape=(edge_count, 3),
    )

    offset = 0
    for query_local, neighbor_local, raw_score, rank in iter_valid_knn_chunks(
        relation,
        k,
        minimum_score,
        chunk_rows,
        drop_query_equals_neighbor=drop_query_equals_neighbor,
    ):
        query_global = query_to_global[query_local]
        neighbor_global = neighbor_to_global[neighbor_local]
        size = int(query_global.size)
        if message_direction == "query-to-neighbor":
            source, destination = query_global, neighbor_global
        else:
            source, destination = neighbor_global, query_global

        stop = offset + size
        edge_index[0, offset:stop] = source
        edge_index[1, offset:stop] = destination
        confidence = np.clip(raw_score, 0.0, 1.0)
        edge_attr[offset:stop, 0] = confidence
        edge_attr[offset:stop, 1] = raw_score
        edge_attr[offset:stop, 2] = 1.0 / (rank.astype(np.float32) + 1.0)
        offset = stop

        if message_direction == "bidirectional":
            stop = offset + size
            edge_index[0, offset:stop] = query_global
            edge_index[1, offset:stop] = neighbor_global
            edge_attr[offset:stop, 0] = confidence
            edge_attr[offset:stop, 1] = raw_score
            edge_attr[offset:stop, 2] = 1.0 / (
                rank.astype(np.float32) + 1.0
            )
            offset = stop

    if offset != edge_count:
        raise RuntimeError(
            f"{relation_name} output count mismatch: "
            f"wrote={offset}, expected={edge_count}"
        )
    edge_index.flush()
    edge_attr.flush()
    record = describe_relation_arrays(
        relation_name,
        edge_index,
        edge_attr,
        num_nodes,
        index_name,
        attr_name,
    )
    del edge_index, edge_attr
    return record, candidate_count


def compile_similar_to(
    stage_dir: Path,
    registry: ProteinRegistry,
    relation: KNNRelation,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if "core" not in registry.role_to_global:
        raise KeyError("protein_registry.csv has no core role")
    core_to_global = registry.role_to_global["core"]
    manifest_core_count = int(relation.manifest.get("core_count", -1))
    if manifest_core_count >= 0 and manifest_core_count != core_to_global.size:
        raise ValueError(
            "core count differs between kNN manifest and protein registry: "
            f"manifest={manifest_core_count}, registry={core_to_global.size}"
        )
    if relation.neighbors.shape[0] != core_to_global.size:
        raise ValueError(
            "core-core query rows do not align with core role rows: "
            f"knn={relation.neighbors.shape[0]}, registry={core_to_global.size}"
        )
    k = min(args.similar_k, relation.neighbors.shape[1])
    if k <= 0:
        raise ValueError("--similar-k must be positive")

    if args.similar_mode == "directed":
        record, directed_candidates = compile_directed_knn(
            stage_dir=stage_dir,
            relation_name="similar_to",
            relation=relation,
            query_to_global=core_to_global,
            neighbor_to_global=core_to_global,
            k=k,
            minimum_score=args.similar_min_score,
            chunk_rows=args.knn_write_chunk_rows,
            num_nodes=registry.num_proteins,
            message_direction=args.similar_message_direction,
            drop_query_equals_neighbor=True,
        )
        source_stats = {
            "retrieval_semantics": "core query -> core neighbor",
            "graph_message_direction": args.similar_message_direction,
            "mode": args.similar_mode,
            "requested_k": int(args.similar_k),
            "resolved_k": int(k),
            "minimum_cosine_score": float(args.similar_min_score),
            "reciprocal_score_reduce": None,
            "directed_candidates_after_threshold": directed_candidates,
            "reciprocal_unordered_pairs_in_candidates": None,
            "streaming_write": {
                "enabled": True,
                "chunk_query_rows": int(args.knn_write_chunk_rows),
                "passes": 2,
            },
            "source_manifest": str(relation.manifest_path),
            "source_neighbors": file_provenance(relation.neighbors_path),
            "source_scores": file_provenance(relation.scores_path),
        }
        return record, source_stats

    n_core = core_to_global.size
    query_local = np.repeat(np.arange(n_core, dtype=np.int64), k)
    neighbor_local = np.asarray(relation.neighbors[:, :k], dtype=np.int64).reshape(-1)
    raw_score = np.asarray(relation.scores[:, :k], dtype=np.float32).reshape(-1)
    rank = np.tile(np.arange(k, dtype=np.int32), n_core)
    keep = (
        np.isfinite(raw_score)
        & (raw_score >= args.similar_min_score)
        & (query_local != neighbor_local)
    )
    query_local = query_local[keep]
    neighbor_local = neighbor_local[keep]
    raw_score = raw_score[keep]
    rank = rank[keep]
    source = core_to_global[query_local]
    destination = core_to_global[neighbor_local]
    rank_weight = (1.0 / (rank.astype(np.float32) + 1.0)).astype(
        np.float32, copy=False
    )

    directed_candidates = int(source.size)
    if source.size == 0:
        source = np.empty(0, dtype=np.int64)
        destination = np.empty(0, dtype=np.int64)
        raw_score = np.empty(0, dtype=np.float32)
        rank_weight = np.empty(0, dtype=np.float32)
        reciprocal_pairs = 0
    else:
        left = np.minimum(source, destination)
        right = np.maximum(source, destination)
        pair_key = left * registry.num_proteins + right
        direction_bit = np.where(source == left, 1, 2).astype(np.uint8)
        order = np.argsort(pair_key, kind="stable")
        pair_key = pair_key[order]
        direction_bit = direction_bit[order]
        raw_score = raw_score[order]
        rank_weight = rank_weight[order]
        starts = np.r_[0, np.flatnonzero(pair_key[1:] != pair_key[:-1]) + 1]
        unique_key = pair_key[starts]
        bitmask = np.bitwise_or.reduceat(direction_bit, starts)
        pair_score = reduce_pair_scores(
            raw_score, starts, args.similar_score_reduce
        ).astype(np.float32, copy=False)
        pair_rank = np.maximum.reduceat(rank_weight, starts).astype(
            np.float32, copy=False
        )
        reciprocal_pairs = int(np.count_nonzero(bitmask == 3))
        if args.similar_mode == "mutual":
            pair_keep = bitmask == 3
            unique_key = unique_key[pair_keep]
            pair_score = pair_score[pair_keep]
            pair_rank = pair_rank[pair_keep]
        left = unique_key // registry.num_proteins
        right = unique_key % registry.num_proteins
        source = np.concatenate((left, right))
        destination = np.concatenate((right, left))
        raw_score = np.concatenate((pair_score, pair_score))
        rank_weight = np.concatenate((pair_rank, pair_rank))

    confidence = np.clip(raw_score, 0.0, 1.0).astype(np.float32, copy=False)
    edge_index = np.stack((source, destination), axis=0)
    edge_attr = np.stack((confidence, raw_score, rank_weight), axis=1)
    record = save_relation_arrays(
        stage_dir, "similar_to", edge_index, edge_attr, registry.num_proteins
    )
    source_stats = {
        "retrieval_semantics": "core query -> core neighbor",
        "graph_message_direction": "explicitly bidirectional",
        "mode": args.similar_mode,
        "requested_k": int(args.similar_k),
        "resolved_k": int(k),
        "minimum_cosine_score": float(args.similar_min_score),
        "reciprocal_score_reduce": args.similar_score_reduce,
        "directed_candidates_after_threshold": directed_candidates,
        "reciprocal_unordered_pairs_in_candidates": reciprocal_pairs,
        "streaming_write": {
            "enabled": False,
            "reason": "union/mutual pair reduction requires reciprocal grouping",
        },
        "source_manifest": str(relation.manifest_path),
        "source_neighbors": file_provenance(relation.neighbors_path),
        "source_scores": file_provenance(relation.scores_path),
    }
    return record, source_stats


def compile_weak_to_core(
    stage_dir: Path,
    registry: ProteinRegistry,
    relation: KNNRelation,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for role in ("core", "weak"):
        if role not in registry.role_to_global:
            raise KeyError(f"protein_registry.csv has no {role!r} role")
    core_to_global = registry.role_to_global["core"]
    weak_to_global = registry.role_to_global["weak"]
    manifest_core_count = int(relation.manifest.get("core_count", -1))
    if manifest_core_count >= 0 and manifest_core_count != core_to_global.size:
        raise ValueError(
            "core count differs between weak-core manifest and protein registry: "
            f"manifest={manifest_core_count}, registry={core_to_global.size}"
        )
    if relation.neighbors.shape[0] != weak_to_global.size:
        raise ValueError(
            "weak-core query rows do not align with weak role rows: "
            f"knn={relation.neighbors.shape[0]}, registry={weak_to_global.size}"
        )
    k = min(args.weak_k, relation.neighbors.shape[1])
    if k <= 0:
        raise ValueError("--weak-k must be positive")

    direction_lookup = {
        "weak-to-core": "query-to-neighbor",
        "core-to-weak": "neighbor-to-query",
        "bidirectional": "bidirectional",
    }
    record, candidate_count = compile_directed_knn(
        stage_dir=stage_dir,
        relation_name="weak_to_core",
        relation=relation,
        query_to_global=weak_to_global,
        neighbor_to_global=core_to_global,
        k=k,
        minimum_score=args.weak_min_score,
        chunk_rows=args.knn_write_chunk_rows,
        num_nodes=registry.num_proteins,
        message_direction=direction_lookup[args.weak_message_direction],
        drop_query_equals_neighbor=False,
    )
    source_stats = {
        "retrieval_direction": "weak->core",
        "graph_message_direction": args.weak_message_direction,
        "intended_two_hop_route": (
            "protein[weak] --weak_to_core--> protein[core] "
            "--annotated_with--> GO"
            if args.weak_message_direction in {"weak-to-core", "bidirectional"}
            else None
        ),
        "reverse_message_edges_emitted": (
            args.weak_message_direction == "bidirectional"
        ),
        "requested_k": int(args.weak_k),
        "resolved_k": int(k),
        "minimum_cosine_score": float(args.weak_min_score),
        "directed_candidates_after_threshold": candidate_count,
        "streaming_write": {
            "enabled": True,
            "chunk_query_rows": int(args.knn_write_chunk_rows),
            "passes": 2,
        },
        "source_manifest": str(relation.manifest_path),
        "source_neighbors": file_provenance(relation.neighbors_path),
        "source_scores": file_provenance(relation.scores_path),
    }
    return record, source_stats


def validate_args(args: argparse.Namespace) -> None:
    invalid_relations = sorted(set(args.relations) - set(RELATION_NAMES))
    if invalid_relations:
        raise ValueError(
            f"Unknown --relations {invalid_relations}; expected {RELATION_NAMES}"
        )
    if len(set(args.relations)) != len(args.relations):
        raise ValueError(f"Duplicate --relations values: {args.relations}")
    if args.ppi_min_score < 0 or args.ppi_score_scale <= 0:
        raise ValueError("--ppi-min-score must be non-negative and --ppi-score-scale positive")
    if args.ppi_min_score > args.ppi_score_scale:
        raise ValueError("--ppi-min-score cannot exceed --ppi-score-scale")
    if args.ppi_insert_batch_size <= 0 or args.ppi_fetch_batch_size <= 0:
        raise ValueError("PPI batch sizes must be positive")
    if args.similar_k <= 0 or args.weak_k <= 0:
        raise ValueError("--similar-k and --weak-k must be positive")
    if args.knn_write_chunk_rows <= 0:
        raise ValueError("--knn-write-chunk-rows must be positive")
    if not 0.0 <= args.min_ppi_mapped_edge_fraction <= 1.0:
        raise ValueError("--min-ppi-mapped-edge-fraction must be within [0, 1]")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.feature_dir = args.feature_dir.expanduser().resolve()
    args.pp_relations_dir = args.pp_relations_dir.expanduser().resolve()
    args.ppi_tsv = args.ppi_tsv.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    validate_args(args)

    representation_manifest_path = args.feature_dir / "representation_manifest.json"
    registry_path = args.feature_dir / "protein_registry.csv"
    pp_manifest_path = args.pp_relations_dir / "pp_relations_manifest.json"
    for path in (representation_manifest_path, registry_path, pp_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if "ppi" in args.relations and not args.ppi_tsv.is_file():
        raise FileNotFoundError(args.ppi_tsv)

    representation_manifest = read_json(representation_manifest_path)
    pp_manifest = read_json(pp_manifest_path)
    if (
        representation_manifest.get("task") is not None
        and pp_manifest.get("task") is not None
        and representation_manifest.get("task") != pp_manifest.get("task")
    ):
        raise ValueError(
            "Task mismatch between representation and pp_relations manifests: "
            f"{representation_manifest.get('task')} != {pp_manifest.get('task')}"
        )
    registry = load_protein_registry(registry_path)
    if registry.num_proteins > np.iinfo(np.int32).max:
        raise OverflowError("Global protein count exceeds int32 edge-index capacity")

    print(
        f"[Implementation] {BUILDER_ID} "
        f"version={BUILDER_VERSION} file={Path(__file__).name}",
        flush=True,
    )

    required_final_paths: List[Path] = [args.output_dir / "pp_edge_types_manifest.json"]
    for relation_name in args.relations:
        required_final_paths.extend(
            [
                args.output_dir / f"pp_{relation_name}_edge_index.i32.npy",
                args.output_dir / f"pp_{relation_name}_edge_attr.f32.npy",
            ]
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_publishable(required_final_paths, args.overwrite)
    stage_dir = Path(tempfile.mkdtemp(prefix=".pp_edge_types_", dir=args.output_dir))

    started = time.time()
    relation_outputs: List[Dict[str, Any]] = []
    source_details: Dict[str, Any] = {}
    try:
        if "ppi" in args.relations:
            record, source = compile_ppi(stage_dir, registry, args)
            relation_outputs.append(record)
            source_details["ppi"] = source
            print(f"[ppi] edges={record['statistics']['edge_count']:,}", flush=True)

        if "similar_to" in args.relations:
            core_relation = load_knn_relation(
                args.pp_relations_dir, pp_manifest, "core"
            )
            record, source = compile_similar_to(
                stage_dir, registry, core_relation, args
            )
            relation_outputs.append(record)
            source_details["similar_to"] = source
            print(
                f"[similar_to] edges={record['statistics']['edge_count']:,}",
                flush=True,
            )

        if "weak_to_core" in args.relations:
            weak_relation = load_knn_relation(
                args.pp_relations_dir, pp_manifest, "weak"
            )
            record, source = compile_weak_to_core(
                stage_dir, registry, weak_relation, args
            )
            relation_outputs.append(record)
            source_details["weak_to_core"] = source
            print(
                f"[weak_to_core] edges={record['statistics']['edge_count']:,}",
                flush=True,
            )

        manifest = {
            "schema_version": 2,
            "builder": {
                "id": BUILDER_ID,
                "version": BUILDER_VERSION,
                "file": Path(__file__).name,
            },
            "task": representation_manifest.get("task"),
            "node_type": "protein",
            "num_proteins": registry.num_proteins,
            "global_protein_index": (
                "protein_idx in protein_registry.csv; edge_index row 0 is message "
                "source and row 1 is message destination"
            ),
            "protein_registry": {
                **file_provenance(registry_path, include_sha256=True),
                "role_counts": dict(registry.role_counts),
            },
            "nbs_edge_attribute_contract": {
                "columns": list(EDGE_ATTR_COLUMNS),
                "column_0_is_absolute_message_weight": True,
                "confidence_range": [0.0, 1.0],
                "source_score_semantics": {
                    "ppi": "combined_score / ppi_score_scale",
                    "similar_to": "raw cosine similarity",
                    "weak_to_core": "raw cosine similarity",
                },
            },
            "routing_contract": {
                "similar_to": {
                    "retrieval_direction": "query->neighbor",
                    "default_message_direction": "neighbor->query",
                    "reason": (
                        "each core query receives its retained top-k evidence; "
                        "mutual filtering is optional rather than required"
                    ),
                },
                "weak_to_core": {
                    "default_message_direction": "weak->core",
                    "two_hop_path": (
                        "weak protein -> core protein -> GO through "
                        "annotated_with"
                    ),
                    "minimum_message_passing_layers": 2,
                },
                "k_hop_closure": {
                    "materialized_as_new_evidence_edges": False,
                    "recommended_k": 2,
                    "recommended_edge_dir_from_go_seeds": "in",
                    "helper_module": "hetero_k_hop_closure_v2.py",
                },
            },
            "relations": relation_outputs,
            "source_details": source_details,
            "source_manifests": {
                "representation": file_provenance(
                    representation_manifest_path, include_sha256=True
                ),
                "pp_relations": file_provenance(
                    pp_manifest_path, include_sha256=True
                ),
            },
            "runtime": {
                "elapsed_seconds": round(time.time() - started, 3),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
            },
        }
        atomic_write_text(
            stage_dir / "pp_edge_types_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )

        for staged_path in sorted(stage_dir.iterdir()):
            if staged_path.name.startswith("."):
                continue
            final_path = args.output_dir / staged_path.name
            os.replace(staged_path, final_path)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()