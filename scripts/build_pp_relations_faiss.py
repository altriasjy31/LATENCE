#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build core-core and weak-core protein relations with cosine kNN.

The FAISS index contains *core proteins only*.  Core proteins query that index
to produce core-core candidates; weak proteins query the same index to produce
weak-core candidates.  The saved neighbor IDs are zero-based row indices in
``core_repr.*.npy``.  Query row IDs are implicit in the first array dimension.

No matrix averaging or implicit symmetrization is performed.  The output is a
directed retrieval result, leaving union-kNN/mutual-kNN/message-direction
choices to the graph compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def csv_items(value: str) -> List[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return items


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build directed core-core / weak-core cosine-kNN candidates"
    )
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--relation", choices=["core-core", "weak-core", "all"], default="all")
    p.add_argument(
        "--query-roles",
        type=csv_items,
        default=csv_items("weak"),
        help="Roles queried against core for weak-core/all mode; e.g. weak,valid,ind_test.",
    )
    p.add_argument("--kmax", type=int, default=100)
    p.add_argument("--query-batch-size", type=int, default=8192)
    p.add_argument("--add-batch-size", type=int, default=32768)
    p.add_argument("--score-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--backend", choices=["faiss", "numpy"], default="faiss")
    p.add_argument("--faiss-gpu-id", type=int, default=-1, help="-1 uses CPU IndexFlatIP.")
    p.add_argument("--faiss-threads", type=int, default=0, help="0 leaves FAISS default unchanged.")
    p.add_argument(
        "--numpy-index-block-size",
        type=int,
        default=32768,
        help="Reference/debug backend only; bounds the query-by-core score matrix.",
    )
    p.add_argument("--allow-zero-norm", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return obj


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def ensure_writable(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        preview = "\n  ".join(existing[:10])
        raise FileExistsError(f"Output already exists; use --overwrite to replace it:\n  {preview}")


def hash_ids_file(path: Path) -> Tuple[str, int]:
    h = hashlib.sha256()
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            protein_id = line.rstrip("\n\r")
            h.update(protein_id.encode("utf-8"))
            h.update(b"\n")
            count += 1
    return h.hexdigest(), count


def role_record(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    matches = [r for r in manifest.get("roles", []) if r.get("role") == role]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one manifest entry for role={role!r}, found {len(matches)}")
    return matches[0]


def load_role(feature_dir: Path, manifest: Mapping[str, Any], role: str) -> Tuple[np.ndarray, Mapping[str, Any]]:
    record = role_record(manifest, role)
    feature_path = feature_dir / str(record["feature_file"])
    ids_path = feature_dir / str(record["protein_ids_file"])
    if not feature_path.is_file() or not ids_path.is_file():
        raise FileNotFoundError(f"Missing role files for {role}: {feature_path}, {ids_path}")
    x = np.load(feature_path, mmap_mode="r")
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D feature array for {role}, got {x.shape}")
    if int(record.get("feature_dim", x.shape[1])) != x.shape[1]:
        raise ValueError(
            f"Role {role} feature dimension differs from manifest: "
            f"array={x.shape[1]}, manifest={record.get('feature_dim')}"
        )
    digest, id_count = hash_ids_file(ids_path)
    if id_count != x.shape[0] or int(record["count"]) != x.shape[0]:
        raise ValueError(
            f"Role {role} alignment failure: features={x.shape[0]}, ids={id_count}, "
            f"manifest={record['count']}"
        )
    expected_digest = record.get("protein_ids_sha256")
    if expected_digest and digest != expected_digest:
        raise ValueError(f"Protein ID hash mismatch for role={role}")
    return x, record


def normalized_float32_chunk(
    x: np.ndarray,
    start: int,
    end: int,
    *,
    role: str,
    allow_zero_norm: bool,
) -> Tuple[np.ndarray, int, float, float]:
    chunk = np.asarray(x[start:end], dtype=np.float32).copy()
    if not np.isfinite(chunk).all():
        bad = int(np.size(chunk) - np.isfinite(chunk).sum())
        raise FloatingPointError(f"Non-finite values in role={role}, rows={start}:{end}, count={bad}")
    norms = np.linalg.norm(chunk, axis=1)
    zero = norms <= 1e-12
    zero_count = int(zero.sum())
    if zero_count and not allow_zero_norm:
        examples = (np.flatnonzero(zero)[:10] + start).tolist()
        raise ValueError(
            f"Zero-norm vectors in role={role}, count={zero_count}, example rows={examples}. "
            "Use --allow-zero-norm only for a deliberate diagnostic."
        )
    safe = np.where(zero, 1.0, norms)
    chunk /= safe[:, None]
    return chunk, zero_count, float(norms.min(initial=np.inf)), float(norms.max(initial=0.0))


class FaissExactIP:
    def __init__(
        self,
        core: np.ndarray,
        *,
        add_batch_size: int,
        gpu_id: int,
        threads: int,
        allow_zero_norm: bool,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS is required for --backend faiss. Install faiss-cpu (or faiss-gpu), "
                "or use --backend numpy only for a small reference test."
            ) from exc
        self.faiss = faiss
        if threads > 0:
            faiss.omp_set_num_threads(int(threads))
        cpu_index = faiss.IndexFlatIP(int(core.shape[1]))
        self.gpu_resources = None
        if gpu_id >= 0:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("Installed FAISS build has no GPU support")
            self.gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, int(gpu_id), cpu_index)
            self.device = f"gpu:{gpu_id}"
        else:
            self.index = cpu_index
            self.device = "cpu"

        self.zero_norm_count = 0
        self.norm_min = math.inf
        self.norm_max = 0.0
        for start in range(0, core.shape[0], add_batch_size):
            end = min(start + add_batch_size, core.shape[0])
            chunk, zero, norm_min, norm_max = normalized_float32_chunk(
                core,
                start,
                end,
                role="core",
                allow_zero_norm=allow_zero_norm,
            )
            self.index.add(np.ascontiguousarray(chunk))
            self.zero_norm_count += zero
            self.norm_min = min(self.norm_min, norm_min)
            self.norm_max = max(self.norm_max, norm_max)
        if self.index.ntotal != core.shape[0]:
            raise RuntimeError(f"FAISS index size {self.index.ntotal} != core rows {core.shape[0]}")

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores, neighbors = self.index.search(np.ascontiguousarray(query), int(k))
        return scores, neighbors


class NumpyExactIP:
    """Memory-bounded exact reference backend for tests and small datasets."""

    def __init__(
        self,
        core: np.ndarray,
        *,
        index_block_size: int,
        allow_zero_norm: bool,
    ) -> None:
        self.core = core
        self.index_block_size = int(index_block_size)
        self.allow_zero_norm = allow_zero_norm
        self.device = "cpu-reference"
        self.zero_norm_count = 0
        self.norm_min = math.inf
        self.norm_max = 0.0
        # Audit once.  Chunks are normalized again during search so the full
        # float32 core matrix never has to reside in RAM.
        for start in range(0, core.shape[0], self.index_block_size):
            end = min(start + self.index_block_size, core.shape[0])
            _, zero, norm_min, norm_max = normalized_float32_chunk(
                core, start, end, role="core", allow_zero_norm=allow_zero_norm
            )
            self.zero_norm_count += zero
            self.norm_min = min(self.norm_min, norm_min)
            self.norm_max = max(self.norm_max, norm_max)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        n_query = query.shape[0]
        best_scores = np.full((n_query, k), -np.inf, dtype=np.float32)
        best_indices = np.full((n_query, k), -1, dtype=np.int64)
        for start in range(0, self.core.shape[0], self.index_block_size):
            end = min(start + self.index_block_size, self.core.shape[0])
            core_chunk, _, _, _ = normalized_float32_chunk(
                self.core,
                start,
                end,
                role="core",
                allow_zero_norm=self.allow_zero_norm,
            )
            block_scores = query @ core_chunk.T
            block_indices = np.broadcast_to(
                np.arange(start, end, dtype=np.int64)[None, :], block_scores.shape
            )
            merged_scores = np.concatenate((best_scores, block_scores), axis=1)
            merged_indices = np.concatenate((best_indices, block_indices), axis=1)
            take = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
            best_scores = np.take_along_axis(merged_scores, take, axis=1)
            best_indices = np.take_along_axis(merged_indices, take, axis=1)
            order = np.argsort(-best_scores, axis=1, kind="stable")
            best_scores = np.take_along_axis(best_scores, order, axis=1)
            best_indices = np.take_along_axis(best_indices, order, axis=1)
        return best_scores, best_indices


def remove_self_neighbors(
    scores: np.ndarray,
    neighbors: np.ndarray,
    query_global_rows: np.ndarray,
    k_out: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove actual row IDs equal to the query; do not assume self is rank 0."""
    keep = neighbors != query_global_rows[:, None]
    packed_rank = np.cumsum(keep, axis=1) - 1
    selected = keep & (packed_rank < k_out)
    rows, cols = np.nonzero(selected)
    out_scores = np.full((scores.shape[0], k_out), -np.inf, dtype=np.float32)
    out_neighbors = np.full((scores.shape[0], k_out), -1, dtype=np.int64)
    destinations = packed_rank[rows, cols]
    out_scores[rows, destinations] = scores[rows, cols]
    out_neighbors[rows, destinations] = neighbors[rows, cols]
    if np.any(out_neighbors < 0):
        bad_rows = np.flatnonzero(np.any(out_neighbors < 0, axis=1))[:10].tolist()
        raise RuntimeError(f"Could not fill {k_out} non-self neighbors; example batch rows={bad_rows}")
    return out_scores, out_neighbors


def build_relation(
    *,
    index: Any,
    query: np.ndarray,
    query_role: str,
    output_prefix: Path,
    kmax: int,
    query_batch_size: int,
    score_dtype: np.dtype,
    remove_self: bool,
    allow_zero_norm: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    n_query = int(query.shape[0])
    n_core = int(index.index.ntotal) if isinstance(index, FaissExactIP) else int(index.core.shape[0])
    k_out = min(int(kmax), n_core - 1 if remove_self else n_core)
    if k_out <= 0:
        raise ValueError(f"Not enough core proteins for relation: n_core={n_core}, remove_self={remove_self}")
    k_search = min(n_core, k_out + 1 if remove_self else k_out)

    neighbor_path = output_prefix.with_name(output_prefix.name + "_neighbors.i32.npy")
    score_suffix = "f16" if score_dtype == np.float16 else "f32"
    score_path = output_prefix.with_name(output_prefix.name + f"_scores.{score_suffix}.npy")
    manifest_path = output_prefix.with_name(output_prefix.name + "_manifest.json")
    ensure_writable((neighbor_path, score_path, manifest_path), overwrite)
    neighbor_partial = neighbor_path.with_name(neighbor_path.name + ".partial")
    score_partial = score_path.with_name(score_path.name + ".partial")
    out_neighbors = np.lib.format.open_memmap(
        neighbor_partial, mode="w+", dtype=np.int32, shape=(n_query, k_out)
    )
    out_scores = np.lib.format.open_memmap(
        score_partial, mode="w+", dtype=score_dtype, shape=(n_query, k_out)
    )

    zero_norm_count = 0
    norm_min = math.inf
    norm_max = 0.0
    score_min = math.inf
    score_max = -math.inf
    started = time.time()
    for start in range(0, n_query, query_batch_size):
        end = min(start + query_batch_size, n_query)
        q, zero, q_norm_min, q_norm_max = normalized_float32_chunk(
            query,
            start,
            end,
            role=query_role,
            allow_zero_norm=allow_zero_norm,
        )
        scores, neighbors = index.search(q, k_search)
        scores = np.asarray(scores, dtype=np.float32)
        neighbors = np.asarray(neighbors, dtype=np.int64)
        if remove_self:
            scores, neighbors = remove_self_neighbors(
                scores,
                neighbors,
                np.arange(start, end, dtype=np.int64),
                k_out,
            )
        else:
            scores = scores[:, :k_out]
            neighbors = neighbors[:, :k_out]
        if np.any(neighbors < 0) or np.any(neighbors >= n_core):
            raise RuntimeError(f"Invalid neighbor index returned for {query_role}, rows={start}:{end}")
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"Non-finite cosine score for {query_role}, rows={start}:{end}")

        out_neighbors[start:end] = neighbors.astype(np.int32, copy=False)
        out_scores[start:end] = scores.astype(score_dtype, copy=False)
        zero_norm_count += zero
        norm_min = min(norm_min, q_norm_min)
        norm_max = max(norm_max, q_norm_max)
        score_min = min(score_min, float(scores.min()))
        score_max = max(score_max, float(scores.max()))
        if start == 0 or end == n_query or (start // query_batch_size + 1) % 20 == 0:
            print(f"[{query_role}->core] rows={end}/{n_query}", flush=True)

    out_neighbors.flush()
    out_scores.flush()
    del out_neighbors, out_scores
    os.replace(neighbor_partial, neighbor_path)
    os.replace(score_partial, score_path)

    manifest = {
        "schema_version": 1,
        "retrieval_direction": f"{query_role}->core",
        "graph_message_direction_recommendation": (
            "directed core->query" if query_role != "core" else "choose directed/union/mutual during graph compilation"
        ),
        "query_role": query_role,
        "query_count": n_query,
        "core_count": n_core,
        "requested_kmax": int(kmax),
        "resolved_k": k_out,
        "search_k": k_search,
        "self_edges_removed_by_row_id": bool(remove_self),
        "neighbor_index_space": "zero-based row index in core feature array",
        "query_index_space": f"implicit zero-based row index in {query_role} feature array",
        "similarity": "cosine (L2-normalized inner product)",
        "neighbors_file": neighbor_path.name,
        "scores_file": score_path.name,
        "score_dtype": np.dtype(score_dtype).name,
        "query_zero_norm_count": zero_norm_count,
        "query_raw_norm_min": norm_min,
        "query_raw_norm_max": norm_max,
        "cosine_score_min": score_min,
        "cosine_score_max": score_max,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.feature_dir = args.feature_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.kmax <= 0 or args.query_batch_size <= 0 or args.add_batch_size <= 0:
        raise ValueError("--kmax, --query-batch-size and --add-batch-size must be positive")
    if len(set(args.query_roles)) != len(args.query_roles):
        raise ValueError(f"Duplicate --query-roles: {args.query_roles}")

    manifest_path = args.feature_dir / "representation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Representation manifest not found: {manifest_path}")
    representation_manifest = read_json(manifest_path)
    core, core_record = load_role(args.feature_dir, representation_manifest, "core")
    if core.shape[0] > np.iinfo(np.int32).max:
        raise OverflowError("Core row count exceeds int32 neighbor index capacity")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "faiss":
        index = FaissExactIP(
            core,
            add_batch_size=args.add_batch_size,
            gpu_id=args.faiss_gpu_id,
            threads=args.faiss_threads,
            allow_zero_norm=args.allow_zero_norm,
        )
    else:
        index = NumpyExactIP(
            core,
            index_block_size=args.numpy_index_block_size,
            allow_zero_norm=args.allow_zero_norm,
        )

    score_dtype = np.float16 if args.score_dtype == "float16" else np.float32
    results: List[Dict[str, Any]] = []
    if args.relation in {"core-core", "all"}:
        result = build_relation(
            index=index,
            query=core,
            query_role="core",
            output_prefix=args.output_dir / "pp_core_core",
            kmax=args.kmax,
            query_batch_size=args.query_batch_size,
            score_dtype=score_dtype,
            remove_self=True,
            allow_zero_norm=args.allow_zero_norm,
            overwrite=args.overwrite,
        )
        results.append(result)

    if args.relation in {"weak-core", "all"}:
        for role in args.query_roles:
            if role == "core":
                raise ValueError("Do not include core in --query-roles; core-core is a separate relation")
            query, query_record = load_role(args.feature_dir, representation_manifest, role)
            if query.shape[1] != core.shape[1]:
                raise ValueError(
                    f"Feature dimension mismatch: core={core.shape[1]}, {role}={query.shape[1]}"
                )
            result = build_relation(
                index=index,
                query=query,
                query_role=role,
                output_prefix=args.output_dir / f"pp_{role}_core",
                kmax=args.kmax,
                query_batch_size=args.query_batch_size,
                score_dtype=score_dtype,
                remove_self=False,
                allow_zero_norm=args.allow_zero_norm,
                overwrite=args.overwrite,
            )
            results.append(result)

    run_manifest = {
        "schema_version": 1,
        "task": representation_manifest.get("task"),
        "representation_manifest": str(manifest_path),
        "representation_checkpoint_sha256": representation_manifest.get("checkpoint_sha256"),
        "backend": args.backend,
        "index_type": "IndexFlatIP" if args.backend == "faiss" else "numpy_exact_blocked_inner_product",
        "index_device": index.device,
        "core_count": int(core.shape[0]),
        "feature_dim": int(core.shape[1]),
        "core_feature_file": core_record["feature_file"],
        "core_zero_norm_count": index.zero_norm_count,
        "core_raw_norm_min": index.norm_min,
        "core_raw_norm_max": index.norm_max,
        "relations": results,
    }
    run_manifest_path = args.output_dir / "pp_relations_manifest.json"
    ensure_writable((run_manifest_path,), args.overwrite)
    atomic_write_text(run_manifest_path, json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(run_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()