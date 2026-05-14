#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Align ESM2GO numpy prediction outputs to the user's namespace_terms.pkl order.

Input produced by ESM2GO_pred_large_multigpu.py:
    working_dir/names.npy
    working_dir/ESM2GO_numpy/BPO.terms.npy
    working_dir/ESM2GO_numpy/BPO.scores.float16.npy
    working_dir/ESM2GO_numpy/BPO.top500.scores.float16.npy      # optional
    working_dir/ESM2GO_numpy/BPO.top500.indices.int32.npy       # optional

namespace_terms.pkl:
    dict with keys:
        cellular_component
        molecular_function
        biological_process
    each value is an ordered list of GO IDs.

Output:
    output_dir/<namespace>.terms.npy
    output_dir/<namespace>.scores.<dtype>.npy
    output_dir/<namespace>.alignment.json
    output_dir/names.npy
    output_dir/term_alignment_summary.tsv

For dense source arrays, all model scores are preserved and reordered.
For top-k source arrays, only saved top-k scores can be reconstructed; all other
entries are filled with --fill_value.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import OrderedDict
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
        # Use the canonical order if possible, then append any extra keys from the pickle.
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


def parse_float_fill_value(text: str) -> float:
    lower = str(text).strip().lower()
    if lower in {"nan", "+nan", "-nan"}:
        return float("nan")
    return float(text)


def find_topk_files(numpy_dir: Path, aspect: str, top_k: Optional[int]) -> Tuple[Optional[Path], Optional[Path], Optional[int]]:
    """Find top-k score/index files for an aspect.

    If top_k is given, uses exactly aspect.top{top_k}.*. Otherwise, scans files.
    """
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


def make_alignment_index(
    source_terms: Sequence[str],
    target_terms: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return source column indices and target column indices in target order."""
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


def align_dense_source(
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
            raise ValueError(
                f"row_indices contains values outside source row range 0..{source_n_rows - 1}"
            )
        n_rows = int(row_indices.shape[0])
    n_target = len(target_terms)
    source_indices, target_indices, missing_terms = make_alignment_index(source_terms, target_terms)

    out = open_output_memmap(output_scores_path, output_dtype, (n_rows, n_target))
    matched = int(len(source_indices))

    for chunk_no, start in enumerate(
        tqdm(range(0, n_rows, row_chunk_size), desc=f"align dense {source_scores_path.name}", ascii=" >=")
    ):
        end = min(start + row_chunk_size, n_rows)
        block = np.empty((end - start, n_target), dtype=output_dtype)
        block[...] = fill_value
        if matched:
            # Fancy indexing keeps target columns in namespace_terms order.
            if row_indices is None:
                values = src[start:end, source_indices]
            else:
                values = src[row_indices[start:end]][:, source_indices]
            if values.dtype != output_dtype:
                values = values.astype(output_dtype, copy=False)
            block[:, target_indices] = values
        out[start:end] = block
        if flush_every > 0 and (chunk_no + 1) % flush_every == 0:
            out.flush()

    out.flush()
    del out
    del src

    return {
        "source_mode": "dense",
        "source_scores_npy": str(source_scores_path),
        "source_rows": int(source_n_rows),
        "output_rows": int(n_rows),
        "source_columns": int(len(source_terms)),
        "target_columns": int(n_target),
        "matched_columns": matched,
        "missing_columns": int(len(missing_terms)),
        "missing_terms_preview": missing_terms[:50],
    }


def align_topk_source(
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
        raise ValueError(
            f"Top-k arrays must be 2-D, got {top_scores_path}: {top_scores.shape}, "
            f"{top_indices_path}: {top_indices.shape}"
        )
    if top_scores.shape != top_indices.shape:
        raise ValueError(
            f"Top-k score/index shape mismatch: {top_scores.shape} vs {top_indices.shape}"
        )

    source_n_rows, top_k = map(int, top_scores.shape)
    if row_indices is None:
        n_rows = source_n_rows
    else:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be 1-D")
        if row_indices.size and (int(row_indices.min()) < 0 or int(row_indices.max()) >= source_n_rows):
            raise ValueError(
                f"row_indices contains values outside source row range 0..{source_n_rows - 1}"
            )
        n_rows = int(row_indices.shape[0])
    n_target = len(target_terms)

    source_pos = {term: i for i, term in enumerate(source_terms)}
    src_index_to_target_index = np.full(len(source_terms), -1, dtype=np.int64)
    missing_terms: List[str] = []
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
            if values.dtype != output_dtype:
                values = values.astype(output_dtype, copy=False)
            block[local_i, target_idx[keep]] = values

        out[start:end] = block
        if flush_every > 0 and (chunk_no + 1) % flush_every == 0:
            out.flush()

    out.flush()
    del out
    del top_scores
    del top_indices

    return {
        "source_mode": "topk",
        "top_scores_npy": str(top_scores_path),
        "top_indices_npy": str(top_indices_path),
        "top_k": int(top_k),
        "source_rows": int(source_n_rows),
        "output_rows": int(n_rows),
        "source_columns": int(len(source_terms)),
        "target_columns": int(n_target),
        "matched_columns": int(matched),
        "missing_columns": int(len(missing_terms)),
        "missing_terms_preview": missing_terms[:50],
        "important_note": (
            "This output was reconstructed from compact top-k predictions only. "
            "Scores not present in the top-k arrays were filled with fill_value."
        ),
    }


def write_term_mapping_rows(
    summary_tsv: Path,
    rows: List[Tuple[str, str, int, str, int, str]],
    append: bool,
) -> None:
    mode = "a" if append and summary_tsv.exists() else "w"
    with open(summary_tsv, mode, encoding="utf-8", newline="") as handle:
        if mode == "w":
            handle.write("namespace\taspect\ttarget_index\tgo_id\tsource_index\tstatus\n")
        for namespace, aspect, target_index, go_id, source_index, status in rows:
            handle.write(f"{namespace}\t{aspect}\t{target_index}\t{go_id}\t{source_index}\t{status}\n")


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
    row_indices: Optional[np.ndarray] = None,
) -> Dict:
    aspect = NAMESPACE_TO_ASPECT[namespace]
    terms_path = numpy_dir / f"{aspect}.terms.npy"
    if not terms_path.exists():
        raise FileNotFoundError(f"Missing model term file: {terms_path}")

    source_terms = normalize_terms(_load_npy(terms_path, allow_pickle=False))
    check_unique(source_terms, f"{terms_path}")
    check_unique(target_terms, f"namespace_terms[{namespace!r}]")

    source_pos = {term: i for i, term in enumerate(source_terms)}
    mapping_rows = []
    missing_terms = []
    for target_j, term in enumerate(target_terms):
        source_j = source_pos.get(term)
        if source_j is None:
            missing_terms.append(term)
            mapping_rows.append((namespace, aspect, target_j, term, -1, "missing_in_model"))
        else:
            mapping_rows.append((namespace, aspect, target_j, term, source_j, "matched"))

    if strict and missing_terms:
        examples = ", ".join(missing_terms[:20])
        raise ValueError(
            f"{namespace} has {len(missing_terms)} target GO terms absent from {terms_path}. "
            f"Examples: {examples}"
        )

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

    if chosen_mode == "dense":
        if not dense_path.exists():
            raise FileNotFoundError(f"Dense source requested but not found: {dense_path}")
        source_meta = align_dense_source(
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
            raise FileNotFoundError(
                f"Top-k source requested but not found for {aspect} in {numpy_dir}. "
                "Use --top_k if the file name is not auto-detected."
            )
        source_meta = align_topk_source(
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
        source_meta["inferred_top_k_from_filename"] = inferred_top_k
    else:
        raise ValueError(f"Unsupported source mode: {chosen_mode}")

    meta = {
        "namespace": namespace,
        "aspect": aspect,
        "source_terms_npy": str(terms_path),
        "output_terms_npy": str(output_terms_path),
        "output_scores_npy": str(output_scores_path),
        "output_dtype": np.dtype(output_dtype).name,
        "fill_value_for_terms_absent_or_unavailable": fill_value if not np.isnan(fill_value) else "nan",
        "row_order": "Rows are identical to output_dir/names.npy. If --target_names_npy was used, rows follow that target label order.",
        "column_order": "Columns are exactly namespace_terms.pkl[namespace] order.",
        **source_meta,
    }
    save_json(output_dir / f"{namespace}.alignment.json", meta)

    return {
        "meta": meta,
        "mapping_rows": mapping_rows,
    }


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
        description="Align ESM2GO numpy prediction matrices to namespace_terms.pkl GO-term order."
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
        help="Output directory. Default: <working_dir>/ESM2GO_numpy_aligned",
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
        help=(
            "Optional target protein ID order, e.g. your label matrix row-order file. "
            "If provided, output rows are reordered to this order."
        ),
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
        help="Value for target GO terms absent from model output, or absent from top-k. Use 'nan' if desired.",
    )
    parser.add_argument(
        "--row_chunk_size",
        type=int,
        default=2048,
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
        help="Fail if any namespace_terms GO ID is absent from the model GO-term list.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also write all_namespaces.scores.<dtype>.npy by concatenating aligned namespace outputs.",
    )
    args = parser.parse_args()

    working_dir = args.working_dir.resolve()
    numpy_dir = (args.numpy_dir or (working_dir / "ESM2GO_numpy")).resolve()
    output_dir = (args.output_dir or (working_dir / "ESM2GO_numpy_aligned")).resolve()
    names_npy = (args.names_npy or (working_dir / "names.npy")).resolve()

    if args.row_chunk_size <= 0:
        raise ValueError("--row_chunk_size must be positive")
    if not names_npy.exists():
        raise FileNotFoundError(f"Missing names.npy: {names_npy}")
    if not numpy_dir.exists():
        raise FileNotFoundError(f"Missing numpy_dir: {numpy_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    namespace_terms = load_namespace_terms(args.namespace_terms.resolve())
    namespaces = resolve_namespaces(args.namespaces, namespace_terms)
    output_dtype = np.dtype(args.output_dtype)
    fill_value = parse_float_fill_value(args.fill_value)

    target_names_npy = args.target_names_npy.resolve() if args.target_names_npy is not None else None
    row_indices, effective_names_npy, n_rows_from_names, row_order_note = make_row_indices(names_npy, target_names_npy)

    # Preserve the effective row-order file in the aligned output directory.
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
    }

    for namespace in namespaces:
        print(f"Aligning {namespace} ({NAMESPACE_TO_ASPECT[namespace]})")
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
        )
        meta = result["meta"]
        if int(meta.get("target_columns", 0)) <= 0:
            raise ValueError(f"No target columns for namespace {namespace}")
        # Check row count against names.npy.
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
    print(f"Done. Aligned outputs saved to: {output_dir}")
    print(f"Term mapping summary: {summary_tsv}")


if __name__ == "__main__":
    main()
