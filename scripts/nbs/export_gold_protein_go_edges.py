#!/usr/bin/env python3
"""Export task-specific gold Protein->GO COO edges for LATENCE NBS.

The source metadata contract matches MSABinaryDataset:

    metadata[mode][normalized_task][name_key]
    metadata[mode][normalized_task][label_key]

For BP defaults this is:

    data["train"]["biological_process"]["proteins"]
    data["train"]["biological_process"]["prop_annotations"]

The output edge index is a standard NumPy .npy array:

    shape [2, E]
    dtype int32
    row 0: global protein_idx from protein_registry.csv
    row 1: task classifier go_idx from go_registry.tsv

The script intentionally rejects candidate/pseudo indices and any source whose
protein-ID set disagrees with the selected registry role. Metadata order may differ
from registry order: alignment_policy=auto first checks exact order, then performs
a strict by-ID alignment only when the two unique protein-ID sets are identical.
The written COO is always sorted by global protein_idx (protein-major). It does not
infer GO indices from GO IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

TASK_KEYS = {
    "bp": "biological_process",
    "biological_process": "biological_process",
    "mf": "molecular_function",
    "molecular_function": "molecular_function",
    "cc": "cellular_component",
    "cellular_component": "cellular_component",
}

EXPORTER_VERSION = "1.1.0-strict-auto-id-alignment"


def normalize_protein_id(value: Any) -> str:
    """Normalize a protein identifier without changing its biological identity."""
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    protein_id = str(value).strip()
    if not protein_id:
        raise ValueError("empty protein identifier")
    return protein_id


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_metadata(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            return pickle.load(handle)
    if suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - only used for .pt input
            raise RuntimeError("PyTorch is required to load .pt metadata") from exc
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    raise ValueError(f"unsupported metadata format: {path}")


def normalize_labels(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.int64)
    if isinstance(value, (int, np.integer)):
        return np.asarray([int(value)], dtype=np.int64)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    array = np.asarray(list(value), dtype=np.int64).reshape(-1)
    if array.size == 0:
        return array
    return np.unique(array)


def load_registry(
    path: Path,
    *,
    role: str,
    dataset_mode: str,
) -> tuple[list[str], np.ndarray, int]:
    records: list[tuple[int, int, str]] = []
    total_rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "protein_idx",
            "protein_id",
            "role",
            "role_row_idx",
            "dataset_mode",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"protein registry must contain {sorted(required)}")
        for expected_global, row in enumerate(reader):
            total_rows += 1
            global_idx = int(row["protein_idx"])
            if global_idx != expected_global:
                raise ValueError(
                    "protein_registry.csv must be row-aligned with contiguous protein_idx"
                )
            if row["role"] == role and row["dataset_mode"] == dataset_mode:
                records.append(
                    (int(row["role_row_idx"]), global_idx, normalize_protein_id(row["protein_id"]))
                )
    if not records:
        raise ValueError(
            f"no registry rows found for role={role!r}, dataset_mode={dataset_mode!r}"
        )
    records.sort(key=lambda item: item[0])
    role_rows = np.asarray([item[0] for item in records], dtype=np.int64)
    if not np.array_equal(role_rows, np.arange(role_rows.size, dtype=np.int64)):
        raise ValueError("selected role_row_idx is not contiguous from zero")
    global_idx = np.asarray([item[1] for item in records], dtype=np.int64)
    protein_ids = [item[2] for item in records]
    if len(set(protein_ids)) != len(protein_ids):
        raise ValueError("duplicate protein_id values in selected registry role")
    return protein_ids, global_idx, total_rows


def load_num_go(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "go_idx" not in (reader.fieldnames or []):
            raise ValueError("go_registry.tsv must contain go_idx")
        for expected, row in enumerate(reader):
            current = int(row["go_idx"])
            if current != expected:
                raise ValueError("go_registry.tsv go_idx must be contiguous and row-aligned")
            count += 1
    if count <= 0:
        raise ValueError("empty GO registry")
    return count


def resolve_alignment(
    metadata_proteins: list[str],
    registry_proteins: list[str],
    registry_global: np.ndarray,
    *,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve metadata rows to global protein indices with strict auditing.

    ``auto`` is safe rather than permissive: it accepts a reordered metadata table
    only when both sides contain exactly the same unique protein identifiers.
    """
    if len(metadata_proteins) != len(set(metadata_proteins)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for protein in metadata_proteins:
            if protein in seen and len(duplicates) < 10:
                duplicates.append(protein)
            seen.add(protein)
        raise ValueError(
            "metadata contains duplicate protein identifiers: "
            f"examples={duplicates}"
        )

    if policy not in {"exact", "by_id", "auto"}:
        raise ValueError("alignment policy must be exact, by_id, or auto")

    ordered_equal = metadata_proteins == registry_proteins
    first_mismatch: dict[str, Any] | None = None
    if not ordered_equal:
        mismatch = next(
            (
                i
                for i, (left, right) in enumerate(
                    zip(metadata_proteins, registry_proteins)
                )
                if left != right
            ),
            min(len(metadata_proteins), len(registry_proteins)),
        )
        first_mismatch = {
            "row": mismatch,
            "metadata": (
                metadata_proteins[mismatch]
                if mismatch < len(metadata_proteins)
                else None
            ),
            "registry": (
                registry_proteins[mismatch]
                if mismatch < len(registry_proteins)
                else None
            ),
        }

    if policy == "exact":
        if not ordered_equal:
            assert first_mismatch is not None
            raise ValueError(
                "metadata train proteins are not exactly aligned with the selected "
                "registry role: first mismatch "
                f"row={first_mismatch['row']}, "
                f"metadata={first_mismatch['metadata']!r}, "
                f"registry={first_mismatch['registry']!r}; "
                "use --alignment-policy auto (recommended) or by_id after auditing"
            )
        return registry_global.copy(), {
            "requested_policy": policy,
            "resolved_policy": "exact",
            "ordered_equal": True,
            "reordered_metadata_rows": 0,
            "first_mismatch": None,
        }

    if policy == "auto" and ordered_equal:
        return registry_global.copy(), {
            "requested_policy": policy,
            "resolved_policy": "exact",
            "ordered_equal": True,
            "reordered_metadata_rows": 0,
            "first_mismatch": None,
        }

    if len(metadata_proteins) != len(registry_proteins):
        raise ValueError(
            "metadata/registry protein counts disagree under strict by-ID alignment: "
            f"metadata={len(metadata_proteins)}, registry={len(registry_proteins)}"
        )

    metadata_set = set(metadata_proteins)
    registry_set = set(registry_proteins)
    missing = sorted(metadata_set - registry_set)
    extra = sorted(registry_set - metadata_set)
    if missing or extra:
        raise ValueError(
            "metadata/registry protein sets disagree under strict by-ID alignment: "
            f"missing_in_registry_count={len(missing)}, "
            f"missing_examples={missing[:10]}, "
            f"extra_in_registry_count={len(extra)}, "
            f"extra_examples={extra[:10]}"
        )

    lookup = {
        protein: int(index)
        for protein, index in zip(registry_proteins, registry_global)
    }
    aligned = np.asarray(
        [lookup[protein] for protein in metadata_proteins], dtype=np.int64
    )
    reordered_rows = int(
        np.count_nonzero(aligned != registry_global)
    ) if aligned.size == registry_global.size else int(aligned.size)
    return aligned, {
        "requested_policy": policy,
        "resolved_policy": "by_id",
        "ordered_equal": False,
        "reordered_metadata_rows": reordered_rows,
        "first_mismatch": first_mismatch,
    }


def atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def export_gold_edges(
    metadata_file: Path,
    protein_registry: Path,
    go_registry: Path,
    output_dir: Path,
    *,
    task: str,
    mode: str,
    role: str,
    name_key: str,
    label_key: str,
    alignment_policy: str,
    overwrite: bool,
) -> dict[str, Any]:
    started = time.time()
    task_key = TASK_KEYS.get(task)
    if task_key is None:
        raise ValueError(f"unknown task: {task!r}")

    for path in (metadata_file, protein_registry, go_registry):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = load_metadata(metadata_file)
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata root must be a mapping, got {type(metadata)}")
    if mode not in metadata:
        raise KeyError(f"metadata has no mode={mode!r}")
    mode_value = metadata[mode]
    if not isinstance(mode_value, Mapping) or task_key not in mode_value:
        raise KeyError(f"metadata[{mode!r}] has no task={task_key!r}")
    sub = mode_value[task_key]
    if not isinstance(sub, Mapping):
        raise TypeError("metadata task entry must be a mapping")
    if name_key not in sub or label_key not in sub:
        raise KeyError(
            f"metadata task entry must contain {name_key!r} and {label_key!r}"
        )

    proteins = [normalize_protein_id(value) for value in list(sub[name_key])]
    raw_labels = list(sub[label_key])
    if len(proteins) != len(raw_labels):
        raise ValueError(
            f"metadata proteins/labels length mismatch: {len(proteins)} vs {len(raw_labels)}"
        )

    registry_proteins, registry_global, total_proteins = load_registry(
        protein_registry,
        role=role,
        dataset_mode=mode,
    )
    metadata_global, alignment_report = resolve_alignment(
        proteins,
        registry_proteins,
        registry_global,
        policy=alignment_policy,
    )
    num_go = load_num_go(go_registry)

    normalized_labels: list[np.ndarray] = []
    num_edges = 0
    zero_degree = 0
    duplicate_labels_removed = 0
    max_degree = 0
    counts = np.zeros(num_go, dtype=np.int64)
    for row, value in enumerate(raw_labels):
        original = np.asarray(list(value), dtype=np.int64).reshape(-1) if value is not None and not isinstance(value, (int, np.integer)) else normalize_labels(value)
        labels = normalize_labels(value)
        duplicate_labels_removed += max(0, int(original.size) - int(labels.size))
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= num_go):
            bad = labels[(labels < 0) | (labels >= num_go)]
            raise IndexError(
                f"gold GO index outside task classifier space at metadata row {row}: "
                f"examples={bad[:10].tolist()}, num_go={num_go}"
            )
        normalized_labels.append(labels)
        degree = int(labels.size)
        num_edges += degree
        max_degree = max(max_degree, degree)
        if degree == 0:
            zero_degree += 1
        else:
            counts[labels] += 1

    # Metadata and registry can contain the same proteins in different orders.
    # Reorder rows by global protein_idx before writing so the output is truly
    # protein-major and can safely feed Protein->GO CSR construction.
    write_order = np.argsort(metadata_global, kind="stable")
    sorted_global = metadata_global[write_order]
    if sorted_global.size and np.any(sorted_global[1:] <= sorted_global[:-1]):
        raise RuntimeError(
            "aligned global protein indices are not strictly increasing after sort"
        )
    sorted_labels = [normalized_labels[int(row)] for row in write_order.tolist()]

    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "gold_protein_go_edge_index.i32.npy"
    manifest_path = output_dir / "gold_annotations_manifest.json"
    if not overwrite:
        existing = [str(path) for path in (edge_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"gold outputs already exist: {existing}")

    temporary_edge = edge_path.with_name(edge_path.stem + ".tmp.npy")
    if temporary_edge.exists():
        temporary_edge.unlink()
    edge = np.lib.format.open_memmap(
        temporary_edge,
        mode="w+",
        dtype=np.int32,
        shape=(2, num_edges),
    )
    cursor = 0
    for protein_idx, labels in zip(sorted_global.tolist(), sorted_labels):
        end = cursor + int(labels.size)
        if end > cursor:
            edge[0, cursor:end] = np.int32(protein_idx)
            edge[1, cursor:end] = labels.astype(np.int32, copy=False)
        cursor = end
    if cursor != num_edges:
        raise RuntimeError("gold edge write cursor mismatch")
    edge.flush()
    del edge
    os.replace(temporary_edge, edge_path)

    verification = np.load(edge_path, mmap_mode="r")
    if verification.shape != (2, num_edges) or verification.dtype != np.int32:
        raise RuntimeError("written gold edge array failed shape/dtype verification")
    if num_edges:
        if int(verification[0].min()) < 0 or int(verification[0].max()) >= total_proteins:
            raise RuntimeError("written protein indices are outside global registry")
        if int(verification[1].min()) < 0 or int(verification[1].max()) >= num_go:
            raise RuntimeError("written GO indices are outside classifier space")
        if verification.shape[1] > 1 and np.any(
            verification[0, 1:] < verification[0, :-1]
        ):
            raise RuntimeError("written gold COO is not protein-major")

    nonzero_go = int(np.count_nonzero(counts))
    result: dict[str, Any] = {
        "schema_version": 2,
        "exporter": {
            "file": "scripts/nbs/export_gold_protein_go_edges.py",
            "version": EXPORTER_VERSION,
        },
        "task": task,
        "metadata_task": task_key,
        "mode": mode,
        "role": role,
        "source": {
            "metadata_file": str(metadata_file.resolve()),
            "metadata_sha256": sha256_file(metadata_file),
            "name_key": name_key,
            "label_key": label_key,
        },
        "protein_registry": {
            "path": str(protein_registry.resolve()),
            "sha256": sha256_file(protein_registry),
            "num_global_proteins": total_proteins,
            "num_selected_proteins": len(proteins),
            "alignment": alignment_report,
            "global_protein_idx_min": int(sorted_global.min()) if sorted_global.size else None,
            "global_protein_idx_max": int(sorted_global.max()) if sorted_global.size else None,
        },
        "go_registry": {
            "path": str(go_registry.resolve()),
            "sha256": sha256_file(go_registry),
            "num_go": num_go,
            "index_contract": "task classifier go_idx; not BoxSquaredEL ontology row",
        },
        "edge_index": {
            "path": str(edge_path.resolve()),
            "sha256": sha256_file(edge_path),
            "shape": [2, num_edges],
            "dtype": "int32",
            "row_0": "global protein_idx",
            "row_1": "task classifier go_idx",
            "message_direction": "protein -> GO",
            "source_layout": "global_protein_idx_major",
        },
        "statistics": {
            "num_edges": num_edges,
            "zero_annotation_proteins": zero_degree,
            "mean_annotations_per_protein": float(num_edges / max(1, len(proteins))),
            "max_annotations_per_protein": max_degree,
            "num_go_with_gold_annotations": nonzero_go,
            "max_go_degree": int(counts.max(initial=0)),
            "duplicate_labels_removed": duplicate_labels_removed,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_json_dump(result, manifest_path)
    result["manifest"] = str(manifest_path.resolve())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export strict gold Protein->GO COO edges for LATENCE NBS"
    )
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--protein-registry", type=Path, required=True)
    parser.add_argument("--go-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASK_KEYS))
    parser.add_argument("--mode", default="train")
    parser.add_argument("--role", default="core")
    parser.add_argument("--name-key", default="proteins")
    parser.add_argument("--label-key", default="prop_annotations")
    parser.add_argument(
        "--alignment-policy",
        choices=["exact", "by_id", "auto"],
        default="auto",
        help=(
            "auto first checks exact order, then uses strict by-ID alignment only "
            "when metadata and registry contain identical unique protein-ID sets"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_gold_edges(
        args.metadata_file,
        args.protein_registry,
        args.go_registry,
        args.output_dir,
        task=args.task,
        mode=args.mode,
        role=args.role,
        name_key=args.name_key,
        label_key=args.label_key,
        alignment_policy=args.alignment_policy,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
