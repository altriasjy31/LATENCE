from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

_GO_PATTERN = re.compile(r"GO[_:](\d{7})", re.IGNORECASE)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def normalize_go_identifier(value: str) -> Optional[str]:
    """Normalize GO compact IDs and OWL IRIs to ``GO:NNNNNNN``."""
    match = _GO_PATTERN.search(str(value))
    if match is None:
        return None
    return f"GO:{match.group(1)}"


def resolve_project_artifact(
    value: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str],
    project_root: Optional[str | os.PathLike[str]] = None,
    must_exist: bool = True,
) -> Path:
    """Resolve paths emitted by project-root-relative BoxSquaredEL manifests.

    The BoxSquaredEL manifest stores paths such as ``outputs/go_emb/...``.  Such
    paths are relative to the LATENCE project root rather than to the manifest
    directory itself.  Several deterministic candidates are checked to remain
    robust when the project is mounted elsewhere.
    """
    raw = Path(value)
    manifest = Path(manifest_path).resolve()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if project_root is not None:
            candidates.append(Path(project_root).resolve() / raw)
        candidates.append(Path.cwd().resolve() / raw)
        candidates.append(manifest.parent / raw)
        candidates.append(manifest.parent / raw.name)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    for candidate in unique:
        if candidate.exists():
            return candidate
    if must_exist:
        attempted = "\n  - ".join(str(item) for item in unique)
        raise FileNotFoundError(
            f"could not resolve BoxSquaredEL artifact {value!r}; attempted:\n  - {attempted}"
        )
    if unique:
        return unique[0]
    return raw.resolve()


@dataclass(frozen=True)
class BoxSquaredELTrainingContract:
    manifest_path: Path
    parser_report_path: Optional[Path]
    embedding_dim: int
    class_vector_width: int
    relation_vector_width: int
    num_ontology_classes: int
    num_relations: int
    selected_epoch: int
    artifact_selection: str
    checkpoint_path: Path
    npy_directory: Optional[Path]
    manifest: Mapping[str, Any]
    parser_report: Mapping[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "model": "BoxSquaredEL",
            "embedding_dim": self.embedding_dim,
            "class_vector_width": self.class_vector_width,
            "relation_vector_width": self.relation_vector_width,
            "num_ontology_classes": self.num_ontology_classes,
            "num_relations": self.num_relations,
            "artifact_selection": self.artifact_selection,
            "selected_epoch": self.selected_epoch,
            "manifest_path": str(self.manifest_path),
            "parser_report_path": (
                None if self.parser_report_path is None else str(self.parser_report_path)
            ),
            "checkpoint_path": str(self.checkpoint_path),
            "npy_directory": None if self.npy_directory is None else str(self.npy_directory),
        }


def load_boxsqel_training_contract(
    manifest_path: str | os.PathLike[str],
    *,
    parser_report_path: Optional[str | os.PathLike[str]] = None,
    project_root: Optional[str | os.PathLike[str]] = None,
    artifact_selection: str = "best",
    require_artifacts: bool = True,
) -> BoxSquaredELTrainingContract:
    manifest_file = Path(manifest_path).resolve()
    manifest = _read_json(manifest_file)
    if manifest.get("model") != "BoxSquaredEL":
        raise ValueError(f"unsupported GO geometry model: {manifest.get('model')!r}")

    architecture = dict(manifest.get("architecture", {}))
    config = dict(manifest.get("config", {}))
    parser = dict(manifest.get("parser", {}))
    training = dict(manifest.get("training", {}))
    artifacts = dict(manifest.get("artifacts", {}))

    embedding_dim = int(config.get("embedding_size", 0))
    class_width = int(architecture.get("class_vector_width", 0))
    relation_width = int(architecture.get("relation_vector_width", 0))
    if embedding_dim <= 0:
        raise ValueError("BoxSquaredEL embedding_size must be positive")
    if class_width != 2 * embedding_dim:
        raise ValueError(
            f"class_vector_width={class_width} does not equal 2*embedding_size"
        )
    if relation_width != 4 * embedding_dim:
        raise ValueError(
            f"relation_vector_width={relation_width} does not equal 4*embedding_size"
        )
    if bool(training.get("nan_detected", False)):
        raise ValueError("BoxSquaredEL manifest reports NaN during training")

    parser_file: Optional[Path] = None
    parser_payload: Mapping[str, Any] = parser
    if parser_report_path is not None:
        parser_file = Path(parser_report_path).resolve()
        parser_payload = _read_json(parser_file)
        embedded_report = parser.get("report", {})
        external_report = parser_payload.get("report", {})
        for key in ("parsed_lines", "expanded_axioms", "unparsed_lines"):
            if key in embedded_report and key in external_report:
                if int(embedded_report[key]) != int(external_report[key]):
                    raise ValueError(
                        f"parser report mismatch for {key}: "
                        f"manifest={embedded_report[key]}, external={external_report[key]}"
                    )
    report = dict(parser_payload.get("report", {}))
    if bool(config.get("strict_parser", False)) and int(report.get("unparsed_lines", 0)) != 0:
        raise ValueError("strict BoxSquaredEL parser report contains unparsed lines")

    selection = str(artifact_selection).lower()
    if selection not in {"selected", "best", "final"}:
        raise ValueError("artifact_selection must be selected, best, or final")
    # The unsuffixed selected artifacts are aliases of the geometry-selected
    # best epoch in the source training script.  Checkpoints retain class IDs,
    # which are required for projection into the classifier GO index space.
    checkpoint_key = "final_checkpoint" if selection == "final" else "best_checkpoint"
    checkpoint_value = artifacts.get(checkpoint_key)
    if not checkpoint_value:
        raise ValueError(f"BoxSquaredEL manifest lacks {checkpoint_key}")
    checkpoint_path = resolve_project_artifact(
        checkpoint_value,
        manifest_path=manifest_file,
        project_root=project_root,
        must_exist=require_artifacts,
    )
    npy_value = artifacts.get("npy_directory")
    npy_directory = None
    if npy_value:
        npy_directory = resolve_project_artifact(
            npy_value,
            manifest_path=manifest_file,
            project_root=project_root,
            must_exist=False,
        )

    selected_epoch = int(
        training.get("last_epoch", 0)
        if selection == "final"
        else training.get("best_epoch", training.get("last_epoch", 0))
    )
    return BoxSquaredELTrainingContract(
        manifest_path=manifest_file,
        parser_report_path=parser_file,
        embedding_dim=embedding_dim,
        class_vector_width=class_width,
        relation_vector_width=relation_width,
        num_ontology_classes=int(parser_payload.get("classes", parser.get("classes", 0))),
        num_relations=int(parser_payload.get("relations", parser.get("relations", 0))),
        selected_epoch=selected_epoch,
        artifact_selection=selection,
        checkpoint_path=checkpoint_path,
        npy_directory=npy_directory,
        manifest=manifest,
        parser_report=parser_payload,
    )


def _class_embedding_from_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict[str, int], np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("BoxSquaredEL checkpoint lacks the class-name mapping")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("BoxSquaredEL checkpoint lacks model_state_dict")
    matches = [key for key in state if str(key).endswith("class_embeds.weight")]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one class_embeds.weight tensor, got " + repr(matches)
        )
    tensor = state[matches[0]]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError("class_embeds.weight must be a rank-2 tensor")
    class_map = {str(key): int(value) for key, value in classes.items()}
    if len(class_map) != tensor.shape[0]:
        raise ValueError(
            f"class mapping size {len(class_map)} != class embedding rows {tensor.shape[0]}"
        )
    return class_map, tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _go_mapping(class_map: Mapping[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_name, row in class_map.items():
        go_id = normalize_go_identifier(raw_name)
        if go_id is None:
            continue
        if go_id in result and result[go_id] != int(row):
            raise ValueError(
                f"multiple BoxSquaredEL rows normalize to {go_id}: "
                f"{result[go_id]} and {row}"
            )
        result[go_id] = int(row)
    return result


def _box_statistics(offset: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    positive = np.abs(offset).clip(min=eps)
    log_offset = np.log(positive)
    stats = np.zeros((offset.shape[0], 6), dtype=np.float32)
    stats[:, 0] = log_offset.mean(axis=1)
    stats[:, 1] = log_offset.std(axis=1)
    stats[:, 2] = np.log(2.0 * positive).mean(axis=1)
    stats[:, 3] = positive.max(axis=1)
    # Columns 4 and 5 are reserved for normalized GO depth and descendant
    # counts.  They remain zero here and may be enriched by the local graph
    # data adapter using gg_relations.
    return stats


def align_boxsqel_to_go_registry(
    manifest_path: str | os.PathLike[str],
    go_registry_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    parser_report_path: Optional[str | os.PathLike[str]] = None,
    project_root: Optional[str | os.PathLike[str]] = None,
    artifact_selection: str = "best",
    strict: bool = True,
    overwrite: bool = False,
) -> Path:
    """Project BoxSquaredEL ontology classes into the classifier GO index space.

    The ontology embedding contains all parsed classes (44,919 in the supplied
    512-dimensional run), whereas the BP classifier contains 21,312 columns.
    This function uses ``go_registry.tsv`` as the immutable target index and
    replicates geometry for canonical/alt-ID duplicate classifier columns.
    """
    contract = load_boxsqel_training_contract(
        manifest_path,
        parser_report_path=parser_report_path,
        project_root=project_root,
        artifact_selection=artifact_selection,
        require_artifacts=True,
    )
    registry_path = Path(go_registry_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    alignment_manifest = out / "go_box_alignment_manifest.json"
    if alignment_manifest.exists() and not overwrite:
        raise FileExistsError(
            f"alignment output already exists: {alignment_manifest}; use overwrite=True"
        )

    class_map, raw = _class_embedding_from_checkpoint(contract.checkpoint_path)
    if raw.shape[1] != contract.class_vector_width:
        raise ValueError(
            f"checkpoint class width {raw.shape[1]} != manifest {contract.class_vector_width}"
        )
    if raw.shape[0] != contract.num_ontology_classes:
        raise ValueError(
            f"checkpoint class rows {raw.shape[0]} != parser classes "
            f"{contract.num_ontology_classes}"
        )
    source_by_go = _go_mapping(class_map)

    registry_rows: list[dict[str, str]] = []
    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"go_idx", "input_go_id", "go_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"GO registry must contain {sorted(required)}, got {reader.fieldnames}"
            )
        registry_rows = [dict(row) for row in reader]
    for expected, row in enumerate(registry_rows):
        if int(row["go_idx"]) != expected:
            raise ValueError("go_registry.tsv go_idx must be contiguous and row-aligned")

    source_rows = np.full(len(registry_rows), -1, dtype=np.int32)
    missing: list[dict[str, Any]] = []
    for go_idx, row in enumerate(registry_rows):
        candidates = []
        for field in ("go_id", "input_go_id"):
            normalized = normalize_go_identifier(row[field])
            if normalized is not None and normalized not in candidates:
                candidates.append(normalized)
        match = next((source_by_go[item] for item in candidates if item in source_by_go), None)
        if match is None:
            missing.append({"go_idx": go_idx, "candidates": candidates})
        else:
            source_rows[go_idx] = int(match)
    if missing and strict:
        raise ValueError(
            f"{len(missing)} classifier GO terms lack BoxSquaredEL geometry; "
            f"examples={missing[:10]}"
        )

    d = contract.embedding_dim
    center = np.zeros((len(registry_rows), d), dtype=np.float32)
    offset = np.ones((len(registry_rows), d), dtype=np.float32) * 1e-6
    found = source_rows >= 0
    center[found] = raw[source_rows[found], :d]
    offset[found] = np.abs(raw[source_rows[found], d:])
    if np.any(offset[found] <= 0):
        # Exact zeros are valid raw outcomes but NBS uses log-offset features.
        offset[found] = np.maximum(offset[found], 1e-8)
    stats = _box_statistics(offset)

    center_file = out / "go_box_center.f32.npy"
    offset_file = out / "go_box_offset.f32.npy"
    stats_file = out / "go_box_stats.f32.npy"
    source_row_file = out / "go_box_source_row.i32.npy"
    np.save(center_file, center)
    np.save(offset_file, offset)
    np.save(stats_file, stats)
    np.save(source_row_file, source_rows)

    payload = {
        "schema_version": 1,
        "builder": "align_boxsqel_to_go_registry_v0.4",
        "model": "Neighborhood-BoxSquare",
        "geometry": "BoxSquaredEL",
        "artifact_selection": contract.artifact_selection,
        "selected_epoch": contract.selected_epoch,
        "embedding_dim": d,
        "classifier_go_terms": len(registry_rows),
        "ontology_classes": raw.shape[0],
        "matched_go_terms": int(found.sum()),
        "missing_go_terms": int((~found).sum()),
        "strict": bool(strict),
        "source": {
            "boxsqel_manifest": str(contract.manifest_path),
            "boxsqel_manifest_sha256": _sha256(contract.manifest_path),
            "parser_report": (
                None if contract.parser_report_path is None else str(contract.parser_report_path)
            ),
            "checkpoint": str(contract.checkpoint_path),
            "checkpoint_sha256": _sha256(contract.checkpoint_path),
            "go_registry": str(registry_path),
            "go_registry_sha256": _sha256(registry_path),
        },
        "arrays": {
            "center": {"file": center_file.name, "shape": list(center.shape), "dtype": "float32"},
            "offset": {"file": offset_file.name, "shape": list(offset.shape), "dtype": "float32"},
            "stats": {
                "file": stats_file.name,
                "shape": list(stats.shape),
                "dtype": "float32",
                "columns": [
                    "mean_log_offset",
                    "std_log_offset",
                    "mean_log_box_width",
                    "max_offset",
                    "normalized_go_depth",
                    "normalized_descendant_count",
                ],
                "topology_columns_initialized_to_zero": True,
            },
            "source_row": {
                "file": source_row_file.name,
                "shape": list(source_rows.shape),
                "dtype": "int32",
            },
        },
        "missing_examples": missing[:100],
    }
    temporary = alignment_manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, alignment_manifest)
    return alignment_manifest


def export_boxsqel_full_ontology(
    manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    parser_report_path: Optional[str | os.PathLike[str]] = None,
    project_root: Optional[str | os.PathLike[str]] = None,
    artifact_selection: str = "best",
    ontology_version: str = "unknown",
    overwrite: bool = False,
) -> Path:
    """Export all BoxSquaredEL classes in checkpoint row order.

    Unlike task alignment, this preserves the complete 44,919-class ontology
    space used by BoxSquaredEL.  Task classifier columns are mapped into this
    space separately through ``go_box_source_row.i32.npy``.
    """
    contract = load_boxsqel_training_contract(
        manifest_path,
        parser_report_path=parser_report_path,
        project_root=project_root,
        artifact_selection=artifact_selection,
        require_artifacts=True,
    )
    class_map, raw = _class_embedding_from_checkpoint(contract.checkpoint_path)
    if raw.shape != (contract.num_ontology_classes, contract.class_vector_width):
        raise ValueError(
            f"checkpoint class embedding shape {raw.shape} != "
            f"{(contract.num_ontology_classes, contract.class_vector_width)}"
        )
    by_row: list[Optional[str]] = [None] * raw.shape[0]
    for class_id, row in class_map.items():
        if row < 0 or row >= len(by_row):
            raise IndexError(f"BoxSquaredEL class row out of range: {class_id} -> {row}")
        if by_row[row] is not None:
            raise ValueError(f"duplicate BoxSquaredEL class row {row}")
        by_row[row] = class_id
    if any(value is None for value in by_row):
        raise ValueError("BoxSquaredEL class mapping does not cover every embedding row")

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    output_manifest = out / "boxsqel_full_ontology_manifest.json"
    if output_manifest.exists() and not overwrite:
        raise FileExistsError(f"full ontology output exists: {output_manifest}")

    d = contract.embedding_dim
    center = np.asarray(raw[:, :d], dtype=np.float32)
    offset = np.maximum(np.abs(raw[:, d:]), 1e-8).astype(np.float32, copy=False)
    stats = _box_statistics(offset)
    center_file = out / "full_go_box_center.f32.npy"
    offset_file = out / "full_go_box_offset.f32.npy"
    stats_file = out / "full_go_box_stats.f32.npy"
    registry_file = out / "boxsqel_class_registry.tsv"
    np.save(center_file, center)
    np.save(offset_file, offset)
    np.save(stats_file, stats)
    with registry_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "ontology_go_idx",
                "boxsqel_source_row",
                "class_id",
                "normalized_go_id",
                "is_go_class",
            ]
        )
        for row, class_id in enumerate(by_row):
            normalized = normalize_go_identifier(str(class_id))
            writer.writerow(
                [row, row, class_id, normalized or "", int(normalized is not None)]
            )

    payload = {
        "schema_version": 1,
        "builder": "export_boxsqel_full_ontology_v0.4",
        "model": "Neighborhood-BoxSquare",
        "geometry": "BoxSquaredEL",
        "ontology_version": str(ontology_version),
        "ontology_index_contract": "BoxSquaredEL checkpoint class row, zero-based",
        "num_ontology_classes": int(raw.shape[0]),
        "embedding_dim": int(d),
        "artifact_selection": contract.artifact_selection,
        "selected_epoch": contract.selected_epoch,
        "registry": {
            "file": registry_file.name,
            "sha256": _sha256(registry_file),
        },
        "source": {
            "boxsqel_manifest": str(contract.manifest_path),
            "boxsqel_manifest_sha256": _sha256(contract.manifest_path),
            "parser_report": None if contract.parser_report_path is None else str(contract.parser_report_path),
            "checkpoint": str(contract.checkpoint_path),
            "checkpoint_sha256": _sha256(contract.checkpoint_path),
            "normalized_ontology": str(contract.manifest.get("config", {}).get("data_file", "")),
        },
        "arrays": {
            "center": {"file": center_file.name, "shape": list(center.shape), "dtype": "float32"},
            "offset": {"file": offset_file.name, "shape": list(offset.shape), "dtype": "float32"},
            "stats": {"file": stats_file.name, "shape": list(stats.shape), "dtype": "float32"},
        },
        "future_go_interface": {
            "version_key": "ontology_version",
            "rebuild_required": [
                "BoxSquaredEL embedding",
                "full normalized G-G relations",
                "task-label-to-ontology mapping",
            ],
            "protein_universe_may_remain_fixed": True,
        },
    }
    output_manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_manifest
