#!/usr/bin/env python3
"""Prepare leakage-safe independent-test inputs for NBS.

The only required biological input is an ordered FASTA file or a trusted
pickle containing ordered protein IDs.  Pickle-only input reuses the configured
ind-test MSA binary; a pickle may also carry sequences directly.  This program:

1. reconstructs the frozen Stage-1 model and reads either the original MSA
   binary or a query-only (singleton) MSA built from FASTA;
2. exports the pooled Stage-1 representation and complete backbone
   probabilities;
3. runs the trained student selector inside the train-defined rare-GO
   vocabulary and exports fixed-K candidate evidence;
4. retrieves core neighbours and emits isolated core->test ``similar_to``
   evidence for NBS inductive message passing.

Labels may exist in the pickle, but this script never reads them into a model
input.  They remain an evaluation-only concern.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_NUM_CLASSES = {"bp": 21312, "mf": 7038, "cc": 2903}
TASK_KEYS = {
    "bp": ("bp", "biological_process"),
    "mf": ("mf", "molecular_function"),
    "cc": ("cc", "cellular_component"),
}
OUTPUT_FILES = {
    "protein_ids": "protein_ids.txt",
    "fasta": "normalized_sequences.fasta",
    "representation": "ind_test_repr.f16.npy",
    "base_probability": "backbone_ind_test_prob.f16.npy",
    "candidate_go": "candidate_go_index.i32.npy",
    "candidate_attr": "candidate_edge_attr.f32.npy",
    "pp_neighbors": "test_core_neighbors.i32.npy",
    "pp_attr": "test_core_edge_attr.f32.npy",
    "manifest": "ind_test_input_manifest.json",
}


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_records(records: Sequence[tuple[str, str | None]]) -> str:
    digest = hashlib.sha256()
    for protein_id, sequence in records:
        digest.update(protein_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(b"<MSA_BINARY>" if sequence is None else sequence.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def import_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_sequence(value: Any, protein_id: str) -> str:
    sequence = "".join(str(value).split()).upper()
    if not sequence:
        raise ValueError(f"empty sequence for protein {protein_id!r}")
    if any(ord(char) > 127 for char in sequence):
        raise ValueError(f"non-ASCII residue in protein {protein_id!r}")
    return sequence


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    parts: list[str] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, normalize_sequence("".join(parts), current_id)))
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"empty FASTA header at {path}:{line_number}")
                current_id = header.split()[0]
                parts = []
            else:
                if current_id is None:
                    raise ValueError(f"sequence before first FASTA header at {path}:{line_number}")
                parts.append(line)
    if current_id is not None:
        records.append((current_id, normalize_sequence("".join(parts), current_id)))
    if not records:
        raise ValueError(f"no FASTA records found in {path}")
    ids = [protein_id for protein_id, _ in records]
    if len(set(ids)) != len(ids):
        raise ValueError("FASTA protein IDs must be unique")
    return records


def _task_block(payload: Any, mode: str, task: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("pickle root must be a mapping")
    mode_block = payload.get(mode)
    if isinstance(mode_block, Mapping):
        for key in TASK_KEYS[task]:
            value = mode_block.get(key)
            if isinstance(value, Mapping):
                return value
    for key in TASK_KEYS[task]:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def _first_present(block: Mapping[str, Any], requested: str | None, candidates: Iterable[str]) -> Any:
    if requested:
        if requested not in block:
            raise KeyError(f"pickle block has no key={requested!r}")
        return block[requested]
    for key in candidates:
        if key in block:
            return block[key]
    return None


def parse_pickle(
    path: Path,
    *,
    mode: str,
    task: str,
    protein_key: str | None,
    sequence_key: str | None,
) -> tuple[list[str], list[str] | None]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    block = _task_block(payload, mode, task)
    proteins = _first_present(block, protein_key, ("proteins", "protein_ids", "ids"))
    if proteins is None:
        raise KeyError("pickle independent-test block has no protein IDs")
    protein_ids = [str(value) for value in list(proteins)]
    if not protein_ids or len(set(protein_ids)) != len(protein_ids):
        raise ValueError("pickle protein IDs must be non-empty and unique")
    raw_sequences = _first_present(
        block,
        sequence_key,
        ("sequences", "seqs", "protein_sequences", "sequence"),
    )
    if raw_sequences is None:
        return protein_ids, None
    sequence_values = list(raw_sequences)
    if len(sequence_values) != len(protein_ids):
        raise ValueError("pickle sequence count does not match protein IDs")
    sequences = [
        normalize_sequence(value, protein_id)
        for protein_id, value in zip(protein_ids, sequence_values)
    ]
    return protein_ids, sequences


def resolve_records(args: argparse.Namespace) -> tuple[list[tuple[str, str | None]], dict[str, Any]]:
    fasta_records = None if args.fasta is None else parse_fasta(args.fasta)
    pickle_ids: list[str] | None = None
    pickle_sequences: list[str] | None = None
    if args.metadata_file is not None:
        pickle_ids, pickle_sequences = parse_pickle(
            args.metadata_file,
            mode=args.mode,
            task=args.task,
            protein_key=args.pickle_protein_key,
            sequence_key=args.pickle_sequence_key,
        )
    if pickle_ids is None:
        if fasta_records is None:
            raise ValueError("provide --fasta or --metadata-file")
        records = fasta_records
    elif fasta_records is None:
        records = (
            [(protein_id, None) for protein_id in pickle_ids]
            if pickle_sequences is None
            else list(zip(pickle_ids, pickle_sequences))
        )
    else:
        fasta_by_id = dict(fasta_records)
        missing = [protein_id for protein_id in pickle_ids if protein_id not in fasta_by_id]
        extra = sorted(set(fasta_by_id) - set(pickle_ids))
        if missing or extra:
            raise ValueError(
                f"FASTA/pickle protein sets differ: missing={missing[:5]}, extra={extra[:5]}"
            )
        records = [(protein_id, fasta_by_id[protein_id]) for protein_id in pickle_ids]
    total_records = len(records)
    if args.limit_proteins:
        records = records[: int(args.limit_proteins)]
    source = {
        "fasta": None if args.fasta is None else str(args.fasta),
        "fasta_sha256": None if args.fasta is None else sha256_file(args.fasta),
        "metadata_file": None if args.metadata_file is None else str(args.metadata_file),
        "metadata_sha256": (
            None if args.metadata_file is None else sha256_file(args.metadata_file)
        ),
        "ordered_input_sha256": sha256_records(records),
        "protein_ids_sha256": sha256_ids([protein_id for protein_id, _ in records]),
        "num_proteins_before_limit": total_records,
        "limit_proteins": None if not args.limit_proteins else int(args.limit_proteins),
    }
    return records, source


def resolve_and_audit_msa_index(
    args: argparse.Namespace,
    records: Sequence[tuple[str, str | None]],
) -> None:
    """Resolve the independent-test MSA index and verify exact ID coverage.

    A Stage-1 training checkpoint normally records the Swiss-Prot training MSA
    index.  That path must never be inherited for metadata-only independent-test
    inference.  The project-local independent-test index is the default unless
    the caller explicitly supplies ``--msa-index``.
    """
    use_binary_msa = any(sequence is None for _, sequence in records)
    if not use_binary_msa:
        return
    if not all(sequence is None for _, sequence in records):
        raise ValueError("cannot mix sequence-backed and MSA-binary-backed rows")
    if args.msa_index is None:
        args.msa_index = (
            args.project_root / "data" / "ind_MSA_bin" / "index.pkl"
        ).resolve()
    if not args.msa_index.is_file():
        raise FileNotFoundError(
            "Independent-test MSA index not found: "
            f"{args.msa_index}. Set STAGE1_MSA_INDEX or --msa-index to the "
            "ind_MSA_bin/index.pkl file; do not use the Stage-1 training "
            "sprot_2204_MSA_bin index."
        )

    with args.msa_index.open("rb") as handle:
        index = pickle.load(handle)
    if not isinstance(index, Mapping) or "proteins" not in index:
        raise KeyError(f"MSA index has no proteins field: {args.msa_index}")
    indexed_ids = {str(value) for value in index["proteins"]}
    requested_ids = [protein_id for protein_id, _ in records]
    missing = [protein_id for protein_id in requested_ids if protein_id not in indexed_ids]
    if missing:
        matched = len(requested_ids) - len(missing)
        raise ValueError(
            "Independent-test protein/MSA index mismatch before model loading: "
            f"matched={matched}/{len(requested_ids)}, "
            f"binary_records={len(indexed_ids)}, missing_examples={missing[:10]}, "
            f"msa_index={args.msa_index}. This commonly means the training "
            "sprot_2204_MSA_bin index was selected instead of ind_MSA_bin."
        )
    print(
        f"[MSA index] exact ID coverage={len(requested_ids)}/{len(requested_ids)}, "
        f"binary_records={len(indexed_ids)}, path={args.msa_index}",
        flush=True,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def resolve_relative(manifest_path: Path, value: str | os.PathLike[str]) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else []
    candidates.extend((manifest_path.parent / raw, manifest_path.parent / raw.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve {value!r} relative to {manifest_path}")


@dataclass(frozen=True)
class Stage1Artifacts:
    representation_manifest: Path
    weak_manifest: Path
    go_registry: Path
    core_repr: Path
    rare_indices: Path
    candidate_topk: int
    candidate_selector_scope: str


def resolve_stage1_artifacts(args: argparse.Namespace) -> Stage1Artifacts:
    data_root = args.data_root.resolve()
    representation_manifest = (
        args.representation_manifest
        if args.representation_manifest is not None
        else data_root / "features" / "representation_manifest.json"
    ).resolve()
    weak_manifest = (
        args.weak_graph_manifest
        if args.weak_graph_manifest is not None
        else data_root / "weak_graph_predictions" / "weak_graph_predictions_manifest.json"
    ).resolve()
    rep = load_json(representation_manifest)
    weak = load_json(weak_manifest)
    core_records = [value for value in rep.get("roles", []) if value.get("role") == "core"]
    if len(core_records) != 1:
        raise KeyError("representation manifest must contain exactly one core role")
    core_repr = resolve_relative(representation_manifest, core_records[0]["feature_file"])
    rare = weak.get("rare_definition", {})
    rare_indices = resolve_relative(weak_manifest, rare["rare_indices_file"])
    go_value = args.go_registry
    if go_value is None:
        go_value = Path(str(weak.get("go_registry", {}).get("path", "")))
        if not go_value.is_file():
            go_value = data_root / "gg_relations" / "go_registry.tsv"
    go_registry = go_value.resolve()
    selector = weak.get("model_semantics", {}).get("protein_go_selector")
    if selector is None:
        selector = weak["model_semantics"]["rare_selector"]
    manifest_scope = str(selector.get("scope", "rare_first"))
    candidate_selector_scope = (
        manifest_scope
        if args.candidate_selector_scope == "auto"
        else str(args.candidate_selector_scope)
    )
    if candidate_selector_scope == "restricted":
        candidate_selector_scope = "rare_first"
    if candidate_selector_scope not in {"rare_first", "full_task"}:
        raise ValueError(
            "inductive candidate preparation supports rare_first or full_task, "
            f"got {candidate_selector_scope!r}"
        )
    candidate_topk = int(args.candidate_topk or selector["requested_topk"])
    return Stage1Artifacts(
        representation_manifest=representation_manifest,
        weak_manifest=weak_manifest,
        go_registry=go_registry,
        core_repr=core_repr,
        rare_indices=rare_indices,
        candidate_topk=candidate_topk,
        candidate_selector_scope=candidate_selector_scope,
    )


def build_cache_signature(
    args: argparse.Namespace,
    records: Sequence[tuple[str, str | None]],
    source: Mapping[str, Any],
    artifacts: Stage1Artifacts,
) -> dict[str, Any]:
    checkpoint_sha256 = sha256_file(args.stage1_checkpoint)
    representation_contract = load_json(artifacts.representation_manifest)
    weak_contract = load_json(artifacts.weak_manifest)
    expected_checkpoint_hashes = {
        "representation_manifest": representation_contract.get("checkpoint_sha256"),
        "weak_graph_manifest": weak_contract.get("checkpoint", {}).get("sha256"),
    }
    mismatched = {
        source_name: expected
        for source_name, expected in expected_checkpoint_hashes.items()
        if expected and str(expected) != checkpoint_sha256
    }
    if mismatched:
        raise RuntimeError(
            "Stage-1 checkpoint does not match the artifacts used to train NBS: "
            f"actual={checkpoint_sha256}, expected={mismatched}"
        )
    if str(weak_contract.get("task", args.task)) != args.task:
        raise ValueError("weak graph manifest task differs from requested task")
    return {
        "task": args.task,
        "num_classes": TASK_NUM_CLASSES[args.task],
        "ordered_input_sha256": source["ordered_input_sha256"],
        "stage1_checkpoint_sha256": checkpoint_sha256,
        "representation_manifest_sha256": sha256_file(artifacts.representation_manifest),
        "weak_graph_manifest_sha256": sha256_file(artifacts.weak_manifest),
        "go_registry_sha256": sha256_file(artifacts.go_registry),
        "rare_indices_sha256": sha256_file(artifacts.rare_indices),
        "msa_index_sha256": (
            None if args.msa_index is None else sha256_file(args.msa_index)
        ),
        "stage1_model_config_sha256": (
            None
            if args.stage1_model_config is None
            else sha256_file(args.stage1_model_config)
        ),
        "num_proteins": len(records),
        "candidate_topk": artifacts.candidate_topk,
        "candidate_selector_scope": artifacts.candidate_selector_scope,
        "selector_affinity_chunk_size": int(args.selector_affinity_chunk_size),
        "pp_topk": int(args.pp_topk),
        "sequence_encoding": (
            "metadata_msa_binary" if any(sequence is None for _, sequence in records) else "singleton_msa"
        ),
    }


def validate_cached(output_dir: Path, signature: Mapping[str, Any]) -> bool:
    manifest_path = output_dir / OUTPUT_FILES["manifest"]
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_json(manifest_path)
        if manifest.get("cache_signature") != dict(signature):
            return False
        n = int(signature["num_proteins"])
        g = int(signature["num_classes"])
        k = int(signature["candidate_topk"])
        ppk = int(signature["pp_topk"])
        expected = {
            "representation": (n, int(manifest["representation"]["feature_dim"])),
            "base_probability": (n, g),
            "candidate_go": (n, k),
            "candidate_attr": (n, k, 3),
            "pp_neighbors": (n, ppk),
            "pp_attr": (n, ppk, 3),
        }
        expected_hashes = {
            "representation": manifest["representation"].get("sha256"),
            "base_probability": manifest["base_probability"].get("sha256"),
            "candidate_go": manifest["candidate_evidence"].get("go_index_sha256"),
            "candidate_attr": manifest["candidate_evidence"].get("edge_attr_sha256"),
            "pp_neighbors": manifest["external_pp"].get("neighbor_index_sha256"),
            "pp_attr": manifest["external_pp"].get("edge_attr_sha256"),
        }
        for key, shape in expected.items():
            path = output_dir / OUTPUT_FILES[key]
            if not path.is_file() or tuple(np.load(path, mmap_mode="r").shape) != shape:
                return False
            expected_hash = expected_hashes[key]
            if not expected_hash or sha256_file(path) != str(expected_hash):
                return False
        protein_ids = output_dir / OUTPUT_FILES["protein_ids"]
        if not protein_ids.is_file() or sha256_file(protein_ids) != str(
            manifest.get("protein_ids_file_sha256", "")
        ):
            return False
        return True
    except Exception:
        return False


def load_stage1_model(args: argparse.Namespace, artifacts: Stage1Artifacts) -> tuple[Any, Any, Any, Any]:
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "msa_models"))
    import torch

    exporter = import_from_path(
        "_nbs_stage1_export_helpers",
        project_root / "scripts" / "export_weak_graph_predictions.py",
    )
    diagnostic_path = project_root / "experiments" / "eval_weak_ind_test_detr_diagnostics.py"
    diagnostic = exporter.import_module_from_path("_nbs_stage1_diagnostic", diagnostic_path)
    weak_module = sys.modules.get("experiments.weak_exp_train_detr")
    exp_module = sys.modules.get("experiments.exp_train")
    if weak_module is None or exp_module is None:
        raise ImportError("Stage-1 diagnostic evaluator did not load LATENCE model modules")

    seed_argv = [
        "--project-root", str(project_root),
        "--diagnostic-eval-script", str(diagnostic_path),
        "--checkpoint", str(args.stage1_checkpoint),
        "--task", args.task,
        "--output-dir", str(args.output_dir),
        "--feature-dir", str(artifacts.representation_manifest.parent),
        "--go-registry", str(artifacts.go_registry),
        "--roles", "core",
        "--modelout-role", "core",
        "--batch-size", str(args.batch_size),
        "--dataloader-num-workers", "0",
        "--device", args.device,
        "--rare-go-topk", str(artifacts.candidate_topk),
    ]
    if args.metadata_file is not None:
        seed_argv += ["--file-address", str(args.metadata_file)]
    if args.msa_index is not None:
        seed_argv += ["--working-address", str(args.msa_index)]
    if args.stage1_model_config is not None:
        seed_argv += ["--model-config", str(args.stage1_model_config)]
    if args.train_args_json != "auto":
        seed_argv += ["--train-args-json", args.train_args_json]
    if args.no_amp:
        seed_argv += ["--no-amp"]
    export_args = exporter.build_parser().parse_args(seed_argv)
    payload = exporter.torch_load(torch, args.stage1_checkpoint)
    training_args, _ = exporter.find_model_args(
        args.stage1_checkpoint, payload, args.train_args_json
    )
    model_args, _ = exporter.make_eval_model_args(
        args=export_args,
        diagnostic_module=diagnostic,
        payload_model_args=training_args,
    )
    device = exporter.resolve_device(torch, args.device)
    if device.type == "cuda" and float(args.min_free_gpu_gb) > 0:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = float(free_bytes) / (1024 ** 3)
        total_gb = float(total_bytes) / (1024 ** 3)
        print(
            f"[CUDA preflight] device={device} free={free_gb:.2f}GB/"
            f"{total_gb:.2f}GB required={float(args.min_free_gpu_gb):.2f}GB",
            flush=True,
        )
        if free_gb < float(args.min_free_gpu_gb):
            raise RuntimeError(
                f"Stage-1 device {device} has only {free_gb:.2f}GB free, below "
                f"--min-free-gpu-gb={float(args.min_free_gpu_gb):.2f}. "
                "Do not run independent inference on a GPU occupied by NBS "
                "training; select another STAGE1_EVAL_DEVICE or wait for training."
            )
    model_args.device = str(device)
    model_args.gpu_ids = "" if device.type == "cpu" else str(device.index or 0)
    opt = weak_module.build_weak_opt_from_config(model_args)
    opt.mode = "eval"
    opt.shuffle = False
    model = weak_module.WeakMSAGOWithDETRDecoder(opt, model_args)
    exporter.load_checkpoint_strict(model, payload, allow_partial=False)
    model = model.to(device).eval()
    return model, model_args, exporter, (weak_module, exp_module, torch, device, opt)


def load_alphabet(project_root: Path) -> str:
    sys.path.insert(0, str(project_root))
    try:
        from msa_models.helper_functions import constants as constants
    except ImportError as exc:
        raise ImportError("cannot import msa_models.helper_functions.constants") from exc
    alphabet = str(constants.C.trR_ALPHABET)
    if not alphabet or len(alphabet) > 255:
        raise ValueError("invalid Stage-1 residue alphabet")
    return alphabet


def encode_singleton_batch(
    records: Sequence[tuple[str, str]],
    *,
    top_k: int,
    max_len: int,
    alphabet: str,
    unknown_policy: str,
) -> tuple[np.ndarray, int, int]:
    mapping = {character: index for index, character in enumerate(alphabet)}
    x_token = mapping.get("X")
    output = np.zeros((len(records), int(top_k), int(max_len)), dtype=np.uint8)
    truncated = 0
    unknown = 0
    for row, (protein_id, sequence) in enumerate(records):
        if len(sequence) > max_len:
            truncated += 1
        for column, residue in enumerate(sequence[:max_len]):
            token = mapping.get(residue)
            if token is None:
                unknown += 1
                if unknown_policy == "error":
                    raise ValueError(
                        f"residue {residue!r} in protein {protein_id!r} is absent from Stage-1 alphabet"
                    )
                if unknown_policy == "x" and x_token is not None:
                    token = x_token
                else:
                    token = 0
            output[row, 0, column] = int(token)
    return output, truncated, unknown


def run_stage1(
    args: argparse.Namespace,
    records: Sequence[tuple[str, str | None]],
    artifacts: Stage1Artifacts,
    stage_dir: Path,
) -> dict[str, Any]:
    model, model_args, exporter, loaded = load_stage1_model(args, artifacts)
    weak_module, exp_module, torch, device, opt = loaded
    use_binary_msa = any(sequence is None for _, sequence in records)
    if use_binary_msa and not all(sequence is None for _, sequence in records):
        raise ValueError("cannot mix sequence-backed and MSA-binary-backed rows")
    alphabet = None if use_binary_msa else load_alphabet(args.project_root)
    rare_indices_np = np.asarray(np.load(artifacts.rare_indices), dtype=np.int64)
    rare_indices = torch.from_numpy(rare_indices_np).to(device=device, dtype=torch.long)
    selector_universe = (
        torch.arange(TASK_NUM_CLASSES[args.task], device=device, dtype=torch.long)
        if artifacts.candidate_selector_scope == "full_task"
        else rare_indices
    )
    if artifacts.candidate_topk > selector_universe.numel():
        raise ValueError(
            "candidate top-k exceeds the configured selector GO vocabulary"
        )

    top_k = int(model_args.top_k)
    max_len = int(model_args.max_len)
    representation_path = stage_dir / OUTPUT_FILES["representation"]
    base_path = stage_dir / OUTPUT_FILES["base_probability"]
    candidate_go_path = stage_dir / OUTPUT_FILES["candidate_go"]
    candidate_attr_path = stage_dir / OUTPUT_FILES["candidate_attr"]
    num_rows = len(records)
    num_classes = TASK_NUM_CLASSES[args.task]
    representation = None
    base_output = np.lib.format.open_memmap(
        base_path, mode="w+", dtype=np.float16, shape=(num_rows, num_classes)
    )
    candidate_go = np.lib.format.open_memmap(
        candidate_go_path,
        mode="w+",
        dtype=np.int32,
        shape=(num_rows, artifacts.candidate_topk),
    )
    candidate_attr = np.lib.format.open_memmap(
        candidate_attr_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_rows, artifacts.candidate_topk, 3),
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    truncated_total = 0
    unknown_total = 0
    feature_dim = None
    if use_binary_msa:
        metadata_task = TASK_KEYS[args.task][1]
        # Build a label-free selection pickle from the already resolved ordered
        # records.  This makes smoke limits effective during Stage-1 encoding
        # and prevents the original evaluation annotations from entering the
        # dataset object at all.
        selection_metadata = stage_dir / "msa_selection.pkl"
        selection_payload = {
            args.mode: {
                metadata_task: {
                    "proteins": [protein_id for protein_id, _ in records],
                    "prop_annotations": [[] for _ in records],
                }
            }
        }
        with selection_metadata.open("wb") as handle:
            pickle.dump(selection_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        dataset_opt = copy.copy(opt)
        dataset_opt.file_address = str(selection_metadata)
        dataset_opt.working_address = str(args.msa_index)
        dataset = exp_module.build_msa_dataset(
            dataset_opt,
            mode=args.mode,
            task=metadata_task,
            need_proteins=True,
        )
        if len(dataset) != num_rows:
            raise ValueError(
                f"MSA binary ind_test rows={len(dataset)} != pickle proteins={num_rows}"
            )
        # Protein IDs from metadata are required to align the binary index, but
        # annotation tensors are not materialized during representation export.
        dataset.return_labels = False
        loader = exp_module.make_loader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            rank=0,
            world_size=1,
            seed=3407,
            persistent_workers=False,
        )
        def binary_batches():
            row_by_id = {protein_id: row for row, (protein_id, _) in enumerate(records)}
            seen: set[int] = set()
            processed = 0
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise RuntimeError("unexpected label-free MSA batch structure")
                proteins, encoded = batch
                batch_size = int(encoded.shape[0])
                protein_ids = [str(value) for value in list(proteins)]
                missing = [protein_id for protein_id in protein_ids if protein_id not in row_by_id]
                if missing:
                    raise RuntimeError(f"MSA binary returned unknown proteins: {missing[:5]}")
                rows = np.asarray([row_by_id[protein_id] for protein_id in protein_ids], dtype=np.int64)
                duplicate_rows = [int(row) for row in rows if int(row) in seen]
                if duplicate_rows:
                    raise RuntimeError(f"MSA binary repeated ind_test rows: {duplicate_rows[:5]}")
                seen.update(int(row) for row in rows)
                yield rows, protein_ids, encoded
                processed += batch_size
            if processed != num_rows or len(seen) != num_rows:
                raise RuntimeError("MSA binary loader did not cover every ind_test protein")

        batch_iterator = binary_batches()
    else:
        assert alphabet is not None
        def singleton_batches():
            nonlocal truncated_total, unknown_total
            for start in range(0, num_rows, args.batch_size):
                end = min(start + args.batch_size, num_rows)
                batch_records = records[start:end]
                singleton_records = [
                    (protein_id, str(sequence)) for protein_id, sequence in batch_records
                ]
                encoded_np, truncated, unknown = encode_singleton_batch(
                    singleton_records,
                    top_k=top_k,
                    max_len=max_len,
                    alphabet=alphabet,
                    unknown_policy=args.unknown_residue,
                )
                truncated_total += truncated
                unknown_total += unknown
                protein_ids = [protein_id for protein_id, _ in batch_records]
                yield np.arange(start, end, dtype=np.int64), protein_ids, torch.from_numpy(encoded_np)

        batch_iterator = singleton_batches()

    with torch.no_grad():
        processed = 0
        for rows, protein_ids, encoded in batch_iterator:
            exp_module.set_model_proteins(model, protein_ids)
            encoded = encoded.to(device=device, dtype=torch.long)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                base_logits, h = model.backbone(
                    encoded,
                    permute_dims=tuple(int(value) for value in model_args.permute_dims),
                    return_embedding=True,
                )
                _, pooled = weak_module.build_backbone_memory(h, mode="pooled")
                selected_idx, selector_score = exporter.rare_first_selector_topk(
                    torch_module=torch,
                    weak_module=weak_module,
                    model=model,
                    base_logits=base_logits,
                    h=h,
                    allowed_go_idx=selector_universe,
                    requested_k=artifacts.candidate_topk,
                    affinity_chunk_size=int(args.selector_affinity_chunk_size),
                )
            pooled_np = pooled.float().cpu().numpy()
            if representation is None:
                feature_dim = int(pooled_np.shape[1])
                representation = np.lib.format.open_memmap(
                    representation_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(num_rows, feature_dim),
                )
            base_prob = torch.sigmoid(base_logits.float())
            selected_prob = torch.gather(base_prob, dim=1, index=selected_idx)
            reciprocal_rank = 1.0 / (
                np.arange(artifacts.candidate_topk, dtype=np.float32) + 1.0
            )
            representation[rows] = pooled_np.astype(np.float16)
            base_output[rows] = base_prob.cpu().numpy().astype(np.float16)
            candidate_go[rows] = selected_idx.cpu().numpy().astype(np.int32)
            candidate_attr[rows, :, 0] = selected_prob.cpu().numpy()
            candidate_attr[rows, :, 1] = selector_score.float().cpu().numpy()
            candidate_attr[rows, :, 2] = reciprocal_rank[None, :]
            processed += len(protein_ids)
            input_mode = "MSA binary" if use_binary_msa else "singleton MSA"
            print(f"[Stage-1 {input_mode}] rows={processed}/{num_rows}", flush=True)
    assert representation is not None and feature_dim is not None
    for value in (representation, base_output, candidate_go, candidate_attr):
        value.flush()
    return {
        "feature_dim": feature_dim,
        "top_k": top_k,
        "max_len": max_len,
        "input_mode": "metadata_msa_binary" if use_binary_msa else "singleton_msa",
        "alphabet": alphabet,
        "truncated_proteins": truncated_total,
        "unknown_residues": unknown_total,
        "checkpoint_type": "weak_detr_decoder_v3",
        "selector_affinity_chunk_size": int(args.selector_affinity_chunk_size),
    }


def normalized_rows(value: np.ndarray, *, label: str) -> np.ndarray:
    output = np.asarray(value, dtype=np.float32).copy()
    if not np.isfinite(output).all():
        raise FloatingPointError(f"non-finite {label} representations")
    norm = np.linalg.norm(output, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError(f"zero-norm {label} representation")
    output /= norm[:, None]
    return output


def retrieve_core_neighbors(
    args: argparse.Namespace,
    query_repr_path: Path,
    core_repr_path: Path,
    stage_dir: Path,
) -> dict[str, Any]:
    query = np.load(query_repr_path, mmap_mode="r")
    core = np.load(core_repr_path, mmap_mode="r")
    if query.ndim != 2 or core.ndim != 2 or query.shape[1] != core.shape[1]:
        raise ValueError("query/core representation shapes are incompatible")
    k = int(args.pp_topk)
    if k <= 0:
        raise ValueError("pp-topk must be positive")
    if k > int(core.shape[0]):
        raise ValueError(
            f"pp-topk={k} exceeds available core proteins={core.shape[0]}"
        )

    if args.faiss_backend == "faiss":
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("FAISS is required; use --faiss-backend numpy only for a small test") from exc
        if args.faiss_threads > 0:
            faiss.omp_set_num_threads(int(args.faiss_threads))
        cpu_index = faiss.IndexFlatIP(int(core.shape[1]))
        resources = None
        if args.faiss_gpu_id >= 0:
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("installed FAISS has no GPU support")
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, int(args.faiss_gpu_id), cpu_index)
            backend = f"faiss-gpu:{args.faiss_gpu_id}"
        else:
            index = cpu_index
            backend = "faiss-cpu"
        for start in range(0, core.shape[0], args.faiss_add_batch):
            end = min(start + args.faiss_add_batch, core.shape[0])
            index.add(np.ascontiguousarray(normalized_rows(core[start:end], label="core")))
        score_chunks: list[np.ndarray] = []
        neighbor_chunks: list[np.ndarray] = []
        for start in range(0, query.shape[0], args.faiss_query_batch):
            end = min(start + args.faiss_query_batch, query.shape[0])
            scores, neighbors = index.search(
                np.ascontiguousarray(normalized_rows(query[start:end], label="query")), k
            )
            score_chunks.append(np.asarray(scores, dtype=np.float32))
            neighbor_chunks.append(np.asarray(neighbors, dtype=np.int32))
        scores = np.concatenate(score_chunks, axis=0)
        neighbors = np.concatenate(neighbor_chunks, axis=0)
        del resources
    else:
        if int(query.shape[0]) * int(core.shape[0]) > int(args.numpy_max_pairs):
            raise RuntimeError(
                "numpy backend safety limit exceeded; install/use FAISS for the complete ind_test"
            )
        core_norm = normalized_rows(core, label="core")
        query_norm = normalized_rows(query, label="query")
        dense = query_norm @ core_norm.T
        take = np.argpartition(dense, -k, axis=1)[:, -k:]
        values = np.take_along_axis(dense, take, axis=1)
        order = np.argsort(-values, axis=1, kind="stable")
        neighbors = np.take_along_axis(take, order, axis=1).astype(np.int32)
        scores = np.take_along_axis(values, order, axis=1).astype(np.float32)
        backend = "numpy-exact"

    if np.any(neighbors < 0) or np.any(neighbors >= core.shape[0]):
        raise RuntimeError("invalid core neighbour index")
    edge_attr = np.empty((query.shape[0], k, 3), dtype=np.float32)
    edge_attr[:, :, 0] = np.clip(scores, 0.0, 1.0)
    edge_attr[:, :, 1] = scores
    edge_attr[:, :, 2] = 1.0 / (np.arange(k, dtype=np.float32)[None, :] + 1.0)
    np.save(stage_dir / OUTPUT_FILES["pp_neighbors"], neighbors)
    np.save(stage_dir / OUTPUT_FILES["pp_attr"], edge_attr)
    return {
        "backend": backend,
        "core_rows": int(core.shape[0]),
        "neighbors_per_test": k,
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "retrieval_direction": "test_to_core",
        "message_direction": "core_to_test",
        "trained_relation_operator": "similar_to",
        "test_to_test_edges": False,
    }


def publish_stage(stage_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, name in OUTPUT_FILES.items():
        if key == "manifest":
            continue
        source = stage_dir / name
        if source.exists():
            os.replace(source, output_dir / name)


def write_normalized_inputs(stage_dir: Path, records: Sequence[tuple[str, str | None]]) -> None:
    atomic_text(
        stage_dir / OUTPUT_FILES["protein_ids"],
        "".join(f"{protein_id}\n" for protein_id, _ in records),
    )
    if all(sequence is not None for _, sequence in records):
        lines: list[str] = []
        for protein_id, sequence in records:
            lines.extend((f">{protein_id}", str(sequence)))
        atomic_text(stage_dir / OUTPUT_FILES["fasta"], "\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--task", choices=sorted(TASK_NUM_CLASSES), required=True)
    parser.add_argument("--mode", default="ind_test")
    source = parser.add_argument_group("biological input")
    source.add_argument("--fasta", type=Path, default=None)
    source.add_argument("--metadata-file", type=Path, default=None)
    source.add_argument("--pickle-protein-key", default=None)
    source.add_argument("--pickle-sequence-key", default=None)

    model = parser.add_argument_group("frozen Stage-1")
    model.add_argument("--stage1-checkpoint", type=Path, required=True)
    model.add_argument("--stage1-model-config", type=Path, default=None)
    model.add_argument("--msa-index", type=Path, default=None)
    model.add_argument("--train-args-json", default="auto")
    model.add_argument("--data-root", type=Path, required=True)
    model.add_argument("--representation-manifest", type=Path, default=None)
    model.add_argument("--weak-graph-manifest", type=Path, default=None)
    model.add_argument("--go-registry", type=Path, default=None)
    model.add_argument("--candidate-topk", type=int, default=0)
    model.add_argument(
        "--candidate-selector-scope",
        choices=("auto", "rare_first", "full_task"),
        default="auto",
        help=(
            "auto follows the weak-graph manifest; full_task selects top-k from "
            "all task GO terms; rare_first retains the historical rare-only universe"
        ),
    )
    model.add_argument("--unknown-residue", choices=("error", "x", "pad"), default="x")

    runtime = parser.add_argument_group("runtime/cache")
    runtime.add_argument("--output-dir", type=Path, required=True)
    runtime.add_argument("--cache-policy", choices=("reuse", "refresh", "require"), default="reuse")
    runtime.add_argument("--batch-size", type=int, default=8)
    runtime.add_argument("--device", default="cuda:0")
    runtime.add_argument(
        "--min-free-gpu-gb",
        type=float,
        default=8.0,
        help="fail before model materialization when the selected CUDA device lacks headroom",
    )
    runtime.add_argument(
        "--selector-affinity-chunk-size",
        type=int,
        default=256,
        help="bound the [batch,prefilter,feature] selector-affinity temporary",
    )
    runtime.add_argument("--no-amp", action="store_true")
    runtime.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    runtime.add_argument(
        "--limit-proteins",
        type=int,
        default=0,
        help="Prepare only the first N ordered inputs for a smoke test (0: all).",
    )

    pp = parser.add_argument_group("isolated test-to-core retrieval")
    pp.add_argument(
        "--pp-topk",
        type=int,
        default=8,
        help="Store at least the maximum trained similar_to fanout (BP v0.5.6: 8).",
    )
    pp.add_argument("--faiss-backend", choices=("faiss", "numpy"), default="faiss")
    pp.add_argument("--faiss-gpu-id", type=int, default=-1)
    pp.add_argument("--faiss-threads", type=int, default=0)
    pp.add_argument("--faiss-add-batch", type=int, default=32768)
    pp.add_argument("--faiss-query-batch", type=int, default=4096)
    pp.add_argument("--numpy-max-pairs", type=int, default=5_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.stage1_checkpoint = args.stage1_checkpoint.expanduser().resolve()
    for name in (
        "fasta",
        "metadata_file",
        "stage1_model_config",
        "msa_index",
        "representation_manifest",
        "weak_graph_manifest",
        "go_registry",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if (
        args.batch_size <= 0
        or args.pp_topk <= 0
        or args.limit_proteins < 0
        or args.min_free_gpu_gb < 0
        or args.selector_affinity_chunk_size < 0
    ):
        raise ValueError(
            "batch-size/pp-topk must be positive; limit/min-free/chunk-size "
            "must be non-negative"
        )

    records, source = resolve_records(args)
    resolve_and_audit_msa_index(args, records)
    if not args.data_root.is_dir():
        raise NotADirectoryError(
            "--data-root must be the NBS data directory containing features/ "
            f"and weak_graph_predictions/, got: {args.data_root}"
        )
    artifacts = resolve_stage1_artifacts(args)
    signature = build_cache_signature(args, records, source, artifacts)
    cached = validate_cached(args.output_dir, signature)
    if args.cache_policy == "require" and not cached:
        raise RuntimeError("compatible independent-test artifacts were not found")
    if args.cache_policy == "reuse" and cached:
        print(f"[Reuse] validated independent-test artifacts: {args.output_dir}")
        print(args.output_dir / OUTPUT_FILES["manifest"])
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".ind_test_prepare_", dir=args.output_dir))
    started = time.time()
    try:
        write_normalized_inputs(stage_dir, records)
        stage1 = run_stage1(args, records, artifacts, stage_dir)
        pp = retrieve_core_neighbors(
            args,
            stage_dir / OUTPUT_FILES["representation"],
            artifacts.core_repr,
            stage_dir,
        )
        publish_stage(stage_dir, args.output_dir)
        manifest = {
            "schema_version": 2,
            "builder": "prepare_nbs_ind_test_inputs_v0.6.0-full-task-candidates",
            "task": args.task,
            "mode": args.mode,
            "cache_signature": signature,
            "input": dict(source),
            "num_proteins": len(records),
            "protein_ids": str(args.output_dir / OUTPUT_FILES["protein_ids"]),
            "protein_ids_sha256": source["protein_ids_sha256"],
            "protein_ids_file_sha256": sha256_file(
                args.output_dir / OUTPUT_FILES["protein_ids"]
            ),
            "sequence_encoding": {
                "mode": stage1["input_mode"],
                "labels_consumed": False,
                **stage1,
            },
            "representation": {
                "path": str(args.output_dir / OUTPUT_FILES["representation"]),
                "shape": [len(records), int(stage1["feature_dim"])],
                "dtype": "float16",
                "feature_dim": int(stage1["feature_dim"]),
                "sha256": sha256_file(args.output_dir / OUTPUT_FILES["representation"]),
            },
            "base_probability": {
                "path": str(args.output_dir / OUTPUT_FILES["base_probability"]),
                "shape": [len(records), TASK_NUM_CLASSES[args.task]],
                "dtype": "float16",
                "sha256": sha256_file(args.output_dir / OUTPUT_FILES["base_probability"]),
            },
            "candidate_evidence": {
                "go_index": str(args.output_dir / OUTPUT_FILES["candidate_go"]),
                "go_index_sha256": sha256_file(
                    args.output_dir / OUTPUT_FILES["candidate_go"]
                ),
                "edge_attr": str(args.output_dir / OUTPUT_FILES["candidate_attr"]),
                "edge_attr_sha256": sha256_file(
                    args.output_dir / OUTPUT_FILES["candidate_attr"]
                ),
                "edge_attr_columns": [
                    "backbone_probability",
                    "selector_score",
                    "reciprocal_rank",
                ],
                "fixed_k": artifacts.candidate_topk,
                "selector_scope": artifacts.candidate_selector_scope,
                "expert_probability_used": False,
                "label_hint_used": False,
            },
            "external_pp": {
                **pp,
                "core_representation": str(artifacts.core_repr),
                "neighbor_index": str(args.output_dir / OUTPUT_FILES["pp_neighbors"]),
                "neighbor_index_sha256": sha256_file(
                    args.output_dir / OUTPUT_FILES["pp_neighbors"]
                ),
                "edge_attr": str(args.output_dir / OUTPUT_FILES["pp_attr"]),
                "edge_attr_sha256": sha256_file(
                    args.output_dir / OUTPUT_FILES["pp_attr"]
                ),
                "edge_attr_columns": ["confidence", "cosine_score", "reciprocal_rank"],
            },
            "registries": {
                "go_registry": str(artifacts.go_registry),
                "go_registry_sha256": sha256_file(artifacts.go_registry),
                "representation_manifest": str(artifacts.representation_manifest),
                "weak_graph_manifest": str(artifacts.weak_manifest),
            },
            "leakage_contract": {
                "ind_test_labels_in_representation": False,
                "ind_test_labels_in_candidate_selection": False,
                "ind_test_labels_in_pp_retrieval": False,
                "expert_probability_in_forward_inputs": False,
                "test_to_test_edges": False,
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }
        atomic_text(
            args.output_dir / OUTPUT_FILES["manifest"],
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
