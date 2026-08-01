#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export weak-model predictions and graph-ready Protein--GO relations.

This exporter is intentionally separate from the diagnostic evaluator.  It
reuses the same LATENCE model classes, checkpoint configuration and forward
semantics, but writes memory-mapped arrays that can be consumed by the
downstream ``nbs_protein_go`` pipeline.

Default scientific contract
---------------------------
* ``backbone_prob`` is computed without external/expert probabilities.
* rare Protein--GO candidates follow an explicit rare-first pipeline:
  determine the train-defined rare-GO vocabulary, run the trained selector
  inside that vocabulary, and retain the requested top-k rare terms.
* weak ``modelout`` uses the external expert probability as decoder input and
  therefore corresponds to:

    modelout::mix_expert_base_anchor::decoderprob::expert

  when the checkpoint's ``query_decoder_logit_base_mode`` is
  ``mix_expert_base_anchor``.
* pseudo labels use the strict comparison ``modelout_prob > threshold``.
* no ground-truth label is passed to the selector/query decoder.

The default role mapping is identical to ``export_protein_universe_repr.py``:

    core -> train
    weak -> exp_train

All Protein indices are checked against ``features/protein_registry.csv``.
All GO indices are checked against ``gg_relations/go_registry.tsv`` and hence
remain aligned with the weak-model classifier output dimension.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


EXPORTER_ID = "export_weak_graph_predictions_v2"
EXPORTER_VERSION = "1.1.0-rare-first-selector"

ROLE_TO_MODE = {
    "core": "train",
    "weak": "exp_train",
    "valid": "valid",
    "ind_test": "ind_test",
}

TASK_TO_METADATA_KEY = {
    "bp": "biological_process",
    "mf": "molecular_function",
    "cc": "cellular_component",
}

TASK_NUM_CLASSES = {
    "bp": 21312,
    "mf": 7038,
    "cc": 2903,
}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def csv_items(value: str) -> List[str]:
    items = [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export backbone predictions, selector-ranked rare Protein--GO edges, "
            "and weak modelout pseudo targets"
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--diagnostic-eval-script", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-args-json", type=str, default="auto")
    parser.add_argument("--task", choices=["bp", "mf", "cc"], required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--go-registry", type=Path, required=True)
    parser.add_argument("--roles", type=csv_items, default=csv_items("core,weak"))
    parser.add_argument(
        "--role-mode",
        action="append",
        default=[],
        metavar="ROLE=MODE",
        help="Override a role-to-MSABinaryDataset mode mapping; may be repeated.",
    )

    # Architecture and dataset paths.  Explicit values take precedence over
    # args.json/checkpoint model_args.
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--file-address", type=str, default=None)
    parser.add_argument("--working-address", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--msa-max-size", type=int, default=None)
    parser.add_argument("--permute-dims", type=int, nargs=4, default=None)
    parser.add_argument(
        "--query-decoder-logit-base-mode",
        choices=[
            "base_residual",
            "expert_base",
            "mix_expert_base",
            "anchor_delta",
            "mix_expert_base_anchor",
        ],
        default=None,
        help="Optional audited override. Normally restored from args.json/model_args.",
    )
    parser.add_argument("--expert-base-mix-alpha", type=float, default=None)

    # Runtime.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", type=parse_bool, default=True)
    parser.add_argument("--pin-memory", type=parse_bool, default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--output-prob-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--msa-read-mode", choices=["full", "rows", "block"], default="full")
    parser.add_argument("--msa-sample-strategy", choices=["random", "block", "head"], default="random")
    parser.add_argument("--msa-shuffle-rows-at-getitem", type=parse_bool, default=False)
    parser.add_argument("--msa-cache-gb", type=float, default=4.0)
    parser.add_argument("--msa-max-open-files", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--sampler-seed", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-batches", type=int, default=None)

    # Rare GO definition and selector behavior.
    parser.add_argument(
        "--rare-policy",
        choices=["train_q33", "train_count_le", "ids_file"],
        default="train_q33",
        help=(
            "train_q33 matches make_frequency_bins(counts)['rare']; "
            "train_count_le uses --rare-max-train-count; ids_file uses --rare-go-ids."
        ),
    )
    parser.add_argument("--rare-max-train-count", type=float, default=5.0)
    parser.add_argument("--rare-go-ids", type=Path, default=None)
    parser.add_argument("--include-zero-train-go", type=parse_bool, default=False)
    parser.add_argument("--rare-go-topk", type=int, default=20)
    parser.add_argument(
        "--rare-selector-scope",
        choices=["rare_first", "restricted", "model_topk_filter"],
        default="rare_first",
        help=(
            "rare_first first restricts the candidate universe to rare GO, then "
            "applies the trained selector and returns K rare terms per protein; "
            "restricted is a backward-compatible alias of rare_first; "
            "model_topk_filter filters rare terms from the model's original full-vocabulary top-k."
        ),
    )
    parser.add_argument(
        "--rare-min-backbone-prob",
        type=float,
        default=0.0,
        help="Optional post-selector edge filter; zero keeps the requested top-k.",
    )

    # Weak modelout and pseudo targets.
    parser.add_argument("--modelout-role", choices=list(ROLE_TO_MODE), default="weak")
    parser.add_argument("--weak-external-prob-path", type=Path, default=None)
    parser.add_argument(
        "--modelout-decoder-prob-source",
        choices=["expert", "base", "mix_expert_base"],
        default="expert",
    )
    parser.add_argument("--modelout-decoder-prob-alpha", type=float, default=0.5)
    parser.add_argument(
        "--modelout-topk-source",
        choices=["base_topk", "external_topk", "blend_topk"],
        default="external_topk",
    )
    parser.add_argument("--modelout-external-prob-blend-alpha", type=float, default=1.0)
    parser.add_argument("--pseudo-threshold", type=float, default=0.5)
    parser.add_argument("--save-dense-backbone", type=parse_bool, default=True)
    parser.add_argument("--save-dense-modelout", type=parse_bool, default=True)
    parser.add_argument(
        "--require-anchor-modelout",
        type=parse_bool,
        default=True,
        help="Reject base_residual checkpoints when modelout export is requested.",
    )

    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    parser.add_argument("--skip-checkpoint-sha256", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(value).__name__}")
    return value


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hash_ids(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for item in ids:
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


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
        preview = "\n  ".join(existing[:20])
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace it:\n  " + preview
        )


def parse_role_modes(items: Iterable[str]) -> Dict[str, str]:
    mapping = dict(ROLE_TO_MODE)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--role-mode expects ROLE=MODE, got {item!r}")
        role, mode = [part.strip() for part in item.split("=", 1)]
        if role not in ROLE_TO_MODE or not mode:
            raise ValueError(f"Invalid --role-mode value: {item!r}")
        mapping[role] = mode
    return mapping


def import_module_from_path(name: str, path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(name, str(path))
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not construct import spec for {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def torch_load(torch_module: Any, path: Path) -> Any:
    try:
        return torch_module.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(str(path), map_location="cpu")


def find_model_args(
    checkpoint: Path,
    payload: Any,
    requested: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """Merge args.json and checkpoint model_args.

    The checkpoint payload has higher priority because it is the state-specific
    architecture record saved together with the weights.  The launcher still
    overrides dataset paths and any explicitly audited CLI values afterward.
    """
    json_args: Dict[str, Any] = {}
    checkpoint_args: Dict[str, Any] = {}
    sources: List[str] = []
    if isinstance(payload, Mapping) and isinstance(payload.get("model_args"), Mapping):
        checkpoint_args.update(dict(payload["model_args"]))

    json_path: Path | None = None
    if requested.lower() == "auto":
        for candidate in (
            checkpoint.parent / "args.json",
            checkpoint.parent.parent / "args.json",
        ):
            if candidate.is_file():
                json_path = candidate
                break
    elif requested.strip():
        json_path = Path(requested).expanduser().resolve()
        if not json_path.is_file():
            raise FileNotFoundError(f"TRAIN_ARGS_JSON not found: {json_path}")

    if json_path is not None:
        json_args.update(read_json(json_path))
        sources.append(str(json_path))
    if checkpoint_args:
        sources.append("checkpoint:model_args (higher priority)")
    merged = dict(json_args)
    merged.update(checkpoint_args)
    return merged, sources


def resolve_project_path(project_root: Path, value: str | Path | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def make_eval_model_args(
    *,
    args: argparse.Namespace,
    diagnostic_module: Any,
    payload_model_args: Mapping[str, Any],
) -> Tuple[argparse.Namespace, Dict[str, str]]:
    """Build the same model namespace as the diagnostic evaluator.

    Priority:
        explicit exporter CLI > args.json/checkpoint model_args > evaluator defaults.
    """
    eval_parser = diagnostic_module.build_argparser()
    seed_argv = [
        "--checkpoint",
        str(args.checkpoint),
        "--task",
        args.task,
        "--output_dir",
        str(args.output_dir),
        "--mode",
        ROLE_TO_MODE[args.modelout_role],
    ]
    eval_args = eval_parser.parse_args(seed_argv)
    applied_sources: Dict[str, str] = {}

    for key, value in payload_model_args.items():
        if hasattr(eval_args, key):
            setattr(eval_args, key, value)
            applied_sources[key] = "training_config"

    explicit_map = {
        "model_config": args.model_config,
        "file_address": args.file_address,
        "working_address": args.working_address,
        "num_classes": args.num_classes,
        "top_k": args.top_k,
        "max_len": args.max_len,
        "msa_max_size": args.msa_max_size,
        "permute_dims": args.permute_dims,
        "query_decoder_logit_base_mode": args.query_decoder_logit_base_mode,
        "expert_base_mix_alpha": args.expert_base_mix_alpha,
    }
    for key, value in explicit_map.items():
        if value is not None:
            setattr(eval_args, key, value)
            applied_sources[key] = "exporter_cli"

    # Runtime settings must never be inherited from training.
    eval_args.batch_size = int(args.batch_size)
    eval_args.eval_batch_size = int(args.batch_size)
    eval_args.dataloader_num_workers = int(args.dataloader_num_workers)
    eval_args.prefetch_factor = int(args.prefetch_factor)
    eval_args.persistent_workers = bool(args.persistent_workers)
    eval_args.pin_memory = bool(args.pin_memory)
    eval_args.device = str(args.device)
    eval_args.gpu_ids = args.gpu_ids
    eval_args.no_amp = bool(args.no_amp)
    eval_args.msa_read_mode = args.msa_read_mode
    eval_args.msa_sample_strategy = args.msa_sample_strategy
    eval_args.msa_shuffle_rows_at_getitem = bool(args.msa_shuffle_rows_at_getitem)
    eval_args.msa_cache_gb = float(args.msa_cache_gb)
    eval_args.msa_max_open_files = int(args.msa_max_open_files)
    eval_args.sample_seed = int(args.sample_seed)
    eval_args.sampler_seed = int(args.sampler_seed)
    eval_args.seed = int(args.seed)
    eval_args.distributed = False
    eval_args.rank = 0
    eval_args.world_size = 1
    eval_args.local_rank = 0
    eval_args.torch_compile = False
    eval_args.train_args_json = args.train_args_json

    for key in ("model_config", "file_address", "working_address"):
        value = resolve_project_path(args.project_root, getattr(eval_args, key, None))
        setattr(eval_args, key, value)
    if getattr(eval_args, "permute_dims", None) is None:
        eval_args.permute_dims = [0, 3, 2, 1]

    required = [
        "model_config",
        "file_address",
        "working_address",
        "num_classes",
        "top_k",
        "max_len",
        "query_decoder_logit_base_mode",
    ]
    missing = [key for key in required if getattr(eval_args, key, None) in {None, ""}]
    if missing:
        raise ValueError(
            "Missing model fields after checkpoint/args overlay: "
            f"{missing}. Restore the training args.json/model_args or pass audited overrides."
        )
    return eval_args, applied_sources


def resolve_device(torch_module: Any, value: str) -> Any:
    if value == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    device = torch_module.device(value)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def normalize_protein_ids(proteins: Any, expected: int) -> List[str]:
    if proteins is None:
        raise RuntimeError("Dataset did not return protein IDs; need_proteins=True is required")
    if hasattr(proteins, "detach"):
        values = proteins.detach().cpu().tolist()
    elif isinstance(proteins, np.ndarray):
        values = proteins.tolist()
    elif isinstance(proteins, (list, tuple)):
        values = list(proteins)
    else:
        values = [proteins]
    result = [str(item) for item in values]
    if len(result) != expected:
        raise RuntimeError(f"Protein ID count {len(result)} != batch size {expected}")
    return result


@dataclass(frozen=True)
class ProteinRegistry:
    protein_ids: Tuple[str, ...]
    role_ids: Mapping[str, Tuple[str, ...]]
    role_global_indices: Mapping[str, np.ndarray]

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
            raise ValueError(f"Protein registry missing columns {missing}: {path}")
        for line_no, row in enumerate(reader, start=2):
            try:
                global_idx = int(row["protein_idx"])
                role_row_idx = int(row["role_row_idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid registry integer at {path}:{line_no}") from exc
            protein_id = str(row["protein_id"]).strip()
            role = str(row["role"]).strip()
            if global_idx < 0 or role_row_idx < 0 or not protein_id or not role:
                raise ValueError(f"Invalid protein registry row at {path}:{line_no}: {row}")
            rows.append((global_idx, protein_id, role, role_row_idx))

    rows.sort()
    if [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError("protein_idx must be contiguous and zero-based")
    protein_ids = tuple(row[1] for row in rows)
    if len(set(protein_ids)) != len(protein_ids):
        duplicate = [key for key, count in Counter(protein_ids).items() if count > 1]
        raise ValueError(f"Duplicate protein IDs in registry: {duplicate[:10]}")

    grouped: MutableMapping[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for global_idx, protein_id, role, role_row_idx in rows:
        grouped[role].append((role_row_idx, global_idx, protein_id))
    role_ids: Dict[str, Tuple[str, ...]] = {}
    role_global: Dict[str, np.ndarray] = {}
    for role, values in grouped.items():
        values.sort()
        if [value[0] for value in values] != list(range(len(values))):
            raise ValueError(f"role_row_idx for role={role!r} is not contiguous")
        role_ids[role] = tuple(value[2] for value in values)
        role_global[role] = np.asarray([value[1] for value in values], dtype=np.int64)
    return ProteinRegistry(
        protein_ids=protein_ids,
        role_ids=role_ids,
        role_global_indices=role_global,
    )


@dataclass(frozen=True)
class GORegistry:
    go_idx: np.ndarray
    input_go_ids: Tuple[str, ...]
    canonical_go_ids: Tuple[str, ...]
    names: Tuple[str, ...]

    @property
    def num_terms(self) -> int:
        return int(self.go_idx.size)


def load_go_registry(path: Path) -> GORegistry:
    required = {"go_idx", "input_go_id", "go_id"}
    rows: List[Tuple[int, str, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"GO registry has no header: {path}")
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"GO registry missing columns {missing}: {path}")
        for line_no, row in enumerate(reader, start=2):
            try:
                idx = int(row["go_idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid go_idx at {path}:{line_no}") from exc
            rows.append(
                (
                    idx,
                    str(row["input_go_id"]).strip(),
                    str(row["go_id"]).strip(),
                    str(row.get("name", "")).strip(),
                )
            )
    rows.sort()
    if [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError("go_idx must be contiguous and zero-based")
    return GORegistry(
        go_idx=np.arange(len(rows), dtype=np.int32),
        input_go_ids=tuple(row[1] for row in rows),
        canonical_go_ids=tuple(row[2] for row in rows),
        names=tuple(row[3] for row in rows),
    )


def label_counts_from_annotation(annotation: Any, num_classes: int) -> Tuple[np.ndarray, int]:
    """Count training labels without assuming dense or sparse annotation storage."""
    counts = np.zeros(num_classes, dtype=np.float64)
    if annotation is None:
        return counts, 0
    if hasattr(annotation, "detach"):
        annotation = annotation.detach().cpu().numpy()
    if isinstance(annotation, np.ndarray):
        if annotation.ndim == 2 and annotation.shape[1] == num_classes:
            return annotation.astype(np.float64, copy=False).sum(axis=0), int(annotation.shape[0])
        if annotation.ndim == 1 and annotation.shape[0] == num_classes:
            return annotation.astype(np.float64, copy=False), 1

    try:
        values = list(annotation)
    except TypeError:
        return counts, 0
    for item in values:
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            if item.ndim == 1 and item.shape[0] == num_classes:
                counts += item.astype(np.float64, copy=False)
            else:
                indices = item.astype(np.int64, copy=False).reshape(-1)
                indices = indices[(indices >= 0) & (indices < num_classes)]
                counts[indices] += 1.0
        elif isinstance(item, (list, tuple, set)):
            item_list = list(item)
            appears_dense = (
                len(item_list) == num_classes
                and all(
                    isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_))
                    for value in item_list[: min(10, len(item_list))]
                )
            )
            if appears_dense:
                counts += np.asarray(item_list, dtype=np.float64)
            else:
                for value in item_list:
                    try:
                        idx = int(value)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < num_classes:
                        counts[idx] += 1.0
    return counts, len(values)


def load_train_label_counts(
    metadata_file: Path,
    metadata_task: str,
    num_classes: int,
) -> Tuple[np.ndarray, int, str]:
    import pickle

    with metadata_file.open("rb") as handle:
        metadata = pickle.load(handle)
    task_block = metadata.get("train", {}).get(metadata_task)
    if not isinstance(task_block, Mapping):
        raise KeyError(f"Metadata missing train/{metadata_task}: {metadata_file}")
    for key in ("annotations", "labels", "prop_annotations"):
        if key in task_block:
            counts, n = label_counts_from_annotation(task_block[key], num_classes)
            if n > 0:
                return counts, n, key
    raise KeyError(
        f"Metadata train/{metadata_task} has no usable annotations/labels/prop_annotations"
    )


def ids_file_mask(path: Path, registry: GORegistry) -> np.ndarray:
    requested: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if token and not token.startswith("#"):
                requested.add(token)
    if not requested:
        raise ValueError(f"Rare GO ID file is empty: {path}")

    mask = np.zeros(registry.num_terms, dtype=bool)
    matched: set[str] = set()
    for idx, (input_id, canonical_id) in enumerate(
        zip(registry.input_go_ids, registry.canonical_go_ids)
    ):
        candidates = {str(idx), input_id, canonical_id}
        if requested.intersection(candidates):
            mask[idx] = True
            matched.update(requested.intersection(candidates))
    missing = sorted(requested - matched)
    if missing:
        raise KeyError(f"{len(missing)} rare GO identifiers were not found: {missing[:20]}")
    return mask


def make_rare_mask(
    counts: np.ndarray,
    *,
    policy: str,
    max_count: float,
    include_zero: bool,
    registry: GORegistry,
    ids_path: Path | None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (registry.num_terms,):
        raise ValueError(
            f"Training count shape {counts.shape} != GO vocabulary {(registry.num_terms,)}"
        )
    positive = counts > 0
    metadata: Dict[str, Any] = {"policy": policy, "include_zero_train_go": include_zero}
    if policy == "train_q33":
        if not positive.any():
            raise ValueError("Cannot define q33 rare GO: no positive training counts")
        q33 = float(np.quantile(counts[positive], 0.33))
        mask = positive & (counts <= q33)
        metadata["positive_count_q33"] = q33
    elif policy == "train_count_le":
        if max_count < 0:
            raise ValueError("--rare-max-train-count must be non-negative")
        mask = positive & (counts <= float(max_count))
        metadata["max_train_count"] = float(max_count)
    elif policy == "ids_file":
        if ids_path is None or not ids_path.is_file():
            raise FileNotFoundError("--rare-policy ids_file requires --rare-go-ids")
        mask = ids_file_mask(ids_path, registry)
        metadata["ids_file"] = str(ids_path)
        metadata["ids_file_sha256"] = sha256_file(ids_path)
    else:
        raise ValueError(f"Unknown rare policy: {policy}")
    if include_zero:
        mask = mask | (counts == 0)
    if not mask.any():
        raise ValueError("Rare GO definition produced an empty set")
    metadata.update(
        {
            "num_rare_terms": int(mask.sum()),
            "num_zero_train_terms_included": int(((counts == 0) & mask).sum()),
            "rare_count_min": float(counts[mask].min()),
            "rare_count_max": float(counts[mask].max()),
        }
    )
    return mask, metadata


def write_rare_registry(
    path: Path,
    registry: GORegistry,
    counts: np.ndarray,
    rare_mask: np.ndarray,
) -> None:
    lines = ["go_idx\tinput_go_id\tgo_id\tname\ttrain_count"]
    for idx in np.flatnonzero(rare_mask):
        safe_name = registry.names[idx].replace("\t", " ").replace("\n", " ")
        lines.append(
            f"{idx}\t{registry.input_go_ids[idx]}\t{registry.canonical_go_ids[idx]}"
            f"\t{safe_name}\t{float(counts[idx]):g}"
        )
    atomic_write_text(path, "\n".join(lines) + "\n")


class RawArrayWriter:
    """Append fixed-width rows to raw storage, then publish a standard .npy."""

    def __init__(self, raw_path: Path, dtype: np.dtype | str, columns: int = 1):
        self.raw_path = raw_path
        self.dtype = np.dtype(dtype)
        self.columns = int(columns)
        if self.columns <= 0:
            raise ValueError("columns must be positive")
        self.handle = raw_path.open("wb")
        self.rows = 0

    def append(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=self.dtype)
        if self.columns == 1:
            array = array.reshape(-1)
            rows = int(array.size)
        else:
            if array.ndim != 2 or array.shape[1] != self.columns:
                raise ValueError(
                    f"Expected [N,{self.columns}] append, got shape={array.shape}"
                )
            rows = int(array.shape[0])
        array.tofile(self.handle)
        self.rows += rows

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()

    def finalize(self, output_path: Path, *, transpose_two_columns: bool = False) -> Path:
        self.close()
        if self.rows == 0:
            shape = (2, 0) if transpose_two_columns else (
                (0,) if self.columns == 1 else (0, self.columns)
            )
            np.save(output_path, np.empty(shape, dtype=self.dtype))
            self.raw_path.unlink(missing_ok=True)
            return output_path

        raw_shape = (self.rows,) if self.columns == 1 else (self.rows, self.columns)
        raw = np.memmap(self.raw_path, mode="r", dtype=self.dtype, shape=raw_shape)
        if transpose_two_columns:
            if self.columns != 2:
                raise ValueError("transpose_two_columns requires columns=2")
            output = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=self.dtype, shape=(2, self.rows)
            )
            for start in range(0, self.rows, 1_000_000):
                end = min(start + 1_000_000, self.rows)
                output[:, start:end] = raw[start:end].T
        else:
            output = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=self.dtype, shape=raw_shape
            )
            for start in range(0, self.rows, 1_000_000):
                end = min(start + 1_000_000, self.rows)
                output[start:end] = raw[start:end]
        output.flush()
        del output
        del raw
        self.raw_path.unlink(missing_ok=True)
        return output_path


def create_dense_memmap(path: Path, shape: Tuple[int, int], dtype: np.dtype) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def shrink_dense_rows(path: Path, target_rows: int) -> None:
    """Shrink a diagnostic memmap when shard-local short batches reduce rows."""
    source = np.load(path, mmap_mode="r")
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D dense prediction array: {path}")
    if not 0 <= int(target_rows) <= int(source.shape[0]):
        raise ValueError(
            f"Cannot shrink {path} from {source.shape[0]} to {target_rows} rows"
        )
    if int(target_rows) == int(source.shape[0]):
        del source
        return
    temporary = path.with_name(path.name + ".shrunk")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=source.dtype,
        shape=(int(target_rows), int(source.shape[1])),
    )
    for start in range(0, int(target_rows), 4096):
        end = min(start + 4096, int(target_rows))
        output[start:end] = source[start:end]
    output.flush()
    del output
    del source
    os.replace(temporary, path)


def load_checkpoint_strict(
    model: Any,
    payload: Any,
    *,
    allow_partial: bool,
) -> Dict[str, Any]:
    """Load a V3 DETR payload and reject random/unloaded selector weights."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}")
    information: Dict[str, Any] = {}
    if isinstance(payload.get("backbone"), Mapping) and isinstance(
        payload.get("query_decoder"), Mapping
    ):
        missing_backbone, unexpected_backbone = model.backbone.load_state_dict(
            payload["backbone"], strict=False
        )
        missing_query, unexpected_query = model.query_decoder.load_state_dict(
            payload["query_decoder"], strict=False
        )
        information = {
            "checkpoint_type": payload.get("checkpoint_type", "detr_payload"),
            "detr_version": payload.get("detr_version"),
            "missing_backbone": list(missing_backbone),
            "unexpected_backbone": list(unexpected_backbone),
            "missing_query_decoder": list(missing_query),
            "unexpected_query_decoder": list(unexpected_query),
        }
    elif any(
        str(key).startswith(("backbone.", "query_decoder.")) for key in payload
    ):
        missing, unexpected = model.load_state_dict(payload, strict=False)
        information = {
            "checkpoint_type": "full_model_state_dict",
            "missing": list(missing),
            "unexpected": list(unexpected),
        }
    else:
        raise RuntimeError(
            "Checkpoint has no query_decoder state. A backbone-only checkpoint cannot "
            "produce trained selector rankings or modelout."
        )

    mismatch_count = sum(
        len(value)
        for key, value in information.items()
        if key.startswith(("missing", "unexpected")) and isinstance(value, list)
    )
    if mismatch_count and not allow_partial:
        raise RuntimeError(
            "Checkpoint did not load exactly; refusing to export graph supervision. "
            f"Details={information}. Use --allow-partial-checkpoint only after auditing."
        )
    return information


def rare_first_selector_topk(
    *,
    torch_module: Any,
    weak_module: Any,
    model: Any,
    base_logits: Any,
    h: Any,
    allowed_go_idx: Any,
    requested_k: int,
) -> Tuple[Any, Any]:
    """Select rare GO first, then run the trained selector inside that universe.

    This preserves the selector architecture used in training:

        rare vocabulary
          -> backbone/static top-M prefilter
          -> learnable selector reranking
          -> graph edge top-k

    ``requested_k`` is the graph degree and is intentionally independent of
    the decoder's training-time ``selector.topk``.  No expert probability or
    label hint is used in this backbone-only graph branch.
    """
    selector = model.query_decoder.selector
    rare_count = int(allowed_go_idx.numel())
    k = min(int(requested_k), rare_count)
    if k <= 0:
        raise ValueError("No rare GO candidates are available")

    _, pooled_feature = weak_module.build_backbone_memory(
        h,
        mode=model.query_decoder.memory_mode,
        memory_grid_h=model.query_decoder.memory_grid_h,
        memory_grid_w=model.query_decoder.memory_grid_w,
    )
    base_for_selector = (
        base_logits.detach()
        if bool(model.query_decoder.selector_detach_base_logits)
        else base_logits
    )
    # For the backbone-only selector path, training canonicalizes
    # ``source="base_topk"`` to ``source="base"`` and therefore defines the
    # static score exactly as sigmoid(base_logits).  Compute that expression
    # directly instead of calling the selector's private _make_static_score()
    # API.  The private method accepts ``alpha`` while the public forward()
    # accepts ``external_prob_blend_alpha``; depending on the training-code
    # revision, forwarding the public keyword to the private method raises a
    # TypeError before the first batch can be exported.
    static_full = torch_module.sigmoid(base_for_selector.float())
    static_rare = static_full.index_select(1, allowed_go_idx)

    if not bool(selector.use_learnable_selector):
        top_score, local_idx = torch_module.topk(static_rare, k=k, dim=1)
        top_idx = allowed_go_idx[local_idx]
        return top_idx, top_score

    prefilter_m = min(max(int(selector.prefilter_topm), k), rare_count)
    prefilter_score, prefilter_local = torch_module.topk(
        static_rare, k=prefilter_m, dim=1
    )
    prefilter_idx = allowed_go_idx[prefilter_local]

    base_logit_selected = torch_module.gather(
        base_for_selector.float(), dim=1, index=prefilter_idx
    )
    base_prob_selected = torch_module.sigmoid(base_logit_selected)
    external_prob_selected = torch_module.zeros_like(base_prob_selected)
    external_logit_selected = torch_module.zeros_like(base_prob_selected)
    has_external_selected = torch_module.zeros_like(base_prob_selected)
    static_prob_selected = prefilter_score.float().clamp(0.0, 1.0)

    features = [
        base_logit_selected,
        base_prob_selected,
        external_prob_selected,
        external_logit_selected,
        static_prob_selected,
        has_external_selected,
    ]
    if bool(selector.use_protein_term_affinity):
        affinity = selector._protein_term_affinity(
            prefilter_idx,
            pooled_feature,
            model.classifier.weight,
        )
        features.append(affinity)

    feature_tensor = torch_module.stack(features, dim=-1)
    residual = selector.score_mlp(feature_tensor).squeeze(-1)
    if selector.term_bias is not None:
        residual = residual + selector.term_bias(prefilter_idx).squeeze(-1)
    static_logit = weak_module.prob_to_logit(static_prob_selected)
    selector_logits = (
        static_logit
        + float(selector.selector_logit_residual_scale) * torch_module.tanh(residual)
    )
    rerank_score = torch_module.sigmoid(selector_logits)
    top_score, top_local = torch_module.topk(rerank_score, k=k, dim=1)
    top_idx = torch_module.gather(prefilter_idx, dim=1, index=top_local)
    return top_idx, top_score


restricted_selector_topk = rare_first_selector_topk


def original_selector_then_filter(
    *,
    torch_module: Any,
    weak_module: Any,
    model: Any,
    base_logits: Any,
    h: Any,
    rare_mask_tensor: Any,
    requested_k: int,
) -> Tuple[Any, Any]:
    """Run the model's original full-vocabulary selector, then retain rare hits.

    The returned matrices are padded with ``go_idx=-1`` and ``score=NaN`` when
    fewer than ``requested_k`` rare terms occur in the model's original top-k.
    """
    selector = model.query_decoder.selector
    _, pooled_feature = weak_module.build_backbone_memory(
        h,
        mode=model.query_decoder.memory_mode,
        memory_grid_h=model.query_decoder.memory_grid_h,
        memory_grid_w=model.query_decoder.memory_grid_w,
    )
    base_for_selector = (
        base_logits.detach()
        if bool(model.query_decoder.selector_detach_base_logits)
        else base_logits
    )
    output = selector(
        base_logits=base_for_selector,
        external_prob=None,
        has_external_prob=None,
        topk_source="base_topk",
        external_prob_blend_alpha=0.0,
        y_hint=None,
        pooled_feature=pooled_feature,
        classifier_weight=model.classifier.weight,
    )
    source_idx = output["topk_idx"]
    source_score = output["topk_score"]
    batch_size = int(source_idx.shape[0])
    result_idx = torch_module.full(
        (batch_size, int(requested_k)),
        -1,
        dtype=source_idx.dtype,
        device=source_idx.device,
    )
    result_score = torch_module.full(
        (batch_size, int(requested_k)),
        float("nan"),
        dtype=source_score.dtype,
        device=source_score.device,
    )
    rare_hit = rare_mask_tensor[source_idx]
    for row in range(batch_size):
        positions = torch_module.nonzero(rare_hit[row], as_tuple=False).flatten()
        positions = positions[: int(requested_k)]
        count = int(positions.numel())
        if count:
            result_idx[row, :count] = source_idx[row, positions]
            result_score[row, :count] = source_score[row, positions]
    return result_idx, result_score


def choose_modelout_decoder_prob(
    torch_module: Any,
    *,
    source: str,
    base_prob: Any,
    expert_prob: Any,
    alpha: float,
) -> Any:
    if source == "expert":
        if expert_prob is None:
            raise ValueError("modelout decoder source=expert requires external probability")
        return expert_prob.float().clamp(0.0, 1.0)
    if source == "base":
        return base_prob.float().clamp(0.0, 1.0)
    if source == "mix_expert_base":
        if expert_prob is None:
            raise ValueError("modelout decoder source=mix_expert_base requires expert probability")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("--modelout-decoder-prob-alpha must be in [0,1]")
        return (
            float(alpha) * expert_prob.float()
            + (1.0 - float(alpha)) * base_prob.float()
        ).clamp(0.0, 1.0)
    raise ValueError(f"Unknown modelout decoder probability source: {source}")


def expected_outputs(args: argparse.Namespace) -> List[Path]:
    outputs = [
        args.output_dir / "rare_go_indices.i32.npy",
        args.output_dir / "rare_go_registry.tsv",
        args.output_dir / "train_go_label_counts.f64.npy",
        args.output_dir / "pg_backbone_rare_edge_index.i32.npy",
        args.output_dir / "pg_backbone_rare_edge_attr.f32.npy",
        args.output_dir / "weak_graph_predictions_manifest.json",
    ]
    if args.save_dense_backbone:
        for role in args.roles:
            outputs.append(args.output_dir / f"backbone_{role}_prob.{args.prob_suffix}.npy")
    if args.modelout_role in args.roles:
        outputs.extend(
            [
                args.output_dir / f"{args.modelout_role}_pseudo_label_indptr.i64.npy",
                args.output_dir / f"{args.modelout_role}_pseudo_label_indices.i32.npy",
                args.output_dir / f"{args.modelout_role}_pseudo_prob.{args.prob_suffix}.npy",
                args.output_dir / "pg_modelout_pseudo_edge_index.i32.npy",
                args.output_dir / "pg_modelout_pseudo_edge_attr.f32.npy",
            ]
        )
        if args.save_dense_modelout:
            outputs.append(
                args.output_dir
                / f"modelout_{args.modelout_role}_prob.{args.prob_suffix}.npy"
            )
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.rare_selector_scope == "restricted":
        print(
            "[Compatibility] --rare-selector-scope=restricted is an alias of "
            "rare_first; manifest records the canonical rare_first mode.",
            flush=True,
        )
        args.rare_selector_scope = "rare_first"
    args.project_root = args.project_root.expanduser().resolve()
    args.diagnostic_eval_script = args.diagnostic_eval_script.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.feature_dir = args.feature_dir.expanduser().resolve()
    args.go_registry = args.go_registry.expanduser().resolve()
    if args.weak_external_prob_path is not None:
        args.weak_external_prob_path = args.weak_external_prob_path.expanduser().resolve()
    if args.rare_go_ids is not None:
        args.rare_go_ids = args.rare_go_ids.expanduser().resolve()
    args.num_classes = args.num_classes or TASK_NUM_CLASSES[args.task]
    output_dtype = np.float16 if args.output_prob_dtype == "float16" else np.float32
    args.prob_suffix = "f16" if output_dtype == np.float16 else "f32"

    if len(set(args.roles)) != len(args.roles):
        raise ValueError(f"Duplicate roles: {args.roles}")
    invalid_roles = sorted(set(args.roles) - set(ROLE_TO_MODE))
    if invalid_roles:
        raise ValueError(f"Unknown roles: {invalid_roles}")
    if args.modelout_role not in args.roles:
        print(
            f"[Warning] modelout role {args.modelout_role!r} is not in roles={args.roles}; "
            "modelout/pseudo outputs will not be generated.",
            flush=True,
        )
    if args.rare_go_topk <= 0:
        raise ValueError("--rare-go-topk must be positive")
    if not 0.0 <= args.rare_min_backbone_prob <= 1.0:
        raise ValueError("--rare-min-backbone-prob must be in [0,1]")
    if not 0.0 <= args.pseudo_threshold <= 1.0:
        raise ValueError("--pseudo-threshold must be in [0,1]")
    if args.batch_size <= 0 or args.dataloader_num_workers < 0:
        raise ValueError("Invalid batch/worker configuration")
    if args.max_batches is not None and int(args.max_batches) <= 0:
        raise ValueError("--max-batches must be positive when supplied")

    required_paths = [
        args.diagnostic_eval_script,
        args.checkpoint,
        args.feature_dir / "protein_registry.csv",
        args.go_registry,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.modelout_role in args.roles:
        if args.weak_external_prob_path is None or not args.weak_external_prob_path.is_file():
            raise FileNotFoundError(
                "Weak modelout export requires --weak-external-prob-path pointing to "
                "the exp_train expert probability array"
            )

    os.chdir(args.project_root)
    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root / "msa_models"))
    import torch
    from torch.amp import autocast

    diagnostic_module = import_module_from_path(
        "_latence_diagnostic_eval_for_graph_export",
        args.diagnostic_eval_script,
    )
    weak_module = sys.modules.get("experiments.weak_exp_train_detr")
    exp_module = sys.modules.get("experiments.exp_train")
    pseudo_module = sys.modules.get("experiments.msaprob")
    if weak_module is None or exp_module is None or pseudo_module is None:
        raise ImportError(
            "The diagnostic evaluator did not import the expected LATENCE modules"
        )

    payload = torch_load(torch, args.checkpoint)
    training_args, training_sources = find_model_args(
        args.checkpoint, payload, args.train_args_json
    )
    model_args, model_arg_sources = make_eval_model_args(
        args=args,
        diagnostic_module=diagnostic_module,
        payload_model_args=training_args,
    )
    if int(model_args.num_classes) != int(args.num_classes):
        raise ValueError(
            f"Checkpoint/model num_classes={model_args.num_classes} != requested {args.num_classes}"
        )

    device = resolve_device(torch, args.device)
    model_args.device = str(device)
    if device.type == "cpu":
        model_args.gpu_ids = ""
    elif model_args.gpu_ids is None:
        model_args.gpu_ids = str(device.index if device.index is not None else 0)

    for path_name in ("model_config", "file_address", "working_address"):
        path = Path(str(getattr(model_args, path_name)))
        if not path.is_file():
            raise FileNotFoundError(f"{path_name} not found: {path}")

    if (
        args.modelout_role in args.roles
        and args.require_anchor_modelout
        and str(model_args.query_decoder_logit_base_mode) == "base_residual"
    ):
        raise RuntimeError(
            "The resolved checkpoint configuration uses query_decoder_logit_base_mode="
            "base_residual, which is not the requested anchor/gated modelout. Restore the "
            "training args/model_args or pass an audited --query-decoder-logit-base-mode."
        )

    exp_module.set_seed(int(args.seed))
    opt = weak_module.build_weak_opt_from_config(model_args)
    opt.mode = "eval"
    opt.shuffle = False
    model = weak_module.WeakMSAGOWithDETRDecoder(opt, model_args)
    checkpoint_info = load_checkpoint_strict(
        model,
        payload,
        allow_partial=bool(args.allow_partial_checkpoint),
    )
    model = model.to(device)
    model.eval()

    protein_registry_path = args.feature_dir / "protein_registry.csv"
    protein_registry = load_protein_registry(protein_registry_path)
    for role in args.roles:
        if role not in protein_registry.role_ids:
            raise KeyError(f"protein_registry.csv has no role={role!r}")
    if protein_registry.num_proteins > np.iinfo(np.int32).max:
        raise OverflowError("Global protein count exceeds int32 edge-index capacity")
    go_registry = load_go_registry(args.go_registry)
    if go_registry.num_terms != int(args.num_classes):
        raise ValueError(
            f"GO registry terms={go_registry.num_terms} != num_classes={args.num_classes}"
        )

    metadata_task = TASK_TO_METADATA_KEY[args.task]
    train_counts, train_count_n, train_count_key = load_train_label_counts(
        Path(model_args.file_address),
        metadata_task,
        int(args.num_classes),
    )
    rare_mask, rare_definition = make_rare_mask(
        train_counts,
        policy=args.rare_policy,
        max_count=float(args.rare_max_train_count),
        include_zero=bool(args.include_zero_train_go),
        registry=go_registry,
        ids_path=args.rare_go_ids,
    )
    rare_indices = np.flatnonzero(rare_mask).astype(np.int32)
    if args.rare_go_topk > len(rare_indices):
        print(
            f"[Warning] RARE_GO_TOPK={args.rare_go_topk} exceeds rare terms="
            f"{len(rare_indices)}; using all rare terms.",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_outputs = expected_outputs(args)
    ensure_publishable(final_outputs, bool(args.overwrite))
    stage_dir = Path(
        tempfile.mkdtemp(prefix=".weak_graph_predictions_", dir=args.output_dir)
    )

    print(
        f"[Implementation] {EXPORTER_ID} version={EXPORTER_VERSION} "
        f"file={Path(__file__).name}",
        flush=True,
    )
    print(
        f"[Model] anchor_mode={model_args.query_decoder_logit_base_mode} "
        f"expert_base_mix_alpha={model_args.expert_base_mix_alpha} "
        f"selector_topk={model.query_decoder.selector.topk} "
        f"selector_prefilter_topm={model.query_decoder.selector.prefilter_topm}",
        flush=True,
    )
    print(
        f"[Rare GO] policy={args.rare_policy} terms={len(rare_indices)} "
        f"edge_topk={args.rare_go_topk} scope={args.rare_selector_scope}",
        flush=True,
    )
    if args.rare_selector_scope == "rare_first":
        effective_prefilter_topm = min(
            max(
                int(model.query_decoder.selector.prefilter_topm),
                min(int(args.rare_go_topk), len(rare_indices)),
            ),
            len(rare_indices),
        )
        print(
            "[Rare selector pipeline] "
            f"rare({len(rare_indices)}) -> "
            f"static_prefilter({effective_prefilter_topm}) -> "
            f"learned_rerank -> topk({min(int(args.rare_go_topk), len(rare_indices))})",
            flush=True,
        )
    else:
        effective_prefilter_topm = min(
            max(
                int(model.query_decoder.selector.prefilter_topm),
                int(model.query_decoder.selector.topk),
            ),
            int(args.num_classes),
        )
    dense_elements = (
        sum(len(protein_registry.role_ids[role]) for role in args.roles)
        * int(args.num_classes)
        * int(bool(args.save_dense_backbone))
    )
    if args.modelout_role in args.roles and args.save_dense_modelout:
        dense_elements += (
            len(protein_registry.role_ids[args.modelout_role])
            * int(args.num_classes)
        )
    print(
        f"[Dense output estimate] "
        f"{dense_elements * np.dtype(output_dtype).itemsize / 1024 ** 3:.2f} GiB "
        "(excluding sparse edges and temporary finalization space)",
        flush=True,
    )

    rare_edge_index_writer = RawArrayWriter(
        stage_dir / ".pg_backbone_rare_edge_index.raw", np.int32, columns=2
    )
    rare_edge_attr_writer = RawArrayWriter(
        stage_dir / ".pg_backbone_rare_edge_attr.raw", np.float32, columns=3
    )
    pseudo_edge_index_writer: RawArrayWriter | None = None
    pseudo_edge_attr_writer: RawArrayWriter | None = None
    pseudo_indices_writer: RawArrayWriter | None = None
    pseudo_probs_writer: RawArrayWriter | None = None
    if args.modelout_role in args.roles:
        pseudo_edge_index_writer = RawArrayWriter(
            stage_dir / ".pg_modelout_pseudo_edge_index.raw", np.int32, columns=2
        )
        pseudo_edge_attr_writer = RawArrayWriter(
            stage_dir / ".pg_modelout_pseudo_edge_attr.raw", np.float32, columns=1
        )
        pseudo_indices_writer = RawArrayWriter(
            stage_dir / f".{args.modelout_role}_pseudo_indices.raw",
            np.int32,
            columns=1,
        )
        pseudo_probs_writer = RawArrayWriter(
            stage_dir / f".{args.modelout_role}_pseudo_probs.raw",
            output_dtype,
            columns=1,
        )

    role_modes = parse_role_modes(args.role_mode)
    role_records: List[Dict[str, Any]] = []
    pseudo_indptr: np.ndarray | None = None
    pseudo_row_offset = 0
    rare_allowed_tensor = torch.from_numpy(rare_indices.astype(np.int64)).to(device)
    rare_mask_tensor = torch.from_numpy(rare_mask).to(device=device, dtype=torch.bool)
    amp_enabled = device.type == "cuda" and not bool(args.no_amp)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    permute_dims = tuple(int(value) for value in model_args.permute_dims)
    started = time.time()

    try:
        np.save(stage_dir / "rare_go_indices.i32.npy", rare_indices)
        np.save(
            stage_dir / "train_go_label_counts.f64.npy",
            train_counts.astype(np.float64, copy=False),
        )
        write_rare_registry(
            stage_dir / "rare_go_registry.tsv",
            go_registry,
            train_counts,
            rare_mask,
        )

        for role in args.roles:
            role_started = time.time()
            mode = role_modes[role]
            base_dataset = exp_module.build_msa_dataset(
                opt,
                mode=mode,
                task=metadata_task,
                need_proteins=True,
            )
            if len(base_dataset) <= 0:
                raise RuntimeError(f"role={role!r} / mode={mode!r} has no proteins")
            expected_role_ids = protein_registry.role_ids[role]
            global_indices = protein_registry.role_global_indices[role]
            if len(base_dataset) != len(expected_role_ids):
                raise ValueError(
                    f"Dataset/registry count mismatch for role={role}: "
                    f"{len(base_dataset)} != {len(expected_role_ids)}"
                )

            use_modelout = role == args.modelout_role
            if use_modelout:
                dataset = pseudo_module.PseudoProbDataset(
                    base_dataset=base_dataset,
                    metadata_file=model_args.file_address,
                    mode=mode,
                    task=metadata_task,
                    prob_path=args.weak_external_prob_path,
                    num_classes=int(args.num_classes),
                )
            else:
                dataset = base_dataset

            loader = exp_module.make_loader(
                dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.dataloader_num_workers),
                pin_memory=bool(args.pin_memory),
                drop_last=False,
                rank=0,
                world_size=1,
                seed=int(args.sampler_seed),
                prefetch_factor=int(args.prefetch_factor),
                persistent_workers=bool(args.persistent_workers),
            )

            n_rows = len(dataset)
            if args.max_batches is not None:
                n_rows = min(n_rows, int(args.max_batches) * int(args.batch_size))
            dense_backbone: np.memmap | None = None
            dense_modelout: np.memmap | None = None
            if args.save_dense_backbone:
                dense_backbone = create_dense_memmap(
                    stage_dir / f"backbone_{role}_prob.{args.prob_suffix}.npy",
                    (n_rows, int(args.num_classes)),
                    output_dtype,
                )
            if use_modelout and args.save_dense_modelout:
                dense_modelout = create_dense_memmap(
                    stage_dir
                    / f"modelout_{args.modelout_role}_prob.{args.prob_suffix}.npy",
                    (n_rows, int(args.num_classes)),
                    output_dtype,
                )
            if use_modelout:
                pseudo_indptr = np.zeros(n_rows + 1, dtype=np.int64)
                pseudo_row_offset = 0

            role_offset = 0
            role_rare_edges_before = rare_edge_index_writer.rows
            role_pseudo_edges_before = (
                0 if pseudo_edge_index_writer is None else pseudo_edge_index_writer.rows
            )
            selector_hit_counts: List[int] = []
            backbone_probability_sum = 0.0
            modelout_probability_sum = 0.0

            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    if (
                        args.max_batches is not None
                        and batch_idx >= int(args.max_batches)
                    ):
                        break

                    if use_modelout:
                        proteins, x, _, expert_prob = exp_module.unpack_pseudo_batch(batch)
                    else:
                        proteins, x, _ = exp_module.unpack_batch(batch)
                        expert_prob = None
                    batch_size = int(x.shape[0])
                    batch_ids = normalize_protein_ids(proteins, batch_size)
                    end = role_offset + batch_size
                    expected_batch_ids = list(expected_role_ids[role_offset:end])
                    if batch_ids != expected_batch_ids:
                        first_bad = next(
                            (
                                idx
                                for idx, (observed, expected) in enumerate(
                                    zip(batch_ids, expected_batch_ids)
                                )
                                if observed != expected
                            ),
                            0,
                        )
                        raise RuntimeError(
                            f"Protein order mismatch for role={role}, row={role_offset + first_bad}: "
                            f"loader={batch_ids[first_bad]!r}, registry={expected_batch_ids[first_bad]!r}. "
                            "Regenerate features/registry with the same dataset and deterministic loader."
                        )
                    exp_module.set_model_proteins(model, proteins)
                    x = x.to(device, non_blocking=True).long()
                    if expert_prob is not None:
                        expert_prob = (
                            expert_prob.to(device, non_blocking=True)
                            .float()
                            .clamp(0.0, 1.0)
                        )

                    with autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=amp_enabled,
                    ):
                        base_logits, h = model.backbone(
                            x,
                            permute_dims=permute_dims,
                            return_embedding=True,
                        )
                        base_prob = torch.sigmoid(base_logits.float())

                        if args.rare_selector_scope == "rare_first":
                            selected_idx, selector_score = rare_first_selector_topk(
                                torch_module=torch,
                                weak_module=weak_module,
                                model=model,
                                base_logits=base_logits,
                                h=h,
                                allowed_go_idx=rare_allowed_tensor,
                                requested_k=int(args.rare_go_topk),
                            )
                        else:
                            selected_idx, selector_score = original_selector_then_filter(
                                torch_module=torch,
                                weak_module=weak_module,
                                model=model,
                                base_logits=base_logits,
                                h=h,
                                rare_mask_tensor=rare_mask_tensor,
                                requested_k=int(args.rare_go_topk),
                            )

                        safe_idx = selected_idx.clamp_min(0)
                        selected_backbone_prob = torch.gather(
                            base_prob, dim=1, index=safe_idx
                        )

                        modelout_prob = None
                        if use_modelout:
                            decoder_prob = choose_modelout_decoder_prob(
                                torch,
                                source=args.modelout_decoder_prob_source,
                                base_prob=base_prob,
                                expert_prob=expert_prob,
                                alpha=float(args.modelout_decoder_prob_alpha),
                            )
                            has_prob = torch.ones(
                                batch_size, dtype=torch.bool, device=device
                            )
                            qout = model.query_decoder(
                                h=h,
                                base_logits=base_logits,
                                classifier_weight=model.classifier.weight,
                                external_prob=decoder_prob,
                                has_external_prob=has_prob,
                                topk_source=args.modelout_topk_source,
                                external_prob_blend_alpha=float(
                                    args.modelout_external_prob_blend_alpha
                                ),
                                y_hint=None,
                            )
                            modelout_prob = torch.sigmoid(qout["logits"].float())

                    base_np = base_prob.detach().cpu().numpy()
                    if not np.isfinite(base_np).all():
                        raise FloatingPointError(
                            f"Non-finite backbone probabilities in role={role}, batch={batch_idx}"
                        )
                    if dense_backbone is not None:
                        dense_backbone[role_offset:end] = base_np.astype(
                            output_dtype, copy=False
                        )
                    backbone_probability_sum += float(base_np.sum(dtype=np.float64))

                    selected_idx_np = selected_idx.detach().cpu().numpy().astype(
                        np.int32, copy=False
                    )
                    selector_score_np = (
                        selector_score.detach().float().cpu().numpy().astype(np.float32)
                    )
                    selected_prob_np = (
                        selected_backbone_prob.detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    if not np.isfinite(selector_score_np[selected_idx_np >= 0]).all():
                        raise FloatingPointError(
                            f"Non-finite selector scores in role={role}, batch={batch_idx}"
                        )
                    if not np.isfinite(selected_prob_np).all():
                        raise FloatingPointError(
                            f"Non-finite selected backbone probabilities in "
                            f"role={role}, batch={batch_idx}"
                        )
                    valid = selected_idx_np >= 0
                    valid &= np.isfinite(selector_score_np)
                    valid &= (
                        selected_prob_np >= float(args.rare_min_backbone_prob)
                    )
                    if np.any(valid & ~rare_mask[selected_idx_np.clip(min=0)]):
                        raise RuntimeError("Selector emitted a non-rare GO index")

                    batch_global = global_indices[role_offset:end].astype(
                        np.int32, copy=False
                    )
                    repeated_protein = np.broadcast_to(
                        batch_global[:, None], selected_idx_np.shape
                    )
                    rank = np.broadcast_to(
                        1.0
                        / (
                            np.arange(selected_idx_np.shape[1], dtype=np.float32)
                            + 1.0
                        )[None, :],
                        selected_idx_np.shape,
                    )
                    edge_index_rows = np.stack(
                        [repeated_protein[valid], selected_idx_np[valid]], axis=1
                    )
                    edge_attr_rows = np.stack(
                        [
                            selected_prob_np[valid],
                            selector_score_np[valid],
                            rank[valid],
                        ],
                        axis=1,
                    )
                    rare_edge_index_writer.append(edge_index_rows)
                    rare_edge_attr_writer.append(edge_attr_rows)
                    selector_hit_counts.extend(valid.sum(axis=1).astype(int).tolist())

                    if use_modelout:
                        assert modelout_prob is not None
                        assert pseudo_indptr is not None
                        assert pseudo_indices_writer is not None
                        assert pseudo_probs_writer is not None
                        assert pseudo_edge_index_writer is not None
                        assert pseudo_edge_attr_writer is not None

                        modelout_np = modelout_prob.detach().cpu().numpy()
                        if not np.isfinite(modelout_np).all():
                            raise FloatingPointError(
                                f"Non-finite modelout probabilities in "
                                f"role={role}, batch={batch_idx}"
                            )
                        if dense_modelout is not None:
                            dense_modelout[role_offset:end] = modelout_np.astype(
                                output_dtype, copy=False
                            )
                        modelout_probability_sum += float(
                            modelout_np.sum(dtype=np.float64)
                        )

                        # Strictly greater than threshold, as requested.
                        positive = modelout_np > float(args.pseudo_threshold)
                        row_counts = positive.sum(axis=1, dtype=np.int64)
                        rows_local, go_indices = np.nonzero(positive)
                        pseudo_values = modelout_np[rows_local, go_indices]
                        pseudo_indices_writer.append(
                            go_indices.astype(np.int32, copy=False)
                        )
                        pseudo_probs_writer.append(
                            pseudo_values.astype(output_dtype, copy=False)
                        )
                        pseudo_indptr[
                            pseudo_row_offset + 1 : pseudo_row_offset + batch_size + 1
                        ] = (
                            pseudo_indptr[pseudo_row_offset]
                            + np.cumsum(row_counts, dtype=np.int64)
                        )

                        pseudo_global_protein = batch_global[rows_local]
                        pseudo_edge_index_writer.append(
                            np.stack(
                                [
                                    pseudo_global_protein,
                                    go_indices.astype(np.int32, copy=False),
                                ],
                                axis=1,
                            )
                        )
                        pseudo_edge_attr_writer.append(
                            pseudo_values.astype(np.float32, copy=False)
                        )
                        pseudo_row_offset += batch_size

                    role_offset = end
                    if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
                        print(
                            f"[{role}] batches={batch_idx + 1} rows={role_offset}/{len(dataset)} "
                            f"rare_edges={rare_edge_index_writer.rows - role_rare_edges_before}",
                            flush=True,
                        )

            if role_offset > n_rows:
                raise RuntimeError(
                    f"role={role}: exported rows={role_offset}, allocated rows={n_rows}"
                )
            if dense_backbone is not None:
                dense_backbone.flush()
                del dense_backbone
                if role_offset < n_rows:
                    shrink_dense_rows(
                        stage_dir
                        / f"backbone_{role}_prob.{args.prob_suffix}.npy",
                        role_offset,
                    )
            if dense_modelout is not None:
                dense_modelout.flush()
                del dense_modelout
                if role_offset < n_rows:
                    shrink_dense_rows(
                        stage_dir
                        / f"modelout_{args.modelout_role}_prob.{args.prob_suffix}.npy",
                        role_offset,
                    )
            if use_modelout:
                assert pseudo_indptr is not None
                assert pseudo_row_offset == role_offset
                np.save(
                    stage_dir / f"{role}_pseudo_label_indptr.i64.npy",
                    pseudo_indptr[: role_offset + 1],
                )
            n_rows = role_offset

            role_rare_edges = rare_edge_index_writer.rows - role_rare_edges_before
            role_pseudo_edges = (
                0
                if pseudo_edge_index_writer is None
                else pseudo_edge_index_writer.rows - role_pseudo_edges_before
            )
            selector_hit_array = np.asarray(selector_hit_counts, dtype=np.int64)
            if (
                args.rare_selector_scope == "rare_first"
                and float(args.rare_min_backbone_prob) == 0.0
            ):
                expected_rare_degree = min(
                    int(args.rare_go_topk), len(rare_indices)
                )
                if not np.all(selector_hit_array == expected_rare_degree):
                    raise RuntimeError(
                        f"role={role}: rare_first must emit exactly "
                        f"{expected_rare_degree} rare GO edges per protein when "
                        "--rare-min-backbone-prob=0, observed degree range="
                        f"[{selector_hit_array.min()}, {selector_hit_array.max()}]"
                    )
            record: Dict[str, Any] = {
                "role": role,
                "dataset_mode": mode,
                "rows": role_offset,
                "protein_ids_sha256": hash_ids(expected_role_ids[:role_offset]),
                "global_protein_idx_min": int(global_indices[:role_offset].min()),
                "global_protein_idx_max": int(global_indices[:role_offset].max()),
                "backbone_dense_file": (
                    f"backbone_{role}_prob.{args.prob_suffix}.npy"
                    if args.save_dense_backbone
                    else None
                ),
                "backbone_probability_mean": (
                    backbone_probability_sum
                    / max(1, role_offset * int(args.num_classes))
                ),
                "rare_edge_count": int(role_rare_edges),
                "rare_degree_min": int(selector_hit_array.min()),
                "rare_degree_mean": float(selector_hit_array.mean()),
                "rare_degree_max": int(selector_hit_array.max()),
                "elapsed_seconds": round(time.time() - role_started, 3),
            }
            if use_modelout:
                record.update(
                    {
                        "modelout_dense_file": (
                            f"modelout_{role}_prob.{args.prob_suffix}.npy"
                            if args.save_dense_modelout
                            else None
                        ),
                        "modelout_probability_mean": (
                            modelout_probability_sum
                            / max(1, role_offset * int(args.num_classes))
                        ),
                        "pseudo_nnz": int(role_pseudo_edges),
                        "pseudo_degree_mean": float(
                            role_pseudo_edges / max(1, role_offset)
                        ),
                    }
                )
            role_records.append(record)

        rare_edge_index_writer.finalize(
            stage_dir / "pg_backbone_rare_edge_index.i32.npy",
            transpose_two_columns=True,
        )
        rare_edge_attr_writer.finalize(
            stage_dir / "pg_backbone_rare_edge_attr.f32.npy"
        )
        if args.modelout_role in args.roles:
            assert pseudo_indices_writer is not None
            assert pseudo_probs_writer is not None
            assert pseudo_edge_index_writer is not None
            assert pseudo_edge_attr_writer is not None
            pseudo_indices_writer.finalize(
                stage_dir / f"{args.modelout_role}_pseudo_label_indices.i32.npy"
            )
            pseudo_probs_writer.finalize(
                stage_dir
                / f"{args.modelout_role}_pseudo_prob.{args.prob_suffix}.npy"
            )
            pseudo_edge_index_writer.finalize(
                stage_dir / "pg_modelout_pseudo_edge_index.i32.npy",
                transpose_two_columns=True,
            )
            pseudo_edge_attr_writer.finalize(
                stage_dir / "pg_modelout_pseudo_edge_attr.f32.npy"
            )

        checkpoint_hash = (
            None
            if args.skip_checkpoint_sha256
            else sha256_file(args.checkpoint)
        )
        modelout_key = (
            "modelout::"
            f"{model_args.query_decoder_logit_base_mode}"
            "::decoderprob::"
            f"{args.modelout_decoder_prob_source}"
        )
        manifest = {
            "schema_version": 1,
            "exporter": {
                "id": EXPORTER_ID,
                "version": EXPORTER_VERSION,
                "file": Path(__file__).name,
            },
            "task": args.task,
            "metadata_task": metadata_task,
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": checkpoint_hash,
                "load": checkpoint_info,
                "training_config_sources": training_sources,
                "resolved_model_arg_sources": model_arg_sources,
            },
            "model_semantics": {
                "backbone": (
                    "sigmoid(backbone logits); no expert probability and no label hint"
                ),
                "rare_selector": {
                    "uses_learnable_selector": bool(
                        model.query_decoder.selector.use_learnable_selector
                    ),
                    "static_source": "backbone/base_topk",
                    "external_probability_used": False,
                    "label_hint_used": False,
                    "scope": args.rare_selector_scope,
                    "candidate_order": (
                        [
                            "train_defined_rare_go_vocabulary",
                            "backbone_static_prefilter_within_rare",
                            "trained_learnable_selector_rerank",
                            "graph_edge_topk",
                        ]
                        if args.rare_selector_scope == "rare_first"
                        else [
                            "trained_full_vocabulary_selector_topk",
                            "filter_selected_terms_to_rare",
                        ]
                    ),
                    "rare_before_selector_topk": (
                        args.rare_selector_scope == "rare_first"
                    ),
                    "rare_candidate_count": int(len(rare_indices)),
                    "selector_prefilter_topm_checkpoint": int(
                        model.query_decoder.selector.prefilter_topm
                    ),
                    "selector_prefilter_topm_effective": int(
                        effective_prefilter_topm
                    ),
                    "selector_topk_checkpoint": int(
                        model.query_decoder.selector.topk
                    ),
                    "requested_topk": int(args.rare_go_topk),
                    "minimum_backbone_probability": float(
                        args.rare_min_backbone_prob
                    ),
                },
                "modelout": {
                    "prediction_key": modelout_key,
                    "query_decoder_logit_base_mode": str(
                        model_args.query_decoder_logit_base_mode
                    ),
                    "expert_base_mix_alpha": float(
                        model_args.expert_base_mix_alpha
                    ),
                    "decoder_prob_source": args.modelout_decoder_prob_source,
                    "decoder_prob_alpha": float(
                        args.modelout_decoder_prob_alpha
                    ),
                    "topk_source": args.modelout_topk_source,
                    "external_prob_blend_alpha": float(
                        args.modelout_external_prob_blend_alpha
                    ),
                    "external_probability_used": (
                        args.modelout_decoder_prob_source
                        in {"expert", "mix_expert_base"}
                    ),
                    "label_hint_used": False,
                },
            },
            "index_contract": {
                "protein": (
                    "zero-based protein_idx in features/protein_registry.csv"
                ),
                "go": (
                    "zero-based go_idx in gg_relations/go_registry.tsv and "
                    "weak-model classifier column"
                ),
                "edge_index": (
                    "[2,E]; row 0 is protein, row 1 is GO; message direction "
                    "protein -> GO"
                ),
                "dense_and_csr_rows": (
                    "role-local role_row_idx in features/protein_registry.csv; "
                    "role is encoded in the filename"
                ),
            },
            "rare_definition": {
                **rare_definition,
                "training_rows": int(train_count_n),
                "training_annotation_key": train_count_key,
                "rare_indices_file": "rare_go_indices.i32.npy",
                "rare_registry_file": "rare_go_registry.tsv",
                "train_counts_file": "train_go_label_counts.f64.npy",
            },
            "backbone_rare_edges": {
                "edge_index_file": "pg_backbone_rare_edge_index.i32.npy",
                "edge_attr_file": "pg_backbone_rare_edge_attr.f32.npy",
                "edge_attr_columns": [
                    "backbone_probability",
                    "selector_score",
                    "reciprocal_rank",
                ],
                "column_0_is_absolute_message_weight": True,
                "edge_count": int(rare_edge_index_writer.rows),
            },
            "weak_pseudo_targets": (
                {
                    "role": args.modelout_role,
                    "comparison": ">",
                    "threshold": float(args.pseudo_threshold),
                    "csr_indptr_file": (
                        f"{args.modelout_role}_pseudo_label_indptr.i64.npy"
                    ),
                    "csr_indices_file": (
                        f"{args.modelout_role}_pseudo_label_indices.i32.npy"
                    ),
                    "csr_probability_file": (
                        f"{args.modelout_role}_pseudo_prob.{args.prob_suffix}.npy"
                    ),
                    "edge_index_file": "pg_modelout_pseudo_edge_index.i32.npy",
                    "edge_attr_file": "pg_modelout_pseudo_edge_attr.f32.npy",
                    "edge_attr_columns": ["modelout_probability"],
                    "column_0_is_absolute_message_weight": True,
                    "nnz": int(
                        0
                        if pseudo_edge_index_writer is None
                        else pseudo_edge_index_writer.rows
                    ),
                }
                if args.modelout_role in args.roles
                else None
            ),
            "protein_registry": {
                "path": str(protein_registry_path),
                "sha256": sha256_file(protein_registry_path),
                "num_proteins": protein_registry.num_proteins,
            },
            "go_registry": {
                "path": str(args.go_registry),
                "sha256": sha256_file(args.go_registry),
                "num_terms": go_registry.num_terms,
            },
            "roles": role_records,
            "runtime": {
                "device": str(device),
                "amp_enabled": amp_enabled,
                "amp_dtype": args.amp_dtype,
                "output_probability_dtype": args.output_prob_dtype,
                "batch_size": int(args.batch_size),
                "msa_sample_strategy": args.msa_sample_strategy,
                "msa_shuffle_rows_at_getitem": bool(
                    args.msa_shuffle_rows_at_getitem
                ),
                "sample_seed": int(args.sample_seed),
                "sampler_seed": int(args.sampler_seed),
                "max_batches": args.max_batches,
                "elapsed_seconds": round(time.time() - started, 3),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
        }
        atomic_write_text(
            stage_dir / "weak_graph_predictions_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )

        for staged_path in sorted(stage_dir.iterdir()):
            if staged_path.name.startswith("."):
                continue
            os.replace(staged_path, args.output_dir / staged_path.name)
    finally:
        rare_edge_index_writer.close()
        rare_edge_attr_writer.close()
        for writer in (
            pseudo_edge_index_writer,
            pseudo_edge_attr_writer,
            pseudo_indices_writer,
            pseudo_probs_writer,
        ):
            if writer is not None:
                writer.close()
        shutil.rmtree(stage_dir, ignore_errors=True)

    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()