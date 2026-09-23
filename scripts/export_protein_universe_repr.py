#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Stage-1 pooled protein representations for Stage-2 P-P retrieval.

This script is designed to run inside the LATENCE project.  It deliberately
uses only the Stage-1 backbone; the DETR/query decoder and external expert
probabilities are not part of the FAISS representation.

Default role-to-dataset mapping
-------------------------------
core      -> train
weak      -> exp_train
valid     -> valid
ind_test  -> ind_test

The feature arrays are standard ``.npy`` files created incrementally with
``numpy.lib.format.open_memmap``.  They can therefore be read later with
``np.load(path, mmap_mode="r")`` without loading the whole universe into RAM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch.amp import autocast


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


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def csv_items(value: str) -> List[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return items


def cli_keys(argv: Sequence[str]) -> set[str]:
    """Return argparse destination-like keys explicitly supplied on the CLI."""
    keys: set[str] = set()
    for token in argv:
        if token.startswith("--"):
            keys.add(token[2:].split("=", 1)[0].replace("-", "_"))
    return keys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export pooled Stage-1 backbone representations for the protein universe"
    )
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--train-args-json",
        type=str,
        default="auto",
        help="Training args.json, or 'auto' to look next to the checkpoint.",
    )
    p.add_argument("--task", required=True, choices=["bp", "mf", "cc"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--roles",
        type=csv_items,
        default=csv_items("core,weak,valid,ind_test"),
        help="Any of: core,weak,valid,ind_test.",
    )
    p.add_argument(
        "--role-mode",
        action="append",
        default=[],
        metavar="ROLE=MODE",
        help="Override a role's MSABinaryDataset mode; may be repeated.",
    )

    # Arch / dataset paths.  When omitted, these are read from args.json or
    # checkpoint['model_args'].
    p.add_argument("--model-config", type=str, default=None)
    p.add_argument("--file-address", type=str, default=None)
    p.add_argument("--working-address", type=str, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--msa-max-size", type=int, default=None)
    p.add_argument("--permute-dims", type=int, nargs=4, default=None)

    # Runtime-only settings are intentionally not inherited from the training
    # launcher unless their CLI value is omitted/None.
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--persistent-workers", type=parse_bool, default=True)
    p.add_argument("--pin-memory", type=parse_bool, default=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--gpu-ids", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--msa-read-mode", choices=["full", "rows", "block"], default="full")
    p.add_argument("--msa-sample-strategy", choices=["random", "block", "head"], default="random")
    p.add_argument("--msa-shuffle-rows-at-getitem", type=parse_bool, default=False)
    p.add_argument("--msa-cache-gb", type=float, default=4.0)
    p.add_argument("--msa-max-open-files", type=int, default=256)
    p.add_argument("--sample-seed", type=int, default=3407)
    p.add_argument("--sampler-seed", type=int, default=3407)
    p.add_argument("--max-batches", type=int, default=None, help="Diagnostic only; do not use for final export.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-partial-checkpoint", action="store_true")
    p.add_argument("--allow-cross-role-overlap", action="store_true")
    return p


def torch_load(path: Path) -> Any:
    # A training checkpoint is trusted project input.  Explicitly request the
    # legacy/full loader because model_args may contain non-tensor values.
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(obj).__name__}")
    return obj


def find_train_args(checkpoint: Path, requested: str, payload: Any) -> Tuple[Dict[str, Any], str]:
    if requested.lower() != "auto":
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Training args JSON not found: {path}")
        return read_json(path), str(path)

    for path in (checkpoint.parent / "args.json", checkpoint.parent.parent / "args.json"):
        if path.is_file():
            return read_json(path), str(path)

    if isinstance(payload, Mapping) and isinstance(payload.get("model_args"), Mapping):
        return dict(payload["model_args"]), "checkpoint:model_args"
    return {}, "none"


def overlay_training_fields(
    args: argparse.Namespace,
    train_args: Mapping[str, Any],
    explicitly_set: set[str],
) -> argparse.Namespace:
    # Only fields required to reconstruct the Stage-1 architecture/dataset are
    # inherited.  Export batch size, workers, AMP, output paths, etc. remain
    # controlled by this script.
    allowed = {
        "model_config",
        "file_address",
        "working_address",
        "num_classes",
        "top_k",
        "max_len",
        "msa_max_size",
        "permute_dims",
        "gpu_ids",
    }
    for key in allowed:
        if key in explicitly_set or key not in train_args:
            continue
        current = getattr(args, key, None)
        if current is None:
            setattr(args, key, train_args[key])
    return args


def infer_num_classes(payload: Any) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    if "num_classes" in payload:
        return int(payload["num_classes"])
    shape = payload.get("query_decoder_classifier_weight_shape")
    if isinstance(shape, (list, tuple)) and shape:
        return int(shape[0])
    model_args = payload.get("model_args")
    if isinstance(model_args, Mapping) and model_args.get("num_classes") is not None:
        return int(model_args["num_classes"])
    return None


def normalize_state_keys(state: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        while key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        out[key] = value
    return out


def extract_backbone_state(payload: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}")

    if isinstance(payload.get("backbone"), Mapping):
        return normalize_state_keys(payload["backbone"]), str(
            payload.get("checkpoint_type", "detr_payload")
        )

    for wrapper in ("state_dict", "model_state_dict", "model"):
        if isinstance(payload.get(wrapper), Mapping):
            candidate = payload[wrapper]
            keys = [str(k).removeprefix("module.") for k in candidate]
            if any(k.startswith("backbone.") for k in keys):
                filtered = {
                    k: v
                    for k, v in candidate.items()
                    if str(k).removeprefix("module.").startswith("backbone.")
                }
                return normalize_state_keys(filtered), f"{wrapper}:full_model"
            return normalize_state_keys(candidate), wrapper

    keys = [str(k).removeprefix("module.") for k in payload]
    if any(k.startswith("backbone.") for k in keys):
        filtered = {
            k: v
            for k, v in payload.items()
            if str(k).removeprefix("module.").startswith("backbone.")
        }
        return normalize_state_keys(filtered), "full_model_state_dict"

    return normalize_state_keys(payload), "backbone_state_dict"


def align_state_keys_to_model(
    state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Align checkpoints saved with/without nested DataParallel wrappers.

    ``Arch`` may wrap ``pre_model``, ``gnet`` and ``rnet`` in DataParallel when
    GPU IDs are supplied.  A CPU export then has keys without the intermediate
    ``module`` component.  Canonical matching lets both layouts load while
    retaining the exact key expected by the newly instantiated model.
    """

    def canonical(key: str) -> str:
        return ".".join(part for part in key.split(".") if part != "module")

    target_by_canonical: Dict[str, List[str]] = {}
    for target_key in model_state:
        target_by_canonical.setdefault(canonical(str(target_key)), []).append(str(target_key))

    aligned: Dict[str, torch.Tensor] = {}
    for source_key, value in state.items():
        candidates = target_by_canonical.get(canonical(str(source_key)), [])
        target_key = candidates[0] if len(candidates) == 1 else str(source_key)
        if target_key in aligned:
            raise RuntimeError(f"Two checkpoint keys map to the same model key: {target_key}")
        aligned[target_key] = value
    return aligned


def parse_role_modes(items: Iterable[str]) -> Dict[str, str]:
    result = dict(ROLE_TO_MODE)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--role-mode expects ROLE=MODE, got {item!r}")
        role, mode = (x.strip() for x in item.split("=", 1))
        if not role or not mode:
            raise ValueError(f"--role-mode expects ROLE=MODE, got {item!r}")
        result[role] = mode
    return result


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def hash_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_ids(ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for protein_id in ids:
        h.update(protein_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def normalize_protein_ids(proteins: Any, expected: int) -> List[str]:
    if proteins is None:
        raise RuntimeError("Dataset did not return protein IDs; need_proteins=True is required")
    if isinstance(proteins, torch.Tensor):
        values = proteins.detach().cpu().tolist()
    elif isinstance(proteins, np.ndarray):
        values = proteins.tolist()
    elif isinstance(proteins, (list, tuple)):
        values = list(proteins)
    else:
        values = [proteins]
    result = [str(x) for x in values]
    if len(result) != expected:
        raise RuntimeError(f"Protein ID count {len(result)} != batch size {expected}")
    for protein_id in result:
        if "\n" in protein_id or "\r" in protein_id or "\t" in protein_id:
            raise ValueError(f"Protein ID contains a tab/newline and cannot be serialized safely: {protein_id!r}")
    return result


def ensure_writable(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        preview = "\n  ".join(existing[:10])
        raise FileExistsError(f"Output already exists; use --overwrite to replace it:\n  {preview}")


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def make_loader(dataset: Any, args: argparse.Namespace, make_loader_fn: Any) -> Any:
    return make_loader_fn(
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


def export_role(
    *,
    role: str,
    mode: str,
    model: torch.nn.Module,
    opt: SimpleNamespace,
    args: argparse.Namespace,
    device: torch.device,
    build_msa_dataset_fn: Any,
    make_loader_fn: Any,
    unpack_batch_fn: Any,
    set_model_proteins_fn: Any,
    build_backbone_memory_fn: Any,
) -> Dict[str, Any]:
    # The public CLI and checkpoint/output naming use bp/mf/cc, while the
    # historical metadata file is indexed by the complete ontology names.
    metadata_task = TASK_TO_METADATA_KEY[args.task]
    dataset = build_msa_dataset_fn(
        opt,
        mode=mode,
        task=metadata_task,
        need_proteins=True,
    )
    n_expected = len(dataset)
    if n_expected <= 0:
        raise RuntimeError(f"Role {role!r} / mode {mode!r} contains no samples")
    loader = make_loader(dataset, args, make_loader_fn)

    dtype = np.float16 if args.feature_dtype == "float16" else np.float32
    final_feature = args.output_dir / f"{role}_repr.{('f16' if dtype == np.float16 else 'f32')}.npy"
    partial_feature = final_feature.with_name(final_feature.name + ".partial")
    final_ids = args.output_dir / f"{role}_protein_ids.txt"
    ensure_writable((final_feature, final_ids), args.overwrite)

    ids: List[str] = []
    mmap: np.memmap | None = None
    feature_dim: int | None = None
    offset = 0
    amp_enabled = device.type == "cuda" and not bool(args.no_amp)
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    started = time.time()

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break
            proteins, x, _ = unpack_batch_fn(batch)
            batch_size = int(x.shape[0])
            batch_ids = normalize_protein_ids(proteins, batch_size)
            set_model_proteins_fn(model, proteins)
            x = x.to(device, non_blocking=True).long()

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                _, h = model(
                    x,
                    permute_dims=tuple(int(v) for v in args.permute_dims),
                    return_embedding=True,
                )
                _, pooled = build_backbone_memory_fn(h, mode="pooled")

            array = pooled.detach().float().cpu().numpy()
            if array.ndim != 2 or array.shape[0] != batch_size:
                raise RuntimeError(f"Unexpected pooled feature shape: {array.shape}")
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Non-finite pooled features in role={role}, batch={batch_idx}")

            if mmap is None:
                feature_dim = int(array.shape[1])
                # A diagnostic --max-batches run intentionally writes only the
                # processed rows; final exports allocate all dataset rows.
                n_rows = n_expected
                if args.max_batches is not None:
                    n_rows = min(n_expected, int(args.max_batches) * int(args.batch_size))
                mmap = np.lib.format.open_memmap(
                    partial_feature,
                    mode="w+",
                    dtype=dtype,
                    shape=(n_rows, feature_dim),
                )
            if array.shape[1] != feature_dim:
                raise RuntimeError(
                    f"Feature dimension changed from {feature_dim} to {array.shape[1]} in role={role}"
                )
            end = offset + batch_size
            if end > mmap.shape[0]:
                raise RuntimeError(f"Export wrote beyond allocated rows: end={end}, rows={mmap.shape[0]}")
            mmap[offset:end] = array.astype(dtype, copy=False)
            ids.extend(batch_ids)
            offset = end

            if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
                print(f"[{role}] batches={batch_idx + 1}, rows={offset}/{n_expected}", flush=True)

    if mmap is None or feature_dim is None:
        raise RuntimeError(f"No batches were produced for role={role}")
    if args.max_batches is None and offset != n_expected:
        raise RuntimeError(f"Role {role}: exported {offset} rows but dataset length is {n_expected}")
    if offset != mmap.shape[0]:
        # This can occur only in a final short diagnostic allocation.  Shrink
        # by making a correctly shaped final partial array.
        shrunk = partial_feature.with_name(partial_feature.name + ".shrunk")
        out = np.lib.format.open_memmap(shrunk, mode="w+", dtype=dtype, shape=(offset, feature_dim))
        out[:] = mmap[:offset]
        out.flush()
        del out
        mmap.flush()
        del mmap
        os.replace(shrunk, partial_feature)
    else:
        mmap.flush()
        del mmap

    if len(set(ids)) != len(ids):
        raise RuntimeError(f"Duplicate protein IDs found within role={role}")
    atomic_write_text(final_ids, "".join(f"{protein_id}\n" for protein_id in ids))
    os.replace(partial_feature, final_feature)

    return {
        "role": role,
        "dataset_mode": mode,
        "count": offset,
        "feature_dim": feature_dim,
        "feature_dtype": np.dtype(dtype).name,
        "feature_file": final_feature.name,
        "protein_ids_file": final_ids.name,
        "protein_ids_sha256": hash_ids(ids),
        "elapsed_seconds": round(time.time() - started, 3),
        "ids": ids,
    }


def write_registry(output_dir: Path, role_results: Sequence[Mapping[str, Any]], overwrite: bool) -> Path:
    path = output_dir / "protein_registry.csv"
    ensure_writable((path,), overwrite)
    tmp = path.with_name(path.name + ".partial")
    global_idx = 0
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["protein_idx", "protein_id", "role", "role_row_idx", "feature_file", "dataset_mode"]
        )
        for result in role_results:
            for role_row_idx, protein_id in enumerate(result["ids"]):
                writer.writerow(
                    [
                        global_idx,
                        protein_id,
                        result["role"],
                        role_row_idx,
                        result["feature_file"],
                        result["dataset_mode"],
                    ]
                )
                global_idx += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    explicit = cli_keys(argv)
    args = parser.parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    invalid_roles = sorted(set(args.roles) - set(ROLE_TO_MODE))
    if invalid_roles:
        raise ValueError(f"Unknown roles: {invalid_roles}; expected {sorted(ROLE_TO_MODE)}")
    if len(set(args.roles)) != len(args.roles):
        raise ValueError(f"Duplicate roles in --roles: {args.roles}")

    payload = torch_load(args.checkpoint)
    train_args, train_args_source = find_train_args(args.checkpoint, args.train_args_json, payload)
    args = overlay_training_fields(args, train_args, explicit)
    if args.num_classes is None:
        args.num_classes = infer_num_classes(payload)
    if args.permute_dims is None:
        args.permute_dims = [0, 3, 2, 1]

    required = ["model_config", "file_address", "working_address", "num_classes", "top_k", "max_len"]
    missing = [name for name in required if getattr(args, name, None) in {None, ""}]
    if missing:
        raise ValueError(
            f"Missing architecture/dataset fields after training-args overlay: {missing}. "
            "Pass them explicitly or restore args.json/model_args."
        )

    device = resolve_device(args.device)
    args.device = str(device)
    args.distributed = False
    args.local_rank = 0
    args.torch_compile = False
    if device.type == "cpu":
        args.gpu_ids = ""
    elif args.gpu_ids is None:
        args.gpu_ids = str(device.index if device.index is not None else 0)

    # Resolve relative project paths exactly as a launcher executed from the
    # LATENCE root would do.
    os.chdir(args.project_root)
    for path_name in ("model_config", "file_address", "working_address"):
        value = Path(str(getattr(args, path_name))).expanduser()
        if not value.is_absolute():
            value = (args.project_root / value).resolve()
        setattr(args, path_name, str(value))

    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root / "msa_models"))
    try:
        from models import Arch
        from experiments.exp_train import (
            build_msa_dataset,
            make_loader,
            set_model_proteins,
            unpack_batch,
        )
        from experiments.weak_exp_train_detr import build_backbone_memory, build_weak_opt_from_config
    except ImportError as exc:
        raise ImportError(
            f"Could not import LATENCE modules from --project-root={args.project_root}. "
            "Run this script from the project root or pass the correct --project-root."
        ) from exc

    opt = build_weak_opt_from_config(args)
    opt.mode = "eval"
    opt.shuffle = False
    model = Arch(opt)
    state, checkpoint_type = extract_backbone_state(payload)
    state = align_state_keys_to_model(state, model.state_dict())
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if (missing_keys or unexpected_keys) and not args.allow_partial_checkpoint:
        raise RuntimeError(
            "Backbone checkpoint did not match exactly. "
            f"missing={list(missing_keys)[:20]}, unexpected={list(unexpected_keys)[:20]}. "
            "Use --allow-partial-checkpoint only after auditing these keys."
        )
    model = model.to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    role_modes = parse_role_modes(args.role_mode)
    results: List[Dict[str, Any]] = []
    seen_role_by_id: MutableMapping[str, str] = {}
    for role in args.roles:
        result = export_role(
            role=role,
            mode=role_modes[role],
            model=model,
            opt=opt,
            args=args,
            device=device,
            build_msa_dataset_fn=build_msa_dataset,
            make_loader_fn=make_loader,
            unpack_batch_fn=unpack_batch,
            set_model_proteins_fn=set_model_proteins,
            build_backbone_memory_fn=build_backbone_memory,
        )
        overlaps: List[Tuple[str, str]] = []
        for protein_id in result["ids"]:
            previous = seen_role_by_id.get(protein_id)
            if previous is not None:
                overlaps.append((protein_id, previous))
            else:
                seen_role_by_id[protein_id] = role
        if overlaps and not args.allow_cross_role_overlap:
            preview = ", ".join(f"{pid}({previous}/{role})" for pid, previous in overlaps[:10])
            raise RuntimeError(
                f"Cross-role protein overlap detected ({len(overlaps)} in role={role}): {preview}. "
                "This usually indicates split leakage; use --allow-cross-role-overlap only if intentional."
            )
        results.append(result)

    feature_dims = {int(r["feature_dim"]) for r in results}
    if len(feature_dims) != 1:
        raise RuntimeError(f"Feature dimensions differ across roles: {sorted(feature_dims)}")
    registry_path = write_registry(args.output_dir, results, args.overwrite)

    manifest_path = args.output_dir / "representation_manifest.json"
    ensure_writable((manifest_path,), args.overwrite)
    manifest = {
        "schema_version": 1,
        "task": args.task,
        "metadata_task": TASK_TO_METADATA_KEY[args.task],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hash_file(args.checkpoint),
        "checkpoint_type": checkpoint_type,
        "checkpoint_missing_keys": list(missing_keys),
        "checkpoint_unexpected_keys": list(unexpected_keys),
        "train_args_source": train_args_source,
        "pooling_method": "build_backbone_memory(mode=pooled)",
        "representation_source": "stage1_backbone_only",
        "feature_dim": next(iter(feature_dims)),
        "feature_dtype": args.feature_dtype,
        "registry_file": registry_path.name,
        "role_order": list(args.roles),
        "roles": [
            {key: value for key, value in result.items() if key != "ids"}
            for result in results
        ],
        "runtime": {
            "device": str(device),
            "amp_enabled": device.type == "cuda" and not args.no_amp,
            "amp_dtype": args.amp_dtype,
            "batch_size": args.batch_size,
            "sample_seed": args.sample_seed,
            "msa_sample_strategy": args.msa_sample_strategy,
            "msa_shuffle_rows_at_getitem": args.msa_shuffle_rows_at_getitem,
            "max_batches": args.max_batches,
        },
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
