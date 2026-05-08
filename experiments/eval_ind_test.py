#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiments/eval_ind_test.py

Evaluation script for MSA-GO / semi-supervised Protein Expansion checkpoints.

Main purpose:
    Load a checkpoint, evaluate a dataset split such as "ind_test",
    and report fmax / auprc using msa_models/helper_functions/helper.py::evalperf_torch.

Notes:
    1. This evaluator instantiates the original Arch backbone directly.
       It does not need SemiSupMSAGO or ProjectionHead.
    2. It can load:
         - original teacher Arch checkpoints
         - semisup_backbone_*.pt
         - semisup_full_*.pt
       by stripping module/backbone prefixes and ignoring projection-head weights.
    3. DDP evaluation is supported without wrapping the model in DDP.
       Each rank evaluates a disjoint shard, saves local predictions, and rank0
       computes global metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
MSA_ROOT = ROOT / "msa_models"

if not MSA_ROOT.exists():
    raise FileNotFoundError(f"Cannot find msa_models directory: {MSA_ROOT}")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MSA_ROOT))

from models import Arch  # noqa: E402
from experiments.msabin import MSABinaryDataset  # noqa: E402

from experiments.exp_train import (  # noqa: E402
    normalize_task,
    parse_gpu_ids_arg,
    init_distributed_mode,
    cleanup_distributed,
    dist_is_initialized,
    get_rank,
    get_world_size,
    is_main_process,
    rank0_print,
    unpack_batch,
    move_to_device,
    set_model_proteins,
    strip_state_dict_prefix,
)

from helper_functions.helper import evalperf_torch  # noqa: E402


# ---------------------------------------------------------------------
# IO utilities
# ---------------------------------------------------------------------

def load_pickle(path: Union[str, Path]):
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def save_json(obj, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_pickle_config(path: Union[str, Path]) -> dict:
    cfg = load_pickle(path)
    if not isinstance(cfg, dict):
        raise TypeError(f"Model config pickle must contain a dict, got {type(cfg)}")
    return cfg


def torch_load_cpu(path: Union[str, Path]):
    """
    Compatibility wrapper for different torch.load signatures.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_optional_path(x):
    """
    Convert None / "" / "none" / "null" to None.
    """
    if x is None:
        return None

    s = str(x).strip()

    if s == "" or s.lower() in {"none", "null"}:
        return None

    return s

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

def build_eval_opt_from_config(args: argparse.Namespace) -> SimpleNamespace:
    """
    Build Arch(opt) config for evaluation.

    This is intentionally smaller than exp_train.build_opt_from_config:
    evaluation does not need projection-head or augmentation fields.
    """
    cfg = load_pickle_config(args.model_config)

    task = normalize_task(args.task)

    runtime_overrides = {
        "file_address": args.file_address,
        "working_address": args.working_address,
        "task": task,
        "num_classes": args.num_classes,

        "batch_size": args.eval_batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
        "permute_dims": tuple(args.permute_dims),
        "torch_compile": args.torch_compile,
        "no_amp": args.no_amp,
        "device": args.device,

        "mode": args.mode,
        "shuffle": False,

        # binary MSA dataset / loader
        "msa_read_mode": args.msa_read_mode,
        "msa_sample_strategy": args.msa_sample_strategy,
        "msa_shuffle_rows_at_getitem": args.msa_shuffle_rows_at_getitem,
        "msa_cache_gb": args.msa_cache_gb,
        "msa_max_open_files": args.msa_max_open_files,
        "sample_seed": args.sample_seed,
    }

    if args.top_k is not None:
        runtime_overrides["top_k"] = args.top_k

    if args.max_len is not None:
        runtime_overrides["max_len"] = args.max_len

    if args.msa_max_size is not None:
        runtime_overrides["msa_max_size"] = args.msa_max_size

    parsed_gpu_ids = parse_gpu_ids_arg(
        args.gpu_ids,
        args.device,
        local_rank=getattr(args, "local_rank", 0),
        distributed=getattr(args, "distributed", False),
    )

    if parsed_gpu_ids is not None:
        runtime_overrides["gpu_ids"] = parsed_gpu_ids

    cfg.update(runtime_overrides)

    cfg.setdefault("mode", args.mode)
    cfg.setdefault("shuffle", False)
    cfg.setdefault("msa_max_size", None)

    required_fields = [
        "top_k",
        "max_len",
        "num_classes",
        "in_channels",
        "out_channels_G",
        "msa_embedding_dim",
        "msa_encoding_strategy",
        "ngf",
        "netG",
        "normG",
        "netD",
        "init_type",
        "init_gain",
        "gpu_ids",
    ]

    missing = [k for k in required_fields if k not in cfg or cfg[k] is None]
    if missing:
        raise ValueError(
            "Missing required fields for Arch(opt): "
            f"{missing}. Check model_config or runner overrides."
        )

    return SimpleNamespace(**cfg)


# ---------------------------------------------------------------------
# Dataset / loader
# ---------------------------------------------------------------------

def build_msa_dataset(
    opt: SimpleNamespace,
    mode: str,
    task: str,
    need_proteins: bool = False,
):
    cache_max_bytes = int(float(getattr(opt, "msa_cache_gb", 0.0)) * 1024 ** 3)

    return MSABinaryDataset(
        index_file=opt.working_address,
        metadata_file=opt.file_address,
        mode=mode,
        task=task,

        num_classes=opt.num_classes,
        topk=opt.top_k,
        max_len=opt.max_len,

        need_proteins=need_proteins,

        read_mode=getattr(opt, "msa_read_mode", "full"),
        sample_strategy=getattr(opt, "msa_sample_strategy", "head"),
        shuffle_rows_at_getitem=getattr(opt, "msa_shuffle_rows_at_getitem", False),

        cache_max_bytes=cache_max_bytes,
        max_open_files=getattr(opt, "msa_max_open_files", 256),

        sample_seed=getattr(opt, "sample_seed", 1),
        avoid_last_sample=True,
    )


class EvalShardBatchSampler(Sampler):
    """
    Exact, no-duplication eval batch sampler.

    It preserves shard grouping for better binary-MSA IO locality,
    but unlike the training DistributedShardShuffleBatchSampler, it does NOT pad
    batches to make every rank have equal length.

    This is safe because the model is not wrapped in DDP during evaluation.
    """

    def __init__(
        self,
        shard_ids,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

        groups = {}
        for i, sid in enumerate(shard_ids):
            sid = int(sid)
            groups.setdefault(sid, []).append(i)

        global_batches = []
        for sid in sorted(groups.keys()):
            indices = groups[sid]
            for start in range(0, len(indices), self.batch_size):
                global_batches.append(indices[start:start + self.batch_size])

        self.global_batches = global_batches
        self.local_batches = [
            b for i, b in enumerate(global_batches)
            if i % self.world_size == self.rank
        ]

    def __iter__(self):
        for batch in self.local_batches:
            yield batch

    def __len__(self):
        return len(self.local_batches)


def make_eval_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    rank: int = 0,
    world_size: int = 1,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
):
    if hasattr(dataset, "sample_shard_ids"):
        shard_ids = dataset.sample_shard_ids
    else:
        shard_ids = [0] * len(dataset)

    batch_sampler = EvalShardBatchSampler(
        shard_ids=shard_ids,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )

    kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_sampler,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(persistent_workers)

    return DataLoader(**kwargs)


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------

def extract_state_dict(raw):
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]

        # Already a state_dict-like object.
        if all(isinstance(k, str) for k in raw.keys()):
            return raw

    raise TypeError(
        "Unsupported checkpoint format. Expected a state_dict or a dict "
        "containing key 'state_dict'."
    )


def load_arch_checkpoint(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    strict_shape: bool = True,
):
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch_load_cpu(ckpt_path)
    sd = extract_state_dict(raw)

    # Handles:
    #   module.xxx
    #   backbone.xxx
    #   module.backbone.xxx
    #   pre_model.module.xxx
    sd = strip_state_dict_prefix(sd)

    model_sd = model.state_dict()

    load_sd = {}
    skipped_shape = []
    ignored = []

    for k, v in sd.items():
        if k not in model_sd:
            ignored.append(k)
            continue

        if tuple(model_sd[k].shape) != tuple(v.shape):
            skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue

        load_sd[k] = v

    if strict_shape and len(load_sd) == 0:
        raise RuntimeError(f"No compatible tensors loaded from {ckpt_path}")

    missing, unexpected = model.load_state_dict(load_sd, strict=False)

    rank0_print(f"[Checkpoint] {ckpt_path}")
    rank0_print(f"[Checkpoint] loaded tensors: {len(load_sd)}")
    rank0_print(f"[Checkpoint] ignored non-Arch tensors: {len(ignored)}")
    rank0_print(f"[Checkpoint] skipped shape-mismatch tensors: {len(skipped_shape)}")
    rank0_print(f"[Checkpoint] missing after partial load: {len(missing)}")
    rank0_print(f"[Checkpoint] unexpected after partial load: {len(unexpected)}")

    if skipped_shape and is_main_process():
        rank0_print("[Checkpoint] First few shape mismatches:")
        for item in skipped_shape[:10]:
            rank0_print("  ", item)


# ---------------------------------------------------------------------
# Prediction / collection
# ---------------------------------------------------------------------

def extract_logits(model_output):
    if isinstance(model_output, dict):
        return model_output["logits"]

    if isinstance(model_output, (tuple, list)):
        return model_output[0]

    return model_output

def _as_model_list(models):
    if isinstance(models, nn.Module):
        return [models]

    if (
        isinstance(models, (list, tuple))
        and len(models) > 0
        and all(isinstance(m, nn.Module) for m in models)
    ):
        return list(models)

    raise TypeError(
        "models must be an nn.Module or a non-empty list/tuple of nn.Module"
    )


def _normalize_model_weights(model_weights, n_models: int):
    if n_models <= 0:
        raise ValueError("n_models must be positive")

    if model_weights is None:
        return [1.0 / n_models for _ in range(n_models)]

    if len(model_weights) != n_models:
        raise ValueError(
            f"model_weights length mismatch: "
            f"len(model_weights)={len(model_weights)}, n_models={n_models}"
        )

    ws = [float(w) for w in model_weights]

    if any(w < 0.0 for w in ws):
        raise ValueError(f"model_weights must be non-negative, got {ws}")

    s = sum(ws)

    if s <= 0.0:
        raise ValueError(f"Sum of model_weights must be positive, got {ws}")

    return [w / s for w in ws]


@torch.no_grad()
def predict_dataset(
    models,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    permute_dims=(0, 3, 2, 1),
    no_amp: bool = False,
    need_proteins: bool = False,
    model_weights=None,
):
    """
    Predict one dataset split.

    If models contains multiple models, final prediction is weighted average of
    sigmoid probabilities:

        p = sum_i w_i * sigmoid(logits_i)

    Default for two models is simple average:
        0.5 * p_student + 0.5 * p_teacher
    """
    models = _as_model_list(models)
    model_weights = _normalize_model_weights(model_weights, len(models))

    for m in models:
        m.eval()

    amp_enabled = device.type == "cuda" and not no_amp

    local_preds = []
    local_targs = []

    pbar = tqdm(
        loader,
        desc=f"predict rank{get_rank()} ({len(models)} model(s))",
        disable=not is_main_process(),
    )

    for batch in pbar:
        proteins, X, y = unpack_batch(batch)

        X, y = move_to_device(X, y, device=device)
        X = X.long()
        y = y.float()

        if need_proteins and proteins is not None:
            for m in models:
                set_model_proteins(m, proteins)

        with autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        ):
            probs_acc = None

            for m, w in zip(models, model_weights):
                out = m(
                    X,
                    permute_dims=permute_dims,
                )
                logits = extract_logits(out)

                if logits.ndim != 2 or logits.shape[1] != num_classes:
                    raise RuntimeError(
                        "Unexpected logits shape: "
                        f"got {tuple(logits.shape)}, expected [B, {num_classes}]"
                    )

                probs_i = torch.sigmoid(logits.float())

                if probs_acc is None:
                    probs_acc = float(w) * probs_i
                else:
                    probs_acc = probs_acc + float(w) * probs_i

            probs = probs_acc

        local_preds.append(probs.detach().cpu().float())
        local_targs.append((y.detach().cpu() > 0.5).to(torch.uint8))

    if len(local_preds) == 0:
        preds = torch.empty((0, num_classes), dtype=torch.float32)
        targs = torch.empty((0, num_classes), dtype=torch.uint8)
    else:
        preds = torch.cat(local_preds, dim=0)
        targs = torch.cat(local_targs, dim=0)

    return targs, preds


def collect_predictions(
    local_targs: torch.Tensor,
    local_preds: torch.Tensor,
    output_dir: Union[str, Path],
    mode: str,
    method: str = "file",
    keep_part_files: bool = False,
):
    """
    Return full targs/preds on rank0, and (None, None) on non-rank0.

    method:
        file:
            Each rank saves a part file. Rank0 loads all parts.
            This avoids gathering huge prediction tensors to every rank.
        all_gather_object:
            Simpler but every rank receives every tensor. More memory-heavy.
    """
    if not dist_is_initialized():
        return local_targs, local_preds

    rank = get_rank()
    world_size = get_world_size()
    output_dir = Path(output_dir)

    method = str(method).lower()

    if method == "all_gather_object":
        obj = {
            "targs": local_targs,
            "preds": local_preds,
        }

        objects = [None for _ in range(world_size)]
        dist.all_gather_object(objects, obj)

        if not is_main_process():
            return None, None

        full_targs = torch.cat([o["targs"] for o in objects], dim=0)
        full_preds = torch.cat([o["preds"] for o in objects], dim=0)
        return full_targs, full_preds

    if method != "file":
        raise ValueError(
            f"Unknown distributed collection method: {method}. "
            "Use 'file' or 'all_gather_object'."
        )

    part_dir = output_dir / ".eval_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    part_path = part_dir / f"{mode}_rank{rank:05d}.pt"

    torch.save(
        {
            "rank": rank,
            "targs": local_targs,
            "preds": local_preds,
        },
        part_path,
    )

    dist.barrier()

    full_targs = None
    full_preds = None

    if is_main_process():
        parts = []
        for r in range(world_size):
            p = part_dir / f"{mode}_rank{r:05d}.pt"
            if not p.is_file():
                raise FileNotFoundError(f"Missing eval part file: {p}")

            part = torch_load_cpu(p)
            parts.append(part)

        full_targs = torch.cat([p["targs"] for p in parts], dim=0)
        full_preds = torch.cat([p["preds"] for p in parts], dim=0)

        if not keep_part_files:
            shutil.rmtree(part_dir, ignore_errors=True)

    dist.barrier()

    return full_targs, full_preds


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_evalperf_metrics(
    targs: torch.Tensor,
    preds: torch.Tensor,
    report_threshold: bool = True,
    no_empty_labels: bool = False,
    no_zero_classes: bool = False,
):
    """
    evalperf_torch expects:
        targs: [N, C] binary / multi-hot labels
        preds: [N, C] probabilities, not logits

    fmax and auprc returned by evalperf_torch are percentages.
    """
    targs = targs.float()
    preds = preds.float()

    metrics = evalperf_torch(
        targs=targs,
        preds=preds,
        threshold=bool(report_threshold),
        auprc=True,
        no_empty_labels=bool(no_empty_labels),
        no_zero_classes=bool(no_zero_classes),
    )

    return {k: float(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def evaluate_one_task(args: argparse.Namespace):
    args.task = normalize_task(args.task)

    device = init_distributed_mode(args)
    set_seed(int(args.seed) + int(getattr(args, "rank", 0)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional teacher checkpoint for probability-average ensemble.
    teacher_ckpt = clean_optional_path(getattr(args, "teacher_ckpt", None))
    args.teacher_ckpt = teacher_ckpt

    if not hasattr(args, "ensemble_student_weight"):
        args.ensemble_student_weight = 1.0

    if not hasattr(args, "ensemble_teacher_weight"):
        args.ensemble_teacher_weight = 1.0

    if teacher_ckpt is not None and not Path(teacher_ckpt).is_file():
        raise FileNotFoundError(f"teacher_ckpt not found: {teacher_ckpt}")

    if is_main_process():
        print("=" * 80)
        print("[Eval] MSA-GO independent test evaluation")
        print("-" * 80)
        print(f"[Device]       {device}")
        print(f"[Distributed]  {getattr(args, 'distributed', False)}")
        print(f"[Rank]         {getattr(args, 'rank', 0)} / {getattr(args, 'world_size', 1)}")
        print(f"[Task]         {args.task}")
        print(f"[Mode]         {args.mode}")
        print(f"[Config]       {args.model_config}")
        print(f"[Student ckpt] {args.ckpt}")
        print(f"[Teacher ckpt] {teacher_ckpt}")
        print(f"[Dataset]      {args.file_address}")
        print(f"[MSA]          {args.working_address}")
        print(f"[Output]       {output_dir}")
        print("=" * 80)

        save_json(vars(args), output_dir / "eval_args.json")

    opt = build_eval_opt_from_config(args)

    if is_main_process():
        save_json(vars(opt), output_dir / "merged_eval_model_opt.json")

    # -------------------------
    # Dataset
    # -------------------------
    dataset = build_msa_dataset(
        opt=opt,
        mode=args.mode,
        task=args.task,
        need_proteins=args.need_proteins,
    )

    loader = make_eval_loader(
        dataset=dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        rank=getattr(args, "rank", 0),
        world_size=getattr(args, "world_size", 1),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    if is_main_process():
        print(f"[Data] split={args.mode}, global_size={len(dataset)}")
        print(f"[Data] local_batches(rank0)={len(loader)}")
        print(f"[Data] num_classes={args.num_classes}")

    # -------------------------
    # Model(s)
    # -------------------------
    # args.ckpt is treated as the trained / student checkpoint.
    student_model = Arch(opt)

    load_arch_checkpoint(
        model=student_model,
        ckpt_path=args.ckpt,
        strict_shape=args.strict_shape,
    )

    student_model = student_model.to(device)
    student_model.eval()

    if args.torch_compile:
        student_model = torch.compile(student_model)

    models = [student_model]
    model_weights = [1.0]
    normalized_model_weights = [1.0]

    if teacher_ckpt is not None:
        teacher_model = Arch(opt)

        load_arch_checkpoint(
            model=teacher_model,
            ckpt_path=teacher_ckpt,
            strict_shape=args.strict_shape,
        )

        teacher_model = teacher_model.to(device)
        teacher_model.eval()

        if args.torch_compile:
            teacher_model = torch.compile(teacher_model)

        models = [student_model, teacher_model]
        model_weights = [
            float(args.ensemble_student_weight),
            float(args.ensemble_teacher_weight),
        ]
        normalized_model_weights = _normalize_model_weights(
            model_weights,
            n_models=2,
        )

        if is_main_process():
            print(
                "[Ensemble] probability average enabled: "
                f"student_weight={normalized_model_weights[0]:.6f}, "
                f"teacher_weight={normalized_model_weights[1]:.6f}"
            )
    else:
        if is_main_process():
            print("[Ensemble] disabled; using student checkpoint only.")

    # -------------------------
    # Inference
    # -------------------------
    local_targs, local_preds = predict_dataset(
        models=models,
        model_weights=normalized_model_weights,
        loader=loader,
        device=device,
        num_classes=args.num_classes,
        permute_dims=tuple(args.permute_dims),
        no_amp=args.no_amp,
        need_proteins=args.need_proteins,
    )

    if is_main_process():
        print(f"[Predict] rank0 local preds: {tuple(local_preds.shape)}")

    # -------------------------
    # Collect
    # -------------------------
    full_targs, full_preds = collect_predictions(
        local_targs=local_targs,
        local_preds=local_preds,
        output_dir=output_dir,
        mode=args.mode,
        method=args.distributed_collect,
        keep_part_files=args.keep_part_files,
    )

    result = None

    # -------------------------
    # Metrics on rank0
    # -------------------------
    if is_main_process():
        if full_targs is None or full_preds is None:
            raise RuntimeError("Rank0 did not receive full predictions.")

        if full_targs.shape[0] != len(dataset):
            print(
                "[Warning] Number of collected samples does not match dataset length: "
                f"collected={full_targs.shape[0]}, dataset={len(dataset)}"
            )

        metrics = compute_evalperf_metrics(
            targs=full_targs,
            preds=full_preds,
            report_threshold=args.report_threshold,
            no_empty_labels=args.no_empty_labels,
            no_zero_classes=args.no_zero_classes,
        )

        ensemble_enabled = teacher_ckpt is not None

        result = {
            "task": args.task,
            "mode": args.mode,

            # Backward-compatible field.
            "checkpoint": str(Path(args.ckpt)),

            # Explicit fields.
            "student_checkpoint": str(Path(args.ckpt)),
            "teacher_checkpoint": str(Path(teacher_ckpt)) if ensemble_enabled else None,
            "ensemble_enabled": bool(ensemble_enabled),
            "ensemble_average_space": "probability",
            "ensemble_weights": (
                {
                    "student": float(normalized_model_weights[0]),
                    "teacher": float(normalized_model_weights[1]),
                }
                if ensemble_enabled
                else {
                    "student": 1.0,
                }
            ),

            "num_samples": int(full_targs.shape[0]),
            "num_classes": int(full_targs.shape[1]),
            "metrics_are_percent": True,
            "no_empty_labels": bool(args.no_empty_labels),
            "no_zero_classes": bool(args.no_zero_classes),
            **metrics,
        }

        save_json(result, output_dir / f"{args.mode}_metrics.json")

        if args.save_predictions:
            torch.save(
                {
                    "targs": full_targs,
                    "preds": full_preds,
                    "result": result,
                },
                output_dir / f"{args.mode}_predictions.pt",
            )

        print("=" * 80)
        print("[Result]")
        for k, v in result.items():
            print(f"{k}: {v}")
        print("=" * 80)

    if dist_is_initialized():
        dist.barrier()

    cleanup_distributed()

    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_argparser():
    parser = argparse.ArgumentParser(
        description="Evaluate MSA-GO checkpoint on ind_test/test split."
    )

    # Required
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=None,
        help=(
            "Optional teacher checkpoint. If provided, final prediction is "
            "weighted average of sigmoid probabilities from --ckpt and teacher."
        ),
    )
    parser.add_argument(
        "--ensemble_student_weight",
        type=float,
        default=1.0,
        help="Student / trained model weight before normalization.",
    )
    parser.add_argument(
        "--ensemble_teacher_weight",
        type=float,
        default=1.0,
        help="Teacher model weight before normalization.",
    )
    parser.add_argument("--file_address", type=str, required=True)
    parser.add_argument("--working_address", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Split
    parser.add_argument("--mode", type=str, default="ind_test")

    # Dataset/model compatibility
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--msa_max_size", type=int, default=None)
    parser.add_argument("--permute_dims", type=int, nargs=4, default=[0, 3, 2, 1])
    parser.add_argument("--torch_compile", action="store_true")

    # Binary MSA loader
    parser.add_argument(
        "--msa_read_mode",
        type=str,
        choices=["full", "rows", "block"],
        default="full",
    )
    parser.add_argument(
        "--msa_sample_strategy",
        type=str,
        choices=["random", "block", "head"],
        default="head",
    )
    parser.add_argument("--msa_shuffle_rows_at_getitem", action="store_true")
    parser.add_argument(
        "--no_msa_shuffle_rows_at_getitem",
        dest="msa_shuffle_rows_at_getitem",
        action="store_false",
    )
    parser.set_defaults(msa_shuffle_rows_at_getitem=False)

    parser.add_argument("--msa_cache_gb", type=float, default=0.0)
    parser.add_argument("--msa_max_open_files", type=int, default=256)
    parser.add_argument("--sample_seed", type=int, default=1)

    # Loader
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)

    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)

    # GPU
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Override opt.gpu_ids. Examples: auto, 0, 0,1, -1, keep.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no_amp", action="store_true")

    # Model-specific optional behavior
    parser.add_argument("--need_proteins", action="store_true")

    # Checkpoint loading
    parser.add_argument("--strict_shape", dest="strict_shape", action="store_true")
    parser.add_argument("--no_strict_shape", dest="strict_shape", action="store_false")
    parser.set_defaults(strict_shape=True)

    # Metrics
    parser.add_argument("--report_threshold", dest="report_threshold", action="store_true")
    parser.add_argument("--no_report_threshold", dest="report_threshold", action="store_false")
    parser.set_defaults(report_threshold=True)

    parser.add_argument("--no_empty_labels", action="store_true")
    parser.add_argument("--no_zero_classes", action="store_true")

    # Distributed collection
    parser.add_argument(
        "--distributed_collect",
        type=str,
        choices=["file", "all_gather_object"],
        default="file",
    )
    parser.add_argument("--keep_part_files", action="store_true")

    # Output
    parser.add_argument("--save_predictions", action="store_true")

    return parser


def main():
    parser = build_argparser()
    args, unknown = parser.parse_known_args()

    # Useful when launched by wrapper with --ddp-child.
    unknown = [x for x in unknown if x != "--ddp-child"]

    if len(unknown) > 0:
        print(f"[Warning] Ignoring unknown arguments: {unknown}")

    evaluate_one_task(args)


if __name__ == "__main__":
    main()