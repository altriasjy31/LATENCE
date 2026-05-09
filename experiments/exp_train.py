#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiments/exp_train.py

LATENCE protein expansion training.

CA-TCC-inspired semi-supervised training for MSA-based GO annotation:

    L = L_true + lambda_u * L_pseudo + lambda_c * L_GOCon + lambda_h * L_hier

This script:
1. loads the original MSA-GO model config from pickle;
2. initializes Arch(opt) with the same architecture as the teacher model;
3. loads teacher/student initial checkpoint;
4. trains on train + exp_train pseudo-labeled data;
5. saves both full semi-supervised checkpoint and backbone-only checkpoint.
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import pickle
import argparse
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union
import contextlib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.amp.autocast_mode import autocast, 
from torch.amp import autocast, GradScaler
import torch.amp as amp
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
# import experiments.msa as D  # noqa: E402
from experiments.msabin import MSABinaryDataset
from experiments.msaprob import PseudoProbDataset
from loss_functions.loss import AsymmetricLossOptimized



TASKS = {
    "cc": "cellular_component",
    "mf": "molecular_function",
    "bp": "biological_process",
    "cellular_component": "cellular_component",
    "molecular_function": "molecular_function",
    "biological_process": "biological_process",
}


# ---------------------------------------------------------------------
# IO and config
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


def append_jsonl(obj: dict, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_pickle_config(path: Union[str, Path]) -> dict:
    cfg = load_pickle(path)
    if not isinstance(cfg, dict):
        raise TypeError(f"Model config pickle must contain a dict, got {type(cfg)}")
    return cfg


def normalize_task(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return TASKS[task]

def clean_optional_path(x):
    if x is None:
        return None

    s = str(x).strip()

    if s == "" or s.lower() in {"none", "null"}:
        return None

    return s


def build_opt_from_config(args: argparse.Namespace) -> SimpleNamespace:
    """
    Load original model config and override runtime fields.

    Important:
    - Architecture-related fields should mostly come from the teacher config.
    - CLI fields override only when explicitly provided.
    - Projection and augmentation fields must be injected into opt because
      SemiSupMSAGO and build_msa_aug_params read them from opt.
    """
    cfg = load_pickle_config(args.model_config)

    task = normalize_task(args.task)

    runtime_overrides = {
        "file_address": args.file_address,
        "working_address": args.working_address,
        "task": task,
        "num_classes": args.num_classes,
        "batch_size": args.batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
        "permute_dims": tuple(args.permute_dims),
        "torch_compile": args.torch_compile,
        "no_amp": args.no_amp,
        "device": args.device,

        # binary MSA dataset / loader
        "msa_read_mode": args.msa_read_mode,
        "msa_sample_strategy": args.msa_sample_strategy,
        "msa_shuffle_rows_at_getitem": args.msa_shuffle_rows_at_getitem,
        "msa_cache_gb": args.msa_cache_gb,
        "msa_max_open_files": args.msa_max_open_files,
        "sample_seed": args.sample_seed,

        # projection head
        "proj_in_dim": args.proj_in_dim,
        "proj_hidden_dim": args.proj_hidden_dim,
        "proj_dim": args.proj_dim,
        "proj_dropout": args.proj_dropout,
    }

    # Only override teacher architecture/data shape fields when explicitly provided.
    if args.top_k is not None:
        runtime_overrides["top_k"] = args.top_k

    if args.max_len is not None:
        runtime_overrides["max_len"] = args.max_len

    if args.msa_max_size is not None:
        runtime_overrides["msa_max_size"] = args.msa_max_size

    # MSA view augmentation parameters.
    aug_fields = [
        "weak_row_drop_p",
        "weak_col_mask_p",
        "weak_block_mask_p",
        "weak_block_mask_min",
        "weak_block_mask_max",
        "weak_n_blocks",
        "weak_shuffle_rows",
        "weak_min_keep_rows",
        "weak_noise_std",

        "strong_row_drop_p",
        "strong_col_mask_p",
        "strong_block_mask_p",
        "strong_block_mask_min",
        "strong_block_mask_max",
        "strong_n_blocks",
        "strong_shuffle_rows",
        "strong_min_keep_rows",
        "strong_noise_std",
    ]

    for name in aug_fields:
        runtime_overrides[name] = getattr(args, name)
    parsed_gpu_ids = parse_gpu_ids_arg(
        args.gpu_ids,
        args.device,
        local_rank=getattr(args, "local_rank", 0),
        distributed=getattr(args, "distributed", False),
    )

    if parsed_gpu_ids is not None:
        runtime_overrides["gpu_ids"] = parsed_gpu_ids

    cfg.update(runtime_overrides)

    # Ensure commonly used fields exist.
    cfg.setdefault("mode", "train")
    cfg.setdefault("shuffle", True)
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
            f"{missing}. Check model_config or provide CLI overrides."
        )

    return SimpleNamespace(**cfg)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

# ---------------------------------------------------------------------
# Distributed utils
# ---------------------------------------------------------------------

def dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not dist_is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist_is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def init_distributed_mode(args: argparse.Namespace) -> torch.device:
    """
    Initialize single-node/multi-node DDP from torchrun environment variables.

    For single-node multi-GPU:
        CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 run_exp_train.py
    """
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0

    has_torchrun_env = (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    )

    if has_torchrun_env:
        args.distributed = True
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available() and args.device != "cpu":
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
        )

        dist.barrier()

        rank0_print(
            f"[DDP] initialized: world_size={args.world_size}, backend={backend}"
        )

        return device

    device = get_device(args.device)

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    return device


def cleanup_distributed():
    if dist_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def reduce_float_mean(x: float, device: torch.device) -> float:
    if not dist_is_initialized():
        return float(x)

    t = torch.tensor([float(x)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return float(t.item())


def reduce_metric_dict_mean(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not dist_is_initialized():
        return metrics

    keys = sorted(metrics.keys())
    vals = torch.tensor(
        [float(metrics[k]) for k in keys],
        device=device,
        dtype=torch.float32,
    )

    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= get_world_size()

    return {
        k: float(v)
        for k, v in zip(keys, vals.detach().cpu().tolist())
    }

def parse_gpu_ids_arg(
    gpu_ids_arg: Optional[str],
    device_arg: str,
    local_rank: int = 0,
    distributed: bool = False,
) -> Optional[List[int]]:
    """
    Return:
        None: keep gpu_ids from config
        []: CPU / no CUDA
        [local_rank]: DDP single-process single-GPU binding
    """
    if distributed:
        if device_arg == "cpu":
            return []

        if gpu_ids_arg is not None:
            s = str(gpu_ids_arg).strip().lower()
            if s == "keep":
                return None
            if s in ("", "-1", "cpu", "none", "[]"):
                return []
            if s == "auto":
                return [int(local_rank)]

        # In DDP, do not reuse a fixed "0" for all ranks.
        return [int(local_rank)]

    if gpu_ids_arg is not None:
        s = str(gpu_ids_arg).strip()

        if s.lower() == "keep":
            return None

        if s.lower() == "auto":
            if device_arg == "cpu":
                return []
            return [0] if torch.cuda.is_available() else []

        if s in ("", "-1") or s.lower() in ("cpu", "none", "[]"):
            return []

        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

    if device_arg == "cpu":
        return []

    if device_arg.startswith("cuda"):
        return [0]

    if device_arg == "auto":
        return [0] if torch.cuda.is_available() else []

    return []

def ddp_all_gather_padded(x: torch.Tensor, pad_value=0):
    if not dist_is_initialized():
        sizes = torch.tensor([x.shape[0]], device=x.device, dtype=torch.long)
        return x, sizes, 0

    world_size = get_world_size()
    rank = get_rank()

    local_n = torch.tensor([x.shape[0]], device=x.device, dtype=torch.long)
    sizes_list = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes_list, local_n)

    sizes = torch.cat(sizes_list, dim=0)
    max_n = int(sizes.max().item())

    if x.shape[0] < max_n:
        pad_shape = (max_n - x.shape[0],) + tuple(x.shape[1:])
        pad = torch.full(
            pad_shape,
            fill_value=pad_value,
            dtype=x.dtype,
            device=x.device,
        )
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    gathered_pad = [torch.empty_like(x_pad) for _ in range(world_size)]
    dist.all_gather(gathered_pad, x_pad)

    gathered = []
    for r, t in enumerate(gathered_pad):
        gathered.append(t[: int(sizes[r].item())])

    gathered = torch.cat(gathered, dim=0)

    return gathered, sizes, rank


def gather_bank_with_local_grad(x: torch.Tensor, pad_value=0):
    """
    Remote ranks are detached.
    Local slice keeps gradient.

    This gives local-anchor/global-bank contrastive learning
    without backpropagating through remote rank tensors.
    """
    if not dist_is_initialized():
        sizes = torch.tensor([x.shape[0]], device=x.device, dtype=torch.long)
        return x, sizes, 0, 0

    gathered, sizes, rank = ddp_all_gather_padded(
        x.detach(),
        pad_value=pad_value,
    )

    start = int(sizes[:rank].sum().item())
    end = start + x.shape[0]

    gathered = torch.cat(
        [
            gathered[:start],
            x,
            gathered[end:],
        ],
        dim=0,
    )

    return gathered, sizes, rank, start

# ---------------------------------------------------------------------
# Dataset helpers
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
        sample_strategy=getattr(opt, "msa_sample_strategy", "random"),
        shuffle_rows_at_getitem=getattr(opt, "msa_shuffle_rows_at_getitem", True),

        cache_max_bytes=cache_max_bytes,
        max_open_files=getattr(opt, "msa_max_open_files", 256),

        sample_seed=getattr(opt, "sample_seed", 1),
        avoid_last_sample=True,
    )

def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 1,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
):
    sampler = DistributedShardShuffleBatchSampler(
        dataset.sample_shard_ids,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**kwargs)


def set_loader_epoch(loader: DataLoader, epoch: int):
    batch_sampler = getattr(loader, "batch_sampler", None)

    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)


def get_next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def split_input(input_data):
    """
    Compatible with existing MSABinaryDataset:
        batch = (X, y)
    or:
        batch = ([proteins, X], y)
    """
    proteins = None

    if isinstance(input_data, torch.Tensor):
        X = input_data
    else:
        proteins, X = input_data
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"Expected tensor X, got {type(X)}")

    return proteins, X


def unpack_batch(batch):
    if len(batch) != 2:
        raise ValueError("Expected batch format: (input_data, y)")
    input_data, y = batch
    proteins, X = split_input(input_data)
    return proteins, X, y

def unpack_pseudo_batch(batch):
    proteins, X, y_pack = unpack_batch(batch)

    prob = None

    if isinstance(y_pack, dict):
        if "hard_y" not in y_pack:
            raise KeyError("Pseudo batch dict must contain key 'hard_y'")

        y = y_pack["hard_y"]
        prob = y_pack.get("prob", None)

    else:
        y = y_pack

    return proteins, X, y, prob


def move_to_device(*xs, device: torch.device):
    out = []
    for x in xs:
        if x is None:
            out.append(None)
        else:
            out.append(x.to(device, non_blocking=True))
    return out


def combine_proteins(p1, p2):
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return list(p2)
    if p2 is None:
        return list(p1)
    return list(p1) + list(p2)

def subsample_contrastive(
    z,
    y,
    mask,
    conf,
    is_true,
    max_n: int,
):
    if max_n is None or int(max_n) <= 0 or z.shape[0] <= int(max_n):
        return z, y, mask, conf, is_true

    max_n = int(max_n)

    idx_true = torch.where(is_true.bool())[0]
    idx_pseudo = torch.where(~is_true.bool())[0]

    keep = []

    n_true_keep = min(len(idx_true), max_n // 2)

    if n_true_keep > 0:
        perm = torch.randperm(len(idx_true), device=z.device)[:n_true_keep]
        keep.append(idx_true[perm])

    remain = max_n - n_true_keep
    n_pseudo_keep = min(len(idx_pseudo), remain)

    if n_pseudo_keep > 0:
        perm = torch.randperm(len(idx_pseudo), device=z.device)[:n_pseudo_keep]
        keep.append(idx_pseudo[perm])

    if len(keep) == 0:
        return z[:0], y[:0], mask[:0], conf[:0], is_true[:0]

    idx = torch.cat(keep, dim=0)

    return z[idx], y[idx], mask[idx], conf[idx], is_true[idx]


def build_msa_aug_params(opt, strength: str):
    """
    Build aug_params for Arch.forward.

    strength:
        "weak" or "strong"
    """

    if strength == "weak":
        return {
            "aug_type": "msa_view",
            "msa_view_params": {
                "row_drop_p": getattr(opt, "weak_row_drop_p", 0.05),
                "col_mask_p": getattr(opt, "weak_col_mask_p", 0.02),
                "block_mask_p": getattr(opt, "weak_block_mask_p", 0.0),
                "block_mask_min": getattr(opt, "weak_block_mask_min", 0.01),
                "block_mask_max": getattr(opt, "weak_block_mask_max", 0.03),
                "n_blocks": getattr(opt, "weak_n_blocks", 1),
                "shuffle_rows": getattr(opt, "weak_shuffle_rows", False),
                "keep_query": True,
                "min_keep_rows": getattr(opt, "weak_min_keep_rows", 2),
                "mask_value": 0.0,
                "noise_std": getattr(opt, "weak_noise_std", 0.0),
            },
        }

    elif strength == "strong":
        return {
            "aug_type": "msa_view",
            "msa_view_params": {
                "row_drop_p": getattr(opt, "strong_row_drop_p", 0.20),
                "col_mask_p": getattr(opt, "strong_col_mask_p", 0.08),
                "block_mask_p": getattr(opt, "strong_block_mask_p", 0.30),
                "block_mask_min": getattr(opt, "strong_block_mask_min", 0.02),
                "block_mask_max": getattr(opt, "strong_block_mask_max", 0.06),
                "n_blocks": getattr(opt, "strong_n_blocks", 1),
                "shuffle_rows": getattr(opt, "strong_shuffle_rows", False),
                "keep_query": True,
                "min_keep_rows": getattr(opt, "strong_min_keep_rows", 2),
                "mask_value": 0.0,
                "noise_std": getattr(opt, "strong_noise_std", 0.0),
            },
        }

    else:
        raise ValueError(f"Unknown MSA augmentation strength: {strength}")


# ---------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def pool_features(self, h: torch.Tensor) -> torch.Tensor:
        """
        Convert backbone feature to [B, D].

        For timm ResNet-like models, forward_features often returns [B, C, H, W].
        In that case use global average pooling rather than flattening H*W.
        """
        if h.ndim == 4:
            h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        elif h.ndim == 3:
            h = h.mean(dim=1)
        elif h.ndim == 2:
            pass
        else:
            h = h.flatten(1)

        return h

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.pool_features(h)
        z = self.net(h)
        return F.normalize(z, dim=-1)

def _is_sequence_of_tensors(x):
    return (
        isinstance(x, (tuple, list))
        and len(x) > 0
        and all(isinstance(v, torch.Tensor) for v in x)
    )


class SemiSupMSAGO(nn.Module):
    """
    backbone: original MSA-GO Arch
    projection head: only used for GO-aware contrastive training
    """

    def __init__(self, opt: SimpleNamespace):
        super().__init__()
        self.backbone = Arch(opt)

        if not hasattr(opt, "proj_in_dim") or opt.proj_in_dim <= 0:
            raise ValueError(
                "Projection head requires --proj_in_dim. "
                "Set it to the embedding dimension returned by Arch(return_embedding=True)."
            )

        self.proj = ProjectionHead(
            in_dim=opt.proj_in_dim,
            hidden_dim=getattr(opt, "proj_hidden_dim", 1024),
            out_dim=getattr(opt, "proj_dim", 128),
            dropout=getattr(opt, "proj_dropout", 0.1),
        )

    def forward(
        self,
        x: torch.Tensor,
        permute_dims=(0, 3, 2, 1),
        return_embedding: bool = False,
        aug_params: Optional[dict] = None,
    ):
        if return_embedding:
            out = self.backbone(
                x,
                permute_dims=permute_dims,
                return_embedding=True,
                aug_params=aug_params,
            )

            # --------------------------------------------------------
            # New multi-view path:
            # backbone returns:
            #   (logits_v1, logits_v2, ...), (h_v1, h_v2, ...)
            # --------------------------------------------------------
            if (
                isinstance(out, (tuple, list))
                and len(out) == 2
                and _is_sequence_of_tensors(out[0])
                and _is_sequence_of_tensors(out[1])
            ):
                logits_views, h_views = out

                if len(logits_views) != len(h_views):
                    raise ValueError(
                        f"Number of logits views ({len(logits_views)}) and "
                        f"embedding views ({len(h_views)}) must match"
                    )

                z_views = tuple(self.proj(h) for h in h_views)
                return tuple(logits_views), z_views

            # --------------------------------------------------------
            # Existing dict path.
            # --------------------------------------------------------
            if isinstance(out, dict):
                logits = out["logits"]
                h = out["embedding"]

            elif isinstance(out, (tuple, list)):
                if len(out) == 2:
                    logits, h = out

                elif len(out) == 3:
                    # Current Arch mixup branch returns:
                    #     logits, y_mix, h
                    logits, _, h = out

                else:
                    raise ValueError(
                        f"Unexpected backbone return tuple length: {len(out)}"
                    )

            else:
                raise TypeError(
                    f"Unexpected backbone return type when return_embedding=True: {type(out)}"
                )

            z = self.proj(h)
            return logits, z

        return self.backbone(
            x,
            permute_dims=permute_dims,
            aug_params=aug_params,
        )


def strip_state_dict_prefix(state_dict: dict) -> dict:
    """
    Normalize checkpoint keys from possible wrappers:
        module.xxx
        backbone.xxx
        module.backbone.xxx
        xxx.module.yyy
    """
    new_sd = {}

    for k, v in state_dict.items():
        kk = k

        while kk.startswith("module."):
            kk = kk[len("module."):]

        if kk.startswith("backbone."):
            kk = kk[len("backbone."):]

        while kk.startswith("module."):
            kk = kk[len("module."):]

        # Handle inner DataParallel modules:
        # pre_model.module.xxx -> pre_model.xxx
        # rnet.module.xxx      -> rnet.xxx
        kk = kk.replace(".module.", ".")

        new_sd[kk] = v

    return new_sd


def load_backbone_checkpoint(
    model: SemiSupMSAGO,
    ckpt_path: Union[str, Path],
    strict_shape: bool = True,
):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    else:
        sd = raw

    sd = strip_state_dict_prefix(sd)

    backbone_sd = model.backbone.state_dict()
    load_sd = {}

    skipped = []
    for k, v in sd.items():
        if k in backbone_sd:
            if backbone_sd[k].shape == v.shape:
                load_sd[k] = v
            else:
                skipped.append((k, tuple(v.shape), tuple(backbone_sd[k].shape)))

    if strict_shape and len(load_sd) == 0:
        raise RuntimeError(f"No compatible tensors loaded from {ckpt_path}")

    missing, unexpected = model.backbone.load_state_dict(load_sd, strict=False)

    print(f"[Checkpoint] loaded {len(load_sd)} tensors into backbone from {ckpt_path}")
    if skipped:
        print(f"[Checkpoint] skipped {len(skipped)} tensors due to shape mismatch")
    if len(missing) > 0:
        print(f"[Checkpoint] missing keys in partial load: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[Checkpoint] unexpected keys in partial load: {len(unexpected)}")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model
def set_model_proteins(model: nn.Module, proteins):
    m = unwrap_model(model)

    if hasattr(m, "set_proteins"):
        m.set_proteins(proteins)


def wrap_model_for_distributed(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
):
    if not getattr(args, "distributed", False):
        return model

    ddp_kwargs = {
        "find_unused_parameters": bool(args.ddp_find_unused_parameters),
    }

    if bool(args.ddp_static_graph):
        ddp_kwargs["static_graph"] = True

    try:
        if device.type == "cuda":
            return DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                **ddp_kwargs,
            )

        return DDP(model, **ddp_kwargs)

    except TypeError:
        # For older PyTorch without static_graph.
        ddp_kwargs.pop("static_graph", None)

        if device.type == "cuda":
            return DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                **ddp_kwargs,
            )

        return DDP(model, **ddp_kwargs)
    


def save_full_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    extra: Optional[dict] = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    m = unwrap_model(model)
    obj = {
        "state_dict": m.state_dict(),
        "extra": extra or {},
    }
    torch.save(obj, path)


def save_backbone_checkpoint(model: nn.Module, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    m = unwrap_model(model)
    torch.save(m.backbone.state_dict(), path)

def extract_logits(model_output):
    if isinstance(model_output, dict):
        return model_output["logits"]

    if isinstance(model_output, (tuple, list)):
        return model_output[0]

    return model_output


def load_arch_checkpoint(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    strict_shape: bool = True,
):
    """
    Load an Arch checkpoint directly into an Arch model.

    Compatible with:
        original teacher Arch checkpoint
        semisup_backbone_*.pt
        semisup_full_*.pt after strip_state_dict_prefix()
    """
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    else:
        sd = raw

    sd = strip_state_dict_prefix(sd)

    model_sd = model.state_dict()
    load_sd = {}

    skipped = []
    ignored = []

    for k, v in sd.items():
        if k not in model_sd:
            ignored.append(k)
            continue

        if tuple(model_sd[k].shape) != tuple(v.shape):
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue

        load_sd[k] = v

    if strict_shape and len(load_sd) == 0:
        raise RuntimeError(f"No compatible tensors loaded from {ckpt_path}")

    missing, unexpected = model.load_state_dict(load_sd, strict=False)

    if is_main_process():
        print(f"[Arch checkpoint] loaded {len(load_sd)} tensors from {ckpt_path}")
        print(f"[Arch checkpoint] ignored={len(ignored)}, skipped_shape={len(skipped)}")
        if len(missing) > 0:
            print(f"[Arch checkpoint] missing after partial load: {len(missing)}")
        if len(unexpected) > 0:
            print(f"[Arch checkpoint] unexpected after partial load: {len(unexpected)}")


def save_arch_checkpoint(model: nn.Module, path: Union[str, Path]):
    """
    Save an Arch model state_dict.

    This is used for EMA teacher backbone.
    eval_ind_test.py can load this directly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


@torch.no_grad()
def ema_update_arch(
    ema_model: nn.Module,
    student_arch: nn.Module,
    decay: float,
):
    """
    EMA update:
        ema = decay * ema + (1 - decay) * student

    Non-floating buffers, e.g. num_batches_tracked, are copied directly.
    """
    decay = float(decay)

    ema_state = ema_model.state_dict()
    stu_state = student_arch.state_dict()

    for k, ema_v in ema_state.items():
        if k not in stu_state:
            continue

        stu_v = stu_state[k].detach()

        if torch.is_floating_point(ema_v):
            ema_v.mul_(decay).add_(stu_v.to(dtype=ema_v.dtype), alpha=1.0 - decay)
        else:
            ema_v.copy_(stu_v)


def set_batchnorm_eval(
    model: nn.Module,
    freeze_affine: bool = False,
):
    """
    Keep BatchNorm running_mean/running_var fixed during fine-tuning.

    Important:
        model.train() will put BN back into train mode,
        so this function should be called after every model.train().
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

            if freeze_affine:
                for p in m.parameters(recurse=False):
                    p.requires_grad_(False)


def build_student_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
):
    """
    Use smaller LR for pretrained backbone and optionally larger LR for projection head.

    model may be DDP-wrapped.
    """
    m = unwrap_model(model)

    backbone_lr = getattr(args, "backbone_lr", None)
    proj_lr = getattr(args, "proj_lr", None)

    if backbone_lr is None or float(backbone_lr) <= 0:
        backbone_lr = args.lr

    if proj_lr is None or float(proj_lr) <= 0:
        proj_lr = args.lr

    param_groups = []

    backbone_params = [
        p for p in m.backbone.parameters()
        if p.requires_grad
    ]

    proj_params = [
        p for p in m.proj.parameters()
        if p.requires_grad
    ]

    if len(backbone_params) > 0:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(backbone_lr),
                "name": "backbone",
            }
        )

    if len(proj_params) > 0:
        param_groups.append(
            {
                "params": proj_params,
                "lr": float(proj_lr),
                "name": "projection",
            }
        )

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(
        param_groups,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )


def sigmoid_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
):
    """
    Multi-label sigmoid distillation loss.

    This is more appropriate than softmax KL for GO multi-label prediction.

    L = BCEWithLogits(student_logits / T, sigmoid(teacher_logits / T)) * T^2
    """
    T = max(float(temperature), 1e-6)

    with torch.no_grad():
        teacher_probs = torch.sigmoid(teacher_logits.float() / T)

    loss = F.binary_cross_entropy_with_logits(
        student_logits.float() / T,
        teacher_probs,
        reduction="mean",
    )

    return loss * (T * T)


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------

def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor):
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    pos_only: bool = True,
    reduction_mode: str = "weighted_mean",
):
    """
    reduction_mode:
        weighted_mean:
            (raw * weight).sum() / weight.sum()
            当前默认，数值较小。

        batch_mean:
            (raw * weight).sum() / batch_size
            pseudo loss 会随每个 protein 的 pseudo positive 数量增加而增大。

        sum:
            (raw * weight).sum()
            不推荐，尺度太大。
    """
    logits = logits.float()
    target = target.float()
    mask = mask.float()

    if conf is None:
        conf = torch.ones_like(mask)
    else:
        conf = conf.float()

    if pos_only:
        mask = mask * (target > 0.5).float()

    raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )

    weight = mask * conf
    numerator = (raw * weight).sum()

    if reduction_mode == "weighted_mean":
        denom = weight.sum().clamp_min(1.0)
        return numerator / denom

    if reduction_mode == "batch_mean":
        denom = torch.tensor(
            logits.shape[0],
            device=logits.device,
            dtype=logits.dtype,
        ).clamp_min(1.0)
        return numerator / denom

    if reduction_mode == "sum":
        return numerator

    raise ValueError(f"Unknown reduction_mode: {reduction_mode}")


class GOAwareSupConLoss(nn.Module):
    """
    IC-weighted soft supervised contrastive loss.

    Positive weight:
        r_ij = IC-weighted Jaccard(y_i, y_j)

    Pair confidence:
        true-true > true-pseudo > pseudo-pseudo
    """
    def __init__(
        self,
        ic: torch.Tensor,
        tau: float = 0.1,
        min_r: float = 1e-6,
        w_true_pseudo: float = 0.7,
        w_pseudo_pseudo: float = 0.4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("ic", ic.float())
        self.tau = tau
        self.min_r = min_r
        self.w_true_pseudo = w_true_pseudo
        self.w_pseudo_pseudo = w_pseudo_pseudo
        self.eps = eps

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        is_true: torch.Tensor,
        term_mask: Optional[torch.Tensor] = None,
        term_conf: Optional[torch.Tensor] = None,
    ):
        device = z.device
        n, c = y.shape

        if n <= 1:
            return z.sum() * 0.0

        # Contrastive logits are more stable in fp32, especially under AMP.
        z = F.normalize(z.float(), dim=-1)
        y = y.float()
        is_true = is_true.bool()

        if term_mask is None:
            term_mask = torch.ones_like(y)
        else:
            term_mask = term_mask.float()

        if term_conf is None:
            term_conf = torch.ones_like(y)
        else:
            term_conf = term_conf.float()

        ic = self.ic.to(device=device, dtype=torch.float32)
        if ic.numel() != c:
            raise ValueError(f"IC dimension mismatch: ic={ic.shape}, y={y.shape}")

        # Trusted positive GO terms only.
        y_pos = ((y > 0.5) & (term_mask > 0.5)).float()

        y_ic = y_pos * ic[None, :]

        inter = y_ic @ y_pos.T
        size = y_ic.sum(dim=1)
        union = size[:, None] + size[None, :] - inter

        r = inter / union.clamp_min(self.eps)

        eye = torch.eye(n, device=device, dtype=torch.bool)
        r = r.masked_fill(eye, 0.0)
        r = torch.where(r >= self.min_r, r, torch.zeros_like(r))

        # Sample confidence: IC-weighted mean confidence over trusted positives.
        conf_num = (term_conf * y_pos * ic[None, :]).sum(dim=1)
        conf_den = (y_pos * ic[None, :]).sum(dim=1)

        sample_conf = conf_num / conf_den.clamp_min(self.eps)
        sample_conf = torch.where(
            conf_den > 0,
            sample_conf,
            torch.zeros_like(sample_conf),
        )

        # True labels are fully trusted.
        sample_conf = torch.where(
            is_true,
            torch.ones_like(sample_conf),
            sample_conf,
        )

        true_i = is_true[:, None]
        true_j = is_true[None, :]

        pair_type_weight = torch.ones_like(r)

        mixed = true_i ^ true_j
        pair_type_weight = torch.where(
            mixed,
            torch.full_like(pair_type_weight, self.w_true_pseudo),
            pair_type_weight,
        )

        pseudo_pseudo = (~true_i) & (~true_j)
        pair_type_weight = torch.where(
            pseudo_pseudo,
            torch.full_like(pair_type_weight, self.w_pseudo_pseudo),
            pair_type_weight,
        )

        pair_conf = torch.sqrt(
            sample_conf[:, None].clamp_min(0.0)
            * sample_conf[None, :].clamp_min(0.0)
        )

        weights = r * pair_type_weight * pair_conf
        weights = weights.masked_fill(eye, 0.0)

        row_sum = weights.sum(dim=1, keepdim=True)
        valid = row_sum.squeeze(1) > self.eps

        if valid.sum() == 0:
            return z.sum() * 0.0

        pi = weights / row_sum.clamp_min(self.eps)

        logits = z @ z.T / float(self.tau)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        neg_large = -torch.finfo(logits.dtype).max
        logits = logits.masked_fill(eye, neg_large)

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = log_prob.masked_fill(eye, 0.0)

        loss_per_anchor = -(pi * log_prob).sum(dim=1)

        return loss_per_anchor[valid].mean()

class GOAwareSupConLocalGlobalLoss(nn.Module):
    def __init__(
        self,
        ic: torch.Tensor,
        tau: float = 0.1,
        min_r: float = 1e-6,
        w_true_pseudo: float = 0.7,
        w_pseudo_pseudo: float = 0.4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("ic", ic.float())
        self.tau = tau
        self.min_r = min_r
        self.w_true_pseudo = w_true_pseudo
        self.w_pseudo_pseudo = w_pseudo_pseudo
        self.eps = eps

    def forward(
        self,
        z_anchor: torch.Tensor,
        y_anchor: torch.Tensor,
        is_true_anchor: torch.Tensor,
        mask_anchor: torch.Tensor,
        conf_anchor: torch.Tensor,
        z_bank: torch.Tensor,
        y_bank: torch.Tensor,
        is_true_bank: torch.Tensor,
        mask_bank: torch.Tensor,
        conf_bank: torch.Tensor,
        local_start: int = 0,
    ):
        if z_anchor.shape[0] <= 0 or z_bank.shape[0] <= 1:
            return z_anchor.float().sum() * 0.0

        device = z_anchor.device
        ic = self.ic.to(device=device, dtype=torch.float32)

        z_anchor = F.normalize(z_anchor.float(), dim=-1)
        z_bank = F.normalize(z_bank.float(), dim=-1)

        y_anchor = y_anchor.float()
        y_bank = y_bank.float()
        mask_anchor = mask_anchor.float()
        mask_bank = mask_bank.float()
        conf_anchor = conf_anchor.float()
        conf_bank = conf_bank.float()

        pos_anchor = ((y_anchor > 0.5) & (mask_anchor > 0.5)).float()
        pos_bank = ((y_bank > 0.5) & (mask_bank > 0.5)).float()

        anchor_ic = pos_anchor * ic[None, :]
        bank_ic = pos_bank * ic[None, :]

        inter = anchor_ic @ pos_bank.T
        size_a = anchor_ic.sum(dim=1)
        size_b = bank_ic.sum(dim=1)
        union = size_a[:, None] + size_b[None, :] - inter

        r = inter / union.clamp_min(self.eps)
        r = torch.where(r >= self.min_r, r, torch.zeros_like(r))

        na = z_anchor.shape[0]
        ng = z_bank.shape[0]

        self_idx = torch.arange(na, device=device) + int(local_start)
        valid_self = self_idx < ng

        if valid_self.any():
            r[
                torch.arange(na, device=device)[valid_self],
                self_idx[valid_self],
            ] = 0.0

        conf_mass_a = (conf_anchor * pos_anchor * ic[None, :]).sum(dim=1)
        conf_den_a = (pos_anchor * ic[None, :]).sum(dim=1).clamp_min(self.eps)
        sample_conf_a = conf_mass_a / conf_den_a

        conf_mass_b = (conf_bank * pos_bank * ic[None, :]).sum(dim=1)
        conf_den_b = (pos_bank * ic[None, :]).sum(dim=1).clamp_min(self.eps)
        sample_conf_b = conf_mass_b / conf_den_b

        sample_conf_a = torch.where(
            conf_den_a > 0,
            sample_conf_a,
            torch.zeros_like(sample_conf_a),
        )

        sample_conf_b = torch.where(
            conf_den_b > 0,
            sample_conf_b,
            torch.zeros_like(sample_conf_b),
        )

        sample_conf_a = torch.where(
            is_true_anchor.bool(),
            torch.ones_like(sample_conf_a),
            sample_conf_a,
        )

        sample_conf_b = torch.where(
            is_true_bank.bool(),
            torch.ones_like(sample_conf_b),
            sample_conf_b,
        )

        true_a = is_true_anchor.bool()[:, None]
        true_b = is_true_bank.bool()[None, :]

        pair_type_weight = torch.ones_like(r)

        mixed = true_a ^ true_b
        pair_type_weight = torch.where(
            mixed,
            torch.full_like(pair_type_weight, self.w_true_pseudo),
            pair_type_weight,
        )

        pseudo_pseudo = (~true_a) & (~true_b)
        pair_type_weight = torch.where(
            pseudo_pseudo,
            torch.full_like(pair_type_weight, self.w_pseudo_pseudo),
            pair_type_weight,
        )

        pair_conf = torch.sqrt(
            sample_conf_a[:, None].clamp_min(0.0)
            * sample_conf_b[None, :].clamp_min(0.0)
        )

        weights = r * pair_type_weight * pair_conf

        row_sum = weights.sum(dim=1, keepdim=True)
        valid = row_sum.squeeze(1) > self.eps

        if valid.sum() == 0:
            return z_anchor.float().sum() * 0.0

        pi = weights / row_sum.clamp_min(self.eps)

        logits = z_anchor @ z_bank.T / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        if valid_self.any():
            logits[
                torch.arange(na, device=device)[valid_self],
                self_idx[valid_self],
            ] = -1e9

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss_per_anchor = -(pi * log_prob).sum(dim=1)

        return loss_per_anchor[valid].mean()


def hierarchy_violation_loss(logits: torch.Tensor, edges: Optional[torch.Tensor]):
    if edges is None or edges.numel() == 0:
        return logits.float().sum() * 0.0

    probs = torch.sigmoid(logits.float())

    child = edges[:, 0]
    parent = edges[:, 1]

    return F.relu(probs[:, child] - probs[:, parent]).mean()


# ---------------------------------------------------------------------
# IC and GO edges
# ---------------------------------------------------------------------

def compute_ic_from_dataset(
    file_address: Union[str, Path],
    task: str,
    num_classes: int,
    modes: Tuple[str, ...] = ("train",),
    alpha: float = 1.0,
    min_count: int = 2,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute IC from train labels only by default.

    Do not use test/ind_test labels for IC to avoid leakage.
    """
    data = load_pickle(file_address)

    counts = np.zeros(num_classes, dtype=np.float64)
    n = 0

    for mode in modes:
        anns = data[mode][task]["prop_annotations"]
        for labels in anns:
            labels = [int(x) for x in set(labels) if 0 <= int(x) < num_classes]
            if len(labels) > 0:
                counts[labels] += 1
            n += 1

    freq = (counts + alpha) / (n + alpha)
    ic = -np.log(np.clip(freq, 1e-12, 1.0))
    ic[counts < min_count] = 0.0

    if normalize and ic.max() > 0:
        ic = ic / ic.max()

    return torch.tensor(ic, dtype=torch.float32)


def load_ic(args, task: str, num_classes: int) -> torch.Tensor:
    if args.ic_path is not None:
        ic = torch.load(args.ic_path, map_location="cpu").float()
        if ic.numel() != num_classes:
            raise ValueError(f"IC size mismatch: {ic.numel()} vs {num_classes}")
        return ic

    return compute_ic_from_dataset(
        file_address=args.file_address,
        task=task,
        num_classes=num_classes,
        modes=("train",),
        alpha=args.ic_alpha,
        min_count=args.ic_min_count,
        normalize=True,
    )


def load_go_edges(path: Optional[str], device: torch.device):
    if path is None:
        return None

    edges = torch.load(path, map_location="cpu").long()
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("GO edges must have shape [E, 2], each row = [child, parent]")

    return edges.to(device)


# ---------------------------------------------------------------------
# Pseudo label utilities
# ---------------------------------------------------------------------

def pseudo_conf_from_labels(
    pseudo_y: torch.Tensor,
    mode: str = "binary",
):
    """
    Current dataset stores prop_annotations only, no probabilities.
    Therefore confidence is approximated as 1 for pseudo positives.

    If later you save soft probabilities in dataset or side files, replace this.
    """
    if mode == "binary":
        return torch.ones_like(pseudo_y)
    raise ValueError(f"Unknown pseudo confidence mode: {mode}")

def build_pseudo_supervision(
    y_u: torch.Tensor,
    prob_u: Optional[torch.Tensor],
    args: argparse.Namespace,
    ic_device: torch.Tensor,
):
    """
    Build pseudo target / mask / confidence.

    Recommended modes:

    pseudo_prob_target = "hard":
        target = hard pseudo label
        conf   = teacher probability
        mask   = pseudo positive set

    pseudo_prob_target = "soft_pos":
        target = teacher probability
        conf   = teacher probability
        mask   = pseudo positive set

    For current issue where pseudo loss quickly vanishes, "soft_pos" is worth testing.
    """

    y_hard = y_u.float()

    if bool(args.pseudo_pos_only):
        mask = (y_hard > 0.5).float()
    else:
        mask = torch.ones_like(y_hard)

    pseudo_min_ic = float(getattr(args, "pseudo_min_ic", 0.0))

    if pseudo_min_ic >= 0.0:
        if ic_device.numel() != y_hard.shape[1]:
            raise ValueError(
                f"IC dimension mismatch: ic={ic_device.shape}, y={y_hard.shape}"
            )

        term_keep = (ic_device > pseudo_min_ic).float()
        mask = mask * term_keep[None, :]

    if prob_u is None:
        target = y_hard
        conf = torch.ones_like(y_hard)

    else:
        p = prob_u.float().clamp(0.0, 1.0)

        if p.shape != y_hard.shape:
            raise ValueError(
                f"Pseudo prob shape mismatch: prob={p.shape}, y={y_hard.shape}"
            )

        min_conf = float(getattr(args, "pseudo_prob_min_conf", 0.0))
        if min_conf > 0.0:
            mask = mask * (p >= min_conf).float()

        conf_power = float(getattr(args, "pseudo_prob_conf_power", 1.0))
        if conf_power <= 0.0:
            conf = torch.ones_like(p)
        else:
            conf = p.pow(conf_power)

        target_mode = str(getattr(args, "pseudo_prob_target", "hard"))

        if target_mode == "hard":
            target = y_hard

        elif target_mode == "soft_pos":
            target = p

        else:
            raise ValueError(f"Unknown pseudo_prob_target: {target_mode}")

    conf = conf * (mask > 0).float()

    return target, mask, conf


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_task(args: argparse.Namespace):
    task = normalize_task(args.task)
    device = init_distributed_mode(args)

    # Different ranks should not share exactly the same RNG stream.
    set_seed(args.seed + args.rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    opt = build_opt_from_config(args)
    opt.mode = "train"

    if is_main_process():
        print(f"[Device] {device}")
        print(f"[DDP] distributed={args.distributed}, rank={args.rank}, world_size={args.world_size}, local_rank={args.local_rank}")
        print(f"[Task] {task}")
        print(f"[Config] {args.model_config}")
        print(f"[Init checkpoint] {args.init_ckpt}")
        print(f"[Dataset] {args.file_address}")
        print(f"[MSA] {args.working_address}")
        print(f"[Output] {output_dir}")
    
        save_json(vars(args), output_dir / "args.json")
        save_json(vars(opt), output_dir / "merged_model_opt.json")

    # -------------------------
    # Datasets
    # -------------------------
    true_dataset = build_msa_dataset(
        opt=opt,
        mode="train",
        task=task,
        need_proteins=args.need_proteins,
    )

    pseudo_base_dataset = build_msa_dataset(
        opt=opt,
        mode="exp_train",
        task=task,
        need_proteins=args.need_proteins,
    )
    
    pseudo_prob_path = clean_optional_path(getattr(args, "pseudo_prob_path", None))
    args.pseudo_prob_path = pseudo_prob_path
    
    if pseudo_prob_path is not None:
        pseudo_dataset = PseudoProbDataset(
            base_dataset=pseudo_base_dataset,
            metadata_file=args.file_address,
            mode="exp_train",
            task=task,
            prob_path=pseudo_prob_path,
            num_classes=args.num_classes,
        )
    
        if is_main_process():
            print(
                "[Pseudo prob] enabled: "
                f"path={pseudo_prob_path}, "
                f"shape={pseudo_dataset.prob_shape}, "
                f"dtype={pseudo_dataset.prob_dtype}, "
                f"target={args.pseudo_prob_target}, "
                f"conf_power={args.pseudo_prob_conf_power}, "
                f"min_conf={args.pseudo_prob_min_conf}, "
                f"pseudo_min_ic={args.pseudo_min_ic}, "
                f"loss_reduction={args.pseudo_loss_reduction}"
            )
    
    else:
        pseudo_dataset = pseudo_base_dataset
    
        if is_main_process():
            print("[Pseudo prob] disabled; using hard prop_annotations only.")

    val_dataset = None
    if not args.no_validation:
        val_dataset = build_msa_dataset(
            opt=opt,
            mode=args.val_mode,
            task=task,
            need_proteins=False,
        )
    true_loader = make_loader(
        true_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.sampler_seed,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    pseudo_loader = make_loader(
        pseudo_dataset,
        batch_size=args.pseudo_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.sampler_seed + 17,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = make_loader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=args.pin_memory,
            drop_last=False,
            rank=args.rank,
            world_size=args.world_size,
            seed=args.sampler_seed + 1009,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
    if is_main_process():
        print(f"[Data] train={len(true_dataset)}, exp_train={len(pseudo_dataset)}")
        print(f"[Data] local steps: train_loader={len(true_loader)}, pseudo_loader={len(pseudo_loader)}")
        if val_dataset is not None:
            print(f"[Data] val({args.val_mode})={len(val_dataset)}")

    # -------------------------
    # Model
    # -------------------------
    model = SemiSupMSAGO(opt)
    
    if args.init_ckpt is not None:
        load_backbone_checkpoint(
            model=model,
            ckpt_path=args.init_ckpt,
            strict_shape=True,
        )
    
        model = model.to(device)
        
        # ------------------------------------------------------------
        # EMA teacher.
        # It is an Arch model, not SemiSupMSAGO, because it only provides logits.
        # Both student and EMA teacher start from the same original teacher checkpoint.
        # ------------------------------------------------------------
        teacher_ema = None
        
        if bool(args.ema_enabled):
            teacher_ema = Arch(opt)
        
            if args.init_ckpt is None:
                raise ValueError("EMA teacher requires args.init_ckpt to initialize.")
        
            load_arch_checkpoint(
                model=teacher_ema,
                ckpt_path=args.init_ckpt,
                strict_shape=True,
            )
        
            teacher_ema = teacher_ema.to(device)
            teacher_ema.eval()
            teacher_ema.requires_grad_(False)
        
            if is_main_process():
                print(
                    f"[EMA teacher] enabled, decay={args.ema_decay}, "
                    f"lambda_kd={args.lambda_kd}, temperature={args.kd_temperature}"
                )
        
        # Freeze BN before optimizer construction if affine parameters are also frozen.
        if bool(args.freeze_bn):
            set_batchnorm_eval(
                model,
                freeze_affine=bool(args.freeze_bn_affine),
            )
        
        model = wrap_model_for_distributed(model, args=args, device=device)
        
        optimizer = build_student_optimizer(model, args)

    steps_per_epoch = max(len(true_loader), len(pseudo_loader))
    total_steps = max(1, steps_per_epoch * args.epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.min_lr,
    )

    amp_enabled = device.type == "cuda" and not args.no_amp

    try:
        scaler = GradScaler(device="cuda", enabled=False)
    except TypeError:
        # For older torch.amp.GradScaler signatures.
        scaler = GradScaler(enabled=False)

    # -------------------------
    # Loss components
    # -------------------------
    ic = load_ic(args, task, args.num_classes)

    go_con_global_loss = GOAwareSupConLocalGlobalLoss(
        ic=ic,
        tau=args.contrast_tau,
        min_r=args.contrast_min_r,
        w_true_pseudo=args.w_true_pseudo,
        w_pseudo_pseudo=args.w_pseudo_pseudo,
    ).to(device)

    go_edges = load_go_edges(args.go_edges_path, device)

    true_loss_func = AsymmetricLossOptimized(
        gamma_neg=args.asl_gamma_neg,
        gamma_pos=args.asl_gamma_pos,
        clip=args.asl_clip,
        disable_torch_grad_focal_loss=True,
    )

    ic_device = ic.to(device=device, dtype=torch.float32)

    permute_dims = tuple(args.permute_dims)

    weak_aug_params = build_msa_aug_params(opt, "weak")
    strong_aug_params = build_msa_aug_params(opt, "strong")

    multi_view_aug_params = {
        "aug_type": "msa_multi_view",
        "views": [
            weak_aug_params["msa_view_params"],
            strong_aug_params["msa_view_params"],
        ],
        "return_format": "tuple",
    }

    best_val_loss = float("inf")

    # -------------------------
    # Training loop
    # -------------------------
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
    
        if bool(args.freeze_bn):
            set_batchnorm_eval(
                model,
                freeze_affine=False,
            )
    
        if teacher_ema is not None:
            teacher_ema.eval()

        set_loader_epoch(true_loader, epoch)
        set_loader_epoch(pseudo_loader, epoch)
    
        if val_loader is not None:
            set_loader_epoch(val_loader, epoch)

        true_iter = iter(true_loader)
        pseudo_iter = iter(pseudo_loader)

        running = {
            "loss": 0.0,
            "loss_true": 0.0,
            "loss_pseudo": 0.0,
            "loss_kd": 0.0,
            "loss_con": 0.0,
            "loss_hier": 0.0,
        
            "w_loss_pseudo": 0.0,
            "w_loss_kd": 0.0,
            "w_loss_con": 0.0,
            "w_loss_hier": 0.0,
        
            "pseudo_mask_terms": 0.0,
            "pseudo_conf_mean": 0.0,
        }
        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"epoch {epoch}/{args.epochs}",
            disable=not is_main_process(),
        )

        debug_anomaly = False
        anomaly_ctx = (
            torch.autograd.set_detect_anomaly(True)
            if debug_anomaly
            else contextlib.nullcontext()
        )
        with anomaly_ctx:
            for step in pbar:
                true_batch, true_iter = get_next_batch(true_iter, true_loader)
                pseudo_batch, pseudo_iter = get_next_batch(pseudo_iter, pseudo_loader)
    
                proteins_t, X_t, y_t = unpack_batch(true_batch)
                proteins_u, X_u, y_u, prob_u = unpack_pseudo_batch(pseudo_batch)
    
                X_t, y_t, X_u, y_u = move_to_device(
                    X_t, y_t, X_u, y_u, device=device
                )
                
                if prob_u is not None:
                    prob_u = prob_u.to(device, non_blocking=True)

                X_t = X_t.long()
                X_u = X_u.long()
    
                y_t = y_t.float()
                y_u = y_u.float()
    
                b_t = X_t.shape[0]
                b_u = X_u.shape[0]
    
                X = torch.cat([X_t, X_u], dim=0)
    
                proteins = combine_proteins(proteins_t, proteins_u)            
                if proteins is not None:
                    set_model_proteins(model, proteins)
    
                # true labels are fully trusted
                mask_t = torch.ones_like(y_t)
                conf_t = torch.ones_like(y_t)
    
                # pseudo labels from exp_train prop_annotations
                # Current MSABinaryDataset gives multi-hot y_u. Trust positive pseudo labels only by default.
                target_u, mask_u, conf_u = build_pseudo_supervision(
                    y_u=y_u,
                    prob_u=prob_u,
                    args=args,
                    ic_device=ic_device,
                )
    
                labels_for_con = torch.cat([y_t, y_u], dim=0)
                masks_for_con = torch.cat([mask_t, mask_u], dim=0)
                confs_for_con = torch.cat([conf_t, conf_u], dim=0)
    
                is_true = torch.cat(
                    [
                        torch.ones(b_t, dtype=torch.bool, device=device),
                        torch.zeros(b_u, dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
    
                optimizer.zero_grad(set_to_none=True)
                
                # ------------------------------------------------------------
                # EMA teacher target.
                #
                # Use clean/no-augmentation teacher prediction as a conservative anchor.
                # This adds one extra forward pass but strongly reduces student drift.
                # ------------------------------------------------------------
                teacher_logits = None
                
                do_kd = (
                    teacher_ema is not None
                    and float(args.lambda_kd) > 0.0
                    and int(args.kd_every_n_steps) > 0
                    and (global_step % int(args.kd_every_n_steps) == 0)
                )
                
                if do_kd:
                    teacher_ema.eval()
                
                    with torch.no_grad():
                        with autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=amp_enabled,
                        ):
                            teacher_out = teacher_ema(
                                X,
                                permute_dims=permute_dims,
                                aug_params=None,
                            )
                
                        teacher_logits = extract_logits(teacher_out).detach()
                
                # ------------------------------------------------------------
                # Student multi-view forward.
                # ------------------------------------------------------------
                with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    logits_views, z_views = model(
                        X,
                        permute_dims=permute_dims,
                        return_embedding=True,
                        aug_params=multi_view_aug_params,
                    )
                    if len(logits_views) != 2 or len(z_views) != 2:
                        raise RuntimeError(
                            f"Expected two views, got logits={len(logits_views)}, z={len(z_views)}"
                        )
                    logits_w, logits_s = logits_views
                    z_w, z_s = z_views
                    logits_t_w = logits_w[:b_t]
                    logits_t_s = logits_s[:b_t]
                    logits_u_s = logits_s[b_t:]
                    loss_true = 0.5 * (
                        true_loss_func(logits_t_w.float(), y_t.float())
                        + true_loss_func(logits_t_s.float(), y_t.float())
                    )
                    loss_pseudo = masked_bce_with_logits(
                        logits=logits_u_s,
                        target=target_u,
                        mask=mask_u,
                        conf=conf_u,
                        pos_only=False,
                        reduction_mode=args.pseudo_loss_reduction,
                    )
    
                    # ----------------------------------------------------
                    # GO-aware contrastive loss
                    # ----------------------------------------------------
                    # Remove samples without any trusted positive GO term from contrastive loss.
                    # Otherwise they still enter the denominator as implicit negatives.
                    trusted_pos = (
                        (labels_for_con > 0.5)
                        & (masks_for_con > 0.5)
                    ).float()
    
                    ic_mass = (trusted_pos * ic_device[None, :]).sum(dim=1)
                    has_pos = ic_mass > 0
    
                    if bool(has_pos.any().item()):
                        z_con = torch.cat(
                            [
                                z_w[has_pos],
                                z_s[has_pos],
                            ],
                            dim=0,
                        )
    
                        labels_for_con = torch.cat([y_t, y_u], dim=0)
                        masks_for_con = torch.cat([mask_t, mask_u], dim=0)
                        confs_for_con = torch.cat([conf_t, conf_u], dim=0)
                        is_true_con = is_true[has_pos].repeat(2)
    
                        z_con, labels_con, masks_con, confs_con, is_true_con = subsample_contrastive(
                            z_con,
                            labels_con,
                            masks_con,
                            confs_con,
                            is_true_con,
                            max_n=int(args.contrast_max_samples_per_rank),
                        )
                        
                        if (
                            bool(args.contrast_all_gather)
                            and dist_is_initialized()
                            and z_con.shape[0] > 0
                        ):
                            labels_local = labels_con.bool()
                            masks_local = masks_con.bool()
                            confs_local = confs_con.to(torch.float16)
                            is_true_local = is_true_con.to(torch.int8)
                        
                            z_bank, sizes, rank, local_start = gather_bank_with_local_grad(
                                z_con.float(),
                                pad_value=0.0,
                            )
                        
                            labels_bank, _, _ = ddp_all_gather_padded(
                                labels_local,
                                pad_value=False,
                            )
                        
                            masks_bank, _, _ = ddp_all_gather_padded(
                                masks_local,
                                pad_value=False,
                            )
                        
                            confs_bank, _, _ = ddp_all_gather_padded(
                                confs_local,
                                pad_value=0.0,
                            )
                        
                            is_true_bank, _, _ = ddp_all_gather_padded(
                                is_true_local,
                                pad_value=0,
                            )
                        
                            loss_con = go_con_global_loss(
                                z_anchor=z_con,
                                y_anchor=labels_local,
                                is_true_anchor=is_true_local.bool(),
                                mask_anchor=masks_local,
                                conf_anchor=confs_local,
                                z_bank=z_bank,
                                y_bank=labels_bank,
                                is_true_bank=is_true_bank.bool(),
                                mask_bank=masks_bank,
                                conf_bank=confs_bank,
                                local_start=local_start,
                            )
                        
                        else:
                            loss_con = go_con_loss(
                                z=z_con,
                                y=labels_con,
                                is_true=is_true_con,
                                term_mask=masks_con,
                                term_conf=confs_con,
                            )
                    else:
                        loss_con = z_w.float().sum() * 0.0
    
                    if teacher_logits is not None:
                        kd_terms = []
                    
                        if bool(args.kd_on_weak):
                            kd_terms.append(
                                sigmoid_kd_loss(
                                    student_logits=logits_w,
                                    teacher_logits=teacher_logits,
                                    temperature=args.kd_temperature,
                                )
                            )
                    
                        if bool(args.kd_on_strong):
                            kd_terms.append(
                                sigmoid_kd_loss(
                                    student_logits=logits_s,
                                    teacher_logits=teacher_logits,
                                    temperature=args.kd_temperature,
                                )
                            )
                    
                        if len(kd_terms) > 0:
                            loss_kd = sum(kd_terms) / float(len(kd_terms))
                        else:
                            loss_kd = logits_s.float().sum() * 0.0
                    else:
                        loss_kd = logits_s.float().sum() * 0.0
                    
                    loss_hier = hierarchy_violation_loss(logits_s, go_edges)
                    
                    loss = (
                        loss_true
                        + args.lambda_u * loss_pseudo
                        + args.lambda_kd * loss_kd
                        + args.lambda_c * loss_con
                        + args.lambda_h * loss_hier
                    )
                    
                scaler.scale(loss).backward()
    
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                
                next_global_step = global_step + 1
                
                # EMA update after student optimizer step.
                if (
                    teacher_ema is not None
                    and next_global_step >= int(args.ema_update_after_step)
                    and int(args.ema_update_every) > 0
                    and next_global_step % int(args.ema_update_every) == 0
                ):
                    ema_update_arch(
                        ema_model=teacher_ema,
                        student_arch=unwrap_model(model).backbone,
                        decay=float(args.ema_decay),
                    )
                
                scheduler.step()
                
                global_step = next_global_step

    
                running["loss"] += float(loss.detach().cpu())
                running["loss_true"] += float(loss_true.detach().cpu())
                running["loss_pseudo"] += float(loss_pseudo.detach().cpu())
                running["loss_kd"] += float(loss_kd.detach().cpu())
                running["loss_con"] += float(loss_con.detach().cpu())
                running["loss_hier"] += float(loss_hier.detach().cpu())

                with torch.no_grad():
                    pseudo_weight = mask_u * conf_u
                    pseudo_terms = mask_u.sum()
                    pseudo_conf_mean = (
                        pseudo_weight.sum() / pseudo_terms.clamp_min(1.0)
                    )
                
                running["w_loss_pseudo"] += float((args.lambda_u * loss_pseudo).detach().cpu())
                running["w_loss_kd"] += float((args.lambda_kd * loss_kd).detach().cpu())
                running["w_loss_con"] += float((args.lambda_c * loss_con).detach().cpu())
                running["w_loss_hier"] += float((args.lambda_h * loss_hier).detach().cpu())
                
                running["pseudo_mask_terms"] += float(pseudo_terms.detach().cpu())
                running["pseudo_conf_mean"] += float(pseudo_conf_mean.detach().cpu())
    
                if is_main_process() and step % args.log_interval == 0:
                    denom = step + 1
                    pbar.set_postfix(
                        {
                            "loss": running["loss"] / denom,
                            "true": running["loss_true"] / denom,
                            "pseudo": running["loss_pseudo"] / denom,
                            "kd": running["loss_kd"] / denom,
                            "con": running["loss_con"] / denom,
                            "hier": running["loss_hier"] / denom,
                            "lr": scheduler.get_last_lr()[0],
                            "w_pseudo": running["w_loss_pseudo"] / denom,
                            "mask": running["pseudo_mask_terms"] / denom,
                            "pconf": running["pseudo_conf_mean"] / denom,
                        }
                    )
        epoch_log = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": scheduler.get_last_lr()[0],
        }
        
        local_loss_means = {
            k: v / max(1, steps_per_epoch)
            for k, v in running.items()
        }
        
        reduced_loss_means = reduce_metric_dict_mean(local_loss_means, device)
        epoch_log.update(reduced_loss_means)

        # Optional lightweight validation: BCE loss only.        if val_loader is not None:
        if val_loader is not None:
            val_loss = evaluate_bce_loss(
                model=model,
                loader=val_loader,
                device=device,
                permute_dims=permute_dims,
                no_amp=args.no_amp,
            )
        
            epoch_log["val_bce_loss"] = val_loss
        
            if is_main_process() and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_full_checkpoint(
                    model,
                    output_dir / "semisup_full_best.pt",
                    extra={"epoch": epoch, "val_bce_loss": val_loss},
                )
                save_backbone_checkpoint(
                    model,
                    output_dir / "semisup_backbone_best.pt",
                )
            
        if is_main_process():
            append_jsonl(epoch_log, output_dir / "train_log.jsonl")
            print("[Epoch]", epoch_log)
        
            if epoch % args.save_interval == 0:
                save_full_checkpoint(
                    model,
                    output_dir / f"semisup_full_epoch{epoch}.pt",
                    extra={"epoch": epoch},
                )
            
                save_backbone_checkpoint(
                    model,
                    output_dir / f"semisup_backbone_epoch{epoch}.pt",
                )
            
                if teacher_ema is not None and bool(args.save_ema_checkpoint):
                    save_arch_checkpoint(
                        teacher_ema,
                        output_dir / f"semisup_ema_backbone_epoch{epoch}.pt",
                    )
        
        if args.distributed:
            dist.barrier()

    if is_main_process():
        save_full_checkpoint(
            model,
            output_dir / "semisup_full_last.pt",
            extra={"epoch": args.epochs},
        )
    
        save_backbone_checkpoint(
            model,
            output_dir / "semisup_backbone_last.pt",
        )
    
        if teacher_ema is not None and bool(args.save_ema_checkpoint):
            save_arch_checkpoint(
                teacher_ema,
                output_dir / "semisup_ema_backbone_last.pt",
            )
    
        print(f"[Done] saved to {output_dir}")
    
        print(f"[Done] saved to {output_dir}")
    
    if args.distributed:
        dist.barrier()
    
    cleanup_distributed()


@torch.no_grad()
def evaluate_bce_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    permute_dims=(0, 3, 2, 1),
    no_amp: bool = False,
):
    model.eval()

    amp_enabled = device.type == "cuda" and not no_amp

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_count = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        proteins, X, y = unpack_batch(batch)
        X, y = move_to_device(X, y, device=device)
        X = X.long()
        y = y.float()

        if proteins is not None:
            set_model_proteins(model, proteins)

        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits = model(X, permute_dims=permute_dims, return_embedding=False)
            loss_sum = F.binary_cross_entropy_with_logits(
                logits,
                y,
                reduction="sum",
            )

        total_loss += loss_sum.detach().double()
        total_count += torch.tensor(y.numel(), device=device, dtype=torch.float64)

    if dist_is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

    model.train()

    if total_count.item() == 0:
        return float("nan")

    return float((total_loss / total_count).item())

class DistributedShardShuffleBatchSampler(torch.utils.data.Sampler):
    """
    DDP-aware shard-grouped batch sampler.

    It builds global batches shard-by-shard, then assigns:
        rank 0: batches[0], batches[world_size], ...
        rank 1: batches[1], batches[world_size + 1], ...

    This keeps all ranks having the same number of batches by padding or dropping
    global batches.
    """

    def __init__(
        self,
        shard_ids,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        groups = {}

        for i, sid in enumerate(shard_ids):
            sid = int(sid)
            if sid not in groups:
                groups[sid] = []
            groups[sid].append(i)

        self.groups = groups
        self.shards = list(groups.keys())

        self._cache_epoch = None
        self._cache_batches = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._cache_epoch = None
        self._cache_batches = None

    def _build_global_batches(self):
        rng = random.Random(self.seed + self.epoch)

        shards = self.shards[:]

        if self.shuffle:
            rng.shuffle(shards)

        batches = []

        for sid in shards:
            indices = self.groups[sid][:]

            if self.shuffle:
                rng.shuffle(indices)

            batch = []

            for idx in indices:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    batches.append(batch)
                    batch = []

            if batch and not self.drop_last:
                batches.append(batch)

        if self.world_size > 1 and len(batches) > 0:
            rem = len(batches) % self.world_size

            if rem != 0:
                if self.drop_last:
                    batches = batches[: len(batches) - rem]
                else:
                    need = self.world_size - rem
                    batches.extend(batches[:need])

        return batches

    def _get_global_batches(self):
        if self._cache_epoch != self.epoch or self._cache_batches is None:
            self._cache_batches = self._build_global_batches()
            self._cache_epoch = self.epoch

        return self._cache_batches

    def __iter__(self):
        batches = self._get_global_batches()

        for i in range(self.rank, len(batches), self.world_size):
            yield batches[i]

    def __len__(self):
        batches = self._get_global_batches()

        if self.world_size <= 1:
            return len(batches)

        return len(batches) // self.world_size


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_argparser():
    parser = argparse.ArgumentParser(
        description="LATENCE semi-supervised MSA-GO expansion training."
    )

    # Required
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--init_ckpt", type=str, required=True)
    parser.add_argument("--file_address", type=str, required=True)
    parser.add_argument("--working_address", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Dataset/model compatibility.
    # Default None means: keep values from teacher config.
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
        default="random",
    )
    parser.add_argument("--msa_shuffle_rows_at_getitem", action="store_true")
    parser.add_argument("--no_msa_shuffle_rows_at_getitem", dest="msa_shuffle_rows_at_getitem", action="store_false")
    parser.set_defaults(msa_shuffle_rows_at_getitem=True)

    parser.add_argument("--msa_cache_gb", type=float, default=0.0)
    parser.add_argument("--msa_max_open_files", type=int, default=256)

    parser.add_argument("--sample_seed", type=int, default=1)
    parser.add_argument("--sampler_seed", type=int, default=1)

    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.set_defaults(persistent_workers=True)

    # GPU handling.
    # Default None means use [0] for cuda/auto, [] for cpu.
    # Use --gpu_ids keep if you explicitly want to keep teacher config gpu_ids.
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Override opt.gpu_ids. Examples: 0, 0,1, -1, keep.",)
    
    # DDP
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")
    parser.add_argument("--ddp_static_graph", action="store_true")

    # BatchNorm fine-tuning policy
    parser.add_argument("--freeze_bn", action="store_true")
    parser.add_argument("--no_freeze_bn", dest="freeze_bn", action="store_false")
    parser.set_defaults(freeze_bn=False)

    parser.add_argument("--freeze_bn_affine", action="store_true")
    parser.add_argument("--no_freeze_bn_affine", dest="freeze_bn_affine", action="store_false")
    parser.set_defaults(freeze_bn_affine=False)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pseudo_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--drop_last", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=None,
        help="LR for pretrained backbone. If None, use --lr.",
    )
    parser.add_argument(
        "--proj_lr",
        type=float,
        default=None,
        help="LR for projection head. If None, use --lr.",
    )
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no_amp", action="store_true")

    # Projection head
    parser.add_argument("--proj_in_dim", type=int, required=True)
    parser.add_argument("--proj_hidden_dim", type=int, default=1024)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_dropout", type=float, default=0.1)

    # Semi-supervised loss weights
    parser.add_argument("--lambda_u", type=float, default=0.5)
    parser.add_argument("--lambda_kd", type=float, default=0.0)
    parser.add_argument("--lambda_c", type=float, default=0.05)
    parser.add_argument("--lambda_h", type=float, default=0.0)

    # EMA teacher / distillation
    parser.add_argument("--ema_enabled", action="store_true")
    parser.add_argument("--no_ema", dest="ema_enabled", action="store_false")
    parser.set_defaults(ema_enabled=False)

    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--ema_update_every", type=int, default=1)

    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--kd_every_n_steps", type=int, default=1)

    parser.add_argument("--kd_on_weak", action="store_true")
    parser.add_argument("--no_kd_on_weak", dest="kd_on_weak", action="store_false")
    parser.set_defaults(kd_on_weak=True)

    parser.add_argument("--kd_on_strong", action="store_true")
    parser.add_argument("--no_kd_on_strong", dest="kd_on_strong", action="store_false")
    parser.set_defaults(kd_on_strong=True)

    parser.add_argument("--save_ema_checkpoint", action="store_true")
    parser.add_argument(
        "--no_save_ema_checkpoint",
        dest="save_ema_checkpoint",
        action="store_false",
    )
    parser.set_defaults(save_ema_checkpoint=True)

    # Contrastive loss
    parser.add_argument("--contrast_tau", type=float, default=0.1)
    parser.add_argument("--contrast_min_r", type=float, default=1e-6)
    parser.add_argument("--w_true_pseudo", type=float, default=0.7)
    parser.add_argument("--w_pseudo_pseudo", type=float, default=0.4)

    # ASL loss
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_gamma_pos", type=float, default=0.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)

    # Pseudo label treatment.
    # Default is positive-only to avoid treating unknown GO terms as negatives.
    parser.add_argument("--pseudo_pos_only", dest="pseudo_pos_only", action="store_true")
    parser.add_argument(
        "--use_pseudo_negatives",
        dest="pseudo_pos_only",
        action="store_false",
        help="Use all pseudo-label dimensions, including zeros, as supervised targets.",
    )
    parser.set_defaults(pseudo_pos_only=True)

    parser.add_argument("--pseudo_prob_path", type=str, default=None)
    
    parser.add_argument(
        "--pseudo_prob_target",
        type=str,
        choices=["hard", "soft_pos"],
        default="hard",
    )
    
    parser.add_argument(
        "--pseudo_prob_conf_power",
        type=float,
        default=1.0,
    )
    
    parser.add_argument(
        "--pseudo_prob_min_conf",
        type=float,
        default=0.0,
    )
    
    parser.add_argument(
        "--pseudo_min_ic",
        type=float,
        default=0.0,
    )
    
    parser.add_argument(
        "--pseudo_loss_reduction",
        type=str,
        choices=["weighted_mean", "batch_mean", "sum"],
        default="weighted_mean",
    )

    # IC / GO hierarchy
    parser.add_argument("--ic_path", type=str, default=None)
    parser.add_argument("--ic_alpha", type=float, default=1.0)
    parser.add_argument("--ic_min_count", type=int, default=2)
    parser.add_argument("--go_edges_path", type=str, default=None)

    # MSA weak augmentation
    parser.add_argument("--weak_row_drop_p", type=float, default=0.05)
    parser.add_argument("--weak_col_mask_p", type=float, default=0.02)
    parser.add_argument("--weak_block_mask_p", type=float, default=0.0)
    parser.add_argument("--weak_block_mask_min", type=float, default=0.01)
    parser.add_argument("--weak_block_mask_max", type=float, default=0.03)
    parser.add_argument("--weak_n_blocks", type=int, default=1)
    parser.add_argument("--weak_shuffle_rows", action="store_true")
    parser.add_argument("--weak_min_keep_rows", type=int, default=2)
    parser.add_argument("--weak_noise_std", type=float, default=0.0)

    # MSA strong augmentation
    parser.add_argument("--strong_row_drop_p", type=float, default=0.20)
    parser.add_argument("--strong_col_mask_p", type=float, default=0.08)
    parser.add_argument("--strong_block_mask_p", type=float, default=0.30)
    parser.add_argument("--strong_block_mask_min", type=float, default=0.02)
    parser.add_argument("--strong_block_mask_max", type=float, default=0.06)
    parser.add_argument("--strong_n_blocks", type=int, default=1)
    parser.add_argument("--strong_shuffle_rows", action="store_true")
    parser.add_argument("--strong_min_keep_rows", type=int, default=2)
    parser.add_argument("--strong_noise_std", type=float, default=0.0)

    # Constrative all gather
    parser.add_argument("--contrast_all_gather", action="store_true")
    parser.add_argument("--no_contrast_all_gather", dest="contrast_all_gather", action="store_false")
    parser.set_defaults(contrast_all_gather=False)
    
    parser.add_argument(
        "--contrast_max_samples_per_rank",
        type=int,
        default=0,
        help="0 means no subsampling before contrastive all_gather.",
    )

    # Validation / logging / saving.
    # Default no_validation=True to avoid accidentally selecting checkpoint on test set.
    parser.add_argument("--no_validation", dest="no_validation", action="store_true")
    parser.add_argument("--do_validation", dest="no_validation", action="store_false")
    parser.set_defaults(no_validation=True)

    parser.add_argument("--val_mode", type=str, default="test")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--need_proteins", action="store_true")

    return parser

def main():
    parser = build_argparser()
    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print(f"[Warning] Ignoring unknown arguments: {unknown}")

    args.task = normalize_task(args.task)

    train_one_task(args)


if __name__ == "__main__":
    main()