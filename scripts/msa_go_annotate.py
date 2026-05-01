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
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast, GradScaler
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
import experiments.msa as D  # noqa: E402


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
        json.dump(obj, f, indent=2, ensure_ascii=False)


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


def build_opt_from_config(args: argparse.Namespace) -> SimpleNamespace:
    """
    Load original model config and override runtime fields.

    This is important: Arch(opt) must be initialized with the same architecture
    parameters used by the teacher model.
    """
    cfg = load_pickle_config(args.model_config)

    # CLI/runtime fields have higher priority.
    runtime_overrides = {
        "file_address": args.file_address,
        "working_dir": args.working_dir,
        "task": normalize_task(args.task),
        "num_classes": args.num_classes,
        "top_k": args.top_k,
        "max_len": args.max_len,
        "msa_max_size": args.msa_max_size,
        "batch_size": args.batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
        "permute_dims": tuple(args.permute_dims),
        "torch_compile": args.torch_compile,
    }

    cfg.update(runtime_overrides)

    # Ensure commonly used fields exist.
    cfg.setdefault("mode", "train")
    cfg.setdefault("shuffle", True)
    cfg.setdefault("no_amp", args.no_amp)
    cfg.setdefault("device", args.device)

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
# Dataset helpers
# ---------------------------------------------------------------------

def build_msa_dataset(
    opt: SimpleNamespace,
    mode: str,
    task: str,
    need_proteins: bool = False,
):
    return D.MSADataset(
        opt.file_address,
        opt.working_dir,
        mode,
        task,
        opt.num_classes,
        opt.top_k,
        opt.max_len,
        need_proteins=need_proteins,
        msa_max_size=opt.msa_max_size,
    )


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def split_input(input_data):
    """
    Compatible with existing MSADataset:
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
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim > 2:
            h = h.flatten(1)
        z = self.net(h)
        return F.normalize(z, dim=-1)


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

            if isinstance(out, dict):
                logits = out["logits"]
                h = out["embedding"]
            else:
                logits, h = out

            z = self.proj(h)
            return logits, z

        return self.backbone(
            x,
            permute_dims=permute_dims,
            aug_params=aug_params,
        )

    def set_proteins(self, proteins):
        if hasattr(self.backbone, "set_proteins"):
            self.backbone.set_proteins(proteins)


def strip_state_dict_prefix(state_dict: dict) -> dict:
    new_sd = {}
    for k, v in state_dict.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module."):]
        if kk.startswith("backbone."):
            kk = kk[len("backbone."):]
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


def save_full_checkpoint(model: nn.Module, path: Union[str, Path], extra: Optional[dict] = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    m = model.module if isinstance(model, nn.DataParallel) else model
    obj = {
        "state_dict": m.state_dict(),
        "extra": extra or {},
    }
    torch.save(obj, path)


def save_backbone_checkpoint(model: nn.Module, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(m.backbone.state_dict(), path)


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
):
    target = target.float()
    mask = mask.float()

    if conf is None:
        conf = torch.ones_like(mask)
    else:
        conf = conf.float()

    if pos_only:
        mask = mask * (target > 0.5).float()

    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = mask * conf
    denom = weight.sum().clamp_min(1.0)
    return (raw * weight).sum() / denom


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

        z = F.normalize(z, dim=-1)
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

        ic = self.ic.to(device)
        if ic.numel() != c:
            raise ValueError(f"IC dimension mismatch: ic={ic.shape}, y={y.shape}")

        y_pos = ((y > 0.5) & (term_mask > 0.5)).float()
        y_ic = y_pos * ic[None, :]

        inter = y_ic @ y_pos.T
        size = y_ic.sum(dim=1)
        union = size[:, None] + size[None, :] - inter
        r = inter / union.clamp_min(self.eps)

        eye = torch.eye(n, device=device, dtype=torch.bool)
        r = r.masked_fill(eye, 0.0)
        r = torch.where(r >= self.min_r, r, torch.zeros_like(r))

        conf_num = (term_conf * y_pos * ic[None, :]).sum(dim=1)
        conf_den = (y_pos * ic[None, :]).sum(dim=1)
        sample_conf = conf_num / conf_den.clamp_min(self.eps)
        sample_conf = torch.where(conf_den > 0, sample_conf, torch.zeros_like(sample_conf))

        # true labels are fully trusted
        sample_conf = torch.where(is_true, torch.ones_like(sample_conf), sample_conf)

        true_i = is_true[:, None]
        true_j = is_true[None, :]
        pair_type_weight = torch.ones((n, n), device=device)

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

        logits = z @ z.T / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        logits = logits.masked_fill(eye, -1e9)

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        log_prob = log_prob.masked_fill(eye, 0.0)

        loss_per_anchor = -(pi * log_prob).sum(dim=1)
        return loss_per_anchor[valid].mean()


def hierarchy_violation_loss(logits: torch.Tensor, edges: Optional[torch.Tensor]):
    if edges is None or edges.numel() == 0:
        return logits.sum() * 0.0

    probs = torch.sigmoid(logits)
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


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_task(args: argparse.Namespace):
    set_seed(args.seed)

    task = normalize_task(args.task)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = build_opt_from_config(args)
    opt.mode = "train"

    print(f"[Device] {device}")
    print(f"[Task] {task}")
    print(f"[Config] {args.model_config}")
    print(f"[Init checkpoint] {args.init_ckpt}")
    print(f"[Dataset] {args.file_address}")
    print(f"[MSA] {args.working_dir}")
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

    pseudo_dataset = build_msa_dataset(
        opt=opt,
        mode="exp_train",
        task=task,
        need_proteins=args.need_proteins,
    )

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
    )

    pseudo_loader = make_loader(
        pseudo_dataset,
        batch_size=args.pseudo_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        drop_last=args.drop_last,
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
        )

    print(f"[Data] train={len(true_dataset)}, exp_train={len(pseudo_dataset)}")
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(len(true_loader), len(pseudo_loader))
    total_steps = max(1, steps_per_epoch * args.epochs)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.min_lr,
    )

    scaler = GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    # -------------------------
    # Loss components
    # -------------------------
    ic = load_ic(args, task, args.num_classes)
    go_con_loss = GOAwareSupConLoss(
        ic=ic,
        tau=args.contrast_tau,
        min_r=args.contrast_min_r,
        w_true_pseudo=args.w_true_pseudo,
        w_pseudo_pseudo=args.w_pseudo_pseudo,
    ).to(device)

    go_edges = load_go_edges(args.go_edges_path, device)

    permute_dims = tuple(args.permute_dims)

    best_val_loss = float("inf")

    # -------------------------
    # Training loop
    # -------------------------
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        true_iter = iter(true_loader)
        pseudo_iter = iter(pseudo_loader)

        running = {
            "loss": 0.0,
            "loss_true": 0.0,
            "loss_pseudo": 0.0,
            "loss_con": 0.0,
            "loss_hier": 0.0,
        }

        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}")

        for step in pbar:
            true_batch, true_iter = get_next_batch(true_iter, true_loader)
            pseudo_batch, pseudo_iter = get_next_batch(pseudo_iter, pseudo_loader)

            proteins_t, X_t, y_t = unpack_batch(true_batch)
            proteins_u, X_u, y_u = unpack_batch(pseudo_batch)

            X_t, y_t, X_u, y_u = move_to_device(
                X_t, y_t, X_u, y_u, device=device
            )

            y_t = y_t.float()
            y_u = y_u.float()

            b_t = X_t.shape[0]
            b_u = X_u.shape[0]

            X = torch.cat([X_t, X_u], dim=0)

            proteins = combine_proteins(proteins_t, proteins_u)
            if proteins is not None and hasattr(model, "set_proteins"):
                model.set_proteins(proteins)

            # true labels are fully trusted
            mask_t = torch.ones_like(y_t)
            conf_t = torch.ones_like(y_t)

            # pseudo labels from exp_train prop_annotations
            # Current MSADataset gives multi-hot y_u. Trust positive pseudo labels only by default.
            mask_u = (y_u > 0.5).float() if args.pseudo_pos_only else torch.ones_like(y_u)
            conf_u = pseudo_conf_from_labels(y_u, mode="binary")

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

            # MSA augmentation:
            # For now, no external tensor augmentation is applied here.
            # If Arch supports aug_params, fill weak_aug_params/strong_aug_params.
            weak_aug_params = None
            strong_aug_params = None

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                logits_w, z_w = model(
                    X,
                    permute_dims=permute_dims,
                    return_embedding=True,
                    aug_params=weak_aug_params,
                )

                logits_s, z_s = model(
                    X,
                    permute_dims=permute_dims,
                    return_embedding=True,
                    aug_params=strong_aug_params,
                )

                logits_t_w = logits_w[:b_t]
                logits_t_s = logits_s[:b_t]
                logits_u_s = logits_s[b_t:]

                loss_true = 0.5 * (
                    bce_with_logits(logits_t_w, y_t)
                    + bce_with_logits(logits_t_s, y_t)
                )

                loss_pseudo = masked_bce_with_logits(
                    logits=logits_u_s,
                    target=y_u,
                    mask=mask_u,
                    conf=conf_u,
                    pos_only=args.pseudo_pos_only,
                )

                z_all = torch.cat([z_w, z_s], dim=0)
                labels_rep = labels_for_con.repeat(2, 1)
                masks_rep = masks_for_con.repeat(2, 1)
                confs_rep = confs_for_con.repeat(2, 1)
                is_true_rep = is_true.repeat(2)

                loss_con = go_con_loss(
                    z=z_all,
                    y=labels_rep,
                    is_true=is_true_rep,
                    term_mask=masks_rep,
                    term_conf=confs_rep,
                )

                loss_hier = hierarchy_violation_loss(logits_s, go_edges)

                loss = (
                    loss_true
                    + args.lambda_u * loss_pseudo
                    + args.lambda_c * loss_con
                    + args.lambda_h * loss_hier
                )

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            running["loss"] += float(loss.detach().cpu())
            running["loss_true"] += float(loss_true.detach().cpu())
            running["loss_pseudo"] += float(loss_pseudo.detach().cpu())
            running["loss_con"] += float(loss_con.detach().cpu())
            running["loss_hier"] += float(loss_hier.detach().cpu())

            if step % args.log_interval == 0:
                denom = step + 1
                pbar.set_postfix(
                    {
                        "loss": running["loss"] / denom,
                        "true": running["loss_true"] / denom,
                        "pseudo": running["loss_pseudo"] / denom,
                        "con": running["loss_con"] / denom,
                        "hier": running["loss_hier"] / denom,
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

        epoch_log = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": scheduler.get_last_lr()[0],
        }

        for k, v in running.items():
            epoch_log[k] = v / max(1, steps_per_epoch)

        # Optional lightweight validation: BCE loss only.
        if val_loader is not None:
            val_loss = evaluate_bce_loss(
                model=model,
                loader=val_loader,
                device=device,
                permute_dims=permute_dims,
                no_amp=args.no_amp,
            )
            epoch_log["val_bce_loss"] = val_loss

            if val_loss < best_val_loss:
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

    save_full_checkpoint(model, output_dir / "semisup_full_last.pt", extra={"epoch": args.epochs})
    save_backbone_checkpoint(model, output_dir / "semisup_backbone_last.pt")

    print(f"[Done] saved to {output_dir}")


@torch.no_grad()
def evaluate_bce_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    permute_dims=(0, 3, 2, 1),
    no_amp: bool = False,
):
    model.eval()
    losses = []

    for batch in loader:
        proteins, X, y = unpack_batch(batch)
        X, y = move_to_device(X, y, device=device)
        y = y.float()

        if proteins is not None and hasattr(model, "set_proteins"):
            model.set_proteins(proteins)

        with autocast(device_type="cuda", enabled=(device.type == "cuda" and not no_amp)):
            logits = model(X, permute_dims=permute_dims, return_embedding=False)
            loss = bce_with_logits(logits, y)

        losses.append(float(loss.detach().cpu()))

    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_argparser():
    parser = argparse.ArgumentParser(
        description="LATENCE CA-TCC-like semi-supervised MSA-GO expansion training."
    )

    # Required
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--init_ckpt", type=str, required=True)
    parser.add_argument("--file_address", type=str, required=True)
    parser.add_argument("--working_dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Dataset/model compatibility
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--msa_max_size", type=int, default=None)
    parser.add_argument("--permute_dims", type=int, nargs=4, default=[0, 3, 2, 1])
    parser.add_argument("--torch_compile", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pseudo_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--drop_last", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
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
    parser.add_argument("--lambda_c", type=float, default=0.05)
    parser.add_argument("--lambda_h", type=float, default=0.0)

    # Contrastive loss
    parser.add_argument("--contrast_tau", type=float, default=0.1)
    parser.add_argument("--contrast_min_r", type=float, default=1e-6)
    parser.add_argument("--w_true_pseudo", type=float, default=0.7)
    parser.add_argument("--w_pseudo_pseudo", type=float, default=0.4)

    # Pseudo label treatment
    parser.add_argument("--pseudo_pos_only", action="store_true")

    # IC / GO hierarchy
    parser.add_argument("--ic_path", type=str, default=None)
    parser.add_argument("--ic_alpha", type=float, default=1.0)
    parser.add_argument("--ic_min_count", type=int, default=2)
    parser.add_argument("--go_edges_path", type=str, default=None)

    # Validation / logging / saving
    parser.add_argument("--no_validation", action="store_true")
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