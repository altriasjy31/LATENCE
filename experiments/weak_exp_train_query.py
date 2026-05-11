#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiments/weak_exp_train.py

Weakly-supervised MSA-GO expansion training.

Purpose:
    Stress-test whether the teacher-initialized MSA-GO model can continue
    training on train + exp_train pseudo annotations without EMA/KD/contrastive
    machinery.

Key differences from experiments/exp_train.py:
    - single Arch model, no SemiSupMSAGO wrapper;
    - no projection head;
    - no EMA teacher/KD;
    - no contrastive loss;
    - one forward pass per step over train + exp_train;
    - OneCycleLR + ASL, closer to original MSA-GO training.

Loss:
    L = lambda_true * L_ASL(true)
      + lambda_pseudo * L_pseudo(prob-aware masked ASL/BCE)
      + lambda_h * L_hierarchy

This file intentionally reuses utilities from experiments.exp_train.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Union, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
MSA_ROOT = ROOT / "msa_models"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MSA_ROOT) not in sys.path:
    sys.path.insert(0, str(MSA_ROOT))

from models import Arch  # noqa: E402
from loss_functions.loss import AsymmetricLossOptimized  # noqa: E402
from experiments.msaprob import PseudoProbDataset  # noqa: E402

# Reuse mature utilities from exp_train.py.
from experiments.exp_train import (  # noqa: E402
    TASKS,
    load_pickle,
    load_pickle_config,
    save_json,
    append_jsonl,
    normalize_task,
    clean_optional_path,
    set_seed,
    init_distributed_mode,
    cleanup_distributed,
    dist_is_initialized,
    is_main_process,
    rank0_print,
    get_rank,
    get_world_size,
    reduce_metric_dict_mean,
    parse_gpu_ids_arg,
    build_msa_dataset,
    make_loader,
    set_loader_epoch,
    get_next_batch,
    unpack_batch,
    unpack_pseudo_batch,
    move_to_device,
    combine_proteins,
    set_model_proteins,
    set_batchnorm_eval,
    load_arch_checkpoint,
    save_arch_checkpoint,
    unwrap_model,
    load_ic,
    load_go_edges,
    hierarchy_violation_loss,
)


def build_weak_opt_from_config(args: argparse.Namespace) -> SimpleNamespace:
    """
    Build opt for Arch(opt), preserving teacher architecture fields while
    overriding dataset/runtime fields for weak expansion training.
    """
    cfg = load_pickle_config(args.model_config)
    task = normalize_task(args.task)

    overrides = {
        "file_address": args.file_address,
        "working_address": args.working_address,
        "task": task,
        "num_classes": int(args.num_classes),
        "batch_size": int(args.batch_size),
        "dataloader_num_workers": int(args.dataloader_num_workers),
        "permute_dims": tuple(args.permute_dims),
        "torch_compile": bool(args.torch_compile),
        "no_amp": bool(args.no_amp),
        "device": args.device,
        "msa_read_mode": args.msa_read_mode,
        "msa_sample_strategy": args.msa_sample_strategy,
        "msa_shuffle_rows_at_getitem": bool(args.msa_shuffle_rows_at_getitem),
        "msa_cache_gb": float(args.msa_cache_gb),
        "msa_max_open_files": int(args.msa_max_open_files),
        "sample_seed": int(args.sample_seed),
    }

    if args.top_k is not None:
        overrides["top_k"] = int(args.top_k)
    if args.max_len is not None:
        overrides["max_len"] = int(args.max_len)
    if args.msa_max_size is not None:
        overrides["msa_max_size"] = args.msa_max_size

    parsed_gpu_ids = parse_gpu_ids_arg(
        args.gpu_ids,
        args.device,
        local_rank=getattr(args, "local_rank", 0),
        distributed=getattr(args, "distributed", False),
    )
    if parsed_gpu_ids is not None:
        overrides["gpu_ids"] = parsed_gpu_ids

    cfg.update(overrides)
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
        raise ValueError(f"Missing required fields for Arch(opt): {missing}")

    return SimpleNamespace(**cfg)


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    reduction_mode: str = "batch_mean",
):
    logits = logits.float()
    target = target.float()
    mask = mask.float()
    conf = torch.ones_like(mask) if conf is None else conf.float()

    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = mask * conf
    numerator = (raw * weight).sum()

    if reduction_mode == "weighted_mean":
        return numerator / weight.sum().clamp_min(1.0)
    if reduction_mode == "batch_mean":
        return numerator / torch.tensor(
            logits.shape[0], device=logits.device, dtype=logits.dtype
        ).clamp_min(1.0)
    if reduction_mode == "sum":
        return numerator
    raise ValueError(f"Unknown reduction_mode: {reduction_mode}")


def masked_asl_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    eps: float = 1e-8,
    reduction: str = "batch_mean",
    disable_torch_grad_focal_loss: bool = True,
):
    """Element-wise masked ASL. Only mask>0 positions contribute."""
    logits = logits.float()
    target = target.float()
    mask = mask.float()
    conf = torch.ones_like(mask) if conf is None else conf.float()

    xs_pos = torch.sigmoid(logits)
    xs_neg = 1.0 - xs_pos

    if clip is not None and float(clip) > 0:
        xs_neg = (xs_neg + float(clip)).clamp(max=1.0)

    pos_loss = target * torch.log(xs_pos.clamp(min=eps))
    neg_loss = (1.0 - target) * torch.log(xs_neg.clamp(min=eps))
    loss = pos_loss + neg_loss

    if gamma_neg > 0 or gamma_pos > 0:
        if disable_torch_grad_focal_loss:
            with torch.no_grad():
                pt = xs_pos * target + xs_neg * (1.0 - target)
                gamma = gamma_pos * target + gamma_neg * (1.0 - target)
                focal_weight = torch.pow(1.0 - pt, gamma)
        else:
            pt = xs_pos * target + xs_neg * (1.0 - target)
            gamma = gamma_pos * target + gamma_neg * (1.0 - target)
            focal_weight = torch.pow(1.0 - pt, gamma)
        loss = loss * focal_weight

    loss = -loss
    weight = mask * conf
    weighted = loss * weight

    if reduction == "weighted_mean" or reduction == "mean_mask":
        return weighted.sum() / weight.sum().clamp_min(1.0)
    if reduction == "batch_mean":
        return weighted.sum() / torch.tensor(
            logits.shape[0], device=logits.device, dtype=logits.dtype
        ).clamp_min(1.0)
    if reduction == "sum":
        return weighted.sum()
    raise ValueError(f"Unknown ASL reduction: {reduction}")


def _topk_mask_per_sample(candidate_mask: torch.Tensor, score: torch.Tensor, topk: int):
    if topk is None or int(topk) <= 0:
        return candidate_mask
    b, c = candidate_mask.shape
    k = min(int(topk), c)
    masked_score = score.masked_fill(~candidate_mask, -1e9)
    idx = torch.topk(masked_score, k=k, dim=1).indices
    valid = torch.gather(candidate_mask, dim=1, index=idx)
    out = torch.zeros_like(candidate_mask)
    out.scatter_(dim=1, index=idx, src=valid)
    return out


def build_pseudo_supervision(
    logits_u: torch.Tensor,
    y_u: torch.Tensor,
    prob_u: Optional[torch.Tensor],
    args: argparse.Namespace,
    ic_device: torch.Tensor,
):
    """
    Build target/mask/conf for pseudo loss.

    Safe default:
        positives = pseudo annotations
        confidence = teacher probability
        negatives = optional teacher-prob-low hard negatives
    """
    y_hard = y_u.float()
    device = y_hard.device
    pos_mask = y_hard > 0.5

    pseudo_min_ic = float(getattr(args, "pseudo_min_ic", 0.0))
    if pseudo_min_ic > 0:
        term_keep = ic_device > pseudo_min_ic
        pos_mask = pos_mask & term_keep[None, :]
    else:
        term_keep = torch.ones(y_hard.shape[1], dtype=torch.bool, device=device)

    target = torch.zeros_like(y_hard)
    mask = torch.zeros_like(y_hard)
    conf = torch.zeros_like(y_hard)

    if prob_u is not None:
        p = prob_u.float().clamp(0.0, 1.0)
        if p.shape != y_hard.shape:
            raise ValueError(f"prob_u shape mismatch: prob={p.shape}, y={y_hard.shape}")

        conf_power = float(getattr(args, "pseudo_prob_conf_power", 1.0))
        pos_conf = p.pow(conf_power)

        pos_keep = pos_mask
        min_conf = float(getattr(args, "pseudo_prob_min_conf", 0.0))
        if min_conf > 0:
            pos_keep = pos_keep & (p >= min_conf)

        target_mode = str(getattr(args, "pseudo_prob_target", "hard"))
        if target_mode == "hard":
            target = torch.where(pos_keep, torch.ones_like(target), target)
        elif target_mode == "soft_pos":
            target = torch.where(pos_keep, p, target)
        else:
            raise ValueError(f"Unknown pseudo_prob_target: {target_mode}")

        mask = torch.where(pos_keep, torch.ones_like(mask), mask)
        conf = torch.where(pos_keep, pos_conf, conf)

        neg_policy = str(getattr(args, "pseudo_negative_policy", "none"))
        if neg_policy == "prob_low":
            neg_max = float(getattr(args, "pseudo_prob_neg_max", 0.01))
            neg_mask = (~pos_mask) & (p <= neg_max) & term_keep[None, :]
            neg_topk = int(getattr(args, "pseudo_neg_topk", 0))
            if neg_topk > 0:
                student_prob = torch.sigmoid(logits_u.detach().float())
                neg_mask = _topk_mask_per_sample(neg_mask, student_prob, neg_topk)
            neg_conf = (1.0 - p).pow(conf_power)
            target = torch.where(neg_mask, torch.zeros_like(target), target)
            mask = torch.where(neg_mask, torch.ones_like(mask), mask)
            conf = torch.where(neg_mask, neg_conf, conf)
        elif neg_policy == "none":
            pass
        else:
            raise ValueError(f"Unknown pseudo_negative_policy: {neg_policy}")
    else:
        target = torch.where(pos_mask, torch.ones_like(target), target)
        mask = torch.where(pos_mask, torch.ones_like(mask), mask)
        conf = torch.where(pos_mask, torch.ones_like(conf), conf)

    with torch.no_grad():
        mask_terms = mask.sum()
        pos_terms = pos_mask.float().sum()
        conf_mean = (conf * mask).sum() / mask_terms.clamp_min(1.0)
        neg_terms = mask_terms - pos_terms.clamp(max=mask_terms)
        stats = {
            "pseudo_mask_terms": float(mask_terms.detach().cpu()),
            "pseudo_pos_terms": float(pos_terms.detach().cpu()),
            "pseudo_neg_terms": float(neg_terms.detach().cpu()),
            "pseudo_conf_mean": float(conf_mean.detach().cpu()),
        }

    return target, mask, conf, stats



# ---------------------------------------------------------------------
# Optional DETR-style top-k ontology query decoder
# ---------------------------------------------------------------------

def _unwrap_module(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def pool_backbone_feature(h: torch.Tensor) -> torch.Tensor:
    """Pool rnet.forward_features output to [B, D]."""
    if h.ndim == 4:
        return F.adaptive_avg_pool2d(h, 1).flatten(1)
    if h.ndim == 3:
        return h.mean(dim=1)
    if h.ndim == 2:
        return h
    return h.flatten(1)

def build_backbone_memory(
    h: torch.Tensor,
    mode: str = "tokens_plus_pooled",
    memory_grid_h: int = 0,
    memory_grid_w: int = 0,
):
    """
    Convert backbone feature to decoder memory.

    Returns
    -------
    memory:
        [B, T, C]
    pooled:
        [B, C]

    mode:
        pooled:
            T = 1
        tokens:
            T = H * W
        tokens_plus_pooled:
            T = 1 + H * W
    """

    if h.ndim == 4:
        # h: [B, C, H, W]

        if memory_grid_h > 0 and memory_grid_w > 0:
            h = F.adaptive_avg_pool2d(h, (memory_grid_h, memory_grid_w))

        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)  # [B, C]

        tokens = h.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, C]

        if mode == "pooled":
            memory = pooled.unsqueeze(1)  # [B, 1, C]

        elif mode == "tokens":
            memory = tokens

        elif mode == "tokens_plus_pooled":
            memory = torch.cat(
                [
                    pooled.unsqueeze(1),
                    tokens,
                ],
                dim=1,
            )

        else:
            raise ValueError(f"Unknown memory mode: {mode}")

        return memory, pooled

    if h.ndim == 3:
        # Usually already token-like.
        # Assume h is [B, T, C].
        memory = h
        pooled = h.mean(dim=1)
        return memory, pooled

    if h.ndim == 2:
        # Already pooled feature [B, C].
        pooled = h
        memory = h.unsqueeze(1)
        return memory, pooled

    # Fallback.
    pooled = h.flatten(1)
    memory = pooled.unsqueeze(1)
    return memory, pooled


def find_classifier_linear(arch_model: nn.Module, num_classes: int) -> nn.Linear:
    """
    Find the final classifier linear layer in Arch/rnet.
    This is used as ontology query embedding source: W[class_id].
    """
    m = _unwrap_module(arch_model)

    candidates = []

    # Prefer Arch.get_fc() when available.
    if hasattr(m, "get_fc"):
        try:
            fc = m.get_fc()
            if isinstance(fc, nn.Linear) and fc.out_features == int(num_classes):
                return fc
            for sub in fc.modules():
                if isinstance(sub, nn.Linear) and sub.out_features == int(num_classes):
                    candidates.append(sub)
        except Exception:
            pass

    # Fallback: search the whole model; choose the last matching linear.
    for sub in m.modules():
        if isinstance(sub, nn.Linear) and sub.out_features == int(num_classes):
            candidates.append(sub)

    if not candidates:
        raise RuntimeError(
            f"Cannot find classifier nn.Linear with out_features={num_classes}. "
            "Top-k ontology decoder requires classifier weights as GO queries."
        )

    return candidates[-1]


class OntologyQueryDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        decoder_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        memory_mode: str = "tokens_plus_pooled",
        memory_grid_h: int = 0,
        memory_grid_w: int = 0,
        detach_query_weight: bool = True,
        delta_max: float = 0.5,
    ):
        super().__init__()

        self.delta_max = float(delta_max)

        self.memory_mode = memory_mode
        self.memory_grid_h = int(memory_grid_h)
        self.memory_grid_w = int(memory_grid_w)
        self.detach_query_weight = bool(detach_query_weight)

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, decoder_dim),
        )

        self.query_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, decoder_dim),
        )

        layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=layer,
            num_layers=num_layers,
        )

        self.out = nn.Linear(decoder_dim, 1)

        # Critical: residual decoder initially does nothing.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        h: torch.Tensor,
        base_logits: torch.Tensor,
        classifier_weight: torch.Tensor,
        topk_idx: torch.Tensor,
    ):
        """
        h:
            backbone feature before pooling.
            Usually [B, C, H, W].

        base_logits:
            [B, num_classes]

        classifier_weight:
            [num_classes, C]

        topk_idx:
            [B, K]
        """

        memory_tokens, pooled_feature = build_backbone_memory(
            h,
            mode=self.memory_mode,
            memory_grid_h=self.memory_grid_h,
            memory_grid_w=self.memory_grid_w,
        )

        memory = self.memory_proj(memory_tokens)  # [B, T, D]

        query_weight = classifier_weight[topk_idx]  # [B, K, C]

        if self.detach_query_weight:
            query_weight = query_weight.detach()

        query = self.query_proj(query_weight)  # [B, K, D]

        decoded = self.decoder(
            tgt=query,
            memory=memory,
        )

        # delta = self.out(decoded).squeeze(-1)  # [B, K]
        raw_delta = self.out(decoded).squeeze(-1)
        delta = self.delta_max * torch.tanh(raw_delta)

        refined_logits = base_logits.clone()
        refined_logits.scatter_add_(
            dim=1,
            index=topk_idx,
            src=delta,
        )

        return {
            "logits": refined_logits,
            "base_logits": base_logits,
            "topk_idx": topk_idx,
            "delta": delta,
            "memory_tokens": memory_tokens,
            "pooled_feature": pooled_feature,
        }

class WeakMSAGOWithQueryDecoder(nn.Module):
    """
    Arch + optional top-k ontology query decoder.

    At train time:
        - true samples use base self top-k, optionally boosted by true labels;
        - pseudo samples can use teacher prob top-k, boosted by pseudo labels.

    At eval time:
        - no teacher prob is available;
        - use base logits top-k for refinement.
    """

    def __init__(self, opt: SimpleNamespace, args: argparse.Namespace):
        super().__init__()
        self.backbone = Arch(opt)
        self.num_classes = int(args.num_classes)
        self.topk = int(args.query_decoder_topk)
        self.topk_source = str(args.query_decoder_topk_source)
        self.include_label_boost = bool(args.query_decoder_include_label_boost)
        self.label_boost = float(args.query_decoder_label_boost)

        classifier = find_classifier_linear(self.backbone, self.num_classes)
        feature_dim = int(classifier.in_features)
        self.classifier = classifier

        self.query_decoder = OntologyQueryDecoder(
            feature_dim=feature_dim,
            decoder_dim=args.query_decoder_dim,
            num_heads=args.query_decoder_heads,
            num_layers=args.query_decoder_layers,
            ffn_dim=args.query_decoder_ffn_dim,
            dropout=args.query_decoder_dropout,
            memory_mode=args.query_decoder_memory_mode,
            memory_grid_h=args.query_decoder_memory_grid_h,
            memory_grid_w=args.query_decoder_memory_grid_w,
            detach_query_weight=args.query_decoder_detach_query_weight,
        )

    @torch.no_grad()
    def build_topk_idx(
        self,
        base_logits: torch.Tensor,
        y_hint: Optional[torch.Tensor] = None,
        prob_hint: Optional[torch.Tensor] = None,
        has_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = min(self.topk, base_logits.shape[1])

        base_score = torch.sigmoid(base_logits.float())

        if self.topk_source == "base":
            score = base_score
        elif self.topk_source == "prob":
            if prob_hint is None:
                score = base_score
            else:
                score = prob_hint.float().clamp(0.0, 1.0)
                if has_prob is not None:
                    hp = has_prob.bool().view(-1, 1)
                    score = torch.where(hp, score, base_score)
        elif self.topk_source == "mixed":
            if prob_hint is None:
                score = base_score
            else:
                prob_score = prob_hint.float().clamp(0.0, 1.0)
                if has_prob is not None:
                    hp = has_prob.bool().view(-1, 1)
                    score = torch.where(hp, prob_score, base_score)
                else:
                    score = prob_score
        else:
            raise ValueError(f"Unknown query_decoder_topk_source: {self.topk_source}")

        if self.include_label_boost and y_hint is not None:
            score = score + y_hint.float() * self.label_boost

        return torch.topk(score, k=k, dim=1).indices

    def forward(
        self,
        x: torch.Tensor,
        permute_dims=(0, 3, 2, 1),
        y_hint: Optional[torch.Tensor] = None,
        prob_hint: Optional[torch.Tensor] = None,
        has_prob: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        base_logits, h = self.backbone(
            x,
            permute_dims=permute_dims,
            return_embedding=True,
        )
        topk_idx = self.build_topk_idx(
            base_logits=base_logits.detach(),
            y_hint=y_hint,
            prob_hint=prob_hint,
            has_prob=has_prob,
        )
        qout = self.query_decoder(
            h=h,
            base_logits=base_logits,
            topk_idx=topk_idx,
            classifier_weight=self.classifier.weight,
        )
        
        refined_logits = qout["logits"]
        delta = qout.get("delta", None)

        if return_dict:
            return {
                "logits": refined_logits,
                "base_logits": base_logits,
                "topk_idx": topk_idx,
                "delta": delta,
            }
        return refined_logits


def save_weak_model(model: nn.Module, path: Union[str, Path]):
    """
    Save backbone-only checkpoint for old eval compatibility.
    If query decoder is enabled, also save a full decoder checkpoint next to it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = _unwrap_module(model)

    if isinstance(m, WeakMSAGOWithQueryDecoder):
        # Old eval compatibility: backbone only.
        torch.save(m.backbone.state_dict(), path)
        full_path = path.parent / path.name.replace("weak_backbone", "weak_query_decoder")
        torch.save(
            {
                "checkpoint_type": "weak_query_decoder",
                "backbone": m.backbone.state_dict(),
                "query_decoder": m.query_decoder.state_dict(),
                "num_classes": m.num_classes,
                "query_decoder_topk": m.topk,
                "query_decoder_classifier_weight_shape": tuple(m.classifier.weight.shape),
            },
            full_path,
        )
    else:
        torch.save(m.state_dict(), path)

def build_optimizer(model: nn.Module, args: argparse.Namespace):
    opt_name = str(args.optim).lower()
    no_weight_decay = set()
    if hasattr(model, "no_weight_decay"):
        try:
            no_weight_decay = set(model.no_weight_decay())
        except Exception:
            no_weight_decay = set()

    if opt_name == "lamb":
        try:
            from timm.optim import create_optimizer_v2
            return create_optimizer_v2(
                model,
                opt="lamb",
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                eps=float(args.optim_eps),
                filter_bias_and_bn=bool(args.no_model_weight_decay),
            )
        except Exception as e:
            rank0_print(f"[Optimizer] Lamb unavailable, fallback AdamW. Reason: {repr(e)}")

    if bool(args.no_model_weight_decay):
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay:
                no_decay.append(p)
            else:
                decay.append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(args.weight_decay)},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(args.lr),
            eps=float(args.optim_eps),
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        eps=float(args.optim_eps),
    )


def build_scheduler(optimizer, args, total_steps: int):
    policy = str(args.lr_policy).lower()
    if policy in {"onecycle", "cycle"}:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(args.lr),
            total_steps=max(1, int(total_steps)),
            pct_start=float(args.lr_pct_start),
            three_phase=bool(args.lr_cycle_three_phase),
            div_factor=float(args.lr_div_factor),
            final_div_factor=float(args.lr_final_div_factor),
            anneal_strategy="cos",
        )
    if policy == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(total_steps)),
            eta_min=float(args.min_lr),
        )
    raise ValueError(f"Unknown lr_policy: {args.lr_policy}")


def save_backbone(model: nn.Module, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = model.module if hasattr(model, "module") else model
    torch.save(m.state_dict(), path)


def wrap_ddp(model: nn.Module, args: argparse.Namespace, device: torch.device):
    if not getattr(args, "distributed", False):
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP
    if device.type == "cuda":
        return DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )
    return DDP(model, find_unused_parameters=bool(args.ddp_find_unused_parameters))

def topk_positive_coverage(topk_idx, y):
    """
    topk_idx: [B, K]
    y:        [B, C]
    """
    b = y.shape[0]
    pos = y > 0.5

    if pos.sum() == 0:
        return torch.tensor(0.0, device=y.device)

    hit = torch.gather(pos, dim=1, index=topk_idx).float().sum()
    total = pos.float().sum().clamp_min(1.0)

    return hit / total

def train_one_task(args: argparse.Namespace):
    task = normalize_task(args.task)
    device = init_distributed_mode(args)
    set_seed(args.seed + args.rank)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = build_weak_opt_from_config(args)
    opt.mode = "train"

    if is_main_process():
        print(f"[WeakTrain] task={task}")
        print(f"[Device] {device}")
        print(f"[DDP] distributed={args.distributed}, rank={args.rank}, world_size={args.world_size}")
        print(f"[Config] {args.model_config}")
        print(f"[Init checkpoint] {args.init_ckpt}")
        print(f"[Dataset] {args.file_address}")
        print(f"[MSA] {args.working_address}")
        print(f"[Output] {output_dir}")
        save_json(vars(args), output_dir / "args.json")
        save_json(vars(opt), output_dir / "merged_model_opt.json")

    true_dataset = build_msa_dataset(opt, mode="train", task=task, need_proteins=args.need_proteins)
    pseudo_base_dataset = build_msa_dataset(opt, mode="exp_train", task=task, need_proteins=args.need_proteins)

    pseudo_prob_path = clean_optional_path(args.pseudo_prob_path)
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
            print(f"[PseudoProb] {pseudo_prob_path}, shape={pseudo_dataset.prob_shape}, dtype={pseudo_dataset.prob_dtype}")
    else:
        pseudo_dataset = pseudo_base_dataset
        if is_main_process():
            print("[PseudoProb] disabled")

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

    if is_main_process():
        print(f"[Data] train={len(true_dataset)}, exp_train={len(pseudo_dataset)}")
        print(f"[Data] local steps: train={len(true_loader)}, pseudo={len(pseudo_loader)}")

    if bool(args.use_query_decoder):
        model = WeakMSAGOWithQueryDecoder(opt, args)
        load_arch_checkpoint(model.backbone, args.init_ckpt, strict_shape=True)
    else:
        model = Arch(opt)
        load_arch_checkpoint(model, args.init_ckpt, strict_shape=True)
    model = model.to(device)

    if bool(args.freeze_bn):
        set_batchnorm_eval(model, freeze_affine=bool(args.freeze_bn_affine))

    model = wrap_ddp(model, args=args, device=device)
    optimizer = build_optimizer(model, args)

    steps_per_epoch = max(len(true_loader), len(pseudo_loader))
    if args.max_steps_per_epoch is not None and int(args.max_steps_per_epoch) > 0:
        steps_per_epoch = min(steps_per_epoch, int(args.max_steps_per_epoch))

    total_update_steps = max(1, (steps_per_epoch * args.epochs + args.accum_steps - 1) // args.accum_steps)
    scheduler = build_scheduler(optimizer, args, total_update_steps)

    amp_enabled = device.type == "cuda" and not args.no_amp
    try:
        scaler = GradScaler(device="cuda", enabled=False)
    except TypeError:
        scaler = GradScaler(enabled=False)

    true_asl = AsymmetricLossOptimized(
        gamma_neg=args.true_asl_gamma_neg,
        gamma_pos=args.true_asl_gamma_pos,
        clip=args.true_asl_clip,
        disable_torch_grad_focal_loss=True,
    )

    ic = load_ic(args, task, args.num_classes)
    ic_device = ic.to(device=device, dtype=torch.float32)
    go_edges = load_go_edges(args.go_edges_path, device)

    permute_dims = tuple(args.permute_dims)
    global_step = 0
    update_step = 0

    if is_main_process():
        print("[Model]", model.__class__.__name__)
        print("[Use query decoder]", getattr(args, "use_query_decoder", False))
        if hasattr(unwrap_model(model), "query_decoder"):
            print("[Query decoder]", unwrap_model(model).query_decoder)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if bool(args.freeze_bn):
            set_batchnorm_eval(model, freeze_affine=bool(args.freeze_bn_affine))

        set_loader_epoch(true_loader, epoch)
        set_loader_epoch(pseudo_loader, epoch)
        true_iter = iter(true_loader)
        pseudo_iter = iter(pseudo_loader)

        running: Dict[str, float] = defaultdict(float)
        running.update({
            "loss": 0.0,
            "loss_true": 0.0,
            "loss_pseudo": 0.0,
            "loss_hier": 0.0,
            "contrib_true": 0.0,
            "contrib_pseudo": 0.0,
            "contrib_hier": 0.0,
            "pseudo_mask_terms": 0.0,
            "pseudo_pos_terms": 0.0,
            "pseudo_neg_terms": 0.0,
            "pseudo_conf_mean": 0.0,
        })

        pbar = tqdm(range(steps_per_epoch), desc=f"weak epoch {epoch}/{args.epochs}", disable=not is_main_process())
        optimizer.zero_grad(set_to_none=True)

        for step in pbar:
            true_batch, true_iter = get_next_batch(true_iter, true_loader)
            pseudo_batch, pseudo_iter = get_next_batch(pseudo_iter, pseudo_loader)

            proteins_t, X_t, y_t = unpack_batch(true_batch)
            proteins_u, X_u, y_u, prob_u = unpack_pseudo_batch(pseudo_batch)

            X_t, y_t, X_u, y_u = move_to_device(X_t, y_t, X_u, y_u, device=device)
            if prob_u is not None:
                prob_u = prob_u.to(device, non_blocking=True)

            X_t = X_t.long()
            X_u = X_u.long()
            y_t = y_t.float()
            y_u = y_u.float()

            b_t = X_t.shape[0]
            X = torch.cat([X_t, X_u], dim=0)

            proteins = combine_proteins(proteins_t, proteins_u)
            if proteins is not None:
                set_model_proteins(model, proteins)

            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                if bool(args.use_query_decoder):
                    # true samples have no teacher prob; pseudo samples use prob_u when available.
                    y_hint = torch.cat([y_t, y_u], dim=0)
                    if prob_u is not None:
                        prob_hint = torch.cat([torch.zeros_like(y_t), prob_u.float()], dim=0)
                        has_prob = torch.cat([
                            torch.zeros(y_t.shape[0], dtype=torch.bool, device=device),
                            torch.ones(y_u.shape[0], dtype=torch.bool, device=device),
                        ], dim=0)
                    else:
                        prob_hint = None
                        has_prob = None
                    out = model(
                        X,
                        permute_dims=permute_dims,
                        y_hint=y_hint,
                        prob_hint=prob_hint,
                        has_prob=has_prob,
                        return_dict=True,
                    )
                    logits = out["logits"]
                else:
                    logits = model(X, permute_dims=permute_dims)
                    if isinstance(logits, dict):
                        logits = logits["logits"]
                    elif isinstance(logits, (tuple, list)):
                        logits = logits[0]

                if is_main_process() and global_step == 0:
                    print("[Query] topk_idx shape:", tuple(out["topk_idx"].shape))
                    if out.get("delta", None) is not None:
                        print("[Query] delta shape:", tuple(out["delta"].shape))

                if isinstance(out, dict) and out.get("delta", None) is not None:
                    delta = out["delta"].detach()
                    delta_abs_mean = delta.abs().mean()
                    delta_abs_max = delta.abs().max()
                else:
                    delta_abs_mean = torch.tensor(0.0, device=device)
                    delta_abs_max = torch.tensor(0.0, device=device)

                if isinstance(out, dict) and "topk_idx" in out:
                    topk_cov = topk_positive_coverage(out["topk_idx"], y_hint)
                else:
                    topk_cov = torch.tensor(0.0, device=device)

                logits_t = logits[:b_t]
                logits_u = logits[b_t:]

                loss_true_raw = true_asl(logits_t.float(), y_t.float())
                if args.true_asl_reduction == "batch_mean":
                    loss_true = loss_true_raw / max(int(logits_t.shape[0]), 1)
                elif args.true_asl_reduction == "raw":
                    loss_true = loss_true_raw
                else:
                    raise ValueError(f"Unknown true_asl_reduction: {args.true_asl_reduction}")

                target_u, mask_u, conf_u, pseudo_stats = build_pseudo_supervision(
                    logits_u=logits_u,
                    y_u=y_u,
                    prob_u=prob_u,
                    args=args,
                    ic_device=ic_device,
                )

                if args.pseudo_loss_type == "asl":
                    loss_pseudo = masked_asl_with_logits(
                        logits=logits_u,
                        target=target_u,
                        mask=mask_u,
                        conf=conf_u,
                        gamma_neg=args.pseudo_asl_gamma_neg,
                        gamma_pos=args.pseudo_asl_gamma_pos,
                        clip=args.pseudo_asl_clip,
                        reduction=args.pseudo_asl_reduction,
                    )
                elif args.pseudo_loss_type == "bce":
                    loss_pseudo = masked_bce_with_logits(
                        logits=logits_u,
                        target=target_u,
                        mask=mask_u,
                        conf=conf_u,
                        reduction_mode=args.pseudo_asl_reduction,
                    )
                else:
                    raise ValueError(f"Unknown pseudo_loss_type: {args.pseudo_loss_type}")

                loss_hier = hierarchy_violation_loss(logits, go_edges)

                loss = (
                    args.lambda_true * loss_true
                    + args.lambda_pseudo * loss_pseudo
                    + args.lambda_h * loss_hier
                )
                loss_for_backward = loss / int(args.accum_steps)

            scaler.scale(loss_for_backward).backward()

            do_update = ((step + 1) % int(args.accum_steps) == 0) or (step + 1 == steps_per_epoch)
            if do_update:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_step += 1

            global_step += 1

            running["loss"] += float(loss.detach().cpu())
            running["loss_true"] += float(loss_true.detach().cpu())
            running["loss_pseudo"] += float(loss_pseudo.detach().cpu())
            running["loss_hier"] += float(loss_hier.detach().cpu())
            running["contrib_true"] += float((args.lambda_true * loss_true).detach().cpu())
            running["contrib_pseudo"] += float((args.lambda_pseudo * loss_pseudo).detach().cpu())
            running["contrib_hier"] += float((args.lambda_h * loss_hier).detach().cpu())
            running["delta_abs_mean"] += float(delta_abs_mean.cpu())
            running["delta_abs_max"] += float(delta_abs_max.cpu())
            running["topk_pos_coverage"] += float(topk_cov.detach().cpu())

            for k, v in pseudo_stats.items():
                running[k] += float(v)

            if is_main_process() and step % args.log_interval == 0:
                denom = step + 1
                pbar.set_postfix({
                    "loss": running["loss"] / denom,
                    "true": running["loss_true"] / denom,
                    "pseudo": running["loss_pseudo"] / denom,
                    "hier": running["loss_hier"] / denom,
                    "c_pseudo": running["contrib_pseudo"] / denom,
                    "p_mask": running["pseudo_mask_terms"] / denom,
                    "pconf": running["pseudo_conf_mean"] / denom,
                    "lr": scheduler.get_last_lr()[0],
                })

        local_means = {k: v / max(1, steps_per_epoch) for k, v in running.items()}
        reduced = reduce_metric_dict_mean(local_means, device)
        epoch_log = {
            "epoch": epoch,
            "global_step": global_step,
            "update_step": update_step,
            "lr": scheduler.get_last_lr()[0],
            **reduced,
        }

        if is_main_process():
            append_jsonl(epoch_log, output_dir / "weak_train_log.jsonl")
            print("[WeakEpoch]", epoch_log)
            if epoch % args.save_interval == 0:
                save_weak_model(model, output_dir / f"weak_backbone_epoch{epoch}.pt")

        if args.distributed:
            torch.distributed.barrier()

    if is_main_process():
        save_weak_model(model, output_dir / "weak_backbone_last.pt")
        print(f"[Done] saved to {output_dir}")

    if args.distributed:
        torch.distributed.barrier()
    cleanup_distributed()


def build_argparser():
    p = argparse.ArgumentParser(description="Weakly-supervised MSA-GO expansion training")

    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--init_ckpt", type=str, required=True)
    p.add_argument("--file_address", type=str, required=True)
    p.add_argument("--working_address", type=str, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--msa_max_size", type=int, default=None)
    p.add_argument("--permute_dims", type=int, nargs=4, default=[0, 3, 2, 1])
    p.add_argument("--torch_compile", action="store_true")

    p.add_argument("--msa_read_mode", type=str, choices=["full", "rows", "block"], default="full")
    p.add_argument("--msa_sample_strategy", type=str, choices=["random", "block", "head"], default="random")
    p.add_argument("--msa_shuffle_rows_at_getitem", action="store_true")
    p.add_argument("--no_msa_shuffle_rows_at_getitem", dest="msa_shuffle_rows_at_getitem", action="store_false")
    p.set_defaults(msa_shuffle_rows_at_getitem=True)
    p.add_argument("--msa_cache_gb", type=float, default=4.0)
    p.add_argument("--msa_max_open_files", type=int, default=256)
    p.add_argument("--sample_seed", type=int, default=1)
    p.add_argument("--sampler_seed", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    p.set_defaults(persistent_workers=True)

    p.add_argument("--gpu_ids", type=str, default=None)
    p.add_argument("--ddp_find_unused_parameters", action="store_true")

    p.add_argument("--freeze_bn", action="store_true")
    p.add_argument("--no_freeze_bn", dest="freeze_bn", action="store_false")
    p.set_defaults(freeze_bn=False)
    p.add_argument("--freeze_bn_affine", action="store_true")
    p.add_argument("--no_freeze_bn_affine", dest="freeze_bn_affine", action="store_false")
    p.set_defaults(freeze_bn_affine=False)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--pseudo_batch_size", type=int, default=64)
    p.add_argument("--dataloader_num_workers", type=int, default=8)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--drop_last", action="store_true")
    p.add_argument("--max_steps_per_epoch", type=int, default=None)

    p.add_argument("--optim", type=str, default="lamb", choices=["lamb", "adamw"])
    p.add_argument("--optim_eps", type=float, default=1e-6)
    p.add_argument("--lr", type=float, default=1.8e-3)
    p.add_argument("--lr_policy", type=str, default="cycle", choices=["cycle", "onecycle", "cosine"])
    p.add_argument("--lr_pct_start", type=float, default=0.1)
    p.add_argument("--lr_cycle_three_phase", action="store_true")
    p.add_argument("--lr_div_factor", type=float, default=25.0)
    p.add_argument("--lr_final_div_factor", type=float, default=120.0)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.009)
    p.add_argument("--no_model_weight_decay", action="store_true")
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--lambda_true", type=float, default=1.0)
    p.add_argument("--lambda_pseudo", type=float, default=0.2)
    p.add_argument("--lambda_h", type=float, default=0.0005)

    p.add_argument("--true_asl_gamma_neg", type=float, default=4.0)
    p.add_argument("--true_asl_gamma_pos", type=float, default=0.0)
    p.add_argument("--true_asl_clip", type=float, default=0.05)
    p.add_argument("--true_asl_reduction", type=str, default="batch_mean", choices=["raw", "batch_mean"])

    p.add_argument("--pseudo_prob_path", type=str, default=None)
    p.add_argument("--pseudo_loss_type", type=str, default="asl", choices=["asl", "bce"])
    p.add_argument("--pseudo_prob_target", type=str, default="soft_pos", choices=["hard", "soft_pos"])
    p.add_argument("--pseudo_prob_conf_power", type=float, default=0.5)
    p.add_argument("--pseudo_prob_min_conf", type=float, default=0.0)
    p.add_argument("--pseudo_negative_policy", type=str, default="none", choices=["none", "prob_low"])
    p.add_argument("--pseudo_prob_neg_max", type=float, default=0.01)
    p.add_argument("--pseudo_neg_topk", type=int, default=0)
    p.add_argument("--pseudo_min_ic", type=float, default=0.0)
    p.add_argument("--pseudo_asl_gamma_neg", type=float, default=4.0)
    p.add_argument("--pseudo_asl_gamma_pos", type=float, default=0.0)
    p.add_argument("--pseudo_asl_clip", type=float, default=0.05)
    p.add_argument("--pseudo_asl_reduction", type=str, default="batch_mean", choices=["weighted_mean", "mean_mask", "batch_mean", "sum"])

    # Optional DETR-style top-k ontology query decoder
    p.add_argument("--use_query_decoder", action="store_true")
    p.add_argument("--no_query_decoder", dest="use_query_decoder", action="store_false")
    p.set_defaults(use_query_decoder=False)
    p.add_argument("--query_decoder_topk", type=int, default=50)
    p.add_argument("--query_decoder_topk_source", type=str, default="mixed", choices=["base", "prob", "mixed"])
    p.add_argument("--query_decoder_mode", type=str, default="residual", choices=["residual", "replace"])
    p.add_argument("--query_decoder_dim", type=int, default=256)
    p.add_argument("--query_decoder_heads", type=int, default=8)
    p.add_argument("--query_decoder_layers", type=int, default=1)
    p.add_argument("--query_decoder_ffn_dim", type=int, default=1024)
    p.add_argument("--query_decoder_dropout", type=float, default=0.1)
    p.add_argument("--query_decoder_detach_query_weight", action="store_true")
    p.add_argument("--no_query_decoder_detach_query_weight", dest="query_decoder_detach_query_weight", action="store_false")
    p.set_defaults(query_decoder_detach_query_weight=True)
    p.add_argument("--query_decoder_include_label_boost", action="store_true")
    p.add_argument("--no_query_decoder_include_label_boost", dest="query_decoder_include_label_boost", action="store_false")
    p.set_defaults(query_decoder_include_label_boost=True)
    p.add_argument("--query_decoder_label_boost", type=float, default=2.0)
    p.add_argument(
        "--query_decoder_memory_mode",
        type=str,
        default="tokens_plus_pooled",
        choices=["pooled", "tokens", "tokens_plus_pooled"],
    )
    
    p.add_argument("--query_decoder_memory_grid_h", type=int, default=0)
    p.add_argument("--query_decoder_memory_grid_w", type=int, default=0)

    p.add_argument("--ic_path", type=str, default=None)
    p.add_argument("--ic_alpha", type=float, default=1.0)
    p.add_argument("--ic_min_count", type=int, default=2)
    p.add_argument("--go_edges_path", type=str, default=None)

    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--need_proteins", action="store_true")

    return p


def main():
    parser = build_argparser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Warning] Ignoring unknown arguments: {unknown}")
    args.task = normalize_task(args.task)
    train_one_task(args)


if __name__ == "__main__":
    main()
