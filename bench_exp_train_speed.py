#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bench_exp_train_speed.py

Benchmark exp_train pipeline.

Measures:
1. DataLoader wait time
2. CPU -> GPU transfer time
3. GPU compute time: weak+strong forward, losses, backward, optimizer step
4. End-to-end step time

Usage, single process:
    python bench_exp_train_speed.py

Usage, DDP:
    CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nproc_per_node=3 bench_exp_train_speed.py
"""

from __future__ import annotations

import os
import sys
import time
import statistics
from pathlib import Path

import torch


# ---------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------

WARMUP_STEPS = 5
BENCH_STEPS = 30

BENCH_LOADER_ONLY = True
BENCH_COMPUTE_ONLY = True
BENCH_END_TO_END = True

# For quick diagnosis, avoid validation/checkpointing here.
OVERRIDE_EPOCHS = 1


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_exp_train as R  # noqa: E402

if R.CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(R.CUDA_VISIBLE_DEVICES)

R.prepare_imports()

import experiments.exp_train as E  # noqa: E402


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(xs):
    xs = list(xs)
    if len(xs) == 0:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    xs_sorted = sorted(xs)
    p95_idx = min(len(xs_sorted) - 1, int(round(0.95 * (len(xs_sorted) - 1))))

    return {
        "mean": statistics.mean(xs),
        "p50": statistics.median(xs),
        "p95": xs_sorted[p95_idx],
        "min": min(xs),
        "max": max(xs),
    }


def fmt_ms(stats):
    return (
        f"mean={stats['mean'] * 1000:.2f} ms, "
        f"p50={stats['p50'] * 1000:.2f} ms, "
        f"p95={stats['p95'] * 1000:.2f} ms, "
        f"min={stats['min'] * 1000:.2f} ms, "
        f"max={stats['max'] * 1000:.2f} ms"
    )


def rank0_print(*args, **kwargs):
    if E.is_main_process():
        print(*args, **kwargs)


def build_loaders(args, opt):
    true_dataset = E.build_msa_dataset(
        opt=opt,
        mode="train",
        task=args.task,
        need_proteins=args.need_proteins,
    )

    pseudo_dataset = E.build_msa_dataset(
        opt=opt,
        mode="exp_train",
        task=args.task,
        need_proteins=args.need_proteins,
    )

    true_loader = E.make_loader(
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

    pseudo_loader = E.make_loader(
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

    return true_loader, pseudo_loader


def prepare_gpu_batch(true_batch, pseudo_batch, device):
    proteins_t, X_t, y_t = E.unpack_batch(true_batch)
    proteins_u, X_u, y_u = E.unpack_batch(pseudo_batch)

    X_t, y_t, X_u, y_u = E.move_to_device(
        X_t, y_t, X_u, y_u, device=device
    )

    X_t = X_t.long()
    X_u = X_u.long()

    y_t = y_t.float()
    y_u = y_u.float()

    proteins = E.combine_proteins(proteins_t, proteins_u)

    return proteins, X_t, y_t, X_u, y_u


def build_model_bundle(args, opt, device):
    model = E.SemiSupMSAGO(opt)

    if args.init_ckpt is not None:
        E.load_backbone_checkpoint(
            model=model,
            ckpt_path=args.init_ckpt,
            strict_shape=True,
        )

    model = model.to(device)
    model = E.wrap_model_for_distributed(model, args=args, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    amp_enabled = device.type == "cuda" and not args.no_amp

    try:
        scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=amp_enabled)

    ic = E.load_ic(args, args.task, args.num_classes)
    go_con_loss = E.GOAwareSupConLoss(
        ic=ic,
        tau=args.contrast_tau,
        min_r=args.contrast_min_r,
        w_true_pseudo=args.w_true_pseudo,
        w_pseudo_pseudo=args.w_pseudo_pseudo,
    ).to(device)

    go_edges = E.load_go_edges(args.go_edges_path, device)
    ic_device = ic.to(device=device, dtype=torch.float32)

    weak_aug_params = E.build_msa_aug_params(opt, "weak")
    strong_aug_params = E.build_msa_aug_params(opt, "strong")

    return {
        "model": model,
        "optimizer": optimizer,
        "scaler": scaler,
        "amp_enabled": amp_enabled,
        "go_con_loss": go_con_loss,
        "go_edges": go_edges,
        "ic_device": ic_device,
        "weak_aug_params": weak_aug_params,
        "strong_aug_params": strong_aug_params,
        "permute_dims": tuple(args.permute_dims),
    }


def train_step_from_gpu_batch(
    args,
    bundle,
    proteins,
    X_t,
    y_t,
    X_u,
    y_u,
):
    model = bundle["model"]
    optimizer = bundle["optimizer"]
    scaler = bundle["scaler"]
    amp_enabled = bundle["amp_enabled"]

    go_con_loss = bundle["go_con_loss"]
    go_edges = bundle["go_edges"]
    ic_device = bundle["ic_device"]

    weak_aug_params = bundle["weak_aug_params"]
    strong_aug_params = bundle["strong_aug_params"]
    permute_dims = bundle["permute_dims"]

    b_t = X_t.shape[0]
    b_u = X_u.shape[0]

    X = torch.cat([X_t, X_u], dim=0)

    if proteins is not None:
        E.set_model_proteins(model, proteins)

    mask_t = torch.ones_like(y_t)
    conf_t = torch.ones_like(y_t)

    mask_u = (y_u > 0.5).float() if args.pseudo_pos_only else torch.ones_like(y_u)
    conf_u = E.pseudo_conf_from_labels(y_u, mode="binary")

    labels_for_con = torch.cat([y_t, y_u], dim=0)
    masks_for_con = torch.cat([mask_t, mask_u], dim=0)
    confs_for_con = torch.cat([conf_t, conf_u], dim=0)

    is_true = torch.cat(
        [
            torch.ones(b_t, dtype=torch.bool, device=X.device),
            torch.zeros(b_u, dtype=torch.bool, device=X.device),
        ],
        dim=0,
    )

    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type=X.device.type, enabled=amp_enabled):
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
            E.bce_with_logits(logits_t_w, y_t)
            + E.bce_with_logits(logits_t_s, y_t)
        )

        loss_pseudo = E.masked_bce_with_logits(
            logits=logits_u_s,
            target=y_u,
            mask=mask_u,
            conf=conf_u,
            pos_only=args.pseudo_pos_only,
        )

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

            labels_con = labels_for_con[has_pos].repeat(2, 1)
            masks_con = masks_for_con[has_pos].repeat(2, 1)
            confs_con = confs_for_con[has_pos].repeat(2, 1)
            is_true_con = is_true[has_pos].repeat(2)

            loss_con = go_con_loss(
                z=z_con,
                y=labels_con,
                is_true=is_true_con,
                term_mask=masks_con,
                term_conf=confs_con,
            )
        else:
            loss_con = z_w.float().sum() * 0.0

        loss_hier = E.hierarchy_violation_loss(logits_s, go_edges)

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

    return loss.detach()


def bench_loader_pair(args, true_loader, pseudo_loader, device):
    rank0_print("\n[Benchmark] DataLoader + H2D")

    true_iter = iter(true_loader)
    pseudo_iter = iter(pseudo_loader)

    wait_times = []
    h2d_times = []

    total = WARMUP_STEPS + BENCH_STEPS

    for i in range(total):
        t0 = time.perf_counter()

        true_batch, true_iter = E.get_next_batch(true_iter, true_loader)
        pseudo_batch, pseudo_iter = E.get_next_batch(pseudo_iter, pseudo_loader)

        t1 = time.perf_counter()

        proteins, X_t, y_t, X_u, y_u = prepare_gpu_batch(
            true_batch,
            pseudo_batch,
            device,
        )

        sync_cuda(device)
        t2 = time.perf_counter()

        if i >= WARMUP_STEPS:
            wait_times.append(t1 - t0)
            h2d_times.append(t2 - t1)

    rank0_print(f"  Data wait: {fmt_ms(summarize(wait_times))}")
    rank0_print(f"  H2D copy : {fmt_ms(summarize(h2d_times))}")


def bench_compute_only(args, true_loader, pseudo_loader, bundle, device):
    rank0_print("\n[Benchmark] Compute only")

    true_iter = iter(true_loader)
    pseudo_iter = iter(pseudo_loader)

    true_batch, true_iter = E.get_next_batch(true_iter, true_loader)
    pseudo_batch, pseudo_iter = E.get_next_batch(pseudo_iter, pseudo_loader)

    proteins, X_t, y_t, X_u, y_u = prepare_gpu_batch(
        true_batch,
        pseudo_batch,
        device,
    )

    sync_cuda(device)

    compute_times = []
    total = WARMUP_STEPS + BENCH_STEPS

    for i in range(total):
        t0 = time.perf_counter()

        loss = train_step_from_gpu_batch(
            args,
            bundle,
            proteins,
            X_t,
            y_t,
            X_u,
            y_u,
        )

        sync_cuda(device)
        t1 = time.perf_counter()

        if i >= WARMUP_STEPS:
            compute_times.append(t1 - t0)

    rank0_print(f"  Compute step: {fmt_ms(summarize(compute_times))}")


def bench_end_to_end(args, true_loader, pseudo_loader, bundle, device):
    rank0_print("\n[Benchmark] End-to-end step")

    true_iter = iter(true_loader)
    pseudo_iter = iter(pseudo_loader)

    data_wait_times = []
    h2d_times = []
    compute_times = []
    total_times = []

    total = WARMUP_STEPS + BENCH_STEPS

    for i in range(total):
        t0 = time.perf_counter()

        true_batch, true_iter = E.get_next_batch(true_iter, true_loader)
        pseudo_batch, pseudo_iter = E.get_next_batch(pseudo_iter, pseudo_loader)

        t1 = time.perf_counter()

        proteins, X_t, y_t, X_u, y_u = prepare_gpu_batch(
            true_batch,
            pseudo_batch,
            device,
        )

        sync_cuda(device)
        t2 = time.perf_counter()

        loss = train_step_from_gpu_batch(
            args,
            bundle,
            proteins,
            X_t,
            y_t,
            X_u,
            y_u,
        )

        sync_cuda(device)
        t3 = time.perf_counter()

        if i >= WARMUP_STEPS:
            data_wait_times.append(t1 - t0)
            h2d_times.append(t2 - t1)
            compute_times.append(t3 - t2)
            total_times.append(t3 - t0)

    rank0_print(f"  Data wait : {fmt_ms(summarize(data_wait_times))}")
    rank0_print(f"  H2D copy  : {fmt_ms(summarize(h2d_times))}")
    rank0_print(f"  Compute   : {fmt_ms(summarize(compute_times))}")
    rank0_print(f"  Total step: {fmt_ms(summarize(total_times))}")


def main():
    args = R.build_args()
    args.epochs = OVERRIDE_EPOCHS
    args.no_validation = True

    device = E.init_distributed_mode(args)
    args.task = E.normalize_task(args.task)

    opt = E.build_opt_from_config(args)
    opt.mode = "train"

    rank0_print("=" * 80)
    rank0_print("[Benchmark config]")
    rank0_print(f"task               = {args.task}")
    rank0_print(f"distributed        = {args.distributed}")
    rank0_print(f"rank/world_size    = {args.rank}/{args.world_size}")
    rank0_print(f"device             = {device}")
    rank0_print(f"batch_size         = {args.batch_size}")
    rank0_print(f"pseudo_batch_size  = {args.pseudo_batch_size}")
    rank0_print(f"top_k              = {args.top_k}")
    rank0_print(f"max_len            = {args.max_len}")
    rank0_print(f"msa_read_mode      = {args.msa_read_mode}")
    rank0_print(f"msa_sample_strategy= {args.msa_sample_strategy}")
    rank0_print(f"msa_cache_gb       = {args.msa_cache_gb}")
    rank0_print(f"num_workers/rank   = {args.dataloader_num_workers}")
    rank0_print("=" * 80)

    true_loader, pseudo_loader = build_loaders(args, opt)

    E.set_loader_epoch(true_loader, 1)
    E.set_loader_epoch(pseudo_loader, 1)

    if BENCH_LOADER_ONLY:
        bench_loader_pair(args, true_loader, pseudo_loader, device)

    bundle = None

    if BENCH_COMPUTE_ONLY or BENCH_END_TO_END:
        bundle = build_model_bundle(args, opt, device)
        bundle["model"].train()

    if BENCH_COMPUTE_ONLY:
        bench_compute_only(args, true_loader, pseudo_loader, bundle, device)

    if BENCH_END_TO_END:
        bench_end_to_end(args, true_loader, pseudo_loader, bundle, device)

    if args.distributed:
        torch.distributed.barrier()

    E.cleanup_distributed()


if __name__ == "__main__":
    main()
