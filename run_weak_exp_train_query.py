#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_weak_exp_train.py

Runner for experiments/weak_exp_train.py.

Usage:
    python run_weak_exp_train.py

The parent process launches torchrun when USE_DDP=True.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
MSA_ROOT = ROOT / "msa_models"
WEAK_EXP_TRAIN = ROOT / "experiments" / "weak_exp_train.py"

# =============================================================================
# USER CONFIG
# =============================================================================

TASK = os.environ.get("TASK", "bp")  # "cc", "mf", "bp"

MODEL_CONFIG = ROOT / "data" / "msa_models" / "configs" / "model_opts" / f"{TASK}_msa_model_config.pkl"
INIT_CKPT = ROOT / "data" / "msa_models" / "checkpoints" / f"{TASK}_msa_model_rank1.pt"
FILE_ADDRESS = ROOT / "data" / "unidata_with_exp_train_pseudo.pkl"
WORKING_ADDRESS = ROOT / "data" / "sprot_2204_MSA_bin" / "index.pkl"
GO_EDGES_PATH = ROOT / "data" / "go_edges" / f"go_edges_{TASK}.pt"
IC_PATH = None

PROB_OUTPUT_DIR = ROOT / "data" / "outputs" / "msa_teacher_25_10_22_probs"
PSEUDO_PROB_PATH = PROB_OUTPUT_DIR / f"{TASK}_exp_train_probs.float16.npy"
USE_PSEUDO_PROB = True

RUN_TAG = f"{TASK}_weak_asl_prob_v1"
OUTPUT_DIR = ROOT / "outputs" / "weak_exp_train_query" / RUN_TAG
FAIL_IF_OUTPUT_EXISTS = False

NUM_CLASSES = None
TOP_K = 64          # teacher reference
MAX_LEN = 2048      # teacher reference
MSA_MAX_SIZE = None
PERMUTE_DIMS = (0, 3, 2, 1)
TORCH_COMPILE = False

SEED = 3407
EPOCHS = 50
# With the provided weak_exp_train.py, true and pseudo batches are concatenated.
# Therefore BATCH_SIZE=32 and PSEUDO_BATCH_SIZE=32 means a per-GPU forward batch of 64.
BATCH_SIZE = 16
PSEUDO_BATCH_SIZE = 48
DATALOADER_NUM_WORKERS = 8
PIN_MEMORY = True
DROP_LAST = False
MAX_STEPS_PER_EPOCH = 1000  # for smoke test: e.g. 200

MSA_READ_MODE = "full"
MSA_SAMPLE_STRATEGY = "random"
MSA_SHUFFLE_ROWS_AT_GETITEM = True
MSA_CACHE_GB = 4.0
MSA_MAX_OPEN_FILES = 256
SAMPLE_SEED = 1
SAMPLER_SEED = 1
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

DDP_FIND_UNUSED_PARAMETERS = False

# Weak training: single model, no EMA/KD/contrastive.
OPTIM = "adamw"          # fallback to AdamW if timm Lamb unavailable
OPTIM_EPS = 1e-6
# LR = 1.8e-3             # teacher reference
LR = 3e-4

LR_POLICY = "cycle"
LR_PCT_START = 0.1
LR_CYCLE_THREE_PHASE = False
LR_DIV_FACTOR = 25.0
LR_FINAL_DIV_FACTOR = 120.0
MIN_LR = 1e-6
WEIGHT_DECAY = 0.009
NO_MODEL_WEIGHT_DECAY = True
ACCUM_STEPS = 1
GRAD_CLIP = 1.0
NO_AMP = False

FREEZE_BN = False       # teacher-like training; set True only if unstable
FREEZE_BN_AFFINE = False

LAMBDA_TRUE = 0.9
LAMBDA_PSEUDO = 0.3
LAMBDA_H = 0.0005

TRUE_ASL_GAMMA_NEG = 4.0
TRUE_ASL_GAMMA_POS = 0.0
TRUE_ASL_CLIP = 0.05
TRUE_ASL_REDUCTION = "batch_mean"  # stabilizes scale vs pseudo loss

PSEUDO_LOSS_TYPE = "asl"
PSEUDO_PROB_TARGET = "soft_pos"    # "hard" or "soft_pos"
PSEUDO_PROB_CONF_POWER = 0.5
PSEUDO_PROB_MIN_CONF = 0.0
PSEUDO_NEGATIVE_POLICY = "none"    # first run: avoid false negatives
PSEUDO_PROB_NEG_MAX = 0.01
PSEUDO_NEG_TOPK = 0
PSEUDO_MIN_IC = 0.0
PSEUDO_ASL_GAMMA_NEG = 4.0
PSEUDO_ASL_GAMMA_POS = 0.0
PSEUDO_ASL_CLIP = 0.05
PSEUDO_ASL_REDUCTION = "batch_mean"

# ---------------------------------------------------------------------
# Optional DETR-style top-k ontology query decoder
# ---------------------------------------------------------------------
USE_QUERY_DECODER = True
QUERY_DECODER_TOPK = 100
QUERY_DECODER_TOPK_SOURCE = "base"      # "base", "prob", "mixed"
QUERY_DECODER_MODE = "residual"          # strongly recommended; "replace" is risky
QUERY_DECODER_DIM = 256
QUERY_DECODER_HEADS = 8
QUERY_DECODER_LAYERS = 1
QUERY_DECODER_FFN_DIM = 1024
QUERY_DECODER_DROPOUT = 0.1
QUERY_DECODER_DETACH_QUERY_WEIGHT = True
QUERY_DECODER_INCLUDE_LABEL_BOOST = False
QUERY_DECODER_LABEL_BOOST = 2.0
QUERY_DECODER_MEMORY_MODE = "tokens_plus_pooled"
QUERY_DECODER_MEMORY_GRID_H = 0
QUERY_DECODER_MEMORY_GRID_W = 0

# =============================================================================
# Multi-node DDP
# =============================================================================

# "single_node" or "multi_node"
LAUNCH_MODE = "single_node"

CUDA_VISIBLE_DEVICES = "1,2"
DEVICE = "auto"
GPU_IDS = "auto"
USE_DDP = True
DDP_STANDALONE = True

# 每台机器使用多少张 GPU
NPROC_PER_NODE = int(os.environ.get("NPROC_PER_NODE", "2"))

# 总节点数
NNODES = int(os.environ.get("NNODES", "1"))

# 当前节点编号：master node = 0, second node = 1, ...
# 建议也允许从环境变量读取，避免每台机器手动改文件。
NODE_RANK = int(os.environ.get("NODE_RANK", "0"))

# master 节点的内网 IP，必须所有节点都能访问
MASTER_ADDR = os.environ.get("MASTER_ADDR", "10.233.128.18")
MASTER_PORT = int(os.environ.get("MASTER_PORT", "46123"))

# rendezvous
RDZV_BACKEND = "c10d"
RDZV_ID = os.environ.get("RDZV_ID", "latence_weak_query_bp_v1")

# 如果使用 --rdzv_endpoint
RDZV_ENDPOINT = f"{MASTER_ADDR}:{MASTER_PORT}"

# 出错是否自动重启。不建议调试阶段自动重启。
DDP_MAX_RESTARTS = 0

# DDP timeout
DDP_TIMEOUT_MINUTES = 180

NCCL_SOCKET_IFNAME = "eth0"  # 例如 "eth0" / "ens3f0" / "ib0"
GLOO_SOCKET_IFNAME = None

IC_ALPHA = 1.0
IC_MIN_COUNT = 2

LOG_INTERVAL = 20
SAVE_INTERVAL = 2
NEED_PROTEINS = False
DRY_RUN = False
PRINT_CONFIG = True
SAVE_RUNNER_CONFIG = True

if CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)


def _load_pickle_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"MODEL_CONFIG must contain a dict, got {type(obj)}")
    return obj


def _infer_value(explicit_value, cfg: Dict[str, Any], key: str, required: bool = True):
    if explicit_value is not None:
        return explicit_value
    value = cfg.get(key, None)
    if required and value is None:
        raise ValueError(f"Could not infer required field '{key}' from MODEL_CONFIG.")
    return value


def _path_or_none(path_like):
    if path_like is None:
        return None
    return str(Path(path_like))


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


def _save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False, default=str)


def _validate_paths():
    if not WEAK_EXP_TRAIN.exists():
        raise FileNotFoundError(f"Cannot find weak_exp_train.py: {WEAK_EXP_TRAIN}")
    if not MSA_ROOT.exists():
        raise FileNotFoundError(f"Cannot find msa_models: {MSA_ROOT}")
    if not Path(MODEL_CONFIG).is_file():
        raise FileNotFoundError(f"MODEL_CONFIG not found: {MODEL_CONFIG}")
    if not Path(INIT_CKPT).is_file():
        raise FileNotFoundError(f"INIT_CKPT not found: {INIT_CKPT}")
    if not Path(FILE_ADDRESS).is_file():
        raise FileNotFoundError(f"FILE_ADDRESS not found: {FILE_ADDRESS}")
    if not Path(WORKING_ADDRESS).is_file():
        warnings.warn(f"WORKING_ADDRESS is not a file: {WORKING_ADDRESS}")
    if USE_PSEUDO_PROB and not Path(PSEUDO_PROB_PATH).is_file():
        raise FileNotFoundError(f"PSEUDO_PROB_PATH not found: {PSEUDO_PROB_PATH}")
    if GO_EDGES_PATH is not None and not Path(GO_EDGES_PATH).is_file():
        raise FileNotFoundError(f"GO_EDGES_PATH not found: {GO_EDGES_PATH}")
    if IC_PATH is not None and not Path(IC_PATH).is_file():
        raise FileNotFoundError(f"IC_PATH not found: {IC_PATH}")


def _validate_output_dir():
    if Path(OUTPUT_DIR).exists() and any(Path(OUTPUT_DIR).iterdir()) and FAIL_IF_OUTPUT_EXISTS:
        raise FileExistsError(f"OUTPUT_DIR exists and is not empty: {OUTPUT_DIR}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _is_rank0_env() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _should_launch_ddp() -> bool:
    return bool(USE_DDP) and "RANK" not in os.environ and int(NPROC_PER_NODE) > 1


def _launch_ddp_and_exit():
    env = os.environ.copy()

    if CUDA_VISIBLE_DEVICES is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)

    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    env.setdefault("NCCL_DEBUG", "INFO")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    # 可选：根据集群网卡名称设置。
    if NCCL_SOCKET_IFNAME is not None:
        env["NCCL_SOCKET_IFNAME"] = str(NCCL_SOCKET_IFNAME)
    if GLOO_SOCKET_IFNAME is not None:
        env["GLOO_SOCKET_IFNAME"] = str(GLOO_SOCKET_IFNAME)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
    ]

    if LAUNCH_MODE == "single_node":
        cmd += [
            "--standalone",
            f"--nproc_per_node={int(NPROC_PER_NODE)}",
            f"--max_restarts={int(DDP_MAX_RESTARTS)}",
        ]

    elif LAUNCH_MODE == "multi_node":
        cmd += [
            f"--nnodes={int(NNODES)}",
            f"--nproc_per_node={int(NPROC_PER_NODE)}",
            f"--node_rank={int(NODE_RANK)}",
            f"--rdzv_id={str(RDZV_ID)}",
            f"--rdzv_backend={str(RDZV_BACKEND)}",
            f"--rdzv_endpoint={str(RDZV_ENDPOINT)}",
            f"--max_restarts={int(DDP_MAX_RESTARTS)}",
        ]

    else:
        raise ValueError(f"Unknown LAUNCH_MODE: {LAUNCH_MODE}")

    cmd.append(str(Path(__file__).resolve()))

    print("=" * 80)
    print("[DDP launcher: weak_exp_train_query]")
    print(f"LAUNCH_MODE          = {LAUNCH_MODE}")
    print(f"NNODES               = {NNODES}")
    print(f"NODE_RANK            = {NODE_RANK}")
    print(f"NPROC_PER_NODE        = {NPROC_PER_NODE}")
    print(f"MASTER_ADDR           = {MASTER_ADDR}")
    print(f"MASTER_PORT           = {MASTER_PORT}")
    print(f"RDZV_ENDPOINT         = {RDZV_ENDPOINT}")
    print(f"CUDA_VISIBLE_DEVICES  = {env.get('CUDA_VISIBLE_DEVICES')}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)

    ret = subprocess.run(cmd, env=env)
    raise SystemExit(ret.returncode)


def build_args() -> argparse.Namespace:
    cfg = _load_pickle_config(Path(MODEL_CONFIG))
    num_classes = _infer_value(NUM_CLASSES, cfg, "num_classes", required=True)
    top_k = _infer_value(TOP_K, cfg, "top_k", required=True)
    max_len = _infer_value(MAX_LEN, cfg, "max_len", required=True)
    msa_max_size = _infer_value(MSA_MAX_SIZE, cfg, "msa_max_size", required=False)

    return argparse.Namespace(
        model_config=str(Path(MODEL_CONFIG)),
        init_ckpt=str(Path(INIT_CKPT)),
        file_address=str(Path(FILE_ADDRESS)),
        working_address=str(Path(WORKING_ADDRESS)),
        task=TASK,
        num_classes=int(num_classes),
        output_dir=str(Path(OUTPUT_DIR)),
        top_k=int(top_k),
        max_len=int(max_len),
        msa_max_size=msa_max_size,
        permute_dims=list(PERMUTE_DIMS),
        torch_compile=bool(TORCH_COMPILE),
        pseudo_prob_path=(str(Path(PSEUDO_PROB_PATH)) if bool(USE_PSEUDO_PROB) else None),
        pseudo_loss_type=str(PSEUDO_LOSS_TYPE),
        pseudo_prob_target=str(PSEUDO_PROB_TARGET),
        pseudo_prob_conf_power=float(PSEUDO_PROB_CONF_POWER),
        pseudo_prob_min_conf=float(PSEUDO_PROB_MIN_CONF),
        pseudo_negative_policy=str(PSEUDO_NEGATIVE_POLICY),
        pseudo_prob_neg_max=float(PSEUDO_PROB_NEG_MAX),
        pseudo_neg_topk=int(PSEUDO_NEG_TOPK),
        pseudo_min_ic=float(PSEUDO_MIN_IC),
        pseudo_asl_gamma_neg=float(PSEUDO_ASL_GAMMA_NEG),
        pseudo_asl_gamma_pos=float(PSEUDO_ASL_GAMMA_POS),
        pseudo_asl_clip=float(PSEUDO_ASL_CLIP),
        pseudo_asl_reduction=str(PSEUDO_ASL_REDUCTION),
        use_query_decoder=bool(USE_QUERY_DECODER),
        query_decoder_topk=int(QUERY_DECODER_TOPK),
        query_decoder_topk_source=str(QUERY_DECODER_TOPK_SOURCE),
        query_decoder_mode=str(QUERY_DECODER_MODE),
        query_decoder_dim=int(QUERY_DECODER_DIM),
        query_decoder_heads=int(QUERY_DECODER_HEADS),
        query_decoder_layers=int(QUERY_DECODER_LAYERS),
        query_decoder_ffn_dim=int(QUERY_DECODER_FFN_DIM),
        query_decoder_dropout=float(QUERY_DECODER_DROPOUT),
        query_decoder_detach_query_weight=bool(QUERY_DECODER_DETACH_QUERY_WEIGHT),
        query_decoder_include_label_boost=bool(QUERY_DECODER_INCLUDE_LABEL_BOOST),
        query_decoder_label_boost=float(QUERY_DECODER_LABEL_BOOST),
        query_decoder_memory_mode=str(QUERY_DECODER_MEMORY_MODE),
        query_decoder_memory_grid_h=int(QUERY_DECODER_MEMORY_GRID_H),
        query_decoder_memory_grid_w=int(QUERY_DECODER_MEMORY_GRID_W),
        true_asl_gamma_neg=float(TRUE_ASL_GAMMA_NEG),
        true_asl_gamma_pos=float(TRUE_ASL_GAMMA_POS),
        true_asl_clip=float(TRUE_ASL_CLIP),
        true_asl_reduction=str(TRUE_ASL_REDUCTION),
        gpu_ids=GPU_IDS,
        epochs=int(EPOCHS),
        batch_size=int(BATCH_SIZE),
        pseudo_batch_size=int(PSEUDO_BATCH_SIZE),
        dataloader_num_workers=int(DATALOADER_NUM_WORKERS),
        pin_memory=bool(PIN_MEMORY),
        drop_last=bool(DROP_LAST),
        max_steps_per_epoch=MAX_STEPS_PER_EPOCH,
        msa_read_mode=str(MSA_READ_MODE),
        msa_sample_strategy=str(MSA_SAMPLE_STRATEGY),
        msa_shuffle_rows_at_getitem=bool(MSA_SHUFFLE_ROWS_AT_GETITEM),
        msa_cache_gb=float(MSA_CACHE_GB),
        msa_max_open_files=int(MSA_MAX_OPEN_FILES),
        sample_seed=int(SAMPLE_SEED),
        sampler_seed=int(SAMPLER_SEED),
        prefetch_factor=int(PREFETCH_FACTOR),
        persistent_workers=bool(PERSISTENT_WORKERS),
        ddp_find_unused_parameters=bool(DDP_FIND_UNUSED_PARAMETERS),
        optim=str(OPTIM),
        optim_eps=float(OPTIM_EPS),
        lr=float(LR),
        lr_policy=str(LR_POLICY),
        lr_pct_start=float(LR_PCT_START),
        lr_cycle_three_phase=bool(LR_CYCLE_THREE_PHASE),
        lr_div_factor=float(LR_DIV_FACTOR),
        lr_final_div_factor=float(LR_FINAL_DIV_FACTOR),
        min_lr=float(MIN_LR),
        weight_decay=float(WEIGHT_DECAY),
        no_model_weight_decay=bool(NO_MODEL_WEIGHT_DECAY),
        accum_steps=int(ACCUM_STEPS),
        grad_clip=float(GRAD_CLIP),
        device=str(DEVICE),
        seed=int(SEED),
        no_amp=bool(NO_AMP),
        freeze_bn=bool(FREEZE_BN),
        freeze_bn_affine=bool(FREEZE_BN_AFFINE),
        lambda_true=float(LAMBDA_TRUE),
        lambda_pseudo=float(LAMBDA_PSEUDO),
        lambda_h=float(LAMBDA_H),
        ic_path=_path_or_none(IC_PATH),
        ic_alpha=float(IC_ALPHA),
        ic_min_count=int(IC_MIN_COUNT),
        go_edges_path=_path_or_none(GO_EDGES_PATH),
        log_interval=int(LOG_INTERVAL),
        save_interval=int(SAVE_INTERVAL),
        need_proteins=bool(NEED_PROTEINS),
    )


def print_run_summary(args: argparse.Namespace):
    print("=" * 80)
    print("[Run] Weak MSA-GO expansion training")
    print(f"TASK                 = {args.task}")
    print(f"MODEL_CONFIG         = {args.model_config}")
    print(f"INIT_CKPT            = {args.init_ckpt}")
    print(f"FILE_ADDRESS         = {args.file_address}")
    print(f"WORKING_ADDRESS      = {args.working_address}")
    print(f"OUTPUT_DIR           = {args.output_dir}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"USE_DDP              = {USE_DDP}")
    print(f"NPROC_PER_NODE       = {NPROC_PER_NODE}")
    print(f"NUM_CLASSES          = {args.num_classes}")
    print(f"TOP_K                = {args.top_k}")
    print(f"MAX_LEN              = {args.max_len}")
    print(f"EPOCHS               = {args.epochs}")
    print(f"BATCH_SIZE           = {args.batch_size}")
    print(f"PSEUDO_BATCH_SIZE    = {args.pseudo_batch_size}")
    print(f"OPTIM                = {args.optim}")
    print(f"LR                   = {args.lr}")
    print(f"LR_POLICY            = {args.lr_policy}")
    print(f"LAMBDA_TRUE          = {args.lambda_true}")
    print(f"LAMBDA_PSEUDO        = {args.lambda_pseudo}")
    print(f"LAMBDA_H             = {args.lambda_h}")
    print(f"PSEUDO_PROB_PATH     = {args.pseudo_prob_path}")
    print(f"PSEUDO_TARGET        = {args.pseudo_prob_target}")
    print(f"PSEUDO_NEG_POLICY    = {args.pseudo_negative_policy}")
    print(f"PSEUDO_ASL_REDUCTION = {args.pseudo_asl_reduction}")
    print(f"TRUE_ASL_REDUCTION   = {args.true_asl_reduction}")
    print(f"USE_QUERY_DECODER    = {args.use_query_decoder}")
    print(f"QUERY_DECODER_TOPK   = {args.query_decoder_topk}")
    print(f"QUERY_DECODER_SOURCE = {args.query_decoder_topk_source}")
    print(f"QUERY_DECODER_MODE   = {args.query_decoder_mode}")
    print(f"QUERY_DECODER_DIM    = {args.query_decoder_dim}")
    print(f"FREEZE_BN            = {args.freeze_bn}")
    print("=" * 80)


def prepare_imports():
    for p in (str(ROOT), str(MSA_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def main():
    _validate_paths()

    if _should_launch_ddp():
        _launch_ddp_and_exit()

    _validate_output_dir()
    args = build_args()

    if PRINT_CONFIG and _is_rank0_env():
        print_run_summary(args)
    if SAVE_RUNNER_CONFIG and _is_rank0_env():
        _save_json(vars(args), Path(args.output_dir) / "runner_config.json")
    if DRY_RUN:
        if _is_rank0_env():
            print("[Dry run] exit without training.")
        return

    prepare_imports()
    from experiments.weak_exp_train_query import train_one_task, normalize_task

    args.task = normalize_task(args.task)
    train_one_task(args)


if __name__ == "__main__":
    main()
