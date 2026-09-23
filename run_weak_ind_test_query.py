#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_weak_ind_test_query.py

Configuration-heavy runner for experiments/eval_weak_ind_test_query.py.

Usage:
    python run_weak_ind_test_query.py

This can evaluate:
    1. weak_query_decoder_epoch*.pt with query decoder enabled;
    2. weak_backbone_epoch*.pt with query decoder disabled;
    3. optional teacher ensemble by probability averaging.
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


# =============================================================================
# Project paths
# =============================================================================

ROOT = Path(__file__).resolve().parent
MSA_ROOT = ROOT / "msa_models"
EVAL_SCRIPT = ROOT / "experiments" / "eval_weak_ind_test_query.py"


# =============================================================================
# USER CONFIG
# =============================================================================

TASK = os.environ.get("TASK", "bp")  # "cc", "mf", "bp"

MODEL_CONFIG = ROOT / "data" / "msa_models" / "configs" / "model_opts" / f"{TASK}_msa_model_config.pkl"
FILE_ADDRESS = ROOT / "data" / "unidata_with_exp_train_pseudo.pkl"
WORKING_ADDRESS = ROOT / "data" / "ind_MSA_bin" / "index.pkl"

# Training output directory.
TRAIN_RUN_TAG = f"{TASK}_weak_asl_prob_v1"
TRAIN_OUTPUT_DIR = ROOT / "outputs" / "weak_exp_train_query" / TRAIN_RUN_TAG

# ---------------------------------------------------------------------
# Checkpoint choice
# ---------------------------------------------------------------------
# Options:
#   "weak_query_epoch"
#   "weak_query_last"
#   "weak_backbone_epoch"
#   "weak_backbone_last"
#   "teacher_only"
CKPT_KIND = "weak_query_epoch"
CKPT_EPOCH = 50

TEACHER_ORIGINAL_CKPT = ROOT / "data" / "msa_models" / "checkpoints" / f"{TASK}_msa_model_rank1.pt"

if CKPT_KIND == "weak_query_epoch":
    if CKPT_EPOCH is None:
        raise ValueError("CKPT_EPOCH is required for weak_query_epoch")
    CKPT = TRAIN_OUTPUT_DIR / f"weak_query_decoder_epoch{CKPT_EPOCH}.pt"
    USE_QUERY_DECODER = True
    CHECKPOINT_ROLE = "weak_query"
elif CKPT_KIND == "weak_query_last":
    CKPT = TRAIN_OUTPUT_DIR / "weak_query_decoder_last.pt"
    USE_QUERY_DECODER = True
    CHECKPOINT_ROLE = "weak_query"
elif CKPT_KIND == "weak_backbone_epoch":
    if CKPT_EPOCH is None:
        raise ValueError("CKPT_EPOCH is required for weak_backbone_epoch")
    CKPT = TRAIN_OUTPUT_DIR / f"weak_backbone_epoch{CKPT_EPOCH}.pt"
    USE_QUERY_DECODER = False
    CHECKPOINT_ROLE = "weak_backbone"
elif CKPT_KIND == "weak_backbone_last":
    CKPT = TRAIN_OUTPUT_DIR / "weak_backbone_last.pt"
    USE_QUERY_DECODER = False
    CHECKPOINT_ROLE = "weak_backbone"
elif CKPT_KIND == "teacher_only":
    CKPT = TEACHER_ORIGINAL_CKPT
    USE_QUERY_DECODER = False
    CHECKPOINT_ROLE = "teacher"
else:
    raise ValueError(f"Unknown CKPT_KIND: {CKPT_KIND}")

# Optional teacher probability-average ensemble.
# Set TEACHER_CKPT = None for primary-only evaluation.
TEACHER_CKPT = None
# TEACHER_CKPT = TEACHER_ORIGINAL_CKPT

ENSEMBLE_PRIMARY_WEIGHT = 1.0
ENSEMBLE_TEACHER_WEIGHT = 0.0
# Example for teacher + query ensemble:
# TEACHER_CKPT = TEACHER_ORIGINAL_CKPT
# ENSEMBLE_PRIMARY_WEIGHT = 0.5
# ENSEMBLE_TEACHER_WEIGHT = 0.5

EVAL_MODE = "ind_test"

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
RUN_TAG = f"{TASK}_ind_test_{CKPT_KIND}"
if CKPT_EPOCH is not None:
    RUN_TAG += f"_epoch{CKPT_EPOCH}"
if TEACHER_CKPT is not None:
    RUN_TAG += f"_ens_p{ENSEMBLE_PRIMARY_WEIGHT}_t{ENSEMBLE_TEACHER_WEIGHT}"

OUTPUT_ROOT = ROOT / "outputs" / "ind_test_weak_query"
OUTPUT_DIR = OUTPUT_ROOT / RUN_TAG
FAIL_IF_OUTPUT_EXISTS = False

# ---------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------
CUDA_VISIBLE_DEVICES = "0"
DEVICE = "auto"
GPU_IDS = "auto"

USE_DDP = True
NPROC_PER_NODE = 1
DDP_STANDALONE = True
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29543
DDP_MAX_RESTARTS = 0

# ---------------------------------------------------------------------
# Dataset/model compatibility
# ---------------------------------------------------------------------
NUM_CLASSES = None
TOP_K = None
MAX_LEN = None
MSA_MAX_SIZE = None
PERMUTE_DIMS = (0, 3, 2, 1)
TORCH_COMPILE = False

# Query decoder settings must match training.
QUERY_DECODER_TOPK = 100
QUERY_DECODER_MODE = "residual"
QUERY_DECODER_DIM = 256
QUERY_DECODER_HEADS = 8
QUERY_DECODER_LAYERS = 1
QUERY_DECODER_FFN_DIM = 1024
QUERY_DECODER_DROPOUT = 0.1
QUERY_DECODER_DETACH_QUERY_WEIGHT = True
QUERY_DECODER_MEMORY_MODE = "tokens_plus_pooled"
QUERY_DECODER_MEMORY_GRID_H = 0
QUERY_DECODER_MEMORY_GRID_W = 0
STRICT_QUERY_DECODER = True

# ---------------------------------------------------------------------
# Eval loader
# ---------------------------------------------------------------------
SEED = 3407
EVAL_BATCH_SIZE = 8
DATALOADER_NUM_WORKERS = 4

MSA_READ_MODE = "full"
MSA_SAMPLE_STRATEGY = "head"
MSA_SHUFFLE_ROWS_AT_GETITEM = False
MSA_CACHE_GB = 2.0
MSA_MAX_OPEN_FILES = 256
SAMPLE_SEED = 1
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
PIN_MEMORY = True
NO_AMP = False

# ---------------------------------------------------------------------
# Metrics / collection
# ---------------------------------------------------------------------
REPORT_THRESHOLD = True
NO_EMPTY_LABELS = False
NO_ZERO_CLASSES = False
DISTRIBUTED_COLLECT = "file"
KEEP_PART_FILES = False
SAVE_PREDICTIONS = False

# ---------------------------------------------------------------------
# Optional
# ---------------------------------------------------------------------
NEED_PROTEINS = False
STRICT_SHAPE = True
DRY_RUN = False
PRINT_CONFIG = True
SAVE_RUNNER_CONFIG = True

if CUDA_VISIBLE_DEVICES is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)


# =============================================================================
# Utilities
# =============================================================================

def _load_pickle_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"MODEL_CONFIG must contain a dict, got {type(obj)}")
    return obj


def _infer_value(explicit_value: Any, cfg: Dict[str, Any], key: str, required: bool = True):
    if explicit_value is not None:
        return explicit_value
    value = cfg.get(key, None)
    if required and value is None:
        raise ValueError(
            f"Could not infer required field '{key}' from MODEL_CONFIG. "
            f"Set {key.upper()} explicitly."
        )
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
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find eval script: {EVAL_SCRIPT}")
    if not MSA_ROOT.exists():
        raise FileNotFoundError(f"Cannot find msa_models: {MSA_ROOT}")
    if not Path(MODEL_CONFIG).is_file():
        raise FileNotFoundError(f"MODEL_CONFIG not found: {MODEL_CONFIG}")
    if not Path(CKPT).is_file():
        raise FileNotFoundError(f"CKPT not found: {CKPT}")
    if TEACHER_CKPT is not None and not Path(TEACHER_CKPT).is_file():
        raise FileNotFoundError(f"TEACHER_CKPT not found: {TEACHER_CKPT}")
    if not Path(FILE_ADDRESS).is_file():
        raise FileNotFoundError(f"FILE_ADDRESS not found: {FILE_ADDRESS}")
    if not Path(WORKING_ADDRESS).is_file():
        warnings.warn(f"WORKING_ADDRESS is not a file: {WORKING_ADDRESS}")


def _validate_output_dir():
    if Path(OUTPUT_DIR).exists() and any(Path(OUTPUT_DIR).iterdir()) and FAIL_IF_OUTPUT_EXISTS:
        raise FileExistsError(f"OUTPUT_DIR exists and is not empty: {OUTPUT_DIR}")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


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
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
    ]

    if DDP_STANDALONE:
        cmd.append("--standalone")
    else:
        cmd += ["--nnodes=1", f"--master_addr={MASTER_ADDR}"]

    cmd += [
        f"--nproc_per_node={int(NPROC_PER_NODE)}",
        f"--master_port={int(MASTER_PORT)}",
        f"--max_restarts={int(DDP_MAX_RESTARTS)}",
        str(Path(__file__).resolve()),
        "--ddp-child",
    ]

    print("=" * 80)
    print("[DDP launcher: eval weak query]")
    print("CUDA_VISIBLE_DEVICES =", env.get("CUDA_VISIBLE_DEVICES"))
    print("Command:", " ".join(cmd))
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
        ckpt=str(Path(CKPT)),
        file_address=str(Path(FILE_ADDRESS)),
        working_address=str(Path(WORKING_ADDRESS)),
        task=TASK,
        num_classes=int(num_classes),
        output_dir=str(Path(OUTPUT_DIR)),
        mode=str(EVAL_MODE),

        teacher_ckpt=_path_or_none(TEACHER_CKPT),
        ensemble_primary_weight=float(ENSEMBLE_PRIMARY_WEIGHT),
        ensemble_teacher_weight=float(ENSEMBLE_TEACHER_WEIGHT),
        checkpoint_role=str(CHECKPOINT_ROLE),

        top_k=int(top_k),
        max_len=int(max_len),
        msa_max_size=msa_max_size,
        permute_dims=list(PERMUTE_DIMS),
        torch_compile=bool(TORCH_COMPILE),

        msa_read_mode=str(MSA_READ_MODE),
        msa_sample_strategy=str(MSA_SAMPLE_STRATEGY),
        msa_shuffle_rows_at_getitem=bool(MSA_SHUFFLE_ROWS_AT_GETITEM),
        msa_cache_gb=float(MSA_CACHE_GB),
        msa_max_open_files=int(MSA_MAX_OPEN_FILES),
        sample_seed=int(SAMPLE_SEED),

        eval_batch_size=int(EVAL_BATCH_SIZE),
        dataloader_num_workers=int(DATALOADER_NUM_WORKERS),
        pin_memory=bool(PIN_MEMORY),
        prefetch_factor=int(PREFETCH_FACTOR),
        persistent_workers=bool(PERSISTENT_WORKERS),

        gpu_ids=str(GPU_IDS),
        device=str(DEVICE),
        seed=int(SEED),
        no_amp=bool(NO_AMP),
        need_proteins=bool(NEED_PROTEINS),
        strict_shape=bool(STRICT_SHAPE),
        strict_query_decoder=bool(STRICT_QUERY_DECODER),

        use_query_decoder=bool(USE_QUERY_DECODER),
        query_decoder_topk=int(QUERY_DECODER_TOPK),
        query_decoder_mode=str(QUERY_DECODER_MODE),
        query_decoder_dim=int(QUERY_DECODER_DIM),
        query_decoder_heads=int(QUERY_DECODER_HEADS),
        query_decoder_layers=int(QUERY_DECODER_LAYERS),
        query_decoder_ffn_dim=int(QUERY_DECODER_FFN_DIM),
        query_decoder_dropout=float(QUERY_DECODER_DROPOUT),
        query_decoder_detach_query_weight=bool(QUERY_DECODER_DETACH_QUERY_WEIGHT),
        query_decoder_memory_mode=str(QUERY_DECODER_MEMORY_MODE),
        query_decoder_memory_grid_h=int(QUERY_DECODER_MEMORY_GRID_H),
        query_decoder_memory_grid_w=int(QUERY_DECODER_MEMORY_GRID_W),

        report_threshold=bool(REPORT_THRESHOLD),
        no_empty_labels=bool(NO_EMPTY_LABELS),
        no_zero_classes=bool(NO_ZERO_CLASSES),
        distributed_collect=str(DISTRIBUTED_COLLECT),
        keep_part_files=bool(KEEP_PART_FILES),
        save_predictions=bool(SAVE_PREDICTIONS),
    )


def print_run_summary(args: argparse.Namespace):
    print("=" * 80)
    print("[Run] weak query ind_test evaluation")
    print("-" * 80)
    print(f"TASK                  = {TASK}")
    print(f"CKPT_KIND             = {CKPT_KIND}")
    print(f"CKPT_EPOCH            = {CKPT_EPOCH}")
    print(f"CKPT                  = {args.ckpt}")
    print(f"CHECKPOINT_ROLE       = {args.checkpoint_role}")
    print(f"USE_QUERY_DECODER     = {args.use_query_decoder}")
    print(f"TEACHER_CKPT          = {args.teacher_ckpt}")
    print(f"ENSEMBLE_WEIGHTS      = primary:{args.ensemble_primary_weight}, teacher:{args.ensemble_teacher_weight}")
    print(f"MODEL_CONFIG          = {args.model_config}")
    print(f"FILE_ADDRESS          = {args.file_address}")
    print(f"WORKING_ADDRESS       = {args.working_address}")
    print(f"OUTPUT_DIR            = {args.output_dir}")
    print(f"CUDA_VISIBLE_DEVICES  = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"USE_DDP               = {USE_DDP}")
    print(f"NPROC_PER_NODE        = {NPROC_PER_NODE}")
    print(f"NUM_CLASSES           = {args.num_classes}")
    print(f"TOP_K                 = {args.top_k}")
    print(f"MAX_LEN               = {args.max_len}")
    print(f"QUERY_TOPK            = {args.query_decoder_topk}")
    print(f"QUERY_MEMORY_MODE     = {args.query_decoder_memory_mode}")
    print(f"EVAL_BATCH_SIZE       = {args.eval_batch_size}")
    print("=" * 80)


def main():
    if _should_launch_ddp():
        _launch_ddp_and_exit()

    _validate_paths()
    _validate_output_dir()

    args = build_args()

    if PRINT_CONFIG and int(os.environ.get("RANK", "0")) == 0:
        print_run_summary(args)

    if SAVE_RUNNER_CONFIG and int(os.environ.get("RANK", "0")) == 0:
        _save_json(vars(args), Path(args.output_dir) / "runner_config.json")

    if DRY_RUN:
        print("[Dry run] Configuration built successfully. Exit without evaluation.")
        return

    root_str = str(ROOT)
    msa_str = str(MSA_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if msa_str not in sys.path:
        sys.path.insert(0, msa_str)

    from experiments.eval_weak_ind_test_query import evaluate_one_task, normalize_task

    args.task = normalize_task(args.task)
    evaluate_one_task(args)


if __name__ == "__main__":
    main()
