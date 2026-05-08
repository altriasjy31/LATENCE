#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_ind_test.py

Configuration-heavy runner for experiments/eval_ind_test.py.

Usage:
    python run_ind_test.py

This runner can optionally launch torchrun automatically, similar to run_exp_train.py.

Important:
    1. This should be used for final or pre-declared evaluation.
       Do not evaluate many checkpoints on ind_test and choose the best one;
       that would tune on the independent test set.
    2. The evaluator reports fmax and auprc via evalperf_torch.
    3. Predictions passed to evalperf_torch are sigmoid probabilities.
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
EVAL_SCRIPT = ROOT / "experiments" / "eval_ind_test.py"


# =============================================================================
# USER CONFIG
# =============================================================================

# ---------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------
# Allowed:
#   "mf", "bp", "cc"
#   "molecular_function", "biological_process", "cellular_component"
# TASK = "mf"
# TASK = "cc"
TASK = "bp"


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
MODEL_CONFIG = ROOT / "data" / "msa_models" / "configs" / "model_opts" / f"{TASK}_msa_model_config.pkl"

FILE_ADDRESS = ROOT / "data" / "unidata_with_exp_train_pseudo.pkl"
WORKING_ADDRESS = ROOT / "data" / "ind_MSA_bin" / "index.pkl"

# Choose one checkpoint.
#
# Compatible examples:
#   outputs/exp_train/<RUN_TAG>/semisup_backbone_last.pt
#   outputs/exp_train/<RUN_TAG>/semisup_backbone_epoch30.pt
#   outputs/exp_train/<RUN_TAG>/semisup_full_last.pt
#   outputs/exp_train/<RUN_TAG>/semisup_full_epoch30.pt
#
# If DO_VALIDATION=False during training, there may be no "best" checkpoint.
TRAIN_RUN_TAG = f"{TASK}_semisup_multi_node_v1"
# CKPT = ROOT / "outputs" / "exp_train" / TRAIN_RUN_TAG / "semisup_backbone_last.pt"
# Trained / student checkpoint.
# CKPT = ROOT / "outputs" / "exp_train" / TRAIN_RUN_TAG / "semisup_backbone_last.pt"
CKPT = ROOT / "data" / "msa_models" / "checkpoints" / f"{TASK}_msa_model_rank1.pt"

# Teacher checkpoint.
# Set to None to disable teacher-student averaging.
# TEACHER_CKPT = ROOT / "data" / "msa_models" / "checkpoints" / f"{TASK}_msa_model_rank1.pt"
TEACHER_CKPT = None

# Probability-average ensemble weights.
# With 1.0 / 1.0, final prediction is:
#   0.5 * p_student + 0.5 * p_teacher
ENSEMBLE_STUDENT_WEIGHT = 1.0
ENSEMBLE_TEACHER_WEIGHT = 1.0

# Evaluation split inside FILE_ADDRESS.
EVAL_MODE = "ind_test"


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
# RUN_TAG = f"{TASK}_ind_test_semisup_backbone_last"
# RUN_TAG = f"{TASK}_ind_test_teacher_student_avg"
RUN_TAG = f"{TASK}_ind_test_teacher_only"
OUTPUT_ROOT = ROOT / "outputs" / "ind_test"
OUTPUT_DIR = OUTPUT_ROOT / RUN_TAG

FAIL_IF_OUTPUT_EXISTS = False


# ---------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------
CUDA_VISIBLE_DEVICES = "0,1"
DEVICE = "auto"

# DDP eval is supported.
# The evaluator does not wrap the model with DDP; each rank evaluates a disjoint shard.
GPU_IDS = "auto"

USE_DDP = True
NPROC_PER_NODE = 2
DDP_STANDALONE = True
MASTER_PORT = 29541


# ---------------------------------------------------------------------
# Dataset/model compatibility
# ---------------------------------------------------------------------
# If None, infer from MODEL_CONFIG.
NUM_CLASSES = None
TOP_K = None
MAX_LEN = None
MSA_MAX_SIZE = None

PERMUTE_DIMS = (0, 3, 2, 1)
TORCH_COMPILE = False


# ---------------------------------------------------------------------
# Eval loader
# ---------------------------------------------------------------------
SEED = 3407

EVAL_BATCH_SIZE = 8
DATALOADER_NUM_WORKERS = 4

# For final evaluation, deterministic MSA sampling is usually preferable.
# If your benchmark protocol requires stochastic row sampling, change this to "random"
# and keep SAMPLE_SEED fixed.
MSA_READ_MODE = "full"
MSA_SAMPLE_STRATEGY = "head"
MSA_SHUFFLE_ROWS_AT_GETITEM = False

# This is per DataLoader worker.
# Eval has one loader, so approximate CPU cache upper bound:
#   world_size × num_workers × MSA_CACHE_GB
MSA_CACHE_GB = 4.0
MSA_MAX_OPEN_FILES = 256

SAMPLE_SEED = 1

PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
PIN_MEMORY = True

NO_AMP = False


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
# evalperf_torch returns fmax and auprc in percent.
REPORT_THRESHOLD = True

# Keep both False to match evalperf_torch default behavior.
# Only set True if you intentionally want those filtering conventions.
NO_EMPTY_LABELS = False
NO_ZERO_CLASSES = False


# ---------------------------------------------------------------------
# Distributed prediction collection
# ---------------------------------------------------------------------
# "file" avoids gathering all predictions to every rank.
# Requires OUTPUT_DIR to be on a filesystem visible to all ranks.
DISTRIBUTED_COLLECT = "file"  # "file" or "all_gather_object"
KEEP_PART_FILES = False


# ---------------------------------------------------------------------
# Optional
# ---------------------------------------------------------------------
NEED_PROTEINS = False
STRICT_SHAPE = True
SAVE_PREDICTIONS = False


# ---------------------------------------------------------------------
# Debug / logging
# ---------------------------------------------------------------------
DRY_RUN = False
PRINT_CONFIG = True
SAVE_RUNNER_CONFIG = True


# =============================================================================
# Apply CUDA_VISIBLE_DEVICES before importing torch / evaluator
# =============================================================================

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


def _infer_value(
    explicit_value: Any,
    cfg: Dict[str, Any],
    key: str,
    required: bool = True,
):
    if explicit_value is not None:
        return explicit_value

    value = cfg.get(key, None)

    if required and value is None:
        raise ValueError(
            f"Could not infer required field '{key}' from MODEL_CONFIG. "
            f"Set {key.upper()} explicitly in run_ind_test.py."
        )

    return value


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


def _rank_from_env() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_rank0_env() -> bool:
    return _rank_from_env() == 0


def _under_torchrun() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


def _is_ddp_child_arg() -> bool:
    return "--ddp-child" in sys.argv


def _should_launch_ddp() -> bool:
    return (
        bool(USE_DDP)
        and int(NPROC_PER_NODE) > 1
        and not _under_torchrun()
        and not _is_ddp_child_arg()
    )


def _validate_paths():
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot find eval script: {EVAL_SCRIPT}")

    if not MSA_ROOT.exists():
        raise FileNotFoundError(f"Cannot find msa_models directory: {MSA_ROOT}")

    if not Path(MODEL_CONFIG).is_file():
        raise FileNotFoundError(f"MODEL_CONFIG not found: {MODEL_CONFIG}")

    if not Path(CKPT).is_file():
        raise FileNotFoundError(f"CKPT not found: {CKPT}")

    if TEACHER_CKPT is not None and not Path(TEACHER_CKPT).is_file():
        raise FileNotFoundError(f"TEACHER_CKPT not found: {TEACHER_CKPT}")

    if not Path(FILE_ADDRESS).is_file():
        raise FileNotFoundError(f"FILE_ADDRESS not found: {FILE_ADDRESS}")

    if not Path(WORKING_ADDRESS).exists():
        warnings.warn(
            f"WORKING_ADDRESS does not exist: {WORKING_ADDRESS}. "
            f"Continue only if MSABinaryDataset can handle this path."
        )


def _validate_output_dir():
    output_dir = Path(OUTPUT_DIR)

    if _is_rank0_env():
        if output_dir.exists() and any(output_dir.iterdir()) and FAIL_IF_OUTPUT_EXISTS:
            raise FileExistsError(
                f"OUTPUT_DIR already exists and is not empty:\n"
                f"  {output_dir}\n"
                f"To reuse it, set FAIL_IF_OUTPUT_EXISTS = False, "
                f"or change RUN_TAG / OUTPUT_DIR."
            )

    output_dir.mkdir(parents=True, exist_ok=True)


def _launch_ddp_and_exit():
    env = os.environ.copy()

    if CUDA_VISIBLE_DEVICES is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
    ]

    if DDP_STANDALONE:
        cmd.append("--standalone")
    else:
        cmd.extend(["--master_port", str(MASTER_PORT)])

    cmd.extend(
        [
            "--nproc_per_node",
            str(NPROC_PER_NODE),
            str(Path(__file__).resolve()),
            "--ddp-child",
        ]
    )

    print("=" * 80)
    print("[DDP launcher]")
    print("CUDA_VISIBLE_DEVICES =", env.get("CUDA_VISIBLE_DEVICES"))
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

    args = argparse.Namespace(
        # Required
        model_config=str(Path(MODEL_CONFIG)),
        ckpt=str(Path(CKPT)),
        teacher_ckpt=(
            None if TEACHER_CKPT is None else str(Path(TEACHER_CKPT))
        ),
        ensemble_student_weight=float(ENSEMBLE_STUDENT_WEIGHT),
        ensemble_teacher_weight=float(ENSEMBLE_TEACHER_WEIGHT),
        file_address=str(Path(FILE_ADDRESS)),
        working_address=str(Path(WORKING_ADDRESS)),
        task=TASK,
        num_classes=int(num_classes),
        output_dir=str(Path(OUTPUT_DIR)),

        # Split
        mode=str(EVAL_MODE),

        # Dataset/model compatibility
        top_k=int(top_k),
        max_len=int(max_len),
        msa_max_size=msa_max_size,
        permute_dims=list(PERMUTE_DIMS),
        torch_compile=bool(TORCH_COMPILE),

        # GPU
        gpu_ids=GPU_IDS,
        device=str(DEVICE),
        seed=int(SEED),
        no_amp=bool(NO_AMP),

        # Loader
        eval_batch_size=int(EVAL_BATCH_SIZE),
        dataloader_num_workers=int(DATALOADER_NUM_WORKERS),
        pin_memory=bool(PIN_MEMORY),
        prefetch_factor=int(PREFETCH_FACTOR),
        persistent_workers=bool(PERSISTENT_WORKERS),

        # Binary MSA
        msa_read_mode=str(MSA_READ_MODE),
        msa_sample_strategy=str(MSA_SAMPLE_STRATEGY),
        msa_shuffle_rows_at_getitem=bool(MSA_SHUFFLE_ROWS_AT_GETITEM),
        msa_cache_gb=float(MSA_CACHE_GB),
        msa_max_open_files=int(MSA_MAX_OPEN_FILES),
        sample_seed=int(SAMPLE_SEED),

        # Optional model behavior
        need_proteins=bool(NEED_PROTEINS),

        # Checkpoint
        strict_shape=bool(STRICT_SHAPE),

        # Metrics
        report_threshold=bool(REPORT_THRESHOLD),
        no_empty_labels=bool(NO_EMPTY_LABELS),
        no_zero_classes=bool(NO_ZERO_CLASSES),

        # Distributed collection
        distributed_collect=str(DISTRIBUTED_COLLECT),
        keep_part_files=bool(KEEP_PART_FILES),

        # Output
        save_predictions=bool(SAVE_PREDICTIONS),
    )

    return args


def print_run_summary(args: argparse.Namespace):
    print("=" * 80)
    print("[Run] MSA-GO independent test evaluation")
    print("-" * 80)
    print(f"ROOT                    = {ROOT}")
    print(f"EVAL_SCRIPT             = {EVAL_SCRIPT}")
    print(f"TASK                    = {args.task}")
    print(f"MODE                    = {args.mode}")
    print(f"MODEL_CONFIG            = {args.model_config}")
    print(f"CKPT                    = {args.ckpt}")
    print(f"TEACHER_CKPT            = {args.teacher_ckpt}")
    print(f"ENSEMBLE_STUDENT_WEIGHT = {args.ensemble_student_weight}")
    print(f"ENSEMBLE_TEACHER_WEIGHT = {args.ensemble_teacher_weight}")
    print(f"FILE_ADDRESS            = {args.file_address}")
    print(f"WORKING_ADDRESS         = {args.working_address}")
    print(f"OUTPUT_DIR              = {args.output_dir}")
    print(f"CUDA_VISIBLE_DEVICES    = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"DEVICE                  = {args.device}")
    print(f"GPU_IDS                 = {args.gpu_ids}")
    print(f"USE_DDP                 = {USE_DDP}")
    print(f"NPROC_PER_NODE          = {NPROC_PER_NODE}")
    print(f"NUM_CLASSES             = {args.num_classes}")
    print(f"TOP_K                   = {args.top_k}")
    print(f"MAX_LEN                 = {args.max_len}")
    print(f"MSA_MAX_SIZE             = {args.msa_max_size}")
    print(f"EVAL_BATCH_SIZE         = {args.eval_batch_size}")
    print(f"MSA_READ_MODE           = {args.msa_read_mode}")
    print(f"MSA_SAMPLE_STRATEGY     = {args.msa_sample_strategy}")
    print(f"MSA_SHUFFLE_ROWS        = {args.msa_shuffle_rows_at_getitem}")
    print(f"MSA_CACHE_GB            = {args.msa_cache_gb}")
    print(f"NO_AMP                  = {args.no_amp}")
    print(f"REPORT_THRESHOLD        = {args.report_threshold}")
    print(f"NO_EMPTY_LABELS         = {args.no_empty_labels}")
    print(f"NO_ZERO_CLASSES         = {args.no_zero_classes}")
    print(f"SAVE_PREDICTIONS        = {args.save_predictions}")
    print(f"DISTRIBUTED_COLLECT     = {args.distributed_collect}")
    print("=" * 80)


def prepare_imports():
    root_str = str(ROOT)
    msa_str = str(MSA_ROOT)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if msa_str not in sys.path:
        sys.path.insert(0, msa_str)


def main():
    _validate_paths()

    # Parent process: launch torchrun and exit.
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
            print("[Dry run] Configuration built successfully. Exit without evaluation.")
        return

    prepare_imports()

    from experiments.eval_ind_test import evaluate_one_task

    evaluate_one_task(args)


if __name__ == "__main__":
    main()