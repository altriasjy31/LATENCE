#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_exp_train.py

Single-task runner for experiments/exp_train.py.

This file is intentionally configuration-heavy:
the constants below are the run record.

Usage:
    python run_exp_train.py

Debug:
    python -m pdb run_exp_train.py

Important:
    1. This runner directly imports and calls train_one_task(args).
       It does not spawn a subprocess.
    2. CUDA_VISIBLE_DEVICES is set before importing experiments.exp_train,
       so torch should see the intended visible GPU.
    3. You should edit the constants in "USER CONFIG" before running.
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# Project paths
# =============================================================================

ROOT = Path(__file__).resolve().parent
MSA_ROOT = ROOT / "msa_models"
EXP_TRAIN = ROOT / "experiments" / "exp_train.py"


# =============================================================================
# USER CONFIG
# =============================================================================
# Edit this section for each run.
# =============================================================================

# ---------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------
# Allowed:
#   "mf", "bp", "cc"
#   "molecular_function", "biological_process", "cellular_component"
TASK = "bp"


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
MODEL_CONFIG = ROOT / "data" / "msa_models" / "configs" / "model_opts" / f"{TASK}_msa_model_config.pkl"
INIT_CKPT = ROOT / "data" / "msa_models" / "checkpoints" / f"{TASK}_msa_model_rank1.pt"

# Dataset pickle should contain:
#   data["train"][task]
#   data["exp_train"][task]
# optionally:
#   data["test"][task] or validation split
FILE_ADDRESS = ROOT / "data" / "dataset_with_exp_train_pseudo.pkl"

# MSA working directory used by MSADataset.
# WORKING_ADDRESS = ROOT / "data" / "sprot_2204_MSA"
WORKING_ADDRESS = ROOT / "data" / "sprot_2204_MSA_bin" / "index.pkl"

# Optional IC and hierarchy files.
# If IC_PATH is None, exp_train.py computes IC from train labels.
IC_PATH = None
GO_EDGES_PATH = None


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
RUN_TAG = f"{TASK}_semisup_v1"

OUTPUT_ROOT = ROOT / "outputs" / "exp_train"
OUTPUT_DIR = OUTPUT_ROOT / RUN_TAG

# Safer default: prevent accidental log/checkpoint mixing.
# Set False if you intentionally want to reuse the directory.
FAIL_IF_OUTPUT_EXISTS = False


# ---------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES is applied before importing torch via exp_train.py.
# Use "0" for one GPU, "1" for another GPU, "0,1" if you later extend to multi-GPU.
CUDA_VISIBLE_DEVICES = "1,2,3"

# exp_train.py --device
# Usually keep "auto".
DEVICE = "auto"

# exp_train.py --gpu_ids
# In a single-process run with CUDA_VISIBLE_DEVICES="0", this should usually be "0".
# Use "-1" for CPU.
GPU_IDS = "0"


# ---------------------------------------------------------------------
# Dataset/model compatibility
# ---------------------------------------------------------------------
# If None, infer from MODEL_CONFIG.
# Keeping these as explicit constants is useful when you want the run file
# to fully record the shape settings.
NUM_CLASSES = None
TOP_K = None
MAX_LEN = None
MSA_MAX_SIZE = None

PERMUTE_DIMS = (0, 3, 2, 1)

# Usually False for this augmentation-heavy script.
TORCH_COMPILE = False


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
SEED = 3407

EPOCHS = 20
BATCH_SIZE = 24
PSEUDO_BATCH_SIZE = 20
EVAL_BATCH_SIZE = 8
DATALOADER_NUM_WORKERS = 4

PIN_MEMORY = True
DROP_LAST = False

LR = 2e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

NO_AMP = False


# ---------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------
# Must match pooled embedding dimension returned by:
#   logits, h = Arch(..., return_embedding=True)
#
# If h.shape == [B, 2048, H, W], and exp_train.py uses adaptive avg pooling,
# then PROJ_IN_DIM should be 2048.
PROJ_IN_DIM = 2048
PROJ_HIDDEN_DIM = 1024
PROJ_DIM = 128
PROJ_DROPOUT = 0.1


# ---------------------------------------------------------------------
# Semi-supervised losses
# ---------------------------------------------------------------------
LAMBDA_U = 0.5
LAMBDA_C = 0.05
LAMBDA_H = 0.0

# Important for GO pseudo labels:
# True means pseudo BCE only supervises pseudo-positive terms.
# This avoids treating unknown GO terms as negatives.
PSEUDO_POS_ONLY = True


# ---------------------------------------------------------------------
# GO-aware contrastive loss
# ---------------------------------------------------------------------
CONTRAST_TAU = 0.1
CONTRAST_MIN_R = 1e-6
W_TRUE_PSEUDO = 0.7
W_PSEUDO_PSEUDO = 0.4


# ---------------------------------------------------------------------
# IC / hierarchy
# ---------------------------------------------------------------------
IC_ALPHA = 1.0
IC_MIN_COUNT = 2


# ---------------------------------------------------------------------
# MSA weak augmentation
# ---------------------------------------------------------------------
WEAK_ROW_DROP_P = 0.05
WEAK_COL_MASK_P = 0.02
WEAK_BLOCK_MASK_P = 0.0
WEAK_BLOCK_MASK_MIN = 0.01
WEAK_BLOCK_MASK_MAX = 0.03
WEAK_N_BLOCKS = 1
WEAK_SHUFFLE_ROWS = False
WEAK_MIN_KEEP_ROWS = 2
WEAK_NOISE_STD = 0.0


# ---------------------------------------------------------------------
# MSA strong augmentation
# ---------------------------------------------------------------------
STRONG_ROW_DROP_P = 0.20
STRONG_COL_MASK_P = 0.08
STRONG_BLOCK_MASK_P = 0.30
STRONG_BLOCK_MASK_MIN = 0.02
STRONG_BLOCK_MASK_MAX = 0.06
STRONG_N_BLOCKS = 1
STRONG_SHUFFLE_ROWS = False
STRONG_MIN_KEEP_ROWS = 2
STRONG_NOISE_STD = 0.0


# ---------------------------------------------------------------------
# Validation / logging / saving
# ---------------------------------------------------------------------
# Default False to avoid accidentally selecting checkpoint on test set.
DO_VALIDATION = False
VAL_MODE = "test"

# If you intentionally want to use test as validation, set this True.
# This is usually not recommended for final evaluation.
ALLOW_TEST_VALIDATION = False

LOG_INTERVAL = 20
SAVE_INTERVAL = 5

# Usually False unless the model needs protein names via set_proteins().
NEED_PROTEINS = False


# ---------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------
DRY_RUN = False
PRINT_CONFIG = True
SAVE_RUNNER_CONFIG = True


# =============================================================================
# Apply CUDA_VISIBLE_DEVICES before importing exp_train.py / torch
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
            f"Set {key.upper()} explicitly in run_exp_train.py."
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
    if not EXP_TRAIN.exists():
        raise FileNotFoundError(f"Cannot find exp_train.py: {EXP_TRAIN}")

    if not MSA_ROOT.exists():
        raise FileNotFoundError(f"Cannot find msa_models directory: {MSA_ROOT}")

    if not Path(MODEL_CONFIG).is_file():
        raise FileNotFoundError(f"MODEL_CONFIG not found: {MODEL_CONFIG}")

    if not Path(INIT_CKPT).is_file():
        raise FileNotFoundError(f"INIT_CKPT not found: {INIT_CKPT}")

    if not Path(FILE_ADDRESS).is_file():
        raise FileNotFoundError(f"FILE_ADDRESS not found: {FILE_ADDRESS}")

    if not Path(WORKING_ADDRESS).exists():
        warnings.warn(
            f"WORKING_ADDRESS does not exist: {WORKING_ADDRESS}. "
            f"Continue only if MSADataset can handle this path."
        )

    if IC_PATH is not None and not Path(IC_PATH).is_file():
        raise FileNotFoundError(f"IC_PATH not found: {IC_PATH}")

    if GO_EDGES_PATH is not None and not Path(GO_EDGES_PATH).is_file():
        raise FileNotFoundError(f"GO_EDGES_PATH not found: {GO_EDGES_PATH}")


def _validate_output_dir():
    output_dir = Path(OUTPUT_DIR)

    if output_dir.exists() and any(output_dir.iterdir()) and FAIL_IF_OUTPUT_EXISTS:
        raise FileExistsError(
            f"OUTPUT_DIR already exists and is not empty:\n"
            f"  {output_dir}\n"
            f"To reuse it, set FAIL_IF_OUTPUT_EXISTS = False, "
            f"or change RUN_TAG / OUTPUT_DIR."
        )

    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_semantics():
    if PROJ_IN_DIM is None or int(PROJ_IN_DIM) <= 0:
        raise ValueError("PROJ_IN_DIM must be a positive integer.")

    if LAMBDA_H != 0.0 and GO_EDGES_PATH is None:
        warnings.warn(
            f"LAMBDA_H={LAMBDA_H}, but GO_EDGES_PATH is None. "
            f"Hierarchy loss will be zero."
        )

    if DO_VALIDATION and VAL_MODE in {"test", "ind_test"} and not ALLOW_TEST_VALIDATION:
        raise ValueError(
            f"DO_VALIDATION=True with VAL_MODE='{VAL_MODE}'. "
            f"This may select checkpoints on a test split. "
            f"If this is intentional, set ALLOW_TEST_VALIDATION=True."
        )

    if not PSEUDO_POS_ONLY:
        warnings.warn(
            "PSEUDO_POS_ONLY=False. This treats unannotated pseudo-label terms "
            "as negatives. For GO annotation this assumption is often unsafe."
        )


def build_args() -> argparse.Namespace:
    """
    Build the argparse.Namespace expected by experiments.exp_train.train_one_task().
    """
    cfg = _load_pickle_config(Path(MODEL_CONFIG))

    num_classes = _infer_value(NUM_CLASSES, cfg, "num_classes", required=True)
    top_k = _infer_value(TOP_K, cfg, "top_k", required=True)
    max_len = _infer_value(MAX_LEN, cfg, "max_len", required=True)

    # msa_max_size is allowed to be None.
    msa_max_size = _infer_value(MSA_MAX_SIZE, cfg, "msa_max_size", required=False)

    args = argparse.Namespace(
        # Required
        model_config=str(Path(MODEL_CONFIG)),
        init_ckpt=str(Path(INIT_CKPT)),
        file_address=str(Path(FILE_ADDRESS)),
        working_address=str(Path(WORKING_ADDRESS)),
        task=TASK,
        num_classes=int(num_classes),
        output_dir=str(Path(OUTPUT_DIR)),

        # Dataset/model compatibility
        top_k=int(top_k),
        max_len=int(max_len),
        msa_max_size=msa_max_size,
        permute_dims=list(PERMUTE_DIMS),
        torch_compile=bool(TORCH_COMPILE),

        # GPU handling
        gpu_ids=GPU_IDS,

        # Training
        epochs=int(EPOCHS),
        batch_size=int(BATCH_SIZE),
        pseudo_batch_size=int(PSEUDO_BATCH_SIZE),
        eval_batch_size=int(EVAL_BATCH_SIZE),
        dataloader_num_workers=int(DATALOADER_NUM_WORKERS),
        pin_memory=bool(PIN_MEMORY),
        drop_last=bool(DROP_LAST),

        lr=float(LR),
        min_lr=float(MIN_LR),
        weight_decay=float(WEIGHT_DECAY),
        grad_clip=float(GRAD_CLIP),

        device=str(DEVICE),
        seed=int(SEED),
        no_amp=bool(NO_AMP),

        # Projection head
        proj_in_dim=int(PROJ_IN_DIM),
        proj_hidden_dim=int(PROJ_HIDDEN_DIM),
        proj_dim=int(PROJ_DIM),
        proj_dropout=float(PROJ_DROPOUT),

        # Semi-supervised loss weights
        lambda_u=float(LAMBDA_U),
        lambda_c=float(LAMBDA_C),
        lambda_h=float(LAMBDA_H),

        # Contrastive loss
        contrast_tau=float(CONTRAST_TAU),
        contrast_min_r=float(CONTRAST_MIN_R),
        w_true_pseudo=float(W_TRUE_PSEUDO),
        w_pseudo_pseudo=float(W_PSEUDO_PSEUDO),

        # Pseudo-label treatment
        pseudo_pos_only=bool(PSEUDO_POS_ONLY),

        # IC / GO hierarchy
        ic_path=_path_or_none(IC_PATH),
        ic_alpha=float(IC_ALPHA),
        ic_min_count=int(IC_MIN_COUNT),
        go_edges_path=_path_or_none(GO_EDGES_PATH),

        # MSA weak augmentation
        weak_row_drop_p=float(WEAK_ROW_DROP_P),
        weak_col_mask_p=float(WEAK_COL_MASK_P),
        weak_block_mask_p=float(WEAK_BLOCK_MASK_P),
        weak_block_mask_min=float(WEAK_BLOCK_MASK_MIN),
        weak_block_mask_max=float(WEAK_BLOCK_MASK_MAX),
        weak_n_blocks=int(WEAK_N_BLOCKS),
        weak_shuffle_rows=bool(WEAK_SHUFFLE_ROWS),
        weak_min_keep_rows=int(WEAK_MIN_KEEP_ROWS),
        weak_noise_std=float(WEAK_NOISE_STD),

        # MSA strong augmentation
        strong_row_drop_p=float(STRONG_ROW_DROP_P),
        strong_col_mask_p=float(STRONG_COL_MASK_P),
        strong_block_mask_p=float(STRONG_BLOCK_MASK_P),
        strong_block_mask_min=float(STRONG_BLOCK_MASK_MIN),
        strong_block_mask_max=float(STRONG_BLOCK_MASK_MAX),
        strong_n_blocks=int(STRONG_N_BLOCKS),
        strong_shuffle_rows=bool(STRONG_SHUFFLE_ROWS),
        strong_min_keep_rows=int(STRONG_MIN_KEEP_ROWS),
        strong_noise_std=float(STRONG_NOISE_STD),

        # Validation / logging / saving
        no_validation=not bool(DO_VALIDATION),
        val_mode=str(VAL_MODE),
        log_interval=int(LOG_INTERVAL),
        save_interval=int(SAVE_INTERVAL),
        need_proteins=bool(NEED_PROTEINS),
    )

    return args


def print_run_summary(args: argparse.Namespace):
    print("=" * 80)
    print("[Run] Semi-supervised MSA-GO expansion training")
    print("-" * 80)
    print(f"ROOT                 = {ROOT}")
    print(f"EXP_TRAIN            = {EXP_TRAIN}")
    print(f"TASK                 = {args.task}")
    print(f"MODEL_CONFIG         = {args.model_config}")
    print(f"INIT_CKPT            = {args.init_ckpt}")
    print(f"FILE_ADDRESS         = {args.file_address}")
    print(f"WORKING_ADDRESS          = {args.working_address}")
    print(f"OUTPUT_DIR           = {args.output_dir}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"DEVICE               = {args.device}")
    print(f"GPU_IDS              = {args.gpu_ids}")
    print(f"NUM_CLASSES          = {args.num_classes}")
    print(f"TOP_K                = {args.top_k}")
    print(f"MAX_LEN              = {args.max_len}")
    print(f"MSA_MAX_SIZE          = {args.msa_max_size}")
    print(f"PROJ_IN_DIM          = {args.proj_in_dim}")
    print(f"EPOCHS               = {args.epochs}")
    print(f"BATCH_SIZE           = {args.batch_size}")
    print(f"PSEUDO_BATCH_SIZE    = {args.pseudo_batch_size}")
    print(f"LR                   = {args.lr}")
    print(f"LAMBDA_U             = {args.lambda_u}")
    print(f"LAMBDA_C             = {args.lambda_c}")
    print(f"LAMBDA_H             = {args.lambda_h}")
    print(f"PSEUDO_POS_ONLY      = {args.pseudo_pos_only}")
    print(f"DO_VALIDATION         = {not args.no_validation}")
    print(f"VAL_MODE             = {args.val_mode}")
    print("=" * 80)


def prepare_imports():
    """
    Add project paths before importing experiments.exp_train.
    """
    root_str = str(ROOT)
    msa_str = str(MSA_ROOT)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if msa_str not in sys.path:
        sys.path.insert(0, msa_str)


def main():
    _validate_paths()
    _validate_output_dir()
    _validate_semantics()

    args = build_args()

    if PRINT_CONFIG:
        print_run_summary(args)

    if SAVE_RUNNER_CONFIG:
        _save_json(vars(args), Path(args.output_dir) / "runner_config.json")

    if DRY_RUN:
        print("[Dry run] Configuration built successfully. Exit without training.")
        return

    prepare_imports()

    # Import after CUDA_VISIBLE_DEVICES and sys.path are configured.
    from experiments.exp_train import train_one_task, normalize_task

    args.task = normalize_task(args.task)

    train_one_task(args)


if __name__ == "__main__":
    main()