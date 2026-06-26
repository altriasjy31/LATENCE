#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_weak_exp_train_detr.py

Runner for experiments/weak_exp_train_detr.py.

The parent process launches torchrun when USE_DDP=True.
This runner is configured for the first full V3 DETR run:
    - expert external probability as exp_train pseudo-prob supervision;
    - trainable ontology query embedding;
    - lightweight learnable selector;
    - residual DETR-style top-k refinement with delta_max=0.5.
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
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parent
MSA_ROOT = ROOT / "msa_models"
WEAK_EXP_TRAIN = ROOT / "experiments" / "weak_exp_train_detr.py"

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

# -----------------------------------------------------------------------------
# Pseudo probability source
# -----------------------------------------------------------------------------
# V3 default: use ESM2-3B expert probabilities for exp_train as the pseudo
# probability supervision signal.  The ind_test expert arrays must not be used
# for training.  Override with:
#   PSEUDO_PROB_PATH=/abs/path/to/{task}_exp_train_predictions.float16.npy TASK=bp python run_weak_exp_train_detr.py

PSEUDO_PROB_SOURCE = os.environ.get("PSEUDO_PROB_SOURCE", "expert")  # "expert" or "teacher" or "precomputed_mix"

TEACHER_PROB_OUTPUT_DIR = ROOT / "data" / "outputs" / "msa_teacher_25_10_22_probs"
TEACHER_PSEUDO_PROB_PATH = TEACHER_PROB_OUTPUT_DIR / f"{TASK}_exp_train_probs.float16.npy"

EXPERT_PROB_OUTPUT_DIR = Path(
    os.environ.get(
        "EXPERT_PROB_OUTPUT_DIR",
        str(ROOT / "data" / "external_probs" / "esm2_3b_exp_train"),
    )
)
EXPERT_PROB_CANDIDATES = [
    EXPERT_PROB_OUTPUT_DIR / f"{TASK}_exp_train_predictions.prop.float16.npy",
    EXPERT_PROB_OUTPUT_DIR / f"{TASK}_exp_train_predictions.float16.npy",
    EXPERT_PROB_OUTPUT_DIR / f"{TASK}_exp_train_probs.float16.npy",
    EXPERT_PROB_OUTPUT_DIR / f"{TASK}_predictions.float16.npy",
]

MIXED_PROB_OUTPUT_DIR = ROOT / "data" / "external_probs" / "esm2_3b_teacher_mix_exp_train"
MIXED_PSEUDO_PROB_PATH = MIXED_PROB_OUTPUT_DIR / f"{TASK}_mix_predictions.float16.npy"
TEACHER_EXPERT_MIX_ALPHA = float(os.environ.get("TEACHER_EXPERT_MIX_ALPHA", "1.0"))


def _first_existing_or_first(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for p in paths:
        if p.is_file():
            return p
    return paths[0]


if "PSEUDO_PROB_PATH" in os.environ:
    PSEUDO_PROB_PATH = Path(os.environ["PSEUDO_PROB_PATH"])
elif PSEUDO_PROB_SOURCE == "expert":
    PSEUDO_PROB_PATH = _first_existing_or_first(EXPERT_PROB_CANDIDATES)
elif PSEUDO_PROB_SOURCE == "teacher":
    PSEUDO_PROB_PATH = TEACHER_PSEUDO_PROB_PATH
elif PSEUDO_PROB_SOURCE == "precomputed_mix":
    PSEUDO_PROB_PATH = MIXED_PSEUDO_PROB_PATH
else:
    raise ValueError(f"Unknown PSEUDO_PROB_SOURCE: {PSEUDO_PROB_SOURCE}")

USE_PSEUDO_PROB = True

RUN_TAG = os.environ.get("RUN_TAG", f"{TASK}_weak_detr_v3_{PSEUDO_PROB_SOURCE}_prob")
OUTPUT_DIR = ROOT / "outputs" / "weak_exp_train_detr" / RUN_TAG
FAIL_IF_OUTPUT_EXISTS = False

NUM_CLASSES = None
TOP_K = 64
MAX_LEN = 2048
MSA_MAX_SIZE = None
PERMUTE_DIMS = (0, 3, 2, 1)
TORCH_COMPILE = False

# First full V3 run.  BP intentionally follows the stronger MF pseudo setting.
TASK_CONFIGS = {
    "bp": {
        "batch_size": 8,
        "pseudo_batch_size": 56,
        "lambda_true": 0.65,
        "lambda_pseudo": 1.0,
        "lr": 2.5e-4,
        "query_topk": 768,
        "selector_topm": 6144,
        "delta_max": 1.0,
        "max_steps": None,
    },
    "mf": {
        "batch_size": 8,
        "pseudo_batch_size": 56,
        "lambda_true": 0.75,
        "lambda_pseudo": 0.8,
        "lr": 2e-4,
        "query_topk": 512,
        "selector_topm": 2048,
        "delta_max": 1.0,
        "max_steps": None,
    },
    "cc": {
        "batch_size": 8,
        "pseudo_batch_size": 56,
        "lambda_true": 0.65,
        "lambda_pseudo": 1.0,
        "lr": 1.5e-4,
        "query_topk": 512,
        "selector_topm": 2048,
        "delta_max": 1.0,
        "max_steps": None,
    },
}

_cfg = TASK_CONFIGS[TASK]

BATCH_SIZE = _cfg["batch_size"]
PSEUDO_BATCH_SIZE = _cfg["pseudo_batch_size"]
LAMBDA_TRUE = _cfg["lambda_true"]
LAMBDA_PSEUDO = _cfg["lambda_pseudo"]
LR = _cfg["lr"]
QUERY_DECODER_TOPK = _cfg["query_topk"]
QUERY_DECODER_DELTA_MAX = _cfg["delta_max"]
SELECTOR_PREFILTER_TOPM = _cfg["selector_topm"]
MAX_STEPS_PER_EPOCH = _cfg["max_steps"]

SEED = 3407
EPOCHS = 400
DATALOADER_NUM_WORKERS = 8
PIN_MEMORY = True
DROP_LAST = False

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

OPTIM = "adamw"
OPTIM_EPS = 1e-6
LR_POLICY = "cycle"
LR_PCT_START = 0.05 # bp using 0.05
LR_CYCLE_THREE_PHASE = False
LR_DIV_FACTOR = 25.0
LR_FINAL_DIV_FACTOR = 120.0
MIN_LR = 1e-6
WEIGHT_DECAY = 0.009
NO_MODEL_WEIGHT_DECAY = True
ACCUM_STEPS = 1
GRAD_CLIP = 1.0
NO_AMP = False

FREEZE_BN = False
FREEZE_BN_AFFINE = False

LAMBDA_H = 0.0005

TRUE_ASL_GAMMA_NEG = 4.0
TRUE_ASL_GAMMA_POS = 0.0
TRUE_ASL_CLIP = 0.05
TRUE_ASL_REDUCTION = "batch_mean"

PSEUDO_LOSS_TYPE = "asl"
PSEUDO_PROB_TARGET = "soft_pos"
PSEUDO_PROB_CONF_POWER = 0.5
PSEUDO_PROB_MIN_CONF = 0.0
PSEUDO_NEGATIVE_POLICY = "none"
PSEUDO_PROB_NEG_MAX = 0.01
PSEUDO_NEG_TOPK = 0
PSEUDO_MIN_IC = 0.0
PSEUDO_ASL_GAMMA_NEG = 4.0
PSEUDO_ASL_GAMMA_POS = 0.0
PSEUDO_ASL_CLIP = 0.05
PSEUDO_ASL_REDUCTION = "batch_mean"

# -----------------------------------------------------------------------------
# DETR-style ontology query decoder V3
# -----------------------------------------------------------------------------
USE_QUERY_DECODER = True
QUERY_DECODER_TOPK_SOURCE = "blend"       # alpha=1.0 => expert/prob top-k for pseudo rows, base top-k for true rows
SELECTOR_STATIC_SOURCE = "blend"
EXTERNAL_PROB_BLEND_ALPHA = 1.0
QUERY_DECODER_MODE = "residual"
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

USE_TRAINABLE_QUERY_EMBEDDING = True
QUERY_EMBED_INIT = "classifier_plus_residual"
QUERY_EMBED_LR = 1e-4
QUERY_EMBED_WEIGHT_DECAY = 1e-4
QUERY_EMBED_RESIDUAL_SCALE = 0.1
USE_QUERY_SCORE_FEATURES = True
QUERY_SCORE_EMBED_SCALE = 0.1
QUERY_SCORE_DETACH = True

QUERY_DECODER_LOGIT_BASE_MODE = os.environ.get(
    "QUERY_DECODER_LOGIT_BASE_MODE",
    "mix_expert_base_anchor",
)

EXPERT_BASE_MIX_ALPHA = float(os.environ.get("EXPERT_BASE_MIX_ALPHA", "0.8"))

ANCHOR_DELTA_GATE_INIT = float(os.environ.get("ANCHOR_DELTA_GATE_INIT", "0.3"))

LAMBDA_ANCHOR_KD = float(os.environ.get("LAMBDA_ANCHOR_KD", "0.3"))
ANCHOR_KD_TOPM = int(os.environ.get("ANCHOR_KD_TOPM", "512"))
ANCHOR_KD_CONF_POWER = float(os.environ.get("ANCHOR_KD_CONF_POWER", "0.5"))
ANCHOR_KD_NEG_WEIGHT = float(os.environ.get("ANCHOR_KD_NEG_WEIGHT", "0.25"))

# Split pseudo supervision: expert pseudo should train both backbone/base and query/refined.
# LAMBDA_PSEUDO_BASE = float(os.environ.get("LAMBDA_PSEUDO_BASE", str(LAMBDA_PSEUDO)))
# LAMBDA_PSEUDO_QUERY = float(os.environ.get("LAMBDA_PSEUDO_QUERY", str(LAMBDA_PSEUDO)))
LAMBDA_PSEUDO_BASE = LAMBDA_PSEUDO
LAMBDA_PSEUDO_QUERY = 0.5 * LAMBDA_PSEUDO

# Expert KD for backbone/base logits.
LAMBDA_BASE_EXPERT_KD = float(os.environ.get("LAMBDA_BASE_EXPERT_KD", "0.2"))
BASE_KD_TOPM = int(os.environ.get("BASE_KD_TOPM", "2048")) # mf 512
BASE_KD_CONF_POWER = float(os.environ.get("BASE_KD_CONF_POWER", "0.5"))
BASE_KD_NEG_WEIGHT = float(os.environ.get("BASE_KD_NEG_WEIGHT", "0.25"))

# Expert KD for query/refined logits.
LAMBDA_QUERY_EXPERT_KD = float(os.environ.get("LAMBDA_QUERY_EXPERT_KD", "0.25"))
QUERY_KD_TOPM = int(os.environ.get("QUERY_KD_TOPM", "512"))
QUERY_KD_CONF_POWER = float(os.environ.get("QUERY_KD_CONF_POWER", "0.5"))
QUERY_KD_NEG_WEIGHT = float(os.environ.get("QUERY_KD_NEG_WEIGHT", "0.25"))

USE_LEARNABLE_SELECTOR = True
SELECTOR_HIDDEN_DIM = 64
SELECTOR_LR = 1e-4
SELECTOR_WEIGHT_DECAY = 1e-4
SELECTOR_POS_WEIGHT = 10.0
SELECTOR_LOGIT_RESIDUAL_SCALE = 1.0
SELECTOR_USE_TERM_BIAS = True
SELECTOR_USE_PROTEIN_TERM_AFFINITY = True
SELECTOR_DETACH_BASE_LOGITS = True
SELECTOR_DETACH_TERM_FEATURES = True

QUERY_DECODER_LR = None
QUERY_DECODER_WEIGHT_DECAY = None

LAMBDA_SELECTOR = 0.05
LAMBDA_DELTA_L2 = 1e-4
LAMBDA_EXTERNAL_KD = 0.0
EXTERNAL_KD_TOPK = 0

# =============================================================================
# Multi-node DDP
# =============================================================================

LAUNCH_MODE = "single_node"  # "single_node" or "multi_node"

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "1,2")
DEVICE = "auto"
GPU_IDS = "auto"
USE_DDP = True
DDP_STANDALONE = True

NPROC_PER_NODE = int(os.environ.get("NPROC_PER_NODE", "2"))
NNODES = int(os.environ.get("NNODES", "1"))
NODE_RANK = int(os.environ.get("NODE_RANK", "0"))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "10.233.128.18")
MASTER_PORT = int(os.environ.get("MASTER_PORT", "46123"))
RDZV_BACKEND = "c10d"
RDZV_ID = os.environ.get("RDZV_ID", f"latence_weak_detr_{TASK}_v3")
RDZV_ENDPOINT = f"{MASTER_ADDR}:{MASTER_PORT}"
DDP_MAX_RESTARTS = 0
DDP_TIMEOUT_MINUTES = 180

NCCL_SOCKET_IFNAME = os.environ.get("NCCL_SOCKET_IFNAME", "eth0")
GLOO_SOCKET_IFNAME = None

IC_ALPHA = 1.0
IC_MIN_COUNT = 2

LOG_INTERVAL = 20
SAVE_INTERVAL = 10
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
        raise FileNotFoundError(f"Cannot find weak_exp_train_detr.py: {WEAK_EXP_TRAIN}")
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
        raise FileNotFoundError(
            f"PSEUDO_PROB_PATH not found: {PSEUDO_PROB_PATH}\n"
            "For V3 expert pseudo supervision, generate ESM2-3B probabilities for exp_train first, "
            "or set PSEUDO_PROB_PATH explicitly. Do not use ind_test external probability arrays for training."
        )
    if USE_PSEUDO_PROB and PSEUDO_PROB_SOURCE == "expert" and "exp_train" not in str(PSEUDO_PROB_PATH):
        warnings.warn(
            f"PSEUDO_PROB_PATH does not contain 'exp_train': {PSEUDO_PROB_PATH}. "
            "Verify this file is aligned to dataset['exp_train'][task]['proteins'], not ind_test."
        )
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

    if NCCL_SOCKET_IFNAME is not None:
        env["NCCL_SOCKET_IFNAME"] = str(NCCL_SOCKET_IFNAME)
    if GLOO_SOCKET_IFNAME is not None:
        env["GLOO_SOCKET_IFNAME"] = str(GLOO_SOCKET_IFNAME)

    cmd = [sys.executable, "-m", "torch.distributed.run"]

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
    print("[DDP launcher: weak_exp_train_detr]")
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
        pseudo_prob_source=str(PSEUDO_PROB_SOURCE),
        teacher_expert_mix_alpha=float(TEACHER_EXPERT_MIX_ALPHA),
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
        external_kd_topk=int(EXTERNAL_KD_TOPK),
        use_query_decoder=bool(USE_QUERY_DECODER),
        query_decoder_topk=int(QUERY_DECODER_TOPK),
        query_decoder_topk_source=str(QUERY_DECODER_TOPK_SOURCE),
        selector_static_source=str(SELECTOR_STATIC_SOURCE),
        external_prob_blend_alpha=float(EXTERNAL_PROB_BLEND_ALPHA),
        query_decoder_delta_max=float(QUERY_DECODER_DELTA_MAX),
        query_decoder_mode=str(QUERY_DECODER_MODE),
        query_decoder_dim=int(QUERY_DECODER_DIM),
        query_decoder_heads=int(QUERY_DECODER_HEADS),
        query_decoder_layers=int(QUERY_DECODER_LAYERS),
        query_decoder_ffn_dim=int(QUERY_DECODER_FFN_DIM),
        query_decoder_dropout=float(QUERY_DECODER_DROPOUT),
        query_decoder_lr=QUERY_DECODER_LR,
        query_decoder_weight_decay=QUERY_DECODER_WEIGHT_DECAY,
        query_decoder_detach_query_weight=bool(QUERY_DECODER_DETACH_QUERY_WEIGHT),
        query_decoder_include_label_boost=bool(QUERY_DECODER_INCLUDE_LABEL_BOOST),
        query_decoder_label_boost=float(QUERY_DECODER_LABEL_BOOST),
        query_decoder_memory_mode=str(QUERY_DECODER_MEMORY_MODE),
        query_decoder_memory_grid_h=int(QUERY_DECODER_MEMORY_GRID_H),
        query_decoder_memory_grid_w=int(QUERY_DECODER_MEMORY_GRID_W),
        use_trainable_query_embedding=bool(USE_TRAINABLE_QUERY_EMBEDDING),
        query_embed_init=str(QUERY_EMBED_INIT),
        query_embed_lr=float(QUERY_EMBED_LR),
        query_embed_weight_decay=float(QUERY_EMBED_WEIGHT_DECAY),
        query_embed_residual_scale=float(QUERY_EMBED_RESIDUAL_SCALE),
        use_query_score_features=bool(USE_QUERY_SCORE_FEATURES),
        query_score_embed_scale=float(QUERY_SCORE_EMBED_SCALE),
        query_score_detach=bool(QUERY_SCORE_DETACH),
        query_decoder_logit_base_mode=str(QUERY_DECODER_LOGIT_BASE_MODE),
        expert_base_mix_alpha=float(EXPERT_BASE_MIX_ALPHA),
        anchor_delta_gate_init=float(ANCHOR_DELTA_GATE_INIT),
        lambda_anchor_kd=float(LAMBDA_ANCHOR_KD),
        anchor_kd_topm=int(ANCHOR_KD_TOPM),
        anchor_kd_conf_power=float(ANCHOR_KD_CONF_POWER),
        anchor_kd_neg_weight=float(ANCHOR_KD_NEG_WEIGHT),
        use_learnable_selector=bool(USE_LEARNABLE_SELECTOR),
        selector_prefilter_topm=int(SELECTOR_PREFILTER_TOPM),
        selector_hidden_dim=int(SELECTOR_HIDDEN_DIM),
        selector_lr=float(SELECTOR_LR),
        selector_weight_decay=float(SELECTOR_WEIGHT_DECAY),
        selector_pos_weight=float(SELECTOR_POS_WEIGHT),
        selector_logit_residual_scale=float(SELECTOR_LOGIT_RESIDUAL_SCALE),
        selector_use_term_bias=bool(SELECTOR_USE_TERM_BIAS),
        selector_use_protein_term_affinity=bool(SELECTOR_USE_PROTEIN_TERM_AFFINITY),
        selector_detach_base_logits=bool(SELECTOR_DETACH_BASE_LOGITS),
        selector_detach_term_features=bool(SELECTOR_DETACH_TERM_FEATURES),
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
        lambda_pseudo_base=float(LAMBDA_PSEUDO_BASE),
        lambda_pseudo_query=float(LAMBDA_PSEUDO_QUERY),

        lambda_base_expert_kd=float(LAMBDA_BASE_EXPERT_KD),
        base_kd_topm=int(BASE_KD_TOPM),
        base_kd_conf_power=float(BASE_KD_CONF_POWER),
        base_kd_neg_weight=float(BASE_KD_NEG_WEIGHT),

        lambda_query_expert_kd=float(LAMBDA_QUERY_EXPERT_KD),
        query_kd_topm=int(QUERY_KD_TOPM),
        query_kd_conf_power=float(QUERY_KD_CONF_POWER),
        query_kd_neg_weight=float(QUERY_KD_NEG_WEIGHT),
        lambda_h=float(LAMBDA_H),
        lambda_selector=float(LAMBDA_SELECTOR),
        lambda_delta_l2=float(LAMBDA_DELTA_L2),
        lambda_external_kd=float(LAMBDA_EXTERNAL_KD),
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
    print("[Run] Weak MSA-GO expansion training with DETR V3")
    print(f"TASK                    = {args.task}")
    print(f"MODEL_CONFIG            = {args.model_config}")
    print(f"INIT_CKPT               = {args.init_ckpt}")
    print(f"FILE_ADDRESS            = {args.file_address}")
    print(f"WORKING_ADDRESS         = {args.working_address}")
    print(f"OUTPUT_DIR              = {args.output_dir}")
    print(f"CUDA_VISIBLE_DEVICES    = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"USE_DDP                 = {USE_DDP}")
    print(f"NPROC_PER_NODE          = {NPROC_PER_NODE}")
    print(f"NUM_CLASSES             = {args.num_classes}")
    print(f"TOP_K                   = {args.top_k}")
    print(f"MAX_LEN                 = {args.max_len}")
    print(f"EPOCHS                  = {args.epochs}")
    print(f"BATCH_SIZE              = {args.batch_size}")
    print(f"PSEUDO_BATCH_SIZE       = {args.pseudo_batch_size}")
    print(f"OPTIM                   = {args.optim}")
    print(f"LR                      = {args.lr}")
    print(f"QUERY_EMBED_LR          = {args.query_embed_lr}")
    print(f"SELECTOR_LR             = {args.selector_lr}")
    print(f"LAMBDA_TRUE             = {args.lambda_true}")
    print(f"LAMBDA_PSEUDO           = {args.lambda_pseudo}")
    print(f"LAMBDA_PSEUDO_BASE       = {args.lambda_pseudo_base}")
    print(f"LAMBDA_PSEUDO_QUERY      = {args.lambda_pseudo_query}")
    print(f"LAMBDA_BASE_EXPERT_KD    = {args.lambda_base_expert_kd}")
    print(f"BASE_KD_TOPM             = {args.base_kd_topm}")
    print(f"LAMBDA_QUERY_EXPERT_KD   = {args.lambda_query_expert_kd}")
    print(f"QUERY_KD_TOPM            = {args.query_kd_topm}")
    print(f"LAMBDA_SELECTOR         = {args.lambda_selector}")
    print(f"LAMBDA_DELTA_L2         = {args.lambda_delta_l2}")
    print(f"LAMBDA_EXTERNAL_KD      = {args.lambda_external_kd}")
    print(f"PSEUDO_PROB_SOURCE      = {args.pseudo_prob_source}")
    print(f"PSEUDO_PROB_PATH        = {args.pseudo_prob_path}")
    print(f"PSEUDO_TARGET           = {args.pseudo_prob_target}")
    print(f"PSEUDO_NEG_POLICY       = {args.pseudo_negative_policy}")
    print(f"USE_QUERY_DECODER       = {args.use_query_decoder}")
    print(f"QUERY_DECODER_TOPK      = {args.query_decoder_topk}")
    print(f"QUERY_DECODER_SOURCE    = {args.query_decoder_topk_source}")
    print(f"EXTERNAL_BLEND_ALPHA    = {args.external_prob_blend_alpha}")
    print(f"QUERY_DECODER_DELTA_MAX = {args.query_decoder_delta_max}")
    print(f"TRAINABLE_QUERY_EMB     = {args.use_trainable_query_embedding}")
    print(f"QUERY_EMBED_INIT        = {args.query_embed_init}")
    print(f"QUERY_DECODER_LOGIT_BASE_MODE = {args.query_decoder_logit_base_mode}")
    print(f"EXPERT_BASE_MIX_ALPHA         = {args.expert_base_mix_alpha}")
    print(f"ANCHOR_DELTA_GATE_INIT        = {args.anchor_delta_gate_init}")
    print(f"LAMBDA_ANCHOR_KD              = {args.lambda_anchor_kd}")
    print(f"ANCHOR_KD_TOPM                = {args.anchor_kd_topm}")
    print(f"LEARNABLE_SELECTOR      = {args.use_learnable_selector}")
    print(f"SELECTOR_PREFILTER_TOPM = {args.selector_prefilter_topm}")
    print(f"SELECTOR_AFFINITY       = {args.selector_use_protein_term_affinity}")
    print(f"FREEZE_BN               = {args.freeze_bn}")
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
    from experiments.weak_exp_train_detr import train_one_task, normalize_task

    args.task = normalize_task(args.task)
    train_one_task(args)


if __name__ == "__main__":
    main()