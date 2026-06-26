#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_eval_weak_ind_test_detr_diagnostics.py

Runner for experiments/eval_weak_ind_test_detr_diagnostics.py.

Examples
--------
TASK=bp RUN_TAG=bp_weak_detr_v3_expert_prob EPOCH=6 python run_eval_weak_ind_test_detr_diagnostics.py

or explicitly:

TASK=bp \
CHECKPOINT=/.../outputs/weak_exp_train_detr/bp_weak_detr_v3_expert_prob/weak_detr_decoder_epoch6.pt \
EXTERNAL_PROB_PATH=/.../data/external_probs/esm2_3b/bp_predictions.float16.npy \
python run_eval_weak_ind_test_detr_diagnostics.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL_SCRIPT = ROOT / "experiments" / "eval_weak_ind_test_detr_diagnostics.py"

TASK = os.environ.get("TASK", "bp")  # cc, mf, bp
MODE = os.environ.get("MODE", "ind_test")
EPOCH = os.environ.get("EPOCH", "6")
RUN_TAG = os.environ.get("RUN_TAG", f"{TASK}_weak_detr_v3_expert_prob")

TASK_NUM_CLASSES = {
    "cc": 2903,
    "mf": 7038,
    "bp": 21312,
}

MODEL_CONFIG = ROOT / "data" / "msa_models" / "configs" / "model_opts" / f"{TASK}_msa_model_config.pkl"
INIT_CKPT = ROOT / "data" / "msa_models" / "checkpoints" / f"{TASK}_msa_model_rank1.pt"
FILE_ADDRESS = ROOT / "data" / "unidata_with_exp_train_pseudo.pkl"
WORKING_ADDRESS = ROOT / "data" / "ind_MSA_bin" / "index.pkl"

DEFAULT_CKPT = ROOT / "outputs" / "weak_exp_train_detr" / RUN_TAG / f"weak_detr_decoder_epoch{EPOCH}.pt"
CHECKPOINT = Path(os.environ.get("CHECKPOINT", str(DEFAULT_CKPT)))

EXTERNAL_PROB_PATH = Path(
    os.environ.get(
        "EXTERNAL_PROB_PATH",
        str(ROOT / "data" / "external_probs" / "esm2_3b" / f"{TASK}_predictions.prop.float16.npy"),
    )
)

ORIGINAL_PROB_PATH = os.environ.get("ORIGINAL_PROB_PATH", "")

DECODER_PROB_SOURCES = os.environ.get(
    "DECODER_PROB_SOURCES",
    "expert,mix_expert_base,mix_expert_original",
)

DECODER_PROB_MIX_ALPHAS = os.environ.get(
    "DECODER_PROB_MIX_ALPHAS",
    "0.5,0.7,0.8,0.9",
)

SKIP_LEGACY_QUERY_SOURCES = os.environ.get("SKIP_LEGACY_QUERY_SOURCES", "0") == "1"

# Important: include decoderprob, otherwise mixed decoder-prob results will not be ensembled.
ENSEMBLE_QUERY_MODES = os.environ.get(
    "ENSEMBLE_QUERY_MODES",
    "external_topk,blend_topk,decoderprob,anchorlogit",
)

OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        str(ROOT / "outputs" / "weak_exp_train_detr_diagnostics" / RUN_TAG / f"epoch{EPOCH}_{MODE}"),
    )
)

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "8"))
NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", "4"))
MAX_BATCHES = os.environ.get("MAX_BATCHES", "")
DEVICE = os.environ.get("DEVICE", "auto")

QUERY_MODES = os.environ.get("QUERY_MODES", "base_topk,external_topk,blend_topk")
DELTA_SCALES = os.environ.get("DELTA_SCALES", "0,0.25,0.5,0.75,1.0")
ENSEMBLE_ALPHAS = os.environ.get("ENSEMBLE_ALPHAS", "0.1,0.3,0.5,0.7,0.9")
ENSEMBLE_DELTA_SCALES = os.environ.get("ENSEMBLE_DELTA_SCALES", "0.5,1.0")

QUERY_DECODER_LOGIT_BASE_MODE = os.environ.get("QUERY_DECODER_LOGIT_BASE_MODE", "")
EXPERT_BASE_MIX_ALPHA = os.environ.get("EXPERT_BASE_MIX_ALPHA", "")
ANCHOR_DELTA_GATE_INIT = os.environ.get("ANCHOR_DELTA_GATE_INIT", "")

# IC Fusion
ENABLE_IC_FUSION = os.environ.get("ENABLE_IC_FUSION", "0") == "1"
IC_FUSION_ALPHA_RARE = float(os.environ.get("IC_FUSION_ALPHA_RARE", "0.5"))
IC_FUSION_ALPHA_MEDIUM = float(os.environ.get("IC_FUSION_ALPHA_MEDIUM", "0.8"))
IC_FUSION_ALPHA_COMMON = float(os.environ.get("IC_FUSION_ALPHA_COMMON", "0.8"))
IC_FUSION_SOURCE = os.environ.get("IC_FUSION_SOURCE", "base")

ENABLE_SIMULATED_IC = os.environ.get("ENABLE_SIMULATED_IC", "0") == "1"

SIMULATED_IC_SOURCE_KEY = os.environ.get(
    "SIMULATED_IC_SOURCE_KEY",
    "backbone_base",
)

SIMULATED_IC_THRESHOLD_MIN = float(os.environ.get(
    "SIMULATED_IC_THRESHOLD_MIN",
    "0.01",
))

SIMULATED_IC_THRESHOLD_MAX = float(os.environ.get(
    "SIMULATED_IC_THRESHOLD_MAX",
    "1.0",
))

SIMULATED_IC_THRESHOLD_STEP = float(os.environ.get(
    "SIMULATED_IC_THRESHOLD_STEP",
    "0.01",
))

SIMULATED_IC_INCLUDE_ZERO_THRESHOLD = os.environ.get(
    "SIMULATED_IC_INCLUDE_ZERO_THRESHOLD",
    "0",
) == "1"

SIMULATED_IC_SAVE_COUNTS = os.environ.get(
    "SIMULATED_IC_SAVE_COUNTS",
    "0",
) == "1"

# hist is fast and enough for diagnosis. Use AUROC_MODE=exact if you need a
# closer micro average-precision value, but it can be slower/more memory heavy.
AUPRC_MODE = os.environ.get("AUPRC_MODE", "hist")  # hist, exact, none
THRESHOLD_STEP = os.environ.get("THRESHOLD_STEP", "0.01")
COMPUTE_SAMPLE_FMAX = os.environ.get("COMPUTE_SAMPLE_FMAX", "0") == "1"
DO_RARE_ANALYSIS = os.environ.get("DO_RARE_ANALYSIS", "1")

# For eval, do not pass y_hint to the decoder unless deliberately doing an
# oracle/leakage ablation.
ALLOW_EVAL_LABEL_BOOST = os.environ.get("ALLOW_EVAL_LABEL_BOOST", "0") == "1"

# Decoder defaults. The eval script will auto-load checkpoint_dir/args.json and
# use the training-time decoder config unless overridden by CLI/env here.
cmd = [
    sys.executable,
    str(EVAL_SCRIPT),
    "--checkpoint", str(CHECKPOINT),
    "--task", TASK,
    "--mode", MODE,
    "--num_classes", str(TASK_NUM_CLASSES[TASK]),
    "--model_config", str(MODEL_CONFIG),
    "--init_ckpt", str(INIT_CKPT),
    "--file_address", str(FILE_ADDRESS),
    "--working_address", str(WORKING_ADDRESS),
    "--external_prob_path", str(EXTERNAL_PROB_PATH),
    "--decoder_prob_sources", str(DECODER_PROB_SOURCES),
    "--decoder_prob_mix_alphas", str(DECODER_PROB_MIX_ALPHAS),
    "--ensemble_query_modes", str(ENSEMBLE_QUERY_MODES),
    "--output_dir", str(OUTPUT_DIR),
    "--eval_batch_size", str(EVAL_BATCH_SIZE),
    "--dataloader_num_workers", str(NUM_WORKERS),
    "--device", DEVICE,
    "--query_modes", QUERY_MODES,
    "--delta_scales", DELTA_SCALES,
    "--ensemble_alphas", ENSEMBLE_ALPHAS,
    "--ensemble_delta_scales", ENSEMBLE_DELTA_SCALES,
    "--auprc_mode", AUPRC_MODE,
    "--threshold_step", str(THRESHOLD_STEP),
    "--do_rare_analysis", DO_RARE_ANALYSIS,
]

cmd.append("--enable_ic_fusion")

if ORIGINAL_PROB_PATH:
    cmd.extend([
        "--original_prob_path",
        str(ORIGINAL_PROB_PATH),
    ])

if SKIP_LEGACY_QUERY_SOURCES:
    cmd.append("--skip_legacy_query_sources")

if QUERY_DECODER_LOGIT_BASE_MODE:
    cmd.extend([
        "--query_decoder_logit_base_mode",
        str(QUERY_DECODER_LOGIT_BASE_MODE),
    ])

if EXPERT_BASE_MIX_ALPHA:
    cmd.extend([
        "--expert_base_mix_alpha",
        str(float(EXPERT_BASE_MIX_ALPHA)),
    ])

if ANCHOR_DELTA_GATE_INIT:
    cmd.extend([
        "--anchor_delta_gate_init",
        str(float(ANCHOR_DELTA_GATE_INIT)),
    ])

if ENABLE_IC_FUSION:
    cmd.append("--enable_ic_fusion")
    cmd.extend([
        "--ic_fusion_alpha_rare", str(IC_FUSION_ALPHA_RARE),
        "--ic_fusion_alpha_medium", str(IC_FUSION_ALPHA_MEDIUM),
        "--ic_fusion_alpha_common", str(IC_FUSION_ALPHA_COMMON),
        "--ic_fusion_source", str(IC_FUSION_SOURCE),
    ])

if ENABLE_SIMULATED_IC:
    cmd.append("--enable_simulated_ic")
    cmd.extend([
        "--simulated_ic_source_key",
        str(SIMULATED_IC_SOURCE_KEY),
        "--simulated_ic_threshold_min",
        str(float(SIMULATED_IC_THRESHOLD_MIN)),
        "--simulated_ic_threshold_max",
        str(float(SIMULATED_IC_THRESHOLD_MAX)),
        "--simulated_ic_threshold_step",
        str(float(SIMULATED_IC_THRESHOLD_STEP)),
    ])

    if SIMULATED_IC_INCLUDE_ZERO_THRESHOLD:
        cmd.append("--simulated_ic_include_zero_threshold")

    if SIMULATED_IC_SAVE_COUNTS:
        cmd.append("--simulated_ic_save_counts")

if NUM_WORKERS <= 0:
    cmd.extend(["--persistent_workers", "0"])
if MAX_BATCHES.strip():
    cmd.extend(["--max_batches", MAX_BATCHES.strip()])
if COMPUTE_SAMPLE_FMAX:
    cmd.append("--compute_sample_fmax")
if ALLOW_EVAL_LABEL_BOOST:
    cmd.append("--allow_eval_label_boost")

print("[Command]")
print(" ".join(cmd))
subprocess.run(cmd, check=True)
