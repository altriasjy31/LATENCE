#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Environment-variable launcher for ``export_weak_graph_predictions_v2.py``.

Typical BP epoch-100 run::

    TASK=bp \
    RUN_TAG=bp_weak_detr_v3_expert_prob_warmstart340_to400 \
    EPOCH=100 \
    OVERWRITE=1 \
    python run_export_weak_graph_predictions_v2.py

The launcher defaults to core+weak proteins.  It deliberately does not include
validation or independent-test proteins in the training graph export.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


LAUNCHER_ID = "run_export_weak_graph_predictions_v2"
LAUNCHER_VERSION = "1.1.0"
SCRIPT_DIR = Path(__file__).resolve().parent


def infer_project_root() -> Path:
    configured = os.environ.get("PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent):
        if (candidate / "experiments").is_dir() and (candidate / "data").is_dir():
            return candidate.resolve()
    return SCRIPT_DIR


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be boolean, got {raw!r}")


def first_existing_or_first(paths: Iterable[Path]) -> Path:
    candidates = list(paths)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve()


ROOT = infer_project_root()
EXPORT_SCRIPT = env_path(
    "EXPORT_SCRIPT",
    first_existing_or_first(
        [
            SCRIPT_DIR / "export_weak_graph_predictions_v2.py",
            SCRIPT_DIR / "export_weak_graph_predictions.py",
        ]
    ),
)
DIAGNOSTIC_EVAL_SCRIPT = env_path(
    "DIAGNOSTIC_EVAL_SCRIPT",
    ROOT / "experiments" / "eval_weak_ind_test_detr_diagnostics.py",
)

TASK = os.environ.get("TASK", "bp").strip().lower()
EPOCH = os.environ.get("EPOCH", "100").strip()
RUN_TAG = os.environ.get(
    "RUN_TAG", f"{TASK}_weak_detr_v3_expert_prob_warmstart340_to400"
).strip()
if TASK not in {"bp", "mf", "cc"}:
    raise ValueError(f"Unknown TASK={TASK!r}; expected bp, mf or cc")
if not EPOCH or not RUN_TAG:
    raise ValueError("EPOCH and RUN_TAG must not be empty")

TASK_NUM_CLASSES = {
    "bp": 21312,
    "mf": 7038,
    "cc": 2903,
}

CHECKPOINT = env_path(
    "CHECKPOINT",
    ROOT
    / "outputs"
    / "weak_exp_train_detr"
    / RUN_TAG
    / f"weak_detr_decoder_epoch{EPOCH}.pt",
)
MODEL_CONFIG = env_path(
    "MODEL_CONFIG",
    ROOT / "data" / "msa_models" / "configs" / "model_opts"
    / f"{TASK}_msa_model_config.pkl",
)
FILE_ADDRESS = env_path(
    "FILE_ADDRESS", ROOT / "data" / "unidata_with_exp_train_pseudo.pkl"
)
WORKING_ADDRESS = env_path(
    "WORKING_ADDRESS", ROOT / "data" / "sprot_2204_MSA_bin" / "index.pkl"
)

expert_dir = env_path(
    "EXPERT_PROB_OUTPUT_DIR",
    ROOT / "data" / "external_probs" / "esm2_3b_exp_train",
)
expert_candidates = [
    expert_dir / f"{TASK}_exp_train_predictions.prop.float16.npy",
    expert_dir / f"{TASK}_exp_train_predictions.float16.npy",
    expert_dir / f"{TASK}_exp_train_probs.float16.npy",
    expert_dir / f"{TASK}_predictions.float16.npy",
]
WEAK_EXTERNAL_PROB_PATH = env_path(
    "WEAK_EXTERNAL_PROB_PATH",
    first_existing_or_first(expert_candidates),
)

DEFAULT_NBS_DIR = (
    ROOT / "outputs" / "latence_nbs" / RUN_TAG / f"epoch{EPOCH}" / TASK
)
FEATURE_DIR = env_path("FEATURE_DIR", DEFAULT_NBS_DIR / "features")
GO_REGISTRY = env_path(
    "GO_REGISTRY", DEFAULT_NBS_DIR / "gg_relations" / "go_registry.tsv"
)
OUTPUT_DIR = env_path(
    "OUTPUT_DIR", DEFAULT_NBS_DIR / "weak_graph_predictions"
)

ROLES = os.environ.get("ROLES", "core,weak").strip()
ROLE_MODES = [
    item.strip()
    for item in os.environ.get("ROLE_MODES", "").split(",")
    if item.strip()
]
TRAIN_ARGS_JSON = os.environ.get("TRAIN_ARGS_JSON", "auto").strip()

TOP_K = int(os.environ.get("TOP_K", "64"))
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
MSA_MAX_SIZE = os.environ.get("MSA_MAX_SIZE", "").strip()
PERMUTE_DIMS = os.environ.get("PERMUTE_DIMS", "0,3,2,1").replace(",", " ").split()

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", "4"))
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", "2"))
PERSISTENT_WORKERS = env_bool("PERSISTENT_WORKERS", NUM_WORKERS > 0)
PIN_MEMORY = env_bool("PIN_MEMORY", True)
DEVICE = os.environ.get("DEVICE", "auto").strip()
GPU_IDS = os.environ.get("GPU_IDS", "").strip()
AMP_DTYPE = os.environ.get("AMP_DTYPE", "bfloat16").strip()
OUTPUT_PROB_DTYPE = os.environ.get("OUTPUT_PROB_DTYPE", "float16").strip()
MSA_READ_MODE = os.environ.get("MSA_READ_MODE", "full").strip()
MSA_SAMPLE_STRATEGY = os.environ.get("MSA_SAMPLE_STRATEGY", "random").strip()
MSA_SHUFFLE_ROWS = env_bool("MSA_SHUFFLE_ROWS_AT_GETITEM", False)
MSA_CACHE_GB = float(os.environ.get("MSA_CACHE_GB", "4.0"))
MSA_MAX_OPEN_FILES = int(os.environ.get("MSA_MAX_OPEN_FILES", "256"))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "1"))
SAMPLER_SEED = int(os.environ.get("SAMPLER_SEED", "1"))
SEED = int(os.environ.get("SEED", "3407"))
MAX_BATCHES = os.environ.get("MAX_BATCHES", "").strip()

RARE_POLICY = os.environ.get("RARE_POLICY", "train_q33").strip()
RARE_MAX_TRAIN_COUNT = float(os.environ.get("RARE_MAX_TRAIN_COUNT", "5"))
RARE_GO_IDS = os.environ.get("RARE_GO_IDS", "").strip()
INCLUDE_ZERO_TRAIN_GO = env_bool("INCLUDE_ZERO_TRAIN_GO", False)
RARE_GO_TOPK = int(os.environ.get("RARE_GO_TOPK", "20"))
RARE_SELECTOR_SCOPE = os.environ.get(
    "RARE_SELECTOR_SCOPE", "rare_first"
).strip()
RARE_MIN_BACKBONE_PROB = float(
    os.environ.get("RARE_MIN_BACKBONE_PROB", "0")
)

MODEL_OUT_ROLE = os.environ.get("MODEL_OUT_ROLE", "weak").strip()
MODEL_OUT_DECODER_PROB_SOURCE = os.environ.get(
    "MODEL_OUT_DECODER_PROB_SOURCE", "expert"
).strip()
MODEL_OUT_DECODER_PROB_ALPHA = float(
    os.environ.get("MODEL_OUT_DECODER_PROB_ALPHA", "0.5")
)
MODEL_OUT_TOPK_SOURCE = os.environ.get(
    "MODEL_OUT_TOPK_SOURCE", "external_topk"
).strip()
MODEL_OUT_EXTERNAL_BLEND_ALPHA = float(
    os.environ.get("MODEL_OUT_EXTERNAL_BLEND_ALPHA", "1.0")
)
PSEUDO_THRESHOLD = float(os.environ.get("PSEUDO_THRESHOLD", "0.5"))
SAVE_DENSE_BACKBONE = env_bool("SAVE_DENSE_BACKBONE", True)
SAVE_DENSE_MODEL_OUT = env_bool("SAVE_DENSE_MODEL_OUT", True)
REQUIRE_ANCHOR_MODEL_OUT = env_bool("REQUIRE_ANCHOR_MODEL_OUT", True)

# These are optional audited overrides.  Empty values mean restore the exact
# training value from args.json/checkpoint model_args.
QUERY_DECODER_LOGIT_BASE_MODE = os.environ.get(
    "QUERY_DECODER_LOGIT_BASE_MODE", ""
).strip()
EXPERT_BASE_MIX_ALPHA = os.environ.get("EXPERT_BASE_MIX_ALPHA", "").strip()

for path, label in (
    (EXPORT_SCRIPT, "EXPORT_SCRIPT"),
    (DIAGNOSTIC_EVAL_SCRIPT, "DIAGNOSTIC_EVAL_SCRIPT"),
    (CHECKPOINT, "CHECKPOINT"),
    (MODEL_CONFIG, "MODEL_CONFIG"),
    (FILE_ADDRESS, "FILE_ADDRESS"),
    (WORKING_ADDRESS, "WORKING_ADDRESS"),
    (FEATURE_DIR / "protein_registry.csv", "protein registry"),
    (GO_REGISTRY, "GO_REGISTRY"),
):
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
if MODEL_OUT_ROLE in {item.strip() for item in ROLES.split(",")}:
    if not WEAK_EXTERNAL_PROB_PATH.is_file():
        attempted = "\n  ".join(str(path) for path in expert_candidates)
        raise FileNotFoundError(
            f"WEAK_EXTERNAL_PROB_PATH not found: {WEAK_EXTERNAL_PROB_PATH}\n"
            f"Set it explicitly. Auto-detection tried:\n  {attempted}"
        )
if len(PERMUTE_DIMS) != 4:
    raise ValueError(f"PERMUTE_DIMS must contain four integers: {PERMUTE_DIMS}")

cmd = [
    sys.executable,
    str(EXPORT_SCRIPT),
    "--project-root",
    str(ROOT),
    "--diagnostic-eval-script",
    str(DIAGNOSTIC_EVAL_SCRIPT),
    "--checkpoint",
    str(CHECKPOINT),
    "--train-args-json",
    TRAIN_ARGS_JSON,
    "--task",
    TASK,
    "--num-classes",
    str(TASK_NUM_CLASSES[TASK]),
    "--output-dir",
    str(OUTPUT_DIR),
    "--feature-dir",
    str(FEATURE_DIR),
    "--go-registry",
    str(GO_REGISTRY),
    "--roles",
    ROLES,
    "--model-config",
    str(MODEL_CONFIG),
    "--file-address",
    str(FILE_ADDRESS),
    "--working-address",
    str(WORKING_ADDRESS),
    "--top-k",
    str(TOP_K),
    "--max-len",
    str(MAX_LEN),
    "--permute-dims",
    *PERMUTE_DIMS,
    "--batch-size",
    str(BATCH_SIZE),
    "--dataloader-num-workers",
    str(NUM_WORKERS),
    "--prefetch-factor",
    str(PREFETCH_FACTOR),
    "--persistent-workers",
    str(PERSISTENT_WORKERS).lower(),
    "--pin-memory",
    str(PIN_MEMORY).lower(),
    "--device",
    DEVICE,
    "--amp-dtype",
    AMP_DTYPE,
    "--output-prob-dtype",
    OUTPUT_PROB_DTYPE,
    "--msa-read-mode",
    MSA_READ_MODE,
    "--msa-sample-strategy",
    MSA_SAMPLE_STRATEGY,
    "--msa-shuffle-rows-at-getitem",
    str(MSA_SHUFFLE_ROWS).lower(),
    "--msa-cache-gb",
    str(MSA_CACHE_GB),
    "--msa-max-open-files",
    str(MSA_MAX_OPEN_FILES),
    "--sample-seed",
    str(SAMPLE_SEED),
    "--sampler-seed",
    str(SAMPLER_SEED),
    "--seed",
    str(SEED),
    "--rare-policy",
    RARE_POLICY,
    "--rare-max-train-count",
    str(RARE_MAX_TRAIN_COUNT),
    "--include-zero-train-go",
    str(INCLUDE_ZERO_TRAIN_GO).lower(),
    "--rare-go-topk",
    str(RARE_GO_TOPK),
    "--rare-selector-scope",
    RARE_SELECTOR_SCOPE,
    "--rare-min-backbone-prob",
    str(RARE_MIN_BACKBONE_PROB),
    "--modelout-role",
    MODEL_OUT_ROLE,
    "--weak-external-prob-path",
    str(WEAK_EXTERNAL_PROB_PATH),
    "--modelout-decoder-prob-source",
    MODEL_OUT_DECODER_PROB_SOURCE,
    "--modelout-decoder-prob-alpha",
    str(MODEL_OUT_DECODER_PROB_ALPHA),
    "--modelout-topk-source",
    MODEL_OUT_TOPK_SOURCE,
    "--modelout-external-prob-blend-alpha",
    str(MODEL_OUT_EXTERNAL_BLEND_ALPHA),
    "--pseudo-threshold",
    str(PSEUDO_THRESHOLD),
    "--save-dense-backbone",
    str(SAVE_DENSE_BACKBONE).lower(),
    "--save-dense-modelout",
    str(SAVE_DENSE_MODEL_OUT).lower(),
    "--require-anchor-modelout",
    str(REQUIRE_ANCHOR_MODEL_OUT).lower(),
]

for role_mode in ROLE_MODES:
    cmd.extend(["--role-mode", role_mode])
if MSA_MAX_SIZE:
    cmd.extend(["--msa-max-size", MSA_MAX_SIZE])
if GPU_IDS:
    cmd.extend(["--gpu-ids", GPU_IDS])
if MAX_BATCHES:
    cmd.extend(["--max-batches", MAX_BATCHES])
if RARE_GO_IDS:
    cmd.extend(["--rare-go-ids", str(Path(RARE_GO_IDS).expanduser().resolve())])
if QUERY_DECODER_LOGIT_BASE_MODE:
    cmd.extend(
        ["--query-decoder-logit-base-mode", QUERY_DECODER_LOGIT_BASE_MODE]
    )
if EXPERT_BASE_MIX_ALPHA:
    cmd.extend(["--expert-base-mix-alpha", EXPERT_BASE_MIX_ALPHA])
if env_bool("NO_AMP", False):
    cmd.append("--no-amp")
if env_bool("ALLOW_PARTIAL_CHECKPOINT", False):
    cmd.append("--allow-partial-checkpoint")
if env_bool("SKIP_CHECKPOINT_SHA256", False):
    cmd.append("--skip-checkpoint-sha256")
if env_bool("OVERWRITE", False):
    cmd.append("--overwrite")

# Explicit command-line arguments are appended last and override scalar values.
cmd.extend(sys.argv[1:])

print("[Export weak graph predictions]")
print(
    f"LAUNCHER={Path(__file__).name} version={LAUNCHER_VERSION} "
    f"EXPORTER={EXPORT_SCRIPT.name}"
)
print(f"TASK={TASK} RUN_TAG={RUN_TAG} EPOCH={EPOCH} ROLES={ROLES}")
print(f"CHECKPOINT={CHECKPOINT}")
print(f"FEATURE_DIR={FEATURE_DIR}")
print(f"GO_REGISTRY={GO_REGISTRY}")
print(f"WEAK_EXTERNAL_PROB_PATH={WEAK_EXTERNAL_PROB_PATH}")
print(f"OUTPUT_DIR={OUTPUT_DIR}")
print(
    f"RARE_POLICY={RARE_POLICY} RARE_GO_TOPK={RARE_GO_TOPK} "
    f"RARE_SELECTOR_SCOPE={RARE_SELECTOR_SCOPE} "
    f"RARE_MIN_BACKBONE_PROB={RARE_MIN_BACKBONE_PROB}"
)
print(
    f"MODEL_OUT_SOURCE={MODEL_OUT_DECODER_PROB_SOURCE} "
    f"MODEL_OUT_TOPK_SOURCE={MODEL_OUT_TOPK_SOURCE} "
    f"PSEUDO_THRESHOLD(strict>)={PSEUDO_THRESHOLD}"
)
print("[Command]")
print(shlex.join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=ROOT)
