#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runner for ``export_protein_universe_repr.py``.

The defaults mirror the Stage-1 LATENCE launchers.  Environment variables are
used for routine configuration, while extra CLI arguments are appended to the
underlying exporter and can therefore override the generated command.

Examples
--------
Export core and weak proteins with the default BP checkpoint::

    python run_export_protein_universe_repr.py

Select another checkpoint/task and run a short diagnostic::

    TASK=mf EPOCH=100 RUN_TAG=mf_weak_detr_v3_expert_prob \
    python run_export_protein_universe_repr.py --max-batches 2 --overwrite

The default MSA index is ``sprot_2204_MSA_bin`` because core and weak proteins
come from the training protein universe.  ``ind_MSA_bin`` should only be used
for a separate ind_test-only export.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def infer_project_root() -> Path:
    configured = os.environ.get("PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    # Support both layouts: scripts copied into the LATENCE root, or kept in a
    # latence_stage2_pp/ subdirectory directly below it.
    for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent):
        if (candidate / "experiments").is_dir() and (candidate / "data").is_dir():
            return candidate.resolve()
    return SCRIPT_DIR


ROOT = infer_project_root()
EXPORT_SCRIPT = Path(
    os.environ.get("EXPORT_SCRIPT", str(SCRIPT_DIR / "export_protein_universe_repr.py"))
).expanduser().resolve()

TASK = os.environ.get("TASK", "bp").strip().lower()
EPOCH = os.environ.get("EPOCH", "6").strip()
RUN_TAG = os.environ.get("RUN_TAG", f"{TASK}_weak_detr_v3_expert_prob").strip()

TASK_NUM_CLASSES = {
    "cc": 2903,
    "mf": 7038,
    "bp": 21312,
}

if TASK not in TASK_NUM_CLASSES:
    raise ValueError(f"Unknown TASK={TASK!r}; expected one of {sorted(TASK_NUM_CLASSES)}")
if not EPOCH:
    raise ValueError("EPOCH must not be empty")
if not RUN_TAG:
    raise ValueError("RUN_TAG must not be empty")


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


MODEL_CONFIG = env_path(
    "MODEL_CONFIG",
    ROOT / "data" / "msa_models" / "configs" / "model_opts" / f"{TASK}_msa_model_config.pkl",
)
FILE_ADDRESS = env_path(
    "FILE_ADDRESS",
    ROOT / "data" / "unidata_with_exp_train_pseudo.pkl",
)
WORKING_ADDRESS = env_path(
    "WORKING_ADDRESS",
    ROOT / "data" / "sprot_2204_MSA_bin" / "index.pkl",
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs"
    / "weak_exp_train_detr"
    / RUN_TAG
    / f"weak_detr_decoder_epoch{EPOCH}.pt"
)
CHECKPOINT = env_path("CHECKPOINT", DEFAULT_CHECKPOINT)

DEFAULT_NBS_DIR = (
    ROOT / "outputs" / "latence_nbs" / RUN_TAG / f"epoch{EPOCH}" / TASK
)
OUTPUT_DIR = env_path("OUTPUT_DIR", DEFAULT_NBS_DIR / "features")

ROLES = os.environ.get("ROLES", "core,weak").strip()
TRAIN_ARGS_JSON = os.environ.get("TRAIN_ARGS_JSON", "auto").strip()
ROLE_MODES = [x.strip() for x in os.environ.get("ROLE_MODES", "").split(",") if x.strip()]

# These values match run_weak_exp_train_detr.py.  They are passed explicitly so
# representation export cannot silently use a different MSA input shape.
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
FEATURE_DTYPE = os.environ.get("FEATURE_DTYPE", "float16").strip()
MSA_READ_MODE = os.environ.get("MSA_READ_MODE", "full").strip()
MSA_SAMPLE_STRATEGY = os.environ.get("MSA_SAMPLE_STRATEGY", "random").strip()
MSA_SHUFFLE_ROWS = env_bool("MSA_SHUFFLE_ROWS_AT_GETITEM", False)
MSA_CACHE_GB = float(os.environ.get("MSA_CACHE_GB", "4.0"))
MSA_MAX_OPEN_FILES = int(os.environ.get("MSA_MAX_OPEN_FILES", "256"))
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "3407"))
SAMPLER_SEED = int(os.environ.get("SAMPLER_SEED", "3407"))
MAX_BATCHES = os.environ.get("MAX_BATCHES", "").strip()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


require_file(EXPORT_SCRIPT, "EXPORT_SCRIPT")
require_file(CHECKPOINT, "CHECKPOINT")
require_file(MODEL_CONFIG, "MODEL_CONFIG")
require_file(FILE_ADDRESS, "FILE_ADDRESS")
require_file(WORKING_ADDRESS, "WORKING_ADDRESS")

if not ROLES:
    raise ValueError("ROLES must not be empty")
if len(PERMUTE_DIMS) != 4:
    raise ValueError(f"PERMUTE_DIMS must contain four integers, got {PERMUTE_DIMS}")
if TOP_K <= 0 or MAX_LEN <= 0 or BATCH_SIZE <= 0:
    raise ValueError("TOP_K, MAX_LEN and BATCH_SIZE must be positive")
if NUM_WORKERS < 0:
    raise ValueError("DATALOADER_NUM_WORKERS must be non-negative")

cmd = [
    sys.executable,
    str(EXPORT_SCRIPT),
    "--project-root", str(ROOT),
    "--checkpoint", str(CHECKPOINT),
    "--train-args-json", TRAIN_ARGS_JSON,
    "--task", TASK,
    "--output-dir", str(OUTPUT_DIR),
    "--roles", ROLES,
    "--model-config", str(MODEL_CONFIG),
    "--file-address", str(FILE_ADDRESS),
    "--working-address", str(WORKING_ADDRESS),
    "--num-classes", str(TASK_NUM_CLASSES[TASK]),
    "--top-k", str(TOP_K),
    "--max-len", str(MAX_LEN),
    "--permute-dims", *PERMUTE_DIMS,
    "--batch-size", str(BATCH_SIZE),
    "--dataloader-num-workers", str(NUM_WORKERS),
    "--prefetch-factor", str(PREFETCH_FACTOR),
    "--persistent-workers", str(PERSISTENT_WORKERS).lower(),
    "--pin-memory", str(PIN_MEMORY).lower(),
    "--device", DEVICE,
    "--amp-dtype", AMP_DTYPE,
    "--feature-dtype", FEATURE_DTYPE,
    "--msa-read-mode", MSA_READ_MODE,
    "--msa-sample-strategy", MSA_SAMPLE_STRATEGY,
    "--msa-shuffle-rows-at-getitem", str(MSA_SHUFFLE_ROWS).lower(),
    "--msa-cache-gb", str(MSA_CACHE_GB),
    "--msa-max-open-files", str(MSA_MAX_OPEN_FILES),
    "--sample-seed", str(SAMPLE_SEED),
    "--sampler-seed", str(SAMPLER_SEED),
]

for role_mode in ROLE_MODES:
    cmd.extend(["--role-mode", role_mode])
if MSA_MAX_SIZE:
    cmd.extend(["--msa-max-size", MSA_MAX_SIZE])
if GPU_IDS:
    cmd.extend(["--gpu-ids", GPU_IDS])
if MAX_BATCHES:
    cmd.extend(["--max-batches", MAX_BATCHES])
if env_bool("NO_AMP", False):
    cmd.append("--no-amp")
if env_bool("OVERWRITE", False):
    cmd.append("--overwrite")
if env_bool("ALLOW_PARTIAL_CHECKPOINT", False):
    cmd.append("--allow-partial-checkpoint")
if env_bool("ALLOW_CROSS_ROLE_OVERLAP", False):
    cmd.append("--allow-cross-role-overlap")

# Explicit CLI arguments take precedence because argparse keeps the last value.
cmd.extend(sys.argv[1:])

print("[Export protein-universe representations]")
print(f"TASK={TASK} RUN_TAG={RUN_TAG} EPOCH={EPOCH}")
print(f"ROLES={ROLES}")
print(f"CHECKPOINT={CHECKPOINT}")
print(f"WORKING_ADDRESS={WORKING_ADDRESS}")
print(f"OUTPUT_DIR={OUTPUT_DIR}")
print("[Command]")
print(shlex.join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=ROOT)
