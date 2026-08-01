#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runner for ``build_pp_relations_faiss.py``.

It uses the same TASK/RUN_TAG/EPOCH convention as
``run_export_protein_universe_repr.py`` so the exporter output is discovered
automatically.

Examples
--------
Build core-core and weak-core exact cosine neighbors::

    python run_build_pp_relations_faiss.py

Use GPU FAISS and override k::

    FAISS_GPU_ID=0 KMAX=100 python run_build_pp_relations_faiss.py

Run only the weak-core relation::

    RELATION=weak-core python run_build_pp_relations_faiss.py --overwrite
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
    for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent):
        if (candidate / "experiments").is_dir() and (candidate / "data").is_dir():
            return candidate.resolve()
    return SCRIPT_DIR


ROOT = infer_project_root()
BUILD_SCRIPT = Path(
    os.environ.get("BUILD_SCRIPT", str(SCRIPT_DIR / "build_pp_relations_faiss.py"))
).expanduser().resolve()

TASK = os.environ.get("TASK", "bp").strip().lower()
EPOCH = os.environ.get("EPOCH", "6").strip()
RUN_TAG = os.environ.get("RUN_TAG", f"{TASK}_weak_detr_v3_expert_prob").strip()

if TASK not in {"bp", "mf", "cc"}:
    raise ValueError(f"Unknown TASK={TASK!r}; expected bp, mf or cc")
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


DEFAULT_NBS_DIR = (
    ROOT / "outputs" / "latence_nbs" / RUN_TAG / f"epoch{EPOCH}" / TASK
)
FEATURE_DIR = env_path("FEATURE_DIR", DEFAULT_NBS_DIR / "features")
OUTPUT_DIR = env_path("OUTPUT_DIR", DEFAULT_NBS_DIR / "pp_relations")

RELATION = os.environ.get("RELATION", "all").strip()
QUERY_ROLES = os.environ.get("QUERY_ROLES", "weak").strip()
KMAX = int(os.environ.get("KMAX", "100"))
QUERY_BATCH_SIZE = int(os.environ.get("QUERY_BATCH_SIZE", "8192"))
ADD_BATCH_SIZE = int(os.environ.get("ADD_BATCH_SIZE", "32768"))
SCORE_DTYPE = os.environ.get("SCORE_DTYPE", "float16").strip()
BACKEND = os.environ.get("BACKEND", "faiss").strip()
FAISS_GPU_ID = int(os.environ.get("FAISS_GPU_ID", "-1"))
FAISS_THREADS = int(os.environ.get("FAISS_THREADS", "0"))
NUMPY_INDEX_BLOCK_SIZE = int(os.environ.get("NUMPY_INDEX_BLOCK_SIZE", "32768"))

if not BUILD_SCRIPT.is_file():
    raise FileNotFoundError(f"BUILD_SCRIPT not found: {BUILD_SCRIPT}")
manifest_path = FEATURE_DIR / "representation_manifest.json"
if not manifest_path.is_file():
    raise FileNotFoundError(
        f"Representation manifest not found: {manifest_path}\n"
        "Run run_export_protein_universe_repr.py first, or set FEATURE_DIR explicitly."
    )
if RELATION not in {"core-core", "weak-core", "all"}:
    raise ValueError(f"RELATION must be core-core, weak-core or all, got {RELATION!r}")
if not QUERY_ROLES:
    raise ValueError("QUERY_ROLES must not be empty")
if KMAX <= 0 or QUERY_BATCH_SIZE <= 0 or ADD_BATCH_SIZE <= 0:
    raise ValueError("KMAX, QUERY_BATCH_SIZE and ADD_BATCH_SIZE must be positive")

cmd = [
    sys.executable,
    str(BUILD_SCRIPT),
    "--feature-dir", str(FEATURE_DIR),
    "--output-dir", str(OUTPUT_DIR),
    "--relation", RELATION,
    "--query-roles", QUERY_ROLES,
    "--kmax", str(KMAX),
    "--query-batch-size", str(QUERY_BATCH_SIZE),
    "--add-batch-size", str(ADD_BATCH_SIZE),
    "--score-dtype", SCORE_DTYPE,
    "--backend", BACKEND,
    "--faiss-gpu-id", str(FAISS_GPU_ID),
    "--faiss-threads", str(FAISS_THREADS),
    "--numpy-index-block-size", str(NUMPY_INDEX_BLOCK_SIZE),
]

if env_bool("ALLOW_ZERO_NORM", False):
    cmd.append("--allow-zero-norm")
if env_bool("OVERWRITE", False):
    cmd.append("--overwrite")

# Explicit CLI arguments take precedence because argparse keeps the last value.
cmd.extend(sys.argv[1:])

print("[Build Protein-Protein FAISS relations]")
print(f"TASK={TASK} RUN_TAG={RUN_TAG} EPOCH={EPOCH}")
print(f"RELATION={RELATION} QUERY_ROLES={QUERY_ROLES} KMAX={KMAX}")
print(f"FEATURE_DIR={FEATURE_DIR}")
print(f"OUTPUT_DIR={OUTPUT_DIR}")
print("[Command]")
print(shlex.join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=ROOT)