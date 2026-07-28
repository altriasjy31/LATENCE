#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Environment-variable launcher for ``build_pp_edge_types_v2.py``.

Typical use::

    TASK=bp \
    RUN_TAG=bp_weak_detr_v3_expert_prob_warmstart340_to400 \
    EPOCH=100 \
    python run_build_pp_edge_types_v2.py

The defaults compile:

* PPI with ``combined_score >= 700`` and explicit reverse edges;
* directed core similarity at ``k=100`` as ``neighbor -> query`` messages;
* weak--core retrieval at ``k=100`` as ``weak -> core`` messages, enabling
  the downstream two-hop route ``weak -> core -> GO``.
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


ROOT = infer_project_root()
BUILD_SCRIPT = env_path("BUILD_SCRIPT", SCRIPT_DIR / "build_pp_edge_types.py")

TASK = os.environ.get("TASK", "bp").strip().lower()
EPOCH = os.environ.get("EPOCH", "100").strip()
RUN_TAG = os.environ.get(
    "RUN_TAG", f"{TASK}_weak_detr_v3_expert_prob_warmstart340_to400"
).strip()
if TASK not in {"bp", "mf", "cc"}:
    raise ValueError(f"Unknown TASK={TASK!r}; expected bp, mf or cc")
if not EPOCH or not RUN_TAG:
    raise ValueError("EPOCH and RUN_TAG must not be empty")

DEFAULT_STAGE2_DIR = (
    ROOT / "outputs" / "latence_nn_pp" / RUN_TAG / f"epoch{EPOCH}" / TASK
)
FEATURE_DIR = env_path("FEATURE_DIR", DEFAULT_STAGE2_DIR / "features")
PP_RELATIONS_DIR = env_path(
    "PP_RELATIONS_DIR", DEFAULT_STAGE2_DIR / "pp_relations"
)
PPI_PATH = env_path(
    "PPI_PATH", ROOT / "data" / "swiss_filtered_ppi_2204.mapped.tsv"
)
OUTPUT_DIR = env_path("OUTPUT_DIR", DEFAULT_STAGE2_DIR / "pp_edge_types")

RELATIONS = os.environ.get(
    "RELATIONS", "ppi,similar_to,weak_to_core"
).strip()
PPI_MIN_SCORE = float(os.environ.get("PPI_MIN_SCORE", "700"))
PPI_SCORE_SCALE = float(os.environ.get("PPI_SCORE_SCALE", "1000"))
PPI_DIRECTION = os.environ.get("PPI_DIRECTION", "bidirectional").strip()
PPI_INSERT_BATCH_SIZE = int(os.environ.get("PPI_INSERT_BATCH_SIZE", "50000"))
PPI_FETCH_BATCH_SIZE = int(os.environ.get("PPI_FETCH_BATCH_SIZE", "100000"))
MIN_PPI_MAPPED_EDGE_FRACTION = float(
    os.environ.get("MIN_PPI_MAPPED_EDGE_FRACTION", "0")
)

SIMILAR_K = int(os.environ.get("SIMILAR_K", "100"))
SIMILAR_MIN_SCORE = float(os.environ.get("SIMILAR_MIN_SCORE", "0"))
SIMILAR_MODE = os.environ.get("SIMILAR_MODE", "directed").strip()
SIMILAR_MESSAGE_DIRECTION = os.environ.get(
    "SIMILAR_MESSAGE_DIRECTION", "neighbor-to-query"
).strip()
SIMILAR_SCORE_REDUCE = os.environ.get("SIMILAR_SCORE_REDUCE", "min").strip()

WEAK_K = int(os.environ.get("WEAK_K", "100"))
WEAK_MIN_SCORE = float(os.environ.get("WEAK_MIN_SCORE", "0"))
WEAK_MESSAGE_DIRECTION = os.environ.get(
    "WEAK_MESSAGE_DIRECTION", "weak-to-core"
).strip()
KNN_WRITE_CHUNK_ROWS = int(os.environ.get("KNN_WRITE_CHUNK_ROWS", "65536"))

if not BUILD_SCRIPT.is_file():
    raise FileNotFoundError(f"BUILD_SCRIPT not found: {BUILD_SCRIPT}")
for path, label in (
    (FEATURE_DIR / "protein_registry.csv", "protein registry"),
    (FEATURE_DIR / "representation_manifest.json", "representation manifest"),
    (PP_RELATIONS_DIR / "pp_relations_manifest.json", "P-P relations manifest"),
):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
if "ppi" in {item.strip() for item in RELATIONS.split(",")} and not PPI_PATH.is_file():
    raise FileNotFoundError(f"PPI_PATH not found: {PPI_PATH}")

cmd = [
    sys.executable,
    str(BUILD_SCRIPT),
    "--feature-dir",
    str(FEATURE_DIR),
    "--pp-relations-dir",
    str(PP_RELATIONS_DIR),
    "--ppi-tsv",
    str(PPI_PATH),
    "--output-dir",
    str(OUTPUT_DIR),
    "--relations",
    RELATIONS,
    "--ppi-min-score",
    str(PPI_MIN_SCORE),
    "--ppi-score-scale",
    str(PPI_SCORE_SCALE),
    "--ppi-direction",
    PPI_DIRECTION,
    "--ppi-insert-batch-size",
    str(PPI_INSERT_BATCH_SIZE),
    "--ppi-fetch-batch-size",
    str(PPI_FETCH_BATCH_SIZE),
    "--min-ppi-mapped-edge-fraction",
    str(MIN_PPI_MAPPED_EDGE_FRACTION),
    "--similar-k",
    str(SIMILAR_K),
    "--similar-min-score",
    str(SIMILAR_MIN_SCORE),
    "--similar-mode",
    SIMILAR_MODE,
    "--similar-message-direction",
    SIMILAR_MESSAGE_DIRECTION,
    "--similar-score-reduce",
    SIMILAR_SCORE_REDUCE,
    "--weak-k",
    str(WEAK_K),
    "--weak-min-score",
    str(WEAK_MIN_SCORE),
    "--weak-message-direction",
    WEAK_MESSAGE_DIRECTION,
    "--knn-write-chunk-rows",
    str(KNN_WRITE_CHUNK_ROWS),
]
if env_bool("SKIP_MALFORMED_PPI_LINES", False):
    cmd.append("--skip-malformed-ppi-lines")
if env_bool("LARGE_INPUT_SHA256", False):
    cmd.append("--large-input-sha256")
if env_bool("OVERWRITE", False):
    cmd.append("--overwrite")

# Appended CLI flags take precedence because argparse keeps the final value.
cmd.extend(sys.argv[1:])

print("[Build evidence-separated Protein-Protein edge types]")
print(
    "LAUNCHER=run_build_pp_edge_types_v2.py "
    f"BUILDER={BUILD_SCRIPT.name}"
)
print(f"TASK={TASK} RUN_TAG={RUN_TAG} EPOCH={EPOCH}")
print(f"FEATURE_DIR={FEATURE_DIR}")
print(f"PP_RELATIONS_DIR={PP_RELATIONS_DIR}")
print(f"PPI_PATH={PPI_PATH}")
print(f"OUTPUT_DIR={OUTPUT_DIR}")
print(
    f"RELATIONS={RELATIONS} PPI_MIN_SCORE={PPI_MIN_SCORE} "
    f"PPI_DIRECTION={PPI_DIRECTION}"
)
print(
    f"SIMILAR_MODE={SIMILAR_MODE} SIMILAR_K={SIMILAR_K} "
    f"SIMILAR_MIN_SCORE={SIMILAR_MIN_SCORE} "
    f"SIMILAR_MESSAGE_DIRECTION={SIMILAR_MESSAGE_DIRECTION} "
    f"SIMILAR_SCORE_REDUCE={SIMILAR_SCORE_REDUCE}"
)
print(
    f"WEAK_K={WEAK_K} WEAK_MIN_SCORE={WEAK_MIN_SCORE} "
    f"WEAK_MESSAGE_DIRECTION={WEAK_MESSAGE_DIRECTION} "
    f"KNN_WRITE_CHUNK_ROWS={KNN_WRITE_CHUNK_ROWS}"
)
print("[Command]")
print(shlex.join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=ROOT)