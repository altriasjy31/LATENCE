#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runner for ``build_gg_relations.py``.

The defaults follow the Stage-2 TASK/RUN_TAG/EPOCH layout used by the protein
relation launchers.  ``GO_TERMS_PATH`` must identify the ordered first-stage
classifier vocabulary; several conventional project paths are auto-detected.

Examples
--------
Build complete BP is_a/part_of closures::

    TASK=bp GO_TERMS_PATH=/path/to/go_terms_bp.txt \
    python run_build_gg_relations.py

Use a trusted pickle vocabulary and an explicit nested key::

    TASK=mf GO_TERMS_PATH=/path/to/labels.pkl GO_TERMS_KEY=classes_ \
    ALLOW_PICKLE_INPUT=1 python run_build_gg_relations.py
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
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path.resolve()
    return None


ROOT = infer_project_root()
BUILD_SCRIPT = env_path("BUILD_GG_SCRIPT", SCRIPT_DIR / "build_gg_relations.py")

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

OBO_PATH = env_path("OBO_PATH", ROOT / "esm_models" / "go-basic.obo")

term_candidate_stems = [
    ROOT / "data" / "go_edges" / f"go_terms_{TASK}",
    ROOT / "data" / "go_edges" / f"{TASK}_go_terms",
    ROOT / "data" / "go_terms" / f"go_terms_{TASK}",
    ROOT / "data" / "go_terms" / TASK,
    ROOT / "data" / f"go_terms_{TASK}",
    ROOT / "esm_models" / f"go_terms_{TASK}",
]
term_candidates = [
    Path(str(stem) + suffix)
    for stem in term_candidate_stems
    for suffix in (".txt", ".tsv", ".csv", ".json", ".npy")
]
configured_terms = os.environ.get("GO_TERMS_PATH", "").strip()
if configured_terms:
    GO_TERMS_PATH = Path(configured_terms).expanduser().resolve()
else:
    GO_TERMS_PATH = first_existing(term_candidates)
    if GO_TERMS_PATH is None:
        attempted = "\n  ".join(str(x) for x in term_candidates)
        raise FileNotFoundError(
            "Could not auto-detect the ordered first-stage GO vocabulary.\n"
            "Set GO_TERMS_PATH explicitly. Tried:\n  " + attempted
        )

GO_TERMS_KEY = os.environ.get("GO_TERMS_KEY", "").strip()
DEFAULT_NBS_DIR = (
    ROOT / "outputs" / "latence_nbs" / RUN_TAG / f"epoch{EPOCH}" / TASK
)
OUTPUT_DIR = env_path("OUTPUT_DIR", DEFAULT_NBS_DIR / "gg_relations")

EXPECTED_NUM_TERMS = int(
    os.environ.get("EXPECTED_NUM_TERMS", str(TASK_NUM_CLASSES[TASK]))
)
MAX_HOPS = int(os.environ.get("MAX_HOPS", "0"))
INCLUDE_PART_OF = env_bool("INCLUDE_PART_OF", True)
OBSOLETE_POLICY = os.environ.get("OBSOLETE_POLICY", "error").strip()
DUPLICATE_POLICY = os.environ.get("DUPLICATE_POLICY", "replicate").strip()

if not BUILD_SCRIPT.is_file():
    raise FileNotFoundError(f"BUILD_GG_SCRIPT not found: {BUILD_SCRIPT}")
if not OBO_PATH.is_file():
    raise FileNotFoundError(f"OBO_PATH not found: {OBO_PATH}")
if not GO_TERMS_PATH.is_file():
    raise FileNotFoundError(f"GO_TERMS_PATH not found: {GO_TERMS_PATH}")
if EXPECTED_NUM_TERMS <= 0:
    raise ValueError("EXPECTED_NUM_TERMS must be positive")
if MAX_HOPS < 0:
    raise ValueError("MAX_HOPS must be zero or positive")
if OBSOLETE_POLICY not in {"error", "keep", "replace"}:
    raise ValueError("OBSOLETE_POLICY must be error, keep or replace")
if DUPLICATE_POLICY not in {"error", "representative", "replicate"}:
    raise ValueError(
        "DUPLICATE_POLICY must be error, representative or replicate"
    )

cmd = [
    sys.executable,
    str(BUILD_SCRIPT),
    "--obo",
    str(OBO_PATH),
    "--go-terms",
    str(GO_TERMS_PATH),
    "--task",
    TASK,
    "--output-dir",
    str(OUTPUT_DIR),
    "--expected-num-terms",
    str(EXPECTED_NUM_TERMS),
    "--max-hops",
    str(MAX_HOPS),
    "--obsolete-policy",
    OBSOLETE_POLICY,
    "--duplicate-policy",
    DUPLICATE_POLICY,
]

if GO_TERMS_KEY:
    cmd.extend(["--go-terms-key", GO_TERMS_KEY])
cmd.append("--include-part-of" if INCLUDE_PART_OF else "--no-part-of")
if env_bool("ALLOW_MISSING_TERMS", False):
    cmd.append("--allow-missing-terms")
if env_bool("ALLOW_CROSS_NAMESPACE_TERMS", False):
    cmd.append("--allow-cross-namespace-terms")
if env_bool("ALLOW_PICKLE_INPUT", False):
    cmd.append("--allow-pickle-input")
if env_bool("OVERWRITE", False):
    cmd.append("--overwrite")

# Explicit CLI arguments are appended last and therefore override scalar values.
cmd.extend(sys.argv[1:])

print("[Build GO-GO ontology relations]")
print(f"TASK={TASK} RUN_TAG={RUN_TAG} EPOCH={EPOCH}")
print(f"OBO_PATH={OBO_PATH}")
print(f"GO_TERMS_PATH={GO_TERMS_PATH}")
print(f"OUTPUT_DIR={OUTPUT_DIR}")
print(
    f"EXPECTED_NUM_TERMS={EXPECTED_NUM_TERMS} "
    f"INCLUDE_PART_OF={INCLUDE_PART_OF} MAX_HOPS={MAX_HOPS} "
    f"DUPLICATE_POLICY={DUPLICATE_POLICY}"
)
print("[Command]")
print(shlex.join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=ROOT)
