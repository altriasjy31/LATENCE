#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LATENCE_PROJECT_ROOT", "/home/dataset-local/data_local/shaojiangyi/latence-project")).resolve()
TASK = os.environ.get("TASK", "bp")
RUN_TAG = os.environ.get("RUN_TAG", "bp_weak_detr_v3_expert_prob_warmstart340_to400")
EPOCH = int(os.environ.get("EPOCH", "100"))
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"
CHUNK = int(os.environ.get("CHUNK_EDGES", "2000000"))
TASK_ROOT = ROOT / "outputs/latence_nbs" / RUN_TAG / f"epoch{EPOCH}" / TASK


def main() -> int:
    argv = [
        sys.executable, str(ROOT / "scripts/nbs/build_pp_sampling_indices.py"),
        "--project-root", str(ROOT),
        "--pp-manifest", str(TASK_ROOT / "pp_edge_types/pp_edge_types_manifest.json"),
        "--output-dir", str(TASK_ROOT / "nbs_indices/pp_sampling"),
        "--chunk-edges", str(CHUNK),
    ]
    if OVERWRITE:
        argv.append("--overwrite")
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
