#!/usr/bin/env python3
"""Direct BP runner for the LATENCE GO->Protein inverted-index build."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from build_go_protein_inverted_index import main

PROJECT_ROOT = Path(
    os.environ.get(
        "LATENCE_PROJECT_ROOT",
        "/home/dataset-local/data_local/shaojiangyi/latence-project",
    )
)
TASK = os.environ.get("TASK", "bp")
RUN_TAG = os.environ.get(
    "RUN_TAG", "bp_weak_detr_v3_expert_prob_warmstart340_to400"
)
EPOCH = int(os.environ.get("EPOCH", "100"))
CHUNK_EDGES = int(os.environ.get("CHUNK_EDGES", "2000000"))
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"

# Renamed NBS data root.  The runner deliberately does not fall back silently
# to outputs/latence_nn_pp after the project-wide migration.
TASK_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "latence_nbs"
    / RUN_TAG
    / f"epoch{EPOCH}"
    / TASK
)
WEAK_GRAPH_MANIFEST = (
    TASK_ROOT / "weak_graph_predictions" / "weak_graph_predictions_manifest.json"
)
PROTEIN_REGISTRY = TASK_ROOT / "features" / "protein_registry.csv"
OUTPUT_DIR = TASK_ROOT / "nbs_indices" / "go_protein"


def run() -> int:
    argv = [
        "--project-root",
        str(PROJECT_ROOT),
        "--weak-graph-manifest",
        str(WEAK_GRAPH_MANIFEST),
        "--protein-registry",
        str(PROTEIN_REGISTRY),
        "--output-dir",
        str(OUTPUT_DIR),
        "--chunk-edges",
        str(CHUNK_EDGES),
    ]
    if OVERWRITE:
        argv.append("--overwrite")
    return main(argv)


if __name__ == "__main__":
    sys.exit(run())
