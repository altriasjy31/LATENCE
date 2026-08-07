#!/usr/bin/env python3
"""Direct LATENCE runner for gold Protein->GO edge export.

Revision (1): default to strict automatic protein-ID alignment.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get(
        "LATENCE_PROJECT_ROOT",
        "/home/dataset-local/data_local/shaojiangyi/latence-project",
    )
).resolve()
TASK = os.environ.get("TASK", "bp")
RUN_TAG = os.environ.get(
    "RUN_TAG", "bp_weak_detr_v3_expert_prob_warmstart340_to400"
)
EPOCH = int(os.environ.get("EPOCH", "100"))
MODE = os.environ.get("GOLD_MODE", "train")
ROLE = os.environ.get("GOLD_ROLE", "core")
NAME_KEY = os.environ.get("GOLD_NAME_KEY", "proteins")
LABEL_KEY = os.environ.get("GOLD_LABEL_KEY", "prop_annotations")
ALIGNMENT_POLICY = os.environ.get("GOLD_ALIGNMENT_POLICY", "auto")
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"

METADATA_FILE = Path(
    os.environ.get(
        "GOLD_METADATA_FILE",
        str(PROJECT_ROOT / "data" / "unidata_with_exp_train_pseudo.pkl"),
    )
).resolve()
TASK_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "latence_nbs"
    / RUN_TAG
    / f"epoch{EPOCH}"
    / TASK
)
PROTEIN_REGISTRY = Path(
    os.environ.get(
        "PROTEIN_REGISTRY",
        str(TASK_ROOT / "features" / "protein_registry.csv"),
    )
).resolve()
GO_REGISTRY = Path(
    os.environ.get(
        "GO_REGISTRY",
        str(TASK_ROOT / "gg_relations" / "go_registry.tsv"),
    )
).resolve()
OUTPUT_DIR = Path(
    os.environ.get(
        "GOLD_OUTPUT_DIR",
        str(TASK_ROOT / "gold_annotations"),
    )
).resolve()
EXPORTER = PROJECT_ROOT / "scripts" / "nbs" / "export_gold_protein_go_edges.py"


def run() -> int:
    if not EXPORTER.exists():
        raise FileNotFoundError(f"gold exporter not found: {EXPORTER}")
    command = [
        sys.executable,
        str(EXPORTER),
        "--metadata-file",
        str(METADATA_FILE),
        "--protein-registry",
        str(PROTEIN_REGISTRY),
        "--go-registry",
        str(GO_REGISTRY),
        "--output-dir",
        str(OUTPUT_DIR),
        "--task",
        TASK,
        "--mode",
        MODE,
        "--role",
        ROLE,
        "--name-key",
        NAME_KEY,
        "--label-key",
        LABEL_KEY,
        "--alignment-policy",
        ALIGNMENT_POLICY,
    ]
    if OVERWRITE:
        command.append("--overwrite")

    print("[NBS gold export]", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)

    edge_path = OUTPUT_DIR / "gold_protein_go_edge_index.i32.npy"
    print("\n[Done] GOLD_EDGE_INDEX:", edge_path)
    print("[Next] add gold dual indices without rebuilding candidate/pseudo:")
    print(
        "GOLD_EDGE_INDEX='{}' GOLD_ONLY=1 MERGE_EXISTING=1 "
        "python scripts/nbs/run_build_go_protein_inverted_index.py".format(edge_path)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
