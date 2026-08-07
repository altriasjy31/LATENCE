#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(os.environ.get("LATENCE_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
    task = os.environ.get("TASK", "bp").strip().lower()
    if task not in {"bp", "mf", "cc"}:
        raise ValueError("TASK must be bp, mf or cc")
    nbs_prob = os.environ.get("NBS_IND_TEST_PROB", "").strip()
    if not nbs_prob:
        raise ValueError(
            "Set NBS_IND_TEST_PROB to a dense [ind_test, GO] NBS probability array. "
            "The NBS ind_test inference exporter is a separate preparation step."
        )
    output = os.environ.get("NBS_EVAL_OUTPUT_DIR", f"outputs/latence_nbs_eval/{task}")
    metadata = os.environ.get("METADATA_FILE", str(root / "data" / "unidata_with_exp_train_pseudo.pkl"))
    cmd = [
        sys.executable,
        str(root / "scripts" / "nbs" / "eval_nbs_ind_test_predictions.py"),
        "--task", task,
        "--metadata-file", metadata,
        "--nbs-prob", nbs_prob,
        "--output-dir", output,
    ]
    optional = {
        "BACKBONE_IND_TEST_PROB": "--backbone-prob",
        "MODELOUT_IND_TEST_PROB": "--modelout-prob",
        "NBS_IND_TEST_PROTEIN_IDS": "--protein-ids",
        "TRAIN_GO_COUNTS": "--train-counts",
    }
    for env_name, flag in optional.items():
        value = os.environ.get(env_name, "").strip()
        if value:
            cmd.extend([flag, value])
    if os.environ.get("NBS_AUPRC_MODE"):
        cmd.extend(["--auprc-mode", os.environ["NBS_AUPRC_MODE"]])
    if os.environ.get("NBS_THRESHOLD_STEP"):
        cmd.extend(["--threshold-step", os.environ["NBS_THRESHOLD_STEP"]])
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
