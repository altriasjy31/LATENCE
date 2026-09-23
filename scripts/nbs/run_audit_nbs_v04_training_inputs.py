#!/usr/bin/env python3
"""Environment-driven wrapper for the NBS v0.4 input auditor.

NBS v0.4.4 fix: TASK, RUN_TAG, and EPOCH are now forwarded to the
underlying auditor. A task-specific config is used when present; otherwise the
BP config is treated only as a shared hyperparameter template and all
 task-dependent paths are overridden at runtime.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

VERSION = "0.4.4"

ROOT = Path(
    os.environ.get(
        "LATENCE_PROJECT_ROOT",
        "/home/dataset-local/data_local/shaojiangyi/latence-project",
    )
).resolve()

TASK = os.environ.get("TASK", "bp").strip().lower()
RUN_TAG = os.environ.get("RUN_TAG", "").strip()
EPOCH = int(os.environ.get("EPOCH", "100"))

if TASK not in {"bp", "mf", "cc"}:
    raise ValueError(f"TASK must be bp, mf, or cc; got {TASK!r}")
if not RUN_TAG:
    if TASK == "bp":
        RUN_TAG = "bp_weak_detr_v3_expert_prob_warmstart340_to400"
    else:
        raise ValueError(
            f"RUN_TAG must be set when TASK={TASK}; refusing to reuse the BP run tag"
        )

_config_from_env = os.environ.get("NBS_TRAIN_CONFIG")
if _config_from_env:
    CONFIG = Path(_config_from_env)
else:
    task_config = ROOT / "nbs_models/nbs_protein_go/configs" / f"{TASK}_fixed_epoch_v0.4.json"
    CONFIG = (
        task_config
        if task_config.is_file()
        else ROOT / "nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.4.json"
    )


def main() -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts/nbs/audit_nbs_v04_training_inputs.py"),
        "--project-root",
        str(ROOT),
        "--config",
        str(CONFIG),
        "--task",
        TASK,
        "--run-tag",
        RUN_TAG,
        "--epoch",
        str(EPOCH),
    ]

    data_root = os.environ.get("NBS_TASK_ROOT")
    if data_root:
        command.extend(["--data-root", data_root])

    alignment = os.environ.get("NBS_GO_ALIGNMENT_MANIFEST")
    if alignment:
        command.extend(["--alignment-manifest", alignment])

    output = os.environ.get("NBS_AUDIT_OUTPUT")
    if output:
        command.extend(["--output", output])

    print(
        f"[NBS audit v{VERSION}] task={TASK} run_tag={RUN_TAG} epoch={EPOCH} "
        f"config_template={CONFIG}",
        flush=True,
        file=sys.stderr,
    )
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
