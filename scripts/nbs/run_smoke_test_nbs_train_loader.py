#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get(
    "LATENCE_PROJECT_ROOT",
    "/home/dataset-local/data_local/shaojiangyi/latence-project",
)).resolve()
CONFIG = Path(os.environ.get(
    "NBS_TRAIN_CONFIG",
    ROOT / "nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.4.json",
))
OUTPUT = os.environ.get("NBS_LOADER_AUDIT_OUTPUT")


def main() -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts/nbs/smoke_test_nbs_train_loader.py"),
        "--project-root", str(ROOT),
        "--config", str(CONFIG),
    ]
    if OUTPUT:
        command.extend(["--output", OUTPUT])
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
