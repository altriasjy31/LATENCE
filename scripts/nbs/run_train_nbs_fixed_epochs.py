#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(
        os.environ.get("LATENCE_PROJECT_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    config = Path(
        os.environ.get(
            "NBS_TRAIN_CONFIG",
            project_root
            / "nbs_models"
            / "nbs_protein_go"
            / "configs"
            / "bp_fixed_epoch_v0.4.json",
        )
    )
    train_script = project_root / "scripts" / "nbs" / "train_nbs_fixed_epochs.py"
    num_gpus = int(os.environ.get("NBS_NUM_GPUS", "1"))
    if num_gpus > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            str(train_script),
            "--config",
            str(config),
        ]
    else:
        command = [sys.executable, str(train_script), "--config", str(config)]
    factory = os.environ.get("NBS_COMPONENT_FACTORY", "")
    if factory:
        command.extend(["--component-factory", factory])
    if os.environ.get("NBS_SAVE_INTERVAL_EPOCHS"):
        command.extend(["--save-interval-epochs", os.environ["NBS_SAVE_INTERVAL_EPOCHS"]])
    if os.environ.get("NBS_EPOCHS"):
        command.extend(["--epochs", os.environ["NBS_EPOCHS"]])
    if os.environ.get("NBS_SAVE_EPOCHS"):
        command.extend(["--save-epochs", os.environ["NBS_SAVE_EPOCHS"]])
    if os.environ.get("NBS_MAX_STEPS_PER_EPOCH"):
        command.extend(["--max-steps-per-epoch", os.environ["NBS_MAX_STEPS_PER_EPOCH"]])
    if os.environ.get("NBS_LOG_INTERVAL"):
        command.extend(["--log-interval", os.environ["NBS_LOG_INTERVAL"]])
    if os.environ.get("NBS_OUTPUT_DIR"):
        command.extend(["--output-dir", os.environ["NBS_OUTPUT_DIR"]])
    if os.environ.get("NBS_RESUME"):
        command.extend(["--resume", os.environ["NBS_RESUME"]])
    if num_gpus == 1 and os.environ.get("NBS_DEVICE"):
        command.extend(["--device", os.environ["NBS_DEVICE"]])
    if os.environ.get("NBS_VALIDATE_CONFIG_ONLY", "0") == "1":
        command.append("--validate-config-only")
    subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
