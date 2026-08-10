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
            / "bp_fixed_epoch_v0.5.6.json",
        )
    )
    train_script = project_root / "scripts" / "nbs" / "train_nbs_fixed_epochs.py"
    num_gpus = int(os.environ.get("NBS_NUM_GPUS", "1"))
    applied_overrides: list[str] = []
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
        applied_overrides.append(f"NBS_COMPONENT_FACTORY={factory}")
    if os.environ.get("NBS_SAVE_INTERVAL_EPOCHS"):
        value = os.environ["NBS_SAVE_INTERVAL_EPOCHS"]
        command.extend(["--save-interval-epochs", value])
        applied_overrides.append(f"NBS_SAVE_INTERVAL_EPOCHS={value}")
    if os.environ.get("NBS_EPOCHS"):
        value = os.environ["NBS_EPOCHS"]
        command.extend(["--epochs", value])
        applied_overrides.append(f"NBS_EPOCHS={value}")
    if os.environ.get("NBS_SAVE_EPOCHS"):
        value = os.environ["NBS_SAVE_EPOCHS"]
        command.extend(["--save-epochs", value])
        applied_overrides.append(f"NBS_SAVE_EPOCHS={value}")
    if os.environ.get("NBS_MAX_STEPS_PER_EPOCH"):
        value = os.environ["NBS_MAX_STEPS_PER_EPOCH"]
        command.extend(["--max-steps-per-epoch", value])
        applied_overrides.append(f"NBS_MAX_STEPS_PER_EPOCH={value}")
    if os.environ.get("NBS_LOG_INTERVAL"):
        value = os.environ["NBS_LOG_INTERVAL"]
        command.extend(["--log-interval", value])
        applied_overrides.append(f"NBS_LOG_INTERVAL={value}")
    if os.environ.get("NBS_OUTPUT_DIR"):
        value = os.environ["NBS_OUTPUT_DIR"]
        command.extend(["--output-dir", value])
        applied_overrides.append(f"NBS_OUTPUT_DIR={value}")
    env_arg_map = {
        "NBS_NUM_QUERIES": "--num-queries",
        "NBS_MAX_CANDIDATES": "--max-candidates",
        "NBS_HARD_CANDIDATE_PER_QUERY": "--hard-candidate-per-query",
        "NBS_BACKGROUND_UNLABELLED_PER_QUERY": "--background-unlabelled-per-query",
        "NBS_BACKGROUND_UNLABELLED_WEIGHT": "--background-unlabelled-weight",
        "NBS_BACKGROUND_BASE_PROBABILITY_MAX": "--background-base-probability-max",
        "NBS_PSEUDO_POSITIVE_PER_QUERY": "--pseudo-positive-per-query",
        "NBS_PSEUDO_SAMPLING_MODE": "--pseudo-sampling-mode",
        "NBS_WEAK_FOCUS_QUERIES": "--weak-focus-queries",
        "NBS_WEAK_FOCUS_TARGETS": "--weak-focus-targets",
        "NBS_WEAK_FOCUS_SCAN_LIMIT": "--weak-focus-scan-limit",
        "NBS_WEAK_FOCUS_SPECIFICITY_POWER": "--weak-focus-specificity-power",
        "NBS_WEAK_FOCUS_MIN_PROBABILITY": "--weak-focus-min-probability",
        "NBS_SUPPORT_PER_QUERY": "--support-per-query",
        "NBS_GOLD_POSITIVE_PER_QUERY": "--gold-positive-per-query",
        "NBS_HIERARCHY_PAIRS_PER_EPISODE": "--hierarchy-pairs-per-episode",
        "NBS_QUERY_SAMPLING_MODE": "--query-sampling-mode",
        "NBS_GOLD_SUPPORT_POLICY": "--gold-support-policy",
        "NBS_SINGLETON_REQUIRES_PSEUDO": "--singleton-requires-pseudo",
        "NBS_CANDIDATE_MESSAGE_TOPK": "--candidate-message-topk",
        "NBS_PSEUDO_MESSAGE_TOPK": "--pseudo-message-topk",
        "NBS_STEPS_PER_EPOCH_PER_RANK": "--steps-per-epoch-per-rank",
        "NBS_COVERAGE_CYCLES_PER_EPOCH": "--coverage-cycles-per-epoch",
        "NBS_EPOCH_UNIT": "--epoch-unit",
        "NBS_WEAK_PSEUDO_PASSES_PER_EPOCH": "--weak-pseudo-passes-per-epoch",
        "NBS_CORE_GOLD_PASSES_PER_EPOCH": "--core-gold-passes-per-epoch",
        "NBS_WEAK_UNIQUE_COVERAGE_TARGET": "--weak-unique-coverage-target",
        "NBS_WEAK_FOCUS_PLANNING_EFFICIENCY": "--weak-focus-planning-efficiency",
        "NBS_PROGRESS_BAR": "--progress-bar",
        "NBS_SCHEDULER": "--scheduler-name",
        "NBS_ONECYCLE_PCT_START": "--onecycle-pct-start",
        "NBS_ONECYCLE_DIV_FACTOR": "--onecycle-div-factor",
        "NBS_ONECYCLE_FINAL_DIV_FACTOR": "--onecycle-final-div-factor",
        "NBS_ONECYCLE_ANNEAL_STRATEGY": "--onecycle-anneal-strategy",
        "NBS_ONECYCLE_THREE_PHASE": "--onecycle-three-phase",
        "NBS_ONECYCLE_CYCLE_MOMENTUM": "--onecycle-cycle-momentum",
    }
    for env_name, flag in env_arg_map.items():
        value = os.environ.get(env_name)
        if value not in (None, ""):
            command.extend([flag, value])
            applied_overrides.append(f"{env_name}={value}")
    fresh_start = os.environ.get("NBS_FRESH_START", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if fresh_start:
        command.append("--fresh-start")
        applied_overrides.append("NBS_FRESH_START=1")
        stale_resume = os.environ.get("NBS_RESUME")
        if stale_resume:
            print(
                f"[NBS launcher] NBS_FRESH_START=1: ignoring stale NBS_RESUME={stale_resume}",
                flush=True,
            )
    elif os.environ.get("NBS_RESUME"):
        value = os.environ["NBS_RESUME"]
        command.extend(["--resume", value])
        applied_overrides.append(f"NBS_RESUME={value}")
    if num_gpus == 1 and os.environ.get("NBS_DEVICE"):
        value = os.environ["NBS_DEVICE"]
        command.extend(["--device", value])
        applied_overrides.append(f"NBS_DEVICE={value}")
    if os.environ.get("NBS_VALIDATE_CONFIG_ONLY", "0") == "1":
        command.append("--validate-config-only")
        applied_overrides.append("NBS_VALIDATE_CONFIG_ONLY=1")
    print(
        f"[NBS launcher] config={config} num_gpus={num_gpus}",
        flush=True,
    )
    if applied_overrides:
        print(
            "[NBS launcher overrides] " + ", ".join(applied_overrides),
            flush=True,
        )
    subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
