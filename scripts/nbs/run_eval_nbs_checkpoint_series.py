#!/usr/bin/env python3
"""Run one fixed NBS independent-test protocol for a checkpoint series.

Required environment variables:

* ``NBS_CHECKPOINT_DIR``: directory containing ``nbs_epoch{epoch}.pt``;
* ``STAGE1_CHECKPOINT``: the exact Stage-1 checkpoint used by NBS inputs.

The existing end-to-end evaluator consumes the remaining Stage-1/metadata
variables.  All epochs share one validated inductive-input cache and receive
separate prediction/evaluation directories.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from summarize_nbs_checkpoint_series import parse_epochs


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def enabled(name: str, default: bool) -> bool:
    value = env(name, "1" if default else "0").lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be boolean")


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def main() -> None:
    root = Path(
        env("LATENCE_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))
    ).resolve()
    checkpoint_value = env("NBS_CHECKPOINT_DIR")
    if not checkpoint_value:
        raise ValueError("NBS_CHECKPOINT_DIR is required")
    checkpoint_dir = resolve(root, checkpoint_value)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(checkpoint_dir)
    if not env("STAGE1_CHECKPOINT"):
        raise ValueError("STAGE1_CHECKPOINT is required")

    task = env("TASK", "bp").lower()
    epochs_text = env("NBS_EVAL_EPOCHS", "1,2,3,4")
    epochs = parse_epochs(epochs_text)
    pattern = env("NBS_CHECKPOINT_PATTERN", "nbs_epoch{epoch}.pt")
    series_root = resolve(
        root,
        env(
            "NBS_EVAL_SERIES_ROOT",
            f"outputs/latence_nbs_eval/{task}_nbs_checkpoint_series",
        ),
    )
    series_root.mkdir(parents=True, exist_ok=True)
    shared_input_dir = resolve(
        root,
        env("NBS_IND_TEST_WORK_DIR", str(series_root / "_shared_inductive_inputs")),
    )
    shared_input_dir.mkdir(parents=True, exist_ok=True)

    config_value = env("NBS_TRAIN_CONFIG")
    if config_value:
        config_path = resolve(root, config_value)
    elif (checkpoint_dir / "resolved_config.json").is_file():
        config_path = (checkpoint_dir / "resolved_config.json").resolve()
    else:
        config_path = resolve(
            root,
            f"nbs_models/nbs_protein_go/configs/{task}_fixed_epoch_v0.6.0_perfopt.json",
        )
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    checkpoints: dict[int, Path] = {}
    for epoch in epochs:
        checkpoint = checkpoint_dir / pattern.format(epoch=epoch)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoints[epoch] = checkpoint.resolve()

    protocol = {
        "schema_version": 1,
        "task": task,
        "epochs": list(epochs),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_pattern": pattern,
        "config": str(config_path),
        "shared_inductive_inputs": str(shared_input_dir),
        "stage1_checkpoint": str(resolve(root, env("STAGE1_CHECKPOINT"))),
        "external_pp": enabled("NBS_EVAL_USE_EXTERNAL_PP", True),
        "candidate_evidence": enabled("NBS_EVAL_USE_CANDIDATE_EVIDENCE", True),
        "metric_backend": env("NBS_METRIC_BACKEND", "stage1"),
        "note": "epochs are reported as a predeclared series; ind_test does not select a best checkpoint",
    }
    protocol_path = series_root / "nbs_epoch_series_protocol.json"
    if protocol_path.is_file():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != protocol and not enabled("NBS_EVAL_ALLOW_PROTOCOL_CHANGE", False):
            raise RuntimeError(
                f"evaluation protocol changed since {protocol_path}; use a new "
                "NBS_EVAL_SERIES_ROOT or explicitly set NBS_EVAL_ALLOW_PROTOCOL_CHANGE=1"
            )
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    runner = root / "scripts" / "nbs" / "run_eval_nbs_ind_test_predictions.py"
    completed: list[int] = []
    skip_completed = enabled("NBS_EVAL_SKIP_COMPLETED", True)
    for epoch in epochs:
        epoch_dir = series_root / f"epoch{epoch}"
        prediction_dir = epoch_dir / "predictions"
        metrics_path = epoch_dir / "nbs_ind_test_metrics.json"
        if skip_completed and metrics_path.is_file():
            print(f"[NBS series] epoch={epoch} status=already_complete", flush=True)
            completed.append(epoch)
            continue
        epoch_dir.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        child_env.update(
            {
                "LATENCE_PROJECT_ROOT": str(root),
                "TASK": task,
                "NBS_TRAIN_CONFIG": str(config_path),
                "NBS_CHECKPOINT": str(checkpoints[epoch]),
                "NBS_EVAL_OUTPUT_DIR": str(epoch_dir),
                "NBS_PRED_OUTPUT_DIR": str(prediction_dir),
                "NBS_IND_TEST_WORK_DIR": str(shared_input_dir),
                "NBS_USE_TMP_WORKSPACE": "0",
                "NBS_RUN_EVALUATION": "1",
                "NBS_SAVE_INFERENCE_DIAGNOSTICS": "1",
                "NBS_EVAL_USE_EXTERNAL_PP": env("NBS_EVAL_USE_EXTERNAL_PP", "1"),
                "NBS_EVAL_USE_CANDIDATE_EVIDENCE": env(
                    "NBS_EVAL_USE_CANDIDATE_EVIDENCE", "1"
                ),
                "NBS_METRIC_BACKEND": env("NBS_METRIC_BACKEND", "stage1"),
                # Build/reuse once; every later epoch must consume the exact cache.
                "NBS_INPUT_CACHE_POLICY": (
                    "require"
                    if (shared_input_dir / "ind_test_input_manifest.json").is_file()
                    else env("NBS_INPUT_CACHE_POLICY", "reuse")
                ),
            }
        )
        child_env.pop("NBS_IND_TEST_PROB", None)
        child_env.pop("NBS_EVAL_LIMIT_PROTEINS", None)
        print(
            f"[NBS series] epoch={epoch} checkpoint={checkpoints[epoch]} "
            f"output={epoch_dir}",
            flush=True,
        )
        subprocess.run([sys.executable, str(runner)], cwd=root, env=child_env, check=True)
        if not metrics_path.is_file():
            raise RuntimeError(f"evaluation finished without {metrics_path}")
        completed.append(epoch)

    summarizer = root / "scripts" / "nbs" / "summarize_nbs_checkpoint_series.py"
    subprocess.run(
        [
            sys.executable,
            str(summarizer),
            "--series-root",
            str(series_root),
            "--epochs",
            epochs_text,
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--checkpoint-pattern",
            pattern,
        ],
        cwd=root,
        check=True,
    )
    print(
        f"[NBS series complete] epochs={completed} summary="
        f"{series_root / 'nbs_epoch_series_primary.tsv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
