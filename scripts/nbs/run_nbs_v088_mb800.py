#!/usr/bin/env python3
"""Run the isolated v088 single-GPU, 800-protein experiment (no torchrun)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TOOLING_VERSION = "0.8.8.2"
MODEL_VERSION = "0.8.8"
CONFIG_REL = Path("nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.8_mb800_single.json")
RUN_ROOT_REL = Path("outputs/latence_nbs_experiments/v088_mb800_single")
RUNNER_REL = Path("scripts/nbs/train_nbs_full_task_v088.py")
MARKER = "mb800_launcher.json"
LOCK = ".mb800_launcher.lock"
STEPS_PER_EPOCH = 765
V088_EPOCH1_TRAINING_SECONDS = 127899.59018327296
DEFAULT_INPUT = "outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2"
DEFAULT_METADATA = "/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl"


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_config(path):
    if path != (ROOT / CONFIG_REL).resolve():
        raise ValueError(f"Use the isolated mb800 config: {ROOT / CONFIG_REL}")
    config = read_json(path)
    cfg = config.get("full_task", {})
    expected = {"weak_batch": 640, "core_batch": 160, "accumulation_steps": 1,
                "epochs": 5, "epoch_unit": "weak", "learning_rate": .006,
                "experiment_preset": "hetero_tuned_mb800_single", "log_every": 5}
    for key, value in expected.items():
        if cfg.get(key) != value or isinstance(cfg.get(key), bool):
            raise ValueError(f"mb800 requires full_task.{key}={value!r}")
    if config.get("release_version") != MODEL_VERSION:
        raise ValueError("mb800 keeps model/loader release_version=0.8.8")
    if config.get("stage", {}).get("name") != "nbs_v088_hetero_tuned":
        raise ValueError("mb800 requires the unchanged nbs_v088_hetero_tuned stage")
    if cfg.get("sampler", {}).get("max_nodes") != 1800000:
        raise ValueError("mb800 requires sampler.max_nodes=1800000")
    scheduler = cfg.get("scheduler", {})
    for key, value in {"name": "epoch_warmup_hold_cosine", "min_lr": .0006,
                       "warmup_epochs": .1, "hold_until_epoch": 3.}.items():
        if scheduler.get(key) != value:
            raise ValueError(f"mb800 requires full_task.scheduler.{key}={value!r}")
    if cfg.get("model", {}).get("encoder_variant") != "hetero_tuned" or cfg.get("model", {}).get("prediction_mode") != "direct":
        raise ValueError("mb800 requires the unchanged hetero_tuned direct model")
    if "runtime" in config:
        raise ValueError("Pass the shipped config, not a resolved runtime config")
    return config


def child_environment(gpu, threads, inherited=None):
    env = dict(os.environ if inherited is None else inherited)
    distributed = {"WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                   "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                   "NODE_RANK", "NNODES"}
    for key in list(env):
        if key in distributed or key.startswith(("TORCHELASTIC_", "TORCHRUN_", "OMPI_", "PMI_", "PMIX_")):
            del env[key]
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               WORLD_SIZE="1", RANK="0", LOCAL_RANK="0",
               OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads))
    return env


def expected_marker(config, config_path, workdir):
    return {"tooling_version": TOOLING_VERSION, "model_version": MODEL_VERSION,
            "config_path": str(config_path), "config_sha256": config_digest(config),
            "work_dir": str(workdir), "world_size": 1,
            "global_weak_batch": 640, "global_core_batch": 160, "accumulation_steps": 1}


def validate_workdir(workdir, marker, config):
    namespace = (ROOT / RUN_ROOT_REL).resolve()
    if not workdir.is_relative_to(namespace) or workdir == namespace:
        raise ValueError(f"Work directory must be inside the new run namespace: {namespace}")
    if not workdir.exists():
        return
    if not workdir.is_dir():
        raise ValueError("Work directory is not a directory")
    # A symlinked checkpoint must never turn the automatic resume into an old run.
    for path in workdir.iterdir():
        if path.is_symlink() and not path.resolve().is_relative_to(workdir):
            raise ValueError(f"Work directory contains an external symlink: {path.name}")
    marker_path = workdir / MARKER
    if marker_path.exists():
        if read_json(marker_path) != marker:
            raise ValueError("Existing run marker does not match this config and work directory")
    elif any(path.name != LOCK for path in workdir.iterdir()):
        raise ValueError("Existing nonempty directory lacks the mb800 run marker; refusing to adopt it")
    resolved = workdir / "resolved_config.json"
    if resolved.exists():
        saved = read_json(resolved)
        runtime = saved.pop("runtime", {})
        if saved != config:
            raise ValueError("Existing resolved_config.json differs from the mb800 config")
        for key, value in {"world_size": 1, "global_weak_batch": 640,
                           "global_core_batch": 160, "accumulation_steps": 1,
                           "steps_per_epoch": STEPS_PER_EPOCH}.items():
            if runtime.get(key) != value:
                raise ValueError(f"Existing runtime {key} differs from this single-GPU experiment")


@contextmanager
def run_lock(workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    with (workdir / LOCK).open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another launcher is using {workdir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_history(workdir):
    path = workdir / "training_history.json"
    if not path.exists():
        return []
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("training_history.json must contain a list")
    return rows


def status(workdir):
    rows = load_history(workdir)
    result = {"work_dir": str(workdir), "tooling_version": TOOLING_VERSION,
              "model_version": MODEL_VERSION, "step": 0, "steps_per_epoch": STEPS_PER_EPOCH,
              "measured_steps": 0, "seconds_per_step": None,
              "rank0_peak_allocated_gb": None,
              "estimated_epoch_speedup_vs_v088_epoch1": None,
              "speedup_scope": "Extrapolated from the currently measured seconds/update and 765 updates/epoch; "
                               "not an observed complete-epoch speedup. Reference v088 epoch1 training time: "
                               "127899.59018327296 seconds.",
              "timing_scope": "Training windows only; excludes model/data loading, checkpoint saving and evaluation. "
                              "First 10 optimizer updates excluded; at most the latest 30 updates. "
                              "Elapsed counters can reset on resume and are not subtracted."}
    if not rows:
        result["state"] = "no training history"
        return result
    windows = []
    previous_step = 0
    for row in rows:
        step = int(row["step"])
        if step <= previous_step:
            raise ValueError("History steps must be strictly increasing")
        duration = float(row["seconds_per_step"])
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("History contains invalid seconds_per_step")
        windows.append((previous_step, step, duration))
        previous_step = step
    last = rows[-1]
    steps_per_epoch = int(last.get("steps_per_epoch", STEPS_PER_EPOCH))
    if steps_per_epoch != STEPS_PER_EPOCH:
        raise ValueError("History epoch length differs from the expected 765 updates")
    # Use each runner-reported window average, weighted by its actual update count.
    # The resumed first window may be shorter than log_every; step differences
    # capture this while cumulative elapsed_seconds resets to zero in the runner.
    start = max(10, previous_step - 30)
    seconds = 0.0
    measured = 0
    for left, right, duration in windows:
        # Do not partially reuse a window that contains excluded warmup updates.
        if left < 10:
            continue
        count = max(0, right - max(left, start))
        measured += count
        seconds += count * duration
    peaks = [float(row["peak_allocated_gb"]) for row in rows if "peak_allocated_gb" in row]
    result.update(state="recorded", step=previous_step,
                  weak_epoch=previous_step / steps_per_epoch,
                  measured_steps=measured,
                  rank0_peak_allocated_gb=max(peaks) if peaks else None)
    if measured:
        sec = seconds / measured
        result.update(seconds_per_step=sec,
                      estimated_epoch_speedup_vs_v088_epoch1=(
                          V088_EPOCH1_TRAINING_SECONDS / (sec * steps_per_epoch) if sec > 0 else None),
                      training_hours_per_epoch=sec * steps_per_epoch / 3600,
                      estimated_training_hours_for_five_epochs=sec * steps_per_epoch * 5 / 3600,
                      remaining_training_hours_to_epoch1=sec * max(0, steps_per_epoch - previous_step) / 3600,
                      remaining_training_hours_to_epoch5=sec * max(0, steps_per_epoch * 5 - previous_step) / 3600)
    return result


def make_commands(args, config_path, workdir):
    runner = ROOT / RUNNER_REL
    if not runner.is_file():
        raise FileNotFoundError(f"Original v088 runner is missing: {runner}")
    common = [sys.executable, str(runner), "--config", str(config_path),
              "--work-dir", str(workdir), "--device", "cuda:0"]
    latest = workdir / "latest.pt"
    history = workdir / "training_history.json"
    if args.action == "probe":
        if latest.exists() or history.exists() or any(workdir.glob("*.pt")):
            raise ValueError("probe requires a fresh run; use train to resume this directory")
        endpoint = ["--stop-step", str(args.probe_steps)]
    elif args.action == "train":
        target = workdir / f"nbs_epoch{args.stop_epoch}.pt"
        if target.exists():
            return []
        if history.exists() and not latest.is_file():
            raise ValueError("Training history exists without latest.pt; refusing to restart over this run")
        if latest.exists() and not (workdir / "resolved_config.json").is_file():
            raise ValueError("Resume requires the matching resolved_config.json in the same directory")
        endpoint = ["--stop-epoch", str(args.stop_epoch)]
        if latest.is_file():
            endpoint.extend(["--resume", str(latest)])
    else:
        checkpoint = workdir / f"nbs_epoch{args.epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Completed-epoch checkpoint is missing: {checkpoint}")
        return [common + ["--stage", "evaluate", "--checkpoint", str(checkpoint),
                          "--input-dir", str(project_path(args.input_dir)),
                          "--metadata-file", str(project_path(args.metadata_file)),
                          "--metric-backend", "stage1", "--auprc-mode", "exact",
                          "--allow-missing-references"]]
    return [common + ["--stage", "prepare", "--backend", "torch"],
            common + ["--stage", "train"] + endpoint
            + ["--input-dir", str(project_path(args.input_dir)),
               "--metadata-file", str(project_path(args.metadata_file))]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "train", "evaluate", "status"))
    parser.add_argument("--config", type=Path, default=CONFIG_REL)
    parser.add_argument("--work-dir", type=Path, default=RUN_ROOT_REL / "hetero_tuned")
    parser.add_argument("--gpu", "--physical-gpu", dest="physical_gpu", type=int, default=1,
                        help="One physical GPU index; no comma-separated lists")
    parser.add_argument("--threads", type=int, default=1, help="OMP, MKL and OpenBLAS thread count")
    parser.add_argument("--probe-steps", type=int, choices=(20, 100),
                        help="probe only: default 20; 100 also covers the 77-update LR warmup")
    parser.add_argument("--stop-epoch", type=int, choices=(1, 5), help="train endpoint, default 1; five-epoch schedule is unchanged")
    parser.add_argument("--epoch", type=int, choices=(1, 5), help="evaluate checkpoint, default 1")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--metadata-file", default=DEFAULT_METADATA)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without creating files")
    args = parser.parse_args(argv)
    if args.physical_gpu < 0 or args.threads < 1:
        parser.error("physical-gpu must be nonnegative and threads must be positive")
    if args.stop_epoch is not None and args.action != "train":
        parser.error("--stop-epoch is only valid for train")
    if args.epoch is not None and args.action != "evaluate":
        parser.error("--epoch is only valid for evaluate")
    if args.probe_steps is not None and args.action != "probe":
        parser.error("--probe-steps is only valid for probe")
    args.stop_epoch = args.stop_epoch or 1
    args.epoch = args.epoch or 1
    args.probe_steps = args.probe_steps or 20
    try:
        config_path, workdir = project_path(args.config), project_path(args.work_dir)
        config = validate_config(config_path)
        marker = expected_marker(config, config_path, workdir)
        validate_workdir(workdir, marker, config)
        if args.action == "status":
            print(json.dumps(status(workdir), indent=2))
            return 0
        env = child_environment(args.physical_gpu, args.threads)
        commands = make_commands(args, config_path, workdir)
        if args.dry_run:
            print(json.dumps({"cwd": str(ROOT), "environment": {name: env[name] for name in
                ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "WORLD_SIZE", "RANK", "LOCAL_RANK", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}, "commands": commands,
                "state": "target epoch already present" if not commands else "dry-run"}, indent=2))
            return 0
        with run_lock(workdir):
            # Recheck after locking: another process may have finished since planning.
            validate_workdir(workdir, marker, config)
            commands = make_commands(args, config_path, workdir)
            if not (workdir / MARKER).exists():
                (workdir / MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
            if not commands:
                print(f"Target epoch {args.stop_epoch} already exists; no training launched.")
            for command in commands:
                print("+ " + shlex.join(command), flush=True)
                subprocess.run(command, cwd=ROOT, env=env, check=True, shell=False)
        if args.action in ("probe", "train"):
            print(json.dumps(status(workdir), indent=2))
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        print(f"v088 child process failed with exit code {exc.returncode}; no other run was resumed.", file=sys.stderr)
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
