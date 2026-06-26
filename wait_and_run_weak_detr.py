#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wait until at least N GPUs have enough free memory, then launch training.

Default behavior:
- Check GPU 0,1,2,3
- Require 2 GPUs
- Require each selected GPU to have >= 70 GiB free memory
- Launch:
    torchrun --standalone --nproc_per_node=2 run_weak_exp_train_detr.py

Example:

python wait_and_run_weak_detr.py \
  --candidate-gpus 0,1,2,3 \
  --num-gpus 2 \
  --min-free-gb 70 \
  --poll-sec 60 \
  --stable-checks 2 \
  -- \
  torchrun --standalone --nproc_per_node=2 run_weak_exp_train_detr.py \
    --task bp \
    --your-other-args ...

If your script is not DDP-based, use:

python wait_and_run_weak_detr.py -- \
  python -u run_weak_exp_train_detr.py --your-other-args ...
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class GPUInfo:
    index: int
    free_mib: int
    used_mib: int
    total_mib: int

    @property
    def free_gib(self) -> float:
        return self.free_mib / 1024.0

    @property
    def used_gib(self) -> float:
        return self.used_mib / 1024.0

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024.0


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def parse_gpu_ids(s: str) -> Optional[set[int]]:
    s = s.strip()
    if not s or s.lower() == "all":
        return None

    ids: set[int] = set()
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        ids.add(int(x))
    return ids


def query_gpus() -> list[GPUInfo]:
    """
    Query GPU memory with nvidia-smi.

    nvidia-smi returns memory in MiB when using nounits.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]

    try:
        out = subprocess.check_output(cmd, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("Cannot find nvidia-smi. Please check NVIDIA driver installation.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi failed with exit code {e.returncode}") from e

    gpus: list[GPUInfo] = []

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue

        idx, free_mib, used_mib, total_mib = map(int, parts)
        gpus.append(
            GPUInfo(
                index=idx,
                free_mib=free_mib,
                used_mib=used_mib,
                total_mib=total_mib,
            )
        )

    return gpus


def format_gpu_status(gpus: list[GPUInfo]) -> str:
    chunks = []
    for g in sorted(gpus, key=lambda x: x.index):
        chunks.append(
            f"GPU{g.index}: free={g.free_gib:.1f}GiB, "
            f"used={g.used_gib:.1f}GiB, total={g.total_gib:.1f}GiB"
        )
    return " | ".join(chunks)


def select_gpus(
    gpus: list[GPUInfo],
    candidate_ids: Optional[set[int]],
    num_gpus: int,
    min_free_mib: int,
    prefer: str,
) -> list[GPUInfo]:
    if candidate_ids is not None:
        gpus = [g for g in gpus if g.index in candidate_ids]

    eligible = [g for g in gpus if g.free_mib >= min_free_mib]

    if prefer == "most_free":
        eligible.sort(key=lambda g: (-g.free_mib, g.index))
    elif prefer == "lowest_index":
        eligible.sort(key=lambda g: g.index)
    else:
        raise ValueError(f"Unknown prefer strategy: {prefer}")

    return eligible[:num_gpus]


def find_free_port() -> int:
    """
    Pick a free TCP port for torch distributed launch.

    This is not a hard reservation, but it avoids many MASTER_PORT conflicts.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = list(args.command)

    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if cmd:
        return cmd

    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.num_gpus}",
        "run_weak_exp_train_detr.py",
    ]


def run_training(
    command: list[str],
    selected_gpus: list[GPUInfo],
    auto_master_port: bool,
    dry_run: bool,
) -> int:
    physical_ids = [g.index for g in selected_gpus]
    visible_devices = ",".join(str(i) for i in physical_ids)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    if auto_master_port and "MASTER_PORT" not in env:
        env["MASTER_PORT"] = str(find_free_port())

    log(f"Selected physical GPUs: {physical_ids}")
    log(f"CUDA_VISIBLE_DEVICES={visible_devices}")
    if "MASTER_PORT" in env:
        log(f"MASTER_PORT={env['MASTER_PORT']}")
    log("Launch command:")
    log(" ".join(command))

    if dry_run:
        log("Dry run enabled. Exit without launching training.")
        return 0

    proc: Optional[subprocess.Popen] = None

    def forward_signal(signum, frame):
        if proc is not None and proc.poll() is None:
            log(f"Forward signal {signum} to child process.")
            proc.send_signal(signum)
        else:
            sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    proc = subprocess.Popen(command, env=env)
    return_code = proc.wait()

    log(f"Training process exited with code {return_code}")
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for enough free GPU memory, then launch run_weak_exp_train_detr.py."
    )

    parser.add_argument(
        "--candidate-gpus",
        type=str,
        default="0,1,2,3",
        help="Candidate physical GPU ids, e.g. 0,1,2,3. Use 'all' for all visible GPUs.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs required for training.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=70.0,
        help="Minimum free memory per selected GPU in GiB.",
    )
    parser.add_argument(
        "--poll-sec",
        type=int,
        default=60,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--stable-checks",
        type=int,
        default=2,
        help=(
            "Require the condition to be satisfied for this many consecutive checks "
            "before launching. Useful to avoid transient false positives."
        ),
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="most_free",
        choices=["most_free", "lowest_index"],
        help="GPU selection strategy.",
    )
    parser.add_argument(
        "--auto-master-port",
        action="store_true",
        default=True,
        help="Automatically set MASTER_PORT if it is not already set.",
    )
    parser.add_argument(
        "--no-auto-master-port",
        dest="auto_master_port",
        action="store_false",
        help="Do not set MASTER_PORT automatically.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected GPUs and command; do not launch training.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help=(
            "Training command after '--'. "
            "Example: -- torchrun --standalone --nproc_per_node=2 run_weak_exp_train_detr.py ..."
        ),
    )

    args = parser.parse_args()

    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive.")

    if args.poll_sec <= 0:
        raise ValueError("--poll-sec must be positive.")

    if args.stable_checks <= 0:
        raise ValueError("--stable-checks must be positive.")

    candidate_ids = parse_gpu_ids(args.candidate_gpus)
    min_free_mib = int(args.min_free_gb * 1024)
    command = build_command(args)

    log("Start GPU watcher.")
    log(f"Candidate GPUs: {args.candidate_gpus}")
    log(f"Need {args.num_gpus} GPU(s), each with >= {args.min_free_gb:.1f} GiB free memory.")
    log(f"Polling every {args.poll_sec} seconds.")
    log(f"Stable checks required: {args.stable_checks}")

    stable_count = 0

    while True:
        gpus = query_gpus()
        selected = select_gpus(
            gpus=gpus,
            candidate_ids=candidate_ids,
            num_gpus=args.num_gpus,
            min_free_mib=min_free_mib,
            prefer=args.prefer,
        )

        status = format_gpu_status(gpus)
        log(status)

        if len(selected) >= args.num_gpus:
            stable_count += 1
            selected_ids = [g.index for g in selected]
            log(
                f"Condition satisfied: selected GPUs {selected_ids} "
                f"({stable_count}/{args.stable_checks})."
            )
        else:
            stable_count = 0
            log("Condition not satisfied. Continue waiting.")

        if stable_count >= args.stable_checks:
            # Final immediate recheck before launch, reducing race condition.
            final_gpus = query_gpus()
            final_selected = select_gpus(
                gpus=final_gpus,
                candidate_ids=candidate_ids,
                num_gpus=args.num_gpus,
                min_free_mib=min_free_mib,
                prefer=args.prefer,
            )

            if len(final_selected) >= args.num_gpus:
                return run_training(
                    command=command,
                    selected_gpus=final_selected,
                    auto_master_port=args.auto_master_port,
                    dry_run=args.dry_run,
                )

            stable_count = 0
            log("Final recheck failed. Continue waiting.")

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())