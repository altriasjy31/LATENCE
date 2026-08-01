#!/usr/bin/env python3
"""
Run build_msa_binary.py for Swiss-Prot and/or independent-test MSA datasets.

Configuration priority:
    command-line arguments > environment variables > built-in defaults

Examples
--------
1. Process Swiss-Prot with default settings:

    python run_build_msa_binary.py --dataset sprot

2. Process the independent-test dataset:

    python run_build_msa_binary.py --dataset ind

3. Process both datasets:

    python run_build_msa_binary.py --dataset all

4. Configure through environment variables:

    export SPROT_MSA_DIR=/path/to/sprot_MSA
    export SPROT_BIN_DIR=/path/to/sprot_MSA_bin
    export IND_MSA_DIR=/path/to/ind_MSA
    export IND_BIN_DIR=/path/to/ind_MSA_bin
    export MSA_NUM_WORKERS=16

    python run_build_msa_binary.py --dataset all

5. Preview commands without executing:

    python run_build_msa_binary.py --dataset all --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_PROJECT_DIR = Path(
    "/home/dataset-local/data_local/shaojiangyi/latence-project"
)

DEFAULT_SPROT_MSA_DIR = DEFAULT_PROJECT_DIR / "data/sprot_2204_MSA"
DEFAULT_SPROT_BIN_DIR = DEFAULT_PROJECT_DIR / "data/sprot_2204_MSA_bin"

DEFAULT_IND_MSA_DIR = DEFAULT_PROJECT_DIR / "data/ind_MSA"
DEFAULT_IND_BIN_DIR = DEFAULT_PROJECT_DIR / "data/ind_MSA_bin"

DEFAULT_BUILD_SCRIPT = Path(__file__).resolve().parent / "build_msa_binary.py"


@dataclass(frozen=True)
class DatasetConfig:
    """Input and output paths for one MSA dataset."""

    name: str
    input_dir: Path
    output_dir: Path


def env_bool(
    name: str,
    default: bool,
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Read a boolean environment variable."""

    value = environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()

    true_values = {"1", "true", "yes", "y", "on"}
    false_values = {"0", "false", "no", "n", "off"}

    if normalized in true_values:
        return True
    if normalized in false_values:
        return False

    raise ValueError(
        f"Invalid boolean value for environment variable {name}: {value!r}. "
        f"Expected one of {sorted(true_values | false_values)}."
    )


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable."""

    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {value!r}."
        ) from exc


def env_float(name: str, default: float) -> float:
    """Read a floating-point environment variable."""

    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be a number, got {value!r}."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run build_msa_binary.py for Swiss-Prot, the independent-test "
            "dataset, or both datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        choices=("sprot", "ind", "all"),
        default=os.getenv("MSA_DATASET", "sprot"),
        help=(
            "Dataset to process. Can also be set through the "
            "MSA_DATASET environment variable."
        ),
    )

    parser.add_argument(
        "--build-script",
        type=Path,
        default=Path(
            os.getenv(
                "BUILD_MSA_BINARY_SCRIPT",
                str(DEFAULT_BUILD_SCRIPT),
            )
        ),
        help=(
            "Path to build_msa_binary.py. Environment variable: "
            "BUILD_MSA_BINARY_SCRIPT."
        ),
    )

    parser.add_argument(
        "--python-executable",
        default=os.getenv("MSA_PYTHON_EXECUTABLE", sys.executable),
        help=(
            "Python executable used to invoke build_msa_binary.py. "
            "Environment variable: MSA_PYTHON_EXECUTABLE."
        ),
    )

    # Dataset paths
    parser.add_argument(
        "--sprot-msa-dir",
        type=Path,
        default=Path(
            os.getenv("SPROT_MSA_DIR", str(DEFAULT_SPROT_MSA_DIR))
        ),
        help="Swiss-Prot MSA directory. Environment variable: SPROT_MSA_DIR.",
    )

    parser.add_argument(
        "--sprot-bin-dir",
        type=Path,
        default=Path(
            os.getenv("SPROT_BIN_DIR", str(DEFAULT_SPROT_BIN_DIR))
        ),
        help=(
            "Swiss-Prot binary output directory. "
            "Environment variable: SPROT_BIN_DIR."
        ),
    )

    parser.add_argument(
        "--ind-msa-dir",
        type=Path,
        default=Path(
            os.getenv("IND_MSA_DIR", str(DEFAULT_IND_MSA_DIR))
        ),
        help=(
            "Independent-test MSA directory. "
            "Environment variable: IND_MSA_DIR."
        ),
    )

    parser.add_argument(
        "--ind-bin-dir",
        type=Path,
        default=Path(
            os.getenv("IND_BIN_DIR", str(DEFAULT_IND_BIN_DIR))
        ),
        help=(
            "Independent-test binary output directory. "
            "Environment variable: IND_BIN_DIR."
        ),
    )

    # build_msa_binary.py arguments
    parser.add_argument(
        "--msa-format",
        default=os.getenv("MSA_FORMAT", "a3m"),
        help="MSA format. Environment variable: MSA_FORMAT.",
    )

    parser.add_argument(
        "--max-msa-size",
        type=int,
        default=env_int("MSA_MAX_MSA_SIZE", -1),
        help=(
            "Maximum number of MSA rows; -1 means no limit. "
            "Environment variable: MSA_MAX_MSA_SIZE."
        ),
    )

    parser.add_argument(
        "--store-max-len",
        type=int,
        default=env_int("MSA_STORE_MAX_LEN", 2048),
        help=(
            "Maximum stored sequence length. "
            "Environment variable: MSA_STORE_MAX_LEN."
        ),
    )

    parser.add_argument(
        "--max-shard-gb",
        type=float,
        default=env_float("MSA_MAX_SHARD_GB", 4.0),
        help=(
            "Maximum shard size in GB. "
            "Environment variable: MSA_MAX_SHARD_GB."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=env_int("MSA_NUM_WORKERS", 8),
        help=(
            "Number of worker processes. "
            "Environment variable: MSA_NUM_WORKERS."
        ),
    )

    parser.set_defaults(
        shuffle_rows=env_bool("MSA_SHUFFLE_ROWS", True),
        overwrite=env_bool("MSA_OVERWRITE", True),
    )

    parser.add_argument(
        "--shuffle-rows",
        dest="shuffle_rows",
        action="store_true",
        help="Shuffle MSA rows before storage.",
    )

    parser.add_argument(
        "--no-shuffle-rows",
        dest="shuffle_rows",
        action="store_false",
        help="Do not shuffle MSA rows.",
    )

    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing output.",
    )

    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing output.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=env_bool("MSA_DRY_RUN", False),
        help=(
            "Print commands without running them. "
            "Environment variable: MSA_DRY_RUN."
        ),
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate global command-line arguments."""

    build_script = args.build_script.expanduser().resolve()

    if not build_script.is_file():
        raise FileNotFoundError(
            f"build_msa_binary.py was not found: {build_script}"
        )

    if args.num_workers <= 0:
        raise ValueError(
            f"--num-workers must be greater than 0, got {args.num_workers}."
        )

    if args.store_max_len <= 0:
        raise ValueError(
            "--store-max-len must be greater than 0, "
            f"got {args.store_max_len}."
        )

    if args.max_msa_size == 0 or args.max_msa_size < -1:
        raise ValueError(
            "--max-msa-size must be -1 or a positive integer, "
            f"got {args.max_msa_size}."
        )

    if args.max_shard_gb <= 0:
        raise ValueError(
            "--max-shard-gb must be greater than 0, "
            f"got {args.max_shard_gb}."
        )


def get_datasets(args: argparse.Namespace) -> list[DatasetConfig]:
    """Resolve the requested dataset configurations."""

    sprot = DatasetConfig(
        name="Swiss-Prot",
        input_dir=args.sprot_msa_dir.expanduser().resolve(),
        output_dir=args.sprot_bin_dir.expanduser().resolve(),
    )

    ind = DatasetConfig(
        name="Independent test",
        input_dir=args.ind_msa_dir.expanduser().resolve(),
        output_dir=args.ind_bin_dir.expanduser().resolve(),
    )

    if args.dataset == "sprot":
        return [sprot]
    if args.dataset == "ind":
        return [ind]
    return [sprot, ind]


def validate_dataset(dataset: DatasetConfig) -> None:
    """Validate paths for one dataset."""

    if not dataset.input_dir.is_dir():
        raise NotADirectoryError(
            f"{dataset.name} input directory does not exist: "
            f"{dataset.input_dir}"
        )

    if dataset.input_dir == dataset.output_dir:
        raise ValueError(
            f"{dataset.name} input and output directories must differ: "
            f"{dataset.input_dir}"
        )

    dataset.output_dir.parent.mkdir(parents=True, exist_ok=True)


def build_command(
    args: argparse.Namespace,
    dataset: DatasetConfig,
) -> list[str]:
    """Build the build_msa_binary.py command."""

    command = [
        args.python_executable,
        str(args.build_script.expanduser().resolve()),
        str(dataset.input_dir),
        str(dataset.output_dir),
        "--msa-format",
        args.msa_format,
        "--max-msa-size",
        str(args.max_msa_size),
        "--store-max-len",
        str(args.store_max_len),
        "--max-shard-gb",
        str(args.max_shard_gb),
        "--num-workers",
        str(args.num_workers),
    ]

    if args.shuffle_rows:
        command.append("--shuffle-rows")

    if args.overwrite:
        command.append("--overwrite")

    return command


def format_command(command: Sequence[str]) -> str:
    """Format a command safely for display."""

    return shlex.join(str(item) for item in command)


def run_dataset(
    args: argparse.Namespace,
    dataset: DatasetConfig,
) -> None:
    """Run binary construction for one dataset."""

    validate_dataset(dataset)
    command = build_command(args, dataset)

    print("=" * 80)
    print(f"Dataset     : {dataset.name}")
    print(f"Input MSA   : {dataset.input_dir}")
    print(f"Output bin  : {dataset.output_dir}")
    print(f"Command     : {format_command(command)}")
    print("=" * 80, flush=True)

    if args.dry_run:
        print(f"[DRY RUN] Skipping {dataset.name}.")
        return

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to process {dataset.name}; "
            f"build_msa_binary.py exited with code {exc.returncode}."
        ) from exc

    print(f"[DONE] Successfully processed {dataset.name}.")


def main() -> int:
    """Program entry point."""

    try:
        parser = build_parser()
        args = parser.parse_args()

        validate_args(args)
        datasets = get_datasets(args)

        print("MSA binary construction configuration:")
        print(f"  dataset        = {args.dataset}")
        print(f"  build_script   = {args.build_script.expanduser().resolve()}")
        print(f"  python         = {args.python_executable}")
        print(f"  msa_format     = {args.msa_format}")
        print(f"  max_msa_size   = {args.max_msa_size}")
        print(f"  store_max_len  = {args.store_max_len}")
        print(f"  max_shard_gb   = {args.max_shard_gb}")
        print(f"  num_workers    = {args.num_workers}")
        print(f"  shuffle_rows   = {args.shuffle_rows}")
        print(f"  overwrite      = {args.overwrite}")
        print(f"  dry_run        = {args.dry_run}")

        for dataset in datasets:
            run_dataset(args, dataset)

        print("=" * 80)
        print("All requested datasets were processed successfully.")
        return 0

    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Operation cancelled by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())