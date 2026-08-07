#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


def _project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def _install_package_path(project_root: Path) -> None:
    package_root = project_root / "nbs_models" / "nbs_protein_go"
    if not package_root.exists():
        raise FileNotFoundError(f"NBS package root not found: {package_root}")
    sys.path.insert(0, str(package_root))
    sys.path.insert(0, str(project_root))


def _load_callable(spec: str) -> Callable[[dict[str, Any]], Any]:
    if ":" not in spec:
        raise ValueError("component factory must be formatted as module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, function_name, None)
    if value is None or not callable(value):
        raise TypeError(f"component factory is not callable: {spec}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LATENCE NBS to fixed epochs without validation-set model "
            "selection or early stopping."
        )
    )
    parser.add_argument("--config", required=True, help="NBS v0.4 JSON config")
    parser.add_argument(
        "--component-factory",
        default=None,
        help="module:function returning nbs_pg.training.NBSRunComponents",
    )
    parser.add_argument("--resume", default=None, help="fixed-epoch checkpoint")
    parser.add_argument(
        "--save-interval-epochs",
        type=int,
        default=None,
        help=(
            "override periodic checkpoint interval; for example 5 or 10. "
            "The saved epochs are the union of this interval, save_epochs, and final epoch."
        ),
    )
    parser.add_argument("--device", default=None, help="single-process override; torchrun uses LOCAL_RANK")
    parser.add_argument("--epochs", type=int, default=None, help="override final epoch")
    parser.add_argument(
        "--save-epochs",
        default=None,
        help="comma-separated fixed snapshots; defaults to the overridden final epoch for smoke runs",
    )
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--progress-bar", type=int, choices=[0, 1], default=None)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--hard-candidate-per-query", type=int, default=None)
    parser.add_argument("--pseudo-positive-per-query", type=int, default=None)
    parser.add_argument("--support-per-query", type=int, default=None)
    parser.add_argument("--gold-positive-per-query", type=int, default=None)
    parser.add_argument("--hierarchy-pairs-per-episode", type=int, default=None)
    parser.add_argument("--candidate-message-topk", type=int, default=None)
    parser.add_argument("--pseudo-message-topk", type=int, default=None)
    parser.add_argument("--steps-per-epoch-per-rank", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--distributed", action="store_true", help="enable DDP even when WORLD_SIZE is not preset")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="validate the fixed-epoch policy without constructing data/model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = _project_root_from_script()
    _install_package_path(project_root)
    from nbs_pg.distributed import (  # pylint: disable=import-outside-toplevel
        NBSDistributedConfig,
        NBSDistributedContext,
        distributed_seed,
    )
    from nbs_pg.training import (  # pylint: disable=import-outside-toplevel
        NBSFixedEpochTrainer,
        NBSFixedEpochTrainingConfig,
        NBSRunComponents,
        seed_everything,
    )

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    distributed_raw = dict(config.get("distributed", {}))
    if args.distributed:
        distributed_raw["enabled"] = True
    distributed_config = NBSDistributedConfig.from_mapping(distributed_raw)
    distributed = NBSDistributedContext.from_environment(distributed_config, initialize=True)
    config["_distributed_runtime"] = {
        "enabled": distributed.enabled,
        "rank": distributed.rank,
        "local_rank": distributed.local_rank,
        "world_size": distributed.world_size,
        "backend": distributed.backend,
    }
    training_raw = config.setdefault("training", {})
    if args.save_interval_epochs is not None:
        training_raw["save_interval_epochs"] = args.save_interval_epochs
    if args.epochs is not None:
        training_raw["epochs"] = int(args.epochs)
        if args.save_epochs is None:
            training_raw["save_epochs"] = [int(args.epochs)]
    if args.save_epochs is not None:
        training_raw["save_epochs"] = [
            int(value) for value in args.save_epochs.split(",") if value.strip()
        ]
    if args.max_steps_per_epoch is not None:
        training_raw["max_steps_per_epoch"] = int(args.max_steps_per_epoch)
    if args.log_interval is not None:
        training_raw["log_interval"] = int(args.log_interval)
    if args.progress_bar is not None:
        training_raw["progress_bar"] = bool(args.progress_bar)

    episode_raw = config.setdefault("episode", {})
    episode_overrides = {
        "num_queries": args.num_queries,
        "max_candidates": args.max_candidates,
        "hard_candidate_per_query": args.hard_candidate_per_query,
        "pseudo_positive_per_query": args.pseudo_positive_per_query,
        "support_per_query": args.support_per_query,
        "gold_positive_per_query": args.gold_positive_per_query,
        "hierarchy_pairs_per_episode": args.hierarchy_pairs_per_episode,
    }
    for key, value in episode_overrides.items():
        if value is not None:
            episode_raw[key] = int(value)

    sampling_raw = config.setdefault("local_sampling", {})
    sampling_overrides = {
        "candidate_message_topk": args.candidate_message_topk,
        "pseudo_message_topk": args.pseudo_message_topk,
        "steps_per_epoch_per_rank": args.steps_per_epoch_per_rank,
    }
    for key, value in sampling_overrides.items():
        if value is not None:
            sampling_raw[key] = int(value)

    if args.output_dir is not None:
        training_raw["output_dir"] = args.output_dir
    training_config = NBSFixedEpochTrainingConfig.from_mapping(training_raw)
    training_config.validate()
    if distributed.is_main_process:
        print(
            "validated fixed-epoch policy: "
            f"epochs={training_config.epochs}, "
            f"save_interval_epochs={training_config.save_interval_epochs}, "
            f"save_epochs={sorted(training_config.epochs_to_save())}, "
            f"world_size={distributed.world_size}, "
            "validation=False, early_stopping=False"
        )
        resolved_output = Path(training_config.output_dir)
        if not resolved_output.is_absolute():
            resolved_output = project_root / resolved_output
        resolved_output.mkdir(parents=True, exist_ok=True)
        (resolved_output / "resolved_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    distributed.barrier()
    if args.validate_config_only:
        distributed.cleanup()
        return

    factory_spec = args.component_factory or config.get("component_factory")
    if not factory_spec:
        raise ValueError(
            "A project-specific component factory is required. Pass "
            "--component-factory module:function or set component_factory in JSON. "
            "The factory must materialize local NBS batches from LATENCE mmap/CSR stores."
        )
    # All ranks construct identical model parameters.  Rank-specific RNG is
    # applied only after DDP synchronization and global GO-cache creation.
    seed_everything(training_config.seed)
    factory = _load_callable(str(factory_spec))
    components = factory(config)
    if not isinstance(components, NBSRunComponents):
        raise TypeError(
            f"factory {factory_spec} returned {type(components).__name__}; "
            "expected NBSRunComponents"
        )
    trainer = NBSFixedEpochTrainer(
        components,
        training_config,
        device=(None if distributed.enabled else args.device or config.get("device", "cuda")),
        distributed_context=distributed,
        distributed_config=distributed_config,
    )
    seed_everything(
        distributed_seed(
            training_config.seed,
            distributed,
            by_rank=distributed_config.seed_by_rank,
        )
    )
    try:
        trainer.fit(resume_from=args.resume)
    finally:
        distributed.cleanup()


if __name__ == "__main__":
    main()
