#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
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
    parser.add_argument("--config", required=True, help="NBS fixed-epoch JSON config")
    parser.add_argument(
        "--component-factory",
        default=None,
        help="module:function returning nbs_pg.training.NBSRunComponents",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", default=None, help="fixed-epoch checkpoint")
    resume_group.add_argument(
        "--fresh-start",
        action="store_true",
        help=(
            "explicitly start from newly initialized NBS weights and ignore any "
            "external resume intent; mutually exclusive with --resume"
        ),
    )
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
    parser.add_argument(
        "--empty-cache-between-epochs",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "release unused PyTorch CUDA allocator segments after each epoch; "
            "live model/optimizer/GO-cache tensors remain on device"
        ),
    )
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--require-full-supervision-retention",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "require max_candidates to retain every sampled gold, pseudo-positive "
            "and hard-negative protein"
        ),
    )
    parser.add_argument("--hard-candidate-per-query", type=int, default=None)
    parser.add_argument("--background-unlabelled-per-query", type=int, default=None)
    parser.add_argument("--background-unlabelled-weight", type=float, default=None)
    parser.add_argument("--background-base-probability-max", type=float, default=None)
    parser.add_argument("--pseudo-positive-per-query", type=int, default=None)
    parser.add_argument(
        "--pseudo-sampling-mode",
        choices=["random", "go_cyclic", "go_cyclic_unique"],
        default=None,
    )
    parser.add_argument("--weak-focus-queries", type=int, default=None)
    parser.add_argument("--weak-focus-targets", type=int, default=None)
    parser.add_argument("--weak-primary-proteins-per-episode", type=int, default=None)
    parser.add_argument(
        "--weak-primary-query-source",
        choices=["pseudo", "pseudo_candidate_intersection"],
        default=None,
    )
    parser.add_argument("--weak-focus-scan-limit", type=int, default=None)
    parser.add_argument("--weak-focus-specificity-power", type=float, default=None)
    parser.add_argument("--weak-focus-min-probability", type=float, default=None)
    parser.add_argument("--support-per-query", type=int, default=None)
    parser.add_argument("--gold-positive-per-query", type=int, default=None)
    parser.add_argument("--hierarchy-pairs-per-episode", type=int, default=None)
    parser.add_argument("--query-sampling-mode", choices=["random", "shuffled_cycle"], default=None)
    parser.add_argument("--gold-support-policy", choices=["fixed", "adaptive_rare"], default=None)
    parser.add_argument("--singleton-requires-pseudo", type=int, choices=[0, 1], default=None)
    parser.add_argument("--candidate-message-topk", type=int, default=None)
    parser.add_argument("--pseudo-message-topk", type=int, default=None)
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        choices=[0, 1],
        default=None,
        help="materialize one CPU batch ahead while the current batch runs on GPU",
    )
    parser.add_argument("--steps-per-epoch-per-rank", type=int, default=None)
    parser.add_argument("--coverage-cycles-per-epoch", type=float, default=None)
    parser.add_argument(
        "--epoch-unit",
        choices=[
            "eligible_go_coverage_cycle",
            "protein_major_with_go_floor",
            "hybrid_go_weak_coverage",
            "weak_primary_exhaustive",
        ],
        default=None,
    )
    parser.add_argument("--weak-pseudo-passes-per-epoch", type=float, default=None)
    parser.add_argument("--core-gold-passes-per-epoch", type=float, default=None)
    parser.add_argument("--weak-unique-coverage-target", type=float, default=None)
    parser.add_argument("--weak-focus-planning-efficiency", type=float, default=None)
    parser.add_argument("--scheduler-name", choices=["none", "onecycle"], default=None)
    parser.add_argument("--onecycle-pct-start", type=float, default=None)
    parser.add_argument("--onecycle-div-factor", type=float, default=None)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=None)
    parser.add_argument("--onecycle-anneal-strategy", choices=["cos", "linear"], default=None)
    parser.add_argument("--onecycle-three-phase", type=int, choices=[0, 1], default=None)
    parser.add_argument("--onecycle-cycle-momentum", type=int, choices=[0, 1], default=None)
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
    # Must be set before importing nbs_pg/torch.  The wrapper sets the same
    # default, but keeping it here also covers direct and torchrun invocation.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    project_root = _project_root_from_script()
    _install_package_path(project_root)
    from nbs_pg.distributed import (  # pylint: disable=import-outside-toplevel
        NBSDistributedConfig,
        NBSDistributedContext,
        distributed_seed,
    )
    from nbs_pg.training import (  # pylint: disable=import-outside-toplevel
        NBSSchedulerConfig,
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
    if args.empty_cache_between_epochs is not None:
        training_raw["empty_cache_between_epochs"] = bool(
            args.empty_cache_between_epochs
        )

    episode_raw = config.setdefault("episode", {})
    episode_overrides = {
        "num_queries": args.num_queries,
        "max_candidates": args.max_candidates,
        "hard_candidate_per_query": args.hard_candidate_per_query,
        "background_unlabelled_per_query": args.background_unlabelled_per_query,
        "pseudo_positive_per_query": args.pseudo_positive_per_query,
        "weak_focus_queries_per_episode": args.weak_focus_queries,
        "weak_focus_targets_per_query": args.weak_focus_targets,
        "weak_primary_proteins_per_episode": (
            args.weak_primary_proteins_per_episode
        ),
        "weak_focus_scan_limit": args.weak_focus_scan_limit,
        "support_per_query": args.support_per_query,
        "gold_positive_per_query": args.gold_positive_per_query,
        "hierarchy_pairs_per_episode": args.hierarchy_pairs_per_episode,
    }
    for key, value in episode_overrides.items():
        if value is not None:
            episode_raw[key] = int(value)
    if args.require_full_supervision_retention is not None:
        episode_raw["require_full_supervision_retention"] = bool(
            args.require_full_supervision_retention
        )
    if args.background_unlabelled_weight is not None:
        episode_raw["background_unlabelled_weight"] = float(
            args.background_unlabelled_weight
        )
    if args.background_base_probability_max is not None:
        episode_raw["background_base_probability_max"] = float(
            args.background_base_probability_max
        )
    if args.weak_focus_specificity_power is not None:
        episode_raw["weak_focus_specificity_power"] = float(
            args.weak_focus_specificity_power
        )
    if args.weak_focus_min_probability is not None:
        episode_raw["weak_focus_min_probability"] = float(
            args.weak_focus_min_probability
        )
    if args.query_sampling_mode is not None:
        episode_raw["query_sampling_mode"] = str(args.query_sampling_mode)
    if args.pseudo_sampling_mode is not None:
        episode_raw["pseudo_sampling_mode"] = str(args.pseudo_sampling_mode)
    if args.weak_primary_query_source is not None:
        episode_raw["weak_primary_query_source"] = str(
            args.weak_primary_query_source
        )
    if args.gold_support_policy is not None:
        episode_raw["gold_support_policy"] = str(args.gold_support_policy)
    if args.singleton_requires_pseudo is not None:
        episode_raw["singleton_requires_pseudo"] = bool(args.singleton_requires_pseudo)

    sampling_raw = config.setdefault("local_sampling", {})
    sampling_overrides = {
        "candidate_message_topk": args.candidate_message_topk,
        "pseudo_message_topk": args.pseudo_message_topk,
        "steps_per_epoch_per_rank": args.steps_per_epoch_per_rank,
        "prefetch_batches": args.prefetch_batches,
    }
    for key, value in sampling_overrides.items():
        if value is not None:
            sampling_raw[key] = int(value)
    if args.coverage_cycles_per_epoch is not None:
        sampling_raw["coverage_cycles_per_epoch"] = float(args.coverage_cycles_per_epoch)
    if args.epoch_unit is not None:
        sampling_raw["epoch_unit"] = str(args.epoch_unit)
    if args.weak_pseudo_passes_per_epoch is not None:
        sampling_raw["weak_pseudo_equivalent_passes_per_epoch"] = float(args.weak_pseudo_passes_per_epoch)
    if args.core_gold_passes_per_epoch is not None:
        sampling_raw["core_gold_equivalent_passes_per_epoch"] = float(args.core_gold_passes_per_epoch)
    if args.weak_unique_coverage_target is not None:
        sampling_raw["weak_unique_coverage_target_per_epoch"] = float(
            args.weak_unique_coverage_target
        )
    if args.weak_focus_planning_efficiency is not None:
        sampling_raw["weak_focus_planning_efficiency"] = float(
            args.weak_focus_planning_efficiency
        )

    scheduler_raw = config.setdefault("scheduler", {})
    if args.scheduler_name is not None:
        scheduler_raw["name"] = str(args.scheduler_name)
    if args.onecycle_pct_start is not None:
        scheduler_raw["pct_start"] = float(args.onecycle_pct_start)
    if args.onecycle_div_factor is not None:
        scheduler_raw["div_factor"] = float(args.onecycle_div_factor)
    if args.onecycle_final_div_factor is not None:
        scheduler_raw["final_div_factor"] = float(args.onecycle_final_div_factor)
    if args.onecycle_anneal_strategy is not None:
        scheduler_raw["anneal_strategy"] = str(args.onecycle_anneal_strategy)
    if args.onecycle_three_phase is not None:
        scheduler_raw["three_phase"] = bool(args.onecycle_three_phase)
    if args.onecycle_cycle_momentum is not None:
        scheduler_raw["cycle_momentum"] = bool(args.onecycle_cycle_momentum)

    if args.output_dir is not None:
        training_raw["output_dir"] = args.output_dir
    training_config = NBSFixedEpochTrainingConfig.from_mapping(training_raw)
    training_config.validate()
    scheduler_config = NBSSchedulerConfig.from_mapping(config.get("scheduler", {}))
    scheduler_config.validate()
    if scheduler_config.name == "onecycle":
        if training_config.scheduler_step != "batch":
            raise ValueError("OneCycleLR requires training.scheduler_step='batch'")
        if not training_config.include_scheduler_state:
            raise ValueError("OneCycleLR requires include_scheduler_state=true")
        if not training_config.include_optimizer_state:
            raise ValueError("OneCycleLR requires include_optimizer_state=true")
    if distributed.is_main_process:
        print(
            "validated fixed-epoch policy: "
            f"epochs={training_config.epochs}, "
            f"save_interval_epochs={training_config.save_interval_epochs}, "
            f"save_epochs={sorted(training_config.epochs_to_save())}, "
            f"world_size={distributed.world_size}, "
            f"scheduler={scheduler_config.name}, "
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
        if distributed.is_main_process and args.fresh_start:
            print("NBS fresh-start mode: checkpoint resume disabled", flush=True)
        trainer.fit(resume_from=(None if args.fresh_start else args.resume))
    finally:
        distributed.cleanup()


if __name__ == "__main__":
    main()
