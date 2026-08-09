from __future__ import annotations

import contextlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional UI dependency fallback
    tqdm = None

from .distributed import (
    NBSDistributedConfig,
    NBSDistributedContext,
    unwrap_distributed_model,
)
from .losses import NBSASLConfig, NBSLossWeights, nbs_training_loss
from .types import NBSGOBoxCache, ProteinGOQueryBatch


@dataclass
class NBSLocalBatch:
    """One fully materialized local NBS training episode.

    The full hundred-million-edge LATENCE graph remains in mmap/CSR stores.
    Only this batch-local graph is moved to the accelerator.
    """

    graph: Any
    query: ProteinGOQueryBatch
    global_go_cache: Optional[NBSGOBoxCache] = None
    hierarchy_edges: Optional[Tensor] = None
    hierarchy_edge_weight: Optional[Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "NBSLocalBatch":
        graph = self.graph.to(device) if hasattr(self.graph, "to") else self.graph
        cache = None
        if self.global_go_cache is not None:
            cache = NBSGOBoxCache(
                semantic=self.global_go_cache.semantic.to(device),
                hierarchy=self.global_go_cache.hierarchy.to(device),
                static=self.global_go_cache.static.to(device),
                context=self.global_go_cache.context.to(device),
                center=self.global_go_cache.center.to(device),
                offset=self.global_go_cache.offset.to(device),
                stats=(
                    None
                    if self.global_go_cache.stats is None
                    else self.global_go_cache.stats.to(device)
                ),
            )
        return NBSLocalBatch(
            graph=graph,
            query=self.query.to(device),
            global_go_cache=cache,
            hierarchy_edges=(
                None if self.hierarchy_edges is None else self.hierarchy_edges.to(device)
            ),
            hierarchy_edge_weight=(
                None
                if self.hierarchy_edge_weight is None
                else self.hierarchy_edge_weight.to(device)
            ),
            metadata=dict(self.metadata),
        )


@dataclass
class NBSLossConfig:
    primary: str = "asl"
    weights: NBSLossWeights = field(default_factory=NBSLossWeights)
    gold_asl: NBSASLConfig = field(default_factory=NBSASLConfig)
    pseudo_asl: NBSASLConfig = field(default_factory=NBSASLConfig)
    anchor_temperature: float = 1.0
    hierarchy_go_axis: int = 0
    require_pseudo_signal_per_epoch: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NBSLossConfig":
        raw = dict(value)
        weight_raw = dict(raw.pop("weights", {}))
        # Backward compatibility with v0.4.x, where ``anchor`` was the public
        # name and one ASL parameter set was shared by gold and pseudo.
        if "anchor" in weight_raw and "base_anchor" not in weight_raw:
            weight_raw["base_anchor"] = weight_raw.pop("anchor")
        legacy_gamma_neg = raw.pop("asl_gamma_neg", None)
        legacy_gamma_pos = raw.pop("asl_gamma_pos", None)
        legacy_clip = raw.pop("asl_clip", None)
        gold_raw = dict(raw.pop("gold_asl", {}))
        pseudo_raw = dict(raw.pop("pseudo_asl", {}))
        legacy_defaults = {
            "gamma_neg": 4.0 if legacy_gamma_neg is None else float(legacy_gamma_neg),
            "gamma_pos": 0.0 if legacy_gamma_pos is None else float(legacy_gamma_pos),
            "clip": 0.05 if legacy_clip is None else float(legacy_clip),
            "reduction": "weighted_mean",
        }
        for key, default in legacy_defaults.items():
            gold_raw.setdefault(key, default)
            pseudo_raw.setdefault(key, default)
        gold_asl = NBSASLConfig(**gold_raw)
        pseudo_asl = NBSASLConfig(**pseudo_raw)
        gold_asl.validate()
        pseudo_asl.validate()
        return cls(
            weights=NBSLossWeights(**weight_raw),
            gold_asl=gold_asl,
            pseudo_asl=pseudo_asl,
            **raw,
        )


@dataclass
class NBSFixedEpochTrainingConfig:
    """Fixed-epoch NBS training with no validation-set model selection.

    This intentionally mirrors the first LATENCE stage: the run proceeds to the
    requested final epoch, and named epoch snapshots (for example 100 and 150)
    are retained for independent post-training evaluation.  Early stopping and
    best-validation checkpoint semantics are rejected rather than silently
    ignored.
    """

    epochs: int = 150
    save_epochs: tuple[int, ...] = (100, 150)
    save_interval_epochs: Optional[int] = None
    output_dir: str = "outputs/nbs_train"
    checkpoint_prefix: str = "nbs"
    save_final: bool = True
    include_optimizer_state: bool = True
    include_scheduler_state: bool = True
    accumulation_steps: int = 1
    grad_clip: Optional[float] = 1.0
    amp: bool = True
    amp_dtype: str = "bfloat16"
    log_interval: int = 20
    progress_bar: bool = True
    progress_mininterval: float = 0.5
    max_steps_per_epoch: Optional[int] = None
    scheduler_step: str = "batch"
    seed: int = 3407
    validation_used: bool = False
    early_stopping: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NBSFixedEpochTrainingConfig":
        raw = dict(value)
        if "save_epochs" in raw:
            raw["save_epochs"] = tuple(int(x) for x in raw["save_epochs"])
        # v0.3 originally exposed ``save_every``.  Keep it as a read-only
        # configuration alias while standardizing the public name.
        if "save_every" in raw:
            legacy = raw.pop("save_every")
            current = raw.get("save_interval_epochs")
            if legacy is not None:
                legacy = int(legacy)
                if current is not None and int(current) != legacy:
                    raise ValueError(
                        "save_every and save_interval_epochs disagree; use only "
                        "save_interval_epochs"
                    )
                raw["save_interval_epochs"] = legacy
        if raw.get("save_interval_epochs") is not None:
            raw["save_interval_epochs"] = int(raw["save_interval_epochs"])
        return cls(**raw)

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.validation_used:
            raise ValueError(
                "NBS v0.4 fixed-epoch training does not use a validation set"
            )
        if self.early_stopping:
            raise ValueError(
                "NBS v0.4 fixed-epoch training does not support early stopping"
            )
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.progress_mininterval <= 0:
            raise ValueError("progress_mininterval must be positive")
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch <= 0:
            raise ValueError("max_steps_per_epoch must be positive or None")
        if self.save_interval_epochs is not None and self.save_interval_epochs <= 0:
            raise ValueError("save_interval_epochs must be positive or None")
        bad = [epoch for epoch in self.save_epochs if epoch <= 0 or epoch > self.epochs]
        if bad:
            raise ValueError(
                f"save_epochs must lie in [1, {self.epochs}], got {bad}"
            )
        if len(set(self.save_epochs)) != len(self.save_epochs):
            raise ValueError("save_epochs must not contain duplicates")
        if self.scheduler_step not in {"batch", "epoch", "none"}:
            raise ValueError("scheduler_step must be 'batch', 'epoch', or 'none'")
        if self.amp_dtype not in {"bfloat16", "float16"}:
            raise ValueError("amp_dtype must be 'bfloat16' or 'float16'")
        if not self.checkpoint_prefix:
            raise ValueError("checkpoint_prefix cannot be empty")

    def epochs_to_save(self) -> set[int]:
        epochs = set(self.save_epochs)
        if self.save_interval_epochs is not None:
            epochs.update(
                range(self.save_interval_epochs, self.epochs + 1, self.save_interval_epochs)
            )
        if self.save_final:
            epochs.add(self.epochs)
        return epochs


@dataclass
class NBSRunComponents:
    """Objects returned by a project-specific LATENCE component factory."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: Iterable[NBSLocalBatch]
    scheduler: Optional[Any] = None
    loss_config: NBSLossConfig = field(default_factory=NBSLossConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    global_go_graph: Optional[Any] = None


ForwardLossFn = Callable[
    [nn.Module, NBSLocalBatch, NBSLossConfig],
    tuple[Tensor, Mapping[str, Tensor]],
]
LogFn = Callable[[str], None]


def default_nbs_forward_loss(
    model: nn.Module,
    batch: NBSLocalBatch,
    loss_config: NBSLossConfig,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    output = model(
        batch.graph,
        batch.query,
        global_go_cache=batch.global_go_cache,
        return_aux=True,
    )
    hierarchy_probabilities = None
    if batch.hierarchy_edges is not None:
        hierarchy_probabilities = torch.sigmoid(output.logits)
    loss, parts = nbs_training_loss(
        output,
        primary=loss_config.primary,
        weights=loss_config.weights,
        gold_asl=loss_config.gold_asl,
        pseudo_asl=loss_config.pseudo_asl,
        anchor_temperature=loss_config.anchor_temperature,
        hierarchy_probabilities=hierarchy_probabilities,
        hierarchy_edges=batch.hierarchy_edges,
        hierarchy_edge_weight=batch.hierarchy_edge_weight,
        hierarchy_go_axis=loss_config.hierarchy_go_axis,
    )
    parts = dict(parts)
    parts["hierarchy_pairs"] = output.logits.new_tensor(
        0.0 if batch.hierarchy_edges is None else float(batch.hierarchy_edges.shape[1])
    )
    return loss, parts


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class NBSFixedEpochTrainer:
    """Fixed-epoch NBS trainer with optional single-node DDP.

    The data loader is rank-sharded and re-iterable.  Every rank must yield the
    same number of local batches per epoch; checkpoint and history writes are
    restricted to rank 0.
    """

    def __init__(
        self,
        components: NBSRunComponents,
        config: NBSFixedEpochTrainingConfig,
        *,
        device: torch.device | str | None = None,
        forward_loss_fn: ForwardLossFn = default_nbs_forward_loss,
        logger: LogFn = print,
        distributed_context: Optional[NBSDistributedContext] = None,
        distributed_config: Optional[NBSDistributedConfig] = None,
    ) -> None:
        config.validate()
        self.components = components
        self.config = config
        self.distributed = distributed_context or NBSDistributedContext(
            enabled=False, rank=0, local_rank=0, world_size=1, backend="none"
        )
        self.distributed_config = distributed_config or NBSDistributedConfig(
            enabled=self.distributed.enabled
        )
        resolved_device = self.distributed.device if self.distributed.enabled else torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = torch.device(resolved_device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.forward_loss_fn = forward_loss_fn
        self._raw_logger = logger
        self.logger = logger if self.distributed.is_main_process else (lambda _message: None)

        self.base_model = components.model.to(self.device)
        self._global_go_graph = components.global_go_graph
        self.global_go_cache: Optional[NBSGOBoxCache] = None
        self.model = self.distributed.wrap_model(self.base_model, self.distributed_config)
        self._refresh_global_go_cache()
        self.optimizer = components.optimizer
        self.scheduler = components.scheduler
        self.loss_config = components.loss_config
        self.output_dir = Path(config.output_dir)
        if self.distributed.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.distributed.barrier()
        self.start_epoch = 1
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        self._amp_enabled = bool(config.amp and self.device.type == "cuda")
        self._amp_dtype = (
            torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        )
        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self._amp_enabled and self._amp_dtype == torch.float16,
        )

    def _refresh_global_go_cache(self) -> None:
        self.global_go_cache = None
        if self._global_go_graph is None:
            return
        if not hasattr(self.base_model, "make_go_cache"):
            raise TypeError("model does not implement make_go_cache for full GO ontology")
        go_graph = self._global_go_graph.to(self.device)
        self.global_go_cache = self.base_model.make_go_cache(go_graph)
        del go_graph

    @property
    def selection_policy(self) -> dict[str, Any]:
        return {
            "mode": "fixed_epoch_snapshots",
            "validation_used": False,
            "early_stopping": False,
            "save_epochs": sorted(self.config.epochs_to_save()),
            "save_interval_epochs": self.config.save_interval_epochs,
            "final_epoch": self.config.epochs,
            "distributed_world_size": self.distributed.world_size,
        }

    def checkpoint_path(self, epoch: int) -> Path:
        return self.output_dir / f"{self.config.checkpoint_prefix}_epoch{epoch}.pt"

    def _checkpoint_payload(
        self,
        epoch: int,
        epoch_metrics: Mapping[str, Any],
        *,
        rng_state_by_rank: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        model_config = getattr(self.base_model, "config", None)
        payload: dict[str, Any] = {
            "checkpoint_type": "latence_nbs_fixed_epoch_v0.4.9",
            "nbs_version": "0.4.9",
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model_state_dict": self.base_model.state_dict(),
            "training_config": asdict(self.config),
            "loss_config": {
                **asdict(self.loss_config),
                "weights": asdict(self.loss_config.weights),
            },
            "selection_policy": self.selection_policy,
            "epoch_metrics": dict(epoch_metrics),
            "history": list(self.history),
            "run_metadata": {
                **dict(self.components.metadata),
                "distributed": {
                    "enabled": self.distributed.enabled,
                    "world_size": self.distributed.world_size,
                    "backend": self.distributed.backend,
                },
            },
            # Episode generation is epoch/global-index deterministic.  Rank-0
            # RNG is retained for model-side stochasticity and single-rank resume.
            "rng_state": _rng_state(),
            "rng_state_by_rank": rng_state_by_rank,
        }
        if model_config is not None:
            try:
                payload["model_config"] = asdict(model_config)
            except TypeError:
                payload["model_config"] = _json_safe(model_config)
        if self.config.include_optimizer_state:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None and self.config.include_scheduler_state:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if self._scaler.is_enabled():
            payload["grad_scaler_state_dict"] = self._scaler.state_dict()
        return payload

    def save_checkpoint(self, epoch: int, epoch_metrics: Mapping[str, Any]) -> Path:
        path = self.checkpoint_path(epoch)
        rng_states = self.distributed.all_gather_object(_rng_state())
        self.distributed.barrier()
        if self.distributed.is_main_process:
            _atomic_torch_save(
                self._checkpoint_payload(
                    epoch, epoch_metrics, rng_state_by_rank=rng_states
                ),
                path,
            )
            pointer = {
                "checkpoint": str(path),
                "epoch": epoch,
                "selection_policy": self.selection_policy,
            }
            pointer_path = self.output_dir / "last_checkpoint.json"
            temporary = pointer_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, pointer_path)
        self.distributed.barrier()
        return path

    def load_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("selection_policy", {}).get("validation_used", False):
            raise ValueError("refusing a validation-selected checkpoint in fixed-epoch mode")
        self.base_model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler and self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self._scaler.is_enabled() and "grad_scaler_state_dict" in checkpoint:
            self._scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        if restore_rng:
            states = checkpoint.get("rng_state_by_rank")
            if isinstance(states, list) and self.distributed.rank < len(states):
                _restore_rng_state(states[self.distributed.rank])
            elif "rng_state" in checkpoint:
                _restore_rng_state(checkpoint["rng_state"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.history = list(checkpoint.get("history", []))
        self._refresh_global_go_cache()
        self.distributed.barrier()
        return checkpoint

    def _optimizer_step(self) -> float:
        grad_norm = 0.0
        if self.config.grad_clip is not None:
            if self._scaler.is_enabled():
                self._scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float(self.config.grad_clip)
            )
            grad_norm = float(norm.detach().cpu())
        if self._scaler.is_enabled():
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None and self.config.scheduler_step == "batch":
            self.scheduler.step()
        return grad_norm

    def _train_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loader = self.components.train_loader
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        started = time.time()
        totals: dict[str, float] = {}
        steps = 0
        optimizer_steps = 0
        last_grad_norm = 0.0
        expected_steps = len(loader) if hasattr(loader, "__len__") else None
        effective_steps = expected_steps
        if self.config.max_steps_per_epoch is not None:
            effective_steps = (
                self.config.max_steps_per_epoch
                if effective_steps is None
                else min(effective_steps, self.config.max_steps_per_epoch)
            )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        batch_totals: dict[str, float] = {}
        query_seen: set[int] = set()
        query_seen_by_bucket: dict[str, set[int]] = {
            "eq1": set(),
            "eq2": set(),
            "3_4": set(),
            "gt4": set(),
        }
        eligible_go_count: Optional[int] = None
        coverage_plan = dict(getattr(loader, "coverage_plan", {}) or {})
        progress = None
        if (
            self.config.progress_bar
            and self.distributed.is_main_process
            and tqdm is not None
            and effective_steps is not None
        ):
            progress = tqdm(
                total=int(effective_steps),
                desc=f"NBS epoch {epoch}/{self.config.epochs}",
                dynamic_ncols=True,
                mininterval=float(self.config.progress_mininterval),
                leave=True,
            )

        for step, raw_batch in enumerate(loader, start=1):
            if (
                self.config.max_steps_per_epoch is not None
                and step > self.config.max_steps_per_epoch
            ):
                break
            batch = raw_batch.to(self.device)
            scalar_metadata_keys = (
                "protein_nodes",
                "go_nodes",
                "candidate_edges_materialized",
                "hierarchy_query_edges",
                "query_gold_count_eq1",
                "query_gold_count_eq2",
                "query_gold_count_3_4",
                "query_gold_count_gt4",
                # These fields were already present in episode metadata and in
                # the tqdm postfix, but v0.4.8 forgot to accumulate them for the
                # epoch summary, which made qcan/qps print as 0.0.
                "query_with_pseudo_pool",
                "query_with_backbone_candidate_pool",
                "support_proteins",
                "candidate_union_before_cap",
                "candidate_union_after_cap",
                "candidate_dropped_by_cap",
                "candidate_truncation_fraction",
                "gold_pairs_requested",
                "gold_pairs_retained",
                "hard_pairs_requested",
                "hard_pairs_retained",
                "pseudo_pairs_requested",
                "pseudo_pairs_retained",
            )
            if isinstance(raw_batch.metadata, Mapping):
                for key in scalar_metadata_keys:
                    value = raw_batch.metadata.get(key)
                    if value is not None:
                        batch_totals[key] = batch_totals.get(key, 0.0) + float(value)
                query_values = raw_batch.metadata.get("query_go_idx")
                query_gold_counts = raw_batch.metadata.get("query_gold_counts")
                if query_values is not None:
                    query_values = [int(value) for value in query_values]
                    query_seen.update(query_values)
                    if query_gold_counts is not None:
                        query_gold_counts = [int(value) for value in query_gold_counts]
                        if len(query_gold_counts) != len(query_values):
                            raise RuntimeError(
                                "query_gold_counts must align with query_go_idx"
                            )
                        for go_idx, count in zip(query_values, query_gold_counts):
                            if count == 1:
                                query_seen_by_bucket["eq1"].add(go_idx)
                            elif count == 2:
                                query_seen_by_bucket["eq2"].add(go_idx)
                            elif 3 <= count <= 4:
                                query_seen_by_bucket["3_4"].add(go_idx)
                            elif count > 4:
                                query_seen_by_bucket["gt4"].add(go_idx)
                            else:
                                raise RuntimeError(
                                    f"eligible query GO has invalid gold count {count}"
                                )
                if raw_batch.metadata.get("eligible_go_count") is not None:
                    current_eligible = int(raw_batch.metadata["eligible_go_count"])
                    if eligible_go_count is None:
                        eligible_go_count = current_eligible
                    elif eligible_go_count != current_eligible:
                        raise RuntimeError("eligible GO count changed within one epoch")
            if batch.global_go_cache is None and self.global_go_cache is not None:
                batch.global_go_cache = self.global_go_cache
            is_last = effective_steps is not None and step >= effective_steps
            should_step = step % self.config.accumulation_steps == 0 or is_last
            sync_context = contextlib.nullcontext()
            if self.distributed.enabled and not should_step and hasattr(self.model, "no_sync"):
                sync_context = self.model.no_sync()
            with sync_context:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._amp_dtype,
                    enabled=self._amp_enabled,
                ):
                    loss, parts = self.forward_loss_fn(self.model, batch, self.loss_config)
                    divisor = self.config.accumulation_steps
                    if effective_steps is not None:
                        group_start = ((step - 1) // self.config.accumulation_steps) * self.config.accumulation_steps + 1
                        group_end = min(
                            group_start + self.config.accumulation_steps - 1,
                            effective_steps,
                        )
                        divisor = group_end - group_start + 1
                    scaled_loss = loss / max(1, divisor)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite NBS loss at epoch={epoch}, step={step}: {loss.item()}"
                    )
                if self._scaler.is_enabled():
                    self._scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            steps += 1
            self.global_step += 1
            for name, value in parts.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            if should_step:
                last_grad_norm = self._optimizer_step()
                optimizer_steps += 1

            if progress is not None:
                progress.update(1)

            if step % self.config.log_interval == 0 or is_last:
                graph_scale = getattr(getattr(self.base_model, "matcher", None), "graph_delta_scale", None)
                scale_value = (
                    float(torch.tanh(graph_scale).detach().cpu())
                    if isinstance(graph_scale, Tensor)
                    else float("nan")
                )

                def _part(name: str) -> float:
                    value = parts.get(name)
                    return float(value.detach().cpu()) if isinstance(value, Tensor) else float("nan")

                memory_gb = (
                    float(torch.cuda.memory_allocated(self.device)) / (1024 ** 3)
                    if self.device.type == "cuda"
                    else 0.0
                )
                meta = raw_batch.metadata if isinstance(raw_batch.metadata, Mapping) else {}
                status = {
                    "loss": f"{float(loss.detach().cpu()):.4f}",
                    "gold": f"{_part('gold_asl'):.3g}",
                    "pseudo": f"{_part('pseudo_asl'):.3g}",
                    "gpos": int(meta.get("gold_pairs_retained", 0)),
                    "hard": int(meta.get("hard_pairs_retained", 0)),
                    "ppos": int(meta.get("pseudo_pairs_retained", 0)),
                    "qps": int(meta.get("query_with_pseudo_pool", 0)),
                    "qcan": int(meta.get("query_with_backbone_candidate_pool", 0)),
                    "h_pairs": int(_part('hierarchy_pairs')),
                    "qseen_r0": len(query_seen),
                    "capdrop": f"{float(meta.get('candidate_truncation_fraction', 0.0)):.1%}",
                    "gscale": f"{scale_value:.4f}",
                }
                if self.device.type == "cuda":
                    status["mem"] = f"{memory_gb:.1f}G"
                if progress is not None:
                    progress.set_postfix(status, refresh=False)
                else:
                    self.logger(
                        f"epoch={epoch} step={step} global_step={self.global_step} "
                        f"loss={float(loss.detach().cpu()):.6f} "
                        f"gold_asl={_part('gold_asl'):.6f} "
                        f"pseudo_asl={_part('pseudo_asl'):.6f} "
                        f"c_gold={_part('contrib_gold_asl'):.6f} "
                        f"c_pseudo={_part('contrib_pseudo_asl'):.6f} "
                        f"c_anchor={_part('contrib_base_anchor'):.6f} "
                        f"hier={_part('hierarchy'):.6f} "
                        f"c_hier={_part('contrib_hierarchy'):.6f} "
                        f"hier_pairs={_part('hierarchy_pairs'):.0f} "
                        f"gold_pairs={_part('gold_supervised_pairs'):.0f} "
                        f"pseudo_pairs={_part('pseudo_supervised_pairs'):.0f} "
                        f"graph_delta_scale={scale_value:.6f} "
                        f"world_size={self.distributed.world_size}"
                    )

        if progress is not None:
            progress.close()
        if steps == 0:
            raise RuntimeError("train_loader produced zero batches")
        if effective_steps is None and steps % self.config.accumulation_steps != 0:
            last_grad_norm = self._optimizer_step()
            optimizer_steps += 1
        if self.scheduler is not None and self.config.scheduler_step == "epoch":
            self.scheduler.step()

        reduction = {f"loss/{name}": value for name, value in totals.items()}
        reduction.update({f"batch/{name}": value for name, value in batch_totals.items()})
        reduction.update({"steps": float(steps), "optimizer_steps": float(optimizer_steps)})
        reduced = self.distributed.reduce_scalar_mapping(reduction, device=self.device)
        global_steps = max(1.0, reduced.pop("steps"))
        global_optimizer_steps = reduced.pop("optimizer_steps")
        metrics: dict[str, Any] = {
            "epoch": epoch,
            "steps_per_rank": steps,
            "global_steps": int(global_steps),
            "optimizer_steps_per_rank": optimizer_steps,
            "global_optimizer_steps": int(global_optimizer_steps),
            "global_step": self.global_step,
            "elapsed_seconds": round(time.time() - started, 3),
            "grad_norm_last": last_grad_norm,
            "learning_rates": [float(group["lr"]) for group in self.optimizer.param_groups],
            "world_size": self.distributed.world_size,
        }
        for name, value in reduced.items():
            if name.startswith("loss/"):
                metrics[name.removeprefix("loss/")] = value / global_steps
            elif name.startswith("batch/"):
                metrics[f"avg_{name.removeprefix('batch/')}"] = value / global_steps

        gathered_query_state = self.distributed.all_gather_object(
            {
                "all": sorted(query_seen),
                **{
                    bucket: sorted(values)
                    for bucket, values in query_seen_by_bucket.items()
                },
            }
        )
        global_query_seen: set[int] = set()
        global_query_seen_by_bucket: dict[str, set[int]] = {
            "eq1": set(),
            "eq2": set(),
            "3_4": set(),
            "gt4": set(),
        }
        for state in gathered_query_state:
            global_query_seen.update(int(value) for value in state.get("all", []))
            for bucket in global_query_seen_by_bucket:
                global_query_seen_by_bucket[bucket].update(
                    int(value) for value in state.get(bucket, [])
                )
        denominator = int(eligible_go_count or coverage_plan.get("eligible_go_count", 0) or 0)
        metrics["unique_query_go_count"] = int(len(global_query_seen))
        metrics["eligible_go_count"] = denominator
        metrics["query_coverage_rate"] = (
            0.0 if denominator <= 0 else float(len(global_query_seen) / denominator)
        )
        metrics["coverage_plan"] = coverage_plan
        metrics["eligibility_summary"] = dict(coverage_plan.get("eligibility_summary", {}) or {})

        eligibility = metrics["eligibility_summary"]
        bucket_specs = {
            "eq1": ("eligible_gold_count_eq1", "query_gold_count_eq1"),
            "eq2": ("eligible_gold_count_eq2", "query_gold_count_eq2"),
            "3_4": ("eligible_gold_count_3_4", "query_gold_count_3_4"),
            "gt4": ("eligible_gold_count_gt4", "query_gold_count_gt4"),
        }
        for bucket, (eligible_key, occurrence_key) in bucket_specs.items():
            eligible_count = int(eligibility.get(eligible_key, 0) or 0)
            unique_count = int(len(global_query_seen_by_bucket[bucket]))
            occurrence_count = int(round(float(
                reduced.get(f"batch/{occurrence_key}", 0.0)
            )))
            metrics[f"eligible_query_go_{bucket}"] = eligible_count
            metrics[f"unique_query_go_{bucket}"] = unique_count
            metrics[f"query_occurrences_{bucket}"] = occurrence_count
            metrics[f"query_coverage_rate_{bucket}"] = (
                0.0 if eligible_count <= 0 else float(unique_count / eligible_count)
            )

        total_query_occurrences = int(sum(
            metrics[f"query_occurrences_{bucket}"]
            for bucket in bucket_specs
        ))
        metrics["query_occurrences_total"] = total_query_occurrences
        candidate_pool_occurrences = int(round(float(
            reduced.get("batch/query_with_backbone_candidate_pool", 0.0)
        )))
        pseudo_pool_occurrences = int(round(float(
            reduced.get("batch/query_with_pseudo_pool", 0.0)
        )))
        metrics["query_with_backbone_candidate_pool_occurrences"] = candidate_pool_occurrences
        metrics["query_with_pseudo_pool_occurrences"] = pseudo_pool_occurrences
        metrics["query_with_backbone_candidate_pool_rate"] = (
            0.0
            if total_query_occurrences <= 0
            else float(candidate_pool_occurrences / total_query_occurrences)
        )
        metrics["query_with_pseudo_pool_rate"] = (
            0.0
            if total_query_occurrences <= 0
            else float(pseudo_pool_occurrences / total_query_occurrences)
        )
        hard_requested = float(metrics.get("avg_hard_pairs_requested", 0.0))
        hard_retained = float(metrics.get("avg_hard_pairs_retained", 0.0))
        metrics["hard_pair_retention_rate"] = (
            1.0 if hard_requested <= 0 else min(1.0, hard_retained / hard_requested)
        )
        candidate_before = float(metrics.get("avg_candidate_union_before_cap", 0.0))
        candidate_after = float(metrics.get("avg_candidate_union_after_cap", 0.0))
        metrics["candidate_capacity_retention_rate"] = (
            1.0 if candidate_before <= 0 else min(1.0, candidate_after / candidate_before)
        )
        if self.device.type == "cuda":
            local_peak_alloc = float(torch.cuda.max_memory_allocated(self.device)) / (1024 ** 3)
            local_peak_reserved = float(torch.cuda.max_memory_reserved(self.device)) / (1024 ** 3)
            gathered = self.distributed.all_gather_object(
                {"allocated": local_peak_alloc, "reserved": local_peak_reserved}
            )
            metrics["gpu_peak_allocated_gb"] = max(float(item["allocated"]) for item in gathered)
            metrics["gpu_peak_reserved_gb"] = max(float(item["reserved"]) for item in gathered)
        graph_scale = getattr(getattr(self.base_model, "matcher", None), "graph_delta_scale", None)
        if isinstance(graph_scale, Tensor):
            metrics["graph_delta_scale"] = float(torch.tanh(graph_scale).detach().cpu())
        if (
            self.loss_config.require_pseudo_signal_per_epoch
            and self.loss_config.weights.pseudo > 0
            and metrics.get("pseudo_supervised_pairs", 0.0) <= 0.0
        ):
            raise RuntimeError(
                "pseudo ASL is enabled but this epoch contained zero pseudo-supervised "
                "pairs across all DDP ranks; audit pseudo indices and episode sampling"
            )
        return metrics

    def fit(self, *, resume_from: Optional[str | os.PathLike[str]] = None) -> list[dict[str, Any]]:
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        if self.start_epoch > self.config.epochs:
            raise ValueError(
                f"resume checkpoint epoch {self.start_epoch - 1} already reaches/exceeds "
                f"configured final epoch {self.config.epochs}"
            )
        self.logger(
            "NBS fixed-epoch training: "
            f"epochs={self.config.epochs}, "
            f"save_interval_epochs={self.config.save_interval_epochs}, "
            f"save_epochs={sorted(self.config.epochs_to_save())}, "
            f"world_size={self.distributed.world_size}, "
            "validation=False, early_stopping=False"
        )
        coverage_plan = dict(getattr(self.components.train_loader, "coverage_plan", {}) or {})
        if coverage_plan and self.distributed.is_main_process:
            eligibility = dict(coverage_plan.get("eligibility_summary", {}) or {})
            (self.output_dir / "query_coverage_plan.json").write_text(
                json.dumps(_json_safe(coverage_plan), indent=2) + "\n", encoding="utf-8"
            )
            self.logger(
                "NBS query coverage plan: "
                f"eligible_go={coverage_plan.get('eligible_go_count')}, "
                f"coverage_slots/episode={coverage_plan.get('coverage_slots_per_episode')}, "
                f"steps/rank={coverage_plan.get('resolved_steps_per_epoch_per_rank')}, "
                f"estimated_cycles={coverage_plan.get('estimated_base_cycle_coverage', 0.0):.3f}, "
                f"eligible_q1={eligibility.get('eligible_gold_count_eq1', 0)}, "
                f"eligible_q2={eligibility.get('eligible_gold_count_eq2', 0)}, "
                f"eligible_q3_4={eligibility.get('eligible_gold_count_3_4', 0)}, "
                f"eligible_qgt4={eligibility.get('eligible_gold_count_gt4', 0)}, "
                f"singleton_all={eligibility.get('gold_count_eq1_all', 0)}, "
                f"singleton_with_pseudo={eligibility.get('gold_count_eq1_with_pseudo', 0)}, "
                f"eligible_with_candidate={eligibility.get('eligible_with_backbone_candidate', 0)}, "
                f"eligible_with_pseudo={eligibility.get('eligible_with_pseudo', 0)}"
            )
        save_epochs = self.config.epochs_to_save()
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            metrics = self._train_epoch(epoch)
            if self.distributed.is_main_process:
                self.history.append(metrics)
            self.logger(
                f"epoch={epoch} complete "
                f"loss={metrics.get('total', float('nan')):.6f} "
                f"gold_asl={metrics.get('gold_asl', float('nan')):.6f} "
                f"pseudo_asl={metrics.get('pseudo_asl', float('nan')):.6f} "
                f"c_gold={metrics.get('contrib_gold_asl', float('nan')):.6f} "
                f"c_pseudo={metrics.get('contrib_pseudo_asl', float('nan')):.6f} "
                f"c_anchor={metrics.get('contrib_base_anchor', float('nan')):.6f} "
                f"hier={metrics.get('hierarchy', float('nan')):.6f} "
                f"c_hier={metrics.get('contrib_hierarchy', float('nan')):.6f} "
                f"hier_pairs={metrics.get('hierarchy_pairs', float('nan')):.1f} "
                f"gpos={metrics.get('avg_gold_pairs_retained', 0.0):.1f} "
                f"hard={metrics.get('avg_hard_pairs_retained', 0.0):.1f} "
                f"ppos={metrics.get('avg_pseudo_pairs_retained', 0.0):.1f} "
                f"qcov={metrics.get('unique_query_go_count', 0)}/"
                f"{metrics.get('eligible_go_count', 0)}"
                f"({metrics.get('query_coverage_rate', 0.0):.3%}) "
                f"q1={metrics.get('unique_query_go_eq1', 0)}/"
                f"{metrics.get('eligible_query_go_eq1', 0)}"
                f"[occ={metrics.get('query_occurrences_eq1', 0)}] "
                f"q2={metrics.get('unique_query_go_eq2', 0)}/"
                f"{metrics.get('eligible_query_go_eq2', 0)} "
                f"q3_4={metrics.get('unique_query_go_3_4', 0)}/"
                f"{metrics.get('eligible_query_go_3_4', 0)} "
                f"qgt4={metrics.get('unique_query_go_gt4', 0)}/"
                f"{metrics.get('eligible_query_go_gt4', 0)} "
                f"qcan={metrics.get('query_with_backbone_candidate_pool_rate', 0.0):.1%} "
                f"qps={metrics.get('query_with_pseudo_pool_rate', 0.0):.1%} "
                f"hard_keep={metrics.get('hard_pair_retention_rate', 1.0):.1%} "
                f"cand_keep={metrics.get('candidate_capacity_retention_rate', 1.0):.1%} "
                f"peak_mem={metrics.get('gpu_peak_allocated_gb', float('nan')):.2f}GB "
                f"elapsed={metrics['elapsed_seconds']:.3f}s"
            )
            if epoch in save_epochs:
                path = self.save_checkpoint(epoch, metrics)
                self.logger(f"saved fixed-epoch checkpoint: {path}")
        if self.distributed.is_main_process:
            history_path = self.output_dir / "training_history.json"
            history_path.write_text(
                json.dumps(_json_safe(self.history), indent=2) + "\n", encoding="utf-8"
            )
        self.distributed.barrier()
        return self.history

def freeze_go_geometry(model: nn.Module, freeze: bool = True) -> None:
    """Freeze/unfreeze GO geometry modules for staged NBS training."""
    names = ("go_box_encoder", "go_tower", "box_calibrator")
    for name in names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = not freeze
