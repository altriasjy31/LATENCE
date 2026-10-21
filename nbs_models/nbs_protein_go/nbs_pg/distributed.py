from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class NBSDistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    init_method: str = "env://"
    find_unused_parameters: bool = True
    broadcast_buffers: bool = False
    gradient_as_bucket_view: bool = True
    seed_by_rank: bool = True

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "NBSDistributedConfig":
        return cls(**dict(value or {}))


@dataclass
class NBSDistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    initialized_here: bool = False

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    @classmethod
    def from_environment(
        cls,
        config: Optional[NBSDistributedConfig] = None,
        *,
        initialize: bool = True,
    ) -> "NBSDistributedContext":
        cfg = config or NBSDistributedConfig()
        env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        enabled = bool(cfg.enabled or env_world_size > 1)
        rank = int(os.environ.get("RANK", "0")) if enabled else 0
        local_rank = int(os.environ.get("LOCAL_RANK", "0")) if enabled else 0
        world_size = env_world_size if enabled else 1
        backend = cfg.backend
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"
        initialized_here = False
        if enabled and initialize and not dist.is_initialized():
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend=backend,
                init_method=cfg.init_method,
                rank=rank,
                world_size=world_size,
            )
            initialized_here = True
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        return cls(
            enabled=enabled,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            initialized_here=initialized_here,
        )

    def barrier(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.barrier()

    def all_reduce_sum(self, tensor: Tensor) -> Tensor:
        if self.enabled and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def reduce_scalar_mapping(
        self,
        values: Mapping[str, float],
        *,
        device: torch.device,
    ) -> dict[str, float]:
        if not values:
            return {}
        names = sorted(values)
        tensor = torch.tensor([float(values[name]) for name in names], device=device)
        self.all_reduce_sum(tensor)
        return {name: float(tensor[i].detach().cpu()) for i, name in enumerate(names)}

    def wrap_model(
        self,
        model: torch.nn.Module,
        config: Optional[NBSDistributedConfig] = None,
    ) -> torch.nn.Module:
        if not self.enabled:
            return model
        cfg = config or NBSDistributedConfig(enabled=True)
        kwargs: dict[str, Any] = {
            "find_unused_parameters": cfg.find_unused_parameters,
            "broadcast_buffers": cfg.broadcast_buffers,
            "gradient_as_bucket_view": cfg.gradient_as_bucket_view,
        }
        if self.device.type == "cuda":
            kwargs.update(device_ids=[self.local_rank], output_device=self.local_rank)
        return DistributedDataParallel(model, **kwargs)

    def all_gather_object(self, value: Any) -> list[Any]:
        if not self.enabled or not dist.is_initialized():
            return [value]
        output: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(output, value)
        return output

    def cleanup(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def unwrap_distributed_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def distributed_seed(base_seed: int, context: NBSDistributedContext, *, by_rank: bool = True) -> int:
    return int(base_seed + context.rank if by_rank else base_seed)
