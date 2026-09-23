from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def segment_sum(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if values.dim() < 1:
        raise ValueError("values must have at least one dimension")
    if index.dim() != 1 or index.numel() != values.size(0):
        raise ValueError("index must be [N] and align with values.shape[0]")
    out = values.new_zeros((dim_size,) + tuple(values.shape[1:]))
    out.index_add_(0, index, values)
    return out


def segment_mean(values: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = segment_sum(values, index, dim_size)
    counts = values.new_zeros(dim_size)
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    shape = (dim_size,) + (1,) * (values.dim() - 1)
    return out / counts.clamp_min(1).view(shape)


def segment_softmax(logits: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Segment softmax over a one-dimensional score tensor."""
    if logits.dim() != 1 or index.dim() != 1 or logits.numel() != index.numel():
        raise ValueError("logits and index must be aligned one-dimensional tensors")
    if logits.numel() == 0:
        return logits
    max_per = logits.new_full((dim_size,), -torch.inf)
    max_per.scatter_reduce_(0, index, logits, reduce="amax", include_self=True)
    exp = torch.exp(logits - max_per[index])
    denom = logits.new_zeros(dim_size)
    denom.index_add_(0, index, exp)
    return exp / denom[index].clamp_min(torch.finfo(logits.dtype).tiny)


def segment_attention_pool(
    values: Tensor,
    logits: Tensor,
    index: Tensor,
    dim_size: int,
) -> tuple[Tensor, Tensor]:
    weights = segment_softmax(logits, index, dim_size)
    pooled = segment_sum(values * weights.unsqueeze(-1), index, dim_size)
    return pooled, weights


def infer_num_segments(index: Tensor, explicit: Optional[int] = None) -> int:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("explicit segment count must be positive")
        return explicit
    if index.numel() == 0:
        raise ValueError("Cannot infer segment count from an empty index")
    return int(index.max().item()) + 1


def validate_local_index(index: Tensor, size: int, name: str) -> None:
    if index.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long")
    if index.numel() == 0:
        return
    lo = int(index.min().item())
    hi = int(index.max().item())
    if lo < 0 or hi >= size:
        raise IndexError(f"{name} contains indices outside [0, {size})")
