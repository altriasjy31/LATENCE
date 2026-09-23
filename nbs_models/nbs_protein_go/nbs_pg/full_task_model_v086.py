"""Matched v084 encoder control and two-hop pre-LayerNorm encoder (v0.8.6).

GO evidence injection and the complete-task matcher are inherited unchanged.
Only message-source normalization and optional, stateless PP DropEdge differ.
The anchor readout remains at layer one, within the seed's two-hop graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .full_task_model_v084 import FullTaskGraphModelV084, FullTaskModelConfigV084


@dataclass
class FullTaskModelConfigV086(FullTaskModelConfigV084):
    encoder_variant: str = "legacy"
    pp_edge_dropout: float = 0.0
    pp_dropout_seed: int = 8086


class FullTaskGraphModelV086(FullTaskGraphModelV084):
    VERSION = "0.8.6"

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfigV086 | None = None):
        config = config or FullTaskModelConfigV086()
        if config.encoder_variant not in {"legacy", "preln"}:
            raise ValueError("encoder_variant must be legacy or preln")
        if not math.isfinite(config.pp_edge_dropout) or not 0 <= config.pp_edge_dropout < 1:
            raise ValueError("pp_edge_dropout must be finite and in [0,1)")
        if config.encoder_variant == "legacy" and config.pp_edge_dropout != 0:
            raise ValueError("legacy encoder requires pp_edge_dropout=0")
        if isinstance(config.pp_dropout_seed, bool) or not isinstance(config.pp_dropout_seed, int):
            raise ValueError("pp_dropout_seed must be an integer")
        # Construct every shared module first: common weights have exactly the
        # same initialization under the same seed as the untouched v084 model.
        super().__init__(protein_dim, ontology, config)
        if config.encoder_variant == "preln":
            self.pp_source_norms = nn.ModuleList([
                nn.LayerNorm(config.hidden_dim) for _ in self.sage_layers
            ])
        self._encoder_step: int | None = None
        self._encoder_rank = 0

    def set_encoder_step(self, step: int, rank: int = 0):
        """Bind a training update; DropEdge has no hidden/restorable RNG state."""
        if (isinstance(step, bool) or not isinstance(step, int) or step < 0 or
                isinstance(rank, bool) or not isinstance(rank, int) or rank < 0):
            raise ValueError("encoder step and rank must be nonnegative integers")
        self._encoder_step, self._encoder_rank = step, rank

    def _pp_keep_mask(self, edge_count: int, device: torch.device, layer: int):
        """Independent layer streams; never consume the global torch RNG."""
        if not self.training or self.config.pp_edge_dropout == 0:
            return torch.ones(edge_count, dtype=torch.bool, device=device)
        if self._encoder_step is None:
            raise RuntimeError("training PP DropEdge requires set_encoder_step(step, rank)")
        value = (f"{self.config.pp_dropout_seed}:{self._encoder_step}:"
                 f"{self._encoder_rank}:{layer}").encode()
        seed = int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return torch.rand(edge_count, device=device, generator=generator) >= self.config.pp_edge_dropout

    def _encode_sampled(self, batch, go, *, use_weak_go, use_core_go,
                        use_pp_context, shuffle_go):
        if self.config.encoder_variant == "legacy":
            return super()._encode_sampled(
                batch, go, use_weak_go=use_weak_go, use_core_go=use_core_go,
                use_pp_context=use_pp_context, shuffle_go=shuffle_go)
        if self.training and self.config.pp_edge_dropout > 0 and self._encoder_step is None:
            raise RuntimeError("training PP DropEdge requires set_encoder_step(step, rank)")
        # Reuse v084's exact feature/evidence initialization, shape checks and
        # association permutations. Disabling only PP avoids duplicating that
        # code or drawing candidate dropout a second time.
        state, _, _, _ = super()._encode_sampled(
            batch, go, use_weak_go=use_weak_go, use_core_go=use_core_go,
            use_pp_context=False, shuffle_go=shuffle_go)
        initial = anchor_state = state
        if not use_pp_context:
            return state, anchor_state, state.new_zeros(()), state.new_zeros(())
        edge = batch["sampled_edge_index"].long()
        attr = batch["sampled_edge_attr"].float()
        kind = batch["sampled_edge_type"].long()
        retained_counts = []
        for layer_index, (layer, norm) in enumerate(zip(self.sage_layers, self.pp_source_norms)):
            # Normalize each source node before confidence-weighted aggregation;
            # normalizing the aggregated message would erase weak confidence.
            with torch.autocast(device_type=state.device.type, enabled=False):
                source_state = norm(state.float())
            keep = self._pp_keep_mask(edge.shape[1], edge.device, layer_index)
            positive = (attr[:, 0] > 0) & keep
            retained_counts.append(positive.sum().float())
            messages, active = [], []
            for relation, name in enumerate(self.PP_RELATIONS):
                selected = (kind == relation) & positive
                message, available = layer[name](source_state, edge[:, selected], attr[selected])
                messages.append(message)
                active.append(available)
            count = torch.stack(active).sum(0).clamp_min(1)
            message = torch.stack(messages).sum(0) / count[:, None]
            state = state + self.config.pp_context_scale * self.dropout(F.silu(message))
            if layer_index == 0:
                anchor_state = state
        # Comparable to v084 when DropEdge=0; with DropEdge this reports the
        # mean retained positive PP edge count per layer, not their union/sum.
        count = torch.stack(retained_counts).mean()
        norm = (state - initial).detach().norm(dim=-1).mean()
        return state, anchor_state, norm, count
