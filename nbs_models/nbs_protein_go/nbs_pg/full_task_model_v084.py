"""Two-layer relation SAGE with GO evidence injection and full-task matching.

The sampled graph contains observed evidence only. All supervision-seed labels
must already be removed by the data builder; modelout/expert probabilities and
training targets are not read. Candidate, binary gold and binary pseudo evidence
have separate transforms. Original task columns remain intact in exact support.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .full_task_model_v083 import FullTaskGraphModelV083, FullTaskModelConfigV083


PP_RELATIONS_V084 = ("ppi", "similar_to", "weak_to_core", "core_to_weak", "cosine")


@dataclass
class FullTaskModelConfigV084(FullTaskModelConfigV083):
    sage_layers: int = 2
    shuffle_seed: int = 8084


class WeightedRelationSAGE(nn.Module):
    """Native-torch equivalent of layers.WeightedSAGEConv's weighted mean.

    Source-to-receiver directions are explicit. Confidence multiplies a learned
    gate (initially one), and the denominator counts incoming positive edges,
    rather than summing weights: weak confidence cannot normalize back to one.
    This removes the PyG scatter dependency without changing SAGE semantics.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.source_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_gate = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.edge_gate[-1].weight)
        nn.init.zeros_(self.edge_gate[-1].bias)

    def forward(self, states: Tensor, edge: Tensor, attr: Tensor):
        n, h = states.shape
        source, receiver = edge.long()
        # All accumulation is FP32 under AMP; absent relations return zero.
        with torch.autocast(device_type=states.device.type, enabled=False):
            if source.numel() >= n:
                message = self.source_proj(states.float())[source]
            else:
                # Sparse/absent relations should project only touched sources.
                # The empty linear call also keeps a zero gradient connection.
                message = self.source_proj(states[source].float())
            attr = attr.float()
            weight = attr[:, :1].clamp(0, 1) * (2 * torch.sigmoid(self.edge_gate(attr)))
            output = states.new_zeros(n, h, dtype=torch.float32)
            output.index_add_(0, receiver, message * weight)
            count = states.new_zeros(n, 1, dtype=torch.float32)
            count.index_add_(0, receiver, (attr[:, :1] > 0).float())
            return output / count.clamp_min(1), count.squeeze(-1) > 0


class FullTaskGraphModelV084(FullTaskGraphModelV083):
    VERSION = "0.8.4"
    PP_RELATIONS = PP_RELATIONS_V084

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfigV084 | None = None):
        config = config or FullTaskModelConfigV084()
        if config.sage_layers != 2:
            raise ValueError("v084 uses exactly two relation SAGE layers")
        super().__init__(protein_dim, ontology, config)
        # Replace v083's context-only PP block and compressed anchor updater.
        del self.pp_context_encoder, self.pp_messages, self.pp_edge_encoders, self.anchor_update
        h = config.hidden_dim
        self.sampled_candidate_value = nn.Linear(h, h, bias=False)
        self.sampled_gold_value = nn.Linear(h, h, bias=False)
        self.sampled_pseudo_value = nn.Linear(h, h, bias=False)
        self.sage_layers = nn.ModuleList([
            nn.ModuleDict({name: WeightedRelationSAGE(h) for name in self.PP_RELATIONS})
            for _ in range(config.sage_layers)])

    def _binary_go_mean(self, edge: Tensor, go: Tensor, n: int, shuffle_go: bool):
        edge = edge.long()
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError("sampled GO edge must be [protein local index, task GO]")
        if edge.numel() and (edge[0].min() < 0 or edge[0].max() >= n or
                             edge[1].min() < 0 or edge[1].max() >= self.num_task_go):
            raise ValueError("sampled GO edge is outside local proteins/task columns")
        ids = self._go_indices(edge[1], shuffle_go)
        with torch.autocast(device_type=go.device.type, enabled=False):
            degree = torch.bincount(edge[0], minlength=n).float().clamp_min(1)
            sparse = torch.sparse_coo_tensor(
                torch.stack((edge[0], ids)), degree[edge[0]].reciprocal(),
                (n, self.num_task_go), device=go.device).coalesce()
            return torch.sparse.mm(sparse, go.float())

    def _encode_sampled(self, batch, go, *, use_weak_go, use_core_go,
                        use_pp_context, shuffle_go):
        x = batch["sampled_protein_x"]
        n = len(x)
        seed = batch["sampled_seed_index"].long()
        anchor = batch["sampled_anchor_index"].long()
        if seed.shape != (len(batch["protein_x"]),) or anchor.shape != (len(batch["anchor_x"]),):
            raise ValueError("sampled seed/anchor maps must match target and anchor rows")
        if any(v.numel() and (v.min() < 0 or v.max() >= n) for v in (seed, anchor)):
            raise ValueError("sampled seed/anchor map is out of range")
        state = self.protein_encoder(x).float()
        if use_weak_go:
            ids = self._go_indices(batch["sampled_candidate_go"], shuffle_go)
            attr = batch["sampled_candidate_attr"].float()
            valid = (ids >= 0) & (ids < self.num_task_go)
            if ids.shape[0] != n or attr.shape != (*ids.shape, 3):
                raise ValueError("sampled candidate arrays must align with sampled proteins")
            if self.training and self.config.candidate_dropout:
                valid = valid & (torch.rand(valid.shape, device=go.device) >= self.config.candidate_dropout)
            evidence = torch.stack((attr[..., 0].clamp(0, 1), attr[..., 1].tanh(),
                                    attr[..., 2].clamp(0, 1)), -1)
            prior = self.weak_edge_encoder(evidence).squeeze(-1).float() + attr[..., 0].clamp_min(1e-5).log()
            weight = self._weights(prior, valid)
            candidate = (go[ids.clamp(0, self.num_task_go - 1)].float() * weight[..., None]).sum(1)
            pseudo = self._binary_go_mean(batch["sampled_pseudo_edge"], go, n, shuffle_go)
            state = state + self.sampled_candidate_value(candidate).float() + self.sampled_pseudo_value(pseudo).float()
        if use_core_go:
            gold = self._binary_go_mean(batch["sampled_gold_edge"], go, n, shuffle_go)
            state = state + self.sampled_gold_value(gold).float()
        initial = state
        anchor_state = state
        edge = batch["sampled_edge_index"].long()
        attr = batch["sampled_edge_attr"].float()
        kind = batch["sampled_edge_type"].long()
        if edge.ndim != 2 or edge.shape[0] != 2 or attr.shape != (edge.shape[1], 3) or kind.shape != (edge.shape[1],):
            raise ValueError("sampled PP edges, types and attributes must align")
        if edge.numel() and (edge.min() < 0 or edge.max() >= n):
            raise ValueError("sampled PP edge is outside local proteins")
        if kind.numel() and (kind.min() < 0 or kind.max() >= len(self.PP_RELATIONS)):
            raise ValueError("unknown sampled PP relation type")
        if use_pp_context:
            for layer_index, layer in enumerate(self.sage_layers):
                messages, active = [], []
                for relation, name in enumerate(self.PP_RELATIONS):
                    selected = (kind == relation) & (attr[:, 0] > 0)
                    message, available = layer[name](state, edge[:, selected], attr[selected])
                    messages.append(message)
                    active.append(available)
                count = torch.stack(active).sum(0).clamp_min(1)
                message = torch.stack(messages).sum(0) / count[:, None]
                state = state + self.config.pp_context_scale * self.dropout(F.silu(message))
                if layer_index == 0:
                    # The anchor readout is one PP hop from the seed. Reading its
                    # second layer would require a third sampled PP hop and make
                    # outputs depend on which other seeds share the inference batch.
                    anchor_state = state
            count = (attr[:, 0] > 0).sum().float()
            norm = (state - initial).detach().norm(dim=-1).mean()
        else:
            norm = count = state.new_zeros(())
        return state, anchor_state, norm, count

    def _prepare_graph_v083(self, batch, go, *, use_weak_go, use_core_go,
                            use_pp_context, shuffle_go):
        sampled, anchor_states, pp_norm, pp_count = self._encode_sampled(
            batch, go, use_weak_go=use_weak_go, use_core_go=use_core_go,
            use_pp_context=use_pp_context, shuffle_go=shuffle_go)
        own_h = sampled[batch["sampled_seed_index"].long()]
        b, h = own_h.shape
        idx = self._go_indices(batch["candidate_go"], shuffle_go)
        valid = (idx >= 0) & (idx < self.num_task_go)
        if not use_weak_go:
            valid = torch.zeros_like(valid)
        elif self.training and self.config.candidate_dropout:
            valid = valid & (torch.rand(valid.shape, device=valid.device) >=
                             self.config.candidate_dropout)
        attr = batch["candidate_attr"].float()
        evidence = torch.stack((attr[..., 0].clamp(0, 1), attr[..., 1].tanh(),
                                attr[..., 2].clamp(0, 1)), -1)
        weak_prior = (self.weak_edge_encoder(evidence).squeeze(-1).float() +
                      attr[..., 0].clamp_min(1e-5).log())
        weak_state = go[idx.clamp(0, self.num_task_go - 1)]
        weak_key = F.normalize(self.weak_key(weak_state).float(), dim=-1)
        weak_value = F.normalize(self.weak_value(weak_state).float(), dim=-1)

        anchor_x, edge = batch["anchor_x"], batch["anchor_go_edge"].long()
        if shuffle_go:
            edge = torch.stack((edge[0], self._go_indices(edge[1], True)))
        c = len(anchor_x)
        annotation = torch.zeros(max(c, 1), self.num_task_go,
                                 dtype=torch.bool, device=go.device)
        if c:
            # The exact column incidence is independent of the mean GO state.
            # Boolean [C,G] avoids a large [B,G,K,H] expansion.
            annotation[edge[0], edge[1]] = True
            anchor = anchor_states[batch["sampled_anchor_index"].long()]
        else:
            anchor = own_h.new_zeros(1, h)
        neighbor = batch["neighbor_index"].long()
        core_valid = (neighbor >= 0) & (neighbor < c)
        if not use_core_go:
            core_valid = torch.zeros_like(core_valid)
        safe_neighbor = neighbor.clamp(0, max(c - 1, 0))
        nattr = batch["neighbor_attr"].float()
        core_prior = (self.core_edge_encoder(nattr).squeeze(-1).float() +
                      nattr[..., 0].clamp_min(1e-5).log())
        core_weight = self._weights(core_prior, core_valid)
        core_state = anchor[safe_neighbor]
        core_key = F.normalize(self.core_key(core_state).float(), dim=-1)
        core_value = F.normalize(self.core_value(core_state).float(), dim=-1)
        exact_value = F.normalize(self.core_exact_value(core_state).float(), dim=-1)
        pair = own_h.new_zeros(b, self.num_task_go, 4, dtype=torch.float32)
        rows = torch.arange(b, device=go.device)[:, None].expand_as(idx)[valid]
        pair[rows, idx[valid], 0] = 1
        pair[rows, idx[valid], 1:] = evidence[valid]
        # Compute the query-independent vote once; query-specific support is
        # read in chunks from the same exact incidence below.
        incidence = own_h.new_zeros(b, max(c, 1), dtype=torch.float32)
        incidence.scatter_add_(1, safe_neighbor, core_weight)
        with torch.autocast(device_type=go.device.type, enabled=False):
            vote = (incidence @ annotation.float()).clamp(0, 1)
        available = torch.stack((torch.ones(b, device=go.device, dtype=torch.bool),
                                 valid.any(-1), core_valid.any(-1)), -1)
        return dict(own=F.normalize(self.own_value(own_h).float(), dim=-1),
                    weak_key=weak_key, weak_value=weak_value, weak_prior=weak_prior,
                    weak_valid=valid, core_key=core_key, core_value=core_value,
                    exact_value=exact_value, core_prior=core_prior,
                    core_valid=core_valid, available=available, pair=pair, vote=vote,
                    annotation=annotation, safe_neighbor=safe_neighbor,
                    pp_message_norm=pp_norm, pp_edge_count=pp_count)
