"""Single full-task matcher with shared PP context and exact core–GO messages.

Only observed graph inputs enter this module. Targets, modelout and expert
probabilities are not part of its forward contract. The final residual starts
at zero; the base logits remain a fixed reference, not a learned second tower.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .full_task_model_v081 import FullTaskGraphModelV081, FullTaskModelConfigV081


@dataclass
class FullTaskModelConfigV083(FullTaskModelConfigV081):
    pp_context_scale: float = 1.0
    shuffle_seed: int = 8083


class FullTaskGraphModelV083(FullTaskGraphModelV081):
    VERSION = "0.8.3"
    SOURCE_NAMES = ("protein", "weak_go", "core_go", "core_exact")
    PP_RELATIONS = ("ppi", "similar_to", "weak_to_core")

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfigV083 | None = None):
        config = config or FullTaskModelConfigV083()
        super().__init__(protein_dim, ontology, config)
        if not math.isfinite(config.pp_context_scale) or config.pp_context_scale < 0:
            raise ValueError("pp_context_scale must be finite and nonnegative")
        h, q, d = config.hidden_dim, config.query_dim, config.decoder_hidden
        # No independently fitted auxiliary graph head: the graph is optimized
        # through its actual contribution to the final prediction.
        del self.graph_output
        self.pp_context_encoder = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(),
                                                 nn.LayerNorm(h))
        self.pp_messages = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(h, h, bias=False), nn.SiLU()) for name in self.PP_RELATIONS})
        self.pp_edge_encoders = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))
            for name in self.PP_RELATIONS})
        self.core_exact_value = nn.Linear(h, q, bias=False)
        self.graph_hidden = nn.Sequential(nn.Linear(4 * q + 4, d), nn.SiLU())
        # hidden + base + four weak-pair features + vote + vote-base + mass
        self.correction_decoder = nn.Sequential(nn.Linear(d + 8, d), nn.SiLU(),
                                                nn.Linear(d, 1))
        nn.init.zeros_(self.correction_decoder[-1].weight)
        nn.init.zeros_(self.correction_decoder[-1].bias)
        generator = torch.Generator().manual_seed(config.shuffle_seed)
        permutation = torch.randperm(self.num_task_go, generator=generator)
        self.register_buffer("association_permutation", permutation, persistent=True)

    def _go_indices(self, ids: Tensor, shuffle_go: bool) -> Tensor:
        """Permute associations while preserving invalid padding and GO columns."""
        ids = ids.long()
        if not shuffle_go:
            return ids
        valid = (ids >= 0) & (ids < self.num_task_go)
        mapped = self.association_permutation[ids.clamp(0, self.num_task_go - 1)]
        return torch.where(valid, mapped, ids)

    def _pp_context(self, batch, go, c, *, use_weak_go, use_pp_context, shuffle_go):
        zero = go.new_zeros(c, self.config.hidden_dim)
        if not use_pp_context or not c or "pp_neighbor_index" not in batch:
            return zero, go.new_zeros(()), go.new_zeros(())
        neighbor = batch["pp_neighbor_index"].long()
        attr = batch["pp_neighbor_attr"].float()
        x = batch["pp_protein_x"]
        if neighbor.ndim != 3 or neighbor.shape[:2] != (c, 3):
            raise ValueError("pp_neighbor_index must be [core anchors,3,fanout]")
        if attr.shape != (*neighbor.shape, 3):
            raise ValueError("pp_neighbor_attr must align with PP neighbours")
        if not len(x) or not neighbor.shape[-1]:
            return zero, go.new_zeros(()), go.new_zeros(())
        idx = self._go_indices(batch["pp_candidate_go"], shuffle_go)
        evidence = batch["pp_candidate_attr"].float()
        valid = (idx >= 0) & (idx < self.num_task_go)
        if not use_weak_go:
            valid = torch.zeros_like(valid)
        features = torch.stack((evidence[..., 0].clamp(0, 1),
                                evidence[..., 1].tanh(),
                                evidence[..., 2].clamp(0, 1)), -1)
        prior = (self.weak_edge_encoder(features).squeeze(-1).float() +
                 evidence[..., 0].clamp_min(1e-5).log())
        weights = self._weights(prior, valid)
        labels = (go[idx.clamp(0, self.num_task_go - 1)] * weights[..., None]).sum(1)
        source = self.pp_context_encoder(torch.cat((self.protein_encoder(x), labels), -1))
        messages, available = [], []
        all_valid = ((neighbor >= 0) & (neighbor < len(x)) &
                     (attr[..., 0] > 0))
        safe_neighbor = neighbor.clamp(0, len(x) - 1)
        for relation, name in enumerate(self.PP_RELATIONS):
            edge_features = attr[:, relation]
            valid_edges = all_valid[:, relation]
            prior = (self.pp_edge_encoders[name](edge_features).squeeze(-1).float() +
                     edge_features[..., 0].clamp_min(1e-5).log())
            # Normalization alone would turn a lone confidence=0.001 edge into
            # a full-strength message. Retain confidence after normalization.
            weight = self._weights(prior, valid_edges) * edge_features[..., 0].clamp(0, 1)
            source_message = self.pp_messages[name](source)[safe_neighbor[:, relation]]
            messages.append((weight[..., None] * source_message).sum(1))
            available.append(valid_edges.any(-1))
        normalizer = torch.stack(available, -1).sum(-1).clamp_min(1)
        message = sum(messages) / normalizer[:, None]
        return (message * self.config.pp_context_scale,
                message.detach().norm(dim=-1).mean(), all_valid.sum().float())

    def _prepare_graph_v083(self, batch, go, *, use_weak_go, use_core_go,
                            use_pp_context, shuffle_go):
        own_h = self.protein_encoder(batch["protein_x"])
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
            degree = torch.bincount(edge[0], minlength=c).float().clamp_min(1)
            mean_annotation = torch.sparse_coo_tensor(
                edge, degree[edge[0]].reciprocal(), (c, self.num_task_go),
                device=go.device).coalesce()
            with torch.autocast(device_type=go.device.type, enabled=False):
                labels = torch.sparse.mm(mean_annotation, go.float())
            pp_message, pp_norm, pp_count = self._pp_context(
                batch, go, c, use_weak_go=use_weak_go,
                use_pp_context=use_pp_context and use_core_go, shuffle_go=shuffle_go)
            anchor = self.anchor_update(torch.cat((
                self.protein_encoder(anchor_x) + pp_message, labels), -1))
        else:
            anchor = own_h.new_zeros(1, h)
            pp_norm = pp_count = own_h.new_zeros(())
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

    def _decode_chunk_v083(self, query, base, own, weak_key, weak_value, weak_prior,
                          weak_valid, core_key, core_value, exact_value, core_prior,
                          core_valid, available, pair, vote, support):
        with torch.autocast(device_type=base.device.type, enabled=False):
            weak = self._attend(query, weak_key, weak_value, weak_prior, weak_valid)
            core = self._attend(query, core_key, core_value, core_prior, core_valid)
            # [B,Q,K] is bounded by go_chunk and target core fanout. The exact
            # message retains both which GO is annotated and which core carries it.
            if core_key.shape[1]:
                scores = torch.einsum("gd,bkd->bgk", query, core_key)
                if not self.config.query_conditioned:
                    scores = scores * 0
                weights = self._weights(scores * math.sqrt(self.config.query_dim) +
                                        core_prior[:, None], core_valid[:, None])
                weights = weights * support.float()
                support_mass = weights.sum(-1)
                exact = torch.einsum("bgk,bkd->bgd", weights, exact_value)
            else:
                support_mass = base.new_zeros(base.shape)
                exact = (exact_value.sum(1)[:, None] * 0).expand(
                    len(base), len(query), self.config.query_dim)
            active = torch.cat((available[:, None].expand(-1, len(query), -1),
                                (support_mass > 0)[..., None]), -1)
            products = torch.stack((query[None] * own[:, None], query[None] * weak,
                                    query[None] * core, query[None] * exact), 2)
            products = products * active[..., None]
            scores = products.sum(-1) * math.sqrt(self.config.query_dim)
            route = torch.softmax(scores.masked_fill(~active, -1e4), -1)
            hidden = self.graph_hidden(torch.cat((products.flatten(-2), route), -1))
            features = torch.cat((hidden, (base / 4).tanh()[..., None], pair,
                                  vote[..., None], (vote - base.sigmoid())[..., None],
                                  support_mass[..., None]), -1)
            delta = self.correction_decoder(features).squeeze(-1)
            return base + delta, delta, route, support_mass

    def forward(self, batch: Mapping[str, Tensor], *, use_weak_go: bool = True,
                use_core_go: bool = True, use_pp_context: bool = True,
                shuffle_go: bool = False, go_encoding: Tensor | None = None,
                return_details: bool = False):
        if go_encoding is not None and self.training:
            raise ValueError("learned GO encodings may only be cached in eval mode")
        go = self.encode_go() if go_encoding is None else go_encoding
        graph = self._prepare_graph_v083(
            batch, go, use_weak_go=use_weak_go, use_core_go=use_core_go,
            use_pp_context=use_pp_context, shuffle_go=shuffle_go)
        base = batch["base_logits"].float()
        if base.shape != (len(graph["own"]), self.num_task_go):
            raise ValueError("base logits must cover ALL task GOs in registry order")
        query = F.normalize(self.query(go).float(), dim=-1)
        names = ("own", "weak_key", "weak_value", "weak_prior", "weak_valid",
                 "core_key", "core_value", "exact_value", "core_prior",
                 "core_valid", "available")
        chunks, deltas, routes, masses = [], [], [], []
        for start in range(0, self.num_task_go, self.config.go_chunk):
            stop = min(start + self.config.go_chunk, self.num_task_go)
            support = graph["annotation"][:, start:stop][graph["safe_neighbor"]].transpose(1, 2)
            args = (query[start:stop], base[:, start:stop],
                    *(graph[name] for name in names), graph["pair"][:, start:stop],
                    graph["vote"][:, start:stop], support)
            if self.training and torch.is_grad_enabled() and self.config.activation_checkpointing:
                result = checkpoint(self._decode_chunk_v083, *args, use_reentrant=False)
            else:
                result = self._decode_chunk_v083(*args)
            chunks.append(result[0])
            if return_details:
                deltas.append(result[1])
                routes.append(result[2])
                masses.append(result[3])
        logits = torch.cat(chunks, 1)
        if return_details:
            return dict(logits=logits, graph_logits=None, delta=torch.cat(deltas, 1),
                        routing=torch.cat(routes, 1), core_vote=graph["vote"],
                        core_support_mass=torch.cat(masses, 1),
                        pp_message_norm=graph["pp_message_norm"],
                        pp_edge_count=graph["pp_edge_count"])
        return logits
