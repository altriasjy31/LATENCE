"""GO-conditioned graph readout with an independently supervised graph score.

Every task column is retained, including aliases of a shared ontology class.
Training and external inference call this same forward. Target labels and expert
probabilities are deliberately absent from the encoder API.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .full_task_model import FullTaskGraphModel, FullTaskModelConfig


@dataclass
class FullTaskModelConfigV081(FullTaskModelConfig):
    candidate_dropout: float = 0.15
    activation_checkpointing: bool = True
    attention_backend: str = "sdpa"
    query_conditioned: bool = True


class FullTaskGraphModelV081(FullTaskGraphModel):
    """Read weak GO and core neighbours separately for each output GO query.

    The auxiliary score sees vector interactions with graph states, including
    weak edge confidence. It cannot read the direct base-logit / candidate-pair /
    core-vote decoder features. The correction head can read both sources and
    starts at exactly zero; the auxiliary graph head starts with live gradients.
    """

    VERSION = "0.8.1"

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfigV081 | None = None):
        config = config or FullTaskModelConfigV081()
        super().__init__(protein_dim, ontology, config)
        if not 0 <= config.candidate_dropout <= 1:
            raise ValueError("candidate_dropout must be in [0,1]")
        if config.attention_backend not in ("sdpa", "math"):
            raise ValueError("attention_backend must be sdpa or math")
        # Remove the superseded modules rather than leaving unused parameters in
        # DDP. Ontology, protein, anchor and edge encoders remain shared.
        del self.weak_update, self.core_update, self.path_keys, self.decoder
        h, q, d = config.hidden_dim, config.query_dim, config.decoder_hidden
        self.own_value = nn.Linear(h, q, bias=False)
        self.weak_key = nn.Linear(h, q, bias=False)
        self.weak_value = nn.Linear(h, q, bias=False)
        self.core_key = nn.Linear(h, q, bias=False)
        self.core_value = nn.Linear(h, q, bias=False)
        self.graph_hidden = nn.Sequential(nn.Linear(3 * q + 3, d), nn.SiLU())
        self.graph_output = nn.Linear(d, 1)
        self.correction_decoder = nn.Sequential(nn.Linear(d + 8, d), nn.SiLU(),
                                                nn.Linear(d, 1))
        nn.init.zeros_(self.correction_decoder[-1].weight)
        nn.init.zeros_(self.correction_decoder[-1].bias)

    def _prepare_graph(self, batch: Mapping[str, Tensor], go: Tensor, *,
                       use_weak_go: bool, use_core_go: bool):
        own = self.protein_encoder(batch["protein_x"])
        b, h = own.shape
        idx = batch["candidate_go"].long()
        valid = (idx >= 0) & (idx < self.num_task_go)
        if not use_weak_go:
            valid = torch.zeros_like(valid)
        elif self.training and self.config.candidate_dropout:
            # Draw once outside activation checkpoints. The same label-independent
            # mask removes an edge from aggregation AND exact pair evidence.
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
        c = len(anchor_x)
        if c:
            degree = torch.bincount(edge[0], minlength=c).float().clamp_min(1)
            annotation = torch.sparse_coo_tensor(
                edge, degree[edge[0]].reciprocal(), (c, self.num_task_go),
                device=go.device).coalesce()
            with torch.autocast(device_type=go.device.type, enabled=False):
                anchor_labels = torch.sparse.mm(annotation, go.float())
            anchor = self.anchor_update(torch.cat((self.protein_encoder(anchor_x),
                                                   anchor_labels), -1))
        else:
            anchor = own.new_zeros(1, h)
        neighbor = batch["neighbor_index"].long()
        neighbor_valid = (neighbor >= 0) & (neighbor < c)
        if not use_core_go:
            neighbor_valid = torch.zeros_like(neighbor_valid)
        safe_neighbor = neighbor.clamp(0, max(c - 1, 0))
        nattr = batch["neighbor_attr"].float()
        core_prior = (self.core_edge_encoder(nattr).squeeze(-1).float() +
                      nattr[..., 0].clamp_min(1e-5).log())
        nweight = self._weights(core_prior, neighbor_valid)
        core_state = anchor[safe_neighbor]
        core_key = F.normalize(self.core_key(core_state).float(), dim=-1)
        core_value = F.normalize(self.core_value(core_state).float(), dim=-1)

        pair = own.new_zeros(b, self.num_task_go, 4, dtype=torch.float32)
        rr = torch.arange(b, device=own.device)[:, None].expand_as(idx)[valid]
        pair[rr, idx[valid], 0] = 1
        pair[rr, idx[valid], 1:] = evidence[valid]
        vote = own.new_zeros(b, self.num_task_go, dtype=torch.float32)
        if c and edge.numel():
            incidence = own.new_zeros(b, c, dtype=torch.float32)
            incidence.scatter_add_(1, safe_neighbor, nweight)
            binary_annotation = torch.sparse_coo_tensor(
                edge, torch.ones(edge.shape[1], device=go.device),
                (c, self.num_task_go), device=go.device).coalesce()
            with torch.autocast(device_type=go.device.type, enabled=False):
                vote = torch.sparse.mm(binary_annotation.transpose(0, 1),
                                        incidence.t()).t().clamp(0, 1)
        available = torch.stack((torch.ones(b, device=go.device, dtype=torch.bool),
                                 valid.any(-1), neighbor_valid.any(-1)), -1)
        own = F.normalize(self.own_value(own).float(), dim=-1)
        return (own, weak_key, weak_value, weak_prior, valid,
                core_key, core_value, core_prior, neighbor_valid, available,
                pair, vote)

    def _attend(self, query: Tensor, key: Tensor, value: Tensor,
                prior: Tensor, valid: Tensor) -> Tensor:
        """[Q,D] + [B,K,D] -> [B,Q,D], with safe empty neighbourhoods.

        Normalized queries/keys use cosine logits scaled by sqrt(D). SDPA can
        select a fused CUDA backend. The explicit implementation is a debugging
        fallback; either backend runs inside a recomputed GO chunk in training.
        """
        b, k, d = key.shape
        if k == 0:
            # Preserve the graph for empty inputs without inventing a neighbour.
            return (value.sum(1)[:, None, :] * 0).expand(b, len(query), d)
        value = value.float() * valid[..., None]
        if not self.config.query_conditioned:
            weight = self._weights(prior, valid)
            # Keep the optional pooling control DDP-safe without changing its
            # output: key projections have an explicit zero gradient.
            context = (weight[..., None] * value).sum(1) + key.sum(1) * 0
            return context[:, None, :].expand(b, len(query), d)
        mask = prior.float().masked_fill(~valid, -1e4)
        # Unlike -inf-only rows, this mask never produces undefined softmax.
        # All-invalid values are zero, so the resulting message is exactly zero.
        if self.config.attention_backend == "sdpa":
            q = (query.float() * d)[None, None].expand(b, 1, -1, -1)
            return F.scaled_dot_product_attention(
                q, key.float()[:, None], value[:, None],
                attn_mask=mask[:, None, None, :], dropout_p=0.0).squeeze(1)
        logits = torch.einsum("gd,bkd->bgk", query.float(), key.float())
        weight = torch.softmax(logits * math.sqrt(d) + mask[:, None, :], -1)
        return torch.einsum("bgk,bkd->bgd", weight, value)

    def _decode_chunk(self, query: Tensor, base: Tensor, own: Tensor,
                      weak_key: Tensor, weak_value: Tensor, weak_prior: Tensor,
                      weak_valid: Tensor, core_key: Tensor, core_value: Tensor,
                      core_prior: Tensor, core_valid: Tensor, available: Tensor,
                      pair: Tensor, vote: Tensor):
        with torch.autocast(device_type=base.device.type, enabled=False):
            weak = self._attend(query, weak_key, weak_value, weak_prior, weak_valid)
            core = self._attend(query, core_key, core_value, core_prior, core_valid)
            products = torch.stack((query[None] * own[:, None], query[None] * weak,
                                    query[None] * core), dim=2)
            products = products * available[:, None, :, None]
            scores = products.sum(-1) * math.sqrt(self.config.query_dim)
            route = torch.softmax(scores.masked_fill(~available[:, None, :], -1e4), -1)
            hidden = self.graph_hidden(torch.cat((products.flatten(-2), route), -1))
            graph_logits = self.graph_output(hidden).squeeze(-1)
            features = torch.cat((hidden, graph_logits[..., None],
                                  (base / 4).tanh()[..., None], pair.float(),
                                  vote[..., None], (vote - base.sigmoid())[..., None]), -1)
            delta = self.correction_decoder(features).squeeze(-1)
            return base + delta, graph_logits, delta, route

    def forward(self, batch: Mapping[str, Tensor], *, use_weak_go: bool = True,
                use_core_go: bool = True, go_encoding: Tensor | None = None,
                return_details: bool = False):
        if go_encoding is not None and self.training:
            raise ValueError("learned GO encodings may only be cached in eval mode")
        go = self.encode_go() if go_encoding is None else go_encoding
        graph = self._prepare_graph(batch, go, use_weak_go=use_weak_go,
                                    use_core_go=use_core_go)
        base = batch["base_logits"].float()
        if base.shape != (len(graph[0]), self.num_task_go):
            raise ValueError("base logits must cover ALL task GOs in registry order")
        query = F.normalize(self.query(go).float(), dim=-1)
        chunks, auxiliary, deltas, routes = [], [], [], []
        for start in range(0, self.num_task_go, self.config.go_chunk):
            stop = min(start + self.config.go_chunk, self.num_task_go)
            args = (query[start:stop], base[:, start:stop], *graph[:10],
                    graph[10][:, start:stop], graph[11][:, start:stop])
            if self.training and torch.is_grad_enabled() and self.config.activation_checkpointing:
                # Checkpoint the *whole decoder chunk*, not merely attention.
                # This prevents retaining [B,all_GO,D] intermediate activations
                # for each path or [B,all_GO,candidate_K] attention probabilities.
                result = checkpoint(self._decode_chunk, *args, use_reentrant=False)
            else:
                result = self._decode_chunk(*args)
            chunks.append(result[0])
            if return_details:
                auxiliary.append(result[1])
                deltas.append(result[2])
                routes.append(result[3])
        logits = torch.cat(chunks, 1)
        if return_details:
            return {"logits": logits, "graph_logits": torch.cat(auxiliary, 1),
                    "delta": torch.cat(deltas, 1), "routing": torch.cat(routes, 1),
                    "core_vote": graph[11]}
        return logits
