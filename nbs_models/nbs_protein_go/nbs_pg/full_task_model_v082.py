"""Complete GO classification with a selective, graph-conditioned matcher.

Both heads share the ontology, weak→GO and weak→core→GO encoders. Classification
has an independent affine row for every original task column. Matching refines
only a fixed-size, label-independent subset; all other outputs are exactly the
classification outputs. Training and external inference use this same forward.
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
class FullTaskModelConfigV082(FullTaskModelConfigV081):
    query_budget: int = 256
    selector_high_fraction: float = 0.375
    selector_uncertain_fraction: float = 0.25
    selector_weak_fraction: float = 0.1875
    selector_core_fraction: float = 0.125
    selector_eval_seed: int = 8082


class FullTaskGraphModelV082(FullTaskGraphModelV081):
    VERSION = "0.8.2"

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfigV082 | None = None, *,
                 variant: str = "dual"):
        config = config or FullTaskModelConfigV082()
        if variant not in ("classifier", "dual"):
            raise ValueError("v082 model variant must be classifier or dual")
        fractions = (config.selector_high_fraction, config.selector_uncertain_fraction,
                     config.selector_weak_fraction, config.selector_core_fraction)
        if config.query_budget < 1 or any(not 0 <= x <= 1 for x in fractions):
            raise ValueError("query_budget must be positive and selector fractions in [0,1]")
        if sum(fractions) > 1:
            raise ValueError("selector fractions must sum to at most one")
        super().__init__(protein_dim, ontology, config)
        self.variant = variant
        # The old all-GO matcher and auxiliary head are not part of this model.
        del self.graph_hidden, self.graph_output, self.correction_decoder
        h, q, d = config.hidden_dim, config.query_dim, config.decoder_hidden
        self.classification_hidden = nn.Sequential(
            nn.Linear(3 * q + 3, h), nn.SiLU(), nn.LayerNorm(h), nn.Dropout(config.dropout))
        self.classification_output = nn.Linear(h, self.num_task_go)
        nn.init.zeros_(self.classification_output.weight)
        nn.init.zeros_(self.classification_output.bias)
        if variant == "dual":
            self.match_hidden = nn.Sequential(nn.Linear(3 * q + 10, d), nn.SiLU())
            self.match_output = nn.Linear(d, 1)
            nn.init.zeros_(self.match_output.weight)
            nn.init.zeros_(self.match_output.bias)
        else:
            # Pooling uses values and edge priors, never query/key projections.
            # Keep the inherited graph preparation API without unused parameters.
            del self.query
            self.weak_key = nn.Identity()
            self.core_key = nn.Identity()
        generator = torch.Generator().manual_seed(config.selector_eval_seed)
        self.register_buffer("selector_eval_scores", torch.rand(self.num_task_go,
                             generator=generator), persistent=False)

    @torch.no_grad()
    def select_queries(self, classification: Tensor, pair: Tensor, vote: Tensor):
        """Select by predictions/visible graph only; targets are not arguments.

        Quotas are applied in order with duplicates removed. Missing weak/core
        evidence leaves spare capacity that is filled by classification rank.
        The exploration share rotates using the saved torch RNG during training;
        eval uses a fixed seeded permutation independent of batching.
        """
        b, g = classification.shape
        budget = min(self.config.query_budget, g)
        selected = torch.zeros(b, g, dtype=torch.bool, device=classification.device)
        fractions = (self.config.selector_high_fraction,
                     self.config.selector_uncertain_fraction,
                     self.config.selector_weak_fraction,
                     self.config.selector_core_fraction)
        counts = [int(budget * fraction) for fraction in fractions]
        counts.append(budget - sum(counts))
        all_valid = torch.ones_like(selected)
        exploration = (torch.rand_like(classification) if self.training else
                       self.selector_eval_scores[None].expand(b, -1))
        score_sets = ((classification, all_valid), (-classification.abs(), all_valid),
                      (pair[..., 1], pair[..., 0] > 0), (vote, vote > 0),
                      (exploration, all_valid))
        for count, (score, eligible) in zip(counts, score_sets):
            if count == 0:
                continue
            valid = eligible & ~selected
            # Stable sort gives original registry order as the tie-breaker.
            indices = score.masked_fill(~valid, -torch.inf).argsort(
                dim=1, descending=True, stable=True)[:, :count]
            selected |= torch.zeros_like(selected).scatter_(1, indices, valid.gather(1, indices))
        remaining = budget - selected.sum(1)
        if bool((remaining > 0).any()):
            order = classification.masked_fill(selected, -torch.inf).argsort(
                dim=1, descending=True, stable=True)
            fill = torch.arange(g, device=classification.device)[None] < remaining[:, None]
            additional = torch.zeros_like(selected).scatter_(1, order, fill)
            selected |= additional
        # nonzero is ordered by protein then GO, retaining immutable task columns.
        indices = selected.nonzero(as_tuple=False)[:, 1].reshape(b, budget)
        return indices, selected

    def _attend_selected(self, query: Tensor, key: Tensor, value: Tensor,
                         prior: Tensor, valid: Tensor) -> Tensor:
        """[B,Q,D] queries; no [B,all_GO,K] attention is constructed."""
        b, k, d = key.shape
        if k == 0:
            return (value.sum(1)[:, None] * 0).expand(b, query.shape[1], d)
        value = value.float() * valid[..., None]
        if not self.config.query_conditioned:
            context = (self._weights(prior, valid)[..., None] * value).sum(1)
            return (context + key.sum(1) * 0)[:, None].expand(b, query.shape[1], d)
        mask = prior.float().masked_fill(~valid, -1e4)
        if self.config.attention_backend == "sdpa":
            return F.scaled_dot_product_attention(
                (query.float() * d)[:, None], key.float()[:, None], value[:, None],
                attn_mask=mask[:, None, None, :], dropout_p=0.0).squeeze(1)
        score = torch.einsum("bqd,bkd->bqk", query.float(), key.float()) * math.sqrt(d)
        weight = torch.softmax(score + mask[:, None], -1)
        return torch.einsum("bqk,bkd->bqd", weight, value)

    def _match_selected(self, query: Tensor, classification: Tensor, own: Tensor,
                        weak_key: Tensor, weak_value: Tensor, weak_prior: Tensor,
                        weak_valid: Tensor, core_key: Tensor, core_value: Tensor,
                        core_prior: Tensor, core_valid: Tensor, available: Tensor,
                        pair: Tensor, vote: Tensor):
        with torch.autocast(device_type=classification.device.type, enabled=False):
            weak = self._attend_selected(query, weak_key, weak_value, weak_prior, weak_valid)
            core = self._attend_selected(query, core_key, core_value, core_prior, core_valid)
            products = torch.stack((query * own[:, None], query * weak, query * core), 2)
            products = products * available[:, None, :, None]
            scores = products.sum(-1) * math.sqrt(self.config.query_dim)
            route = torch.softmax(scores.masked_fill(~available[:, None], -1e4), -1)
            features = torch.cat((products.flatten(-2), route,
                                  (classification / 4).tanh()[..., None], pair.float(),
                                  vote[..., None],
                                  (vote - classification.sigmoid())[..., None]), -1)
            return self.match_output(self.match_hidden(features)).squeeze(-1), route

    def forward(self, batch: Mapping[str, Tensor], *, use_weak_go: bool = True,
                use_core_go: bool = True, go_encoding: Tensor | None = None,
                return_details: bool = False):
        if go_encoding is not None and self.training:
            raise ValueError("learned GO encodings may only be cached in eval mode")
        go = self.encode_go() if go_encoding is None else go_encoding
        graph = self._prepare_graph(batch, go, use_weak_go=use_weak_go,
                                    use_core_go=use_core_go)
        own, _, weak_value, weak_prior, weak_valid = graph[:5]
        core_value, core_prior, core_valid, available = graph[6:10]
        pair, vote = graph[10:]
        base = batch["base_logits"].float()
        if base.shape != (len(own), self.num_task_go):
            raise ValueError("base logits must cover ALL task GOs in registry order")
        with torch.autocast(device_type=base.device.type, enabled=False):
            weak = (self._weights(weak_prior, weak_valid)[..., None] * weak_value).sum(1)
            core = (self._weights(core_prior, core_valid)[..., None] * core_value).sum(1)
            pooled = torch.cat((own, weak, core, available.float()), -1)
            classification = base + self.classification_output(self.classification_hidden(pooled))
        if self.variant == "classifier":
            indices = torch.empty(len(base), 0, dtype=torch.long, device=base.device)
            mask = torch.zeros_like(base, dtype=torch.bool)
            match_delta = torch.zeros_like(base)
            logits = classification
            routes = base.new_empty(len(base), 0, 3)
        else:
            indices, mask = self.select_queries(classification.detach(), pair.detach(), vote.detach())
            query = F.normalize(self.query(go).float(), dim=-1)[indices]
            args = (query, classification.gather(1, indices), *graph[:10],
                    pair.gather(1, indices[..., None].expand(-1, -1, 4)), vote.gather(1, indices))
            if self.training and torch.is_grad_enabled() and self.config.activation_checkpointing:
                delta, routes = checkpoint(self._match_selected, *args, use_reentrant=False)
            else:
                delta, routes = self._match_selected(*args)
            match_delta = torch.zeros_like(classification).scatter(1, indices, delta)
            logits = classification + match_delta
        if return_details:
            return {"logits": logits, "classification_logits": classification,
                    "selected_mask": mask, "selected_indices": indices,
                    "delta_logits": logits - base, "delta": logits - base,
                    "match_delta_logits": match_delta, "core_vote": vote,
                    "routing": routes, "graph_logits": None}
        return logits
