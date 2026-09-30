"""Absolute graph prediction and coherent protein-level evidence controls.

The residual/no-source-dropout mode is exactly the v086 model.  Direct mode
retains the same two-layer graph and GO matcher, but its readout has no dense
backbone channels or backbone skip.  Backbone-generated candidate evidence is
still an input; this is deliberately not a claim of a backbone-free model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .full_task_model_v086 import FullTaskGraphModelV086, FullTaskModelConfigV086


@dataclass
class FullTaskModelConfigV087(FullTaskModelConfigV086):
    prediction_mode: str = "residual"
    source_dropout_query_candidate: float = 0.0
    source_dropout_neighbor_candidate: float = 0.0
    source_dropout_neighbor_pseudo: float = 0.0
    source_dropout_seed: int = 8087
    direct_bias_init: float = -4.0


class FullTaskGraphModelV087(FullTaskGraphModelV086):
    VERSION = "0.8.7"
    EVIDENCE_SOURCES = ("query_candidate", "neighbor_candidate", "neighbor_pseudo")

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfigV087 | None = None):
        config = config or FullTaskModelConfigV087()
        if config.prediction_mode not in {"residual", "direct"}:
            raise ValueError("prediction_mode must be residual or direct")
        for source in self.EVIDENCE_SOURCES:
            value = getattr(config, "source_dropout_" + source)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("source dropout probabilities must be finite and in [0,1]")
        if isinstance(config.source_dropout_seed, bool) or not isinstance(config.source_dropout_seed, int):
            raise ValueError("source_dropout_seed must be an integer")
        if (isinstance(config.direct_bias_init, bool) or
                not isinstance(config.direct_bias_init, (int, float)) or
                not math.isfinite(config.direct_bias_init)):
            raise ValueError("direct_bias_init must be finite")
        super().__init__(protein_dim, ontology, config)
        if config.prediction_mode == "direct":
            d = config.decoder_hidden
            # Four candidate-pair features, exact core vote and support mass.
            # No tanh(base/4), vote-sigmoid(base), or final base-logit skip.
            self.correction_decoder = nn.Sequential(nn.Linear(d + 6, d), nn.SiLU(),
                                                    nn.Linear(d, 1))
            nn.init.normal_(self.correction_decoder[-1].weight, std=0.01)
            nn.init.zeros_(self.correction_decoder[-1].bias)
            # Fixed initialization, not an estimated biological class prior.
            # Keep immutable task columns, including canonical/alias columns.
            self.direct_go_bias = nn.Parameter(torch.full(
                (self.num_task_go,), float(config.direct_bias_init)))

    def _source_keep_mask(self, count: int, device: torch.device, source: str,
                          *, enabled: bool = True) -> Tensor:
        """One independent stateless stream per source; no global RNG draws."""
        if source not in self.EVIDENCE_SOURCES:
            raise ValueError("unknown evidence source")
        if not enabled:
            return torch.zeros(count, dtype=torch.bool, device=device)
        probability = getattr(self.config, "source_dropout_" + source)
        if not self.training or probability == 0:
            return torch.ones(count, dtype=torch.bool, device=device)
        if self._encoder_step is None:
            raise RuntimeError("training source dropout requires set_encoder_step(step, rank)")
        value = (f"{self.config.source_dropout_seed}:{self._encoder_step}:"
                 f"{self._encoder_rank}:{source}").encode()
        seed = int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.rand(count, device=device, generator=generator) >= probability

    def _source_batch(self, batch: Mapping[str, Tensor], *, use_weak_go: bool,
                      use_query_candidate: bool, use_neighbor_candidate: bool,
                      use_neighbor_pseudo: bool):
        """Copy only edited tensors; bind a source decision to its local node.

        Query candidates are removed from both the sampled encoder and direct
        query readout.  Gold, graph topology and PU loss-support tensors are
        untouched.  This runs before any decoder activation checkpoint.
        """
        sampled = batch["sampled_candidate_go"]
        n, device = len(batch["sampled_protein_x"]), sampled.device
        seed = batch["sampled_seed_index"].long()
        if sampled.ndim != 2 or len(sampled) != n:
            raise ValueError("sampled candidate rows must match sampled proteins")
        if (seed.shape != (len(batch["protein_x"]),) or
                (seed.numel() and (seed.min() < 0 or seed.max() >= n)) or
                len(seed.unique()) != len(seed)):
            raise ValueError("sampled supervision seed indices must be unique valid local nodes")
        query_nodes = torch.zeros(n, dtype=torch.bool, device=device)
        query_nodes[seed] = True
        flags = {"query_candidate": use_query_candidate,
                 "neighbor_candidate": use_neighbor_candidate,
                 "neighbor_pseudo": use_neighbor_pseudo}
        keep = {source: self._source_keep_mask(n, device, source,
                    enabled=use_weak_go and flags[source]) for source in self.EVIDENCE_SOURCES}
        candidate_keep = torch.where(query_nodes, keep["query_candidate"],
                                     keep["neighbor_candidate"])
        result = dict(batch)
        if not candidate_keep.all():
            result["sampled_candidate_go"] = sampled.masked_fill(~candidate_keep[:, None], -1)
            result["sampled_candidate_attr"] = batch["sampled_candidate_attr"].masked_fill(
                ~candidate_keep[:, None, None], 0)
        if not keep["query_candidate"][seed].all():
            result["candidate_go"] = batch["candidate_go"].masked_fill(
                ~keep["query_candidate"][seed, None], -1)
            result["candidate_attr"] = batch["candidate_attr"].masked_fill(
                ~keep["query_candidate"][seed, None, None], 0)
        pseudo = batch["sampled_pseudo_edge"].long()
        if (pseudo.ndim != 2 or pseudo.shape[0] != 2 or
                (pseudo.numel() and (pseudo[0].min() < 0 or pseudo[0].max() >= n))):
            raise ValueError("sampled pseudo edges must have valid local protein indices")
        # Current data already excludes all seed labels.  Fail closed if a
        # future loader violates that contract instead of exposing supervision.
        if pseudo.numel() and query_nodes[pseudo[0]].any():
            raise ValueError("supervision seed pseudo labels must already be excluded")
        pseudo_keep = keep["neighbor_pseudo"][pseudo[0]]
        if not pseudo_keep.all():
            result["sampled_pseudo_edge"] = pseudo[:, pseudo_keep]

        sampled_available = ((sampled >= 0) & (sampled < self.num_task_go)).any(1)
        query_available = sampled_available.clone()
        direct = batch["candidate_go"]
        query_available[seed] |= ((direct >= 0) & (direct < self.num_task_go)).any(1)
        pseudo_available = torch.zeros(n, dtype=torch.bool, device=device)
        pseudo_available[pseudo[0]] = True
        available = {"query_candidate": query_available & query_nodes,
                     "neighbor_candidate": sampled_available & ~query_nodes,
                     "neighbor_pseudo": pseudo_available & ~query_nodes}
        diagnostics = {}
        for source in self.EVIDENCE_SOURCES:
            count = available[source].sum().float()
            retained = (available[source] & keep[source]).sum().float()
            prefix = "source_" + source
            diagnostics[prefix + "_input_nodes"] = count
            diagnostics[prefix + "_kept_nodes"] = retained
            diagnostics[prefix + "_removed_fraction"] = (count - retained) / count.clamp_min(1)
        return result, diagnostics

    def _decode_chunk_v083(self, query, base, own, weak_key, weak_value, weak_prior,
                          weak_valid, core_key, core_value, exact_value, core_prior,
                          core_valid, available, pair, vote, support):
        if self.config.prediction_mode == "residual":
            return super()._decode_chunk_v083(query, base, own, weak_key, weak_value,
                weak_prior, weak_valid, core_key, core_value, exact_value, core_prior,
                core_valid, available, pair, vote, support)
        # ``base`` is part of the inherited chunk API, but its values do not
        # enter this branch. Allocate from graph-derived tensors exclusively.
        with torch.autocast(device_type=query.device.type, enabled=False):
            weak = self._attend(query, weak_key, weak_value, weak_prior, weak_valid)
            core = self._attend(query, core_key, core_value, core_prior, core_valid)
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
                support_mass = vote.new_zeros(vote.shape)
                exact = (exact_value.sum(1)[:, None] * 0).expand(
                    len(own), len(query), self.config.query_dim)
            active = torch.cat((available[:, None].expand(-1, len(query), -1),
                                (support_mass > 0)[..., None]), -1)
            products = torch.stack((query[None] * own[:, None], query[None] * weak,
                                    query[None] * core, query[None] * exact), 2)
            products = products * active[..., None]
            scores = products.sum(-1) * math.sqrt(self.config.query_dim)
            route = torch.softmax(scores.masked_fill(~active, -1e4), -1)
            hidden = self.graph_hidden(torch.cat((products.flatten(-2), route), -1))
            features = torch.cat((hidden, pair, vote[..., None],
                                  support_mass[..., None]), -1)
            absolute = self.correction_decoder(features).squeeze(-1)
            return absolute, absolute, route, support_mass

    def forward(self, batch: Mapping[str, Tensor], *, use_weak_go: bool = True,
                use_core_go: bool = True, use_pp_context: bool = True,
                shuffle_go: bool = False, go_encoding: Tensor | None = None,
                return_details: bool = False, use_query_candidate: bool = True,
                use_neighbor_candidate: bool = True, use_neighbor_pseudo: bool = True):
        masked, diagnostics = self._source_batch(batch, use_weak_go=use_weak_go,
            use_query_candidate=use_query_candidate,
            use_neighbor_candidate=use_neighbor_candidate,
            use_neighbor_pseudo=use_neighbor_pseudo)
        weak_enabled = use_weak_go and any((use_query_candidate,
                                           use_neighbor_candidate, use_neighbor_pseudo))
        result = super().forward(masked, use_weak_go=weak_enabled,
            use_core_go=use_core_go, use_pp_context=use_pp_context,
            shuffle_go=shuffle_go, go_encoding=go_encoding, return_details=return_details)
        if self.config.prediction_mode == "direct":
            if return_details:
                result["logits"] = result["logits"] + self.direct_go_bias.float()[None]
                result["graph_logits"] = result["logits"]
                result["delta"] = None  # An absolute readout has no residual delta.
            else:
                result = result + self.direct_go_bias.float()[None]
        if return_details:
            result["source_diagnostics"] = diagnostics
        return result
