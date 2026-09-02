from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import NBSConfig
from .ops import segment_mean, validate_local_index
from .types import NBSMatchOutput, NBSNeighborhoodHierarchy, NBSQueryCondition


class NBSGatedDeltaAttnRes(nn.Module):
    """NBS-GDAR: query-conditioned routing plus base-logit residual refinement.

    The graph context correction is factorized and never materializes a
    ``[B, N, D]`` tensor. When base logits are supplied, NBS initializes as an
    exact baseline and learns a gated graph delta rather than replacing it.
    """

    INDUCTIVE_ROUTING_API_VERSION = 1

    def __init__(self, config: NBSConfig, num_sources: int) -> None:
        super().__init__()
        if num_sources <= 0:
            raise ValueError("NBS-GDAR needs at least one source")
        self.config = config
        self.num_sources = num_sources
        d = config.hidden_dim
        self.source_embedding = nn.Parameter(torch.zeros(num_sources, d))
        self.gate_net = nn.Sequential(
            nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.delta_norm = nn.LayerNorm(d)
        self.route_query = nn.Linear(d, d, bias=False)
        self.route_delta = nn.Linear(d, d, bias=False)
        self.route_score = nn.Linear(d, 1, bias=False)
        self.null_logit = nn.Parameter(torch.tensor(float(config.null_logit_init)))
        self.query_scale = nn.Parameter(torch.tensor(float(config.query_residual_init)))
        self.context_scale = nn.Parameter(torch.tensor(float(config.context_residual_init)))
        self.graph_delta_scale = nn.Parameter(
            torch.tensor(float(config.graph_delta_scale_init))
        )
        # Four stable gate channels are always used.  Channel 2 is either
        # first-stage candidate evidence, zero (student-only), or expert
        # probability for an explicitly requested legacy ablation.
        self.delta_gate = nn.Sequential(
            nn.Linear(4, config.delta_gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.delta_gate_hidden_dim, 1),
        )
        self.candidate_evidence_residual = nn.Sequential(
            nn.Linear(config.candidate_evidence_dim, config.candidate_evidence_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.candidate_evidence_hidden_dim, 1),
        )
        self.candidate_evidence_scale = nn.Parameter(
            torch.tensor(float(config.candidate_evidence_residual_scale_init))
        )

        nn.init.zeros_(self.source_embedding)
        nn.init.zeros_(self.route_score.weight)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -2.0)
        nn.init.zeros_(self.delta_gate[-1].weight)
        nn.init.constant_(self.delta_gate[-1].bias, float(config.delta_gate_bias_init))
        nn.init.zeros_(self.candidate_evidence_residual[-1].weight)
        nn.init.zeros_(self.candidate_evidence_residual[-1].bias)

    def _route(
        self,
        hierarchy: NBSNeighborhoodHierarchy,
        condition: NBSQueryCondition,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        hierarchy.validate()
        if hierarchy.source_contexts.size(0) != self.num_sources:
            raise ValueError(
                f"Expected {self.num_sources} sources, got {hierarchy.source_contexts.size(0)}"
            )
        validate_local_index(condition.seed_index, hierarchy.final_context.size(0), "seed_index")
        query_deltas = torch.stack(
            [
                segment_mean(
                    hierarchy.source_contexts[s, condition.seed_index],
                    condition.seed_query_index,
                    condition.num_queries,
                )
                for s in range(self.num_sources)
            ],
            dim=1,
        )
        q = condition.base_query
        source_emb = self.source_embedding.unsqueeze(0).expand(q.size(0), -1, -1)
        q_expand = q.unsqueeze(1).expand(-1, self.num_sources, -1)
        gates = torch.sigmoid(
            self.gate_net(torch.cat([q_expand, query_deltas, source_emb], dim=-1))
        )
        gated_delta = gates * query_deltas
        route_hidden = torch.tanh(
            self.route_query(q).unsqueeze(1)
            + self.route_delta(self.delta_norm(gated_delta))
            + source_emb
        )
        logits = self.route_score(route_hidden).squeeze(-1) / self.config.route_temperature
        if self.config.use_null_source:
            all_weights = torch.softmax(
                torch.cat([self.null_logit.expand(q.size(0), 1), logits], dim=1),
                dim=1,
            )
            weights = all_weights[:, 1:]
            null_weight = all_weights[:, 0]
        else:
            weights = torch.softmax(logits, dim=1)
            null_weight = torch.zeros(q.size(0), device=q.device, dtype=q.dtype)

        selected_query_delta = torch.einsum("bs,bsd->bd", weights, gated_delta)
        q_out = q + torch.tanh(self.query_scale) * selected_query_delta
        return q_out, weights, gates, null_weight

    def _score_slice(
        self,
        q_out: Tensor,
        weights: Tensor,
        gates: Tensor,
        final_context: Tensor,
        source_contexts: Tensor,
    ) -> Tensor:
        graph_score = torch.einsum("bd,nd->bn", q_out, final_context)
        if not self.config.use_context_routing:
            return graph_score
        routed_query = weights.unsqueeze(-1) * gates * q_out.unsqueeze(1)
        correction = torch.einsum("bsd,snd->bn", routed_query, source_contexts)
        return graph_score + torch.tanh(self.context_scale) * correction

    def _score_candidates(
        self,
        hierarchy: NBSNeighborhoodHierarchy,
        condition: NBSQueryCondition,
        q_out: Tensor,
        weights: Tensor,
        gates: Tensor,
    ) -> Tensor:
        if condition.candidate_index is None:
            candidate_index = torch.arange(
                hierarchy.final_context.size(0), device=hierarchy.final_context.device
            )
        else:
            candidate_index = condition.candidate_index
            validate_local_index(candidate_index, hierarchy.final_context.size(0), "candidate_index")

        chunk = self.config.score_chunk_size
        if chunk is None or candidate_index.numel() <= chunk:
            return self._score_slice(
                q_out,
                weights,
                gates,
                hierarchy.final_context[candidate_index],
                hierarchy.source_contexts[:, candidate_index],
            )
        outputs: List[Tensor] = []
        for start in range(0, candidate_index.numel(), chunk):
            idx = candidate_index[start : start + chunk]
            outputs.append(
                self._score_slice(
                    q_out,
                    weights,
                    gates,
                    hierarchy.final_context[idx],
                    hierarchy.source_contexts[:, idx],
                )
            )
        return torch.cat(outputs, dim=1)


    def _encode_candidate_evidence(self, evidence: Tensor, base_prob: Tensor) -> Tensor:
        """Compress candidate evidence to one bounded gate channel.

        A two-dimensional tensor is treated as an already-compressed scalar
        feature.  The production three-column LATENCE tensor starts exactly
        from reciprocal rank, while a zero-initialized residual can learn from
        backbone probability and selector score without duplicating the base
        probability channel at initialization.
        """
        evidence = evidence.to(base_prob.device, base_prob.dtype)
        if evidence.dim() == 2:
            if evidence.shape != base_prob.shape:
                raise ValueError("scalar candidate_evidence must align with base_logits")
            return evidence.clamp(0.0, 1.0)
        if evidence.dim() != 3 or evidence.shape[:2] != base_prob.shape:
            raise ValueError(
                "candidate_evidence must be [Q,C] or [Q,C,F] aligned with base_logits"
            )
        if evidence.size(-1) != self.config.candidate_evidence_dim:
            raise ValueError(
                f"candidate_evidence feature dim {evidence.size(-1)} != "
                f"configured {self.config.candidate_evidence_dim}"
            )
        initial_channel = 2 if evidence.size(-1) >= 3 else evidence.size(-1) - 1
        initial = evidence[..., initial_channel].clamp(1e-5, 1.0 - 1e-5)
        initial_logit = torch.logit(initial)
        residual = self.candidate_evidence_residual(evidence).squeeze(-1)
        return torch.sigmoid(
            initial_logit + torch.tanh(self.candidate_evidence_scale) * residual
        )

    def _refine_base_logits(
        self,
        graph_logits: Tensor,
        condition: NBSQueryCondition,
    ) -> Tuple[Tensor, Tensor]:
        base = condition.base_logits
        if base is None or not self.config.use_base_logit_residual:
            return graph_logits, torch.ones_like(graph_logits)
        base = base.to(graph_logits.device, graph_logits.dtype)
        if base.shape != graph_logits.shape:
            raise ValueError(
                f"base_logits shape {tuple(base.shape)} != graph logits {tuple(graph_logits.shape)}"
            )
        if self.config.use_candidate_delta_gate:
            base_prob = torch.sigmoid(base)
            mode = self.config.delta_gate_feature_mode
            if mode == "student_candidate":
                if condition.candidate_evidence is None:
                    evidence = torch.zeros_like(base_prob)
                else:
                    evidence = self._encode_candidate_evidence(
                        condition.candidate_evidence, base_prob
                    )
            elif mode == "student_only":
                evidence = torch.zeros_like(base_prob)
            elif mode == "legacy_expert":
                if condition.expert_prob is None:
                    evidence = torch.zeros_like(base_prob)
                else:
                    evidence = condition.expert_prob.to(base_prob.device, base_prob.dtype)
                    if evidence.shape != base_prob.shape:
                        raise ValueError("expert_prob must align with base_logits")
            else:  # validated by NBSConfig
                raise ValueError(f"unsupported delta_gate_feature_mode: {mode}")
            if condition.query_go_frequency is None:
                frequency = torch.zeros(base.size(0), 1, device=base.device, dtype=base.dtype)
            else:
                frequency = condition.query_go_frequency.to(base.device, base.dtype).reshape(base.size(0), -1)
                if frequency.size(1) != 1:
                    frequency = frequency.mean(dim=1, keepdim=True)
                frequency = torch.log1p(frequency.clamp_min(0.0))
                frequency = frequency / (1.0 + frequency)
            frequency = frequency.expand_as(base)
            features = torch.stack(
                [base_prob, evidence, torch.tanh(graph_logits), frequency], dim=-1
            )
            gate = torch.sigmoid(self.delta_gate(features).squeeze(-1))
        else:
            gate = torch.ones_like(graph_logits)
        correction = torch.tanh(self.graph_delta_scale) * gate * graph_logits
        return base + correction, gate

    def forward(
        self,
        hierarchy: NBSNeighborhoodHierarchy,
        condition: NBSQueryCondition,
        return_aux: bool = False,
        *,
        routing_hierarchy: Optional[NBSNeighborhoodHierarchy] = None,
    ) -> NBSMatchOutput:
        # Training uses one hierarchy for both support-conditioned routing and
        # candidate scoring.  Isolated inductive inference is different: query
        # seeds remain indices in the sampled support graph, while the scored
        # candidates live in a separate external-protein index space.  Keeping
        # these hierarchies explicit prevents support seed indices from being
        # interpreted as external candidate indices and preserves batching
        # invariance for independent-test proteins.
        score_hierarchy = hierarchy
        route_hierarchy = (
            hierarchy if routing_hierarchy is None else routing_hierarchy
        )
        score_hierarchy.validate()
        route_hierarchy.validate()
        if score_hierarchy.source_contexts.size(0) != self.num_sources:
            raise ValueError(
                f"Expected {self.num_sources} scoring sources, got "
                f"{score_hierarchy.source_contexts.size(0)}"
            )
        if score_hierarchy.source_names != route_hierarchy.source_names:
            raise ValueError(
                "routing and scoring hierarchies must use the same source order"
            )
        q_out, weights, gates, null_weight = self._route(
            route_hierarchy, condition
        )
        graph_logits = self._score_candidates(
            score_hierarchy, condition, q_out, weights, gates
        )
        logits, delta_gate = self._refine_base_logits(graph_logits, condition)

        for name, value in (
            ("labels", condition.labels),
            ("mask", condition.mask),
            ("confidence", condition.confidence),
            ("pseudo_mask", condition.pseudo_mask),
            ("supervision_weight", condition.supervision_weight),
        ):
            if value is not None and value.shape != logits.shape:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} != logits shape {tuple(logits.shape)}"
                )

        auxiliary: Optional[Dict[str, Tensor]] = None
        if return_aux:
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
            auxiliary = {
                "source_weights": weights,
                "null_weight": null_weight,
                "dimension_gates": gates,
                "routing_entropy": entropy,
                "query_scale": torch.tanh(self.query_scale),
                "context_scale": torch.tanh(self.context_scale),
                "graph_delta_scale": torch.tanh(self.graph_delta_scale),
                "query_embedding": q_out,
                "graph_logits": graph_logits,
                "delta_gate": delta_gate,
                "applied_graph_delta": logits - (
                    condition.base_logits.to(logits.device, logits.dtype)
                    if condition.base_logits is not None
                    and self.config.use_base_logit_residual
                    else torch.zeros_like(logits)
                ),
            }
            if condition.base_logits is not None:
                auxiliary["base_logits"] = condition.base_logits.to(logits.device, logits.dtype)
            if condition.auxiliary:
                auxiliary.update({f"query/{k}": v for k, v in condition.auxiliary.items()})

        return NBSMatchOutput(
            logits=logits,
            labels=condition.labels,
            mask=condition.mask,
            confidence=condition.confidence,
            pseudo_mask=condition.pseudo_mask,
            supervision_weight=condition.supervision_weight,
            auxiliary=auxiliary,
        )
