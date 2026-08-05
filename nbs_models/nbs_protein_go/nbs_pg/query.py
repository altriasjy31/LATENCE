from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .config import NBSConfig
from .layers import NBSMLP
from .ops import segment_attention_pool, segment_mean, validate_local_index
from .types import (
    BoxGOEncoding,
    NBSGOBoxCache,
    NBSQueryCondition,
    ProteinGOQueryBatch,
)


class NBSProteinGOQueryEncoder(nn.Module):
    """Build GO-conditioned queries from protein seeds and box-aware GO states."""

    def __init__(self, config: NBSConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.external_proj = (
            nn.Linear(config.external_query_dim, d)
            if config.external_query_dim > 0
            else None
        )
        self.go_attention_seed = nn.Linear(d, d, bias=False)
        self.go_attention_key = nn.Linear(d, d, bias=False)
        self.semantic_fusion = NBSMLP(
            5 * d,
            2 * d,
            d,
            activation=config.activation,
            dropout=config.residual_dropout,
        )
        self.hierarchy_fusion = NBSMLP(
            3 * d,
            2 * d,
            d,
            activation=config.activation,
            dropout=config.residual_dropout,
        )
        self.semantic_gate = nn.Sequential(
            nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.hierarchy_gate = nn.Sequential(
            nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d)
        )
        nn.init.zeros_(self.semantic_gate[-1].weight)
        nn.init.constant_(
            self.semantic_gate[-1].bias,
            float(config.query_semantic_gate_bias_init),
        )
        nn.init.zeros_(self.hierarchy_gate[-1].weight)
        nn.init.constant_(
            self.hierarchy_gate[-1].bias,
            float(config.query_hierarchy_gate_bias_init),
        )
        self.external_gate = nn.Parameter(torch.tensor(-2.0))
        self.frequency_proj = NBSMLP(1, d, d, config.activation, config.residual_dropout)
        self.frequency_gate = nn.Parameter(torch.tensor(-2.0))
        self.output_norm = nn.LayerNorm(d)

    def _resolve_go_occurrences(
        self,
        go_context: Tensor,
        local_box: BoxGOEncoding,
        query: ProteinGOQueryBatch,
        global_cache: Optional[NBSGOBoxCache],
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        if query.query_go_index is not None:
            idx = query.query_go_index
            validate_local_index(idx, go_context.size(0), "query_go_index")
            return (
                local_box.static[idx],
                local_box.semantic[idx],
                local_box.hierarchy[idx],
                go_context[idx],
            )
        if query.query_go_global_index is not None:
            if global_cache is None:
                raise ValueError(
                    "query_go_global_index requires a full NBSGOBoxCache"
                )
            idx = query.query_go_global_index
            validate_local_index(idx, global_cache.context.size(0), "query_go_global_index")
            return (
                global_cache.static[idx],
                global_cache.semantic[idx],
                global_cache.hierarchy[idx],
                global_cache.context[idx],
            )
        return None

    def _pool_go(
        self,
        seed_pool: Tensor,
        static: Tensor,
        semantic: Tensor,
        hierarchy: Tensor,
        context: Tensor,
        go_query_index: Tensor,
        num_queries: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if go_query_index.numel() != static.size(0):
            raise ValueError("go_query_index must align with the query GO occurrences")
        if self.config.go_query_pool == "mean":
            static_pool = segment_mean(static, go_query_index, num_queries)
            semantic_pool = segment_mean(semantic, go_query_index, num_queries)
            hierarchy_pool = segment_mean(hierarchy, go_query_index, num_queries)
            context_pool = segment_mean(context, go_query_index, num_queries)
            weights = static.new_ones(static.size(0))
        else:
            score = (
                self.go_attention_seed(seed_pool)[go_query_index]
                * self.go_attention_key(static + context)
            ).sum(dim=-1) / max(static.size(-1) ** 0.5, 1.0)
            static_pool, weights = segment_attention_pool(
                static, score, go_query_index, num_queries
            )
            semantic_pool = segment_mean(
                semantic * weights.unsqueeze(-1), go_query_index, num_queries
            )
            hierarchy_pool = segment_mean(
                hierarchy * weights.unsqueeze(-1), go_query_index, num_queries
            )
            context_pool = segment_mean(
                context * weights.unsqueeze(-1), go_query_index, num_queries
            )
            # segment_mean divided by the count; undo it because attention weights
            # already sum to one in each query segment.
            counts = static.new_zeros(num_queries)
            counts.index_add_(0, go_query_index, torch.ones_like(go_query_index, dtype=static.dtype))
            scale = counts.clamp_min(1).unsqueeze(-1)
            semantic_pool = semantic_pool * scale
            hierarchy_pool = hierarchy_pool * scale
            context_pool = context_pool * scale
        return static_pool, semantic_pool, hierarchy_pool, context_pool, weights

    def forward(
        self,
        protein_context: Tensor,
        go_context: Tensor,
        local_box: BoxGOEncoding,
        query: ProteinGOQueryBatch,
        *,
        global_cache: Optional[NBSGOBoxCache] = None,
    ) -> NBSQueryCondition:
        validate_local_index(
            query.seed_protein_index, protein_context.size(0), "seed_protein_index"
        )
        if query.seed_query_index.numel() != query.seed_protein_index.numel():
            raise ValueError("seed_query_index must align with seed_protein_index")
        seed_pool = segment_mean(
            protein_context[query.seed_protein_index],
            query.seed_query_index,
            query.num_queries,
        )

        resolved = self._resolve_go_occurrences(
            go_context, local_box, query, global_cache
        )
        go_present = seed_pool.new_zeros(query.num_queries, 1)
        go_attention = seed_pool.new_zeros(0)
        if resolved is None:
            static_pool = semantic_pool = hierarchy_pool = context_pool = torch.zeros_like(seed_pool)
        else:
            if query.go_query_index is None:
                raise ValueError("go_query_index is required when query GO terms are supplied")
            static, semantic, hierarchy, context = resolved
            static_pool, semantic_pool, hierarchy_pool, context_pool, go_attention = self._pool_go(
                seed_pool,
                static,
                semantic,
                hierarchy,
                context,
                query.go_query_index,
                query.num_queries,
            )
            counts = seed_pool.new_zeros(query.num_queries)
            counts.index_add_(
                0,
                query.go_query_index,
                torch.ones_like(query.go_query_index, dtype=seed_pool.dtype),
            )
            go_present = (counts > 0).to(seed_pool.dtype).unsqueeze(-1)

        semantic_delta = self.semantic_fusion(
            torch.cat(
                [
                    seed_pool,
                    static_pool,
                    context_pool,
                    seed_pool * context_pool,
                    (seed_pool - context_pool).abs(),
                ],
                dim=-1,
            )
        ) * go_present
        hierarchy_delta = self.hierarchy_fusion(
            torch.cat([hierarchy_pool, static_pool, context_pool], dim=-1)
        ) * go_present
        semantic_gate = torch.sigmoid(
            self.semantic_gate(torch.cat([seed_pool, static_pool, context_pool], dim=-1))
        ) * go_present
        hierarchy_gate = torch.sigmoid(
            self.hierarchy_gate(
                torch.cat([seed_pool, hierarchy_pool, context_pool], dim=-1)
            )
        ) * go_present

        base_query = seed_pool + semantic_gate * semantic_delta + hierarchy_gate * hierarchy_delta
        frequency_encoded: Optional[Tensor] = None
        if query.query_go_frequency is not None:
            freq = query.query_go_frequency.to(seed_pool.device, seed_pool.dtype).reshape(query.num_queries, -1)
            if freq.size(1) != 1:
                freq = freq.mean(dim=1, keepdim=True)
            frequency_encoded = self.frequency_proj(torch.log1p(freq.clamp_min(0.0)))
            base_query = base_query + torch.sigmoid(self.frequency_gate) * frequency_encoded

        external_encoded: Optional[Tensor] = None
        if self.external_proj is not None:
            if query.external_query_features is None:
                external_encoded = torch.zeros_like(seed_pool)
            else:
                ext = query.external_query_features.to(seed_pool.device, seed_pool.dtype)
                if ext.shape != (query.num_queries, self.config.external_query_dim):
                    raise ValueError(
                        "external_query_features must be "
                        f"[{query.num_queries}, {self.config.external_query_dim}]"
                    )
                external_encoded = self.external_proj(ext)
                base_query = base_query + torch.sigmoid(self.external_gate) * external_encoded
        elif query.external_query_features is not None:
            raise ValueError("external_query_features were provided but external_query_dim=0")

        base_query = self.output_norm(base_query)
        auxiliary: Dict[str, Tensor] = {
            "seed_pool": seed_pool,
            "go_static_pool": static_pool,
            "go_semantic_pool": semantic_pool,
            "go_hierarchy_pool": hierarchy_pool,
            "go_context_pool": context_pool,
            "go_present": go_present,
            "go_attention": go_attention,
            "semantic_gate": semantic_gate,
            "hierarchy_gate": hierarchy_gate,
            "semantic_delta": semantic_delta,
            "hierarchy_delta": hierarchy_delta,
        }
        if frequency_encoded is not None:
            auxiliary["frequency_encoded"] = frequency_encoded
        if external_encoded is not None:
            auxiliary["external_encoded"] = external_encoded

        return NBSQueryCondition(
            base_query=base_query,
            seed_index=query.seed_protein_index,
            seed_query_index=query.seed_query_index,
            num_queries=query.num_queries,
            candidate_index=query.candidate_protein_index,
            base_logits=query.base_logits,
            candidate_evidence=query.candidate_evidence,
            expert_prob=query.expert_prob,
            query_go_frequency=query.query_go_frequency,
            labels=query.labels,
            mask=query.mask,
            confidence=query.confidence,
            pseudo_mask=query.pseudo_mask,
            auxiliary=auxiliary,
        )
