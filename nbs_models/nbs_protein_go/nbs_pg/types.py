from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class NBSNeighborhoodHierarchy:
    """Graph-agnostic additive protein neighbourhood hierarchy."""

    final_context: Tensor                    # [N_target, D]
    source_contexts: Tensor                  # [S, N_target, D]
    source_names: Tuple[str, ...]
    layer_contexts: Optional[Tensor] = None  # [L, N_target, D]
    target_global_ids: Optional[Tensor] = None

    def validate(self) -> None:
        if self.final_context.dim() != 2:
            raise ValueError("final_context must be [N, D]")
        if self.source_contexts.dim() != 3:
            raise ValueError("source_contexts must be [S, N, D]")
        if self.source_contexts.shape[1:] != self.final_context.shape:
            raise ValueError("source_contexts must align with final_context")
        if self.source_contexts.size(0) != len(self.source_names):
            raise ValueError("source_names and source_contexts disagree")


@dataclass
class NBSHeteroBackboneOutput:
    final_states: Dict[str, Tensor]
    layer_states: List[Dict[str, Tensor]]
    layer_deltas: List[Dict[str, Tensor]]
    target_sources: List[Tensor]
    target_source_names: Tuple[str, ...]
    relation_gate_means: Dict[str, Tensor]


@dataclass
class NBSHomogeneousBackboneOutput:
    final_state: Tensor
    layer_states: List[Tensor]
    layer_deltas: List[Tensor]
    source_names: Tuple[str, ...]


@dataclass
class BoxGOEncoding:
    semantic: Tensor       # [G, D]
    hierarchy: Tensor      # [G, D]
    static: Tensor         # [G, D]
    hierarchy_gate: Tensor # [G, D]
    center: Tensor         # [G, box_dim]
    offset: Tensor         # [G, box_dim]
    log_offset: Tensor     # [G, box_dim]
    stats: Optional[Tensor] = None


@dataclass
class NBSGOBoxCache:
    """Full-GO cache usable when a protein-rooted sample omits query GO nodes."""

    semantic: Tensor
    hierarchy: Tensor
    static: Tensor
    context: Tensor
    center: Tensor
    offset: Tensor
    stats: Optional[Tensor] = None

    def index(self, ids: Tensor) -> BoxGOEncoding:
        return BoxGOEncoding(
            semantic=self.semantic[ids],
            hierarchy=self.hierarchy[ids],
            static=self.static[ids],
            hierarchy_gate=torch.ones_like(self.static[ids]),
            center=self.center[ids],
            offset=self.offset[ids],
            log_offset=torch.log(self.offset[ids].clamp_min(1e-8)),
            stats=None if self.stats is None else self.stats[ids],
        )


@dataclass
class ProteinGOQueryBatch:
    """One or more GO-conditioned protein-neighbourhood queries.

    Local query GO indices are preferred when present. ``query_go_global_index``
    enables a full-GO cache fallback for protein-rooted sampled subgraphs.
    ``base_logits`` and optional ``candidate_evidence`` must align with the
    selected candidates. ``expert_prob`` is retained only for legacy ablations.
    Unknown labels should be excluded through ``mask`` rather than encoded as 0.
    """

    seed_protein_index: Tensor
    seed_query_index: Tensor
    num_queries: int
    query_go_index: Optional[Tensor] = None
    query_go_global_index: Optional[Tensor] = None
    go_query_index: Optional[Tensor] = None
    external_query_features: Optional[Tensor] = None
    candidate_protein_index: Optional[Tensor] = None
    base_logits: Optional[Tensor] = None
    candidate_evidence: Optional[Tensor] = None
    # Legacy compatibility only.  The default NBS gate never reads expert_prob.
    expert_prob: Optional[Tensor] = None
    query_go_frequency: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    confidence: Optional[Tensor] = None
    pseudo_mask: Optional[Tensor] = None

    def to(self, device: torch.device | str) -> "ProteinGOQueryBatch":
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            kwargs[f.name] = value.to(device) if isinstance(value, Tensor) else value
        return ProteinGOQueryBatch(**kwargs)


@dataclass
class NBSQueryCondition:
    base_query: Tensor
    seed_index: Tensor
    seed_query_index: Tensor
    num_queries: int
    candidate_index: Optional[Tensor]
    base_logits: Optional[Tensor]
    candidate_evidence: Optional[Tensor]
    expert_prob: Optional[Tensor]
    query_go_frequency: Optional[Tensor]
    labels: Optional[Tensor]
    mask: Optional[Tensor]
    confidence: Optional[Tensor]
    pseudo_mask: Optional[Tensor]
    auxiliary: Optional[Dict[str, Tensor]] = None


@dataclass
class NBSMatchOutput:
    logits: Tensor
    labels: Optional[Tensor]
    mask: Optional[Tensor]
    confidence: Optional[Tensor] = None
    pseudo_mask: Optional[Tensor] = None
    auxiliary: Optional[Dict[str, Tensor]] = None

    @property
    def flat_logits(self) -> Tensor:
        return self.logits.reshape(-1)


@dataclass
class ProteinGOEncodedGraph:
    hierarchy: NBSNeighborhoodHierarchy
    final_states: Dict[str, Tensor]
    layer_states: List[Dict[str, Tensor]]
    local_go_box: BoxGOEncoding
    global_go_cache: Optional[NBSGOBoxCache] = None
    backbone_aux: Optional[Dict[str, Tensor]] = None
