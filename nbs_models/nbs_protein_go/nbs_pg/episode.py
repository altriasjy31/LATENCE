from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from .latence_graph_stores import RoleLocalProteinGOCSRStore
from .latence_stores import (
    FixedDegreeCandidateAttributeStore,
    GOProteinCSRStore,
    RoleAwareBaseLogitStore,
)
from .types import ProteinGOQueryBatch


def select_background_unlabelled_columns(
    *,
    candidate_protein_idx: np.ndarray,
    base_logits_row: np.ndarray,
    occupied_mask: np.ndarray,
    excluded_protein_idx: np.ndarray,
    count: int,
    max_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select low-confidence PU background pairs from an existing candidate union.

    The returned indices are columns in ``candidate_protein_idx``.  This helper
    deliberately does not add proteins or edges to the local graph.  It only
    opens additional supervision positions in the already materialized [Q,C]
    score matrix.  Gold, pseudo-positive and backbone-candidate proteins for
    the current GO must be supplied through ``excluded_protein_idx``.
    """
    candidate = np.asarray(candidate_protein_idx, dtype=np.int64)
    logits = np.asarray(base_logits_row, dtype=np.float64)
    occupied = np.asarray(occupied_mask, dtype=bool)
    excluded = np.asarray(excluded_protein_idx, dtype=np.int64)
    if candidate.ndim != 1 or logits.shape != candidate.shape or occupied.shape != candidate.shape:
        raise ValueError("background candidate/logit/mask arrays must be one-dimensional and aligned")
    if count <= 0 or candidate.size == 0:
        return np.empty(0, dtype=np.int64)
    if not 0.0 <= float(max_probability) <= 0.5:
        raise ValueError("background max_probability must lie in [0,0.5]")
    # Stable sigmoid; the threshold is intentionally evaluated in float64 so
    # fp16 base-probability quantization cannot silently broaden the PU pool.
    clipped = np.clip(logits, -40.0, 40.0)
    probability = 1.0 / (1.0 + np.exp(-clipped))
    eligible = ~occupied
    eligible &= probability <= float(max_probability)
    if excluded.size:
        eligible &= ~np.isin(candidate, excluded, assume_unique=False)
    columns = np.flatnonzero(eligible).astype(np.int64, copy=False)
    if columns.size <= int(count):
        return columns
    chosen = rng.choice(columns, size=int(count), replace=False)
    return np.asarray(chosen, dtype=np.int64)


@dataclass
class NBSQueryEpisodeConfig:
    num_queries: int = 8
    support_per_query: int = 2
    gold_positive_per_query: int = 1
    hard_candidate_per_query: int = 16
    pseudo_positive_per_query: int = 0
    # Repeated GO queries can either resample weak positives randomly or walk
    # the GO-major pseudo CSR in deterministic non-overlapping chunks.  The
    # latter makes repeated GO coverage act as a weak-protein coverage cycle.
    pseudo_sampling_mode: str = "random"
    # ``go_cyclic_unique`` keeps the GO-major cyclic walk but prioritizes
    # weak proteins not yet used as pseudo targets in the current epoch.
    # With DDP, each rank first consumes its deterministic weak-ID partition,
    # reducing cross-rank duplicate pseudo supervision.
    pseudo_rank_partition: bool = True
    # Hybrid GO/weak scheduling.  These slots are still GO queries, but their
    # GO identity is chosen from previously unseen weak proteins through the
    # protein-major modelout CSR.  This preserves the NBS GO-query architecture
    # while preventing repeated full GO cycles from supervising the same
    # multi-label weak proteins over and over.
    weak_focus_queries_per_episode: int = 0
    weak_focus_targets_per_query: int = 0
    weak_focus_scan_limit: int = 8192
    weak_focus_specificity_power: float = 0.5
    weak_focus_min_probability: float = 0.5
    max_candidates: int = 256
    sampled_unlabelled_weight: float = 0.2
    # Low-weight PU contrast from proteins already present in the shared
    # candidate union.  These are not asserted biological negatives.  They are
    # selected only when the first-stage backbone is very low-confidence for
    # the current GO and are down-weighted far below hard candidate evidence.
    background_unlabelled_per_query: int = 0
    background_unlabelled_weight: float = 0.05
    background_base_probability_max: float = 0.05
    pseudo_confidence_power: float = 1.0
    # Number of direct child-parent query pairs that must be co-sampled in one
    # episode.  This makes the query-axis hierarchy loss observable instead of
    # relying on an extremely unlikely random co-occurrence.
    hierarchy_pairs_per_episode: int = 0
    # Query scheduling is a mini-batch policy, not a prediction-vocabulary cap.
    # ``shuffled_cycle`` deterministically walks the eligible task-GO space
    # across global DDP episodes before repeating it.
    query_sampling_mode: str = "random"
    # ``adaptive_rare`` makes train-count 2 terms use 1 support + 1 held-out
    # gold target, and train-count 1 terms use one support plus pseudo/graph
    # supervision.  This prevents rare GO terms from being silently excluded
    # merely because the default fixed policy needs 2+1 gold proteins.
    gold_support_policy: str = "fixed"
    singleton_requires_pseudo: bool = True
    max_query_resample_attempts: int = 100

    def validate(self) -> None:
        for name in (
            "num_queries",
            "support_per_query",
            "gold_positive_per_query",
            "hard_candidate_per_query",
            "max_candidates",
            "max_query_resample_attempts",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pseudo_positive_per_query < 0:
            raise ValueError("pseudo_positive_per_query cannot be negative")
        if self.pseudo_sampling_mode not in {"random", "go_cyclic", "go_cyclic_unique"}:
            raise ValueError("pseudo_sampling_mode must be random, go_cyclic or go_cyclic_unique")
        if self.weak_focus_queries_per_episode < 0:
            raise ValueError("weak_focus_queries_per_episode cannot be negative")
        if self.weak_focus_targets_per_query < 0:
            raise ValueError("weak_focus_targets_per_query cannot be negative")
        if self.weak_focus_scan_limit <= 0:
            raise ValueError("weak_focus_scan_limit must be positive")
        if self.weak_focus_specificity_power < 0:
            raise ValueError("weak_focus_specificity_power cannot be negative")
        if not 0.5 <= self.weak_focus_min_probability <= 1.0:
            raise ValueError("weak_focus_min_probability must lie in [0.5,1]")
        if self.weak_focus_queries_per_episode > 0:
            if self.pseudo_sampling_mode != "go_cyclic_unique":
                raise ValueError("weak-focus scheduling requires pseudo_sampling_mode=go_cyclic_unique")
            if self.query_sampling_mode != "shuffled_cycle":
                raise ValueError("weak-focus scheduling requires query_sampling_mode=shuffled_cycle")
            if self.weak_focus_targets_per_query <= 0:
                raise ValueError("weak_focus_targets_per_query must be positive when weak focus is enabled")
            if self.weak_focus_targets_per_query > self.pseudo_positive_per_query:
                raise ValueError("weak_focus_targets_per_query cannot exceed pseudo_positive_per_query")
        if self.hierarchy_pairs_per_episode < 0:
            raise ValueError("hierarchy_pairs_per_episode cannot be negative")
        reserved_queries = (
            2 * self.hierarchy_pairs_per_episode
            + self.weak_focus_queries_per_episode
        )
        if reserved_queries >= self.num_queries:
            raise ValueError(
                "hierarchy and weak-focus query slots must leave at least one "
                "shuffled-cycle GO coverage slot"
            )
        if not 0.0 <= self.sampled_unlabelled_weight <= 1.0:
            raise ValueError("sampled_unlabelled_weight must lie in [0,1]")
        if self.background_unlabelled_per_query < 0:
            raise ValueError("background_unlabelled_per_query cannot be negative")
        if not 0.0 <= self.background_unlabelled_weight <= 1.0:
            raise ValueError("background_unlabelled_weight must lie in [0,1]")
        if self.background_unlabelled_weight > self.sampled_unlabelled_weight:
            raise ValueError(
                "background_unlabelled_weight should not exceed sampled_unlabelled_weight"
            )
        if not 0.0 <= self.background_base_probability_max <= 0.5:
            raise ValueError("background_base_probability_max must lie in [0,0.5]")
        if self.pseudo_confidence_power < 0:
            raise ValueError("pseudo_confidence_power cannot be negative")
        if self.query_sampling_mode not in {"random", "shuffled_cycle"}:
            raise ValueError("query_sampling_mode must be random or shuffled_cycle")
        if self.gold_support_policy not in {"fixed", "adaptive_rare"}:
            raise ValueError("gold_support_policy must be fixed or adaptive_rare")


@dataclass
class NBSGlobalEpisode:
    # Task classifier GO columns used by base logits and supervision.
    query_go_idx: np.ndarray
    seed_protein_idx: np.ndarray
    seed_query_idx: np.ndarray
    candidate_protein_idx: np.ndarray
    base_logits: np.ndarray
    candidate_evidence: np.ndarray
    query_go_frequency: np.ndarray
    labels: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    pseudo_mask: np.ndarray
    supervision_weight: np.ndarray
    # Full BoxSquaredEL ontology rows used by the GO tower.  ``None`` keeps
    # backward compatibility for task-only toy tests.
    query_ontology_go_idx: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        q = self.query_go_idx.size
        c = self.candidate_protein_idx.size
        if self.query_ontology_go_idx is not None and self.query_ontology_go_idx.shape != (q,):
            raise ValueError("query_ontology_go_idx must align with task queries")
        if self.seed_protein_idx.shape != self.seed_query_idx.shape:
            raise ValueError("seed protein/query arrays must align")
        if self.seed_query_idx.size and (
            self.seed_query_idx.min() < 0 or self.seed_query_idx.max() >= q
        ):
            raise IndexError("seed_query_idx outside episode query range")
        for name in (
            "base_logits",
            "labels",
            "mask",
            "confidence",
            "pseudo_mask",
            "supervision_weight",
        ):
            value = getattr(self, name)
            if value.shape != (q, c):
                raise ValueError(f"{name} shape {value.shape} != {(q, c)}")
        if self.candidate_evidence.shape[:2] != (q, c):
            raise ValueError("candidate_evidence must start with [Q,C]")
        if self.query_go_frequency.shape != (q,):
            raise ValueError("query_go_frequency must be [Q]")
        if np.intersect1d(self.seed_protein_idx, self.candidate_protein_idx).size:
            raise ValueError("support and candidate proteins must be episode-disjoint")

    def to_query_batch(
        self,
        *,
        seed_protein_local: Tensor,
        candidate_protein_local: Tensor,
        query_go_local: Optional[Tensor] = None,
        use_global_go_fallback: bool = False,
        device: torch.device | str = "cpu",
    ) -> ProteinGOQueryBatch:
        self.validate()
        query_go_global = None
        if use_global_go_fallback:
            query_go_global = torch.as_tensor(
                (self.query_go_idx if self.query_ontology_go_idx is None else self.query_ontology_go_idx),
                dtype=torch.long, device=device
            )
            query_go_local = None
        elif query_go_local is None:
            raise ValueError("query_go_local is required unless global GO fallback is enabled")
        return ProteinGOQueryBatch(
            seed_protein_index=seed_protein_local.to(device),
            seed_query_index=torch.as_tensor(
                self.seed_query_idx, dtype=torch.long, device=device
            ),
            num_queries=int(self.query_go_idx.size),
            query_go_index=(None if query_go_local is None else query_go_local.to(device)),
            query_go_global_index=query_go_global,
            go_query_index=torch.arange(
                self.query_go_idx.size, dtype=torch.long, device=device
            ),
            candidate_protein_index=candidate_protein_local.to(device),
            base_logits=torch.as_tensor(self.base_logits, dtype=torch.float32, device=device),
            candidate_evidence=torch.as_tensor(
                self.candidate_evidence, dtype=torch.float32, device=device
            ),
            query_go_frequency=torch.as_tensor(
                self.query_go_frequency, dtype=torch.float32, device=device
            ),
            labels=torch.as_tensor(self.labels, dtype=torch.float32, device=device),
            mask=torch.as_tensor(self.mask, dtype=torch.bool, device=device),
            confidence=torch.as_tensor(
                self.confidence, dtype=torch.float32, device=device
            ),
            pseudo_mask=torch.as_tensor(
                self.pseudo_mask, dtype=torch.bool, device=device
            ),
            supervision_weight=torch.as_tensor(
                self.supervision_weight, dtype=torch.float32, device=device
            ),
        )


class GOQueryEpisodeSampler:
    """Construct GO-query episodes from gold/candidate/pseudo inverted stores.

    This sampler defines supervision and candidate evidence only.  A project-
    specific local graph materializer must subsequently gather P--P, P--GO and
    GO--GO neighbourhoods and map these global IDs to local PyG indices.
    """

    def __init__(
        self,
        *,
        gold: GOProteinCSRStore,
        candidate: GOProteinCSRStore,
        base_logits: RoleAwareBaseLogitStore,
        train_go_counts: np.ndarray,
        task_to_ontology_go: Optional[np.ndarray] = None,
        candidate_attributes: Optional[FixedDegreeCandidateAttributeStore] = None,
        pseudo: Optional[GOProteinCSRStore] = None,
        pseudo_by_protein: Optional[RoleLocalProteinGOCSRStore] = None,
        pseudo_active_role_rows: Optional[np.ndarray] = None,
        hierarchy_pairs: Optional[np.ndarray] = None,
        config: Optional[NBSQueryEpisodeConfig] = None,
        seed: int = 3407,
        weak_global_start: Optional[int] = None,
        weak_global_end: Optional[int] = None,
    ) -> None:
        self.gold = gold
        self.candidate = candidate
        self.pseudo = pseudo
        self.pseudo_by_protein = pseudo_by_protein
        self.base_logit_store = base_logits
        self.candidate_attributes = candidate_attributes
        self.train_go_counts = np.asarray(train_go_counts, dtype=np.float64)
        if hierarchy_pairs is None:
            self.hierarchy_pairs = np.empty((2, 0), dtype=np.int64)
        else:
            pairs = np.asarray(hierarchy_pairs, dtype=np.int64)
            if pairs.ndim != 2 or pairs.shape[0] != 2:
                raise ValueError("hierarchy_pairs must have shape [2,E]")
            self.hierarchy_pairs = pairs
        if task_to_ontology_go is None:
            self.task_to_ontology_go = np.arange(self.gold.num_go, dtype=np.int64)
        else:
            self.task_to_ontology_go = np.asarray(task_to_ontology_go, dtype=np.int64)
            if self.task_to_ontology_go.shape != (self.gold.num_go,):
                raise ValueError("task_to_ontology_go must align with task GO space")
            if np.any(self.task_to_ontology_go < 0):
                raise ValueError("task_to_ontology_go contains missing ontology rows")
        self.config = config or NBSQueryEpisodeConfig()
        self.config.validate()
        if self.gold.num_go != self.candidate.num_go:
            raise ValueError("gold and candidate stores use different GO spaces")
        if self.pseudo is not None and self.pseudo.num_go != self.gold.num_go:
            raise ValueError("pseudo and gold stores use different GO spaces")
        if self.train_go_counts.shape != (self.gold.num_go,):
            raise ValueError("train_go_counts must align with GO index space")
        self.gold_degree = np.diff(np.asarray(self.gold.indptr, dtype=np.int64))
        self.pseudo_degree = (
            np.diff(np.asarray(self.pseudo.indptr, dtype=np.int64))
            if self.pseudo is not None
            else np.zeros(self.gold.num_go, dtype=np.int64)
        )
        if self.config.gold_support_policy == "fixed":
            min_gold = self.config.support_per_query + self.config.gold_positive_per_query
            eligible = self.gold_degree >= min_gold
        else:
            # Every query still needs at least one manually annotated core
            # protein to define a concrete functional neighbourhood.  Singleton
            # terms may enter only when weak modelout supplies a positive signal
            # if singleton_requires_pseudo is enabled.
            eligible = self.gold_degree >= 1
            if self.config.singleton_requires_pseudo:
                singleton = self.gold_degree == 1
                eligible[singleton] &= self.pseudo_degree[singleton] > 0
        self.eligible_go = np.flatnonzero(eligible).astype(np.int64)
        self.candidate_degree = np.diff(np.asarray(self.candidate.indptr, dtype=np.int64))
        eligible_degree = self.gold_degree[self.eligible_go]
        eligible_pseudo_degree = self.pseudo_degree[self.eligible_go]
        eligible_candidate_degree = self.candidate_degree[self.eligible_go]
        zero_gold = self.gold_degree == 0
        singleton_all = self.gold_degree == 1
        singleton_with_pseudo = singleton_all & (self.pseudo_degree > 0)
        singleton_without_pseudo = singleton_all & (self.pseudo_degree <= 0)
        ineligible_mask = ~eligible
        ineligible_other = ineligible_mask & ~zero_gold & ~singleton_without_pseudo
        self.eligibility_summary = {
            "num_task_go": int(self.gold.num_go),
            "gold_positive_go": int(np.sum(self.gold_degree > 0)),
            "gold_zero_count_all": int(np.sum(zero_gold)),
            "gold_count_eq1_all": int(np.sum(singleton_all)),
            "gold_count_eq1_with_pseudo": int(np.sum(singleton_with_pseudo)),
            "gold_count_eq1_without_pseudo": int(np.sum(singleton_without_pseudo)),
            "eligible_go_count": int(self.eligible_go.size),
            "ineligible_go_count": int(np.sum(ineligible_mask)),
            "ineligible_zero_gold": int(np.sum(ineligible_mask & zero_gold)),
            "ineligible_singleton_without_pseudo": int(
                np.sum(ineligible_mask & singleton_without_pseudo)
            ),
            "ineligible_other": int(np.sum(ineligible_other)),
            "eligible_gold_count_eq1": int(np.sum(eligible_degree == 1)),
            "eligible_gold_count_eq2": int(np.sum(eligible_degree == 2)),
            "eligible_gold_count_3_4": int(np.sum((eligible_degree >= 3) & (eligible_degree <= 4))),
            "eligible_gold_count_gt4": int(np.sum(eligible_degree > 4)),
            "eligible_with_pseudo": int(np.sum(eligible_pseudo_degree > 0)),
            "eligible_with_backbone_candidate": int(np.sum(eligible_candidate_degree > 0)),
            "eligible_without_backbone_candidate": int(np.sum(eligible_candidate_degree == 0)),
        }
        if self.eligible_go.size < self.config.num_queries:
            raise ValueError("not enough eligible GO terms for the requested query episode")
        eligible_mask = np.zeros(self.gold.num_go, dtype=bool)
        eligible_mask[self.eligible_go] = True
        self.eligible_go_mask = eligible_mask
        if self.hierarchy_pairs.size:
            keep = (
                eligible_mask[self.hierarchy_pairs[0]]
                & eligible_mask[self.hierarchy_pairs[1]]
                & (self.hierarchy_pairs[0] != self.hierarchy_pairs[1])
            )
            self.hierarchy_pairs = self.hierarchy_pairs[:, keep]
            if self.hierarchy_pairs.size:
                # Remove exact duplicate task-level edges while keeping direction.
                self.hierarchy_pairs = np.unique(self.hierarchy_pairs.T, axis=0).T
        if (
            self.config.hierarchy_pairs_per_episode > 0
            and self.hierarchy_pairs.shape[1] < self.config.hierarchy_pairs_per_episode
        ):
            raise ValueError(
                "not enough eligible task-level hierarchy pairs for "
                f"hierarchy_pairs_per_episode={self.config.hierarchy_pairs_per_episode}"
            )
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.weak_global_start = None if weak_global_start is None else int(weak_global_start)
        self.weak_global_end = None if weak_global_end is None else int(weak_global_end)
        if self.config.pseudo_sampling_mode == "go_cyclic_unique":
            if self.weak_global_start is None or self.weak_global_end is None:
                raise ValueError("go_cyclic_unique requires weak_global_start/weak_global_end")
            if self.weak_global_end <= self.weak_global_start:
                raise ValueError("invalid weak global protein range")
        if self.config.weak_focus_queries_per_episode > 0:
            if self.pseudo_by_protein is None:
                raise ValueError("weak-focus scheduling requires the protein-major pseudo CSR")
            if self.pseudo_by_protein.role != "weak":
                raise ValueError("weak-focus protein-major pseudo CSR must use role='weak'")
        if pseudo_active_role_rows is None:
            self.pseudo_active_role_rows = np.empty(0, dtype=np.int64)
        else:
            active_rows = np.unique(np.asarray(pseudo_active_role_rows, dtype=np.int64))
            if active_rows.size and (
                int(active_rows.min()) < 0
                or self.weak_global_start is None
                or self.weak_global_end is None
                or int(active_rows.max()) >= self.weak_global_end - self.weak_global_start
            ):
                raise IndexError("pseudo_active_role_rows leaves the weak role-local space")
            self.pseudo_active_role_rows = active_rows
        self._pseudo_seen_epoch: Optional[int] = None
        self._pseudo_seen: Optional[np.ndarray] = None
        self._weak_focus_epoch: Optional[int] = None
        self._weak_focus_cursor: int = 0
        self._weak_focus_owned_cache: dict[tuple[int, int], np.ndarray] = {}

    @property
    def coverage_slots_per_episode(self) -> int:
        """Number of query slots reserved for deterministic GO-cycle coverage."""
        return max(
            1,
            int(self.config.num_queries)
            - 2 * int(self.config.hierarchy_pairs_per_episode)
            - int(self.config.weak_focus_queries_per_episode),
        )

    @staticmethod
    def _coprime_stride(size: int, seed: int) -> int:
        """Return a deterministic stride coprime to ``size``.

        This defines a memory-free affine permutation over a potentially very
        large GO-major pseudo list.  It avoids materialising a full random
        permutation for high-degree GO terms.
        """
        import math

        if size <= 1:
            return 1
        candidate = int(seed % size)
        if candidate <= 0:
            candidate = 1
        if candidate % 2 == 0:
            candidate += 1
        while math.gcd(candidate, size) != 1:
            candidate += 2
            if candidate >= size:
                candidate = 1
        return candidate

    def _cyclic_pseudo_indices(
        self,
        *,
        size: int,
        count: int,
        go_idx: int,
        epoch: int,
        global_episode: int,
    ) -> np.ndarray:
        """Select a deterministic pseudo-positive chunk for one GO.

        The base GO-cycle index is derived from the global DDP episode cursor.
        Successive complete GO cycles therefore move to the next pseudo chunk
        instead of repeatedly drawing the same weak proteins.  A per-epoch
        affine permutation changes ordering without allocating O(degree) RAM.
        """
        if count <= 0 or size <= 0:
            return np.empty(0, dtype=np.int64)
        count = min(int(count), int(size))
        cycle_index = (
            int(global_episode) * int(self.coverage_slots_per_episode)
        ) // max(1, int(self.eligible_go.size))
        absolute_start = int(cycle_index) * int(count)
        result: list[int] = []
        remaining = int(count)
        cursor = int(absolute_start)
        while remaining > 0:
            block = cursor // int(size)
            offset = cursor % int(size)
            take = min(remaining, int(size) - offset)
            permutation_seed = (
                self.base_seed
                + int(epoch) * 1_000_003
                + int(go_idx) * 104_729
                + int(block) * 65_537
            )
            stride = self._coprime_stride(int(size), permutation_seed * 2 + 1)
            shift = int((permutation_seed * 2_654_435_761) % int(size))
            positions = np.arange(offset, offset + take, dtype=np.int64)
            mapped = (stride * positions + shift) % int(size)
            result.extend(mapped.astype(np.int64).tolist())
            cursor += take
            remaining -= take
        return np.asarray(result, dtype=np.int64)

    def _ensure_pseudo_seen_epoch(self, epoch: int) -> None:
        if self.config.pseudo_sampling_mode != "go_cyclic_unique":
            return
        if self._pseudo_seen_epoch == int(epoch) and self._pseudo_seen is not None:
            return
        assert self.weak_global_start is not None and self.weak_global_end is not None
        self._pseudo_seen_epoch = int(epoch)
        self._pseudo_seen = np.zeros(
            self.weak_global_end - self.weak_global_start, dtype=np.bool_
        )

    def _take_cyclic_pool(
        self,
        pool: np.ndarray,
        *,
        count: int,
        go_idx: int,
        epoch: int,
        global_episode: int,
        salt: int,
    ) -> np.ndarray:
        if count <= 0 or pool.size == 0:
            return np.empty(0, dtype=np.int64)
        take = min(int(count), int(pool.size))
        # Reuse the memory-free affine permutation over the compact pool.
        choice = self._cyclic_pseudo_indices(
            size=int(pool.size),
            count=take,
            go_idx=int(go_idx) + int(salt) * 1_000_003,
            epoch=int(epoch),
            global_episode=int(global_episode),
        )
        return pool[choice].astype(np.int64, copy=False)

    def _unique_pseudo_indices(
        self,
        proteins: np.ndarray,
        *,
        count: int,
        go_idx: int,
        epoch: int,
        global_episode: int,
        rank: int,
        world_size: int,
    ) -> np.ndarray:
        """Prefer unseen weak proteins, then fall back deterministically.

        The selection priority is:
          unseen + rank-owned -> unseen -> rank-owned -> any.
        This preserves GO-conditioned sampling while making a protein-major
        epoch target meaningful in a multi-label pseudo graph.
        """
        if count <= 0 or proteins.size == 0:
            return np.empty(0, dtype=np.int64)
        self._ensure_pseudo_seen_epoch(int(epoch))
        assert self._pseudo_seen is not None
        assert self.weak_global_start is not None and self.weak_global_end is not None
        local = proteins.astype(np.int64, copy=False) - int(self.weak_global_start)
        if np.any(local < 0) or np.any(local >= self._pseudo_seen.size):
            raise IndexError("pseudo protein leaves configured weak global range")
        unseen = ~self._pseudo_seen[local]
        if bool(self.config.pseudo_rank_partition) and int(world_size) > 1:
            owner = (local % int(world_size)) == int(rank)
        else:
            owner = np.ones(local.shape, dtype=np.bool_)

        selected_positions: list[np.ndarray] = []
        selected_mask = np.zeros(proteins.size, dtype=np.bool_)
        remaining = min(int(count), int(proteins.size))
        priorities = (
            unseen & owner,
            unseen,
            owner,
            np.ones(proteins.size, dtype=np.bool_),
        )
        for priority, mask in enumerate(priorities):
            if remaining <= 0:
                break
            pool = np.flatnonzero(mask & ~selected_mask).astype(np.int64)
            if pool.size == 0:
                continue
            chosen = self._take_cyclic_pool(
                pool,
                count=remaining,
                go_idx=int(go_idx),
                epoch=int(epoch),
                global_episode=int(global_episode),
                salt=priority + 1,
            )
            if chosen.size:
                selected_positions.append(chosen)
                selected_mask[chosen] = True
                remaining -= int(chosen.size)

        if not selected_positions:
            return np.empty(0, dtype=np.int64)
        selected = np.concatenate(selected_positions).astype(np.int64, copy=False)
        self._pseudo_seen[local[selected]] = True
        return selected


    def _ensure_weak_focus_epoch(self, epoch: int) -> None:
        if self._weak_focus_epoch == int(epoch):
            return
        self._weak_focus_epoch = int(epoch)
        self._weak_focus_cursor = 0

    def _owned_active_role_rows(self, *, rank: int, world_size: int) -> np.ndarray:
        key = (int(rank), int(world_size))
        cached = self._weak_focus_owned_cache.get(key)
        if cached is not None:
            return cached
        rows = self.pseudo_active_role_rows
        if bool(self.config.pseudo_rank_partition) and int(world_size) > 1:
            rows = rows[(rows % int(world_size)) == int(rank)]
        rows = np.asarray(rows, dtype=np.int64)
        self._weak_focus_owned_cache[key] = rows
        return rows

    def _next_weak_focus_role_row(
        self,
        *,
        epoch: int,
        rank: int,
        world_size: int,
    ) -> Optional[int]:
        """Walk a deterministic affine permutation of rank-owned weak rows.

        The cursor is stateful only within one epoch.  Checkpoints are written at
        epoch boundaries, so this preserves deterministic resume semantics while
        avoiding an O(Nweak) permutation allocation.
        """
        self._ensure_weak_focus_epoch(int(epoch))
        rows = self._owned_active_role_rows(rank=int(rank), world_size=int(world_size))
        if rows.size == 0:
            return None
        cursor = int(self._weak_focus_cursor)
        self._weak_focus_cursor += 1
        block = cursor // int(rows.size)
        offset = cursor % int(rows.size)
        seed = (
            self.base_seed
            + int(epoch) * 1_000_003
            + int(rank) * 104_729
            + int(block) * 65_537
        )
        stride = self._coprime_stride(int(rows.size), seed * 2 + 1)
        shift = int((seed * 2_654_435_761) % int(rows.size))
        position = int((stride * offset + shift) % int(rows.size))
        return int(rows[position])

    def _focus_go_score(
        self,
        go_idx: np.ndarray,
        probability: np.ndarray,
    ) -> np.ndarray:
        """Balance modelout confidence, pseudo capacity and GO specificity.

        Very broad GO terms often have enormous pseudo degree.  Confidence alone
        would repeatedly choose these generic terms for weak-first scheduling.
        The specificity factor keeps them as a fallback while preferring more
        informative GO labels when the same weak protein has alternatives.
        """
        go = np.asarray(go_idx, dtype=np.int64)
        prob = np.asarray(probability, dtype=np.float64)
        target = max(1, int(self.config.weak_focus_targets_per_query))
        capacity = np.minimum(self.pseudo_degree[go], target).astype(np.float64) / target
        specificity = np.power(
            1.0 + np.log1p(np.maximum(self.train_go_counts[go], 0.0)),
            -float(self.config.weak_focus_specificity_power),
        )
        return prob * capacity * specificity

    def _sample_weak_focus_anchors(
        self,
        *,
        epoch: int,
        rank: int,
        world_size: int,
        excluded_task: set[int],
        excluded_ontology: set[int],
    ) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], int]:
        """Choose GO queries from unseen weak proteins and retain one anchor.

        Only the GO identity is weak-first.  The resulting object is still a
        standard GO-conditioned NBS query and receives core support exactly like
        every other query.
        """
        requested = int(self.config.weak_focus_queries_per_episode)
        if requested <= 0:
            return {}, 0
        if self.pseudo_by_protein is None:
            raise RuntimeError("weak-focus scheduling requires pseudo_by_protein")
        self._ensure_pseudo_seen_epoch(int(epoch))
        assert self._pseudo_seen is not None
        anchors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        scanned = 0
        used_role_rows: set[int] = set()
        while len(anchors) < requested and scanned < int(self.config.weak_focus_scan_limit):
            role_row = self._next_weak_focus_role_row(
                epoch=int(epoch), rank=int(rank), world_size=int(world_size)
            )
            if role_row is None:
                break
            scanned += 1
            if role_row in used_role_rows or bool(self._pseudo_seen[role_row]):
                continue
            values = self.pseudo_by_protein.get_role_row(role_row)
            go = np.asarray(values["go_idx"], dtype=np.int64)
            probability = np.asarray(values["probability"], dtype=np.float32)
            if go.size == 0:
                continue
            keep = self.eligible_go_mask[go]
            keep &= probability >= float(self.config.weak_focus_min_probability)
            if excluded_task:
                keep &= ~np.isin(go, np.fromiter(excluded_task, dtype=np.int64))
            if not np.any(keep):
                continue
            go = go[keep]
            probability = probability[keep]
            ontology = self.task_to_ontology_go[go]
            if excluded_ontology:
                keep_ontology = ~np.isin(
                    ontology, np.fromiter(excluded_ontology, dtype=np.int64)
                )
                go = go[keep_ontology]
                probability = probability[keep_ontology]
                ontology = ontology[keep_ontology]
            if go.size == 0:
                continue
            score = self._focus_go_score(go, probability)
            # Stable deterministic tie-break: larger score, then smaller task GO.
            order = np.lexsort((go, -score))
            choice = int(order[0])
            selected_go = int(go[choice])
            selected_probability = float(probability[choice])
            protein = int(self.pseudo_by_protein.global_protein_for_role_row(role_row))
            anchors[selected_go] = (
                np.asarray([protein], dtype=np.int64),
                np.asarray([selected_probability], dtype=np.float32),
            )
            used_role_rows.add(role_row)
            excluded_task.add(selected_go)
            excluded_ontology.add(int(self.task_to_ontology_go[selected_go]))
        return anchors, scanned

    def _mark_pseudo_seen(self, proteins: np.ndarray, *, epoch: int) -> int:
        if proteins.size == 0 or self.config.pseudo_sampling_mode != "go_cyclic_unique":
            return 0
        self._ensure_pseudo_seen_epoch(int(epoch))
        assert self._pseudo_seen is not None
        assert self.weak_global_start is not None
        local = np.asarray(proteins, dtype=np.int64) - int(self.weak_global_start)
        if np.any(local < 0) or np.any(local >= self._pseudo_seen.size):
            raise IndexError("pseudo protein leaves configured weak global range")
        new_count = int(np.count_nonzero(~self._pseudo_seen[local]))
        self._pseudo_seen[local] = True
        return new_count

    def _gold_requirements(self, go_idx: int) -> tuple[int, int]:
        degree = int(self.gold_degree[int(go_idx)])
        if self.config.gold_support_policy == "fixed":
            return int(self.config.support_per_query), int(self.config.gold_positive_per_query)
        # Adaptive rare-GO policy:
        #   count=1 -> support=1, gold-positive=0
        #   count=2 -> support=1, gold-positive=1
        #   count>=3 -> default support/positive, capped by available gold.
        if degree <= 0:
            raise ValueError("adaptive query has no core gold support")
        if degree == 1:
            return 1, 0
        positive = min(int(self.config.gold_positive_per_query), max(1, degree - 1))
        support = min(int(self.config.support_per_query), degree - positive)
        support = max(1, support)
        return support, positive

    def _cycle_candidates(
        self,
        *,
        epoch: int,
        global_episode: int,
        count: int,
        excluded_task: set[int],
        excluded_ontology: set[int],
    ) -> list[int]:
        if count <= 0:
            return []
        n = int(self.eligible_go.size)
        if n <= 0:
            raise RuntimeError("empty eligible GO space")
        slots = self.coverage_slots_per_episode
        cursor = int(global_episode) * slots
        selected: list[int] = []
        # Continue across successive deterministic permutations so the tail of
        # one cycle can fill an episode without introducing a special case.
        max_scan = n * 4 + count * 4
        scanned = 0
        while len(selected) < count and scanned < max_scan:
            cycle_index = cursor // n
            offset = cursor % n
            cycle_seed = (
                self.base_seed
                + int(epoch) * 1_000_003
                + int(cycle_index) * 104_729
            )
            perm = np.random.default_rng(cycle_seed).permutation(self.eligible_go)
            go_idx = int(perm[offset])
            cursor += 1
            scanned += 1
            ontology_idx = int(self.task_to_ontology_go[go_idx])
            if go_idx in excluded_task or ontology_idx in excluded_ontology:
                continue
            selected.append(go_idx)
            excluded_task.add(go_idx)
            excluded_ontology.add(ontology_idx)
        if len(selected) < count:
            raise RuntimeError("could not complete shuffled-cycle GO query selection")
        return selected

    def _sample_query_plan(
        self,
        *,
        epoch: Optional[int] = None,
        global_episode: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]], int]:
        """Build hierarchy, weak-focus and shuffled-cycle query slots."""
        cfg = self.config
        chosen: list[int] = []
        chosen_ontology: set[int] = set()

        n_pairs = int(cfg.hierarchy_pairs_per_episode)
        if n_pairs > 0:
            order = self.rng.permutation(self.hierarchy_pairs.shape[1])
            used_edges = 0
            for edge_index in order.tolist():
                child = int(self.hierarchy_pairs[0, edge_index])
                parent = int(self.hierarchy_pairs[1, edge_index])
                ontology_rows = (
                    int(self.task_to_ontology_go[child]),
                    int(self.task_to_ontology_go[parent]),
                )
                if child in chosen or parent in chosen:
                    continue
                if ontology_rows[0] == ontology_rows[1]:
                    continue
                if ontology_rows[0] in chosen_ontology or ontology_rows[1] in chosen_ontology:
                    continue
                chosen.extend([child, parent])
                chosen_ontology.update(ontology_rows)
                used_edges += 1
                if used_edges >= n_pairs:
                    break
            if used_edges < n_pairs:
                raise RuntimeError(
                    "could not sample the requested number of disjoint hierarchy pairs"
                )

        focus_anchors: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        focus_scanned = 0
        if int(cfg.weak_focus_queries_per_episode) > 0:
            if epoch is None or global_episode is None:
                raise ValueError("weak-focus scheduling requires epoch/global_episode context")
            focus_anchors, focus_scanned = self._sample_weak_focus_anchors(
                epoch=int(epoch),
                rank=int(rank),
                world_size=int(world_size),
                excluded_task=set(chosen),
                excluded_ontology=set(chosen_ontology),
            )
            chosen.extend(int(go_idx) for go_idx in focus_anchors)
            chosen_ontology.update(
                int(self.task_to_ontology_go[int(go_idx)]) for go_idx in focus_anchors
            )

        coverage_count = int(self.coverage_slots_per_episode)
        if coverage_count > 0:
            if (
                cfg.query_sampling_mode == "shuffled_cycle"
                and epoch is not None
                and global_episode is not None
            ):
                coverage = self._cycle_candidates(
                    epoch=int(epoch),
                    global_episode=int(global_episode),
                    count=coverage_count,
                    excluded_task=set(chosen),
                    excluded_ontology=set(chosen_ontology),
                )
            else:
                candidates = [
                    int(go_idx)
                    for go_idx in self.eligible_go.tolist()
                    if int(go_idx) not in chosen
                    and int(self.task_to_ontology_go[int(go_idx)]) not in chosen_ontology
                ]
                if len(candidates) < coverage_count:
                    raise RuntimeError("not enough GO terms for coverage query slots")
                coverage = self.rng.choice(
                    np.asarray(candidates, dtype=np.int64),
                    size=coverage_count,
                    replace=False,
                ).astype(np.int64).tolist()
            chosen.extend(coverage)
            chosen_ontology.update(
                int(self.task_to_ontology_go[int(go_idx)]) for go_idx in coverage
            )

        # Focus selection may occasionally find fewer distinct eligible GO terms
        # than requested. Fill those non-coverage slots randomly; the fixed
        # shuffled-cycle cursor remains untouched, so GO coverage accounting is
        # still exact.
        remaining = int(cfg.num_queries) - len(chosen)
        if remaining > 0:
            candidates = [
                int(go_idx)
                for go_idx in self.eligible_go.tolist()
                if int(go_idx) not in chosen
                and int(self.task_to_ontology_go[int(go_idx)]) not in chosen_ontology
            ]
            if len(candidates) < remaining:
                raise RuntimeError("not enough GO terms to complete hybrid query episode")
            extra = self.rng.choice(
                np.asarray(candidates, dtype=np.int64), size=remaining, replace=False
            ).astype(np.int64)
            chosen.extend(extra.tolist())

        if len(chosen) != int(cfg.num_queries):
            raise RuntimeError(
                f"hybrid query plan produced {len(chosen)} rows, expected {cfg.num_queries}"
            )
        query = np.asarray(chosen, dtype=np.int64)
        query = query[self.rng.permutation(query.size)]
        return query, focus_anchors, int(focus_scanned)

    def _sample_query_indices(
        self, *, epoch: Optional[int] = None, global_episode: Optional[int] = None
    ) -> np.ndarray:
        """Backward-compatible query-only wrapper used by older tests."""
        query, _, _ = self._sample_query_plan(
            epoch=epoch, global_episode=global_episode, rank=0, world_size=1
        )
        return query

    def _sample_queries_and_support(
        self,
        *,
        epoch: Optional[int] = None,
        global_episode: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[
        np.ndarray,
        list[np.ndarray],
        np.ndarray,
        dict[int, tuple[np.ndarray, np.ndarray]],
        int,
    ]:
        cfg = self.config
        for _ in range(cfg.max_query_resample_attempts):
            query, focus_anchors, focus_scanned = self._sample_query_plan(
                epoch=epoch,
                global_episode=global_episode,
                rank=int(rank),
                world_size=int(world_size),
            )
            # Canonical/alt-ID classifier columns may map to the same full
            # BoxSquaredEL class. Keep ontology queries unique inside an
            # episode while preserving the immutable task label space.
            if np.unique(self.task_to_ontology_go[query]).size != query.size:
                continue
            support: list[np.ndarray] = []
            gold_positive_counts: list[int] = []
            for go_idx in query.tolist():
                gold = np.unique(self.gold.get(go_idx)["protein_idx"].astype(np.int64))
                support_count, positive_count = self._gold_requirements(go_idx)
                chosen = self.rng.choice(
                    gold, size=support_count, replace=False
                ).astype(np.int64)
                support.append(chosen)
                gold_positive_counts.append(int(positive_count))
            union = np.unique(np.concatenate(support))
            valid = True
            for row, go_idx in enumerate(query.tolist()):
                gold = np.unique(self.gold.get(go_idx)["protein_idx"].astype(np.int64))
                remaining = np.setdiff1d(gold, union, assume_unique=False)
                if remaining.size < gold_positive_counts[row]:
                    valid = False
                    break
            if valid:
                return (
                    query,
                    support,
                    np.asarray(gold_positive_counts, dtype=np.int64),
                    focus_anchors,
                    int(focus_scanned),
                )
        raise RuntimeError("failed to sample episode-disjoint support/query positives")

    def sample(
        self,
        *,
        seed: Optional[int] = None,
        epoch: Optional[int] = None,
        global_episode: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> NBSGlobalEpisode:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        cfg = self.config
        (
            query_go,
            support_by_query,
            gold_positive_counts,
            weak_focus_anchors,
            weak_focus_scanned,
        ) = self._sample_queries_and_support(
            epoch=epoch,
            global_episode=global_episode,
            rank=int(rank),
            world_size=int(world_size),
        )
        support_union = np.unique(np.concatenate(support_by_query))
        # Reserve every weak-focus anchor before any per-query fill selection.
        # Without this pre-marking, an anchor belonging to a later query could
        # be consumed as an ordinary pseudo positive by an earlier GO, reducing
        # unique weak-protein coverage inside the same episode.
        weak_focus_anchor_new = 0
        if weak_focus_anchors:
            if epoch is None:
                raise ValueError("weak-focus pseudo targets require an epoch context")
            anchor_parts = [
                np.asarray(values[0], dtype=np.int64)
                for values in weak_focus_anchors.values()
                if np.asarray(values[0]).size
            ]
            if anchor_parts:
                weak_focus_anchor_new = self._mark_pseudo_seen(
                    np.unique(np.concatenate(anchor_parts)), epoch=int(epoch)
                )
        per_query_gold_positive: list[np.ndarray] = []
        per_query_hard: list[np.ndarray] = []
        per_query_hard_rank: list[np.ndarray] = []
        per_query_pseudo: list[np.ndarray] = []
        per_query_pseudo_prob: list[np.ndarray] = []
        # Full evidence pools are retained only while constructing this episode.
        # Background-PU sampling excludes every known gold/pseudo/candidate
        # protein for the current GO, not just the subset selected as direct
        # supervision in this mini-batch.
        per_query_all_gold: list[np.ndarray] = []
        per_query_all_pseudo: list[np.ndarray] = []
        per_query_all_candidate: list[np.ndarray] = []

        for query_row, go_idx in enumerate(query_go.tolist()):
            gold = np.unique(self.gold.get(go_idx)["protein_idx"].astype(np.int64))
            per_query_all_gold.append(gold)
            remaining_gold = np.setdiff1d(gold, support_union, assume_unique=False)
            positive_count = int(gold_positive_counts[query_row])
            if positive_count > 0:
                gold_positive = self.rng.choice(
                    remaining_gold,
                    size=positive_count,
                    replace=False,
                ).astype(np.int64)
            else:
                gold_positive = np.empty(0, dtype=np.int64)
            per_query_gold_positive.append(gold_positive)

            # The weak-set supervision contract is independent of whether this
            # stage currently assigns a non-zero pseudo loss.  Every pair in
            # the pseudo CSR is a modelout-positive annotation (p > 0.5), so it
            # must NEVER re-enter the sampled-unlabelled/hard-negative pool.
            all_pseudo_protein = np.empty(0, dtype=np.int64)
            all_pseudo_prob = np.empty(0, dtype=np.float32)
            if self.pseudo is not None:
                pseudo_values = self.pseudo.get(go_idx)
                all_pseudo_protein = pseudo_values["protein_idx"].astype(np.int64)
                all_pseudo_prob = np.asarray(
                    pseudo_values.get(
                        "probability",
                        np.ones(all_pseudo_protein.size, dtype=np.float32),
                    ),
                    dtype=np.float32,
                )
                if all_pseudo_prob.shape != all_pseudo_protein.shape:
                    raise ValueError("pseudo probability payload does not align with protein_idx")
                if all_pseudo_prob.size:
                    if not np.all(np.isfinite(all_pseudo_prob)):
                        raise ValueError("pseudo probability contains non-finite values")
                    if np.any(all_pseudo_prob < 0.5) or np.any(all_pseudo_prob > 1.0):
                        raise ValueError(
                            "pseudo CSR must contain only modelout annotations with "
                            "0.5 <= stored probability <= 1.0; CSR membership itself records the pre-quantization modelout > 0.5 decision"
                        )

            per_query_all_pseudo.append(all_pseudo_protein.astype(np.int64, copy=False))

            candidate_values = self.candidate.get(go_idx)
            hard_protein = candidate_values["protein_idx"].astype(np.int64)
            per_query_all_candidate.append(hard_protein.astype(np.int64, copy=False))
            hard_rank = candidate_values.get(
                "source_rank", np.zeros(hard_protein.size, dtype=np.uint16)
            )
            excluded = np.union1d(gold, support_union)
            if all_pseudo_protein.size:
                excluded = np.union1d(excluded, all_pseudo_protein)
            keep = ~np.isin(hard_protein, excluded, assume_unique=False)
            hard_protein = hard_protein[keep]
            hard_rank = np.asarray(hard_rank)[keep]
            if hard_protein.size:
                count = min(cfg.hard_candidate_per_query, hard_protein.size)
                choice = self.rng.choice(hard_protein.size, size=count, replace=False)
                per_query_hard.append(hard_protein[choice])
                per_query_hard_rank.append(hard_rank[choice])
            else:
                per_query_hard.append(np.empty(0, dtype=np.int64))
                per_query_hard_rank.append(np.empty(0, dtype=np.uint16))

            pseudo_protein = np.empty(0, dtype=np.int64)
            pseudo_prob = np.empty(0, dtype=np.float32)
            forced_protein, forced_prob = weak_focus_anchors.get(
                int(go_idx),
                (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)),
            )
            forced_protein = np.asarray(forced_protein, dtype=np.int64)
            forced_prob = np.asarray(forced_prob, dtype=np.float32)
            if forced_protein.size and epoch is None:
                raise ValueError("weak-focus pseudo targets require an epoch context")
            # All focus anchors were pre-marked before entering this loop so
            # ordinary pseudo filling cannot consume another query's anchor.
            if cfg.pseudo_positive_per_query > 0 and all_pseudo_protein.size:
                keep = ~np.isin(
                    all_pseudo_protein, np.union1d(gold, support_union),
                    assume_unique=False,
                )
                available_protein = all_pseudo_protein[keep]
                available_prob = all_pseudo_prob[keep]
                if forced_protein.size:
                    keep_forced = ~np.isin(
                        available_protein, forced_protein, assume_unique=False
                    )
                    available_protein = available_protein[keep_forced]
                    available_prob = available_prob[keep_forced]

                desired = int(cfg.pseudo_positive_per_query)
                if int(go_idx) in weak_focus_anchors:
                    desired = min(
                        desired, int(cfg.weak_focus_targets_per_query)
                    )
                remaining = max(0, desired - int(forced_protein.size))
                selected_protein = np.empty(0, dtype=np.int64)
                selected_prob = np.empty(0, dtype=np.float32)
                if remaining > 0 and available_protein.size:
                    count = min(remaining, int(available_protein.size))
                    if (
                        cfg.pseudo_sampling_mode == "go_cyclic_unique"
                        and epoch is not None
                        and global_episode is not None
                    ):
                        choice = self._unique_pseudo_indices(
                            available_protein,
                            count=int(count),
                            go_idx=int(go_idx),
                            epoch=int(epoch),
                            global_episode=int(global_episode),
                            rank=int(rank),
                            world_size=int(world_size),
                        )
                    elif (
                        cfg.pseudo_sampling_mode == "go_cyclic"
                        and epoch is not None
                        and global_episode is not None
                    ):
                        choice = self._cyclic_pseudo_indices(
                            size=int(available_protein.size),
                            count=int(count),
                            go_idx=int(go_idx),
                            epoch=int(epoch),
                            global_episode=int(global_episode),
                        )
                    else:
                        choice = self.rng.choice(
                            available_protein.size, size=count, replace=False
                        )
                    selected_protein = available_protein[choice]
                    selected_prob = available_prob[choice]
                pseudo_protein = np.concatenate(
                    [forced_protein, selected_protein]
                ).astype(np.int64, copy=False)
                pseudo_prob = np.concatenate(
                    [forced_prob, selected_prob]
                ).astype(np.float32, copy=False)
            elif forced_protein.size:
                # This path is mainly a defensive invariant.  Formal weak-focus
                # configs require pseudo_positive_per_query > 0.
                pseudo_protein = forced_protein
                pseudo_prob = forced_prob
            per_query_pseudo.append(pseudo_protein)
            per_query_pseudo_prob.append(pseudo_prob)

        candidate_union = np.unique(
            np.concatenate(
                per_query_gold_positive + per_query_hard + per_query_pseudo
            )
        )
        candidate_union = np.setdiff1d(candidate_union, support_union, assume_unique=False)
        candidate_union_before_cap = int(candidate_union.size)
        if candidate_union.size > cfg.max_candidates:
            required_parts = list(per_query_gold_positive) + [
                values for values in per_query_pseudo if values.size
            ]
            required = np.unique(np.concatenate(required_parts))
            optional = np.setdiff1d(candidate_union, required, assume_unique=False)
            remaining = cfg.max_candidates - required.size
            if remaining < 0:
                raise RuntimeError("max_candidates is smaller than required gold positives")
            if optional.size > remaining:
                optional = self.rng.choice(optional, size=remaining, replace=False)
            candidate_union = np.unique(np.concatenate([required, optional])).astype(np.int64)
        candidate_position = {
            int(protein): column for column, protein in enumerate(candidate_union.tolist())
        }
        q, c = query_go.size, candidate_union.size
        # Base logits are available for the complete shared candidate union and
        # are also used to define conservative PU-background supervision.
        base_logits = self.base_logit_store.gather_matrix(candidate_union, query_go)
        labels = np.zeros((q, c), dtype=np.float32)
        mask = np.zeros((q, c), dtype=bool)
        confidence = np.ones((q, c), dtype=np.float32)
        pseudo_mask = np.zeros((q, c), dtype=bool)
        supervision_weight = np.zeros((q, c), dtype=np.float32)
        candidate_evidence = np.zeros((q, c, 3), dtype=np.float32)
        retained_gold_pairs = 0
        retained_hard_pairs = 0
        retained_pseudo_pairs = 0
        retained_background_pairs = 0
        retained_weak_focus_pairs = 0
        retained_gold_proteins: list[int] = []
        retained_hard_proteins: list[int] = []
        retained_pseudo_proteins: list[int] = []
        retained_background_proteins: list[int] = []
        background_rows = 0

        for row, go_idx in enumerate(query_go.tolist()):
            for protein in per_query_gold_positive[row].tolist():
                column = candidate_position.get(int(protein))
                if column is not None:
                    labels[row, column] = 1.0
                    mask[row, column] = True
                    supervision_weight[row, column] = 1.0
                    retained_gold_pairs += 1
                    retained_gold_proteins.append(int(protein))
            hard = per_query_hard[row]
            hard_rank = per_query_hard_rank[row]
            hard_attr = None
            if self.candidate_attributes is not None and hard.size:
                hard_attr = self.candidate_attributes.gather(hard, hard_rank)
            for index, protein in enumerate(hard.tolist()):
                column = candidate_position.get(int(protein))
                if column is None:
                    continue
                mask[row, column] = True
                supervision_weight[row, column] = cfg.sampled_unlabelled_weight
                retained_hard_pairs += 1
                retained_hard_proteins.append(int(protein))
                if hard_attr is not None:
                    candidate_evidence[row, column] = hard_attr[index]
                else:
                    rank = float(hard_rank[index])
                    candidate_evidence[row, column, 2] = 1.0 / (1.0 + rank)
            for protein, probability in zip(
                per_query_pseudo[row].tolist(), per_query_pseudo_prob[row].tolist()
            ):
                column = candidate_position.get(int(protein))
                if column is None:
                    continue
                labels[row, column] = float(probability)
                mask[row, column] = True
                pseudo_mask[row, column] = True
                normalized = max(0.0, min(1.0, (float(probability) - 0.5) / 0.5))
                confidence[row, column] = normalized ** cfg.pseudo_confidence_power
                supervision_weight[row, column] = 1.0
                retained_pseudo_pairs += 1
                if int(go_idx) in weak_focus_anchors:
                    retained_weak_focus_pairs += 1
                retained_pseudo_proteins.append(int(protein))

        # Add a small amount of conservative PU background contrast for every
        # GO query.  These pairs are sampled only from proteins already present
        # in the shared candidate union, so they do not expand the graph.  The
        # full gold, modelout-positive and backbone-candidate pools for this GO
        # are excluded before applying a very-low base-probability threshold.
        if int(cfg.background_unlabelled_per_query) > 0:
            for row, _go_idx in enumerate(query_go.tolist()):
                excluded_parts = [
                    per_query_all_gold[row],
                    per_query_all_pseudo[row],
                    per_query_all_candidate[row],
                ]
                excluded_nonempty = [part for part in excluded_parts if part.size]
                excluded = (
                    np.unique(np.concatenate(excluded_nonempty)).astype(np.int64)
                    if excluded_nonempty
                    else np.empty(0, dtype=np.int64)
                )
                columns = select_background_unlabelled_columns(
                    candidate_protein_idx=candidate_union,
                    base_logits_row=base_logits[row],
                    occupied_mask=mask[row],
                    excluded_protein_idx=excluded,
                    count=int(cfg.background_unlabelled_per_query),
                    max_probability=float(cfg.background_base_probability_max),
                    rng=self.rng,
                )
                if columns.size:
                    background_rows += 1
                    mask[row, columns] = True
                    # labels are initialized to zero; retain an explicit low PU
                    # weight rather than asserting these pairs as true negatives.
                    supervision_weight[row, columns] = float(
                        cfg.background_unlabelled_weight
                    )
                    retained_background_pairs += int(columns.size)
                    retained_background_proteins.extend(
                        candidate_union[columns].astype(np.int64).tolist()
                    )

        positive_mask = mask & (labels > 0.0)
        negative_mask = mask & (~pseudo_mask) & (labels <= 0.0)
        positive_rows_mask = np.any(positive_mask, axis=1)
        negative_rows_mask = np.any(negative_mask, axis=1)
        positive_rows = int(np.count_nonzero(positive_rows_mask))
        negative_rows = int(np.count_nonzero(negative_rows_mask))
        positive_only_rows = int(np.count_nonzero(positive_rows_mask & ~negative_rows_mask))
        negative_only_rows = int(np.count_nonzero(negative_rows_mask & ~positive_rows_mask))
        mixed_rows = int(np.count_nonzero(positive_rows_mask & negative_rows_mask))
        positive_pairs = int(np.count_nonzero(positive_mask))
        negative_pairs = int(np.count_nonzero(negative_mask))
        neg_pos_pair_ratio = (
            0.0 if positive_pairs <= 0 else float(negative_pairs / positive_pairs)
        )

        seed_protein = np.concatenate(support_by_query).astype(np.int64)
        seed_query = np.concatenate(
            [np.full(values.size, row, dtype=np.int64) for row, values in enumerate(support_by_query)]
        )
        episode = NBSGlobalEpisode(
            query_go_idx=query_go.astype(np.int64),
            query_ontology_go_idx=self.task_to_ontology_go[query_go].astype(np.int64),
            seed_protein_idx=seed_protein,
            seed_query_idx=seed_query,
            candidate_protein_idx=candidate_union.astype(np.int64),
            base_logits=base_logits.astype(np.float32),
            candidate_evidence=candidate_evidence,
            query_go_frequency=self.train_go_counts[query_go].astype(np.float32),
            labels=labels,
            mask=mask,
            confidence=confidence,
            pseudo_mask=pseudo_mask,
            supervision_weight=supervision_weight,
            metadata={
                "query_go_idx": query_go.astype(np.int64).tolist(),
                # Keep exact per-query train-gold frequencies so epoch-level
                # diagnostics can report unique bucket coverage instead of an
                # average that rounds singleton exposure to zero.
                "query_gold_counts": self.gold_degree[query_go].astype(np.int64).tolist(),
                "eligible_go_count": int(self.eligible_go.size),
                "query_sampling_mode": str(cfg.query_sampling_mode),
                "gold_support_policy": str(cfg.gold_support_policy),
                "pseudo_sampling_mode": str(cfg.pseudo_sampling_mode),
                "weak_focus_queries_requested": int(cfg.weak_focus_queries_per_episode),
                "weak_focus_queries_realized": int(len(weak_focus_anchors)),
                "weak_focus_scan_count": int(weak_focus_scanned),
                "weak_focus_anchor_new_count": int(weak_focus_anchor_new),
                "weak_focus_target_capacity_requested": int(
                    len(weak_focus_anchors) * int(cfg.weak_focus_targets_per_query)
                ),
                # Selected before the global candidate-union cap.  Kept under
                # the legacy key for compatibility with v0.5.4 probes.
                "weak_focus_targets_requested": int(sum(
                    per_query_pseudo[row].size
                    for row, query_go_idx in enumerate(query_go.tolist())
                    if int(query_go_idx) in weak_focus_anchors
                )),
                "weak_focus_targets_retained": int(retained_weak_focus_pairs),
                "pseudo_seen_rank": (
                    0 if self._pseudo_seen is None else int(np.count_nonzero(self._pseudo_seen))
                ),
                "query_gold_count_eq1": int(np.sum(self.gold_degree[query_go] == 1)),
                "query_gold_count_eq2": int(np.sum(self.gold_degree[query_go] == 2)),
                "query_gold_count_3_4": int(np.sum((self.gold_degree[query_go] >= 3) & (self.gold_degree[query_go] <= 4))),
                "query_gold_count_gt4": int(np.sum(self.gold_degree[query_go] > 4)),
                "query_with_pseudo_pool": int(np.sum(self.pseudo_degree[query_go] > 0)),
                "query_with_backbone_candidate_pool": int(np.sum(self.candidate_degree[query_go] > 0)),
                "support_proteins": int(seed_protein.size),
                # Exact global protein IDs are retained only in transient batch
                # metadata.  They are consumed by epoch-level coverage audits
                # and are never written into checkpoints or dense graph stores.
                "support_protein_idx": np.unique(seed_protein).astype(np.int64),
                "candidate_protein_idx": candidate_union.astype(np.int64),
                "root_protein_idx": np.unique(
                    np.concatenate([seed_protein, candidate_union])
                ).astype(np.int64),
                "gold_target_protein_idx": np.unique(
                    np.asarray(retained_gold_proteins, dtype=np.int64)
                ),
                "hard_target_protein_idx": np.unique(
                    np.asarray(retained_hard_proteins, dtype=np.int64)
                ),
                "pseudo_target_protein_idx": np.unique(
                    np.asarray(retained_pseudo_proteins, dtype=np.int64)
                ),
                "candidate_union_before_cap": int(candidate_union_before_cap),
                "candidate_union_after_cap": int(candidate_union.size),
                "candidate_dropped_by_cap": int(max(0, candidate_union_before_cap - int(candidate_union.size))),
                "candidate_truncation_fraction": (
                    0.0 if candidate_union_before_cap <= 0 else float(max(0, candidate_union_before_cap - int(candidate_union.size)) / candidate_union_before_cap)
                ),
                "gold_pairs_requested": int(sum(values.size for values in per_query_gold_positive)),
                "gold_pairs_retained": int(retained_gold_pairs),
                "hard_pairs_requested": int(sum(values.size for values in per_query_hard)),
                "hard_pairs_retained": int(retained_hard_pairs),
                "pseudo_pairs_requested": int(sum(values.size for values in per_query_pseudo)),
                "pseudo_pairs_retained": int(retained_pseudo_pairs),
                "background_pairs_requested": int(
                    query_go.size * int(cfg.background_unlabelled_per_query)
                ),
                "background_pairs_retained": int(retained_background_pairs),
                "background_query_rows": int(background_rows),
                "positive_query_rows": int(positive_rows),
                "negative_query_rows": int(negative_rows),
                "positive_only_query_rows": int(positive_only_rows),
                "negative_only_query_rows": int(negative_only_rows),
                "mixed_query_rows": int(mixed_rows),
                "positive_supervision_pairs": int(positive_pairs),
                "negative_supervision_pairs": int(negative_pairs),
                "negative_positive_pair_ratio": float(neg_pos_pair_ratio),
                "background_target_protein_idx": np.unique(
                    np.asarray(retained_background_proteins, dtype=np.int64)
                ),
            },
        )
        episode.validate()
        return episode
