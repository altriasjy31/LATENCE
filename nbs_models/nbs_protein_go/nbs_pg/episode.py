from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from .latence_stores import (
    FixedDegreeCandidateAttributeStore,
    GOProteinCSRStore,
    RoleAwareBaseLogitStore,
)
from .types import ProteinGOQueryBatch


@dataclass
class NBSQueryEpisodeConfig:
    num_queries: int = 8
    support_per_query: int = 2
    gold_positive_per_query: int = 1
    hard_candidate_per_query: int = 16
    pseudo_positive_per_query: int = 0
    max_candidates: int = 256
    sampled_unlabelled_weight: float = 0.2
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
        if self.hierarchy_pairs_per_episode < 0:
            raise ValueError("hierarchy_pairs_per_episode cannot be negative")
        if 2 * self.hierarchy_pairs_per_episode > self.num_queries:
            raise ValueError(
                "2 * hierarchy_pairs_per_episode cannot exceed num_queries"
            )
        if not 0.0 <= self.sampled_unlabelled_weight <= 1.0:
            raise ValueError("sampled_unlabelled_weight must lie in [0,1]")
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
        hierarchy_pairs: Optional[np.ndarray] = None,
        config: Optional[NBSQueryEpisodeConfig] = None,
        seed: int = 3407,
    ) -> None:
        self.gold = gold
        self.candidate = candidate
        self.pseudo = pseudo
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
        singleton_all = self.gold_degree == 1
        singleton_with_pseudo = singleton_all & (self.pseudo_degree > 0)
        self.eligibility_summary = {
            "num_task_go": int(self.gold.num_go),
            "gold_positive_go": int(np.sum(self.gold_degree > 0)),
            "gold_count_eq1_all": int(np.sum(singleton_all)),
            "gold_count_eq1_with_pseudo": int(np.sum(singleton_with_pseudo)),
            "eligible_go_count": int(self.eligible_go.size),
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

    @property
    def coverage_slots_per_episode(self) -> int:
        """Number of query slots reserved for deterministic GO-cycle coverage."""
        if self.config.query_sampling_mode != "shuffled_cycle":
            return int(self.config.num_queries)
        return max(1, int(self.config.num_queries) - 2 * int(self.config.hierarchy_pairs_per_episode))

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

    def _sample_query_indices(
        self, *, epoch: Optional[int] = None, global_episode: Optional[int] = None
    ) -> np.ndarray:
        """Sample task GO columns, optionally forcing direct hierarchy pairs."""
        cfg = self.config
        n_pairs = int(cfg.hierarchy_pairs_per_episode)
        if n_pairs <= 0:
            if (
                cfg.query_sampling_mode == "shuffled_cycle"
                and epoch is not None
                and global_episode is not None
            ):
                chosen = self._cycle_candidates(
                    epoch=int(epoch),
                    global_episode=int(global_episode),
                    count=int(cfg.num_queries),
                    excluded_task=set(),
                    excluded_ontology=set(),
                )
                return np.asarray(chosen, dtype=np.int64)
            return self.rng.choice(
                self.eligible_go, size=cfg.num_queries, replace=False
            ).astype(np.int64)

        order = self.rng.permutation(self.hierarchy_pairs.shape[1])
        chosen: list[int] = []
        chosen_ontology: set[int] = set()
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

        remaining = int(cfg.num_queries) - len(chosen)
        if remaining > 0:
            if (
                cfg.query_sampling_mode == "shuffled_cycle"
                and epoch is not None
                and global_episode is not None
            ):
                extra = self._cycle_candidates(
                    epoch=int(epoch),
                    global_episode=int(global_episode),
                    count=remaining,
                    excluded_task=set(chosen),
                    excluded_ontology=set(chosen_ontology),
                )
                chosen.extend(extra)
            else:
                candidates = [
                    int(go_idx)
                    for go_idx in self.eligible_go.tolist()
                    if int(go_idx) not in chosen
                    and int(self.task_to_ontology_go[int(go_idx)]) not in chosen_ontology
                ]
                if len(candidates) < remaining:
                    raise RuntimeError("not enough GO terms to complete hierarchy-aware episode")
                extra = self.rng.choice(
                    np.asarray(candidates, dtype=np.int64),
                    size=remaining,
                    replace=False,
                ).astype(np.int64)
                chosen.extend(extra.tolist())
        query = np.asarray(chosen, dtype=np.int64)
        # Shuffle row order; hierarchy edge recovery later is index-based.
        return query[self.rng.permutation(query.size)]

    def _sample_queries_and_support(
        self, *, epoch: Optional[int] = None, global_episode: Optional[int] = None
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        cfg = self.config
        for _ in range(cfg.max_query_resample_attempts):
            query = self._sample_query_indices(epoch=epoch, global_episode=global_episode)
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
                return query, support, np.asarray(gold_positive_counts, dtype=np.int64)
        raise RuntimeError("failed to sample episode-disjoint support/query positives")

    def sample(
        self,
        *,
        seed: Optional[int] = None,
        epoch: Optional[int] = None,
        global_episode: Optional[int] = None,
    ) -> NBSGlobalEpisode:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        cfg = self.config
        query_go, support_by_query, gold_positive_counts = self._sample_queries_and_support(
            epoch=epoch, global_episode=global_episode
        )
        support_union = np.unique(np.concatenate(support_by_query))
        per_query_gold_positive: list[np.ndarray] = []
        per_query_hard: list[np.ndarray] = []
        per_query_hard_rank: list[np.ndarray] = []
        per_query_pseudo: list[np.ndarray] = []
        per_query_pseudo_prob: list[np.ndarray] = []

        for query_row, go_idx in enumerate(query_go.tolist()):
            gold = np.unique(self.gold.get(go_idx)["protein_idx"].astype(np.int64))
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

            candidate_values = self.candidate.get(go_idx)
            hard_protein = candidate_values["protein_idx"].astype(np.int64)
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
            if cfg.pseudo_positive_per_query > 0 and all_pseudo_protein.size:
                keep = ~np.isin(
                    all_pseudo_protein, np.union1d(gold, support_union),
                    assume_unique=False,
                )
                pseudo_protein = all_pseudo_protein[keep]
                pseudo_prob = all_pseudo_prob[keep]
                if pseudo_protein.size:
                    count = min(cfg.pseudo_positive_per_query, pseudo_protein.size)
                    choice = self.rng.choice(
                        pseudo_protein.size, size=count, replace=False
                    )
                    pseudo_protein = pseudo_protein[choice]
                    pseudo_prob = pseudo_prob[choice]
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
        labels = np.zeros((q, c), dtype=np.float32)
        mask = np.zeros((q, c), dtype=bool)
        confidence = np.ones((q, c), dtype=np.float32)
        pseudo_mask = np.zeros((q, c), dtype=bool)
        supervision_weight = np.zeros((q, c), dtype=np.float32)
        candidate_evidence = np.zeros((q, c, 3), dtype=np.float32)
        retained_gold_pairs = 0
        retained_hard_pairs = 0
        retained_pseudo_pairs = 0

        for row, go_idx in enumerate(query_go.tolist()):
            for protein in per_query_gold_positive[row].tolist():
                column = candidate_position.get(int(protein))
                if column is not None:
                    labels[row, column] = 1.0
                    mask[row, column] = True
                    supervision_weight[row, column] = 1.0
                    retained_gold_pairs += 1
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

        base_logits = self.base_logit_store.gather_matrix(candidate_union, query_go)
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
                "query_gold_count_eq1": int(np.sum(self.gold_degree[query_go] == 1)),
                "query_gold_count_eq2": int(np.sum(self.gold_degree[query_go] == 2)),
                "query_gold_count_3_4": int(np.sum((self.gold_degree[query_go] >= 3) & (self.gold_degree[query_go] <= 4))),
                "query_gold_count_gt4": int(np.sum(self.gold_degree[query_go] > 4)),
                "query_with_pseudo_pool": int(np.sum(self.pseudo_degree[query_go] > 0)),
                "query_with_backbone_candidate_pool": int(np.sum(self.candidate_degree[query_go] > 0)),
                "support_proteins": int(seed_protein.size),
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
            },
        )
        episode.validate()
        return episode
