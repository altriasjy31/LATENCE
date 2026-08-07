from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        if not 0.0 <= self.sampled_unlabelled_weight <= 1.0:
            raise ValueError("sampled_unlabelled_weight must lie in [0,1]")
        if self.pseudo_confidence_power < 0:
            raise ValueError("pseudo_confidence_power cannot be negative")


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
        config: Optional[NBSQueryEpisodeConfig] = None,
        seed: int = 3407,
    ) -> None:
        self.gold = gold
        self.candidate = candidate
        self.pseudo = pseudo
        self.base_logit_store = base_logits
        self.candidate_attributes = candidate_attributes
        self.train_go_counts = np.asarray(train_go_counts, dtype=np.float64)
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
        min_gold = self.config.support_per_query + self.config.gold_positive_per_query
        degree = np.diff(np.asarray(self.gold.indptr, dtype=np.int64))
        self.eligible_go = np.flatnonzero(degree >= min_gold).astype(np.int64)
        if self.eligible_go.size < self.config.num_queries:
            raise ValueError("not enough GO terms with support and held-out gold proteins")
        self.rng = np.random.default_rng(seed)

    def _sample_queries_and_support(self) -> tuple[np.ndarray, list[np.ndarray]]:
        cfg = self.config
        for _ in range(cfg.max_query_resample_attempts):
            query = self.rng.choice(
                self.eligible_go, size=cfg.num_queries, replace=False
            ).astype(np.int64)
            # Canonical/alt-ID classifier columns may map to the same full
            # BoxSquaredEL class. Keep ontology queries unique inside an
            # episode while preserving the immutable task label space.
            if np.unique(self.task_to_ontology_go[query]).size != query.size:
                continue
            support: list[np.ndarray] = []
            for go_idx in query.tolist():
                gold = np.unique(self.gold.get(go_idx)["protein_idx"].astype(np.int64))
                chosen = self.rng.choice(
                    gold, size=cfg.support_per_query, replace=False
                ).astype(np.int64)
                support.append(chosen)
            union = np.unique(np.concatenate(support))
            valid = True
            for go_idx in query.tolist():
                gold = np.unique(self.gold.get(go_idx)["protein_idx"].astype(np.int64))
                remaining = np.setdiff1d(gold, union, assume_unique=False)
                if remaining.size < cfg.gold_positive_per_query:
                    valid = False
                    break
            if valid:
                return query, support
        raise RuntimeError("failed to sample episode-disjoint support/query positives")

    def sample(self, *, seed: Optional[int] = None) -> NBSGlobalEpisode:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        cfg = self.config
        query_go, support_by_query = self._sample_queries_and_support()
        support_union = np.unique(np.concatenate(support_by_query))
        per_query_gold_positive: list[np.ndarray] = []
        per_query_hard: list[np.ndarray] = []
        per_query_hard_rank: list[np.ndarray] = []
        per_query_pseudo: list[np.ndarray] = []
        per_query_pseudo_prob: list[np.ndarray] = []

        for go_idx in query_go.tolist():
            gold = np.unique(self.gold.get(go_idx)["protein_idx"].astype(np.int64))
            remaining_gold = np.setdiff1d(gold, support_union, assume_unique=False)
            gold_positive = self.rng.choice(
                remaining_gold,
                size=cfg.gold_positive_per_query,
                replace=False,
            ).astype(np.int64)
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

        for row, go_idx in enumerate(query_go.tolist()):
            for protein in per_query_gold_positive[row].tolist():
                column = candidate_position.get(int(protein))
                if column is not None:
                    labels[row, column] = 1.0
                    mask[row, column] = True
                    supervision_weight[row, column] = 1.0
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
        )
        episode.validate()
        return episode
