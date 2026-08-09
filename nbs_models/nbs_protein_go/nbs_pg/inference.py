from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .episode import NBSGlobalEpisode
from .local_loader import LatenceNBSLocalGraphMaterializer, LatenceNBSStores
from .types import ProteinGOQueryBatch


@dataclass
class FullTaskInferenceConfig:
    go_chunk_size: int = 256
    protein_batch_size: int = 256
    support_per_query: int = 2
    support_seed: int = 3407
    probability_clip: float = 1e-5

    def validate(self) -> None:
        if self.go_chunk_size <= 0 or self.protein_batch_size <= 0:
            raise ValueError("GO/protein inference chunk sizes must be positive")
        if self.support_per_query < 0:
            raise ValueError("support_per_query cannot be negative")
        if not 0.0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must lie in (0,0.5)")


class ExternalCandidateEvidenceStore:
    """Optional fixed-K first-stage candidate evidence for external proteins.

    ``go_index`` is [N,K] task-level GO columns and ``edge_attr`` is [N,K,F].
    Missing Protein--GO pairs produce zero evidence, which is the normal NBS
    semantics for non-candidate task labels.
    """

    def __init__(self, go_index_path: str | Path, edge_attr_path: str | Path) -> None:
        self.go_index = np.load(go_index_path, mmap_mode="r")
        raw_attr = np.load(edge_attr_path, mmap_mode="r")
        if self.go_index.ndim != 2:
            raise ValueError("external candidate GO index must be [N,K]")
        if raw_attr.ndim == 2 and raw_attr.shape[0] == self.go_index.size:
            raw_attr = raw_attr.reshape(self.go_index.shape[0], self.go_index.shape[1], -1)
        if raw_attr.ndim != 3 or raw_attr.shape[:2] != self.go_index.shape:
            raise ValueError("external candidate edge_attr must be [N,K,F]")
        self.edge_attr = raw_attr
        self.feature_dim = int(raw_attr.shape[2])

    def gather(self, rows: np.ndarray, go_idx: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        go_idx = np.asarray(go_idx, dtype=np.int64)
        out = np.zeros((go_idx.size, rows.size, self.feature_dim), dtype=np.float32)
        for col, row in enumerate(rows.tolist()):
            row_go = np.asarray(self.go_index[row], dtype=np.int64)
            row_attr = np.asarray(self.edge_attr[row], dtype=np.float32)
            # K is small (typically 512), so a Python dict is cheaper than a
            # dense N x G evidence tensor and preserves the sparse contract.
            pos = {int(go): i for i, go in enumerate(row_go.tolist())}
            for q, go in enumerate(go_idx.tolist()):
                idx = pos.get(int(go))
                if idx is not None:
                    out[q, col] = row_attr[idx]
        return out


def _deterministic_support(
    stores: LatenceNBSStores,
    query_go: np.ndarray,
    *,
    support_per_query: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    proteins: list[np.ndarray] = []
    query_rows: list[np.ndarray] = []
    for row, go_idx in enumerate(query_go.tolist()):
        gold = np.unique(stores.episode_sampler.gold.get(int(go_idx))["protein_idx"].astype(np.int64))
        take = min(int(support_per_query), int(gold.size))
        if take <= 0:
            continue
        # Stable, GO-specific selection avoids inference-time randomness while
        # not systematically preferring low registry IDs.
        rng = np.random.default_rng(int(seed) + int(go_idx) * 104_729)
        choice = rng.choice(gold.size, size=take, replace=False)
        selected = gold[choice].astype(np.int64, copy=False)
        proteins.append(selected)
        query_rows.append(np.full(selected.size, row, dtype=np.int64))
    if not proteins:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(proteins), np.concatenate(query_rows)


def build_support_episode(
    stores: LatenceNBSStores,
    query_go: np.ndarray,
    *,
    support_per_query: int,
    seed: int,
) -> NBSGlobalEpisode:
    query_go = np.asarray(query_go, dtype=np.int64)
    seed_protein, seed_query = _deterministic_support(
        stores, query_go, support_per_query=support_per_query, seed=seed
    )
    q = int(query_go.size)
    empty_2d = np.zeros((q, 0), dtype=np.float32)
    episode = NBSGlobalEpisode(
        query_go_idx=query_go,
        query_ontology_go_idx=np.asarray(stores.task_to_ontology[query_go], dtype=np.int64),
        seed_protein_idx=seed_protein,
        seed_query_idx=seed_query,
        candidate_protein_idx=np.empty(0, dtype=np.int64),
        base_logits=empty_2d.copy(),
        candidate_evidence=np.zeros((q, 0, 3), dtype=np.float32),
        query_go_frequency=np.asarray(
            stores.episode_sampler.train_go_counts[query_go], dtype=np.float32
        ),
        labels=empty_2d.copy(),
        mask=np.zeros((q, 0), dtype=np.bool_),
        confidence=empty_2d.copy(),
        pseudo_mask=np.zeros((q, 0), dtype=np.bool_),
        supervision_weight=empty_2d.copy(),
        metadata={"inference_support_only": True},
    )
    episode.validate()
    return episode


def probability_to_logit(probability: np.ndarray, eps: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)
    clipped = np.clip(probability, eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


@torch.no_grad()
def export_full_task_probabilities(
    *,
    model,
    stores: LatenceNBSStores,
    materializer: LatenceNBSLocalGraphMaterializer,
    global_go_cache,
    external_repr: np.ndarray,
    base_values: np.ndarray,
    output_path: str | Path,
    device: torch.device | str,
    config: FullTaskInferenceConfig,
    base_values_are_logits: bool = False,
    candidate_evidence_store: Optional[ExternalCandidateEvidenceStore] = None,
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> np.memmap:
    """Export [N_external, num_task_GO] NBS probabilities in GO chunks.

    Training-time Q is irrelevant here: every task classifier column is visited
    exactly once.  ``go_chunk_size`` is only an inference memory/throughput knob.
    """
    config.validate()
    external_repr = np.asarray(external_repr)
    base_values = np.asarray(base_values)
    if external_repr.ndim != 2:
        raise ValueError("external_repr must be [N,D]")
    if base_values.ndim != 2:
        raise ValueError("base logits/probabilities must be [N,G]")
    if external_repr.shape[0] != base_values.shape[0]:
        raise ValueError("external representation/base rows disagree")
    if base_values.shape[1] != int(stores.num_task_go):
        raise ValueError("base prediction columns do not match task GO space")
    if candidate_evidence_store is not None and candidate_evidence_store.go_index.shape[0] != external_repr.shape[0]:
        raise ValueError("external candidate evidence rows disagree with external proteins")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(external_repr.shape[0], stores.num_task_go),
    )
    device = torch.device(device)
    model.eval()
    num_go = int(stores.num_task_go)
    num_protein = int(external_repr.shape[0])

    for go_start in range(0, num_go, int(config.go_chunk_size)):
        go_end = min(num_go, go_start + int(config.go_chunk_size))
        query_go = np.arange(go_start, go_end, dtype=np.int64)
        support_episode = build_support_episode(
            stores,
            query_go,
            support_per_query=int(config.support_per_query),
            seed=int(config.support_seed),
        )
        local = materializer.materialize(
            support_episode,
            seed=int(config.support_seed) + go_start * 31,
        )
        graph = local.graph.to(device)
        encoded = model.encode_graph(graph, global_go_cache=global_go_cache)
        template = local.query.to(device)

        for p_start in range(0, num_protein, int(config.protein_batch_size)):
            p_end = min(num_protein, p_start + int(config.protein_batch_size))
            rows = np.arange(p_start, p_end, dtype=np.int64)
            raw_base = np.asarray(base_values[p_start:p_end, go_start:go_end], dtype=np.float32)
            logits = raw_base if base_values_are_logits else probability_to_logit(raw_base, config.probability_clip)
            if candidate_evidence_store is None:
                evidence = np.zeros((query_go.size, rows.size, model.config.candidate_evidence_dim), dtype=np.float32)
            else:
                evidence = candidate_evidence_store.gather(rows, query_go)
            query = ProteinGOQueryBatch(
                seed_protein_index=template.seed_protein_index,
                seed_query_index=template.seed_query_index,
                num_queries=int(query_go.size),
                query_go_index=template.query_go_index,
                query_go_global_index=template.query_go_global_index,
                go_query_index=template.go_query_index,
                candidate_protein_index=None,
                base_logits=torch.as_tensor(logits.T, dtype=torch.float32, device=device),
                candidate_evidence=torch.as_tensor(evidence, dtype=torch.float32, device=device),
                query_go_frequency=torch.as_tensor(
                    stores.episode_sampler.train_go_counts[query_go],
                    dtype=torch.float32,
                    device=device,
                ),
            )
            candidate_x = torch.as_tensor(
                np.asarray(external_repr[p_start:p_end], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            autocast_enabled = device.type == "cuda" and amp_dtype is not None
            with torch.autocast(
                device_type=device.type,
                dtype=(amp_dtype if amp_dtype is not None else torch.float32),
                enabled=autocast_enabled,
            ):
                result = model.score_external_candidates(
                    encoded,
                    query,
                    candidate_x,
                    return_aux=False,
                )
            probability = torch.sigmoid(result.logits).T.float().cpu().numpy()
            output[p_start:p_end, go_start:go_end] = probability.astype(np.float16)
        output.flush()
    return output
