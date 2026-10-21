"""Three genuine relation-specific blocks with inductive ghost queries.

V084 caches remain immutable candidate pools.  Each block samples independent
incoming relations, preserving a protein pair that has several relation types.
All supervision proteins in the global DDP macro batch are removed from the
support graph, not merely stripped of labels.  Ghosts retain their own frozen
features/candidates and only retrieve cosine core neighbors at every layer.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .full_task_data import FullTaskData, _sha256
from .full_task_data_v084 import FullTaskDataV084, RELATIONS, NATIVE_RELATIONS, POOL_MANIFEST


DEFAULT_FANOUTS = (
    {name: 1 for name in RELATIONS},
    {name: 2 for name in RELATIONS},
    {name: (32 if name == "cosine" else 0) for name in RELATIONS},
)


class FullTaskDataV088(FullTaskDataV084):
    VERSION = "0.8.8"
    NUM_LAYERS = 3

    def __init__(self, config: Mapping[str, Any], stores=None):
        super().__init__(config, stores=stores)
        supplied = self.sampler.get("fanouts", DEFAULT_FANOUTS)
        if not isinstance(supplied, (list, tuple)) or len(supplied) != self.NUM_LAYERS:
            raise ValueError("v088 sampler.fanouts must contain three outer-to-inner relation mappings")
        fanouts = []
        for layer, item in enumerate(supplied):
            if not isinstance(item, Mapping) or set(item) - set(RELATIONS):
                raise ValueError("v088 fanouts contain an unknown relation or non-mapping layer")
            row = {}
            for relation in RELATIONS:
                count = item.get(relation, 0)
                if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0:
                    raise ValueError("v088 relation fanouts must be nonnegative integers")
                limit = self.retrieval_pool_size if relation == "cosine" else self.pool_size
                if count > limit:
                    raise ValueError(f"v088 layer {layer} {relation} fanout exceeds immutable pool size")
                row[relation] = int(count)
            if not sum(row.values()):
                raise ValueError("each v088 layer must request at least one relation")
            fanouts.append(row)
        self.fanouts = tuple(fanouts)
        limit = self.sampler.get("max_nodes", 100_000)
        if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, Integral) or limit <= 0:
            raise ValueError("v088 sampler.max_nodes must be a positive integer")
        self.max_nodes = int(limit)

    def data_contract(self):
        # Do not describe ignored v084 two-hop/native-dropout settings as if
        # they controlled the new blocks. Pool identity remains exactly v084.
        result = FullTaskData.data_contract(self)
        result.update(
            v088_sampling={"mode": self.mode, "seed": self.seed,
                "fanouts_outer_to_inner": deepcopy(list(self.fanouts)),
                "stable_fraction": self.stable_fraction,
                "candidate_topk": self.candidate_topk, "max_nodes": self.max_nodes,
                "relations": list(RELATIONS), "relation_identity_preserved": True,
                "eval": "deterministic_topk_per_receiver_layer_relation"},
            v088_query_view="negative_id_ghost; cosine_only_at_every_layer",
            v088_support_exclusion="all_global_DDP_macro_supervision_original_IDs_and_holdout",
            v088_supervision="binary_core_gold_and_weak_CSR_membership",
            v084_pool_signature=self._pool_signature,
        )
        manifest = self.pool_dir / POOL_MANIFEST
        result["v084_pool_manifest_sha256"] = _sha256(manifest) if manifest.exists() else None
        return result

    def _graph(self, protein_x, base_logits, candidate_go, candidate_attr,
               neighbors, attrs, device):
        """Base reader supplies query arrays/labels; no obsolete anchor graph.

        In particular the base reader never gathers annotation-bearing anchors
        before the global macro-batch exclusion has been applied by v088.
        """
        arrays = dict(protein_x=protein_x, base_logits=base_logits,
                      candidate_go=candidate_go, candidate_attr=candidate_attr)
        return {name: torch.as_tensor(np.array(value, copy=True), device=device)
                for name, value in arrays.items()}

    def _valid_ids(self, values, name, *, nonempty=True):
        ids = np.asarray(values)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or (nonempty and not len(ids))
                or np.any(ids < 0) or np.any(ids >= self.registry.num_proteins)):
            raise ValueError(f"{name} must be a vector of valid integer protein IDs")
        ids = ids.astype(np.int64, copy=False)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"{name} must contain unique protein IDs")
        allowed = np.isin(self.registry.role_code[ids],
                         [self.registry.role_to_code["core"], self.registry.role_to_code["weak"]])
        if not allowed.all():
            raise ValueError(f"{name} must contain only core or weak proteins")
        return ids

    def _receiver_pool(self, protein, relation, ghosts):
        if protein < 0:
            if relation != "cosine":
                return np.empty(0, np.int64), np.empty((0, 3), np.float32)
            return ghosts[protein]
        if relation == "cosine":
            return self._retrieval_neighbors[protein], self._retrieval_attrs[protein]
        lookup, neighbors, attrs = self._native[RELATIONS.index(relation)]
        row = int(lookup[protein])
        if row < 0:
            return np.empty(0, np.int64), np.empty((0, 3), np.float32)
        return neighbors[row], attrs[row]

    def _sample_relation(self, protein, layer, relation, ghosts, forbidden):
        budget = self.fanouts[layer][relation]
        if not budget or (protein < 0 and relation != "cosine"):
            return [], 0
        neighbors, attrs = self._receiver_pool(protein, relation, ghosts)
        # The immutable pools are already ranked. Filtering before selecting
        # refills a removed query from the remainder of that same bounded pool.
        eligible, seen = [], set()
        for source, attr in zip(neighbors, attrs):
            source = int(source)
            if source < 0 or source == protein or source in forbidden or source in seen:
                continue
            if source >= self.registry.num_proteins or not self._permitted[source]:
                continue
            if not np.isfinite(attr).all():
                raise ValueError("v088 candidate edge attributes must be finite")
            if attr[0] <= 0:
                continue
            seen.add(source)
            eligible.append((source, np.array(attr, dtype=np.float32, copy=True)))
        take = min(budget, len(eligible))
        indices = np.arange(take)
        if self.mode == "dynamic" and self._sampling_training and len(eligible) > take:
            stable = min(take, int(take * self.stable_fraction))
            payload = f"{self.seed}:{self._sampling_step}:{self._sampling_rank}:{protein}:{layer}:{relation}".encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))
            tail = np.arange(stable, len(eligible))
            weights = 1 / np.sqrt(np.arange(len(tail), dtype=np.float64) + 1)
            chosen = rng.choice(tail, size=take - stable, replace=False, p=weights / weights.sum())
            indices = np.concatenate((np.arange(stable), np.sort(chosen)))
        return [eligible[int(index)] for index in indices], int(np.count_nonzero(indices >= take))

    def _build_blocks(self, query_ids, retrieval, forbidden):
        self._require_pools()
        query_ids = np.asarray(query_ids, np.int64)
        if not len(query_ids) or np.any(query_ids >= 0) or len(np.unique(query_ids)) != len(query_ids):
            raise ValueError("v088 query IDs must be nonempty unique negative ghost IDs")
        neighbors, attributes = retrieval
        if (neighbors.shape != (len(query_ids), self.retrieval_pool_size)
                or attributes.shape != (*neighbors.shape, 3)):
            raise ValueError("v088 ghost retrieval must match the immutable cosine pool shape")
        ghosts = {int(q): (neighbors[row], attributes[row]) for row, q in enumerate(query_ids)}
        current, reversed_blocks, new_count = query_ids.tolist(), [], 0
        for layer in reversed(range(self.NUM_LAYERS)):
            if len(current) > self.max_nodes:
                raise RuntimeError("v088 max_nodes exceeded before feature gathering; reduce microbatch/fanouts")
            edges, extra = [], set()
            current_set = set(current)
            for target, protein in enumerate(current):
                for code, relation in enumerate(RELATIONS):
                    selected, changed = self._sample_relation(protein, layer, relation, ghosts, forbidden)
                    new_count += changed
                    for source, attr in selected:
                        edges.append((source, target, code, attr))
                        if source not in current_set:
                            extra.add(source)
                    if len(current) + len(extra) > self.max_nodes:
                        raise RuntimeError("v088 max_nodes exceeded before feature gathering; reduce microbatch/fanouts")
            sources = current + sorted(extra)
            positions = {protein: index for index, protein in enumerate(sources)}
            block = dict(
                src_global_ids=np.asarray(sources, np.int64),
                dst_global_ids=np.asarray(current, np.int64),
                dst_in_src=np.asarray([positions[p] for p in current], np.int64),
                edge_index=np.asarray([(positions[s], t) for s, t, _, _ in edges], np.int64).reshape(-1, 2).T,
                edge_type=np.asarray([r for _, _, r, _ in edges], np.int64),
                edge_attr=np.asarray([a for _, _, _, a in edges], np.float32).reshape(-1, 3),
            )
            reversed_blocks.append(block)
            current = sources
        blocks = list(reversed(reversed_blocks))
        for left, right in zip(blocks, blocks[1:]):
            if not np.array_equal(left["dst_global_ids"], right["src_global_ids"]):
                raise AssertionError("v088 block destination/source chain is inconsistent")
        return blocks, new_count

    def _attach_blocks(self, result, query_ids, retrieval, excluded_support_ids, device):
        query_ids = np.asarray(query_ids, np.int64)
        forbidden = set(map(int, excluded_support_ids)) | set(map(int, self.validation_ids))
        blocks, new_count = self._build_blocks(query_ids, retrieval, forbidden)
        outer = blocks[0]["src_global_ids"]
        known_positions = np.flatnonzero(outer >= 0)
        known_ids = outer[known_positions]
        if forbidden.intersection(map(int, known_ids)):
            raise AssertionError("v088 global supervision/holdout protein entered support graph")
        query_positions = {int(q): row for row, q in enumerate(query_ids)}
        ghost_positions = np.flatnonzero(outer < 0)
        ghost_rows = np.asarray([query_positions[int(outer[i])] for i in ghost_positions], np.int64)
        protein_x = np.empty((len(outer), self.feature_dim), np.float32)
        if len(known_ids):
            protein_x[known_positions] = self.stores.features.gather(known_ids)
        protein_x[ghost_positions] = result["protein_x"].detach().cpu().numpy()[ghost_rows]
        query_go, query_attr = self._truncate_candidates(
            result["candidate_go"].detach().cpu().numpy(), result["candidate_attr"].detach().cpu().numpy())
        candidate_go = np.full((len(outer), query_go.shape[1]), -1, np.int64)
        candidate_attr = np.zeros((*candidate_go.shape, 3), np.float32)
        if len(known_ids):
            go, attr = self._small_candidates(known_ids)
            if go.shape[1] != candidate_go.shape[1]:
                raise ValueError("v088 support/query candidate widths must share the same contract")
            candidate_go[known_positions], candidate_attr[known_positions] = go, attr
        candidate_go[ghost_positions], candidate_attr[ghost_positions] = query_go[ghost_rows], query_attr[ghost_rows]
        gold, pseudo = self._label_edges(outer, forbidden)
        node_type = np.full(len(outer), 2, np.int64)
        if len(known_ids):
            weak = self.registry.role_code[known_ids] == self.registry.role_to_code["weak"]
            node_type[known_positions] = weak.astype(np.int64)
        mapping = {int(p): i for i, p in enumerate(outer)}
        arrays = dict(sampled_global_ids=outer, sampled_protein_x=protein_x,
                      sampled_candidate_go=candidate_go, sampled_candidate_attr=candidate_attr,
                      sampled_gold_edge=gold, sampled_pseudo_edge=pseudo,
                      sampled_node_type=node_type,
                      sampled_seed_index=np.asarray([mapping[int(q)] for q in query_ids], np.int64))
        result.update({name: torch.as_tensor(np.array(value, copy=True), device=device)
                       for name, value in arrays.items()})
        result["blocks"] = [{name: torch.as_tensor(np.array(value, copy=True), device=device)
                             for name, value in block.items()} for block in blocks]
        edge_count = sum(block["edge_index"].shape[1] for block in blocks)
        stats = dict(sampled_nodes=len(outer), sampled_edges=edge_count,
                     sampled_support_nodes=len(known_ids), sampled_ghost_queries=len(query_ids),
                     sampled_new_neighbor_fraction=new_count / max(edge_count, 1),
                     sampled_first_neighbors=blocks[-1]["edge_index"].shape[1] / len(query_ids),
                     sampled_forbidden_support_nodes=0)
        for layer, block in enumerate(blocks):
            stats.update({f"sampled_layer_{layer}_src_nodes": len(block["src_global_ids"]),
                          f"sampled_layer_{layer}_dst_nodes": len(block["dst_global_ids"]),
                          f"sampled_layer_{layer}_edges": block["edge_index"].shape[1]})
        for code, relation in enumerate(RELATIONS):
            stats[f"sampled_relation_{relation}_edges"] = sum(int((b["edge_type"] == code).sum()) for b in blocks)
        result["sampler_diagnostics"] = {key: torch.tensor(float(value), device=device) for key, value in stats.items()}
        return result

    def batch(self, protein_ids, device="cpu", *, supervision_seed_ids=None):
        local = self._valid_ids(protein_ids, "v088 local queries")
        global_ids = local if supervision_seed_ids is None else self._valid_ids(
            supervision_seed_ids, "v088 global supervision seeds")
        if not np.isin(local, global_ids).all():
            raise ValueError("v088 global supervision seeds must include every local query")
        # Call the original reader directly: bypass both old sampled-graph and
        # v087 local label-mask adapters. Targets are binarized by membership.
        result = FullTaskData.batch(self, local, device=device)
        result["targets"] = result["positive_mask"].float()
        self._require_pools()
        return self._attach_blocks(result, -local - 1,
            (self._retrieval_neighbors[local], self._retrieval_attrs[local]), global_ids, device)

    def inference_batch(self, input_dir, row_ids, device="cpu"):
        context = (self._sampling_step, self._sampling_rank, self._sampling_training)
        self.set_sampling_context(0, 0, False)
        try:
            result = FullTaskData.inference_batch(self, input_dir, row_ids, device=device)
            arrays = self._inference[str(Path(input_dir).resolve())]
            rows = np.asarray(row_ids, np.int64).reshape(-1)
            if len(np.unique(rows)) != len(rows):
                raise ValueError("v088 external query row IDs must be unique")
            return self._attach_blocks(result, -rows - 1,
                (arrays["v084_neighbors"][rows], arrays["v084_attrs"][rows]), (), device)
        finally:
            self.set_sampling_context(*context)
