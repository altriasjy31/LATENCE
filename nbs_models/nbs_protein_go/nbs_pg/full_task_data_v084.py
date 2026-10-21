"""Bounded two-hop, typed ego graphs with label-safe GO evidence (v0.8.4).

The immutable cache is a candidate POOL, not a sampled training graph. Incoming
PPI/similarity/weak->core edges and an explicit core->weak inverse are sampled
with the same budgets in fixed and dynamic runs. A separate cosine pool uses
identical train/external retrieval. Dynamic sampling is stateless, so resume
requires no hidden sampler RNG. Modelout magnitudes are never model inputs or
optimization targets; OTHER weak proteins contribute only pseudo membership.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .full_task_data import FullTaskData, _sha256
from .full_task_data_v083 import _array_identity, _merge_incoming_topk

RELATIONS = ("ppi", "similar_to", "weak_to_core", "core_to_weak", "cosine")
NATIVE_RELATIONS = RELATIONS[:4]
POOL_MANIFEST = "v084_pool_manifest.json"


def _rng(seed, step, rank, protein, stream=0):
    value = f"{seed}:{step}:{rank}:{protein}:{stream}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(value).digest()[:8], "little"))


def _atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


class FullTaskDataV084(FullTaskData):
    def __init__(self, config: Mapping[str, Any], stores=None):
        super().__init__(config, stores=stores)
        self.sampler = dict(self.options.get("sampler", {}))
        self.mode = str(self.sampler.get("mode", "fixed"))
        self.seed = int(self.sampler.get("seed", 8084))
        self.first_hop = int(self.sampler.get("first_hop", 32))
        self.second_hop = int(self.sampler.get("second_hop", 4))
        self.pool_size = int(self.sampler.get("native_pool_per_relation", 64))
        self.retrieval_pool_size = int(self.sampler.get("retrieval_pool_size", 64))
        self.candidate_topk = int(self.sampler.get("candidate_topk", 16))
        self.block_size = int(self.sampler.get("scan_block_size", 1_000_000))
        self.stable_fraction = float(self.sampler.get("stable_fraction", .5))
        self.native_dropout = float(self.sampler.get("native_dropout", .25))
        if self.mode not in {"fixed", "dynamic"}:
            raise ValueError("v084 sampler mode must be fixed or dynamic")
        if min(self.first_hop, self.second_hop, self.pool_size,
               self.retrieval_pool_size, self.candidate_topk, self.block_size) <= 0:
            raise ValueError("v084 sampler budgets must be positive")
        if not 0 <= self.stable_fraction <= 1 or not 0 <= self.native_dropout <= 1:
            raise ValueError("sampler stable_fraction/native_dropout must be in [0,1]")
        if self.retrieval_pool_size >= len(self.anchor_core_ids):
            raise ValueError("retrieval_pool_size must be smaller than training anchor count")
        self.pool_dir = Path(self.sampler.get("cache_dir", Path(config["data"]["root"]) /
                                             "nbs_indices/full_task_pools_v084")).resolve()
        self._native = None
        self._retrieval_neighbors = self._retrieval_attrs = None
        self._sampling_step, self._sampling_rank, self._sampling_training = 0, 0, False
        self._permitted = np.zeros(self.registry.num_proteins, dtype=bool)
        self._permitted[self.anchor_core_ids] = True
        self._permitted[np.asarray(self.registry.role_global_indices.get("weak", []), np.int64)] = True
        self._pool_signature = self._build_pool_signature()

    def set_sampling_context(self, step=0, rank=0, training=False):
        self._sampling_step = int(step)
        self._sampling_rank = int(rank)
        self._sampling_training = bool(training)

    def _build_pool_signature(self):
        pp = getattr(self.stores, "pp", {})
        sources = {name: None if pp.get(name) is None else {
            "edge_index": _array_identity(pp[name].edge_index),
            "edge_attr": _array_identity(pp[name].edge_attr),
        } for name in NATIVE_RELATIONS[:3]}
        return {
            "version": "0.8.4", "algorithm": "typed_incoming_topk_explicit_weak_inverse_v1",
            "source": dict(self._signature), "native_sources": sources,
            "relations": list(RELATIONS), "pool_size": self.pool_size,
            "retrieval_pool_size": self.retrieval_pool_size,
            "permitted_sha256": hashlib.sha256(self._permitted.tobytes()).hexdigest(),
            "ranking": "confidence_score_reciprocal_rank_desc_source_id_asc",
        }

    def data_contract(self):
        result = super().data_contract()
        result.update(v084_supervision="binary_csr_membership", v084_pool_signature=self._pool_signature,
                      v084_sampling={"mode": self.mode, "seed": self.seed, "first_hop": self.first_hop,
                          "second_hop": self.second_hop, "stable_fraction": self.stable_fraction,
                          "native_dropout": self.native_dropout, "candidate_topk": self.candidate_topk},
                      v084_label_mask="all_batch_supervision_seeds_and_all_holdout_labels_excluded",
                      v084_relations=list(RELATIONS))
        manifest = self.pool_dir / POOL_MANIFEST
        result["v084_pool_manifest_sha256"] = _sha256(manifest) if manifest.exists() else None
        return result

    def _load_pools(self):
        path = self.pool_dir / POOL_MANIFEST
        if not path.exists():
            return False
        payload = json.loads(path.read_text())
        if payload.get("signature") != self._pool_signature:
            raise ValueError("v084 pool cache differs from graph/data/split; use a new sampler.cache_dir")
        required = {f"{r}_{suffix}" for r in NATIVE_RELATIONS for suffix in
                    ("centers.i32.npy", "neighbors.i32.npy", "attrs.f32.npy")}
        required.update(("retrieval_neighbors.i32.npy", "retrieval_attrs.f32.npy"))
        if set(payload.get("arrays", {})) != required:
            raise ValueError("v084 pool cache must bind every candidate array hash")
        for name, digest in payload.get("arrays", {}).items():
            if not (self.pool_dir / name).exists() or _sha256(self.pool_dir / name) != digest:
                raise ValueError(f"v084 pool cache hash mismatch: {name}; force prepare to repair")
        native = []
        for relation in NATIVE_RELATIONS:
            centers = np.load(self.pool_dir / f"{relation}_centers.i32.npy", mmap_mode="r")
            neighbors = np.load(self.pool_dir / f"{relation}_neighbors.i32.npy", mmap_mode="r")
            attrs = np.load(self.pool_dir / f"{relation}_attrs.f32.npy", mmap_mode="r")
            if neighbors.shape != (len(centers), self.pool_size) or attrs.shape != (*neighbors.shape, 3):
                raise ValueError("v084 native pool shape mismatch")
            lookup = np.full(self.registry.num_proteins, -1, np.int32)
            lookup[centers] = np.arange(len(centers), dtype=np.int32)
            native.append((lookup, neighbors, attrs))
        neighbors = np.load(self.pool_dir / "retrieval_neighbors.i32.npy", mmap_mode="r")
        attrs = np.load(self.pool_dir / "retrieval_attrs.f32.npy", mmap_mode="r")
        shape = (self.registry.num_proteins, self.retrieval_pool_size)
        if neighbors.shape != shape or attrs.shape != (*shape, 3):
            raise ValueError("v084 retrieval pool shape mismatch")
        self._native, self._retrieval_neighbors, self._retrieval_attrs = native, neighbors, attrs
        return True

    def prepare_neighbors(self, cache_dir=None, *, backend="torch", device="cpu",
                          query_batch_size=256, force=False):
        result = super().prepare_neighbors(cache_dir, backend=backend, device=device,
                                           query_batch_size=query_batch_size, force=force)
        self.prepare_pools(backend=backend, device=device, query_batch_size=query_batch_size, force=force)
        return result

    def _edge_blocks(self, relation):
        store = getattr(self.stores, "pp", {}).get("weak_to_core" if relation == "core_to_weak" else relation)
        if store is None:
            return
        edges, attributes = store.edge_index, store.edge_attr
        if edges.ndim != 2 or edges.shape[0] != 2 or attributes.shape != (edges.shape[1], 3):
            raise ValueError(f"{relation} must have [2,E] edges and [E,3] attributes")
        for start in range(0, edges.shape[1], self.block_size):
            edge = np.asarray(edges[:, start:start + self.block_size], dtype=np.int64)
            attr = np.asarray(attributes[start:start + self.block_size], dtype=np.float32)
            if np.any(edge < 0) or np.any(edge >= self.registry.num_proteins) or not np.isfinite(attr).all():
                raise ValueError(f"invalid native graph {relation}")
            source, target = edge[::-1] if relation == "core_to_weak" else edge
            valid = self._permitted[source] & self._permitted[target] & (source != target) & (attr[:, 0] > 0)
            yield source[valid], target[valid], attr[valid]

    def prepare_pools(self, *, backend="torch", device="cpu", query_batch_size=256, force=False):
        if not force and self._load_pools():
            print(f"[v084 pools] reuse verified pool cache {self.pool_dir}", flush=True)
            return self.pool_dir / POOL_MANIFEST
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.pool_dir / POOL_MANIFEST
        manifest.unlink(missing_ok=True)
        files, coverage = [], {}
        for relation in NATIVE_RELATIONS:
            present = np.zeros(self.registry.num_proteins, dtype=bool)
            edge_count = 0
            print(f"[v084 pools] indexing incoming {relation}, per-center pool={self.pool_size}", flush=True)
            for _, targets, _ in self._edge_blocks(relation):
                present[targets] = True
                edge_count += len(targets)
            centers = np.flatnonzero(present).astype(np.int32)
            lookup = np.full(self.registry.num_proteins, -1, np.int32)
            lookup[centers] = np.arange(len(centers), dtype=np.int32)
            cn, nn, an = (f"{relation}_centers.i32.npy", f"{relation}_neighbors.i32.npy", f"{relation}_attrs.f32.npy")
            np.save(self.pool_dir / cn, centers)
            neighbors = np.lib.format.open_memmap(self.pool_dir / nn, mode="w+", dtype=np.int32,
                                                  shape=(len(centers), self.pool_size))
            attrs = np.lib.format.open_memmap(self.pool_dir / an, mode="w+", dtype=np.float32,
                                              shape=(len(centers), self.pool_size, 3))
            neighbors[:] = -1
            attrs[:] = 0
            for sources, targets, values in self._edge_blocks(relation):
                _merge_incoming_topk(neighbors, attrs, lookup[targets], sources, values)
            neighbors.flush(); attrs.flush()
            kept = sum(int((neighbors[start:start + 4096] >= 0).sum()) for start in range(0, len(centers), 4096))
            coverage[relation] = {"centers": len(centers), "eligible_edges": edge_count, "retained_edges": kept}
            print(f"[v084 pools] {relation}: {len(centers)} centers, {kept} retained edges", flush=True)
            files.extend((cn, nn, an))
            del neighbors, attrs
        neighbors = np.lib.format.open_memmap(self.pool_dir / "retrieval_neighbors.i32.npy", mode="w+",
            dtype=np.int32, shape=(self.registry.num_proteins, self.retrieval_pool_size))
        attrs = np.lib.format.open_memmap(self.pool_dir / "retrieval_attrs.f32.npy", mode="w+",
            dtype=np.float32, shape=(self.registry.num_proteins, self.retrieval_pool_size, 3))
        neighbors[:] = -1; attrs[:] = 0
        query_ids = np.union1d(self.all_core_ids, self.registry.role_global_indices.get("weak", []))
        search = self._search_index(backend, device)
        for start in range(0, len(query_ids), query_batch_size):
            ids = query_ids[start:start + query_batch_size]
            indices, values = search.search(self.stores.features.gather(ids), self.retrieval_pool_size,
                                            self._anchor_local[ids])
            neighbors[ids] = self.anchor_core_ids[indices]
            attrs[ids] = values
            if start == 0 or start // query_batch_size % 100 == 0:
                print(f"[v084 cosine pools] {min(start + len(ids), len(query_ids))}/{len(query_ids)}", flush=True)
        neighbors.flush(); attrs.flush()
        files.extend(("retrieval_neighbors.i32.npy", "retrieval_attrs.f32.npy"))
        _atomic_json(manifest, {"signature": self._pool_signature, "coverage": coverage,
                               "arrays": {name: _sha256(self.pool_dir / name) for name in files}})
        self._load_pools()
        return manifest

    def _require_pools(self):
        if self._native is None and not self._load_pools():
            raise RuntimeError("v084 candidate pools absent; run prepare once")

    def ensure_prepared(self):
        """Read-only load for DDP/evaluation; never constructs a cache implicitly."""
        if self._neighbors is None and not super()._load_neighbors():
            raise RuntimeError("core-neighbor cache absent; run prepare once")
        self._require_pools()
        return True

    def _select(self, protein, budget, *, seed_node=False, external=None):
        """Relation-round-robin ordering, then stable + rank-tempered exploration."""
        groups = []
        rng = _rng(self.seed, self._sampling_step, self._sampling_rank, protein)
        drop = self._sampling_training and seed_node and rng.random() < self.native_dropout
        if external is None and not drop:
            for code, (lookup, neighbors, attrs) in enumerate(self._native):
                row = lookup[protein]
                groups.append([] if row < 0 else [(int(p), code, np.array(a, copy=True))
                    for p, a in zip(neighbors[row], attrs[row]) if p >= 0])
        else:
            groups = [[] for _ in NATIVE_RELATIONS]
        cos, values = external if external is not None else (
            self._retrieval_neighbors[protein], self._retrieval_attrs[protein])
        groups.append([(int(p), 4, np.array(a, copy=True)) for p, a in zip(cos, values) if p >= 0 and p != protein])
        # A relation's highly ranked entries cannot exhaust the whole budget.
        ordered, seen = [], set()
        for rank in range(max((len(g) for g in groups), default=0)):
            for group in groups:
                if rank < len(group) and group[rank][0] not in seen:
                    ordered.append(group[rank]); seen.add(group[rank][0])
        fixed = ordered[:budget]
        if self.mode == "dynamic" and self._sampling_training and len(ordered) > budget:
            stable = min(budget, int(budget * self.stable_fraction))
            tail = np.arange(stable, len(ordered))
            # Rank-tempered draws preserve every retained candidate's nonzero chance.
            weights = 1 / np.sqrt(np.arange(len(tail), dtype=np.float64) + 1)
            chosen = rng.choice(tail, size=budget - stable, replace=False, p=weights / weights.sum())
            chosen.sort()
            selected = ordered[:stable] + [ordered[int(i)] for i in chosen]
        else:
            selected = fixed
        new = len({x[0] for x in selected} - {x[0] for x in fixed})
        native_missing = not any(groups[:-1])
        return selected, new, bool(drop), native_missing

    def _small_candidates(self, ids):
        go, attr = self._candidate_rows(ids)
        return self._truncate_candidates(go, attr)

    def _truncate_candidates(self, go, attr):
        valid = (go >= 0) & (go < self.num_task_go)
        rank = np.where(valid, attr[..., 2], -np.inf)
        order = np.lexsort((go, -attr[..., 0], -rank), axis=1)[:, :min(self.candidate_topk, go.shape[1])]
        go = np.take_along_axis(go, order, axis=1)
        attr = np.take_along_axis(attr, order[..., None], axis=1)
        valid = (go >= 0) & (go < self.num_task_go)
        return np.where(valid, go, -1), np.where(valid[..., None], attr, 0)

    def _label_edges(self, nodes, excluded):
        gold, pseudo = [], []
        for local, protein in enumerate(nodes):
            if protein < 0 or protein in excluded or not self._permitted[protein]:
                continue
            role = int(self.registry.role_code[protein])
            if role == self.registry.role_to_code["core"]:
                store = self.stores.gold_messages
                go = np.asarray(store.go_idx[store.indptr[protein]:store.indptr[protein + 1]])
                gold.extend((local, int(g)) for g in go)
            elif role == self.registry.role_to_code["weak"]:
                row = self.stores.pseudo_messages.get_role_row(int(self.registry.role_row[protein]))
                pseudo.extend((local, int(g)) for g in row["go_idx"])
        return (np.asarray(gold, np.int64).reshape(-1, 2).T,
                np.asarray(pseudo, np.int64).reshape(-1, 2).T)

    def _sample_graph(self, result, seed_ids, device, *, external=None, loss_anchor_ids=None):
        self._require_pools()
        seed_ids = np.asarray(seed_ids, np.int64)
        if len(np.unique(seed_ids)) != len(seed_ids):
            raise ValueError("v084 requires distinct supervision seeds within each batch")
        first, second, diagnostics = {}, {}, {"new": 0, "drop": 0, "missing": 0}
        for row, protein in enumerate(seed_ids):
            ext = None if external is None else (external[0][row], external[1][row])
            chosen, new, dropped, missing = self._select(int(protein), self.first_hop, seed_node=True, external=ext)
            first[int(protein)] = chosen
            diagnostics["new"] += new; diagnostics["drop"] += dropped; diagnostics["missing"] += missing
        first_nodes = sorted({x[0] for neighbors in first.values() for x in neighbors})
        for protein in first_nodes:
            # A node that is another current seed already has the same first-hop
            # receiver graph; do not add a competing truncated receiver graph.
            if protein not in first:
                second[protein] = self._select(protein, self.second_hop)[0]
        remaining = sorted(({x[0] for neighbors in second.values() for x in neighbors} | set(first_nodes)) - set(seed_ids))
        nodes = list(map(int, seed_ids)) + remaining
        mapping = {protein: index for index, protein in enumerate(nodes)}
        edge_values = {}
        for receiver, neighbors in {**second, **first}.items():
            for source, code, attr in neighbors:
                edge_values[(mapping[source], mapping[receiver], code)] = attr
        edge_keys = sorted(edge_values)
        edges = np.asarray([[s, t] for s, t, _ in edge_keys], np.int64).reshape(-1, 2).T
        edge_type = np.asarray([r for _, _, r in edge_keys], np.int64)
        edge_attr = np.asarray([edge_values[key] for key in edge_keys], np.float32).reshape(-1, 3)
        known_positions = np.asarray([i for i, p in enumerate(nodes) if p >= 0], np.int64)
        known_ids = np.asarray([p for p in nodes if p >= 0], np.int64)
        sampled_x = np.zeros((len(nodes), self.feature_dim), np.float32)
        candidate_shape = min(self.candidate_topk, result["candidate_go"].shape[1])
        sampled_go = np.full((len(nodes), candidate_shape), -1, np.int64)
        sampled_attr = np.zeros((len(nodes), candidate_shape, 3), np.float32)
        if len(known_ids):
            sampled_x[known_positions] = self.stores.features.gather(known_ids)
            cg, ca = self._small_candidates(known_ids)
            sampled_go[known_positions], sampled_attr[known_positions] = cg, ca
        if external is not None:
            sampled_x[:len(seed_ids)] = result["protein_x"].detach().cpu().numpy()
            cg, ca = self._truncate_candidates(result["candidate_go"].detach().cpu().numpy(),
                                              result["candidate_attr"].detach().cpu().numpy())
            sampled_go[:len(seed_ids)], sampled_attr[:len(seed_ids)] = cg, ca
        excluded = set(map(int, seed_ids)) | set(map(int, self.validation_ids))
        gold, pseudo = self._label_edges(nodes, excluded)
        core_first = [[item for item in first[int(p)] if self._anchor_local[item[0]] >= 0] for p in seed_ids]
        anchors = sorted({p for row in core_first for p, _, _ in row})
        anchor_lookup = {p: i for i, p in enumerate(anchors)}
        width = max(1, max((len(row) for row in core_first), default=0))
        neighbor_index = np.full((len(seed_ids), width), -1, np.int64)
        neighbor_attr = np.zeros((len(seed_ids), width, 3), np.float32)
        for row, selected in enumerate(core_first):
            for col, (protein, _, attr) in enumerate(selected):
                neighbor_index[row, col] = anchor_lookup[protein]; neighbor_attr[row, col] = attr
        anchor_gold, _ = self._label_edges(anchors, excluded)
        arrays = dict(sampled_protein_x=sampled_x, sampled_candidate_go=sampled_go,
            sampled_candidate_attr=sampled_attr, sampled_gold_edge=gold, sampled_pseudo_edge=pseudo,
            sampled_seed_index=np.arange(len(seed_ids), dtype=np.int64),
            sampled_anchor_index=np.asarray([mapping[p] for p in anchors], np.int64),
            sampled_edge_index=edges, sampled_edge_attr=edge_attr, sampled_edge_type=edge_type,
            sampled_global_ids=np.asarray(nodes, np.int64), anchor_x=self.stores.features.gather(np.asarray(anchors, np.int64)),
            anchor_go_edge=anchor_gold, neighbor_index=neighbor_index, neighbor_attr=neighbor_attr)
        for key in ("anchor_x", "anchor_go_edge", "neighbor_index", "neighbor_attr"):
            result["loss_" + key] = result[key]
        # PU support keeps the original cosine retrieval in BOTH sampler modes,
        # but is not allowed to reintroduce a current seed's labels via another
        # seed's anchor. This exclusion does not depend on fixed/dynamic choices.
        if loss_anchor_ids is not None:
            loss_edges = result["loss_anchor_go_edge"]
            masked = np.isin(np.asarray(loss_anchor_ids), np.asarray(sorted(excluded)))
            keep = ~torch.as_tensor(masked, device=device)[loss_edges[0]]
            result["loss_anchor_go_edge"] = loss_edges[:, keep]
        result.update({key: torch.as_tensor(np.array(value, copy=True), device=device) for key, value in arrays.items()})
        total = sum(map(len, first.values()))
        scalar = dict(sampled_nodes=len(nodes), sampled_edges=len(edge_keys),
            sampled_new_neighbor_fraction=diagnostics["new"] / max(total, 1),
            sampled_native_dropout_fraction=diagnostics["drop"] / len(seed_ids),
            sampled_native_missing_fraction=diagnostics["missing"] / len(seed_ids),
            sampled_first_neighbors=total / len(seed_ids), sampled_cap_dropped=0,
            sampled_fallback_fraction=sum(code == 4 for row in first.values() for _, code, _ in row) / max(total, 1))
        scalar.update({f"sampled_relation_{name}_edges": int((edge_type == code).sum())
                       for code, name in enumerate(RELATIONS)})
        result.update({key: torch.tensor(float(value), device=device) for key, value in scalar.items()})
        result["sampler_diagnostics"] = {key: result[key] for key in scalar}
        return result

    def batch(self, protein_ids, device="cpu"):
        result = super().batch(protein_ids, device=device)
        result["targets"] = result["positive_mask"].float()
        old_anchor_ids = self.anchor_core_ids[np.unique(self._neighbors[np.asarray(protein_ids, np.int64)])]
        return self._sample_graph(result, protein_ids, device, loss_anchor_ids=old_anchor_ids)

    def _open_inference(self, input_dir, device):
        super()._open_inference(input_dir, device)
        arrays = self._inference[str(input_dir)]
        manifest = input_dir / "ind_test_input_manifest.json"
        digest = hashlib.sha256((json.dumps(self._pool_signature, sort_keys=True) + _sha256(manifest)).encode()).hexdigest()[:20]
        path = self.pool_dir / f"external_retrieval_{digest}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as payload:
                neighbors, attrs = payload["neighbors"], payload["attrs"]
                if (str(payload["neighbors_sha256"].item()) != hashlib.sha256(neighbors.tobytes()).hexdigest()
                    or str(payload["attrs_sha256"].item()) != hashlib.sha256(attrs.tobytes()).hexdigest()):
                    raise ValueError("v084 external retrieval pool hash mismatch")
        else:
            n = len(arrays["protein_x"])
            neighbors = np.empty((n, self.retrieval_pool_size), np.int32)
            attrs = np.empty((n, self.retrieval_pool_size, 3), np.float32)
            search = self._search_index(self.options.get("retrieval_backend", "torch"), device)
            for start in range(0, n, 256):
                indices, values = search.search(arrays["protein_x"][start:start + 256], self.retrieval_pool_size)
                neighbors[start:start + 256] = self.anchor_core_ids[indices]; attrs[start:start + 256] = values
            self.pool_dir.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, neighbors=neighbors, attrs=attrs,
                neighbors_sha256=hashlib.sha256(neighbors.tobytes()).hexdigest(),
                attrs_sha256=hashlib.sha256(attrs.tobytes()).hexdigest())
            os.replace(temporary, path)
        shape = (len(arrays["protein_x"]), self.retrieval_pool_size)
        if neighbors.shape != shape or attrs.shape != (*shape, 3):
            raise ValueError("v084 external retrieval pool shape mismatch")
        arrays["v084_neighbors"], arrays["v084_attrs"] = neighbors, attrs

    def inference_batch(self, input_dir, row_ids, device="cpu"):
        context = (self._sampling_step, self._sampling_rank, self._sampling_training)
        self.set_sampling_context(0, 0, False)
        try:
            result = super().inference_batch(input_dir, row_ids, device=device)
            arrays = self._inference[str(Path(input_dir).resolve())]
            rows = np.asarray(row_ids, np.int64).reshape(-1)
            return self._sample_graph(result, -rows - 1, device,
                external=(arrays["v084_neighbors"][rows], arrays["v084_attrs"][rows]),
                loss_anchor_ids=self.anchor_core_ids[np.unique(arrays["neighbors"][rows])])
        finally:
            self.set_sampling_context(*context)
