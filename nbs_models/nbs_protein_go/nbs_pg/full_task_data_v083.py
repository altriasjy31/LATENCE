"""Shared native protein-graph context for v0.8.3 train/inductive queries.

A query retrieves the same core anchor pool used in v081. Each anchor additionally
receives bounded INCOMING, native P-P edges. Context proteins contribute raw
representations and backbone candidate GO only: their gold/pseudo labels never
enter this context, including when a context protein happens to be the target.
Weak supervision retains the audited pseudo-CSR membership but binarizes targets;
there is no dense modelout loader or probability target in this module.
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

PP_RELATIONS = ("ppi", "similar_to", "weak_to_core")
PP_INDEX_FILE = "pp_context_neighbors.i32.npy"
PP_ATTR_FILE = "pp_context_attr.f32.npy"
PP_MANIFEST_FILE = "pp_context_manifest.json"


def _array_identity(array):
    filename = getattr(array, "filename", None)
    if filename:
        path = Path(filename).resolve()
        stat = path.stat()
        return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "shape": list(array.shape), "dtype": str(array.dtype)}
    value = np.asarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def _merge_incoming_topk(neighbors, attributes, destinations, sources, edge_attr):
    """Merge an edge block into exact per-center top-k, deduplicating sources.

    Confidence, raw score, reciprocal rank (descending), then source ID
    (ascending) define a total deterministic order. Work is vectorized over
    edges and restricted to centers touched by this block.
    """
    if not len(destinations):
        return
    touched = np.unique(destinations)
    old = neighbors[touched]
    valid = old >= 0
    dst = np.concatenate((np.broadcast_to(touched[:, None], old.shape)[valid], destinations))
    src = np.concatenate((old[valid], sources))
    attr = np.concatenate((attributes[touched][valid], edge_attr))
    # Order identical (center, source) pairs together, best duplicate first.
    order = np.lexsort((-attr[:, 2], -attr[:, 1], -attr[:, 0], src, dst))
    dst, src, attr = dst[order], src[order], attr[order]
    unique = np.ones(len(dst), dtype=bool)
    unique[1:] = (dst[1:] != dst[:-1]) | (src[1:] != src[:-1])
    dst, src, attr = dst[unique], src[unique], attr[unique]
    order = np.lexsort((src, -attr[:, 2], -attr[:, 1], -attr[:, 0], dst))
    dst, src, attr = dst[order], src[order], attr[order]
    group_start = np.r_[True, dst[1:] != dst[:-1]]
    position = np.arange(len(dst))
    rank = position - np.maximum.accumulate(np.where(group_start, position, 0))
    take = rank < neighbors.shape[1]
    neighbors[touched] = -1
    attributes[touched] = 0
    neighbors[dst[take], rank[take]] = src[take]
    attributes[dst[take], rank[take]] = attr[take]


class FullTaskDataV083(FullTaskData):
    def __init__(self, config: Mapping[str, Any], stores=None):
        super().__init__(config, stores=stores)
        self.use_pp_context = bool(self.options.get("use_pp_context", True))
        self.pp_fanout = int(self.options.get("pp_context_fanout", 4))
        self.pp_candidate_topk = int(self.options.get("pp_candidate_topk", 16))
        self.pp_block_size = int(self.options.get("pp_scan_block_size", 1_000_000))
        if min(self.pp_fanout, self.pp_candidate_topk, self.pp_block_size) <= 0:
            raise ValueError("PP context fanout, candidate top-k and scan block size must be positive")
        self.pp_cache_dir = Path(self.options.get(
            "pp_context_cache", Path(config["data"]["root"]) / "nbs_indices/full_task_pp_context_v083"
        )).resolve()
        self._pp_neighbors = self._pp_attrs = None
        self._preparing_core_cache = False
        self._pp_signature = self._pp_source_signature() if self.use_pp_context else None

    def _pp_source_signature(self):
        stores = getattr(self.stores, "pp", {})
        if not any(name in stores for name in PP_RELATIONS):
            raise ValueError("v083 PP context requires the native pp sampling stores; use_pp_context=False is the local ablation")
        relation_sources = {}
        for name in PP_RELATIONS:
            store = stores.get(name)
            relation_sources[name] = None if store is None else {
                "edge_index": _array_identity(store.edge_index),
                "edge_attr": _array_identity(store.edge_attr),
            }
        data = self.config["data"]
        manifest_hash = None
        if data.get("pp_sampling_indices_manifest"):
            path = Path(data["pp_sampling_indices_manifest"])
            if not path.is_absolute():
                path = Path(data["root"]) / path
            manifest_hash = _sha256(path)
        return {
            "schema": 1, "algorithm": "native_source_to_destination_incoming_exact_topk",
            "ranking": "confidence,score,reciprocal_rank_descending_then_source_id_ascending",
            "relations": list(PP_RELATIONS), "relation_sources": relation_sources,
            "pp_sampling_manifest_sha256": manifest_hash,
            "registry_sha256": self._signature["registry_sha256"],
            "anchor_ids_sha256": self._signature["anchor_ids_sha256"],
            "validation_ids_sha256": hashlib.sha256(self.validation_ids.tobytes()).hexdigest(),
            "num_proteins": self.registry.num_proteins, "fanout": self.pp_fanout,
            "context_content": "raw_features_and_backbone_candidate_go_only_no_neighbor_labels",
        }

    def data_contract(self):
        contract = super().data_contract()
        contract.update({
            "v083_supervision": "binary_original_pseudo_csr_membership_and_core_gold",
            "use_pp_context": self.use_pp_context,
            "pp_context_signature": self._pp_signature,
            "pp_candidate_topk": self.pp_candidate_topk,
        })
        manifest = self.pp_cache_dir / PP_MANIFEST_FILE
        contract["pp_context_manifest_sha256"] = (
            _sha256(manifest) if self.use_pp_context and manifest.exists() else None
        )
        return contract

    def _load_pp_context(self):
        if not self.use_pp_context:
            return True
        path = self.pp_cache_dir / PP_MANIFEST_FILE
        if not path.exists():
            return False
        manifest = json.loads(path.read_text())
        if manifest.get("signature") != self._pp_signature:
            raise ValueError("PP-context cache differs from native graph/split/fanout; use a new pp_context_cache or force prepare")
        for name in (PP_INDEX_FILE, PP_ATTR_FILE):
            file = self.pp_cache_dir / name
            if not file.exists() or _sha256(file) != manifest.get("arrays", {}).get(name):
                raise ValueError(f"PP-context cache hash mismatch: {name}; rerun prepare with force=True")
        neighbors = np.load(self.pp_cache_dir / PP_INDEX_FILE, mmap_mode="r")
        attrs = np.load(self.pp_cache_dir / PP_ATTR_FILE, mmap_mode="r")
        shape = (len(self.anchor_core_ids), len(PP_RELATIONS), self.pp_fanout)
        if neighbors.shape != shape or attrs.shape != (*shape, 3):
            raise ValueError("PP-context cache shape differs from native graph contract")
        if neighbors.dtype != np.int32 or attrs.dtype != np.float32:
            raise ValueError("PP-context cache dtype differs from native graph contract")
        if np.any(neighbors < -1) or np.any(neighbors >= self.registry.num_proteins) or not np.isfinite(attrs).all():
            raise ValueError("PP-context cache contains invalid protein IDs/attributes")
        self._pp_neighbors, self._pp_attrs = neighbors, attrs
        return True

    def _load_neighbors(self):
        core_ok = super()._load_neighbors()
        if self._preparing_core_cache or not core_ok:
            return core_ok
        return self._load_pp_context()

    def prepare_neighbors(self, cache_dir=None, *, backend="torch", device="cpu",
                          query_batch_size=256, force=False):
        # The parent calls self._load_neighbors while building its cosine cache;
        # an absent PP cache must not cause a costly valid cosine cache rebuild.
        self._preparing_core_cache = True
        try:
            result = super().prepare_neighbors(cache_dir, backend=backend, device=device,
                                               query_batch_size=query_batch_size, force=force)
        finally:
            self._preparing_core_cache = False
        if self.use_pp_context:
            self.prepare_pp_context(force=force)
        return result

    def prepare_pp_context(self, *, force=False):
        if not self.use_pp_context:
            return None
        if not force and self._load_pp_context():
            manifest_path = self.pp_cache_dir / PP_MANIFEST_FILE
            cached = json.loads(manifest_path.read_text())
            for name, coverage in cached.get("coverage", {}).items():
                print(f"[v083 PP context reuse] {name}: {coverage['retained_unique_edges']} edges, "
                      f"{coverage['centers_with_context']}/{coverage['anchor_centers']} centers "
                      f"({coverage['center_coverage']:.1%})", flush=True)
            return manifest_path
        if self.registry.num_proteins > np.iinfo(np.int32).max:
            raise ValueError("PP-context cache int32 capacity exceeded")
        self.pp_cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.pp_cache_dir / PP_MANIFEST_FILE
        manifest_path.unlink(missing_ok=True)
        shape = (len(self.anchor_core_ids), len(PP_RELATIONS), self.pp_fanout)
        neighbors = np.full(shape, -1, dtype=np.int32)
        attrs = np.zeros((*shape, 3), dtype=np.float32)
        # Never sample heldout/test roles as context. Core validation proteins
        # are excluded too, even though no label would be read from them.
        permitted = np.zeros(self.registry.num_proteins, dtype=bool)
        permitted[self.anchor_core_ids] = True
        permitted[np.asarray(self.registry.role_global_indices.get("weak", []), np.int64)] = True
        report = {}
        for relation, name in enumerate(PP_RELATIONS):
            store = getattr(self.stores, "pp", {}).get(name)
            raw_count = accepted_count = 0
            if store is not None:
                edges, source_attrs = store.edge_index, store.edge_attr
                if edges.ndim != 2 or edges.shape[0] != 2 or source_attrs.shape != (edges.shape[1], 3):
                    raise ValueError(f"native PP relation {name} must have edge_index [2,E] and edge_attr [E,3]")
                raw_count = edges.shape[1]
                print(f"[v083 PP scan] {name}: {raw_count} native edges; incoming fanout={self.pp_fanout}", flush=True)
                for start in range(0, raw_count, self.pp_block_size):
                    block = np.asarray(edges[:, start:start + self.pp_block_size], dtype=np.int64)
                    values = np.asarray(source_attrs[start:start + self.pp_block_size], dtype=np.float32)
                    if np.any(block < 0) or np.any(block >= self.registry.num_proteins) or not np.isfinite(values).all():
                        raise ValueError(f"native PP relation {name} contains invalid IDs or nonfinite attributes")
                    # Native edge direction is immutable. A CSR key_axis is an
                    # indexing choice, never permission to reverse an edge.
                    source, target = block
                    local = self._anchor_local[target]
                    valid = (local >= 0) & (source != target) & permitted[source]
                    accepted_count += int(valid.sum())
                    _merge_incoming_topk(neighbors[:, relation], attrs[:, relation],
                                         local[valid], source[valid], values[valid])
                    completed = min(start + self.pp_block_size, raw_count)
                    if start == 0 or (start // self.pp_block_size + 1) % 10 == 0 or completed == raw_count:
                        print(f"[v083 PP scan] {name}: {completed}/{raw_count}", flush=True)
            valid = neighbors[:, relation] >= 0
            report[name] = {
                "present": store is not None, "source_edges": int(raw_count),
                "eligible_incoming_edges_before_dedup": accepted_count,
                "retained_unique_edges": int(valid.sum()),
                "centers_with_context": int(valid.any(1).sum()),
                "center_coverage": float(valid.any(1).mean()) if len(valid) else 0.0,
                "anchor_centers": len(self.anchor_core_ids), "fanout": self.pp_fanout,
            }
            print(f"[v083 PP context] {name}: {report[name]['retained_unique_edges']} edges, "
                  f"{report[name]['centers_with_context']}/{len(self.anchor_core_ids)} centers "
                  f"({report[name]['center_coverage']:.1%})", flush=True)
        for name, array in ((PP_INDEX_FILE, neighbors), (PP_ATTR_FILE, attrs)):
            temporary = self.pp_cache_dir / (name + ".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, array)
            os.replace(temporary, self.pp_cache_dir / name)
        payload = {"signature": self._pp_signature, "coverage": report,
                   "edge_attr_columns": ["confidence", "score", "reciprocal_rank"],
                   "arrays": {name: _sha256(self.pp_cache_dir / name) for name in (PP_INDEX_FILE, PP_ATTR_FILE)}}
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, manifest_path)
        self._load_pp_context()
        return manifest_path

    def _pp_candidate_rows(self, protein_ids):
        candidates, attributes = self._candidate_rows(protein_ids)
        count = min(self.pp_candidate_topk, candidates.shape[1])
        # Original reciprocal rank is more informative than file storage order.
        # Invalid candidates remain padding, with their attributes zeroed.
        valid = (candidates >= 0) & (candidates < self.num_task_go)
        ranks = np.where(valid, attributes[..., 2], -np.inf)
        order = np.lexsort((candidates, -attributes[..., 0], -ranks), axis=1)[:, :count]
        candidates = np.take_along_axis(candidates, order, axis=1)
        attributes = np.take_along_axis(attributes, order[..., None], axis=1)
        valid = (candidates >= 0) & (candidates < self.num_task_go)
        return np.where(valid, candidates, -1), np.where(valid[..., None], attributes, 0)

    def _graph(self, protein_x, base_logits, candidate_go, candidate_attr, neighbors, attrs, device):
        result = super()._graph(protein_x, base_logits, candidate_go, candidate_attr, neighbors, attrs, device)
        if not self.use_pp_context:
            return result
        if self._pp_neighbors is None and not self._load_pp_context():
            raise RuntimeError("native PP-context cache is absent; run v083 prepare once")
        # Identical unique-anchor order to FullTaskData._graph, train and test.
        anchors = np.unique(neighbors)
        global_nodes = np.asarray(self._pp_neighbors[anchors], dtype=np.int64)
        valid = global_nodes >= 0
        unique_nodes, inverse = np.unique(global_nodes[valid], return_inverse=True)
        local = np.full(global_nodes.shape, -1, dtype=np.int64)
        local[valid] = inverse
        candidates, candidate_attributes = self._pp_candidate_rows(unique_nodes)
        arrays = {
            "pp_protein_x": self.stores.features.gather(unique_nodes),
            "pp_candidate_go": candidates,
            "pp_candidate_attr": candidate_attributes,
            "pp_neighbor_index": local,
            "pp_neighbor_attr": np.asarray(self._pp_attrs[anchors]),
        }
        result.update({key: torch.from_numpy(np.array(value, copy=True)).to(device) for key, value in arrays.items()})
        return result

    def batch(self, protein_ids, device="cpu"):
        result = super().batch(protein_ids, device=device)
        # Membership, including >.5 values rounded to FP16 .5, is preserved.
        # Modelout magnitudes cannot enter the optimization target.
        result["targets"] = result["positive_mask"].to(torch.float32)
        return result
