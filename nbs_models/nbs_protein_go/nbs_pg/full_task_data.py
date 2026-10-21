"""Target-protein batches over every task GO; labels never enter target graphs.

The only protein--GO inputs are Stage-1 candidate predictions and annotations
of OTHER training core proteins. Weak pseudo annotations are supervision only.
Core neighbours use the same normalized-cosine retrieval at train and test.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .local_loader import build_latence_nbs_stores


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.all(np.isfinite(rows)) or np.any(norm <= 0):
        raise ValueError("core retrieval requires finite, nonzero protein representations")
    return np.ascontiguousarray(rows / norm)


def _attrs(scores: np.ndarray) -> np.ndarray:
    out = np.empty((*scores.shape, 3), dtype=np.float32)
    out[..., 0] = np.clip(scores, 0.0, 1.0)
    out[..., 1] = scores
    out[..., 2] = 1.0 / (np.arange(scores.shape[1], dtype=np.float32) + 1)
    return out


class _CoreSearch:
    """One index per preparation pass; no all-protein matrix or ANN approximation."""

    def __init__(self, core: np.ndarray, *, backend: str, device: str | torch.device):
        self.backend = backend
        self.device = torch.device(device)
        self.count = core.shape[0]
        if backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise ImportError("Install FAISS or prepare with --backend torch") from exc
            self.index = faiss.IndexFlatIP(core.shape[1])
            self.resources = None
            if self.device.type == "cuda":
                if not hasattr(faiss, "StandardGpuResources"):
                    raise RuntimeError("CPU-only FAISS: use --backend torch for GPU preparation")
                self.resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(
                    self.resources, self.device.index or 0, self.index
                )
            for start in range(0, len(core), 8192):
                self.index.add(_normalize(core[start:start + 8192]))
        elif backend == "torch":
            self.core = torch.empty(core.shape, device=self.device, dtype=torch.float32)
            for start in range(0, len(core), 8192):
                self.core[start:start + 8192] = torch.from_numpy(
                    _normalize(core[start:start + 8192])
                ).to(self.device)
        else:
            raise ValueError("retrieval backend must be torch or faiss")

    @torch.no_grad()
    def search(self, query: np.ndarray, k: int, excluded: np.ndarray | None = None):
        # One extra result removes a core target's own annotation-bearing node.
        take = min(self.count, k + int(excluded is not None))
        if self.count < k + int(excluded is not None and np.any(excluded >= 0)):
            raise ValueError("not enough core anchors after excluding the target itself")
        query = _normalize(query)
        if self.backend == "faiss":
            scores, neighbors = self.index.search(query, take)
        else:
            query_t = torch.from_numpy(query).to(self.device)
            scores_t = torch.empty((len(query), 0), device=self.device)
            indices_t = torch.empty((len(query), 0), device=self.device, dtype=torch.long)
            # Bound score memory independently of the total core population.
            for start in range(0, self.count, 16384):
                block = query_t @ self.core[start:start + 16384].T
                score, index = block.topk(min(take, block.shape[1]), dim=1)
                score_all = torch.cat((scores_t, score), dim=1)
                index_all = torch.cat((indices_t, index + start), dim=1)
                scores_t, order = score_all.topk(min(take, score_all.shape[1]), dim=1)
                indices_t = index_all.gather(1, order)
            scores = scores_t.cpu().numpy()
            neighbors = indices_t.cpu().numpy()
        output = np.empty((len(query), k), dtype=np.int32)
        similarity = np.empty((len(query), k), dtype=np.float32)
        for row in range(len(query)):
            valid = np.ones(take, dtype=bool) if excluded is None else neighbors[row] != excluded[row]
            output[row] = neighbors[row, valid][:k]
            similarity[row] = scores[row, valid][:k]
        return output, _attrs(similarity)


class FullTaskData:
    def __init__(self, config: Mapping[str, Any], stores=None):
        self.config = dict(config)
        self.options = dict(config.get("full_task", {}))
        self.stores = stores if stores is not None else build_latence_nbs_stores(config)
        self.feature_dim = int(self.stores.feature_dim)
        self.num_task_go = int(self.stores.num_task_go)
        self.registry = self.stores.registry
        self.all_core_ids = np.asarray(self.registry.role_global_indices["core"], dtype=np.int64)
        weak = np.asarray(self.registry.role_global_indices.get("weak", []), dtype=np.int64)
        pseudo = self.stores.pseudo_messages
        if pseudo is None:
            raise ValueError("full-task training requires the audited weak pseudo CSR")
        # CSR membership records the ORIGINAL >0.5 selection. A stored FP16 0.5
        # can have been rounded from >0.5; re-thresholding would discard positives.
        self.weak_ids = weak[np.diff(np.asarray(pseudo.indptr, dtype=np.int64)) > 0]
        gold_degree = np.diff(np.asarray(self.stores.gold_messages.indptr, dtype=np.int64))
        labeled_core = self.all_core_ids[gold_degree[self.all_core_ids] > 0]
        holdout = int(self.options.get("holdout_core_count", 0))
        if holdout < 0 or holdout >= len(labeled_core):
            raise ValueError("holdout_core_count must leave at least one labeled training core")
        rng = np.random.default_rng(int(self.options.get("holdout_seed", 8080)))
        self.validation_ids = np.sort(rng.permutation(labeled_core)[:holdout])
        self.core_ids = np.setdiff1d(labeled_core, self.validation_ids)
        self.anchor_core_ids = np.setdiff1d(self.all_core_ids, self.validation_ids)
        self.k = int(self.options.get("core_neighbors", 8))
        if self.k <= 0 or self.k >= len(self.anchor_core_ids):
            raise ValueError("core_neighbors must be positive and smaller than the anchor pool")
        self.cache_dir = Path(self.options.get(
            "neighbor_cache", Path(config["data"]["root"]) / "nbs_indices/full_task_core_neighbors"
        )).resolve()
        self._neighbors = self._neighbor_attrs = None
        self._inference: dict[str, dict[str, np.ndarray]] = {}
        self._anchor_local = np.full(self.registry.num_proteins, -1, dtype=np.int64)
        self._anchor_local[self.anchor_core_ids] = np.arange(len(self.anchor_core_ids))
        self._signature = self._source_signature()
        self._contract = None
        self._go_registry_sha256 = None
        root = Path(config["data"]["root"])
        if "weak_graph_predictions_manifest" in config["data"]:
            weak_manifest_path = root / config["data"]["weak_graph_predictions_manifest"]
            weak_manifest = json.loads(weak_manifest_path.read_text())
            go_path = root / "gg_relations/go_registry.tsv"
            if not go_path.exists():
                go_path = Path(weak_manifest["go_registry"]["path"])
                if not go_path.is_absolute():
                    go_path = weak_manifest_path.parent / go_path
            self._go_registry_sha256 = _sha256(go_path)

    def data_contract(self) -> dict[str, Any]:
        """Checkpoint identity for features, label columns, supervision and split."""
        if self._contract is not None:
            return dict(self._contract)
        result = dict(self._signature)
        result["validation_ids_sha256"] = hashlib.sha256(self.validation_ids.tobytes()).hexdigest()
        mapping = np.asarray(self.stores.task_to_ontology, dtype=np.int64)
        result["task_to_ontology_sha256"] = hashlib.sha256(mapping.tobytes()).hexdigest()
        result["num_task_go"] = self.num_task_go
        result["feature_dim"] = self.feature_dim
        result["go_registry_sha256"] = self._go_registry_sha256
        data = self.config["data"]
        for key in ("weak_graph_predictions_manifest", "go_protein_inverted_index_manifest",
                    "full_go_box_manifest", "boxsqel_gg_relations_manifest"):
            if key in data:
                path = Path(data[key])
                if not path.is_absolute():
                    path = (Path(data["root"]) if key in {
                        "weak_graph_predictions_manifest", "go_protein_inverted_index_manifest"
                    } else Path.cwd()) / path
                result[key + "_sha256"] = _sha256(path)
        for name, store, attributes in (
            ("gold", self.stores.gold_messages, ("indptr", "go_idx")),
            ("pseudo", self.stores.pseudo_messages, ("indptr", "go_idx", "probability")),
        ):
            for attribute in attributes:
                array = getattr(store, attribute)
                filename = getattr(array, "filename", None)
                result[f"{name}_{attribute}_sha256"] = (
                    _sha256(filename) if filename else hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
                )
        self._contract = result
        return dict(result)

    def _source_signature(self) -> dict[str, Any]:
        feature_files = []
        for role, value in self.stores.features.arrays.items():
            filename = getattr(value, "filename", None)
            if filename:
                path = Path(filename).resolve()
                stat = path.stat()
                feature_files.append([role, str(path), stat.st_size, stat.st_mtime_ns])
        return {
            "schema": 1, "retrieval": "normalized_cosine_core_only_exclude_self",
            "registry_sha256": _sha256(self.registry.path),
            "representation_manifest_sha256": _sha256(self.stores.features.manifest_path),
            "feature_files": feature_files,
            "anchor_ids_sha256": hashlib.sha256(self.anchor_core_ids.tobytes()).hexdigest(),
            "core_neighbors": self.k, "num_proteins": self.registry.num_proteins,
        }

    def _load_neighbors(self) -> bool:
        manifest = self.cache_dir / "core_neighbor_manifest.json"
        if not manifest.exists():
            return False
        payload = json.loads(manifest.read_text())
        if payload.get("signature") != self._signature:
            raise ValueError("core-neighbor cache differs from this split/data; use a new neighbor_cache directory")
        for name in ("core_neighbors.i32.npy", "core_edge_attr.f32.npy"):
            if _sha256(self.cache_dir / name) != payload.get("arrays", {}).get(name):
                raise ValueError(f"core-neighbor cache hash mismatch: {name}; rerun prepare with force=True")
        neighbors = np.load(self.cache_dir / "core_neighbors.i32.npy", mmap_mode="r")
        attrs = np.load(self.cache_dir / "core_edge_attr.f32.npy", mmap_mode="r")
        expected = (self.registry.num_proteins, self.k)
        if neighbors.shape != expected or attrs.shape != (*expected, 3):
            raise ValueError("incomplete core-neighbor cache; rerun prepare with force=True")
        self._neighbors, self._neighbor_attrs = neighbors, attrs
        return True

    def _search_index(self, backend: str, device):
        return _CoreSearch(
            self.stores.features.gather(self.anchor_core_ids), backend=backend, device=device
        )

    def prepare_neighbors(self, cache_dir=None, *, backend="torch", device="cpu",
                          query_batch_size=256, force=False) -> Path:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        if not force and self._load_neighbors():
            return self.cache_dir / "core_neighbor_manifest.json"
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Publish the manifest last. A failed preparation cannot look complete.
        manifest = self.cache_dir / "core_neighbor_manifest.json"
        manifest.unlink(missing_ok=True)
        neighbors = np.lib.format.open_memmap(
            self.cache_dir / "core_neighbors.i32.npy", mode="w+", dtype=np.int32,
            shape=(self.registry.num_proteins, self.k)
        )
        attrs = np.lib.format.open_memmap(
            self.cache_dir / "core_edge_attr.f32.npy", mode="w+", dtype=np.float32,
            shape=(self.registry.num_proteins, self.k, 3)
        )
        neighbors[:] = -1
        attrs[:] = 0
        query_ids = np.union1d(self.all_core_ids, self.weak_ids)
        search = self._search_index(backend, device)
        print(f"[full-task prepare] {len(query_ids)} queries / {len(self.anchor_core_ids)} core anchors; {backend} {device}", flush=True)
        for start in range(0, len(query_ids), query_batch_size):
            rows = query_ids[start:start + query_batch_size]
            index, attributes = search.search(
                self.stores.features.gather(rows), self.k, self._anchor_local[rows]
            )
            neighbors[rows], attrs[rows] = index, attributes
            if start == 0 or (start // query_batch_size) % 50 == 0:
                print(f"[full-task prepare] {min(start + len(rows), len(query_ids))}/{len(query_ids)}", flush=True)
        neighbors.flush()
        attrs.flush()
        payload = {"signature": self._signature, "backend": backend, "device": str(device),
                   "validation_core_count": len(self.validation_ids),
                   "edge_attr_columns": ["confidence", "cosine_score", "reciprocal_rank"],
                   "arrays": {name: _sha256(self.cache_dir / name) for name in (
                       "core_neighbors.i32.npy", "core_edge_attr.f32.npy")}}
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, manifest)
        self._load_neighbors()
        return manifest

    def _graph(self, protein_x, base_logits, candidate_go, candidate_attr, neighbors, attrs, device):
        if np.any(neighbors < 0) or np.any(neighbors >= len(self.anchor_core_ids)):
            raise ValueError("missing or invalid prepared core neighbor for a target")
        anchors, inverse = np.unique(neighbors, return_inverse=True)
        global_anchors = self.anchor_core_ids[anchors]
        edges = self.stores.gold_messages.gather(global_anchors)
        local_edges = edges.copy()
        if edges.shape[1]:
            local_edges[0] = np.searchsorted(global_anchors, edges[0])
        arrays = {
            "protein_x": protein_x, "base_logits": base_logits,
            "candidate_go": candidate_go, "candidate_attr": candidate_attr,
            "anchor_x": self.stores.features.gather(global_anchors),
            "anchor_go_edge": local_edges,
            "neighbor_index": inverse.reshape(neighbors.shape).astype(np.int64),
            "neighbor_attr": attrs,
        }
        return {key: torch.from_numpy(np.array(value, copy=True)).to(device) for key, value in arrays.items()}

    def _candidate_rows(self, ids):
        store = self.stores.candidate_messages
        k = store.fixed_degree
        output = np.full((len(ids), k), -1, dtype=np.int64)
        attrs = np.zeros((len(ids), k, 3), dtype=np.float32)
        local = ids - store.source_start
        valid = (local >= 0) & (local < store.num_sources)
        offsets = local[valid, None] * k + np.arange(k)[None, :]
        if offsets.size:
            edges = np.asarray(store.edge_index[:, offsets.reshape(-1)])
            if not np.array_equal(edges[0], np.repeat(ids[valid], k)):
                raise ValueError("candidate store lost protein-major row alignment")
            output[valid] = edges[1].reshape(-1, k)
            attrs[valid] = np.asarray(store.edge_attr[offsets.reshape(-1)]).reshape(-1, k, 3)
        return output, attrs

    def batch(self, protein_ids, device="cpu"):
        ids = np.asarray(protein_ids, dtype=np.int64).reshape(-1)
        if not len(ids) or np.any(ids < 0) or np.any(ids >= self.registry.num_proteins):
            raise ValueError("a target batch must contain valid protein IDs")
        if self._neighbors is None and not self._load_neighbors():
            raise RuntimeError("core-neighbor cache is absent; run the full-task prepare stage once")
        is_weak = self.registry.role_code[ids] == self.registry.role_to_code["weak"]
        is_core = self.registry.role_code[ids] == self.registry.role_to_code["core"]
        if not np.all(is_core | is_weak):
            raise ValueError("only training core and weak targets are permitted")
        neighbors = np.asarray(self._neighbors[ids], dtype=np.int64)
        if np.any(self.anchor_core_ids[neighbors] == ids[:, None]):
            raise ValueError("core target self-edge would leak its gold supervision")
        candidate, candidate_attr = self._candidate_rows(ids)
        base = self.stores.episode_sampler.base_logit_store.gather_matrix(
            ids, np.arange(self.num_task_go, dtype=np.int64)
        ).T.copy()
        result = self._graph(
            self.stores.features.gather(ids), base, candidate, candidate_attr,
            neighbors, np.asarray(self._neighbor_attrs[ids]), device
        )
        targets = np.zeros((len(ids), self.num_task_go), dtype=np.float32)
        positive = np.zeros_like(targets, dtype=bool)
        for row, protein in enumerate(ids):
            if is_weak[row]:
                payload = self.stores.pseudo_messages.get_role_row(int(self.registry.role_row[protein]))
                go, probability = payload["go_idx"], payload["probability"]
                if np.any(~np.isfinite(probability)) or np.any((probability < 0.5) | (probability > 1)):
                    raise ValueError("pseudo CSR must contain the audited modelout >0.5 soft targets")
            else:
                gold = self.stores.gold_messages
                go = np.asarray(gold.go_idx[gold.indptr[protein]:gold.indptr[protein + 1]])
                probability = np.ones(len(go), dtype=np.float32)
            if np.any(go < 0) or np.any(go >= self.num_task_go):
                raise ValueError("target GO outside immutable task classifier columns")
            targets[row, go], positive[row, go] = probability, True
        result.update(targets=torch.from_numpy(targets).to(device),
                      positive_mask=torch.from_numpy(positive).to(device),
                      is_weak=torch.from_numpy(is_weak).to(device))
        return result

    def ontology(self, device="cpu"):
        boxes = self.stores.full_boxes.gather(np.arange(self.stores.full_boxes.num_go))
        result = {key: torch.from_numpy(value).to(device) for key, value in boxes.items()}
        result["task_to_ontology"] = torch.as_tensor(np.array(self.stores.task_to_ontology), device=device, dtype=torch.long)
        result["edges"] = {name: torch.as_tensor(np.array(store.edge.T), dtype=torch.long, device=device)
                           for name, store in self.stores.go_relations.items()}
        return result

    def _open_inference(self, input_dir: Path, device):
        manifest_path = input_dir / "ind_test_input_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        evidence = manifest.get("candidate_evidence", {})
        if evidence.get("selector_scope") != "full_task":
            raise ValueError("inductive cache must declare candidate_evidence.selector_scope=full_task")
        if evidence.get("expert_probability_used", True) or evidence.get("label_hint_used", True):
            raise ValueError("inductive candidate inputs must not contain expert probabilities or label hints")
        signature = manifest.get("cache_signature", {})
        if signature.get("representation_manifest_sha256") != self._signature["representation_manifest_sha256"]:
            raise ValueError("inductive input and training representations have different provenance")
        if self._go_registry_sha256 is not None and signature.get("go_registry_sha256") != self._go_registry_sha256:
            raise ValueError("inductive and training GO classifier columns have different provenance")
        core_path = Path(manifest.get("external_pp", {}).get("core_representation", "")).resolve()
        actual_core = getattr(self.stores.features.arrays["core"], "filename", None)
        if actual_core is not None and core_path != Path(actual_core).resolve():
            raise ValueError("inductive cache refers to a different core representation/order")
        filenames = {
            "protein_x": ("ind_test_repr.f16.npy", manifest["representation"].get("sha256")),
            "probability": ("backbone_ind_test_prob.f16.npy", manifest["base_probability"].get("sha256")),
            "candidate_go": ("candidate_go_index.i32.npy", evidence.get("go_index_sha256")),
            "candidate_attr": ("candidate_edge_attr.f32.npy", evidence.get("edge_attr_sha256")),
        }
        arrays = {}
        for key, (name, digest) in filenames.items():
            path = input_dir / name
            if not digest or _sha256(path) != digest:
                raise ValueError(f"inductive cache hash mismatch or missing hash: {name}")
            arrays[key] = np.load(path, mmap_mode="r")
        n = len(arrays["protein_x"])
        if arrays["protein_x"].shape != (n, self.feature_dim) or arrays["probability"].shape != (n, self.num_task_go):
            raise ValueError("inductive feature/probability dimensions differ from training")
        if arrays["candidate_go"].shape[0] != n or arrays["candidate_attr"].shape != (*arrays["candidate_go"].shape, 3):
            raise ValueError("inductive candidate arrays are misaligned")
        candidate = arrays["candidate_go"]
        if np.any(candidate < 0) or np.any(candidate >= self.num_task_go):
            raise ValueError("inductive candidate GO outside task columns")
        # Recompute only the small independent set when heldout core nodes change
        # the anchor pool. This avoids filtering a global top-k into fewer edges.
        if len(self.validation_ids):
            digest = hashlib.sha256((json.dumps(self._signature, sort_keys=True) + _sha256(manifest_path)).encode()).hexdigest()[:20]
            path = self.cache_dir / f"inductive_core_neighbors_{digest}.npz"
            cache_valid = False
            if path.exists():
                try:
                    with np.load(path, allow_pickle=False) as cached:
                        neighbors, attrs = cached["neighbors"], cached["attrs"]
                        cache_valid = (
                            neighbors.shape == (n, self.k) and attrs.shape == (n, self.k, 3)
                            and neighbors.dtype.kind in "iu" and np.isfinite(attrs).all()
                            and np.all(neighbors >= 0) and np.all(neighbors < len(self.anchor_core_ids))
                            and "neighbors_sha256" in cached and "attrs_sha256" in cached
                            and str(cached["neighbors_sha256"].item()) == hashlib.sha256(neighbors.tobytes()).hexdigest()
                            and str(cached["attrs_sha256"].item()) == hashlib.sha256(attrs.tobytes()).hexdigest()
                        )
                except (ValueError, KeyError, OSError):
                    cache_valid = False
            if not cache_valid:
                print(f"[full-task prepare] {n} inductive queries against the training-only core pool", flush=True)
                search = self._search_index(self.options.get("retrieval_backend", "torch"), device)
                neighbors = np.empty((n, self.k), dtype=np.int32)
                attrs = np.empty((n, self.k, 3), dtype=np.float32)
                for start in range(0, n, 256):
                    neighbors[start:start + 256], attrs[start:start + 256] = search.search(
                        arrays["protein_x"][start:start + 256], self.k
                    )
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp.npz")
                np.savez(temporary, neighbors=neighbors, attrs=attrs,
                         neighbors_sha256=hashlib.sha256(neighbors.tobytes()).hexdigest(),
                         attrs_sha256=hashlib.sha256(attrs.tobytes()).hexdigest())
                os.replace(temporary, path)
        else:
            pp = manifest.get("external_pp", {})
            if pp.get("edge_attr_columns") != ["confidence", "cosine_score", "reciprocal_rank"]:
                raise ValueError("prepared core retrieval does not declare the expected cosine attributes")
            neighbors = np.load(input_dir / "test_core_neighbors.i32.npy", mmap_mode="r")
            attrs = np.load(input_dir / "test_core_edge_attr.f32.npy", mmap_mode="r")
            if neighbors.shape != (n, self.k) or attrs.shape != (n, self.k, 3):
                raise ValueError("inductive cache core-neighbor k differs from training")
            for filename, key in (("test_core_neighbors.i32.npy", "neighbor_index_sha256"),
                                  ("test_core_edge_attr.f32.npy", "edge_attr_sha256")):
                if _sha256(input_dir / filename) != pp.get(key):
                    raise ValueError(f"inductive core cache hash mismatch: {filename}")
            if np.any(neighbors < 0) or np.any(neighbors >= len(self.all_core_ids)):
                raise ValueError("inductive core neighbor outside the representation's core row order")
            # The legacy preparation stores ROLE-LOCAL core rows. Anchor pools
            # are globally sorted, which need not be the same registry order.
            neighbors = self._anchor_local[self.all_core_ids[neighbors]]
        arrays["neighbors"], arrays["neighbor_attrs"] = neighbors, attrs
        self._inference[str(input_dir)] = arrays

    def inference_batch(self, input_dir, row_ids, device="cpu"):
        input_dir = Path(input_dir).resolve()
        if str(input_dir) not in self._inference:
            self._open_inference(input_dir, device)
        arrays = self._inference[str(input_dir)]
        rows = np.asarray(row_ids, dtype=np.int64).reshape(-1)
        if not len(rows) or np.any(rows < 0) or np.any(rows >= len(arrays["protein_x"])):
            raise ValueError("invalid independent-test row indices")
        clip = float(self.stores.episode_sampler.base_logit_store.probability_clip)
        probability = np.clip(np.asarray(arrays["probability"][rows], dtype=np.float32), clip, 1 - clip)
        return self._graph(
            np.asarray(arrays["protein_x"][rows], dtype=np.float32),
            np.log(probability) - np.log1p(-probability),
            np.asarray(arrays["candidate_go"][rows], dtype=np.int64),
            np.asarray(arrays["candidate_attr"][rows], dtype=np.float32),
            np.asarray(arrays["neighbors"][rows], dtype=np.int64),
            np.asarray(arrays["neighbor_attrs"][rows], dtype=np.float32), device
        )
