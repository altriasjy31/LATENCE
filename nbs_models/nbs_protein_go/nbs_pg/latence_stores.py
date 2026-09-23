from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


class GOProteinCSRStore:
    """Read-only GO-major CSR over global protein indices."""

    def __init__(
        self,
        indptr_path: str | Path,
        protein_idx_path: str | Path,
        *,
        payload_paths: Optional[Mapping[str, str | Path]] = None,
    ) -> None:
        self.indptr = np.load(Path(indptr_path), mmap_mode="r")
        self.protein_idx = np.load(Path(protein_idx_path), mmap_mode="r")
        self.payloads = {
            name: np.load(Path(path), mmap_mode="r")
            for name, path in (payload_paths or {}).items()
        }
        if self.indptr.ndim != 1 or self.protein_idx.ndim != 1:
            raise ValueError("GO CSR arrays must be one-dimensional")
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != self.protein_idx.size:
            raise ValueError("invalid GO CSR endpoints")
        if np.any(np.diff(np.asarray(self.indptr)) < 0):
            raise ValueError("GO CSR indptr must be nondecreasing")
        for name, payload in self.payloads.items():
            if payload.shape != self.protein_idx.shape:
                raise ValueError(f"payload {name!r} does not align with protein_idx")

    @property
    def num_go(self) -> int:
        return int(self.indptr.size - 1)

    @property
    def num_edges(self) -> int:
        return int(self.protein_idx.size)

    def degree(self, go_idx: int) -> int:
        start, end = int(self.indptr[go_idx]), int(self.indptr[go_idx + 1])
        return end - start

    def get(self, go_idx: int) -> dict[str, np.ndarray]:
        if go_idx < 0 or go_idx >= self.num_go:
            raise IndexError(f"GO index {go_idx} outside [0, {self.num_go})")
        start, end = int(self.indptr[go_idx]), int(self.indptr[go_idx + 1])
        result = {"protein_idx": np.asarray(self.protein_idx[start:end])}
        result.update(
            {name: np.asarray(payload[start:end]) for name, payload in self.payloads.items()}
        )
        return result

    def sample(
        self,
        go_idx: int,
        count: int,
        *,
        rng: np.random.Generator,
        replace: bool = False,
    ) -> dict[str, np.ndarray]:
        values = self.get(go_idx)
        size = values["protein_idx"].size
        if count < 0:
            raise ValueError("count cannot be negative")
        if count == 0 or size == 0:
            return {name: value[:0] for name, value in values.items()}
        if not replace:
            count = min(count, size)
        choice = rng.choice(size, size=count, replace=replace)
        return {name: value[choice] for name, value in values.items()}


class FixedDegreeCandidateAttributeStore:
    """Lookup three-column candidate attributes using protein and source rank."""

    def __init__(
        self,
        edge_attr_path: str | Path,
        *,
        fixed_degree: int,
        source_protein_start: int,
    ) -> None:
        self.edge_attr = np.load(Path(edge_attr_path), mmap_mode="r")
        self.fixed_degree = int(fixed_degree)
        self.source_protein_start = int(source_protein_start)
        if self.edge_attr.ndim != 2 or self.edge_attr.shape[1] != 3:
            raise ValueError("candidate edge_attr must be [E,3]")
        if self.fixed_degree <= 0:
            raise ValueError("fixed_degree must be positive")
        if self.edge_attr.shape[0] % self.fixed_degree != 0:
            raise ValueError("edge_attr rows are not divisible by fixed_degree")

    @property
    def num_source_proteins(self) -> int:
        return int(self.edge_attr.shape[0] // self.fixed_degree)

    def source_rows(self, protein_idx: np.ndarray, source_rank: np.ndarray) -> np.ndarray:
        protein_idx = np.asarray(protein_idx, dtype=np.int64)
        source_rank = np.asarray(source_rank, dtype=np.int64)
        if protein_idx.shape != source_rank.shape:
            raise ValueError("protein_idx and source_rank must align")
        if source_rank.size and (source_rank.min() < 0 or source_rank.max() >= self.fixed_degree):
            raise IndexError("candidate source_rank outside fixed degree")
        local = protein_idx - self.source_protein_start
        if local.size and (local.min() < 0 or local.max() >= self.num_source_proteins):
            raise IndexError("candidate protein_idx outside source range")
        return local * self.fixed_degree + source_rank

    def gather(self, protein_idx: np.ndarray, source_rank: np.ndarray) -> np.ndarray:
        rows = self.source_rows(protein_idx, source_rank)
        return np.asarray(self.edge_attr[rows], dtype=np.float32)


@dataclass(frozen=True)
class RoleProbabilitySlice:
    role: str
    global_start: int
    global_end: int
    probability_path: str

    @property
    def count(self) -> int:
        return self.global_end - self.global_start


class RoleAwareBaseLogitStore:
    """Gather first-stage base logits from role-specific dense probabilities.

    This is the safe v0.4 fallback until exact logits are exported or rebuilt
    from pooled representations and the first-stage classifier.  Probability
    clipping is explicit and recorded by callers in training metadata.
    """

    def __init__(
        self,
        slices: Iterable[RoleProbabilitySlice],
        *,
        num_go: int,
        probability_clip: float = 1e-4,
    ) -> None:
        self.num_go = int(num_go)
        self.probability_clip = float(probability_clip)
        if not 0.0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must lie in (0, 0.5)")
        self.slices = sorted(slices, key=lambda item: item.global_start)
        self.arrays: dict[str, np.ndarray] = {}
        previous_end: Optional[int] = None
        for item in self.slices:
            if item.global_end <= item.global_start:
                raise ValueError("invalid role global range")
            if previous_end is not None and item.global_start < previous_end:
                raise ValueError("role global ranges overlap")
            array = np.load(item.probability_path, mmap_mode="r")
            if array.shape != (item.count, self.num_go):
                raise ValueError(
                    f"{item.role} probability shape {array.shape} != {(item.count, self.num_go)}"
                )
            self.arrays[item.role] = array
            previous_end = item.global_end

    def _slice_for(self, protein_idx: int) -> RoleProbabilitySlice:
        for item in self.slices:
            if item.global_start <= protein_idx < item.global_end:
                return item
        raise IndexError(f"protein_idx {protein_idx} is outside configured role slices")

    def gather_matrix(self, protein_idx: np.ndarray, go_idx: np.ndarray) -> np.ndarray:
        protein_idx = np.asarray(protein_idx, dtype=np.int64).reshape(-1)
        go_idx = np.asarray(go_idx, dtype=np.int64).reshape(-1)
        if go_idx.size and (go_idx.min() < 0 or go_idx.max() >= self.num_go):
            raise IndexError("GO index outside classifier range")
        probabilities = np.empty((go_idx.size, protein_idx.size), dtype=np.float32)
        for column, protein in enumerate(protein_idx.tolist()):
            item = self._slice_for(protein)
            local = protein - item.global_start
            probabilities[:, column] = np.asarray(
                self.arrays[item.role][local, go_idx], dtype=np.float32
            )
        p = np.clip(
            probabilities, self.probability_clip, 1.0 - self.probability_clip
        )
        return np.log(p) - np.log1p(-p)


class GOBoxStore:
    @classmethod
    def from_alignment_manifest(cls, manifest_path: str | Path) -> "GOBoxStore":
        path = Path(manifest_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        arrays = payload.get("arrays", {})
        try:
            center = path.parent / arrays["center"]["file"]
            offset = path.parent / arrays["offset"]["file"]
            stats_spec = arrays.get("stats")
            stats = None if stats_spec is None else path.parent / stats_spec["file"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid GO box alignment manifest: {path}") from exc
        store = cls(center, offset, stats_path=stats)
        expected_dim = int(payload.get("embedding_dim", store.center.shape[1]))
        if store.center.shape[1] != expected_dim:
            raise ValueError(
                f"GO box dimension {store.center.shape[1]} != manifest {expected_dim}"
            )
        expected_go = int(payload.get("classifier_go_terms", store.center.shape[0]))
        if store.center.shape[0] != expected_go:
            raise ValueError(
                f"GO box rows {store.center.shape[0]} != manifest {expected_go}"
            )
        return store

    def __init__(
        self,
        center_path: str | Path,
        offset_path: str | Path,
        *,
        stats_path: Optional[str | Path] = None,
    ) -> None:
        self.center = np.load(Path(center_path), mmap_mode="r")
        self.offset = np.load(Path(offset_path), mmap_mode="r")
        self.stats = (
            None if stats_path is None else np.load(Path(stats_path), mmap_mode="r")
        )
        if self.center.shape != self.offset.shape or self.center.ndim != 2:
            raise ValueError("GO center and offset must align as [G,box_dim]")
        if np.any(np.asarray(self.offset[: min(1024, self.offset.shape[0])]) <= 0):
            raise ValueError("GO offsets must be positive")
        if self.stats is not None and self.stats.shape[0] != self.center.shape[0]:
            raise ValueError("GO stats must align with center rows")

    def gather(self, go_idx: np.ndarray) -> dict[str, np.ndarray]:
        go_idx = np.asarray(go_idx, dtype=np.int64)
        result = {
            "center": np.asarray(self.center[go_idx], dtype=np.float32),
            "offset": np.asarray(self.offset[go_idx], dtype=np.float32),
        }
        if self.stats is not None:
            result["stats"] = np.asarray(self.stats[go_idx], dtype=np.float32)
        return result


def load_go_protein_stores(
    manifest_path: str | Path,
) -> dict[str, GOProteinCSRStore]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stores: dict[str, GOProteinCSRStore] = {}
    for name, spec in manifest.get("indices", {}).items():
        payloads = dict(spec.get("payloads", {}))
        stores[name] = GOProteinCSRStore(
            spec["indptr"], spec["protein_idx"], payload_paths=payloads
        )
    return stores
