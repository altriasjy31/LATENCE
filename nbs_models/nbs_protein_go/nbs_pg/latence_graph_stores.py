from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np


def resolve_data_path(base: str | Path, value: str | Path) -> Path:
    base_path = Path(base).resolve()
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else []
    candidates.extend([base_path / raw, base_path / raw.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else raw.resolve()


class ProteinRegistryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        protein_idx: list[int] = []
        roles: list[str] = []
        role_rows: list[int] = []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"protein_idx", "role", "role_row_idx"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"protein registry must contain {sorted(required)}")
            for expected, row in enumerate(reader):
                current = int(row["protein_idx"])
                if current != expected:
                    raise ValueError("protein_idx must be contiguous and row-aligned")
                protein_idx.append(current)
                roles.append(str(row["role"]))
                role_rows.append(int(row["role_row_idx"]))
        self.num_proteins = len(protein_idx)
        unique_roles = sorted(set(roles))
        self.role_to_code = {role: index for index, role in enumerate(unique_roles)}
        self.code_to_role = unique_roles
        self.role_code = np.asarray([self.role_to_code[item] for item in roles], dtype=np.int16)
        self.role_row = np.asarray(role_rows, dtype=np.int64)
        self.role_global_indices: dict[str, np.ndarray] = {}
        for role, code in self.role_to_code.items():
            indices = np.flatnonzero(self.role_code == code).astype(np.int64)
            local = self.role_row[indices]
            order = np.argsort(local, kind="stable")
            indices = indices[order]
            local = local[order]
            if not np.array_equal(local, np.arange(local.size, dtype=np.int64)):
                raise ValueError(f"role_row_idx for {role} is not contiguous")
            self.role_global_indices[role] = indices

    def role_of(self, protein_idx: int) -> str:
        return self.code_to_role[int(self.role_code[int(protein_idx)])]

    def role_local(self, protein_idx: int) -> tuple[str, int]:
        protein_idx = int(protein_idx)
        return self.role_of(protein_idx), int(self.role_row[protein_idx])


class RoleAwareProteinFeatureStore:
    def __init__(
        self,
        representation_manifest: str | Path,
        registry: ProteinRegistryStore,
    ) -> None:
        self.manifest_path = Path(representation_manifest).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.feature_dim = int(payload["feature_dim"])
        self.registry = registry
        self.arrays: dict[str, np.ndarray] = {}
        for role_spec in payload["roles"]:
            role = str(role_spec["role"])
            path = resolve_data_path(self.manifest_path.parent, role_spec["feature_file"])
            array = np.load(path, mmap_mode="r")
            expected = (int(role_spec["count"]), self.feature_dim)
            if array.shape != expected:
                raise ValueError(f"{role} feature shape {array.shape} != {expected}")
            self.arrays[role] = array

    def gather(self, protein_idx: np.ndarray) -> np.ndarray:
        indices = np.asarray(protein_idx, dtype=np.int64).reshape(-1)
        output = np.empty((indices.size, self.feature_dim), dtype=np.float32)
        codes = self.registry.role_code[indices]
        for role, code in self.registry.role_to_code.items():
            selected = codes == code
            if not np.any(selected):
                continue
            rows = self.registry.role_row[indices[selected]]
            output[selected] = np.asarray(self.arrays[role][rows], dtype=np.float32)
        return output


class EdgeOffsetCSRStore:
    """Node-keyed CSR over source COO edge offsets.

    The original edge_index and edge_attr remain memory-mapped.  ``key_axis``
    records whether the adjacency is keyed by message source (0) or destination
    (1); returned edges always preserve their original message direction.
    """

    def __init__(
        self,
        *,
        indptr_path: str | Path,
        edge_offset_path: str | Path,
        edge_index_path: str | Path,
        edge_attr_path: str | Path,
        key_axis: int,
    ) -> None:
        self.indptr = np.load(Path(indptr_path), mmap_mode="r")
        self.edge_offset = np.load(Path(edge_offset_path), mmap_mode="r")
        self.edge_index = np.load(Path(edge_index_path), mmap_mode="r")
        self.edge_attr = np.load(Path(edge_attr_path), mmap_mode="r")
        self.key_axis = int(key_axis)
        if self.key_axis not in (0, 1):
            raise ValueError("key_axis must be 0 or 1")
        if self.indptr.ndim != 1 or self.edge_offset.ndim != 1:
            raise ValueError("sampling CSR arrays must be one-dimensional")
        if int(self.indptr[-1]) != self.edge_offset.size:
            raise ValueError("sampling CSR endpoints do not align")
        if self.edge_index.shape[0] != 2 or self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise ValueError("source edge arrays are misaligned")

    @property
    def num_nodes(self) -> int:
        return int(self.indptr.size - 1)

    def sample(
        self,
        keys: Iterable[int],
        fanout: int,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if fanout <= 0:
            return np.empty((2, 0), np.int64), np.empty((0, self.edge_attr.shape[1]), np.float32)
        chosen_offsets: list[np.ndarray] = []
        for key in np.unique(np.fromiter((int(x) for x in keys), dtype=np.int64)):
            if key < 0 or key >= self.num_nodes:
                continue
            start, end = int(self.indptr[key]), int(self.indptr[key + 1])
            offsets = np.asarray(self.edge_offset[start:end], dtype=np.int64)
            if offsets.size > fanout:
                offsets = offsets[rng.choice(offsets.size, size=fanout, replace=False)]
            if offsets.size:
                chosen_offsets.append(offsets)
        if not chosen_offsets:
            return np.empty((2, 0), np.int64), np.empty((0, self.edge_attr.shape[1]), np.float32)
        offsets = np.concatenate(chosen_offsets)
        return (
            np.asarray(self.edge_index[:, offsets], dtype=np.int64),
            np.asarray(self.edge_attr[offsets], dtype=np.float32),
        )


class FixedDegreeProteinGOStore:
    def __init__(
        self,
        edge_index_path: str | Path,
        edge_attr_path: str | Path,
        *,
        fixed_degree: int,
        source_protein_start: int,
    ) -> None:
        self.edge_index = np.load(Path(edge_index_path), mmap_mode="r")
        self.edge_attr = np.load(Path(edge_attr_path), mmap_mode="r")
        self.fixed_degree = int(fixed_degree)
        self.source_start = int(source_protein_start)
        if self.edge_index.shape[0] != 2 or self.edge_attr.shape != (self.edge_index.shape[1], 3):
            raise ValueError("fixed Protein-GO source arrays are misaligned")
        if self.edge_index.shape[1] % self.fixed_degree != 0:
            raise ValueError("candidate edges are not divisible by fixed degree")
        self.num_sources = self.edge_index.shape[1] // self.fixed_degree

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def gather(
        self,
        protein_idx: Iterable[int],
        *,
        topk: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        degree = self.fixed_degree if topk is None else min(int(topk), self.fixed_degree)
        if degree <= 0:
            return np.empty((2, 0), np.int64), np.empty((0, 3), np.float32)
        proteins = np.unique(
            np.fromiter((int(x) for x in protein_idx), dtype=np.int64)
        )
        local = proteins - self.source_start
        valid = (local >= 0) & (local < self.num_sources)
        proteins = proteins[valid]
        local = local[valid]
        if proteins.size == 0:
            return np.empty((2, 0), np.int64), np.empty((0, 3), np.float32)

        # Candidate files are protein-major fixed-degree arrays.  Build all
        # requested mmap offsets in one vectorized operation instead of doing
        # tens of thousands of tiny Python slices and concatenations.
        offsets = (
            local[:, None] * self.fixed_degree
            + np.arange(degree, dtype=np.int64)[None, :]
        ).reshape(-1)
        edge = np.asarray(self.edge_index[:, offsets], dtype=np.int64)
        expected_source = np.repeat(proteins, degree)
        if edge.size and not np.array_equal(edge[0], expected_source):
            raise ValueError("candidate source is not protein-major fixed-degree")
        attr = np.asarray(self.edge_attr[offsets], dtype=np.float32)
        return edge, attr


class RoleLocalProteinGOCSRStore:
    def __init__(
        self,
        indptr_path: str | Path,
        go_index_path: str | Path,
        *,
        role: str,
        registry: ProteinRegistryStore,
        probability_path: Optional[str | Path] = None,
    ) -> None:
        self.indptr = np.load(Path(indptr_path), mmap_mode="r")
        self.go_idx = np.load(Path(go_index_path), mmap_mode="r")
        self.probability = None if probability_path is None else np.load(Path(probability_path), mmap_mode="r")
        self.role = str(role)
        self.registry = registry
        expected_rows = registry.role_global_indices[self.role].size
        if self.indptr.shape != (expected_rows + 1,) or int(self.indptr[-1]) != self.go_idx.size:
            raise ValueError("role-local Protein-GO CSR shape mismatch")
        if self.probability is not None and self.probability.shape != self.go_idx.shape:
            raise ValueError("Protein-GO probability payload mismatch")

    @property
    def num_role_rows(self) -> int:
        return int(self.indptr.size - 1)

    def get_role_row(self, role_row: int) -> dict[str, np.ndarray]:
        """Return one role-local Protein->GO row without copying the full CSR.

        This accessor is used by the weak-first query scheduler.  The returned
        arrays are small views/copies for a single protein; the complete weak
        pseudo matrix remains memory-mapped.
        """
        row = int(role_row)
        if row < 0 or row >= self.num_role_rows:
            raise IndexError(f"role-local protein row outside [0,{self.num_role_rows}): {row}")
        start, end = int(self.indptr[row]), int(self.indptr[row + 1])
        go = np.asarray(self.go_idx[start:end], dtype=np.int64)
        if self.probability is None:
            probability = np.ones(go.size, dtype=np.float32)
        else:
            probability = np.asarray(self.probability[start:end], dtype=np.float32)
        return {"go_idx": go, "probability": probability}

    def global_protein_for_role_row(self, role_row: int) -> int:
        row = int(role_row)
        indices = self.registry.role_global_indices[self.role]
        if row < 0 or row >= indices.size:
            raise IndexError(f"role-local protein row outside [0,{indices.size}): {row}")
        return int(indices[row])

    def active_role_rows_for_go_mask(
        self,
        go_mask: np.ndarray,
        *,
        chunk_rows: int = 65536,
    ) -> np.ndarray:
        """Return role-local rows with at least one GO selected by ``go_mask``.

        The computation is row-chunked and never materializes an edge-to-row
        array for the full pseudo graph.  It therefore remains practical for
        the ~10.6M weak pseudo annotations in BP.
        """
        mask = np.asarray(go_mask, dtype=np.bool_)
        if mask.ndim != 1:
            raise ValueError("go_mask must be one-dimensional")
        if self.go_idx.size and int(np.max(self.go_idx)) >= mask.size:
            raise IndexError("go_mask does not cover the Protein-GO CSR GO space")
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        active = np.zeros(self.num_role_rows, dtype=np.bool_)
        for row_start in range(0, self.num_role_rows, int(chunk_rows)):
            row_end = min(self.num_role_rows, row_start + int(chunk_rows))
            edge_start = int(self.indptr[row_start])
            edge_end = int(self.indptr[row_end])
            if edge_end <= edge_start:
                continue
            degrees = np.diff(np.asarray(self.indptr[row_start : row_end + 1], dtype=np.int64))
            nonempty = np.flatnonzero(degrees > 0)
            if nonempty.size == 0:
                continue
            relative_starts = (
                np.asarray(self.indptr[row_start:row_end], dtype=np.int64)[nonempty]
                - edge_start
            )
            selected_edge = mask[np.asarray(self.go_idx[edge_start:edge_end], dtype=np.int64)]
            counts = np.add.reduceat(selected_edge.astype(np.int32), relative_starts)
            active[row_start + nonempty] = counts > 0
        return np.flatnonzero(active).astype(np.int64)

    def gather(
        self,
        protein_idx: Iterable[int],
        *,
        topk: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gather role-local Protein->GO messages.

        When a probability payload exists (the NBS weak pseudo store), ``topk``
        means the highest-probability modelout annotations for each protein,
        not the first ``topk`` CSR entries.  This preserves the intended
        semantics of ``pseudo_message_topk`` as a confidence cap.
        """
        edges: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        for protein in np.unique(np.fromiter((int(x) for x in protein_idx), dtype=np.int64)):
            if self.registry.role_of(protein) != self.role:
                continue
            local = int(self.registry.role_row[protein])
            start, end = int(self.indptr[local]), int(self.indptr[local + 1])
            go_all = np.asarray(self.go_idx[start:end], dtype=np.int64)
            if self.probability is None:
                probability_all = np.ones(go_all.size, dtype=np.float32)
            else:
                probability_all = np.asarray(self.probability[start:end], dtype=np.float32)
                if probability_all.size and (
                    not np.all(np.isfinite(probability_all))
                    or np.any(probability_all < 0.5)
                    or np.any(probability_all > 1.0)
                ):
                    raise ValueError(
                        "pseudo Protein-GO CSR must contain only finite "
                        "modelout probabilities in [0.5, 1.0]; CSR membership records the original >0.5 threshold before float16 quantization"
                    )

            if topk is not None and go_all.size > int(topk):
                k = int(topk)
                if k <= 0:
                    raise ValueError("topk must be positive or None")
                if self.probability is None:
                    selected = np.arange(k, dtype=np.int64)
                else:
                    # O(degree) selection followed by deterministic ordering.
                    selected = np.argpartition(-probability_all, k - 1)[:k]
                    selected = selected[
                        np.lexsort((go_all[selected], -probability_all[selected]))
                    ]
                go = go_all[selected]
                probability = probability_all[selected]
            else:
                go = go_all
                probability = probability_all
                if self.probability is not None and go.size > 1:
                    order = np.lexsort((go, -probability))
                    go = go[order]
                    probability = probability[order]

            if go.size:
                edges.append(np.stack([np.full(go.size, protein, np.int64), go], axis=0))
                probabilities.append(probability.astype(np.float32, copy=False))
        if not edges:
            return np.empty((2, 0), np.int64), np.empty(0, np.float32)
        return np.concatenate(edges, axis=1), np.concatenate(probabilities)


class FullGOBoxStore:
    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "FullGOBoxStore":
        path = Path(manifest_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        arrays = payload["arrays"]
        return cls(
            path.parent / arrays["center"]["file"],
            path.parent / arrays["offset"]["file"],
            stats_path=path.parent / arrays["stats"]["file"],
        )

    def __init__(self, center_path: str | Path, offset_path: str | Path, *, stats_path: str | Path) -> None:
        self.center = np.load(Path(center_path), mmap_mode="r")
        self.offset = np.load(Path(offset_path), mmap_mode="r")
        self.stats = np.load(Path(stats_path), mmap_mode="r")
        if self.center.shape != self.offset.shape or self.stats.shape[0] != self.center.shape[0]:
            raise ValueError("full GO box arrays are misaligned")

    @property
    def num_go(self) -> int:
        return int(self.center.shape[0])

    def gather(self, go_idx: np.ndarray) -> dict[str, np.ndarray]:
        go = np.asarray(go_idx, dtype=np.int64)
        return {
            "center": np.asarray(self.center[go], dtype=np.float32),
            "offset": np.asarray(self.offset[go], dtype=np.float32),
            "stats": np.asarray(self.stats[go], dtype=np.float32),
        }


class DirectGORelationStore:
    def __init__(self, relation_path: str | Path, *, num_go: int) -> None:
        raw = np.load(Path(relation_path), mmap_mode="r")
        if raw.ndim != 2 or raw.shape[1] < 2:
            raise ValueError("GO relation array must be [E,>=2]")
        self.edge = np.asarray(raw[:, :2], dtype=np.int64)
        if self.edge.size and (self.edge.min() < 0 or self.edge.max() >= num_go):
            raise IndexError("GO relation endpoint outside full ontology space")
        order = np.argsort(self.edge[:, 0], kind="stable")
        self.edge = self.edge[order]
        counts = np.bincount(self.edge[:, 0], minlength=num_go)
        self.indptr = np.empty(num_go + 1, dtype=np.int64)
        self.indptr[0] = 0
        np.cumsum(counts, out=self.indptr[1:])

    def sample(
        self,
        sources: Iterable[int],
        fanout: int,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        selected: list[np.ndarray] = []
        for source in np.unique(np.fromiter((int(x) for x in sources), dtype=np.int64)):
            start, end = int(self.indptr[source]), int(self.indptr[source + 1])
            rows = self.edge[start:end]
            if rows.shape[0] > fanout:
                rows = rows[rng.choice(rows.shape[0], size=fanout, replace=False)]
            if rows.size:
                selected.append(rows)
        return np.concatenate(selected, axis=0) if selected else np.empty((0, 2), np.int64)

class GlobalProteinGOCSRStore:
    """Global Protein->GO CSR used for local gold annotation messages."""

    def __init__(
        self,
        indptr_path: str | Path,
        go_index_path: str | Path,
        *,
        num_go: int,
    ) -> None:
        self.indptr = np.load(Path(indptr_path), mmap_mode="r")
        self.go_idx = np.load(Path(go_index_path), mmap_mode="r")
        self.num_go = int(num_go)
        if self.indptr.ndim != 1 or self.go_idx.ndim != 1:
            raise ValueError("global Protein-GO CSR arrays must be one-dimensional")
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != self.go_idx.size:
            raise ValueError("global Protein-GO CSR endpoints are invalid")
        if self.go_idx.size and (
            int(np.min(self.go_idx)) < 0 or int(np.max(self.go_idx)) >= self.num_go
        ):
            raise IndexError("gold GO index outside task label space")

    @property
    def num_proteins(self) -> int:
        return int(self.indptr.size - 1)

    def gather(
        self,
        protein_idx: Iterable[int],
        *,
        topk: Optional[int] = None,
    ) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for protein in np.unique(
            np.fromiter((int(x) for x in protein_idx), dtype=np.int64)
        ):
            if protein < 0 or protein >= self.num_proteins:
                continue
            start, end = int(self.indptr[protein]), int(self.indptr[protein + 1])
            if topk is not None:
                end = min(end, start + int(topk))
            go = np.asarray(self.go_idx[start:end], dtype=np.int64)
            if go.size:
                blocks.append(
                    np.stack([np.full(go.size, protein, np.int64), go], axis=0)
                )
        return (
            np.concatenate(blocks, axis=1)
            if blocks
            else np.empty((2, 0), dtype=np.int64)
        )
