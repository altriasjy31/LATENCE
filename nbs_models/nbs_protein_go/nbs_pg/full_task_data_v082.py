"""v0.8.2: dense Stage-1 modelout is supervision, never a graph input.

The old >0.5 CSR is not a complete teacher distribution.  Weak rows therefore
read every task column from the exporter-declared dense modelout mmap.  Core
rows retain gold supervision; their unavailable teacher rows are explicitly
masked instead of being interpreted as an all-negative teacher.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .full_task_data import FullTaskData, _sha256


VERSION = "0.8.2"
_MISSING = (
    "v0.8.2 requires dense Stage-1 modelout for every weak protein. "
    "The configured weak_graph_predictions_manifest must declare the weak "
    "role's modelout_dense_file. Re-export training weak graph predictions with "
    "--save-dense-modelout true using the same Stage-1 checkpoint and registries. "
    "Independent-test modelout and the thresholded pseudo CSR cannot substitute "
    "for this training teacher."
)


def _ids_digest(ids) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class FullTaskDataV082(FullTaskData):
    def __init__(self, config: Mapping[str, Any], stores=None, *, require_teacher=True):
        super().__init__(config, stores=stores)
        # Eligibility no longer depends on whether Stage 1 happened to exceed
        # an arbitrary threshold on at least one output column.
        self.weak_ids = np.asarray(
            self.registry.role_global_indices.get("weak", []), dtype=np.int64
        ).copy()
        self.teacher_probability = None
        self.teacher_path = None
        self._teacher_contract = None
        self._teacher_source = {"available": False, "version": VERSION}
        if require_teacher:
            self._open_teacher()

    def _open_teacher(self):
        data = self.config["data"]
        value = data.get("weak_graph_predictions_manifest")
        if not value:
            raise ValueError(_MISSING)
        path = Path(value)
        if not path.is_absolute():
            path = Path(data["root"]) / path
        path = path.resolve()
        manifest = json.loads(path.read_text())
        roles = [row for row in manifest.get("roles", []) if row.get("role") == "weak"]
        if len(roles) != 1 or not roles[0].get("modelout_dense_file"):
            raise ValueError(_MISSING)
        role = roles[0]
        if int(role.get("rows", -1)) != len(self.weak_ids):
            raise ValueError("dense teacher weak row count differs from the protein registry")
        registry = manifest.get("protein_registry", {})
        registry_digest = _sha256(self.registry.path)
        if registry.get("sha256") != registry_digest:
            raise ValueError("dense teacher protein registry hash differs from training")
        if int(registry.get("num_proteins", -1)) != self.registry.num_proteins:
            raise ValueError("dense teacher protein registry size differs from training")
        with self.registry.path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "protein_id" not in rows[0]:
            raise ValueError("dense teacher alignment requires protein_id in the protein registry")
        ordered = sorted((row for row in rows if row["role"] == "weak"),
                         key=lambda row: int(row["role_row_idx"]))
        protein_ids = [row["protein_id"].strip() for row in ordered]
        if any(not value for value in protein_ids) or len(set(protein_ids)) != len(protein_ids):
            raise ValueError("dense teacher weak protein IDs must be nonempty and unique")
        ids_digest = _ids_digest(protein_ids)
        if role.get("protein_ids_sha256") != ids_digest:
            raise ValueError("dense teacher weak protein order hash differs from role_row_idx order")
        for key, actual in (("global_protein_idx_min", self.weak_ids.min(initial=self.registry.num_proteins)),
                            ("global_protein_idx_max", self.weak_ids.max(initial=-1))):
            if int(role.get(key, -2)) != int(actual):
                raise ValueError(f"dense teacher {key} differs from the weak registry")
        go = manifest.get("go_registry", {})
        if int(go.get("num_terms", -1)) != self.num_task_go:
            raise ValueError("dense teacher GO column count differs from training")
        if not self._go_registry_sha256 or go.get("sha256") != self._go_registry_sha256:
            raise ValueError("dense teacher GO registry hash differs from immutable task columns")
        semantics = manifest.get("model_semantics", {}).get("modelout", {})
        if semantics.get("label_hint_used") is not False:
            raise ValueError("dense training modelout must explicitly declare label_hint_used=false")
        if not str(semantics.get("prediction_key", "")).startswith("modelout::"):
            raise ValueError("dense teacher must declare Stage-1 modelout prediction semantics")
        checkpoint = manifest.get("checkpoint", {})
        digest = checkpoint.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("dense teacher manifest requires the Stage-1 checkpoint SHA256")
        source = Path(role["modelout_dense_file"])
        if not source.is_absolute():
            source = path.parent / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing exporter-declared dense teacher: {source}. {_MISSING}")
        probability = np.load(source, mmap_mode="r", allow_pickle=False)
        expected = (len(self.weak_ids), self.num_task_go)
        if probability.shape != expected or probability.dtype.kind != "f":
            raise ValueError(f"dense teacher shape/dtype {probability.shape}/{probability.dtype} != {expected}/float")
        self.teacher_probability, self.teacher_path = probability, source
        self._teacher_source = {
            "available": True, "version": VERSION, "role": "weak",
            "source": "stage1_dense_modelout_all_task_columns",
            "manifest_path": str(path), "manifest_sha256": _sha256(path),
            "probability_path": str(source), "shape": list(probability.shape),
            "dtype": str(probability.dtype), "protein_registry_sha256": registry_digest,
            "weak_protein_ids_sha256": ids_digest,
            "go_registry_sha256": self._go_registry_sha256,
            "stage1_checkpoint_sha256": digest, "modelout_semantics": semantics,
            "declared_probability_sha256": role.get("modelout_dense_sha256"),
            "forward_use": "none; supervision_only",
        }

    def data_contract(self) -> dict[str, Any]:
        result = super().data_contract()
        if self._teacher_contract is None:
            teacher = dict(self._teacher_source)
            if self.teacher_path is not None:
                digest = _sha256(self.teacher_path)
                declared = teacher.pop("declared_probability_sha256", None)
                if declared is not None and declared != digest:
                    raise ValueError("dense teacher probability file hash differs from its manifest")
                teacher["probability_sha256"] = digest
            self._teacher_contract = teacher
        result["teacher_v082"] = dict(self._teacher_contract)
        result["weak_eligibility"] = "all_weak_registry_rows"
        result["weak_ids_sha256"] = hashlib.sha256(self.weak_ids.tobytes()).hexdigest()
        return result

    def _load_neighbors(self) -> bool:
        if not super()._load_neighbors():
            return False
        # A v0.8.1 cache may have left no-positive weak rows unprepared.  It is
        # reusable only when every newly eligible row already has neighbours.
        if np.any(np.asarray(self._neighbors[self.weak_ids]) < 0):
            self._neighbors = self._neighbor_attrs = None
            return False
        return True

    def batch(self, protein_ids, device="cpu"):
        result = super().batch(protein_ids, device=device)
        ids = np.asarray(protein_ids, dtype=np.int64).reshape(-1)
        weak = self.registry.role_code[ids] == self.registry.role_to_code["weak"]
        probability = np.zeros((len(ids), self.num_task_go), dtype=np.float32)
        available = np.zeros(len(ids), dtype=bool)
        if np.any(weak):
            if self.teacher_probability is None:
                raise RuntimeError("weak training batches require the dense teacher; construct with require_teacher=True")
            rows = self.registry.role_row[ids[weak]]
            selected = np.asarray(self.teacher_probability[rows], dtype=np.float32)
            if not np.all(np.isfinite(selected)) or np.any((selected < 0) | (selected > 1)):
                raise ValueError("dense teacher contains non-finite or out-of-range probabilities")
            # Both files originate from exactly the same exporter pass. This
            # catches a swapped dense file even when its dimensions match.
            positive = result["positive_mask"][result["is_weak"]].detach().cpu().numpy()
            csr_target = result["targets"][result["is_weak"]].detach().cpu().numpy()
            if np.any(np.abs(selected[positive] - csr_target[positive]) > 1e-3):
                raise ValueError("dense teacher disagrees with the same manifest's pseudo CSR probabilities")
            probability[weak] = selected
            available[weak] = True
        result["teacher_prob"] = torch.from_numpy(probability).to(device)
        result["teacher_available"] = torch.from_numpy(available).to(device)
        return result
