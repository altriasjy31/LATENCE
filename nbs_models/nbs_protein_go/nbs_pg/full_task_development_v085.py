"""Optional, external development set; labels never enter graph/model inputs.

The contract binds supplied artifacts. It cannot establish the completeness of a
user-supplied Stage-1 training-ID list or manufacture unseen annotated proteins.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = "nbs_v085_external_development_1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path):
    path = Path(path).expanduser().resolve()
    return {"path": str(path), "sha256": sha256(path)}


def _ids(path):
    values = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"ID file must be nonempty and unique: {path}")
    return values


def _resolve(path, parent):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(parent) / path).resolve()


def _check(path, expected, description):
    if not expected or sha256(path) != expected:
        raise ValueError(f"{description} SHA256 missing or mismatched: {path}")


def _indices(source, target, description):
    lookup = {value: index for index, value in enumerate(source)}
    if len(lookup) != len(source) or set(source) != set(target):
        raise ValueError(f"{description} IDs must match the prepared input exactly")
    return np.asarray([lookup[value] for value in target], np.int64)


def _array(path, shape, description, *, binary=False):
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{description} shape/nonfinite values: expected {shape}")
    if binary:
        valid = np.all((value == 0) | (value == 1)) and np.any(value == 1)
    else:
        valid = np.all((value >= 0) & (value <= 1))
    if not valid:
        raise ValueError(f"{description} must be {'binary with positives' if binary else 'probabilities in [0,1]'}")
    return value


def build_contract(*, input_dir, labels, label_protein_ids, label_go_ids,
                   reference_dir, stage1_training_ids, final_test_ids):
    """Validate existing inputs and return a JSON-safe, immutable source contract."""
    input_dir, reference_dir = Path(input_dir).resolve(), Path(reference_dir).resolve()
    sources = {
        "input_manifest": input_dir / "ind_test_input_manifest.json",
        "input_protein_ids": input_dir / "protein_ids.txt",
        "labels": Path(labels), "label_protein_ids": Path(label_protein_ids),
        "label_go_ids": Path(label_go_ids), "stage1_training_ids": Path(stage1_training_ids),
        "final_test_ids": Path(final_test_ids),
        "reference_manifest": reference_dir / "stage1_reference_manifest.json",
        "B": input_dir / "backbone_ind_test_prob.f16.npy",
    }
    input_manifest = json.loads(sources["input_manifest"].read_text())
    proteins = _ids(sources["input_protein_ids"])
    _check(sources["input_protein_ids"], input_manifest.get("protein_ids_file_sha256"), "Input protein IDs")
    signature = input_manifest.get("cache_signature", {})
    if input_manifest.get("num_proteins", signature.get("num_proteins")) != len(proteins):
        raise ValueError("Prepared input protein count differs from ID file")
    for name in ("stage1_training_ids", "final_test_ids"):
        overlap = set(proteins).intersection(_ids(sources[name]))
        if overlap:
            raise ValueError(f"Development overlaps {name}: {sorted(overlap)[:8]}")
    registry = input_manifest.get("registries", {})
    sources["go_registry"] = _resolve(registry["go_registry"], input_dir)
    _check(sources["go_registry"], signature.get("go_registry_sha256"), "GO registry")
    with sources["go_registry"].open(newline="") as stream:
        rows = sorted(csv.DictReader(stream, delimiter="\t"), key=lambda row: int(row["go_idx"]))
    if [int(row["go_idx"]) for row in rows] != list(range(len(rows))):
        raise ValueError("GO registry must retain contiguous classifier columns")
    gos = [row["input_go_id"].strip() for row in rows]
    if not gos or len(gos) != len(set(gos)) or signature.get("num_classes") != len(gos):
        raise ValueError("Original GO classifier IDs must be unique and match input shape")
    _indices(_ids(sources["label_protein_ids"]), proteins, "Label protein")
    _indices(_ids(sources["label_go_ids"]), gos, "Label GO")
    shape = (len(proteins), len(gos))
    _array(sources["labels"], shape, "Development labels", binary=True)
    _check(sources["B"], input_manifest.get("base_probability", {}).get("sha256"), "Backbone")
    _array(sources["B"], shape, "Backbone")
    evidence = input_manifest.get("candidate_evidence", {})
    if (evidence.get("selector_scope") != "full_task" or
            evidence.get("expert_probability_used", True) or evidence.get("label_hint_used", True)):
        raise ValueError("Development graph inputs require label-free, expert-free full-task candidates")
    for name, filename, expected in (
        ("representation", "ind_test_repr.f16.npy", input_manifest.get("representation", {}).get("sha256")),
        ("candidate_go", "candidate_go_index.i32.npy", evidence.get("go_index_sha256")),
        ("candidate_attr", "candidate_edge_attr.f32.npy", evidence.get("edge_attr_sha256")),
    ):
        sources[name] = input_dir / filename
        _check(sources[name], expected, name)
    references = json.loads(sources["reference_manifest"].read_text())
    ref_sources = references.get("cache_signature", {}).get("sources", {})
    _check(sources["input_manifest"], ref_sources.get("input_manifest", {}).get("sha256"), "Reference input manifest")
    _check(sources["B"], ref_sources.get("cached_backbone", {}).get("sha256"), "Reference backbone")
    semantics = references.get("semantics", {})
    if semantics.get("labels_consumed_by_model", True) or semantics.get("ind_test_label_boost", True):
        raise ValueError("Stage-1 references must declare label-free inference")
    outputs = references.get("outputs", {})
    for name, key in (("E", "expert_prob"), ("M", "stage1_modelout"),
                      ("reference_protein_ids", "protein_ids"), ("reference_go_ids", "go_ids")):
        if key not in outputs:
            raise ValueError(f"Stage-1 reference manifest missing {key}")
        sources[name] = _resolve(outputs[key]["path"], reference_dir)
        _check(sources[name], outputs[key].get("sha256"), f"Reference {name}")
    if _ids(sources["reference_protein_ids"]) != proteins or _ids(sources["reference_go_ids"]) != gos:
        raise ValueError("Reference E/M row and original GO-column order must equal prepared inputs")
    _array(sources["E"], shape, "Expert E")
    _array(sources["M"], shape, "Modelout M")
    return {"schema": SCHEMA, "input_dir": str(input_dir), "reference_dir": str(reference_dir),
            "shape": list(shape), "sources": {name: _record(path) for name, path in sources.items()},
            "label_policy": "binary_gold; never passed to model", "selection_split": "external_development",
            "stage1_training_ids_policy": "supplied complete list; disjoint IDs, not sequence homology audit"}


def _registry_ids(data):
    path = Path(data.registry.path)
    with path.open(newline="") as stream:
        sample = stream.read(4096); stream.seek(0)
        reader = csv.DictReader(stream, delimiter="\t" if "\t" in sample.splitlines()[0] else ",")
        if "protein_id" not in (reader.fieldnames or []):
            raise ValueError("Training protein registry lacks protein_id; cannot audit development overlap")
        result = [row["protein_id"].strip() for row in reader]
    if len(result) != data.registry.num_proteins or not all(result):
        raise ValueError("Training registry ID rows are incomplete")
    return result


class DevelopmentSetV085:
    """Rank-zero development evaluation with deterministic external graph construction."""
    @classmethod
    def load(cls, manifest_path, data):
        path = Path(manifest_path).resolve()
        declared = json.loads(path.read_text())
        if declared.get("schema") != SCHEMA:
            raise ValueError("Unsupported v085 development contract")
        sources = declared["sources"]
        verified = build_contract(input_dir=declared["input_dir"], reference_dir=declared["reference_dir"],
            **{key: sources[key]["path"] for key in (
                "labels", "label_protein_ids", "label_go_ids", "stage1_training_ids", "final_test_ids")})
        if verified != declared:
            raise ValueError("Development contract/source files changed; create a new experiment contract")
        proteins = _ids(sources["input_protein_ids"]["path"])
        overlap = set(proteins).intersection(_registry_ids(data))
        if overlap:
            raise ValueError(f"Development proteins appear in Stage-2 registry: {sorted(overlap)[:8]}")
        if data.num_task_go != declared["shape"][1]:
            raise ValueError("Development task columns differ from training")
        self = cls()
        self.input_dir = Path(declared["input_dir"])
        self.contract = {**declared, "manifest_sha256": sha256(path),
                         "training_registry_sha256": sha256(data.registry.path)}
        gos = _ids(sources["reference_go_ids"]["path"])
        labels = np.load(sources["labels"]["path"], allow_pickle=False)
        rows = _indices(_ids(sources["label_protein_ids"]["path"]), proteins, "Label protein")
        columns = _indices(_ids(sources["label_go_ids"]["path"]), gos, "Label GO")
        self.labels = labels[np.ix_(rows, columns)].astype(np.uint8)
        self.references = {name: np.load(sources[name]["path"], mmap_mode="r", allow_pickle=False)
                           for name in ("B", "E", "M")}
        self._reference_metrics = None
        return self

    @torch.no_grad()
    def evaluate(self, model, data, device, batch_size, forward_flags=None):
        from .full_task_metrics_v085 import compute_standard_metrics
        if int(batch_size) <= 0:
            raise ValueError("Development batch_size must be positive")
        previous_training = model.training
        cpu_rng = torch.get_rng_state()
        cuda_device = torch.device(device)
        cuda_rng = torch.cuda.get_rng_state(cuda_device) if cuda_device.type == "cuda" else None
        context = (data._sampling_step, data._sampling_rank, data._sampling_training)
        model.eval()
        data.set_sampling_context(0, 0, False)
        try:
            go = model.encode_go()
            predictions = []
            for start in range(0, len(self.labels), int(batch_size)):
                rows = np.arange(start, min(start + int(batch_size), len(self.labels)))
                # inference_batch verifies cache provenance/array hashes and never
                # receives development labels, training roles, or pseudo targets.
                batch = data.inference_batch(self.input_dir, rows, device=device)
                if "positive_mask" in batch or "targets" in batch:
                    raise ValueError("Development inference unexpectedly contains supervision targets")
                output = model(batch, go_encoding=go, **(forward_flags or {}))
                values = output.float().sigmoid().cpu().numpy()
                if values.shape != (len(rows), self.labels.shape[1]) or not np.isfinite(values).all():
                    raise ValueError("Development predictions have invalid shape/nonfinite values")
                predictions.append(values)
            if self._reference_metrics is None:
                self._reference_metrics = {name: compute_standard_metrics(self.labels, values)
                                           for name, values in self.references.items()}
            methods = {**self._reference_metrics,
                       "G": compute_standard_metrics(self.labels, np.concatenate(predictions))}
            keys = ("standard_protein_fmax", "standard_micro_ap", "standard_micro_pr_auc")
            deltas = {f"G_minus_{name}": {key: methods["G"][key] - methods[name][key] for key in keys}
                      for name in ("B", "E", "M")}
            return {"scope": "external_development", "selection_role": "checkpoint_selection",
                    "num_proteins": len(self.labels), "methods": methods, "deltas": deltas,
                    "eligible_for_selection": True,
                    "selection_score": methods["G"]["standard_micro_pr_auc"],
                    "selection_guard": methods["G"]["standard_protein_fmax"] >= methods["B"]["standard_protein_fmax"]}
        finally:
            data.set_sampling_context(*context)
            model.train(previous_training)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, cuda_device)
