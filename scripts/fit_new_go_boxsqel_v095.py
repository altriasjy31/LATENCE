#!/usr/bin/env python3
"""Fit new GO boxes in a frozen BoxSquaredEL space and export a v095 ontology.

Install under scripts/ and run from the LATENCE project root. See --help for
the two axiom sources.
The program consumes ontology information only. Do not pass a list of gold GO.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import gzip
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = {
    "biological_process": "bp",
    "molecular_function": "mf",
    "cellular_component": "cc",
}
GO_ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
CLASS_COLUMNS = {"nf1": (0, 1), "nf2": (0, 1, 2), "nf3": (0, 2),
                 "nf4": (1, 2), "disjoint": (0, 1)}
AXIOM_WIDTH = {"nf1": 2, "nf2": 3, "nf3": 3, "nf4": 3, "disjoint": 2}
ROLE_COLUMN = {"nf3": 1, "nf4": 0}
ROLE_ALIASES = {
    "part_of": ("BFO_0000050", "RO_0000050"),
    "occurs_in": ("BFO_0000066",),
    "regulates": ("RO_0002211",),
    "negatively_regulates": ("RO_0002212",),
    "positively_regulates": ("RO_0002213",),
    "has_part": ("BFO_0000051",),
}


@dataclass
class OBOTerm:
    go: str
    name: str = ""
    namespace: str = ""
    obsolete: bool = False
    aliases: set[str] = field(default_factory=set)
    replacements: set[str] = field(default_factory=set)
    parents: set[str] = field(default_factory=set)
    relationships: set[tuple[str, str]] = field(default_factory=set)


def read_obo(path):
    """Read GO Term stanzas, retaining aliases and explicit typed edges."""
    opener = gzip.open if path.suffix == ".gz" else open
    version = None
    terms = {}
    record = None

    def save():
        if record is not None and record.go:
            if record.go in terms:
                raise ValueError(f"{path}: duplicate GO term {record.go}")
            terms[record.go] = record

    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith("["):
                save()
                record = OBOTerm("") if line == "[Term]" else None
                continue
            key, sep, value = line.partition(": ")
            if not sep or not value:
                continue
            if key == "data-version" and record is None:
                version = value
            if record is None:
                continue
            tokens = value.split()
            if key == "name":
                record.name = value
            elif key == "id" and re.fullmatch(r"GO:\d{7}", tokens[0]):
                record.go = tokens[0]
            elif key == "namespace":
                record.namespace = NAMESPACES.get(tokens[0], "")
            elif key == "is_obsolete":
                record.obsolete = tokens[0] == "true"
            elif key == "alt_id":
                record.aliases.add(tokens[0])
            elif key in {"replaced_by", "consider"}:
                record.replacements.add(tokens[0])
            elif key == "is_a":
                record.parents.add(tokens[0])
            elif key == "relationship" and len(tokens) >= 2:
                record.relationships.add(tuple(tokens[:2]))
    save()
    if not terms or not version:
        raise ValueError(f"{path}: expected GO [Term] stanzas and a data-version")
    return version, terms


def identity_owners(terms):
    owners = defaultdict(set)
    for term in terms.values():
        for identity in {term.go} | term.aliases:
            owners[identity].add(term.go)
    return owners


def canonical_active(identity, terms, owners):
    possible = owners.get(identity, ())
    if len(possible) == 1:
        go = next(iter(possible))
        if not terms[go].obsolete:
            return go
    return None


def select_candidates(old_terms, new_terms, new_owners, registry, namespaces):
    old_owners = identity_owners(old_terms)
    old_identities = set(old_owners) | set(registry._canonical) | set(registry._row)
    migrations = {
        canonical_active(ref, new_terms, new_owners)
        for term in (*old_terms.values(), *new_terms.values())
        if ({term.go} | term.aliases) & old_identities
        for ref in term.replacements
    }
    audit = {}
    for go, term in sorted(new_terms.items()):
        if term.obsolete or term.namespace not in namespaces or go in old_terms:
            continue
        identities = {go} | term.aliases
        if identities & old_identities:
            status = "old_identity_overlap"
        elif go in migrations:
            status = "replacement_or_consider"
        elif any(len(new_owners[identity]) != 1 for identity in identities):
            status = "ambiguous_alias"
        else:
            status = "candidate"
        audit[go] = {"go_id": go, "namespace": term.namespace, "name": term.name,
                     "status": status, "ontology_row": None, "anchor_hops": None,
                     "anchor_go_ids": []}
    return audit


def read_trainer(path):
    spec = importlib.util.spec_from_file_location("_latence_original_boxsqel_trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original BoxSquaredEL trainer {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for symbol in ("BoxSquaredELModel", "parse_axiom_line", "strip_inline_comment", "shorten_identifier"):
        if not hasattr(module, symbol):
            raise ImportError(f"{path}: original trainer must expose {symbol}")
    return module


def role_key(raw, trainer):
    return trainer.shorten_identifier(raw).replace(":", "_")


def role_lookup(checkpoint_relations, trainer):
    lookup = {}
    for raw, row in checkpoint_relations.items():
        key = role_key(raw, trainer)
        if key in lookup and lookup[key] != row:
            raise ValueError(f"Ambiguous checkpoint relation {key}")
        lookup[key] = int(row)
    for name, synonyms in ROLE_ALIASES.items():
        rows = {lookup[key] for key in synonyms if key in lookup}
        if name not in lookup and len(rows) == 1:
            lookup[name] = rows.pop()
        if name in lookup:
            for key in synonyms:
                lookup.setdefault(key, lookup[name])
    return lookup


def read_checkpoint(old, path, trainer, training_manifest):
    if path is None:
        raise FileNotFoundError("Original BoxSquaredEL checkpoint is missing; pass --box-checkpoint")
    actual_hash = digest_file(path)
    if actual_hash != old.box_checkpoint_sha256:
        raise ValueError("BoxSquaredEL checkpoint SHA256 differs from the v095-bound full geometry")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = payload["model_state_dict"]
    class_map = payload["classes"]
    roles = payload["relations"]
    source_config = payload["config"]
    if training_manifest is not None:
        expected = json.loads(training_manifest.read_text(encoding="utf-8"))["config"]
        for key in ("embedding_size", "margin", "reg_factor", "learning_rate", "loss_weights"):
            if key in expected and expected[key] != source_config.get(key):
                raise ValueError(f"Training manifest and bound checkpoint disagree on {key}")

    def tensor(key):
        return state[key].detach().cpu().numpy()

    n, dim = old.center.shape
    raw = tensor("class_embeds.weight")
    if (raw.shape != (n, 2 * dim) or set(class_map.values()) != set(range(n))
            or int(source_config.get("embedding_size", dim)) != dim):
        raise ValueError("Checkpoint class space differs from the v095-bound arrays")
    if not np.array_equal(raw[:, :dim], old.center):
        raise ValueError("Frozen centers differ from the original BoxSquaredEL checkpoint")
    if not np.array_equal(np.maximum(np.abs(raw[:, dim:]), 1e-8), old.offset):
        raise ValueError("Frozen offsets differ from the original BoxSquaredEL checkpoint")
    for record in old.registry_records:
        if record.get("class_id"):
            row = int(record["ontology_go_idx"])
            name = record["class_id"]
            if class_map.get(name) != row:
                raise ValueError(f"Original class row disagrees with registry for {name}")

    bumps = tensor("bumps.weight")
    heads = tensor("relation_heads.weight")
    tails = tensor("relation_tails.weight")
    if (bumps.shape != (n, dim) or heads.shape != tails.shape
            or heads.shape != (max(1, len(roles)), 2 * dim)
            or set(roles.values()) != set(range(len(roles)))):
        raise ValueError("Inconsistent bump or role parameter shapes in BoxSquaredEL checkpoint")
    return source_config, class_map, role_lookup(roles, trainer), bumps, heads, tails, actual_hash


def known_go_rows(registry, new_terms, owners):
    """Use the historical registry's resolution without rewriting it."""
    resolved = {}

    def old_row(identity):
        target = registry._canonical.get(identity)
        if target in registry._row and not registry._obsolete[target][0]:
            return int(registry._row[target])
        return None

    for go, term in new_terms.items():
        if term.obsolete:
            continue
        row = old_row(go)
        if row is None:
            options = {old_row(alias) for alias in term.aliases if owners[alias] == {go}}
            options.discard(None)
            if len(options) == 1:
                row = options.pop()
        if row is not None:
            resolved[go] = row
    return resolved


def collect_axioms(new_terms, owners, go_rows, trial_rows, class_map, roles, trainer, normalized):
    """Build direct G-G edges and geometry constraints in the trial row space."""
    axioms = {name: set() for name in AXIOM_WIDTH}
    edges = {"is_a": set(), "part_of": set()}
    counters = Counter()
    new_rows = set(trial_rows.values())
    part_keys = {"part_of", *ROLE_ALIASES["part_of"]}
    part_rows = {roles[key] for key in part_keys if key in roles}

    def row_for_go(raw):
        return go_rows.get(canonical_active(raw, new_terms, owners))

    def add(form, rows):
        if form == "nf1" and rows[0] == rows[1]:
            return
        if form in {"nf2", "disjoint"}:
            rows = (*sorted(rows[:2]), *rows[2:])
        axioms[form].add(rows)
        if form == "nf1":
            edges["is_a"].add(rows)
        elif form == "nf3" and rows[1] in part_rows:
            edges["part_of"].add((rows[0], rows[2]))

    for term in new_terms.values():
        source = go_rows.get(term.go)
        if source is None:
            continue
        for target_id in term.parents:
            target = row_for_go(target_id)
            if target is not None and (source in new_rows or target in new_rows):
                add("nf1", (source, target))
        for role, target_id in term.relationships:
            target = row_for_go(target_id)
            if target is None or (source not in new_rows and target not in new_rows):
                continue
            key = role_key(role, trainer)
            relation = roles.get(key)
            if relation is None:
                counters[f"obo_unknown_role:{key}"] += 1
                continue
            add("nf3", (source, relation, target))

    if normalized is None:
        return axioms, edges, dict(counters)
    stable_classes = {
        trainer.shorten_identifier(name).replace(":", "_"): int(row)
        for name, row in class_map.items()
        if re.fullmatch(r"[A-Z][A-Z0-9]*[_:]\d+|owl:Thing",
                        trainer.shorten_identifier(name))
        and not re.fullmatch(r"GO[_:]\d{7}", trainer.shorten_identifier(name))
    }

    def class_row(raw):
        key = trainer.shorten_identifier(raw).replace(":", "_")
        if re.fullmatch(r"GO_\d{7}", key):
            return row_for_go(key.replace("_", ":"))
        return stable_classes.get(key)

    opener = gzip.open if normalized.suffix == ".gz" else open
    with opener(normalized, "rt", encoding="utf-8-sig") as stream:
        for number, raw in enumerate(stream, 1):
            line = trainer.strip_inline_comment(raw.strip())
            if not line or line.startswith(("#", "//")):
                continue
            parsed = trainer.parse_axiom_line(line)
            if parsed is None:
                raise ValueError(f"{normalized}:{number}: not a supported normalized axiom: {line[:140]}")
            for form, values in parsed:
                if form not in CLASS_COLUMNS:
                    counters[f"normalized_ignored_{form}"] += 1
                    continue
                columns = CLASS_COLUMNS[form]
                resolved = {col: class_row(values[col]) for col in columns}
                if not any(row in new_rows for row in resolved.values()):
                    continue
                if any(row is None for row in resolved.values()):
                    counters[f"normalized_unresolved_class_{form}"] += 1
                    continue
                if form in ROLE_COLUMN:
                    col = ROLE_COLUMN[form]
                    role = roles.get(role_key(values[col], trainer))
                    if role is None:
                        counters[f"normalized_unknown_role_{form}"] += 1
                        continue
                    resolved[col] = role
                add(form, tuple(resolved[i] for i in range(len(values))))
                counters[f"normalized_accepted_{form}"] += 1
    return axioms, edges, dict(counters)


def anchored_rows(known, trial, audit, new_terms, edges):
    """Follow same-namespace is_a/part_of paths to nearest non-root old rows."""
    old_by_row = defaultdict(set)
    node_ns = defaultdict(set)
    for go, row in known.items():
        if go not in GO_ROOTS:
            old_by_row[row].add(go)
        node_ns[row].add(new_terms[go].namespace)
    for go, row in trial.items():
        node_ns[row].add(new_terms[go].namespace)
    parents = defaultdict(set)
    for child, parent in edges["is_a"] | edges["part_of"]:
        if node_ns[child] & node_ns[parent] and child != parent:
            parents[child].add(parent)
    keep = []
    anchors = []
    for go, row in trial.items():
        aspect = new_terms[go].namespace
        visited = {row}
        frontier = {row}
        depth = 0
        while frontier:
            found = {
                node for node in frontier
                if any(new_terms[old_go].namespace == aspect for old_go in old_by_row.get(node, ()))
            }
            if found:
                audit[go].update(
                    status="fitted", anchor_hops=depth,
                    anchor_go_ids=sorted({g for x in found for g in old_by_row[x]
                                          if new_terms[g].namespace == aspect}),
                )
                keep.append(go)
                anchors.append(sorted(found))
                break
            frontier = {parent for node in frontier for parent in parents.get(node, ())
                        if parent not in visited and aspect in node_ns[parent]}
            visited.update(frontier)
            depth += 1
        else:
            audit[go]["status"] = "no_nonroot_old_anchor"
    return keep, anchors


def restrict_rows(axioms, edges, trial_rows, keep, num_old):
    new_index = {trial_rows[go]: num_old + i for i, go in enumerate(keep)}
    excluded = set(trial_rows.values()) - new_index.keys()
    remapped = {}
    for name, rows in axioms.items():
        columns = CLASS_COLUMNS[name]
        remapped[name] = np.asarray([
            tuple(new_index.get(row, row) if col in columns else row
                  for col, row in enumerate(values))
            for values in sorted(rows) if all(values[col] not in excluded for col in columns)
        ], dtype=np.int64).reshape(-1, AXIOM_WIDTH[name])
    remapped_edges = {
        name: sorted({(new_index.get(a, a), new_index.get(b, b))
                      for a, b in rows if a != b and a not in excluded and b not in excluded})
        for name, rows in edges.items()
    }
    return remapped, remapped_edges


def digest_file(path):
    with Path(path).open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(destination, manifest, audit, go_ids, bundle, center, offset, bump):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as tmp:
        stage = Path(tmp) / "output"
        stage.mkdir()
        torch.save(bundle, stage / "ontology_bundle.pt")
        np.savez(stage / "new_go_embeddings.npz", go_ids=np.asarray(go_ids),
                 center=center, offset=offset, bump=bump)
        (stage / "new_go_ids.txt").write_text("".join(go + "\n" for go in go_ids), encoding="utf-8")
        fields = ["go_id", "namespace", "name", "status", "ontology_row", "anchor_hops", "anchor_go_ids"]
        with (stage / "candidate_go.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for record in audit.values():
                writer.writerow({**record, "anchor_go_ids": ";".join(record["anchor_go_ids"])})
        manifest["artifacts"] = {
            path.name: {"sha256": digest_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(stage.iterdir())
        }
        (stage / "extension_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Output directory already exists: {destination}")
        os.rename(stage, destination)


# Read and preserve the exact old v095 ontology resources.
NBS_RELATION_NAMES = ("is_a", "has_child", "part_of", "has_part")


@dataclass
class V095Ontology:
    project_root: Path
    task: str
    center: np.ndarray
    offset: np.ndarray
    edges: dict
    task_to_ontology: np.ndarray
    registry_records: list
    registry: object
    box_checkpoint: Path | None
    box_checkpoint_sha256: str
    paths: dict
    hashes: dict


def _ontology_path(base, value):
    path = Path(value).expanduser()
    return (Path(base) / path).resolve()


def _ontology_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ontology_array(manifest_path, specification):
    path = _ontology_path(manifest_path.parent, specification["file"])
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if "shape" in specification and data.shape != tuple(specification["shape"]):
        raise ValueError(f"Array shape differs from {manifest_path}: {path}")
    if "dtype" in specification and data.dtype != np.dtype(specification["dtype"]):
        raise ValueError(f"Array dtype differs from {manifest_path}: {path}")
    return path, data


def _ontology_source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import project ontology contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_v095_ontology(config_path, project_root):
    """Resolve the same GO arrays and task registry as the v095 worker.

    The v095 loader bootstraps ontology resources from the v094 dataset's
    snapshot of the original configuration. Reading only those resources
    avoids loading protein features, supervision, or a trained NBS model.
    """
    root = Path(project_root).expanduser().resolve(strict=True)
    config_path = Path(config_path).expanduser().resolve(strict=True)
    config = _ontology_json(config_path)
    if config.get("release_version") != "0.9.5":
        raise ValueError(f"Expected a v095 resolved_config.json: {config_path}")

    dataset_path = _ontology_path(root, config["data"]["unified_dataset_manifest"])
    dataset_hash = digest_file(dataset_path)
    recorded_hash = config.get("v095_experiment", {}).get("unified_dataset_manifest_sha256")
    if recorded_hash != dataset_hash:
        raise ValueError(f"v095 unified dataset manifest hash differs: {dataset_path}")
    dataset = _ontology_json(dataset_path)
    if dataset.get("schema") != "nbs-weak-dataset-v094-r1" or dataset.get("complete") is not True:
        raise ValueError(f"Expected a complete v094 unified dataset: {dataset_path}")
    source = dataset["source_config"]
    snapshot_path = _ontology_path(dataset_path.parent,
                                  dataset.get("files", {}).get("source_config_snapshot", source["path"]))
    snapshot_hash = digest_file(snapshot_path)
    if snapshot_hash != source["sha256"]:
        raise ValueError(f"Original bootstrap configuration changed: {snapshot_path}")
    bootstrap = _ontology_json(snapshot_path)
    task = config["task"]
    if bootstrap.get("task") != task or dataset.get("task", task) != task:
        raise ValueError("v095 task and bootstrap ontology task differ")

    paths = {"v095_config": config_path, "dataset_manifest": dataset_path,
             "bootstrap_config": snapshot_path}
    hashes = {"v095_config": digest_file(config_path), "dataset_manifest": dataset_hash,
              "bootstrap_config": snapshot_hash}

    def record_file(name, path):
        digest = digest_file(path)
        identity = dataset.get("sources", {}).get(name)
        if identity is not None and digest != identity.get("sha256"):
            raise ValueError(f"v094-bound ontology source differs for {name}: {path}")
        paths[name] = path
        hashes[name] = digest
        return digest

    data = bootstrap["data"]
    data_root = _ontology_path(root, data["root"])
    weak_path = _ontology_path(data_root, data["weak_graph_predictions_manifest"])
    record_file("weak_manifest", weak_path)
    weak = _ontology_json(weak_path)
    task_path = (data_root / "gg_relations/go_registry.tsv").resolve()
    if not task_path.is_file():
        task_path = _ontology_path(weak_path.parent, weak["go_registry"]["path"])
    task_hash = record_file("task_registry", task_path)
    expected_task_hash = weak["go_registry"].get("sha256")
    if expected_task_hash is not None and expected_task_hash != task_hash:
        raise ValueError(f"Task registry hash differs from the weak source: {task_path}")

    full_path = _ontology_path(root, data["full_go_box_manifest"])
    alignment_path = _ontology_path(root, bootstrap["go_boxsqel"]["alignment_manifest"])
    gg_path = _ontology_path(root, data["boxsqel_gg_relations_manifest"])
    record_file("full_ontology_manifest", full_path)
    record_file("alignment_manifest", alignment_path)
    record_file("gg_manifest", gg_path)
    full, alignment, gg = (_ontology_json(path) for path in (full_path, alignment_path, gg_path))
    registry_path = _ontology_path(full_path.parent, full["registry"]["file"])
    record_file("ontology_registry", registry_path)
    mapping_path, mapping = _ontology_array(alignment_path, alignment["arrays"]["source_row"])
    record_file("alignment_mapping", mapping_path)

    registry_source = root / "nbs_models/nbs_protein_go/nbs_pg/go_registry_v093.py"
    registry_type = _ontology_source_module(registry_source, "_anchored_go_registry").GORegistryV093
    registry = registry_type.from_manifests(full_path, task_path, alignment_path)
    if registry.contract() != dataset.get("GO_registry_contract"):
        raise ValueError("Installed GO registry policy does not match the v094 dataset contract")
    if (mapping.shape != (int(dataset["num_task_go"]),)
            or not np.array_equal(registry.task_to_ontology, mapping)):
        raise ValueError("Aligned ontology mapping differs from the immutable task columns")
    with registry_path.open(encoding="utf-8", newline="") as stream:
        registry_records = list(csv.DictReader(stream, delimiter="\t"))

    boxes = {}
    for name in ("center", "offset", "stats"):
        path, array = _ontology_array(full_path, full["arrays"][name])
        record_file("old_" + name, path)
        boxes[name] = np.asarray(array, dtype=np.float32)
        if not np.isfinite(boxes[name]).all():
            raise ValueError(f"The v095 {name} array contains nonfinite values: {path}")
    n = registry.num_ontology_rows
    if (boxes["center"].shape != (n, int(full["embedding_dim"]))
            or boxes["offset"].shape != boxes["center"].shape
            or boxes["stats"].shape[0] != n or int(gg["num_classes"]) != n):
        raise ValueError("The v095 ontology geometry, registry and G-G graph row counts differ")

    edges = {}
    for name in NBS_RELATION_NAMES:
        path, array = _ontology_array(gg_path, gg["relations"][name])
        record_file("old_edge_" + name, path)
        if array.ndim != 2 or array.shape[1] < 2 or array.dtype.kind not in "iu":
            raise ValueError(f"G-G relation {name} requires integer [E,>=2] edges: {path}")
        columns = np.asarray(array[:, :2], dtype=np.int64)
        if columns.size and (columns.min() < 0 or columns.max() >= n):
            raise ValueError(f"G-G relation {name} has an endpoint outside the old ontology")
        # DirectGORelationStore applies exactly this stable source sort.
        edges[name] = columns[np.argsort(columns[:, 0], kind="stable")].T.copy()

    box_source = full["source"]
    checkpoint_hash = box_source["checkpoint_sha256"]
    gg_checkpoint_hash = gg.get("source", {}).get("boxsqel_checkpoint_sha256")
    if not checkpoint_hash or (gg_checkpoint_hash and checkpoint_hash != gg_checkpoint_hash):
        raise ValueError("Old BoxSquaredEL geometry and G-G graph use different checkpoints")
    checkpoint = _ontology_path(full_path.parent, box_source["checkpoint"])
    if not checkpoint.is_file():
        checkpoint = _ontology_path(root, box_source["checkpoint"])
    paths["box_checkpoint_recorded"] = checkpoint
    return V095Ontology(
        project_root=root, task=task, center=boxes["center"], offset=boxes["offset"],
        edges=edges, task_to_ontology=registry.task_to_ontology,
        registry_records=registry_records, registry=registry,
        box_checkpoint=checkpoint if checkpoint.is_file() else None,
        box_checkpoint_sha256=checkpoint_hash, paths=paths, hashes=hashes,
    )


def make_ontology_bundle(old, new_ids, new_center, new_offset, new_edges, *, new_aliases=None):
    """Append new geometry and direct relations without changing old rows.

    `new_edges` holds child/parent row pairs in the extended ontology for
    is_a and part_of. Reverse edges are generated using v095 relation names.
    """
    old_n = len(old.center)
    new_ids = list(new_ids)
    expected_shape = (len(new_ids), old.center.shape[1])
    new_center = np.asarray(new_center, dtype=np.float32)
    new_offset = np.asarray(new_offset, dtype=np.float32)
    if not new_ids or len(set(new_ids)) != len(new_ids):
        raise ValueError("The ontology extension requires nonempty unique new GO identities")
    if (new_center.shape != expected_shape or new_offset.shape != expected_shape
            or not np.isfinite(new_center).all() or not np.isfinite(new_offset).all()
            or (new_offset <= 0).any()):
        raise ValueError("New geometry must be finite with positive offsets and the old embedding width")

    aliases = new_aliases or {}
    if set(aliases).difference(new_ids):
        raise ValueError("Alias targets must be new canonical GO identities")
    records = [dict(record) for record in old.registry_records]
    for row, go_id in enumerate(new_ids, old_n):
        if go_id in old.registry._canonical:
            raise ValueError(f"Extension GO is already known to the old registry: {go_id}")
        records.append({"go_id": go_id, "ontology_row": row,
                        "alt_ids": sorted(set(aliases.get(go_id, ())))})

    edges = dict(old.edges)
    if set(new_edges).difference({"is_a", "part_of"}):
        raise ValueError("New G-G edges may only use is_a and part_of")
    for name, inverse in (("is_a", "has_child"), ("part_of", "has_part")):
        pairs = np.asarray(list(new_edges.get(name, ())))
        if not pairs.size:
            continue
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
            raise ValueError(f"New {name} pairs must contain integer child/parent row indices")
        if (pairs.min() < 0 or pairs.max() >= old_n + len(new_ids)
                or (pairs.max(axis=1) < old_n).any() or (pairs[:, 0] == pairs[:, 1]).any()):
            raise ValueError(f"New {name} pairs require valid, distinct endpoints and a new ontology row")
        direct = np.unique(pairs.astype(np.int64, copy=False), axis=0).T
        edges[name] = np.concatenate((edges[name], direct), axis=1)
        edges[inverse] = np.concatenate((edges[inverse], direct[::-1]), axis=1)

    center = np.concatenate((old.center, new_center), axis=0)
    offset = np.concatenate((old.offset, new_offset), axis=0)
    old_rows = np.arange(old_n, dtype=np.int64)
    payload = {
        "ontology": {"center": center, "offset": offset, "edges": edges,
                     "task_to_ontology": old.task_to_ontology.copy()},
        "old_ontology_rows": old_rows,
        "registry_records": records,
    }
    predictor_path = old.project_root / "scripts/nbs/predict_nbs_go_v095.py"
    predictor = _ontology_source_module(predictor_path, "_anchored_go_predictor_v095")
    updated = predictor.updated_registry(old.registry, payload, old_rows, len(center), type(old.registry))
    expected = np.arange(old_n, len(center), dtype=np.int64)
    if not np.array_equal(updated.resolve_target_go(new_ids), expected):
        raise ValueError("Extended registry does not resolve the new GO IDs in append order")
    return payload


# Optimize only newly appended BoxSquaredEL rows.


class _FrozenOldRows(nn.Module):
    """Look up old and new rows while exposing only the latter to Adam."""

    def __init__(self, old_rows, new_rows):
        super().__init__()
        self.register_buffer("old_rows", old_rows)
        self.new_rows = nn.Parameter(new_rows)

    def forward(self, row_ids):
        old_count = len(self.old_rows)
        old = F.embedding(row_ids.clamp_max(old_count - 1), self.old_rows)
        new = F.embedding((row_ids - old_count).clamp_min(0), self.new_rows)
        return torch.where((row_ids >= old_count).unsqueeze(-1), new, old)


def _localize_axioms(axioms, anchors, old_count, relation_count, loss_weights):
    used_old = {row for group in anchors for row in group}
    involved = set()
    selected = {}
    for form, class_columns in CLASS_COLUMNS.items():
        rows = np.asarray(axioms.get(form, []))
        if rows.size == 0:
            rows = np.empty((0, AXIOM_WIDTH[form]), dtype=np.int64)
        if rows.ndim != 2 or rows.shape[1] != AXIOM_WIDTH[form] or rows.dtype.kind not in "iu":
            raise ValueError(f"{form} requires integer axioms of width {AXIOM_WIDTH[form]}")
        rows = rows.astype(np.int64, copy=True)
        class_ids = rows[:, class_columns]
        if np.any(class_ids < 0) or np.any(class_ids >= old_count + len(anchors)):
            raise ValueError(f"{form} contains an out-of-range class row")
        if form in ROLE_COLUMN:
            relation_ids = rows[:, ROLE_COLUMN[form]]
            if np.any(relation_ids < 0) or np.any(relation_ids >= relation_count):
                raise ValueError(f"{form} contains an out-of-range relation row")
        if np.any(np.all(class_ids < old_count, axis=1)):
            raise ValueError(f"{form} contains an axiom with only frozen classes")
        used_old.update(class_ids[class_ids < old_count].tolist())
        if loss_weights[form] > 0:
            involved.update(class_ids[class_ids >= old_count].tolist())
        selected[form] = rows

    missing = sorted(set(range(old_count, old_count + len(anchors))) - involved)
    if missing:
        raise ValueError(f"New rows have no positively weighted ontology constraint: {missing[:20]}")

    old_rows = np.asarray(sorted(used_old), dtype=np.int64)
    localized = {}
    for form, rows in selected.items():
        compact = rows.copy()
        class_columns = CLASS_COLUMNS[form]
        class_ids = rows[:, class_columns]
        compact[:, class_columns] = np.where(
            class_ids < old_count,
            np.searchsorted(old_rows, class_ids),
            class_ids - old_count + len(old_rows),
        )
        localized[form] = compact
    return old_rows, selected, localized


def _initial_new_geometry(old_center, old_offset, old_bumps, anchors, min_offset):
    centers, offsets, bumps = [], [], []
    for group in anchors:
        anchor_centers = old_center[group]
        anchor_offsets = old_offset[group]
        lower = (anchor_centers - anchor_offsets).max(axis=0)
        upper = (anchor_centers + anchor_offsets).min(axis=0)
        overlapping = lower <= upper
        centers.append(np.where(overlapping, (lower + upper) / 2, anchor_centers.mean(axis=0)))
        half_width = np.where(overlapping, (upper - lower) / 2, anchor_offsets.mean(axis=0))
        offsets.append(np.maximum(0.5 * half_width, min_offset))
        bumps.append(old_bumps[group].mean(axis=0))
    return np.stack(centers), np.stack(offsets), np.stack(bumps)


def _residual_summary(values):
    if not len(values):
        return {"mean": None, "max": None}
    return {"mean": float(np.mean(values)), "max": float(np.max(values))}


def fit_new_boxes(
    trainer,
    *,
    old_center,
    old_offset,
    old_bumps,
    relation_heads,
    relation_tails,
    axioms,
    anchors,
    config,
    steps=1000,
    learning_rate=None,
    device="cpu",
    anchor_weight=0.01,
    min_offset=1e-8,
):
    """Fit new BoxSquaredEL parameters using the original trainer's positive losses.

    Axiom class IDs index old rows followed by new rows. Relation IDs index the
    checkpoint's relation tensors in original order. Only old rows referenced by
    the local constraints are moved to the requested device.
    """
    old_center, old_offset, old_bumps, relation_heads, relation_tails = (
        np.asarray(array, dtype=np.float32)
        for array in (old_center, old_offset, old_bumps, relation_heads, relation_tails)
    )
    if old_center.ndim != 2 or not all(old_center.shape):
        raise ValueError("Old class centers must be a nonempty [classes, dimension] array")
    old_count, dimension = old_center.shape
    if old_offset.shape != old_center.shape or old_bumps.shape != old_center.shape:
        raise ValueError("Old class centers, offsets and bumps must have identical shapes")
    if relation_heads.ndim != 2 or relation_heads.shape[1] != 2 * dimension or relation_tails.shape != relation_heads.shape:
        raise ValueError("Frozen relation heads and tails must be [relations, 2 * dimension]")
    anchors = [np.asarray(sorted(set(group)), dtype=np.int64) for group in anchors]
    if not anchors or any(not len(group) or group[0] < 0 or group[-1] >= old_count for group in anchors):
        raise ValueError("Every new class needs at least one valid frozen anchor row")
    if steps < 0 or not isinstance(steps, int):
        raise ValueError("steps must be a nonnegative integer")
    learning_rate = float(config.get("learning_rate", 0.001) if learning_rate is None else learning_rate)
    margin = float(config.get("margin", 0.0))
    reg_factor = float(config.get("reg_factor", 0.05))
    clip_norm = float(config.get("clip_norm", 1.0))
    loss_weights = {form: float(config.get("loss_weights", {}).get(form, 1.0)) for form in CLASS_COLUMNS}
    scalars = [learning_rate, margin, reg_factor, clip_norm, anchor_weight, min_offset, *loss_weights.values()]
    if not all(math.isfinite(value) and value >= 0 for value in scalars) or not learning_rate or not min_offset:
        raise ValueError("Geometry fit weights must be finite and nonnegative, with positive learning rate and offset floor")

    old_rows, selected_axioms, local_axioms = _localize_axioms(
        axioms, anchors, old_count, len(relation_heads), loss_weights,
    )
    frozen_center = old_center[old_rows]
    frozen_offset = old_offset[old_rows]
    frozen_bumps = old_bumps[old_rows]
    if not all(np.isfinite(array).all() for array in (frozen_center, frozen_offset, frozen_bumps, relation_heads, relation_tails)):
        raise ValueError("Referenced checkpoint geometry contains nonfinite values")
    if np.any(frozen_offset < 0):
        raise ValueError("Old class offsets must be absolute widths")
    initial_center, initial_offset, initial_bumps = _initial_new_geometry(
        old_center, old_offset, old_bumps, anchors, min_offset,
    )

    def on_device(array):
        return torch.as_tensor(array, device=device)

    initial_boxes = on_device(np.concatenate((initial_center, initial_offset), axis=1))
    initial_bumps = on_device(initial_bumps)

    class AnchoredBoxSquaredEL(trainer.BoxSquaredELModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.embedding_dim = dimension
            self.margin = margin
            self.class_embeds = _FrozenOldRows(
                on_device(np.concatenate((frozen_center, frozen_offset), axis=1)), initial_boxes.clone(),
            )
            self.bumps = _FrozenOldRows(on_device(frozen_bumps), initial_bumps.clone())
            self.relation_heads = nn.Embedding.from_pretrained(on_device(relation_heads))
            self.relation_tails = nn.Embedding.from_pretrained(on_device(relation_tails))

        def zero(self):
            return self.class_embeds.new_rows.new_zeros(())

    model = AnchoredBoxSquaredEL()
    parameters = (model.class_embeds.new_rows, model.bumps.new_rows)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    tensors = {form: on_device(rows) for form, rows in local_axioms.items() if len(rows)}

    def evaluate():
        distances = {form: getattr(model, f"{form}_loss")(rows) for form, rows in tensors.items()}
        losses = {form: model.squared_mean(values) for form, values in distances.items()}
        losses["bump_reg"] = torch.linalg.vector_norm(parameters[1], dim=1).mean()
        box_shift = torch.cat((parameters[0][:, :dimension], parameters[0][:, dimension:].abs()), dim=1) - initial_boxes
        losses["anchor_tether"] = box_shift.square().sum(dim=1).mean()
        total = sum(loss_weights[form] * losses[form] for form in tensors)
        total = total + reg_factor * losses["bump_reg"] + anchor_weight * losses["anchor_tether"]
        return total, losses, distances

    def snapshot(losses, distances):
        return ({name: value.item() for name, value in losses.items()},
                {name: value.detach().cpu().numpy().ravel() for name, value in distances.items()})

    with torch.no_grad():
        initial_objective, losses, distances = evaluate()
        initial_losses, initial_distances = snapshot(losses, distances)
        first_score = initial_objective.item()
    best_score = math.inf
    best_step = 0
    best_parameters = None
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = evaluate()
        value = total.item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite anchored objective at step {step}")
        if value < best_score:
            best_score, best_step = value, step
            best_parameters = [param.detach().clone() for param in parameters]
        if step == steps:
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip_norm or math.inf, error_if_nonfinite=True)
        optimizer.step()

    with torch.no_grad():
        for parameter, best in zip(parameters, best_parameters):
            parameter.copy_(best)
        parameters[0][:, dimension:].copy_(parameters[0][:, dimension:].abs().clamp_min(min_offset))
        final_objective, losses, distances = evaluate()
        final_losses, final_distances = snapshot(losses, distances)
        fitted = parameters[0].cpu().numpy().copy()
        fitted_bumps = parameters[1].cpu().numpy().copy()

    per_term = [
        {"row": old_count + index, "anchor_rows": group.tolist(), "axioms": {}}
        for index, group in enumerate(anchors)
    ]
    per_form = {}
    for form, rows in selected_axioms.items():
        before = initial_distances.get(form, np.empty(0))
        after = final_distances.get(form, np.empty(0))
        per_form[form] = {"count": len(rows), "initial": _residual_summary(before), "final": _residual_summary(after)}
        new_to_axioms = {}
        for index, class_ids in enumerate(rows[:, CLASS_COLUMNS[form]]):
            for row_id in set(class_ids[class_ids >= old_count].tolist()):
                new_to_axioms.setdefault(row_id, []).append(index)
        for row_id, indices in new_to_axioms.items():
            per_term[row_id - old_count]["axioms"][form] = {
                "count": len(indices), "initial": _residual_summary(before[indices]),
                "final": _residual_summary(after[indices]),
            }
    for report, offset in zip(per_term, fitted[:, dimension:]):
        report["offset_min"] = float(offset.min())
        report["offset_max"] = float(offset.max())
        report["offset_floor_dimensions"] = int(np.count_nonzero(offset <= min_offset))
    shifts = np.linalg.norm(fitted - np.concatenate((initial_center, initial_offset), axis=1), axis=1)
    for report, shift in zip(per_term, shifts):
        report["box_shift_l2"] = float(shift)

    report = {
        "steps": steps,
        "best_step": best_step,
        "learning_rate": learning_rate,
        "margin": margin,
        "reg_factor": reg_factor,
        "clip_norm": clip_norm,
        "loss_weights": loss_weights,
        "anchor_weight": anchor_weight,
        "min_offset": min_offset,
        "stochastic_sampling": False,
        "initial_objective": first_score,
        "final_objective": final_objective.item(),
        "initial_losses": initial_losses,
        "final_losses": final_losses,
        "per_form": per_form,
        "per_term": per_term,
    }
    return fitted[:, :dimension], fitted[:, dimension:], fitted_bumps, report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v095-config", required=True, type=Path, help="Actual v095 resolved_config.json")
    p.add_argument("--old-obo", required=True, type=Path, help="Historical OBO, used to audit new GO identities")
    p.add_argument("--new-obo", required=True, type=Path, help="Future OBO, used for GO identities and direct edges")
    p.add_argument("--trainer", type=Path, help="Original boxsqel_go_only_torch.py; defaults to scripts/ in project")
    p.add_argument("--box-checkpoint", type=Path, help="Explicit copy of the checkpoint bound to the old full geometry")
    p.add_argument("--training-manifest", type=Path, help="Optional original BoxSquaredEL training config cross-check")
    p.add_argument("--new-normalized", type=Path, help="Optional EL-normalized axioms from the same release as --new-obo")
    p.add_argument("--namespaces", nargs="+", choices=("bp", "mf", "cc"), help="Default: namespace of the v095 config")
    p.add_argument("--output-dir", required=True, type=Path, help="New output directory")
    p.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    p.add_argument("--steps", default=1000, type=int, help="Full-batch local optimization steps")
    p.add_argument("--learning-rate", type=float, help="Default: original BoxSquaredEL checkpoint learning rate")
    p.add_argument("--anchor-weight", default=0.01, type=float, help="Regularization toward anchored initialization")
    p.add_argument("--threads", default=4, type=int, help="CPU PyTorch threads")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Output directory already exists: {out}")
    if args.steps <= 0 or args.threads <= 0:
        raise ValueError("--steps and --threads must be positive")
    torch.set_num_threads(args.threads)
    trainer_path = (args.trainer or ROOT / "scripts/boxsqel_go_only_torch.py").resolve(strict=True)
    trainer = read_trainer(trainer_path)
    old = load_v095_ontology(args.v095_config.resolve(strict=True), ROOT)
    namespaces = set(args.namespaces or [str(old.task).lower()])
    if namespaces - set(NAMESPACES.values()):
        raise ValueError(f"Cannot infer namespace from v095 task {old.task!r}; pass --namespaces")
    box_path = args.box_checkpoint.resolve(strict=True) if args.box_checkpoint else old.box_checkpoint
    cfg, classes, roles, bumps, heads, tails, checkpoint_hash = read_checkpoint(
        old, box_path, trainer, args.training_manifest
    )
    old_version, old_terms = read_obo(args.old_obo)
    new_version, new_terms = read_obo(args.new_obo)
    old_date = re.search(r"\d{4}-\d{2}-\d{2}", old_version)
    new_date = re.search(r"\d{4}-\d{2}-\d{2}", new_version)
    if old_version == new_version or (old_date and new_date and old_date[0] >= new_date[0]):
        raise ValueError(f"Expected an older historical OBO, got {old_version!r} and {new_version!r}")
    owners = identity_owners(new_terms)
    audit = select_candidates(old_terms, new_terms, owners, old.registry, namespaces)
    n_old = len(old.center)
    candidates = [go for go, record in audit.items() if record["status"] == "candidate"]
    known = known_go_rows(old.registry, new_terms, owners)
    trial = {go: n_old + i for i, go in enumerate(candidates)}
    axioms, edges, ignored = collect_axioms(
        new_terms, owners, {**known, **trial}, trial, classes, roles, trainer, args.new_normalized
    )
    selected, anchors = anchored_rows(known, trial, audit, new_terms, edges)
    if not selected:
        raise ValueError(f"No new GO with a modeled non-root historical anchor; {dict(Counter(r['status'] for r in audit.values()))}")
    for i, go in enumerate(selected):
        audit[go]["ontology_row"] = n_old + i
    axioms, edges = restrict_rows(axioms, edges, trial, selected, n_old)
    print(f"[anchored GO] {old_version} -> {new_version}; {len(selected)}/{len(candidates)} candidates to fit", flush=True)
    print(f"[anchored GO] local axioms { {name: len(rows) for name, rows in axioms.items()} }", flush=True)
    center, offset, bump, fit = fit_new_boxes(
        trainer, old_center=old.center, old_offset=old.offset, old_bumps=bumps,
        relation_heads=heads, relation_tails=tails, axioms=axioms, anchors=anchors,
        config=cfg, steps=args.steps, learning_rate=args.learning_rate,
        device=args.device, anchor_weight=args.anchor_weight,
    )
    for go, diagnostics in zip(selected, fit["per_term"]):
        diagnostics["go_id"] = go
    fit["unique_final_box_count"] = int(len(np.unique(np.concatenate((center, offset), axis=1), axis=0)))
    aliases = {go: sorted(new_terms[go].aliases) for go in selected}
    bundle = make_ontology_bundle(old, selected, center, offset, edges, new_aliases=aliases)
    inputs = {
        "old_obo": args.old_obo, "new_obo": args.new_obo, "trainer": trainer_path,
        "new_normalized": args.new_normalized, "training_manifest": args.training_manifest,
    }
    manifest = {
        "schema": "latence-v095-anchored-new-go-v1",
        "complete": True,
        "old_obo_version": old_version, "new_obo_version": new_version,
        "task": old.task, "namespaces": sorted(namespaces), "embedding_dim": old.center.shape[1],
        "old_rows": n_old, "new_rows": len(selected),
        "candidate_status": dict(Counter(record["status"] for record in audit.values())),
        "axioms": {name: len(rows) for name, rows in axioms.items()},
        "appended_direct_edges": {name: len(rows) for name, rows in edges.items()},
        "axiom_notes": ignored, "fit": fit,
        "ontology_only": True,
        "sampled_negative_axioms": False,
        "sources": {
            **{name: {"path": str(path.resolve()), "sha256": digest_file(path)}
               for name, path in inputs.items() if path is not None},
            "box_checkpoint": {"path": str(box_path), "sha256": checkpoint_hash},
            "v095_bindings": {
                name: {"path": str(old.paths[name]), "sha256": checksum}
                for name, checksum in old.hashes.items()
            },
        },
    }
    write_outputs(out, manifest, audit, selected, bundle, center, offset, bump)
    print(f"[anchored GO] saved {len(selected)} new GO boxes to {out}", flush=True)


if __name__ == "__main__":
    main()
