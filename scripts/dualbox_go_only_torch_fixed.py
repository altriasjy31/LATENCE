#!/usr/bin/env python
"""
Train BoxSquaredEL-style box embeddings from normalized GO axioms.

This is a standalone PyTorch implementation aligned with the public
KRR-Oxford/BoxSquaredEL model design:

* each class is one axis-aligned box
      class row = [center (d) | raw_offset (d)]
  and the geometric half-width is abs(raw_offset);
* each object property has a head box and a tail box;
* each class has one bump vector used by existential-restriction losses;
* NF1, NF2, NF3, NF4, disjointness, role inclusion, and role-chain losses
  follow the BoxSquaredEL conventions.

The exported class pickle is directly compatible with:

    python evaluate_go_embedding.py \
      --embedding-file go_boxsqel_classes.pkl \
      --go-file go.obo \
      --geometry box \
      --layout center_offset \
      --embedding-dim <d>

The pickle stores non-negative offsets:

    classes | embeddings
    GO_...  | [center_1, ..., center_d, abs(offset_1), ..., abs(offset_d)]

Input formats
-------------
The parser accepts one normalized axiom per line in OWL functional syntax:

    SubClassOf(C D)                                  # NF1
    SubClassOf(ObjectIntersectionOf(C D) E)          # NF2
    SubClassOf(C ObjectSomeValuesFrom(R D))          # NF3
    SubClassOf(ObjectSomeValuesFrom(R C) D)          # NF4
    SubClassOf(ObjectIntersectionOf(C D) owl:Nothing)# disjoint
    DisjointClasses(C D)
    SubObjectPropertyOf(R S)                         # role inclusion
    SubObjectPropertyOf(ObjectPropertyChain(R1 R2) S)# role chain
    TransitiveObjectProperty(R)                      # R o R <= R

It also accepts the simplified forms used by the earlier GO scripts:

    C SubClassOf D
    C and D SubClassOf E
    C SubClassOf R some D
    R some C SubClassOf D
    R SubPropertyOf S
    R1 o R2 SubPropertyOf S

Important behavior
------------------
* No artificial disjoint axiom is created when the ontology has none.
* NF3 negatives are regenerated dynamically for every optimizer step.
* Known positive NF3 triples are filtered from corrupted negatives by default.
* ``best`` and ``final`` artifacts are always saved separately.
* The unsuffixed class/relation/bump pickle paths are copies of the selected
  best artifacts, never accidental final-epoch overwrites.
* Training-time diagnostics measure strict NF1 containment, matched-negative
  false containment, containment specificity, and box-scale statistics.

This script intentionally implements the official single-box BoxSquaredEL
architecture. It does not implement the experimental positive/negative
four-part class representation from the earlier DualBox prototype.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import click as ck
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger("boxsqel_go_only")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    elif torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def resolve_device(device: str) -> torch.device:
    text = str(device).strip().lower()
    if text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text.startswith("/"):
        text = text[1:]
    if text.startswith("gpu:"):
        text = "cuda:" + text.split(":", 1)[1]
    if text.startswith("cpu:"):
        text = "cpu"

    resolved = torch.device(text)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return resolved


def tagged_path(path: str | Path, tag: str) -> Path:
    p = Path(path)
    if p.suffix:
        return p.with_name(f"{p.stem}_{tag}{p.suffix}")
    return p.with_name(f"{p.name}_{tag}")


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_int_list(text: str) -> List[int]:
    text = str(text).strip()
    if not text:
        return []
    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("Scheduler milestones must be positive integers.")
        values.append(value)
    return sorted(set(values))


def shorten_identifier(value: str) -> str:
    """Shorten an OWL IRI while preserving compact identifiers."""
    text = str(value).strip()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1]
    if text.startswith("http://") or text.startswith("https://"):
        text = re.split(r"[/#]", text.rstrip("/#"))[-1]
    return text.replace("GO:", "GO_") if re.fullmatch(r"GO:\d+", text) else text


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    source = Path(source)
    destination = Path(destination)
    ensure_parent(destination)
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)


def safe_json_dump(payload: Mapping[str, Any], path: str | Path) -> None:
    ensure_parent(path)

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(convert(dict(payload)), handle, indent=2, ensure_ascii=False)
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# Normalized-axiom parsing
# ---------------------------------------------------------------------------


@dataclass
class ParserReport:
    total_lines: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    ignored_lines: int = 0
    parsed_lines: int = 0
    expanded_axioms: int = 0
    unparsed_lines: int = 0
    nf1: int = 0
    nf2: int = 0
    nf3: int = 0
    nf4: int = 0
    disjoint: int = 0
    role_inclusion: int = 0
    role_chain: int = 0


@dataclass
class OntologyTrainingData:
    arrays: Dict[str, np.ndarray]
    classes: Dict[str, int]
    relations: Dict[str, int]
    report: ParserReport
    unparsed_examples: List[str]


AXIOM_WIDTHS: Dict[str, int] = {
    "nf1": 2,
    "nf2": 3,
    "nf3": 3,
    "nf4": 3,
    "disjoint": 2,
    "role_inclusion": 2,
    "role_chain": 3,
}


def strip_inline_comment(line: str) -> str:
    """Remove shell-style comments only when '#' starts a token.

    Full IRIs may legally contain '#', so an arbitrary split('#') is unsafe.
    """
    for idx, char in enumerate(line):
        if char == "#" and (idx == 0 or line[idx - 1].isspace()):
            return line[:idx].rstrip()
    return line.rstrip()


def split_top_level(text: str) -> List[str]:
    """Split on whitespace outside parentheses and angle-bracket IRIs."""
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    in_angle = False

    for char in text.strip():
        if char == "<" and depth >= 0:
            in_angle = True
        elif char == ">" and in_angle:
            in_angle = False
        elif not in_angle:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError(f"Unbalanced parentheses in: {text}")

        if char.isspace() and depth == 0 and not in_angle:
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(char)

    if depth != 0 or in_angle:
        raise ValueError(f"Unbalanced expression: {text}")
    if current:
        parts.append("".join(current))
    return parts


def unwrap_call(expression: str, name: str) -> Optional[str]:
    prefix = name + "("
    if expression.startswith(prefix) and expression.endswith(")"):
        return expression[len(prefix):-1].strip()
    return None


def parse_functional_axiom(line: str) -> Optional[List[Tuple[str, Tuple[str, ...]]]]:
    inner = unwrap_call(line, "SubClassOf")
    if inner is not None:
        args = split_top_level(inner)
        if len(args) != 2:
            return None
        lhs, rhs = args

        lhs_inter = unwrap_call(lhs, "ObjectIntersectionOf")
        if lhs_inter is not None:
            members = split_top_level(lhs_inter)
            if len(members) == 2:
                if rhs in {"owl:Nothing", "<http://www.w3.org/2002/07/owl#Nothing>"}:
                    return [("disjoint", (members[0], members[1]))]
                return [("nf2", (members[0], members[1], rhs))]
            return None

        lhs_some = unwrap_call(lhs, "ObjectSomeValuesFrom")
        if lhs_some is not None:
            members = split_top_level(lhs_some)
            if len(members) == 2:
                return [("nf4", (members[0], members[1], rhs))]
            return None

        rhs_some = unwrap_call(rhs, "ObjectSomeValuesFrom")
        if rhs_some is not None:
            members = split_top_level(rhs_some)
            if len(members) == 2:
                return [("nf3", (lhs, members[0], members[1]))]
            return None

        return [("nf1", (lhs, rhs))]

    inner = unwrap_call(line, "DisjointClasses")
    if inner is not None:
        members = split_top_level(inner)
        if len(members) < 2:
            return None
        pairs: List[Tuple[str, Tuple[str, ...]]] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append(("disjoint", (members[i], members[j])))
        return pairs

    inner = unwrap_call(line, "EquivalentClasses")
    if inner is not None:
        members = split_top_level(inner)
        if len(members) == 2 and all("(" not in member for member in members):
            return [
                ("nf1", (members[0], members[1])),
                ("nf1", (members[1], members[0])),
            ]
        return None

    inner = unwrap_call(line, "SubObjectPropertyOf")
    if inner is not None:
        args = split_top_level(inner)
        if len(args) != 2:
            return None
        lhs, rhs = args
        chain = unwrap_call(lhs, "ObjectPropertyChain")
        if chain is not None:
            members = split_top_level(chain)
            if len(members) == 2:
                return [("role_chain", (members[0], members[1], rhs))]
            return None
        return [("role_inclusion", (lhs, rhs))]

    inner = unwrap_call(line, "EquivalentObjectProperties")
    if inner is not None:
        members = split_top_level(inner)
        if len(members) == 2:
            return [
                ("role_inclusion", (members[0], members[1])),
                ("role_inclusion", (members[1], members[0])),
            ]
        return None

    inner = unwrap_call(line, "TransitiveObjectProperty")
    if inner is not None:
        members = split_top_level(inner)
        if len(members) == 1:
            relation = members[0]
            return [("role_chain", (relation, relation, relation))]
        return None

    inner = unwrap_call(line, "ObjectPropertyDomain")
    if inner is not None:
        members = split_top_level(inner)
        if len(members) == 2:
            relation, domain = members
            return [("nf4", (relation, "owl:Thing", domain))]
        return None

    ignored_prefixes = (
        "Prefix(",
        "Ontology(",
        "Import(",
        "Declaration(",
        "Annotation(",
        "AnnotationAssertion(",
        "SubAnnotationPropertyOf(",
        "AnnotationPropertyDomain(",
        "AnnotationPropertyRange(",
    )
    if line.startswith(ignored_prefixes) or line == ")":
        return []

    return None


def parse_simplified_axiom(line: str) -> Optional[List[Tuple[str, Tuple[str, ...]]]]:
    tokens = line.split()

    if len(tokens) == 3 and tokens[1] == "SubClassOf":
        return [("nf1", (tokens[0], tokens[2]))]

    if len(tokens) == 5 and tokens[1] == "and" and tokens[3] == "SubClassOf":
        if tokens[4] == "owl:Nothing":
            return [("disjoint", (tokens[0], tokens[2]))]
        return [("nf2", (tokens[0], tokens[2], tokens[4]))]

    if len(tokens) == 5 and tokens[1] == "SubClassOf" and tokens[3] == "some":
        return [("nf3", (tokens[0], tokens[2], tokens[4]))]

    if len(tokens) == 5 and tokens[1] == "some" and tokens[3] == "SubClassOf":
        return [("nf4", (tokens[0], tokens[2], tokens[4]))]

    if len(tokens) == 3 and tokens[1] in {"SubPropertyOf", "SubObjectPropertyOf"}:
        return [("role_inclusion", (tokens[0], tokens[2]))]

    if (
        len(tokens) == 5
        and tokens[1] in {"o", "compose", "chain"}
        and tokens[3] in {"SubPropertyOf", "SubObjectPropertyOf"}
    ):
        return [("role_chain", (tokens[0], tokens[2], tokens[4]))]

    return None


def parse_axiom_line(line: str) -> Optional[List[Tuple[str, Tuple[str, ...]]]]:
    parsed = parse_functional_axiom(line)
    if parsed is not None:
        return parsed
    return parse_simplified_axiom(line)


def load_normalized_ontology(
    filename: str | Path,
    strict_parser: bool,
    max_unparsed_examples: int,
) -> OntologyTrainingData:
    classes: Dict[str, int] = {}
    relations: Dict[str, int] = {}
    rows: Dict[str, List[Tuple[int, ...]]] = {key: [] for key in AXIOM_WIDTHS}
    report = ParserReport()
    unparsed_examples: List[str] = []

    def class_id(name: str) -> int:
        if name not in classes:
            classes[name] = len(classes)
        return classes[name]

    def relation_id(name: str) -> int:
        if name not in relations:
            relations[name] = len(relations)
        return relations[name]

    with Path(filename).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            report.total_lines += 1
            stripped = raw_line.strip()
            if not stripped:
                report.blank_lines += 1
                continue
            if stripped.startswith("#") or stripped.startswith("//"):
                report.comment_lines += 1
                continue

            line = strip_inline_comment(stripped)
            if not line:
                report.comment_lines += 1
                continue

            try:
                parsed = parse_axiom_line(line)
            except ValueError:
                parsed = None

            if parsed == []:
                report.ignored_lines += 1
                continue
            if parsed is None:
                report.unparsed_lines += 1
                if len(unparsed_examples) < max_unparsed_examples:
                    unparsed_examples.append(line)
                continue

            report.parsed_lines += 1
            report.expanded_axioms += len(parsed)

            for form, values in parsed:
                if form == "nf1":
                    rows[form].append((class_id(values[0]), class_id(values[1])))
                elif form == "nf2":
                    rows[form].append(
                        (class_id(values[0]), class_id(values[1]), class_id(values[2]))
                    )
                elif form == "nf3":
                    rows[form].append(
                        (class_id(values[0]), relation_id(values[1]), class_id(values[2]))
                    )
                elif form == "nf4":
                    rows[form].append(
                        (relation_id(values[0]), class_id(values[1]), class_id(values[2]))
                    )
                elif form == "disjoint":
                    rows[form].append((class_id(values[0]), class_id(values[1])))
                elif form == "role_inclusion":
                    rows[form].append((relation_id(values[0]), relation_id(values[1])))
                elif form == "role_chain":
                    rows[form].append(
                        (relation_id(values[0]), relation_id(values[1]), relation_id(values[2]))
                    )
                else:
                    raise RuntimeError(f"Internal parser error: unsupported form {form!r}")
                setattr(report, form, getattr(report, form) + 1)

    if strict_parser and report.unparsed_lines > 0:
        examples = "\n".join(f"  - {item}" for item in unparsed_examples)
        raise ValueError(
            f"Strict parsing failed: {report.unparsed_lines} input lines were not parsed.\n"
            f"Examples:\n{examples}"
        )

    arrays: Dict[str, np.ndarray] = {}
    for key, width in AXIOM_WIDTHS.items():
        array = np.asarray(rows[key], dtype=np.int64)
        arrays[key] = array.reshape((-1, width))

    if not classes:
        raise ValueError("No ontology classes were parsed from the input file.")

    return OntologyTrainingData(
        arrays=arrays,
        classes=classes,
        relations=relations,
        report=report,
        unparsed_examples=unparsed_examples,
    )


def invert_mapping(mapping: Mapping[str, int]) -> List[str]:
    inverse = [""] * len(mapping)
    for name, idx in mapping.items():
        inverse[idx] = name
    if any(not value for value in inverse):
        raise ValueError("Mapping indices are not contiguous from zero.")
    return inverse


# ---------------------------------------------------------------------------
# Batching and dynamic negative sampling
# ---------------------------------------------------------------------------


class AxiomBatcher:
    """Independently sample each normal form with replacement.

    This mirrors the original BoxSquaredEL training behavior while allowing
    enough optimizer steps per epoch to cover the largest axiom group.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray], batch_size: int, seed: int):
        self.arrays = dict(arrays)
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(seed)

    def sample(self, key: str) -> np.ndarray:
        array = self.arrays[key]
        width = AXIOM_WIDTHS[key]
        if len(array) == 0:
            return np.zeros((0, width), dtype=np.int64)
        if len(array) <= self.batch_size:
            return array.copy()
        indices = self.rng.choice(len(array), size=self.batch_size, replace=True)
        return array[indices]

    def next_batch(self) -> Dict[str, np.ndarray]:
        return {key: self.sample(key) for key in AXIOM_WIDTHS}

    def get_state(self) -> tuple:
        return self.rng.get_state()

    def set_state(self, state: tuple) -> None:
        self.rng.set_state(state)


class NF3NegativeSampler:
    def __init__(
        self,
        num_classes: int,
        known_positive_nf3: np.ndarray,
        num_negatives: int,
        seed: int,
        filter_known: bool = True,
        max_resample_rounds: int = 20,
    ):
        self.num_classes = int(num_classes)
        self.num_negatives = int(num_negatives)
        self.filter_known = bool(filter_known)
        self.max_resample_rounds = int(max_resample_rounds)
        self.rng = np.random.RandomState(seed)
        self.known: Set[Tuple[int, int, int]] = {
            tuple(map(int, row)) for row in np.asarray(known_positive_nf3)
        }

    def _draw_class(self, original: int, relation: int, other: int, corrupt_head: bool) -> int:
        for _ in range(self.max_resample_rounds):
            candidate = int(self.rng.randint(0, self.num_classes))
            if candidate == original:
                continue
            triple = (candidate, relation, other) if corrupt_head else (other, relation, candidate)
            if self.filter_known and triple in self.known:
                continue
            return candidate

        # Extremely dense edge cases: fall back to a deterministic scan.
        for candidate in range(self.num_classes):
            if candidate == original:
                continue
            triple = (candidate, relation, other) if corrupt_head else (other, relation, candidate)
            if not self.filter_known or triple not in self.known:
                return candidate
        raise RuntimeError("Could not construct a valid corrupted NF3 triple.")

    def sample(self, positive_batch: np.ndarray) -> np.ndarray:
        positive_batch = np.asarray(positive_batch, dtype=np.int64).reshape((-1, 3))
        if len(positive_batch) == 0 or self.num_negatives <= 0:
            return np.zeros((0, 3), dtype=np.int64)

        negatives: List[Tuple[int, int, int]] = []
        for c, r, d in positive_batch.tolist():
            for _ in range(self.num_negatives):
                corrupted_d = self._draw_class(d, r, c, corrupt_head=False)
                negatives.append((c, r, corrupted_d))
                corrupted_c = self._draw_class(c, r, d, corrupt_head=True)
                negatives.append((corrupted_c, r, d))
        return np.asarray(negatives, dtype=np.int64).reshape((-1, 3))

    def get_state(self) -> tuple:
        return self.rng.get_state()

    def set_state(self, state: tuple) -> None:
        self.rng.set_state(state)


def move_batch_to_device(
    batch: Mapping[str, np.ndarray],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.long, device=device)
        for key, value in batch.items()
    }


# ---------------------------------------------------------------------------
# BoxSquaredEL model
# ---------------------------------------------------------------------------


@dataclass
class Boxes:
    centers: torch.Tensor
    offsets: torch.Tensor

    def intersect(self, other: "Boxes") -> Tuple["Boxes", torch.Tensor, torch.Tensor]:
        lower = torch.maximum(self.centers - self.offsets, other.centers - other.offsets)
        upper = torch.minimum(self.centers + self.offsets, other.centers + other.offsets)
        center = (lower + upper) / 2.0
        # abs matches the public BoxSquaredEL implementation; lower > upper is
        # separately penalized in NF2.
        offset = torch.abs(upper - lower) / 2.0
        return Boxes(center, offset), lower, upper

    def translate(self, direction: torch.Tensor) -> "Boxes":
        return Boxes(self.centers + direction, self.offsets)


class BoxSquaredELModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_relations: int,
        embedding_dim: int,
        margin: float = 0.0,
        neg_dist: float = 2.0,
        reg_factor: float = 0.05,
        negative_loss_mode: str = "official",
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if margin < 0:
            raise ValueError("BoxSquaredEL margin must be >= 0.")
        if negative_loss_mode not in {"official", "hinge"}:
            raise ValueError("negative_loss_mode must be 'official' or 'hinge'.")

        self.num_classes = int(num_classes)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)
        self.margin = float(margin)
        self.neg_dist = float(neg_dist)
        self.reg_factor = float(reg_factor)
        self.negative_loss_mode = negative_loss_mode

        # Official BoxSquaredEL parameterization: one box per class.
        self.class_embeds = self._init_embeddings(num_classes, 2 * embedding_dim)
        self.bumps = self._init_embeddings(num_classes, embedding_dim)
        self.relation_heads = self._init_embeddings(max(num_relations, 1), 2 * embedding_dim)
        self.relation_tails = self._init_embeddings(max(num_relations, 1), 2 * embedding_dim)

    @staticmethod
    def _init_embeddings(num_embeddings: int, width: int) -> nn.Embedding:
        embedding = nn.Embedding(num_embeddings, width)
        nn.init.uniform_(embedding.weight, a=-1.0, b=1.0)
        with torch.no_grad():
            denominator = torch.linalg.norm(
                embedding.weight, dim=1, keepdim=True
            ).clamp_min(1e-12)
            embedding.weight.div_(denominator)
        return embedding

    def get_boxes(self, embedding: torch.Tensor) -> Boxes:
        d = self.embedding_dim
        return Boxes(
            centers=embedding[:, :d],
            offsets=torch.abs(embedding[:, d:]),
        )

    def class_boxes(self, ids: torch.Tensor) -> Boxes:
        return self.get_boxes(self.class_embeds(ids))

    def relation_boxes(self, ids: torch.Tensor) -> Tuple[Boxes, Boxes]:
        return self.get_boxes(self.relation_heads(ids)), self.get_boxes(self.relation_tails(ids))

    def inclusion_distance(self, first: Boxes, second: Boxes) -> torch.Tensor:
        difference = torch.abs(first.centers - second.centers)
        violation = F.relu(difference + first.offsets - second.offsets - self.margin)
        return torch.linalg.norm(violation, dim=1, keepdim=True)

    def disjoint_distance(self, first: Boxes, second: Boxes) -> torch.Tensor:
        difference = torch.abs(first.centers - second.centers)
        overlap = F.relu(-difference + first.offsets + second.offsets - self.margin)
        return torch.linalg.norm(overlap, dim=1, keepdim=True)

    def separation_distance(self, first: Boxes, second: Boxes) -> torch.Tensor:
        difference = torch.abs(first.centers - second.centers)
        separation = F.relu(difference - first.offsets - second.offsets + self.margin)
        return torch.linalg.norm(separation, dim=1, keepdim=True)

    def nf1_loss(self, data: torch.Tensor) -> torch.Tensor:
        child = self.class_boxes(data[:, 0])
        parent = self.class_boxes(data[:, 1])
        return self.inclusion_distance(child, parent)

    def nf2_loss(self, data: torch.Tensor) -> torch.Tensor:
        first = self.class_boxes(data[:, 0])
        second = self.class_boxes(data[:, 1])
        target = self.class_boxes(data[:, 2])
        intersection, lower, upper = first.intersect(second)
        empty_intersection_penalty = torch.linalg.norm(
            F.relu(lower - upper), dim=1, keepdim=True
        )
        return self.inclusion_distance(intersection, target) + empty_intersection_penalty

    def disjoint_loss(self, data: torch.Tensor) -> torch.Tensor:
        first = self.class_boxes(data[:, 0])
        second = self.class_boxes(data[:, 1])
        return self.disjoint_distance(first, second)

    def nf3_loss(self, data: torch.Tensor) -> torch.Tensor:
        source = self.class_boxes(data[:, 0])
        target = self.class_boxes(data[:, 2])
        source_bump = self.bumps(data[:, 0])
        target_bump = self.bumps(data[:, 2])
        head, tail = self.relation_boxes(data[:, 1])
        first = self.inclusion_distance(source.translate(target_bump), head)
        second = self.inclusion_distance(target.translate(source_bump), tail)
        return (first + second) / 2.0

    def nf3_negative_distances(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        source = self.class_boxes(data[:, 0])
        target = self.class_boxes(data[:, 2])
        source_bump = self.bumps(data[:, 0])
        target_bump = self.bumps(data[:, 2])
        head, tail = self.relation_boxes(data[:, 1])
        first = self.separation_distance(source.translate(target_bump), head)
        second = self.separation_distance(target.translate(source_bump), tail)
        return first, second

    def nf4_loss(self, data: torch.Tensor) -> torch.Tensor:
        target = self.class_boxes(data[:, 2])
        head, _ = self.relation_boxes(data[:, 0])
        source_bump = self.bumps(data[:, 1])
        return self.inclusion_distance(head.translate(-source_bump), target)

    def role_inclusion_loss(self, data: torch.Tensor) -> torch.Tensor:
        sub_head, sub_tail = self.relation_boxes(data[:, 0])
        super_head, super_tail = self.relation_boxes(data[:, 1])
        return (
            self.inclusion_distance(sub_head, super_head)
            + self.inclusion_distance(sub_tail, super_tail)
        ) / 2.0

    def role_chain_loss(self, data: torch.Tensor) -> torch.Tensor:
        first_head, _ = self.relation_boxes(data[:, 0])
        _, second_tail = self.relation_boxes(data[:, 1])
        super_head, super_tail = self.relation_boxes(data[:, 2])
        return (
            self.inclusion_distance(first_head, super_head)
            + self.inclusion_distance(second_tail, super_tail)
        ) / 2.0

    def zero(self) -> torch.Tensor:
        return self.class_embeds.weight.new_tensor(0.0)

    def squared_mean(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return self.zero()
        return values.square().mean()

    def negative_penalty(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.numel() == 0:
            return self.zero()
        if self.negative_loss_mode == "official":
            residual = self.neg_dist - distances
        else:
            residual = F.relu(self.neg_dist - distances)
        return residual.square().mean()

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        nf3_negatives: torch.Tensor,
        loss_weights: Mapping[str, float],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        components: Dict[str, torch.Tensor] = {
            "nf1": self.squared_mean(self.nf1_loss(batch["nf1"])),
            "nf2": self.squared_mean(self.nf2_loss(batch["nf2"])),
            "nf3": self.squared_mean(self.nf3_loss(batch["nf3"])),
            "nf4": self.squared_mean(self.nf4_loss(batch["nf4"])),
            "disjoint": self.squared_mean(self.disjoint_loss(batch["disjoint"])),
            "role_inclusion": self.squared_mean(
                self.role_inclusion_loss(batch["role_inclusion"])
            ),
            "role_chain": self.squared_mean(self.role_chain_loss(batch["role_chain"])),
        }

        if nf3_negatives.numel() > 0:
            neg_first, neg_second = self.nf3_negative_distances(nf3_negatives)
            components["nf3_neg"] = (
                self.negative_penalty(neg_first) + self.negative_penalty(neg_second)
            )
        else:
            components["nf3_neg"] = self.zero()

        components["bump_reg"] = torch.linalg.norm(
            self.bumps.weight, dim=1
        ).mean()

        total = self.zero()
        for name, component in components.items():
            weight = self.reg_factor if name == "bump_reg" else float(loss_weights[name])
            total = total + weight * component
        return total, components


# ---------------------------------------------------------------------------
# Geometry diagnostics used during training
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticPairs:
    positive: np.ndarray
    negative: np.ndarray
    depths: np.ndarray


class GeometryDiagnostics:
    def __init__(
        self,
        nf1: np.ndarray,
        num_classes: int,
        sample_size: int,
        class_sample_size: int,
        seed: int,
    ):
        self.nf1 = np.asarray(nf1, dtype=np.int64).reshape((-1, 2))
        self.num_classes = int(num_classes)
        self.sample_size = int(sample_size)
        self.class_sample_size = int(class_sample_size)
        self.rng = np.random.RandomState(seed)
        self.parents: List[List[int]] = [[] for _ in range(num_classes)]
        self.children: List[List[int]] = [[] for _ in range(num_classes)]
        for child, parent in self.nf1.tolist():
            self.parents[child].append(parent)
            self.children[parent].append(child)
        self.depth = self._compute_min_depth()
        self.depth_buckets = self._build_depth_buckets()
        self.pairs = self._build_pairs()
        self.class_sample = self._sample_classes()

    def _compute_min_depth(self) -> np.ndarray:
        depth = np.full(self.num_classes, -1, dtype=np.int64)
        graph_nodes = {
            int(node) for edge in self.nf1.tolist() for node in edge
        }
        roots = [node for node in graph_nodes if len(self.parents[node]) == 0]
        queue: deque[int] = deque()
        for root in roots:
            depth[root] = 0
            queue.append(root)
        while queue:
            parent = queue.popleft()
            for child in self.children[parent]:
                candidate = depth[parent] + 1
                if depth[child] < 0 or candidate < depth[child]:
                    depth[child] = candidate
                    queue.append(child)
        return depth

    def _build_depth_buckets(self) -> Dict[int, np.ndarray]:
        buckets: Dict[int, List[int]] = defaultdict(list)
        for node, value in enumerate(self.depth.tolist()):
            if value >= 0:
                buckets[int(value)].append(node)
        return {
            key: np.asarray(values, dtype=np.int64)
            for key, values in buckets.items()
        }

    def _ancestors(self, start: int) -> Set[int]:
        visited: Set[int] = set()
        stack = list(self.parents[start])
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.parents[node])
        return visited

    def _descendants(self, start: int) -> Set[int]:
        visited: Set[int] = set()
        stack = list(self.children[start])
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self.children[node])
        return visited

    def _candidate_pool(self, desired_depth: int) -> np.ndarray:
        pools: List[np.ndarray] = []
        for delta in (0, -1, 1, -2, 2):
            candidate_depth = desired_depth + delta
            if candidate_depth in self.depth_buckets:
                pools.append(self.depth_buckets[candidate_depth])
        if not pools:
            return np.arange(self.num_classes, dtype=np.int64)
        return np.unique(np.concatenate(pools))

    def _build_pairs(self) -> DiagnosticPairs:
        if len(self.nf1) == 0 or self.sample_size <= 0:
            empty = np.zeros((0, 2), dtype=np.int64)
            return DiagnosticPairs(empty, empty.copy(), np.zeros((0,), dtype=np.int64))

        size = min(self.sample_size, len(self.nf1))
        indices = self.rng.choice(len(self.nf1), size=size, replace=False)
        positive = self.nf1[indices]
        negative_rows: List[Tuple[int, int]] = []
        kept_positive: List[Tuple[int, int]] = []
        kept_depths: List[int] = []

        relation_pairs = {tuple(map(int, row)) for row in self.nf1}
        ancestor_cache: Dict[int, Set[int]] = {}
        descendant_cache: Dict[int, Set[int]] = {}

        for child, true_parent in positive.tolist():
            ancestor_cache.setdefault(child, self._ancestors(child))
            descendant_cache.setdefault(child, self._descendants(child))
            forbidden = ancestor_cache[child] | descendant_cache[child] | {child}
            desired_depth = int(self.depth[true_parent])
            pool = self._candidate_pool(desired_depth)
            if len(pool) == 0:
                continue

            candidate: Optional[int] = None
            for _ in range(100):
                trial = int(pool[self.rng.randint(0, len(pool))])
                if trial in forbidden or (child, trial) in relation_pairs:
                    continue
                candidate = trial
                break
            if candidate is None:
                for trial in pool.tolist():
                    trial = int(trial)
                    if trial not in forbidden and (child, trial) not in relation_pairs:
                        candidate = trial
                        break
            if candidate is None:
                continue

            kept_positive.append((child, true_parent))
            negative_rows.append((child, candidate))
            kept_depths.append(desired_depth)

        return DiagnosticPairs(
            positive=np.asarray(kept_positive, dtype=np.int64).reshape((-1, 2)),
            negative=np.asarray(negative_rows, dtype=np.int64).reshape((-1, 2)),
            depths=np.asarray(kept_depths, dtype=np.int64),
        )

    def _sample_classes(self) -> np.ndarray:
        if self.class_sample_size <= 0 or self.class_sample_size >= self.num_classes:
            return np.arange(self.num_classes, dtype=np.int64)
        return self.rng.choice(
            self.num_classes,
            size=self.class_sample_size,
            replace=False,
        ).astype(np.int64)

    @staticmethod
    def _pair_statistics(
        model: BoxSquaredELModel,
        pairs: np.ndarray,
        device: torch.device,
        tolerance: float,
        temperature: float,
    ) -> Tuple[float, float, float, float]:
        if len(pairs) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        tensor = torch.as_tensor(pairs, dtype=torch.long, device=device)
        first = model.class_boxes(tensor[:, 0])
        second = model.class_boxes(tensor[:, 1])
        raw = torch.abs(first.centers - second.centers) + first.offsets - second.offsets
        worst_constraint = raw.amax(dim=1)
        per_pair_violation = F.relu(worst_constraint)
        strict_satisfied = (worst_constraint <= tolerance).float().mean()
        margin_satisfied = ((worst_constraint - model.margin) <= tolerance).float().mean()
        soft_satisfaction = torch.sigmoid(-worst_constraint / max(temperature, 1e-12)).mean()
        return (
            float(strict_satisfied.detach().cpu()),
            float(margin_satisfied.detach().cpu()),
            float(per_pair_violation.mean().detach().cpu()),
            float(soft_satisfaction.detach().cpu()),
        )

    @torch.no_grad()
    def compute(
        self,
        model: BoxSquaredELModel,
        device: torch.device,
        tolerance: float,
        temperature: float,
    ) -> Dict[str, float]:
        was_training = model.training
        model.eval()

        true_strict, true_margin, true_violation, true_soft = self._pair_statistics(
            model, self.pairs.positive, device, tolerance, temperature
        )
        false_strict, false_margin, false_violation, false_soft = self._pair_statistics(
            model, self.pairs.negative, device, tolerance, temperature
        )

        class_ids = torch.as_tensor(self.class_sample, dtype=torch.long, device=device)
        boxes = model.class_boxes(class_ids)
        offsets = boxes.offsets.reshape(-1)
        centers = boxes.centers
        bumps = model.bumps(class_ids)

        metrics = {
            "diag_true_containment": true_strict,
            "diag_true_containment_train_margin": true_margin,
            "diag_false_containment": false_strict,
            "diag_false_containment_train_margin": false_margin,
            "diag_containment_specificity": true_strict - false_strict,
            "diag_true_soft_containment": true_soft,
            "diag_false_soft_containment": false_soft,
            "diag_soft_containment_specificity": true_soft - false_soft,
            "diag_true_violation_mean": true_violation,
            "diag_false_violation_mean": false_violation,
            "diag_offset_mean": float(offsets.mean().detach().cpu()),
            "diag_offset_median": float(offsets.median().detach().cpu()),
            "diag_offset_p95": float(torch.quantile(offsets, 0.95).detach().cpu()),
            "diag_offset_near_zero_fraction": float(
                (offsets <= tolerance).float().mean().detach().cpu()
            ),
            "diag_center_norm_mean": float(
                torch.linalg.norm(centers, dim=1).mean().detach().cpu()
            ),
            "diag_bump_norm_mean": float(
                torch.linalg.norm(bumps, dim=1).mean().detach().cpu()
            ),
            "diag_positive_pairs": float(len(self.pairs.positive)),
            "diag_negative_pairs": float(len(self.pairs.negative)),
        }

        if was_training:
            model.train()
        return metrics


# ---------------------------------------------------------------------------
# Artifact serialization
# ---------------------------------------------------------------------------


def class_embedding_arrays(model: BoxSquaredELModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = model.class_embeds.weight.detach().cpu().numpy().astype(np.float32)
    d = model.embedding_dim
    centers = raw[:, :d]
    offsets = np.abs(raw[:, d:])
    combined = np.concatenate([centers, offsets], axis=1).astype(np.float32, copy=False)
    return centers, offsets, combined


def relation_embedding_arrays(
    model: BoxSquaredELModel,
    relation_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = model.embedding_dim
    raw_heads = model.relation_heads.weight.detach().cpu().numpy().astype(np.float32)[:relation_count]
    raw_tails = model.relation_tails.weight.detach().cpu().numpy().astype(np.float32)[:relation_count]
    heads = np.concatenate(
        [raw_heads[:, :d], np.abs(raw_heads[:, d:])], axis=1
    ).astype(np.float32, copy=False)
    tails = np.concatenate(
        [raw_tails[:, :d], np.abs(raw_tails[:, d:])], axis=1
    ).astype(np.float32, copy=False)
    combined = np.concatenate([heads, tails], axis=1).astype(np.float32, copy=False)
    return heads, tails, combined


def save_pickle_artifacts(
    model: BoxSquaredELModel,
    class_names: Sequence[str],
    relation_names: Sequence[str],
    classes_path: str | Path,
    relations_path: str | Path,
    bumps_path: str | Path,
    shorten_iris: bool,
    add_box_columns: bool,
) -> None:
    ensure_parent(classes_path)
    ensure_parent(relations_path)
    ensure_parent(bumps_path)

    display_classes = [shorten_identifier(item) if shorten_iris else item for item in class_names]
    display_relations = [shorten_identifier(item) if shorten_iris else item for item in relation_names]

    centers, offsets, combined = class_embedding_arrays(model)
    class_data: Dict[str, Any] = {
        "classes": pd.Series(display_classes, dtype=object),
        "embeddings": list(combined),
    }
    if add_box_columns:
        class_data["center"] = list(centers)
        class_data["offset"] = list(offsets)
    pd.DataFrame(class_data).to_pickle(classes_path)

    heads, tails, relation_combined = relation_embedding_arrays(model, len(relation_names))
    relation_data: Dict[str, Any] = {
        "relations": pd.Series(display_relations, dtype=object),
        "embeddings": list(relation_combined),
    }
    if add_box_columns:
        d = model.embedding_dim
        relation_data["head_center"] = list(heads[:, :d])
        relation_data["head_offset"] = list(heads[:, d:])
        relation_data["tail_center"] = list(tails[:, :d])
        relation_data["tail_offset"] = list(tails[:, d:])
    pd.DataFrame(relation_data).to_pickle(relations_path)

    bumps = model.bumps.weight.detach().cpu().numpy().astype(np.float32)
    pd.DataFrame(
        {
            "classes": pd.Series(display_classes, dtype=object),
            "bumps": list(bumps),
        }
    ).to_pickle(bumps_path)


def export_repository_npy(model: BoxSquaredELModel, folder: str | Path, best: bool) -> None:
    """Export raw parameters using the official repository naming convention."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "_best" if best else ""
    np.save(
        folder / f"class_embeds{suffix}.npy",
        model.class_embeds.weight.detach().cpu().numpy(),
    )
    np.save(
        folder / f"bumps{suffix}.npy",
        model.bumps.weight.detach().cpu().numpy(),
    )
    np.save(
        folder / f"rel_heads{suffix}.npy",
        model.relation_heads.weight.detach().cpu().numpy(),
    )
    np.save(
        folder / f"rel_tails{suffix}.npy",
        model.relation_tails.weight.detach().cpu().numpy(),
    )


def capture_rng_state(
    batcher: AxiomBatcher,
    negative_sampler: NF3NegativeSampler,
) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "batcher": batcher.get_state(),
        "negative_sampler": negative_sampler.get_state(),
    }


def restore_rng_state(
    state: Mapping[str, Any],
    batcher: AxiomBatcher,
    negative_sampler: NF3NegativeSampler,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    batcher.set_state(state["batcher"])
    negative_sampler.set_state(state["negative_sampler"])


def save_checkpoint(
    path: str | Path,
    model: BoxSquaredELModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    best_score: float,
    best_epoch: int,
    classes: Mapping[str, int],
    relations: Mapping[str, int],
    config: Mapping[str, Any],
    batcher: AxiomBatcher,
    negative_sampler: NF3NegativeSampler,
) -> None:
    ensure_parent(path)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "classes": dict(classes),
        "relations": dict(relations),
        "config": dict(config),
        "rng_state": capture_rng_state(batcher, negative_sampler),
    }
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def copy_selected_aliases(
    best_classes: Path,
    best_relations: Path,
    best_bumps: Path,
    base_classes: Path,
    base_relations: Path,
    base_bumps: Path,
) -> None:
    atomic_copy(best_classes, base_classes)
    atomic_copy(best_relations, base_relations)
    atomic_copy(best_bumps, base_bumps)


# ---------------------------------------------------------------------------
# CSV history
# ---------------------------------------------------------------------------


HISTORY_FIELDS = [
    "epoch",
    "loss",
    "nf1",
    "nf2",
    "nf3",
    "nf4",
    "disjoint",
    "role_inclusion",
    "role_chain",
    "nf3_neg",
    "bump_reg",
    "lr",
    "grad_norm",
    "time_sec",
    "diag_true_containment",
    "diag_true_containment_train_margin",
    "diag_false_containment",
    "diag_false_containment_train_margin",
    "diag_containment_specificity",
    "diag_true_soft_containment",
    "diag_false_soft_containment",
    "diag_soft_containment_specificity",
    "diag_true_violation_mean",
    "diag_false_violation_mean",
    "diag_offset_mean",
    "diag_offset_median",
    "diag_offset_p95",
    "diag_offset_near_zero_fraction",
    "diag_center_norm_mean",
    "diag_bump_norm_mean",
    "diag_positive_pairs",
    "diag_negative_pairs",
    "selection_score",
    "is_best",
]


def initialize_history(path: str | Path, append: bool) -> None:
    ensure_parent(path)
    target = Path(path)
    if append and target.exists():
        return
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()


def append_history(path: str | Path, row: Mapping[str, Any]) -> None:
    output = {key: row.get(key, "") for key in HISTORY_FIELDS}
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writerow(output)


# ---------------------------------------------------------------------------
# Training CLI
# ---------------------------------------------------------------------------


@ck.command(context_settings={"show_default": True})
@ck.option(
    "--data-file",
    "-df",
    required=True,
    type=ck.Path(exists=True, dir_okay=False, path_type=Path),
    help="Normalized ontology axiom file.",
)
@ck.option(
    "--out-classes-file",
    "-ocf",
    default="go_boxsqel_classes.pkl",
    type=ck.Path(dir_okay=False, path_type=Path),
)
@ck.option(
    "--out-relations-file",
    "-orf",
    default="go_boxsqel_relations.pkl",
    type=ck.Path(dir_okay=False, path_type=Path),
)
@ck.option(
    "--out-bumps-file",
    default="go_boxsqel_bumps.pkl",
    type=ck.Path(dir_okay=False, path_type=Path),
)
@ck.option("--embedding-size", "-es", default=200, type=ck.IntRange(min=1))
@ck.option("--batch-size", "-bs", default=512, type=ck.IntRange(min=1))
@ck.option("--epochs", "-e", default=1000, type=ck.IntRange(min=1))
@ck.option("--learning-rate", "-lr", default=1e-3, type=ck.FloatRange(min=0.0, min_open=True))
@ck.option("--margin", "-m", default=0.0, type=ck.FloatRange(min=0.0))
@ck.option("--neg-dist", default=2.0, type=ck.FloatRange(min=0.0))
@ck.option("--reg-factor", default=0.05, type=ck.FloatRange(min=0.0))
@ck.option("--num-negatives", default=2, type=ck.IntRange(min=0))
@ck.option("--use-negatives/--no-negatives", default=True)
@ck.option("--filter-known-negatives/--allow-known-negatives", default=True)
@ck.option(
    "--negative-loss-mode",
    type=ck.Choice(["official", "hinge"], case_sensitive=False),
    default="official",
    help="official: (neg_dist-distance)^2; hinge: relu(neg_dist-distance)^2.",
)
@ck.option("--nf1-weight", default=1.0, type=ck.FloatRange(min=0.0))
@ck.option("--nf2-weight", default=1.0, type=ck.FloatRange(min=0.0))
@ck.option("--nf3-weight", default=1.0, type=ck.FloatRange(min=0.0))
@ck.option("--nf4-weight", default=1.0, type=ck.FloatRange(min=0.0))
@ck.option("--disjoint-weight", default=1.0, type=ck.FloatRange(min=0.0))
@ck.option("--role-inclusion-weight", default=1.0, type=ck.FloatRange(min=0.0))
@ck.option("--role-chain-weight", default=1.0, type=ck.FloatRange(min=0.0))
@ck.option("--negative-weight", default=1.0, type=ck.FloatRange(min=0.0))
@ck.option("--clip-norm", default=1.0, type=float)
@ck.option("--device", "-d", default="auto")
@ck.option("--seed", default=100, type=int)
@ck.option("--deterministic/--no-deterministic", default=False)
@ck.option("--num-threads", default=0, type=ck.IntRange(min=0))
@ck.option(
    "--scheduler",
    type=ck.Choice(["none", "multistep", "cosine"], case_sensitive=False),
    default="none",
)
@ck.option(
    "--lr-milestones",
    default="600,800",
    help="Comma-separated epoch milestones for the multistep scheduler.",
)
@ck.option("--lr-gamma", default=0.1, type=ck.FloatRange(min=0.0, min_open=True))
@ck.option("--diagnostic-every", default=10, type=ck.IntRange(min=1))
@ck.option("--diagnostic-pairs", default=4096, type=ck.IntRange(min=0))
@ck.option("--diagnostic-class-sample-size", default=4096, type=ck.IntRange(min=0))
@ck.option("--diagnostic-tolerance", default=1e-7, type=ck.FloatRange(min=0.0))
@ck.option(
    "--diagnostic-temperature",
    default=0.1,
    type=ck.FloatRange(min=0.0, min_open=True),
    help="Temperature for the continuous soft-containment score used in geometry-based selection.",
)
@ck.option(
    "--selection-metric",
    type=ck.Choice(["geometry", "loss"], case_sensitive=False),
    default="geometry",
    help="geometry maximizes strict true-minus-false NF1 containment; loss minimizes training loss.",
)
@ck.option("--min-delta", default=0.0, type=ck.FloatRange(min=0.0))
@ck.option(
    "--early-stopping-patience",
    default=0,
    type=ck.IntRange(min=0),
    help="Diagnostic evaluations without improvement; 0 disables early stopping.",
)
@ck.option("--save-every", default=50, type=ck.IntRange(min=0))
@ck.option("--save-init/--no-save-init", default=False)
@ck.option("--add-box-columns/--no-add-box-columns", default=False)
@ck.option("--shorten-iris/--keep-full-iris", default=True)
@ck.option(
    "--export-npy-dir",
    default=None,
    type=ck.Path(file_okay=False, path_type=Path),
    help="Also export official-repository class_embeds/bumps/rel_heads/rel_tails .npy files.",
)
@ck.option(
    "--checkpoint-file",
    default="go_boxsqel_checkpoint.pt",
    type=ck.Path(dir_okay=False, path_type=Path),
)
@ck.option(
    "--resume-from",
    default=None,
    type=ck.Path(exists=True, dir_okay=False, path_type=Path),
)
@ck.option(
    "--history-file",
    default="go_boxsqel_training_history.csv",
    type=ck.Path(dir_okay=False, path_type=Path),
)
@ck.option(
    "--manifest-file",
    default="go_boxsqel_manifest.json",
    type=ck.Path(dir_okay=False, path_type=Path),
)
@ck.option(
    "--parser-report-file",
    default="go_boxsqel_parser_report.json",
    type=ck.Path(dir_okay=False, path_type=Path),
)
@ck.option("--strict-parser/--allow-unparsed", default=False)
@ck.option("--max-unparsed-examples", default=20, type=ck.IntRange(min=0))
@ck.option("--validate-data-only/--train", default=False)
@ck.option("--log-every", default=1, type=ck.IntRange(min=1))
def main(
    data_file: Path,
    out_classes_file: Path,
    out_relations_file: Path,
    out_bumps_file: Path,
    embedding_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    margin: float,
    neg_dist: float,
    reg_factor: float,
    num_negatives: int,
    use_negatives: bool,
    filter_known_negatives: bool,
    negative_loss_mode: str,
    nf1_weight: float,
    nf2_weight: float,
    nf3_weight: float,
    nf4_weight: float,
    disjoint_weight: float,
    role_inclusion_weight: float,
    role_chain_weight: float,
    negative_weight: float,
    clip_norm: float,
    device: str,
    seed: int,
    deterministic: bool,
    num_threads: int,
    scheduler: str,
    lr_milestones: str,
    lr_gamma: float,
    diagnostic_every: int,
    diagnostic_pairs: int,
    diagnostic_class_sample_size: int,
    diagnostic_tolerance: float,
    diagnostic_temperature: float,
    selection_metric: str,
    min_delta: float,
    early_stopping_patience: int,
    save_every: int,
    save_init: bool,
    add_box_columns: bool,
    shorten_iris: bool,
    export_npy_dir: Optional[Path],
    checkpoint_file: Path,
    resume_from: Optional[Path],
    history_file: Path,
    manifest_file: Path,
    parser_report_file: Path,
    strict_parser: bool,
    max_unparsed_examples: int,
    validate_data_only: bool,
    log_every: int,
) -> None:
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    set_seed(seed, deterministic=deterministic)
    resolved_device = resolve_device(device)

    LOGGER.info("PyTorch: %s", torch.__version__)
    LOGGER.info("Device: %s", resolved_device)
    if resolved_device.type == "cuda":
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(resolved_device))

    ontology = load_normalized_ontology(
        data_file,
        strict_parser=strict_parser,
        max_unparsed_examples=max_unparsed_examples,
    )
    arrays = ontology.arrays
    class_names = invert_mapping(ontology.classes)
    relation_names = invert_mapping(ontology.relations)

    parser_payload = {
        "input_file": str(data_file),
        "report": asdict(ontology.report),
        "classes": len(class_names),
        "relations": len(relation_names),
        "axiom_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "unparsed_examples": ontology.unparsed_examples,
        "artificial_disjoint_axioms": 0,
    }
    safe_json_dump(parser_payload, parser_report_file)

    LOGGER.info(
        "Parsed ontology: %d classes, %d relations, %d parsed lines, %d unparsed lines",
        len(class_names),
        len(relation_names),
        ontology.report.parsed_lines,
        ontology.report.unparsed_lines,
    )
    for key in AXIOM_WIDTHS:
        LOGGER.info("%-16s %s", key + ":", arrays[key].shape)
    if ontology.unparsed_examples:
        LOGGER.warning("Unparsed examples (up to %d):", max_unparsed_examples)
        for example in ontology.unparsed_examples:
            LOGGER.warning("  %s", example)

    if validate_data_only:
        LOGGER.info("Data validation completed; --validate-data-only requested, so training is skipped.")
        return

    nonempty_sizes = [len(value) for value in arrays.values() if len(value) > 0]
    if not nonempty_sizes:
        raise ValueError("No trainable axioms were parsed.")
    steps_per_epoch = max(1, int(math.ceil(max(nonempty_sizes) / float(batch_size))))
    LOGGER.info("Batch size: %d | Steps per epoch: %d", batch_size, steps_per_epoch)

    effective_num_negatives = num_negatives if use_negatives else 0
    if len(arrays["nf3"]) == 0 and effective_num_negatives > 0:
        LOGGER.warning("NF3 is empty; dynamic negative sampling is disabled.")
        effective_num_negatives = 0

    batcher = AxiomBatcher(arrays, batch_size=batch_size, seed=seed + 11)
    negative_sampler = NF3NegativeSampler(
        num_classes=len(class_names),
        known_positive_nf3=arrays["nf3"],
        num_negatives=effective_num_negatives,
        seed=seed + 23,
        filter_known=filter_known_negatives,
    )

    model = BoxSquaredELModel(
        num_classes=len(class_names),
        num_relations=len(relation_names),
        embedding_dim=embedding_size,
        margin=margin,
        neg_dist=neg_dist,
        reg_factor=reg_factor,
        negative_loss_mode=negative_loss_mode.lower(),
    ).to(resolved_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_name = scheduler.lower()
    scheduler_object: Optional[torch.optim.lr_scheduler.LRScheduler]
    if scheduler_name == "none":
        scheduler_object = None
    elif scheduler_name == "multistep":
        milestones = parse_int_list(lr_milestones)
        scheduler_object = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_gamma
        )
    elif scheduler_name == "cosine":
        scheduler_object = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}")

    loss_weights = {
        "nf1": nf1_weight,
        "nf2": nf2_weight,
        "nf3": nf3_weight,
        "nf4": nf4_weight,
        "disjoint": disjoint_weight,
        "role_inclusion": role_inclusion_weight,
        "role_chain": role_chain_weight,
        "nf3_neg": negative_weight,
    }

    config: Dict[str, Any] = {
        "data_file": str(data_file),
        "embedding_size": embedding_size,
        "class_vector_width": 2 * embedding_size,
        "relation_vector_width": 4 * embedding_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "margin": margin,
        "neg_dist": neg_dist,
        "reg_factor": reg_factor,
        "num_negatives": effective_num_negatives,
        "filter_known_negatives": filter_known_negatives,
        "negative_loss_mode": negative_loss_mode.lower(),
        "loss_weights": loss_weights,
        "clip_norm": clip_norm,
        "device": str(resolved_device),
        "seed": seed,
        "deterministic": deterministic,
        "scheduler": scheduler_name,
        "lr_milestones": parse_int_list(lr_milestones),
        "lr_gamma": lr_gamma,
        "diagnostic_every": diagnostic_every,
        "diagnostic_pairs": diagnostic_pairs,
        "diagnostic_class_sample_size": diagnostic_class_sample_size,
        "diagnostic_tolerance": diagnostic_tolerance,
        "diagnostic_temperature": diagnostic_temperature,
        "selection_metric": selection_metric.lower(),
        "strict_parser": strict_parser,
        "steps_per_epoch": steps_per_epoch,
    }

    diagnostics = GeometryDiagnostics(
        nf1=arrays["nf1"],
        num_classes=len(class_names),
        sample_size=diagnostic_pairs,
        class_sample_size=diagnostic_class_sample_size,
        seed=seed + 37,
    )
    LOGGER.info(
        "Diagnostic pairs: %d true NF1 and %d matched negatives",
        len(diagnostics.pairs.positive),
        len(diagnostics.pairs.negative),
    )

    # Artifact paths.
    best_classes = tagged_path(out_classes_file, "best")
    final_classes = tagged_path(out_classes_file, "final")
    init_classes = tagged_path(out_classes_file, "init")
    best_relations = tagged_path(out_relations_file, "best")
    final_relations = tagged_path(out_relations_file, "final")
    init_relations = tagged_path(out_relations_file, "init")
    best_bumps = tagged_path(out_bumps_file, "best")
    final_bumps = tagged_path(out_bumps_file, "final")
    init_bumps = tagged_path(out_bumps_file, "init")
    best_checkpoint = tagged_path(checkpoint_file, "best")
    final_checkpoint = tagged_path(checkpoint_file, "final")

    if save_init and resume_from is None:
        save_pickle_artifacts(
            model,
            class_names,
            relation_names,
            init_classes,
            init_relations,
            init_bumps,
            shorten_iris=shorten_iris,
            add_box_columns=add_box_columns,
        )

    start_epoch = 1
    selection_name = selection_metric.lower()
    best_score = -float("inf") if selection_name == "geometry" else float("inf")
    best_epoch = 0
    diagnostics_without_improvement = 0

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=resolved_device, weights_only=False)
        if checkpoint.get("classes") != ontology.classes:
            raise ValueError("Checkpoint class mapping does not match the current ontology input.")
        if checkpoint.get("relations") != ontology.relations:
            raise ValueError("Checkpoint relation mapping does not match the current ontology input.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler_object is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler_object.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        if checkpoint.get("rng_state") is not None:
            restore_rng_state(checkpoint["rng_state"], batcher, negative_sampler)
        LOGGER.info(
            "Resumed from %s at epoch %d (best epoch %d, best score %.6f)",
            resume_from,
            start_epoch,
            best_epoch,
            best_score,
        )

    initialize_history(history_file, append=resume_from is not None)

    component_names = [
        "nf1",
        "nf2",
        "nf3",
        "nf4",
        "disjoint",
        "role_inclusion",
        "role_chain",
        "nf3_neg",
        "bump_reg",
    ]

    nan_detected = False
    stopped_early = False
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_start = time.time()
        running = {"loss": 0.0, **{name: 0.0 for name in component_names}}
        grad_norm_sum = 0.0

        for _step in range(steps_per_epoch):
            batch_np = batcher.next_batch()
            nf3_negative_np = negative_sampler.sample(batch_np["nf3"])
            batch = move_batch_to_device(batch_np, resolved_device)
            nf3_negative = torch.as_tensor(
                nf3_negative_np,
                dtype=torch.long,
                device=resolved_device,
            )

            optimizer.zero_grad(set_to_none=True)
            loss, components = model(batch, nf3_negative, loss_weights)
            if not torch.isfinite(loss):
                LOGGER.error("Non-finite loss at epoch %d. Training is stopped.", epoch)
                nan_detected = True
                break

            loss.backward()
            if clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_norm
                )
                grad_norm_sum += float(grad_norm.detach().cpu())
            optimizer.step()

            running["loss"] += float(loss.detach().cpu())
            for name in component_names:
                running[name] += float(components[name].detach().cpu())

        if nan_detected:
            break

        if scheduler_object is not None:
            scheduler_object.step()

        logs: Dict[str, Any] = {
            key: value / steps_per_epoch for key, value in running.items()
        }
        logs.update(
            epoch=epoch,
            lr=optimizer.param_groups[0]["lr"],
            grad_norm=grad_norm_sum / steps_per_epoch if clip_norm > 0 else float("nan"),
            time_sec=time.time() - epoch_start,
        )

        run_diagnostics = (
            epoch == start_epoch
            or epoch % diagnostic_every == 0
            or epoch == epochs
        )
        diagnostic_metrics: Dict[str, float] = {}
        if run_diagnostics:
            diagnostic_metrics = diagnostics.compute(
                model,
                resolved_device,
                tolerance=diagnostic_tolerance,
                temperature=diagnostic_temperature,
            )
            logs.update(diagnostic_metrics)

        if selection_name == "geometry":
            selection_score = (
                diagnostic_metrics.get("diag_soft_containment_specificity", float("nan"))
                if run_diagnostics
                else float("nan")
            )
            improved = (
                run_diagnostics
                and math.isfinite(selection_score)
                and selection_score > best_score + min_delta
            )
        else:
            selection_score = -float(logs["loss"])
            # Internally maximize -loss while keeping the manifest intuitive.
            current_loss = float(logs["loss"])
            improved = current_loss < best_score - min_delta

        logs["selection_score"] = selection_score
        logs["is_best"] = int(improved)

        if improved:
            if selection_name == "geometry":
                best_score = float(selection_score)
            else:
                best_score = float(logs["loss"])
            best_epoch = epoch
            diagnostics_without_improvement = 0

            save_pickle_artifacts(
                model,
                class_names,
                relation_names,
                best_classes,
                best_relations,
                best_bumps,
                shorten_iris=shorten_iris,
                add_box_columns=add_box_columns,
            )
            if export_npy_dir is not None:
                export_repository_npy(model, export_npy_dir, best=True)
            save_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                scheduler_object,
                epoch,
                best_score,
                best_epoch,
                ontology.classes,
                ontology.relations,
                config,
                batcher,
                negative_sampler,
            )
        elif run_diagnostics:
            diagnostics_without_improvement += 1

        append_history(history_file, logs)

        if epoch % log_every == 0:
            message = (
                f"Epoch {epoch:05d}/{epochs} | loss={logs['loss']:.6f} | "
                f"nf1={logs['nf1']:.6f} | nf2={logs['nf2']:.6f} | "
                f"nf3={logs['nf3']:.6f} | nf4={logs['nf4']:.6f} | "
                f"dis={logs['disjoint']:.6f} | ri={logs['role_inclusion']:.6f} | "
                f"rc={logs['role_chain']:.6f} | neg={logs['nf3_neg']:.6f} | "
                f"lr={logs['lr']:.3g} | time={logs['time_sec']:.2f}s"
            )
            if run_diagnostics:
                message += (
                    f" | true_cont={logs.get('diag_true_containment', float('nan')):.4f}"
                    f" | false_cont={logs.get('diag_false_containment', float('nan')):.4f}"
                    f" | specificity={logs.get('diag_containment_specificity', float('nan')):.4f}"
                    f" | soft_spec={logs.get('diag_soft_containment_specificity', float('nan')):.4f}"
                    f" | offset_med={logs.get('diag_offset_median', float('nan')):.4g}"
                )
            if improved:
                message += " | BEST"
            LOGGER.info(message)

        if save_every > 0 and epoch % save_every == 0:
            save_pickle_artifacts(
                model,
                class_names,
                relation_names,
                tagged_path(out_classes_file, f"epoch{epoch:05d}"),
                tagged_path(out_relations_file, f"epoch{epoch:05d}"),
                tagged_path(out_bumps_file, f"epoch{epoch:05d}"),
                shorten_iris=shorten_iris,
                add_box_columns=add_box_columns,
            )

        if (
            early_stopping_patience > 0
            and run_diagnostics
            and diagnostics_without_improvement >= early_stopping_patience
        ):
            LOGGER.info(
                "Early stopping at epoch %d after %d diagnostic checks without improvement.",
                epoch,
                diagnostics_without_improvement,
            )
            stopped_early = True
            break

    # Always preserve the last finite model separately.
    if last_epoch >= start_epoch and not nan_detected:
        save_pickle_artifacts(
            model,
            class_names,
            relation_names,
            final_classes,
            final_relations,
            final_bumps,
            shorten_iris=shorten_iris,
            add_box_columns=add_box_columns,
        )
        if export_npy_dir is not None:
            # No suffix is the official repository's final-model convention.
            export_repository_npy(model, export_npy_dir, best=False)
        save_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            scheduler_object,
            last_epoch,
            best_score,
            best_epoch,
            ontology.classes,
            ontology.relations,
            config,
            batcher,
            negative_sampler,
        )

    # If geometry selection had no NF1 diagnostic pairs, fall back to final.
    if best_epoch == 0 and final_classes.exists():
        LOGGER.warning(
            "No best epoch was selected (usually because NF1 diagnostics were unavailable). "
            "The final finite model is used as the selected artifact."
        )
        atomic_copy(final_classes, best_classes)
        atomic_copy(final_relations, best_relations)
        atomic_copy(final_bumps, best_bumps)
        atomic_copy(final_checkpoint, best_checkpoint)
        best_epoch = last_epoch
        best_score = float("nan")

    if best_classes.exists():
        copy_selected_aliases(
            best_classes,
            best_relations,
            best_bumps,
            out_classes_file,
            out_relations_file,
            out_bumps_file,
        )

    manifest = {
        "model": "BoxSquaredEL",
        "architecture": {
            "class_representation": "one axis-aligned box per class",
            "class_layout": "[center(d), abs(offset)(d)]",
            "class_vector_width": 2 * embedding_size,
            "relation_layout": "[head_center(d), head_abs_offset(d), tail_center(d), tail_abs_offset(d)]",
            "relation_vector_width": 4 * embedding_size,
            "bump_width": embedding_size,
        },
        "config": config,
        "parser": parser_payload,
        "training": {
            "start_epoch": start_epoch,
            "last_epoch": last_epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "selection_metric": selection_name,
            "nan_detected": nan_detected,
            "stopped_early": stopped_early,
        },
        "artifacts": {
            "selected_classes": str(out_classes_file),
            "selected_relations": str(out_relations_file),
            "selected_bumps": str(out_bumps_file),
            "best_classes": str(best_classes),
            "best_relations": str(best_relations),
            "best_bumps": str(best_bumps),
            "final_classes": str(final_classes),
            "final_relations": str(final_relations),
            "final_bumps": str(final_bumps),
            "best_checkpoint": str(best_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "history": str(history_file),
            "parser_report": str(parser_report_file),
            "npy_directory": str(export_npy_dir) if export_npy_dir is not None else None,
        },
        "evaluation_command": (
            f"python evaluate_go_embedding.py --embedding-file {out_classes_file} "
            f"--go-file <go.obo> --geometry box --layout center_offset "
            f"--embedding-dim {embedding_size} --edge-types is_a --output-dir <output_dir>"
        ),
    }
    safe_json_dump(manifest, manifest_file)

    LOGGER.info("Selected class embeddings: %s", out_classes_file)
    LOGGER.info("Best class embeddings: %s", best_classes)
    LOGGER.info("Final class embeddings: %s", final_classes)
    LOGGER.info("Manifest: %s", manifest_file)


if __name__ == "__main__":
    main()
