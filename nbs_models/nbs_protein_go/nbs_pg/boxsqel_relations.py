from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from .boxsqel_manifest import normalize_go_identifier

BOXSQEL_RELATIONS_VERSION = "0.4.2"

_GO_UNDERSCORE = re.compile(r"GO_(\d{7})", re.IGNORECASE)
_REL_ID = re.compile(r"(?:RO|BFO|GOREL|GO)[:_](\d{7})", re.IGNORECASE)
_ROLE_AXIOM = re.compile(
    r"^(?P<left>.+?)\s+"
    r"(?P<marker>SubObjectPropertyOf|SubPropertyOf)"
    r"\s*:?\s*"
    r"(?P<right>\S+)\s*$",
    re.IGNORECASE,
)
_ROLE_CHAIN_SEPARATOR = re.compile(r"\s+(?:o|and|\*)\s+", re.IGNORECASE)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_class_map(checkpoint_path: Path) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("BoxSquaredEL checkpoint lacks classes mapping")
    return {str(key): int(value) for key, value in classes.items()}


def _normalize_line(line: str) -> str:
    return _GO_UNDERSCORE.sub(lambda match: f"GO:{match.group(1)}", line.strip())


def _identifier_variants(value: str) -> list[str]:
    raw = value.strip()
    variants = [raw, raw.strip("<>"), raw.replace("_", ":")]
    go_id = normalize_go_identifier(raw)
    if go_id is not None:
        variants.extend([go_id, go_id.replace(":", "_")])
    seen: set[str] = set()
    return [item for item in variants if item and not (item in seen or seen.add(item))]


def _class_aliases(class_map: Mapping[str, int]) -> dict[str, int]:
    aliases: dict[str, int] = {}
    for value, row in class_map.items():
        for alias in _identifier_variants(value):
            previous = aliases.get(alias)
            if previous is not None and previous != int(row):
                continue
            aliases[alias] = int(row)
    return aliases


def _resolve_class(value: str, aliases: Mapping[str, int]) -> int:
    for candidate in _identifier_variants(value):
        if candidate in aliases:
            return int(aliases[candidate])
    raise KeyError(f"normalized ontology class not present in BoxSquaredEL checkpoint: {value!r}")


def normalize_relation_identifier(value: str) -> str:
    raw = value.strip().strip("<>")
    raw = raw.replace("_", ":")
    match = _REL_ID.search(raw)
    if match:
        prefix_match = re.search(r"(RO|BFO|GOREL|GO)[:_]", raw, re.IGNORECASE)
        prefix = prefix_match.group(1).upper() if prefix_match else "REL"
        return f"{prefix}:{match.group(1)}"
    return raw


def _is_part_of(relation: str) -> bool:
    normalized = normalize_relation_identifier(relation).lower()
    return normalized in {"bfo:0000050", "ro:0000050"} or "part_of" in normalized


def _split_role_axiom(line: str) -> Optional[tuple[str, str]]:
    """Split a normalized role inclusion or role-chain axiom.

    The BoxSquaredEL normalizer writes these axioms using Manchester-like
    spellings such as::

        R1 SubPropertyOf: R2
        R1 o R2 SubPropertyOf: R3

    Earlier NBS code only accepted a whitespace-delimited marker without the
    colon (``" SubPropertyOf "``), so all three role inclusions and all six
    role chains in the 512-dimensional GO normalization report were rejected.
    This parser deliberately accepts both colon and non-colon variants, and
    the longer ``SubObjectPropertyOf`` spelling, while keeping strict
    validation for the relation operands themselves.
    """

    match = _ROLE_AXIOM.fullmatch(line.strip())
    if match is None:
        return None
    return match.group("left").strip(), match.group("right").strip()


@dataclass
class BoxSquaredELNormalizedAxioms:
    nf1: list[tuple[int, int, int]]
    nf2: list[tuple[int, int, int, int]]
    nf3: list[tuple[int, int, int, int]]
    nf4: list[tuple[int, int, int, int]]
    role_inclusion: list[tuple[int, int, int]]
    role_chain: list[tuple[int, int, int, int]]
    relations: dict[str, int]
    total_lines: int
    unparsed: list[dict[str, Any]]


def parse_boxsqel_normalized_axioms(
    normalized_path: str | os.PathLike[str],
    class_map: Mapping[str, int],
    *,
    strict: bool = True,
) -> BoxSquaredELNormalizedAxioms:
    aliases = _class_aliases(class_map)
    relations: dict[str, int] = {}

    def relation_index(value: str) -> int:
        key = normalize_relation_identifier(value)
        if key not in relations:
            relations[key] = len(relations)
        return relations[key]

    nf1: list[tuple[int, int, int]] = []
    nf2: list[tuple[int, int, int, int]] = []
    nf3: list[tuple[int, int, int, int]] = []
    nf4: list[tuple[int, int, int, int]] = []
    role_inclusion: list[tuple[int, int, int]] = []
    role_chain: list[tuple[int, int, int, int]] = []
    unparsed: list[dict[str, Any]] = []
    total = 0
    with Path(normalized_path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = _normalize_line(raw_line)
            if not line:
                continue
            total += 1
            try:
                if " SubClassOf " in line:
                    left, right = line.split(" SubClassOf ", 1)
                    if " and " in left:
                        first, second = left.split(" and ", 1)
                        nf2.append(
                            (
                                _resolve_class(first, aliases),
                                _resolve_class(second, aliases),
                                _resolve_class(right, aliases),
                                line_number,
                            )
                        )
                    # BoxSquaredEL follows the standard EL normal-form naming:
                    #   NF3: C SubClassOf R some D
                    #   NF4: R some C SubClassOf D
                    #
                    # v0.4/v0.4.1 assigned these two branches in reverse.  The
                    # axioms were parsed, but their counts disagreed with the
                    # BoxSquaredEL parser report (19,117 NF3 vs 11,582 NF4).
                    elif " some " in left:
                        relation, filler = left.split(" some ", 1)
                        nf4.append(
                            (
                                relation_index(relation),
                                _resolve_class(filler, aliases),
                                _resolve_class(right, aliases),
                                line_number,
                            )
                        )
                    elif " some " in right:
                        relation, filler = right.split(" some ", 1)
                        nf3.append(
                            (
                                _resolve_class(left, aliases),
                                relation_index(relation),
                                _resolve_class(filler, aliases),
                                line_number,
                            )
                        )
                    else:
                        nf1.append(
                            (
                                _resolve_class(left, aliases),
                                _resolve_class(right, aliases),
                                line_number,
                            )
                        )
                    continue

                role_axiom = _split_role_axiom(line)
                if role_axiom is not None:
                    left, right = role_axiom
                    chain_parts = _ROLE_CHAIN_SEPARATOR.split(left.strip())
                    if any(not part.strip() for part in chain_parts):
                        raise ValueError("empty relation in normalized role axiom")
                    if len(chain_parts) == 1:
                        role_inclusion.append(
                            (relation_index(chain_parts[0]), relation_index(right), line_number)
                        )
                    elif len(chain_parts) == 2:
                        role_chain.append(
                            (
                                relation_index(chain_parts[0]),
                                relation_index(chain_parts[1]),
                                relation_index(right),
                                line_number,
                            )
                        )
                    else:
                        raise ValueError("unsupported role chain length")
                    continue
                raise ValueError("unsupported normalized axiom")
            except Exception as exc:  # preserve examples for data audit
                unparsed.append(
                    {"line_number": line_number, "line": line, "error": str(exc)}
                )
    if strict and unparsed:
        raise ValueError(
            f"failed to parse {len(unparsed)} normalized axioms; examples={unparsed[:5]}"
        )
    return BoxSquaredELNormalizedAxioms(
        nf1=nf1,
        nf2=nf2,
        nf3=nf3,
        nf4=nf4,
        role_inclusion=role_inclusion,
        role_chain=role_chain,
        relations=relations,
        total_lines=total,
        unparsed=unparsed,
    )


def build_boxsqel_gg_relations(
    normalized_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    parser_report_path: Optional[str | os.PathLike[str]] = None,
    ontology_version: str = "unknown",
    strict: bool = True,
    overwrite: bool = False,
) -> Path:
    """Build full 44,919-class G-G relations from the normalized EL source.

    The message graph uses direct NF1 subclass edges and direct NF3 ``part_of``
    edges.  All NF2/NF3/NF4 and role axioms are also retained as structured
    arrays so later NBS versions can add relation-specific or hypergraph logic
    without reparsing the ontology.
    """
    normalized = Path(normalized_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "gg_boxsqel_relations_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"BoxSquaredEL G-G output exists: {manifest_path}")
    class_map = _load_class_map(checkpoint)
    axioms = parse_boxsqel_normalized_axioms(normalized, class_map, strict=strict)

    expected_report = None
    if parser_report_path is not None:
        expected_report = json.loads(Path(parser_report_path).read_text(encoding="utf-8"))
        report = expected_report.get("report", {})
        expected_classes = expected_report.get("classes")
        if strict and expected_classes is not None and int(expected_classes) != len(class_map):
            raise ValueError(
                f"BoxSquaredEL checkpoint class count {len(class_map)} != parser report {expected_classes}"
            )
        actual = {
            "nf1": len(axioms.nf1),
            "nf2": len(axioms.nf2),
            "nf3": len(axioms.nf3),
            "nf4": len(axioms.nf4),
            "role_inclusion": len(axioms.role_inclusion),
            "role_chain": len(axioms.role_chain),
        }
        mismatches = {
            key: (int(report.get(key, -1)), value)
            for key, value in actual.items()
            if key in report and int(report[key]) != value
        }
        if strict and mismatches:
            raise ValueError(f"normalized parser counts disagree with BoxSquaredEL report: {mismatches}")
        expected_relations = expected_report.get("relations")
        if strict and expected_relations is not None and int(expected_relations) != len(axioms.relations):
            raise ValueError(
                f"normalized relation count {len(axioms.relations)} != parser report {expected_relations}"
            )
        expected_lines = report.get("parsed_lines")
        if strict and expected_lines is not None and int(expected_lines) != axioms.total_lines:
            raise ValueError(
                f"normalized parsed lines {axioms.total_lines} != parser report {expected_lines}"
            )

    nf1 = np.asarray(axioms.nf1, dtype=np.int32).reshape(-1, 3)
    nf2 = np.asarray(axioms.nf2, dtype=np.int32).reshape(-1, 4)
    nf3 = np.asarray(axioms.nf3, dtype=np.int32).reshape(-1, 4)
    nf4 = np.asarray(axioms.nf4, dtype=np.int32).reshape(-1, 4)
    role_inclusion = np.asarray(axioms.role_inclusion, dtype=np.int32).reshape(-1, 3)
    role_chain = np.asarray(axioms.role_chain, dtype=np.int32).reshape(-1, 4)

    is_a = np.column_stack(
        [nf1[:, 0], nf1[:, 1], np.ones(len(nf1), np.int32), np.ones(len(nf1), np.int32)]
    ) if len(nf1) else np.empty((0, 4), np.int32)
    has_child = is_a[:, [1, 0, 2, 3]].copy()
    relation_by_index = {index: name for name, index in axioms.relations.items()}
    # ``A SubClassOf part_of some B`` is NF3 in the BoxSquaredEL parser.
    part_rows = [row for row in nf3.tolist() if _is_part_of(relation_by_index[int(row[1])])]
    part_core = np.asarray(part_rows, dtype=np.int32).reshape(-1, 4)
    part_of = np.column_stack(
        [part_core[:, 0], part_core[:, 2], np.ones(len(part_core), np.int32), np.ones(len(part_core), np.int32)]
    ) if len(part_core) else np.empty((0, 4), np.int32)
    has_part = part_of[:, [1, 0, 2, 3]].copy()

    arrays = {
        "is_a": ("gg_boxsqel_is_a.i32.npy", is_a, ["src_go_idx", "dst_go_idx", "hop_distance", "is_direct"]),
        "has_child": ("gg_boxsqel_has_child.i32.npy", has_child, ["src_go_idx", "dst_go_idx", "hop_distance", "is_direct"]),
        "part_of": ("gg_boxsqel_part_of.i32.npy", part_of, ["src_go_idx", "dst_go_idx", "hop_distance", "is_direct"]),
        "has_part": ("gg_boxsqel_has_part.i32.npy", has_part, ["src_go_idx", "dst_go_idx", "hop_distance", "is_direct"]),
        "nf1": ("boxsqel_nf1.i32.npy", nf1, ["subclass", "superclass", "source_line"]),
        "nf2": ("boxsqel_nf2.i32.npy", nf2, ["left_1", "left_2", "superclass", "source_line"]),
        "nf3": ("boxsqel_nf3.i32.npy", nf3, ["subclass", "relation_idx", "filler", "source_line"]),
        "nf4": ("boxsqel_nf4.i32.npy", nf4, ["relation_idx", "filler", "superclass", "source_line"]),
        "role_inclusion": ("boxsqel_role_inclusion.i32.npy", role_inclusion, ["sub_role", "super_role", "source_line"]),
        "role_chain": ("boxsqel_role_chain.i32.npy", role_chain, ["left_role", "right_role", "super_role", "source_line"]),
    }
    for filename, array, _columns in arrays.values():
        np.save(out / filename, array)
    relation_registry = out / "boxsqel_relation_registry.tsv"
    with relation_registry.open("w", encoding="utf-8") as handle:
        handle.write("relation_idx\trelation_id\tis_part_of\n")
        for index in range(len(relation_by_index)):
            name = relation_by_index[index]
            handle.write(f"{index}\t{name}\t{int(_is_part_of(name))}\n")

    payload = {
        "schema_version": 1,
        "builder": f"build_boxsqel_gg_relations_v{BOXSQEL_RELATIONS_VERSION}",
        "ontology_version": str(ontology_version),
        "node_index_space": "BoxSquaredEL checkpoint class row",
        "num_classes": len(class_map),
        "num_relations": len(axioms.relations),
        "message_graph_policy": {
            "is_a": "direct NF1 only",
            "part_of": "direct NF3 part_of only",
            "closure_materialized": False,
            "other_normal_forms": "retained as structured auxiliary arrays",
        },
        "source": {
            "normalized_file": str(normalized),
            "normalized_file_sha256": _sha256(normalized),
            "boxsqel_checkpoint": str(checkpoint),
            "boxsqel_checkpoint_sha256": _sha256(checkpoint),
            "parser_report": None if parser_report_path is None else str(Path(parser_report_path).resolve()),
            "parser_report_sha256": (
                None if parser_report_path is None else _sha256(Path(parser_report_path).resolve())
            ),
        },
        "relations": {
            name: {
                "file": filename,
                "shape": list(array.shape),
                "dtype": "int32",
                "columns": columns,
            }
            for name, (filename, array, columns) in arrays.items()
        },
        "relation_registry": relation_registry.name,
        "unparsed_lines": len(axioms.unparsed),
        "unparsed_examples": axioms.unparsed[:20],
        "future_go_interface": {
            "rebuild_from": ["future go.obo -> go.norm", "future BoxSquaredEL checkpoint"],
            "stable_contract": "class-row registry + normalized axiom arrays + task mapping",
            "task_classifier_indices_are_not_reordered": True,
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path
