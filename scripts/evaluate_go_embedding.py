#!/usr/bin/env python3
"""
Evaluate geometric Gene Ontology embeddings without retraining a downstream model.

The evaluator focuses on three questions:

1. Geometry health
   - numerical validity, collapse, effective rank, distance concentration, hubness
2. GO hierarchy fidelity
   - size-depth association, parent/ancestor containment, matched false containment
3. GO neighborhood quality
   - parent/ancestor retrieval, sibling separation, kNN graph-neighborhood preservation

Supported embedding geometries
------------------------------
- sphere: class = center + scalar radius
- box:    class = center + per-dimension half-width/offset
- vector: class = ordinary point vector

Expected input
--------------
The embedding file is normally a pandas pickle produced by the supplied
ELEmbeddings/ELBox scripts. At minimum it should contain an ID column such as
"classes" and an array-valued column such as "embeddings".

Typical examples:

Sphere stored as one vector of length d+1:
    classes | embeddings
    GO_...  | [center_1, ..., center_d, radius]

Sphere stored in separate columns:
    classes | center | radius

Box stored as one vector of length 2d:
    classes | embeddings
    GO_...  | [center_1, ..., center_d, offset_1, ..., offset_d]

Box stored in separate columns:
    classes | center | offset

Dependencies
------------
    numpy pandas scipy scikit-learn networkx matplotlib

Example
-------
python evaluate_go_embedding.py \
    --embedding-file go_elem_classes.pkl \
    --go-file go-basic.obo \
    --geometry sphere \
    --layout center_radius \
    --embedding-dim 50 \
    --edge-types is_a \
    --output-dir results/go_elem

For ELBox/ELBE, verify how the exported vector is laid out. If it is
[center, offset], use:

python evaluate_go_embedding.py \
    --embedding-file go_elbox_classes.pkl \
    --go-file go-basic.obo \
    --geometry box \
    --layout center_offset \
    --embedding-dim 50 \
    --output-dir results/go_elbox

Notes
-----
- The default hierarchy relation is only "is_a".
- "part_of" can be included with --edge-types is_a,part_of, but its results
  should normally be reported separately from pure is_a results.
- The evaluation uses the current GO graph as a structural reference. It is a
  representation diagnostic, not a held-out logical generalization test.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import random
import re
import sys
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("go_embedding_evaluator")
EPS = 1e-12


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(data: Mapping[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(jsonable(dict(data)), handle, indent=2, ensure_ascii=False)


def percentile_summary(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_p01": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p25": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p75": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_p99": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    q = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(q[0]),
        f"{prefix}_p01": float(q[1]),
        f"{prefix}_p05": float(q[2]),
        f"{prefix}_p25": float(q[3]),
        f"{prefix}_median": float(q[4]),
        f"{prefix}_p75": float(q[5]),
        f"{prefix}_p95": float(q[6]),
        f"{prefix}_p99": float(q[7]),
        f"{prefix}_max": float(q[8]),
    }


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, int]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    n = int(mask.sum())
    if n < 3 or np.unique(x_arr[mask]).size < 2 or np.unique(y_arr[mask]).size < 2:
        return float("nan"), float("nan"), n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat = spearmanr(x_arr[mask], y_arr[mask])
    return float(stat.statistic), float(stat.pvalue), n


def gini_coefficient(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.min(x) < 0:
        x = x - np.min(x)
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x + EPS)
    n = x.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2 * index - n - 1) * x) / (n * np.sum(x)))


def canonical_go_id(value: Any) -> str:
    """Convert common GO ID/IRI variants to GO:0000000 form."""
    text = str(value).strip()
    match = re.search(r"GO[_:](\d{1,7})", text, flags=re.IGNORECASE)
    if match:
        return f"GO:{int(match.group(1)):07d}"
    return text


def parse_array_value(value: Any) -> np.ndarray:
    """Parse an in-memory array or a CSV string representation of one."""
    if isinstance(value, str):
        text = value.strip()
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Also accept whitespace-separated NumPy-style strings such as
            # "[0.1 0.2 0.3]".
            stripped = text.strip("[]()")
            value = np.fromstring(stripped.replace(",", " "), sep=" ")
    return np.asarray(value, dtype=np.float64).reshape(-1)


def stack_array_column(series: pd.Series, column_name: str) -> np.ndarray:
    rows: List[np.ndarray] = []
    expected: Optional[int] = None
    for row_idx, value in enumerate(series):
        arr = parse_array_value(value)
        if expected is None:
            expected = int(arr.size)
        elif arr.size != expected:
            raise ValueError(
                f"Column '{column_name}' has inconsistent vector lengths: "
                f"row 0 has {expected}, row {row_idx} has {arr.size}."
            )
        rows.append(arr)
    if not rows:
        raise ValueError(f"Column '{column_name}' is empty.")
    return np.stack(rows, axis=0)


def transform_size(raw: np.ndarray, mode: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if mode == "abs":
        return np.abs(raw)
    if mode == "softplus":
        # Stable softplus.
        return np.log1p(np.exp(-np.abs(raw))) + np.maximum(raw, 0)
    if mode == "none":
        if np.any(raw < 0):
            LOGGER.warning(
                "Negative radius/offset values were retained because --size-transform none was used."
            )
        return raw
    raise ValueError(f"Unknown size transform: {mode}")


# ---------------------------------------------------------------------------
# GO OBO parsing and graph preparation
# ---------------------------------------------------------------------------


@dataclass
class GOTerm:
    term_id: str
    name: str = ""
    namespace: str = "unknown"
    is_obsolete: bool = False
    alt_ids: Tuple[str, ...] = ()
    parents: Dict[str, Tuple[str, ...]] | None = None


@dataclass
class OntologyData:
    terms: Dict[str, GOTerm]
    alt_to_primary: Dict[str, str]
    parent_map: Dict[str, Set[str]]
    child_map: Dict[str, Set[str]]
    graph_child_parent: nx.DiGraph
    graph_parent_child: nx.DiGraph
    graph_undirected: nx.Graph
    min_depth: Dict[str, int]
    max_depth: Dict[str, int]
    roots: List[str]
    selected_edge_types: Tuple[str, ...]


def parse_obo(path: str | Path) -> Tuple[Dict[str, GOTerm], Dict[str, str]]:
    """Minimal GO OBO parser sufficient for hierarchy diagnostics."""
    terms: Dict[str, GOTerm] = {}
    alt_to_primary: Dict[str, str] = {}

    current: Optional[MutableMapping[str, Any]] = None

    def flush(record: Optional[MutableMapping[str, Any]]) -> None:
        if not record or "id" not in record:
            return
        term_id = canonical_go_id(record["id"])
        parents = {
            relation: tuple(canonical_go_id(v) for v in values)
            for relation, values in record.get("parents", {}).items()
        }
        alt_ids = tuple(canonical_go_id(x) for x in record.get("alt_ids", []))
        term = GOTerm(
            term_id=term_id,
            name=str(record.get("name", "")),
            namespace=str(record.get("namespace", "unknown")),
            is_obsolete=bool(record.get("is_obsolete", False)),
            alt_ids=alt_ids,
            parents=parents,
        )
        terms[term_id] = term
        for alt_id in alt_ids:
            alt_to_primary[alt_id] = term_id

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line == "[Term]":
                flush(current)
                current = {"parents": defaultdict(list), "alt_ids": []}
                continue
            if line.startswith("["):
                flush(current)
                current = None
                continue
            if current is None or not line or line.startswith("!"):
                continue

            if line.startswith("id:"):
                current["id"] = line.split("id:", 1)[1].strip()
            elif line.startswith("name:"):
                current["name"] = line.split("name:", 1)[1].strip()
            elif line.startswith("namespace:"):
                current["namespace"] = line.split("namespace:", 1)[1].strip()
            elif line.startswith("alt_id:"):
                current["alt_ids"].append(line.split("alt_id:", 1)[1].strip())
            elif line.startswith("is_obsolete:"):
                current["is_obsolete"] = line.split("is_obsolete:", 1)[1].strip().lower() == "true"
            elif line.startswith("is_a:"):
                parent = line.split("is_a:", 1)[1].split("!", 1)[0].strip()
                current["parents"]["is_a"].append(parent)
            elif line.startswith("relationship:"):
                payload = line.split("relationship:", 1)[1].split("!", 1)[0].strip()
                parts = payload.split()
                if len(parts) >= 2:
                    relation, target = parts[0], parts[1]
                    current["parents"][relation].append(target)

    flush(current)
    return terms, alt_to_primary


def build_ontology(path: str | Path, edge_types: Sequence[str]) -> OntologyData:
    terms, alt_to_primary = parse_obo(path)
    selected = tuple(x.strip() for x in edge_types if x.strip())
    if not selected:
        raise ValueError("At least one edge type must be selected.")

    active_terms = {k: v for k, v in terms.items() if not v.is_obsolete}
    parent_map: Dict[str, Set[str]] = {term_id: set() for term_id in active_terms}
    child_map: Dict[str, Set[str]] = {term_id: set() for term_id in active_terms}

    graph_cp = nx.DiGraph()
    graph_cp.add_nodes_from(active_terms)

    for term_id, term in active_terms.items():
        assert term.parents is not None
        for relation in selected:
            for raw_parent in term.parents.get(relation, ()):
                parent = alt_to_primary.get(raw_parent, raw_parent)
                if parent not in active_terms or parent == term_id:
                    continue
                parent_map[term_id].add(parent)
                child_map[parent].add(term_id)
                graph_cp.add_edge(term_id, parent, relation=relation)

    graph_pc = graph_cp.reverse(copy=True)
    graph_u = graph_cp.to_undirected(as_view=False)
    roots = sorted(node for node in graph_cp.nodes if graph_cp.out_degree(node) == 0)

    # Minimum depth: breadth-first search from all roots in parent->child direction.
    min_depth: Dict[str, int] = {}
    queue: deque[Tuple[str, int]] = deque((root, 0) for root in roots)
    while queue:
        node, depth = queue.popleft()
        old = min_depth.get(node)
        if old is not None and old <= depth:
            continue
        min_depth[node] = depth
        for child in child_map.get(node, ()):
            queue.append((child, depth + 1))

    # Longest root-to-node depth when the selected graph is acyclic.
    if nx.is_directed_acyclic_graph(graph_pc):
        max_depth: Dict[str, int] = {node: 0 for node in graph_pc.nodes}
        for node in nx.topological_sort(graph_pc):
            base = max_depth.get(node, 0)
            for child in graph_pc.successors(node):
                max_depth[child] = max(max_depth.get(child, 0), base + 1)
    else:
        LOGGER.warning(
            "The selected relation graph contains cycles. max_depth falls back to min_depth. "
            "For strict hierarchy diagnostics, prefer --edge-types is_a."
        )
        max_depth = dict(min_depth)

    for node in graph_cp.nodes:
        min_depth.setdefault(node, -1)
        max_depth.setdefault(node, min_depth[node])

    LOGGER.info(
        "Loaded GO: %d active terms, %d selected edges, %d roots, edge types=%s",
        len(active_terms),
        graph_cp.number_of_edges(),
        len(roots),
        ",".join(selected),
    )

    return OntologyData(
        terms=active_terms,
        alt_to_primary=alt_to_primary,
        parent_map=parent_map,
        child_map=child_map,
        graph_child_parent=graph_cp,
        graph_parent_child=graph_pc,
        graph_undirected=graph_u,
        min_depth=min_depth,
        max_depth=max_depth,
        roots=roots,
        selected_edge_types=selected,
    )


# ---------------------------------------------------------------------------
# Embedding loading and geometric adapters
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingData:
    ids: List[str]
    centers: np.ndarray
    sizes: Optional[np.ndarray]
    geometry: str
    layout: str
    source_rows: int
    dropped_duplicate_ids: int
    metadata: Dict[str, Any]


@dataclass
class EvaluationView:
    name: str
    centers: np.ndarray
    sizes: Optional[np.ndarray]
    geometry: str


def detect_column(df: pd.DataFrame, explicit: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested column '{explicit}' is absent. Available columns: {list(df.columns)}")
        return explicit
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def load_embedding_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported embedding file extension '{suffix}'. Use .pkl/.pickle, .csv, or .parquet."
        )
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, found {type(df)!r}.")
    return df


def load_embeddings(
    path: str | Path,
    geometry: str,
    layout: str,
    embedding_dim: Optional[int],
    id_column: Optional[str],
    embedding_column: Optional[str],
    center_column: Optional[str],
    radius_column: Optional[str],
    offset_column: Optional[str],
    size_transform: str,
) -> EmbeddingData:
    df = load_embedding_table(path)
    source_rows = len(df)

    id_col = detect_column(df, id_column, ["classes", "go_id", "id", "term", "terms"])
    if id_col is None:
        raise ValueError(
            "Could not identify the GO ID column. Use --id-column. "
            f"Available columns: {list(df.columns)}"
        )

    emb_col = detect_column(df, embedding_column, ["embeddings", "embedding", "vector", "vectors"])
    center_col = detect_column(df, center_column, ["center", "centers"])
    radius_col = detect_column(df, radius_column, ["radius", "radii"])
    offset_col = detect_column(df, offset_column, ["offset", "offsets", "half_width", "half_widths"])

    raw: Optional[np.ndarray] = None
    centers_from_col: Optional[np.ndarray] = None
    radius_from_col: Optional[np.ndarray] = None
    offset_from_col: Optional[np.ndarray] = None

    if emb_col is not None:
        raw = stack_array_column(df[emb_col], emb_col)
    if center_col is not None:
        centers_from_col = stack_array_column(df[center_col], center_col)
    if radius_col is not None:
        radius_values = []
        for value in df[radius_col]:
            arr = parse_array_value(value)
            if arr.size != 1:
                raise ValueError(f"Radius column '{radius_col}' must contain scalars.")
            radius_values.append(float(arr[0]))
        radius_from_col = np.asarray(radius_values, dtype=np.float64)
    if offset_col is not None:
        offset_from_col = stack_array_column(df[offset_col], offset_col)

    # Infer geometry/layout conservatively.
    inferred_geometry = geometry
    inferred_layout = layout

    if inferred_geometry == "auto":
        if radius_from_col is not None or inferred_layout == "center_radius":
            inferred_geometry = "sphere"
        elif offset_from_col is not None or inferred_layout in {"center_offset", "min_max"}:
            inferred_geometry = "box"
        elif raw is not None and embedding_dim is not None and raw.shape[1] == embedding_dim + 1:
            inferred_geometry = "sphere"
            inferred_layout = "center_radius"
        elif raw is not None and embedding_dim is not None and raw.shape[1] == 2 * embedding_dim:
            inferred_geometry = "box"
            inferred_layout = "center_offset"
        else:
            inferred_geometry = "vector"

    if inferred_layout == "auto":
        if inferred_geometry == "sphere":
            inferred_layout = "separate" if radius_from_col is not None else "center_radius"
        elif inferred_geometry == "box":
            inferred_layout = "separate" if offset_from_col is not None else "center_offset"
        else:
            inferred_layout = "vector"

    centers: np.ndarray
    sizes: Optional[np.ndarray]

    if inferred_geometry == "vector":
        if centers_from_col is not None:
            centers = centers_from_col
        elif raw is not None:
            centers = raw
        else:
            raise ValueError("Vector geometry requires an embedding or center column.")
        sizes = None

    elif inferred_geometry == "sphere":
        if radius_from_col is not None:
            if centers_from_col is not None:
                centers = centers_from_col
            elif raw is not None:
                centers = raw
            else:
                raise ValueError("Sphere geometry with a radius column also needs center/embedding values.")
            radii = radius_from_col
        else:
            if raw is None:
                raise ValueError("Sphere center_radius layout requires an embedding vector column.")
            d = embedding_dim if embedding_dim is not None else raw.shape[1] - 1
            if d <= 0 or raw.shape[1] != d + 1:
                raise ValueError(
                    f"Sphere vector has width {raw.shape[1]}, but center_radius layout requires d+1. "
                    "Specify the correct --embedding-dim or use separate center/radius columns."
                )
            centers = raw[:, :d]
            radii = raw[:, d]
        radii = transform_size(np.asarray(radii, dtype=np.float64), size_transform)
        sizes = radii.reshape(-1, 1)

    elif inferred_geometry == "box":
        if offset_from_col is not None:
            if centers_from_col is not None:
                centers = centers_from_col
            elif raw is not None:
                centers = raw
            else:
                raise ValueError("Box geometry with an offset column also needs center/embedding values.")
            offsets = offset_from_col
        else:
            if raw is None:
                raise ValueError("Box geometry requires an embedding vector column or separate center/offset columns.")
            d = embedding_dim if embedding_dim is not None else raw.shape[1] // 2
            if d <= 0 or raw.shape[1] != 2 * d:
                raise ValueError(
                    f"Box vector has width {raw.shape[1]}, but the selected layout requires 2d. "
                    "Specify the correct --embedding-dim."
                )
            first = raw[:, :d]
            second = raw[:, d:]
            if inferred_layout == "center_offset":
                centers = first
                offsets = second
            elif inferred_layout == "min_max":
                lower = np.minimum(first, second)
                upper = np.maximum(first, second)
                centers = (lower + upper) / 2.0
                offsets = (upper - lower) / 2.0
            else:
                raise ValueError(
                    f"Unsupported box layout '{inferred_layout}'. Use center_offset, min_max, or separate."
                )
        offsets = transform_size(np.asarray(offsets, dtype=np.float64), size_transform)
        if centers.shape != offsets.shape:
            raise ValueError(
                f"Box centers shape {centers.shape} does not match offsets shape {offsets.shape}."
            )
        sizes = offsets

    else:
        raise ValueError(f"Unsupported geometry '{inferred_geometry}'.")

    ids = [canonical_go_id(x) for x in df[id_col].tolist()]
    if centers.shape[0] != len(ids):
        raise ValueError("Embedding row count does not match ID count.")

    # Keep the first occurrence of each canonicalized ID.
    keep: List[int] = []
    seen: Set[str] = set()
    for idx, term_id in enumerate(ids):
        if term_id in seen:
            continue
        seen.add(term_id)
        keep.append(idx)
    dropped = len(ids) - len(keep)
    if dropped:
        LOGGER.warning("Dropped %d duplicate GO IDs after canonicalization.", dropped)

    ids = [ids[i] for i in keep]
    centers = np.asarray(centers[keep], dtype=np.float64)
    if sizes is not None:
        sizes = np.asarray(sizes[keep], dtype=np.float64)

    metadata = {
        "id_column": id_col,
        "embedding_column": emb_col,
        "center_column": center_col,
        "radius_column": radius_col,
        "offset_column": offset_col,
        "raw_columns": list(df.columns),
    }

    LOGGER.info(
        "Loaded embeddings: %d unique IDs, geometry=%s, layout=%s, center_dim=%d",
        len(ids), inferred_geometry, inferred_layout, centers.shape[1],
    )

    return EmbeddingData(
        ids=ids,
        centers=centers,
        sizes=sizes,
        geometry=inferred_geometry,
        layout=inferred_layout,
        source_rows=source_rows,
        dropped_duplicate_ids=dropped,
        metadata=metadata,
    )


def align_embeddings_to_go(
    embedding: EmbeddingData,
    ontology: OntologyData,
) -> Tuple[EmbeddingData, List[str]]:
    mapped_ids: List[str] = []
    keep: List[int] = []
    missing: List[str] = []
    seen: Set[str] = set()

    for idx, raw_id in enumerate(embedding.ids):
        term_id = ontology.alt_to_primary.get(raw_id, raw_id)
        if term_id not in ontology.terms:
            missing.append(raw_id)
            continue
        if term_id in seen:
            continue
        seen.add(term_id)
        mapped_ids.append(term_id)
        keep.append(idx)

    if not keep:
        raise ValueError("No embedding IDs could be matched to active GO terms.")

    aligned = EmbeddingData(
        ids=mapped_ids,
        centers=embedding.centers[keep],
        sizes=embedding.sizes[keep] if embedding.sizes is not None else None,
        geometry=embedding.geometry,
        layout=embedding.layout,
        source_rows=embedding.source_rows,
        dropped_duplicate_ids=embedding.dropped_duplicate_ids + (len(embedding.ids) - len(keep) - len(missing)),
        metadata=dict(embedding.metadata),
    )

    LOGGER.info(
        "Aligned embeddings to GO: %d matched terms; %d unmatched embedding IDs.",
        len(mapped_ids), len(missing),
    )
    return aligned, missing


def make_evaluation_views(
    embedding: EmbeddingData,
    requested_modes: Sequence[str],
    rng: np.random.Generator,
) -> Dict[str, EvaluationView]:
    views: Dict[str, EvaluationView] = {}
    n = len(embedding.ids)

    for mode in requested_modes:
        mode = mode.strip()
        if not mode:
            continue
        if mode == "full":
            views[mode] = EvaluationView(
                name=mode,
                centers=embedding.centers,
                sizes=embedding.sizes,
                geometry=embedding.geometry,
            )
        elif mode == "center":
            views[mode] = EvaluationView(
                name=mode,
                centers=embedding.centers,
                sizes=None,
                geometry="vector",
            )
        elif mode == "size_shuffled":
            if embedding.sizes is None:
                LOGGER.warning("Skipping size_shuffled: the selected geometry has no size parameters.")
                continue
            perm = rng.permutation(n)
            views[mode] = EvaluationView(
                name=mode,
                centers=embedding.centers,
                sizes=embedding.sizes[perm],
                geometry=embedding.geometry,
            )
        elif mode == "node_shuffled":
            perm = rng.permutation(n)
            views[mode] = EvaluationView(
                name=mode,
                centers=embedding.centers[perm],
                sizes=embedding.sizes[perm] if embedding.sizes is not None else None,
                geometry=embedding.geometry,
            )
        else:
            raise ValueError(
                f"Unknown ablation '{mode}'. Choose from full,center,size_shuffled,node_shuffled."
            )

    if "full" not in views:
        views["full"] = EvaluationView(
            name="full",
            centers=embedding.centers,
            sizes=embedding.sizes,
            geometry=embedding.geometry,
        )
    return views


def euclidean_distance_one_to_many(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    diff = candidates - query[None, :]
    return np.sqrt(np.maximum(np.einsum("ij,ij->i", diff, diff), 0.0))


def asymmetric_scores(view: EvaluationView, query_idx: int, candidate_idx: np.ndarray) -> np.ndarray:
    """Higher score means a better candidate parent/ancestor."""
    q = view.centers[query_idx]
    c = view.centers[candidate_idx]

    if view.geometry == "vector" or view.sizes is None:
        return -euclidean_distance_one_to_many(q, c)

    if view.geometry == "sphere":
        q_radius = float(view.sizes[query_idx, 0])
        parent_radius = view.sizes[candidate_idx, 0]
        return parent_radius - q_radius - euclidean_distance_one_to_many(q, c)

    if view.geometry == "box":
        q_offset = view.sizes[query_idx]
        parent_offset = view.sizes[candidate_idx]
        per_dim_margin = parent_offset - q_offset[None, :] - np.abs(c - q[None, :])
        return np.min(per_dim_margin, axis=1)

    raise ValueError(f"Unsupported geometry: {view.geometry}")


def containment_margin_pairs(
    view: EvaluationView,
    child_idx: np.ndarray,
    parent_idx: np.ndarray,
) -> np.ndarray:
    child = view.centers[child_idx]
    parent = view.centers[parent_idx]

    if view.geometry == "vector" or view.sizes is None:
        diff = parent - child
        return -np.sqrt(np.maximum(np.einsum("ij,ij->i", diff, diff), 0.0))

    if view.geometry == "sphere":
        diff = parent - child
        distance = np.sqrt(np.maximum(np.einsum("ij,ij->i", diff, diff), 0.0))
        return view.sizes[parent_idx, 0] - view.sizes[child_idx, 0] - distance

    if view.geometry == "box":
        margins = (
            view.sizes[parent_idx]
            - view.sizes[child_idx]
            - np.abs(parent - child)
        )
        return np.min(margins, axis=1)

    raise ValueError(f"Unsupported geometry: {view.geometry}")


def symmetric_distances_pairs(
    view: EvaluationView,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
) -> np.ndarray:
    left = view.centers[left_idx]
    right = view.centers[right_idx]
    diff = right - left

    if view.geometry == "vector" or view.sizes is None:
        return np.sqrt(np.maximum(np.einsum("ij,ij->i", diff, diff), 0.0))

    if view.geometry == "sphere":
        center_distance = np.sqrt(np.maximum(np.einsum("ij,ij->i", diff, diff), 0.0))
        gap = center_distance - view.sizes[left_idx, 0] - view.sizes[right_idx, 0]
        return np.maximum(gap, 0.0)

    if view.geometry == "box":
        gap = np.maximum(
            np.abs(diff) - view.sizes[left_idx] - view.sizes[right_idx],
            0.0,
        )
        return np.sqrt(np.maximum(np.einsum("ij,ij->i", gap, gap), 0.0))

    raise ValueError(f"Unsupported geometry: {view.geometry}")


def symmetric_distances_one_to_many(
    view: EvaluationView,
    query_idx: int,
    candidate_idx: np.ndarray,
) -> np.ndarray:
    q = np.full(candidate_idx.shape[0], query_idx, dtype=np.int64)
    return symmetric_distances_pairs(view, q, candidate_idx)


def size_scalar(embedding: EmbeddingData) -> Optional[np.ndarray]:
    if embedding.sizes is None:
        return None
    if embedding.geometry == "sphere":
        return np.log(np.maximum(embedding.sizes[:, 0], EPS))
    if embedding.geometry == "box":
        return np.sum(np.log(np.maximum(2.0 * embedding.sizes, EPS)), axis=1)
    return None


# ---------------------------------------------------------------------------
# Geometry health
# ---------------------------------------------------------------------------


@dataclass
class GeometryHealthArtifacts:
    eigenvalues: np.ndarray
    sampled_pair_distances: np.ndarray
    query_indices: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray


def evaluate_geometry_health(
    embedding: EmbeddingData,
    namespaces: Sequence[str],
    rng: np.random.Generator,
    pair_sample_size: int,
    query_sample_size: int,
    knn_k: int,
    pca_sample_size: int,
) -> Tuple[Dict[str, Any], GeometryHealthArtifacts]:
    centers = np.asarray(embedding.centers, dtype=np.float64)
    n, d = centers.shape

    finite_mask = np.isfinite(centers)
    nan_count = int(np.isnan(centers).sum())
    inf_count = int(np.isinf(centers).sum())
    if embedding.sizes is not None:
        nan_count += int(np.isnan(embedding.sizes).sum())
        inf_count += int(np.isinf(embedding.sizes).sum())

    # Geometry calculations require finite values. Fail early rather than silently
    # replacing malformed values.
    if not np.all(finite_mask) or (embedding.sizes is not None and not np.all(np.isfinite(embedding.sizes))):
        raise ValueError(
            f"Embedding contains non-finite values: NaN={nan_count}, Inf={inf_count}. "
            "Fix the embedding before structural evaluation."
        )

    variances = np.var(centers, axis=0)
    near_zero_variance_dims = int(np.sum(variances <= 1e-12))
    norms = np.linalg.norm(centers, axis=1)

    rounded = np.round(centers, decimals=8)
    unique_rows = np.unique(rounded, axis=0).shape[0]
    duplicate_rate = 1.0 - unique_rows / max(n, 1)

    # PCA/effective rank on a bounded sample for speed.
    pca_n = min(n, max(2, pca_sample_size))
    pca_idx = rng.choice(n, size=pca_n, replace=False) if pca_n < n else np.arange(n)
    x = centers[pca_idx] - np.mean(centers[pca_idx], axis=0, keepdims=True)
    singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    eigenvalues = (singular ** 2) / max(x.shape[0] - 1, 1)
    total_var = float(np.sum(eigenvalues))
    if total_var > 0:
        effective_rank = float(total_var ** 2 / np.sum(eigenvalues ** 2))
        explained = eigenvalues / total_var
    else:
        effective_rank = 0.0
        explained = np.zeros_like(eigenvalues)

    # Random pair distances.
    pair_n = min(max(pair_sample_size, 1), max(n * 20, 1_000)) if n > 1 else 0
    if pair_n:
        left = rng.integers(0, n, size=pair_n)
        right = rng.integers(0, n, size=pair_n)
        equal = left == right
        while np.any(equal):
            right[equal] = rng.integers(0, n, size=int(equal.sum()))
            equal = left == right
        diff = centers[left] - centers[right]
        pair_distances = np.sqrt(np.maximum(np.einsum("ij,ij->i", diff, diff), 0.0))
    else:
        pair_distances = np.array([], dtype=np.float64)

    # Estimated nearest-neighbor and hubness statistics using sampled queries.
    qn = min(query_sample_size, n)
    query_idx = rng.choice(n, size=qn, replace=False) if qn < n else np.arange(n)
    actual_k = min(knn_k, max(n - 1, 1))
    model = NearestNeighbors(n_neighbors=min(actual_k + 1, n), metric="euclidean", n_jobs=-1)
    model.fit(centers)
    raw_dist, raw_ind = model.kneighbors(centers[query_idx], return_distance=True)

    neighbor_ind = np.empty((qn, actual_k), dtype=np.int64)
    neighbor_dist = np.empty((qn, actual_k), dtype=np.float64)
    for row, q_idx in enumerate(query_idx):
        mask = raw_ind[row] != q_idx
        inds = raw_ind[row][mask][:actual_k]
        dists = raw_dist[row][mask][:actual_k]
        if inds.size < actual_k:
            # This only occurs for tiny datasets.
            pad_n = actual_k - inds.size
            inds = np.pad(inds, (0, pad_n), constant_values=q_idx)
            dists = np.pad(dists, (0, pad_n), constant_values=np.nan)
        neighbor_ind[row] = inds
        neighbor_dist[row] = dists

    hub_counts = np.bincount(neighbor_ind.reshape(-1), minlength=n)
    nonzero_hub = hub_counts[hub_counts > 0]

    metrics: Dict[str, Any] = {
        "n_terms": n,
        "center_dim": d,
        "geometry": embedding.geometry,
        "layout": embedding.layout,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duplicate_vector_rate_round8": float(duplicate_rate),
        "near_zero_variance_dims": near_zero_variance_dims,
        "near_zero_variance_dim_ratio": float(near_zero_variance_dims / max(d, 1)),
        "effective_rank": effective_rank,
        "effective_rank_ratio": float(effective_rank / max(d, 1)),
        "pca_explained_top1": float(np.sum(explained[:1])),
        "pca_explained_top5": float(np.sum(explained[:5])),
        "pca_explained_top10": float(np.sum(explained[:10])),
        "distance_cv": float(np.std(pair_distances) / max(np.mean(pair_distances), EPS)) if pair_distances.size else float("nan"),
        "nearest_random_distance_ratio": float(
            np.nanmean(neighbor_dist[:, 0]) / max(np.mean(pair_distances), EPS)
        ) if pair_distances.size and neighbor_dist.size else float("nan"),
        "hubness_gini": gini_coefficient(hub_counts),
        "hubness_max_count": int(np.max(hub_counts)) if hub_counts.size else 0,
        "hubness_top1pct_share": float(
            np.sort(hub_counts)[-max(1, int(math.ceil(0.01 * n))):].sum() / max(hub_counts.sum(), 1)
        ),
        "hubness_nonzero_nodes": int(nonzero_hub.size),
    }
    metrics.update(percentile_summary(norms, "center_norm"))
    metrics.update(percentile_summary(pair_distances, "random_pair_distance"))
    metrics.update(percentile_summary(neighbor_dist[:, 0], "nearest_neighbor_distance"))

    if embedding.sizes is not None:
        if embedding.geometry == "sphere":
            radii = embedding.sizes[:, 0]
            metrics.update(percentile_summary(radii, "radius"))
            metrics["near_zero_radius_rate"] = float(np.mean(radii <= 1e-8))
        elif embedding.geometry == "box":
            offsets = embedding.sizes
            log_volume = np.sum(np.log(np.maximum(2.0 * offsets, EPS)), axis=1)
            metrics.update(percentile_summary(offsets.reshape(-1), "box_offset"))
            metrics.update(percentile_summary(log_volume, "box_log_volume"))
            metrics["near_zero_offset_rate"] = float(np.mean(offsets <= 1e-8))
            metrics["degenerate_box_rate"] = float(np.mean(np.any(offsets <= 1e-8, axis=1)))

    # Namespace purity among center-space nearest neighbors.
    ns = np.asarray(list(namespaces), dtype=object)
    if ns.size == n and neighbor_ind.size:
        same = ns[neighbor_ind] == ns[query_idx, None]
        metrics[f"center_knn_namespace_purity_at_{actual_k}"] = float(np.mean(same))

    artifacts = GeometryHealthArtifacts(
        eigenvalues=eigenvalues,
        sampled_pair_distances=pair_distances,
        query_indices=query_idx,
        neighbor_indices=neighbor_ind,
        neighbor_distances=neighbor_dist,
    )
    return metrics, artifacts


# ---------------------------------------------------------------------------
# Hierarchy metadata and size diagnostics
# ---------------------------------------------------------------------------


def sample_descendant_counts(
    ontology: OntologyData,
    term_ids: Sequence[str],
    rng: np.random.Generator,
    sample_size: int,
) -> Dict[str, int]:
    if sample_size <= 0:
        return {}
    n = min(sample_size, len(term_ids))
    sampled = rng.choice(np.asarray(term_ids, dtype=object), size=n, replace=False)
    counts: Dict[str, int] = {}
    for pos, term_id in enumerate(sampled, start=1):
        # In child->parent orientation, graph ancestors are ontology descendants.
        counts[str(term_id)] = len(nx.ancestors(ontology.graph_child_parent, str(term_id)))
        if pos % 1000 == 0:
            LOGGER.info("Computed exact descendant counts for %d/%d sampled terms.", pos, n)
    return counts


def build_term_table(
    embedding: EmbeddingData,
    ontology: OntologyData,
    descendant_counts: Mapping[str, int],
) -> pd.DataFrame:
    scalar = size_scalar(embedding)
    rows: List[Dict[str, Any]] = []
    for idx, term_id in enumerate(embedding.ids):
        term = ontology.terms[term_id]
        rows.append({
            "embedding_index": idx,
            "go_id": term_id,
            "name": term.name,
            "namespace": term.namespace,
            "min_depth": ontology.min_depth.get(term_id, -1),
            "max_depth": ontology.max_depth.get(term_id, -1),
            "direct_parent_count": len(ontology.parent_map.get(term_id, ())),
            "direct_child_count": len(ontology.child_map.get(term_id, ())),
            "is_leaf": len(ontology.child_map.get(term_id, ())) == 0,
            "descendant_count": descendant_counts.get(term_id, np.nan),
            "size_scalar": float(scalar[idx]) if scalar is not None else np.nan,
            "center_norm": float(np.linalg.norm(embedding.centers[idx])),
        })
    return pd.DataFrame(rows)


def evaluate_size_hierarchy(term_df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if "size_scalar" not in term_df or not np.isfinite(term_df["size_scalar"]).any():
        return {"available": False}

    rho, p, n = safe_spearman(term_df["max_depth"], term_df["size_scalar"])
    metrics.update({
        "available": True,
        "size_max_depth_spearman": rho,
        "size_max_depth_pvalue": p,
        "size_max_depth_n": n,
    })
    rho, p, n = safe_spearman(term_df["min_depth"], term_df["size_scalar"])
    metrics.update({
        "size_min_depth_spearman": rho,
        "size_min_depth_pvalue": p,
        "size_min_depth_n": n,
    })
    rho, p, n = safe_spearman(term_df["direct_child_count"], term_df["size_scalar"])
    metrics.update({
        "size_child_count_spearman": rho,
        "size_child_count_pvalue": p,
        "size_child_count_n": n,
    })
    rho, p, n = safe_spearman(
        np.log1p(term_df["descendant_count"].to_numpy(dtype=float)),
        term_df["size_scalar"],
    )
    metrics.update({
        "size_log_descendant_count_spearman": rho,
        "size_log_descendant_count_pvalue": p,
        "size_log_descendant_count_n": n,
    })

    leaf = term_df.loc[term_df["is_leaf"], "size_scalar"].to_numpy(dtype=float)
    nonleaf = term_df.loc[~term_df["is_leaf"], "size_scalar"].to_numpy(dtype=float)
    if leaf.size and nonleaf.size:
        pooled = math.sqrt((np.var(leaf) + np.var(nonleaf)) / 2.0 + EPS)
        metrics["leaf_mean_size"] = float(np.mean(leaf))
        metrics["nonleaf_mean_size"] = float(np.mean(nonleaf))
        metrics["leaf_nonleaf_cohens_d"] = float((np.mean(leaf) - np.mean(nonleaf)) / pooled)

    by_namespace: Dict[str, Any] = {}
    for namespace, group in term_df.groupby("namespace"):
        ns_metrics: Dict[str, Any] = {"n": len(group)}
        for x_col, key in [
            ("max_depth", "size_max_depth_spearman"),
            ("direct_child_count", "size_child_count_spearman"),
            ("descendant_count", "size_descendant_count_spearman"),
        ]:
            x = group[x_col].to_numpy(dtype=float)
            if x_col == "descendant_count":
                x = np.log1p(x)
            rho, p, n = safe_spearman(x, group["size_scalar"])
            ns_metrics[key] = rho
            ns_metrics[f"{key}_pvalue"] = p
            ns_metrics[f"{key}_n"] = n
        by_namespace[str(namespace)] = ns_metrics
    metrics["by_namespace"] = by_namespace
    return metrics


# ---------------------------------------------------------------------------
# Pair sampling and containment diagnostics
# ---------------------------------------------------------------------------


def upward_distances(term_id: str, parent_map: Mapping[str, Set[str]]) -> Dict[str, int]:
    distances: Dict[str, int] = {}
    queue: deque[Tuple[str, int]] = deque((parent, 1) for parent in parent_map.get(term_id, ()))
    while queue:
        node, distance = queue.popleft()
        old = distances.get(node)
        if old is not None and old <= distance:
            continue
        distances[node] = distance
        for parent in parent_map.get(node, ()):
            queue.append((parent, distance + 1))
    return distances


def direct_edge_pairs(
    embedding_ids: Sequence[str],
    id_to_idx: Mapping[str, int],
    ontology: OntologyData,
    rng: np.random.Generator,
    max_pairs: int,
) -> List[Tuple[str, str]]:
    represented = set(embedding_ids)
    pairs = [
        (child, parent)
        for child in embedding_ids
        for parent in ontology.parent_map.get(child, ())
        if parent in represented
    ]
    if len(pairs) > max_pairs > 0:
        choice = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in choice]
    return pairs


def ancestor_pairs(
    embedding_ids: Sequence[str],
    ontology: OntologyData,
    rng: np.random.Generator,
    max_pairs: int,
) -> List[Tuple[str, str, int]]:
    represented = set(embedding_ids)
    if max_pairs <= 0:
        return []

    result: Set[Tuple[str, str, int]] = set()
    cache: Dict[str, Dict[str, int]] = {}
    attempts = 0
    max_attempts = max_pairs * 20 + 1000

    eligible = [x for x in embedding_ids if ontology.parent_map.get(x)]
    if not eligible:
        return []

    while len(result) < max_pairs and attempts < max_attempts:
        attempts += 1
        child = str(rng.choice(eligible))
        if child not in cache:
            cache[child] = {
                ancestor: distance
                for ancestor, distance in upward_distances(child, ontology.parent_map).items()
                if ancestor in represented and distance >= 2
            }
        if not cache[child]:
            continue
        ancestors = list(cache[child].items())
        ancestor, distance = ancestors[int(rng.integers(0, len(ancestors)))]
        result.add((child, ancestor, int(distance)))

    return list(result)


def build_depth_buckets(
    embedding_ids: Sequence[str],
    ontology: OntologyData,
) -> Dict[Tuple[str, int], List[str]]:
    buckets: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for term_id in embedding_ids:
        namespace = ontology.terms[term_id].namespace
        depth = ontology.max_depth.get(term_id, -1)
        buckets[(namespace, depth)].append(term_id)
    return buckets


def sample_matched_negative_pairs(
    positive_pairs: Sequence[Tuple[str, str]],
    embedding_ids: Sequence[str],
    ontology: OntologyData,
    rng: np.random.Generator,
    negatives_per_positive: int,
    depth_tolerance: int,
) -> List[Tuple[str, str]]:
    represented = set(embedding_ids)
    buckets = build_depth_buckets(embedding_ids, ontology)
    ancestor_cache: Dict[str, Set[str]] = {}
    negatives: List[Tuple[str, str]] = []

    for child, true_parent in positive_pairs:
        if child not in ancestor_cache:
            ancestor_cache[child] = set(upward_distances(child, ontology.parent_map))
        forbidden = ancestor_cache[child] | {child}
        namespace = ontology.terms[true_parent].namespace
        target_depth = ontology.max_depth.get(true_parent, -1)

        candidates: List[str] = []
        for delta in range(depth_tolerance + 1):
            depth_values = [target_depth] if delta == 0 else [target_depth - delta, target_depth + delta]
            for depth in depth_values:
                candidates.extend(buckets.get((namespace, depth), ()))
            candidates = [x for x in candidates if x in represented and x not in forbidden]
            if candidates:
                break

        if not candidates:
            candidates = [
                x for x in embedding_ids
                if ontology.terms[x].namespace == namespace and x not in forbidden
            ]
        if not candidates:
            continue

        for _ in range(max(1, negatives_per_positive)):
            negative_parent = str(rng.choice(candidates))
            negatives.append((child, negative_parent))

    return negatives


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": float("nan"), "aupr": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
    }


def evaluate_containment(
    views: Mapping[str, EvaluationView],
    embedding: EmbeddingData,
    ontology: OntologyData,
    rng: np.random.Generator,
    max_direct_pairs: int,
    max_ancestor_pairs: int,
    negatives_per_positive: int,
    negative_depth_tolerance: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    id_to_idx = {term_id: idx for idx, term_id in enumerate(embedding.ids)}
    direct_pairs = direct_edge_pairs(
        embedding.ids, id_to_idx, ontology, rng, max_direct_pairs
    )
    negative_pairs = sample_matched_negative_pairs(
        direct_pairs,
        embedding.ids,
        ontology,
        rng,
        negatives_per_positive,
        negative_depth_tolerance,
    )
    anc_pairs = ancestor_pairs(
        embedding.ids,
        ontology,
        rng,
        max_ancestor_pairs,
    )

    LOGGER.info(
        "Containment pairs: direct=%d, matched_negative=%d, non-direct_ancestor=%d",
        len(direct_pairs), len(negative_pairs), len(anc_pairs),
    )

    direct_child_idx = np.asarray([id_to_idx[a] for a, _ in direct_pairs], dtype=np.int64)
    direct_parent_idx = np.asarray([id_to_idx[b] for _, b in direct_pairs], dtype=np.int64)
    neg_child_idx = np.asarray([id_to_idx[a] for a, _ in negative_pairs], dtype=np.int64)
    neg_parent_idx = np.asarray([id_to_idx[b] for _, b in negative_pairs], dtype=np.int64)
    anc_child_idx = np.asarray([id_to_idx[a] for a, _, _ in anc_pairs], dtype=np.int64)
    anc_parent_idx = np.asarray([id_to_idx[b] for _, b, _ in anc_pairs], dtype=np.int64)
    anc_distance = np.asarray([d for _, _, d in anc_pairs], dtype=np.int64)

    summary: Dict[str, Any] = {
        "n_direct_pairs": len(direct_pairs),
        "n_matched_negative_pairs": len(negative_pairs),
        "n_ancestor_pairs": len(anc_pairs),
        "views": {},
    }
    pair_rows: List[Dict[str, Any]] = []
    plot_arrays: Dict[str, np.ndarray] = {}

    for view_name, view in views.items():
        direct_margin = containment_margin_pairs(view, direct_child_idx, direct_parent_idx) if len(direct_pairs) else np.array([])
        negative_margin = containment_margin_pairs(view, neg_child_idx, neg_parent_idx) if len(negative_pairs) else np.array([])
        ancestor_margin = containment_margin_pairs(view, anc_child_idx, anc_parent_idx) if len(anc_pairs) else np.array([])

        labels = np.concatenate([
            np.ones(direct_margin.size, dtype=np.int64),
            np.zeros(negative_margin.size, dtype=np.int64),
        ])
        scores = np.concatenate([direct_margin, negative_margin])
        classification = binary_metrics(labels, scores)

        view_metrics: Dict[str, Any] = {
            **classification,
            "direct_margin_mean": float(np.mean(direct_margin)) if direct_margin.size else float("nan"),
            "negative_margin_mean": float(np.mean(negative_margin)) if negative_margin.size else float("nan"),
            "ancestor_margin_mean": float(np.mean(ancestor_margin)) if ancestor_margin.size else float("nan"),
        }
        view_metrics.update(percentile_summary(direct_margin, "direct_margin"))
        view_metrics.update(percentile_summary(negative_margin, "negative_margin"))
        view_metrics.update(percentile_summary(ancestor_margin, "ancestor_margin"))

        if view.geometry in {"sphere", "box"} and view.sizes is not None:
            direct_rate = float(np.mean(direct_margin >= 0)) if direct_margin.size else float("nan")
            false_rate = float(np.mean(negative_margin >= 0)) if negative_margin.size else float("nan")
            ancestor_rate = float(np.mean(ancestor_margin >= 0)) if ancestor_margin.size else float("nan")
            view_metrics.update({
                "direct_containment_rate": direct_rate,
                "false_containment_rate": false_rate,
                "containment_specificity": direct_rate - false_rate,
                "ancestor_containment_rate": ancestor_rate,
            })

        distance_bins: Dict[str, Any] = {}
        if ancestor_margin.size:
            bin_defs = [
                ("2", lambda x: x == 2),
                ("3_4", lambda x: (x >= 3) & (x <= 4)),
                ("5_8", lambda x: (x >= 5) & (x <= 8)),
                ("gt8", lambda x: x > 8),
            ]
            for label, selector in bin_defs:
                mask = selector(anc_distance)
                values = ancestor_margin[mask]
                item: Dict[str, Any] = {
                    "n": int(mask.sum()),
                    "margin_mean": float(np.mean(values)) if values.size else float("nan"),
                    "margin_median": float(np.median(values)) if values.size else float("nan"),
                }
                if view.geometry in {"sphere", "box"} and view.sizes is not None:
                    item["containment_rate"] = float(np.mean(values >= 0)) if values.size else float("nan")
                distance_bins[label] = item
        view_metrics["ancestor_by_graph_distance"] = distance_bins
        summary["views"][view_name] = view_metrics

        if view_name == "full":
            plot_arrays["direct_margin"] = direct_margin
            plot_arrays["negative_margin"] = negative_margin
            plot_arrays["ancestor_margin"] = ancestor_margin

        # Keep pair-level output bounded to the sampled pairs already selected.
        for (child, parent), margin in zip(direct_pairs, direct_margin):
            pair_rows.append({
                "view": view_name,
                "pair_type": "direct_parent",
                "child": child,
                "parent_or_candidate": parent,
                "graph_distance": 1,
                "score_or_margin": float(margin),
            })
        for (child, candidate), margin in zip(negative_pairs, negative_margin):
            pair_rows.append({
                "view": view_name,
                "pair_type": "matched_negative",
                "child": child,
                "parent_or_candidate": candidate,
                "graph_distance": np.nan,
                "score_or_margin": float(margin),
            })
        for (child, ancestor, distance), margin in zip(anc_pairs, ancestor_margin):
            pair_rows.append({
                "view": view_name,
                "pair_type": "non_direct_ancestor",
                "child": child,
                "parent_or_candidate": ancestor,
                "graph_distance": distance,
                "score_or_margin": float(margin),
            })

    return summary, pd.DataFrame(pair_rows), plot_arrays


# ---------------------------------------------------------------------------
# Parent and ancestor retrieval
# ---------------------------------------------------------------------------


def average_precision_from_ranks(ranks: Sequence[int], n_relevant: int) -> float:
    if n_relevant <= 0:
        return float("nan")
    ordered = sorted(int(r) for r in ranks)
    precisions = [(idx + 1) / rank for idx, rank in enumerate(ordered)]
    return float(np.sum(precisions) / n_relevant)


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    rel = np.asarray(list(relevances)[:k], dtype=np.float64)
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2, dtype=np.float64))
    return float(np.sum(rel / discounts))


def ndcg_at_k(relevances: Sequence[float], ideal_relevances: Sequence[float], k: int) -> float:
    ideal = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if ideal <= 0:
        return float("nan")
    return float(dcg_at_k(relevances, k) / ideal)


def namespace_candidate_pools(
    embedding: EmbeddingData,
    ontology: OntologyData,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, int]]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, term_id in enumerate(embedding.ids):
        groups[ontology.terms[term_id].namespace].append(idx)
    pools = {namespace: np.asarray(indices, dtype=np.int64) for namespace, indices in groups.items()}
    local_maps = {
        namespace: {int(global_idx): local_idx for local_idx, global_idx in enumerate(pool)}
        for namespace, pool in pools.items()
    }
    return pools, local_maps


def choose_retrieval_queries(
    embedding: EmbeddingData,
    ontology: OntologyData,
    rng: np.random.Generator,
    query_sample_size: int,
) -> List[str]:
    represented = set(embedding.ids)
    eligible = [
        term_id for term_id in embedding.ids
        if any(parent in represented for parent in ontology.parent_map.get(term_id, ()))
    ]
    if len(eligible) <= query_sample_size or query_sample_size <= 0:
        return eligible

    # Rough namespace stratification.
    groups: Dict[str, List[str]] = defaultdict(list)
    for term_id in eligible:
        groups[ontology.terms[term_id].namespace].append(term_id)

    selected: List[str] = []
    for namespace, items in groups.items():
        quota = max(1, int(round(query_sample_size * len(items) / len(eligible))))
        quota = min(quota, len(items))
        selected.extend(rng.choice(np.asarray(items, dtype=object), size=quota, replace=False).tolist())

    if len(selected) > query_sample_size:
        selected = rng.choice(np.asarray(selected, dtype=object), size=query_sample_size, replace=False).tolist()
    elif len(selected) < query_sample_size:
        remaining = list(set(eligible) - set(selected))
        add = min(query_sample_size - len(selected), len(remaining))
        if add:
            selected.extend(rng.choice(np.asarray(remaining, dtype=object), size=add, replace=False).tolist())
    return [str(x) for x in selected]


def evaluate_retrieval(
    views: Mapping[str, EvaluationView],
    embedding: EmbeddingData,
    ontology: OntologyData,
    rng: np.random.Generator,
    query_sample_size: int,
    parent_ks: Sequence[int],
    ancestor_ks: Sequence[int],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    id_to_idx = {term_id: idx for idx, term_id in enumerate(embedding.ids)}
    represented = set(embedding.ids)
    pools, local_maps = namespace_candidate_pools(embedding, ontology)
    queries = choose_retrieval_queries(embedding, ontology, rng, query_sample_size)
    max_ancestor_k = max(ancestor_ks) if ancestor_ks else 0

    summary: Dict[str, Any] = {"n_queries": len(queries), "views": {}}
    per_term_rows: List[Dict[str, Any]] = []

    for view_name, view in views.items():
        aggregate: Dict[str, List[float]] = defaultdict(list)
        namespace_aggregate: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        for pos, query_id in enumerate(queries, start=1):
            q_idx = id_to_idx[query_id]
            namespace = ontology.terms[query_id].namespace
            pool = pools[namespace]
            candidate_idx = pool[pool != q_idx]
            if candidate_idx.size == 0:
                continue

            scores = asymmetric_scores(view, q_idx, candidate_idx)
            order = np.argsort(-scores, kind="mergesort")
            ranked_idx = candidate_idx[order]
            rank_lookup = {int(global_idx): rank + 1 for rank, global_idx in enumerate(ranked_idx)}

            true_parents = [
                p for p in ontology.parent_map.get(query_id, ())
                if p in represented and ontology.terms[p].namespace == namespace
            ]
            parent_ranks = [rank_lookup[id_to_idx[p]] for p in true_parents if id_to_idx[p] in rank_lookup]
            if not parent_ranks:
                continue

            row: Dict[str, Any] = {
                "view": view_name,
                "go_id": query_id,
                "namespace": namespace,
                "max_depth": ontology.max_depth.get(query_id, -1),
                "n_true_parents": len(true_parents),
                "parent_mrr": 1.0 / min(parent_ranks),
                "parent_map": average_precision_from_ranks(parent_ranks, len(true_parents)),
            }
            aggregate["parent_mrr"].append(row["parent_mrr"])
            aggregate["parent_map"].append(row["parent_map"])
            namespace_aggregate[namespace]["parent_mrr"].append(row["parent_mrr"])
            namespace_aggregate[namespace]["parent_map"].append(row["parent_map"])

            for k in parent_ks:
                recall = sum(rank <= k for rank in parent_ranks) / len(true_parents)
                key = f"parent_recall_at_{k}"
                row[key] = recall
                aggregate[key].append(recall)
                namespace_aggregate[namespace][key].append(recall)

            # Parent nDCG with binary relevance.
            max_parent_k = max(parent_ks) if parent_ks else 10
            parent_set_idx = {id_to_idx[p] for p in true_parents}
            parent_rel = [1.0 if int(idx) in parent_set_idx else 0.0 for idx in ranked_idx[:max_parent_k]]
            row[f"parent_ndcg_at_{max_parent_k}"] = ndcg_at_k(
                parent_rel,
                [1.0] * len(true_parents),
                max_parent_k,
            )
            aggregate[f"parent_ndcg_at_{max_parent_k}"].append(row[f"parent_ndcg_at_{max_parent_k}"])
            namespace_aggregate[namespace][f"parent_ndcg_at_{max_parent_k}"].append(row[f"parent_ndcg_at_{max_parent_k}"])

            # Ancestors with inverse graph-distance relevance.
            up = upward_distances(query_id, ontology.parent_map)
            ancestor_distance = {
                id_to_idx[term_id]: distance
                for term_id, distance in up.items()
                if term_id in represented and ontology.terms[term_id].namespace == namespace
            }
            ancestor_ranks = [rank_lookup[idx] for idx in ancestor_distance if idx in rank_lookup]
            if ancestor_ranks:
                for k in ancestor_ks:
                    recall = sum(rank <= k for rank in ancestor_ranks) / len(ancestor_ranks)
                    key = f"ancestor_recall_at_{k}"
                    row[key] = recall
                    aggregate[key].append(recall)
                    namespace_aggregate[namespace][key].append(recall)

                top = ranked_idx[:max_ancestor_k]
                relevance = [
                    1.0 / ancestor_distance[int(idx)] if int(idx) in ancestor_distance else 0.0
                    for idx in top
                ]
                ideal = [1.0 / distance for distance in ancestor_distance.values()]
                ndcg_key = f"ancestor_weighted_ndcg_at_{max_ancestor_k}"
                row[ndcg_key] = ndcg_at_k(relevance, ideal, max_ancestor_k)
                aggregate[ndcg_key].append(row[ndcg_key])
                namespace_aggregate[namespace][ndcg_key].append(row[ndcg_key])

                penalty = max(ontology.max_depth.get(query_id, 0) + 2, 10)
                graph_distances_top = [
                    ancestor_distance.get(int(idx), penalty) for idx in top
                ]
                mgd_key = f"mean_penalized_graph_distance_at_{max_ancestor_k}"
                row[mgd_key] = float(np.mean(graph_distances_top))
                aggregate[mgd_key].append(row[mgd_key])
                namespace_aggregate[namespace][mgd_key].append(row[mgd_key])

            per_term_rows.append(row)

            if pos % 200 == 0:
                LOGGER.info(
                    "Retrieval view=%s: processed %d/%d queries.",
                    view_name, pos, len(queries),
                )

        view_summary: Dict[str, Any] = {
            key: float(np.nanmean(values)) if values else float("nan")
            for key, values in aggregate.items()
        }
        view_summary["n_evaluated_queries"] = len(aggregate.get("parent_mrr", []))
        view_summary["by_namespace"] = {
            namespace: {
                key: float(np.nanmean(values)) if values else float("nan")
                for key, values in metric_map.items()
            }
            for namespace, metric_map in namespace_aggregate.items()
        }
        summary["views"][view_name] = view_summary

    return summary, pd.DataFrame(per_term_rows)


# ---------------------------------------------------------------------------
# Sibling and local-neighborhood diagnostics
# ---------------------------------------------------------------------------


def sample_sibling_pairs(
    embedding: EmbeddingData,
    ontology: OntologyData,
    rng: np.random.Generator,
    max_pairs: int,
    max_pairs_per_parent: int,
) -> List[Tuple[str, str]]:
    represented = set(embedding.ids)
    result: Set[Tuple[str, str]] = set()

    parents = list(ontology.child_map)
    rng.shuffle(parents)
    for parent in parents:
        children = [x for x in ontology.child_map[parent] if x in represented]
        if len(children) < 2:
            continue
        total_possible = len(children) * (len(children) - 1) // 2
        n_take = min(max_pairs_per_parent, total_possible)
        attempts = 0
        taken = 0
        while taken < n_take and attempts < n_take * 20 + 20:
            attempts += 1
            a, b = rng.choice(np.asarray(children, dtype=object), size=2, replace=False)
            pair = tuple(sorted((str(a), str(b))))
            if pair not in result:
                result.add(pair)
                taken += 1
                if len(result) >= max_pairs:
                    return list(result)
    return list(result)


def share_parent(a: str, b: str, ontology: OntologyData) -> bool:
    return bool(ontology.parent_map.get(a, set()) & ontology.parent_map.get(b, set()))


def sample_sibling_negatives(
    positive_pairs: Sequence[Tuple[str, str]],
    embedding: EmbeddingData,
    ontology: OntologyData,
    rng: np.random.Generator,
    depth_tolerance: int,
) -> List[Tuple[str, str]]:
    buckets = build_depth_buckets(embedding.ids, ontology)
    negatives: List[Tuple[str, str]] = []

    for a, b in positive_pairs:
        namespace = ontology.terms[b].namespace
        target_depth = ontology.max_depth.get(b, -1)
        candidates: List[str] = []
        for delta in range(depth_tolerance + 1):
            depths = [target_depth] if delta == 0 else [target_depth - delta, target_depth + delta]
            for depth in depths:
                candidates.extend(buckets.get((namespace, depth), ()))
            candidates = [
                x for x in candidates
                if x != a and not share_parent(a, x, ontology)
            ]
            if candidates:
                break
        if not candidates:
            candidates = [
                x for x in embedding.ids
                if ontology.terms[x].namespace == namespace
                and x != a
                and not share_parent(a, x, ontology)
            ]
        if candidates:
            negatives.append((a, str(rng.choice(candidates))))
    return negatives


def evaluate_siblings(
    views: Mapping[str, EvaluationView],
    embedding: EmbeddingData,
    ontology: OntologyData,
    rng: np.random.Generator,
    max_pairs: int,
    max_pairs_per_parent: int,
    negative_depth_tolerance: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    id_to_idx = {term_id: idx for idx, term_id in enumerate(embedding.ids)}
    positives = sample_sibling_pairs(
        embedding, ontology, rng, max_pairs, max_pairs_per_parent
    )
    negatives = sample_sibling_negatives(
        positives, embedding, ontology, rng, negative_depth_tolerance
    )

    LOGGER.info("Sibling pairs: positive=%d, matched_negative=%d", len(positives), len(negatives))

    pos_l = np.asarray([id_to_idx[a] for a, _ in positives], dtype=np.int64)
    pos_r = np.asarray([id_to_idx[b] for _, b in positives], dtype=np.int64)
    neg_l = np.asarray([id_to_idx[a] for a, _ in negatives], dtype=np.int64)
    neg_r = np.asarray([id_to_idx[b] for _, b in negatives], dtype=np.int64)

    summary: Dict[str, Any] = {
        "n_sibling_pairs": len(positives),
        "n_matched_negative_pairs": len(negatives),
        "views": {},
    }
    rows: List[Dict[str, Any]] = []

    for view_name, view in views.items():
        pos_distance = symmetric_distances_pairs(view, pos_l, pos_r) if positives else np.array([])
        neg_distance = symmetric_distances_pairs(view, neg_l, neg_r) if negatives else np.array([])
        labels = np.concatenate([
            np.ones(pos_distance.size, dtype=np.int64),
            np.zeros(neg_distance.size, dtype=np.int64),
        ])
        scores = -np.concatenate([pos_distance, neg_distance])
        metrics = binary_metrics(labels, scores)
        metrics.update({
            "sibling_distance_mean": float(np.mean(pos_distance)) if pos_distance.size else float("nan"),
            "matched_negative_distance_mean": float(np.mean(neg_distance)) if neg_distance.size else float("nan"),
        })
        if pos_distance.size and neg_distance.size:
            pooled = math.sqrt((np.var(pos_distance) + np.var(neg_distance)) / 2.0 + EPS)
            metrics["distance_cohens_d_positive_minus_negative"] = float(
                (np.mean(pos_distance) - np.mean(neg_distance)) / pooled
            )

        if view.geometry in {"sphere", "box"} and view.sizes is not None and positives:
            margin_ab = containment_margin_pairs(view, pos_l, pos_r)
            margin_ba = containment_margin_pairs(view, pos_r, pos_l)
            metrics["sibling_any_direction_containment_rate"] = float(
                np.mean((margin_ab >= 0) | (margin_ba >= 0))
            )
            metrics["sibling_both_direction_containment_rate"] = float(
                np.mean((margin_ab >= 0) & (margin_ba >= 0))
            )

        summary["views"][view_name] = metrics

        for (a, b), distance in zip(positives, pos_distance):
            rows.append({"view": view_name, "pair_type": "sibling", "left": a, "right": b, "distance": float(distance)})
        for (a, b), distance in zip(negatives, neg_distance):
            rows.append({"view": view_name, "pair_type": "matched_nonsibling", "left": a, "right": b, "distance": float(distance)})

    return summary, pd.DataFrame(rows)


def sibling_set(term_id: str, ontology: OntologyData, represented: Set[str]) -> Set[str]:
    result: Set[str] = set()
    for parent in ontology.parent_map.get(term_id, ()):
        result.update(x for x in ontology.child_map.get(parent, ()) if x in represented)
    result.discard(term_id)
    return result


def evaluate_knn_structure(
    views: Mapping[str, EvaluationView],
    embedding: EmbeddingData,
    ontology: OntologyData,
    rng: np.random.Generator,
    query_sample_size: int,
    k: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    represented = set(embedding.ids)
    id_to_idx = {term_id: idx for idx, term_id in enumerate(embedding.ids)}
    pools, _ = namespace_candidate_pools(embedding, ontology)

    qn = min(query_sample_size, len(embedding.ids))
    query_ids = rng.choice(np.asarray(embedding.ids, dtype=object), size=qn, replace=False).tolist()

    summary: Dict[str, Any] = {"n_queries": qn, "k": k, "views": {}}
    rows: List[Dict[str, Any]] = []

    for view_name, view in views.items():
        aggregates: Dict[str, List[float]] = defaultdict(list)
        for pos, query_id_raw in enumerate(query_ids, start=1):
            query_id = str(query_id_raw)
            q_idx = id_to_idx[query_id]
            namespace = ontology.terms[query_id].namespace
            pool = pools[namespace]
            candidates = pool[pool != q_idx]
            if candidates.size == 0:
                continue
            distances = symmetric_distances_one_to_many(view, q_idx, candidates)
            actual_k = min(k, candidates.size)
            top_local = np.argpartition(distances, actual_k - 1)[:actual_k]
            top_local = top_local[np.argsort(distances[top_local])]
            top_ids = {embedding.ids[int(idx)] for idx in candidates[top_local]}

            parents = {x for x in ontology.parent_map.get(query_id, ()) if x in represented}
            children = {x for x in ontology.child_map.get(query_id, ()) if x in represented}
            siblings = sibling_set(query_id, ontology, represented)
            graph2 = {
                x for x, distance in nx.single_source_shortest_path_length(
                    ontology.graph_undirected, query_id, cutoff=2
                ).items()
                if x in represented and x != query_id and distance <= 2
            }

            row: Dict[str, Any] = {
                "view": view_name,
                "go_id": query_id,
                "namespace": namespace,
                "max_depth": ontology.max_depth.get(query_id, -1),
            }
            for label, reference in [
                ("parent", parents),
                ("child", children),
                ("sibling", siblings),
                ("graph_distance_le_2", graph2),
            ]:
                purity = len(top_ids & reference) / actual_k
                recall = len(top_ids & reference) / len(reference) if reference else np.nan
                row[f"{label}_purity_at_{k}"] = purity
                row[f"{label}_recall_at_{k}"] = recall
                aggregates[f"{label}_purity_at_{k}"].append(purity)
                if np.isfinite(recall):
                    aggregates[f"{label}_recall_at_{k}"].append(float(recall))

            rows.append(row)
            if pos % 200 == 0:
                LOGGER.info("kNN view=%s: processed %d/%d queries.", view_name, pos, qn)

        summary["views"][view_name] = {
            key: float(np.mean(values)) if values else float("nan")
            for key, values in aggregates.items()
        }

    return summary, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Graph distance versus embedding distance
# ---------------------------------------------------------------------------


def sample_graph_distance_pairs(
    embedding: EmbeddingData,
    ontology: OntologyData,
    rng: np.random.Generator,
    n_pairs: int,
    n_sources: int,
) -> List[Tuple[str, str, int]]:
    if n_pairs <= 0 or not embedding.ids:
        return []
    sources_n = min(n_sources, len(embedding.ids))
    sources = rng.choice(np.asarray(embedding.ids, dtype=object), size=sources_n, replace=False)
    per_source = int(math.ceil(n_pairs / sources_n))
    represented = set(embedding.ids)
    pairs: List[Tuple[str, str, int]] = []

    for source_raw in sources:
        source = str(source_raw)
        namespace = ontology.terms[source].namespace
        lengths = nx.single_source_shortest_path_length(ontology.graph_undirected, source)
        candidates = [
            (target, distance)
            for target, distance in lengths.items()
            if target in represented
            and target != source
            and ontology.terms[target].namespace == namespace
        ]
        if not candidates:
            continue
        take = min(per_source, len(candidates))
        chosen = rng.choice(len(candidates), size=take, replace=False)
        for idx in chosen:
            target, distance = candidates[int(idx)]
            pairs.append((source, target, int(distance)))
            if len(pairs) >= n_pairs:
                return pairs
    return pairs


def graph_distance_bin(distance: int) -> str:
    if distance == 1:
        return "1"
    if distance == 2:
        return "2"
    if distance == 3:
        return "3"
    if distance <= 5:
        return "4_5"
    if distance <= 10:
        return "6_10"
    return "gt10"


def evaluate_graph_distance(
    views: Mapping[str, EvaluationView],
    embedding: EmbeddingData,
    ontology: OntologyData,
    rng: np.random.Generator,
    n_pairs: int,
    n_sources: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    pairs = sample_graph_distance_pairs(
        embedding, ontology, rng, n_pairs, n_sources
    )
    id_to_idx = {term_id: idx for idx, term_id in enumerate(embedding.ids)}
    left_idx = np.asarray([id_to_idx[a] for a, _, _ in pairs], dtype=np.int64)
    right_idx = np.asarray([id_to_idx[b] for _, b, _ in pairs], dtype=np.int64)
    graph_dist = np.asarray([d for _, _, d in pairs], dtype=np.int64)

    summary: Dict[str, Any] = {"n_pairs": len(pairs), "views": {}}
    rows: List[Dict[str, Any]] = []

    for view_name, view in views.items():
        emb_dist = symmetric_distances_pairs(view, left_idx, right_idx) if pairs else np.array([])
        rho, p, n = safe_spearman(graph_dist, emb_dist)
        view_metrics: Dict[str, Any] = {
            "graph_embedding_distance_spearman": rho,
            "graph_embedding_distance_pvalue": p,
            "graph_embedding_distance_n": n,
            "by_graph_distance_bin": {},
        }
        for label in ["1", "2", "3", "4_5", "6_10", "gt10"]:
            mask = np.asarray([graph_distance_bin(int(d)) == label for d in graph_dist])
            values = emb_dist[mask]
            view_metrics["by_graph_distance_bin"][label] = {
                "n": int(mask.sum()),
                "mean": float(np.mean(values)) if values.size else float("nan"),
                "median": float(np.median(values)) if values.size else float("nan"),
            }
        summary["views"][view_name] = view_metrics

        for (a, b, distance), e_distance in zip(pairs, emb_dist):
            rows.append({
                "view": view_name,
                "left": a,
                "right": b,
                "graph_distance": distance,
                "graph_distance_bin": graph_distance_bin(distance),
                "embedding_distance": float(e_distance),
            })

    return summary, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-depth aggregation and plots
# ---------------------------------------------------------------------------


def aggregate_per_depth(
    term_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    knn_df: pd.DataFrame,
) -> pd.DataFrame:
    base = term_df.groupby(["namespace", "max_depth"], dropna=False).agg(
        n_terms=("go_id", "size"),
        mean_size=("size_scalar", "mean"),
        median_size=("size_scalar", "median"),
        mean_center_norm=("center_norm", "mean"),
        mean_direct_children=("direct_child_count", "mean"),
        leaf_rate=("is_leaf", "mean"),
    ).reset_index()

    if not retrieval_df.empty:
        numeric = [
            column for column in retrieval_df.columns
            if column not in {"view", "go_id", "namespace", "max_depth"}
            and pd.api.types.is_numeric_dtype(retrieval_df[column])
        ]
        ret = retrieval_df.groupby(["view", "namespace", "max_depth"])[numeric].mean().reset_index()
        base = base.merge(ret, on=["namespace", "max_depth"], how="right")
    else:
        base["view"] = "full"

    if not knn_df.empty:
        numeric = [
            column for column in knn_df.columns
            if column not in {"view", "go_id", "namespace", "max_depth"}
            and pd.api.types.is_numeric_dtype(knn_df[column])
        ]
        knn = knn_df.groupby(["view", "namespace", "max_depth"])[numeric].mean().reset_index()
        base = base.merge(knn, on=["view", "namespace", "max_depth"], how="outer")

    return base.sort_values(["view", "namespace", "max_depth"]).reset_index(drop=True)


def save_plots(
    output_dir: Path,
    embedding: EmbeddingData,
    term_df: pd.DataFrame,
    health_artifacts: GeometryHealthArtifacts,
    containment_plot_arrays: Mapping[str, np.ndarray],
    graph_distance_df: pd.DataFrame,
    max_plot_points: int,
    rng: np.random.Generator,
) -> None:
    figure_dir = ensure_dir(output_dir / "figures")

    # PCA scree.
    eigen = health_artifacts.eigenvalues
    if eigen.size and np.sum(eigen) > 0:
        explained = eigen / np.sum(eigen)
        plt.figure(figsize=(7, 5))
        plt.plot(np.arange(1, len(explained) + 1), explained, marker="o", markersize=3)
        plt.xlabel("Principal component")
        plt.ylabel("Explained variance ratio")
        plt.title("Embedding center PCA spectrum")
        plt.tight_layout()
        plt.savefig(figure_dir / "pca_scree.png", dpi=180)
        plt.close()

    # Random pair distance histogram.
    distances = health_artifacts.sampled_pair_distances
    if distances.size:
        plt.figure(figsize=(7, 5))
        plt.hist(distances, bins=60)
        plt.xlabel("Center Euclidean distance")
        plt.ylabel("Count")
        plt.title("Random GO-term pair distances")
        plt.tight_layout()
        plt.savefig(figure_dir / "random_pair_distance_hist.png", dpi=180)
        plt.close()

    # Size versus depth.
    plot_df = term_df[np.isfinite(term_df["size_scalar"]) & (term_df["max_depth"] >= 0)].copy()
    if not plot_df.empty:
        if len(plot_df) > max_plot_points:
            plot_df = plot_df.iloc[rng.choice(len(plot_df), size=max_plot_points, replace=False)]
        plt.figure(figsize=(7, 5))
        plt.scatter(plot_df["max_depth"], plot_df["size_scalar"], s=8, alpha=0.35)
        plt.xlabel("GO maximum depth")
        plt.ylabel("Log radius / log box volume")
        plt.title("Geometric size versus GO depth")
        plt.tight_layout()
        plt.savefig(figure_dir / "size_vs_depth.png", dpi=180)
        plt.close()

    # Direct versus matched-negative containment margins.
    direct = np.asarray(containment_plot_arrays.get("direct_margin", []), dtype=float)
    negative = np.asarray(containment_plot_arrays.get("negative_margin", []), dtype=float)
    if direct.size and negative.size:
        plt.figure(figsize=(7, 5))
        plt.hist(direct, bins=60, alpha=0.55, label="Direct parent")
        plt.hist(negative, bins=60, alpha=0.55, label="Matched negative")
        plt.xlabel("Containment margin / score")
        plt.ylabel("Count")
        plt.title("Hierarchy containment-score separation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "containment_margin_hist.png", dpi=180)
        plt.close()

    # Graph distance bins versus embedding distance, full view.
    if not graph_distance_df.empty:
        full_df = graph_distance_df[graph_distance_df["view"] == "full"]
        order = ["1", "2", "3", "4_5", "6_10", "gt10"]
        values = [
            full_df.loc[full_df["graph_distance_bin"] == label, "embedding_distance"].to_numpy(dtype=float)
            for label in order
        ]
        nonempty = [(label, vals) for label, vals in zip(order, values) if vals.size]
        if nonempty:
            plt.figure(figsize=(8, 5))
            plt.boxplot([vals for _, vals in nonempty], tick_labels=[label for label, _ in nonempty], showfliers=False)
            plt.xlabel("GO undirected shortest-path distance")
            plt.ylabel("Symmetric geometric distance")
            plt.title("GO graph distance versus embedding distance")
            plt.tight_layout()
            plt.savefig(figure_dir / "graph_vs_embedding_distance.png", dpi=180)
            plt.close()


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parse_int_list(text: str) -> List[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers.")
    return values


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate sphere/box/vector GO embeddings from their geometry and GO hierarchy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embedding-file", required=True, help="Pickle/CSV/Parquet embedding table.")
    parser.add_argument("--go-file", required=True, help="GO OBO file, preferably go-basic.obo.")
    parser.add_argument("--output-dir", default="go_embedding_evaluation", help="Output directory.")

    parser.add_argument(
        "--geometry",
        choices=["auto", "sphere", "box", "vector"],
        default="auto",
        help="Geometric interpretation of each GO class embedding.",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "center_radius", "center_offset", "min_max", "separate", "vector"],
        default="auto",
        help="Layout of concatenated embedding vectors.",
    )
    parser.add_argument("--embedding-dim", type=int, default=None, help="Center dimension d for d+1 or 2d layouts.")
    parser.add_argument("--id-column", default=None, help="GO ID column; auto-detected when omitted.")
    parser.add_argument("--embedding-column", default=None, help="Array-valued embedding column.")
    parser.add_argument("--center-column", default=None, help="Optional array-valued center column.")
    parser.add_argument("--radius-column", default=None, help="Optional scalar radius column.")
    parser.add_argument("--offset-column", default=None, help="Optional array-valued box half-width column.")
    parser.add_argument(
        "--size-transform",
        choices=["abs", "softplus", "none"],
        default="abs",
        help="Transform applied to radius/offset parameters before geometry evaluation.",
    )

    parser.add_argument(
        "--edge-types",
        type=parse_str_list,
        default=["is_a"],
        help="Comma-separated OBO relation types used as hierarchy edges.",
    )
    parser.add_argument(
        "--ablations",
        type=parse_str_list,
        default=["full", "center", "size_shuffled"],
        help="Comma-separated views: full,center,size_shuffled,node_shuffled.",
    )

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--pair-sample-size", type=int, default=100000, help="Random center-distance pairs.")
    parser.add_argument("--health-query-sample-size", type=int, default=3000, help="Queries for NN/hubness estimates.")
    parser.add_argument("--health-knn-k", type=int, default=10)
    parser.add_argument("--pca-sample-size", type=int, default=10000)
    parser.add_argument("--descendant-sample-size", type=int, default=5000, help="Terms with exact descendant counts.")

    parser.add_argument("--max-direct-pairs", type=int, default=50000)
    parser.add_argument("--max-ancestor-pairs", type=int, default=50000)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--negative-depth-tolerance", type=int, default=1)

    parser.add_argument("--retrieval-query-sample-size", type=int, default=1000)
    parser.add_argument("--parent-ks", type=parse_int_list, default=[1, 5, 10, 50])
    parser.add_argument("--ancestor-ks", type=parse_int_list, default=[10, 50, 100])

    parser.add_argument("--max-sibling-pairs", type=int, default=30000)
    parser.add_argument("--max-sibling-pairs-per-parent", type=int, default=20)

    parser.add_argument("--knn-query-sample-size", type=int, default=500)
    parser.add_argument("--structure-knn-k", type=int, default=20)

    parser.add_argument("--graph-pair-sample-size", type=int, default=10000)
    parser.add_argument("--graph-pair-source-size", type=int, default=300)

    parser.add_argument("--max-plot-points", type=int, default=10000)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG diagnostic plots.")
    parser.add_argument("--save-pair-tables", action="store_true", help="Save sampled pair-level CSV files.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    nonnegative = [
        "pair_sample_size",
        "health_query_sample_size",
        "pca_sample_size",
        "descendant_sample_size",
        "max_direct_pairs",
        "max_ancestor_pairs",
        "retrieval_query_sample_size",
        "max_sibling_pairs",
        "knn_query_sample_size",
        "graph_pair_sample_size",
    ]
    for name in nonnegative:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.embedding_dim is not None and args.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be positive.")
    if args.health_knn_k <= 0 or args.structure_knn_k <= 0:
        raise ValueError("k values must be positive.")
    if args.negatives_per_positive <= 0:
        raise ValueError("--negatives-per-positive must be positive.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        validate_args(args)
        rng = set_seed(args.seed)
        output_dir = ensure_dir(args.output_dir)

        ontology = build_ontology(args.go_file, args.edge_types)
        raw_embedding = load_embeddings(
            path=args.embedding_file,
            geometry=args.geometry,
            layout=args.layout,
            embedding_dim=args.embedding_dim,
            id_column=args.id_column,
            embedding_column=args.embedding_column,
            center_column=args.center_column,
            radius_column=args.radius_column,
            offset_column=args.offset_column,
            size_transform=args.size_transform,
        )
        embedding, unmatched_ids = align_embeddings_to_go(raw_embedding, ontology)

        id_to_idx = {term_id: idx for idx, term_id in enumerate(embedding.ids)}
        namespaces = [ontology.terms[term_id].namespace for term_id in embedding.ids]
        views = make_evaluation_views(embedding, args.ablations, rng)

        LOGGER.info("Evaluating geometry health.")
        geometry_health, health_artifacts = evaluate_geometry_health(
            embedding=embedding,
            namespaces=namespaces,
            rng=rng,
            pair_sample_size=args.pair_sample_size,
            query_sample_size=args.health_query_sample_size,
            knn_k=args.health_knn_k,
            pca_sample_size=args.pca_sample_size,
        )

        LOGGER.info("Computing sampled exact descendant counts.")
        descendant_counts = sample_descendant_counts(
            ontology,
            embedding.ids,
            rng,
            args.descendant_sample_size,
        )
        term_df = build_term_table(embedding, ontology, descendant_counts)
        size_hierarchy = evaluate_size_hierarchy(term_df)

        LOGGER.info("Evaluating hierarchy containment.")
        containment, containment_pairs_df, containment_plot_arrays = evaluate_containment(
            views=views,
            embedding=embedding,
            ontology=ontology,
            rng=rng,
            max_direct_pairs=args.max_direct_pairs,
            max_ancestor_pairs=args.max_ancestor_pairs,
            negatives_per_positive=args.negatives_per_positive,
            negative_depth_tolerance=args.negative_depth_tolerance,
        )

        LOGGER.info("Evaluating parent and ancestor retrieval.")
        retrieval, retrieval_df = evaluate_retrieval(
            views=views,
            embedding=embedding,
            ontology=ontology,
            rng=rng,
            query_sample_size=args.retrieval_query_sample_size,
            parent_ks=args.parent_ks,
            ancestor_ks=args.ancestor_ks,
        )

        LOGGER.info("Evaluating sibling structure.")
        siblings, sibling_pairs_df = evaluate_siblings(
            views=views,
            embedding=embedding,
            ontology=ontology,
            rng=rng,
            max_pairs=args.max_sibling_pairs,
            max_pairs_per_parent=args.max_sibling_pairs_per_parent,
            negative_depth_tolerance=args.negative_depth_tolerance,
        )

        LOGGER.info("Evaluating kNN graph-neighborhood preservation.")
        knn_structure, knn_df = evaluate_knn_structure(
            views=views,
            embedding=embedding,
            ontology=ontology,
            rng=rng,
            query_sample_size=args.knn_query_sample_size,
            k=args.structure_knn_k,
        )

        LOGGER.info("Evaluating GO graph distance versus geometric distance.")
        graph_distance, graph_distance_df = evaluate_graph_distance(
            views=views,
            embedding=embedding,
            ontology=ontology,
            rng=rng,
            n_pairs=args.graph_pair_sample_size,
            n_sources=args.graph_pair_source_size,
        )

        # A clean per-term hierarchy table independent of view.
        base_term_df = build_term_table(embedding, ontology, descendant_counts)
        per_depth_df = aggregate_per_depth(base_term_df, retrieval_df, knn_df)

        summary: Dict[str, Any] = {
            "input": {
                "embedding_file": str(Path(args.embedding_file).resolve()),
                "go_file": str(Path(args.go_file).resolve()),
                "output_dir": str(output_dir.resolve()),
                "selected_edge_types": args.edge_types,
                "requested_ablations": args.ablations,
                "seed": args.seed,
            },
            "embedding": {
                "source_rows": raw_embedding.source_rows,
                "source_unique_ids": len(raw_embedding.ids),
                "matched_go_terms": len(embedding.ids),
                "unmatched_embedding_ids": len(unmatched_ids),
                "unmatched_embedding_id_examples": unmatched_ids[:50],
                "geometry": embedding.geometry,
                "layout": embedding.layout,
                "center_dim": int(embedding.centers.shape[1]),
                "size_shape": list(embedding.sizes.shape) if embedding.sizes is not None else None,
                "metadata": embedding.metadata,
            },
            "ontology": {
                "active_terms": len(ontology.terms),
                "selected_edges": ontology.graph_child_parent.number_of_edges(),
                "roots": ontology.roots,
                "selected_edge_types": ontology.selected_edge_types,
            },
            "geometry_health": geometry_health,
            "size_hierarchy": size_hierarchy,
            "containment": containment,
            "retrieval": retrieval,
            "siblings": siblings,
            "knn_structure": knn_structure,
            "graph_distance": graph_distance,
        }

        write_json(summary, output_dir / "summary.json")
        base_term_df.to_csv(output_dir / "per_term_geometry.csv", index=False)
        retrieval_df.to_csv(output_dir / "per_term_retrieval.csv", index=False)
        knn_df.to_csv(output_dir / "per_term_knn.csv", index=False)
        per_depth_df.to_csv(output_dir / "per_depth_metrics.csv", index=False)
        graph_distance_df.to_csv(output_dir / "graph_distance_pairs.csv", index=False)

        if args.save_pair_tables:
            containment_pairs_df.to_csv(output_dir / "containment_pairs.csv", index=False)
            sibling_pairs_df.to_csv(output_dir / "sibling_pairs.csv", index=False)

        if not args.no_plots:
            save_plots(
                output_dir=output_dir,
                embedding=embedding,
                term_df=base_term_df,
                health_artifacts=health_artifacts,
                containment_plot_arrays=containment_plot_arrays,
                graph_distance_df=graph_distance_df,
                max_plot_points=args.max_plot_points,
                rng=rng,
            )

        LOGGER.info("Evaluation complete. Main report: %s", output_dir / "summary.json")
        print(json.dumps(jsonable({
            "output_dir": str(output_dir),
            "summary": str(output_dir / "summary.json"),
            "matched_go_terms": len(embedding.ids),
            "geometry": embedding.geometry,
            "views": list(views),
        }), indent=2, ensure_ascii=False))
        return 0

    except Exception as exc:
        LOGGER.exception("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
