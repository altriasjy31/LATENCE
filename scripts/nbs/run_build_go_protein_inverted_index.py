#!/usr/bin/env python3
"""Environment launcher for the LATENCE GO->Protein inverted-index build.

The v0.6 contract is intentionally strict: the default candidate source must
be the full-task, fixed-degree top-512 Protein--GO export.  Historical
rare-first/top-20 manifests can still be processed by the lower-level builder,
but this production launcher refuses them unless the caller explicitly turns
off the v0.6 contract check.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from build_go_protein_inverted_index import main


RUNNER_ID = "run_build_go_protein_inverted_index"
RUNNER_VERSION = "0.6.1-full-task-top512"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be boolean, got {raw!r}")


def _infer_project_root() -> Path:
    configured = (
        os.environ.get("LATENCE_PROJECT_ROOT")
        or os.environ.get("PROJECT_ROOT")
    )
    if configured:
        return Path(configured).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "nbs_models" / "nbs_protein_go").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]


def _resolve_under(base: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "weak-graph manifest does not exist: "
            f"{path}\nRun scripts/run_export_weak_graph_predictions.py with "
            "full_task/top-512 before building the inverted index."
        ) from error
    if not isinstance(value, Mapping):
        raise TypeError(f"weak-graph manifest must contain a JSON object: {path}")
    return value


def _validate_v060_candidate_contract(
    manifest: Mapping[str, Any],
    *,
    expected_scope: str,
    expected_topk: int,
) -> None:
    selector = (
        manifest.get("model_semantics", {})
        .get("protein_go_selector", {})
    )
    candidate = manifest.get("backbone_candidate_edges", {})
    scope = selector.get("scope", candidate.get("selector_scope"))
    requested_topk = selector.get("requested_topk")
    if scope != expected_scope:
        raise ValueError(
            "weak-graph candidate scope mismatch: "
            f"expected {expected_scope!r}, observed {scope!r}. "
            "The manifest is probably a historical rare-first export."
        )
    if requested_topk is None or int(requested_topk) != expected_topk:
        raise ValueError(
            "weak-graph candidate top-k mismatch: "
            f"expected {expected_topk}, observed {requested_topk!r}."
        )
    universe = selector.get("candidate_universe_count")
    num_go = int(manifest.get("go_registry", {}).get("num_terms", -1))
    if universe is None or int(universe) != num_go:
        raise ValueError(
            "full-task candidate universe mismatch: "
            f"candidate_universe_count={universe!r}, task GO terms={num_go}."
        )
    roles = manifest.get("roles", [])
    if not roles:
        raise ValueError("weak-graph manifest contains no role records")
    for role in roles:
        degree_min = int(role.get("candidate_degree_min", -1))
        degree_max = int(role.get("candidate_degree_max", -1))
        if degree_min != expected_topk or degree_max != expected_topk:
            raise ValueError(
                f"role={role.get('role')!r} candidate degree mismatch: "
                f"min={degree_min}, max={degree_max}, expected={expected_topk}."
            )


PROJECT_ROOT = _infer_project_root()
TASK = os.environ.get("TASK", "bp")
RUN_TAG = os.environ.get(
    "RUN_TAG", "bp_weak_detr_v3_expert_prob_warmstart340_to400"
)
EPOCH = int(os.environ.get("EPOCH", "100"))
CHUNK_EDGES = int(os.environ.get("CHUNK_EDGES", "2000000"))
OVERWRITE = _env_bool("OVERWRITE", False)
GOLD_EDGE_INDEX = os.environ.get("GOLD_EDGE_INDEX")
MERGE_EXISTING = _env_bool("MERGE_EXISTING", True)
GOLD_ONLY = _env_bool("GOLD_ONLY", True)
REQUIRE_V060_CANDIDATES = _env_bool("REQUIRE_V060_CANDIDATES", True)
EXPECTED_CANDIDATE_SCOPE = os.environ.get(
    "PROTEIN_GO_SELECTOR_SCOPE", "full_task"
).strip()
EXPECTED_CANDIDATE_TOPK = int(os.environ.get("PROTEIN_GO_TOPK", "512"))

# Renamed NBS data root.  The runner deliberately does not fall back silently
# to outputs/latence_nn_pp after the project-wide migration.
TASK_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "latence_nbs"
    / RUN_TAG
    / f"epoch{EPOCH}"
    / TASK
)
WEAK_GRAPH_DIR = os.environ.get(
    "WEAK_GRAPH_DIR", "weak_graph_predictions_full_task_top512"
)
weak_graph_root = _resolve_under(TASK_ROOT, WEAK_GRAPH_DIR)
WEAK_GRAPH_MANIFEST = _resolve_under(
    TASK_ROOT,
    os.environ.get(
        "WEAK_GRAPH_MANIFEST",
        weak_graph_root / "weak_graph_predictions_manifest.json",
    ),
)
PROTEIN_REGISTRY = _resolve_under(
    TASK_ROOT,
    os.environ.get("PROTEIN_REGISTRY", TASK_ROOT / "features/protein_registry.csv"),
)
OUTPUT_DIR = _resolve_under(
    TASK_ROOT,
    os.environ.get(
        "GO_PROTEIN_INDEX_OUTPUT_DIR",
        TASK_ROOT / "nbs_indices" / "go_protein_full_task_top512",
    ),
)


def run() -> int:
    if CHUNK_EDGES <= 0:
        raise ValueError("CHUNK_EDGES must be positive")
    if GOLD_ONLY and not GOLD_EDGE_INDEX:
        raise ValueError("GOLD_ONLY=1 requires GOLD_EDGE_INDEX")
    for path, label in (
        (PROTEIN_REGISTRY, "protein registry"),
        (Path(GOLD_EDGE_INDEX).expanduser().resolve() if GOLD_EDGE_INDEX else None,
         "gold Protein-GO edge index"),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    weak_manifest = _load_manifest(WEAK_GRAPH_MANIFEST)
    if REQUIRE_V060_CANDIDATES:
        _validate_v060_candidate_contract(
            weak_manifest,
            expected_scope=EXPECTED_CANDIDATE_SCOPE,
            expected_topk=EXPECTED_CANDIDATE_TOPK,
        )

    print(
        f"[{RUNNER_ID}] version={RUNNER_VERSION}\n"
        f"PROJECT_ROOT={PROJECT_ROOT}\n"
        f"WEAK_GRAPH_MANIFEST={WEAK_GRAPH_MANIFEST}\n"
        f"PROTEIN_REGISTRY={PROTEIN_REGISTRY}\n"
        f"OUTPUT_DIR={OUTPUT_DIR}\n"
        f"GOLD_EDGE_INDEX={GOLD_EDGE_INDEX}\n"
        f"GOLD_ONLY={int(GOLD_ONLY)} MERGE_EXISTING={int(MERGE_EXISTING)} "
        f"OVERWRITE={int(OVERWRITE)}\n"
        f"REQUIRE_V060_CANDIDATES={int(REQUIRE_V060_CANDIDATES)} "
        f"EXPECTED_SCOPE={EXPECTED_CANDIDATE_SCOPE} "
        f"EXPECTED_TOPK={EXPECTED_CANDIDATE_TOPK}",
        flush=True,
    )
    argv = [
        "--project-root",
        str(PROJECT_ROOT),
        "--weak-graph-manifest",
        str(WEAK_GRAPH_MANIFEST),
        "--protein-registry",
        str(PROTEIN_REGISTRY),
        "--output-dir",
        str(OUTPUT_DIR),
        "--chunk-edges",
        str(CHUNK_EDGES),
    ]
    if GOLD_EDGE_INDEX:
        argv.extend(["--gold-edge-index", GOLD_EDGE_INDEX])
        if GOLD_ONLY:
            argv.extend(["--skip-candidate", "--skip-pseudo"])
    if MERGE_EXISTING and (OUTPUT_DIR / "go_protein_inverted_index_manifest.json").exists():
        argv.append("--merge-existing-manifest")
    if OVERWRITE:
        argv.append("--overwrite")
    return main(argv)


if __name__ == "__main__":
    sys.exit(run())
