#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def _add_model_path(project_root: Path) -> None:
    model_root = project_root / "nbs_models" / "nbs_protein_go"
    if not model_root.exists():
        raise FileNotFoundError(f"NBS model package not found: {model_root}")
    sys.path.insert(0, str(model_root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build memory-mapped GO->Protein inverted indices for LATENCE NBS. "
            "The candidate builder uses two streaming passes and is suitable "
            "for the 281,457,664-edge BP candidate relation."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--weak-graph-manifest", type=Path, required=True)
    parser.add_argument("--protein-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-edge-index", type=Path)
    parser.add_argument("--chunk-edges", type=int, default=2_000_000)
    parser.add_argument("--skip-candidate", action="store_true")
    parser.add_argument("--skip-pseudo", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _add_model_path(args.project_root.resolve())
    from nbs_pg.inverted_index import build_latence_go_protein_indices

    result = build_latence_go_protein_indices(
        args.weak_graph_manifest,
        args.protein_registry,
        args.output_dir,
        build_candidate=not args.skip_candidate,
        build_pseudo=not args.skip_pseudo,
        gold_edge_index_path=args.gold_edge_index,
        chunk_edges=args.chunk_edges,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
