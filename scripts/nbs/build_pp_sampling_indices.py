#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build direction-safe P-P sampling CSR indices for NBS")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--pp-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-edges", type=int, default=2_000_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "nbs_models" / "nbs_protein_go"))
    from nbs_pg.sampling_indices import build_pp_sampling_indices
    print(build_pp_sampling_indices(
        args.pp_manifest,
        args.output_dir,
        chunk_edges=args.chunk_edges,
        overwrite=args.overwrite,
    ))


if __name__ == "__main__":
    main()
