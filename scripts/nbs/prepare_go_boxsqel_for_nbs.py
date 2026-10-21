#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align a trained BoxSquaredEL ontology embedding to the immutable "
            "LATENCE classifier GO index space."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parser-report", default=None)
    parser.add_argument("--go-registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--artifact-selection",
        choices=("selected", "best", "final"),
        default="best",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root or _project_root_from_script()).resolve()
    package_root = project_root / "nbs_models" / "nbs_protein_go"
    sys.path.insert(0, str(package_root))
    from nbs_pg.boxsqel_manifest import align_boxsqel_to_go_registry

    output = align_boxsqel_to_go_registry(
        args.manifest,
        args.go_registry,
        args.output_dir,
        parser_report_path=args.parser_report,
        project_root=project_root,
        artifact_selection=args.artifact_selection,
        strict=not args.allow_missing,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
