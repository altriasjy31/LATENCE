#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all BoxSquaredEL classes and boxes for NBS")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parser-report", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ontology-version", default="2022")
    parser.add_argument("--artifact-selection", choices=("selected", "best", "final"), default="best")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "nbs_models" / "nbs_protein_go"))
    from nbs_pg.boxsqel_manifest import export_boxsqel_full_ontology
    output = export_boxsqel_full_ontology(
        args.manifest,
        args.output_dir,
        parser_report_path=args.parser_report,
        project_root=root,
        artifact_selection=args.artifact_selection,
        ontology_version=args.ontology_version,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
