#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build full BoxSquaredEL-class G-G relations from go.norm. "
            "The node space is the checkpoint's complete class mapping rather "
            "than a BP/MF/CC classifier subset."
        )
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--normalized-go", required=True)
    parser.add_argument("--boxsqel-checkpoint", required=True)
    parser.add_argument("--parser-report", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ontology-version", default="2022")
    parser.add_argument("--allow-unparsed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root / "nbs_models" / "nbs_protein_go"))
    from nbs_pg.boxsqel_relations import build_boxsqel_gg_relations

    manifest = build_boxsqel_gg_relations(
        args.normalized_go,
        args.boxsqel_checkpoint,
        args.output_dir,
        parser_report_path=args.parser_report,
        ontology_version=args.ontology_version,
        strict=not args.allow_unparsed,
        overwrite=args.overwrite,
    )
    print(manifest)


if __name__ == "__main__":
    main()
