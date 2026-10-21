#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "LATENCE_PROJECT_ROOT",
    "/home/dataset-local/data_local/shaojiangyi/latence-project",
)).resolve()
ONTOLOGY_VERSION = os.environ.get("GO_ONTOLOGY_VERSION", "2022")
MANIFEST = Path(os.environ.get("BOXSQEL_MANIFEST", PROJECT_ROOT / "outputs/go_emb/go_boxsqel_manifest_512.json"))
PARSER_REPORT = Path(os.environ.get("BOXSQEL_PARSER_REPORT", PROJECT_ROOT / "outputs/go_emb/go_boxsqel_parser_report_512.json"))
OUTPUT_DIR = Path(os.environ.get(
    "NBS_FULL_GO_DIR",
    PROJECT_ROOT / "outputs/latence_nbs/ontology" / f"go_{ONTOLOGY_VERSION}" / "boxsqel_full",
))
ARTIFACT_SELECTION = os.environ.get("BOXSQEL_ARTIFACT_SELECTION", "best")
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"


def main() -> int:
    script = PROJECT_ROOT / "scripts/nbs/prepare_boxsqel_full_ontology_for_nbs.py"
    argv = [
        sys.executable, str(script),
        "--project-root", str(PROJECT_ROOT),
        "--manifest", str(MANIFEST),
        "--parser-report", str(PARSER_REPORT),
        "--output-dir", str(OUTPUT_DIR),
        "--ontology-version", ONTOLOGY_VERSION,
        "--artifact-selection", ARTIFACT_SELECTION,
    ]
    if OVERWRITE:
        argv.append("--overwrite")
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
