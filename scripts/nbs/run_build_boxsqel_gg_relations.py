#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "LATENCE_PROJECT_ROOT",
    "/home/dataset-local/data_local/shaojiangyi/latence-project",
)).resolve()
ONTOLOGY_VERSION = os.environ.get("GO_ONTOLOGY_VERSION", "2022")
MANIFEST = Path(os.environ.get(
    "BOXSQEL_MANIFEST",
    PROJECT_ROOT / "outputs/go_emb/go_boxsqel_manifest_512.json",
))
PARSER_REPORT = Path(os.environ.get(
    "BOXSQEL_PARSER_REPORT",
    PROJECT_ROOT / "outputs/go_emb/go_boxsqel_parser_report_512.json",
))
OUTPUT_DIR = Path(os.environ.get(
    "NBS_FULL_GO_DIR",
    PROJECT_ROOT / "outputs/latence_nbs/ontology" / f"go_{ONTOLOGY_VERSION}" / "boxsqel_full",
))
ARTIFACT_SELECTION = os.environ.get("BOXSQEL_ARTIFACT_SELECTION", "best")
CHECKPOINT_OVERRIDE = os.environ.get("BOXSQEL_CHECKPOINT")
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"


def _resolve_checkpoint() -> Path:
    if CHECKPOINT_OVERRIDE:
        return Path(CHECKPOINT_OVERRIDE).resolve()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    key = f"{ARTIFACT_SELECTION}_checkpoint"
    if key not in payload["artifacts"]:
        raise KeyError(
            f"BoxSquaredEL manifest has no {key!r}; set BOXSQEL_CHECKPOINT explicitly"
        )
    path = Path(payload["artifacts"][key])
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    normalized = Path(payload["config"]["data_file"])
    if not normalized.is_absolute():
        normalized = PROJECT_ROOT / normalized
    script = PROJECT_ROOT / "scripts/nbs/build_boxsqel_gg_relations.py"
    argv = [
        sys.executable,
        str(script),
        "--project-root", str(PROJECT_ROOT),
        "--normalized-go", str(normalized),
        "--boxsqel-checkpoint", str(_resolve_checkpoint()),
        "--parser-report", str(PARSER_REPORT),
        "--output-dir", str(OUTPUT_DIR),
        "--ontology-version", ONTOLOGY_VERSION,
    ]
    if OVERWRITE:
        argv.append("--overwrite")
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
