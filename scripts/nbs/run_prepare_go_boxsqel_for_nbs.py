#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(
        os.environ.get("LATENCE_PROJECT_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    task = os.environ.get("TASK", "bp")
    run_tag = os.environ.get(
        "RUN_TAG", "bp_weak_detr_v3_expert_prob_warmstart340_to400"
    )
    epoch = int(os.environ.get("EPOCH", "100"))
    data_root = project_root / "outputs" / "latence_nbs" / run_tag / f"epoch{epoch}" / task
    manifest = Path(
        os.environ.get(
            "GO_BOXSQEL_MANIFEST",
            project_root / "outputs" / "go_emb" / "go_boxsqel_manifest_512.json",
        )
    )
    parser_report = Path(
        os.environ.get(
            "GO_BOXSQEL_PARSER_REPORT",
            project_root / "outputs" / "go_emb" / "go_boxsqel_parser_report_512.json",
        )
    )
    go_registry = Path(
        os.environ.get(
            "GO_REGISTRY",
            data_root / "gg_relations" / "go_registry.tsv",
        )
    )
    output_dir = Path(
        os.environ.get(
            "GO_BOXSQEL_ALIGNMENT_DIR",
            data_root / "nbs_indices" / "go_boxsqel_512",
        )
    )
    command = [
        sys.executable,
        str(project_root / "scripts" / "nbs" / "prepare_go_boxsqel_for_nbs.py"),
        "--manifest",
        str(manifest),
        "--parser-report",
        str(parser_report),
        "--go-registry",
        str(go_registry),
        "--output-dir",
        str(output_dir),
        "--project-root",
        str(project_root),
        "--artifact-selection",
        os.environ.get("GO_BOXSQEL_SELECTION", "best"),
    ]
    if os.environ.get("GO_BOXSQEL_ALLOW_MISSING", "0") == "1":
        command.append("--allow-missing")
    if os.environ.get("OVERWRITE", "0") == "1":
        command.append("--overwrite")
    subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
