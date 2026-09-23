#!/usr/bin/env python3
"""Prepare v0.4 full-ontology and direction-safe sampling inputs.

This does not rebuild the already generated GO->Protein inverted index.  Gold
supervision must be present in that manifest before training.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get(
    "LATENCE_PROJECT_ROOT",
    "/home/dataset-local/data_local/shaojiangyi/latence-project",
)).resolve()


def _run(script: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/nbs" / script)], cwd=ROOT, check=True)


def main() -> None:
    _run("run_prepare_boxsqel_full_ontology_for_nbs.py")
    _run("run_build_boxsqel_gg_relations.py")
    _run("run_build_pp_sampling_indices.py")


if __name__ == "__main__":
    main()
