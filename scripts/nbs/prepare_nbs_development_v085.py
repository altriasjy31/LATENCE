#!/usr/bin/env python3
"""Bind an existing external development set; this script does not create labels.

E/M must already be exported through prepare_nbs_stage1_references_v081.py for
this prepared input directory. ID lists are one protein / original GO ID per line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nbs_models/nbs_protein_go"))
from nbs_pg.full_task_development_v085 import build_contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-dir", "reference-dir", "labels", "label-protein-ids", "label-go-ids",
                 "stage1-training-ids", "final-test-ids", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    contract = build_contract(**{key: value for key, value in vars(args).items() if key != "output"})
    content = json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    if output.exists():
        if json.loads(output.read_text()) != contract:
            raise FileExistsError(f"Development contract already exists with different inputs: {output}")
        print(f"[development contract verified] {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an existing experiment's source contract.
    with output.open("x") as stream:
        stream.write(content)
    print(f"[development contract ready] {output}")
    print("Registry overlap and training/input provenance are additionally checked at runtime.")


if __name__ == "__main__":
    main()
