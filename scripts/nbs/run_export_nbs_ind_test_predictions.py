#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    task = os.environ.get("TASK", "bp")
    config = Path(os.environ.get(
        "NBS_TRAIN_CONFIG",
        f"nbs_models/nbs_protein_go/configs/{task}_fixed_epoch_v0.6.0.json",
    ))
    checkpoint = os.environ.get("NBS_CHECKPOINT")
    protein_repr = os.environ.get("NBS_IND_TEST_REPR")
    base_values = os.environ.get("NBS_IND_TEST_BASE_VALUES")
    output_dir = os.environ.get("NBS_PRED_OUTPUT_DIR", f"outputs/latence_nbs_eval/{task}_nbs")
    missing = [name for name, value in (
        ("NBS_CHECKPOINT", checkpoint),
        ("NBS_IND_TEST_REPR", protein_repr),
        ("NBS_IND_TEST_BASE_VALUES", base_values),
    ) if not value]
    if missing:
        raise SystemExit(
            "Missing required environment variables: " + ", ".join(missing) + "\n"
            "NBS_IND_TEST_REPR must be the Stage-1 pooled 2048-D representation for ind_test; "
            "NBS_IND_TEST_BASE_VALUES must be the complete Stage-1 backbone probability/logit matrix."
        )

    cmd = [
        sys.executable,
        str(root / "scripts/nbs/export_nbs_full_task_predictions.py"),
        "--config", str(config),
        "--checkpoint", str(checkpoint),
        "--protein-repr", str(protein_repr),
        "--base-values", str(base_values),
        "--output-dir", str(output_dir),
        "--go-chunk-size", os.environ.get("NBS_EVAL_GO_CHUNK", "256"),
        "--protein-batch-size", os.environ.get("NBS_EVAL_PROTEIN_BATCH", "256"),
        "--support-per-query", os.environ.get("NBS_EVAL_SUPPORT_PER_QUERY", "2"),
        "--device", os.environ.get("NBS_EVAL_DEVICE", "cuda:0"),
    ]
    protein_ids = os.environ.get("NBS_IND_TEST_PROTEIN_IDS")
    if protein_ids:
        cmd += ["--protein-ids", protein_ids]
    if os.environ.get("NBS_IND_TEST_BASE_IS_LOGITS", "0") == "1":
        cmd += ["--base-values-are-logits"]
    candidate_go = os.environ.get("NBS_IND_TEST_CANDIDATE_GO")
    candidate_attr = os.environ.get("NBS_IND_TEST_CANDIDATE_ATTR")
    if candidate_go or candidate_attr:
        if not candidate_go or not candidate_attr:
            raise SystemExit("Set both NBS_IND_TEST_CANDIDATE_GO and NBS_IND_TEST_CANDIDATE_ATTR")
        cmd += ["--candidate-go-index", candidate_go, "--candidate-edge-attr", candidate_attr]
    pp_core = os.environ.get("NBS_IND_TEST_CORE_REPR")
    pp_neighbors = os.environ.get("NBS_IND_TEST_PP_NEIGHBORS")
    pp_attr = os.environ.get("NBS_IND_TEST_PP_ATTR")
    if pp_core or pp_neighbors or pp_attr:
        if not (pp_core and pp_neighbors and pp_attr):
            raise SystemExit(
                "Set NBS_IND_TEST_CORE_REPR, NBS_IND_TEST_PP_NEIGHBORS and "
                "NBS_IND_TEST_PP_ATTR together"
            )
        cmd += [
            "--external-pp-core-repr", pp_core,
            "--external-pp-neighbors", pp_neighbors,
            "--external-pp-edge-attr", pp_attr,
        ]
    input_manifest = os.environ.get("NBS_IND_TEST_INPUT_MANIFEST")
    if input_manifest:
        cmd += ["--input-manifest", input_manifest]
    if os.environ.get("NBS_SAVE_INFERENCE_DIAGNOSTICS", "1") == "1":
        cmd += ["--save-diagnostics"]
    limit = os.environ.get("NBS_EVAL_LIMIT_PROTEINS")
    if limit:
        cmd += ["--limit-proteins", limit]
    print("[NBS full-task inference]", " ".join(cmd))
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
