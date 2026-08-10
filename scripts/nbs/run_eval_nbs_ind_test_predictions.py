#!/usr/bin/env python3
"""Environment-driven end-to-end NBS independent-test inference/evaluation.

Backward compatibility: when ``NBS_IND_TEST_PROB`` is supplied this remains an
evaluation-only wrapper.  Otherwise it prepares Stage-1 artifacts from FASTA or
pickle, retrieves isolated test-to-core neighbours, exports full-task NBS
predictions, and optionally runs diagnostic evaluation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def enabled(name: str, default: bool) -> bool:
    value = env(name, "1" if default else "0").lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def run(cmd: list[str], root: Path, label: str) -> None:
    print(f"[{label}]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, check=True)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def resolve_data_root(root: Path, config_path: Path) -> Path:
    explicit = env("NBS_DATA_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    config = load_json(config_path)
    value = Path(str(config["data"]["root"])).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def evaluation_command(
    *,
    root: Path,
    task: str,
    metadata: Path,
    nbs_prob: Path,
    output_dir: Path,
    protein_ids: Path | None,
    backbone_prob: Path | None,
    prediction_dir: Path | None,
    input_dir: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(root / "scripts" / "nbs" / "eval_nbs_ind_test_predictions.py"),
        "--task", task,
        "--metadata-file", str(metadata),
        "--nbs-prob", str(nbs_prob),
        "--output-dir", str(output_dir),
    ]
    optional_paths: list[tuple[str, Path | None]] = [
        ("--protein-ids", protein_ids),
        ("--backbone-prob", backbone_prob),
    ]
    modelout = env("MODELOUT_IND_TEST_PROB")
    if modelout:
        optional_paths.append(("--modelout-prob", Path(modelout)))
    train_counts = env("TRAIN_GO_COUNTS")
    if train_counts:
        optional_paths.append(("--train-counts", Path(train_counts)))
    go_registry = env("NBS_GO_REGISTRY")
    if go_registry:
        optional_paths.append(("--go-registry", Path(go_registry)))
    elif input_dir is not None and (input_dir / "ind_test_input_manifest.json").is_file():
        input_contract = load_json(input_dir / "ind_test_input_manifest.json")
        recorded_go_registry = input_contract.get("registries", {}).get("go_registry")
        if recorded_go_registry:
            optional_paths.append(("--go-registry", Path(str(recorded_go_registry))))

    if input_dir is not None:
        optional_paths.extend(
            [
                ("--candidate-go-index", input_dir / "candidate_go_index.i32.npy"),
                ("--input-manifest", input_dir / "ind_test_input_manifest.json"),
            ]
        )
    if prediction_dir is not None:
        optional_paths.extend(
            [
                ("--eligible-go-index", prediction_dir / "eligible_go_indices.i32.npy"),
                ("--applied-logit-delta", prediction_dir / "nbs_applied_logit_delta.f16.npy"),
                ("--delta-gate", prediction_dir / "nbs_delta_gate.f16.npy"),
                ("--routing-source-weights", prediction_dir / "nbs_routing_source_weights.f16.npy"),
                ("--routing-null-weight", prediction_dir / "nbs_routing_null_weight.f16.npy"),
                ("--routing-source-names", prediction_dir / "nbs_routing_source_names.json"),
                ("--prediction-manifest", prediction_dir / "nbs_full_task_prediction_manifest.json"),
            ]
        )
    for flag, path in optional_paths:
        if path is not None and path.is_file():
            cmd.extend([flag, str(path)])
    if env("NBS_AUPRC_MODE"):
        cmd.extend(["--auprc-mode", env("NBS_AUPRC_MODE")])
    if env("NBS_THRESHOLD_STEP"):
        cmd.extend(["--threshold-step", env("NBS_THRESHOLD_STEP")])
    return cmd


def main() -> None:
    root = Path(env("LATENCE_PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
    task = env("TASK", "bp").lower()
    if task not in {"bp", "mf", "cc"}:
        raise ValueError("TASK must be bp, mf or cc")
    output = Path(env("NBS_EVAL_OUTPUT_DIR", f"outputs/latence_nbs_eval/{task}")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_override = env("METADATA_FILE")
    metadata = Path(
        metadata_override or str(root / "data" / "unidata_with_exp_train_pseudo.pkl")
    ).resolve()
    run_evaluation = enabled("NBS_RUN_EVALUATION", True)

    # Legacy/precomputed result path: evaluate exactly as before, with new
    # diagnostics accepted when their paths are supplied.
    existing_nbs_prob = env("NBS_IND_TEST_PROB")
    if existing_nbs_prob:
        protein_ids = env("NBS_IND_TEST_PROTEIN_IDS")
        backbone = env("BACKBONE_IND_TEST_PROB")
        cmd = evaluation_command(
            root=root,
            task=task,
            metadata=metadata,
            nbs_prob=Path(existing_nbs_prob).resolve(),
            output_dir=output,
            protein_ids=None if not protein_ids else Path(protein_ids).resolve(),
            backbone_prob=None if not backbone else Path(backbone).resolve(),
            prediction_dir=None,
            input_dir=None,
        )
        run(cmd, root, "NBS evaluation-only")
        return

    config_path = Path(
        env(
            "NBS_TRAIN_CONFIG",
            f"nbs_models/nbs_protein_go/configs/{task}_fixed_epoch_v0.5.6.json",
        )
    )
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    nbs_checkpoint = env("NBS_CHECKPOINT")
    stage1_checkpoint = env("STAGE1_CHECKPOINT")
    fasta = env("NBS_IND_TEST_FASTA")
    if not nbs_checkpoint or not stage1_checkpoint:
        raise ValueError("NBS_CHECKPOINT and STAGE1_CHECKPOINT are required")
    if not fasta and not metadata.is_file():
        raise ValueError("provide NBS_IND_TEST_FASTA or a METADATA_FILE containing sequences")

    prediction_dir = Path(
        env("NBS_PRED_OUTPUT_DIR", str(output / "predictions"))
    ).resolve()
    prediction_dir.mkdir(parents=True, exist_ok=True)
    limit = env("NBS_EVAL_LIMIT_PROTEINS")
    if limit and run_evaluation:
        raise ValueError(
            "NBS_EVAL_LIMIT_PROTEINS is a smoke-inference option; set "
            "NBS_RUN_EVALUATION=0 because the label file contains the full test set"
        )
    use_tmp = enabled("NBS_USE_TMP_WORKSPACE", False)
    keep_tmp = enabled("NBS_KEEP_TMP_WORKSPACE", False)
    temporary_created = False
    if env("NBS_IND_TEST_WORK_DIR"):
        input_dir = Path(env("NBS_IND_TEST_WORK_DIR")).resolve()
    elif use_tmp:
        tmp_root_value = env("NBS_TMP_ROOT")
        tmp_root = None if not tmp_root_value else Path(tmp_root_value).resolve()
        if tmp_root is not None:
            tmp_root.mkdir(parents=True, exist_ok=True)
        input_dir = Path(tempfile.mkdtemp(prefix="latence_nbs_ind_", dir=tmp_root))
        temporary_created = True
    else:
        input_dir = (output / "inductive_inputs").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    try:
        data_root = resolve_data_root(root, config_path)
        resolved_config = load_json(config_path)
        similar_fanouts = [
            int(value)
            for value in resolved_config.get("local_sampling", {}).get(
                "similar_to_fanouts", []
            )
            if int(value) > 0
        ]
        if not similar_fanouts:
            raise ValueError("NBS config has no positive local_sampling.similar_to_fanouts")
        pp_topk = env("NBS_IND_TEST_PP_TOPK", str(max(similar_fanouts)))
        prepare_cmd = [
            sys.executable,
            str(root / "scripts" / "nbs" / "prepare_nbs_ind_test_inputs.py"),
            "--project-root", str(root),
            "--task", task,
            "--stage1-checkpoint", str(Path(stage1_checkpoint).resolve()),
            "--data-root", str(data_root),
            "--output-dir", str(input_dir),
            "--cache-policy", env("NBS_INPUT_CACHE_POLICY", "reuse"),
            "--batch-size", env("STAGE1_EVAL_BATCH", "8"),
            "--device", env("STAGE1_EVAL_DEVICE", env("NBS_EVAL_DEVICE", "cuda:0")),
            "--pp-topk", pp_topk,
            "--faiss-backend", env("NBS_IND_TEST_FAISS_BACKEND", "faiss"),
            "--faiss-gpu-id", env("NBS_IND_TEST_FAISS_GPU_ID", "-1"),
        ]
        if fasta:
            prepare_cmd.extend(["--fasta", str(Path(fasta).resolve())])
        # Formal evaluation aligns FASTA to metadata. A service request with
        # evaluation disabled accepts arbitrary FASTA unless the caller
        # explicitly supplied METADATA_FILE as an alignment contract.
        if metadata.is_file() and (
            not fasta or run_evaluation or bool(metadata_override)
        ):
            prepare_cmd.extend(["--metadata-file", str(metadata)])
        train_args = env("STAGE1_TRAIN_ARGS_JSON")
        if train_args:
            prepare_cmd.extend(["--train-args-json", train_args])
        for env_name, flag in (
            ("NBS_REPRESENTATION_MANIFEST", "--representation-manifest"),
            ("NBS_WEAK_GRAPH_MANIFEST", "--weak-graph-manifest"),
            ("NBS_GO_REGISTRY", "--go-registry"),
            ("NBS_PICKLE_PROTEIN_KEY", "--pickle-protein-key"),
            ("NBS_PICKLE_SEQUENCE_KEY", "--pickle-sequence-key"),
            ("STAGE1_MODEL_CONFIG", "--stage1-model-config"),
            ("STAGE1_MSA_INDEX", "--msa-index"),
        ):
            value = env(env_name)
            if value:
                prepare_cmd.extend([flag, value])
        if enabled("STAGE1_EVAL_NO_AMP", False):
            prepare_cmd.append("--no-amp")
        run(prepare_cmd, root, "Prepare inductive inputs")

        input_manifest = load_json(input_dir / "ind_test_input_manifest.json")
        core_repr = Path(input_manifest["external_pp"]["core_representation"])
        export_cmd = [
            sys.executable,
            str(root / "scripts" / "nbs" / "export_nbs_full_task_predictions.py"),
            "--config", str(config_path),
            "--checkpoint", str(Path(nbs_checkpoint).resolve()),
            "--protein-repr", str(input_dir / "ind_test_repr.f16.npy"),
            "--base-values", str(input_dir / "backbone_ind_test_prob.f16.npy"),
            "--protein-ids", str(input_dir / "protein_ids.txt"),
            "--candidate-go-index", str(input_dir / "candidate_go_index.i32.npy"),
            "--candidate-edge-attr", str(input_dir / "candidate_edge_attr.f32.npy"),
            "--external-pp-core-repr", str(core_repr),
            "--external-pp-neighbors", str(input_dir / "test_core_neighbors.i32.npy"),
            "--external-pp-edge-attr", str(input_dir / "test_core_edge_attr.f32.npy"),
            "--input-manifest", str(input_dir / "ind_test_input_manifest.json"),
            "--output-dir", str(prediction_dir),
            "--go-chunk-size", env("NBS_EVAL_GO_CHUNK", "256"),
            "--protein-batch-size", env("NBS_EVAL_PROTEIN_BATCH", "256"),
            "--support-per-query", env("NBS_EVAL_SUPPORT_PER_QUERY", "2"),
            "--device", env("NBS_EVAL_DEVICE", "cuda:0"),
        ]
        if enabled("NBS_SAVE_INFERENCE_DIAGNOSTICS", True):
            export_cmd.append("--save-diagnostics")
        if limit:
            export_cmd.extend(["--limit-proteins", limit])
        run(export_cmd, root, "NBS full-task inductive inference")

        if run_evaluation:
            if not metadata.is_file():
                raise FileNotFoundError("METADATA_FILE with independent-test labels is required for evaluation")
            eval_cmd = evaluation_command(
                root=root,
                task=task,
                metadata=metadata,
                nbs_prob=prediction_dir / "nbs_full_task_prob.f16.npy",
                output_dir=output,
                protein_ids=prediction_dir / "protein_ids.txt",
                backbone_prob=input_dir / "backbone_ind_test_prob.f16.npy",
                prediction_dir=prediction_dir,
                input_dir=input_dir,
            )
            run(eval_cmd, root, "NBS independent-test diagnostics")
    finally:
        if temporary_created and not keep_tmp:
            shutil.rmtree(input_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
