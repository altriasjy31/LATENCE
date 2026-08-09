#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export complete task-space NBS probabilities for external/independent-test "
            "proteins. Training Q is only a mini-batch width: inference visits every task GO "
            "column in deterministic GO chunks."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protein-repr", type=Path, required=True)
    parser.add_argument("--base-values", type=Path, required=True)
    parser.add_argument("--base-values-are-logits", action="store_true")
    parser.add_argument("--protein-ids", type=Path, default=None)
    parser.add_argument("--candidate-go-index", type=Path, default=None)
    parser.add_argument("--candidate-edge-attr", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--go-chunk-size", type=int, default=256)
    parser.add_argument("--protein-batch-size", type=int, default=256)
    parser.add_argument("--support-per-query", type=int, default=2)
    parser.add_argument("--support-seed", type=int, default=3407)
    parser.add_argument("--probability-clip", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-proteins", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    from nbs_pg import NBSConfig, ProteinGONBSModel  # pylint: disable=import-outside-toplevel
    from nbs_pg.inference import (  # pylint: disable=import-outside-toplevel
        ExternalCandidateEvidenceStore,
        FullTaskInferenceConfig,
        export_full_task_probabilities,
    )
    from nbs_pg.local_loader import (  # pylint: disable=import-outside-toplevel
        LatenceNBSLocalGraphMaterializer,
        NBSLocalGraphSamplingConfig,
        build_latence_nbs_stores,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Inference is single-process by default. The graph/data contract is shared
    # with training, but no optimizer or scheduler is constructed.
    config["_distributed_runtime"] = {"rank": 0, "world_size": 1}
    stores = build_latence_nbs_stores(config)
    sampling_cfg = NBSLocalGraphSamplingConfig.from_mapping(config.get("local_sampling"))
    materializer = LatenceNBSLocalGraphMaterializer(stores, sampling_cfg)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config") or config.get("model", {})
    model = ProteinGONBSModel(
        NBSConfig(**model_config),
        protein_input_dim=int(config["model_inputs"]["protein_input_dim"]),
        go_box_dim=int(config["model_inputs"]["go_box_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    global_go_graph = materializer.build_global_go_graph().to(device)
    global_go_cache = model.make_go_cache(global_go_graph)

    representation = np.load(args.protein_repr, mmap_mode="r")
    base_values = np.load(args.base_values, mmap_mode="r")
    if args.limit_proteins is not None:
        n = min(int(args.limit_proteins), int(representation.shape[0]))
        representation = representation[:n]
        base_values = base_values[:n]
    if representation.shape[0] != base_values.shape[0]:
        raise ValueError("protein representation and base prediction rows disagree")

    evidence = None
    if (args.candidate_go_index is None) ^ (args.candidate_edge_attr is None):
        raise ValueError("candidate-go-index and candidate-edge-attr must be supplied together")
    if args.candidate_go_index is not None:
        evidence = ExternalCandidateEvidenceStore(
            args.candidate_go_index, args.candidate_edge_attr
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability_path = args.output_dir / "nbs_full_task_prob.f16.npy"
    inference_config = FullTaskInferenceConfig(
        go_chunk_size=int(args.go_chunk_size),
        protein_batch_size=int(args.protein_batch_size),
        support_per_query=int(args.support_per_query),
        support_seed=int(args.support_seed),
        probability_clip=float(args.probability_clip),
    )
    output = export_full_task_probabilities(
        model=model,
        stores=stores,
        materializer=materializer,
        global_go_cache=global_go_cache,
        external_repr=representation,
        base_values=base_values,
        output_path=probability_path,
        device=device,
        config=inference_config,
        base_values_are_logits=bool(args.base_values_are_logits),
        candidate_evidence_store=evidence,
        amp_dtype=torch.bfloat16,
    )

    output_ids = None
    if args.protein_ids is not None:
        ids = [line.strip() for line in args.protein_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = ids[: output.shape[0]]
        if len(ids) != output.shape[0]:
            raise ValueError("protein ID count does not match exported prediction rows")
        output_ids = args.output_dir / "protein_ids.txt"
        output_ids.write_text("\n".join(ids) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "exporter": "NBS full-task inductive feature-candidate inference v0.5.3",
        "task": config.get("task"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "inference_mode": "inductive_feature_candidate",
        "prediction_space": "complete_task_classifier_columns",
        "num_proteins": int(output.shape[0]),
        "num_task_go": int(output.shape[1]),
        "go_chunk_size": int(args.go_chunk_size),
        "protein_batch_size": int(args.protein_batch_size),
        "support_per_query": int(args.support_per_query),
        "base_values_are_logits": bool(args.base_values_are_logits),
        "candidate_evidence": "sparse_fixed_k" if evidence is not None else "absent_zero_channel",
        "output_probability": str(probability_path),
        "protein_ids": None if output_ids is None else str(output_ids),
        "uses_expert_probability_in_nbs_forward": False,
        "note": (
            "Every task GO column is scored. Training Q does not cap prediction labels. "
            "External candidates use the learned protein input projection with zero graph-relation "
            "source residuals; GO queries use the trained NBS support graph and full BoxSquaredEL cache."
        ),
    }
    manifest_path = args.output_dir / "nbs_full_task_prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
