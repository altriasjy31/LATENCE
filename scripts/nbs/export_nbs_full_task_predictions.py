#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


EXPECTED_INDUCTIVE_API_VERSION = 5
EXPECTED_INDUCTIVE_ROUTING_API_VERSION = 1


def validate_inductive_inference_api(
    inference_api: ModuleType | Any,
    model_class: type[Any],
    matcher_class: type[Any],
) -> int:
    """Reject mixed exporter/inference/model deployments with one clear error."""
    required_inference = (
        "ExternalCandidateEvidenceStore",
        "ExternalPPNeighborhoodStore",
        "FullTaskInferenceConfig",
        "export_full_task_probabilities",
    )
    missing = [name for name in required_inference if not hasattr(inference_api, name)]
    inference_version = getattr(
        inference_api, "NBS_INDUCTIVE_INFERENCE_API_VERSION", None
    )
    model_version = getattr(model_class, "INDUCTIVE_INFERENCE_API_VERSION", None)
    score = getattr(model_class, "score_external_candidates", None)
    score_parameters = set() if score is None else set(inspect.signature(score).parameters)
    missing_score_parameters = sorted(
        {"neighbor_x", "neighbor_edge_attr", "neighbor_fanouts"} - score_parameters
    )
    matcher_version = getattr(
        matcher_class, "INDUCTIVE_ROUTING_API_VERSION", None
    )
    matcher_forward = getattr(matcher_class, "forward", None)
    matcher_parameters = (
        set()
        if matcher_forward is None
        else set(inspect.signature(matcher_forward).parameters)
    )
    problems: list[str] = []
    if missing:
        problems.append(f"inference.py missing symbols={missing}")
    if inference_version != EXPECTED_INDUCTIVE_API_VERSION:
        problems.append(
            "inference.py API version="
            f"{inference_version!r}, expected={EXPECTED_INDUCTIVE_API_VERSION}"
        )
    if model_version != EXPECTED_INDUCTIVE_API_VERSION:
        problems.append(
            f"model.py API version={model_version!r}, "
            f"expected={EXPECTED_INDUCTIVE_API_VERSION}"
        )
    if missing_score_parameters:
        problems.append(
            "ProteinGONBSModel.score_external_candidates missing parameters="
            f"{missing_score_parameters}"
        )
    if matcher_version != EXPECTED_INDUCTIVE_ROUTING_API_VERSION:
        problems.append(
            f"matcher.py routing API version={matcher_version!r}, "
            f"expected={EXPECTED_INDUCTIVE_ROUTING_API_VERSION}"
        )
    if "routing_hierarchy" not in matcher_parameters:
        problems.append(
            "NBSGatedDeltaAttnRes.forward missing parameter='routing_hierarchy'"
        )
    if problems:
        detail = "; ".join(problems)
        raise RuntimeError(
            "Incompatible NBS isolated-inductive inference deployment: "
            f"{detail}. This normally means a newer exporter was copied over an "
            "older nbs_pg package. Replace inference.py, model.py and matcher.py "
            "together with the exporter/launcher patch; do not delete "
            "ExternalPPNeighborhoodStore "
            "or fall back to feature-only inference for the formal result."
        )
    return EXPECTED_INDUCTIVE_API_VERSION


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
    parser.add_argument("--external-pp-core-repr", type=Path, default=None)
    parser.add_argument("--external-pp-neighbors", type=Path, default=None)
    parser.add_argument("--external-pp-edge-attr", type=Path, default=None)
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--go-chunk-size", type=int, default=256)
    parser.add_argument("--protein-batch-size", type=int, default=256)
    parser.add_argument("--support-per-query", type=int, default=2)
    parser.add_argument("--support-seed", type=int, default=3407)
    parser.add_argument("--probability-clip", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--min-free-gpu-gb",
        type=float,
        default=8.0,
        help="fail before model materialization when the selected CUDA device lacks headroom",
    )
    parser.add_argument("--limit-proteins", type=int, default=None)
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Save applied logit delta, delta gate and per-GO routing arrays.",
    )
    args = parser.parse_args()
    if args.min_free_gpu_gb < 0:
        raise ValueError("--min-free-gpu-gb must be non-negative")

    project_root = Path(__file__).resolve().parents[2]
    import sys

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "nbs_models" / "nbs_protein_go"))
    from nbs_pg import NBSConfig, ProteinGONBSModel  # pylint: disable=import-outside-toplevel
    import nbs_pg.inference as inference_api  # pylint: disable=import-outside-toplevel
    from nbs_pg.matcher import (  # pylint: disable=import-outside-toplevel
        NBSGatedDeltaAttnRes,
    )

    api_version = validate_inductive_inference_api(
        inference_api, ProteinGONBSModel, NBSGatedDeltaAttnRes
    )
    ExternalCandidateEvidenceStore = inference_api.ExternalCandidateEvidenceStore
    ExternalPPNeighborhoodStore = inference_api.ExternalPPNeighborhoodStore
    FullTaskInferenceConfig = inference_api.FullTaskInferenceConfig
    export_full_task_probabilities = inference_api.export_full_task_probabilities
    print(f"[NBS inference API] version={api_version} status=ok", flush=True)
    from nbs_pg.local_loader import (  # pylint: disable=import-outside-toplevel
        LatenceNBSLocalGraphMaterializer,
        NBSLocalGraphSamplingConfig,
        build_latence_nbs_stores,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and float(args.min_free_gpu_gb) > 0:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = float(free_bytes) / (1024 ** 3)
        total_gb = float(total_bytes) / (1024 ** 3)
        print(
            f"[CUDA preflight] device={device} free={free_gb:.2f}GB/"
            f"{total_gb:.2f}GB required={float(args.min_free_gpu_gb):.2f}GB",
            flush=True,
        )
        if free_gb < float(args.min_free_gpu_gb):
            raise RuntimeError(
                f"NBS inference device {device} has only {free_gb:.2f}GB free, "
                f"below --min-free-gpu-gb={float(args.min_free_gpu_gb):.2f}. "
                "Select another NBS_EVAL_DEVICE or wait for training."
            )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen_support_block = int(config.get("inference", {}).get("go_chunk_size", 256))
    if int(args.go_chunk_size) != frozen_support_block:
        raise ValueError(
            "--go-chunk-size changes the sampled support graph and is therefore a scientific "
            f"inference parameter, not a free memory knob. Use the frozen value "
            f"{frozen_support_block} from the resolved config."
        )
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
        if evidence.feature_dim != int(model.config.candidate_evidence_dim):
            raise ValueError(
                "candidate evidence width differs from the trained NBS contract: "
                f"input={evidence.feature_dim}, model={model.config.candidate_evidence_dim}"
            )

    pp_values = (
        args.external_pp_core_repr,
        args.external_pp_neighbors,
        args.external_pp_edge_attr,
    )
    if any(value is not None for value in pp_values) and not all(
        value is not None for value in pp_values
    ):
        raise ValueError(
            "external-pp-core-repr, external-pp-neighbors and external-pp-edge-attr "
            "must be supplied together"
        )
    pp_neighborhood = None
    pp_neighbor_fanouts = None
    if all(value is not None for value in pp_values):
        pp_neighborhood = ExternalPPNeighborhoodStore(*pp_values)
        pp_neighbor_fanouts = tuple(
            int(value)
            for value in sampling_cfg.similar_to_fanouts[: model.config.num_layers]
        )
        if len(pp_neighbor_fanouts) != model.config.num_layers:
            raise ValueError(
                "local_sampling.similar_to_fanouts must cover every NBS layer"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability_path = args.output_dir / "nbs_full_task_prob.f16.npy"
    logit_delta_path = (
        args.output_dir / "nbs_applied_logit_delta.f16.npy"
        if args.save_diagnostics
        else None
    )
    delta_gate_path = (
        args.output_dir / "nbs_delta_gate.f16.npy" if args.save_diagnostics else None
    )
    routing_source_path = (
        args.output_dir / "nbs_routing_source_weights.f16.npy"
        if args.save_diagnostics
        else None
    )
    routing_null_path = (
        args.output_dir / "nbs_routing_null_weight.f16.npy"
        if args.save_diagnostics
        else None
    )
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
        pp_neighborhood_store=pp_neighborhood,
        pp_neighbor_fanouts=pp_neighbor_fanouts,
        logit_delta_output_path=logit_delta_path,
        delta_gate_output_path=delta_gate_path,
        routing_source_weight_output_path=routing_source_path,
        routing_null_weight_output_path=routing_null_path,
        amp_dtype=torch.bfloat16,
    )

    eligible_path = args.output_dir / "eligible_go_indices.i32.npy"
    np.save(
        eligible_path,
        np.asarray(stores.episode_sampler.eligible_go, dtype=np.int32),
    )
    routing_names_path = None
    if args.save_diagnostics:
        routing_names_path = args.output_dir / "nbs_routing_source_names.json"
        if model.config.source_mode == "layer":
            source_names = [
                f"layer:{i + 1}|target:protein" for i in range(model.config.num_layers)
            ]
        else:
            source_names = [
                f"layer:{i + 1}|relation:{'-'.join(edge_type)}"
                for i in range(model.config.num_layers)
                for edge_type in model.backbone.incoming_target_relations
            ]
        routing_names_path.write_text(
            json.dumps(source_names, indent=2) + "\n",
            encoding="utf-8",
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
        "schema_version": 3,
        "exporter": "NBS full-task isolated inductive inference v0.6.0-inductive-r5",
        "inductive_inference_api_version": api_version,
        "task": config.get("task"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "inference_mode": "_".join(
            [
                "inductive",
                *([] if pp_neighborhood is None else ["pp"]),
                "feature",
                *([] if evidence is None else ["candidate"]),
            ]
        ),
        "prediction_space": "complete_task_classifier_columns",
        "num_proteins": int(output.shape[0]),
        "num_task_go": int(output.shape[1]),
        "go_chunk_size": int(args.go_chunk_size),
        "go_chunk_semantics": "frozen_deterministic_support_graph_block",
        "protein_batch_size": int(args.protein_batch_size),
        "support_per_query": int(args.support_per_query),
        "base_values_are_logits": bool(args.base_values_are_logits),
        "candidate_evidence": "sparse_fixed_k" if evidence is not None else "absent_zero_channel",
        "external_pp": {
            "enabled": pp_neighborhood is not None,
            "relation_operator": "similar_to" if pp_neighborhood is not None else None,
            "message_direction": "core_to_test" if pp_neighborhood is not None else None,
            "test_to_test_edges": False,
            "per_layer_fanouts": pp_neighbor_fanouts,
            "neighbors": (
                None if args.external_pp_neighbors is None else str(args.external_pp_neighbors)
            ),
            "neighbors_sha256": (
                None if args.external_pp_neighbors is None else _sha256(args.external_pp_neighbors)
            ),
            "edge_attr": (
                None if args.external_pp_edge_attr is None else str(args.external_pp_edge_attr)
            ),
            "edge_attr_sha256": (
                None if args.external_pp_edge_attr is None else _sha256(args.external_pp_edge_attr)
            ),
        },
        "routing_contract": {
            "query_routing_hierarchy": "sampled_training_support_graph",
            "candidate_scoring_hierarchy": "isolated_external_proteins",
            "index_spaces_separate": True,
        },
        "output_probability": str(probability_path),
        "output_probability_sha256": _sha256(probability_path),
        "applied_logit_delta": None if logit_delta_path is None else str(logit_delta_path),
        "delta_gate": None if delta_gate_path is None else str(delta_gate_path),
        "routing_source_weights": (
            None if routing_source_path is None else str(routing_source_path)
        ),
        "routing_null_weight": None if routing_null_path is None else str(routing_null_path),
        "routing_source_names": (
            None if routing_names_path is None else str(routing_names_path)
        ),
        "eligible_go_indices": str(eligible_path),
        "eligible_go_count": int(stores.episode_sampler.eligible_go.size),
        "protein_ids": None if output_ids is None else str(output_ids),
        "protein_ids_sha256": None if output_ids is None else _sha256(output_ids),
        "input_manifest": None if args.input_manifest is None else str(args.input_manifest),
        "input_manifest_sha256": (
            None if args.input_manifest is None else _sha256(args.input_manifest)
        ),
        "uses_expert_probability_in_nbs_forward": False,
        "note": (
            "Every task GO column is scored. Training Q does not cap prediction labels. "
            "External proteins are always isolated from one another. When external_pp.enabled, "
            "retrieved core neighbours are aggregated through the trained similar_to operator; "
            "otherwise relation-source residuals are zero. GO queries use the trained NBS support "
            "graph and full BoxSquaredEL cache."
        ),
    }
    manifest_path = args.output_dir / "nbs_full_task_prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
