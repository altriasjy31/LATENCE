#!/usr/bin/env python3
"""Export an explicit GO list using existing, checkpoint-bound ontology rows.

No gold metadata is read. This is a dynamic-target inference interface, not a
claim that a trained model has passed a new-GO or temporal zero-shot benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nbs_models/nbs_protein_go")]

import numpy as np
import torch

from scripts.nbs import train_nbs_full_task_v090 as runner
from nbs_pg.local_loader_v090 import build_latence_nbs_stores, load_go_registry
from nbs_pg.full_task_data_v090 import FullTaskDataV090
from nbs_pg.full_task_model_v090 import FullTaskModelConfigV090
from nbs_pg.full_task_phase_v090 import PhasePlanV090


def verified_model(checkpoint, config, data, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("runner_version") != "0.9.0":
        raise ValueError("dynamic target export requires a v0.9.0 checkpoint")
    phase = PhasePlanV090(**saved["phase_plan"])
    progress = runner._phase_fields(phase, saved["step"])
    if (saved["step"] % phase.steps_per_epoch
            or Path(checkpoint).name != f"nbs_epoch{progress['cumulative_completed_epochs']}.pt"
            or any(saved.get(key) != value for key, value in progress.items())):
        raise ValueError("use a matching fixed completed-epoch checkpoint")
    data.ensure_prepared()
    ontology = data.ontology("cpu")
    actual = {"data": data.data_contract(), "ontology": runner.ontology_contract(ontology)}
    if saved["contract"] != actual:
        raise ValueError("target export data/ontology/split differs from checkpoint")
    if saved["training_baseline_identity"] != runner.training_baseline_identity(data):
        raise ValueError("target export source identities differ from checkpoint")
    variant = saved["variant"]
    implementation = runner.prediction_implementation(variant)
    if any(saved["training_implementation"].get(k) != v for k, v in implementation.items()):
        raise ValueError("target export implementation differs from training")
    if asdict(FullTaskModelConfigV090(**config["full_task"]["model"])) != saved["model_config"]:
        raise ValueError("target export model configuration differs from checkpoint")
    if config["full_task"].get("sampler") != saved["config"]["full_task"].get("sampler"):
        raise ValueError("target export sampler configuration differs from checkpoint")
    model, _ = runner.build_model(data.feature_dim, ontology, saved["model_config"],
                                  variant, device, data.seen_task_mask)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    data.set_sampling_context(0, 0, False)
    return model, saved


@torch.no_grad()
def export_targets(model, data, registry, target_ids, input_dir, output_dir,
                   *, device="cpu", batch_size=16, go_chunk_size=1024):
    if batch_size < 1 or go_chunk_size < 1:
        raise ValueError("batch and GO chunk sizes must be positive")
    targets = registry.resolve_target_go(target_ids)
    if not np.array_equal(registry.task_to_ontology, data.task_to_ontology_rows):
        raise ValueError("target GO identity mapping differs from training data")
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("target export requires a new empty output directory")
    n = len(np.load(input_dir / "ind_test_repr.f16.npy", mmap_mode="r"))
    # Validate prepared input arrays and their hashes before creating an output.
    data.inference_batch(input_dir, np.arange(min(1, n)), device)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "target_go_prob.f32.npy"
    temporary = output_dir / "target_go_prob.tmp.npy"
    full_go = model.encode_ontology()
    task_go = full_go[model.task_to_ontology]
    probability = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                           shape=(n, len(targets)))
    try:
        for start in range(0, n, batch_size):
            batch = data.inference_batch(input_dir,
                np.arange(start, min(n, start + batch_size)), device)
            values = model(batch, target_ontology_ids=torch.as_tensor(targets, device=device),
                go_encoding=task_go, ontology_encoding=full_go,
                go_chunk_size=go_chunk_size).sigmoid().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("nonfinite dynamic GO predictions")
            probability[start:start + len(values)] = values
        probability.flush()
    except Exception:
        del probability
        temporary.unlink(missing_ok=True)
        raise
    del probability
    temporary.replace(output)
    column_file = output_dir / "target_go_columns.tsv"
    canonical = [registry.canonical_go_id(go) for go in target_ids]
    seen_rows = set(map(int, data.task_to_ontology_rows[data.seen_task_mask]))
    column_file.write_text("column\trequested_go_id\tcanonical_go_id\tontology_row\tstage2_supervised\n" +
        "".join(f"{i}\t{go}\t{canonical[i]}\t{row}\t{int(int(row) in seen_rows)}\n"
                for i, (go, row) in enumerate(zip(target_ids, targets))), encoding="utf-8")
    return output, column_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "input-dir", "target-go-file", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--go-chunk-size", type=int, default=1024)
    args = p.parse_args(argv)
    initial_hash = runner.sha256(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = json.loads(args.config.read_text()) if args.config else saved["config"]
    if config.get("release_version") != "0.9.0":
        p.error("config is not a v0.9.0 consumer")
    targets = [line.strip() for line in args.target_go_file.read_text().splitlines() if line.strip()]
    stores = build_latence_nbs_stores(config)
    registry = load_go_registry(config, stores)
    data = FullTaskDataV090(config, stores=stores)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, saved = verified_model(args.checkpoint, config, data, device)
    if runner.sha256(args.checkpoint) != initial_hash:
        raise ValueError("checkpoint changed while opening it")
    output, columns = export_targets(model, data, registry, targets,
        args.input_dir, args.output_dir, device=device,
        batch_size=config["full_task"].get("eval_batch", 16),
        go_chunk_size=args.go_chunk_size)
    runner.write_json(args.output_dir / "target_go_prediction_manifest.json", {
        "schema": "nbs-v090-dynamic-target-export-v1", "runner_version": "0.9.0",
        "checkpoint_sha256": initial_hash, "checkpoint_path": str(args.checkpoint.resolve()),
        "input_manifest_sha256": runner.sha256(args.input_dir / "ind_test_input_manifest.json"),
        "protein_ids_sha256": runner.sha256(args.input_dir / "protein_ids.txt"),
        "target_columns_sha256": runner.sha256(columns), "registry": registry.contract(),
        "probability_sha256": runner.sha256(output), "probability_shape": list(np.load(output, mmap_mode="r").shape),
        "exporter_sha256": runner.sha256(Path(__file__)), "label_split": data.label_split_contract,
        "output_fusion": "none", "uses_gold_at_inference": False,
        "zero_shot_performance": "未验证", "biological_validation": "未验证",
        "scope": "explicit existing ontology rows; no future geometry construction or automatic temporal benchmark"})
    print(output)


if __name__ == "__main__":
    main()
