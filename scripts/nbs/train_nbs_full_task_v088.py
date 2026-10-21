#!/usr/bin/env python3
"""v0.8.8: inductive heterogeneous tunedGraphSAGE, full-axis BCE and standalone GO classification."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nbs_models/nbs_protein_go")]

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from nbs_pg.full_task_data_v088 import FullTaskDataV088 as FullTaskData
from nbs_pg.local_loader_v088 import build_latence_nbs_stores
from nbs_pg.full_task_loss_v088 import FullTaskLossConfigV088, full_task_loss_v088, compute_training_prior
from nbs_pg.full_task_model_v088 import FullTaskGraphModelV088, FullTaskModelConfigV088
from nbs_pg.full_task_schedule_v087 import ScheduleConfigV087, EpochScheduleV087
from nbs_pg.full_task_epochs_v087 import EpochPlanV087, EpochSeedStreamV087
from nbs_pg.full_task_metrics_v085 import compute_standard_metrics, metric_definitions
from nbs_pg.full_task_development_v085 import DevelopmentSetV085
from nbs_pg.full_task_evidence_v081 import alias_pu_exclusions


def build_model(feature_dim, ontology, model_options, variant, device):
    if variant not in ("fixed", "dynamic"):
        raise ValueError("full_task.variant must be fixed or dynamic")
    cfg = FullTaskModelConfigV088(**model_options)
    model = FullTaskGraphModelV088(feature_dim, ontology, cfg).to(device)
    model.training_variant = variant
    return model, cfg


def forward_flags(variant, ablation="full"):
    if variant not in ("fixed", "dynamic") or ablation != "full":
        raise ValueError("v088 supports only the complete model; paused ablations are not available")
    return {}


def source_flags(ablation):
    forward_flags("dynamic", ablation)
    return {name: True for name in ("query_candidate", "neighbor_candidate", "neighbor_pseudo", "core", "pp")}


def training_objective(model, batch, loss_config, *, global_role_counts, world_size=1):
    output = model(batch, return_details=True)
    logits = output["logits"]
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    excluded = alias_pu_exclusions(batch["positive_mask"], raw_model.task_to_ontology)
    if not torch.equal(batch["targets"], batch["positive_mask"].float()):
        raise ValueError("v088 requires binary training membership, not teacher probabilities")
    loss, parts = full_task_loss_v088(logits, batch["targets"], batch["is_weak"], loss_config,
        pu_exclusion_mask=excluded, global_role_counts=global_role_counts, world_size=world_size)
    return logits, loss, parts


def split_microbatches(weak, core, count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("accumulation_steps must be a positive integer")
    if len(core) < count:
        raise ValueError("core_batch must be >= accumulation_steps, including every rank")
    return [np.concatenate((w, c)) for w, c in zip(np.array_split(weak, count), np.array_split(core, count))]


@torch.no_grad()
def synchronize_batchnorm_buffers(model, world):
    """One collective: canonical rank-0 running stats, also used at inference.

    Normalization during training uses each rank's local microbatch. Sparse GO
    adjacency buffers are deliberately not passed to DDP buffer broadcasting.
    """
    if world == 1:
        return
    buffers = [buf for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
               for buf in (module.running_mean, module.running_var, module.num_batches_tracked) if buf is not None]
    if not buffers:
        return
    packed = torch.cat([buf.reshape(-1).float() for buf in buffers])
    dist.broadcast(packed, src=0)
    offset = 0
    for buf in buffers:
        buf.copy_(packed[offset:offset + buf.numel()].reshape(buf.shape).to(buf.dtype))
        offset += buf.numel()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prediction_implementation(variant):
    names = ["full_task_model.py", "full_task_model_v088.py", "full_task_data.py",
             "full_task_data_v083.py", "full_task_data_v084.py", "full_task_data_v088.py",
             "local_loader.py", "local_loader_v088.py", "latence_graph_stores.py"]
    folder = ROOT / "nbs_models/nbs_protein_go/nbs_pg"
    result = {name: sha256(folder / name) for name in names}
    result["train_nbs_full_task_v088.py"] = sha256(Path(__file__))
    return result


def training_implementation(variant):
    result = prediction_implementation(variant)
    folder = ROOT / "nbs_models/nbs_protein_go/nbs_pg"
    for name in ("full_task_loss_v088.py", "full_task_evidence_v081.py", "full_task_schedule_v087.py",
                 "full_task_epochs_v087.py", "full_task_metrics_v085.py", "full_task_development_v085.py"):
        result[name] = sha256(folder / name)
    return result


def training_baseline_identity(data):
    store = data.stores.episode_sampler.base_logit_store
    files = []
    for role, value in sorted(store.arrays.items()):
        path = Path(value.filename).resolve()
        stat = path.stat()
        files.append({"role": role, "path": str(path), "size": stat.st_size,
                      "mtime_ns": stat.st_mtime_ns, "shape": list(value.shape), "dtype": str(value.dtype)})
    return {"files": files, "probability_clip": store.probability_clip,
            "role_slices": [asdict(value) for value in store.slices]}


def ontology_contract(ontology):
    result = {}
    tensors = {k: v for k, v in ontology.items() if k != "edges"}
    tensors.update({"edge_" + k: v for k, v in ontology["edges"].items()})
    for key, value in tensors.items():
        result[key] = hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
    return result


def metric_pack(labels, probabilities):
    # The SAME existing protein-Fmax kernel, grid and strict comparison. No
    # ranking diagnostics or unrelated ablation metrics are computed here.
    from nbs_pg.full_task_metrics_v085 import _validate, _thresholds, _protein_curve
    y, p = _validate(labels, probabilities)
    thresholds = _thresholds(.001)
    f1, precision, recall, eligible = _protein_curve(y, p, thresholds)
    best = int(np.argmax(f1))
    return {"standard_protein_fmax": 100 * float(f1[best]),
            "standard_protein_fmax_threshold": float(thresholds[best]),
            "standard_protein_precision_at_fmax": 100 * float(precision[best]),
            "standard_protein_recall_at_fmax": 100 * float(recall[best]),
            "standard_protein_fmax_threshold_step": .001,
            "standard_num_proteins": len(y), "standard_num_go": y.shape[1],
            "standard_num_positive_labels": int(y.sum()),
            "standard_num_protein_fmax_eligible": eligible}


@torch.no_grad()
def validate(model, data, device, batch_size):
    if not len(data.validation_ids):
        raise ValueError("set holdout_core_count > 0")
    previous = model.training
    context = (data._sampling_step, data._sampling_rank, data._sampling_training)
    data.set_sampling_context(0, 0, False)
    model.eval()
    try:
        go = model.encode_go()
        predictions, labels = [], []
        for start in range(0, len(data.validation_ids), batch_size):
            batch = data.batch(data.validation_ids[start:start + batch_size], device)
            predictions.append(model(batch, go_encoding=go).sigmoid().cpu().numpy())
            labels.append(batch["positive_mask"].cpu().numpy())
        return {"full": metric_pack(np.concatenate(labels), np.concatenate(predictions)),
                "scope": "stage2_core_holdout; Stage1-seen; training monitor only",
                "selection_role": "monitor_only; never acceptance", "num_proteins": len(data.validation_ids),
                "standard_metric_definitions": metric_definitions()}
    finally:
        model.train(previous)
        data.set_sampling_context(*context)


@torch.no_grad()
def evaluate_development(development, model, data, device, batch_size):
    previous = model.training
    model.eval()
    try:
        go = model.encode_go()
        values = []
        for start in range(0, len(development.labels), batch_size):
            rows = np.arange(start, min(start + batch_size, len(development.labels)))
            batch = data.inference_batch(development.input_dir, rows, device)
            if "targets" in batch or "positive_mask" in batch:
                raise ValueError("development labels entered inference")
            values.append(model(batch, go_encoding=go).sigmoid().cpu().numpy())
        return {"eligible_for_selection": True, "scope": "prevalidated_external_development",
                "methods": {"G": metric_pack(development.labels, np.concatenate(values))}}
    finally:
        model.train(previous)


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(state, temporary)
    temporary.replace(path)


def load_development(cfg, data):
    spec = cfg.get("development")
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {"manifest"} or not spec["manifest"]:
        raise ValueError("full_task.development must contain only a nonempty manifest path")
    return DevelopmentSetV085.load(spec["manifest"], data)


def development_selection(result, previous_best):
    score = result["methods"]["G"]["standard_protein_fmax"]
    improved = result.get("eligible_for_selection") is True and score > previous_best
    return improved, score if improved else previous_best


def train(args, config, data, device, rank, world):
    cfg = config["full_task"]
    out = args.work_dir.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if not args.resume and (out / "training_history.json").exists():
            raise FileExistsError(f"Run exists: use --resume {out / 'latest.pt'} or a new --work-dir")
    if world > 1:
        dist.barrier()
    data.ensure_prepared()
    development = load_development(cfg, data)
    development_contract = development.contract if development is not None else None
    ontology = data.ontology("cpu")
    contract = {"data": data.data_contract(), "ontology": ontology_contract(ontology)}
    baseline_identity = training_baseline_identity(data)
    variant = cfg.get("variant", "fixed")
    if cfg.get("sampler", {}).get("mode", variant) != variant:
        raise ValueError("variant must match sampler.mode for an interpretable fixed/dynamic comparison")
    model, model_config = build_model(data.feature_dim, ontology, cfg.get("model", {}), variant, device)
    loss_config = FullTaskLossConfigV088(**cfg.get("loss", {}))
    accumulation = cfg.get("accumulation_steps", 4)
    # Validate even before data/model work reaches its first optimizer update.
    split_microbatches(np.arange(cfg["weak_batch"]), np.arange(cfg["core_batch"]), accumulation)
    prior_metadata = None
    if not args.resume:
        prior, prior_metadata = compute_training_prior(data, loss_config,
            task_to_ontology=model.task_to_ontology.detach().cpu().numpy())
        model.initialize_output_prior(torch.as_tensor(prior, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg.get("weight_decay", 1e-4))
    if cfg.get("epoch_unit") != "weak":
        raise ValueError("v088 requires epoch_unit=weak")
    if args.epochs is not None and args.epochs != cfg["epochs"]:
        raise ValueError("--epochs must match config; it cannot rewrite the schedule horizon")
    wb, cb = int(cfg["weak_batch"]), int(cfg["core_batch"])
    plan = EpochPlanV087(len(data.weak_ids), len(data.core_ids), wb, cb, world, epochs=cfg["epochs"])
    stream = EpochSeedStreamV087(data.weak_ids, data.core_ids, plan, seed=cfg["seed"], rank=rank)
    schedule_config = ScheduleConfigV087.from_full_task(cfg, plan.steps_per_epoch)
    scheduler = EpochScheduleV087(optimizer, schedule_config)
    total = plan.total_steps
    if args.stop_epoch is not None:
        if not 1 <= args.stop_epoch <= plan.epochs:
            raise ValueError("--stop-epoch must be within the configured horizon")
        total = args.stop_epoch * plan.steps_per_epoch
    if args.stop_step is not None:
        total = args.stop_step
    if not 0 < total <= plan.total_steps:
        raise ValueError("stop budget must be positive and no greater than the fixed epoch horizon")
    start = 0
    history, validations, development_history = [], [], []
    best = best_development = -float("inf")
    if args.resume:
        if args.resume.resolve().parent != out:
            raise ValueError("resume continues its original --work-dir; use the checkpoint's parent directory")
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("runner_version") != "0.8.8":
            raise ValueError("v0.8.8 needs a fresh run; older checkpoints lack this runner schedule/development contract")
        if saved.get("training_implementation") != training_implementation(variant):
            raise ValueError("resume requires unchanged v0.8.8 training implementation")
        if saved.get("training_baseline_identity") != baseline_identity:
            raise ValueError("resume backbone probability files or clipping changed")
        if saved.get("development_contract") != development_contract:
            raise ValueError("resume requires identical development data, references and exclusion contract")
        if saved["contract"] != contract or saved["world_size"] != world:
            raise ValueError("resume requires the same data, core split and number of ranks")
        if saved["model_config"] != asdict(model_config) or saved["loss_config"] != asdict(loss_config):
            raise ValueError("resume requires identical model/loss configuration")
        old_cfg = saved["config"]["full_task"]
        if old_cfg.get("variant", "fixed") != variant or old_cfg.get("sampler", {}) != cfg.get("sampler", {}):
            raise ValueError("resume requires identical variant and sampler")
        if any(old_cfg.get(k) != cfg.get(k) for k in ("weak_batch", "core_batch", "seed", "epochs", "epoch_unit", "learning_rate", "weight_decay", "grad_clip", "accumulation_steps")):
            raise ValueError("resume may extend steps but must preserve batch, RNG and optimization settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved.get("scheduler", {}), expected_step=saved["step"])
        stream.load_state_dict(saved["epoch_stream"], expected_step=saved["step"])
        start, best = saved["step"], saved["best_score"]
        prior_metadata = saved.get("prior_metadata")
        history, validations = saved["history"], saved["validations"]
        development_history = saved.get("development_history", [])
        best_development = saved.get("best_development_score", -float("inf"))
        rng = saved["rng"][rank]
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
    if total <= start:
        raise ValueError(f"requested steps={total} must exceed resumed step={start}")
    if rank == 0:
        config["runtime"] = {"requested_steps": total, "world_size": world,
                             "global_weak_batch": wb * world, "global_core_batch": cb * world,
                             "scheduler_contract": asdict(schedule_config),
                             "model_architecture_version": "0.8.8",
                             "data_architecture_version": "0.8.8",
                             "encoder_variant": model_config.encoder_variant,
                             "prediction_mode": model_config.prediction_mode,
                             "epoch_unit": "weak", "epochs": plan.epochs, "steps_per_epoch": plan.steps_per_epoch,
                             "total_planned_steps": plan.total_steps,
                             "training_control": "inductive heterogeneous tunedGraphSAGE; full-axis BCE",
                             "accumulation_steps": accumulation, "batchnorm": "local microbatch; rank0 canonical running stats",
                             "parameters": sum(p.numel() for p in model.parameters()),
                             "prior_metadata": prior_metadata,
                             "weak_population": len(data.weak_ids), "core_training_population": len(data.core_ids),
                             "development_contract": development_contract,
                             "development_selection_rule": "maximize standard_protein_fmax on external dev only; fixed epoch checkpoint otherwise"}
        write_json(out / "resolved_config.json", config)
    distributed = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=False,
    ) if world > 1 else model
    checkpoint_epochs = cfg.get("checkpoint_epochs", list(range(1, plan.epochs + 1)))
    if not isinstance(checkpoint_epochs, list) or any(isinstance(e, bool) or not isinstance(e, int) or not 1 <= e <= plan.epochs for e in checkpoint_epochs):
        raise ValueError("checkpoint_epochs must be a list of epochs within the configured horizon")
    points = {e * plan.steps_per_epoch for e in checkpoint_epochs if e * plan.steps_per_epoch <= total} | {total}
    interval = int(cfg.get("checkpoint_interval", 0))
    if interval > 0:
        points.update(range(interval, total + 1, interval))
    if rank == 0 and start == 0:
        write_json(out / "training_prior.json", prior_metadata)
    # No validation at untrained initialization: the fixed prior is not a
    # performance result. Measure only actual checkpoints and the final model.
    if world > 1:
        dist.barrier()
    # DDP constructor synchronizes parameters; dropout RNG differs by rank.
    if not args.resume:
        torch.manual_seed(cfg["seed"] + rank)
    begin = window = time.monotonic()
    aggregate = {}
    window_steps = 0
    for step in range(start + 1, total + 1):
        seeds = stream.next_batch()
        if seeds["step"] != step:
            raise RuntimeError("epoch stream and optimizer step diverged")
        weak, core = seeds["weak_ids"], seeds["core_ids"]
        data.set_sampling_context(step=step, rank=rank, training=True)
        model.set_encoder_step(step=step, rank=rank)
        synchronize_batchnorm_buffers(model, world)
        global_seeds = np.concatenate((seeds["global_weak_ids"], seeds["global_core_ids"]))
        microbatches = split_microbatches(weak, core, accumulation)
        lr = scheduler.apply(step)
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)
        max_nodes = max_edges = 0
        for micro, ids in enumerate(microbatches):
            batch = data.batch(ids, device, supervision_seed_ids=global_seeds)
            max_nodes = max(max_nodes, len(batch["sampled_protein_x"]))
            max_edges = max(max_edges, sum(block["edge_index"].shape[1] for block in batch["blocks"]))
            sync_context = distributed.no_sync() if world > 1 and micro < len(microbatches) - 1 else nullcontext()
            with sync_context:
                logits, loss, parts = training_objective(distributed, batch, loss_config,
                    global_role_counts=(seeds["global_weak_count"], seeds["global_core_count"]), world_size=world)
                finite = torch.isfinite(loss).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError(f"nonfinite loss at step {step}; no optimizer update applied")
                # The loss already uses GLOBAL MACRO role denominators. Dividing
                # again by accumulation_steps would silently shrink gradients.
                loss.backward()
            step_loss += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.), error_if_nonfinite=True)
        optimizer.step()
        for key, value in {"loss": step_loss, "grad_norm": norm.detach(),
                           "max_micro_nodes": torch.tensor(float(max_nodes), device=device),
                           "max_micro_edges": torch.tensor(float(max_edges), device=device)}.items():
            aggregate[key] = aggregate.get(key, torch.zeros_like(value)) + value
        window_steps += 1
        if step % cfg.get("log_every", 25) == 0 or step in points:
            names = sorted(aggregate)
            values = torch.stack([aggregate[k] / window_steps for k in names])
            if world > 1:
                dist.all_reduce(values)
                values /= world
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = dict(zip(names, values.cpu().tolist()))
            row.update(step=step, seconds_per_step=(time.monotonic() - window) / window_steps,
                       elapsed_seconds=time.monotonic() - begin, learning_rate=lr,
                       weak_equivalent_passes=seeds["weak_exposures"] / len(data.weak_ids),
                       core_equivalent_passes=seeds["core_exposures"] / len(data.core_ids),
                       weak_completed_passes=seeds["weak_exposures"] // len(data.weak_ids),
                       core_completed_passes=seeds["core_exposures"] // len(data.core_ids),
                       epoch=seeds["epoch"], step_in_epoch=seeds["step_in_epoch"],
                       steps_per_epoch=plan.steps_per_epoch,
                       weak_exposures=seeds["weak_exposures"], core_exposures=seeds["core_exposures"],
                       scheduler_horizon_steps=schedule_config.horizon_steps,
                       optimizer_updates=step,
                       full_go_per_protein=data.num_task_go,
                       peak_allocated_gb=(torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0))
            if rank == 0:
                history.append(row)
                write_json(out / "training_history.json", history)
                print(f"[epoch {seeds['epoch']}/{plan.epochs} {seeds['step_in_epoch']}/{plan.steps_per_epoch}; step {step}/{total}] "
                      f"BCE={row['loss']:.6f} grad={row['grad_norm']:.4f} "
                      f"sec/step={row['seconds_per_step']:.3f} GPU={row['peak_allocated_gb']:.2f}GB "
                      f"lr={lr:.3g} passes(weak/core)={row['weak_equivalent_passes']:.3f}/{row['core_equivalent_passes']:.3f}", flush=True)
            aggregate, window_steps = {}, 0
            window = time.monotonic()
        if step in points:
            synchronize_batchnorm_buffers(model, world)
            if rank == 0:
                improved = development_improved = False
                # A short operational smoke stop still saves a resumable state,
                # but only completed-epoch checkpoints run the full monitor.
                if step % plan.steps_per_epoch == 0:
                    result = validate(model, data, device, cfg.get("eval_batch", 16))
                    validations.append({"step": step, **result})
                    score = result["full"]["standard_protein_fmax"]
                    improved = score > best
                    best = max(best, score)
                    write_json(out / "validation_history.json", validations)
                    if development is not None:
                        dev_result = evaluate_development(development, model, data, device, cfg.get("eval_batch", 16))
                        development_history.append({"step": step, **dev_result})
                        development_improved, best_development = development_selection(dev_result, best_development)
                        write_json(out / "development_history.json", development_history)
                        print(f"[development {step}] protein_Fmax={dev_result['methods']['G']['standard_protein_fmax']:.4f} "
                              f"best_checkpoint={development_improved}", flush=True)
                    print(f"[holdout monitor {step}] protein_Fmax={score:.4f}; this is not external acceptance", flush=True)
            rng = {"cpu": torch.get_rng_state(),
                   "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            all_rng = [None] * world
            if world > 1:
                dist.all_gather_object(all_rng, rng)
            else:
                all_rng[0] = rng
            if rank == 0:
                state = {"version": "0.8.8", "runner_version": "0.8.8", "variant": variant,
                         "training_implementation": training_implementation(variant),
                         "training_baseline_identity": baseline_identity,
                         "step": step, "model": model.state_dict(),
                         "model_config": asdict(model_config), "loss_config": asdict(loss_config),
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "model_architecture_version": "0.8.8", "data_architecture_version": "0.8.8",
                         "encoder_variant": model_config.encoder_variant,
                         "config": config, "contract": contract,
                         "epoch_stream": stream.state_dict(), "epoch_plan": asdict(plan),
                         "epoch": seeds["epoch"], "step_in_epoch": seeds["step_in_epoch"],
                         "rng": all_rng, "world_size": world, "best_score": best,
                         "history": history, "validations": validations,
                         "development_contract": development_contract,
                         "development_history": development_history,
                         "best_development_score": best_development, "prior_metadata": prior_metadata}
                path = out / f"nbs_step{step}.pt"
                save_checkpoint(path, state)
                shutil.copyfile(path, out / "latest.pt")
                if step % plan.steps_per_epoch == 0:
                    shutil.copyfile(path, out / f"nbs_epoch{step // plan.steps_per_epoch}.pt")
                if improved:
                    shutil.copyfile(path, out / "best_core_monitor.pt")
                if development_improved:
                    shutil.copyfile(path, out / "best_development.pt")
            if world > 1:
                dist.barrier()
            window = time.monotonic()
    if rank == 0:
        print(f"[done] variant={variant} fixed endpoint={out / 'latest.pt'}. "
              "best_core_monitor.pt is a monitor only; no automatic pass/fail from Stage1-seen core. "
              "Evaluate the standalone model on the unchanged external Fmax protocol.", flush=True)


@torch.no_grad()
def evaluate(args, config, data, device):
    if not args.checkpoint or not args.input_dir:
        raise ValueError("export/evaluate needs --checkpoint and --input-dir")
    if args.stage == "evaluate" and not args.metadata_file:
        raise ValueError("evaluate needs --metadata-file")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("runner_version") != "0.8.8":
        raise ValueError("v0.8.8 evaluator requires a v0.8.8 checkpoint; use the original runner for old runs")
    if saved.get("training_baseline_identity") != training_baseline_identity(data):
        raise ValueError("evaluation backbone probability files or clipping differ from the checkpoint")
    data.ensure_prepared()
    ontology = data.ontology("cpu")
    contract = {"data": data.data_contract(), "ontology": ontology_contract(ontology)}
    if saved["contract"] != contract:
        raise ValueError("checkpoint data/ontology/split differs from the evaluation config")
    variant = saved.get("variant", saved["config"]["full_task"].get("variant", "fixed"))
    implementation = prediction_implementation(variant)
    trained_implementation = saved.get("training_implementation", {})
    if any(trained_implementation.get(key) != value for key, value in implementation.items()):
        raise ValueError("prediction implementation differs from the code that trained this v088 checkpoint")
    requested_model_config = asdict(FullTaskModelConfigV088(**config["full_task"].get("model", {})))
    if requested_model_config != saved["model_config"]:
        raise ValueError("evaluation requires the checkpoint model/encoder configuration")
    if config["full_task"].get("sampler", {}) != saved["config"]["full_task"].get("sampler", {}):
        raise ValueError("evaluation requires the checkpoint sampler configuration")
    model, _ = build_model(data.feature_dim, ontology, saved["model_config"], variant, device)
    model.load_state_dict(saved["model"])
    data.set_sampling_context(step=0, rank=0, training=False)
    model.eval()
    go = model.encode_go()
    inp = args.input_dir.resolve()
    rows = len(np.load(inp / "ind_test_repr.f16.npy", mmap_mode="r"))
    destination = args.work_dir.resolve() / f"eval_step{saved['step']}" / args.ablation
    destination.mkdir(parents=True, exist_ok=True)
    probability_path = destination / "nbs_ind_test_prob.f32.npy"
    manifest_path = destination / "nbs_full_task_prediction_manifest.json"
    reuse = False
    if probability_path.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"incomplete prediction without manifest: use a new --work-dir ({destination})")
        recorded = json.loads(manifest_path.read_text())
        reuse = (recorded.get("checkpoint_sha256") == sha256(args.checkpoint)
                 and recorded.get("input_manifest_sha256") == sha256(inp / "ind_test_input_manifest.json")
                 and recorded.get("output_probability_sha256") == sha256(probability_path)
                 and recorded.get("prediction_implementation") == implementation
                 and recorded.get("contract") == contract and recorded.get("ablation") == args.ablation
                 and recorded.get("branch") == "final" and recorded.get("variant") == variant
                 and recorded.get("version") == "0.8.8" and recorded.get("runner_version") == "0.8.8"
                 and recorded.get("model_architecture_version") == "0.8.8"
                 and recorded.get("data_architecture_version") == "0.8.8"
                 and recorded.get("prediction_mode") == saved["model_config"]["prediction_mode"]
                 and recorded.get("source_flags") == source_flags(args.ablation)
                 and recorded.get("step") == saved["step"]
                 and recorded.get("epoch") == saved.get("epoch")
                 and recorded.get("epoch_plan") == saved.get("epoch_plan")
                 and recorded.get("encoder_variant") == saved["model_config"]["encoder_variant"]
                 and recorded.get("output_fusion") == "none"
                 and recorded.get("model_config") == saved["model_config"]
                 and recorded.get("sampler_config") == saved["config"]["full_task"].get("sampler", {}))
        if not reuse:
            raise FileExistsError(f"existing predictions differ from this request: use a new --work-dir ({destination})")
        # Revalidate the actual input arrays as well as their manifest before
        # comparing cached predictions with the current backbone file.
        data.inference_batch(inp, np.arange(min(1, rows)), device)
        print("[predict] verified existing predictions; rerunning metrics only", flush=True)
    if not reuse:
        probability = np.lib.format.open_memmap(probability_path, mode="w+", dtype=np.float32,
                                               shape=(rows, data.num_task_go))

    try:
        size = config["full_task"].get("eval_batch", 80)
        for start in range(0, 0 if reuse else rows, size):
            batch = data.inference_batch(inp, np.arange(start, min(start + size, rows)), device)
            logits = model(batch, go_encoding=go, **forward_flags(variant, args.ablation))
            values = logits.sigmoid().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("nonfinite independent predictions")
            probability[start:start + len(values)] = values
            print(f"[predict] {start + len(values)}/{rows}", flush=True)
        if not reuse:
            probability.flush()
    except Exception:
        if not reuse:
            del probability
            probability_path.unlink(missing_ok=True)
        raise
    manifest = {"version": "0.8.8", "runner_version": "0.8.8", "variant": variant, "step": saved["step"],
                "branch": "final", "model_architecture_version": "0.8.8",
                "data_architecture_version": "0.8.8",
                "encoder_variant": saved["model_config"]["encoder_variant"],
                "prediction_mode": saved["model_config"]["prediction_mode"],
                "source_flags": source_flags(args.ablation),
                "epoch_plan": saved.get("epoch_plan"), "epoch": saved.get("epoch"),
                "model_config": saved["model_config"],
                "sampler_config": saved["config"]["full_task"].get("sampler", {}),
                "scheduler": saved["scheduler"], "checkpoint_path": str(args.checkpoint.resolve()),
                "probability_path": str(probability_path),
                "prediction_implementation": implementation,
                "num_task_go": data.num_task_go, "prediction_space": "complete_task_classifier_columns",
                "uses_expert_probability_in_nbs_forward": False,
                "uses_modelout_probability_in_nbs_forward": False,
                "uses_dense_backbone_logits_in_forward": False, "output_fusion": "none",
                "checkpoint_sha256": sha256(args.checkpoint),
                "input_manifest_sha256": sha256(inp / "ind_test_input_manifest.json"),
                "output_probability_sha256": sha256(probability_path), "ablation": args.ablation,
                "graph_forward": "GO evidence then genuine typed layer blocks; SAGE self+relation mean; raw residual; BN/ReLU/dropout; independent Linear(H,G)",
                "supervision": "binary_core_gold_and_weak_pseudo_membership",
                "uses_modelout_probability_as_target": False,
                "uses_neighbor_weak_pseudo_membership": True,
                "sampling_inference": "deterministic_typed_layer_blocks_ghost_queries",
                                "contract": contract}
    write_json(manifest_path, manifest)
    if args.stage == "export":
        print(f"[export] {manifest_path}", flush=True)
        return
    command = [sys.executable, str(ROOT / "scripts/nbs/eval_nbs_full_task_v088.py"),
               "--task", config["task"], "--metadata-file", str(args.metadata_file.resolve()),
               "--nbs-prob", str(probability_path), "--backbone-prob", str(inp / "backbone_ind_test_prob.f16.npy"),
               "--protein-ids", str(inp / "protein_ids.txt"), "--candidate-go-index", str(inp / "candidate_go_index.i32.npy"),
               "--input-manifest", str(inp / "ind_test_input_manifest.json"), "--prediction-manifest", str(manifest_path),
               "--precision-k", "10,50,100", "--metric-backend", args.metric_backend,
               "--auprc-mode", getattr(args, "auprc_mode", "exact"),
               "--output-dir", str(destination / "metrics")]
    for name in ("expert_prob", "modelout_prob", "expert_protein_ids", "modelout_protein_ids",
                 "expert_go_ids", "modelout_go_ids", "stage1_reference_manifest"):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(Path(value).resolve())))
    if getattr(args, "references_aligned_to_input", False):
        command.append("--references-aligned-to-input")
    if getattr(args, "allow_missing_references", False):
        command.append("--allow-missing-references")
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "train", "export", "evaluate"), required=True)
    p.add_argument("--config", type=Path, default=ROOT / "nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.8_hetero_tuned.json")
    p.add_argument("--work-dir", type=Path, default=ROOT / "outputs/latence_nbs_experiments/v088/hetero_tuned")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", choices=("torch", "faiss"), default="torch")
    p.add_argument("--epochs", type=int, help="Must match configured fixed horizon")
    stop = p.add_mutually_exclusive_group()
    stop.add_argument("--stop-step", type=int, help="Cumulative optimizer step endpoint, not an increment")
    stop.add_argument("--stop-epoch", type=int, help="Cumulative completed weak epoch endpoint")
    p.add_argument("--resume", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--metadata-file", type=Path)
    p.add_argument("--development-manifest", type=Path,
                   help="Optional prevalidated external development contract; bound permanently to a fresh run")
    p.add_argument("--ablation", choices=("full",), default="full")
    p.add_argument("--metric-backend", choices=("stage1", "local_micro"), default="stage1")
    p.add_argument("--auprc-mode", choices=("exact", "hist", "none"), default="exact")
    for name in ("expert-prob", "modelout-prob", "expert-protein-ids", "modelout-protein-ids",
                 "expert-go-ids", "modelout-go-ids", "stage1-reference-manifest"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--references-aligned-to-input", action="store_true")
    p.add_argument("--allow-missing-references", action="store_true")
    args = p.parse_args()
    # E/M are optional comparison columns only; never inputs to the model.
    if args.expert_prob is None and args.modelout_prob is None:
        args.allow_missing_references = True
    if args.stage == "evaluate":
        for name in ("expert_prob", "modelout_prob", "expert_protein_ids", "modelout_protein_ids",
                     "expert_go_ids", "modelout_go_ids", "stage1_reference_manifest", "checkpoint", "metadata_file"):
            value = getattr(args, name)
            if value is not None and not value.is_file():
                p.error(f"{name}: file not found: {value}")
    os.chdir(ROOT)
    config = json.loads(args.config.resolve().read_text())
    cfg = config["full_task"]
    if args.development_manifest is not None:
        if not args.development_manifest.is_file():
            p.error(f"development_manifest: file not found: {args.development_manifest}")
        cfg["development"] = {"manifest": str(args.development_manifest.resolve())}
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    device = torch.device(args.device)
    if world > 1:
        if args.stage != "train":
            raise ValueError("only train uses torchrun; prepare/evaluate use a single process")
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg["seed"])
    # All prepare/train/evaluate/dev paths share the explicit v088 provenance loader.
    data = FullTaskData(config, stores=build_latence_nbs_stores(config))
    if args.stage == "prepare":
        print(data.prepare_neighbors(backend=args.backend, device=device), flush=True)
    elif args.stage == "train":
        train(args, config, data, device, rank, world)
    else:
        evaluate(args, config, data, device)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
