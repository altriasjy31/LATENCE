#!/usr/bin/env python3
"""v0.9.0: shared protein–GO scoring with strict independent training phases."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from copy import deepcopy
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

from nbs_pg.full_task_data_v090 import FullTaskDataV090 as FullTaskData
from nbs_pg.local_loader_v090 import build_latence_nbs_stores
from nbs_pg.full_task_loss_v090 import FullTaskLossConfigV090, full_task_loss_v090
from nbs_pg.full_task_model_v090 import FullTaskGraphModelV090, FullTaskModelConfigV090, migrate_encoder_state_dict
from nbs_pg.full_task_model_v088 import FullTaskModelConfigV088
from nbs_pg.full_task_loss_v088 import FullTaskLossConfigV088
from nbs_pg.full_task_schedule_v087 import ScheduleConfigV087, EpochScheduleV087
from nbs_pg.full_task_epochs_v087 import EpochPlanV087, EpochSeedStreamV087
from nbs_pg.full_task_metrics_v085 import compute_standard_metrics, metric_definitions
from nbs_pg.full_task_development_v085 import DevelopmentSetV085
from nbs_pg.full_task_evidence_v081 import alias_pu_exclusions
from nbs_pg.full_task_phase_v090 import (PhasePlanV090, resume_contract,
                                         validate_resume_contract, RELEASE_VERSION, MODEL_VERSION, DATA_VERSION)
from scripts.nbs import train_nbs_full_task_v089 as source_runner


def build_model(feature_dim, ontology, model_options, variant, device, seen_task_mask=None):
    if variant not in ("fixed", "dynamic"):
        raise ValueError("full_task.variant must be fixed or dynamic")
    cfg = FullTaskModelConfigV090(**model_options)
    model = FullTaskGraphModelV090(feature_dim, ontology, cfg,
        seen_task_mask=None if seen_task_mask is None else np.array(seen_task_mask, dtype=bool, copy=True)).to(device)
    model.training_variant = variant
    return model, cfg


def forward_flags(variant, ablation="full"):
    if variant not in ("fixed", "dynamic") or ablation != "full":
        raise ValueError("v090 supports only the complete model; paused ablations are not available")
    return {}


def source_flags(ablation):
    forward_flags("dynamic", ablation)
    return {name: True for name in ("query_candidate", "neighbor_candidate", "neighbor_pseudo", "core", "pp")}


def training_objective(model, batch, loss_config, *, global_role_counts, world_size=1):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    seen = raw_model.seen_task_mask
    supplied_seen = batch.get("seen_task_mask")
    if supplied_seen is not None and (not torch.is_tensor(supplied_seen) or
            supplied_seen.dtype != torch.bool or not torch.equal(supplied_seen, seen)):
        raise ValueError("batch seen_task_mask differs from immutable model supervision mask")
    output = model(batch, return_details=True)
    logits = output["logits"]
    excluded = alias_pu_exclusions(batch["positive_mask"], raw_model.task_to_ontology)
    if not torch.equal(batch["targets"], batch["positive_mask"].float()):
        raise ValueError("v090 requires binary training membership, not teacher probabilities")
    loss, parts = full_task_loss_v090(logits, batch["targets"], batch["is_weak"], loss_config,
        pu_exclusion_mask=excluded, global_role_counts=global_role_counts, world_size=world_size,
        seen_task_mask=seen, task_to_ontology=raw_model.task_to_ontology)
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
             "local_loader.py", "local_loader_v088.py", "local_loader_v089.py", "latence_graph_stores.py",
             "full_task_model_v090.py", "full_task_data_v090.py", "local_loader_v090.py", "go_registry_v090.py"]
    folder = ROOT / "nbs_models/nbs_protein_go/nbs_pg"
    result = {name: sha256(folder / name) for name in names}
    result["train_nbs_full_task_v090.py"] = sha256(Path(__file__))
    result["full_task_phase_v090.py"] = sha256(folder / "full_task_phase_v090.py")
    return result


def training_implementation(variant):
    result = prediction_implementation(variant)
    folder = ROOT / "nbs_models/nbs_protein_go/nbs_pg"
    for name in ("full_task_loss_v088.py", "full_task_loss_v090.py", "full_task_evidence_v081.py", "full_task_schedule_v087.py",
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


def _source_contract(value):
    # max_nodes only raises on excess; it never truncates the sampled graph.
    result = deepcopy(value)
    result.get("data", {}).get("v088_sampling", {}).pop("max_nodes", None)
    return result


def _sampler_identity(value):
    result = deepcopy(value)
    result.pop("max_nodes", None)
    return result


def initialize_encoder_from(saved, source_path, source_sha, config, data, model, model_config,
                            loss_config, contract, baseline_identity, development_contract, variant):
    """Verify original old contracts, then transfer only shared encoder tensors.

    The new scorer, optimizer, RNG, phase counters and history start from zero.
    An encoder supervised on held-out GO cannot initialize a label-holdout run.
    """
    if not np.asarray(data.seen_task_mask, dtype=bool).all():
        raise ValueError("--encoder-from is forbidden with unseen held-out GO: old supervision may contaminate them")
    old_model_config = FullTaskModelConfigV088(**saved.get("model_config", {}))
    current_model = asdict(model_config)
    for key, value in asdict(old_model_config).items():
        if key != "prediction_mode" and current_model.get(key) != value:
            raise ValueError("encoder source architecture differs: " + key)
    if model_config.jk_mode != "last" or model_config.edge_dropout != 0:
        raise ValueError("controlled encoder transfer requires jk_mode=last and edge_dropout=0")
    if old_model_config.prediction_mode != "direct":
        raise ValueError("encoder source must be the original direct classifier")
    old_loss_config = FullTaskLossConfigV088(**asdict(loss_config))
    source_contract = deepcopy(contract)
    source_contract["data"].pop("v090_label_split", None)
    source = source_runner.validate_source(saved, config, data, old_model_config, old_loss_config,
        source_contract, baseline_identity, development_contract, variant)
    if Path(source_path).name != f"nbs_epoch{source['origin_epochs']}.pt":
        raise ValueError("encoder-from requires a matching fixed nbs_epoch<number>.pt source checkpoint")
    state = saved.get("model", {})
    for key, shape in (("classifier.weight", (data.num_task_go, old_model_config.hidden_dim)),
                       ("classifier.bias", (data.num_task_go,))):
        value = state.get(key)
        if (not torch.is_tensor(value) or tuple(value.shape) != shape or
                value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError("source is not a complete finite original classifier checkpoint: " + key)
    transferred = transfer_encoder_state(model, state)
    return dict(mode="encoder_only", source_checkpoint=str(Path(source_path).resolve()),
        source_checkpoint_sha256=source_sha, source_runner_version=saved["runner_version"],
        source_encoder_epochs=source["origin_epochs"],
        source_encoder_weak_exposures=source["origin_weak_exposures"],
        source_encoder_core_exposures=source["origin_core_exposures"],
        source_encoder_optimizer_updates=source["origin_optimizer_updates"],
        transferred_state_keys=transferred, old_head_transferred=False,
        optimizer_state_reused=False, rng_state_reused=False,
        new_scorer_training_exposures_at_initialization=0)


def transfer_encoder_state(model, old_state):
    """Use the shared model's exact, fail-closed encoder migration contract."""
    return migrate_encoder_state_dict(model, old_state)["reused_encoder_keys"]


def _atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _phase_fields(phase, step):
    progress = phase.progress(step)
    return {**progress, "epoch": progress["cumulative_epoch"],
            "step_in_epoch": progress["phase_step_in_epoch"]}


def _fresh_directory(out, init_from):
    if init_from is not None and Path(init_from).resolve().parent == out:
        raise ValueError("encoder-from requires a NEW work directory, not the source checkpoint directory")
    if out.exists():
        existing = [p for p in out.iterdir() if p.name != ".launcher_train.lock"]
        if existing:
            raise FileExistsError("new phase requires an empty work directory; use --resume for the same phase")


def _load_source_checkpoint(path):
    before = sha256(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if sha256(path) != before:
        raise ValueError("source checkpoint changed while initializing; use a fixed completed-epoch checkpoint")
    return saved, before


def train(args, config, data, device, rank, world):
    cfg = config["full_task"]
    out = args.work_dir.resolve()
    init_from = getattr(args, "encoder_from", None)
    if init_from is not None and args.resume is not None:
        raise ValueError("encoder-from and resume are mutually exclusive")
    if cfg.get("expected_world_size", world) != world:
        raise ValueError("training world size differs from the selected v090 preset")
    if not args.resume:
        _fresh_directory(out, init_from)
    elif args.resume.resolve().parent != out:
        raise ValueError("resume continues its original work directory")
    # No rank writes the new phase before every rank has finished preflight.
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
        raise ValueError("variant must match sampler.mode")
    model, model_config = build_model(data.feature_dim, ontology, cfg.get("model", {}), variant, device, data.seen_task_mask)
    loss_config = FullTaskLossConfigV090(**cfg.get("loss", {}))
    accumulation = cfg.get("accumulation_steps", 1)
    split_microbatches(np.arange(cfg["weak_batch"]), np.arange(cfg["core_batch"]), accumulation)
    if cfg.get("epoch_unit") != "weak":
        raise ValueError("v090 requires epoch_unit=weak")
    if args.epochs is not None and args.epochs != cfg["epochs"]:
        raise ValueError("--epochs must match the cumulative configured target")
    wb, cb = cfg["weak_batch"], cfg["core_batch"]
    target = cfg["epochs"]
    saved = None
    origin = dict(origin_epochs=0, origin_weak_exposures=0, origin_core_exposures=0,
                  origin_optimizer_updates=0)
    prior_metadata = {"mode": "shared_scalar_bias", "initial_bias": model_config.scorer_initial_bias,
                      "per_go_frequency_prior": False}
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("runner_version") != RELEASE_VERSION:
            raise ValueError("v090 resume requires an unchanged v090 checkpoint; old encoders use --encoder-from")
        if saved.get("model_architecture_version") != MODEL_VERSION or saved.get("data_architecture_version") != DATA_VERSION:
            raise ValueError("resume model/data architecture metadata differs")
        phase = PhasePlanV090(**saved["phase_plan"])
        lineage = saved["phase_origin"]
    else:
        transfer = None
        if init_from is not None:
            source_saved, source_sha = _load_source_checkpoint(init_from)
            transfer = initialize_encoder_from(source_saved, init_from, source_sha, config, data, model,
                model_config, loss_config, contract, baseline_identity, development_contract, variant)
            del source_saved
        provisional = EpochPlanV087(len(data.weak_ids), len(data.core_ids), wb, cb, world, epochs=target)
        phase = PhasePlanV090(**origin, target_epochs=target,
            steps_per_epoch=provisional.steps_per_epoch, n_weak=len(data.weak_ids),
            n_core=len(data.core_ids), global_weak_batch=provisional.global_weak_batch_size,
            global_core_batch=provisional.global_core_batch_size)
        lineage = dict(initialization="fresh", source_checkpoint=None,
            source_checkpoint_sha256=None, source_runner_version=None, **origin,
            source_phase_step=0, source_world_size=0, source_optimizer_state_reused=False,
            source_rng_state_reused=False, new_training_seed=cfg["seed"], encoder_transfer=transfer)
    plan = EpochPlanV087(len(data.weak_ids), len(data.core_ids), wb, cb, world,
                         epochs=phase.remaining_epochs)
    phase_contract = resume_contract(cfg, phase, world)
    if world > 1:
        # Each process may load a source independently. Bind all ranks to the
        # SAME immutable source bytes/origin before DDP broadcasts weights.
        identity = {"phase_plan": asdict(phase), "phase_origin": lineage,
                    "phase_training_contract": phase_contract, "data_contract": contract}
        identities = [None] * world
        dist.all_gather_object(identities, identity)
        canonical = json.dumps(identity, sort_keys=True, allow_nan=False)
        if any(json.dumps(value, sort_keys=True, allow_nan=False) != canonical for value in identities):
            raise ValueError("all ranks must initialize/resume the identical source checkpoint and phase contract")
    phase_cfg = {**cfg, "epochs": phase.remaining_epochs}
    schedule_config = ScheduleConfigV087.from_full_task(phase_cfg, plan.steps_per_epoch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = EpochScheduleV087(optimizer, schedule_config)
    stream = EpochSeedStreamV087(data.weak_ids, data.core_ids, plan, seed=cfg["seed"], rank=rank)
    total = plan.total_steps
    if args.stop_epoch is not None:
        if isinstance(args.stop_epoch, bool) or not phase.origin_epochs < args.stop_epoch <= phase.target_epochs:
            raise ValueError("stop-epoch must exceed the source completed epoch and not exceed the cumulative target")
        total = (args.stop_epoch - phase.origin_epochs) * plan.steps_per_epoch
    if args.stop_step is not None:
        total = args.stop_step
    if isinstance(total, bool) or not isinstance(total, int) or not 0 < total <= plan.total_steps:
        raise ValueError("stop-step must be positive within the frozen PHASE horizon")
    start = 0
    history, validations, development_history = [], [], []
    best = best_development = -float("inf")
    if args.resume:
        if saved.get("training_implementation") != training_implementation(variant):
            raise ValueError("resume requires unchanged v090 training implementation")
        if saved.get("contract") != contract or saved.get("world_size") != world:
            raise ValueError("resume requires identical data, split and ranks")
        if saved.get("training_baseline_identity") != baseline_identity or saved.get("development_contract") != development_contract:
            raise ValueError("resume baseline/development identities differ")
        if saved.get("model_config") != asdict(model_config) or saved.get("loss_config") != asdict(loss_config):
            raise ValueError("resume model/loss configuration differs")
        validate_resume_contract(saved.get("phase_training_contract"), phase_contract)
        phase_manifest = json.loads((out / "phase_manifest.json").read_text())
        if phase_manifest != {"phase_plan": asdict(phase), "phase_origin": lineage}:
            raise ValueError("resume phase lineage differs from the immutable phase manifest")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved.get("scheduler", {}), expected_step=saved["step"])
        stream.load_state_dict(saved.get("epoch_stream", {}), expected_step=saved["step"])
        start = saved["step"]
        for key, value in _phase_fields(phase, start).items():
            if saved.get(key) != value:
                raise ValueError("resume checkpoint phase/lifetime counters differ: " + key)
        prior_metadata = saved.get("prior_metadata")
        history, validations = saved["history"], saved["validations"]
        development_history = saved.get("development_history", [])
        best, best_development = saved.get("best_score", best), saved.get("best_development_score", best_development)
        rng = saved["rng"][rank]
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
    if total <= start:
        raise ValueError("stop endpoint must exceed the resumed phase step")
    # All compatibility and origin checks precede output writes.
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        config["runtime"] = dict(runner_version=RELEASE_VERSION, model_architecture_version=MODEL_VERSION,
            data_architecture_version=DATA_VERSION, world_size=world, requested_phase_steps=total,
            global_weak_batch=plan.global_weak_batch_size, global_core_batch=plan.global_core_batch_size,
            steps_per_epoch=plan.steps_per_epoch, phase_epochs=plan.epochs, cumulative_target_epochs=target,
            phase_plan=asdict(phase), phase_origin=lineage, total_planned_phase_steps=plan.total_steps,
            scheduler_contract=asdict(schedule_config), accumulation_steps=accumulation,
            batchnorm="local microbatch; rank0 canonical running stats", monitor_core=cfg.get("monitor_core", False),
            parameters=sum(p.numel() for p in model.parameters()), prior_metadata=prior_metadata,
            training_control="v090 shared GO-conditioned bilinear scorer; unchanged main-preset typed GraphSAGE and binary role-weighted BCE",
            acceptance_control="fixed external protein Fmax; test is descriptive, never scheduler input")
        write_json(out / "resolved_config.json", config)
        if not args.resume:
            write_json(out / "phase_manifest.json", {"phase_plan": asdict(phase), "phase_origin": lineage})
            write_json(out / "training_prior.json", prior_metadata)
    distributed = DistributedDataParallel(model,
        device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
    points = {s for s in phase.checkpoint_steps(cfg.get("checkpoint_every_epochs", 5)) if s <= total} | {total}
    if world > 1:
        dist.barrier()
    if not args.resume:
        torch.manual_seed(cfg["seed"] + rank)
    # Source optimizer/RNG/history are not part of this phase.
    if not args.resume:
        del saved
    begin = window = time.monotonic()
    aggregate, window_steps = {}, 0
    for step in range(start + 1, total + 1):
        seeds = stream.next_batch()
        progress = _phase_fields(phase, step)
        if seeds["step"] != step or seeds["weak_exposures"] != progress["phase_weak_exposures"] or seeds["core_exposures"] != progress["phase_core_exposures"]:
            raise RuntimeError("phase seed stream and cumulative counters diverged")
        # Offset hash-based dropout/sampling from preceding phases as well.
        sampler_step = progress["lifetime_optimizer_updates"]
        data.set_sampling_context(step=sampler_step, rank=rank, training=True)
        model.set_encoder_step(step=sampler_step, rank=rank)
        synchronize_batchnorm_buffers(model, world)
        global_seeds = np.concatenate((seeds["global_weak_ids"], seeds["global_core_ids"]))
        microbatches = split_microbatches(seeds["weak_ids"], seeds["core_ids"], accumulation)
        lr = scheduler.apply(step)
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)
        step_parts = {}
        max_nodes = max_edges = 0
        for micro, ids in enumerate(microbatches):
            batch = data.batch(ids, device, supervision_seed_ids=global_seeds)
            max_nodes = max(max_nodes, len(batch["sampled_protein_x"]))
            max_edges = max(max_edges, sum(block["edge_index"].shape[1] for block in batch["blocks"]))
            context = distributed.no_sync() if world > 1 and micro < len(microbatches) - 1 else nullcontext()
            with context:
                logits, loss, parts = training_objective(distributed, batch, loss_config,
                    global_role_counts=(seeds["global_weak_count"], seeds["global_core_count"]), world_size=world)
                finite = torch.isfinite(loss).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("nonfinite loss; no optimizer update applied")
                loss.backward()
            step_loss += loss.detach()
            for key in ("contrib_positive", "contrib_negative", "contrib_core_positive", "contrib_core_negative",
                        "contrib_weak_positive", "contrib_weak_negative"):
                step_parts[key] = step_parts.get(key, torch.zeros((), device=device)) + parts[key].detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.), error_if_nonfinite=True)
        optimizer.step()
        for key, value in {"loss": step_loss, **step_parts, "grad_norm": norm.detach(),
                           "max_micro_nodes": torch.tensor(float(max_nodes), device=device),
                           "max_micro_edges": torch.tensor(float(max_edges), device=device)}.items():
            aggregate[key] = aggregate.get(key, torch.zeros_like(value)) + value
        window_steps += 1
        if step % cfg.get("log_every", 5) == 0 or step in points:
            names = sorted(aggregate)
            values = torch.stack([aggregate[k] / window_steps for k in names])
            if world > 1:
                dist.all_reduce(values)
                values /= world
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.monotonic() - window
            row = dict(zip(names, values.cpu().tolist()))
            row.update(progress, step=step, optimizer_updates=step, window_updates=window_steps,
                seconds_per_step=seconds/window_steps, window_training_seconds=seconds,
                elapsed_seconds=time.monotonic()-begin, learning_rate=lr,
                weak_exposures=progress["lifetime_weak_exposures"], core_exposures=progress["lifetime_core_exposures"],
                weak_equivalent_passes=progress["lifetime_weak_equivalent_passes"],
                core_equivalent_passes=progress["lifetime_core_equivalent_passes"],
                steps_per_epoch=plan.steps_per_epoch, scheduler_horizon_steps=schedule_config.horizon_steps,
                full_go_per_protein=data.num_task_go, seen_go_per_protein=int(data.seen_task_mask.sum()),
                unseen_go_per_protein=int((~data.seen_task_mask).sum()),
                peak_allocated_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type == "cuda" else 0.)
            if rank == 0:
                history.append(row)
                write_json(out / "training_history.json", history)
                timing = [r for r in history if r["step"]-r["window_updates"] >= 10][-6:]
                if timing:
                    measured = sum(r["window_updates"] for r in timing)
                    mean = sum(r["window_training_seconds"] for r in timing)/measured
                    write_json(out / "training_runtime.json", dict(runner_version=RELEASE_VERSION,
                        device=str(device), world_size=world, measured_updates=measured,
                        seconds_per_update=mean, training_hours_per_epoch=mean*plan.steps_per_epoch/3600,
                        estimated_remaining_training_hours=mean*(plan.total_steps-step)/3600,
                        timing_scope="training windows only; excludes preparation, checkpoint/monitor/evaluation; first10 phase updates excluded",
                        phase_step=step, cumulative_epoch=progress["epoch"], phase_plan=asdict(phase)))
                print(f"[epoch {progress['epoch']}/{target}; phase {progress['phase_epoch']}/{plan.epochs} "
                      f"{seeds['step_in_epoch']}/{plan.steps_per_epoch}; phase-step {step}/{total}; "
                      f"lifetime-updates {progress['lifetime_optimizer_updates']}] "
                      f"BCE={row['loss']:.6f} grad={row['grad_norm']:.4f} sec/step={row['seconds_per_step']:.3f} "
                      f"GPU={row['peak_allocated_gb']:.2f}GB lr={lr:.3g} "
                      f"passes(weak/core)={row['weak_equivalent_passes']:.3f}/{row['core_equivalent_passes']:.3f}", flush=True)
            aggregate, window_steps = {}, 0
            window = time.monotonic()
        if step in points:
            synchronize_batchnorm_buffers(model, world)
            improved = development_improved = False
            if rank == 0 and step % plan.steps_per_epoch == 0:
                if cfg.get("monitor_core", False):
                    result = validate(model, data, device, cfg.get("eval_batch", 16))
                    validations.append({"step": step, "epoch": progress["epoch"], **result})
                    score = result["full"]["standard_protein_fmax"]
                    improved, best = score > best, max(best, score)
                    write_json(out / "validation_history.json", validations)
                if development is not None:
                    result = evaluate_development(development, model, data, device, cfg.get("eval_batch", 16))
                    development_history.append({"step": step, "epoch": progress["epoch"], **result})
                    development_improved, best_development = development_selection(result, best_development)
                    write_json(out / "development_history.json", development_history)
            rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            all_rng = [None]*world
            if world > 1:
                dist.all_gather_object(all_rng, rng)
            else:
                all_rng[0] = rng
            if rank == 0:
                state = dict(version=RELEASE_VERSION, runner_version=RELEASE_VERSION, variant=variant,
                    training_implementation=training_implementation(variant), training_baseline_identity=baseline_identity,
                    step=step, **progress, phase_plan=asdict(phase), phase_origin=lineage,
                    phase_training_contract=phase_contract, model=model.state_dict(),
                    model_config=asdict(model_config), loss_config=asdict(loss_config), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), model_architecture_version=MODEL_VERSION,
                    data_architecture_version=DATA_VERSION, encoder_variant=model_config.encoder_variant,
                    config=config, contract=contract, epoch_stream=stream.state_dict(), epoch_plan=asdict(plan),
                    rng=all_rng, world_size=world, best_score=best, history=history, validations=validations,
                    development_contract=development_contract, development_history=development_history,
                    best_development_score=best_development, prior_metadata=prior_metadata)
                path = out / f"nbs_step{step}.pt"
                save_checkpoint(path, state)
                _atomic_copy(path, out / "latest.pt")
                if step % plan.steps_per_epoch == 0:
                    _atomic_copy(path, out / f"nbs_epoch{progress['epoch']}.pt")
                if improved:
                    _atomic_copy(path, out / "best_core_monitor.pt")
                if development_improved:
                    _atomic_copy(path, out / "best_development.pt")
            if world > 1:
                dist.barrier()
            window = time.monotonic()
    if rank == 0:
        print(f"[done] cumulative weak epoch={progress['cumulative_completed_epochs']}; "
              f"phase step={step}; latest={out/'latest.pt'}. "
              "Evaluate fixed nbs_epoch*.pt checkpoints on the unchanged external Fmax protocol.", flush=True)


@torch.no_grad()
def evaluate(args, config, data, device):
    if not args.checkpoint or not args.input_dir:
        raise ValueError("export/evaluate needs --checkpoint and --input-dir")
    if args.stage == "evaluate" and not args.metadata_file:
        raise ValueError("evaluate needs --metadata-file")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("runner_version") != RELEASE_VERSION:
        raise ValueError("v090 evaluator requires a v090 checkpoint; evaluate source checkpoints with their original runner")
    if saved.get("model_architecture_version") != MODEL_VERSION or saved.get("data_architecture_version") != DATA_VERSION:
        raise ValueError("export model/data architecture metadata differs")
    phase = PhasePlanV090(**saved["phase_plan"])
    phase_fields = _phase_fields(phase, saved["step"])
    if saved["step"] % phase.steps_per_epoch or args.checkpoint.name != f"nbs_epoch{phase_fields['cumulative_completed_epochs']}.pt":
        raise ValueError("export/evaluate requires a matching fixed completed-epoch nbs_epoch<number>.pt checkpoint")
    if any(saved.get(k) != v for k, v in phase_fields.items()):
        raise ValueError("export checkpoint phase/lifetime counters differ")
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
        raise ValueError("prediction implementation differs from the code that trained this v090 checkpoint")
    requested_model_config = asdict(FullTaskModelConfigV090(**config["full_task"].get("model", {})))
    if requested_model_config != saved["model_config"]:
        raise ValueError("evaluation requires the checkpoint model/encoder configuration")
    if config["full_task"].get("sampler", {}) != saved["config"]["full_task"].get("sampler", {}):
        raise ValueError("evaluation requires the checkpoint sampler configuration")
    model, _ = build_model(data.feature_dim, ontology, saved["model_config"], variant, device, data.seen_task_mask)
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
                 and recorded.get("version") == RELEASE_VERSION and recorded.get("runner_version") == RELEASE_VERSION
                 and recorded.get("model_architecture_version") == MODEL_VERSION
                 and recorded.get("data_architecture_version") == "0.8.8"
                 and recorded.get("prediction_mode") == saved["model_config"]["prediction_mode"]
                 and recorded.get("source_flags") == source_flags(args.ablation)
                 and recorded.get("step") == saved["step"]
                 and recorded.get("epoch") == saved.get("epoch")
                 and recorded.get("epoch_plan") == saved.get("epoch_plan")
                 and recorded.get("phase_plan") == saved.get("phase_plan")
                 and recorded.get("phase_origin") == saved.get("phase_origin")
                 and all(recorded.get(k) == v for k, v in phase_fields.items())
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
    manifest = {"version": RELEASE_VERSION, "runner_version": RELEASE_VERSION, "variant": variant, "step": saved["step"],
                "branch": "final", "model_architecture_version": MODEL_VERSION,
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
                "graph_forward": "GO evidence then genuine typed layer blocks; SAGE self+relation mean; raw residual; BN/ReLU/dropout; shared protein–GO bilinear scoring",
                "supervision": "binary_core_gold_and_weak_pseudo_membership",
                "uses_modelout_probability_as_target": False,
                "uses_neighbor_weak_pseudo_membership": True,
                "sampling_inference": "deterministic_typed_layer_blocks_ghost_queries",
                                "contract": contract}
    manifest.update(phase_fields, phase_plan=saved["phase_plan"], phase_origin=saved["phase_origin"])
    write_json(manifest_path, manifest)
    if args.stage == "export":
        print(f"[export] {manifest_path}", flush=True)
        return
    comparison_dir = destination / "metrics_fmax_only"
    command = [sys.executable, str(ROOT / "scripts/nbs/eval_nbs_full_task_v090.py"),
               "--task", config["task"], "--fmax-only", "--metadata-file", str(args.metadata_file.resolve()),
               "--nbs-prob", str(probability_path), "--backbone-prob", str(inp / "backbone_ind_test_prob.f16.npy"),
               "--protein-ids", str(inp / "protein_ids.txt"),
               "--input-manifest", str(inp / "ind_test_input_manifest.json"), "--prediction-manifest", str(manifest_path),
               "--output-dir", str(comparison_dir)]
    subprocess.run(command, cwd=ROOT, check=True)
    acceptance = args.work_dir.resolve() / f"acceptance_epoch{saved['epoch']}.json"
    gate = [sys.executable, str(ROOT / "scripts/nbs/check_nbs_v090_acceptance.py"),
            "--baseline-comparison", str(args.baseline_comparison.resolve()),
            "--comparison", str(comparison_dir / "nbs_w2s_comparison.json"), "--output", str(acceptance)]
    run_acceptance_gate(gate, acceptance, comparison_dir / "nbs_w2s_comparison.json", args.baseline_comparison)


def run_acceptance_gate(command, output, comparison, baseline):
    """An evaluated 未达标 model is a completed evaluation, not a CLI crash.

    Exit 1 is tolerated only with a newly verified, matching acceptance artifact;
    a checker exception or stale artifact cannot masquerade as a normal result.
    """
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in (0, 1):
        completed.check_returncode()
    path = Path(output)
    if not path.is_file():
        raise RuntimeError("acceptance checker failed without a verified output artifact")
    result = json.loads(path.read_text())
    expected_status = "达标" if completed.returncode == 0 else "未达标"
    if (result.get("status") != expected_status or result.get("fixed_conditions_verified") is not True
            or result.get("new_comparison", {}).get("sha256") != sha256(comparison)
            or result.get("baseline_comparison", {}).get("sha256") != sha256(baseline)):
        raise RuntimeError("acceptance checker output is stale, invalid, or differs from this evaluation")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "train", "export", "evaluate"), required=True)
    p.add_argument("--config", type=Path, default=ROOT / "nbs_models/nbs_protein_go/configs/bp_full_task_v0.9.0_mb12800_ddp_ep100.json")
    p.add_argument("--work-dir", type=Path, default=ROOT / "outputs/latence_nbs_experiments/v090/ddp")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", choices=("torch", "faiss"), default="torch")
    p.add_argument("--epochs", type=int, help="Must match configured fixed horizon")
    stop = p.add_mutually_exclusive_group()
    stop.add_argument("--stop-step", type=int, help="Phase-local optimizer step endpoint, not an increment")
    stop.add_argument("--stop-epoch", type=int, help="Cumulative completed weak epoch endpoint")
    init = p.add_mutually_exclusive_group()
    init.add_argument("--resume", type=Path)
    init.add_argument("--encoder-from", type=Path, help="Encoder-only fresh run from a verified fixed completed v088/v089 epoch; no old head/optimizer/RNG")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--metadata-file", type=Path)
    p.add_argument("--baseline-comparison", type=Path, default=ROOT / "BASELINE_CONTRACT_v088.json")
    p.add_argument("--development-manifest", type=Path,
                   help="Optional prevalidated external development contract; bound permanently to a fresh run")
    p.add_argument("--ablation", choices=("full",), default="full")
    p.add_argument("--metric-backend", choices=("stage1",), default="stage1")
    p.add_argument("--auprc-mode", choices=("exact",), default="exact")
    args = p.parse_args()
    if args.stage != "train" and (args.resume is not None or args.encoder_from is not None):
        p.error("encoder-from/resume are valid only for train")
    if args.encoder_from is not None and not args.encoder_from.is_file():
        p.error("encoder-from checkpoint does not exist")
    if args.stage == "evaluate":
        if not args.baseline_comparison.is_file():
            p.error("fixed baseline comparison does not exist")
        for name in ("checkpoint", "metadata_file"):
            value = getattr(args, name)
            if value is not None and not value.is_file():
                p.error(f"{name}: file not found: {value}")
    os.chdir(ROOT)
    config = json.loads(args.config.resolve().read_text())
    cfg = config["full_task"]
    if config.get("release_version") != RELEASE_VERSION:
        p.error("v090 requires a release_version=0.9.0 config")
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
    if args.stage == "train" and cfg.get("expected_world_size", world) != world:
        p.error("world size differs from config; use the matching single/ddp launcher")
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
