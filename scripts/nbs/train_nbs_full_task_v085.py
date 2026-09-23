#!/usr/bin/env python3
"""v0.8.5: controlled LR schedules over unchanged v084 graph/data/loss."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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

from nbs_pg.full_task_data_v084 import FullTaskDataV084 as FullTaskData
from nbs_pg.full_task_loss_v081 import FullTaskLossConfigV081, full_task_loss_v081
from nbs_pg.full_task_model_v084 import FullTaskGraphModelV084, FullTaskModelConfigV084
from nbs_pg.full_task_schedule_v085 import ScheduleConfigV085, WarmupScheduleV085
from nbs_pg.full_task_metrics_v085 import compute_standard_metrics, metric_definitions
from nbs_pg.full_task_development_v085 import DevelopmentSetV085
from nbs_pg.full_task_evidence_v081 import (
    StructuralSupportConfig, structural_core_support, alias_pu_exclusions,
)


def build_model(feature_dim, ontology, model_options, variant, device):
    if variant not in ("fixed", "dynamic"):
        raise ValueError("full_task.variant must be fixed or dynamic")
    cfg = FullTaskModelConfigV084(**model_options)
    model = FullTaskGraphModelV084(feature_dim, ontology, cfg).to(device)
    model.training_variant = variant
    return model, cfg


def forward_flags(variant, ablation="full"):
    if variant not in ("fixed", "dynamic"):
        raise ValueError("variant must be fixed or dynamic")
    if ablation not in ("full", "weak_off", "core_off", "pp_off", "graph_off", "go_shuffle"):
        raise ValueError("unknown graph intervention")
    return {
        "use_weak_go": ablation not in ("weak_off", "graph_off"),
        "use_core_go": ablation not in ("core_off", "graph_off"),
        "use_pp_context": ablation not in ("pp_off", "graph_off"),
        "shuffle_go": ablation == "go_shuffle",
    }


def loss_support_batch(batch):
    # Identical cosine-only PU policy across fixed/dynamic graph sampling.
    required = ("neighbor_index", "neighbor_attr", "anchor_go_edge", "anchor_x")
    present = ["loss_" + key in batch for key in required]
    if any(present) and not all(present):
        raise ValueError("incomplete fixed loss-support graph")
    return {key: batch["loss_" + key] for key in required} if all(present) else batch


def training_objective(model, batch, variant, loss_config, support_config):
    output = model(batch, return_details=True, **forward_flags(variant))
    logits = output["logits"]
    # Hold PU support and the loss policy constant in fixed/dynamic controls.
    support = structural_core_support(loss_support_batch(batch), logits.shape[1], support_config)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    excluded = alias_pu_exclusions(batch["positive_mask"], raw_model.task_to_ontology)
    if not torch.equal(batch["targets"], batch["positive_mask"].float()):
        raise ValueError("v085 supervision must be binary membership, without modelout probabilities")
    loss, parts = full_task_loss_v081(
        logits, batch["base_logits"], batch["targets"], batch["positive_mask"],
        batch["is_weak"], loss_config, structural_support=support,
        graph_logits=None, pu_exclusion_mask=excluded,
    )
    for key in ("pp_message_norm", "pp_edge_count"):
        if key in output:
            parts[key] = output[key].detach().float()
    if "core_support_mass" in output:
        core_supported = output["core_support_mass"].detach() > 0
        known = batch["positive_mask"].bool()
        parts["core_query_supported_fraction"] = core_supported.float().mean()
        parts["core_prior_positive_coverage"] = (core_supported & known).sum().float() / known.sum().clamp_min(1)
        for role, rows in (("weak", batch["is_weak"].bool()), ("core", ~batch["is_weak"].bool())):
            role_positive = known & rows[:, None]
            parts[role + "_core_prior_positive_coverage"] = (core_supported & role_positive).sum().float() / role_positive.sum().clamp_min(1)
        unknown_supported = core_supported & ~known
        probability_delta = logits.detach().sigmoid() - batch["base_logits"].sigmoid()
        parts["core_supported_unknown_probability_delta"] = (probability_delta * unknown_supported).sum() / unknown_supported.sum().clamp_min(1)
    for key, value in batch.get("sampler_diagnostics", {}).items():
        parts[key] = value.detach().float()
    # Gold-only difficulty diagnostics do not change the loss objective.
    difficult = batch["positive_mask"].bool() & ~batch["is_weak"].bool()[:, None]
    difficult &= batch["base_logits"].sigmoid() < 0.5
    parts["difficult_gold_pairs"] = difficult.sum().float()
    parts["core_proteins"] = (~batch["is_weak"].bool()).sum().float()
    parts["core_gold_positive_pairs"] = (batch["positive_mask"].bool() & ~batch["is_weak"].bool()[:, None]).sum().float()
    if "core_support_mass" in output:
        parts["difficult_gold_core_supported_pairs"] = ((output["core_support_mass"].detach() > 0) & difficult).sum().float()
    parts["binary_positive_targets"] = logits.new_tensor(1.)
    return logits, loss, parts


def finalize_count_diagnostics(row, window_steps, world):
    """Reconstruct global window totals before forming ratios across ranks/steps."""
    for key in ("difficult_gold_pairs", "difficult_gold_core_supported_pairs",
                "core_gold_positive_pairs", "core_proteins"):
        if key in row:
            row[key + "_total"] = int(round(row[key] * window_steps * world))
    total = row.get("difficult_gold_pairs_total", 0)
    if "difficult_gold_core_supported_pairs_total" in row:
        row["difficult_gold_core_coverage"] = (row["difficult_gold_core_supported_pairs_total"] / total
                                                 if total else None)
    core = row.get("core_proteins_total", 0)
    gold = row.get("core_gold_positive_pairs_total", 0)
    row["difficult_gold_per_core_protein"] = total / core if core else None
    row["difficult_gold_fraction_of_core_positive"] = total / gold if gold else None
    row["count_scope"] = "window_total_over_all_ranks; *_pairs without _total are per-rank batch means"
    return row


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
    names = ["full_task_model.py", "full_task_model_v081.py", "full_task_model_v083.py", "full_task_model_v084.py",
             "full_task_data.py", "full_task_data_v083.py", "full_task_data_v084.py", "local_loader.py", "latence_graph_stores.py"]
    folder = ROOT / "nbs_models/nbs_protein_go/nbs_pg"
    result = {name: sha256(folder / name) for name in names}
    result["train_nbs_full_task_v085.py"] = sha256(Path(__file__))
    return result


def training_implementation(variant):
    result = prediction_implementation(variant)
    folder = ROOT / "nbs_models/nbs_protein_go/nbs_pg"
    for name in ("full_task_loss_v081.py", "full_task_evidence_v081.py", "full_task_schedule_v085.py",
                 "full_task_metrics_v085.py", "full_task_development_v085.py"):
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


class ProteinCycle:
    """All ranks draw the same global shuffled order and take disjoint slices."""
    def __init__(self, ids, seed):
        self.ids = np.asarray(ids, dtype=np.int64)
        if not len(self.ids) or len(np.unique(self.ids)) != len(self.ids):
            raise ValueError("training role must have nonempty unique protein IDs")
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.ids)
        self.cursor = 0
        self.passes = 0

    def take(self, n):
        # Crossing a shuffled pass boundary must not repeat a supervision seed
        # in the same global DDP batch. Defer collisions within the new order;
        # never discard proteins or draw replacement examples.
        n = int(n)
        if n < 0 or n > len(self.ids):
            raise ValueError("role batch must be nonnegative and no larger than its protein population")
        result, selected = [], set()
        while len(result) < n:
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.ids)
                self.cursor = 0
                self.passes += 1
            if int(self.order[self.cursor]) in selected:
                swap = self.cursor + 1
                while swap < len(self.order) and int(self.order[swap]) in selected:
                    swap += 1
                if swap == len(self.order):
                    raise RuntimeError("cannot form a unique role batch; duplicate protein IDs in role population")
                self.order[self.cursor], self.order[swap] = self.order[swap], self.order[self.cursor]
            protein = int(self.order[self.cursor])
            self.cursor += 1
            selected.add(protein)
            result.append(protein)
        return np.asarray(result, dtype=np.int64)

    def state(self):
        return {"order": self.order, "cursor": self.cursor, "passes": self.passes,
                "rng": self.rng.bit_generator.state}

    def restore(self, state):
        self.order, self.cursor, self.passes = state["order"], state["cursor"], state["passes"]
        self.rng.bit_generator.state = state["rng"]


def metric_pack(labels, probabilities):
    # Reuse the existing evaluator; this fast validation report explicitly uses
    # histogram micro metrics, while the final independent report uses Stage1.
    from scripts.nbs.eval_nbs_ind_test_predictions import (
        _compute_metric_pack, _per_protein_ranking_metrics,
    )
    metrics = _compute_metric_pack(labels, probabilities, threshold_step=.001, auprc_mode="hist")
    ks = tuple(k for k in (10, 50, 100) if k <= probabilities.shape[1])
    for k, values in _per_protein_ranking_metrics(labels, probabilities, ks).items():
        for name, value in values.items():
            metrics[f"{name}@{k}"] = 100 * float(value.mean())
    return metrics


@torch.no_grad()
def validate(model, data, device, batch_size, *, ablations=False):
    if not len(data.validation_ids):
        raise ValueError("set full_task.holdout_core_count > 0 for checkpoint selection")
    data.set_sampling_context(step=0, rank=0, training=False)
    model.eval()
    go = model.encode_go()
    modes = ["full"]
    if ablations:
        modes += ["weak_off", "core_off", "pp_off", "graph_off", "go_shuffle"]
    outputs = {name: [] for name in modes}
    base, labels = [], []
    for start in range(0, len(data.validation_ids), batch_size):
        batch = data.batch(data.validation_ids[start:start + batch_size], device)
        for name in modes:
            z = model(batch, go_encoding=go, **forward_flags(model.training_variant, name))
            outputs[name].append(z.sigmoid().cpu().numpy())
        base.append(batch["base_logits"].sigmoid().cpu().numpy())
        labels.append(batch["positive_mask"].cpu().numpy())
    y = np.concatenate(labels)
    metrics = {}
    for name, values in outputs.items():
        probability = np.concatenate(values)
        metrics[name] = metric_pack(y, probability)
        if name == "full":
            metrics[name].update(compute_standard_metrics(y, probability))
    base_probability = np.concatenate(base)
    metrics["backbone"] = metric_pack(y, base_probability)
    metrics["backbone"].update(compute_standard_metrics(y, base_probability))
    metrics["standard_metric_definitions"] = metric_definitions()
    metrics["standard_metric_scope"] = "full and backbone only; intervention monitors retain explicitly named legacy micro fields"
    metrics["scope"] = "stage2_core_holdout; Stage1 has previously seen core train proteins"
    metrics["selection_role"] = "core_monitor_only; not evidence of exceeding expert or modelout"
    metrics["num_proteins"] = len(data.validation_ids)
    model.train()
    return metrics


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
    """Prespecified dev selection; never a claim of passing final evaluation."""
    methods = result["methods"]
    graph, base = methods["G"], methods["B"]
    score = graph["standard_micro_pr_auc"]
    eligible = (result.get("eligible_for_selection") is True
                and graph["standard_protein_fmax"] >= base["standard_protein_fmax"]
                and score > base["standard_micro_pr_auc"])
    improved = eligible and score > previous_best
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
    loss_config = FullTaskLossConfigV081(**cfg.get("loss", {}))
    if loss_config.graph_weight != 0:
        raise ValueError("v085 keeps the old own-feature graph auxiliary disabled: set loss.graph_weight=0")
    support_config = StructuralSupportConfig(**cfg.get("structural_support", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg.get("weight_decay", 1e-4))
    schedule_config = ScheduleConfigV085.from_full_task(cfg)
    scheduler = WarmupScheduleV085(optimizer, schedule_config)
    weak_cycle = ProteinCycle(data.weak_ids, cfg["seed"])
    core_cycle = ProteinCycle(data.core_ids, cfg["seed"] + 1)
    total = int(cfg["steps"] if args.steps is None else args.steps)
    wb, cb = int(cfg["weak_batch"]), int(cfg["core_batch"])
    if min(wb, cb, total) <= 0:
        raise ValueError("steps and both role batch sizes must be positive")
    if total > schedule_config.horizon_steps:
        raise ValueError("--steps is a stop budget and must not exceed the immutable scheduler.horizon_steps")
    start = 0
    history, validations, development_history = [], [], []
    best = best_development = -float("inf")
    if args.resume:
        if args.resume.resolve().parent != out:
            raise ValueError("resume continues its original --work-dir; use the checkpoint's parent directory")
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("runner_version") != "0.8.5":
            raise ValueError("v0.8.5 needs a fresh run; older checkpoints lack this runner schedule/development contract")
        if saved.get("training_implementation") != training_implementation(variant):
            raise ValueError("resume requires unchanged v0.8.5 training implementation")
        if saved.get("training_baseline_identity") != baseline_identity:
            raise ValueError("resume backbone probability files or clipping changed")
        if saved.get("development_contract") != development_contract:
            raise ValueError("resume requires identical development data, references and exclusion contract")
        if saved["contract"] != contract or saved["world_size"] != world:
            raise ValueError("resume requires the same data, core split and number of ranks")
        if saved["model_config"] != asdict(model_config) or saved["loss_config"] != asdict(loss_config):
            raise ValueError("resume requires identical model/loss configuration")
        old_cfg = saved["config"]["full_task"]
        if old_cfg.get("variant", "fixed") != variant or old_cfg.get("structural_support", {}) != cfg.get("structural_support", {}):
            raise ValueError("resume requires identical variant and structural-support policy")
        if any(old_cfg.get(k) != cfg.get(k) for k in ("weak_batch", "core_batch", "seed", "warmup_steps", "learning_rate", "weight_decay", "grad_clip")):
            raise ValueError("resume may extend steps but must preserve batch, RNG and optimization settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved.get("scheduler", {}), expected_step=saved["step"])
        weak_cycle.restore(saved["weak_cycle"])
        core_cycle.restore(saved["core_cycle"])
        start, best = saved["step"], saved["best_score"]
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
                             "model_architecture_version": "0.8.4",
                             "weak_population": len(data.weak_ids), "core_training_population": len(data.core_ids),
                             "development_contract": development_contract,
                             "development_selection_rule": "maximize standard_micro_pr_auc among G>B on this metric and G protein_Fmax>=B; monitor only if development absent"}
        write_json(out / "resolved_config.json", config)
    distributed = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=True,
    ) if world > 1 else model
    points = {int(v) for v in cfg.get("checkpoints", [600, 2000, 4000]) if 0 < int(v) <= total} | {total}
    interval = int(cfg.get("checkpoint_interval", 1000))
    if interval > 0:
        points.update(range(interval, total + 1, interval))
    if rank == 0 and start == 0:
        initial = validate(model, data, device, cfg.get("eval_batch", wb + cb))
        validations.append({"step": 0, **initial})
        write_json(out / "validation_history.json", validations)
        if development is not None:
            result = development.evaluate(model, data, device, cfg.get("eval_batch", wb + cb),
                                          forward_flags=forward_flags(variant))
            development_history.append({"step": 0, **result})
            write_json(out / "development_history.json", development_history)
        print(f"[baseline] holdout={len(data.validation_ids)} full_GO={data.num_task_go} "
              f"micro_AP={initial['backbone']['auprc_micro_hist']:.4f}", flush=True)
    if world > 1:
        dist.barrier()
    # DDP constructor synchronizes parameters; stochastic PU selection differs by rank.
    if not args.resume:
        torch.manual_seed(cfg["seed"] + rank)
    begin = window = time.monotonic()
    aggregate = {}
    window_steps = 0
    for step in range(start + 1, total + 1):
        weak = weak_cycle.take(wb * world).reshape(world, wb)[rank]
        core = core_cycle.take(cb * world).reshape(world, cb)[rank]
        data.set_sampling_context(step=step, rank=rank, training=True)
        batch = data.batch(np.concatenate((weak, core)), device)
        lr = scheduler.apply(step)
        optimizer.zero_grad(set_to_none=True)
        logits, loss, parts = training_objective(distributed, batch, variant, loss_config, support_config)
        finite = torch.isfinite(loss).to(torch.int32)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"nonfinite loss at step {step}; no optimizer update applied")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.),
                                             error_if_nonfinite=True)
        optimizer.step()
        parts["grad_norm"] = norm.detach()
        parts["absolute_logit_delta"] = (logits.detach() - batch["base_logits"]).abs().mean()
        for key, value in parts.items():
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
            finalize_count_diagnostics(row, window_steps, world)
            row.update(step=step, seconds_per_step=(time.monotonic() - window) / window_steps,
                       elapsed_seconds=time.monotonic() - begin, learning_rate=lr,
                       weak_equivalent_passes=(step * wb * world / len(data.weak_ids)),
                       core_equivalent_passes=(step * cb * world / len(data.core_ids)),
                       weak_completed_passes=weak_cycle.passes + int(weak_cycle.cursor == len(weak_cycle.ids)),
                       core_completed_passes=core_cycle.passes + int(core_cycle.cursor == len(core_cycle.ids)),
                       scheduler_horizon_steps=schedule_config.horizon_steps,
                       optimizer_updates=step,
                       full_go_per_protein=data.num_task_go,
                       peak_allocated_gb=(torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0))
            if rank == 0:
                history.append(row)
                write_json(out / "training_history.json", history)
                print(f"[step {step}/{total}] loss={row['loss']:.5f} base={row['base_objective']:.5f} "
                      f"fit={row.get('comparable_objective', row['loss']):.5f} "
                      f"gain={row['objective_gain']:+.5f} "
                      f"pos/protein={row['positive_pairs_per_protein']:.1f} "
                      f"hard/protein={row['hard_pu_pairs_per_protein']:.1f} "
                      f"|delta|={row['absolute_logit_delta']:.4f} sec/step={row['seconds_per_step']:.3f} "
                      f"GPU={row['peak_allocated_gb']:.2f}GB lr={lr:.3g} "
                      f"passes(weak/core)={row['weak_equivalent_passes']:.3f}/{row['core_equivalent_passes']:.3f}", flush=True)
            aggregate, window_steps = {}, 0
            window = time.monotonic()
        if step in points:
            if rank == 0:
                result = validate(model, data, device, cfg.get("eval_batch", wb + cb), ablations=step == total)
                validations.append({"step": step, **result})
                score = result["full"]["auprc_micro_hist"]
                improved = score > best
                best = max(best, score)
                write_json(out / "validation_history.json", validations)
                development_improved = False
                if development is not None:
                    dev_result = development.evaluate(model, data, device, cfg.get("eval_batch", wb + cb),
                                                      forward_flags=forward_flags(variant))
                    development_history.append({"step": step, **dev_result})
                    development_improved, best_development = development_selection(dev_result, best_development)
                    write_json(out / "development_history.json", development_history)
                    print(f"[development {step}] PR_AUC={dev_result['methods']['G']['standard_micro_pr_auc']:.4f} "
                          f"protein_Fmax={dev_result['methods']['G']['standard_protein_fmax']:.4f} "
                          f"best_checkpoint={development_improved}", flush=True)
                print(f"[holdout {step}] micro_AP={score:.4f} "
                      f"base={result['backbone']['auprc_micro_hist']:.4f} "
                      f"best_checkpoint={improved}", flush=True)
            rng = {"cpu": torch.get_rng_state(),
                   "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            all_rng = [None] * world
            if world > 1:
                dist.all_gather_object(all_rng, rng)
            else:
                all_rng[0] = rng
            if rank == 0:
                state = {"version": "0.8.5", "runner_version": "0.8.5", "variant": variant,
                         "training_implementation": training_implementation(variant),
                         "training_baseline_identity": baseline_identity,
                         "step": step, "model": model.state_dict(),
                         "model_config": asdict(model_config), "loss_config": asdict(loss_config),
                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                         "model_architecture_version": "0.8.4", "config": config, "contract": contract,
                         "weak_cycle": weak_cycle.state(), "core_cycle": core_cycle.state(),
                         "rng": all_rng, "world_size": world, "best_score": best,
                         "history": history, "validations": validations,
                         "development_contract": development_contract,
                         "development_history": development_history,
                         "best_development_score": best_development}
                path = out / f"nbs_step{step}.pt"
                save_checkpoint(path, state)
                shutil.copyfile(path, out / "latest.pt")
                if improved:
                    shutil.copyfile(path, out / "best_core_holdout.pt")
                if development_improved:
                    shutil.copyfile(path, out / "best_development.pt")
            if world > 1:
                dist.barrier()
            window = time.monotonic()
    if rank == 0:
        print(f"[done] variant={variant} fixed endpoint={out / 'latest.pt'}. "
              "best_core_holdout.pt is a monitor only; no automatic pass/fail from Stage1-seen core. "
              "Evaluate fixed checkpoints against expert_prob AND stage1_modelout.", flush=True)


@torch.no_grad()
def evaluate(args, config, data, device):
    if not args.checkpoint or not args.input_dir or not args.metadata_file:
        raise ValueError("evaluate needs --checkpoint, --input-dir and --metadata-file")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("runner_version") != "0.8.5":
        raise ValueError("v0.8.5 evaluator requires a v0.8.5 checkpoint; use the original runner for old runs")
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
        raise ValueError("prediction implementation differs from the code that trained this v085 checkpoint")
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
                 and recorded.get("branch") == "final" and recorded.get("variant") == variant)
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
    manifest = {"version": "0.8.5", "runner_version": "0.8.5", "variant": variant, "step": saved["step"],
                "branch": "final", "model_architecture_version": "0.8.4",
                "scheduler": saved["scheduler"], "checkpoint_path": str(args.checkpoint.resolve()),
                "probability_path": str(probability_path),
                "prediction_implementation": implementation,
                "num_task_go": data.num_task_go, "prediction_space": "complete_task_classifier_columns",
                "uses_expert_probability_in_nbs_forward": False,
                "checkpoint_sha256": sha256(args.checkpoint),
                "input_manifest_sha256": sha256(inp / "ind_test_input_manifest.json"),
                "output_probability_sha256": sha256(probability_path), "ablation": args.ablation,
                "graph_forward": "GO_evidence_injection_then_two_relation_SAGE_layers",
                "supervision": "binary_core_gold_and_weak_pseudo_membership",
                "uses_modelout_probability_as_target": False,
                "uses_neighbor_weak_pseudo_membership": True,
                "sampling_inference": "deterministic_fixed_budget",
                "go_shuffle_semantics": "GO_association_permutation; not protein_topology_permutation",
                "contract": contract}
    write_json(manifest_path, manifest)
    command = [sys.executable, str(ROOT / "scripts/nbs/eval_nbs_full_task_v085.py"),
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
    p.add_argument("--stage", choices=("prepare", "train", "evaluate"), required=True)
    p.add_argument("--config", type=Path, default=ROOT / "nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.5_schedule.json")
    p.add_argument("--work-dir", type=Path, default=ROOT / "outputs/latence_nbs_experiments/v085/schedule")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", choices=("torch", "faiss"), default="torch")
    p.add_argument("--steps", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--metadata-file", type=Path)
    p.add_argument("--development-manifest", type=Path,
                   help="Optional prevalidated external development contract; bound permanently to a fresh run")
    p.add_argument("--ablation", choices=("full", "weak_off", "core_off", "pp_off", "graph_off", "go_shuffle"), default="full")
    p.add_argument("--metric-backend", choices=("stage1", "local_micro"), default="stage1")
    p.add_argument("--auprc-mode", choices=("exact", "hist", "none"), default="exact")
    for name in ("expert-prob", "modelout-prob", "expert-protein-ids", "modelout-protein-ids",
                 "expert-go-ids", "modelout-go-ids", "stage1-reference-manifest"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--references-aligned-to-input", action="store_true")
    p.add_argument("--allow-missing-references", action="store_true")
    args = p.parse_args()
    if args.stage == "evaluate":
        if not args.allow_missing_references and (args.expert_prob is None or args.modelout_prob is None):
            p.error("evaluation requires --expert-prob AND --modelout-prob; use --allow-missing-references only for incomplete diagnostics")
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
    data = FullTaskData(config)
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
