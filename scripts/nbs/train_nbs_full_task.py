#!/usr/bin/env python3
"""One entry for core retrieval preparation, full-GO training and evaluation."""
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

from nbs_pg.full_task_data import FullTaskData
from nbs_pg.full_task_loss import FullTaskLossConfig, full_task_loss
from nbs_pg.full_task_model import FullTaskGraphModel, FullTaskModelConfig


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
        if not len(self.ids):
            raise ValueError("training role has no labeled proteins")
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.ids)
        self.cursor = 0
        self.passes = 0

    def take(self, n):
        result = []
        while n:
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.ids)
                self.cursor = 0
                self.passes += 1
            take = min(n, len(self.order) - self.cursor)
            result.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            n -= take
        return np.concatenate(result) if result else np.empty(0, dtype=np.int64)

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
    model.eval()
    go = model.encode_go()
    modes = {"full": (True, True)}
    if ablations:
        modes.update(weak_off=(False, True), core_off=(True, False))
    outputs = {name: [] for name in modes}
    base, labels = [], []
    for start in range(0, len(data.validation_ids), batch_size):
        batch = data.batch(data.validation_ids[start:start + batch_size], device)
        for name, (weak_on, core_on) in modes.items():
            z = model(batch, go_encoding=go, use_weak_go=weak_on, use_core_go=core_on)
            outputs[name].append(z.sigmoid().cpu().numpy())
        base.append(batch["base_logits"].sigmoid().cpu().numpy())
        labels.append(batch["positive_mask"].cpu().numpy())
    y = np.concatenate(labels)
    metrics = {name: metric_pack(y, np.concatenate(values)) for name, values in outputs.items()}
    metrics["backbone"] = metric_pack(y, np.concatenate(base))
    metrics["scope"] = "stage2_core_holdout; Stage1 has previously seen core train proteins"
    metrics["num_proteins"] = len(data.validation_ids)
    model.train()
    return metrics


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(state, temporary)
    temporary.replace(path)


def train(args, config, data, device, rank, world):
    cfg = config["full_task"]
    out = args.work_dir.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if not args.resume and (out / "training_history.json").exists():
            raise FileExistsError(f"Run exists: use --resume {out / 'latest.pt'} or a new --work-dir")
    if world > 1:
        dist.barrier()
    if not data._load_neighbors():
        raise FileNotFoundError("Run prepare first; the shell pilot command does this automatically")
    ontology = data.ontology("cpu")
    contract = {"data": data.data_contract(), "ontology": ontology_contract(ontology)}
    model_config = FullTaskModelConfig(**cfg.get("model", {}))
    model = FullTaskGraphModel(data.feature_dim, ontology, model_config).to(device)
    loss_config = FullTaskLossConfig(**cfg.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg.get("weight_decay", 1e-4))
    weak_cycle = ProteinCycle(data.weak_ids, cfg["seed"])
    core_cycle = ProteinCycle(data.core_ids, cfg["seed"] + 1)
    total = int(args.steps or cfg["steps"])
    wb, cb = int(cfg["weak_batch"]), int(cfg["core_batch"])
    if min(wb, cb, total) <= 0:
        raise ValueError("steps and both role batch sizes must be positive")
    start = 0
    history, validations = [], []
    best = -float("inf")
    if args.resume:
        if args.resume.resolve().parent != out:
            raise ValueError("resume continues its original --work-dir; use the checkpoint's parent directory")
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["contract"] != contract or saved["world_size"] != world:
            raise ValueError("resume requires the same data, core split and number of ranks")
        if saved["model_config"] != asdict(model_config) or saved["loss_config"] != asdict(loss_config):
            raise ValueError("resume requires identical model/loss configuration")
        old_cfg = saved["config"]["full_task"]
        if any(old_cfg.get(k) != cfg.get(k) for k in ("weak_batch", "core_batch", "seed", "warmup_steps", "learning_rate", "weight_decay", "grad_clip")):
            raise ValueError("resume may extend steps but must preserve batch, RNG and optimization settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        weak_cycle.restore(saved["weak_cycle"])
        core_cycle.restore(saved["core_cycle"])
        start, best = saved["step"], saved["best_score"]
        history, validations = saved["history"], saved["validations"]
        rng = saved["rng"][rank]
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
    if total <= start:
        raise ValueError(f"requested steps={total} must exceed resumed step={start}")
    if rank == 0:
        config["runtime"] = {"requested_steps": total, "world_size": world,
                             "global_weak_batch": wb * world, "global_core_batch": cb * world}
        write_json(out / "resolved_config.json", config)
    distributed = DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False, find_unused_parameters=True,
    ) if world > 1 else model
    points = {int(v) for v in cfg.get("checkpoints", [100, 300, 600]) if int(v) <= total} | {total}
    if rank == 0 and start == 0:
        initial = validate(model, data, device, cfg.get("eval_batch", wb + cb))
        validations.append({"step": 0, **initial})
        write_json(out / "validation_history.json", validations)
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
        batch = data.batch(np.concatenate((weak, core)), device)
        lr = cfg["learning_rate"] * min(1., step / max(1, cfg.get("warmup_steps", 25)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = distributed(batch)
        loss, parts = full_task_loss(logits, batch["base_logits"], batch["targets"],
                                     batch["positive_mask"], batch["is_weak"], loss_config)
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
            row.update(step=step, seconds_per_step=(time.monotonic() - window) / window_steps,
                       elapsed_seconds=time.monotonic() - begin, learning_rate=lr,
                       weak_equivalent_passes=(step * wb * world / len(data.weak_ids)),
                       full_go_per_protein=data.num_task_go,
                       peak_allocated_gb=(torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0))
            if rank == 0:
                history.append(row)
                write_json(out / "training_history.json", history)
                print(f"[step {step}/{total}] loss={row['loss']:.5f} base={row['base_objective']:.5f} "
                      f"gain={row['objective_gain']:+.5f} pos/protein={row['positive_pairs_per_protein']:.1f} "
                      f"hard/protein={row['hard_pu_pairs_per_protein']:.1f} "
                      f"|delta|={row['absolute_logit_delta']:.4f} sec/step={row['seconds_per_step']:.3f} "
                      f"GPU={row['peak_allocated_gb']:.2f}GB", flush=True)
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
                state = {"version": FullTaskGraphModel.VERSION, "step": step, "model": model.state_dict(),
                         "model_config": asdict(model_config), "loss_config": asdict(loss_config),
                         "optimizer": optimizer.state_dict(), "config": config, "contract": contract,
                         "weak_cycle": weak_cycle.state(), "core_cycle": core_cycle.state(),
                         "rng": all_rng, "world_size": world, "best_score": best,
                         "history": history, "validations": validations}
                path = out / f"nbs_step{step}.pt"
                save_checkpoint(path, state)
                shutil.copyfile(path, out / "latest.pt")
                if improved:
                    shutil.copyfile(path, out / "best.pt")
            if world > 1:
                dist.barrier()
            window = time.monotonic()
    if rank == 0:
        print(f"[done] {out / 'best.pt'} selected by Stage2 holdout, not independent test. "
              "Check validation_history.json against backbone before extending training.", flush=True)
        baseline_score = validations[0]["backbone"]["auprc_micro_hist"]
        if best <= baseline_score:
            print("[pilot decision] No heldout micro-AP gain over backbone. Do not extend "
                  "training unchanged; inspect the existing full/weak_off/core_off comparison.", flush=True)


@torch.no_grad()
def evaluate(args, config, data, device):
    if not args.checkpoint or not args.input_dir or not args.metadata_file:
        raise ValueError("evaluate needs --checkpoint, --input-dir and --metadata-file")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ontology = data.ontology("cpu")
    contract = {"data": data.data_contract(), "ontology": ontology_contract(ontology)}
    if saved["contract"] != contract:
        raise ValueError("checkpoint data/ontology/split differs from the evaluation config")
    model = FullTaskGraphModel(data.feature_dim, ontology, FullTaskModelConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"])
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
                 and recorded.get("contract") == contract and recorded.get("ablation") == args.ablation)
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
            logits = model(batch, go_encoding=go, use_weak_go=args.ablation != "weak_off",
                           use_core_go=args.ablation != "core_off")
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
    manifest = {"version": FullTaskGraphModel.VERSION, "step": saved["step"],
                "num_task_go": data.num_task_go, "prediction_space": "complete_task_classifier_columns",
                "uses_expert_probability_in_nbs_forward": False,
                "checkpoint_sha256": sha256(args.checkpoint),
                "input_manifest_sha256": sha256(inp / "ind_test_input_manifest.json"),
                "output_probability_sha256": sha256(probability_path), "ablation": args.ablation,
                "graph_forward": "shared_train_external_target_local", "contract": contract}
    write_json(manifest_path, manifest)
    command = [sys.executable, str(ROOT / "scripts/nbs/eval_nbs_ind_test_predictions.py"),
               "--task", config["task"], "--metadata-file", str(args.metadata_file.resolve()),
               "--nbs-prob", str(probability_path), "--backbone-prob", str(inp / "backbone_ind_test_prob.f16.npy"),
               "--protein-ids", str(inp / "protein_ids.txt"), "--candidate-go-index", str(inp / "candidate_go_index.i32.npy"),
               "--input-manifest", str(inp / "ind_test_input_manifest.json"), "--prediction-manifest", str(manifest_path),
               "--precision-k", "10,50,100", "--metric-backend", args.metric_backend,
               "--bootstrap-replicates", "0", "--output-dir", str(destination / "metrics")]
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "train", "evaluate"), required=True)
    p.add_argument("--config", type=Path, default=ROOT / "nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.0.json")
    p.add_argument("--work-dir", type=Path, default=ROOT / "outputs/latence_nbs_experiments/v080/main")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", choices=("torch", "faiss"), default="torch")
    p.add_argument("--steps", type=int)
    p.add_argument("--resume", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--input-dir", type=Path)
    p.add_argument("--metadata-file", type=Path)
    p.add_argument("--ablation", choices=("full", "weak_off", "core_off"), default="full")
    p.add_argument("--metric-backend", choices=("stage1", "local_micro"), default="stage1")
    args = p.parse_args()
    os.chdir(ROOT)
    config = json.loads(args.config.resolve().read_text())
    cfg = config["full_task"]
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
