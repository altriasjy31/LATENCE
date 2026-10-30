#!/usr/bin/env python3
"""Train only a shared GO decoder from immutable, frozen-v089 feature caches.

This is a new stage with its own optimizer/update counter, not a continuation
of the v089 schedule or a six-layer v090 run. The original classifier and all
GNN/ontology parameters remain unchanged. No test labels enter optimization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nbs_models/nbs_protein_go")]
VERSION = "0.9.1-shared-head-stage1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validated_config(config, manifest):
    """Resolve inherited source loss without changing the source experiment."""
    result = deepcopy(config)
    if result.get("schema") not in (None, "nbs-shared-head-v091"):
        raise ValueError("unsupported shared-head research config schema")
    if type(result.get("source_epoch")) is not int or result["source_epoch"] != 60:
        raise ValueError("this preset binds source_epoch=60")
    if manifest.get("source_epoch") != result["source_epoch"]:
        raise ValueError("cache source epoch differs from requested epoch60")
    if manifest.get("complete") is not True:
        raise ValueError("training requires a complete frozen feature cache")
    train = result["train"]
    for name in ("seed", "weak_batch", "core_batch", "steps", "save_every"):
        value = train.get(name)
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"train.{name} is invalid")
    if type(train.get("warmup_steps")) is not int or not 0 <= train["warmup_steps"] <= train["steps"]:
        raise ValueError("warmup_steps must be an integer within the new stage horizon")
    for name in ("learning_rate", "weight_decay", "grad_clip"):
        value = train.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"train.{name} must be finite")
        if value < 0 or (name in ("learning_rate", "grad_clip") and value == 0):
            raise ValueError(f"train.{name} is outside its valid range")
    if result.get("loss") is None:
        result["loss"] = deepcopy(manifest["source_config"]["full_task"].get("loss", {}))
    for name in ("core_weight", "weak_weight"):
        value = result["loss"].get(name, 1.0 if name == "core_weight" else .25)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"loss.{name} must be finite nonnegative")
    if sum(result["loss"].get(k, d) for k, d in (("core_weight", 1.), ("weak_weight", .25))) <= 0:
        raise ValueError("at least one role must have positive loss weight")
    return result


def stage_learning_rate(step, cfg):
    """The head phase has a fresh, explicitly recorded warmup/constant LR."""
    if not 1 <= step <= cfg["steps"]:
        raise ValueError("step must lie within the frozen new-stage horizon")
    return float(cfg["learning_rate"]) * min(1., step / max(cfg["warmup_steps"], 1))


def sample_rows(rng, weak_rows, core_rows, weak_batch, core_batch):
    if weak_batch > len(weak_rows) or core_batch > len(core_rows):
        raise ValueError("head minibatch exceeds cached population")
    return np.concatenate((rng.choice(weak_rows, weak_batch, replace=False),
                           rng.choice(core_rows, core_batch, replace=False))).astype(np.int64)


def dense_membership(indptr, go_idx, rows, num_go):
    """Preserve CSR binary membership, including duplicates and immutable aliases."""
    rows = np.asarray(rows, np.int64)
    result = np.zeros((len(rows), num_go), np.float32)
    for index, row in enumerate(rows):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        columns = np.asarray(go_idx[start:stop], np.int64)
        if np.any(columns < 0) or np.any(columns >= num_go):
            raise ValueError("cached label column outside immutable task space")
        result[index, columns] = 1.
    return result


def assert_resume_contract(saved, identity, work_dir):
    if saved.get("version") != VERSION or saved.get("identity") != identity:
        raise ValueError("resume requires unchanged head config, source, cache and implementation")
    if saved.get("work_dir") != str(Path(work_dir).resolve()):
        raise ValueError("resume must continue the original shared-head work directory")
    if type(saved.get("step")) is not int or not 0 < saved["step"] <= identity["config"]["train"]["steps"]:
        raise ValueError("invalid shared-head checkpoint update counter")


def training_implementation():
    module = ROOT / "nbs_models/nbs_protein_go/nbs_pg"
    paths = [Path(__file__), module / "shared_head_v091.py", module / "shared_head_io_v091.py",
             module / "full_task_loss_v088.py", module / "full_task_evidence_v081.py"]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def train(args):
    import torch
    from nbs_pg.shared_head_v091 import SharedGeometryHeadV091
    from nbs_pg.shared_head_io_v091 import load_cache
    from nbs_pg.full_task_loss_v088 import FullTaskLossConfigV088, full_task_loss_v088
    from nbs_pg.full_task_evidence_v081 import alias_pu_exclusions

    config_path, out = args.config.resolve(), args.work_dir.resolve()
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    cache_dir = Path(raw_config["cache_dir"]).resolve()
    manifest, arrays = load_cache(cache_dir)
    config = validated_config(raw_config, manifest)
    source = Path(config["source_checkpoint"]).resolve()
    if not source.is_file() or sha256(source) != manifest["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint differs from the frozen cache")
    device = torch.device(args.device)
    identity = {"config": config, "config_sha256": sha256(config_path),
                "device_type": device.type, "cache_manifest_sha256": sha256(cache_dir / "cache_manifest.json"),
                "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
                "implementation": training_implementation()}
    if args.resume is None:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("new shared-head training needs an empty work directory")
    elif args.resume.resolve().parent != out:
        raise ValueError("--resume must point into the same shared-head work directory")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config["train"]["seed"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = SharedGeometryHeadV091(np.array(arrays["geometry.f32.npy"], copy=True), **config["head"]).to(device)
    hidden = arrays["hidden.f32.npy"]
    if hidden.shape[1] != model.hidden_dim:
        raise ValueError("head hidden dimension differs from source GNN representation")
    mapping = torch.as_tensor(np.array(arrays["task_to_ontology.i64.npy"], copy=True), device=device)
    role = np.asarray(arrays["is_weak.bool.npy"], bool)
    weak_rows, core_rows = np.flatnonzero(role), np.flatnonzero(~role)
    cfg = config["train"]
    if cfg["weak_batch"] > len(weak_rows) or cfg["core_batch"] > len(core_rows):
        raise ValueError("cached role populations are smaller than the selected head batch")
    loss_config = FullTaskLossConfigV088(**config["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    history, start = [], 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert_resume_contract(saved, identity, out)
        model.load_state_dict(saved["head_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            if saved.get("cuda_rng") is None:
                raise ValueError("CUDA resume requires its original device RNG state")
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        start, history = saved["step"], saved["history"]
    endpoint = cfg["steps"] if args.stop_step is None else args.stop_step
    if type(endpoint) is not int or not start < endpoint <= cfg["steps"]:
        raise ValueError("stop-step must exceed resumed step within the frozen 500-update horizon")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "resolved_config.json", config)
    model.train()
    started = time.monotonic()
    for step in range(start + 1, endpoint + 1):
        rows = sample_rows(rng, weak_rows, core_rows, cfg["weak_batch"], cfg["core_batch"])
        x = torch.as_tensor(np.array(hidden[rows], copy=True), device=device)
        target = torch.as_tensor(dense_membership(arrays["label_indptr.i64.npy"], arrays["label_go_idx.i32.npy"],
                                                 rows, model.geometry.shape[0]), device=device)
        is_weak = torch.as_tensor(role[rows], dtype=torch.bool, device=device)
        logits = model(x)
        excluded = alias_pu_exclusions(target.bool(), mapping)
        loss, parts = full_task_loss_v088(logits, target, is_weak, loss_config,
            pu_exclusion_mask=excluded, global_role_counts=(cfg["weak_batch"], cfg["core_batch"]), world_size=1)
        if not torch.isfinite(loss):
            raise FloatingPointError("shared head produced nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        if not torch.isfinite(grad):
            raise FloatingPointError("shared head produced nonfinite gradient norm")
        lr = stage_learning_rate(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad), "lr": lr,
               "head_weak_exposures": step * cfg["weak_batch"], "head_core_exposures": step * cfg["core_batch"],
               "elapsed_seconds_since_resume": time.monotonic() - started,
               **{key: float(value) for key, value in parts.items() if key.startswith("contrib_")}}
        for key in ("positive_pairs_per_protein", "positive_probability", "negative_probability",
                    "alias_excluded_pairs_per_protein", "core_proteins", "weak_proteins"):
            row[key] = float(parts[key])
        positive_counts = target.sum(dim=1)
        row["core_positive_pairs_per_protein"] = float(positive_counts[~is_weak].mean())
        row["weak_positive_pairs_per_protein"] = float(positive_counts[is_weak].mean())
        history.append(row)
        if step % 10 == 0 or step == endpoint:
            write_json(out / "training_history.json", history)
            print(f"[shared-head {step}/{cfg['steps']}] BCE={row['loss']:.7f} grad={row['grad_norm']:.5g} lr={lr:.5g}", flush=True)
        if step % cfg["save_every"] == 0 or step == endpoint:
            state = {"version": VERSION, "step": step, "identity": identity, "work_dir": str(out),
                     "head_state": model.state_dict(), "head_config": config["head"],
                     "config_sha256": identity["config_sha256"],
                     "training_implementation": identity["implementation"],
                     "cache_manifest_sha256": identity["cache_manifest_sha256"],
                     "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
                     "source_epoch": manifest["source_epoch"], "source_step": manifest["source_step"],
                     "optimizer": optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                     "history": history, "scope": "frozen-encoder shared-head pretraining; full v089 unchanged; no zero-shot claim"}
            path = out / f"shared_head_step{step}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            latest = out / "latest.pt.tmp"
            shutil.copyfile(path, latest)
            latest.replace(out / "latest.pt")
    print(f"[done] shared-head checkpoint: {path}; original v089 classifier and encoders unchanged", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-step", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    os.chdir(ROOT)
    train(args)


if __name__ == "__main__":
    main()
