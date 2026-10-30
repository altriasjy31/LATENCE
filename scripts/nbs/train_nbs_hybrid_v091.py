#!/usr/bin/env python3
"""Single-GPU joint fine-tuning of a checked v089-epoch60 hybrid initialization.

New update counter and optimizer; no claim of resuming old epoch61. Reuse the
source global supervision batch, loss and sampler, splitting only computation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
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
VERSION = "0.9.1-hybrid-joint"


def validate_joint_config(cfg, source_config, source_world):
    cfg = deepcopy(cfg)
    for name in ("seed", "steps", "weak_batch", "core_batch", "accumulation_steps", "save_every", "warmup_steps"):
        if type(cfg.get(name)) is not int or cfg[name] < (0 if name in ("seed", "warmup_steps") else 1):
            raise ValueError("invalid joint." + name)
    if cfg["warmup_steps"] > cfg["steps"] or cfg["core_batch"] < cfg["accumulation_steps"]:
        raise ValueError("joint warmup/accumulation exceeds available steps/core rows")
    for name in ("encoder_lr", "shared_lr", "residual_lr", "grad_clip", "shared_aux_weight", "weight_decay"):
        value = cfg.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid joint." + name)
        if name in ("encoder_lr", "shared_lr", "residual_lr", "grad_clip") and value == 0:
            raise ValueError("joint learning rates and grad_clip must be positive")
    if type(cfg.get("detach_shared_hidden")) is not bool:
        raise ValueError("detach_shared_hidden must be boolean")
    ft = source_config["full_task"]
    if any(cfg[k] != ft[k] * source_world for k in ("weak_batch", "core_batch")):
        raise ValueError("first joint preset must retain source GLOBAL weak/core batch counts")
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--initialization", type=Path)
    group.add_argument("--resume", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--stop-step", type=int)
    args = p.parse_args(argv)
    os.chdir(ROOT)
    import torch
    from nbs_pg.shared_head_io_v091 import sha256, write_json
    from nbs_pg.full_task_epochs_v087 import EpochPlanV087, EpochSeedStreamV087
    from nbs_pg.full_task_loss_v088 import FullTaskLossConfigV088, full_task_loss_v088
    from nbs_pg.full_task_evidence_v081 import alias_pu_exclusions
    from scripts.nbs.train_nbs_full_task_v089 import split_microbatches
    from scripts.nbs.convert_nbs_shared_head_v091 import load_hybrid, INIT_VERSION
    from scripts.nbs.eval_nbs_shared_head_v091 import save_checkpoint
    out = args.work_dir.resolve()
    if args.resume is None and out.exists() and any(out.iterdir()):
        raise FileExistsError("fresh joint phase requires an empty WORK_DIR")
    if args.resume is not None and args.resume.resolve().parent != out:
        raise ValueError("joint resume must continue its original WORK_DIR")
    device = torch.device(args.device)
    checkpoint = args.resume if args.resume is not None else args.initialization
    cfg, manifest, arrays, saved, model, data = load_hybrid(args.config, checkpoint, device)
    stage = validate_joint_config(cfg["joint"], manifest["source_config"], manifest["source_world_size"])
    if args.resume is None and (saved["version"] != INIT_VERSION or not saved["conversion_check"]["passed"]):
        raise ValueError("new joint phase requires a passed hybrid_init.pt")
    # Includes source objective, seed stream and this new training algorithm.
    implementation = dict(saved["implementation"])
    for path in ("scripts/nbs/train_nbs_hybrid_v091.py", "nbs_models/nbs_protein_go/nbs_pg/full_task_loss_v088.py",
                 "nbs_models/nbs_protein_go/nbs_pg/full_task_evidence_v081.py",
                 "nbs_models/nbs_protein_go/nbs_pg/full_task_epochs_v087.py"):
        implementation[path] = sha256(ROOT / path)
    identity = {"config_sha256": sha256(args.config), "cache_manifest_sha256": saved["cache_manifest_sha256"],
                "source_sha256": manifest["source_checkpoint_sha256"], "implementation": implementation,
                "device_type": device.type, "joint_config": stage}
    torch.manual_seed(stage["seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    groups = model.optimizer_groups(stage)
    base_lrs = [g["lr"] for g in groups]
    optimizer = torch.optim.AdamW(groups, weight_decay=stage["weight_decay"])
    nsteps = (len(data.weak_ids) + stage["weak_batch"] - 1) // stage["weak_batch"]
    plan = EpochPlanV087(len(data.weak_ids), len(data.core_ids), stage["weak_batch"], stage["core_batch"], 1,
                        epochs=(stage["steps"]+nsteps-1)//nsteps)
    stream = EpochSeedStreamV087(data.weak_ids, data.core_ids, plan, seed=stage["seed"], rank=0)
    loss_cfg = FullTaskLossConfigV088(**(cfg.get("loss") or manifest["source_config"]["full_task"].get("loss", {})))
    start, history = 0, []
    if args.resume is not None:
        if saved["version"] != VERSION or saved.get("identity") != identity or saved.get("work_dir") != str(out):
            raise ValueError("joint resume requires its exact stage, source, code, cache and work directory")
        start = saved["step"]
        stream.load_state_dict(saved["seed_stream"], expected_step=start)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        history = saved["history"]
    endpoint = stage["steps"] if args.stop_step is None else args.stop_step
    if not start < endpoint <= stage["steps"]:
        raise ValueError("STOP_STEP must exceed current joint step within its fixed horizon")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "resolved_config.json", cfg)
    write_json(out / "joint_contract.json", identity)
    model.train()
    go = model.encode_go()
    parameter_list = [p for p in model.parameters() if p.requires_grad]
    for step in range(start+1, endpoint+1):
        started = time.monotonic()
        seeds = stream.next_batch()
        global_ids = np.concatenate((seeds["global_weak_ids"], seeds["global_core_ids"]))
        data.set_sampling_context(step=manifest["source_step"]+step, rank=0, training=True)
        model.backbone.set_encoder_step(step=manifest["source_step"]+step, rank=0)
        microbatches = split_microbatches(seeds["weak_ids"], seeds["core_ids"], stage["accumulation_steps"])
        optimizer.zero_grad(set_to_none=True)
        totals = {"full_bce": 0., "shared_bce": 0., "objective": 0.}
        for ids in microbatches:
            batch = data.batch(ids, device, supervision_seed_ids=global_ids)
            if not torch.equal(batch["targets"], batch["positive_mask"].float()):
                raise ValueError("joint loss requires original binary membership")
            result = model(batch, go_encoding=go, detach_shared_hidden=stage["detach_shared_hidden"])
            mask = alias_pu_exclusions(batch["positive_mask"], model.task_to_ontology)
            kwargs = dict(pu_exclusion_mask=mask,
                          global_role_counts=(seeds["global_weak_count"], seeds["global_core_count"]), world_size=1)
            full, _ = full_task_loss_v088(result["full"], batch["targets"], batch["is_weak"], loss_cfg, **kwargs)
            shared, _ = full_task_loss_v088(result["shared"], batch["targets"], batch["is_weak"], loss_cfg, **kwargs)
            loss = full + stage["shared_aux_weight"] * shared
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite joint objective")
            # Loss already uses global role denominators. Do not divide again.
            loss.backward()
            for key, value in (("full_bce", full), ("shared_bce", shared), ("objective", loss)):
                totals[key] += float(value.detach())
            del result, batch, loss, full, shared, mask
        grad = torch.nn.utils.clip_grad_norm_(parameter_list, stage["grad_clip"], error_if_nonfinite=True)
        factor = min(1., step/max(1, stage["warmup_steps"]))
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * factor
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {"step": step, "joint_weak_epoch": seeds["epoch_float"], "source_epoch": 60,
               "weak_exposures": seeds["weak_exposures"], "core_exposures": seeds["core_exposures"],
               "grad_norm": float(grad), "seconds": time.monotonic()-started,
               "learning_rates": {g["group_name"]:g["lr"] for g in optimizer.param_groups}, **totals}
        history.append(row)
        write_json(out / "training_history.json", history)
        print(f"[joint {step}/{stage['steps']}] full={totals['full_bce']:.7f} shared={totals['shared_bce']:.7f} sec={row['seconds']:.1f}", flush=True)
        if step % stage["save_every"] == 0 or step == endpoint:
            state = {key:saved[key] for key in ("source_epoch", "source_step", "source_checkpoint_sha256",
                "research_config_sha256", "cache_manifest_sha256", "head_step", "head_checkpoint_sha256", "conversion_check")}
            state.update(version=VERSION, step=step, research_config=cfg, implementation=implementation,
                identity=identity, work_dir=str(out), model=model.state_dict(), optimizer=optimizer.state_dict(),
                seed_stream=stream.state_dict(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                history=history, zero_shot=False)
            path = out / f"hybrid_step{step}.pt"
            save_checkpoint(path, state)
            latest = out / "latest.pt.tmp"
            shutil.copyfile(path, latest); latest.replace(out / "latest.pt")


if __name__ == "__main__":
    main()
