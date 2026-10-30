#!/usr/bin/env python3
"""Evaluate fixed shared/hybrid checkpoints with original Fmax and micro AP.
All task columns are retained. No test-based selection or GO propagation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nbs_models/nbs_protein_go")]
SCHEMA = "nbs-shared-head-evaluation-v091"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def log(message):
    print(f"[shared-eval {time.strftime('%H:%M:%S')}] {message}", flush=True)


def metric_helpers():
    candidates = (Path(__file__).resolve().parent / "diagnose_nbs_knn_v090.py",
                  ROOT / "scripts/nbs/diagnose_nbs_knn_v090.py")
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("diagnose_nbs_knn_v090.py is required in scripts/nbs")
    spec = importlib.util.spec_from_file_location("_shared_v091_metrics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compose_residuals(original_weight, original_bias, shared_weight, shared_bias):
    """Pure-array counterpart of the residual initialization; aliases retained."""
    old_w, old_b, q, b = (np.asarray(x, np.float32) for x in
                          (original_weight, original_bias, shared_weight, shared_bias))
    if old_w.ndim != 2 or old_w.shape != q.shape or old_b.shape != (len(old_w),) or b.shape != old_b.shape:
        raise ValueError("source/shared head dimensions differ")
    if any(not np.isfinite(x).all() for x in (old_w, old_b, q, b)):
        raise ValueError("source/shared head parameters must be finite")
    return old_w - q, old_b - b


def validate_head_binding(saved, config, config_sha, manifest_sha, source_sha):
    """A head may only be evaluated against its exact cache and research config."""
    if saved.get("version") != "0.9.1-shared-head-stage1":
        raise ValueError("evaluation requires a v091 frozen shared-head checkpoint")
    for key, value in (("config_sha256", config_sha), ("cache_manifest_sha256", manifest_sha),
                       ("source_checkpoint_sha256", source_sha)):
        if saved.get(key) != value:
            raise ValueError("shared head binding differs: " + key)
    if saved.get("source_epoch") != config.get("source_epoch", 60):
        raise ValueError("shared head source epoch differs")
    if saved.get("head_config") != config.get("head"):
        raise ValueError("shared head architecture differs from research config")
    step = saved.get("step")
    if type(step) is not int or not 0 < step <= config["train"]["steps"]:
        raise ValueError("invalid fixed shared head update")
    if "head_state" not in saved or not isinstance(saved.get("training_implementation"), dict) or not saved["training_implementation"]:
        raise ValueError("shared head lacks state or implementation identity")


def validate_training_implementation(recorded, root):
    for name, digest in recorded.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError("invalid training implementation record")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("training implementation path must be project relative")
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("shared training implementation differs: " + name)


def compare_composed(source_prob, composed_prob, source_metrics, composed_metrics, *, probability_atol=1e-3, metric_atol=.02):
    """Engineering consistency check, not a statistical equivalence claim."""
    source = np.asarray(source_prob, np.float32)
    composed = np.asarray(composed_prob, np.float32)
    if source.shape != composed.shape or not source.size or not np.isfinite(source).all() or not np.isfinite(composed).all():
        raise ValueError("invalid source/composed probability matrices")
    if np.any((source < 0) | (source > 1) | (composed < 0) | (composed > 1)):
        raise ValueError("source/composed probabilities must lie in [0,1]")
    if any(not np.isfinite(x) or x < 0 for x in (probability_atol, metric_atol)):
        raise ValueError("composition tolerances must be finite and nonnegative")
    delta = {name: abs(source_metrics[name] - composed_metrics[name]) for name in ("fmax", "micro_ap")}
    maximum = float(np.abs(source - composed).max())
    return {"max_abs_probability_difference": maximum, "probability_atol": probability_atol,
            "metric_atol_percentage_points": metric_atol, "absolute_metric_delta_percentage_points": delta,
            "passed": maximum <= probability_atol and all(v <= metric_atol for v in delta.values()),
            "scope": "residual initialization check only; not a trained improved model"}


def core_counts_from_cache(arrays, num_go):
    ptr = np.asarray(arrays["label_indptr.i64.npy"])
    weak = np.asarray(arrays["is_weak.bool.npy"], bool)
    go = np.asarray(arrays["label_go_idx.i32.npy"])
    if ptr.shape != (len(weak) + 1,) or ptr[-1] != len(go):
        raise ValueError("cache label dimensions differ")
    label_is_weak = np.repeat(weak, np.diff(ptr))
    return np.bincount(go[~label_is_weak], minlength=num_go)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--head-checkpoint", type=Path)
    g.add_argument("--hybrid-checkpoint", type=Path)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--metadata-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    os.chdir(ROOT)
    import torch
    from nbs_pg.shared_head_io_v091 import load_source
    from nbs_pg.hybrid_head_v091 import HybridGraphModelV091
    from scripts.nbs.convert_nbs_shared_head_v091 import load_bound_head, load_hybrid, INIT_VERSION
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("evaluation needs a new/empty output directory")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    source = None
    if args.head_checkpoint is not None:
        cfg, manifest, arrays, saved, head = load_bound_head(args.config, args.head_checkpoint, device)
        source, data, _, metadata = load_source(cfg["source_config"], cfg["source_checkpoint"], device,
            sampler=cfg["cache"]["sampler"], expected_epoch=60)
        if metadata["source_checkpoint_sha256"] != manifest["source_checkpoint_sha256"]:
            raise ValueError("source differs from cache")
        model = HybridGraphModelV091(source, head).to(device).eval()
        methods = ("source", "shared", "hybrid_initialization", "shared_weight_original_bias", "original_weight_shared_bias")
        selected = args.head_checkpoint
        phase = "frozen_shared_head"
    else:
        cfg, manifest, arrays, saved, model, data = load_hybrid(args.config, args.hybrid_checkpoint, device)
        if args.hybrid_checkpoint.name != ("hybrid_init.pt" if saved["version"] == INIT_VERSION else f"hybrid_step{saved['step']}.pt"):
            raise ValueError("select a fixed hybrid checkpoint, not latest.pt")
        methods = ("full", "shared")
        selected = args.hybrid_checkpoint
        phase = "hybrid_initialization" if saved["version"] == INIT_VERSION else "joint"
    helpers = metric_helpers()
    test = helpers.load_test_inputs(args.input_dir, args.metadata_file, manifest["source_config"]["task"],
        data._signature["representation_manifest_sha256"])
    labels = test["labels"]
    rows, num_go = labels.shape
    if num_go != len(arrays["geometry.f32.npy"]):
        raise ValueError("independent task columns differ from source")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {"schema": SCHEMA, "complete": False, "phase": phase,
              "source_epoch": 60, "checkpoint_step": saved["step"], "checkpoint_sha256": sha256(selected),
              "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
              "all_go_columns_retained": True, "shared_only_is_zero_shot": False,
              "scope": "fixed-checkpoint independent-test evaluation; never automatic model selection",
              "input_manifest_sha256": sha256(args.input_dir / "ind_test_input_manifest.json")}
    write_json(args.output_dir / "evaluation_manifest.json", record)
    probabilities = {name:np.lib.format.open_memmap(args.output_dir / (name+"_ind_test_prob.f32.npy"),
                         mode="w+", dtype=np.float32, shape=(rows, num_go)) for name in methods}
    batch_size = manifest["source_config"]["full_task"].get("eval_batch", 16)
    model.eval()
    data.set_sampling_context(0, 0, False)
    from nbs_pg.shared_head_io_v091 import candidate_statistics, update_candidate_statistics, finish_candidate_statistics
    candidate = candidate_statistics()
    with torch.no_grad():
        go = model.encode_go()
        prototypes = model.shared.prototypes()
        for start in range(0, rows, batch_size):
            ids = np.arange(start, min(start+batch_size, rows))
            batch = data.inference_batch(args.input_dir, ids, device)
            hidden = model.backbone(batch, go_encoding=go)
            values = model.score_hidden(hidden, prototypes=prototypes)
            if source is not None:
                values = {"source":torch.nn.functional.linear(hidden, model.original_weight, model.original_bias),
                          "shared":values["shared"], "hybrid_initialization":values["full"],
                          "shared_weight_original_bias":torch.nn.functional.linear(hidden, prototypes[0], model.original_bias),
                          "original_weight_shared_bias":torch.nn.functional.linear(hidden, model.original_weight, prototypes[1])}
            for name in methods:
                p = values[name].sigmoid().cpu().numpy()
                if not np.isfinite(p).all():
                    raise FloatingPointError("nonfinite evaluation probabilities")
                probabilities[name][start:start+len(ids)] = p
            update_candidate_statistics(candidate, batch["sampled_candidate_go"][batch["sampled_seed_index"]].cpu().numpy(),
                batch["sampled_candidate_attr"][batch["sampled_seed_index"]].cpu().numpy()[...,0], [np.flatnonzero(row) for row in labels[ids]])
            if start // batch_size % 20 == 0:
                log(f"predicted {start+len(ids)}/{rows}")
    for p in probabilities.values():
        p.flush()
    results = {name:{**helpers.protein_fmax(labels, p), "micro_ap":helpers.grouped_ap(labels,p)}
               for name,p in probabilities.items()}
    counts = core_counts_from_cache(arrays,num_go)
    summary = {**record, "metrics": results,
               "frequency_strata":helpers.frequency_strata(labels,{n:(p,results[n]["threshold"]) for n,p in probabilities.items()},counts),
               "candidate_quality":finish_candidate_statistics(candidate)}
    if source is not None:
        check = compare_composed(probabilities["source"],probabilities["hybrid_initialization"],
                                 results["source"],results["hybrid_initialization"], probability_atol=2e-6, metric_atol=0.0001)
        summary["composition_check"] = check
        write_json(args.output_dir / "shared_head_metrics.json", summary)
        if not check["passed"]:
            raise ValueError("source/hybrid initialization consistency failed")
    summary["complete"] = True
    write_json(args.output_dir / "shared_head_metrics.json", summary)
    record.update(complete=True, probability_files={n:{"filename":Path(p.filename).name,"sha256":sha256(p.filename)} for n,p in probabilities.items()})
    write_json(args.output_dir / "evaluation_manifest.json",record)
    log(json.dumps(results))


if __name__ == "__main__":
    main()
