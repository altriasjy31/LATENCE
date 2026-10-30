#!/usr/bin/env python3
"""Convert a fixed shared-head checkpoint into a checked full hybrid model.

No independent-test labels are read. R2 separates strict same-hidden head
equivalence from repeated graph-forward numerical variation. Encoder state,
structure and eval modes must remain unchanged. No tolerance was widened.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nbs_models/nbs_protein_go")]
INIT_VERSION = "0.9.1-hybrid-init"
CHECK_VERSION = "v091-convert-r2-same-hidden"


def _error(first, second):
    """Never let NaN disappear through Python max(0., nan)."""
    import torch
    if first.shape != second.shape:
        raise ValueError("comparison tensor shapes differ")
    if not first.numel() or not torch.isfinite(first).all() or not torch.isfinite(second).all():
        return None
    value = float((first - second).abs().max())
    return value if math.isfinite(value) else None


def _within(value, tolerance):
    return value is not None and math.isfinite(value) and value <= tolerance


def _encoder_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()
            if not name.startswith("classifier.")}


def _encoder_structure(model):
    return {name: (id(module), type(module).__module__, type(module).__qualname__, module.training)
            for name, module in model.named_modules() if not name.startswith("classifier")}


def _forward_and_hidden(model, classifier, batch, go):
    captured = []
    handle = classifier.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()))
    try:
        output = model(batch, go_encoding=go)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("expected exactly one classifier input capture")
    return captured[0], output["full"] if isinstance(output, dict) else output


def run_conversion_checks(source, head, batch, go, cached_hidden, *, hybrid_factory=None):
    """Check actual hidden tensors, state identity, and immutable cached rows.

    Repeated source and hybrid forwards are measurements, not head-equivalence
    gates: they need not present the classifier with identical hidden values.
    Each actual forward is instead checked against the original linear head
    on THAT forward's captured hidden tensor, under the original tolerances.
    """
    import torch
    import torch.nn.functional as F
    if hybrid_factory is None:
        from nbs_pg.hybrid_head_v091 import HybridGraphModelV091
        hybrid_factory = HybridGraphModelV091
    if source.training or head.training:
        raise ValueError("conversion requires source and shared head in eval mode")
    with torch.no_grad():
        original_head = source.classifier
        before = _encoder_state(source)
        structure = _encoder_structure(source)
        check = {"check_version": CHECK_VERSION, "logit_atol": 2e-5, "probability_atol": 2e-6,
                 "cached_hidden_rows": min(4096, len(cached_hidden)),
                 "cached_max_abs_logit": 0., "cached_max_abs_probability": 0.,
                 "cached_nonfinite_batches": 0, "source_repeat_comparisons": [],
                 "hybrid_repeat_comparisons": [],
                 "independent_forward_comparison_role": "diagnostic_only; different hidden tensors",
                 "numerical_cause": "not_identified_by_this_check",
                 "independent_test_metrics_checked": False}
        device = original_head.weight.device
        original_hidden, original = _forward_and_hidden(source, original_head, batch, go)
        original_reconstructed = F.linear(original_hidden, original_head.weight, original_head.bias)
        check["source_head_reconstruction_max_abs_logit"] = _error(original, original_reconstructed)
        check["source_head_reconstruction_max_abs_probability"] = _error(original.sigmoid(), original_reconstructed.sigmoid())
        # A source-vs-source control measures repeatability without conversion.
        for _ in range(2):
            h, z = _forward_and_hidden(source, original_head, batch, go)
            reference = F.linear(h, original_head.weight, original_head.bias)
            check["source_repeat_comparisons"].append({
                "max_abs_hidden": _error(original_hidden, h),
                "max_abs_logit": _error(original, z),
                "max_abs_probability": _error(original.sigmoid(), z.sigmoid()),
                "same_hidden_max_abs_logit": _error(reference, z),
                "same_hidden_max_abs_probability": _error(reference.sigmoid(), z.sigmoid())})
        hybrid = hybrid_factory(source, head).to(device).eval()
        q = hybrid.shared.prototypes()
        weight, bias = hybrid.effective_parameters(q)
        check["effective_weight_equal"] = bool(torch.equal(weight, original_head.weight))
        check["effective_bias_equal"] = bool(torch.equal(bias, original_head.bias))
        check["original_weight_equal"] = bool(torch.equal(hybrid.original_weight, original_head.weight))
        check["original_bias_equal"] = bool(torch.equal(hybrid.original_bias, original_head.bias))
        check["residual_delta_is_zero"] = bool(
            torch.count_nonzero(hybrid.residual_delta_weight) == 0 and
            torch.count_nonzero(hybrid.residual_delta_bias) == 0)
        same_hidden = hybrid.score_hidden(original_hidden)["full"]
        check["same_hidden_max_abs_logit"] = _error(original, same_hidden)
        check["same_hidden_max_abs_probability"] = _error(original.sigmoid(), same_hidden.sigmoid())
        for repeat in range(2):
            h, z = _forward_and_hidden(hybrid, hybrid.backbone.classifier, batch, go)
            reference = F.linear(h, original_head.weight, original_head.bias)
            row = {"max_abs_hidden_vs_source": _error(original_hidden, h),
                   "max_abs_logit_vs_source": _error(original, z),
                   "max_abs_probability_vs_source": _error(original.sigmoid(), z.sigmoid()),
                   "same_hidden_max_abs_logit": _error(reference, z),
                   "same_hidden_max_abs_probability": _error(reference.sigmoid(), z.sigmoid())}
            check["hybrid_repeat_comparisons"].append(row)
            if repeat == 0:
                # Retain R1 field names, but explicitly change their role.
                check["real_batch_max_abs_logit"] = row["max_abs_logit_vs_source"]
                check["real_batch_max_abs_probability"] = row["max_abs_probability_vs_source"]
        for start in range(0, check["cached_hidden_rows"], 128):
            x = torch.as_tensor(np.array(cached_hidden[start:min(start+128, check["cached_hidden_rows"])], copy=True), device=device)
            old = F.linear(x, original_head.weight, original_head.bias)
            new = hybrid.score_hidden(x, prototypes=q)["full"]
            logit_error, probability_error = _error(old,new), _error(old.sigmoid(),new.sigmoid())
            if logit_error is None or probability_error is None:
                check["cached_nonfinite_batches"] += 1
            else:
                check["cached_max_abs_logit"] = max(check["cached_max_abs_logit"], logit_error)
                check["cached_max_abs_probability"] = max(check["cached_max_abs_probability"], probability_error)
        after = _encoder_state(hybrid.backbone)
        check["encoder_state_unchanged"] = (before.keys() == after.keys() and
            all(torch.equal(before[k],after[k]) for k in before))
        check["encoder_structure_and_modes_unchanged"] = structure == _encoder_structure(hybrid.backbone)
        strict = {name:check[name] for name in (
            "effective_weight_equal", "effective_bias_equal", "original_weight_equal", "original_bias_equal",
            "residual_delta_is_zero", "encoder_state_unchanged", "encoder_structure_and_modes_unchanged")}
        strict["source_reconstruction"] = (
            _within(check["source_head_reconstruction_max_abs_logit"],check["logit_atol"]) and
            _within(check["source_head_reconstruction_max_abs_probability"],check["probability_atol"]))
        strict["same_hidden_readout"] = (
            _within(check["same_hidden_max_abs_logit"],check["logit_atol"]) and
            _within(check["same_hidden_max_abs_probability"],check["probability_atol"]))
        repeated = check["source_repeat_comparisons"] + check["hybrid_repeat_comparisons"]
        strict["repeated_forwards_finite_and_readouts_correct"] = all(
            all(value is not None for value in row.values()) and
            _within(row["same_hidden_max_abs_logit"],check["logit_atol"]) and
            _within(row["same_hidden_max_abs_probability"],check["probability_atol"])
            for row in repeated)
        strict["cached_readout"] = (
            check["cached_hidden_rows"] > 0 and check["cached_nonfinite_batches"] == 0 and
            _within(check["cached_max_abs_logit"],check["logit_atol"]) and
            _within(check["cached_max_abs_probability"],check["probability_atol"]))
        check["strict_checks"] = strict
        check["failed_checks"] = [name for name,passed in strict.items() if not passed]
        check["passed"] = not check["failed_checks"]
        check["interpretation"] = (
            "State-preserving conversion and same-hidden head checks passed; independent-forward numerical repeatability is reported separately."
            if check["passed"] else "Conversion failed one or more strict checks; no initialization saved.")
        return hybrid, check


def load_bound_head(config_path, checkpoint, device):
    import torch
    from nbs_pg.shared_head_io_v091 import load_cache, sha256
    from nbs_pg.shared_head_v091 import SharedGeometryHeadV091
    from scripts.nbs.eval_nbs_shared_head_v091 import validate_head_binding, validate_training_implementation
    config = json.loads(Path(config_path).read_text())
    manifest, arrays = load_cache(config["cache_dir"])
    cache_sha = sha256(Path(config["cache_dir"]) / "cache_manifest.json")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_head_binding(saved, config, sha256(config_path), cache_sha, manifest["source_checkpoint_sha256"])
    if Path(checkpoint).name != f"shared_head_step{saved['step']}.pt":
        raise ValueError("choose a fixed shared_head_step<N>.pt, not latest.pt")
    validate_training_implementation(saved["training_implementation"], ROOT)
    if (sha256(config["source_checkpoint"]) != manifest["source_checkpoint_sha256"] or
        json.loads(Path(config["source_config"]).read_text()) != manifest["source_config"]):
        raise ValueError("cache/source checkpoint or config differs")
    head = SharedGeometryHeadV091(np.array(arrays["geometry.f32.npy"], copy=True), **saved["head_config"]).to(device)
    head.load_state_dict(saved["head_state"], strict=True)
    head.eval()
    return config, manifest, arrays, saved, head


def hybrid_implementation():
    from nbs_pg.shared_head_io_v091 import sha256
    paths = ["nbs_models/nbs_protein_go/nbs_pg/" + n for n in
             ("shared_head_v091.py", "shared_head_io_v091.py", "hybrid_head_v091.py", "full_task_sampler_fast.py")]
    paths += ["scripts/nbs/convert_nbs_shared_head_v091.py"]
    return {p: sha256(ROOT / p) for p in paths}


def load_hybrid(config_path, checkpoint, device):
    import torch
    from nbs_pg.shared_head_io_v091 import load_cache, load_source, sha256
    from nbs_pg.shared_head_v091 import SharedGeometryHeadV091
    from nbs_pg.hybrid_head_v091 import HybridGraphModelV091
    from scripts.nbs.eval_nbs_shared_head_v091 import validate_training_implementation
    cfg = json.loads(Path(config_path).read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("version") not in (INIT_VERSION, "0.9.1-hybrid-joint"):
        raise ValueError("requires new hybrid init or joint checkpoint, never old v089 RESUME")
    manifest, arrays = load_cache(cfg["cache_dir"])
    if (state.get("research_config_sha256") != sha256(config_path) or
        state.get("cache_manifest_sha256") != sha256(Path(cfg["cache_dir"]) / "cache_manifest.json") or
        state.get("source_checkpoint_sha256") != manifest["source_checkpoint_sha256"]):
        raise ValueError("hybrid checkpoint/source/config/cache binding differs")
    validate_training_implementation(state["implementation"], ROOT)
    source, data, ontology, metadata = load_source(cfg["source_config"], cfg["source_checkpoint"], device,
        sampler=cfg["cache"]["sampler"], expected_epoch=60)
    if metadata["source_checkpoint_sha256"] != manifest["source_checkpoint_sha256"]:
        raise ValueError("source changed since caching")
    head = SharedGeometryHeadV091(np.array(arrays["geometry.f32.npy"], copy=True), **cfg["head"]).to(device)
    model = HybridGraphModelV091(source, head).to(device)
    model.load_state_dict(state["model"], strict=True)
    if not torch.equal(model.original_weight.cpu(), torch.from_numpy(np.array(arrays["original_weight.f32.npy"]))):
        raise ValueError("hybrid anchor differs from original source weights")
    if not torch.equal(model.original_bias.cpu(), torch.from_numpy(np.array(arrays["original_bias.f32.npy"]))):
        raise ValueError("hybrid anchor differs from original source bias")
    model.eval()
    return cfg, manifest, arrays, state, model, data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--head-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)
    os.chdir(ROOT)
    import torch
    from nbs_pg.shared_head_io_v091 import load_source, sha256, write_json
    from nbs_pg.hybrid_head_v091 import HybridGraphModelV091
    from scripts.nbs.eval_nbs_shared_head_v091 import save_checkpoint
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("conversion needs a new/empty output directory")
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    cfg, manifest, arrays, saved, head = load_bound_head(args.config, args.head_checkpoint, device)
    source, data, _, metadata = load_source(cfg["source_config"], cfg["source_checkpoint"], device,
        sampler=cfg["cache"]["sampler"], expected_epoch=60)
    if metadata["source_checkpoint_sha256"] != manifest["source_checkpoint_sha256"]:
        raise ValueError("source differs from cached representation")
    for key, param in (("original_weight.f32.npy", source.classifier.weight), ("original_bias.f32.npy", source.classifier.bias)):
        if not np.array_equal(arrays[key], param.detach().cpu().numpy()):
            raise ValueError("cached original classifier differs")
    global_count = cfg["cache"]["global_weak_batch"] + cfg["cache"]["global_core_batch"]
    ids = np.asarray(arrays["protein_ids.i64.npy"][:min(16, global_count, len(arrays["protein_ids.i64.npy"]))])
    data.set_sampling_context(step=0, rank=0, training=False)
    with torch.no_grad():
        go = source.encode_go()
        global_count = cfg["cache"]["global_weak_batch"] + cfg["cache"]["global_core_batch"]
        forbidden = np.asarray(arrays["protein_ids.i64.npy"][:global_count])
        batch = data.batch(ids, device, supervision_seed_ids=forbidden)
        hybrid, check = run_conversion_checks(source, head, batch, go, arrays["hidden.f32.npy"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "conversion_check.json", check)
    print(json.dumps(check, ensure_ascii=False), flush=True)
    if not check["passed"]:
        raise ValueError("hybrid conversion consistency failed: " + ", ".join(check["failed_checks"]) + "; see conversion_check.json")
    state = {"version": INIT_VERSION, "step": 0, "source_epoch": 60, "source_step": manifest["source_step"],
        "source_checkpoint_sha256": manifest["source_checkpoint_sha256"], "research_config": cfg,
        "research_config_sha256": sha256(args.config),
        "cache_manifest_sha256": sha256(Path(cfg["cache_dir"]) / "cache_manifest.json"),
        "head_step": saved["step"], "head_checkpoint_sha256": sha256(args.head_checkpoint),
        "model": {k:v.detach().cpu() for k,v in hybrid.state_dict().items()}, "implementation": hybrid_implementation(),
        "conversion_check": check, "zero_shot": False}
    save_checkpoint(args.output_dir / "hybrid_init.pt", state)
    print(f"[convert] saved {args.output_dir / 'hybrid_init.pt'}", flush=True)


if __name__ == "__main__":
    main()
