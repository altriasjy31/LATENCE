#!/usr/bin/env python3
"""Reproduce Stage-1 external_only/modelout references without evaluation labels.

The decoder's actual qout['logits'] is the modelout, including its learned gate
and trained anchor.  This program does not call the diagnostics evaluation loop,
which also performs label-dependent analyses unrelated to reference inference.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "v081-stage1-references-1"
FILES = {"expert_prob": "expert_prob.f32.npy", "stage1_modelout": "stage1_modelout.f32.npy",
         "backbone_recomputed": "backbone_recomputed.f32.npy", "protein_ids": "protein_ids.txt",
         "go_ids": "go_ids.txt"}
MANIFEST = "stage1_reference_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_path(value, relative: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else relative / path).resolve()


def ids_from_file(path: Path) -> list[str]:
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim != 1:
            raise ValueError(f"Expected one-dimensional ID file: {path}")
        ids = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in array.tolist()]
    else:
        ids = [v.strip() for v in path.read_text().splitlines() if v.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"IDs must be nonempty and unique: {path}")
    return ids


def ordered_indices(source: list[str], target: list[str], label: str) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(source)}
    if len(lookup) != len(source):
        raise ValueError(f"Duplicate {label} IDs")
    missing = [value for value in target if value not in lookup]
    if missing:
        raise ValueError(f"Missing {label} IDs: {missing[:8]}")
    return np.asarray([lookup[value] for value in target], dtype=np.int64)


def source_record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def check_hash(path: Path, expected: str | None, label: str) -> None:
    if expected and sha256(path) != expected:
        raise ValueError(f"{label} SHA256 differs from the independent input contract: {path}")


def resolve_contract(args):
    root = args.project_root = args.project_root.expanduser().resolve()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.metadata_file = args.metadata_file.expanduser().resolve()
    args.output_dir = (args.output_dir or args.input_dir / "stage1_references_v081").expanduser().resolve()
    if args.output_dir == args.input_dir:
        raise ValueError("Reference output-dir must differ from the independent input directory")
    manifest_path = args.input_dir / "ind_test_input_manifest.json"
    manifest = read_json(manifest_path)
    signature = manifest.get("cache_signature", {})
    if manifest.get("task", signature.get("task")) != args.task:
        raise ValueError("Task differs from independent input manifest")
    if manifest.get("mode", "ind_test") != "ind_test":
        raise ValueError("Only independent-test input caches are supported")
    encoding = manifest.get("sequence_encoding", {}).get("mode", signature.get("sequence_encoding"))
    if encoding != "metadata_msa_binary":
        raise ValueError("Automatic references require the same metadata_msa_binary inputs as Stage-1 diagnostics")
    registries = manifest["registries"]
    weak_path = resolve_path(registries["weak_graph_manifest"], manifest_path.parent)
    weak = read_json(weak_path)
    check_hash(weak_path, signature.get("weak_graph_manifest_sha256"), "Weak graph manifest")
    checkpoint = args.stage1_checkpoint or weak.get("checkpoint", {}).get("path")
    if not checkpoint:
        raise ValueError("No Stage-1 checkpoint in input weak manifest; provide --stage1-checkpoint")
    args.stage1_checkpoint = resolve_path(checkpoint, root)
    expected_checkpoint = signature.get("stage1_checkpoint_sha256")
    if not expected_checkpoint:
        raise ValueError("Independent input manifest lacks stage1_checkpoint_sha256")
    check_hash(args.stage1_checkpoint, expected_checkpoint, "Stage-1 checkpoint")
    check_hash(args.stage1_checkpoint, weak.get("checkpoint", {}).get("sha256"), "Weak graph checkpoint")
    args.msa_index = resolve_path(args.msa_index or root / "data/ind_MSA_bin/index.pkl", root)
    check_hash(args.msa_index, signature.get("msa_index_sha256"), "MSA index")
    args.diagnostic_script = resolve_path(args.diagnostic_script or root / "experiments/eval_weak_ind_test_detr_diagnostics.py", root)
    args.external_prob_path = resolve_path(args.external_prob_path or root / f"data/external_probs/esm2_3b/{args.task}_predictions.prop.float16.npy", root)
    if not args.external_prob_path.is_file():
        raise FileNotFoundError(f"Stage-1 external/expert probability not found: {args.external_prob_path}. "
                                "Set --external-prob-path to the EXTERNAL_PROB_PATH used by Stage-1 diagnostics; no fallback is attempted.")
    if args.stage1_model_config:
        args.stage1_model_config = resolve_path(args.stage1_model_config, root)
    if args.train_args_json == "auto":
        args_path = next((p for p in (args.stage1_checkpoint.parent / "args.json", args.stage1_checkpoint.parent.parent / "args.json") if p.is_file()), None)
    elif args.train_args_json:
        args_path = resolve_path(args.train_args_json, root)
        args.train_args_json = str(args_path)
    else:
        args_path = None

    prepare = import_file("_stage1_references_input_helpers", root / "scripts/nbs/prepare_nbs_ind_test_inputs.py")
    with args.metadata_file.open("rb") as stream:
        block = prepare._task_block(pickle.load(stream), "ind_test", args.task)
    values = prepare._first_present(block, None, ("proteins", "protein_ids", "ids"))
    if values is None:
        raise ValueError("Metadata ind_test block has no protein IDs")
    metadata_ids = [str(v) for v in values]
    if not metadata_ids or len(metadata_ids) != len(set(metadata_ids)):
        raise ValueError("Metadata independent-test protein IDs must be unique")
    protein_path = args.input_dir / "protein_ids.txt"
    proteins = ids_from_file(protein_path)
    ordered_indices(metadata_ids, proteins, "metadata protein")
    if manifest.get("num_proteins", signature.get("num_proteins")) != len(proteins):
        raise ValueError("Input protein count differs from manifest")
    check_hash(protein_path, manifest.get("protein_ids_file_sha256"), "Input protein IDs")
    prepare.resolve_and_audit_msa_index(args, [(p, None) for p in proteins])
    go_path = resolve_path(registries["go_registry"], manifest_path.parent)
    check_hash(go_path, registries.get("go_registry_sha256") or signature.get("go_registry_sha256"), "GO registry")
    with go_path.open(newline="") as stream:
        rows = sorted(csv.DictReader(stream, delimiter="\t"), key=lambda row: int(row["go_idx"]))
    if [int(row["go_idx"]) for row in rows] != list(range(len(rows))):
        raise ValueError("GO registry must retain contiguous original task columns")
    gos = [row["input_go_id"].strip() for row in rows]
    if not gos or len(gos) != len(set(gos)) or len(gos) != signature.get("num_classes"):
        raise ValueError("GO registry original input_go_id columns must be unique and match input class count")
    base_path = args.input_dir / "backbone_ind_test_prob.f16.npy"
    check_hash(base_path, manifest.get("base_probability", {}).get("sha256"), "Cached backbone probability")
    base = np.load(base_path, mmap_mode="r", allow_pickle=False)
    if base.shape != (len(proteins), len(gos)):
        raise ValueError("Cached backbone probability dimensions differ from task IDs")

    semantics = weak.get("model_semantics", {}).get("modelout", {})
    modelout_key = args.modelout_key or semantics.get("prediction_key") or "modelout::mix_expert_base_anchor::decoderprob::expert"
    if not modelout_key.startswith("modelout::"):
        modelout_key = "modelout::" + modelout_key
    parts = modelout_key.split("::")
    if len(parts) != 4 or parts[2:] != ["decoderprob", "expert"]:
        raise ValueError(f"Unsupported modelout key {modelout_key!r}; this exporter reproduces decoderprob::expert only. "
                         "Other decoder sources must be implemented explicitly, never silently substituted.")
    if semantics.get("prediction_key") and modelout_key != semantics["prediction_key"]:
        raise ValueError("Requested modelout key differs from NBS weak supervision manifest")
    if semantics.get("decoder_prob_source", "expert") != "expert" or semantics.get("topk_source", "external_topk") != "external_topk" or float(semantics.get("external_prob_blend_alpha", 1.0)) != 1.0 or semantics.get("label_hint_used", False):
        raise ValueError("Weak modelout semantics differ from label-free decoderprob::expert external_topk inference")

    expert = np.load(args.external_prob_path, mmap_mode="r", allow_pickle=False)
    if expert.ndim != 2:
        raise ValueError("External probability must be a two-dimensional .npy array")
    if args.external_protein_ids:
        args.external_protein_ids = resolve_path(args.external_protein_ids, root)
        external_ids = ids_from_file(args.external_protein_ids)
        row_contract = "explicit_external_protein_ids"
    else:
        external_ids = metadata_ids
        row_contract = "Stage1_PseudoProbDataset_metadata_ind_test_order"
    if len(external_ids) != expert.shape[0]:
        raise ValueError("External probability rows do not match metadata ind_test; a universe-sized file requires --external-protein-ids")
    row_index = ordered_indices(external_ids, proteins, "external protein")
    if args.external_go_ids:
        args.external_go_ids = resolve_path(args.external_go_ids, root)
        external_gos = ids_from_file(args.external_go_ids)
        if len(external_gos) != expert.shape[1]:
            raise ValueError("External GO ID count differs from matrix columns")
        column_index = ordered_indices(external_gos, gos, "external GO")
        column_contract = "explicit_external_go_ids_matched_to_original_input_go_id"
    else:
        if expert.shape[1] != len(gos):
            raise ValueError("External probability columns differ from Stage-1 task; provide --external-go-ids")
        column_index = np.arange(len(gos), dtype=np.int64)
        column_contract = "inherited_Stage1_ordered_task_columns_no_external_GO_sidecar"

    sources = {"input_manifest": manifest_path, "weak_manifest": weak_path, "checkpoint": args.stage1_checkpoint,
               "metadata": args.metadata_file, "msa_index": args.msa_index, "go_registry": go_path,
               "input_proteins": protein_path, "cached_backbone": base_path, "external_probability": args.external_prob_path,
               "reference_script": Path(__file__).resolve(), "input_helper": Path(prepare.__file__),
               "export_helper": root / "scripts/export_weak_graph_predictions.py", "diagnostic_script": args.diagnostic_script}
    for name, path in (("train_args_json", args_path), ("model_config_override", args.stage1_model_config),
                       ("external_protein_ids", args.external_protein_ids), ("external_go_ids", args.external_go_ids)):
        if path:
            sources[name] = Path(path)
    contract = {"version": VERSION, "sources": {key: source_record(path) for key, path in sources.items()},
                "task": args.task, "shape": [len(proteins), len(gos)], "modelout_key": modelout_key,
                "batch_size": args.batch_size, "device": args.device, "no_amp": args.no_amp,
                "amp_dtype": args.amp_dtype, "backbone_atol": args.backbone_atol,
                "external_row_contract": row_contract, "external_column_contract": column_contract,
                "external_column_identity_independently_verified": bool(args.external_go_ids)}
    return {"contract": contract, "input_manifest": manifest, "semantics": semantics, "prepare": prepare,
            "proteins": proteins, "gos": gos, "go_path": go_path, "weak_path": weak_path,
            "expert": expert, "row_index": row_index, "column_index": column_index, "base": base,
            "args_path": args_path}


def cached_manifest(output_dir: Path, contract: dict) -> dict | None:
    path = output_dir / MANIFEST
    if not path.is_file():
        return None
    try:
        manifest = read_json(path)
        if manifest.get("cache_signature") != contract:
            return None
        records = list(manifest["outputs"].values()) + list(manifest["runtime_source_files"].values())
        for record in records:
            file = Path(record["path"])
            if not file.is_file() or sha256(file) != record["sha256"]:
                return None
        return manifest
    except (KeyError, ValueError, OSError, TypeError):
        return None


def load_runtime(args, context):
    """Use the strict production Stage-1 loader, without its evaluation loop."""
    root = args.project_root
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "msa_models"))
    import torch
    exporter = import_file("_nbs_reference_export_helpers", root / "scripts/export_weak_graph_predictions.py")
    diagnostic = import_file("_nbs_reference_stage1_diagnostic", args.diagnostic_script)
    weak = sys.modules.get("experiments.weak_exp_train_detr")
    exp = sys.modules.get("experiments.exp_train")
    if weak is None or exp is None:
        raise ImportError("Stage-1 diagnostic did not import its model and MSA dataset modules")
    seed = ["--project-root", str(root), "--diagnostic-eval-script", str(args.diagnostic_script),
            "--checkpoint", str(args.stage1_checkpoint), "--task", args.task,
            "--output-dir", str(args.output_dir), "--feature-dir", str(args.input_dir),
            "--go-registry", str(context["go_path"]), "--roles", "ind_test", "--modelout-role", "ind_test",
            "--file-address", str(args.metadata_file), "--working-address", str(args.msa_index),
            "--batch-size", str(args.batch_size), "--dataloader-num-workers", "0", "--device", args.device,
            "--train-args-json", args.train_args_json, "--amp-dtype", args.amp_dtype]
    if args.stage1_model_config:
        seed += ["--model-config", str(args.stage1_model_config)]
    if args.no_amp:
        seed += ["--no-amp"]
    export_args = exporter.build_parser().parse_args(seed)
    payload = exporter.torch_load(torch, args.stage1_checkpoint)
    model_config, config_sources = exporter.find_model_args(args.stage1_checkpoint, payload, args.train_args_json)
    if not model_config.get("model_config") and not args.stage1_model_config:
        default_config = root / f"data/msa_models/configs/model_opts/{args.task}_msa_model_config.pkl"
        if default_config.is_file():
            export_args.model_config = str(default_config)
    model_args, resolved_sources = exporter.make_eval_model_args(args=export_args, diagnostic_module=diagnostic, payload_model_args=model_config)
    model_args.mode = "ind_test"
    model_args.allow_eval_label_boost = False
    check_hash(Path(model_args.model_config), context["input_manifest"].get("cache_signature", {}).get("stage1_model_config_sha256"), "Stage-1 model config")
    model_args.num_classes = int(model_args.num_classes)
    if model_args.num_classes != len(context["gos"]):
        raise ValueError("Restored Stage-1 class count differs from original task columns")
    mode = str(model_args.query_decoder_logit_base_mode)
    if context["contract"]["modelout_key"].split("::")[1] != mode:
        raise ValueError(f"Restored checkpoint anchor mode {mode!r} differs from required modelout key")
    semantics = context["semantics"]
    if "expert_base_mix_alpha" in semantics and not np.isclose(float(model_args.expert_base_mix_alpha), float(semantics["expert_base_mix_alpha"]), rtol=0, atol=1e-12):
        raise ValueError("Restored checkpoint expert_base_mix_alpha differs from weak supervision manifest")
    device = exporter.resolve_device(torch, args.device)
    if device.type == "cuda" and args.min_free_gpu_gb > 0:
        free, _ = torch.cuda.mem_get_info(device)
        if free / 1024**3 < args.min_free_gpu_gb:
            raise RuntimeError(f"Stage-1 {device} has less than {args.min_free_gpu_gb} GiB free; choose --device on a free GPU")
    model_args.device = str(device)
    model_args.gpu_ids = "" if device.type == "cpu" else str(device.index or 0)
    exp.set_seed(int(model_args.seed))
    opt = weak.build_weak_opt_from_config(model_args)
    opt.mode, opt.shuffle = "ind_test", False
    model = weak.WeakMSAGOWithDETRDecoder(opt, model_args)
    load_info = exporter.load_checkpoint_strict(model, payload, allow_partial=False)
    model = model.to(device).eval()
    runtime_sources = {"resolved_model_config": source_record(Path(model_args.model_config)),
                       "weak_model_module": source_record(Path(weak.__file__)),
                       "msa_runtime_module": source_record(Path(exp.__file__))}
    if hasattr(exp, "MSABinaryDataset"):
        runtime_sources["msa_binary_dataset_module"] = source_record(Path(inspect.getfile(exp.MSABinaryDataset)))
    runtime_sources["backbone_model_module"] = source_record(Path(inspect.getfile(type(model.backbone))))
    # Record checkpoint/args disagreements: production export correctly gives the
    # checkpoint architecture precedence over an older adjacent args.json.
    conflicts = {}
    if context["args_path"] is not None and isinstance(payload, dict):
        disk_args = read_json(context["args_path"])
        for key, value in payload.get("model_args", {}).items():
            if key in disk_args and disk_args[key] != value:
                conflicts[key] = {"args_json": disk_args[key], "checkpoint_used": value}
    return {"torch": torch, "device": device, "model": model, "model_args": model_args, "opt": opt, "exp": exp,
            "runtime_source_files": runtime_sources, "checkpoint_load": load_info,
            "config_sources": config_sources, "resolved_arg_sources": resolved_sources, "configuration_conflicts": conflicts,
            "runtime_environment": {"torch_version": torch.__version__, "device": str(device),
                                    "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}


def forward_probabilities(runtime, encoded, expert, args):
    """Exact canonical Stage-1 modelout; never reconstruct base + delta."""
    torch, model, device = runtime["torch"], runtime["model"], runtime["device"]
    dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    encoded = encoded.to(device=device, dtype=torch.long)
    expert = torch.from_numpy(expert).to(device=device, dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and not args.no_amp):
        base_logits, hidden = model.backbone(encoded, permute_dims=tuple(runtime["model_args"].permute_dims), return_embedding=True)
        output = model.query_decoder(h=hidden, base_logits=base_logits, classifier_weight=model.classifier.weight,
                                     external_prob=expert, has_external_prob=torch.ones(encoded.shape[0], device=device, dtype=torch.bool),
                                     topk_source="external_topk", external_prob_blend_alpha=1.0, y_hint=None)
        if "logits" not in output:
            raise KeyError("Stage-1 query decoder did not return final logits")
        base = torch.sigmoid(base_logits.float()).cpu().numpy()
        modelout = torch.sigmoid(output["logits"].float()).cpu().numpy()
    return base, modelout


def infer_references(args, context, stage: Path):
    runtime = load_runtime(args, context)
    exp, opt = runtime["exp"], copy.copy(runtime["opt"])
    proteins, gos = context["proteins"], context["gos"]
    selection = stage / "msa_selection.pkl"
    metadata_task = context["prepare"].TASK_KEYS[args.task][1]
    with selection.open("wb") as stream:
        pickle.dump({"ind_test": {metadata_task: {"proteins": proteins, "prop_annotations": [[] for _ in proteins]}}}, stream)
    opt.file_address, opt.working_address = str(selection), str(args.msa_index)
    dataset = exp.build_msa_dataset(opt, mode="ind_test", task=metadata_task, need_proteins=True)
    dataset.return_labels = False
    if len(dataset) != len(proteins):
        raise ValueError("MSA dataset length differs from requested protein IDs")
    loader = exp.make_loader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False,
                             drop_last=False, rank=0, world_size=1, seed=3407, persistent_workers=False)
    arrays = {key: np.lib.format.open_memmap(stage / FILES[key], mode="w+", dtype=np.float32,
              shape=(len(proteins), len(gos))) for key in ("expert_prob", "stage1_modelout", "backbone_recomputed")}
    row_by_id = {value: i for i, value in enumerate(proteins)}
    seen = set()
    parity_max, parity_sum, parity_exceed, parity_count = 0.0, 0.0, 0, 0
    try:
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                raise ValueError("Expected label-free (protein_ids, encoded_MSA) batches")
            batch_proteins, encoded = batch
            batch_proteins = [str(p) for p in batch_proteins]
            if any(p not in row_by_id for p in batch_proteins):
                raise ValueError("MSA loader returned a protein outside the independent input")
            rows = np.asarray([row_by_id[p] for p in batch_proteins], dtype=np.int64)
            if len(rows) != len(set(rows.tolist())) or any(int(i) in seen for i in rows):
                raise ValueError("MSA loader repeated independent-test proteins")
            seen.update(rows.tolist())
            ext = np.asarray(context["expert"][context["row_index"][rows]][:, context["column_index"]], dtype=np.float32)
            if not np.isfinite(ext).all() or np.any((ext < 0) | (ext > 1)):
                raise ValueError("External probability contains nonfinite/out-of-range values")
            # PseudoProbDataset stores/returns external values through float16.
            ext = ext.astype(np.float16).astype(np.float32)
            exp.set_model_proteins(runtime["model"], batch_proteins)
            base, modelout = forward_probabilities(runtime, encoded, ext, args)
            if base.shape != ext.shape or modelout.shape != ext.shape or not np.isfinite(base).all() or not np.isfinite(modelout).all():
                raise ValueError("Stage-1 output is nonfinite or has incorrect shape")
            error = np.abs(base.astype(np.float64) - np.asarray(context["base"][rows], dtype=np.float64))
            parity_max = max(parity_max, float(error.max()))
            parity_sum += float(error.sum())
            parity_exceed += int((error > args.backbone_atol).sum())
            parity_count += int(error.size)
            arrays["expert_prob"][rows], arrays["stage1_modelout"][rows], arrays["backbone_recomputed"][rows] = ext, modelout, base
            print(f"[Stage-1 references] proteins={len(seen)}/{len(proteins)} B_max_abs_error={parity_max:.7f}", flush=True)
        if len(seen) != len(proteins):
            raise ValueError("MSA loader did not cover every requested protein")
        if parity_exceed:
            raise ValueError(f"Recomputed backbone differs from cached B: max_abs={parity_max:.7f}, "
                             f"pairs_above_atol={parity_exceed}/{parity_count}, atol={args.backbone_atol}. "
                             "Check checkpoint, independent MSA sampling, model options and AMP; cached B was not changed.")
        for array in arrays.values():
            array.flush()
    finally:
        arrays.clear()
    selection.unlink()
    parity = {"max_abs_error": parity_max, "mean_abs_error": parity_sum / parity_count,
              "pairs_above_atol": parity_exceed, "pair_count": parity_count, "atol": args.backbone_atol,
              "cached_backbone_replaced": False}
    info = {key: runtime[key] for key in ("runtime_source_files", "checkpoint_load", "config_sources", "resolved_arg_sources", "configuration_conflicts")}
    info["resolved_model_args"] = vars(runtime["model_args"])
    info["runtime_environment"] = runtime.get("runtime_environment", {})
    info["backbone_parity"] = parity
    return info


def check_output_ownership(output_dir: Path):
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"Reference output-dir is not a directory: {output_dir}")
    contents = {path.name for path in output_dir.iterdir()}
    if not contents:
        return
    known = set(FILES.values()) | {MANIFEST}
    try:
        old = read_json(output_dir / MANIFEST)
    except (OSError, ValueError, TypeError):
        old = {}
    owned = old.get("schema_version") == 1 and str(old.get("builder", "")).startswith("v081-stage1-references-")
    if not owned or contents - known:
        raise ValueError(f"Refusing to replace non-owned/non-reference contents in output-dir: {output_dir}. "
                         "Choose a dedicated Stage-1 reference directory; existing files were not changed.")


def publish_directory(stage: Path, output_dir: Path):
    """Manifest is complete before directory swap; failed generation preserves old cache."""
    check_output_ownership(output_dir)
    backup = None
    if output_dir.exists():
        backup = Path(tempfile.mkdtemp(prefix=".stage1_refs_previous_", dir=output_dir.parent))
        backup.rmdir()
        os.replace(output_dir, backup)
    try:
        os.replace(stage, output_dir)
    except BaseException:
        if backup is not None:
            os.replace(backup, output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def prepare_references(args) -> dict:
    if args.batch_size < 1 or not 0 <= args.backbone_atol <= 0.05:
        raise ValueError("batch-size must be positive; backbone-atol must be within [0,0.05]")
    context = resolve_contract(args)
    check_output_ownership(args.output_dir)
    cached = cached_manifest(args.output_dir, context["contract"])
    if args.cache_policy != "refresh" and cached is not None:
        print(f"[Reuse Stage-1 references] {args.output_dir / MANIFEST}", flush=True)
        return cached
    if args.cache_policy == "require":
        raise RuntimeError("Compatible Stage-1 E/M references were not found; run with --cache-policy reuse to create them")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stage1_refs_prepare_", dir=args.output_dir.parent))
    try:
        info = infer_references(args, context, stage)
        (stage / FILES["protein_ids"]).write_text("\n".join(context["proteins"]) + "\n")
        (stage / FILES["go_ids"]).write_text("\n".join(context["gos"]) + "\n")
        outputs = {key: {"path": str(args.output_dir / filename), "sha256": sha256(stage / filename)} for key, filename in FILES.items()}
        manifest = {"schema_version": 1, "builder": VERSION, "cache_signature": context["contract"],
                    "outputs": outputs, **info,
                    "semantics": {"expert_prob": "Stage-1 external_only, source quantized float16 then converted to float32",
                                  "stage1_modelout": context["contract"]["modelout_key"],
                                  "modelout_formula": "sigmoid(query_decoder(..., external_prob=E, y_hint=None)['logits'].float())",
                                  "decoder_topk": "restored from Stage-1 checkpoint/model_args; not NBS candidate_topk",
                                  "output_dtype": "float32", "historical_diagnostic_store_dtype": "float16 by default; modelout output quantization may differ",
                                  "labels_consumed_by_model": False, "ind_test_label_boost": False,
                                  "ic_fusion_computed": False}}
        (stage / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n")
        publish_directory(stage, args.output_dir)
        print(f"[Stage-1 references ready] {args.output_dir / MANIFEST}", flush=True)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--metadata-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task", choices=["bp", "mf", "cc"], default="bp")
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--external-prob-path", type=Path)
    parser.add_argument("--external-protein-ids", type=Path)
    parser.add_argument("--external-go-ids", type=Path)
    parser.add_argument("--msa-index", type=Path)
    parser.add_argument("--stage1-model-config", type=Path)
    parser.add_argument("--train-args-json", default="auto")
    parser.add_argument("--diagnostic-script", type=Path)
    parser.add_argument("--modelout-key")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-free-gpu-gb", type=float, default=8.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--backbone-atol", type=float, default=0.002,
                        help="Maximum probability difference from cached B; tolerates small AMP/kernel and float16-storage changes, never relabels M as B")
    parser.add_argument("--cache-policy", choices=["reuse", "require", "refresh"], default="reuse")
    return parser


def main(argv=None):
    return prepare_references(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
