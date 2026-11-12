#!/usr/bin/env python3
"""Export frozen ESM2ResNet features immediately before output_layer[0].

Inputs are the existing esm_models PLM cache (mean layers 34,35,36), not GO
probabilities. A single explicitly selected checkpoint is the default protocol.
Independent checkpoint hidden coordinates MUST NOT be averaged; explicit
--ensemble concat preserves them as separate feature segments.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

import numpy as np

FORMAT = "latence-esm-penultimate-v096-v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_ids(path, fasta=False):
    path = Path(path)
    if fasta:
        values = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.startswith(">"):
                    parts = line[1:].strip().split()
                    if not parts:
                        raise ValueError("Empty FASTA identifier")
                    values.append(parts[0])
    elif path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 1 or arr.dtype.kind not in "US":
            raise ValueError("IDs NPY must be a 1D Unicode/bytes array, not pickle")
        values = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in arr]
    else:
        values = path.read_text(encoding="utf-8").splitlines()
    if not values:
        raise ValueError("Empty protein ID list")
    seen = set()
    for value in values:
        if not value or value != value.strip() or any(c.isspace() for c in value):
            raise ValueError(f"Invalid protein ID: {value!r}")
        if value in seen:
            raise ValueError(f"Duplicate protein ID: {value}")
        if value in {".", ".."} or any(c in value for c in "/\\\x00"):
            raise ValueError(f"Unsafe cache filename ID: {value!r}")
        seen.add(value)
    return np.asarray(values, dtype=str)


def load_plm_batch(features_dir, ids, layers, embed_dim, digest=None):
    """Bounded reader; bypasses the native Dataset's broken low_memory branch.

    Legacy esm_models caches contain pickled dictionaries. Only use cache files
    you generated or trust. Output artifacts themselves never require pickle.
    """
    directories = [features_dir] if isinstance(features_dir, (str, Path)) else list(features_dir)
    batch = np.empty((len(ids), len(layers) * embed_dim), dtype=np.float32)
    for row, pid in enumerate(ids):
        candidates = [Path(folder) / (str(pid) + ".npy") for folder in directories]
        matches = [p for p in candidates if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"Missing ESM cache for {pid!r} in {directories}")
        path = matches[0]
        payload = path.read_bytes()
        for duplicate in matches[1:]:
            if duplicate.read_bytes() != payload:
                raise ValueError(f"Conflicting duplicated cache for {pid!r}: {path} and {duplicate}")
        if digest is not None:
            digest.update(str(pid).encode("utf-8") + b"\0" + hashlib.sha256(payload).digest())
        raw = np.load(io.BytesIO(payload), allow_pickle=True)
        if raw.shape != ():
            raise ValueError(f"Expected scalar dictionary cache: {path}")
        raw = raw.item()
        if not isinstance(raw, dict) or not isinstance(raw.get("mean"), dict):
            raise ValueError(f"Missing mean representation dictionary: {path}")
        for j, layer in enumerate(layers):
            if layer not in raw["mean"]:
                raise ValueError(f"Missing ESM layer {layer}: {path}")
            vector = np.asarray(raw["mean"][layer])
            if vector.shape != (embed_dim,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid layer {layer} shape/values: {path}: {vector.shape}")
            batch[row, j * embed_dim:(j + 1) * embed_dim] = vector
    return batch


def import_source(source_dir):
    path = Path(source_dir) / "model.py"
    spec = importlib.util.spec_from_file_location("latence_v096_source_esm_model", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import model source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ESM2ResNet, path


def load_model(model_type, checkpoint, aspect, trust_checkpoint=False):
    import torch
    config = torch.load(checkpoint, map_location="cpu", weights_only=not trust_checkpoint)
    if not isinstance(config, dict):
        raise ValueError(f"Expected ESM2ResNet.save_config dictionary: {checkpoint}")
    fields = ["aspect", "layer_list", "embed_dim", "go_term_list", "dropout",
              "activation", "seed", "add_res"]
    missing = [key for key in fields + ["state_dict"] if key not in config]
    if missing:
        raise ValueError(f"Missing checkpoint configuration: {missing}")
    aliases = {"BP": "BPO", "CC": "CCO", "MF": "MFO"}
    observed = aliases.get(str(config["aspect"]).upper(), str(config["aspect"]).upper())
    expected = aliases.get(aspect.upper(), aspect.upper())
    if observed != expected:
        raise ValueError(f"Checkpoint aspect {observed!r} != requested {expected!r}")
    # save_config preserves the caller's container types. In real checkpoints,
    # np.load often supplies an ndarray GO vocabulary; ndarray truth testing is
    # ambiguous, and bytes IDs also need decoding for stable JSON provenance.
    config = dict(config)
    hidden_dims = np.asarray(config["layer_list"], dtype=object)
    go_terms = np.asarray(config["go_term_list"], dtype=object)
    if hidden_dims.ndim != 1 or hidden_dims.size == 0:
        raise ValueError("Checkpoint hidden dimensions must be a nonempty 1D sequence")
    if go_terms.ndim != 1 or go_terms.size == 0:
        raise ValueError("Checkpoint GO vocabulary must be a nonempty 1D sequence")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
           or int(x) <= 0 for x in hidden_dims):
        raise ValueError("Checkpoint hidden dimensions must be positive integers")
    normalized_terms = []
    for term in go_terms:
        if isinstance(term, (bytes, np.bytes_)):
            term = term.decode("utf-8")
        elif isinstance(term, (str, np.str_)):
            term = str(term)
        else:
            raise ValueError("Checkpoint GO vocabulary entries must be strings")
        if not term:
            raise ValueError("Checkpoint GO vocabulary contains an empty ID")
        normalized_terms.append(term)
    config["layer_list"] = [int(x) for x in hidden_dims]
    config["go_term_list"] = normalized_terms
    model = model_type(**{key: config[key] for key in fields}, prediction_mode=False)
    model.load_state_dict(config["state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    if len(model.output_layer) != 1 or not isinstance(model.output_layer[0], torch.nn.Linear):
        raise ValueError("Expected output_layer=Sequential(Linear); refusing ambiguous extraction")
    return model


class _Captured(Exception):
    pass


def capture_penultimate(model, inputs, verify=False):
    """Read the actual classifier input without maintaining a second forward.

    Most batches stop at the pre-hook, avoiding allocation of N x GO logits.
    On the first probe, execute the full forward and verify that the captured
    vector exactly reconstructs the model's logits through its final Linear.
    """
    import torch
    captured = []
    linear = model.output_layer[0]

    def hook(_module, args):
        if len(args) != 1 or not torch.is_tensor(args[0]):
            raise ValueError("Unexpected final classifier input signature")
        captured.append(args[0].detach())
        if not verify:
            raise _Captured()

    handle = linear.register_forward_pre_hook(hook)
    try:
        try:
            with torch.inference_mode():
                logits = model(inputs)
        except _Captured:
            logits = None
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError(f"Final classifier was reached {len(captured)} times")
    features = captured[0]
    if features.ndim != 2 or features.shape != (inputs.shape[0], linear.in_features):
        raise ValueError("Invalid penultimate tensor shape")
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite penultimate features")
    max_abs = None
    if verify:
        with torch.inference_mode():
            reconstructed = torch.nn.functional.linear(features, linear.weight, linear.bias)
        if logits is None or not torch.isfinite(logits).all():
            raise ValueError("Invalid verification logits")
        torch.testing.assert_close(logits, reconstructed, atol=2e-5, rtol=2e-5)
        max_abs = float((logits - reconstructed).abs().max().item())
    return features, max_abs


def resolve_checkpoints(args):
    if args.checkpoint:
        paths = [Path(p).resolve() for p in args.checkpoint]
    else:
        paths = sorted(Path(args.model_dir).resolve().glob("*.pt"))
    if not paths:
        raise ValueError("No classifier checkpoints found")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate checkpoints")
    if len(paths) > 1 and args.ensemble != "concat":
        raise ValueError("Multiple checkpoints require --ensemble concat; latent averaging is invalid")
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("A selected checkpoint is missing")
    return paths


def run(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if len(args.repr_layers) != 3 or len(set(args.repr_layers)) != 3:
        raise ValueError("ESM2ResNet requires three distinct ordered representation layers")
    if args.repr_layers != [34, 35, 36] and not args.allow_nonstandard_layers:
        raise ValueError("Expected ESM2-3B layers 34 35 36; use explicit override only with matching training inputs")
    ids_path = args.ids or args.fasta
    ids = load_ids(ids_path, fasta=bool(args.fasta))
    checkpoints = resolve_checkpoints(args)
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {out}")
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly use --device cpu if intended")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_hash = sha256_file(Path(args.esm_source_dir) / "model.py")
    model_type, source_path = import_source(args.esm_source_dir)
    if sha256_file(source_path) != source_hash:
        raise ValueError("ESM model source changed during import")
    provenance = None
    if args.provenance_json:
        provenance = json.loads(Path(args.provenance_json).read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".partial-", dir=out.parent))
    try:
        # Preflight dimensions one checkpoint at a time, without keeping ensembles on GPU.
        segments = []
        offset = 0
        expected_embed_dim = None
        for path in checkpoints:
            checkpoint_hash = sha256_file(path)
            model = load_model(model_type, path, args.aspect, args.trust_checkpoint)
            if sha256_file(path) != checkpoint_hash:
                raise ValueError("Checkpoint changed during preflight")
            dim = int(model.output_layer[0].in_features)
            if expected_embed_dim is None:
                expected_embed_dim = int(model.embed_dim)
            elif model.embed_dim != expected_embed_dim:
                raise ValueError("Ensemble checkpoints have different ESM input dimensions")
            if expected_embed_dim != 2560 and not args.allow_nonstandard_layers:
                raise ValueError("ESM2-3B checkpoints must use embed_dim=2560")
            terms_hash = hashlib.sha256(json.dumps(list(model.go_term_list), separators=(",", ":")).encode()).hexdigest()
            segments.append({"checkpoint": str(path), "checkpoint_sha256": checkpoint_hash,
                             "start": offset, "end": offset + dim, "dim": dim,
                             "seed": int(model.seed), "aspect": str(model.aspect),
                             "go_terms_sha256": terms_hash, "add_res": bool(model.add_res)})
            offset += dim
            del model
        dtype = np.dtype(args.dtype)
        features = np.lib.format.open_memmap(staging / "features.npy", mode="w+",
                                             dtype=dtype, shape=(len(ids), offset))
        input_digest_value = None
        for segment, path in zip(segments, checkpoints):
            input_digest = hashlib.sha256()
            if sha256_file(path) != segment["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during export")
            model = load_model(model_type, path, args.aspect, args.trust_checkpoint).to(args.device)
            for start in range(0, len(ids), args.batch_size):
                stop = min(len(ids), start + args.batch_size)
                batch = load_plm_batch(args.features_dir, ids[start:stop], args.repr_layers,
                                       expected_embed_dim, input_digest)
                tensor = torch.from_numpy(batch).to(args.device)
                if start == 0:
                    # Tiny probe verifies the actual model forward; no GO logits exported.
                    _, err = capture_penultimate(model, tensor[:min(2, len(tensor))], verify=True)
                    segment["linear_reconstruction_max_abs"] = err
                hidden, _ = capture_penultimate(model, tensor)
                array = hidden.float().cpu().numpy()
                if dtype == np.float16 and np.abs(array).max() > np.finfo(np.float16).max:
                    raise ValueError("float16 feature overflow; export with --dtype float32")
                features[start:stop, segment["start"]:segment["end"]] = array.astype(dtype)
                if start == 0 or stop == len(ids) or (start // args.batch_size) % 100 == 0:
                    print(f"{path.name}: {stop}/{len(ids)} proteins", flush=True)
            if sha256_file(path) != segment["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during export")
            current_digest = input_digest.hexdigest()
            if input_digest_value is None:
                input_digest_value = current_digest
            elif current_digest != input_digest_value:
                raise ValueError("PLM cache changed between ensemble exports")
            del model
        if sha256_file(source_path) != source_hash:
            raise ValueError("ESM model source changed during export")
        features.flush()
        del features
        np.save(staging / "protein_ids.npy", ids, allow_pickle=False)
        features_hash = sha256_file(staging / "features.npy")
        ids_hash = sha256_file(staging / "protein_ids.npy")
        manifest = {
            "format": FORMAT, "schema": FORMAT, "complete": True, "created_at": datetime.now(timezone.utc).isoformat(),
            "features_file": "features.npy", "ids_file": "protein_ids.npy",
            "shape": [int(len(ids)), offset], "num_proteins": int(len(ids)), "feature_dim": offset,
            "dtype": args.dtype, "features_sha256": features_hash,
            "ids_sha256": ids_hash,
            "feature_definition": "ESM2ResNet.output_layer[0] input, after concat_layer and optional residual block; no sigmoid, no GO propagation",
            "ensemble": "concat" if len(checkpoints) > 1 else "single", "segments": segments,
            "repr_layers": args.repr_layers, "esm_embed_dim": expected_embed_dim,
            "esm_model": "esm2_t36_3B_UR50D" if expected_embed_dim == 2560 and args.repr_layers == [34, 35, 36] else "explicit_nonstandard",
            "source_model_sha256": source_hash, "source_model_path": str(source_path.resolve()),
            "plm_cache_dirs": [str(Path(folder).resolve()) for folder in args.features_dir],
            "plm_cache_ordered_content_sha256": input_digest_value,
            "input_ids_sha256": sha256_file(ids_path),
            "files": {
                "features": {"path": "features.npy", "sha256": features_hash},
                "protein_ids": {"path": "protein_ids.npy", "sha256": ids_hash},
            },
            "representation_identity": {
                "checkpoints_sha256": [s["checkpoint_sha256"] for s in segments],
                "repr_layers": args.repr_layers,
                "ensemble": "concat" if len(checkpoints) > 1 else "single",
                "feature_layer": "output_layer.0.input",
                "dimensions": [s["dim"] for s in segments],
                "model_source_sha256": source_hash,
            },
            "training_provenance": provenance,
            "training_provenance_status": "supplied_not_independently_verified" if provenance is not None else "unknown_requires_split_and_time_audit",
            "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": args.device,
                        "model_eval": True, "autocast": False, "compute_dtype": "float32", "tf32": False},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, out)
        print(f"Exported {len(ids)} x {offset} {args.dtype} features: {out}")
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--esm-source-dir", required=True, help="Directory containing original esm_models model.py")
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--checkpoint", nargs="+", help="Explicit classifier checkpoint(s), in feature segment order")
    selected.add_argument("--model-dir", help="Aspect checkpoint directory (e.g. MODEL/BPO), sorted *.pt")
    parser.add_argument("--ensemble", choices=["single", "concat"], default="single")
    parser.add_argument("--aspect", choices=["BPO", "CCO", "MFO", "BP", "CC", "MF"], default="BPO")
    parser.add_argument("--features-dir", action="append", required=True, help="Trusted PLM cache directory; repeat for core/weak/test. Conflicting duplicate IDs fail.")
    ids = parser.add_mutually_exclusive_group(required=True)
    ids.add_argument("--ids", help="IDs NPY (no pickle), or text file with one exact ID per line")
    ids.add_argument("--fasta", help="FASTA whose exact first-token IDs define output row order")
    parser.add_argument("--output-dir", required=True, help="New output directory; existing paths are never overwritten")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--repr-layers", nargs=3, type=int, default=[34, 35, 36])
    parser.add_argument("--allow-nonstandard-layers", action="store_true", help="Explicit override for a classifier trained on different layers/dimensions")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow unrestricted torch pickle only for a trusted checkpoint rejected by weights_only")
    parser.add_argument("--provenance-json", help="Optional record of teacher training split/date/label snapshot; supplied claims are not automatically verified")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
