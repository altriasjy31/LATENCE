"""Strict source loading and immutable, complete-only frozen-feature caches.

This module imports PyTorch only inside load_source. Numerical cache validation
and role/macro helpers can be tested without production data or PyTorch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "nbs-shared-cache-v091"
CACHE_FILES = (
    "hidden.f32.npy", "protein_ids.i64.npy", "is_weak.bool.npy",
    "label_indptr.i64.npy", "label_go_idx.i32.npy", "geometry.f32.npy",
    "task_to_ontology.i64.npy", "original_weight.f32.npy", "original_bias.f32.npy",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _finite(array, name):
    if array.dtype.kind != "f":
        return
    for start in range(0, len(array), 4096):
        if not np.isfinite(array[start:start + 4096]).all():
            raise ValueError(f"non-finite cache values: {name}")


def load_cache(cache_dir):
    """Verify all immutable bytes and structural invariants before training."""
    folder = Path(cache_dir).resolve()
    manifest = json.loads((folder / "cache_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise ValueError("a complete nbs-shared-cache-v091 cache is required")
    records = manifest.get("files", {})
    if set(records) != set(CACHE_FILES):
        raise ValueError("cache manifest file set differs from the v091 schema")
    arrays = {}
    for name in CACHE_FILES:
        path = folder / name
        record = records[name]
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"cache file missing or changed: {name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != record.get("shape") or str(value.dtype) != record.get("dtype"):
            raise ValueError(f"cache shape/dtype differs: {name}")
        arrays[name] = value
    validate_cache_arrays(arrays)
    return manifest, arrays


def validate_cache_arrays(arrays):
    h, ids, weak = (arrays[name] for name in CACHE_FILES[:3])
    ptr, go = arrays["label_indptr.i64.npy"], arrays["label_go_idx.i32.npy"]
    geometry, mapping = arrays["geometry.f32.npy"], arrays["task_to_ontology.i64.npy"]
    weight, bias = arrays["original_weight.f32.npy"], arrays["original_bias.f32.npy"]
    if h.ndim != 2 or not len(h) or ids.shape != (len(h),) or weak.shape != (len(h),):
        raise ValueError("invalid hidden/protein/role cache axes")
    if (h.dtype != np.float32 or ids.dtype != np.int64 or weak.dtype != np.bool_
            or ptr.dtype != np.int64 or go.dtype != np.int32):
        raise ValueError("hidden, ID, role or CSR cache dtype differs")
    if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("cache protein IDs must be unique and nonnegative")
    if not weak.any() or weak.all():
        raise ValueError("cache must contain both weak and core training proteins")
    if geometry.ndim != 2 or not len(geometry) or geometry.shape[1] != 1024:
        raise ValueError("v091 source requires task GO geometry of width 1024")
    n_go = len(geometry)
    if mapping.shape != (n_go,) or mapping.dtype != np.int64 or np.any(mapping < 0):
        raise ValueError("invalid task-to-ontology cache mapping")
    if weight.shape != (n_go, h.shape[1]) or bias.shape != (n_go,):
        raise ValueError("source classifier does not match hidden/GO axes")
    if any(arrays[n].dtype != np.float32 for n in
           ("geometry.f32.npy", "original_weight.f32.npy", "original_bias.f32.npy")):
        raise ValueError("geometry and classifier cache dtype must be float32")
    if (ptr.shape != (len(h) + 1,) or go.ndim != 1 or ptr[0] != 0 or ptr[-1] != len(go)
            or np.any(np.diff(ptr) < 0) or np.any(go < 0) or np.any(go >= n_go)):
        raise ValueError("invalid training label CSR rows")
    for name, value in arrays.items():
        _finite(value, name)


def macro_id_groups(weak_ids, core_ids, weak_batch, core_batch, seed):
    """Each training protein appears once; a macro's entire seed set is excluded.

    The final role tails can be uneven. This is one fixed cache view, not a
    replay or continuation of the original DDP epoch stream.
    """
    if any(type(x) is not int or x < 1 for x in (weak_batch, core_batch)):
        raise ValueError("cache macro role batch sizes must be positive integers")
    weak_ids, core_ids = np.asarray(weak_ids, np.int64), np.asarray(core_ids, np.int64)
    if (weak_ids.ndim != 1 or core_ids.ndim != 1 or not len(weak_ids) or not len(core_ids)
            or len(np.unique(np.concatenate((weak_ids, core_ids)))) != len(weak_ids) + len(core_ids)):
        raise ValueError("training role IDs must be nonempty, disjoint and unique")
    rng = np.random.default_rng(seed)
    weak_ids, core_ids = rng.permutation(weak_ids), rng.permutation(core_ids)
    count = max((len(weak_ids) + weak_batch - 1) // weak_batch,
                (len(core_ids) + core_batch - 1) // core_batch)
    for index in range(count):
        w = weak_ids[index * weak_batch:(index + 1) * weak_batch]
        c = core_ids[index * core_batch:(index + 1) * core_batch]
        yield np.concatenate((w, c)), np.r_[np.ones(len(w), bool), np.zeros(len(c), bool)]


def split_cache_macro(ids, weak, query_batch):
    """Keep weak/core proportions while bounding each graph query microbatch."""
    count = max(1, int(np.ceil(len(ids) / query_batch)))
    while True:
        pieces = [np.concatenate((w, c)) for w, c in zip(
            np.array_split(ids[weak], count), np.array_split(ids[~weak], count))]
        if max(map(len, pieces)) <= query_batch:
            return [p for p in pieces if len(p)]
        count += 1


def candidate_statistics():
    return dict(proteins=0, candidates=0, correct=0, positives=0, macro_precision_sum=0.,
                macro_recall_sum=0., positive_proteins=0, correct_count_histogram=[0]*17,
                confidence_counts=[0]*10, confidence_correct=[0]*10)


def update_candidate_statistics(stats, columns, confidence, labels):
    """Raw task-column top16 membership; aliases are not promoted to positives."""
    for go, prob, target in zip(columns, confidence, labels):
        target = set(map(int, target))
        seen, correct = set(), 0
        for col, value in zip(go[:16], prob[:16]):
            col = int(col)
            if col < 0 or value <= 0 or col in seen:
                continue
            seen.add(col)
            hit = int(col in target)
            correct += hit
            bin_id = min(9, max(0, int(float(value)*10)))
            stats["confidence_counts"][bin_id] += 1
            stats["confidence_correct"][bin_id] += hit
        stats["proteins"] += 1
        stats["candidates"] += len(seen)
        stats["correct"] += correct
        stats["positives"] += len(target)
        stats["macro_precision_sum"] += correct / max(1,len(seen))
        if target:
            stats["macro_recall_sum"] += correct / len(target)
            stats["positive_proteins"] += 1
        stats["correct_count_histogram"][correct] += 1


def finish_candidate_statistics(stats):
    result = dict(stats)
    result.update(micro_precision=stats["correct"]/max(1,stats["candidates"]),
                  micro_recall=stats["correct"]/max(1,stats["positives"]),
                  macro_precision=stats["macro_precision_sum"]/max(1,stats["proteins"]),
                  macro_recall=stats["macro_recall_sum"]/max(1,stats["positive_proteins"]),
                  confidence_accuracy=[h/n if n else None for h,n in zip(stats["confidence_correct"],stats["confidence_counts"])])
    result["scope"] = "top16 raw task-column membership; weak measured against pseudo labels, not experimental truth"
    return result


def labels_for_ids(data, ids):
    """Same binary CSR membership as v089, without probabilities or new labels."""
    registry = data.registry
    core_code, weak_code = registry.role_to_code["core"], registry.role_to_code["weak"]
    rows = []
    for protein in np.asarray(ids, np.int64):
        protein = int(protein)
        code = int(registry.role_code[protein])
        if code == core_code:
            store, row = data.stores.gold_messages, protein
            values = store.go_idx[int(store.indptr[row]):int(store.indptr[row + 1])]
        elif code == weak_code:
            store, row = data.stores.pseudo_messages, int(registry.role_row[protein])
            values = store.get_role_row(row)["go_idx"]
        else:
            raise ValueError("cache source ID is not a core or weak training protein")
        go = np.unique(np.asarray(values, np.int64))
        if len(go) and (go[0] < 0 or go[-1] >= data.num_task_go):
            raise ValueError("cache label row is outside immutable task columns")
        rows.append(go.astype(np.int32))
    return rows


def validate_source_header(saved, expected_epoch=60):
    """Reject wrong experiments before building any production data objects."""
    if type(expected_epoch) is not int or expected_epoch != 60:
        raise ValueError("this v091 experiment is fixed to source epoch60")
    if saved.get("runner_version") != "0.8.9.1" or saved.get("epoch") != expected_epoch:
        raise ValueError("source is not the requested v089 completed-epoch checkpoint")
    options = saved.get("model_config", {})
    if options.get("sage_layers") != 3 or options.get("prediction_mode") != "direct":
        raise ValueError("v091 first stage requires the original three-layer direct v089 source")


def load_source(config_path, checkpoint_path, device, sampler="fast", expected_epoch=60):
    """Load the actual frozen v089 source; never treat it as v090 six-layer resume."""
    import torch
    from nbs_pg.local_loader_v089 import build_latence_nbs_stores
    from scripts.nbs import train_nbs_full_task_v089 as runner
    from nbs_pg.full_task_phase_v089 import PhasePlanV089
    if sampler == "fast":
        from nbs_pg.full_task_sampler_fast import FullTaskDataV088Fast as Data
    elif sampler == "original":
        from nbs_pg.full_task_data_v088 import FullTaskDataV088 as Data
    else:
        raise ValueError("sampler must be fast or original")
    device = torch.device(device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    path = Path(checkpoint_path).resolve()
    if not path.is_file() or path.name != f"nbs_epoch{expected_epoch}.pt":
        raise ValueError(f"source must be a fixed nbs_epoch{expected_epoch}.pt checkpoint")
    before = sha256(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if sha256(path) != before:
        raise ValueError("source checkpoint changed while loading")
    validate_source_header(saved, expected_epoch)
    phase = PhasePlanV089(**saved["phase_plan"])
    fields = runner._phase_fields(phase, saved["step"])
    if any(saved.get(key) != value for key, value in fields.items()):
        raise ValueError("source checkpoint epoch/phase counters disagree")
    if fields["cumulative_completed_epochs"] != expected_epoch:
        raise ValueError("source checkpoint is not at the requested completed epoch")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config != saved["config"]:
        raise ValueError("use the source run's unchanged resolved_config.json")
    variant = saved.get("variant", config["full_task"].get("variant", "fixed"))
    expected_implementation = runner.prediction_implementation(variant)
    trained = saved.get("training_implementation", {})
    if any(trained.get(key) != value for key, value in expected_implementation.items()):
        raise ValueError("source prediction implementation differs from the checkpoint")
    options = saved["model_config"]
    torch.manual_seed(config["full_task"]["seed"])
    data = Data(config, stores=build_latence_nbs_stores(config))
    data.ensure_prepared()
    ontology = data.ontology("cpu")
    contract = {"data": data.data_contract(), "ontology": runner.ontology_contract(ontology)}
    if saved.get("contract") != contract:
        raise ValueError("source data, split or ontology contract differs")
    if saved.get("training_baseline_identity") != runner.training_baseline_identity(data):
        raise ValueError("source backbone inputs/clipping differ")
    model, _ = runner.build_model(data.feature_dim, ontology, options, variant, device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    metadata = {
        "source_checkpoint": str(path), "source_checkpoint_sha256": before,
        "source_config_path": str(Path(config_path).resolve()), "source_config": config,
        "source_epoch": int(saved["epoch"]), "source_step": int(saved["step"]),
        "source_model_config": options, "source_implementation": trained,
        "data_contract": contract, "source_world_size": saved["world_size"],
        "strict_pipeline_zero_shot": False,
    }
    return model, data, ontology, metadata
