#!/usr/bin/env python3
"""Read-only NBS diagnostics before the next long run: no training, no checkpoint.

All computations use the frozen Stage-1 representations, the existing training
CSR labels (core gold, weak binary >0.5 pseudo membership) and existing caches.

  shift    B quality, label density and retrieval similarity for a training-core
           sample and the Stage-2 core holdout (both Stage-1-seen) versus the
           independent test set.
  knn      Similarity-weighted label transfer from exact cosine neighbours in
           three pools (training core gold, weak pseudo, core+weak), full k/power
           grid, for all three query sets.
  mix      alpha*B + (1-alpha)*kNN and alpha*G + (1-alpha)*kNN on the test set.
  strata   micro AP / recall by training-core GO frequency bin (test set).
  overlap  exact protein-ID and near-duplicate representation overlap.
  profile  (optional, CPU) cProfile of one real training batch build.

Every grid cell is written; nothing is selected on the test set. kNN and mixtures
are measurements of the signal available to a model, NOT models and NOT
acceptance metrics; the fixed no-fusion acceptance protocol is unchanged.
"""
from __future__ import annotations

import argparse
import cProfile
import csv
import importlib
import importlib.util
import io
import json
import pickle
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "nbs_models/nbs_protein_go"
SCHEMA = "nbs-knn-diagnostics-v1"
POOLS = ("core", "weak", "core_weak")
FREQUENCY_BINS = ((0, 0), (1, 9), (10, 99), (100, 999), (1000, None))
NUM_CLASSES = {"bp": 21312, "mf": 7038, "cc": 2903}
# Release -> (store loader module, data class module, data class).
RELEASES = {
    "0.8.9.1": ("nbs_pg.local_loader_v089", "nbs_pg.full_task_data_v088", "FullTaskDataV088"),
    "0.9.0": ("nbs_pg.local_loader_v090", "nbs_pg.full_task_data_v090", "FullTaskDataV090"),
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Loaded by path: the nbs_pg package __init__ imports torch-dependent modules.
_METRICS = load_module(MODEL_ROOT / "nbs_pg/full_task_metrics_v085.py", "_nbs_diag_v085_metrics")
_LEGACY = load_module(ROOT / "scripts/nbs/eval_nbs_ind_test_predictions.py", "_nbs_diag_legacy_eval")


def log(message):
    print(f"[diagnose {time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------- metrics

def protein_fmax(labels, scores):
    """The unchanged acceptance kernel: 0.001 grid, strict >, percent units."""
    y, p = _METRICS._validate(labels, scores)
    thresholds = _METRICS._thresholds(.001)
    f1, precision, recall, eligible = _METRICS._protein_curve(y, p, thresholds)
    best = int(np.argmax(f1))
    return {"fmax": 100 * float(f1[best]), "threshold": float(thresholds[best]),
            "precision": 100 * float(precision[best]), "recall": 100 * float(recall[best]),
            "eligible_proteins": int(eligible)}


def grouped_ap(labels, scores):
    """Non-interpolated micro AP in percent; equal scores form one group."""
    y = np.asarray(labels).reshape(-1).astype(bool)
    positives = int(y.sum())
    if not positives:
        return None
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    true_positive = np.cumsum(y)[last]
    gained = np.diff(np.r_[0, true_positive])
    return 100 * float(np.sum(gained * true_positive / (last + 1)) / positives)


def normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    if not np.isfinite(rows).all() or np.any(norm <= 0):
        raise ValueError("cosine retrieval requires finite nonzero representations")
    return rows / norm


# ---------------------------------------------------------------- retrieval

def _torch_for(device):
    if device == "numpy":
        return None
    import torch
    return torch


def search_pool(queries, pool_ids, gather, k, *, self_ids=None, chunk=32768, device="numpy"):
    """Exact cosine top-k of every query over pool_ids, similarity descending.

    self_ids[q] (or -1) is removed from that query's list: one extra candidate
    is retrieved so exactly k remain. Returns (similarity[nq,k], global_id[nq,k]).
    """
    pool_ids = np.asarray(pool_ids, dtype=np.int64)
    if len(pool_ids) < k + 1:
        raise ValueError("retrieval pool is smaller than k + 1")
    q = normalize(queries)
    self_ids = np.full(len(q), -1, np.int64) if self_ids is None else np.asarray(self_ids, np.int64)
    take = k + 1
    torch = _torch_for(device)
    query_t = torch.from_numpy(q).to(device) if torch is not None else None
    best_s = np.empty((len(q), 0), np.float32)
    best_i = np.empty((len(q), 0), np.int64)
    for start in range(0, len(pool_ids), chunk):
        ids = pool_ids[start:start + chunk]
        candidates = normalize(gather(ids))
        width = min(take, len(ids))
        if torch is not None:
            scores = query_t @ torch.from_numpy(candidates).to(device).T
            value, index = scores.topk(width, dim=1)
            value, index = value.cpu().numpy(), index.cpu().numpy()
        else:
            scores = q @ candidates.T
            index = np.argpartition(-scores, width - 1, axis=1)[:, :width]
            value = np.take_along_axis(scores, index, axis=1)
        best_s = np.concatenate((best_s, value.astype(np.float32)), axis=1)
        best_i = np.concatenate((best_i, ids[index]), axis=1)
        order = np.argsort(-best_s, axis=1, kind="stable")[:, :take]
        best_s = np.take_along_axis(best_s, order, axis=1)
        best_i = np.take_along_axis(best_i, order, axis=1)
    sims = np.empty((len(q), k), np.float32)
    neighbors = np.empty((len(q), k), np.int64)
    for row in range(len(q)):
        keep = np.flatnonzero(best_i[row] != self_ids[row])[:k]
        sims[row], neighbors[row] = best_s[row, keep], best_i[row, keep]
    return sims, neighbors


def merge_pools(first, second, k):
    """Top-k of two (similarity, id) neighbour lists by similarity."""
    sims = np.concatenate((first[0], second[0]), axis=1)
    ids = np.concatenate((first[1], second[1]), axis=1)
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(sims, order, axis=1), np.take_along_axis(ids, order, axis=1)


def knn_transfer(sims, neighbors, labels_of, ks, power, num_go):
    """{k: [nq,G]} = sum_i w_i y_i / sum_i w_i over the k nearest, w=max(sim,0)**power."""
    ks = sorted(set(int(k) for k in ks))
    if ks[0] < 1 or sims.shape[1] < ks[-1]:
        raise ValueError("k must lie in [1, retrieved neighbours]")
    out = {k: np.zeros((len(sims), num_go), np.float32) for k in ks}
    for row in range(len(sims)):
        accumulated, total = np.zeros(num_go, np.float64), 0.
        for rank in range(ks[-1]):
            weight = max(float(sims[row, rank]), 0.) ** power
            if weight > 0:
                accumulated[labels_of(int(neighbors[row, rank]))] += weight
                total += weight
            if rank + 1 in out and total > 0:
                out[rank + 1][row] = np.minimum(accumulated / total, 1.)
    return out


def similarity_summary(sims):
    top1 = sims[:, 0]
    return {"top1_mean": float(top1.mean()), "top1_median": float(np.median(top1)),
            "top1_p10": float(np.quantile(top1, .1)), "topk_mean": float(sims.mean()),
            "top1_ge_0.99_fraction": float((top1 >= .99).mean()),
            "top1_ge_0.999_fraction": float((top1 >= .999).mean())}


def recoverable(labels, scores):
    """Share of gold pairs that at least one retrieved neighbour carries (score>0)."""
    y = np.asarray(labels, bool)
    hit = y & (np.asarray(scores) > 0)
    per_protein = hit.sum(1) / np.maximum(y.sum(1), 1)
    return {"micro_percent": 100 * float(hit.sum() / max(int(y.sum()), 1)),
            "protein_mean_percent": 100 * float(per_protein[y.any(1)].mean())}


# ---------------------------------------------------------------- strata / mixtures

def frequency_strata(labels, methods, column_counts, bins=FREQUENCY_BINS):
    """methods: name -> (scores, threshold). Per training-core frequency bin."""
    y = np.asarray(labels, bool)
    rows = []
    for low, high in bins:
        columns = (column_counts >= low) & (True if high is None else column_counts <= high)
        sub_y = y[:, columns]
        entry = {"train_core_count_min": low, "train_core_count_max": high,
                 "columns": int(columns.sum()), "test_positive_pairs": int(sub_y.sum()),
                 "columns_with_test_positive": int(sub_y.any(0).sum()), "methods": {}}
        for name, (scores, threshold) in methods.items():
            sub = np.asarray(scores)[:, columns]
            predicted = sub > threshold
            entry["methods"][name] = {
                "micro_ap": grouped_ap(sub_y, sub) if sub_y.any() else None,
                "recall_at_method_fmax_threshold": (100 * float((predicted & sub_y).sum() / sub_y.sum())
                                                    if sub_y.any() else None),
                "predicted_pairs_at_threshold": int(predicted.sum()),
                "columns_predicted_at_threshold": int(predicted.any(0).sum())}
        rows.append(entry)
    return rows


def mixture_grid(labels, first, second, alphas):
    return [{"alpha_first": float(a), **protein_fmax(labels, a * first + (1 - a) * second)}
            for a in alphas]


# ---------------------------------------------------------------- data access

class TrainingSources:
    """Duck-typed view over FullTaskData: labels, features and dense B rows."""

    def __init__(self, data):
        self.data = data
        self.registry, self.stores = data.registry, data.stores
        self.num_go = int(data.num_task_go)
        self.core_code = self.registry.role_to_code["core"]
        self.weak_code = self.registry.role_to_code["weak"]
        self._cache = {}

    def _raw_labels(self, protein):
        code = int(self.registry.role_code[protein])
        if code == self.core_code:
            store, row = self.stores.gold_messages, protein
        elif code == self.weak_code:
            store, row = self.stores.pseudo_messages, int(self.registry.role_row[protein])
        else:
            raise ValueError(f"protein {protein} is neither training core nor weak")
        go = np.unique(np.asarray(store.go_idx[int(store.indptr[row]):int(store.indptr[row + 1])], np.int64))
        if go.size and (go[0] < 0 or go[-1] >= self.num_go):
            raise ValueError("training label outside task columns")
        return go

    def labels_of(self, protein):
        value = self._cache.get(protein)
        if value is None:
            value = self._cache[protein] = self._raw_labels(protein)
        return value

    def dense_labels(self, ids):
        out = np.zeros((len(ids), self.num_go), bool)
        for row, protein in enumerate(ids):
            out[row, self._raw_labels(int(protein))] = True
        return out

    def features(self, ids):
        return self.stores.features.gather(np.asarray(ids, np.int64))

    def backbone(self, ids):
        store = self.stores.episode_sampler.base_logit_store
        out = np.empty((len(ids), self.num_go), np.float32)
        for row, protein in enumerate(ids):
            item = store._slice_for(int(protein))
            out[row] = store.arrays[item.role][int(protein) - item.global_start]
        return np.clip(out, 0., 1.)

    def core_column_counts(self, core_ids):
        counts = np.zeros(self.num_go, np.int64)
        for protein in core_ids:
            counts[self._raw_labels(int(protein))] += 1
        return counts


def load_test_inputs(input_dir, metadata_file, task, representation_sha256=None):
    input_dir = Path(input_dir)
    manifest = json.loads((input_dir / "ind_test_input_manifest.json").read_text())
    if representation_sha256 is not None:
        declared = manifest.get("cache_signature", {}).get("representation_manifest_sha256")
        if declared != representation_sha256:
            raise ValueError("test representations differ in provenance from the training features")
    ids = [line.strip() for line in (input_dir / "protein_ids.txt").read_text(encoding="utf-8").splitlines()
           if line.strip()]
    features = np.load(input_dir / "ind_test_repr.f16.npy", mmap_mode="r")
    backbone = np.load(input_dir / "backbone_ind_test_prob.f16.npy", mmap_mode="r")
    num_go = NUM_CLASSES[task]
    if len(features) != len(ids) or backbone.shape != (len(ids), num_go) or len(set(ids)) != len(ids):
        raise ValueError("test IDs, representations and B probabilities are misaligned")
    with Path(metadata_file).open("rb") as handle:
        block = _LEGACY._task_dict(pickle.load(handle), "ind_test", task)
    metadata_ids = [str(x) for x in block["proteins"]]
    key = next((k for k in ("prop_annotations", "prop_annotation", "annotations", "labels") if k in block), None)
    if key is None or set(metadata_ids) != set(ids):
        raise ValueError("metadata lacks test annotations or has a different protein set")
    labels = _LEGACY._dense_labels(block[key], num_go, len(metadata_ids))
    lookup = {protein: row for row, protein in enumerate(metadata_ids)}
    labels = labels[np.asarray([lookup[protein] for protein in ids], np.int64)]
    return {"ids": ids, "features": np.asarray(features, np.float32),
            "backbone": np.clip(np.asarray(backbone, np.float32), 0., 1.), "labels": labels}


def registry_protein_ids(registry):
    """protein_idx -> protein_id if the registry CSV carries that column."""
    with Path(registry.path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "protein_id" not in (reader.fieldnames or []):
            return None
        return [row["protein_id"] for row in reader]


# ---------------------------------------------------------------- report

def _knn_grid(name, labels, pools, sources, ks, powers, with_ap, keep=None):
    """Evaluate every grid cell; retain only the ``keep`` cell's matrix (memory)."""
    rows, kept = [], {}
    for pool in POOLS:
        sims, neighbors = pools[pool]
        for power in powers:
            for k, scores in knn_transfer(sims, neighbors, sources.labels_of, ks, power, sources.num_go).items():
                row = {"pool": pool, "k": k, "power": power, **protein_fmax(labels, scores)}
                if with_ap:
                    row["micro_ap"] = grouped_ap(labels, scores)
                if k == max(ks) and power == powers[0]:
                    row["recoverable_gold"] = recoverable(labels, scores)
                rows.append(row)
                if (pool, k, power) == keep:
                    kept[keep] = scores
        log(f"{name}: kNN pool={pool} done")
    return rows, kept


def run(data, test, *, ks=(1, 5, 10, 20, 50), powers=(1, 4), train_core_sample=1024, seed=20261024,
        strata_knn=("core_weak", 20, 1), mix_alphas=(0, .25, .5, .75, 1), nbs=None,
        chunk=32768, device="numpy"):
    """Return the complete diagnostics report; inputs are never modified."""
    sources = TrainingSources(data)
    ks, powers = tuple(sorted(set(ks))), tuple(powers)
    if strata_knn[0] not in POOLS or strata_knn[1] not in ks or strata_knn[2] not in powers:
        raise ValueError("strata kNN setting must be one cell of the configured grid")
    rng = np.random.default_rng(seed)
    core_ids = np.asarray(data.core_ids, np.int64)
    sample = np.sort(rng.choice(core_ids, min(train_core_sample, len(core_ids)), replace=False))
    holdout = np.asarray(data.validation_ids, np.int64)
    sets = {"test": dict(features=test["features"], labels=test["labels"], backbone=test["backbone"],
                         self_ids=np.full(len(test["labels"]), -1, np.int64))}
    if len(holdout):
        sets["stage2_core_holdout"] = dict(ids=holdout, self_ids=np.full(len(holdout), -1, np.int64))
    sets["train_core_sample"] = dict(ids=sample, self_ids=sample)
    for name, entry in sets.items():
        if "ids" in entry:
            entry.update(features=sources.features(entry["ids"]), labels=sources.dense_labels(entry["ids"]),
                         backbone=sources.backbone(entry["ids"]))
    names = list(sets)
    queries = np.concatenate([sets[n]["features"] for n in names])
    self_ids = np.concatenate([sets[n]["self_ids"] for n in names])
    bounds = np.cumsum([0] + [len(sets[n]["features"]) for n in names])
    kmax = max(ks)
    retrieved = {}
    for pool, ids in (("core", data.anchor_core_ids), ("weak", data.weak_ids)):
        log(f"exact cosine search over {pool} pool ({len(ids)} proteins), {len(queries)} queries")
        retrieved[pool] = search_pool(queries, ids, sources.features, kmax, self_ids=self_ids,
                                      chunk=chunk, device=device)
    report = {"schema": SCHEMA, "settings": {
        "ks": list(ks), "powers": list(powers), "train_core_sample": int(len(sample)), "seed": seed,
        "pools": {"core": "training anchor core (holdout excluded), gold labels",
                  "weak": "weak with >=1 binary >0.5 pseudo label, pseudo labels",
                  "core_weak": "similarity-merged top-k of both"},
        "knn_score": "sum_i w_i y_i / sum_i w_i, w=max(cos,0)**power",
        "selection": "none; every grid cell is reported; diagnostics only, not acceptance"},
        "sets": {}}
    for index, name in enumerate(names):
        entry = sets[name]
        part = slice(bounds[index], bounds[index + 1])
        pools = {pool: (retrieved[pool][0][part], retrieved[pool][1][part]) for pool in ("core", "weak")}
        pools["core_weak"] = merge_pools(pools["core"], pools["weak"], kmax)
        labels = entry["labels"]
        result = {"num_proteins": int(len(labels)),
                  "gold_positives_per_protein": float(labels.sum(1).mean()),
                  "backbone_B": protein_fmax(labels, entry["backbone"]),
                  "similarity": {pool: similarity_summary(pools[pool][0]) for pool in POOLS}}
        grid, kept = _knn_grid(name, labels, pools, sources, ks, powers, with_ap=name == "test",
                               keep=tuple(strata_knn) if name == "test" else None)
        result["knn"] = grid
        if name == "test":
            result["backbone_B"]["micro_ap"] = grouped_ap(labels, entry["backbone"])
            knn = kept[tuple(strata_knn)]
            knn_fmax = next(r for r in grid if (r["pool"], r["k"], r["power"]) == tuple(strata_knn))
            methods = {"B": (entry["backbone"], result["backbone_B"]["threshold"]),
                       f"kNN_{strata_knn[0]}_k{strata_knn[1]}_p{strata_knn[2]}": (knn, knn_fmax["threshold"])}
            result["mixtures"] = {"B_with_kNN": mixture_grid(labels, entry["backbone"], knn, mix_alphas)}
            result["nbs"] = {}
            for label, path in (nbs or {}).items():
                values = np.clip(np.asarray(np.load(path, mmap_mode="r"), np.float32), 0., 1.)
                if values.shape != labels.shape:
                    raise ValueError(f"NBS prediction {label} has shape {values.shape}, expected {labels.shape}")
                result["nbs"][label] = {**protein_fmax(labels, values), "micro_ap": grouped_ap(labels, values),
                                        "path": str(path)}
                methods["G_" + label] = (values, result["nbs"][label]["threshold"])
                result["mixtures"][f"G_{label}_with_kNN"] = mixture_grid(labels, values, knn, mix_alphas)
            counts = sources.core_column_counts(core_ids)
            result["frequency_strata"] = frequency_strata(labels, methods, counts)
            result["strata_knn_setting"] = list(strata_knn)
            log("test: mixtures and frequency strata done")
        report["sets"][name] = result
    return report


def overlap_report(test_ids, registry, data, test_similarity):
    names = registry_protein_ids(registry)
    result = {"near_duplicate_representations": {pool: {k: v for k, v in summary.items() if "ge_0.99" in k}
                                                 for pool, summary in test_similarity.items()}}
    if names is None:
        result["exact_id_overlap"] = "registry has no protein_id column"
        return result
    lookup = {protein: index for index, protein in enumerate(names)}
    shared = [lookup[p] for p in test_ids if p in lookup]
    holdout = set(map(int, data.validation_ids))
    roles = {}
    for protein in shared:
        role = "stage2_core_holdout" if protein in holdout else registry.code_to_role[int(registry.role_code[protein])]
        roles[role] = roles.get(role, 0) + 1
    result["exact_id_overlap"] = {"test_proteins": len(test_ids), "found_in_registry": len(shared), "by_role": roles}
    return result


def profile_sampler(data, *, weak, core, seed, top=30):
    """cProfile one training batch build exactly as the runner calls it (CPU)."""
    rng = np.random.default_rng(seed)
    ids = np.concatenate((rng.choice(data.weak_ids, weak, replace=False),
                          rng.choice(data.core_ids, core, replace=False)))
    data.set_sampling_context(step=1, rank=0, training=True)
    started = time.perf_counter()
    dense = data.stores.episode_sampler.base_logit_store.gather_matrix(ids, np.arange(data.num_task_go))
    base_seconds = time.perf_counter() - started
    base_bytes = int(dense.nbytes)
    del dense
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    batch = data.batch(ids, "cpu", supervision_seed_ids=ids)
    profiler.disable()
    total = time.perf_counter() - started
    texts = {}
    for key in ("cumulative", "tottime"):
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats(key).print_stats(top)
        texts[key] = stream.getvalue()
    diagnostics = {key: float(value) for key, value in batch.get("sampler_diagnostics", {}).items()}
    return {"weak": int(weak), "core": int(core), "batch_build_seconds": total,
            "dense_base_logit_gather_seconds_alone": base_seconds, "dense_base_logit_bytes": base_bytes,
            "sampler_diagnostics": diagnostics}, texts


def summary_markdown(report):
    lines = ["# NBS kNN / shift diagnostics", "", "Diagnostics only: no selection, no acceptance claim.", ""]
    for name, result in report["sets"].items():
        b = result["backbone_B"]
        lines += [f"## {name} ({result['num_proteins']} proteins, "
                  f"{result['gold_positives_per_protein']:.1f} labels/protein)", "",
                  f"B Fmax {b['fmax']:.3f} (thr {b['threshold']:.3f}, P {b['precision']:.2f}, R {b['recall']:.2f})"
                  + (f", micro AP {b['micro_ap']:.3f}" if b.get("micro_ap") is not None else ""), "",
                  "| pool | top1 cos mean | top1 p10 | top-k mean | top1>=0.99 |", "|---|---|---|---|---|"]
        for pool, s in result["similarity"].items():
            lines.append(f"| {pool} | {s['top1_mean']:.4f} | {s['top1_p10']:.4f} | {s['topk_mean']:.4f} | "
                         f"{s['top1_ge_0.99_fraction']:.3f} |")
        lines += ["", "| pool | k | power | Fmax | thr | P | R | AP |", "|---|---|---|---|---|---|---|---|"]
        for r in result["knn"]:
            ap = "" if r.get("micro_ap") is None else f"{r['micro_ap']:.2f}"
            lines.append(f"| {r['pool']} | {r['k']} | {r['power']} | {r['fmax']:.3f} | {r['threshold']:.3f} | "
                         f"{r['precision']:.2f} | {r['recall']:.2f} | {ap} |")
        for r in result["knn"]:
            if "recoverable_gold" in r:
                g = r["recoverable_gold"]
                lines.append(f"- recoverable gold (pool {r['pool']}, k={r['k']}): micro {g['micro_percent']:.2f}%, "
                             f"protein-mean {g['protein_mean_percent']:.2f}%")
        for label, value in result.get("nbs", {}).items():
            lines.append(f"- G {label}: Fmax {value['fmax']:.3f} (thr {value['threshold']:.3f}), AP {value['micro_ap']:.3f}")
        for label, grid in result.get("mixtures", {}).items():
            lines.append(f"- mixture {label}: " + ", ".join(f"a={m['alpha_first']:g}:{m['fmax']:.3f}" for m in grid))
        if "frequency_strata" in result:
            lines += ["", "| train-core count | columns | test pos | method | AP | recall@thr | cols predicted |",
                      "|---|---|---|---|---|---|---|"]
            for row in result["frequency_strata"]:
                span = f"{row['train_core_count_min']}-{row['train_core_count_max'] or 'inf'}"
                for method, m in row["methods"].items():
                    ap = "" if m["micro_ap"] is None else f"{m['micro_ap']:.2f}"
                    rec = "" if m["recall_at_method_fmax_threshold"] is None else f"{m['recall_at_method_fmax_threshold']:.2f}"
                    lines.append(f"| {span} | {row['columns']} | {row['test_positive_pairs']} | {method} | {ap} | "
                                 f"{rec} | {m['columns_predicted_at_threshold']} |")
        lines.append("")
    if "overlap" in report:
        lines += ["## overlap", "", "```", json.dumps(report["overlap"], indent=2), "```", ""]
    if "profile" in report:
        p = report["profile"]
        lines += ["## sampler profile", "", f"- batch build {p['batch_build_seconds']:.1f} s for "
                  f"{p['weak']} weak + {p['core']} core; dense base_logits gather alone "
                  f"{p['dense_base_logit_gather_seconds_alone']:.1f} s ({p['dense_base_logit_bytes'] / 1e9:.2f} GB)", ""]
    return "\n".join(lines)


def prepare_output_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _int_list(text):
    return tuple(int(x) for x in text.split(",") if x.strip())


def _float_list(text):
    return tuple(float(x) for x in text.split(",") if x.strip())


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="resolved_config.json of an existing run")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--metadata-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task", default="bp", choices=sorted(NUM_CLASSES))
    p.add_argument("--device", default="cuda:0", help="torch device, or 'numpy' for a pure CPU path")
    p.add_argument("--nbs-prob", action="append", default=[], metavar="LABEL=PATH",
                   help="existing exported NBS test probabilities (input row order); repeatable")
    p.add_argument("--ks", type=_int_list, default=(1, 5, 10, 20, 50))
    p.add_argument("--powers", type=_int_list, default=(1, 4))
    p.add_argument("--train-core-sample", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--strata-knn", default="core_weak:20:1", help="pool:k:power, one grid cell")
    p.add_argument("--mix-alphas", type=_float_list, default=(0, .25, .5, .75, 1))
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--profile-sampler", action="store_true", help="also cProfile one CPU training batch build")
    p.add_argument("--profile-weak", type=int, default=5120)
    p.add_argument("--profile-core", type=int, default=1280)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    out = prepare_output_dir(args.output_dir)
    sys.path[:0] = [str(ROOT), str(MODEL_ROOT)]
    config = json.loads(args.config.read_text())
    release = config.get("release_version")
    if release not in RELEASES:
        raise ValueError(f"unsupported release_version {release!r}; expected one of {sorted(RELEASES)}")
    loader, data_module, data_class = RELEASES[release]
    log(f"building stores with {loader} (release {release})")
    stores = importlib.import_module(loader).build_latence_nbs_stores(config)
    data = getattr(importlib.import_module(data_module), data_class)(config, stores=stores)
    nbs = {}
    for item in args.nbs_prob:
        label, _, path = item.partition("=")
        if not label or not path:
            raise ValueError("--nbs-prob expects LABEL=PATH")
        nbs[label] = Path(path)
    pool, k, power = args.strata_knn.split(":")
    test = load_test_inputs(args.input_dir, args.metadata_file, args.task,
                            data._signature["representation_manifest_sha256"])
    started = time.perf_counter()
    report = run(data, test, ks=args.ks, powers=args.powers, train_core_sample=args.train_core_sample,
                 seed=args.seed, strata_knn=(pool, int(k), int(power)), mix_alphas=args.mix_alphas,
                 nbs=nbs, chunk=args.chunk, device=args.device)
    report["overlap"] = overlap_report(test["ids"], data.registry, data, report["sets"]["test"]["similarity"])
    report["provenance"] = {"config": str(args.config.resolve()), "release_version": release,
                            "input_dir": str(args.input_dir.resolve()),
                            "metadata_file": str(args.metadata_file.resolve()),
                            "seconds": time.perf_counter() - started}
    if args.profile_sampler:
        log("profiling one training batch build (CPU, read-only caches)")
        data.ensure_prepared()
        report["profile"], texts = profile_sampler(data, weak=args.profile_weak, core=args.profile_core,
                                                   seed=args.seed)
        for key, text in texts.items():
            (out / f"sampler_profile_{key}.txt").write_text(text, encoding="utf-8")
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    log(f"done: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
