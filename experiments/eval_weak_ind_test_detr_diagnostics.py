#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
experiments/eval_weak_ind_test_detr_diagnostics.py


Diagnostic evaluator for weak_exp_train_detr.py / WeakMSAGOWithDETRDecoder.


Main goals
----------
1. Compare backbone-only, external-prob-only, query-refined, and
   external/query ensemble predictions.
2. Diagnose whether saturated delta values are useful calibration or a
   degenerate global suppressor.
3. Run eval-time delta-scale ablations without retraining:
       logits(scale) = base_logits + scale * scatter_add(delta, topk_idx)
4. Report delta statistics conditioned on true labels, external probabilities,
   and base probabilities.
5. Optionally report rare/common GO term metrics using training-label frequency.


This script is intentionally independent from the old query V1 eval scripts.
It instantiates the V3 DETR decoder architecture, loads a checkpoint payload
saved by save_weak_model(), and evaluates mode='ind_test' by default.
"""


from __future__ import annotations


import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple


import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
MSA_ROOT = ROOT / "msa_models"


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MSA_ROOT) not in sys.path:
    sys.path.insert(0, str(MSA_ROOT))


from experiments.weak_exp_train_detr import (  # noqa: E402
    WeakMSAGOWithDETRDecoder,
    build_weak_opt_from_config,
)
from utils.util_functions import (
    make_frequency_bins, 
    compute_simulated_ic,
    compute_ic_from_pred_map,
    resolve_pred_map_source_key,
    compute_simulated_counts_from_scores,
    compute_simulated_bins_from_pred_map,
    sanitize_key_for_filename,
)

from experiments.msaprob import PseudoProbDataset  # noqa: E402
from experiments.exp_train import (  # noqa: E402
    normalize_task,
    load_pickle,
    build_msa_dataset,
    make_loader,
    unpack_batch,
    unpack_pseudo_batch,
    set_model_proteins,
    set_seed,
    clean_optional_path,
)




# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------




def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse bool: {v}")




def parse_csv_floats(s: str) -> List[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x) for x in str(s).replace(";", ",").split(",") if x.strip() != ""]




def parse_csv_strings(s: str) -> List[str]:
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).replace(";", ",").split(",") if x.strip() != ""]


#=============revise probability for adding in query decoder=========


def get_prob_batch(
    prob_np,
    offset: int,
    batch_size: int,
    device: torch.device,
    row_indices: Optional[np.ndarray] = None,
):
    if prob_np is None:
        return None
    if row_indices is None:
        rows = slice(offset, offset + batch_size)
    else:
        rows = row_indices[offset: offset + batch_size]
    arr = np.asarray(prob_np[rows], dtype=np.float32)
    return torch.from_numpy(arr).to(device=device, non_blocking=True)




def make_decoder_prob_variants(
    *,
    expert_prob: torch.Tensor | None,
    original_prob_file: torch.Tensor | None,
    base_logits: torch.Tensor,
    sources: list[str],
    mix_alphas: list[float],
):
    """
    Returns list of (prob_name, prob_tensor).


    `prob_tensor` replaces the current expert/external probability input to the
    DETR query decoder.  With topk_source='external_topk' and blend alpha=1.0,
    the decoder top-k proposal is exactly based on this probability tensor.
    """
    out = []
    base_prob = torch.sigmoid(base_logits.float()).detach()


    for src in sources:
        src = src.strip()


        if src == "expert":
            if expert_prob is not None:
                out.append(("expert", expert_prob.float().clamp(0.0, 1.0)))


        elif src == "base":
            out.append(("baseprob", base_prob))


        elif src == "original":
            if original_prob_file is not None:
                out.append(("origprob", original_prob_file.float().clamp(0.0, 1.0)))


        elif src == "mix_expert_base":
            if expert_prob is None:
                continue
            expert = expert_prob.float().clamp(0.0, 1.0)
            for a in mix_alphas:
                a = float(a)
                mixed = a * expert + (1.0 - a) * base_prob
                out.append((f"mix_exp_base_a{a:.3g}", mixed.clamp(0.0, 1.0)))


        elif src == "mix_expert_original":
            if expert_prob is None or original_prob_file is None:
                continue
            expert = expert_prob.float().clamp(0.0, 1.0)
            orig = original_prob_file.float().clamp(0.0, 1.0)
            for a in mix_alphas:
                a = float(a)
                mixed = a * expert + (1.0 - a) * orig
                out.append((f"mix_exp_orig_a{a:.3g}", mixed.clamp(0.0, 1.0)))


        else:
            raise ValueError(f"Unknown decoder_prob_source: {src}")


    return out
#======================================================


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")




def sigmoid_np(x: np.ndarray) -> np.ndarray:
    # Stable enough for logits in normal model ranges.
    return 1.0 / (1.0 + np.exp(-x))




def prob_to_logit_torch(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = prob.float().clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)




def apply_delta_scale(qout: Dict[str, torch.Tensor], scale: float) -> torch.Tensor:
    """Reconstruct refined logits using an eval-time delta scale."""
    base = qout["base_logits"]
    topk_idx = qout.get("topk_idx", None)
    delta = qout.get("delta", None)
    if topk_idx is None or delta is None or float(scale) == 0.0:
        return base
    out = base.clone()
    out.scatter_add_(dim=1, index=topk_idx, src=float(scale) * delta)
    return out

def apply_delta_scale_on_anchor_prob(
    qout: Dict[str, torch.Tensor],
    anchor_prob: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Eval-time ablation:

        anchor_logits = logit(anchor_prob)
        final_logits  = anchor_logits + scale * scatter(delta, topk_idx)

    This differs from apply_delta_scale(), which uses qout["base_logits"] as base.
    Here the base is a strong expert/base mixed probability anchor.

    Cases:
        scale = 0:
            final_logits = logit(anchor_prob)
        scale = 1:
            final_logits = logit(anchor_prob) + learned query delta
    """
    anchor_prob = anchor_prob.float().clamp(float(eps), 1.0 - float(eps))
    anchor_logits = torch.log(anchor_prob) - torch.log1p(-anchor_prob)

    topk_idx = qout.get("topk_idx", None)
    delta = qout.get("delta", None)

    if topk_idx is None or delta is None or float(scale) == 0.0:
        return anchor_logits

    out = anchor_logits.clone()
    out.scatter_add_(
        dim=1,
        index=topk_idx,
        src=float(scale) * delta.float(),
    )
    return out



def get_cli_provided_keys(argv: List[str]) -> set[str]:
    keys = set()
    for tok in argv:
        if not tok.startswith("--"):
            continue
        key = tok[2:].split("=", 1)[0].replace("-", "_")
        keys.add(key)
    return keys




def load_json_if_exists(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_model_args(checkpoint: str | Path) -> Tuple[Dict[str, Any], bool]:
    """Return embedded construction args and whether this is a full DETR checkpoint."""
    try:
        try:
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(str(checkpoint), map_location="cpu")
    except Exception:
        return {}, False

    if not isinstance(payload, dict):
        return {}, False

    model_args = payload.get("model_args", {})
    if not isinstance(model_args, dict):
        model_args = {}

    keys = [str(k) for k in payload.keys()]
    is_full_detr = (
        "query_decoder" in payload
        or any(k.startswith("query_decoder.") for k in keys)
        or any(k.startswith("module.query_decoder.") for k in keys)
    )
    return model_args, is_full_detr




def auto_train_args_path(checkpoint: str | Path) -> Optional[Path]:
    p = Path(checkpoint)
    cands = [p.parent / "args.json", p.parent.parent / "args.json"]
    for c in cands:
        if c.is_file():
            return c
    return None




def overlay_train_args(args: argparse.Namespace, cli_keys: set[str]) -> argparse.Namespace:
    """Load args.json from the training run and fill eval args not set on CLI."""
    train_args_path: Optional[Path]
    if str(args.train_args_json).lower() == "auto":
        train_args_path = auto_train_args_path(args.checkpoint)
    else:
        train_args_path = Path(args.train_args_json) if args.train_args_json else None


    train_args = load_json_if_exists(train_args_path)
    if train_args_path is not None and train_args_path.is_file():
        print(f"[Train args] loaded {train_args_path}")
        args.resolved_train_args_source = str(train_args_path)
    else:
        checkpoint_args, is_full_detr = load_checkpoint_model_args(args.checkpoint)
        if checkpoint_args:
            train_args = checkpoint_args
            args.resolved_train_args_source = "checkpoint:model_args"
            print("[Train args] loaded from checkpoint model_args")
        elif is_full_detr and not bool(getattr(args, "allow_missing_train_args", False)):
            raise RuntimeError(
                "Full DETR checkpoint found, but neither args.json nor embedded model_args "
                "is available. Refusing to evaluate with silent parser defaults because "
                "delta_max/anchor mode/expert alpha and selector settings may change model "
                "semantics. Restore the training args.json, or explicitly use "
                "--allow_missing_train_args only for a deliberate diagnostic run."
            )
        else:
            args.resolved_train_args_source = None
            print("[Train args] not found; using CLI/default eval arguments")


    # Do not override eval-specific keys or keys explicitly passed on CLI.
    never_overlay = {
        "checkpoint", "external_prob_path", "original_prob_path", "output_dir",
        "mode", "eval_batch_size",
        "query_modes", "delta_scales",
        "ensemble_alphas", "ensemble_query_modes", "ensemble_delta_scales",
        "decoder_prob_sources", "decoder_prob_mix_alphas",
        "skip_legacy_query_sources",
        "auprc_mode", "threshold_step", "compute_sample_fmax",
        "do_rare_analysis", "rare_metric_keys", "max_batches", "train_args_json",
        "allow_missing_train_args", "resolved_train_args_source",
    }


    for k, v in train_args.items():
        if not hasattr(args, k):
            continue
        if k in cli_keys or k in never_overlay:
            continue
        setattr(args, k, v)


    return args




def infer_num_classes_from_checkpoint(checkpoint: str | Path) -> Optional[int]:
    try:
        payload = torch.load(str(checkpoint), map_location="cpu")
    except Exception:
        return None
    if isinstance(payload, dict):
        if "num_classes" in payload:
            return int(payload["num_classes"])
        if "query_decoder_classifier_weight_shape" in payload:
            return int(payload["query_decoder_classifier_weight_shape"][0])
    return None




# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------




def load_detr_checkpoint(model: WeakMSAGOWithDETRDecoder, checkpoint: str | Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = Path(checkpoint)
    payload = torch.load(str(checkpoint), map_location=device)
    info: Dict[str, Any] = {"checkpoint": str(checkpoint)}


    if isinstance(payload, dict) and "backbone" in payload and "query_decoder" in payload:
        missing_b, unexpected_b = model.backbone.load_state_dict(payload["backbone"], strict=False)
        missing_q, unexpected_q = model.query_decoder.load_state_dict(payload["query_decoder"], strict=False)
        info.update({
            "checkpoint_type": payload.get("checkpoint_type", "weak_detr_decoder_payload"),
            "missing_backbone": list(missing_b),
            "unexpected_backbone": list(unexpected_b),
            "missing_query_decoder": list(missing_q),
            "unexpected_query_decoder": list(unexpected_q),
            "detr_version": payload.get("detr_version", None),
        })
        return info


    # Full model state dict.
    if isinstance(payload, dict) and any(str(k).startswith("backbone.") or str(k).startswith("query_decoder.") for k in payload.keys()):
        missing, unexpected = model.load_state_dict(payload, strict=False)
        info.update({
            "checkpoint_type": "full_model_state_dict",
            "missing": list(missing),
            "unexpected": list(unexpected),
        })
        return info


    # Backbone-only checkpoint. Query decoder remains randomly initialized.
    if isinstance(payload, dict):
        missing_b, unexpected_b = model.backbone.load_state_dict(payload, strict=False)
        info.update({
            "checkpoint_type": "backbone_only_state_dict",
            "missing_backbone": list(missing_b),
            "unexpected_backbone": list(unexpected_b),
            "warning": "query_decoder was not loaded; only backbone diagnostics are meaningful",
        })
        return info


    raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload)}")




# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------




def fmax_from_histograms(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_step: float = 0.01,
    compute_sample_fmax: bool = False,
) -> Dict[str, float]:
    """
    Fast threshold-grid Fmax using probability histograms.


    Returns micro-Fmax always. Optionally returns sample/protein-centric Fmax.
    The sample-Fmax is useful for CAFA-style diagnosis, but may not exactly
    match legacy project code. Use the old official eval for final reporting.
    """
    y = y_true.astype(bool, copy=False)
    p = np.asarray(y_prob, dtype=np.float32)
    step = float(threshold_step)
    nbins = int(round(1.0 / step)) + 1
    thresholds = np.arange(nbins, dtype=np.float32) * step
    thresholds[-1] = 1.0


    idx = np.floor(p / step).astype(np.int32)
    idx = np.clip(idx, 0, nbins - 1)


    flat_idx = idx.ravel()
    flat_y = y.ravel()
    pred_hist = np.bincount(flat_idx, minlength=nbins).astype(np.float64)
    tp_hist = np.bincount(flat_idx[flat_y], minlength=nbins).astype(np.float64)


    pred_cum = np.cumsum(pred_hist[::-1])[::-1]
    tp_cum = np.cumsum(tp_hist[::-1])[::-1]
    total_pos = float(flat_y.sum())
    fp_cum = pred_cum - tp_cum
    fn_cum = total_pos - tp_cum


    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
    recall = tp_cum / max(total_pos, 1.0)
    f = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)


    best = int(np.nanargmax(f))
    out = {
        "fmax_micro": float(f[best]),
        "threshold_micro": float(thresholds[best]),
        "precision_micro": float(precision[best]),
        "recall_micro": float(recall[best]),
    }


    # Histogram approximate AP. Good enough for diagnostics; exact AP is optional.
    order = np.arange(nbins - 1, -1, -1)
    rec_desc = recall[order]
    prec_desc = precision[order]
    rec_prev = 0.0
    ap = 0.0
    for r, pr in zip(rec_desc, prec_desc):
        if r > rec_prev:
            ap += (r - rec_prev) * pr
            rec_prev = r
    out["auprc_micro_hist"] = float(ap)


    if compute_sample_fmax:
        n = y.shape[0]
        sum_prec = np.zeros(nbins, dtype=np.float64)
        sum_rec = np.zeros(nbins, dtype=np.float64)
        valid_prec_count = np.zeros(nbins, dtype=np.float64)
        valid_rec_count = 0.0


        for i in range(n):
            yi = y[i]
            idxi = idx[i]
            pred_hist_i = np.bincount(idxi, minlength=nbins).astype(np.float64)
            tp_hist_i = np.bincount(idxi[yi], minlength=nbins).astype(np.float64)
            pred_cum_i = np.cumsum(pred_hist_i[::-1])[::-1]
            tp_cum_i = np.cumsum(tp_hist_i[::-1])[::-1]
            total_pos_i = float(yi.sum())


            pred_nonzero = pred_cum_i > 0
            prec_i = np.zeros(nbins, dtype=np.float64)
            prec_i[pred_nonzero] = tp_cum_i[pred_nonzero] / pred_cum_i[pred_nonzero]
            sum_prec += prec_i
            valid_prec_count += pred_nonzero.astype(np.float64)


            if total_pos_i > 0:
                sum_rec += tp_cum_i / total_pos_i
                valid_rec_count += 1.0


        avg_prec = sum_prec / np.maximum(valid_prec_count, 1.0)
        avg_rec = sum_rec / max(valid_rec_count, 1.0)
        fs = 2.0 * avg_prec * avg_rec / np.maximum(avg_prec + avg_rec, 1e-12)
        best_s = int(np.nanargmax(fs))
        out.update({
            "fmax_sample": float(fs[best_s]),
            "threshold_sample": float(thresholds[best_s]),
            "precision_sample": float(avg_prec[best_s]),
            "recall_sample": float(avg_rec[best_s]),
        })


    return out




def exact_micro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return None
    y = y_true.astype(np.int8, copy=False).ravel()
    p = np.asarray(y_prob, dtype=np.float32).ravel()
    if y.sum() == 0:
        return None
    return float(average_precision_score(y, p))




def filter_metric_arrays(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    no_empty_labels: bool = False,
    no_zero_classes: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    y = y_true
    p = y_prob
    if no_empty_labels:
        row_keep = y.sum(axis=1) > 0
        y = y[row_keep]
        p = p[row_keep]
    if no_zero_classes:
        col_keep = y.sum(axis=0) > 0
        y = y[:, col_keep]
        p = p[:, col_keep]
    return y, p




def compute_metric_pack(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_step: float,
    auprc_mode: str,
    compute_sample_fmax: bool,
    no_empty_labels: bool,
    no_zero_classes: bool,
    metrics_are_percent: bool = True,
) -> Dict[str, float]:
    y, p = filter_metric_arrays(y_true, y_prob, no_empty_labels, no_zero_classes)
    out = fmax_from_histograms(y, p, threshold_step=threshold_step, compute_sample_fmax=compute_sample_fmax)


    if auprc_mode == "exact":
        ap = exact_micro_auprc(y, p)
        if ap is not None:
            out["auprc_micro_exact"] = ap
    elif auprc_mode == "hist":
        pass
    elif auprc_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown auprc_mode: {auprc_mode}")


    if metrics_are_percent:
        for k in list(out.keys()):
            if k.startswith("fmax") or k.startswith("precision") or k.startswith("recall") or k.startswith("auprc"):
                out[k] = 100.0 * float(out[k])
    return out




# -----------------------------------------------------------------------------
# Delta diagnostics
# -----------------------------------------------------------------------------




class WeightedStat:
    def __init__(self) -> None:
        self.sum = 0.0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.count = 0.0
        self.min = float("inf")
        self.max = float("-inf")


    def update(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return
        self.sum += float(v.sum())
        self.abs_sum += float(np.abs(v).sum())
        self.sq_sum += float(np.square(v).sum())
        self.count += float(v.size)
        self.min = min(self.min, float(v.min()))
        self.max = max(self.max, float(v.max()))


    def to_dict(self, prefix: str) -> Dict[str, float]:
        if self.count <= 0:
            return {
                f"{prefix}_count": 0.0,
                f"{prefix}_mean": float("nan"),
                f"{prefix}_abs_mean": float("nan"),
                f"{prefix}_std": float("nan"),
                f"{prefix}_min": float("nan"),
                f"{prefix}_max": float("nan"),
            }
        mean = self.sum / self.count
        var = max(self.sq_sum / self.count - mean * mean, 0.0)
        return {
            f"{prefix}_count": float(self.count),
            f"{prefix}_mean": float(mean),
            f"{prefix}_abs_mean": float(self.abs_sum / self.count),
            f"{prefix}_std": float(math.sqrt(var)),
            f"{prefix}_min": float(self.min),
            f"{prefix}_max": float(self.max),
        }




class DeltaDiagnostics:
    def __init__(self, delta_max: float = 0.5) -> None:
        self.delta_max = float(delta_max)
        self.all_delta_values: List[np.ndarray] = []
        self.all_delta = WeightedStat()
        self.true_pos = WeightedStat()
        self.true_neg = WeightedStat()
        self.ext_high = WeightedStat()
        self.ext_mid = WeightedStat()
        self.ext_low = WeightedStat()
        self.base_high = WeightedStat()
        self.base_mid = WeightedStat()
        self.base_low = WeightedStat()
        self.shift_all = WeightedStat()
        self.shift_true_pos = WeightedStat()
        self.shift_true_neg = WeightedStat()
        self.delta_ext_pairs: List[np.ndarray] = []
        self.delta_base_pairs: List[np.ndarray] = []
        self.delta_true_pairs: List[np.ndarray] = []
        self.total_true_pos = 0.0
        self.topk_true_hits = 0.0
        self.prefilter_true_hits = 0.0
        self.total_ext_high = 0.0
        self.topk_ext_high_hits = 0.0
        self.prefilter_ext_high_hits = 0.0
        self.total_elements = 0.0
        self.neg_count = 0.0
        self.pos_count = 0.0
        self.sat045_count = 0.0
        self.sat049_count = 0.0
        self.sat095max_count = 0.0
        self.batches = 0

        self.all_gate_values: List[np.ndarray] = []
        self.gate_all = WeightedStat()


    def update(
        self,
        qout: Dict[str, torch.Tensor],
        y_true: torch.Tensor,
        base_logits: torch.Tensor,
        external_prob: Optional[torch.Tensor] = None,
        external_high_threshold: float = 0.5,
    ) -> None:
        if qout.get("delta", None) is None or qout.get("topk_idx", None) is None:
            return

        gate_t = qout.get("delta_gate", None)
        if gate_t is not None:
            gate_np = gate_t.detach().float().cpu().numpy()
            self.all_gate_values.append(gate_np.reshape(-1).copy())
            self.gate_all.update(gate_np.reshape(-1))


        delta_t = qout["delta"].detach().float()
        topk_idx = qout["topk_idx"].detach()
        prefilter_idx = qout.get("prefilter_idx", None)
        if prefilter_idx is not None:
            prefilter_idx = prefilter_idx.detach()


        y_t = y_true.detach().float()
        base_prob_t = torch.sigmoid(base_logits.detach().float())
        base_sel_t = torch.gather(base_prob_t, dim=1, index=topk_idx)
        label_sel_t = torch.gather(y_t, dim=1, index=topk_idx)
        base_logit_sel_t = torch.gather(base_logits.detach().float(), dim=1, index=topk_idx)
        prob_shift_t = torch.sigmoid(base_logit_sel_t + delta_t) - torch.sigmoid(base_logit_sel_t)


        d = delta_t.cpu().numpy().astype(np.float32)
        ysel = label_sel_t.cpu().numpy() > 0.5
        bsel = base_sel_t.cpu().numpy().astype(np.float32)
        shift = prob_shift_t.cpu().numpy().astype(np.float32)


        self.all_delta_values.append(d.reshape(-1).copy())
        self.all_delta.update(d)
        self.shift_all.update(shift)
        self.true_pos.update(d[ysel])
        self.true_neg.update(d[~ysel])
        self.shift_true_pos.update(shift[ysel])
        self.shift_true_neg.update(shift[~ysel])


        self.base_high.update(d[bsel >= 0.5])
        self.base_mid.update(d[(bsel >= 0.1) & (bsel < 0.5)])
        self.base_low.update(d[bsel < 0.1])


        self.total_elements += float(d.size)
        self.neg_count += float((d < 0).sum())
        self.pos_count += float((d > 0).sum())
        self.sat045_count += float((np.abs(d) >= 0.45).sum())
        self.sat049_count += float((np.abs(d) >= 0.49).sum())
        self.sat095max_count += float((np.abs(d) >= 0.95 * self.delta_max).sum())


        # Coverage with true labels.
        true_pos_mask = y_t > 0.5
        self.total_true_pos += float(true_pos_mask.sum().item())
        self.topk_true_hits += float(torch.gather(true_pos_mask, dim=1, index=topk_idx).float().sum().item())
        if prefilter_idx is not None:
            self.prefilter_true_hits += float(torch.gather(true_pos_mask, dim=1, index=prefilter_idx).float().sum().item())


        # Correlations. Store only selected top-k arrays, which are small.
        self.delta_base_pairs.append(np.stack([d.reshape(-1), bsel.reshape(-1)], axis=1))
        self.delta_true_pairs.append(np.stack([d.reshape(-1), ysel.astype(np.float32).reshape(-1)], axis=1))


        if external_prob is not None:
            ext_t = external_prob.detach().float().clamp(0.0, 1.0)
            ext_sel_t = torch.gather(ext_t, dim=1, index=topk_idx)
            ext_sel = ext_sel_t.cpu().numpy().astype(np.float32)
            self.ext_high.update(d[ext_sel >= 0.5])
            self.ext_mid.update(d[(ext_sel >= 0.1) & (ext_sel < 0.5)])
            self.ext_low.update(d[ext_sel < 0.1])
            self.delta_ext_pairs.append(np.stack([d.reshape(-1), ext_sel.reshape(-1)], axis=1))


            ext_high_mask = ext_t >= float(external_high_threshold)
            self.total_ext_high += float(ext_high_mask.sum().item())
            self.topk_ext_high_hits += float(torch.gather(ext_high_mask, dim=1, index=topk_idx).float().sum().item())
            if prefilter_idx is not None:
                self.prefilter_ext_high_hits += float(torch.gather(ext_high_mask, dim=1, index=prefilter_idx).float().sum().item())


        self.batches += 1


    @staticmethod
    def _pearson_from_pairs(pairs: List[np.ndarray]) -> float:
        if not pairs:
            return float("nan")
        arr = np.concatenate(pairs, axis=0).astype(np.float64)
        if arr.shape[0] < 2:
            return float("nan")
        x = arr[:, 0]
        y = arr[:, 1]
        sx = x.std()
        sy = y.std()
        if sx <= 1e-12 or sy <= 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])


    def to_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out.update(self.all_delta.to_dict("delta_all"))
        out.update(self.true_pos.to_dict("delta_true_pos"))
        out.update(self.true_neg.to_dict("delta_true_neg"))
        out.update(self.ext_high.to_dict("delta_ext_ge_0p5"))
        out.update(self.ext_mid.to_dict("delta_ext_0p1_0p5"))
        out.update(self.ext_low.to_dict("delta_ext_lt_0p1"))
        out.update(self.base_high.to_dict("delta_base_ge_0p5"))
        out.update(self.base_mid.to_dict("delta_base_0p1_0p5"))
        out.update(self.base_low.to_dict("delta_base_lt_0p1"))
        out.update(self.shift_all.to_dict("prob_shift_all"))
        out.update(self.shift_true_pos.to_dict("prob_shift_true_pos"))
        out.update(self.shift_true_neg.to_dict("prob_shift_true_neg"))


        if self.all_delta_values:
            d = np.concatenate(self.all_delta_values, axis=0).astype(np.float32)
            out.update({
                "delta_abs_p50": float(np.quantile(np.abs(d), 0.50)),
                "delta_abs_p90": float(np.quantile(np.abs(d), 0.90)),
                "delta_abs_p95": float(np.quantile(np.abs(d), 0.95)),
                "delta_abs_p99": float(np.quantile(np.abs(d), 0.99)),
                "delta_signed_p05": float(np.quantile(d, 0.05)),
                "delta_signed_p50": float(np.quantile(d, 0.50)),
                "delta_signed_p95": float(np.quantile(d, 0.95)),
            })

        if self.all_gate_values:
            gate_all = np.concatenate(self.all_gate_values)
            out.update({
                "delta_gate_mean": float(gate_all.mean()),
                "delta_gate_p05": float(np.quantile(gate_all, 0.05)),
                "delta_gate_p50": float(np.quantile(gate_all, 0.50)),
                "delta_gate_p95": float(np.quantile(gate_all, 0.95)),
            })


        denom = max(self.total_elements, 1.0)
        out.update({
            "batches": float(self.batches),
            "delta_neg_frac": float(self.neg_count / denom),
            "delta_pos_frac": float(self.pos_count / denom),
            "delta_sat_frac_abs_ge_0p45": float(self.sat045_count / denom),
            "delta_sat_frac_abs_ge_0p49": float(self.sat049_count / denom),
            "delta_sat_frac_abs_ge_0p95max": float(self.sat095max_count / denom),
            "topk_true_coverage": float(self.topk_true_hits / max(self.total_true_pos, 1.0)),
            "prefilter_true_coverage": float(self.prefilter_true_hits / max(self.total_true_pos, 1.0)),
            "topk_ext_ge_0p5_coverage": float(self.topk_ext_high_hits / max(self.total_ext_high, 1.0)),
            "prefilter_ext_ge_0p5_coverage": float(self.prefilter_ext_high_hits / max(self.total_ext_high, 1.0)),
            "delta_vs_base_prob_pearson": self._pearson_from_pairs(self.delta_base_pairs),
            "delta_vs_external_prob_pearson": self._pearson_from_pairs(self.delta_ext_pairs),
            "delta_vs_true_label_pearson": self._pearson_from_pairs(self.delta_true_pairs),
        })
        return out




# -----------------------------------------------------------------------------
# Rare/common GO bins
# -----------------------------------------------------------------------------




def _task_keys(task: str) -> List[str]:
    task = normalize_task(task)
    aliases = {
        "cc": ["cc", "cellular_component"],
        "mf": ["mf", "molecular_function"],
        "bp": ["bp", "biological_process"],
    }
    return aliases.get(task, [task])




def _get_mode_task_dict(data: Any, mode: str, task: str) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict) or mode not in data:
        return None
    mode_d = data[mode]
    if not isinstance(mode_d, dict):
        return None
    for k in _task_keys(task):
        if k in mode_d:
            return mode_d[k]
    return None




def _label_counts_from_annotation_object(ann: Any, num_classes: int) -> Tuple[np.ndarray, int]:
    counts = np.zeros(int(num_classes), dtype=np.float64)
    n = 0


    if ann is None:
        return counts, n


    if isinstance(ann, np.ndarray):
        arr = ann
        if arr.ndim == 2 and arr.shape[1] == num_classes:
            return arr.astype(np.float32).sum(axis=0).astype(np.float64), int(arr.shape[0])
        if arr.ndim == 1 and arr.shape[0] == num_classes:
            return arr.astype(np.float32).astype(np.float64), 1


    if isinstance(ann, torch.Tensor):
        return _label_counts_from_annotation_object(ann.cpu().numpy(), num_classes)


    # List-like cases: each item may be a dense binary vector or a list of class ids.
    try:
        iterator = list(ann)
    except Exception:
        return counts, n


    for item in iterator:
        n += 1
        if isinstance(item, torch.Tensor):
            item = item.cpu().numpy()
        if isinstance(item, np.ndarray):
            if item.ndim == 1 and item.shape[0] == num_classes:
                counts += item.astype(np.float32)
            else:
                idx = item.astype(np.int64).reshape(-1)
                idx = idx[(idx >= 0) & (idx < num_classes)]
                counts[idx] += 1.0
        elif isinstance(item, (list, tuple, set)):
            vals = list(item)
            if len(vals) == num_classes and all(isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)) for v in vals[: min(10, len(vals))]):
                counts += np.asarray(vals, dtype=np.float32)
            else:
                for j in vals:
                    try:
                        jj = int(j)
                    except Exception:
                        continue
                    if 0 <= jj < num_classes:
                        counts[jj] += 1.0
    return counts, n




def load_train_label_counts(file_address: str | Path, task: str, num_classes: int) -> Tuple[np.ndarray, int, str]:
    data = load_pickle(str(file_address))
    d = _get_mode_task_dict(data, "train", task)
    if d is None:
        return np.zeros(num_classes, dtype=np.float64), 0, "none"


    # Try common keys in order. True labels should usually be 'annotations', but
    # some project pickles use 'prop_annotations'.
    for key in ["annotations", "labels", "prop_annotations"]:
        if key in d:
            counts, n = _label_counts_from_annotation_object(d[key], num_classes)
            if n > 0:
                return counts, n, key
    return np.zeros(num_classes, dtype=np.float64), 0, "none"


def build_ic_alpha_vector(
    num_classes: int,
    bins: Dict[str, np.ndarray],
    alpha_rare: float = 0.5,
    alpha_medium: float = 0.5,
    alpha_common: float = 0.8,
) -> np.ndarray:
    """
    Build per-class alpha vector for:
        p = alpha_i * expert + (1-alpha_i) * base_or_model

    bins values are boolean masks or index arrays from make_frequency_bins().
    """

    alpha = np.full((num_classes,), float(alpha_medium), dtype=np.float32)

    def _assign(bin_name: str, value: float):
        if bin_name not in bins:
            return
        m = bins[bin_name]
        if m.dtype == bool:
            alpha[m] = float(value)
        else:
            alpha[np.asarray(m, dtype=np.int64)] = float(value)

    _assign("rare", alpha_rare)
    _assign("medium", alpha_medium)
    _assign("common", alpha_common)

    # More specific bins override broader bins if present.
    if "rare_le_5" in bins:
        m = bins["rare_le_5"]
        alpha[m if getattr(m, "dtype", None) == bool else np.asarray(m, dtype=np.int64)] = float(alpha_rare)

    if "common_ge_50" in bins:
        m = bins["common_ge_50"]
        alpha[m if getattr(m, "dtype", None) == bool else np.asarray(m, dtype=np.int64)] = float(alpha_common)

    return alpha



def rare_analysis_metrics(
    y_true: np.ndarray,
    pred_map: Dict[str, np.ndarray],
    class_bins: Dict[str, np.ndarray],
    keys: List[str],
    threshold_step: float,
    auprc_mode: str,
    metrics_are_percent: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in pred_map:
            continue
        out[key] = {}
        for bin_name, mask in class_bins.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            yy = y_true[:, mask]
            pp = pred_map[key][:, mask]
            if yy.size == 0 or yy.sum() == 0:
                continue
            m = compute_metric_pack(
                yy,
                pp,
                threshold_step=threshold_step,
                auprc_mode=auprc_mode,
                compute_sample_fmax=False,
                no_empty_labels=False,
                no_zero_classes=False,
                metrics_are_percent=metrics_are_percent,
            )
            m["num_classes"] = int(mask.sum())
            m["num_positives"] = int(yy.sum())
            out[key][bin_name] = m
    return out




# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------




def should_use_external(mode: str) -> bool:
    m = str(mode).lower()
    return m in {"external", "external_topk", "blend", "blend_topk", "mixed", "external_or_base", "prob"}




def query_mode_to_source(mode: str) -> str:
    m = str(mode).lower()
    if m == "base_topk":
        return "base_topk"
    if m == "external_topk":
        return "external_topk"
    if m == "blend_topk":
        return "blend_topk"
    return m




def model_forward_once(
    model: WeakMSAGOWithDETRDecoder,
    x: torch.Tensor,
    permute_dims: Tuple[int, int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_logits, h = model.backbone(x, permute_dims=permute_dims, return_embedding=True)
    return base_logits, h



def run_eval(args: argparse.Namespace) -> Dict[str, Any]:
    task = normalize_task(args.task)
    set_seed(int(args.seed))


    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)


    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0
    args.task = task
    args.batch_size = int(args.eval_batch_size)
    args.dataloader_num_workers = int(args.dataloader_num_workers)


    if args.num_classes is None:
        n = infer_num_classes_from_checkpoint(args.checkpoint)
        if n is not None:
            args.num_classes = int(n)
    if args.num_classes is None:
        raise ValueError("num_classes is required or must be inferable from checkpoint payload")


    opt = build_weak_opt_from_config(args)
    opt.mode = args.mode
    opt.shuffle = False


    model = WeakMSAGOWithDETRDecoder(opt, args)
    load_info = load_detr_checkpoint(model, args.checkpoint, device=device)
    model = model.to(device)
    model.eval()


    print(f"[Eval] task={task}, mode={args.mode}, device={device}")
    print(f"[Checkpoint] {args.checkpoint}")
    print(f"[Checkpoint load] {json.dumps(load_info, indent=2, ensure_ascii=False)[:2000]}")
    print(f"[External prob] {args.external_prob_path}")


    base_dataset = build_msa_dataset(opt, mode=args.mode, task=task, need_proteins=True)
    external_path = clean_optional_path(args.external_prob_path)
    if external_path is not None:
        dataset = PseudoProbDataset(
            base_dataset=base_dataset,
            metadata_file=args.file_address,
            mode=args.mode,
            task=task,
            prob_path=external_path,
            num_classes=args.num_classes,
        )
        print(f"[ExternalProbDataset] shape={dataset.prob_shape}, dtype={dataset.prob_dtype}")
    else:
        dataset = base_dataset
        print("[ExternalProbDataset] disabled")


    original_np = None
    original_row_indices = None
    original_path = clean_optional_path(args.original_prob_path)
    if original_path is not None:
        # Reuse the same protein-name alignment and validation as external
        # probabilities. Shape equality alone is insufficient when the binary
        # MSA index has dropped proteins present in metadata.
        original_alignment = PseudoProbDataset(
            base_dataset=base_dataset,
            metadata_file=args.file_address,
            mode=args.mode,
            task=task,
            prob_path=original_path,
            num_classes=args.num_classes,
        )
        original_np = np.load(original_path, mmap_mode="r")
        original_row_indices = original_alignment.prob_indices
        print(
            f"[Original prob] {original_path}, shape={original_np.shape}, "
            f"dtype={original_np.dtype}, aligned_rows={len(original_row_indices)}"
        )


    loader = make_loader(
        dataset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        num_workers=int(args.dataloader_num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        rank=0,
        world_size=1,
        seed=int(args.seed),
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
    )


    query_modes = parse_csv_strings(args.query_modes)
    delta_scales = parse_csv_floats(args.delta_scales)
    ensemble_alphas = parse_csv_floats(args.ensemble_alphas)
    ensemble_query_modes = set(parse_csv_strings(args.ensemble_query_modes))
    ensemble_delta_scales = set([round(x, 6) for x in parse_csv_floats(args.ensemble_delta_scales)])


    decoder_prob_sources = parse_csv_strings(args.decoder_prob_sources)
    if not decoder_prob_sources:
        decoder_prob_sources = ["expert", "mix_expert_base", "mix_expert_original"]
    decoder_prob_mix_alphas = parse_csv_floats(args.decoder_prob_mix_alphas)
    if not decoder_prob_mix_alphas:
        decoder_prob_mix_alphas = [0.5, 0.7, 0.8, 0.9]


    pred_chunks: Dict[str, List[np.ndarray]] = defaultdict(list)
    label_chunks: List[np.ndarray] = []
    diagnostics: Dict[str, DeltaDiagnostics] = {}


    def get_diagnostics(name: str) -> DeltaDiagnostics:
        if name not in diagnostics:
            diagnostics[name] = DeltaDiagnostics(delta_max=float(args.query_decoder_delta_max))
        return diagnostics[name]


    amp_enabled = device.type == "cuda" and not bool(args.no_amp)
    permute_dims = tuple(int(x) for x in args.permute_dims)
    store_dtype = np.float16 if bool(args.store_float16) else np.float32


    start = time.time()
    sample_offset = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(loader, desc="detr-diagnostic-eval")):
            if args.max_batches is not None and int(args.max_batches) > 0 and batch_i >= int(args.max_batches):
                break


            if external_path is not None:
                proteins, x, y, ext_prob = unpack_pseudo_batch(batch)
            else:
                proteins, x, y = unpack_batch(batch)
                ext_prob = None


            batch_size = int(y.shape[0])
            batch_start = sample_offset
            sample_offset += batch_size
            orig_prob = get_prob_batch(
                original_np,
                batch_start,
                batch_size,
                device,
                row_indices=original_row_indices,
            )


            if proteins is not None:
                set_model_proteins(model, proteins)


            x = x.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True).float()
            if ext_prob is not None:
                ext_prob = ext_prob.to(device, non_blocking=True).float().clamp(0.0, 1.0)
                has_prob = torch.ones(y.shape[0], dtype=torch.bool, device=device)
            else:
                has_prob = None


            label_chunks.append(y.detach().cpu().numpy().astype(np.bool_))


            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                base_logits, h = model_forward_once(model, x, permute_dims=permute_dims)


                base_prob = torch.sigmoid(base_logits.float())
                pred_chunks["backbone_base"].append(base_prob.detach().cpu().numpy().astype(store_dtype))


                if ext_prob is not None:
                    pred_chunks["external_only"].append(ext_prob.detach().cpu().numpy().astype(store_dtype))


                if orig_prob is not None:
                    pred_chunks["original_prob_only"].append(orig_prob.detach().cpu().numpy().astype(store_dtype))


                # Direct probability-mix baselines, without query decoder.
                # These rows tell whether mixing itself is stronger than expert-only.
                if ext_prob is not None:
                    for alpha in decoder_prob_mix_alphas:
                        a = float(alpha)
                        mix_base = (a * ext_prob.float() + (1.0 - a) * base_prob.float()).clamp(0.0, 1.0)
                        pred_chunks[f"prob_mix::expert_base::alpha_{a:g}"].append(
                            mix_base.detach().cpu().numpy().astype(store_dtype)
                        )
                        if orig_prob is not None:
                            mix_orig = (a * ext_prob.float() + (1.0 - a) * orig_prob.float()).clamp(0.0, 1.0)
                            pred_chunks[f"prob_mix::expert_original::alpha_{a:g}"].append(
                                mix_orig.detach().cpu().numpy().astype(store_dtype)
                            )


                # New strategy: replace decoder external_prob input with a chosen
                # expert/base/original/mixed probability tensor.  topk_source is
                # external_topk and blend_alpha=1.0, so candidate proposal comes
                # exactly from decoder_prob.
                decoder_prob_variants = make_decoder_prob_variants(
                    expert_prob=ext_prob,
                    original_prob_file=orig_prob,
                    base_logits=base_logits,
                    sources=decoder_prob_sources,
                    mix_alphas=decoder_prob_mix_alphas,
                )
                for prob_name, decoder_prob in decoder_prob_variants:
                    q_has = torch.ones(y.shape[0], dtype=torch.bool, device=device)
                    qout = model.query_decoder(
                        h=h,
                        base_logits=base_logits,
                        classifier_weight=model.classifier.weight,
                        external_prob=decoder_prob,
                        has_external_prob=q_has,
                        topk_source="external_topk",
                        external_prob_blend_alpha=1.0,
                        y_hint=y if bool(args.allow_eval_label_boost) else None,
                    )
                    diag_key = f"decoderprob::{prob_name}"
                    get_diagnostics(diag_key).update(
                        qout=qout,
                        y_true=y,
                        base_logits=base_logits,
                        external_prob=decoder_prob,
                        external_high_threshold=float(args.external_high_threshold),
                    )
                    mode_name = str(getattr(args, "query_decoder_logit_base_mode", "base_residual"))
                    
                    if mode_name != "base_residual":
                        # Actual model output under the trained anchor/gated mode.
                        model_prob = torch.sigmoid(qout["logits"].float())
                        model_key = f"modelout::{mode_name}::{diag_key}"
                        pred_chunks[model_key].append(
                            model_prob.detach().cpu().numpy().astype(store_dtype)
                        )
                    
                        # Anchor generated by the model itself.
                        if "anchor_prob" in qout and qout["anchor_prob"] is not None:
                            anchor_prob = qout["anchor_prob"].float().clamp(0.0, 1.0)
                            anchor_key = f"modelanchor::{mode_name}::{diag_key}"
                            pred_chunks[anchor_key].append(
                                anchor_prob.detach().cpu().numpy().astype(store_dtype)
                            )
                    for scale in delta_scales:
                        logits_scaled = apply_delta_scale(qout, scale=float(scale))
                        prob_scaled = torch.sigmoid(logits_scaled.float())
                        key = f"query::decoderprob::{prob_name}::delta_scale_{float(scale):g}"
                        pred_chunks[key].append(prob_scaled.detach().cpu().numpy().astype(store_dtype))
                    # Eval-time anchor ablation:
                    #   anchor only:
                    #       logit(decoder_prob)
                    #   anchor + delta:
                    #       logit(decoder_prob) + scatter(delta)
                    #
                    # This tests whether the delta learned on weak base logits still improves
                    # a strong expert/base mixed probability anchor.
                    for scale in delta_scales:
                        anchor_logits_scaled = apply_delta_scale_on_anchor_prob(
                            qout=qout,
                            anchor_prob=decoder_prob,
                            scale=float(scale),
                        )
                        anchor_prob_scaled = torch.sigmoid(anchor_logits_scaled.float())
                    
                        anchor_key = (
                            f"anchorlogit::decoderprob::{prob_name}"
                            f"::delta_scale_{float(scale):g}"
                        )
                    
                        pred_chunks[anchor_key].append(
                            anchor_prob_scaled.detach().cpu().numpy().astype(store_dtype)
                        )


                if not bool(args.skip_legacy_query_sources):
                    legacy_query_modes_iter = query_modes
                else:
                    legacy_query_modes_iter = []


                for qmode in legacy_query_modes_iter:
                    use_ext = should_use_external(qmode)
                    if use_ext and ext_prob is None:
                        continue
                    q_ext = ext_prob if use_ext else None
                    q_has = has_prob if use_ext else None


                    qout = model.query_decoder(
                        h=h,
                        base_logits=base_logits,
                        classifier_weight=model.classifier.weight,
                        external_prob=q_ext,
                        has_external_prob=q_has,
                        topk_source=query_mode_to_source(qmode),
                        external_prob_blend_alpha=float(args.external_prob_blend_alpha),
                        y_hint=y if bool(args.allow_eval_label_boost) else None,
                    )


                    get_diagnostics(qmode).update(
                        qout=qout,
                        y_true=y,
                        base_logits=base_logits,
                        external_prob=q_ext,
                        external_high_threshold=float(args.external_high_threshold),
                    )


                    for scale in delta_scales:
                        logits_scaled = apply_delta_scale(qout, scale=float(scale))
                        prob_scaled = torch.sigmoid(logits_scaled.float())
                        key = f"query::{qmode}::delta_scale_{float(scale):g}"
                        pred_chunks[key].append(prob_scaled.detach().cpu().numpy().astype(store_dtype))


    y_true = np.concatenate(label_chunks, axis=0)
    pred_map: Dict[str, np.ndarray] = {}
    for key, chunks in pred_chunks.items():
        pred_map[key] = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    simulated_ic_meta = None

    if bool(args.enable_ic_fusion):
        if "external_only" not in pred_map:
            raise RuntimeError("IC fusion requires external_only in pred_map.")
        if "backbone_base" not in pred_map:
            raise RuntimeError("IC fusion requires backbone_base in pred_map.")

        counts, n_train, count_key = load_train_label_counts(args.file_address, task, int(args.num_classes))
        if n_train <= 0 and not bool(args.enable_simulated_ic):
            raise RuntimeError(
                "IC fusion requires GO label counts from the training split, but no usable "
                "train annotations were found. Refusing to derive bins from evaluation labels."
            )
            
        # Build bins using existing frequency bin helper.
        if bool(args.enable_simulated_ic):
            bins_to_use, simulated_counts, simulated_ic_meta = compute_simulated_bins_from_pred_map(
                pred_map=pred_map,
                source_key=str(args.simulated_ic_source_key),
                threshold_min=float(args.simulated_ic_threshold_min),
                threshold_max=float(args.simulated_ic_threshold_max),
                threshold_step=float(args.simulated_ic_threshold_step),
                include_zero_threshold=bool(args.simulated_ic_include_zero_threshold),
            )
        
            count_key = simulated_ic_meta["count_source_key"]
            n_train = simulated_ic_meta["num_samples"]
        
            if bool(getattr(args, "simulated_ic_save_counts", False)):
                sim_name = (
                    "simulated_ic_counts__"
                    + sanitize_key_for_filename(simulated_ic_meta["resolved_source_key"])
                    + ".npy"
                )
                np.save(Path(args.output_dir) / sim_name, simulated_counts)
        
            print("[ICFusion] using simulated IC bins")
            print(f"[ICFusion] source_key={args.simulated_ic_source_key}")
            print(f"[ICFusion] resolved_source_key={simulated_ic_meta['resolved_source_key']}")
            print(f"[ICFusion] thresholds={simulated_ic_meta['threshold_min']}..{simulated_ic_meta['threshold_max']} "
                  f"step={simulated_ic_meta['threshold_step']} "
                  f"n={simulated_ic_meta['num_thresholds']}")
            print(f"[ICFusion] bin_sizes={simulated_ic_meta['bin_sizes']}")
        
        else:
            bins_to_use = make_frequency_bins(counts)  # 传统训练集 counts
            simulated_counts = None
            simulated_ic_meta = None
        
        alpha_vec = build_ic_alpha_vector(
            num_classes=int(args.num_classes),
            bins=bins_to_use,
            alpha_rare=args.ic_fusion_alpha_rare,
            alpha_medium=args.ic_fusion_alpha_medium,
            alpha_common=args.ic_fusion_alpha_common,
        )
        alpha_mat = alpha_vec.reshape(1, -1)

        expert = pred_map["external_only"]

        if args.ic_fusion_source == "base":
            src = pred_map["backbone_base"]
            src_name = "base"
        elif args.ic_fusion_source == "modelout_a05":
            src_name = "modelout::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.5"
            src = pred_map[src_name]
        elif args.ic_fusion_source == "modelout_a08":
            src_name = "modelout::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.8"
            src = pred_map[src_name]
        elif args.ic_fusion_source == "modelanchor_a05":
            src_name = "modelanchor::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.5"
            src = pred_map[src_name]
        elif args.ic_fusion_source == "modelanchor_a08":
            src_name = "modelanchor::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.8"
            src = pred_map[src_name]
        else:
            raise ValueError(args.ic_fusion_source)

        ic_fused = alpha_mat * expert + (1.0 - alpha_mat) * src

        key = (
            f"ic_fusion::{args.ic_fusion_source}"
            f"::rare{args.ic_fusion_alpha_rare:g}"
            f"_medium{args.ic_fusion_alpha_medium:g}"
            f"_common{args.ic_fusion_alpha_common:g}"
        )

        pred_map[key] = ic_fused.astype(np.float32, copy=False)

    # Ensembles are computed after prediction collection to avoid extra forward passes.
    if "external_only" in pred_map and ensemble_alphas:
        ext = pred_map["external_only"]
        # query_keys = [k for k in list(pred_map.keys()) if k.startswith("query::")]
        query_keys = [
            k for k in list(pred_map.keys())
            if (
                k.startswith("query::")
                or k.startswith("anchorlogit::")
                or k.startswith("modelout::")
            )
        ]
        for k in query_keys:
            # Normal keys:
            #   query::{qmode}::delta_scale_1
            # Mixed-prob keys:
            #   query::decoderprob::{prob_name}::delta_scale_1
            parts = k.split("::")
            
            if k.startswith("anchorlogit::"):
                qmode = "anchorlogit"
            elif k.startswith("modelout::"):
                qmode = "modelout"
            elif len(parts) > 1:
                qmode = parts[1]
            else:
                qmode = ""
            scale_str = parts[-1].replace("delta_scale_", "") if len(parts) > 2 else "1"
            try:
                scale_val = round(float(scale_str), 6)
            except Exception:
                scale_val = 1.0
            if ensemble_query_modes and qmode not in ensemble_query_modes:
                continue
            if ensemble_delta_scales and scale_val not in ensemble_delta_scales:
                continue
            qp = pred_map[k]
            for alpha in ensemble_alphas:
                ens = float(alpha) * ext + (1.0 - float(alpha)) * qp
                pred_map[f"ensemble_ext_alpha_{float(alpha):g}::{k}"] = ens.astype(np.float32, copy=False)


    metric_results: Dict[str, Any] = {}
    for key, pred in pred_map.items():
        metric_results[key] = compute_metric_pack(
            y_true,
            pred,
            threshold_step=float(args.threshold_step),
            auprc_mode=str(args.auprc_mode),
            compute_sample_fmax=bool(args.compute_sample_fmax),
            no_empty_labels=bool(args.no_empty_labels),
            no_zero_classes=bool(args.no_zero_classes),
            metrics_are_percent=bool(args.metrics_are_percent),
        )


    # Optional rare/common analysis. By default use only a small set of keys.
    rare_results: Dict[str, Any] = {}
    frequency_info: Dict[str, Any] = {}
    rare_best: Dict[str, Any] = {}
    alpha_sweep: Dict[str, Any] = {}

    if bool(args.do_rare_analysis):
        counts, n_train, count_key = load_train_label_counts(args.file_address, task, int(args.num_classes))
        if n_train <= 0:
            raise RuntimeError(
                "Rare/common analysis requires GO label counts from the training split, but "
                "no usable train annotations were found. Refusing to derive bins from "
                "evaluation labels."
            )
        bins = make_frequency_bins(counts)
        rare_keys = parse_csv_strings(args.rare_metric_keys)
        
        if not rare_keys:
            rare_prefixes = (
                "prob_mix::expert_base::",
                "query::decoderprob::",
                "anchorlogit::decoderprob::",
                "modelanchor::",
                "modelout::",
                "ensemble_ext_alpha_",
                "ic_fusion::",
            )

            rare_exact = {
                "backbone_base",
                "external_only",
                "query::external_topk::delta_scale_1",
                "query::blend_topk::delta_scale_1",
                "query::external_topk::delta_scale_0.5",
                "query::blend_topk::delta_scale_0.5",
            }
        
            rare_keys = [
                k for k in pred_map.keys()
                if k in rare_exact or any(k.startswith(p) for p in rare_prefixes)
            ]

        rare_results = rare_analysis_metrics(
            y_true=y_true,
            pred_map=pred_map,
            class_bins=bins,
            keys=rare_keys,
            threshold_step=float(args.threshold_step),
            auprc_mode="hist" if args.auprc_mode != "none" else "none",
            metrics_are_percent=bool(args.metrics_are_percent),
        )

        for bin_name in ["rare", "rare_le_5", "medium", "common", "common_ge_50"]:
            rows = []
            for key, bin_metrics in rare_results.items():
                if bin_name not in bin_metrics:
                    continue
                m = bin_metrics[bin_name]
                rows.append((key, m["fmax_micro"], m["auprc_micro_hist"]))
        
            if rows:
                rare_best[bin_name] = {
                    "best_fmax": max(rows, key=lambda x: x[1]),
                    "best_auprc": max(rows, key=lambda x: x[2]),
                }
            else:
                rare_best[bin_name] = {
                    "best_fmax": None,
                    "best_auprc": None,
                    "reason": "no valid metrics for this bin",
                }

        for bin_name in bins:
            candidates = []
            for key in rare_results:
                if key.startswith("prob_mix::expert_base::alpha_"):
                    alpha = float(key.split("alpha_")[-1])
                    m = rare_results[key].get(bin_name)
                    if m is None:
                        continue
                    candidates.append((alpha, m["fmax_micro"], m["auprc_micro_hist"]))

            if candidates:
                alpha_sweep[bin_name] = {
                    "best_alpha_by_fmax": max(candidates, key=lambda x: x[1]),
                    "best_alpha_by_auprc": max(candidates, key=lambda x: x[2]),
                }
            else:
                alpha_sweep[bin_name] = {
                    "best_alpha_by_fmax": None,
                    "best_alpha_by_auprc": None,
                    "reason": "no probability-mix metrics for this bin",
                }

        frequency_info = {
            "count_source_key": count_key,
            "num_train_samples_for_counts": int(n_train),
            "num_zero_train_classes": int((counts == 0).sum()),
            "num_positive_train_classes": int((counts > 0).sum()),
            "bin_sizes": {k: int(v.sum()) for k, v in bins.items()},
        }


    delta_results = {m: d.to_dict() for m, d in diagnostics.items() if d.batches > 0}


    result = {
        "task": task,
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "external_prob_path": str(args.external_prob_path) if args.external_prob_path else None,
        "original_prob_path": str(args.original_prob_path) if args.original_prob_path else None,
        "decoder_prob_sources": decoder_prob_sources,
        "decoder_prob_mix_alphas": decoder_prob_mix_alphas,
        "skip_legacy_query_sources": bool(args.skip_legacy_query_sources),
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(y_true.shape[1]),
        "metrics_are_percent": bool(args.metrics_are_percent),
        "threshold_step": float(args.threshold_step),
        "auprc_mode": str(args.auprc_mode),
        "query_modes": query_modes,
        "delta_scales": delta_scales,
        "ensemble_alphas": ensemble_alphas,

        "ic_fusion_enabled": bool(args.enable_ic_fusion),
        "simulated_ic_enabled": bool(args.enable_simulated_ic),
        "simulated_ic_source_key": str(args.simulated_ic_source_key),
        "simulated_ic_meta": simulated_ic_meta,

        "load_info": load_info,
        "metrics": metric_results,
        "delta_diagnostics": delta_results,
        "frequency_info": frequency_info,
        "rare_analysis": rare_results,
        "elapsed_sec": time.time() - start,
    }


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "detr_diagnostic_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=True)
    with (output_dir / "detr_metric_results.json").open("w", encoding="utf-8") as f:
        json.dump(metric_results, f, indent=2, ensure_ascii=False, allow_nan=True)
    with (output_dir / "detr_delta_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(delta_results, f, indent=2, ensure_ascii=False, allow_nan=True)
    with (output_dir / "effective_eval_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, default=str)
    if rare_results:
        with (output_dir / "detr_rare_analysis.json").open("w", encoding="utf-8") as f:
            json.dump(rare_results, f, indent=2, ensure_ascii=False, allow_nan=True)
    if rare_best:
        with (output_dir / "detr_rare_best_by_bin.json").open("w", encoding="utf-8") as f:
            json.dump(rare_best, f, indent=2, ensure_ascii=False, allow_nan=True)

    if alpha_sweep:
        with (output_dir / "detr_rare_alpha_sweep.json").open("w", encoding="utf-8") as f:
            json.dump(alpha_sweep, f, indent=2, ensure_ascii=False, allow_nan=True)


    write_text_summary(result, output_dir / "detr_diagnostic_summary.txt")
    print(f"[Saved] {output_dir}")
    print_compact_summary(result)
    return result




def write_text_summary(result: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append(f"task={result['task']} mode={result['mode']}")
    lines.append(f"checkpoint={result['checkpoint']}")
    lines.append(f"external_prob_path={result['external_prob_path']}")
    lines.append(f"original_prob_path={result.get('original_prob_path')}")
    lines.append(f"decoder_prob_sources={result.get('decoder_prob_sources')}")
    lines.append(f"decoder_prob_mix_alphas={result.get('decoder_prob_mix_alphas')}")
    lines.append(f"num_samples={result['num_samples']} num_classes={result['num_classes']}")
    lines.append("")
    lines.append("[Metrics]")
    for key, m in result["metrics"].items():
        parts = [
            f"fmax_micro={m.get('fmax_micro', float('nan')):.4f}",
            f"thr={m.get('threshold_micro', float('nan')):.3f}",
        ]
        if "auprc_micro_exact" in m:
            parts.append(f"auprc_exact={m['auprc_micro_exact']:.4f}")
        if "auprc_micro_hist" in m:
            parts.append(f"auprc_hist={m['auprc_micro_hist']:.4f}")
        lines.append(f"  {key}: " + ", ".join(parts))


    lines.append("")
    lines.append("[Delta diagnostics]")
    for key, d in result["delta_diagnostics"].items():
        parts = [
            f"mean={d.get('delta_all_mean', float('nan')):.4f}",
            f"abs_mean={d.get('delta_all_abs_mean', float('nan')):.4f}",
            f"abs_p95={d.get('delta_abs_p95', float('nan')):.4f}",
            f"neg_frac={d.get('delta_neg_frac', float('nan')):.4f}",
            f"sat049={d.get('delta_sat_frac_abs_ge_0p49', float('nan')):.4f}",
            f"true_pos_mean={d.get('delta_true_pos_mean', float('nan')):.4f}",
            f"true_neg_mean={d.get('delta_true_neg_mean', float('nan')):.4f}",
            f"topk_true_cov={d.get('topk_true_coverage', float('nan')):.4f}",
            f"prefilter_true_cov={d.get('prefilter_true_coverage', float('nan')):.4f}",
        ]
        lines.append(f"  {key}: " + ", ".join(parts))


    if result.get("frequency_info"):
        lines.append("")
        lines.append("[Frequency bins]")
        lines.append(json.dumps(result["frequency_info"], ensure_ascii=False))

    lines.append(f"ic_fusion_enabled={result.get('ic_fusion_enabled')}")
    lines.append(f"simulated_ic_enabled={result.get('simulated_ic_enabled')}")
    lines.append(f"simulated_ic_source_key={result.get('simulated_ic_source_key')}")
    if result.get("simulated_ic_meta") is not None:
        lines.append(f"simulated_ic_meta={json.dumps(result['simulated_ic_meta'], ensure_ascii=False)}")


    path.write_text("\n".join(lines) + "\n", encoding="utf-8")




def print_compact_summary(result: Dict[str, Any]) -> None:
    print("\n[Compact metrics]")
    preferred = [
        "backbone_base",
        "external_only",
        "query::base_topk::delta_scale_1",
        "query::external_topk::delta_scale_1",
        "query::blend_topk::delta_scale_1",
        "query::external_topk::delta_scale_0.5",
        "query::blend_topk::delta_scale_0.5",
    ]

    printed = set()
    def _print_metric_row(key: str) -> None:
        if key not in result["metrics"] or key in printed:
            return
        printed.add(key)
        m = result["metrics"][key]
        ap_key = "auprc_micro_exact" if "auprc_micro_exact" in m else "auprc_micro_hist"
        print(
            f"  {key:62s} "
            f"Fmicro={m.get('fmax_micro', float('nan')):8.4f} "
            f"thr={m.get('threshold_micro', float('nan')):5.3f} "
            f"AuPRC={m.get(ap_key, float('nan')):8.4f}"
        )


    for key in preferred:
        _print_metric_row(key)


    # Print newly added probability-mix and decoder-prob rows.
    for key in sorted(result["metrics"].keys()):
        if (
            key.startswith("original_prob_only")
            or key.startswith("original_only")
            or key.startswith("prob_mix::")
            or key.startswith("query::decoderprob::")
            or key.startswith("anchorlogit::decoderprob::")
            or key.startswith("modelanchor::")
            or key.startswith("modelout::")
            or key.startswith("ensemble_ext_alpha_")
        ):
            _print_metric_row(key)


    print("\n[Compact delta]")
    for key, d in result["delta_diagnostics"].items():
        print(
            f"  {key:14s} "
            f"mean={d.get('delta_all_mean', float('nan')):+.4f} "
            f"abs={d.get('delta_all_abs_mean', float('nan')):.4f} "
            f"p95={d.get('delta_abs_p95', float('nan')):.4f} "
            f"neg={d.get('delta_neg_frac', float('nan')):.3f} "
            f"sat049={d.get('delta_sat_frac_abs_ge_0p49', float('nan')):.3f} "
            f"d_pos={d.get('delta_true_pos_mean', float('nan')):+.4f} "
            f"d_neg={d.get('delta_true_neg_mean', float('nan')):+.4f} "
            f"tcov={d.get('topk_true_coverage', float('nan')):.3f}"
        )




# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------




def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DETR weak query decoder diagnostic evaluator")


    # Checkpoint/eval fields.
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train_args_json", type=str, default="auto")
    p.add_argument(
        "--allow_missing_train_args",
        action="store_true",
        help="Allow full DETR evaluation without args.json/embedded model_args. Unsafe by default.",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", type=str, default="ind_test")
    p.add_argument("--external_prob_path", type=str, default=None)
    
    # probability path
    p.add_argument(
        "--original_prob_path",
        type=str,
        default=None,
        help="Optional original/teacher/raw probability file aligned to the eval dataset.",
    )
    p.add_argument(
        "--decoder_prob_sources",
        type=str,
        default="expert,mix_expert_base,mix_expert_original",
        help=(
            "Comma-separated decoder probability sources. Options: "
            "expert, base, original, mix_expert_base, mix_expert_original."
        ),
    )
    p.add_argument(
        "--decoder_prob_mix_alphas",
        type=str,
        default="0.5,0.7,0.8,0.9",
        help="alpha for alpha*expert_prob + (1-alpha)*original_prob/base_prob.",
    )
    p.add_argument(
        "--skip_legacy_query_sources",
        action="store_true",
        help="If set, skip the old base_topk/external_topk/blend_topk loop and only run decoder_prob variants.",
    )


    # eval config
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=None)


    # Required Arch/dataset fields. Can be auto-filled from training args.json.
    p.add_argument("--model_config", type=str, default=None)
    p.add_argument("--init_ckpt", type=str, default="")
    p.add_argument("--file_address", type=str, default=None)
    p.add_argument("--working_address", type=str, default=None)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--msa_max_size", type=int, default=None)
    p.add_argument("--permute_dims", type=int, nargs=4, default=[0, 3, 2, 1])
    p.add_argument("--torch_compile", action="store_true")


    # MSA/dataloader fields.
    p.add_argument("--msa_read_mode", type=str, choices=["full", "rows", "block"], default="full")
    p.add_argument("--msa_sample_strategy", type=str, choices=["random", "block", "head"], default="random")
    p.add_argument("--msa_shuffle_rows_at_getitem", type=str2bool, default=False)
    p.add_argument("--msa_cache_gb", type=float, default=4.0)
    p.add_argument("--msa_max_open_files", type=int, default=256)
    p.add_argument("--sample_seed", type=int, default=1)
    p.add_argument("--sampler_seed", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", type=str2bool, default=True)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=str2bool, default=True)
    p.add_argument("--gpu_ids", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--no_amp", action="store_true")


    # Decoder fields; default values match weak_exp_train_detr.py / runner.
    p.add_argument("--use_query_decoder", type=str2bool, default=True)
    p.add_argument("--query_decoder_topk", type=int, default=100)
    p.add_argument("--query_decoder_topk_source", type=str, default="blend")
    p.add_argument("--selector_static_source", type=str, default="blend")
    p.add_argument("--external_prob_blend_alpha", type=float, default=1.0)
    p.add_argument("--query_decoder_mode", type=str, default="residual", choices=["residual", "replace"])
    p.add_argument("--query_decoder_dim", type=int, default=256)
    p.add_argument("--query_decoder_heads", type=int, default=8)
    p.add_argument("--query_decoder_layers", type=int, default=1)
    p.add_argument("--query_decoder_ffn_dim", type=int, default=1024)
    p.add_argument("--query_decoder_dropout", type=float, default=0.1)
    p.add_argument("--query_decoder_delta_max", type=float, default=0.5)
    p.add_argument("--query_decoder_detach_query_weight", type=str2bool, default=True)
    p.add_argument("--query_decoder_include_label_boost", type=str2bool, default=False)
    p.add_argument("--query_decoder_label_boost", type=float, default=2.0)
    p.add_argument("--query_decoder_memory_mode", type=str, default="tokens_plus_pooled", choices=["pooled", "tokens", "tokens_plus_pooled"])
    p.add_argument("--query_decoder_memory_grid_h", type=int, default=0)
    p.add_argument("--query_decoder_memory_grid_w", type=int, default=0)

    p.add_argument(
        "--query_decoder_logit_base_mode",
        type=str,
        default="base_residual",
        choices=[
            "base_residual",
            "expert_base",
            "mix_expert_base",
            "anchor_delta",
            "mix_expert_base_anchor",
        ],
    )
    
    p.add_argument(
        "--expert_base_mix_alpha",
        type=float,
        default=0.8,
    )
    
    p.add_argument(
        "--anchor_delta_gate_init",
        type=float,
        default=0.1,
    )


    p.add_argument("--use_trainable_query_embedding", type=str2bool, default=True)
    p.add_argument("--query_embed_init", type=str, default="classifier_plus_residual", choices=["classifier_plus_residual", "classifier_weight", "random"])
    p.add_argument("--query_embed_residual_scale", type=float, default=0.1)
    p.add_argument("--use_query_score_features", type=str2bool, default=True)
    p.add_argument("--query_score_embed_scale", type=float, default=0.1)
    p.add_argument("--query_score_detach", type=str2bool, default=True)


    p.add_argument("--use_learnable_selector", type=str2bool, default=True)
    p.add_argument("--selector_prefilter_topm", type=int, default=1024)
    p.add_argument("--selector_hidden_dim", type=int, default=64)
    p.add_argument("--selector_pos_weight", type=float, default=10.0)
    p.add_argument("--selector_logit_residual_scale", type=float, default=1.0)
    p.add_argument("--selector_use_term_bias", type=str2bool, default=True)
    p.add_argument("--selector_use_protein_term_affinity", type=str2bool, default=True)
    p.add_argument("--selector_detach_base_logits", type=str2bool, default=True)
    p.add_argument("--selector_detach_term_features", type=str2bool, default=True)


    # Diagnostic settings.
    p.add_argument("--query_modes", type=str, default="base_topk,external_topk,blend_topk")
    p.add_argument("--delta_scales", type=str, default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--external_high_threshold", type=float, default=0.5)
    p.add_argument("--allow_eval_label_boost", action="store_true")
    p.add_argument("--store_float16", type=str2bool, default=True)


    # Metric settings.
    p.add_argument("--threshold_step", type=float, default=0.01)
    p.add_argument("--auprc_mode", type=str, default="hist", choices=["hist", "exact", "none"])
    p.add_argument("--compute_sample_fmax", action="store_true")
    p.add_argument("--no_empty_labels", action="store_true")
    p.add_argument("--no_zero_classes", action="store_true")
    p.add_argument("--metrics_are_percent", type=str2bool, default=True)


    # Ensembles: alpha is weight on external_prob.
    p.add_argument("--ensemble_alphas", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--ensemble_query_modes", type=str, default="external_topk,blend_topk,decoderprob,anchorlogit,modelout")
    p.add_argument("--ensemble_delta_scales", type=str, default="0.5,1.0")


    # Rare/common GO analysis.
    p.add_argument("--do_rare_analysis", type=str2bool, default=True)
    p.add_argument("--rare_metric_keys", type=str, default="")

    # IC aware fusion
    p.add_argument(
        "--enable_ic_fusion",
        action="store_true",
    )
    
    p.add_argument(
        "--ic_fusion_alpha_rare",
        type=float,
        default=0.5,
    )
    
    p.add_argument(
        "--ic_fusion_alpha_medium",
        type=float,
        default=0.5,
    )
    
    p.add_argument(
        "--ic_fusion_alpha_common",
        type=float,
        default=0.8,
    )
    
    p.add_argument(
        "--ic_fusion_source",
        type=str,
        default="base",
        choices=["base", "modelout_a05", "modelout_a08", "modelanchor_a05", "modelanchor_a08"],
    )

    p.add_argument(
        "--enable_simulated_ic",
        action="store_true",
        help="Use pred_map[source] threshold-sweep simulated counts for IC/frequency bins.",
    )

    p.add_argument(
        "--simulated_ic_source_key",
        type=str,
        default="backbone_base",
        help=(
            "Which pred_map key is used to simulate counts. "
            "Aliases: base, expert, probmix_a08, modelout_a05, modelanchor_a05, etc."
        ),
    )

    p.add_argument(
        "--simulated_ic_threshold_min",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--simulated_ic_threshold_max",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--simulated_ic_threshold_step",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--simulated_ic_include_zero_threshold",
        action="store_true",
        help=(
            "Include threshold 0.0 in simulated IC. Usually not recommended because "
            "it gives every GO term an artificial positive count."
        ),
    )

    p.add_argument(
        "--simulated_ic_save_counts",
        action="store_true",
        help="Save simulated counts as .npy for later inspection.",
    )


    return p




def validate_args(args: argparse.Namespace) -> None:
    required = ["model_config", "file_address", "working_address"]
    missing = [k for k in required if getattr(args, k, None) in {None, ""}]
    if missing:
        raise ValueError(
            f"Missing required fields after args.json overlay: {missing}. "
            "Pass them explicitly or ensure checkpoint_dir/args.json exists."
        )
    if clean_optional_path(args.external_prob_path) is None:
        q_modes = parse_csv_strings(args.query_modes)
        decoder_sources = parse_csv_strings(getattr(args, "decoder_prob_sources", ""))
        if any(should_use_external(m) for m in q_modes):
            print("[Warning] external query modes requested but external_prob_path is None; they will be skipped.")
        if any(s in {"expert", "mix_expert_base", "mix_expert_original"} for s in decoder_sources):
            print("[Warning] decoder_prob_sources request expert probability but external_prob_path is None; expert/mix variants will be skipped.")
    if clean_optional_path(getattr(args, "original_prob_path", None)) is None:
        decoder_sources = parse_csv_strings(getattr(args, "decoder_prob_sources", ""))
        if any(s in {"original", "mix_expert_original"} for s in decoder_sources):
            print("[Warning] original/mix_expert_original requested but original_prob_path is None; those variants will be skipped.")




def main() -> None:
    parser = build_argparser()
    cli_keys = get_cli_provided_keys(sys.argv[1:])
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Warning] Ignoring unknown arguments: {unknown}")
    args.task = normalize_task(args.task)
    args = overlay_train_args(args, cli_keys=cli_keys)
    args.task = normalize_task(args.task)
    validate_args(args)
    run_eval(args)




if __name__ == "__main__":
    main()
