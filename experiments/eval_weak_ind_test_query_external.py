#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiments/eval_weak_ind_test_query.py

Evaluate weak_exp_train_query checkpoints on ind_test / test splits.

Supported checkpoints:
    1. weak_query_decoder_epoch*.pt / weak_query_decoder_last.pt
       Expected keys:
           - backbone
           - query_decoder

    2. weak_backbone_epoch*.pt / weak_backbone_last.pt
       Evaluated as a normal Arch backbone when --no_use_query_decoder is set.

This evaluator reuses the mature distributed evaluation, binary-MSA loader,
metric, and checkpoint utilities from experiments/eval_ind_test.py, but builds
an Arch + DETR-style ontology query decoder model for full query-decoder eval.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader, Sampler, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
MSA_ROOT = ROOT / "msa_models"

if not MSA_ROOT.exists():
    raise FileNotFoundError(f"Cannot find msa_models directory: {MSA_ROOT}")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MSA_ROOT))

from models import Arch  # noqa: E402

# Reuse existing evaluator utilities.
from experiments.eval_ind_test import (  # noqa: E402
    load_pickle,
    save_json,
    load_pickle_config,
    torch_load_cpu,
    set_seed,
    clean_optional_path,
    build_eval_opt_from_config,
    build_msa_dataset,
    make_eval_loader,
    extract_state_dict,
    load_arch_checkpoint,
    predict_dataset,
    collect_predictions,
    compute_evalperf_metrics,
)

from experiments.exp_train import (  # noqa: E402
    normalize_task,
    parse_gpu_ids_arg,
    init_distributed_mode,
    cleanup_distributed,
    dist_is_initialized,
    get_rank,
    get_world_size,
    is_main_process,
    rank0_print,
    strip_state_dict_prefix,
)

# Import query-decoder building blocks from the training file.
# This avoids duplicating the decoder definition, while the eval model below
# owns a safe eval-only forward implementation.
from experiments.weak_exp_train_query import (  # noqa: E402
    OntologyQueryDecoder,
    find_classifier_linear,
)


# ---------------------------------------------------------------------
# State-dict helpers
# ---------------------------------------------------------------------

def _compatible_state_dict(
    model: nn.Module,
    sd: dict,
    strict_shape: bool = True,
    name: str = "model",
):
    sd = strip_state_dict_prefix(sd)
    model_sd = model.state_dict()

    load_sd = {}
    ignored = []
    skipped_shape = []

    for k, v in sd.items():
        if k not in model_sd:
            ignored.append(k)
            continue

        if tuple(model_sd[k].shape) != tuple(v.shape):
            skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue

        load_sd[k] = v

    if strict_shape and len(load_sd) == 0:
        raise RuntimeError(f"No compatible tensors loaded for {name}")

    missing, unexpected = model.load_state_dict(load_sd, strict=False)

    rank0_print(f"[Load:{name}] loaded tensors: {len(load_sd)}")
    rank0_print(f"[Load:{name}] ignored keys: {len(ignored)}")
    rank0_print(f"[Load:{name}] skipped shape mismatch: {len(skipped_shape)}")
    rank0_print(f"[Load:{name}] missing after partial load: {len(missing)}")
    rank0_print(f"[Load:{name}] unexpected after partial load: {len(unexpected)}")

    if skipped_shape and is_main_process():
        rank0_print(f"[Load:{name}] First shape mismatches:")
        for item in skipped_shape[:10]:
            rank0_print("  ", item)


def _load_query_checkpoint(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Query checkpoint not found: {path}")

    raw = torch_load_cpu(path)

    if not isinstance(raw, dict):
        raise TypeError(
            "Query checkpoint must be a dict with keys 'backbone' and "
            f"'query_decoder', got {type(raw)}"
        )

    if "backbone" not in raw or "query_decoder" not in raw:
        keys = list(raw.keys())
        raise KeyError(
            "Query decoder eval requires checkpoint with keys 'backbone' and "
            f"'query_decoder'. Got keys: {keys[:20]}"
        )

    return raw

def unpack_eval_batch_with_external_prob(batch):
    input_data, y_pack = batch

    external_prob = None

    if isinstance(y_pack, dict):
        y = y_pack["target"]
        external_prob = y_pack.get("external_prob", None)
    else:
        y = y_pack

    return input_data, y, external_prob


class ExternalProbEvalDataset(Dataset):
    """
    Wrap an eval dataset and attach external probability vectors.

    Default alignment assumes external_prob_path rows follow the exact order of
    the base dataset. For safer use, provide --external_prob_protein_ids and make
    sure base_dataset.proteins is available.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        external_prob_path: Union[str, Path],
        num_classes: int,
        external_prob_protein_ids: Optional[Union[str, Path]] = None,
    ):
        self.base = base_dataset
        self.external_prob_path = str(external_prob_path)
        self.num_classes = int(num_classes)
        self.external_prob_protein_ids = None if external_prob_protein_ids is None else str(external_prob_protein_ids)

        prob = np.load(self.external_prob_path, mmap_mode="r")
        if prob.ndim != 2:
            raise ValueError(f"external prob must be 2D, got {prob.shape}")
        if prob.shape[1] != self.num_classes:
            raise ValueError(
                f"external prob class mismatch: prob={prob.shape[1]}, "
                f"num_classes={self.num_classes}"
            )

        self.prob_shape = tuple(prob.shape)
        self.prob_dtype = str(prob.dtype)

        if self.external_prob_protein_ids is None:
            if prob.shape[0] != len(base_dataset):
                raise ValueError(
                    "Exact-order external prob requires rows == dataset length: "
                    f"prob rows={prob.shape[0]}, dataset={len(base_dataset)}. "
                    "Provide --external_prob_protein_ids for protein-name alignment."
                )
            self.row_indices = np.arange(len(base_dataset), dtype=np.int64)
        else:
            protein_ids = self._load_protein_ids(self.external_prob_protein_ids)
            if len(protein_ids) != prob.shape[0]:
                raise ValueError(
                    f"protein id count={len(protein_ids)} but prob rows={prob.shape[0]}"
                )
            if not hasattr(base_dataset, "proteins"):
                raise RuntimeError(
                    "Protein-id aligned external prob requires base_dataset.proteins."
                )
            protein_to_row = {}
            for i, p in enumerate(protein_ids):
                p = str(p)
                if p in protein_to_row:
                    raise RuntimeError(f"duplicated protein id in external ids: {p}")
                protein_to_row[p] = i
            rows = []
            missing = []
            for p in base_dataset.proteins:
                p = str(p)
                if p not in protein_to_row:
                    missing.append(p)
                else:
                    rows.append(protein_to_row[p])
            if missing:
                raise RuntimeError(
                    f"{len(missing)} dataset proteins missing from external ids. "
                    f"Examples: {missing[:10]}"
                )
            self.row_indices = np.asarray(rows, dtype=np.int64)

        self._prob = None
        del prob

    @staticmethod
    def _load_protein_ids(path: Union[str, Path]):
        path = Path(path)
        if path.suffix.lower() in {".json"}:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                # Accept common keys.
                for k in ("proteins", "protein_ids", "ids"):
                    if k in obj:
                        return [str(x) for x in obj[k]]
                raise KeyError(f"No proteins/protein_ids/ids key found in {path}")
            return [str(x) for x in obj]
        with path.open("r", encoding="utf-8") as f:
            return [line.strip().split()[0] for line in f if line.strip()]

    @property
    def prob(self):
        if self._prob is None:
            self._prob = np.load(self.external_prob_path, mmap_mode="r")
        return self._prob

    @property
    def proteins(self):
        return getattr(self.base, "proteins", None)

    @property
    def sample_shard_ids(self):
        return getattr(self.base, "sample_shard_ids", None)

    def __len__(self):
        return len(self.base)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_prob"] = None
        return state

    def __getitem__(self, idx):
        input_data, y = self.base[idx]
        row = int(self.row_indices[idx])
        p = np.array(self.prob[row], dtype=np.float16, copy=True)
        return input_data, {
            "target": y,
            "external_prob": torch.from_numpy(p),
            "external_prob_index": torch.tensor(row, dtype=torch.long),
        }


# ---------------------------------------------------------------------
# Query-decoder eval model
# ---------------------------------------------------------------------

class EvalWeakMSAGOWithQueryDecoder(nn.Module):
    """
    Eval-only wrapper: Arch backbone + ontology query decoder.

    In evaluation, there is no teacher prob and no label hint. Therefore top-k
    query indices are always selected from base logits.
    """

    def __init__(self, opt: SimpleNamespace, args: argparse.Namespace):
        super().__init__()

        self.backbone = Arch(opt)
        self.num_classes = int(args.num_classes)
        self.topk = int(args.query_decoder_topk)
        self.query_decoder_mode = str(args.query_decoder_mode)
        self.query_decoder_topk_source = str(getattr(args, "query_decoder_topk_source", "base"))
        self.external_prob_blend_alpha = float(getattr(args, "external_prob_blend_alpha", 1.0))
        self.uses_external_query_prob = True

        classifier = find_classifier_linear(self.backbone, self.num_classes)
        self.classifier = classifier
        feature_dim = int(classifier.in_features)

        self.query_decoder = OntologyQueryDecoder(
            feature_dim=feature_dim,
            decoder_dim=int(args.query_decoder_dim),
            num_heads=int(args.query_decoder_heads),
            num_layers=int(args.query_decoder_layers),
            ffn_dim=int(args.query_decoder_ffn_dim),
            dropout=float(args.query_decoder_dropout),
            memory_mode=str(args.query_decoder_memory_mode),
            memory_grid_h=int(args.query_decoder_memory_grid_h),
            memory_grid_w=int(args.query_decoder_memory_grid_w),
            detach_query_weight=bool(args.query_decoder_detach_query_weight),
        )

    @torch.no_grad()
    def build_topk_idx(
        self,
        base_logits: torch.Tensor,
        external_prob: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = min(self.topk, base_logits.shape[1])
        base_score = torch.sigmoid(base_logits.float())

        source = self.query_decoder_topk_source
        if source == "base":
            score = base_score
        elif source == "external_prob":
            if external_prob is None:
                raise RuntimeError("query_decoder_topk_source='external_prob' but external_prob is None")
            score = external_prob.float().to(device=base_logits.device)
        elif source == "external_or_base":
            if external_prob is None:
                score = base_score
            else:
                score = external_prob.float().to(device=base_logits.device)
        elif source == "blend":
            if external_prob is None:
                score = base_score
            else:
                ext = external_prob.float().to(device=base_logits.device)
                alpha = max(0.0, min(1.0, self.external_prob_blend_alpha))
                score = alpha * ext + (1.0 - alpha) * base_score
        else:
            raise ValueError(f"Unknown query_decoder_topk_source: {source}")

        if score.shape != base_score.shape:
            raise ValueError(f"top-k score shape mismatch: score={score.shape}, base={base_score.shape}")
        return score.to(base_logits.dtype), torch.topk(score, k=k, dim=1).indices

    def forward(self, x: torch.Tensor, permute_dims=(0, 3, 2, 1), external_prob: Optional[torch.Tensor] = None):
        out = self.backbone(
            x,
            permute_dims=permute_dims,
            return_embedding=True,
        )

        if isinstance(out, dict):
            base_logits = out["logits"]
            h = out.get("embedding", out.get("h", None))
            if h is None:
                raise RuntimeError("Backbone dict output lacks embedding/h.")
        elif isinstance(out, (tuple, list)):
            if len(out) < 2:
                raise RuntimeError("Backbone return_embedding=True should return (logits, h).")
            base_logits, h = out[0], out[1]
        else:
            raise RuntimeError("Backbone return_embedding=True returned unsupported output.")

        base_logits, topk_idx = self.build_topk_idx(base_logits.detach(), external_prob=external_prob)

        qout = self.query_decoder(
            h=h,
            base_logits=base_logits,
            classifier_weight=self.classifier.weight,
            topk_idx=topk_idx,
        )

        if isinstance(qout, dict):
            refined_logits = qout["logits"]
            delta = qout.get("delta", None)
        elif isinstance(qout, (tuple, list)):
            refined_logits = qout[0]
            delta = qout[1] if len(qout) > 1 else None
        else:
            refined_logits = qout
            delta = None

        if self.query_decoder_mode == "residual":
            logits = refined_logits
        elif self.query_decoder_mode == "replace":
            # The current OntologyQueryDecoder implementation is residual.
            # True replace mode would require reconstructing a full logit tensor
            # with non-topk classes masked or set to base logits. To avoid silent
            # behavior mismatch, keep residual unless training code implements replace.
            logits = refined_logits
        else:
            raise ValueError(f"Unknown query_decoder_mode: {self.query_decoder_mode}")

        return {
            "logits": logits,
            "base_logits": base_logits,
            "topk_idx": topk_idx,
            "delta": delta,
            "query_decoder_topk_source": self.query_decoder_topk_source,
        }


def build_query_model(opt: SimpleNamespace, args: argparse.Namespace, device: torch.device):
    model = EvalWeakMSAGOWithQueryDecoder(opt, args)
    ckpt = _load_query_checkpoint(args.ckpt)

    _compatible_state_dict(
        model.backbone,
        ckpt["backbone"],
        strict_shape=bool(args.strict_shape),
        name="backbone",
    )

    # Query decoder should normally match exactly. Keep strict=True by default.
    if bool(args.strict_query_decoder):
        missing, unexpected = model.query_decoder.load_state_dict(ckpt["query_decoder"], strict=True)
        rank0_print(f"[Load:query_decoder] strict load; missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        _compatible_state_dict(
            model.query_decoder,
            ckpt["query_decoder"],
            strict_shape=bool(args.strict_shape),
            name="query_decoder",
        )

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_dataset_with_optional_external_prob(
    primary_model: nn.Module,
    teacher_model: Optional[nn.Module],
    primary_weight: float,
    teacher_weight: float,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    permute_dims=(0, 3, 2, 1),
    no_amp: bool = False,
    need_proteins: bool = False,
):
    primary_model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    all_targs = []
    all_preds = []

    amp_enabled = (device.type == "cuda" and not no_amp)
    rank = get_rank()

    iterator = tqdm(loader, desc=f"predict-rank{rank}", disable=not is_main_process())

    for batch in iterator:
        input_data, y, external_prob = unpack_eval_batch_with_external_prob(batch)

        proteins = None
        if isinstance(input_data, torch.Tensor):
            X = input_data
        else:
            proteins, X = input_data

        X = X.to(device, non_blocking=True)
        X = X.long()
        y = y.to(device, non_blocking=True)
        if external_prob is not None:
            external_prob = external_prob.to(device, non_blocking=True)

        if proteins is not None and hasattr(primary_model, "set_proteins"):
            primary_model.set_proteins(proteins)
        if proteins is not None and teacher_model is not None and hasattr(teacher_model, "set_proteins"):
            teacher_model.set_proteins(proteins)

        with autocast(device_type="cuda", enabled=amp_enabled):
            try:
                out_primary = primary_model(
                    X,
                    permute_dims=permute_dims,
                    external_prob=external_prob,
                )
            except TypeError:
                out_primary = primary_model(X, permute_dims=permute_dims)

            if isinstance(out_primary, dict):
                logits_primary = out_primary["logits"]
            elif isinstance(out_primary, (tuple, list)):
                logits_primary = out_primary[0]
            else:
                logits_primary = out_primary

            prob = torch.sigmoid(logits_primary.float()) * float(primary_weight)

            if teacher_model is not None and float(teacher_weight) > 0:
                out_teacher = teacher_model(X, permute_dims=permute_dims)
                if isinstance(out_teacher, dict):
                    logits_teacher = out_teacher["logits"]
                elif isinstance(out_teacher, (tuple, list)):
                    logits_teacher = out_teacher[0]
                else:
                    logits_teacher = out_teacher
                prob = prob + torch.sigmoid(logits_teacher.float()) * float(teacher_weight)

        if y.ndim != 2 or y.shape[1] != num_classes:
            raise ValueError(f"target shape mismatch: y={tuple(y.shape)}, num_classes={num_classes}")
        if prob.ndim != 2 or prob.shape[1] != num_classes:
            raise ValueError(f"pred shape mismatch: pred={tuple(prob.shape)}, num_classes={num_classes}")

        all_targs.append(y.detach().cpu())
        all_preds.append(prob.detach().cpu())

    if not all_targs:
        return torch.empty((0, num_classes)), torch.empty((0, num_classes))

    return torch.cat(all_targs, dim=0), torch.cat(all_preds, dim=0)


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def evaluate_one_task(args: argparse.Namespace):
    args.task = normalize_task(args.task)
    set_seed(args.seed)

    device = init_distributed_mode(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opt = build_eval_opt_from_config(args)

    if is_main_process():
        print(f"[EvalWeakQuery] task={args.task}, mode={args.mode}")
        print(f"[Device] {device}")
        print(f"[DDP] distributed={args.distributed}, rank={args.rank}, world_size={args.world_size}")
        print(f"[Checkpoint] {args.ckpt}")
        print(f"[Use query decoder] {args.use_query_decoder}")
        save_json(vars(args), output_dir / "eval_args.json")
        save_json(vars(opt), output_dir / "eval_merged_model_opt.json")

    dataset = build_msa_dataset(
        opt=opt,
        mode=args.mode,
        task=args.task,
        need_proteins=args.need_proteins,
    )

    external_prob_path = clean_optional_path(getattr(args, "external_prob_path", None))
    args.external_prob_path = external_prob_path
    if external_prob_path is not None:
        dataset = ExternalProbEvalDataset(
            base_dataset=dataset,
            external_prob_path=external_prob_path,
            num_classes=args.num_classes,
            external_prob_protein_ids=clean_optional_path(getattr(args, "external_prob_protein_ids", None)),
        )
        if is_main_process():
            print(
                "[ExternalProb] enabled: "
                f"path={external_prob_path}, shape={dataset.prob_shape}, dtype={dataset.prob_dtype}, "
                f"topk_source={getattr(args, 'query_decoder_topk_source', 'base')}"
            )

    loader = make_eval_loader(
        dataset=dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        rank=args.rank,
        world_size=args.world_size,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    if is_main_process():
        print(f"[Data] split={args.mode}, samples={len(dataset)}, local_batches={len(loader)}")

    # Primary model: query decoder or plain Arch.
    if bool(args.use_query_decoder):
        primary_model = build_query_model(opt, args, device)
        primary_role = args.checkpoint_role
    else:
        primary_model = Arch(opt)
        load_arch_checkpoint(
            model=primary_model,
            ckpt_path=args.ckpt,
            strict_shape=args.strict_shape,
        )
        primary_model = primary_model.to(device)
        primary_model.eval()
        primary_role = args.checkpoint_role

    if args.torch_compile:
        primary_model = torch.compile(primary_model)

    models = [primary_model]
    model_weights = [1.0]
    normalized_model_weights = [1.0]

    teacher_ckpt = clean_optional_path(args.teacher_ckpt)

    if teacher_ckpt is not None:
        teacher_model = Arch(opt)
        load_arch_checkpoint(
            model=teacher_model,
            ckpt_path=teacher_ckpt,
            strict_shape=args.strict_shape,
        )
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

        if args.torch_compile:
            teacher_model = torch.compile(teacher_model)

        models = [primary_model, teacher_model]
        model_weights = [
            float(args.ensemble_primary_weight),
            float(args.ensemble_teacher_weight),
        ]

        # Reuse private normalizer from eval_ind_test indirectly by simple local implementation.
        s = sum(float(w) for w in model_weights)
        if s <= 0:
            raise ValueError(f"Invalid ensemble weights: {model_weights}")
        normalized_model_weights = [float(w) / s for w in model_weights]

        if is_main_process():
            print(
                "[Ensemble] probability average enabled: "
                f"{primary_role}_weight={normalized_model_weights[0]:.6f}, "
                f"teacher_weight={normalized_model_weights[1]:.6f}"
            )
    else:
        if is_main_process():
            print(f"[Ensemble] disabled; using {primary_role} checkpoint only.")

    teacher_model_for_predict = models[1] if len(models) > 1 else None
    local_targs, local_preds = predict_dataset_with_optional_external_prob(
        primary_model=primary_model,
        teacher_model=teacher_model_for_predict,
        primary_weight=float(normalized_model_weights[0]),
        teacher_weight=(float(normalized_model_weights[1]) if len(normalized_model_weights) > 1 else 0.0),
        loader=loader,
        device=device,
        num_classes=args.num_classes,
        permute_dims=tuple(args.permute_dims),
        no_amp=args.no_amp,
        need_proteins=args.need_proteins,
    )

    if is_main_process():
        print(f"[Predict] rank0 local preds: {tuple(local_preds.shape)}")

    full_targs, full_preds = collect_predictions(
        local_targs=local_targs,
        local_preds=local_preds,
        output_dir=output_dir,
        mode=args.mode,
        method=args.distributed_collect,
        keep_part_files=args.keep_part_files,
    )

    result = None

    if is_main_process():
        if full_targs is None or full_preds is None:
            raise RuntimeError("Rank0 did not receive full predictions.")

        if full_targs.shape[0] != len(dataset):
            print(
                "[Warning] collected samples != dataset length: "
                f"collected={full_targs.shape[0]}, dataset={len(dataset)}"
            )

        metrics = compute_evalperf_metrics(
            targs=full_targs,
            preds=full_preds,
            report_threshold=args.report_threshold,
            no_empty_labels=args.no_empty_labels,
            no_zero_classes=args.no_zero_classes,
        )

        ensemble_enabled = teacher_ckpt is not None

        result = {
            "task": args.task,
            "mode": args.mode,
            "checkpoint": str(Path(args.ckpt)),
            "checkpoint_role": str(primary_role),
            "use_query_decoder": bool(args.use_query_decoder),
            "primary_checkpoint": str(Path(args.ckpt)),
            "teacher_checkpoint": str(Path(teacher_ckpt)) if ensemble_enabled else None,
            "ensemble_enabled": bool(ensemble_enabled),
            "ensemble_average_space": "probability",
            "ensemble_weights": (
                {
                    str(primary_role): float(normalized_model_weights[0]),
                    "teacher": float(normalized_model_weights[1]),
                }
                if ensemble_enabled
                else {
                    str(primary_role): 1.0,
                }
            ),
            "query_decoder_topk": int(args.query_decoder_topk) if args.use_query_decoder else None,
            "query_decoder_topk_source": str(getattr(args, "query_decoder_topk_source", "base")) if args.use_query_decoder else None,
            "external_prob_path": str(args.external_prob_path) if args.external_prob_path is not None else None,
            "external_prob_blend_alpha": float(getattr(args, "external_prob_blend_alpha", 1.0)),
            "query_decoder_mode": str(args.query_decoder_mode) if args.use_query_decoder else None,
            "query_decoder_memory_mode": str(args.query_decoder_memory_mode) if args.use_query_decoder else None,
            "query_decoder_memory_grid_h": int(args.query_decoder_memory_grid_h) if args.use_query_decoder else None,
            "query_decoder_memory_grid_w": int(args.query_decoder_memory_grid_w) if args.use_query_decoder else None,
            "num_samples": int(full_targs.shape[0]),
            "num_classes": int(full_targs.shape[1]),
            "metrics_are_percent": True,
            "no_empty_labels": bool(args.no_empty_labels),
            "no_zero_classes": bool(args.no_zero_classes),
            **metrics,
        }

        save_json(result, output_dir / f"{args.mode}_metrics.json")

        if args.save_predictions:
            torch.save(
                {
                    "targs": full_targs,
                    "preds": full_preds,
                    "result": result,
                },
                output_dir / f"{args.mode}_predictions.pt",
            )

        print("=" * 80)
        print("[Result]")
        for k, v in result.items():
            print(f"{k}: {v}")
        print("=" * 80)

    if dist_is_initialized():
        dist.barrier()

    cleanup_distributed()
    return result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="Evaluate weak query-decoder MSA-GO checkpoint on ind_test/test split."
    )

    # Required
    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--file_address", type=str, required=True)
    p.add_argument("--working_address", type=str, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # Optional teacher ensemble
    p.add_argument("--teacher_ckpt", type=str, default=None)
    p.add_argument("--ensemble_primary_weight", type=float, default=1.0)
    p.add_argument("--ensemble_teacher_weight", type=float, default=1.0)
    p.add_argument("--checkpoint_role", type=str, default="weak_query")

    # Split
    p.add_argument("--mode", type=str, default="ind_test")

    # Dataset/model compatibility
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--msa_max_size", type=int, default=None)
    p.add_argument("--permute_dims", type=int, nargs=4, default=[0, 3, 2, 1])
    p.add_argument("--torch_compile", action="store_true")

    # Binary MSA loader
    p.add_argument("--msa_read_mode", type=str, choices=["full", "rows", "block"], default="full")
    p.add_argument("--msa_sample_strategy", type=str, choices=["random", "block", "head"], default="head")
    p.add_argument("--msa_shuffle_rows_at_getitem", action="store_true")
    p.add_argument("--no_msa_shuffle_rows_at_getitem", dest="msa_shuffle_rows_at_getitem", action="store_false")
    p.set_defaults(msa_shuffle_rows_at_getitem=False)
    p.add_argument("--msa_cache_gb", type=float, default=0.0)
    p.add_argument("--msa_max_open_files", type=int, default=256)
    p.add_argument("--sample_seed", type=int, default=1)

    # Loader
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    p.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    p.set_defaults(persistent_workers=True)

    # GPU / DDP
    p.add_argument("--gpu_ids", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--no_amp", action="store_true")

    # Model-specific optional behavior
    p.add_argument("--need_proteins", action="store_true")

    # Checkpoint loading
    p.add_argument("--strict_shape", dest="strict_shape", action="store_true")
    p.add_argument("--no_strict_shape", dest="strict_shape", action="store_false")
    p.set_defaults(strict_shape=True)
    p.add_argument("--strict_query_decoder", dest="strict_query_decoder", action="store_true")
    p.add_argument("--no_strict_query_decoder", dest="strict_query_decoder", action="store_false")
    p.set_defaults(strict_query_decoder=True)

    # Query decoder
    p.add_argument("--use_query_decoder", dest="use_query_decoder", action="store_true")
    p.add_argument("--no_use_query_decoder", dest="use_query_decoder", action="store_false")
    p.set_defaults(use_query_decoder=True)
    p.add_argument("--query_decoder_topk", type=int, default=50)
    p.add_argument(
        "--query_decoder_topk_source",
        type=str,
        default="base",
        choices=["base", "external_prob", "external_or_base", "blend"],
        help="Source used to select top-k ontology queries at evaluation time.",
    )
    p.add_argument("--external_prob_path", type=str, default=None)
    p.add_argument("--external_prob_protein_ids", type=str, default=None)
    p.add_argument(
        "--external_prob_blend_alpha",
        type=float,
        default=1.0,
        help="For topk_source='blend': score = alpha*external_prob + (1-alpha)*base_prob.",
    )
    p.add_argument("--query_decoder_mode", type=str, default="residual", choices=["residual", "replace"])
    p.add_argument("--query_decoder_dim", type=int, default=256)
    p.add_argument("--query_decoder_heads", type=int, default=8)
    p.add_argument("--query_decoder_layers", type=int, default=1)
    p.add_argument("--query_decoder_ffn_dim", type=int, default=1024)
    p.add_argument("--query_decoder_dropout", type=float, default=0.1)
    p.add_argument("--query_decoder_detach_query_weight", dest="query_decoder_detach_query_weight", action="store_true")
    p.add_argument("--no_query_decoder_detach_query_weight", dest="query_decoder_detach_query_weight", action="store_false")
    p.set_defaults(query_decoder_detach_query_weight=True)
    p.add_argument("--query_decoder_memory_mode", type=str, default="tokens_plus_pooled", choices=["pooled", "tokens", "tokens_plus_pooled"])
    p.add_argument("--query_decoder_memory_grid_h", type=int, default=0)
    p.add_argument("--query_decoder_memory_grid_w", type=int, default=0)

    # Metrics
    p.add_argument("--report_threshold", dest="report_threshold", action="store_true")
    p.add_argument("--no_report_threshold", dest="report_threshold", action="store_false")
    p.set_defaults(report_threshold=True)
    p.add_argument("--no_empty_labels", action="store_true")
    p.add_argument("--no_zero_classes", action="store_true")

    # Distributed collection
    p.add_argument("--distributed_collect", type=str, choices=["file", "all_gather_object"], default="file")
    p.add_argument("--keep_part_files", action="store_true")

    # Output
    p.add_argument("--save_predictions", action="store_true")

    return p


def main():
    parser = build_argparser()
    args, unknown = parser.parse_known_args()
    unknown = [x for x in unknown if x != "--ddp-child"]
    if unknown:
        print(f"[Warning] Ignoring unknown arguments: {unknown}")
    evaluate_one_task(args)


if __name__ == "__main__":
    main()
