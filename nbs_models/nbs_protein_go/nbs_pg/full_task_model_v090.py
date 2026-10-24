"""Shared target-GO scoring over the unchanged v088 heterogeneous SAGE encoder.

The main configuration retains v089 encoder/dropout settings: raw residuals,
typed BN, last-layer output and no DropEdge. Optional JK sum and stochastic
edge dropout are independent research switches, not reported improvements.

Dynamic targets are full-ontology *row indices*, resolved by the caller's
versioned GO/alias registry. A target must have supplied finite geometry and
legal relation indices. No per-label trainable rows or label-frequency bias
exist. Availability of this interface does not establish strict zero-shot
provenance or biological predictive validity.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Integral
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .full_task_loss_v090 import validate_seen_task_mask
from .full_task_model_v088 import FullTaskGraphModelV088, FullTaskModelConfigV088


@dataclass
class FullTaskModelConfigV090(FullTaskModelConfigV088):
    prediction_mode: str = "shared_bilinear"
    jk_mode: str = "last"
    edge_dropout: float = 0.0
    scorer_initial_bias: float = -6.0
    inference_go_chunk_size: int = 1024


class SharedProteinGOScorerV090(nn.Module):
    """p W g / sqrt(H) + a(g), with parameters shared across every GO."""
    def __init__(self, hidden_dim: int, initial_bias: float = -6.0):
        super().__init__()
        self.protein_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.label_bias = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.protein_projection.weight, std=.01)
        nn.init.zeros_(self.label_bias.weight)
        nn.init.constant_(self.label_bias.bias, initial_bias)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, proteins: Tensor, labels: Tensor) -> Tensor:
        return self.score_projected(self.project_proteins(proteins), labels)

    def project_proteins(self, proteins: Tensor) -> Tensor:
        with torch.autocast(device_type=proteins.device.type, enabled=False):
            return self.protein_projection(proteins.float())

    def score_projected(self, projected_proteins: Tensor, labels: Tensor) -> Tensor:
        with torch.autocast(device_type=projected_proteins.device.type, enabled=False):
            return (projected_proteins.float() @ labels.float().T /
                    self.scale + self.label_bias(labels.float()).T)


class FullTaskGraphModelV090(FullTaskGraphModelV088):
    VERSION = "0.9.0"

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfigV090 | None = None, *, seen_task_mask=None):
        cfg = config or FullTaskModelConfigV090()
        if cfg.prediction_mode != "shared_bilinear":
            raise ValueError("v090 implements shared_bilinear prediction")
        if cfg.jk_mode not in ("last", "sum"):
            raise ValueError("jk_mode must be last or sum")
        if (isinstance(cfg.edge_dropout, bool) or not math.isfinite(cfg.edge_dropout)
                or not 0 <= cfg.edge_dropout < 1):
            raise ValueError("edge_dropout must be finite and in [0,1)")
        if isinstance(cfg.scorer_initial_bias, bool) or not math.isfinite(cfg.scorer_initial_bias):
            raise ValueError("scorer_initial_bias must be finite")
        self._validate_chunk_size(cfg.inference_go_chunk_size)
        raw_mapping = torch.as_tensor(ontology["task_to_ontology"])
        if raw_mapping.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("task_to_ontology must contain integer row indices")
        for name, value in ontology["edges"].items():
            edge = torch.as_tensor(value)
            if edge.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError(f"ontology relation {name} must contain integer row indices")
        legacy = {field.name: getattr(cfg, field.name) for field in fields(FullTaskModelConfigV088)}
        legacy["prediction_mode"] = "direct"
        super().__init__(protein_dim, ontology, FullTaskModelConfigV088(**legacy))
        del self.classifier
        self.config = cfg
        self.scorer = SharedProteinGOScorerV090(cfg.hidden_dim, cfg.scorer_initial_bias)
        seen = validate_seen_task_mask(seen_task_mask, self.task_to_ontology,
                                      num_task_go=self.num_task_go,
                                      device=self.task_to_ontology.device)
        self.register_buffer("seen_task_mask", seen.clone())

    @staticmethod
    def _validate_chunk_size(size):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError("GO chunk size must be a positive integer")

    @property
    def num_ontology_go(self):
        return len(self.ontology_encoder.box_center)

    def initialize_output_prior(self, probability):
        raise ValueError("v090 has no per-GO output prior; use the configured shared scorer bias")

    def encode_ontology(self) -> Tensor:
        """Encode every supplied legal ontology node with shared parameters."""
        encoder = self.ontology_encoder
        h = encoder.norm(encoder.center_encoder(encoder.box_center) +
                         encoder.offset_encoder(encoder.box_log_offset))
        for layer, norm in zip(encoder.layers, encoder.layer_norms):
            if not encoder.relations:
                continue
            with torch.autocast(device_type=h.device.type, enabled=False):
                x = norm(h.float())
                messages = [torch.sparse.mm(getattr(encoder, "adj_" + name), layer[name](x))
                            for name in encoder.relations]
                h = h.float() + encoder.dropout(F.silu(sum(messages) / len(messages)))
        return h

    def encode_go(self) -> Tensor:
        """Compatibility view: immutable closed-task columns, including aliases."""
        return self.encode_ontology()[self.task_to_ontology]

    def _check_cache(self, value, rows: int, name: str):
        if value is not None:
            if self.training:
                raise ValueError("GO encoding may only be cached in evaluation")
            if (value.shape != (rows, self.config.hidden_dim) or not value.is_floating_point()
                    or not torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite and match its GO row space")

    def _resolve_go(self, go_encoding=None, ontology_encoding=None, *, require_full=False):
        self._check_cache(go_encoding, self.num_task_go, "task GO encoding")
        self._check_cache(ontology_encoding, self.num_ontology_go, "full ontology encoding")
        full = ontology_encoding
        if full is None and (go_encoding is None or require_full):
            full = self.encode_ontology()
        task = full[self.task_to_ontology] if go_encoding is None else go_encoding
        if go_encoding is not None and ontology_encoding is not None:
            if not torch.equal(go_encoding, ontology_encoding[self.task_to_ontology]):
                raise ValueError("task and ontology caches use inconsistent row mappings")
        return task, full

    def _edge_go_indices(self, edge, name):
        if (edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype not in
                (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError(f"{name} evidence must have integer shape [2, edges]")
        if edge.numel() and ((edge[1] < 0).any() or (edge[1] >= self.num_task_go).any()):
            raise ValueError(f"{name} evidence GO index leaves immutable task columns")
        return edge[1].long()

    def sanitize_evidence(self, batch: Mapping[str, object], *, filter_targets=True):
        """Return a shallow copy with unseen evidence/targets explicitly removed.

        Source tensors are never mutated. This adapter does not audit upstream
        teacher/features or erase information previously learned by Stage1.
        """
        result = dict(batch)
        candidate = batch["sampled_candidate_go"]
        attr = batch["sampled_candidate_attr"]
        if (candidate.ndim != 2 or attr.shape != (*candidate.shape, 3)
                or candidate.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)):
            raise ValueError("candidate evidence must have aligned integer GO indices and attributes")
        valid = candidate >= 0
        if (candidate[valid] >= self.num_task_go).any():
            raise ValueError("candidate GO index leaves immutable task columns")
        seen = self.seen_task_mask.to(candidate.device)
        removed = valid & ~seen[candidate.clamp_min(0).long()]
        result["sampled_candidate_go"] = candidate.masked_fill(removed, -1)
        result["sampled_candidate_attr"] = attr.masked_fill(removed[..., None], 0)
        for name in ("gold", "pseudo"):
            key = "sampled_" + name + "_edge"
            edge = batch[key]
            go = self._edge_go_indices(edge, name)
            result[key] = edge[:, self.seen_task_mask.to(go.device)[go]]
        if filter_targets:
            for name in ("targets", "positive_mask"):
                if name in result:
                    target = result[name]
                    if target.ndim != 2 or target.shape[1] != self.num_task_go:
                        raise ValueError("target/positive_mask must retain all immutable task columns")
                    result[name] = target.masked_fill(~self.seen_task_mask.to(target.device)[None], 0)
        return result

    def _assert_seen_evidence(self, batch):
        candidate, attr = batch["sampled_candidate_go"], batch["sampled_candidate_attr"]
        if candidate.ndim != 2 or attr.shape != (*candidate.shape, 3):
            raise ValueError("candidate GO indices and attributes must align")
        if candidate.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("candidate GO indices must be integers")
        valid = candidate >= 0
        if (candidate[valid] >= self.num_task_go).any():
            raise ValueError("candidate GO index leaves immutable task columns")
        active = valid & (attr[..., 0] > 0)
        if (~self.seen_task_mask.to(candidate.device)[candidate[active].long()]).any():
            raise ValueError("unseen candidate evidence remains; sanitize evidence before use")
        for name in ("gold", "pseudo"):
            go = self._edge_go_indices(batch["sampled_" + name + "_edge"], name)
            if (~self.seen_task_mask.to(go.device)[go]).any():
                raise ValueError(f"unseen {name} evidence remains; sanitize evidence before use")

    def _encode_proteins(self, batch, task_go):
        self._assert_seen_evidence(batch)
        h, types = self._initial_states(batch, task_go)
        blocks = batch["blocks"]
        if len(blocks) != len(self.sage_layers):
            raise ValueError("one message-flow block is required per SAGE layer")
        previous_ids, raw_last, aligned = None, None, []
        edge_count = 0
        for block, layer, norm in zip(blocks, self.sage_layers, self.norms):
            src, dst = block["src_global_ids"].long(), block["dst_global_ids"].long()
            mapping = block["dst_in_src"].long()
            edge, kind, attr = block["edge_index"].long(), block["edge_type"].long(), block["edge_attr"]
            if src.ndim != 1 or dst.ndim != 1 or len(src) != len(h) or not len(dst):
                raise ValueError("block node axes must match current source states")
            if previous_ids is not None and not torch.equal(src, previous_ids):
                raise ValueError("previous block destinations must equal next block sources in order")
            if (mapping.shape != dst.shape or (mapping < 0).any() or (mapping >= len(src)).any()
                    or not torch.equal(src[mapping], dst)):
                raise ValueError("dst_in_src must map destination identities into source order")
            if (edge.ndim != 2 or edge.shape[0] != 2 or kind.shape != (edge.shape[1],)
                    or attr.shape != (edge.shape[1], 3) or not torch.isfinite(attr).all()
                    or (edge.numel() and (edge[0].min() < 0 or edge[0].max() >= len(src)
                        or edge[1].min() < 0 or edge[1].max() >= len(dst)))
                    or (kind.numel() and (kind.min() < 0 or kind.max() >= len(self.PP_RELATIONS)))):
                raise ValueError("message-flow block edge indices/types/attributes are invalid")
            used_attr = attr
            if self.training and self.config.edge_dropout > 0 and len(attr):
                kept = torch.rand(len(attr), device=attr.device) >= self.config.edge_dropout
                used_attr = torch.cat((attr[:, :1] * kept[:, None], attr[:, 1:]), dim=1)
            # Existing RAW residual follows the current source order unchanged.
            types = types[mapping]
            raw = layer(h, mapping, types, edge, kind, used_attr, raw_skip=raw_last)
            h = self.hidden_drop(F.relu(norm(raw, types)))
            if self.config.jk_mode == "sum":
                aligned = [value[mapping] for value in aligned]
                aligned.append(h)
            raw_last, previous_ids = raw, dst
            edge_count += edge.shape[1]
        if (types != 2).any():
            raise ValueError("final block must contain only query nodes in output order")
        if self.config.jk_mode == "sum":
            h = sum(aligned)
        return h, edge_count

    def encode_proteins(self, batch, *, go_encoding=None, ontology_encoding=None) -> Tensor:
        task, _ = self._resolve_go(go_encoding, ontology_encoding)
        return self._encode_proteins(batch, task)[0]

    def _target_rows(self, ids):
        rows = torch.as_tensor(ids, device=self.task_to_ontology.device)
        if (rows.ndim != 1 or not len(rows) or rows.dtype not in
                (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
                or (rows < 0).any() or (rows >= self.num_ontology_go).any()):
            raise ValueError("target_ontology_ids must be nonempty valid integer ontology rows")
        return rows.long()

    def forward(self, batch: Mapping[str, object], *, target_ontology_ids=None,
                go_encoding: Tensor | None = None, ontology_encoding: Tensor | None = None,
                go_chunk_size=None, return_details=False):
        rows = None if target_ontology_ids is None else self._target_rows(target_ontology_ids)
        task, full = self._resolve_go(go_encoding, ontology_encoding, require_full=rows is not None)
        proteins, edge_count = self._encode_proteins(batch, task)
        labels = task if rows is None else full[rows]
        # Training retains the full supervised output for BCE anyway: use one
        # GEMM by default. Chunking is an inference/explicit caller contract.
        chunk = (len(labels) if self.training else self.config.inference_go_chunk_size) \
            if go_chunk_size is None else go_chunk_size
        self._validate_chunk_size(chunk)
        projected = self.scorer.project_proteins(proteins)
        logits = torch.cat([self.scorer.score_projected(projected, labels[start:start + chunk])
                            for start in range(0, len(labels), chunk)], dim=1)
        if return_details:
            return {"logits": logits, "model_diagnostics": {
                "sampled_nodes": logits.new_tensor(len(batch["sampled_protein_x"])),
                "sampled_edges": logits.new_tensor(edge_count),
                "query_count": logits.new_tensor(len(proteins)),
                "query_hidden_norm": proteins.detach().float().norm(dim=-1).mean(),
                "target_go_columns": logits.new_tensor(len(labels))}}
        return logits

    def load_encoder_state_dict(self, state_dict):
        return migrate_encoder_state_dict(self, state_dict)


@torch.no_grad()
def migrate_encoder_state_dict(model: FullTaskGraphModelV090, state_dict: Mapping[str, Tensor]):
    """Copy matching v088/v089 encoder and BN state; never classifier/head state.

    Geometry/registry hashes and source checkpoint provenance are nonpersistent
    external resources and must be verified by the runner before migration.
    Optimizer, schedule and RNG migration are intentionally outside this API.
    """
    current = model.state_dict()
    expected = {name for name in current if not name.startswith("scorer.") and name != "seen_task_mask"}
    allowed_discard = {"classifier.weight", "classifier.bias"}
    missing, extra = expected - set(state_dict), set(state_dict) - expected - allowed_discard
    if missing or extra:
        raise ValueError(f"encoder migration key mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}")
    for name in sorted(expected):
        value = state_dict[name]
        if not isinstance(value, Tensor) or value.shape != current[name].shape or value.dtype != current[name].dtype:
            raise ValueError(f"encoder migration shape/dtype mismatch for {name}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"encoder migration nonfinite tensor: {name}")
    current.update({name: state_dict[name] for name in expected})
    model.load_state_dict(current, strict=True)
    return {"reused_encoder_keys": sorted(expected),
            "discarded_classifier_keys": sorted(set(state_dict) & allowed_discard),
            "scorer_reinitialized": True, "optimizer_reused": False}
