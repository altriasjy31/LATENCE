"""Three-block heterogeneous GraphSAGE with a full-task absolute classifier.

The training skeleton follows tunedGNN large_graph/models.py at 23f9604:
SAGE -> previous RAW layer residual -> BN -> ReLU -> dropout. The first
layer has no raw residual. Relation-specific means, node-type self maps and
node-type local BN adapt that skeleton to the NBS heterogeneous graph.

Only sparse GO evidence enters the initial node states. There is no backbone
skip, dense teacher input, GO-query decoder, exact vote or candidate-pair head.
The fixed-column classifier is not an open-vocabulary/zero-shot GO model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .full_task_model import mean_adjacency


PP_RELATIONS_V088 = ("ppi", "similar_to", "weak_to_core", "core_to_weak", "cosine")
NODE_TYPES_V088 = ("core", "weak", "query")


@dataclass
class FullTaskModelConfigV088:
    encoder_variant: str = "hetero_tuned"
    prediction_mode: str = "direct"
    hidden_dim: int = 320
    sage_layers: int = 3
    ontology_layers: int = 2
    input_dropout: float = 0.1
    dropout: float = 0.25
    ontology_dropout: float = 0.0
    bn_momentum: float = 0.1


class OntologyEncoderV088(nn.Module):
    """Shared geometry/ontology encoding; no trainable per-GO embedding table."""
    def __init__(self, ontology: Mapping[str, object], hidden_dim: int,
                 layers: int, dropout: float):
        super().__init__()
        center = torch.as_tensor(ontology["center"]).float()
        offset = torch.as_tensor(ontology["offset"]).float()
        mapping = torch.as_tensor(ontology["task_to_ontology"]).long()
        if center.ndim != 2 or center.shape != offset.shape or not len(center):
            raise ValueError("ontology center/offset must be matching nonempty matrices")
        if not torch.isfinite(center).all() or not torch.isfinite(offset).all():
            raise ValueError("ontology geometry must be finite")
        if (mapping.ndim != 1 or not mapping.numel() or mapping.min() < 0 or
                mapping.max() >= len(center)):
            raise ValueError("task_to_ontology must be a nonempty valid vector")
        self.register_buffer("box_center", center, persistent=False)
        self.register_buffer("box_log_offset", offset.abs().clamp_min(1e-8).log(), persistent=False)
        self.register_buffer("task_to_ontology", mapping, persistent=False)
        self.center_encoder = nn.Sequential(nn.LayerNorm(center.shape[1]),
                                            nn.Linear(center.shape[1], hidden_dim), nn.SiLU())
        self.offset_encoder = nn.Sequential(nn.LayerNorm(center.shape[1]),
                                            nn.Linear(center.shape[1], hidden_dim), nn.SiLU())
        self.norm = nn.LayerNorm(hidden_dim)
        self.relations = tuple(sorted(ontology["edges"]))
        for name in self.relations:
            if not name.replace("_", "").isalnum():
                raise ValueError("ontology relation names must be simple identifiers")
            self.register_buffer("adj_" + name,
                mean_adjacency(ontology["edges"][name], len(center)), persistent=False)
        self.layers = nn.ModuleList([nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for name in self.relations}) for _ in range(layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self):
        h = self.norm(self.center_encoder(self.box_center) + self.offset_encoder(self.box_log_offset))
        for layer, norm in zip(self.layers, self.layer_norms):
            if not self.relations:
                continue
            with torch.autocast(device_type=h.device.type, enabled=False):
                x = norm(h.float())
                messages = [torch.sparse.mm(getattr(self, "adj_" + name), layer[name](x))
                            for name in self.relations]
                h = h.float() + self.dropout(F.silu(sum(messages) / len(messages)))
        return h[self.task_to_ontology]


class TypedRelationSAGEV088(nn.Module):
    """One typed self transform plus the SUM of relation-specific means.

    Positive confidence scales each incoming message; the denominator counts
    positive edges, so a weak-confidence edge cannot renormalize to certainty.
    With one relation and unit confidence this is ordinary mean GraphSAGE.
    ``raw_skip`` is the previous layer's pre-BN tensor in this source order.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                          for _ in NODE_TYPES_V088])
        self.neighbor_linears = nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim, bias=False)
                                               for name in PP_RELATIONS_V088})

    def forward(self, states: Tensor, dst_in_src: Tensor, dst_type: Tensor,
                edge_index: Tensor, edge_type: Tensor, edge_attr: Tensor,
                raw_skip: Tensor | None = None):
        with torch.autocast(device_type=states.device.type, enabled=False):
            states = states.float()
            destination = states[dst_in_src]
            raw = torch.zeros_like(destination)
            # Empty calls intentionally preserve zero gradient connections for
            # every type/relation on DDP ranks with different node compositions.
            for code, linear in enumerate(self.self_linears):
                rows = torch.where(dst_type == code)[0]
                raw = raw.index_add(0, rows, linear(destination[rows]))
            source, receiver = edge_index.long()
            positive = edge_attr[:, 0] > 0
            for code, name in enumerate(PP_RELATIONS_V088):
                selected = (edge_type == code) & positive
                src, dst = source[selected], receiver[selected]
                confidence = edge_attr[selected, :1].float().clamp(0, 1)
                aggregate = states.new_zeros(len(destination), states.shape[1])
                aggregate.index_add_(0, dst, states[src] * confidence)
                count = torch.bincount(dst, minlength=len(destination)).float().clamp_min(1)
                # Projection after mean is equivalent to projecting neighbors
                # before aggregation because the relation map has no bias.
                raw = raw + self.neighbor_linears[name](aggregate / count[:, None])
            if raw_skip is not None:
                if raw_skip.shape != states.shape:
                    raise ValueError("raw residual must follow this block's source order")
                raw = raw + raw_skip.float()[dst_in_src]
            return raw


class TypedBatchNormV088(nn.Module):
    """Rank-local per-type BN; runner synchronizes buffers before evaluation.

    No SyncBatchNorm collective is used because some ranks lack a node type.
    Singleton types use running statistics without updating those statistics.
    """
    def __init__(self, hidden_dim: int, momentum: float = .1):
        super().__init__()
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim, momentum=momentum)
                                   for _ in NODE_TYPES_V088])

    def forward(self, values: Tensor, node_type: Tensor):
        with torch.autocast(device_type=values.device.type, enabled=False):
            values = values.float()
            result = torch.zeros_like(values)
            for code, norm in enumerate(self.norms):
                rows = torch.where(node_type == code)[0]
                x = values[rows]
                if not len(rows):
                    output = x * norm.weight + norm.bias
                elif self.training and len(rows) == 1:
                    output = F.batch_norm(x, norm.running_mean, norm.running_var,
                        norm.weight, norm.bias, training=False, momentum=0., eps=norm.eps)
                else:
                    output = norm(x)
                result = result.index_add(0, rows, output)
            return result


class FullTaskGraphModelV088(nn.Module):
    VERSION = "0.8.8"
    PP_RELATIONS = PP_RELATIONS_V088
    NODE_TYPES = NODE_TYPES_V088

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfigV088 | None = None):
        super().__init__()
        self.config = config or FullTaskModelConfigV088()
        cfg = self.config
        if cfg.encoder_variant != "hetero_tuned" or cfg.prediction_mode != "direct":
            raise ValueError("v088 implements only hetero_tuned direct prediction")
        for name in ("hidden_dim", "sage_layers", "ontology_layers"):
            value = getattr(cfg, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "ontology_layers" else 1):
                raise ValueError("model dimensions/layer counts must be valid integers")
        for name in ("input_dropout", "dropout", "ontology_dropout"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError("dropout probabilities must be finite and in [0,1)")
        if not math.isfinite(cfg.bn_momentum) or not 0 < cfg.bn_momentum <= 1:
            raise ValueError("bn_momentum must be finite and in (0,1]")
        if protein_dim < 1:
            raise ValueError("protein feature dimension must be positive")
        self.protein_dim = protein_dim
        self.ontology_encoder = OntologyEncoderV088(ontology, cfg.hidden_dim,
                                                     cfg.ontology_layers, cfg.ontology_dropout)
        self.num_task_go = len(self.task_to_ontology)
        self.node_encoder = nn.Linear(protein_dim, cfg.hidden_dim)
        self.evidence_norms = nn.ModuleDict({name: nn.LayerNorm(cfg.hidden_dim)
                                            for name in ("candidate", "gold", "pseudo")})
        self.evidence_linears = nn.ModuleDict({name: nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
                                              for name in self.evidence_norms})
        self.sage_layers = nn.ModuleList([TypedRelationSAGEV088(cfg.hidden_dim)
                                          for _ in range(cfg.sage_layers)])
        self.norms = nn.ModuleList([TypedBatchNormV088(cfg.hidden_dim, cfg.bn_momentum)
                                    for _ in self.sage_layers])
        self.input_drop = nn.Dropout(cfg.input_dropout)
        self.hidden_drop = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_dim, self.num_task_go)
        nn.init.normal_(self.classifier.weight, std=.01)
        nn.init.zeros_(self.classifier.bias)

    @property
    def task_to_ontology(self):
        return self.ontology_encoder.task_to_ontology

    def encode_go(self):
        return self.ontology_encoder()

    @torch.no_grad()
    def initialize_output_prior(self, probability: Tensor):
        probability = torch.as_tensor(probability, dtype=self.classifier.bias.dtype,
                                      device=self.classifier.bias.device)
        if (probability.shape != (self.num_task_go,) or not torch.isfinite(probability).all()
                or (probability <= 0).any() or (probability >= 1).any()):
            raise ValueError("output prior must have one finite probability strictly inside (0,1) per task GO")
        self.classifier.bias.copy_(torch.logit(probability))

    def set_encoder_step(self, step: int, rank: int = 0):
        # Compatibility with the runner; all v088 stochastic operations use the
        # checkpointed rank RNG, not a mutable evidence-mask/sampling context.
        if (isinstance(step, bool) or not isinstance(step, int) or step < 0 or
                isinstance(rank, bool) or not isinstance(rank, int) or rank < 0):
            raise ValueError("encoder step/rank must be nonnegative integers")

    def _go_mean(self, edge: Tensor, weights: Tensor, go: Tensor, n: int):
        if (edge.ndim != 2 or edge.shape[0] != 2 or weights.shape != (edge.shape[1],)
                or (edge.numel() and (edge[0].min() < 0 or edge[0].max() >= n
                    or edge[1].min() < 0 or edge[1].max() >= self.num_task_go))):
            raise ValueError("GO incidence must have valid local-node and task-GO indices")
        with torch.autocast(device_type=go.device.type, enabled=False):
            count = go.new_zeros(n, dtype=torch.float32)
            count.index_add_(0, edge[0], weights.float())
            values = weights.float() / count[edge[0]].clamp_min(1e-8)
            matrix = torch.sparse_coo_tensor(edge, values, (n, self.num_task_go),
                                             device=go.device).coalesce()
            return torch.sparse.mm(matrix, go.float()), count > 0

    def _initial_states(self, batch, go):
        x, types = batch["sampled_protein_x"], batch["sampled_node_type"].long()
        n = len(x)
        if x.ndim != 2 or x.shape[1] != self.protein_dim or types.shape != (n,):
            raise ValueError("sampled feature/type arrays must align")
        if (types < 0).any() or (types >= len(NODE_TYPES_V088)).any():
            raise ValueError("sampled node type must be core=0, weak=1 or query=2")
        candidate, attr = batch["sampled_candidate_go"].long(), batch["sampled_candidate_attr"]
        if candidate.ndim != 2 or candidate.shape[0] != n or attr.shape != (*candidate.shape, 3):
            raise ValueError("sampled candidate arrays must align with outer source nodes")
        if (candidate >= self.num_task_go).any() or not torch.isfinite(attr).all():
            raise ValueError("candidate GO IDs/attributes must be valid")
        valid = (candidate >= 0) & (attr[..., 0] > 0)
        rows = torch.arange(n, device=x.device)[:, None].expand_as(candidate)[valid]
        incidence = torch.stack((rows, candidate[valid]))
        evidence = {"candidate": self._go_mean(incidence, attr[..., 0][valid].clamp(0, 1), go, n)}
        for name, role in (("gold", 0), ("pseudo", 1)):
            edge = batch["sampled_" + name + "_edge"].long()
            mean, available = self._go_mean(edge, go.new_ones(edge.shape[1]), go, n)
            if edge.numel() and (types[edge[0]] != role).any():
                raise ValueError("gold/pseudo evidence must belong to support core/weak nodes, never query ghosts")
            evidence[name] = mean, available
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = self.node_encoder(x.float())
            for name, (mean, available) in evidence.items():
                message = self.evidence_linears[name](self.evidence_norms[name](mean.float()))
                h = h + message * available[:, None]
            return self.input_drop(F.relu(h)), types

    def forward(self, batch: Mapping[str, object], *, go_encoding: Tensor | None = None,
                return_details: bool = False):
        if go_encoding is not None and self.training:
            raise ValueError("GO encoding may only be cached in evaluation")
        go = self.encode_go() if go_encoding is None else go_encoding
        if go.shape != (self.num_task_go, self.config.hidden_dim):
            raise ValueError("GO encoding must match all immutable task columns")
        h, types = self._initial_states(batch, go)
        blocks = batch["blocks"]
        if len(blocks) != len(self.sage_layers):
            raise ValueError("one message-flow block is required per SAGE layer")
        previous_ids = None
        raw_last = None
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
                raise ValueError("dst_in_src must map every destination identity into source order")
            if (edge.ndim != 2 or edge.shape[0] != 2 or kind.shape != (edge.shape[1],)
                    or attr.shape != (edge.shape[1], 3) or not torch.isfinite(attr).all()
                    or (edge.numel() and (edge[0].min() < 0 or edge[0].max() >= len(src)
                        or edge[1].min() < 0 or edge[1].max() >= len(dst)))
                    or (kind.numel() and (kind.min() < 0 or kind.max() >= len(self.PP_RELATIONS)))):
                raise ValueError("message-flow block edge indices/types/attributes are invalid")
            types = types[mapping]
            raw = layer(h, mapping, types, edge, kind, attr, raw_skip=raw_last)
            h = self.hidden_drop(F.relu(norm(raw, types)))
            raw_last, previous_ids = raw, dst
            edge_count += edge.shape[1]
        if (types != 2).any():
            raise ValueError("final block must contain only query nodes in output order")
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = self.classifier(h.float())
        if return_details:
            return {"logits": logits, "model_diagnostics": {
                "sampled_nodes": logits.new_tensor(len(batch["sampled_protein_x"])),
                "sampled_edges": logits.new_tensor(edge_count),
                "query_count": logits.new_tensor(len(h)),
                "query_hidden_norm": h.detach().float().norm(dim=-1).mean()}}
        return logits
