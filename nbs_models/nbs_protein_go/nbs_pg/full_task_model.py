"""Protein-centred graph encoding and complete-task GO-query decoding.

Only raw BoxSquaredEL geometry and ontology adjacency are cached. Learned GO
states are recomputed on every training step. The *same* forward is used for
training proteins and external proteins; target annotations are never read.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class FullTaskModelConfig:
    hidden_dim: int = 128
    query_dim: int = 64
    ontology_layers: int = 2
    decoder_hidden: int = 32
    go_chunk: int = 512
    dropout: float = 0.0


def mean_adjacency(edge: Tensor, n: int) -> Tensor:
    """edge[0] sends a message to edge[1]; normalize incoming neighbours."""
    edge = edge.long()
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("ontology edges must be [2,E]")
    if edge.numel() and (edge.min() < 0 or edge.max() >= n):
        raise ValueError("ontology edge index is out of range")
    count = torch.bincount(edge[1], minlength=n).float().clamp_min(1)
    return torch.sparse_coo_tensor(
        edge.flip(0), count[edge[1]].reciprocal(), (n, n)
    ).coalesce()


class FullTaskGraphModel(nn.Module):
    """A heterogenous GO→weak and GO→core→target graph link predictor.

    Three query-conditioned paths (self, predicted weak–GO, core neighbourhood)
    share ontology queries. Pair evidence retains exact GO identity in addition
    to the neighbourhood embeddings. There is no learned per-class classifier
    matrix. A single zero-initialized final affine layer starts at Stage1 and
    permits gradients into the graph encoder after its first update.
    """
    VERSION = "0.8.0"
    SOURCE_NAMES = ("protein", "weak_go", "core_go")

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfig | None = None):
        super().__init__()
        self.config = config or FullTaskModelConfig()
        cfg = self.config
        if min(cfg.hidden_dim, cfg.query_dim, cfg.decoder_hidden, cfg.go_chunk) < 1:
            raise ValueError("model dimensions and go_chunk must be positive")
        if cfg.ontology_layers < 0:
            raise ValueError("ontology_layers must be nonnegative")
        center = torch.as_tensor(ontology["center"]).float()
        offset = torch.as_tensor(ontology["offset"]).float().abs().clamp_min(1e-8).log()
        mapping = torch.as_tensor(ontology["task_to_ontology"]).long()
        if center.ndim != 2 or center.shape != offset.shape:
            raise ValueError("aligned center/offset must be [ontology GO,box dimension]")
        if mapping.ndim != 1:
            raise ValueError("task GO mapping must be one-dimensional")
        # Canonical GO IDs and alt-ID classifier columns may share an ontology
        # row (align_boxsqel_to_go_registry explicitly supports this). Keep
        # EVERY task column in its original order: only geometry is shared;
        # base logits, evidence, targets and exported predictions remain in
        # the immutable task vocabulary. Never unique()/compress this mapping.
        if mapping.numel() == 0 or mapping.min() < 0 or mapping.max() >= len(center):
            raise ValueError("task GO mapping is outside ontology")
        self.register_buffer("box_center", center, persistent=False)
        self.register_buffer("box_log_offset", offset, persistent=False)
        self.register_buffer("task_to_ontology", mapping, persistent=False)
        self.num_task_go = len(mapping)
        edges = ontology["edges"]
        self.relations = tuple(sorted(edges))
        for name in self.relations:
            if not name.replace("_", "").isalnum():
                raise ValueError("use simple relation names, e.g. is_a")
            self.register_buffer("adj_" + name, mean_adjacency(edges[name], len(center)),
                                 persistent=False)
        h, q = cfg.hidden_dim, cfg.query_dim
        self.box_center_encoder = nn.Sequential(nn.LayerNorm(center.shape[1]),
                                               nn.Linear(center.shape[1], h), nn.SiLU())
        self.box_offset_encoder = nn.Sequential(nn.LayerNorm(center.shape[1]),
                                               nn.Linear(center.shape[1], h), nn.SiLU())
        self.go_norm = nn.LayerNorm(h)
        self.go_layers = nn.ModuleList([
            nn.ModuleDict({name: nn.Linear(h, h, bias=False) for name in self.relations})
            for _ in range(cfg.ontology_layers)
        ])
        self.go_layer_norms = nn.ModuleList([nn.LayerNorm(h) for _ in self.go_layers])
        self.protein_encoder = nn.Sequential(nn.LayerNorm(protein_dim),
                                            nn.Linear(protein_dim, h), nn.SiLU(), nn.LayerNorm(h))
        self.weak_edge_encoder = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))
        self.core_edge_encoder = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))
        self.weak_update = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.LayerNorm(h))
        self.anchor_update = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.LayerNorm(h))
        self.core_update = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.LayerNorm(h))
        self.query = nn.Linear(h, q, bias=False)
        self.path_keys = nn.ModuleList([nn.Linear(h, q, bias=False) for _ in range(3)])
        self.decoder = nn.Sequential(nn.Linear(11, cfg.decoder_hidden), nn.SiLU(),
                                     nn.Linear(cfg.decoder_hidden, 1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def encode_go(self) -> Tensor:
        """Full ontology remains shared; sparse multiplication avoids [E,H] tensors."""
        h = self.go_norm(self.box_center_encoder(self.box_center) +
                         self.box_offset_encoder(self.box_log_offset))
        for layer, norm in zip(self.go_layers, self.go_layer_norms):
            if not self.relations:
                continue
            x = norm(h)
            messages = []
            # sparse mm is explicitly FP32 on CPU and CUDA, including under AMP.
            with torch.autocast(device_type=h.device.type, enabled=False):
                for name in self.relations:
                    messages.append(torch.sparse.mm(getattr(self, "adj_" + name),
                                                    layer[name](x.float())))
            h = h + self.dropout(F.silu(sum(messages) / len(messages)))
        return h[self.task_to_ontology]

    @staticmethod
    def _weights(logits: Tensor, valid: Tensor) -> Tensor:
        # Defined even for entirely empty neighbourhoods.
        weights = torch.softmax(logits.float().masked_fill(~valid, -1e4), dim=-1) * valid
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)

    def encode_proteins(self, batch: Mapping[str, Tensor], go: Tensor, *,
                        use_weak_go: bool, use_core_go: bool):
        own = self.protein_encoder(batch["protein_x"])
        b, h = own.shape
        idx = batch["candidate_go"].long()
        valid = (idx >= 0) & (idx < self.num_task_go)
        if not use_weak_go:
            valid = torch.zeros_like(valid)
        attr = batch["candidate_attr"].float()
        evidence = torch.stack((attr[..., 0].clamp(0, 1), attr[..., 1].tanh(),
                                attr[..., 2].clamp(0, 1)), dim=-1)
        weight = self._weights(self.weak_edge_encoder(evidence).squeeze(-1) +
                               attr[..., 0].clamp_min(1e-5).log(), valid)
        weak_message = (go[idx.clamp(0, self.num_task_go - 1)] * weight[..., None]).sum(1)
        weak = self.weak_update(torch.cat((own, weak_message), -1))

        anchor_x = batch["anchor_x"]
        edge = batch["anchor_go_edge"].long()
        c = len(anchor_x)
        if c:
            degree = torch.bincount(edge[0], minlength=c).float().clamp_min(1)
            annotation = torch.sparse_coo_tensor(
                edge, degree[edge[0]].reciprocal(), (c, self.num_task_go), device=go.device
            ).coalesce()
            with torch.autocast(device_type=go.device.type, enabled=False):
                anchor_labels = torch.sparse.mm(annotation, go.float())
            anchor = self.anchor_update(torch.cat((self.protein_encoder(anchor_x), anchor_labels), -1))
        else:
            # A dummy is used only to make gathering padded indices well-defined.
            anchor = own.new_zeros(1, h)
        neighbor = batch["neighbor_index"].long()
        neighbor_valid = (neighbor >= 0) & (neighbor < c)
        if not use_core_go:
            neighbor_valid = torch.zeros_like(neighbor_valid)
        nattr = batch["neighbor_attr"].float()
        nweight = self._weights(self.core_edge_encoder(nattr).squeeze(-1) +
                                nattr[..., 0].clamp_min(1e-5).log(), neighbor_valid)
        safe_neighbor = neighbor.clamp(0, max(c - 1, 0))
        core_message = (anchor[safe_neighbor] * nweight[..., None]).sum(1)
        core = self.core_update(torch.cat((own, core_message), -1))

        # Exact identity evidence is retained alongside compressed graph states.
        candidate_pair = own.new_zeros(b, self.num_task_go, 4, dtype=torch.float32)
        rr = torch.arange(b, device=own.device)[:, None].expand_as(idx)[valid]
        candidate_pair[rr, idx[valid], 0] = 1
        candidate_pair[rr, idx[valid], 1:] = evidence[valid]
        core_vote = own.new_zeros(b, self.num_task_go, dtype=torch.float32)
        if c and edge.numel():
            incidence = own.new_zeros(b, c, dtype=torch.float32)
            incidence.scatter_add_(1, safe_neighbor, nweight)
            # Sparse incidence @ annotations, keeping all gold labels of anchors.
            binary_annotation = torch.sparse_coo_tensor(
                edge, torch.ones(edge.shape[1], device=go.device),
                (c, self.num_task_go), device=go.device
            ).coalesce()
            with torch.autocast(device_type=go.device.type, enabled=False):
                core_vote = torch.sparse.mm(binary_annotation.transpose(0, 1),
                                             incidence.t()).t().clamp(0, 1)
        path_valid = torch.stack((torch.ones(b, device=own.device, dtype=torch.bool),
                                  valid.any(-1), neighbor_valid.any(-1)), -1)
        paths = (own, weak, core)
        keys = torch.stack([F.normalize(proj(value).float(), dim=-1)
                            for proj, value in zip(self.path_keys, paths)], 1)
        return keys, path_valid, candidate_pair, core_vote

    def forward(self, batch: Mapping[str, Tensor], *, use_weak_go: bool = True,
                use_core_go: bool = True, go_encoding: Tensor | None = None,
                return_details: bool = False):
        # No access to targets, positive_mask, is_weak or protein annotation labels.
        if go_encoding is not None and self.training:
            raise ValueError("learned GO encodings may only be cached in eval mode")
        go = self.encode_go() if go_encoding is None else go_encoding
        keys, available, evidence, core_vote = self.encode_proteins(
            batch, go, use_weak_go=use_weak_go, use_core_go=use_core_go)
        query = F.normalize(self.query(go).float(), dim=-1)
        base = batch["base_logits"].float()
        if base.shape != (keys.shape[0], self.num_task_go):
            raise ValueError("base logits must cover ALL task GOs in registry order")
        chunks, routes = [], []
        for start in range(0, self.num_task_go, self.config.go_chunk):
            end = min(start + self.config.go_chunk, self.num_task_go)
            with torch.autocast(device_type=base.device.type, enabled=False):
                scores = torch.einsum("bph,gh->bgp", keys.float(), query[start:end].float())
                scores = scores * math.sqrt(self.config.query_dim)
                scores = scores.masked_fill(~available[:, None, :], 0)
                route = torch.softmax(scores.masked_fill(~available[:, None, :], -1e4), -1)
                vote = core_vote[:, start:end]
                features = torch.cat((
                    (base[:, start:end] / 4).tanh()[..., None], scores,
                    (route * scores).sum(-1, keepdim=True), evidence[:, start:end],
                    vote[..., None], (vote - base[:, start:end].sigmoid())[..., None],
                ), -1)
                delta = self.decoder(features.float()).squeeze(-1)
                chunks.append(base[:, start:end] + delta)
                if return_details:
                    routes.append(route)
        logits = torch.cat(chunks, 1)
        if return_details:
            return {"logits": logits, "delta": logits - base,
                    "routing": torch.cat(routes, 1), "core_vote": core_vote}
        return logits
