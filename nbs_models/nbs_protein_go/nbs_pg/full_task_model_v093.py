"""Backbone-preserving shared correction and explicit unseen-GO inference.

The general shared head is trained separately by the v093 objective. Seen task
columns use backbone logits plus a zero-initialized, fully shared correction;
unseen task columns and ontology-only targets use the general head alone.
Geometry extension requires fixed coordinates for every old ontology node.
This interface does not generate BoxSquaredEL embeddings or establish strict
pipeline zero-shot provenance for reused Stage1 representations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .full_task_model import mean_adjacency
from .full_task_model_v090 import SharedProteinGOScorerV090
from .full_task_model_v092 import FullTaskGraphModelV092, FullTaskModelConfigV092


@dataclass
class FullTaskModelConfigV093(FullTaskModelConfigV092):
    """Retain the v092 encoder fields and declare the v093 output contract."""
    output_strategy: str = 'backbone_shared_residual'


class FullTaskGraphModelV093(FullTaskGraphModelV092):
    VERSION = '0.9.3'

    def __init__(self, protein_dim: int, ontology: Mapping[str, object],
                 config: FullTaskModelConfigV093 | None = None, *, seen_task_mask=None):
        cfg = config or FullTaskModelConfigV093()
        if cfg.output_strategy != 'backbone_shared_residual':
            raise ValueError('v093 requires backbone_shared_residual output_strategy')
        super().__init__(protein_dim, ontology, cfg, seen_task_mask=seen_task_mask)
        self.residual_scorer = SharedProteinGOScorerV090(cfg.hidden_dim, initial_bias=0.)
        # Zero the entire readout, not an upstream gate: first-step gradients
        # can train these shared parameters while output is exactly backbone.
        nn.init.zeros_(self.residual_scorer.protein_projection.weight)
        nn.init.zeros_(self.residual_scorer.label_bias.weight)
        nn.init.zeros_(self.residual_scorer.label_bias.bias)
        self._ontology_revision = 0

    def _base_columns_for_targets(self, rows: Tensor | None) -> Tensor:
        """Return source task columns, or -1 for every general-head target."""
        if rows is None:
            columns = torch.arange(self.num_task_go, device=self.task_to_ontology.device)
            return columns.masked_fill(~self.seen_task_mask, -1)
        lookup = torch.full((self.num_ontology_go,), self.num_task_go, dtype=torch.long,
                            device=self.task_to_ontology.device)
        # Canonical aliases can have different backbone columns. For the
        # canonical-row API use the first seen immutable task column, stable
        # across batches and input target order. Closed-task output retains
        # every alias's original backbone column.
        columns = torch.where(self.seen_task_mask)[0]
        lookup.scatter_reduce_(0, self.task_to_ontology[columns], columns,
                               reduce='amin', include_self=True)
        return lookup[rows].masked_fill(lookup[rows] == self.num_task_go, -1)

    def _aligned_base(self, batch, projected_shape, columns, device):
        result = torch.zeros(projected_shape, dtype=torch.float32, device=device)
        available = columns >= 0
        if not available.any():
            return result, available
        if 'base_logits' not in batch:
            raise ValueError('seen GO correction requires aligned base_logits [protein, task GO]')
        source = batch['base_logits']
        if (not torch.is_tensor(source) or source.shape != (projected_shape[0], self.num_task_go)
                or not source.is_floating_point()):
            raise ValueError('base_logits must cover all immutable task GO columns in query order')
        selected = source[:, columns[available]].detach().float().to(device)
        if not torch.isfinite(selected).all():
            raise ValueError('backbone logits for seen targets must be finite')
        result[:, available] = selected
        return result, available

    def forward(self, batch: Mapping[str, object], *, target_ontology_ids=None,
                go_encoding: Tensor | None = None, ontology_encoding: Tensor | None = None,
                go_chunk_size=None, return_details=False):
        rows = None if target_ontology_ids is None else self._target_rows(target_ontology_ids)
        task, full = self._resolve_go(go_encoding, ontology_encoding, require_full=rows is not None)
        proteins, edge_count = self._encode_proteins(batch, task)
        labels = task if rows is None else full[rows]
        chunk = (len(labels) if self.training else self.config.inference_go_chunk_size) \
            if go_chunk_size is None else go_chunk_size
        self._validate_chunk_size(chunk)
        general_projected = self.scorer.project_proteins(proteins)
        residual_projected = self.residual_scorer.project_proteins(proteins)
        shared = torch.cat([self.scorer.score_projected(general_projected, labels[start:start + chunk])
                            for start in range(0, len(labels), chunk)], dim=1)
        correction = torch.cat([self.residual_scorer.score_projected(residual_projected, labels[start:start + chunk])
                                for start in range(0, len(labels), chunk)], dim=1)
        source_columns = self._base_columns_for_targets(rows)
        base, available = self._aligned_base(batch, shared.shape, source_columns, shared.device)
        logits = torch.where(available[None], base + correction, shared)
        if not return_details:
            return logits
        return {'logits': logits, 'shared_logits': shared, 'base_logits': base,
                'residual_logits': correction, 'base_available_mask': available,
                'model_diagnostics': {
                    'sampled_nodes': logits.new_tensor(len(batch['sampled_protein_x'])),
                    'sampled_edges': logits.new_tensor(edge_count),
                    'query_count': logits.new_tensor(len(proteins)),
                    'query_hidden_norm': proteins.detach().float().norm(dim=-1).mean(),
                    'target_go_columns': logits.new_tensor(len(labels)),
                    'backbone_target_columns': available.sum().to(logits.dtype)}}

    @torch.no_grad()
    def replace_ontology_for_inference(self, updated_ontology: Mapping[str, object], *,
                                       old_ontology_rows=None):
        """Install new nodes/edges with explicit, exact old-coordinate anchors.

        old_ontology_rows[i] is the updated row of current ontology node i.
        Pass arange(old_N) explicitly for an append-only update. The registry
        caller must verify GO identities; numeric coordinates alone cannot
        identify semantic nodes. Geometry/adjacency buffers are nonpersistent;
        updated sources and their hashes must accompany prediction output.
        New relations need training and are rejected here. Changed relations
        between existing nodes can change their encodings even with fixed boxes.
        Cached ontology/task encodings must be recomputed after this call.
        """
        if self.training:
            raise ValueError('ontology replacement is inference-only; call eval() first')
        if old_ontology_rows is None:
            raise ValueError('old_ontology_rows is required to anchor every current ontology node')
        if not isinstance(updated_ontology, Mapping):
            raise ValueError('updated_ontology must be a geometry/relation mapping')
        encoder, device = self.ontology_encoder, self.task_to_ontology.device
        center = torch.as_tensor(updated_ontology['center'], device=device).float()
        # The established runner constructs models from CPU ontology tensors
        # and moves the resulting buffers to CUDA afterward. Reproduce that
        # FP32 CPU log transform before exact coordinate checks; recomputing
        # log on CUDA can differ by an ulp for identical source offsets.
        offset_cpu = torch.as_tensor(updated_ontology['offset']).detach().to('cpu', dtype=torch.float32)
        offset = offset_cpu.to(device)
        if (center.ndim != 2 or center.shape != offset.shape or not len(center)
                or center.shape[1] != encoder.box_center.shape[1]
                or not torch.isfinite(center).all() or not torch.isfinite(offset).all()):
            raise ValueError('updated finite geometry must preserve the training geometry width')
        raw_rows = torch.as_tensor(old_ontology_rows, device=device)
        if (raw_rows.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or raw_rows.shape != (self.num_ontology_go,)):
            raise ValueError('old_ontology_rows must be one integer updated row per current node')
        old_rows = raw_rows.long()
        if ((old_rows < 0).any() or (old_rows >= len(center)).any()
                or len(torch.unique(old_rows)) != len(old_rows)):
            raise ValueError('old ontology anchors must be distinct valid updated rows')
        log_offset = offset_cpu.abs().clamp_min(1e-8).log().to(device)
        if (not torch.equal(center[old_rows], encoder.box_center)
                or not torch.equal(log_offset[old_rows], encoder.box_log_offset)):
            raise ValueError('old geometry coordinates changed; provide anchored BoxSquaredEL geometry')
        mapping = torch.as_tensor(updated_ontology['task_to_ontology'], device=device)
        if (mapping.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or mapping.shape != self.task_to_ontology.shape
                or not torch.equal(mapping.long(), old_rows[self.task_to_ontology])):
            raise ValueError('updated ontology must preserve every immutable task GO identity')
        edges = updated_ontology['edges']
        if not isinstance(edges, Mapping) or tuple(sorted(edges)) != tuple(encoder.relations):
            raise ValueError('updated relation names must equal trained relation names')
        adjacency = {}
        for name, value in edges.items():
            edge = torch.as_tensor(value, device=device)
            if (edge.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                    or edge.ndim != 2 or edge.shape[0] != 2):
                raise ValueError(f'updated relation {name} requires integer [2,E] edges')
            adjacency[name] = mean_adjacency(edge.long(), len(center)).to(device)
        # Commit only after all validation succeeds. No learned parameter is
        # created, resized or overwritten, and raw normalization is unchanged.
        encoder.box_center = center.clone()
        encoder.box_log_offset = log_offset.clone()
        encoder.task_to_ontology = mapping.long().clone()
        for name, value in adjacency.items():
            setattr(encoder, 'adj_' + name, value)
        self._ontology_revision += 1
        return {'schema': 'nbs-v093-anchored-ontology-replacement-v1',
                'ontology_revision': self._ontology_revision,
                'num_ontology_go': self.num_ontology_go,
                'num_task_go': self.num_task_go,
                'old_coordinate_anchors_verified': True,
                'raw_geometry_statistics_refitted': False,
                'new_trainable_parameter_rows_added': False,
                'old_predictions_guaranteed_unchanged': False,
                'strict_pipeline_zero_shot_provenance_established': False}

    def representation_description(self):
        result = super().representation_description()
        result.update(output_strategy=self.config.output_strategy,
                      seen_GO='backbone_logit_plus_shared_zero_initialized_correction',
                      unseen_GO='shared_general_head_without_backbone',
                      dynamic_alias_rule='first_seen_immutable_task_column')
        return result
