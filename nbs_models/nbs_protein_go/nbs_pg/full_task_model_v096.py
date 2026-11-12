"""v095 hetero tunedGNN with an additive frozen-ESM feature adapter."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

from .full_task_model_v095 import FullTaskGraphModelV095, FullTaskModelConfigV095
from .full_task_model_v088 import NODE_TYPES_V088


@dataclass
class FullTaskModelConfigV096(FullTaskModelConfigV095):
    esm_dim: int = 1024
    esm_dropout: float = .1
    esm_gate_init: float = 0.


class FullTaskGraphModelV096(FullTaskGraphModelV095):
    VERSION = '0.9.6'

    def __init__(self, protein_dim, ontology, config=None, *, seen_task_mask=None):
        cfg = config or FullTaskModelConfigV096()
        if type(cfg.esm_dim) is not int or cfg.esm_dim < 1:
            raise ValueError('esm_dim must be a positive integer')
        if (isinstance(cfg.esm_dropout, bool) or not math.isfinite(cfg.esm_dropout)
                or not 0 <= cfg.esm_dropout < 1):
            raise ValueError('esm_dropout must lie in [0,1)')
        if isinstance(cfg.esm_gate_init, bool) or not math.isfinite(cfg.esm_gate_init):
            raise ValueError('esm_gate_init must be finite')
        super().__init__(protein_dim, ontology, cfg, seen_task_mask=seen_task_mask)
        # Keep node_encoder and every v095 state key unchanged for auditable
        # weight migration. Row-wise LN uses no train/test population statistics.
        self.esm_norm = nn.LayerNorm(cfg.esm_dim)
        self.esm_projection = nn.Linear(cfg.esm_dim, cfg.hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.esm_projection.weight)
        self.esm_input_drop = nn.Dropout(cfg.esm_dropout)
        self.esm_gate = nn.Parameter(torch.tensor(float(cfg.esm_gate_init)))

    @staticmethod
    def is_esm_state_key(key):
        return key == 'esm_gate' or key.startswith(('esm_norm.', 'esm_projection.'))

    def load_v095_weights(self, source):
        """Exact old-key migration; unexpected/missing old keys are fatal."""
        current = self.state_dict()
        old_keys = {name for name in current if not self.is_esm_state_key(name)}
        if set(source) != old_keys:
            missing, extra = sorted(old_keys - set(source)), sorted(set(source) - old_keys)
            raise ValueError(f'v095 migration state keys differ; missing={missing}, extra={extra}')
        for name in old_keys:
            if current[name].shape != source[name].shape or current[name].dtype != source[name].dtype:
                raise ValueError('v095 migration state shape/dtype differs: ' + name)
        if not torch.equal(current['seen_task_mask'].cpu(), source['seen_task_mask'].cpu()):
            raise ValueError('v095 migration changes immutable seen GO mask')
        if float(self.esm_gate.detach()) != 0.:
            raise ValueError('v095 weight migration requires esm_gate_init=0 for exact initial preservation')
        merged = {name: source[name] if name in old_keys else value for name, value in current.items()}
        self.load_state_dict(merged, strict=True)
        return {'loaded_v095_keys': len(old_keys),
                'new_esm_keys': sorted(set(current) - old_keys),
                'initial_fusion_gate': 0.}

    def _initial_states(self, batch, go):
        """Original v088 evidence composition plus ESM before ReLU/dropout."""
        x, types = batch['sampled_protein_x'], batch['sampled_node_type'].long()
        n = len(x)
        if x.ndim != 2 or x.shape[1] != self.protein_dim or types.shape != (n,):
            raise ValueError('sampled feature/type arrays must align')
        if (types < 0).any() or (types >= len(NODE_TYPES_V088)).any():
            raise ValueError('sampled node type must be core=0, weak=1 or query=2')
        esm = batch.get('sampled_esm_x')
        if (not torch.is_tensor(esm) or esm.shape != (n, self.config.esm_dim)
                or not esm.is_floating_point() or esm.device != x.device
                or not torch.isfinite(esm).all()):
            raise ValueError('sampled_esm_x must cover every sampled node with finite ESM features')
        candidate, attr = batch['sampled_candidate_go'].long(), batch['sampled_candidate_attr']
        if candidate.ndim != 2 or candidate.shape[0] != n or attr.shape != (*candidate.shape, 3):
            raise ValueError('sampled candidate arrays must align with outer source nodes')
        if (candidate >= self.num_task_go).any() or not torch.isfinite(attr).all():
            raise ValueError('candidate GO IDs/attributes must be valid')
        valid = (candidate >= 0) & (attr[..., 0] > 0)
        rows = torch.arange(n, device=x.device)[:, None].expand_as(candidate)[valid]
        incidence = torch.stack((rows, candidate[valid]))
        evidence = {'candidate': self._go_mean(incidence, attr[..., 0][valid].clamp(0, 1), go, n)}
        for name, role in (('gold', 0), ('pseudo', 1)):
            edge = batch['sampled_' + name + '_edge'].long()
            mean, available = self._go_mean(edge, go.new_ones(edge.shape[1]), go, n)
            if edge.numel() and (types[edge[0]] != role).any():
                raise ValueError('gold/pseudo evidence must belong to support core/weak nodes, never query ghosts')
            evidence[name] = mean, available
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = self.node_encoder(x.float())
            esm_hidden = self.esm_projection(self.esm_input_drop(self.esm_norm(esm.detach().float())))
            h = h + self.esm_gate.tanh() * esm_hidden
            for name, (mean, available) in evidence.items():
                message = self.evidence_linears[name](self.evidence_norms[name](mean.float()))
                h = h + message * available[:, None]
            return self.input_drop(F.relu(h)), types

    def representation_description(self):
        result = super().representation_description()
        result.update(protein_esm_source='frozen_ESM2ResNet_output_layer.0_input',
                      protein_fusion='old_projection + tanh(gate) * Linear(Dropout(LayerNorm(ESM)))',
                      esm_gate=float(self.esm_gate.detach().cpu().tanh()),
                      esm_feature_dim=self.config.esm_dim,
                      esm_trunk_finetuned=False, esm_classifier_logits_used=False,
                      retrieval_space_changed=False)
        return result

    def replace_ontology_for_inference(self, updated_ontology, *, old_ontology_rows=None):
        result = super().replace_ontology_for_inference(updated_ontology, old_ontology_rows=old_ontology_rows)
        result['schema'] = 'nbs-v096-anchored-ontology-replacement-v1'
        return result
