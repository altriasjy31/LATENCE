"""Scratch-trained shared GO scoring with an explicit raw-geometry bypass.

Both ontology and ontology_raw controls share the same final fusion LayerNorm.
The bypass uses center + log(abs(offset)), without replacing the ontology path.
Optional fixed random Fourier features are computed from Stage1 x; they are
available for new proteins and are not trainable per-protein ID embeddings.
They add no independent biological information and do not undo Stage1 exposure.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from .full_task_model_v090 import FullTaskGraphModelV090, FullTaskModelConfigV090


@dataclass
class FullTaskModelConfigV092(FullTaskModelConfigV090):
    go_input_mode: str = 'ontology_raw'
    raw_geometry_width: int = 512
    protein_random_features: str = 'none'
    random_feature_dim: int = 128
    random_feature_scale: float = .1
    random_feature_seed: int = 1092


class FixedRandomFeatureProjectionV092(nn.Module):
    """Original trainable projection + a shared random-feature adapter.

    P and phase are frozen seeded buffers. The same input x produces the same
    random features in any batch or protein role. No registry IDs are inputs.
    """
    def __init__(self, stage1_projection, *, feature_dim, hidden_dim, random_dim, seed, scale):
        super().__init__()
        self.stage1 = stage1_projection
        self.scale = float(scale)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.register_buffer('random_matrix', torch.randn(feature_dim,random_dim//2,generator=generator)/math.sqrt(feature_dim))
        self.register_buffer('random_phase', torch.rand(random_dim//2,generator=generator)*(2*math.pi))
        self.random_projection = nn.Linear(random_dim, hidden_dim, bias=False)
        nn.init.normal_(self.random_projection.weight,std=.01)
        self.feature_dim = feature_dim

    def random_features(self, x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            normalized = F.layer_norm(x.float(),(self.feature_dim,))
            angles = normalized @ self.random_matrix.float() + self.random_phase.float()
            # The cos/sin coordinates have bounded magnitude independent of x.
            return torch.cat((angles.cos(),angles.sin()),dim=-1)

    def forward(self,x):
        with torch.autocast(device_type=x.device.type,enabled=False):
            return self.stage1(x.float()) + self.scale*self.random_projection(self.random_features(x))


class FullTaskGraphModelV092(FullTaskGraphModelV090):
    VERSION = '0.9.2'

    def __init__(self, protein_dim, ontology, config=None, *, seen_task_mask=None):
        cfg = config or FullTaskModelConfigV092()
        if cfg.go_input_mode not in ('ontology','ontology_raw'):
            raise ValueError('go_input_mode must be ontology or ontology_raw')
        if type(cfg.raw_geometry_width) is not int or cfg.raw_geometry_width < 1:
            raise ValueError('raw_geometry_width must be positive integer')
        if cfg.protein_random_features not in ('none','rff'):
            raise ValueError('protein_random_features must be none or rff')
        if type(cfg.random_feature_dim) is not int or cfg.random_feature_dim < 2 or cfg.random_feature_dim % 2:
            raise ValueError('random_feature_dim must be a positive even integer')
        if type(cfg.random_feature_seed) is not int or cfg.random_feature_seed < 0:
            raise ValueError('random_feature_seed must be a nonnegative integer')
        if (isinstance(cfg.random_feature_scale,bool) or not isinstance(cfg.random_feature_scale,(int,float))
                or not math.isfinite(cfg.random_feature_scale) or cfg.random_feature_scale < 0):
            raise ValueError('random_feature_scale must be finite nonnegative')
        super().__init__(protein_dim,ontology,cfg,seen_task_mask=seen_task_mask)
        self.go_fusion_norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.go_input_mode == 'ontology_raw':
            # Statistics depend only on supplied ontology geometry, never GO
            # frequencies or any core/weak/independent-test labels.
            with torch.no_grad():
                raw = torch.cat((self.ontology_encoder.box_center,self.ontology_encoder.box_log_offset),dim=-1)
                self.register_buffer('raw_geometry_mean',raw.mean(0))
                self.register_buffer('raw_geometry_scale',raw.std(0,unbiased=False).clamp_min(1e-6))
            self.raw_geometry_encoder = nn.Sequential(
                nn.Linear(raw.shape[1],cfg.raw_geometry_width),nn.SiLU(),
                nn.Linear(cfg.raw_geometry_width,cfg.hidden_dim))
        if cfg.protein_random_features == 'rff':
            self.node_encoder = FixedRandomFeatureProjectionV092(self.node_encoder,
                feature_dim=protein_dim,hidden_dim=cfg.hidden_dim,random_dim=cfg.random_feature_dim,
                seed=cfg.random_feature_seed,scale=cfg.random_feature_scale)

    def encode_ontology(self):
        encoded = super().encode_ontology()
        with torch.autocast(device_type=encoded.device.type,enabled=False):
            if self.config.go_input_mode == 'ontology_raw':
                raw = torch.cat((self.ontology_encoder.box_center,self.ontology_encoder.box_log_offset),dim=-1).float()
                bypass = self.raw_geometry_encoder((raw-self.raw_geometry_mean)/self.raw_geometry_scale)
                encoded = encoded.float() + bypass
            return self.go_fusion_norm(encoded.float())

    def representation_description(self):
        return {'go_input_mode':self.config.go_input_mode,
            'raw_geometry': 'center_plus_log_abs_offset' if self.config.go_input_mode=='ontology_raw' else None,
            'raw_statistics_scope':'supplied_ontology_geometry_only',
            'protein_random_features':self.config.protein_random_features,
            'random_feature_scope':'fixed_seed_mapping_of_Stage1_x; no trainable protein ID table',
            'stage1_prior_erased':False,'strict_zero_shot_provenance_established':False}
