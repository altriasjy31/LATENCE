"""Shared geometry decoder over a frozen v089 protein representation.

No trainable per-GO table exists. Supplying a new finite geometry row permits
scoring it, but does not establish strict zero-shot provenance or accuracy.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SharedGeometryHeadV091(nn.Module):
    VERSION = "0.9.1-shared-head-stage1"

    def __init__(self, geometry, hidden_dim=320, mlp_width=512, initial_bias=-6.0):
        super().__init__()
        if any(type(x) is not int or x < 1 for x in (hidden_dim, mlp_width)):
            raise ValueError("hidden_dim and mlp_width must be positive integers")
        if isinstance(initial_bias, bool) or not math.isfinite(initial_bias):
            raise ValueError("initial_bias must be finite")
        geometry = torch.as_tensor(geometry, dtype=torch.float32).detach().clone()
        if geometry.ndim != 2 or min(geometry.shape) < 1 or not torch.isfinite(geometry).all():
            raise ValueError("geometry must have finite nonempty [GO, geometry-feature] shape")
        self.hidden_dim = hidden_dim
        self.mlp_width = mlp_width
        self.geometry_dim = geometry.shape[1]
        # The supplied geometry is immutable cache data, never a trainable
        # label table. Checkpoints bind its cache-file hash separately.
        self.register_buffer("geometry", geometry, persistent=False)
        self.register_buffer("geometry_mean", geometry.mean(dim=0))
        self.register_buffer("geometry_scale", geometry.std(dim=0, unbiased=False).clamp_min(1e-6))
        self.decoder = nn.Sequential(nn.Linear(self.geometry_dim, mlp_width), nn.SiLU(),
                                     nn.Linear(mlp_width, hidden_dim + 1))
        nn.init.normal_(self.decoder[-1].weight, std=.001)
        nn.init.zeros_(self.decoder[-1].bias)
        with torch.no_grad():
            self.decoder[-1].weight[-1].zero_()
            self.decoder[-1].bias[-1] = float(initial_bias)

    def prototypes(self, geometry=None):
        values = self.geometry if geometry is None else torch.as_tensor(
            geometry, dtype=torch.float32, device=self.geometry_mean.device)
        if values.ndim != 2 or values.shape[1] != self.geometry_dim or not len(values):
            raise ValueError("geometry rows must match the cached geometry feature dimension")
        if not torch.isfinite(values).all():
            raise ValueError("geometry rows must be finite")
        with torch.autocast(device_type=values.device.type, enabled=False):
            decoded = self.decoder((values.float() - self.geometry_mean) / self.geometry_scale)
        return decoded[:, :self.hidden_dim], decoded[:, self.hidden_dim]

    def forward(self, hidden):
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_dim:
            raise ValueError("hidden must have [protein, frozen-hidden-dimension] shape")
        weight, bias = self.prototypes()
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(hidden.float(), weight.float(), bias.float())

    def score_geometry(self, hidden, geometry):
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_dim:
            raise ValueError("hidden must match the frozen protein encoder")
        weight, bias = self.prototypes(geometry)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return F.linear(hidden.float(), weight.float(), bias.float())
