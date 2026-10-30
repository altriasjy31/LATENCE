"""Anchored shared+residual head; original v088 encoder implementation is reused.

W = W0 + (q-q0) + delta_r equals q + (W0-q0+delta_r).
Unlike subtracting and re-adding q0 to W0, this evaluates to W0 exactly at
initialization on the same backend. The bias has the identical decomposition.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HybridGraphModelV091(nn.Module):
    def __init__(self, backbone, shared):
        super().__init__()
        if not isinstance(backbone.classifier, nn.Linear):
            raise ValueError("conversion requires the original direct linear classifier")
        self.backbone, self.shared = backbone, shared
        with torch.no_grad():
            q, b = shared.prototypes()
            w0, b0 = backbone.classifier.weight, backbone.classifier.bias
            if w0.shape != q.shape or b0.shape != b.shape:
                raise ValueError("shared and source classifier dimensions differ")
            self.register_buffer("original_weight", w0.detach().clone())
            self.register_buffer("original_bias", b0.detach().clone())
            self.register_buffer("shared_origin_weight", q.detach().clone())
            self.register_buffer("shared_origin_bias", b.detach().clone())
        self.residual_delta_weight = nn.Parameter(torch.zeros_like(q))
        self.residual_delta_bias = nn.Parameter(torch.zeros_like(b))
        # The unchanged source forward now returns its final hidden states.
        self.backbone.classifier = nn.Identity()
        self.backbone.requires_grad_(True)
        self.backbone.ontology_encoder.requires_grad_(False)
        self.train(False)

    @property
    def task_to_ontology(self):
        return self.backbone.task_to_ontology

    def train(self, mode=True):
        super().train(mode)
        # eval() does not disable gradients. Preserve source BN statistics and
        # disable encoder dropout throughout this first joint experiment.
        self.backbone.eval()
        self.backbone.ontology_encoder.eval()
        return self

    def encode_go(self):
        with torch.no_grad():
            return self.backbone.encode_go().detach()

    def effective_parameters(self, prototypes=None):
        q, b = self.shared.prototypes() if prototypes is None else prototypes
        w = self.original_weight + ((q - self.shared_origin_weight) + self.residual_delta_weight)
        bias = self.original_bias + ((b - self.shared_origin_bias) + self.residual_delta_bias)
        return w, bias

    def score_hidden(self, hidden, *, detach_shared_hidden=True, prototypes=None):
        q, b = self.shared.prototypes() if prototypes is None else prototypes
        w, bias = self.effective_parameters((q, b))
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            full = F.linear(hidden.float(), w.float(), bias.float())
            x = hidden.detach() if detach_shared_hidden else hidden
            shared = F.linear(x.float(), q.float(), b.float())
        return {"full": full, "shared": shared}

    def forward(self, batch, *, go_encoding=None, detach_shared_hidden=True, prototypes=None):
        hidden = self.backbone(batch, go_encoding=go_encoding)
        return self.score_hidden(hidden, detach_shared_hidden=detach_shared_hidden, prototypes=prototypes)

    def optimizer_groups(self, cfg):
        return [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": cfg["encoder_lr"], "group_name": "protein_encoder"},
            {"params": list(self.shared.parameters()), "lr": cfg["shared_lr"], "group_name": "shared_head"},
            {"params": [self.residual_delta_weight, self.residual_delta_bias],
             "lr": cfg["residual_lr"], "weight_decay": 0., "group_name": "residual_delta"},
        ]
