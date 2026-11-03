"""Process-local AdamW policy for the independent v095 worker.

The legacy worker keeps its public training loop. Only its module-local torch
reference is proxied; torch.optim and other imported workers are never mutated.
"""
from __future__ import annotations

import math
import torch


def adamw_groups(parameters, weight_decay):
    params = [p for p in parameters if p.requires_grad]
    if len({id(p) for p in params}) != len(params):
        raise ValueError('Duplicate optimizer parameters')
    return [dict(params=[p for p in params if p.ndim >= 2], weight_decay=weight_decay),
            dict(params=[p for p in params if p.ndim < 2], weight_decay=0.)]


class _OptimProxy:
    def __init__(self, spec):
        self.spec = spec

    def AdamW(self, parameters, *, lr, weight_decay):
        return torch.optim.AdamW(adamw_groups(parameters, weight_decay), lr=lr,
                                 eps=self.spec['eps'], betas=tuple(self.spec['betas']))

    def __getattr__(self, name):
        return getattr(torch.optim, name)


class _TorchProxy:
    def __init__(self, spec):
        self.optim = _OptimProxy(spec)

    def __getattr__(self, name):
        return getattr(torch, name)


def configure_training_runtime(base, config):
    spec = config['full_task']['optimizer']
    if set(spec) != {'name', 'eps', 'betas', 'decay_policy'}:
        raise ValueError('Unexpected v095 optimizer fields')
    if spec['name'] != 'adamw' or spec['decay_policy'] != 'matrices_only':
        raise ValueError('v095 requires AdamW with matrix-only weight decay')
    if not math.isfinite(spec['eps']) or spec['eps'] <= 0:
        raise ValueError('Invalid optimizer epsilon')
    if len(spec['betas']) != 2 or any(not math.isfinite(x) or not 0 <= x < 1 for x in spec['betas']):
        raise ValueError('Invalid AdamW betas')
    base.torch = _TorchProxy(spec)
    base._v095_optimizer_contract = dict(spec)
