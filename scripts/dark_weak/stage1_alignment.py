"""Protein-name keyed row placement for shard-grouped LATENCE Stage-1 loaders.

The source batch sampler groups samples by binary MSA shard even when its
shuffle flag is false.  This class prevents silently conflating iteration
order with metadata/prediction-matrix order.
"""
from __future__ import annotations

import numpy as np


class ProteinRowPlacement:
    def __init__(self, expected):
        self.expected = [str(x) for x in expected]
        if not self.expected or len(set(self.expected)) != len(self.expected):
            raise ValueError("Expected protein IDs must be nonempty and unique")
        self.index = {name: i for i, name in enumerate(self.expected)}
        self.seen = np.zeros(len(self.expected), dtype=np.bool_)

    def locate_batch(self, proteins):
        names = [str(x) for x in proteins]
        if not names or len(names) != len(set(names)):
            raise ValueError("Batch contains duplicate or missing protein IDs")
        missing = [name for name in names if name not in self.index]
        if missing:
            raise ValueError(f"MSA batch contains {len(missing)} unrequested proteins: {missing[:5]}")
        rows = np.asarray([self.index[name] for name in names], dtype=np.int64)
        if np.any(self.seen[rows]):
            repeated = [name for name, row in zip(names, rows) if self.seen[row]]
            raise ValueError(f"MSA loader emitted a protein multiple times: {repeated[:5]}")
        self.seen[rows] = True
        return rows

    def assert_complete(self):
        missing = np.flatnonzero(~self.seen)
        if len(missing):
            names = [self.expected[i] for i in missing[:8]]
            raise ValueError(f"Stage-1 inference missing {len(missing)} proteins: {names}")
