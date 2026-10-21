"""Exact weak epochs and independently cycling core seeds for v0.8.7.

Every rank holds the same global stream and slices it locally. No padding,
replacement, or dropping is used for weak tails. This stream owns NumPy RNGs;
it never advances NumPy's global RNG, torch RNG, or the PU sampler RNG.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from numbers import Integral

import numpy as np


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class EpochPlanV087:
    n_weak: int
    n_core: int
    weak_batch_size: int
    core_batch_size: int
    world_size: int
    epochs: int = 5

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.global_core_batch_size > self.n_core:
            raise ValueError("global core batch must not exceed the core population")

    @property
    def global_weak_batch_size(self):
        return self.weak_batch_size * self.world_size

    @property
    def global_core_batch_size(self):
        return self.core_batch_size * self.world_size

    @property
    def steps_per_epoch(self):
        return (self.n_weak + self.global_weak_batch_size - 1) // self.global_weak_batch_size

    @property
    def total_steps(self):
        return self.epochs * self.steps_per_epoch


def _population(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer population")
    if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} has IDs outside int64")
    array = np.array(array, dtype=np.int64, copy=True)
    if np.any(array < 0) or np.unique(array).size != array.size:
        raise ValueError(f"{name} IDs must be nonnegative and unique")
    return array


def _digest(values):
    # Byte order is fixed, so checkpoint identities are machine independent.
    return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


class EpochSeedStreamV087:
    VERSION = 1

    def __init__(self, weak_ids, core_ids, plan, *, seed, rank=0):
        self.plan = plan
        self.weak_ids = _population(weak_ids, "weak")
        self.core_ids = _population(core_ids, "core")
        if len(self.weak_ids) != plan.n_weak or len(self.core_ids) != plan.n_core:
            raise ValueError("seed populations differ from epoch plan sizes")
        if np.intersect1d(self.weak_ids, self.core_ids).size:
            raise ValueError("weak and core seed populations must be disjoint")
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, Integral) or not 0 <= rank < plan.world_size:
            raise ValueError("rank must be an integer within world_size")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.rank, self.seed = int(rank), int(seed)
        weak_seed, core_seed = np.random.SeedSequence(self.seed).spawn(2)
        self._weak_rng = np.random.default_rng(weak_seed)
        self._core_rng = np.random.default_rng(core_seed)
        self._weak_order = self._weak_rng.permutation(self.weak_ids)
        self._core_order = self._core_rng.permutation(self.core_ids)
        self._weak_cursor = self._core_cursor = 0
        self._weak_epoch = self._core_cycle = self.completed_steps = 0
        self._contract = {"plan": asdict(plan), "seed": self.seed,
                          "weak_ids_sha256": _digest(self.weak_ids),
                          "core_ids_sha256": _digest(self.core_ids)}

    def _next_core(self):
        count = self.plan.global_core_batch_size
        parts = []
        used = np.empty(0, dtype=np.int64)
        while count:
            if self._core_cursor == self.plan.n_core:
                order = self._core_rng.permutation(self.core_ids)
                # Preserve an exact permutation while avoiding a repeated ID
                # when one global batch straddles two core cycles.
                if len(used):
                    excluded = np.isin(order, used)
                    order = np.concatenate((order[~excluded], order[excluded]))
                self._core_order = order
                self._core_cursor = 0
                self._core_cycle += 1
            take = min(count, self.plan.n_core - self._core_cursor)
            part = self._core_order[self._core_cursor:self._core_cursor + take]
            parts.append(part)
            used = np.concatenate((used, part))
            self._core_cursor += take
            count -= take
        return np.concatenate(parts)

    def next_batch(self):
        """Return local IDs plus global counts, all referring to this update.

        ``epoch`` and ``step_in_epoch`` are one-based. ``epoch_progress`` is
        the fraction of weak IDs exposed in this epoch *after* this batch;
        ``epoch_float`` adds previously completed epochs. At a short tail,
        local weak_ids may be empty, but the global weak count stays positive.
        """
        if self.completed_steps >= self.plan.total_steps:
            raise StopIteration("all configured weak epochs are complete")
        if self._weak_cursor == self.plan.n_weak:
            self._weak_epoch += 1
            self._weak_order = self._weak_rng.permutation(self.weak_ids)
            self._weak_cursor = 0
        end = min(self._weak_cursor + self.plan.global_weak_batch_size, self.plan.n_weak)
        global_weak = self._weak_order[self._weak_cursor:end]
        global_core = self._next_core()
        self._weak_cursor = end
        self.completed_steps += 1
        wstart = self.rank * self.plan.weak_batch_size
        cstart = self.rank * self.plan.core_batch_size
        progress = self._weak_cursor / self.plan.n_weak
        return {"weak_ids": global_weak[wstart:wstart + self.plan.weak_batch_size].copy(),
                "core_ids": global_core[cstart:cstart + self.plan.core_batch_size].copy(),
                "global_weak_ids": global_weak.copy(), "global_core_ids": global_core.copy(),
                "global_weak_count": len(global_weak), "global_core_count": len(global_core),
                "step": self.completed_steps, "epoch": self._weak_epoch + 1,
                "step_in_epoch": (self.completed_steps - 1) % self.plan.steps_per_epoch + 1,
                "steps_per_epoch": self.plan.steps_per_epoch,
                "epoch_progress": progress, "epoch_float": self._weak_epoch + progress,
                "completed_epochs": self._weak_epoch + int(self._weak_cursor == self.plan.n_weak),
                "weak_exposures": self._weak_epoch * self.plan.n_weak + self._weak_cursor,
                "core_exposures": self.completed_steps * self.plan.global_core_batch_size}

    def state_dict(self):
        return {"version": self.VERSION, "contract": deepcopy(self._contract),
                "completed_steps": self.completed_steps,
                "weak_epoch": self._weak_epoch, "weak_cursor": self._weak_cursor,
                "core_cycle": self._core_cycle, "core_cursor": self._core_cursor,
                "weak_order": self._weak_order.copy(), "core_order": self._core_order.copy(),
                "weak_rng": deepcopy(self._weak_rng.bit_generator.state),
                "core_rng": deepcopy(self._core_rng.bit_generator.state)}

    def load_state_dict(self, state, *, expected_step):
        """Restore rank-independent stream state, rejecting incompatible data."""
        if state.get("version") != self.VERSION or state.get("contract") != self._contract:
            raise ValueError("resume requires identical seed IDs/order, seed, batches, world size and epoch plan")
        step = state.get("completed_steps")
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral) or step != expected_step or not 0 <= step <= self.plan.total_steps:
            raise ValueError("seed stream position differs from checkpoint optimizer step")
        step = int(step)
        if step:
            weak_epoch = (step - 1) // self.plan.steps_per_epoch
            weak_cursor = min(((step - 1) % self.plan.steps_per_epoch + 1) *
                              self.plan.global_weak_batch_size, self.plan.n_weak)
            core_exposures = step * self.plan.global_core_batch_size
            core_cycle, core_zero_cursor = divmod(core_exposures - 1, self.plan.n_core)
            core_cursor = core_zero_cursor + 1
        else:
            weak_epoch = weak_cursor = core_cycle = core_cursor = 0
        expected = {"weak_epoch": weak_epoch, "weak_cursor": weak_cursor,
                    "core_cycle": core_cycle, "core_cursor": core_cursor}
        if any(state.get(name) != value for name, value in expected.items()):
            raise ValueError("seed stream cursors do not match the next optimizer step")
        weak_order = _population(state.get("weak_order"), "saved weak order")
        core_order = _population(state.get("core_order"), "saved core order")
        if not np.array_equal(np.sort(weak_order), np.sort(self.weak_ids)) or not np.array_equal(np.sort(core_order), np.sort(self.core_ids)):
            raise ValueError("saved seed orders are not permutations of current populations")
        # Validate both RNG payloads before mutating the live stream.
        weak_rng, core_rng = np.random.default_rng(), np.random.default_rng()
        try:
            weak_rng.bit_generator.state = deepcopy(state["weak_rng"])
            core_rng.bit_generator.state = deepcopy(state["core_rng"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid saved seed RNG state") from exc
        self._weak_rng, self._core_rng = weak_rng, core_rng
        self._weak_order, self._core_order = weak_order, core_order
        self._weak_epoch, self._weak_cursor = weak_epoch, weak_cursor
        self._core_cycle, self._core_cursor = core_cycle, core_cursor
        self.completed_steps = step
