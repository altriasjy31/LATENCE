"""Exact v0.9.0 scorer-phase counters with immutable same-phase resume.

The shared scorer always begins at zero training exposures. Optional imported
encoder training counts are separate lineage and cannot advance this schedule.
This module imports neither torch nor NumPy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from numbers import Integral
from typing import Mapping


RELEASE_VERSION = "0.9.0"
MODEL_VERSION = "0.9.0"
DATA_VERSION = "0.8.8"


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


@dataclass(frozen=True)
class PhasePlanV090:
    origin_epochs: int
    target_epochs: int
    origin_weak_exposures: int
    origin_core_exposures: int
    origin_optimizer_updates: int
    steps_per_epoch: int
    n_weak: int
    n_core: int
    global_core_batch: int
    global_weak_batch: int

    def __post_init__(self):
        positive = {"target_epochs", "steps_per_epoch", "n_weak", "n_core",
                    "global_core_batch", "global_weak_batch"}
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _integer(getattr(self, name), name,
                                                   minimum=int(name in positive)))
        if any((self.origin_epochs, self.origin_weak_exposures, self.origin_core_exposures, self.origin_optimizer_updates)):
            raise ValueError("v090 shared-head phases start at zero; encoder-transfer counters belong to separate lineage")
        if self.target_epochs <= self.origin_epochs:
            raise ValueError("target_epochs must exceed completed origin_epochs")
        expected_steps = (self.n_weak + self.global_weak_batch - 1) // self.global_weak_batch
        if self.steps_per_epoch != expected_steps:
            raise ValueError("steps_per_epoch differs from the exact weak population and global batch")
        if self.global_core_batch > self.n_core:
            raise ValueError("global core batch must not exceed the core population")
        if self.origin_weak_exposures != self.origin_epochs * self.n_weak:
            raise ValueError("origin weak exposures must represent exactly completed weak epochs")
        if self.origin_epochs == 0:
            if self.origin_core_exposures or self.origin_optimizer_updates:
                raise ValueError("a fresh phase cannot inherit core exposures or optimizer updates")
        elif self.origin_core_exposures == 0 or self.origin_optimizer_updates == 0:
            raise ValueError("an initialized phase requires actual source core and update counters")

    @property
    def remaining_epochs(self):
        return self.target_epochs - self.origin_epochs

    @property
    def phase_total_steps(self):
        return self.remaining_epochs * self.steps_per_epoch

    def progress(self, phase_step):
        """Counters after an update, including exact short weak tails.

        Epoch display fields are one-based for nonzero steps. At step zero
        phase display is zero and cumulative display equals the completed
        origin. ``cumulative_completed_epochs`` always counts only full passes.
        """
        phase_step = _integer(phase_step, "phase_step")
        if phase_step > self.phase_total_steps:
            raise ValueError("phase_step exceeds the frozen phase horizon")
        completed, remainder = divmod(phase_step, self.steps_per_epoch)
        weak = completed * self.n_weak + min(remainder * self.global_weak_batch, self.n_weak)
        core = phase_step * self.global_core_batch
        phase_epoch = (phase_step - 1) // self.steps_per_epoch + 1 if phase_step else 0
        in_epoch = (phase_step - 1) % self.steps_per_epoch + 1 if phase_step else 0
        lifetime_weak = self.origin_weak_exposures + weak
        lifetime_core = self.origin_core_exposures + core
        return {"phase_step": phase_step,
                "phase_epoch": phase_epoch, "phase_step_in_epoch": in_epoch,
                "cumulative_epoch": self.origin_epochs + phase_epoch,
                "cumulative_completed_epochs": self.origin_epochs + completed,
                "phase_weak_exposures": weak, "phase_core_exposures": core,
                "lifetime_weak_exposures": lifetime_weak,
                "lifetime_core_exposures": lifetime_core,
                "lifetime_optimizer_updates": self.origin_optimizer_updates + phase_step,
                "phase_weak_equivalent_passes": weak / self.n_weak,
                "phase_core_equivalent_passes": core / self.n_core,
                "lifetime_weak_equivalent_passes": lifetime_weak / self.n_weak,
                "lifetime_core_equivalent_passes": lifetime_core / self.n_core}

    def checkpoint_steps(self, every_epochs=5):
        """Map phase-local updates to fixed cumulative completed epochs.

        For origin 20 and target 100, interval 5 saves 25, 30, ..., 100. An
        unaligned final target is still saved. The origin itself is not saved.
        Operational short stops are handled separately by the trainer.
        """
        every_epochs = _integer(every_epochs, "checkpoint_every_epochs", minimum=1)
        first = (self.origin_epochs // every_epochs + 1) * every_epochs
        epochs = set(range(first, self.target_epochs + 1, every_epochs))
        epochs.add(self.target_epochs)
        return {(epoch - self.origin_epochs) * self.steps_per_epoch: epoch
                for epoch in sorted(epochs)}


# A short alias lets callers use PhasePlan without importing legacy plan names.
PhasePlan = PhasePlanV090


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def resume_contract(full_task, phase_plan, world_size):
    """Freeze the complete training config, including checkpoint cadence.

    JSON serialization rejects nonfinite values and provides an isolated value
    snapshot. The root trainer separately verifies implementation and datasets.
    """
    full_task = _mapping(full_task, "full_task")
    if not isinstance(phase_plan, PhasePlanV090):
        raise ValueError("phase_plan must be a PhasePlanV090")
    world_size = _integer(world_size, "world_size", minimum=1)
    cadence = _integer(full_task.get("checkpoint_every_epochs", 5), "checkpoint_every_epochs", minimum=1)
    if (phase_plan.global_weak_batch != _integer(full_task.get("weak_batch"), "weak_batch", minimum=1) * world_size or
            phase_plan.global_core_batch != _integer(full_task.get("core_batch"), "core_batch", minimum=1) * world_size):
        raise ValueError("phase_plan global batches differ from the config and world_size")
    if _integer(full_task.get("epochs"), "epochs", minimum=1) != phase_plan.target_epochs:
        raise ValueError("full_task.epochs must equal the cumulative target_epochs")
    if full_task.get("epoch_unit") != "weak":
        raise ValueError("v090 requires epoch_unit=weak")
    try:
        snapshot = json.loads(json.dumps(dict(full_task), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("training config must contain finite JSON values") from exc
    return {"runner_version": RELEASE_VERSION, "model_version": MODEL_VERSION,
            "full_task": snapshot, "phase_plan": asdict(phase_plan),
            "world_size": world_size, "checkpoint_every_epochs": cadence}


def validate_resume_contract(saved_contract, expected_contract):
    saved_contract = _mapping(saved_contract, "saved phase training contract")
    expected_contract = _mapping(expected_contract, "expected phase training contract")
    # Canonical JSON keeps boolean and integer values distinct, unlike Python's
    # True == 1 comparison. It also rejects forged NaN/Infinity payloads.
    try:
        old = json.dumps(dict(saved_contract), sort_keys=True, allow_nan=False)
        new = json.dumps(dict(expected_contract), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("resume contract must contain finite JSON values") from exc
    if old != new:
        raise ValueError("resume requires identical phase origin, target, batches, ranks, config and checkpoint cadence")
    return deepcopy(dict(expected_contract))
