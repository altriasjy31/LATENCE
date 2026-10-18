"""Exact lifetime bookkeeping for independently scheduled v0.8.9.1 phases.

This module imports neither torch nor NumPy. A phase may change batch size or
rank count, but its inherited counters are measured from a completed source
epoch, never reconstructed using the *new* batch size or a checkpoint filename.
``target_epochs`` is the lifetime weak-epoch target; ``step`` stays phase-local.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from numbers import Integral
from typing import Mapping


RELEASE_VERSION = "0.8.9.1"
MODEL_VERSION = "0.8.8"


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


@dataclass(frozen=True)
class PhasePlanV089:
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
PhasePlan = PhasePlanV089


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _saved_epoch_plan(saved):
    values = _mapping(saved.get("epoch_plan"), "checkpoint epoch_plan")
    names = ("n_weak", "n_core", "weak_batch_size", "core_batch_size", "world_size", "epochs")
    if set(values) != set(names):
        raise ValueError("checkpoint epoch_plan has an incomplete or unknown schema")
    return {name: _integer(values[name], "epoch_plan." + name, minimum=1) for name in names}


def checkpoint_origin(saved, *, n_weak=None, n_core=None,
                      weak_ids_sha256=None, core_ids_sha256=None):
    """Validate a source epoch boundary and derive actual inherited counters.

    The trainer additionally restores the legacy seed stream to validate its
    permutations and RNG payloads, and checks data/ontology/model contracts.
    This function verifies serialized counters, populations, and phase chaining
    without requiring model tensors, production arrays, or a GPU.
    """
    saved = _mapping(saved, "checkpoint")
    version = saved.get("runner_version")
    if version not in (MODEL_VERSION, RELEASE_VERSION):
        raise ValueError("init-from requires a v0.8.8 or v0.8.9.1 checkpoint")
    plan = _saved_epoch_plan(saved)
    if plan["core_batch_size"] * plan["world_size"] > plan["n_core"]:
        raise ValueError("source global core batch exceeds its population")
    for name, requested in (("n_weak", n_weak), ("n_core", n_core)):
        if requested is not None and _integer(requested, name, minimum=1) != plan[name]:
            raise ValueError(f"source {name} population differs from current training data")
    world = _integer(saved.get("world_size"), "checkpoint world_size", minimum=1)
    if world != plan["world_size"]:
        raise ValueError("source world_size differs from epoch_plan")
    stream = _mapping(saved.get("epoch_stream"), "checkpoint epoch_stream")
    if _integer(stream.get("version"), "epoch_stream version", minimum=1) != 1:
        raise ValueError("source epoch_stream version differs from the supported schema")
    contract = _mapping(stream.get("contract"), "epoch_stream contract")
    contract_plan = _saved_epoch_plan({"epoch_plan": contract.get("plan")})
    if contract_plan != plan:
        raise ValueError("source epoch_stream and epoch_plan differ")
    for name, expected in (("weak_ids_sha256", weak_ids_sha256),
                           ("core_ids_sha256", core_ids_sha256)):
        value = contract.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"source {name} is missing or malformed")
        if expected is not None and value != expected:
            raise ValueError(f"source {name} differs from current seed population")
    step = _integer(saved.get("step"), "checkpoint step", minimum=1)
    if _integer(stream.get("completed_steps"), "epoch_stream completed_steps") != step:
        raise ValueError("source seed stream and optimizer update disagree")
    global_weak = plan["weak_batch_size"] * world
    global_core = plan["core_batch_size"] * world
    steps_per_epoch = (plan["n_weak"] + global_weak - 1) // global_weak
    if step > plan["epochs"] * steps_per_epoch or step % steps_per_epoch:
        raise ValueError("init-from requires an exact completed weak epoch, not a partial checkpoint")
    completed = step // steps_per_epoch
    weak_epoch = _integer(stream.get("weak_epoch"), "epoch_stream weak_epoch")
    weak_cursor = _integer(stream.get("weak_cursor"), "epoch_stream weak_cursor")
    if weak_epoch != completed - 1 or weak_cursor != plan["n_weak"]:
        raise ValueError("source weak stream counters do not describe a completed epoch")
    weak_exposures = weak_epoch * plan["n_weak"] + weak_cursor
    core_exposures = step * global_core
    core_cycle, core_zero_cursor = divmod(core_exposures - 1, plan["n_core"])
    if (_integer(stream.get("core_cycle"), "epoch_stream core_cycle") != core_cycle or
            _integer(stream.get("core_cursor"), "epoch_stream core_cursor") != core_zero_cursor + 1):
        raise ValueError("source core stream counters disagree with actual update exposure")
    expected_epoch = completed
    updates = step
    if version == RELEASE_VERSION:
        phase_values = dict(_mapping(saved.get("phase_plan"), "checkpoint phase_plan"))
        try:
            phase = PhasePlanV089(**phase_values)
        except TypeError as exc:
            raise ValueError("source phase_plan has an incomplete or unknown schema") from exc
        if (phase.n_weak != plan["n_weak"] or phase.n_core != plan["n_core"] or
                phase.global_weak_batch != global_weak or phase.global_core_batch != global_core or
                phase.steps_per_epoch != steps_per_epoch or phase.remaining_epochs != plan["epochs"]):
            raise ValueError("source phase_plan and epoch_plan differ")
        progress = phase.progress(step)
        expected_epoch = progress["cumulative_epoch"]
        weak_exposures = progress["lifetime_weak_exposures"]
        core_exposures = progress["lifetime_core_exposures"]
        updates = progress["lifetime_optimizer_updates"]
        for name in ("lifetime_weak_exposures", "lifetime_core_exposures", "lifetime_optimizer_updates"):
            if _integer(saved.get(name), "checkpoint " + name) != progress[name]:
                raise ValueError(f"source {name} differs from the frozen phase origin and actual progress")
        for name in ("phase_step", "phase_epoch", "phase_step_in_epoch", "cumulative_completed_epochs"):
            if name in saved and _integer(saved[name], "checkpoint " + name) != progress[name]:
                raise ValueError(f"source {name} disagrees with actual phase progress")
    if _integer(saved.get("epoch"), "checkpoint epoch", minimum=1) != expected_epoch:
        raise ValueError("source epoch display differs from actual completed weak epochs")
    if _integer(saved.get("step_in_epoch"), "checkpoint step_in_epoch", minimum=1) != steps_per_epoch:
        raise ValueError("source step_in_epoch does not describe an exact epoch boundary")
    return {"origin_epochs": expected_epoch,
            "origin_weak_exposures": weak_exposures,
            "origin_core_exposures": core_exposures,
            "origin_optimizer_updates": updates,
            "source_runner_version": version, "source_phase_step": step,
            "source_world_size": world, "source_steps_per_epoch": steps_per_epoch,
            "weak_ids_sha256": contract["weak_ids_sha256"],
            "core_ids_sha256": contract["core_ids_sha256"]}


source_summary = checkpoint_origin


def resume_contract(full_task, phase_plan, world_size):
    """Freeze the complete training config, including checkpoint cadence.

    JSON serialization rejects nonfinite values and provides an isolated value
    snapshot. The root trainer separately verifies implementation and datasets.
    """
    full_task = _mapping(full_task, "full_task")
    if not isinstance(phase_plan, PhasePlanV089):
        raise ValueError("phase_plan must be a PhasePlanV089")
    world_size = _integer(world_size, "world_size", minimum=1)
    cadence = _integer(full_task.get("checkpoint_every_epochs", 5), "checkpoint_every_epochs", minimum=1)
    if (phase_plan.global_weak_batch != _integer(full_task.get("weak_batch"), "weak_batch", minimum=1) * world_size or
            phase_plan.global_core_batch != _integer(full_task.get("core_batch"), "core_batch", minimum=1) * world_size):
        raise ValueError("phase_plan global batches differ from the config and world_size")
    if _integer(full_task.get("epochs"), "epochs", minimum=1) != phase_plan.target_epochs:
        raise ValueError("full_task.epochs must equal the cumulative target_epochs")
    if full_task.get("epoch_unit") != "weak":
        raise ValueError("v089 requires epoch_unit=weak")
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
