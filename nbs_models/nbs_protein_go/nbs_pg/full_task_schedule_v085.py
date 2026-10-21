"""Versioned, resumable learning-rate schedule independent of a run stop budget."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ScheduleConfigV085:
    name: str
    max_lr: float
    min_lr: float
    warmup_steps: int
    horizon_steps: int

    def __post_init__(self):
        if self.name not in ("constant", "warmup_cosine"):
            raise ValueError("scheduler.name must be constant or warmup_cosine")
        if not math.isfinite(self.max_lr) or self.max_lr <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.min_lr) or not 0 <= self.min_lr <= self.max_lr:
            raise ValueError("scheduler.min_lr must be finite and between 0 and learning_rate")
        if self.horizon_steps <= 0 or not 0 <= self.warmup_steps < self.horizon_steps:
            raise ValueError("scheduler.horizon_steps must exceed nonnegative warmup_steps")

    @classmethod
    def from_full_task(cls, config):
        spec = config.get("scheduler")
        if not isinstance(spec, dict):
            raise ValueError("v085 requires explicit full_task.scheduler; --steps never defines the schedule horizon")
        unknown = set(spec) - {"name", "horizon_steps", "min_lr"}
        if unknown:
            raise ValueError(f"unknown scheduler settings: {sorted(unknown)}")
        if "name" not in spec or "horizon_steps" not in spec:
            raise ValueError("scheduler requires explicit name and horizon_steps")
        rate = float(config["learning_rate"])
        return cls(name=spec["name"], max_lr=rate,
                   min_lr=float(spec.get("min_lr", .1 * rate)),
                   warmup_steps=int(config.get("warmup_steps", 25)),
                   horizon_steps=int(spec["horizon_steps"]))

    def learning_rate(self, step):
        step = int(step)
        if not 1 <= step <= self.horizon_steps:
            raise ValueError(f"optimizer step must be between 1 and fixed horizon {self.horizon_steps}")
        if self.warmup_steps and step <= self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        if self.name == "constant":
            return self.max_lr
        # The last warmup step reaches max_lr; the fixed final step reaches min_lr.
        progress = (step - self.warmup_steps) / (self.horizon_steps - self.warmup_steps)
        return self.min_lr + .5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


class WarmupScheduleV085:
    """Apply the LR for a numbered optimizer update and persist exact resume state."""
    VERSION = 1

    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.last_step = 0
        self.last_lr = None

    def apply(self, step):
        if int(step) != self.last_step + 1:
            raise ValueError("scheduler requires consecutive optimizer updates")
        lr = self.config.learning_rate(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_step, self.last_lr = int(step), lr
        return lr

    def state_dict(self):
        return {"version": self.VERSION, "contract": asdict(self.config),
                "last_step": self.last_step, "last_lr": self.last_lr}

    def load_state_dict(self, state, *, expected_step):
        if state.get("version") != self.VERSION or state.get("contract") != asdict(self.config):
            raise ValueError("resume requires identical scheduler name, horizon, warmup and learning rates")
        last_step = state.get("last_step")
        if last_step != expected_step or not 0 <= last_step <= self.config.horizon_steps:
            raise ValueError("scheduler position differs from checkpoint optimizer step")
        expected_lr = self.config.learning_rate(last_step) if last_step else None
        if state.get("last_lr") != expected_lr:
            raise ValueError("scheduler saved learning rate is inconsistent")
        if last_step and any(group["lr"] != expected_lr for group in self.optimizer.param_groups):
            raise ValueError("optimizer and scheduler saved learning rates differ")
        self.last_step, self.last_lr = last_step, expected_lr
