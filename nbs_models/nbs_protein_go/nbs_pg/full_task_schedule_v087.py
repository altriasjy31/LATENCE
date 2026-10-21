"""Epoch-defined warmup/hold/cosine schedule with a fixed resume contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral


@dataclass(frozen=True)
class ScheduleConfigV087:
    max_lr: float
    min_lr: float
    steps_per_epoch: int
    epochs: int = 5
    warmup_epochs: float = .1
    hold_until_epoch: float = 3.0

    def __post_init__(self):
        for name in ("steps_per_epoch", "epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"scheduler.{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if not math.isfinite(self.max_lr) or self.max_lr <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.min_lr) or not 0 <= self.min_lr <= self.max_lr:
            raise ValueError("scheduler.min_lr must be finite and between zero and learning_rate")
        if not all(math.isfinite(x) for x in (self.warmup_epochs, self.hold_until_epoch)) or not 0 <= self.warmup_epochs < self.hold_until_epoch < self.epochs:
            raise ValueError("scheduler requires 0 <= warmup_epochs < hold_until_epoch < epochs")

    @property
    def horizon_steps(self):
        return self.steps_per_epoch * self.epochs

    @classmethod
    def from_full_task(cls, config, steps_per_epoch):
        spec = config.get("scheduler")
        if not isinstance(spec, dict):
            raise ValueError("v087 requires explicit full_task.scheduler")
        allowed = {"name", "min_lr", "warmup_epochs", "hold_until_epoch"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"unknown v087 scheduler settings: {sorted(unknown)}")
        if spec.get("name") != "epoch_warmup_hold_cosine":
            raise ValueError("v087 scheduler.name must be epoch_warmup_hold_cosine")
        if "epochs" not in config:
            raise ValueError("v087 requires full_task.epochs, independent of a stop-step override")
        rate = float(config["learning_rate"])
        return cls(max_lr=rate, min_lr=float(spec.get("min_lr", .1 * rate)),
                   steps_per_epoch=steps_per_epoch, epochs=config["epochs"],
                   warmup_epochs=float(spec.get("warmup_epochs", .1)),
                   hold_until_epoch=float(spec.get("hold_until_epoch", 3.0)))

    def learning_rate(self, step):
        if isinstance(step, bool) or not isinstance(step, Integral) or not 1 <= step <= self.horizon_steps:
            raise ValueError(f"optimizer step must be an integer in [1, {self.horizon_steps}]")
        # Fractional epoch boundaries are intentionally not rounded to steps.
        position = step / self.steps_per_epoch
        if self.warmup_epochs and position <= self.warmup_epochs:
            return self.max_lr * position / self.warmup_epochs
        if position <= self.hold_until_epoch:
            return self.max_lr
        progress = (position - self.hold_until_epoch) / (self.epochs - self.hold_until_epoch)
        return self.min_lr + .5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


class EpochScheduleV087:
    VERSION = 1

    def __init__(self, optimizer, config):
        self.optimizer, self.config = optimizer, config
        self.last_step = 0
        self.last_lr = None

    def apply(self, step):
        if isinstance(step, bool) or not isinstance(step, Integral) or step != self.last_step + 1:
            raise ValueError("scheduler requires consecutive integer optimizer updates")
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
            raise ValueError("resume requires identical epochs, steps per epoch, warmup, hold and learning rates")
        step = state.get("last_step")
        if isinstance(step, bool) or not isinstance(step, Integral) or step != expected_step or not 0 <= step <= self.config.horizon_steps:
            raise ValueError("scheduler position differs from checkpoint optimizer step")
        lr = self.config.learning_rate(step) if step else None
        saved_lr = state.get("last_lr")
        if (lr is None and saved_lr is not None) or (lr is not None and
                (not isinstance(saved_lr, (int, float)) or not math.isclose(saved_lr, lr, rel_tol=1e-9, abs_tol=1e-15))):
            raise ValueError("scheduler saved learning rate is inconsistent")
        if step and any(not math.isclose(float(group["lr"]), lr, rel_tol=1e-9, abs_tol=1e-15)
                        for group in self.optimizer.param_groups):
            raise ValueError("optimizer and scheduler saved learning rates differ")
        self.last_step, self.last_lr = int(step), lr
