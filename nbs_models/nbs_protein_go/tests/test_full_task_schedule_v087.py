from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from nbs_pg.full_task_schedule_v087 import ScheduleConfigV087, EpochScheduleV087


def cfg():
    return ScheduleConfigV087(1e-4, 1e-5, 100)


def optimizer(lr=1e-4):
    return SimpleNamespace(param_groups=[{"lr": lr}, {"lr": lr}])


def test_schedule_has_warmup_hold_and_fixed_end_and_monotone_decay():
    c = cfg()
    assert c.horizon_steps == 500
    assert c.learning_rate(1) == pytest.approx(1e-5)
    assert c.learning_rate(10) == pytest.approx(1e-4)
    assert c.learning_rate(300) == pytest.approx(1e-4)
    assert c.learning_rate(400) == pytest.approx(5.5e-5)
    assert c.learning_rate(500) == pytest.approx(1e-5)
    assert len({c.learning_rate(i) for i in range(10, 301)}) == 1
    values = [c.learning_rate(i) for i in range(300, 501)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_fractional_warmup_boundary_is_continuous_without_flooring():
    c = replace(cfg(), steps_per_epoch=3823)
    assert c.learning_rate(382) == pytest.approx(c.max_lr * 382 / 382.3)
    assert c.learning_rate(383) == c.max_lr
    assert c.learning_rate(19115) == c.min_lr


def test_from_full_task_config_does_not_use_steps_for_horizon():
    base = {"epochs": 5, "steps": 4, "learning_rate": 1e-4,
            "scheduler": {"name": "epoch_warmup_hold_cosine", "min_lr": 1e-5,
                          "warmup_epochs": .1, "hold_until_epoch": 3.}}
    c = ScheduleConfigV087.from_full_task(base, 3823)
    assert c.horizon_steps == 19115
    assert asdict(c)["epochs"] == 5
    with pytest.raises(ValueError, match="unknown"):
        ScheduleConfigV087.from_full_task({**base, "scheduler": {**base["scheduler"], "horizon_steps": 500}}, 3823)
    with pytest.raises(ValueError, match="explicit"):
        ScheduleConfigV087.from_full_task({}, 3823)


def test_exact_resume_and_next_update_matches_uninterrupted():
    first_opt, second_opt = optimizer(), optimizer()
    first, second = EpochScheduleV087(first_opt, cfg()), EpochScheduleV087(second_opt, cfg())
    for step in range(1, 324):
        first.apply(step)
    second_opt.param_groups = deepcopy(first_opt.param_groups)
    second.load_state_dict(first.state_dict(), expected_step=323)
    for step in range(324, 501):
        assert first.apply(step) == second.apply(step)
    assert first.state_dict() == second.state_dict()


def test_resume_requires_same_horizon_optimizer_lr_and_expected_step():
    opt = optimizer()
    sched = EpochScheduleV087(opt, cfg())
    sched.apply(1)
    saved = sched.state_dict()
    for c in [replace(cfg(), epochs=6), replace(cfg(), steps_per_epoch=99),
              replace(cfg(), hold_until_epoch=2), replace(cfg(), warmup_epochs=.2)]:
        with pytest.raises(ValueError, match="identical"):
            EpochScheduleV087(opt, c).load_state_dict(saved, expected_step=1)
    with pytest.raises(ValueError, match="position"):
        EpochScheduleV087(opt, cfg()).load_state_dict(saved, expected_step=2)
    with pytest.raises(ValueError, match="optimizer"):
        EpochScheduleV087(optimizer(), cfg()).load_state_dict(saved, expected_step=1)
    saved["last_lr"] = 1e-3
    with pytest.raises(ValueError, match="saved learning"):
        EpochScheduleV087(opt, cfg()).load_state_dict(saved, expected_step=1)


def test_resume_float_tolerance_and_reject_nan():
    opt = optimizer()
    sched = EpochScheduleV087(opt, cfg())
    sched.apply(1)
    saved = sched.state_dict()
    saved["last_lr"] *= 1 + 1e-10
    for group in opt.param_groups:
        group["lr"] *= 1 + 1e-10
    EpochScheduleV087(opt, cfg()).load_state_dict(saved, expected_step=1)
    saved["last_lr"] = float("nan")
    with pytest.raises(ValueError, match="saved learning"):
        EpochScheduleV087(opt, cfg()).load_state_dict(saved, expected_step=1)


@pytest.mark.parametrize("updates", [{"epochs": 0}, {"epochs": 2.5}, {"steps_per_epoch": 0},
                                      {"max_lr": float("nan")}, {"min_lr": -1},
                                      {"warmup_epochs": 3}, {"hold_until_epoch": 5},
                                      {"warmup_epochs": -.1}])
def test_invalid_schedule_contracts_rejected(updates):
    with pytest.raises(ValueError):
        replace(cfg(), **updates)


def test_consecutive_integer_updates_and_horizon_are_enforced():
    sched = EpochScheduleV087(optimizer(), cfg())
    for invalid in [0, 2, True, 1.0]:
        with pytest.raises(ValueError, match="consecutive"):
            sched.apply(invalid)
    for invalid in [0, 501, 1.2]:
        with pytest.raises(ValueError, match="integer"):
            cfg().learning_rate(invalid)
