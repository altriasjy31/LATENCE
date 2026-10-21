import math
import pytest
import torch
from nbs_pg.full_task_schedule_v085 import ScheduleConfigV085, WarmupScheduleV085


def config(name="warmup_cosine", horizon=4000):
    return ScheduleConfigV085(name, 3e-4, 3e-5, 25, horizon)


def test_warmup_cosine_endpoints_and_monotonicity():
    cfg = config()
    assert cfg.learning_rate(1) == pytest.approx(3e-4 / 25)
    assert cfg.learning_rate(25) == 3e-4
    assert cfg.learning_rate(4000) == 3e-5
    values = [cfg.learning_rate(i) for i in range(25, 4001)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert cfg.learning_rate(2000) == pytest.approx(3e-5 + .5*(3e-4-3e-5)*(1+math.cos(math.pi*1975/3975)))


def test_constant_control_preserves_v084_warmup_formula():
    cfg = config("constant")
    for step in [1, 12, 25, 600, 2000, 4000]:
        assert cfg.learning_rate(step) == pytest.approx(3e-4 * min(1, step / 25))


def test_optimizer_schedule_resume_exact():
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    first = WarmupScheduleV085(optimizer, config())
    for step in range(1, 601):
        first.apply(step)
    saved = first.state_dict()
    later = WarmupScheduleV085(optimizer, config())
    later.load_state_dict(saved, expected_step=600)
    for step in range(601, 801):
        assert first.apply(step) == later.apply(step)
    assert first.state_dict() == later.state_dict()


def test_schedule_requires_valid_contract_and_sequential_updates():
    with pytest.raises(ValueError, match="explicit full_task.scheduler"):
        ScheduleConfigV085.from_full_task({"learning_rate": .01})
    for opts in [("bad", 1., .1, 1, 4), ("warmup_cosine", float("nan"), .1, 1, 4),
                 ("warmup_cosine", 1., 2., 1, 4), ("warmup_cosine", 1., .1, 4, 4)]:
        with pytest.raises(ValueError):
            ScheduleConfigV085(*opts)
    parameter = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([parameter], lr=3e-4)
    scheduler = WarmupScheduleV085(optimizer, config())
    with pytest.raises(ValueError, match="consecutive"):
        scheduler.apply(2)
    for step in [0, 4001]:
        with pytest.raises(ValueError, match="fixed horizon"):
            scheduler.config.learning_rate(step)
