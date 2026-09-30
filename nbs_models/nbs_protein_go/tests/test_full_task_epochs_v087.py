from copy import deepcopy

import numpy as np
import pytest

from nbs_pg.full_task_epochs_v087 import EpochPlanV087, EpochSeedStreamV087


def make_stream(rank=0, *, nweak=11, ncore=7, wb=2, cb=1, world=3, epochs=5, seed=17):
    plan = EpochPlanV087(nweak, ncore, wb, cb, world, epochs)
    return EpochSeedStreamV087(np.arange(nweak), np.arange(ncore) + 100,
                              plan, seed=seed, rank=rank)


def compare_batch(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key])
        else:
            assert a[key] == b[key]


@pytest.mark.parametrize("nweak,world,wb", [(11, 3, 2), (7, 4, 2), (1, 3, 2), (12, 3, 2)])
def test_exact_weak_epochs_disjoint_ranks_and_tail(nweak, world, wb):
    streams = [make_stream(rank, nweak=nweak, world=world, wb=wb, ncore=13) for rank in range(world)]
    plan = streams[0].plan
    for epoch in range(1, plan.epochs + 1):
        weak = []
        for local_step in range(1, plan.steps_per_epoch + 1):
            batches = [s.next_batch() for s in streams]
            global_weak = np.concatenate([b["weak_ids"] for b in batches])
            global_core = np.concatenate([b["core_ids"] for b in batches])
            assert len(np.unique(global_weak)) == len(global_weak)
            assert len(np.unique(global_core)) == len(global_core)
            assert len(global_core) == plan.global_core_batch_size
            assert all(b["global_weak_count"] == len(global_weak) for b in batches)
            assert all(b["global_core_count"] == len(global_core) for b in batches)
            for rank, batch in enumerate(batches):
                np.testing.assert_array_equal(batch["global_weak_ids"], global_weak)
                np.testing.assert_array_equal(batch["global_core_ids"], global_core)
                np.testing.assert_array_equal(batch["weak_ids"], batch["global_weak_ids"][rank * wb:(rank + 1) * wb])
                np.testing.assert_array_equal(batch["core_ids"], batch["global_core_ids"][rank * plan.core_batch_size:(rank + 1) * plan.core_batch_size])
            assert all(b["epoch"] == epoch and b["step_in_epoch"] == local_step for b in batches)
            weak.extend(global_weak.tolist())
        assert sorted(weak) == list(range(nweak))
        assert batches[0]["epoch_float"] == epoch
        assert batches[0]["completed_epochs"] == epoch
    assert streams[0].completed_steps == plan.total_steps
    with pytest.raises(StopIteration):
        streams[0].next_batch()


def test_core_cross_cycle_batches_unique_and_each_cycle_exhaustive():
    streams = [make_stream(i, epochs=17) for i in range(3)]
    all_core = []
    for _ in range(streams[0].plan.total_steps):
        batch = np.concatenate([s.next_batch()["core_ids"] for s in streams])
        assert len(np.unique(batch)) == 3
        all_core.extend(batch.tolist())
    for offset in range(0, len(all_core) - 6, 7):
        assert sorted(all_core[offset:offset + 7]) == list(range(100, 107))


@pytest.mark.parametrize("stop", [0, 1, 2, 3, 7, 10])
def test_resume_matches_next_batch_including_epoch_and_core_boundaries(stop):
    original = make_stream()
    for _ in range(stop):
        original.next_batch()
    saved = original.state_dict()
    resumed = make_stream()
    resumed.load_state_dict(saved, expected_step=stop)
    for _ in range(stop, original.plan.total_steps):
        compare_batch(original.next_batch(), resumed.next_batch())


def test_rank_zero_state_resumes_other_ranks_with_global_same_order():
    rank_zero, rank_two = make_stream(0), make_stream(2)
    for _ in range(3):
        rank_zero.next_batch()
        rank_two.next_batch()
    restored = make_stream(2)
    restored.load_state_dict(rank_zero.state_dict(), expected_step=3)
    compare_batch(rank_two.next_batch(), restored.next_batch())


def test_stream_does_not_advance_global_numpy_or_torch_rng():
    import torch
    np.random.seed(552)
    torch.manual_seed(789)
    np_state = deepcopy(np.random.get_state())
    torch_state = torch.random.get_rng_state().clone()
    stream = make_stream()
    for _ in range(4):
        stream.next_batch()
    resumed = make_stream()
    resumed.load_state_dict(stream.state_dict(), expected_step=4)
    resumed.next_batch()
    after = np.random.get_state()
    assert np_state[0] == after[0]
    np.testing.assert_array_equal(np_state[1], after[1])
    assert np_state[2:] == after[2:]
    assert torch.equal(torch_state, torch.random.get_rng_state())


def test_resume_rejects_identity_same_length_and_world_batch_seed_changes():
    old = make_stream()
    old.next_batch()
    saved = old.state_dict()
    changed_ids = EpochSeedStreamV087(np.arange(11) + 20, np.arange(7) + 100,
                                     old.plan, seed=17)
    for new in [changed_ids, make_stream(seed=18), make_stream(wb=3),
                make_stream(world=2), make_stream(epochs=6)]:
        with pytest.raises(ValueError, match="identical"):
            new.load_state_dict(saved, expected_step=1)


@pytest.mark.parametrize("field,value", [("weak_cursor", 0), ("weak_epoch", 1),
                                         ("core_cursor", 0), ("core_cycle", 1)])
def test_resume_rejects_bad_cursor(field, value):
    original = make_stream()
    original.next_batch()
    state = original.state_dict()
    state[field] = value
    with pytest.raises(ValueError, match="cursors"):
        make_stream().load_state_dict(state, expected_step=1)


def test_bad_order_and_expected_step_are_rejected_without_mutation():
    stream = make_stream()
    stream.next_batch()
    state = stream.state_dict()
    with pytest.raises(ValueError, match="position"):
        make_stream().load_state_dict(state, expected_step=2)
    state["weak_order"][0] = 77
    target = make_stream()
    with pytest.raises(ValueError, match="permutations"):
        target.load_state_dict(state, expected_step=1)
    compare_batch(target.next_batch(), make_stream().next_batch())


def test_plan_uses_ceil_and_does_not_accept_invalid_dimensions():
    plan = EpochPlanV087(489222, 59468, 64, 16, 2, 5)
    assert plan.steps_per_epoch == 3823
    assert plan.total_steps == 19115
    for args in [(0, 7, 2, 1, 3), (11, 2, 2, 1, 3), (11, 7, 2., 1, 3), (True, 7, 2, 1, 3)]:
        with pytest.raises(ValueError):
            EpochPlanV087(*args)


def test_returned_global_seed_arrays_cannot_mutate_stream_permutations():
    stream = make_stream()
    batch = stream.next_batch()
    saved = stream.state_dict()
    batch["global_weak_ids"][:] = 999
    batch["global_core_ids"][:] = 999
    batch["weak_ids"][:] = 999
    batch["core_ids"][:] = 999
    after = stream.state_dict()
    np.testing.assert_array_equal(saved["weak_order"], after["weak_order"])
    np.testing.assert_array_equal(saved["core_order"], after["core_order"])
