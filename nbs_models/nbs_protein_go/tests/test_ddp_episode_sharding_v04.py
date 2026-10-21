from __future__ import annotations

from dataclasses import dataclass

from nbs_pg.local_loader import LatenceNBSLocalBatchLoader


class FakeSampler:
    def __init__(self):
        self.seeds = []

    def sample(self, *, seed):
        self.seeds.append(seed)
        return seed


@dataclass
class FakeMaterializer:
    def build_global_go_graph(self):
        return "go-graph"

    def materialize(self, episode, *, seed):
        return (episode, seed)


def test_rank_shards_are_disjoint_and_reproducible():
    sampler0, sampler1 = FakeSampler(), FakeSampler()
    loader0 = LatenceNBSLocalBatchLoader(
        sampler0, FakeMaterializer(), rank=0, world_size=2,
        steps_per_epoch_per_rank=4, base_seed=10,
    )
    loader1 = LatenceNBSLocalBatchLoader(
        sampler1, FakeMaterializer(), rank=1, world_size=2,
        steps_per_epoch_per_rank=4, base_seed=10,
    )
    loader0.set_epoch(3); loader1.set_epoch(3)
    list(loader0); list(loader1)
    assert set(sampler0.seeds).isdisjoint(sampler1.seeds)
    repeat = FakeSampler()
    loader_repeat = LatenceNBSLocalBatchLoader(
        repeat, FakeMaterializer(), rank=0, world_size=2,
        steps_per_epoch_per_rank=4, base_seed=10,
    )
    loader_repeat.set_epoch(3)
    list(loader_repeat)
    assert sampler0.seeds == repeat.seeds


def test_one_batch_prefetch_preserves_episode_order_and_seed_contract():
    synchronous_sampler = FakeSampler()
    synchronous = LatenceNBSLocalBatchLoader(
        synchronous_sampler, FakeMaterializer(), rank=1, world_size=2,
        steps_per_epoch_per_rank=5, base_seed=17, prefetch_batches=0,
    )
    prefetched_sampler = FakeSampler()
    prefetched = LatenceNBSLocalBatchLoader(
        prefetched_sampler, FakeMaterializer(), rank=1, world_size=2,
        steps_per_epoch_per_rank=5, base_seed=17, prefetch_batches=1,
    )
    synchronous.set_epoch(4)
    prefetched.set_epoch(4)

    assert list(synchronous) == list(prefetched)
    assert synchronous_sampler.seeds == prefetched_sampler.seeds
