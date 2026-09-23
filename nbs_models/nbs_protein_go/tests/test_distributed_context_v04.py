from __future__ import annotations

from nbs_pg.distributed import NBSDistributedConfig, NBSDistributedContext, distributed_seed


def test_single_process_distributed_context():
    context = NBSDistributedContext.from_environment(
        NBSDistributedConfig(enabled=False), initialize=False
    )
    assert context.rank == 0
    assert context.world_size == 1
    assert context.is_main_process
    assert distributed_seed(100, context, by_rank=True) == 100
