import torch

from nbs_pg.config import NBSConfig
from nbs_pg.go_encoder import BoxSquaredGOEncoder
from nbs_pg.query import NBSProteinGOQueryEncoder
from nbs_pg.types import NBSGOBoxCache, ProteinGOQueryBatch


def test_query_can_use_global_go_cache():
    torch.manual_seed(4)
    cfg = NBSConfig(hidden_dim=8, go_stat_dim=6, go_tower_layers=0)
    box_encoder = BoxSquaredGOEncoder(cfg, box_dim=4)
    center = torch.randn(6, 4)
    offset = torch.rand(6, 4) + 0.01
    stats = torch.randn(6, 6)
    box = box_encoder(center, offset, stats)
    cache = NBSGOBoxCache(
        semantic=box.semantic,
        hierarchy=box.hierarchy,
        static=box.static,
        context=box.static + 0.1,
        center=center,
        offset=offset,
        stats=stats,
    )
    query = ProteinGOQueryBatch(
        seed_protein_index=torch.tensor([0, 1]),
        seed_query_index=torch.tensor([0, 0]),
        num_queries=1,
        query_go_global_index=torch.tensor([2, 3]),
        go_query_index=torch.tensor([0, 0]),
    )
    encoder = NBSProteinGOQueryEncoder(cfg)
    local_box = box_encoder(center[:2], offset[:2], stats[:2])
    condition = encoder(
        torch.randn(5, 8),
        torch.randn(2, 8),
        local_box,
        query,
        global_cache=cache,
    )
    assert condition.base_query.shape == (1, 8)
