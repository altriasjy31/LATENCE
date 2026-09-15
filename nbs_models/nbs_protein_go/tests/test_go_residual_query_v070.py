from __future__ import annotations

import torch

from nbs_pg.config import NBSConfig
from nbs_pg.query import NBSProteinGOQueryEncoder
from nbs_pg.types import BoxGOEncoding, ProteinGOQueryBatch


def test_go_tower_layer_deltas_are_attended_into_query():
    config = NBSConfig(
        hidden_dim=8,
        num_layers=1,
        go_tower_layers=2,
        use_go_residual_query=True,
    )
    encoder = NBSProteinGOQueryEncoder(config)
    protein_context = torch.randn(4, 8)
    go_context = torch.randn(2, 8)
    center = torch.randn(2, 3)
    offset = torch.rand(2, 3) + 0.1
    local_box = BoxGOEncoding(
        semantic=torch.randn(2, 8),
        hierarchy=torch.randn(2, 8),
        static=torch.randn(2, 8),
        hierarchy_gate=torch.ones(2, 8),
        center=center,
        offset=offset,
        log_offset=torch.log(offset),
        tower_deltas=torch.randn(2, 2, 8),
    )
    query = ProteinGOQueryBatch(
        seed_protein_index=torch.tensor([0, 1, 2, 3]),
        seed_query_index=torch.tensor([0, 0, 1, 1]),
        num_queries=2,
        query_go_index=torch.tensor([0, 1]),
        go_query_index=torch.tensor([0, 1]),
    )
    condition = encoder(protein_context, go_context, local_box, query)
    assert condition.auxiliary is not None
    assert condition.auxiliary["go_tower_layer_attention"].shape == (2, 2)
    torch.testing.assert_close(
        condition.auxiliary["go_tower_layer_attention"].sum(dim=1),
        torch.ones(2),
    )
    assert torch.count_nonzero(condition.auxiliary["go_residual_gate"]) > 0
