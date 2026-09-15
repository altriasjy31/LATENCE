from __future__ import annotations

import torch

from nbs_pg import NBSConfig, ProteinGONBSModel


def test_external_candidate_neighbor_mask_supports_feature_only_rows():
    model = ProteinGONBSModel(
        NBSConfig(
            hidden_dim=8,
            num_layers=2,
            backbone_type="sage",
            go_tower_layers=0,
        ),
        protein_input_dim=4,
        go_box_dim=3,
    )
    protein_x = torch.randn(2, 4)
    neighbor_x = torch.randn(2, 3, 4)
    neighbor_attr = torch.rand(2, 3, 3)
    neighbor_mask = torch.tensor(
        [[True, True, False], [False, False, False]]
    )
    hierarchy = model.encode_external_protein_candidates(
        protein_x,
        neighbor_x=neighbor_x,
        neighbor_edge_attr=neighbor_attr,
        neighbor_mask=neighbor_mask,
        neighbor_fanouts=(3, 2),
    )
    assert torch.isfinite(hierarchy.final_context).all()
    assert torch.isfinite(hierarchy.source_contexts).all()
    assert torch.count_nonzero(hierarchy.source_contexts[:, 1]) == 0
