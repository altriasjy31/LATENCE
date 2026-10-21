"""Requires torch-geometric."""

import torch

from nbs_pg import NBSConfig, ProteinGONBSModel
from examples.toy_train import make_toy_graph


def test_relation_sources_add_to_target_residual():
    graph = make_toy_graph()
    cfg = NBSConfig(
        hidden_dim=32,
        num_layers=2,
        go_tower_layers=0,
        source_mode="layer_relation",
        residual_dropout=0.25,
    )
    model = ProteinGONBSModel(
        cfg,
        protein_input_dim=graph["protein"].x.size(1),
        go_box_dim=graph["go"].center.size(1),
    )
    model.train()
    box = model._encode_local_box(graph)
    edge_index_dict, edge_attr_dict = model._edge_dicts(graph)
    raw = model.backbone(
        {"protein": graph["protein"].x, "go": box.static},
        edge_index_dict,
        edge_attr_dict,
    )
    incoming = len(model.backbone.incoming_target_relations)
    for layer in range(cfg.num_layers):
        start = layer * incoming
        relation_sum = torch.stack(raw.target_sources[start : start + incoming]).sum(0)
        torch.testing.assert_close(relation_sum, raw.layer_deltas[layer]["protein"])
