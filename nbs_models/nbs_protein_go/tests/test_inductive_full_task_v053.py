import torch

from nbs_pg import NBSConfig, ProteinGONBSModel


def test_external_candidate_encoder_has_correct_shape_and_zero_sources():
    config = NBSConfig(hidden_dim=16, num_layers=2, backbone_type="sage")
    model = ProteinGONBSModel(config, protein_input_dim=8, go_box_dim=4)
    x = torch.randn(5, 8)
    hierarchy = model.encode_external_protein_candidates(x)
    assert hierarchy.final_context.shape == (5, 16)
    assert hierarchy.source_contexts.shape[1:] == (5, 16)
    assert hierarchy.source_contexts.shape[0] == model.backbone.num_target_sources
    assert torch.count_nonzero(hierarchy.source_contexts) == 0
