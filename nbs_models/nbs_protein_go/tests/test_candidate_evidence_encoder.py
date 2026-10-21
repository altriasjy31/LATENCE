import torch

from nbs_pg.config import NBSConfig
from nbs_pg.matcher import NBSGatedDeltaAttnRes


def test_three_column_candidate_evidence_starts_from_reciprocal_rank():
    matcher = NBSGatedDeltaAttnRes(
        NBSConfig(hidden_dim=8, num_layers=1, candidate_evidence_dim=3),
        num_sources=1,
    )
    base_prob = torch.full((1, 3), 0.4)
    evidence = torch.tensor(
        [[[0.9, 4.0, 1.0], [0.1, -3.0, 0.5], [0.7, 2.0, 0.25]]]
    )
    encoded = matcher._encode_candidate_evidence(evidence, base_prob)
    assert torch.allclose(encoded, evidence[..., 2], atol=1e-5)


def test_scalar_candidate_evidence_remains_supported():
    matcher = NBSGatedDeltaAttnRes(
        NBSConfig(hidden_dim=8, num_layers=1), num_sources=1
    )
    base_prob = torch.full((2, 2), 0.4)
    evidence = torch.tensor([[0.1, 0.5], [0.8, 1.2]])
    encoded = matcher._encode_candidate_evidence(evidence, base_prob)
    assert torch.allclose(encoded, evidence.clamp(0, 1))
