import torch

from nbs_pg.config import NBSConfig
from nbs_pg.matcher import NBSGatedDeltaAttnRes
from nbs_pg.types import NBSNeighborhoodHierarchy, NBSQueryCondition


def _condition(expert):
    return NBSQueryCondition(
        base_query=torch.randn(1, 8),
        seed_index=torch.tensor([0]),
        seed_query_index=torch.tensor([0]),
        num_queries=1,
        candidate_index=None,
        base_logits=torch.randn(1, 3),
        candidate_evidence=torch.full((1, 3), 0.7),
        expert_prob=expert,
        query_go_frequency=torch.tensor([3.0]),
        labels=None,
        mask=None,
        confidence=None,
        pseudo_mask=None,
    )


def test_default_delta_gate_is_expert_free():
    torch.manual_seed(5)
    cfg = NBSConfig(hidden_dim=8, num_layers=1, delta_gate_feature_mode="student_candidate")
    matcher = NBSGatedDeltaAttnRes(cfg, num_sources=1)
    matcher.graph_delta_scale.data.fill_(0.5)
    hierarchy = NBSNeighborhoodHierarchy(
        final_context=torch.randn(3, 8),
        source_contexts=torch.randn(1, 3, 8),
        source_names=("source",),
    )
    c1 = _condition(torch.zeros(1, 3))
    c2 = NBSQueryCondition(**{**c1.__dict__, "expert_prob": torch.ones(1, 3)})
    out1 = matcher(hierarchy, c1)
    out2 = matcher(hierarchy, c2)
    torch.testing.assert_close(out1.logits, out2.logits)
