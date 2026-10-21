import torch

from nbs_pg.config import NBSConfig
from nbs_pg.matcher import NBSGatedDeltaAttnRes
from nbs_pg.types import NBSNeighborhoodHierarchy, NBSQueryCondition


def make_condition(base_logits):
    return NBSQueryCondition(
        base_query=torch.randn(2, 8),
        seed_index=torch.tensor([0, 1]),
        seed_query_index=torch.tensor([0, 1]),
        num_queries=2,
        candidate_index=None,
        base_logits=base_logits,
        candidate_evidence=None,
        expert_prob=None,
        query_go_frequency=torch.tensor([0.01, 0.2]),
        labels=None,
        mask=None,
        confidence=None,
        pseudo_mask=None,
    )


def test_zero_initialized_graph_delta_is_exact_baseline():
    torch.manual_seed(2)
    cfg = NBSConfig(hidden_dim=8, num_layers=1, graph_delta_scale_init=0.0)
    matcher = NBSGatedDeltaAttnRes(cfg, num_sources=1)
    hierarchy = NBSNeighborhoodHierarchy(
        final_context=torch.randn(4, 8),
        source_contexts=torch.randn(1, 4, 8),
        source_names=("source",),
    )
    base = torch.randn(2, 4)
    out = matcher(hierarchy, make_condition(base), return_aux=True)
    torch.testing.assert_close(out.logits, base)
    assert out.auxiliary is not None
    assert out.auxiliary["graph_delta_scale"].item() == 0.0
