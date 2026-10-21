import pytest
import torch

from nbs_pg.config import NBSConfig
from nbs_pg.matcher import NBSGatedDeltaAttnRes
from nbs_pg.types import NBSNeighborhoodHierarchy, NBSQueryCondition


def _hierarchy(num_nodes: int, source_names: tuple[str, ...]):
    hidden = 8
    return NBSNeighborhoodHierarchy(
        final_context=torch.randn(num_nodes, hidden),
        source_contexts=torch.randn(len(source_names), num_nodes, hidden),
        source_names=source_names,
    )


def _condition(num_candidates: int) -> NBSQueryCondition:
    # The two query seeds deliberately live at support rows 4 and 5.  They are
    # invalid in a two-row external candidate hierarchy and reproduce the r4
    # smoke-test failure if routing and scoring index spaces are conflated.
    return NBSQueryCondition(
        base_query=torch.randn(2, 8),
        seed_index=torch.tensor([4, 5]),
        seed_query_index=torch.tensor([0, 1]),
        num_queries=2,
        candidate_index=None,
        base_logits=torch.randn(2, num_candidates),
        candidate_evidence=None,
        expert_prob=None,
        query_go_frequency=torch.tensor([1.0, 3.0]),
        labels=None,
        mask=None,
        confidence=None,
        pseudo_mask=None,
    )


def test_external_scoring_uses_support_hierarchy_for_seed_routing() -> None:
    torch.manual_seed(58)
    names = ("layer:1|relation:similar_to", "layer:2|relation:similar_to")
    matcher = NBSGatedDeltaAttnRes(
        NBSConfig(hidden_dim=8, num_layers=2), num_sources=2
    ).eval()
    support = _hierarchy(6, names)
    external = _hierarchy(2, names)

    with pytest.raises(IndexError, match="seed_index"):
        matcher(external, _condition(2))

    result = matcher(
        external,
        _condition(2),
        return_aux=True,
        routing_hierarchy=support,
    )
    assert result.logits.shape == (2, 2)
    assert result.auxiliary is not None
    assert result.auxiliary["source_weights"].shape == (2, 2)


def test_support_routing_is_invariant_to_external_candidate_count() -> None:
    torch.manual_seed(59)
    names = ("source:0", "source:1")
    matcher = NBSGatedDeltaAttnRes(
        NBSConfig(hidden_dim=8, num_layers=2), num_sources=2
    ).eval()
    support = _hierarchy(6, names)
    condition_two = _condition(2)
    condition_one = NBSQueryCondition(
        **{
            **condition_two.__dict__,
            "base_logits": condition_two.base_logits[:, :1],
        }
    )
    result_two = matcher(
        _hierarchy(2, names),
        condition_two,
        return_aux=True,
        routing_hierarchy=support,
    )
    result_one = matcher(
        _hierarchy(1, names),
        condition_one,
        return_aux=True,
        routing_hierarchy=support,
    )
    assert result_two.auxiliary is not None
    assert result_one.auxiliary is not None
    torch.testing.assert_close(
        result_two.auxiliary["source_weights"],
        result_one.auxiliary["source_weights"],
    )


def test_routing_and_scoring_source_order_must_match() -> None:
    matcher = NBSGatedDeltaAttnRes(
        NBSConfig(hidden_dim=8, num_layers=2), num_sources=2
    ).eval()
    support = _hierarchy(6, ("source:0", "source:1"))
    external = _hierarchy(2, ("source:1", "source:0"))
    with pytest.raises(ValueError, match="same source order"):
        matcher(
            external,
            _condition(2),
            routing_hierarchy=support,
        )
