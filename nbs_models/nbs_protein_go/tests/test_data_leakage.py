"""Requires torch-geometric."""

import torch

from nbs_pg import mask_candidate_evidence_edges
from examples.toy_train import make_toy_graph


def test_candidate_annotations_are_removed_in_both_directions():
    graph = make_toy_graph()
    candidates = torch.arange(24, 48)
    masked = mask_candidate_evidence_edges(
        graph,
        candidates,
        query_go_index=torch.tensor([0, 1]),
        gold_mode="all",
        pseudo_mode="all",
        candidate_mode="query_only",
    )
    for relation in ("gold_annotated_with", "pseudo_annotated_with"):
        edge = masked[("protein", relation, "go")].edge_index
        assert not torch.isin(edge[0], candidates).any()
    for relation in ("has_gold_annotation", "has_pseudo_annotation"):
        edge = masked[("go", relation, "protein")].edge_index
        assert not torch.isin(edge[1], candidates).any()
