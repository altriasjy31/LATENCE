import torch

from nbs_pg.losses import hierarchy_violation_loss, query_hierarchy_violation_loss


def test_query_axis_hierarchy_loss_uses_go_rows():
    probabilities = torch.tensor(
        [
            [0.9, 0.2],  # child query
            [0.8, 0.3],  # parent query
        ]
    )
    edges = torch.tensor([[0], [1]])
    loss = query_hierarchy_violation_loss(probabilities, edges)
    assert torch.allclose(loss, torch.tensor(0.05))
    assert torch.allclose(
        loss,
        hierarchy_violation_loss(probabilities, edges, go_axis=0),
    )


def test_conventional_hierarchy_loss_keeps_go_columns():
    probabilities = torch.tensor([[0.9, 0.8], [0.2, 0.3]])
    edges = torch.tensor([[0], [1]])
    loss = hierarchy_violation_loss(probabilities, edges, go_axis=1)
    assert torch.allclose(loss, torch.tensor(0.05))
