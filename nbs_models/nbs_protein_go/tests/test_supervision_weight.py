import torch

from nbs_pg.losses import NBSLossWeights, nbs_training_loss
from nbs_pg.types import NBSMatchOutput


def test_supervision_weight_can_remove_sampled_unlabelled_negative():
    logits = torch.tensor([[0.0, 10.0]], requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    output = NBSMatchOutput(
        logits=logits,
        labels=labels,
        mask=mask,
        supervision_weight=torch.tensor([[1.0, 0.0]]),
    )
    loss, _ = nbs_training_loss(
        output,
        primary="bce",
        weights=NBSLossWeights(
            pseudo=0.0,
            anchor=0.0,
            hierarchy=0.0,
            delta_l2=0.0,
            routing_balance=0.0,
            null_collapse=0.0,
        ),
    )
    assert torch.allclose(loss, torch.tensor(0.6931472), atol=1e-5)
