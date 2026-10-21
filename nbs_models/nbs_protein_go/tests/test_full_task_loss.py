import torch

from nbs_pg.full_task_loss import FullTaskLossConfig, full_task_loss


def test_all_positive_queries_learn_even_outside_a_small_candidate_prefix():
    logits = torch.zeros(2, 257, requires_grad=True)
    labels = torch.zeros_like(logits)
    labels[0, [0, 17, 256]] = 1
    labels[1, [4, 19, 200, 255]] = 0.8
    loss, metrics = full_task_loss(
        logits, logits.detach(), labels, labels > 0,
        torch.tensor([False, True]),
        FullTaskLossConfig(hard_pu_k=4, background_pu_k=4),
    )
    loss.backward()
    assert (logits.grad[labels > 0] < 0).all()
    assert torch.isfinite(logits.grad).all()
    assert metrics["positive_pairs_per_protein"] == 3.5
    assert metrics["positive_coverage"] == 1
    assert metrics["objective_gain"].abs() < 1e-7


def test_pu_budget_never_relabels_known_positives_and_handles_short_rows():
    logits = torch.full((3, 5), 1.0, requires_grad=True)
    positive = torch.tensor([
        [True, True, True, True, True],
        [True, False, True, True, True],
        [False, False, False, False, False],
    ])
    loss, metrics = full_task_loss(
        logits, torch.zeros_like(logits), positive.float(), positive,
        torch.tensor([True, True, False]),
        FullTaskLossConfig(
            positive_weight=0, anchor_weight=0, hard_pu_k=3,
            background_pu_k=8,
        ),
    )
    loss.backward()
    assert (logits.grad[positive] == 0).all()
    assert (logits.grad[~positive] > 0).all()
    assert torch.allclose(metrics["hard_pu_pairs_per_protein"], torch.tensor(4 / 3))
    assert torch.allclose(metrics["background_pu_pairs_per_protein"], torch.tensor(2 / 3))


def test_soft_pseudo_targets_keep_calibration_and_role_weights_are_respected():
    logits = torch.tensor([[2.0, -1.0], [2.0, -1.0]], requires_grad=True)
    labels = torch.tensor([[0.6, 0.0], [1.0, 0.0]])
    config = FullTaskLossConfig(
        core_weight=0, hard_pu_weight=0, background_pu_weight=0,
        anchor_weight=0,
    )
    loss, _ = full_task_loss(
        logits, torch.zeros_like(logits), labels, labels > 0,
        torch.tensor([True, False]), config,
    )
    loss.backward()
    # A 0.6 pseudo target should reduce an overconfident 0.88 prediction.
    assert logits.grad[0, 0] > 0
    assert (logits.grad[1] == 0).all()


def test_role_balance_does_not_change_when_identical_weak_examples_are_repeated():
    config = FullTaskLossConfig(hard_pu_k=0, background_pu_k=0, anchor_weight=0)
    logits = torch.tensor([[-1.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([[0.8, 0.0], [1.0, 0.0]])
    original, _ = full_task_loss(
        logits, logits, labels, labels > 0, torch.tensor([True, False]), config,
    )
    repeat = torch.tensor([0, 0, 0, 0, 1])
    expanded, _ = full_task_loss(
        logits[repeat], logits[repeat], labels[repeat], labels[repeat] > 0,
        torch.tensor([True, True, True, True, False]), config,
    )
    assert torch.allclose(original, expanded)


def test_positive_and_hard_gradients_are_not_diluted_by_unused_go_vocabulary():
    config = FullTaskLossConfig(
        hard_pu_k=1, background_pu_k=0, anchor_weight=0,
    )
    gradients = []
    for width in (3, 21312):
        logits = torch.full((1, width), -8.0)
        logits[0, :2] = torch.tensor([-1.0, 1.0])
        logits.requires_grad_()
        target = torch.zeros_like(logits)
        target[0, 0] = 1
        loss, _ = full_task_loss(
            logits, logits.detach(), target, target > 0,
            torch.tensor([True]), config,
        )
        loss.backward()
        gradients.append(logits.grad[0, :2].clone())
    assert torch.allclose(gradients[0], gradients[1], atol=1e-8)


def test_empty_positive_and_all_positive_rows_are_finite_with_ranking():
    logits = torch.tensor([[20.0, -20.0], [1.0, -1.0]], requires_grad=True)
    positive = torch.tensor([[False, False], [True, True]])
    loss, metrics = full_task_loss(
        logits, torch.zeros_like(logits), positive.float(), positive,
        torch.tensor([True, False]),
        FullTaskLossConfig(ranking_weight=0.1, hard_pu_k=5, background_pu_k=5),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert all(value.ndim == 0 and torch.isfinite(value) for value in metrics.values())


def test_fixed_toy_problem_optimizes_positives_and_hard_unknown_ranking():
    torch.manual_seed(4)
    # The only positive initially ranks last.  Optimisation must move the
    # positive up AND decrease high-scoring unknowns, not simply lift all GOs.
    base = torch.tensor([[-2.0, 2.0, 1.5, 1.0], [-1.5, 1.0, 1.5, 2.0]])
    logits = torch.nn.Parameter(base.clone())
    labels = torch.zeros_like(base)
    labels[:, 0] = torch.tensor([1.0, 0.85])
    config = FullTaskLossConfig(hard_pu_k=3, background_pu_k=0)
    optimizer = torch.optim.Adam([logits], lr=0.12)
    initial, _ = full_task_loss(
        logits, base, labels, labels > 0, torch.tensor([False, True]), config,
    )
    for _ in range(80):
        optimizer.zero_grad()
        loss, metrics = full_task_loss(
            logits, base, labels, labels > 0, torch.tensor([False, True]), config,
        )
        loss.backward()
        optimizer.step()
    assert loss < initial * 0.2
    assert (logits.argmax(1) == 0).all()
    assert (logits[:, 1:] < base[:, 1:] - 1).all()
    assert metrics["objective_gain"] > 0


def test_bfloat16_input_keeps_loss_arithmetic_in_fp32():
    logits = torch.zeros(2, 8, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.zeros_like(logits)
    labels[:, -1] = 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = full_task_loss(
            logits, logits.detach(), labels, labels > 0,
            torch.tensor([True, False]),
        )
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(logits.grad).all()
