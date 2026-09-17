import pytest
import torch

from nbs_pg.full_task_loss_v082 import FullTaskLossConfigV082, full_task_loss_v082


def call(z, teacher, *, weak=None, base=None, targets=None, cfg=None, available=None, **kwargs):
    weak = torch.ones(z.shape[0], dtype=torch.bool) if weak is None else weak
    targets = torch.zeros_like(z) if targets is None else targets
    return full_task_loss_v082(
        z, z.detach() if base is None else base, targets, targets > 0, weak,
        cfg or FullTaskLossConfigV082(), teacher_prob=teacher,
        teacher_available=weak if available is None else available, **kwargs,
    )


def test_subthreshold_modelout_is_soft_teacher_not_unlabelled_negative_or_base_anchor():
    teacher = torch.tensor([[.49, .18]], requires_grad=True)
    gradients = []
    for base_value in (-5., 5.):
        z = torch.full((1, 2), -2., requires_grad=True)
        loss, metrics = call(z, teacher, base=torch.full_like(z, base_value))
        loss.backward()
        assert (z.grad < 0).all()  # Every target exceeds sigmoid(-2).
        assert metrics['contrib_core'] == 0
        assert metrics['hard_pu_pairs_per_protein'] == 0
        gradients.append(z.grad.clone())
    assert torch.equal(gradients[0], gradients[1])
    assert teacher.grad is None
    optimum = torch.logit(teacher.detach()).requires_grad_()
    loss, _ = call(optimum, teacher)
    loss.backward()
    assert loss < 1e-6
    assert optimum.grad.abs().max() < 1e-6


def test_confidence_strata_prevent_easy_nearzero_classes_from_diluting_useful_signal():
    gradients = []
    for width in (3, 21312):
        teacher = torch.full((1, width), .001)
        teacher[0, :2] = torch.tensor([.49, .9])
        z = torch.zeros_like(teacher, requires_grad=True)
        loss, metrics = call(z, teacher)
        loss.backward()
        gradients.append(z.grad[0, :2].clone())
        assert (z.grad[0, 2:] > 0).all()
        assert metrics['teacher_pairs_per_weak_protein'] == width
    assert torch.allclose(gradients[0], gradients[1], atol=1e-8)


def test_weak_and_core_role_means_are_independent_of_batch_proportions():
    cfg = FullTaskLossConfigV082(background_pu_k=0, hard_pu_k=2)
    z = torch.tensor([[1., -1.], [1., -1.]])
    teacher = torch.tensor([[.1, .8], [0., 0.]])
    gold = torch.tensor([[0., 0.], [1., 0.]])
    weak = torch.tensor([True, False])
    first, parts = call(z, teacher, weak=weak, targets=gold, cfg=cfg)
    repeat = torch.tensor([0, 0, 0, 0, 1])
    second, parts_repeat = call(z[repeat], teacher[repeat], weak=weak[repeat], targets=gold[repeat], cfg=cfg)
    assert torch.allclose(first, second)
    assert parts['weak_role_mass'] == parts_repeat['weak_role_mass'] == .5
    assert parts['core_role_mass'] == parts_repeat['core_role_mass'] == .5


def test_core_gold_and_alias_exclusion_preserved_and_support_attenuates_pu():
    gradients = []
    for support_value in (0., 1.):
        z = torch.zeros(1, 4, requires_grad=True)
        gold = torch.tensor([[1., 0., 0., 0.]])
        excluded = torch.tensor([[False, False, True, False]])
        support = torch.full_like(z, support_value, requires_grad=True)
        loss, parts = call(z, torch.zeros_like(z), weak=torch.tensor([False]), targets=gold,
                           structural_support=support, pu_exclusion_mask=excluded,
                           cfg=FullTaskLossConfigV082(hard_pu_k=4, background_pu_k=0))
        loss.backward()
        assert z.grad[0, 0] < 0
        assert z.grad[0, 2] == 0
        assert (z.grad[0, [1, 3]] > 0).all()
        assert support.grad is None
        assert parts['teacher_kl'] == 0
        gradients.append(z.grad.clone())
    assert torch.allclose(gradients[1][0, [1, 3]], gradients[0][0, [1, 3]] * .25)
    assert torch.equal(gradients[0][:, :1], gradients[1][:, :1])


def test_classification_has_full_supervision_refinement_has_only_selected_gradient():
    classification = torch.zeros(1, 5, requires_grad=True)
    final = torch.zeros_like(classification, requires_grad=True)
    selected = torch.tensor([[False, True, False, True, False]])
    teacher = torch.tensor([[.9, .8, .4, .2, .01]])
    loss, parts = call(final, teacher, classification_logits=classification, selected_mask=selected,
                       cfg=FullTaskLossConfigV082(refine_weight=1))
    loss.backward()
    assert (classification.grad != 0).all()
    assert (final.grad[selected] != 0).all()
    assert (final.grad[~selected] == 0).all()
    assert parts['selected_pairs_per_protein'] == 2
    assert parts['selected_high_teacher_coverage'] == .5
    assert torch.allclose(loss, parts['contrib_teacher'] + parts['contrib_core'] + parts['contrib_refine'])


@pytest.mark.parametrize('core_positive', [False, True])
def test_empty_selection_and_empty_or_all_core_positives_stay_finite(core_positive):
    final = torch.tensor([[20., -20.], [1., -1.]], requires_grad=True)
    classification = final.detach().clone().requires_grad_()
    teacher = torch.tensor([[0., 1.], [0., 0.]])
    gold = torch.zeros_like(final)
    gold[1] = float(core_positive)
    loss, parts = call(final, teacher, weak=torch.tensor([True, False]), targets=gold,
                       classification_logits=classification, selected_mask=torch.zeros_like(final, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(classification.grad).all()
    assert (final.grad == 0).all()
    assert all(value.ndim == 0 and torch.isfinite(value) for value in parts.values())
    assert parts['contrib_refine'] == 0


def test_baseline_comparison_uses_identical_pairs_and_refinement_is_separate():
    torch.manual_seed(1)
    classification = torch.tensor([[2., -1., 0.], [-2., 1., 0.]], requires_grad=True)
    final = classification + torch.tensor([[0., .2, 0.], [0., .2, 0.]])
    teacher = torch.tensor([[.9, .3, .01], [0., 0., 0.]])
    gold = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    loss, parts = call(final, teacher, base=classification.detach(), weak=torch.tensor([True, False]), targets=gold,
                       classification_logits=classification, selected_mask=torch.tensor([[False, True, False]] * 2))
    assert parts['gain_vs_backbone_objective'].abs() < 1e-7
    assert parts['contrib_refine'] > 0
    assert torch.allclose(loss, parts['fit_loss_comparable'] + parts['contrib_refine'])
    assert parts['gain_excludes_refine'] == 1


def test_missing_teacher_rows_or_invalid_probabilities_fail_instead_of_falling_back():
    z = torch.zeros(2, 3)
    teacher = torch.full_like(z, .2)
    with pytest.raises(ValueError, match='complete, aligned'):
        call(z, teacher, available=torch.tensor([True, False]))
    teacher[0, 1] = float('nan')
    with pytest.raises(ValueError, match='teacher_prob'):
        call(z, teacher)
    # Core rows do not consume the teacher; missing core values are harmless.
    teacher[0] = .2
    teacher[1] = float('nan')
    loss, _ = call(z, teacher, weak=torch.tensor([True, False]))
    assert torch.isfinite(loss)


def test_bfloat16_logits_keep_fp32_objective_and_nonzero_soft_target_gradient():
    z = torch.full((1, 3), -1., dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.tensor([[.49, .8, .01]])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, _ = call(z, teacher)
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(z.grad).all()
    assert z.grad[0, 0] < 0


def test_toy_optimization_recovers_teacher_order_and_continuous_subthreshold_probability():
    teacher = torch.tensor([[.95, .85, .49, .02], [.03, .4, .8, .92]])
    base = torch.tensor([[-2., -1., 2., 3.], [3., 2., -1., -2.]])
    logits = torch.nn.Parameter(base.clone())
    opt = torch.optim.Adam([logits], lr=.15)
    first = None
    for step in range(180):
        opt.zero_grad()
        loss, _ = call(logits, teacher, base=base)
        first = loss.detach() if first is None else first
        loss.backward()
        opt.step()
    assert loss < first * .002
    assert torch.max((logits.sigmoid() - teacher).abs()) < .012
    assert torch.equal(logits.argsort(1), teacher.argsort(1))
    assert .47 < logits.sigmoid()[0, 2] < .51


@pytest.mark.parametrize('options', [dict(refine_weight=-1), dict(teacher_low_threshold=.7),
                                     dict(support_pu_floor=0), dict(hard_pu_k=-1),
                                     dict(weak_weight=0, core_weight=0)])
def test_invalid_config_rejected(options):
    with pytest.raises(ValueError):
        FullTaskLossConfigV082(**options)
