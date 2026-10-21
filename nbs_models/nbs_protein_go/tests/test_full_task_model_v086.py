"""v086 controls, locality and stateless PP DropEdge contracts."""
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from nbs_pg.full_task_model_v084 import FullTaskGraphModelV084, FullTaskModelConfigV084
from nbs_pg.full_task_model_v086 import FullTaskGraphModelV086, FullTaskModelConfigV086
from test_full_task_model import _ontology
from test_full_task_model_v084 import _sampled_batch, _chain_batch, _open


def _options(**kwargs):
    result = dict(hidden_dim=16, query_dim=8, decoder_hidden=16, ontology_layers=1,
                  go_chunk=3, candidate_dropout=0., activation_checkpointing=False)
    result.update(kwargs)
    return result


def _model(variant="preln", **kwargs):
    return FullTaskGraphModelV086(6, _ontology(),
                                 FullTaskModelConfigV086(encoder_variant=variant, **_options(**kwargs)))


def test_legacy_is_exact_v084_in_initialization_outputs_and_gradients():
    torch.manual_seed(86)
    old = FullTaskGraphModelV084(6, _ontology(), FullTaskModelConfigV084(**_options()))
    torch.manual_seed(86)
    new = _model("legacy")
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    batch = _sampled_batch()
    _open(old)
    new.load_state_dict(old.state_dict())
    left, right = old(batch), new(batch)
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    F.binary_cross_entropy_with_logits(left, batch['targets']).backward()
    F.binary_cross_entropy_with_logits(right, batch['targets']).backward()
    for (name, p), (other, q) in zip(old.named_parameters(), new.named_parameters()):
        assert name == other
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    assert asdict(new.config)['encoder_variant'] == 'legacy'
    assert asdict(new.config)['pp_edge_dropout'] == 0


def test_preln_preserves_common_initialization_and_zero_residual():
    torch.manual_seed(86)
    legacy = _model('legacy')
    torch.manual_seed(86)
    tuned = _model()
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, tuned.state_dict()[name]), name
    additional = sum(p.numel() for name, p in tuned.named_parameters()
                     if name.startswith('pp_source_norms.'))
    assert additional == 4 * tuned.config.hidden_dim  # 512 at hidden_dim=128.
    assert torch.equal(tuned(_sampled_batch()), _sampled_batch()['base_logits'])


def test_nonzero_head_propagates_gradients_into_both_norms_and_relations():
    torch.manual_seed(861)
    model, batch = _model(), _sampled_batch()
    _open(model)
    F.binary_cross_entropy_with_logits(model(batch), batch['targets']).backward()
    for prefix in ('pp_source_norms.0', 'pp_source_norms.1', 'sampled_candidate_value',
                   'sampled_gold_value', 'sampled_pseudo_value'):
        gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert all(g.norm() > 0 for g in gradients), prefix
    for layer in range(2):
        for relation in model.PP_RELATIONS:
            prefix = f'sage_layers.{layer}.{relation}'
            assert any(p.grad is not None and p.grad.norm() > 0
                       for n, p in model.named_parameters() if n.startswith(prefix)), prefix


def test_go_to_core2_to_core1_to_seed_remains_learnable():
    torch.manual_seed(864)
    model, batch = _model(), _chain_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    start = F.binary_cross_entropy_with_logits(model(batch), batch['targets']).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(batch), batch['targets'])
        loss.backward()
        optimizer.step()
    model.eval()
    full = F.binary_cross_entropy_with_logits(model(batch), batch['targets']).item()
    off = F.binary_cross_entropy_with_logits(model(batch, use_pp_context=False), batch['targets']).item()
    assert full < start * .5 and off > full + .1, (start, full, off)


def test_anchor_readout_excludes_third_hop():
    torch.manual_seed(862)
    model = _model().eval()
    _open(model)
    batch = _chain_batch()
    for key in ('protein_x', 'base_logits', 'candidate_go', 'candidate_attr',
                'targets', 'positive_mask', 'is_weak'):
        batch[key] = batch[key][:1]
    batch.update(anchor_x=torch.zeros(1, 6), anchor_go_edge=torch.tensor([[0], [0]]),
                 neighbor_index=torch.tensor([[0]]), neighbor_attr=torch.ones(1, 1, 3),
                 sampled_protein_x=torch.zeros(4, 6), sampled_seed_index=torch.tensor([0]),
                 sampled_anchor_index=torch.tensor([1]),
                 sampled_candidate_go=torch.empty(4, 0, dtype=torch.long),
                 sampled_candidate_attr=torch.empty(4, 0, 3),
                 sampled_gold_edge=torch.tensor([[1, 2, 3], [0, 2, 4]]),
                 sampled_edge_index=torch.tensor([[2, 1], [1, 0]]),
                 sampled_edge_type=torch.tensor([0, 3]), sampled_edge_attr=torch.ones(2, 3))
    expanded = dict(batch, sampled_edge_index=torch.tensor([[2, 1, 3], [1, 0, 2]]),
                    sampled_edge_type=torch.tensor([0, 3, 1]), sampled_edge_attr=torch.ones(3, 3))
    torch.testing.assert_close(model(batch), model(expanded), atol=1e-7, rtol=0)


def test_empty_pp_is_identity_and_first_layer_keeps_confidence_attenuation():
    torch.manual_seed(863)
    model, batch = _model().eval(), _sampled_batch()
    go = model.encode_go()
    flags = dict(use_weak_go=True, use_core_go=True, shuffle_go=False)
    initial = model._encode_sampled(batch, go, use_pp_context=False, **flags)[0]
    empty = dict(batch, sampled_edge_index=torch.empty(2, 0, dtype=torch.long),
                 sampled_edge_type=torch.empty(0, dtype=torch.long), sampled_edge_attr=torch.empty(0, 3))
    state, anchor, norm, count = model._encode_sampled(empty, go, use_pp_context=True, **flags)
    assert torch.equal(state, initial) and torch.equal(anchor, initial)
    assert norm == 0 and count == 0
    captured = []
    handle = model.sage_layers[0]['ppi'].register_forward_hook(
        lambda module, args, result: captured.append(result[0].detach().clone()))
    model._encode_sampled(batch, go, use_pp_context=True, **flags)
    weak_attr = batch['sampled_edge_attr'].clone()
    # Original positive confidences are <=1.2: use an unambiguous baseline.
    full_attr = weak_attr.clone()
    full_attr[:, 0] = 1.
    model._encode_sampled(dict(batch, sampled_edge_attr=full_attr), go, use_pp_context=True, **flags)
    weak_attr[:, 0] = .001
    model._encode_sampled(dict(batch, sampled_edge_attr=weak_attr), go, use_pp_context=True, **flags)
    handle.remove()
    # Gates initialize to one; per-node pre-LN does not cancel confidence.
    torch.testing.assert_close(captured[-1], captured[-2] * .001, atol=1e-8, rtol=1e-5)


def test_external_predictions_ignore_batch_partition_and_dropout_in_eval(tmp_path):
    from test_full_task_data_v084 import data_v084, inference_fixture
    data = data_v084(tmp_path, holdout=1, variant='dynamic', dropout=.75)
    data.prepare_neighbors()
    input_dir = inference_fixture(tmp_path, data)
    ontology = _ontology()
    ontology['task_to_ontology'] = torch.arange(data.num_task_go)
    model = FullTaskGraphModelV086(data.feature_dim, ontology,
        FullTaskModelConfigV086(encoder_variant='preln', pp_edge_dropout=.1, **_options())).eval()
    _open(model)
    for shuffle in (False, True):
        combined = model(data.inference_batch(input_dir, [0, 1]), shuffle_go=shuffle)
        separate = torch.cat([model(data.inference_batch(input_dir, [r]), shuffle_go=shuffle)
                              for r in (0, 1)])
        torch.testing.assert_close(combined, separate, atol=1e-6, rtol=1e-5)


def test_dropedge_masks_are_stateless_distinct_and_do_not_consume_global_rng():
    model = _model(pp_edge_dropout=.1).train()
    with pytest.raises(RuntimeError, match='set_encoder_step'):
        model(_sampled_batch())
    model.set_encoder_step(10, rank=1)
    torch.manual_seed(11)
    rng = torch.get_rng_state().clone()
    first = model._pp_keep_mask(4096, torch.device('cpu'), 0)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(first, model._pp_keep_mask(4096, torch.device('cpu'), 0))
    assert not torch.equal(first, model._pp_keep_mask(4096, torch.device('cpu'), 1))
    model.set_encoder_step(11, rank=1)
    assert not torch.equal(first, model._pp_keep_mask(4096, torch.device('cpu'), 0))
    model.set_encoder_step(10, rank=2)
    assert not torch.equal(first, model._pp_keep_mask(4096, torch.device('cpu'), 0))
    model.set_encoder_step(10, rank=1)
    batch = _sampled_batch()
    _open(model)
    before = torch.get_rng_state().clone()
    a, b = model(batch), model(batch)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_dropedge_preserves_direct_vote_and_reports_mean_retained_edges():
    model, batch = _model(pp_edge_dropout=.1), _sampled_batch()
    model.set_encoder_step(51)
    output = model(batch, return_details=True)
    valid = batch['sampled_edge_attr'][:, 0] > 0
    counts = [(model._pp_keep_mask(len(valid), valid.device, layer) & valid).sum()
              for layer in range(2)]
    assert output['pp_edge_count'] == torch.stack(counts).float().mean()
    model.eval()
    stable = model(batch, return_details=True)
    torch.testing.assert_close(output['core_vote'], stable['core_vote'], atol=0, rtol=0)
    assert stable['pp_edge_count'] == valid.sum()


def test_forward_ignores_targets_and_teacher_probability():
    model, batch = _model().eval(), _sampled_batch()
    _open(model)
    changed = dict(batch, targets=1-batch['targets'], positive_mask=~batch['positive_mask'],
                   is_weak=~batch['is_weak'], expert_prob=torch.rand(4, 8), modelout=torch.rand(4, 8))
    torch.testing.assert_close(model(batch), model(changed), atol=0, rtol=0)


def test_decoder_checkpoint_with_dropedge_preserves_outputs_gradients_and_rng():
    left, batch = _model(pp_edge_dropout=.1, candidate_dropout=.2), _sampled_batch()
    right = _model(pp_edge_dropout=.1, candidate_dropout=.2, activation_checkpointing=True)
    _open(left)
    right.load_state_dict(left.state_dict())
    left.set_encoder_step(5, rank=1)
    right.set_encoder_step(5, rank=1)
    torch.manual_seed(86)
    a = left(batch)
    F.binary_cross_entropy_with_logits(a, batch['targets']).backward()
    rng = torch.get_rng_state()
    torch.manual_seed(86)
    b = right(batch)
    F.binary_cross_entropy_with_logits(b, batch['targets']).backward()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for (name, p), (_, q) in zip(left.named_parameters(), right.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5, msg=name)


@pytest.mark.parametrize('options', [dict(encoder_variant='other'),
    dict(encoder_variant='legacy', pp_edge_dropout=.1), dict(pp_edge_dropout=1),
    dict(pp_edge_dropout=-.1), dict(pp_edge_dropout=float('nan')), dict(pp_dropout_seed=1.5)])
def test_invalid_new_config_rejected(options):
    with pytest.raises(ValueError):
        FullTaskGraphModelV086(6, _ontology(), FullTaskModelConfigV086(**_options(**options)))


def test_preln_empty_graph_keeps_fp32_small_correction_under_bfloat16():
    model, batch = _model(pp_edge_dropout=.1), _chain_batch()
    model.set_encoder_step(1)
    batch.update(sampled_edge_index=torch.empty(2, 0, dtype=torch.long),
                 sampled_edge_type=torch.empty(0, dtype=torch.long),
                 sampled_edge_attr=torch.empty(0, 3))
    batch['base_logits'].fill_(2)
    with torch.no_grad():
        model.correction_decoder[-1].bias.fill_(.000411)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(batch, return_details=True)
        F.binary_cross_entropy_with_logits(output['logits'], batch['targets']).backward()
    assert output['logits'].dtype == torch.float32
    assert torch.all(output['logits'] > 2.0004)
    assert output['pp_edge_count'] == 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
