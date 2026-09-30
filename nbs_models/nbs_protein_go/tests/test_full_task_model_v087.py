"""Absolute prediction, source-isolation and exact v086 compatibility contracts."""
import copy

import pytest
import torch
import torch.nn.functional as F

from nbs_pg.full_task_model_v086 import FullTaskGraphModelV086, FullTaskModelConfigV086
from nbs_pg.full_task_model_v087 import FullTaskGraphModelV087, FullTaskModelConfigV087
from test_full_task_model import _ontology
from test_full_task_model_v084 import _sampled_batch, _open


def _options(**kwargs):
    result = dict(hidden_dim=16, query_dim=8, decoder_hidden=16, ontology_layers=1,
                  go_chunk=3, candidate_dropout=0., activation_checkpointing=False)
    result.update(kwargs)
    return result


def _model(**kwargs):
    return FullTaskGraphModelV087(6, _ontology(), FullTaskModelConfigV087(**_options(**kwargs)))


@pytest.mark.parametrize("variant", ["legacy", "preln"])
def test_residual_matches_v086_initialization_forward_and_gradients(variant):
    opts = _options(encoder_variant=variant, candidate_dropout=.2)
    torch.manual_seed(8701)
    old = FullTaskGraphModelV086(6, _ontology(), FullTaskModelConfigV086(**opts))
    torch.manual_seed(8701)
    new = FullTaskGraphModelV087(6, _ontology(), FullTaskModelConfigV087(**opts))
    assert old.state_dict().keys() == new.state_dict().keys()
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name
    _open(old)
    new.load_state_dict(old.state_dict())
    batch = _sampled_batch()
    torch.manual_seed(71)
    left = old(batch, return_details=True)
    F.binary_cross_entropy_with_logits(left['logits'], batch['targets']).backward()
    rng = torch.get_rng_state()
    torch.manual_seed(71)
    right = new(batch, return_details=True)
    F.binary_cross_entropy_with_logits(right['logits'], batch['targets']).backward()
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in left.items():
        if value is None:
            assert right[name] is None
        else:
            torch.testing.assert_close(value, right[name], rtol=0, atol=0)
    for (name, p), (name2, q) in zip(old.named_parameters(), new.named_parameters()):
        assert name == name2
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("mode", ["legacy", "preln"])
def test_direct_never_depends_on_dense_base_or_teacher_values(mode):
    model, batch = _model(prediction_mode='direct', encoder_variant=mode).eval(), _sampled_batch()
    # NaNs make an accidentally retained B readout/skip immediately visible.
    changed = dict(batch, base_logits=torch.full_like(batch['base_logits'], float('nan')),
                   targets=1-batch['targets'], positive_mask=~batch['positive_mask'],
                   modelout=torch.randn(4, 8), expert_prob=torch.randn(4, 8))
    a, b = model(batch, return_details=True), model(changed, return_details=True)
    assert a['logits'].shape == (4, 8) and torch.isfinite(a['logits']).all()
    assert a['delta'] is b['delta'] is None
    for name in ('logits', 'graph_logits', 'routing', 'core_vote', 'core_support_mass'):
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert torch.equal(model(batch), a['logits'])
    assert model.correction_decoder[0].in_features == model.config.decoder_hidden + 6
    assert torch.equal(model.direct_go_bias, torch.full((8,), -4.))
    assert len(model.sage_layers) == 2 and not hasattr(model, 'graph_output')


def test_direct_initial_head_reaches_both_pp_layers_go_and_all_evidence():
    torch.manual_seed(8712)
    model, batch = _model(prediction_mode='direct', encoder_variant='preln'), _sampled_batch()
    F.binary_cross_entropy_with_logits(model(batch), batch['targets']).backward()
    for prefix in ('go_layers', 'box_center_encoder', 'query', 'protein_encoder',
                   'sampled_candidate_value', 'sampled_gold_value', 'sampled_pseudo_value',
                   'direct_go_bias'):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), prefix
        assert any(g.norm() > 0 for g in grads), prefix
    for layer in range(2):
        for relation in model.PP_RELATIONS:
            prefix = f'sage_layers.{layer}.{relation}'
            assert any(p.grad is not None and p.grad.norm() > 0
                       for n, p in model.named_parameters() if n.startswith(prefix)), prefix


def _source_batch(model, batch, **kwargs):
    flags = dict(use_weak_go=True, use_query_candidate=True,
                 use_neighbor_candidate=True, use_neighbor_pseudo=True)
    flags.update(kwargs)
    return model._source_batch(batch, **flags)


def test_query_candidate_mask_covers_both_entry_points_without_mutation():
    model, batch = _model().eval(), _sampled_batch()
    original = {k: v.clone() for k, v in batch.items()}
    masked, stats = _source_batch(model, batch, use_query_candidate=False)
    seed = batch['sampled_seed_index']
    assert (masked['candidate_go'] == -1).all()
    assert (masked['sampled_candidate_go'][seed] == -1).all()
    assert not masked['candidate_attr'].any()
    assert not masked['sampled_candidate_attr'][seed].any()
    assert torch.equal(masked['sampled_candidate_go'][4:], batch['sampled_candidate_go'][4:])
    assert torch.equal(masked['sampled_pseudo_edge'], batch['sampled_pseudo_edge'])
    assert stats['source_query_candidate_removed_fraction'] == 1
    for key in batch:
        assert torch.equal(batch[key], original[key]), key
    _open(model)
    changed = dict(batch, candidate_go=(batch['candidate_go']+1) % 8,
                   sampled_candidate_go=batch['sampled_candidate_go'].clone())
    changed['sampled_candidate_go'][seed] = (changed['sampled_candidate_go'][seed]+2) % 8
    torch.testing.assert_close(model(batch, use_query_candidate=False),
                               model(changed, use_query_candidate=False), rtol=0, atol=0)
    assert not torch.allclose(model(batch), model(changed))


def test_neighbor_candidate_and_pseudo_masks_preserve_other_sources():
    model, batch = _model().eval(), _sampled_batch()
    candidates, stats = _source_batch(model, batch, use_neighbor_candidate=False)
    assert (candidates['sampled_candidate_go'][4:] == -1).all()
    assert torch.equal(candidates['sampled_candidate_go'][:4], batch['sampled_candidate_go'][:4])
    assert torch.equal(candidates['candidate_go'], batch['candidate_go'])
    assert torch.equal(candidates['sampled_pseudo_edge'], batch['sampled_pseudo_edge'])
    assert stats['source_neighbor_candidate_removed_fraction'] == 1
    pseudo, stats = _source_batch(model, batch, use_neighbor_pseudo=False)
    assert pseudo['sampled_pseudo_edge'].shape == (2, 0)
    assert torch.equal(pseudo['sampled_candidate_go'], batch['sampled_candidate_go'])
    assert torch.equal(pseudo['sampled_gold_edge'], batch['sampled_gold_edge'])
    assert stats['source_neighbor_pseudo_removed_fraction'] == 1
    _open(model)
    changed = dict(batch, sampled_pseudo_edge=torch.tensor([[10, 12], [6, 0]]))
    torch.testing.assert_close(model(batch, use_neighbor_pseudo=False),
                               model(changed, use_neighbor_pseudo=False), rtol=0, atol=0)
    changed = dict(batch, sampled_candidate_go=batch['sampled_candidate_go'].clone())
    changed['sampled_candidate_go'][4:] = (changed['sampled_candidate_go'][4:]+3) % 8
    torch.testing.assert_close(model(batch, use_neighbor_candidate=False),
                               model(changed, use_neighbor_candidate=False), rtol=0, atol=0)


@pytest.mark.parametrize('prediction', ['residual', 'direct'])
def test_finegrained_all_off_exactly_matches_weak_off_even_with_training_dropout(prediction):
    model, batch = _model(prediction_mode=prediction, candidate_dropout=.25,
                         dropout=.1), _sampled_batch()
    _open(model)
    torch.manual_seed(21)
    weak_off = model(batch, use_weak_go=False)
    rng = torch.get_rng_state()
    torch.manual_seed(21)
    granular = model(batch, use_query_candidate=False,
                     use_neighbor_candidate=False, use_neighbor_pseudo=False)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(weak_off, granular, rtol=0, atol=0)
    # Coarse switch dominates every finer switch.
    torch.manual_seed(21)
    torch.testing.assert_close(weak_off, model(batch, use_weak_go=False,
        use_query_candidate=False, use_neighbor_pseudo=False), rtol=0, atol=0)


def test_node_level_masks_are_shared_stateless_and_resume_reproducible():
    options = dict(prediction_mode='direct', source_dropout_query_candidate=.5,
                   source_dropout_neighbor_candidate=.5, source_dropout_neighbor_pseudo=.5)
    model, batch = _model(**options), _sampled_batch()
    with pytest.raises(RuntimeError, match='set_encoder_step'):
        model(batch)
    model.set_encoder_step(31, rank=1)
    torch.manual_seed(66)
    rng = torch.get_rng_state().clone()
    first = model._source_keep_mask(4096, torch.device('cpu'), 'query_candidate')
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(first, model._source_keep_mask(4096, torch.device('cpu'), 'query_candidate'))
    assert not torch.equal(first, model._source_keep_mask(4096, torch.device('cpu'), 'neighbor_candidate'))
    masked, _ = _source_batch(model, batch)
    keep = model._source_keep_mask(13, torch.device('cpu'), 'query_candidate')[:4]
    assert torch.equal((masked['candidate_go'] >= 0).any(1), keep)
    assert torch.equal((masked['sampled_candidate_go'][:4] >= 0).any(1), keep)
    a = model(batch)
    assert torch.equal(rng, torch.get_rng_state())
    resumed = _model(**options)
    resumed.load_state_dict(copy.deepcopy(model.state_dict()))
    resumed.set_encoder_step(31, rank=1)
    torch.testing.assert_close(a, resumed(batch), rtol=0, atol=0)
    model.set_encoder_step(32, rank=1)
    assert not torch.equal(first, model._source_keep_mask(4096, torch.device('cpu'), 'query_candidate'))
    model.set_encoder_step(31, rank=2)
    assert not torch.equal(first, model._source_keep_mask(4096, torch.device('cpu'), 'query_candidate'))


def test_dropout_probability_one_removes_routes_training_only():
    model, batch = _model(source_dropout_query_candidate=1,
        source_dropout_neighbor_candidate=1, source_dropout_neighbor_pseudo=1), _sampled_batch()
    model.set_encoder_step(4)
    masked, stats = _source_batch(model, batch)
    assert (masked['candidate_go'] == -1).all() and (masked['sampled_candidate_go'] == -1).all()
    assert masked['sampled_pseudo_edge'].shape[1] == 0
    assert all(v == 1 for k, v in stats.items() if k.endswith('_removed_fraction'))
    model.eval()
    full, stats = _source_batch(model, batch)
    assert torch.equal(full['candidate_go'], batch['candidate_go'])
    assert torch.equal(full['sampled_candidate_go'], batch['sampled_candidate_go'])
    assert torch.equal(full['sampled_pseudo_edge'], batch['sampled_pseudo_edge'])
    assert all(v == 0 for k, v in stats.items() if k.endswith('_removed_fraction'))


def test_source_dropout_and_checkpoint_recompute_agree_with_pp_dropedge():
    options = dict(prediction_mode='direct', encoder_variant='preln', pp_edge_dropout=.1,
        candidate_dropout=.2, source_dropout_query_candidate=.3,
        source_dropout_neighbor_candidate=.3, source_dropout_neighbor_pseudo=.3)
    left, right, batch = _model(**options), _model(**options, activation_checkpointing=True), _sampled_batch()
    right.load_state_dict(left.state_dict())
    for model in (left, right):
        model.set_encoder_step(5, rank=1)
    torch.manual_seed(87)
    a = left(batch)
    F.binary_cross_entropy_with_logits(a, batch['targets']).backward()
    rng = torch.get_rng_state()
    torch.manual_seed(87)
    b = right(batch)
    F.binary_cross_entropy_with_logits(b, batch['targets']).backward()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)
    for (name, p), (_, q) in zip(left.named_parameters(), right.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-6, msg=name)


def test_seed_pseudo_label_is_rejected_even_if_route_disabled():
    model, batch = _model(), _sampled_batch()
    batch['sampled_pseudo_edge'] = torch.tensor([[0], [1]])
    with pytest.raises(ValueError, match='seed pseudo labels'):
        model(batch, use_neighbor_pseudo=False)


def test_all_core_or_all_weak_batches_and_empty_evidence_stay_finite():
    model, batch = _model(prediction_mode='direct'), _sampled_batch()
    # A union epoch can emit a role-only local batch; roles are supervision,
    # never model features. Empty evidence still leaves sequence + GO readout.
    all_core = dict(batch, is_weak=torch.zeros(4, dtype=torch.bool))
    all_weak = dict(batch, is_weak=torch.ones(4, dtype=torch.bool))
    torch.testing.assert_close(model(all_core), model(all_weak), rtol=0, atol=0)
    empty = dict(all_core, candidate_go=torch.empty(4, 0, dtype=torch.long),
        candidate_attr=torch.empty(4, 0, 3),
        sampled_candidate_go=torch.empty(13, 0, dtype=torch.long),
        sampled_candidate_attr=torch.empty(13, 0, 3),
        sampled_gold_edge=torch.empty(2, 0, dtype=torch.long),
        sampled_pseudo_edge=torch.empty(2, 0, dtype=torch.long),
        sampled_edge_index=torch.empty(2, 0, dtype=torch.long),
        sampled_edge_type=torch.empty(0, dtype=torch.long), sampled_edge_attr=torch.empty(0, 3),
        anchor_x=torch.empty(0, 6), anchor_go_edge=torch.empty(2, 0, dtype=torch.long),
        sampled_anchor_index=torch.empty(0, dtype=torch.long),
        neighbor_index=torch.empty(4, 0, dtype=torch.long), neighbor_attr=torch.empty(4, 0, 3))
    output = model(empty, return_details=True)
    F.binary_cross_entropy_with_logits(output['logits'], batch['targets']).backward()
    assert output['logits'].shape == (4, 8) and torch.isfinite(output['logits']).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(v == 0 for v in output['source_diagnostics'].values())


@pytest.mark.parametrize('options', [dict(prediction_mode='unknown'),
    dict(source_dropout_query_candidate=-.1), dict(source_dropout_neighbor_candidate=1.1),
    dict(source_dropout_neighbor_pseudo=float('nan')), dict(source_dropout_seed=1.5),
    dict(direct_bias_init=float('inf'))])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        _model(**options)
