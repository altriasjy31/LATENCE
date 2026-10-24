"""Dynamic labels, exact graph identity, migration and contamination contracts."""
import copy

import pytest
import torch
import torch.nn.functional as F

from nbs_pg.full_task_model_v088 import FullTaskGraphModelV088, FullTaskModelConfigV088
from nbs_pg.full_task_model_v090 import (FullTaskGraphModelV090, FullTaskModelConfigV090,
                                       migrate_encoder_state_dict)
from test_full_task_model import _ontology
from test_full_task_model_v088 import _batch


def _model(ontology=None, seen=None, **kwargs):
    options = dict(hidden_dim=16, ontology_layers=1, input_dropout=0., dropout=0.)
    options.update(kwargs)
    return FullTaskGraphModelV090(6, _ontology() if ontology is None else ontology,
        FullTaskModelConfigV090(**options), seen_task_mask=seen)


def test_closed_default_order_aliases_and_dynamic_target_order_share_one_scorer():
    ontology = _ontology()
    ontology['task_to_ontology'] = torch.tensor([4, 1, 4, 2, 8, 5, 6, 7])
    model, batch = _model(ontology).eval(), _batch()
    default = model(batch)
    assert default.shape == (2, 8)
    torch.testing.assert_close(default[:, 0], default[:, 2], rtol=0, atol=0)
    dynamic = model(batch, target_ontology_ids=[7, 4, 9, 1, 4])
    torch.testing.assert_close(dynamic[:, [0, 1, 3, 4]], default[:, [7, 0, 1, 2]])
    assert not hasattr(model, 'classifier')
    assert sum(p.numel() for p in model.scorer.parameters()) == 16*16+16+1
    with pytest.raises(ValueError, match='no per-GO'):
        model.initialize_output_prior(torch.ones(8)*.1)


def test_supplied_new_node_geometry_is_legal_without_new_parameter_rows():
    original = _model().eval()
    extended = _ontology()
    extended['center'] = torch.cat([extended['center'], torch.full((1, 6), .25)])
    extended['offset'] = torch.cat([extended['offset'], torch.full((1, 6), .6)])
    # New label-side node and legal relations are supplied explicitly.
    extended['edges']['is_a'] = torch.cat([extended['edges']['is_a'], torch.tensor([[10], [8]])], dim=1)
    extended['edges']['has_child'] = torch.cat([extended['edges']['has_child'], torch.tensor([[8], [10]])], dim=1)
    added = _model(extended).eval()
    added.load_state_dict(original.state_dict())
    result = added(_batch(), target_ontology_ids=[10, 0])
    assert result.shape == (2, 2) and torch.isfinite(result).all()
    assert sum(p.numel() for p in added.parameters()) == sum(p.numel() for p in original.parameters())


@pytest.mark.parametrize('chunk', [1, 2, 3, 8, 100])
def test_chunking_and_full_cache_preserve_logits_and_target_column_order(chunk):
    model, batch = _model().eval(), _batch()
    expected = model(batch, target_ontology_ids=[9, 0, 8, 4, 0], go_chunk_size=100)
    full = model.encode_ontology()
    actual = model(batch, target_ontology_ids=[9, 0, 8, 4, 0],
                   ontology_encoding=full, go_chunk_size=chunk)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model(batch, go_encoding=model.encode_go()), model(batch))
    model.train()
    with pytest.raises(ValueError, match='cached'):
        model(batch, ontology_encoding=full)


def test_jk_sum_uses_every_nonprefix_identity_mapping_and_keeps_raw_residual():
    model, batch = _model(jk_mode='sum').eval(), _batch()
    post = []
    handles = [norm.register_forward_hook(lambda module, args, output:
        post.append(F.relu(output).detach().clone())) for norm in model.norms]
    raw = []
    handles += [norm.register_forward_pre_hook(lambda module, args:
        raw.append(args[0].detach().clone())) for norm in model.norms]
    encoded = model.encode_proteins(batch)
    for handle in handles:
        handle.remove()
    map1 = batch['blocks'][1]['dst_in_src']
    map2 = batch['blocks'][2]['dst_in_src']
    expected = post[0][map1][map2] + post[1][map2] + post[2]
    torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
    # A direct structural check that inherited RAW residual is not post-BN h.
    with torch.no_grad():
        for index, layer in enumerate(model.sage_layers):
            for linear in layer.self_linears:
                linear.weight.zero_(); linear.bias.fill_(-2 if index == 0 else 0)
            for linear in layer.neighbor_linears.values():
                linear.weight.zero_()
    captured = []
    handles = [norm.register_forward_pre_hook(lambda module, args:
        captured.append(args[0].detach().clone())) for norm in model.norms]
    model(batch)
    for handle in handles:
        handle.remove()
    assert torch.equal(captured[0], torch.full_like(captured[0], -2))
    torch.testing.assert_close(captured[1], captured[0][map1], rtol=0, atol=0)
    torch.testing.assert_close(captured[2], captured[1][map2], rtol=0, atol=0)


def test_partial_seen_evidence_rejected_then_explicitly_sanitized_without_mutation():
    seen = torch.tensor([True, True, False, True, True, True, False, True])
    model, batch = _model(seen=seen).eval(), _batch()
    batch.update(targets=torch.ones(2, 8), positive_mask=torch.ones(2, 8, dtype=torch.bool))
    original = copy.deepcopy(batch)
    with pytest.raises(ValueError, match='unseen'):
        model(batch)
    cleaned = model.sanitize_evidence(batch)
    assert torch.equal(batch['sampled_candidate_go'], original['sampled_candidate_go'])
    assert torch.equal(batch['sampled_pseudo_edge'], original['sampled_pseudo_edge'])
    assert not cleaned['targets'][:, ~seen].any()
    assert not cleaned['positive_mask'][:, ~seen].any()
    assert torch.isfinite(model(cleaned, target_ontology_ids=[2, 6, 9])).all()
    for name in ['sampled_gold_edge', 'sampled_pseudo_edge']:
        assert seen[cleaned[name][1]].all()
    assert not (cleaned['sampled_candidate_go'] == 2).any()
    assert not (cleaned['sampled_candidate_go'] == 6).any()


def test_alias_split_and_nonfinite_or_missing_new_geometry_are_rejected():
    ontology = _ontology(); ontology['task_to_ontology'][2] = 0
    seen = torch.ones(8, dtype=torch.bool); seen[2] = False
    with pytest.raises(ValueError, match='alias'):
        _model(ontology, seen)
    ontology = _ontology(); ontology['center'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        _model(ontology)
    with pytest.raises(KeyError):
        _model({'task_to_ontology': torch.arange(8)})
    ontology = _ontology(); ontology['edges']['is_a'] = ontology['edges']['is_a'].float() + .5
    with pytest.raises(ValueError, match='integer'):
        _model(ontology)
    ontology = _ontology(); ontology['edges']['is_a'][0, 0] = 10
    with pytest.raises(ValueError, match='out of range'):
        _model(ontology)


def test_empty_evidence_and_pp_edges_keep_ddp_zero_gradient_connections():
    model, batch = _model(), _batch()
    batch.update(sampled_candidate_go=torch.empty(6, 0, dtype=torch.long),
                 sampled_candidate_attr=torch.empty(6, 0, 3),
                 sampled_gold_edge=torch.empty(2, 0, dtype=torch.long),
                 sampled_pseudo_edge=torch.empty(2, 0, dtype=torch.long))
    for block in batch['blocks']:
        block.update(edge_index=torch.empty(2, 0, dtype=torch.long),
                     edge_type=torch.empty(0, dtype=torch.long), edge_attr=torch.empty(0, 3))
    model(batch).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_all_encoder_and_shared_scorer_parameters_receive_finite_gradients():
    model, batch = _model(), _batch()
    target = torch.zeros(2, 8); target[0, 0] = target[1, 1] = 1
    F.binary_cross_entropy_with_logits(model(batch), target).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for prefix in ['ontology_encoder', 'node_encoder', 'sage_layers.0', 'sage_layers.1',
                   'sage_layers.2', 'scorer.protein_projection', 'scorer.label_bias']:
        assert any(p.grad.norm() > 0 for name, p in model.named_parameters() if name.startswith(prefix)), prefix


def test_encoder_migration_is_exact_but_never_copies_old_classifier_or_shared_head():
    old = FullTaskGraphModelV088(6, _ontology(), FullTaskModelConfigV088(
        hidden_dim=16, ontology_layers=1, input_dropout=0, dropout=0)).eval()
    model = _model().eval()
    before = {name: value.clone() for name, value in model.scorer.state_dict().items()}
    result = migrate_encoder_state_dict(model, old.state_dict())
    assert result['discarded_classifier_keys'] == ['classifier.bias', 'classifier.weight']
    assert result['scorer_reinitialized'] is True
    for name, value in before.items():
        torch.testing.assert_close(model.scorer.state_dict()[name], value, rtol=0, atol=0)
    for name, value in old.state_dict().items():
        if not name.startswith('classifier.'):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    task = old.encode_go()
    old_hidden = []
    handle = old.classifier.register_forward_pre_hook(lambda module, args: old_hidden.append(args[0]))
    old(_batch(), go_encoding=task); handle.remove()
    torch.testing.assert_close(model.encode_proteins(_batch()), old_hidden[0], rtol=0, atol=0)
    bad = dict(old.state_dict()); bad['node_encoder.weight'] = bad['node_encoder.weight'][:1]
    with pytest.raises(ValueError, match='shape'):
        model.load_encoder_state_dict(bad)
    bad = dict(old.state_dict()); bad['scorer.label_bias.bias'] = torch.zeros(1)
    with pytest.raises(ValueError, match='key'):
        model.load_encoder_state_dict(bad)


def test_dropedge_is_training_only_and_zero_confidence_edges_leave_mean_denominator(monkeypatch):
    model, batch = _model(edge_dropout=.5), _batch()
    original = copy.deepcopy(batch)
    masks = []
    handles = [layer.register_forward_pre_hook(lambda module, args:
        masks.append(args[5].detach().clone())) for layer in model.sage_layers]
    monkeypatch.setattr(torch, 'rand', lambda count, device=None: torch.zeros(count, device=device))
    model.train(); model(batch)
    assert all(torch.equal(attr[:, 0], torch.zeros_like(attr[:, 0])) for attr in masks)
    masks.clear(); model.eval(); model(batch)
    assert all(torch.equal(attr, block['edge_attr']) for attr, block in zip(masks, batch['blocks']))
    for handle in handles:
        handle.remove()
    for before, after in zip(original['blocks'], batch['blocks']):
        assert torch.equal(before['edge_attr'], after['edge_attr'])


@pytest.mark.parametrize('target', [[], [-1], [10], [0.5], [[0]], [True]])
def test_invalid_target_row_list_rejected(target):
    with pytest.raises(ValueError, match='target_ontology_ids'):
        _model().eval()(_batch(), target_ontology_ids=target)


@pytest.mark.parametrize('options', [dict(jk_mode='concat'), dict(edge_dropout=1),
    dict(edge_dropout=-.1), dict(scorer_initial_bias=float('nan')),
    dict(inference_go_chunk_size=0), dict(prediction_mode='direct')])
def test_invalid_new_model_options_rejected(options):
    with pytest.raises(ValueError):
        _model(**options)
