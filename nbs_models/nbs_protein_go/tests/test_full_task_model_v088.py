"""Formula, block locality, typed-BN and absolute full-axis model contracts."""
import copy

import pytest
import torch
import torch.nn.functional as F

from nbs_pg.full_task_model_v088 import (FullTaskGraphModelV088,
    FullTaskModelConfigV088, TypedRelationSAGEV088, TypedBatchNormV088)
from test_full_task_model import _ontology


def _model(**kwargs):
    options = dict(hidden_dim=16, ontology_layers=1, input_dropout=0., dropout=0.)
    options.update(kwargs)
    return FullTaskGraphModelV088(6, _ontology(), FullTaskModelConfigV088(**options))


def _block(src, mapping):
    src = torch.tensor(src, dtype=torch.long)
    mapping = torch.tensor(mapping, dtype=torch.long)
    dst = src[mapping]
    sources = torch.where(src >= 0)[0].tolist()
    edges, relations = [], []
    for row in range(len(dst)):
        for kind in range(5):
            edges.append((sources[kind % len(sources)], row))
            relations.append(kind)
    return dict(src_global_ids=src, dst_global_ids=dst, dst_in_src=mapping,
        edge_index=torch.tensor(edges, dtype=torch.long).T,
        edge_type=torch.tensor(relations, dtype=torch.long),
        edge_attr=torch.ones(len(edges), 3))


def _batch():
    generator = torch.Generator().manual_seed(8801)
    # Destinations are deliberately non-prefix/reordered to test identity maps.
    blocks = [_block([-1, -2, 0, 1, 4, 5], [1, 2, 0, 4]),
              _block([-2, 0, -1, 4], [1, 0, 2]),
              _block([0, -2, -1], [2, 1])]
    return dict(sampled_protein_x=torch.randn(6, 6, generator=generator),
        sampled_node_type=torch.tensor([2, 2, 0, 0, 1, 1]),
        sampled_candidate_go=torch.randint(8, (6, 2), generator=generator),
        sampled_candidate_attr=torch.rand(6, 2, 3, generator=generator),
        sampled_gold_edge=torch.tensor([[2, 2, 3], [0, 1, 4]]),
        sampled_pseudo_edge=torch.tensor([[4, 5], [2, 6]]), blocks=blocks)


def test_one_relation_is_exact_mean_sage_with_one_self_map_and_raw_skip():
    layer = TypedRelationSAGEV088(2)
    with torch.no_grad():
        for linear in layer.self_linears:
            linear.weight.copy_(2*torch.eye(2)); linear.bias.fill_(.5)
        for linear in layer.neighbor_linears.values():
            linear.weight.copy_(3*torch.eye(2))
    state = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    mapping, types = torch.tensor([2, 0]), torch.tensor([0, 0])
    edge, kind = torch.tensor([[0, 1, 2], [0, 0, 1]]), torch.zeros(3, dtype=torch.long)
    attr = torch.ones(3, 3)
    expected = 2*state[mapping]+.5 + 3*torch.stack(((state[0]+state[1])/2, state[2]))
    actual = layer(state, mapping, types, edge, kind, attr)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    raw = torch.tensor([[10., 20.], [30., 40.], [50., 60.]])
    torch.testing.assert_close(layer(state, mapping, types, edge, kind, attr, raw),
                               expected+raw[mapping], rtol=0, atol=0)
    attr[1, 0] = .5
    expected = 2*state[mapping]+.5 + 3*torch.stack(((state[0]+.5*state[1])/2, state[2]))
    torch.testing.assert_close(layer(state, mapping, types, edge, kind, attr), expected)


def test_relations_sum_and_empty_relations_do_not_duplicate_self_or_create_messages():
    layer = TypedRelationSAGEV088(2)
    with torch.no_grad():
        for linear in layer.self_linears:
            linear.weight.zero_(); linear.bias.fill_(1)
        for linear in layer.neighbor_linears.values():
            linear.weight.copy_(torch.eye(2))
    h = torch.ones(3, 2)
    mapping, types = torch.tensor([2]), torch.tensor([2])
    output = layer(h, mapping, types, torch.tensor([[0, 1], [0, 0]]),
                   torch.tensor([0, 1]), torch.ones(2, 3))
    assert torch.equal(output, torch.full((1, 2), 3.))
    empty = layer(h, mapping, types, torch.empty(2, 0, dtype=torch.long),
                   torch.empty(0, dtype=torch.long), torch.empty(0, 3))
    assert torch.equal(empty, torch.ones(1, 2))
    empty.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_model_raw_residual_is_pre_bn_not_activated_hidden():
    model, batch = _model().eval(), _batch()
    with torch.no_grad():
        for i, layer in enumerate(model.sage_layers):
            for linear in layer.self_linears:
                linear.weight.zero_(); linear.bias.fill_(-2 if i == 0 else 0)
            for linear in layer.neighbor_linears.values():
                linear.weight.zero_()
    seen = []
    handles = [norm.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
               for norm in model.norms]
    model(batch)
    for handle in handles:
        handle.remove()
    assert torch.equal(seen[0], torch.full_like(seen[0], -2))
    assert torch.equal(seen[1], seen[0][batch['blocks'][1]['dst_in_src']])
    assert torch.equal(seen[2], seen[1][batch['blocks'][2]['dst_in_src']])


def test_type_specific_bn_empty_and_singleton_keep_finite_gradients_and_buffers():
    norm = TypedBatchNormV088(3).train()
    x = torch.tensor([[1., 2., 3.], [3., 6., 9.], [8., 7., 6.]], requires_grad=True)
    output = norm(x, torch.tensor([0, 0, 2]))
    assert norm.norms[0].num_batches_tracked == 1
    assert norm.norms[1].num_batches_tracked == norm.norms[2].num_batches_tracked == 0
    expected = x[2] / torch.sqrt(torch.ones(3)+norm.norms[2].eps)
    torch.testing.assert_close(output[2], expected)
    output.square().sum().backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in norm.parameters())


def test_full_head_alias_columns_and_train_prior_are_independent():
    ontology = _ontology()
    ontology['task_to_ontology'] = torch.tensor([0, 1, 0, 3, 4, 5, 6, 7])
    model = FullTaskGraphModelV088(6, ontology,
        FullTaskModelConfigV088(hidden_dim=16, ontology_layers=1, input_dropout=0, dropout=0)).eval()
    prior = torch.linspace(.01, .08, 8)
    original_weight = model.classifier.weight.detach().clone()
    model.initialize_output_prior(prior)
    assert torch.equal(model.classifier.weight, original_weight)
    torch.testing.assert_close(model.classifier.bias.sigmoid(), prior)
    assert model.classifier.out_features == 8
    assert torch.equal(model.encode_go()[0], model.encode_go()[2])
    with torch.no_grad():
        model.classifier.weight.zero_()
    result = model(_batch(), return_details=True)
    torch.testing.assert_close(result['logits'], torch.logit(prior)[None].expand(2, -1))
    assert result['logits'][0, 0] != result['logits'][0, 2]
    assert set(result) == {'logits', 'model_diagnostics'}


def test_forward_has_no_dense_base_teacher_or_target_dependency():
    model, batch = _model().eval(), _batch()
    before = model(batch)
    changed = dict(batch, base_logits=torch.full((2, 8), float('nan')),
        expert_prob=torch.full((2, 8), float('nan')), modelout=torch.full((2, 8), float('nan')),
        targets=torch.ones(2, 8), positive_mask=torch.ones(2, 8, dtype=torch.bool))
    torch.testing.assert_close(model(changed), before, rtol=0, atol=0)
    assert not hasattr(model, 'correction_decoder') and not hasattr(model, 'direct_go_bias')


def test_eval_batch_partition_does_not_change_logits():
    model, batch = _model().eval(), _batch()
    together = model(batch)
    separate = []
    for row in (0, 1):
        single = copy.deepcopy(batch)
        block = single['blocks'][-1]
        keep = block['edge_index'][1] == row
        block['edge_index'] = torch.stack((block['edge_index'][0, keep],
                                          torch.zeros(int(keep.sum()), dtype=torch.long)))
        block['edge_type'], block['edge_attr'] = block['edge_type'][keep], block['edge_attr'][keep]
        for name in ('dst_global_ids', 'dst_in_src'):
            block[name] = block[name][row:row+1]
        separate.append(model(single))
    torch.testing.assert_close(together, torch.cat(separate), atol=1e-6, rtol=1e-6)


def test_all_three_layers_ontology_evidence_and_classifier_receive_gradients():
    torch.manual_seed(882)
    model, batch = _model(), _batch()
    model.initialize_output_prior(torch.full((8,), .2))
    target = torch.tensor([[1., 0, 1, 0, 0, 0, 0, 0], [0., 1, 0, 0, 0, 1, 0, 0]])
    loss = F.binary_cross_entropy_with_logits(model(batch), target)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for prefix in ('node_encoder', 'ontology_encoder', 'evidence_linears.candidate',
                   'evidence_linears.gold', 'evidence_linears.pseudo', 'classifier'):
        assert any(p.grad.norm() > 0 for n, p in model.named_parameters() if n.startswith(prefix)), prefix
    for i in range(3):
        assert any(p.grad.norm() > 0 for n, p in model.named_parameters()
                   if n.startswith(f'sage_layers.{i}')), i


def test_completely_empty_evidence_and_edges_preserve_ddp_parameter_connections():
    model, batch = _model(), _batch()
    batch.update(sampled_candidate_go=torch.empty(6, 0, dtype=torch.long),
                 sampled_candidate_attr=torch.empty(6, 0, 3),
                 sampled_gold_edge=torch.empty(2, 0, dtype=torch.long),
                 sampled_pseudo_edge=torch.empty(2, 0, dtype=torch.long))
    for block in batch['blocks']:
        block.update(edge_index=torch.empty(2, 0, dtype=torch.long),
                     edge_type=torch.empty(0, dtype=torch.long), edge_attr=torch.empty(0, 3))
    out = model(batch)
    out.square().mean().backward()
    assert out.shape == (2, 8) and torch.isfinite(out).all()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_cached_go_evaluation_matches_uncached_and_training_rejects_cache():
    model, batch = _model().eval(), _batch()
    go = model.encode_go()
    torch.testing.assert_close(model(batch), model(batch, go_encoding=go), rtol=0, atol=0)
    model.train()
    with pytest.raises(ValueError, match='cached'):
        model(batch, go_encoding=go)


@pytest.mark.parametrize('kind', ['chain', 'mapping', 'query_gold'])
def test_malformed_block_or_query_label_contract_is_rejected(kind):
    model, batch = _model(), _batch()
    if kind == 'chain':
        batch['blocks'][1]['src_global_ids'][0] = -99
    elif kind == 'mapping':
        batch['blocks'][0]['dst_in_src'][0] = 0
    else:
        batch['sampled_gold_edge'] = torch.tensor([[0], [1]])
    with pytest.raises(ValueError):
        model(batch)


@pytest.mark.parametrize('prior', [torch.zeros(8), torch.ones(8), torch.ones(7)*.2,
                                  torch.full((8,), float('nan'))])
def test_invalid_output_prior_is_rejected(prior):
    with pytest.raises(ValueError, match='output prior'):
        _model().initialize_output_prior(prior)


@pytest.mark.parametrize('options', [dict(encoder_variant='legacy'), dict(prediction_mode='residual'),
    dict(hidden_dim=0), dict(sage_layers=0), dict(ontology_layers=-1), dict(input_dropout=1),
    dict(dropout=-.1), dict(ontology_dropout=float('nan')), dict(bn_momentum=0)])
def test_invalid_config_is_rejected(options):
    with pytest.raises(ValueError):
        _model(**options)
