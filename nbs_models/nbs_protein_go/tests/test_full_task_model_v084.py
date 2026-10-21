"""Meaningful graph propagation, control, optimization and numerical contracts."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nbs_pg.full_task_model_v084 import (
    FullTaskGraphModelV084, FullTaskModelConfigV084, WeightedRelationSAGE)
from test_full_task_model import _batch, _ontology


def _model(**kwargs):
    options = dict(hidden_dim=16, query_dim=8, decoder_hidden=16, ontology_layers=1,
                   go_chunk=3, candidate_dropout=0.0, activation_checkpointing=False)
    options.update(kwargs)
    return FullTaskGraphModelV084(6, _ontology(), FullTaskModelConfigV084(**options))


def _sampled_batch():
    batch = _batch()
    g = torch.Generator().manual_seed(184)
    batch.update(sampled_protein_x=torch.cat((batch['protein_x'], batch['anchor_x'],
                                            torch.randn(5, 6, generator=g))),
                 sampled_seed_index=torch.arange(4), sampled_anchor_index=torch.arange(4, 8),
                 sampled_candidate_go=torch.randint(8, (13, 3), generator=g),
                 sampled_candidate_attr=torch.rand(13, 3, 3, generator=g),
                 sampled_gold_edge=torch.cat((batch['anchor_go_edge'] + torch.tensor([[4], [0]]),
                                               torch.tensor([[8, 8, 9], [0, 4, 2]])), 1),
                 sampled_pseudo_edge=torch.tensor([[10, 10, 11, 12], [1, 5, 3, 7]]))
    # Every relation reaches a seed, and every relation also exists upstream.
    edges, types = [], []
    for relation in range(5):
        for seed in range(4):
            edges += [(8 + relation, 4 + seed), (4 + (seed + relation) % 4, seed)]
            types += [relation, relation]
    batch['sampled_edge_index'] = torch.tensor(edges).T
    batch['sampled_edge_type'] = torch.tensor(types)
    batch['sampled_edge_attr'] = torch.rand(len(edges), 3, generator=g) + 0.2
    return batch


def _open(model):
    with torch.no_grad():
        model.correction_decoder[-1].weight.normal_(std=0.2)


def test_initial_base_identity_full_go_and_only_single_match_head():
    model, batch = _model(), _sampled_batch()
    out = model(batch, return_details=True)
    assert torch.equal(out['logits'], batch['base_logits'])
    assert out['logits'].shape == (4, 8) and out['routing'].shape == (4, 8, 4)
    assert out['graph_logits'] is None
    assert out['pp_edge_count'] == 40 and out['pp_message_norm'] > 0
    assert not hasattr(model, 'graph_output')
    assert len(model.sage_layers) == 2


def test_all_relation_layers_and_evidence_sources_receive_real_gradients():
    torch.manual_seed(23)
    model, batch = _model(), _sampled_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(3):
        optimizer.zero_grad()
        F.binary_cross_entropy_with_logits(model(batch), batch['targets']).backward()
        if step == 2:
            for layer in range(2):
                for relation in model.PP_RELATIONS:
                    prefix = f'sage_layers.{layer}.{relation}'
                    grads = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
                    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), prefix
                    assert all(g.norm() > 0 for g in grads), prefix
            for name in ('sampled_candidate_value', 'sampled_gold_value', 'sampled_pseudo_value',
                         'protein_encoder', 'core_exact_value', 'weak_value', 'go_layers'):
                assert any(p.grad is not None and p.grad.norm() > 0
                           for key, p in model.named_parameters() if key.startswith(name)), name
        optimizer.step()


def test_forward_never_reads_targets_or_dense_teacher_probabilities():
    model, batch = _model().eval(), _sampled_batch()
    _open(model)
    expected = model(batch)
    changed = dict(batch, targets=1 - batch['targets'], positive_mask=~batch['positive_mask'],
                   is_weak=~batch['is_weak'], modelout=torch.rand(4, 8),
                   expert_prob=torch.rand(4, 8), teacher_prob=torch.rand(4, 8))
    assert torch.equal(model(changed), expected)


def test_ablation_removes_all_evidence_of_its_source_only():
    torch.manual_seed(31)
    model, batch = _model().eval(), _sampled_batch()
    _open(model)
    changed = dict(batch, candidate_go=(batch['candidate_go'] + 1) % 8,
                   sampled_candidate_go=(batch['sampled_candidate_go'] + 2) % 8,
                   sampled_pseudo_edge=torch.tensor([[10, 12], [6, 0]]))
    torch.testing.assert_close(model(batch, use_weak_go=False), model(changed, use_weak_go=False),
                               atol=0, rtol=0)
    assert not torch.allclose(model(batch), model(changed))
    changed = dict(batch, anchor_go_edge=torch.tensor([[0, 1, 2, 3], [0, 2, 4, 6]]),
                   sampled_gold_edge=torch.tensor([[4, 8], [7, 5]]))
    torch.testing.assert_close(model(batch, use_core_go=False), model(changed, use_core_go=False),
                               atol=0, rtol=0)
    changed = dict(batch, sampled_edge_index=batch['sampled_edge_index'].flip(0))
    torch.testing.assert_close(model(batch, use_pp_context=False), model(changed, use_pp_context=False),
                               atol=0, rtol=0)
    # core_off still allows PP sequence/pseudo signals; graph_off explicitly removes PP.
    assert model(batch, use_core_go=False, return_details=True)['pp_edge_count'] > 0
    flags = dict(use_core_go=False, use_weak_go=False, use_pp_context=False)
    out = model(batch, return_details=True, **flags)
    assert not out['core_vote'].any() and not out['core_support_mass'].any()
    assert out['pp_edge_count'] == 0
    assert not torch.equal(out['logits'], batch['base_logits'])
    changed = dict(batch, sampled_protein_x=batch['sampled_protein_x'].clone())
    changed['sampled_protein_x'][4:] += 10
    torch.testing.assert_close(model(changed, **flags), out['logits'], atol=0, rtol=0)


def test_confidence_attenuation_zero_and_missing_relations_are_exact():
    torch.manual_seed(14)
    conv = WeightedRelationSAGE(5)
    x = torch.randn(3, 5)
    edges = torch.tensor([[0, 1], [2, 2]])
    attr = torch.ones(2, 3)
    full, valid = conv(x, edges, attr)
    attr[:, 0] = 0.001
    small, _ = conv(x, edges, attr)
    torch.testing.assert_close(small, full * 0.001)
    attr[:, 0] = 0
    zero, active = conv(x, edges, attr)
    assert not zero.any() and not active.any()
    empty, active = conv(x, torch.empty(2, 0, dtype=torch.long), torch.empty(0, 3))
    assert not empty.any() and not active.any()


def test_alias_go_exact_incidence_is_not_compressed_with_geometry():
    ontology = _ontology()
    ontology['task_to_ontology'] = torch.tensor([0, 1, 0, 3, 4, 5, 6, 7])
    model = FullTaskGraphModelV084(6, ontology, _model().config).eval()
    batch = _sampled_batch()
    batch['anchor_go_edge'] = torch.tensor([[0, 1, 2, 3], [0, 3, 5, 7]])
    batch['sampled_gold_edge'] = batch['anchor_go_edge'] + torch.tensor([[4], [0]])
    changed = dict(batch, anchor_go_edge=batch['anchor_go_edge'].clone(),
                   sampled_gold_edge=batch['sampled_gold_edge'].clone())
    changed['anchor_go_edge'][1, 0] = changed['sampled_gold_edge'][1, 0] = 2
    before, after = model(batch, return_details=True), model(changed, return_details=True)
    assert before['core_support_mass'][0, 0] > 0 and after['core_support_mass'][0, 0] == 0
    assert after['core_support_mass'][0, 2] > 0


def test_shuffle_permutates_sampled_associations_and_is_batch_independent():
    model, batch = _model().eval(), _sampled_batch()
    _open(model)
    shuffled = model(batch, shuffle_go=True)
    assert not torch.allclose(shuffled, model(batch))
    assert torch.equal(shuffled, model(batch, shuffle_go=True))
    single = dict(batch)
    for name in ('protein_x', 'base_logits', 'candidate_go', 'candidate_attr', 'neighbor_index',
                 'neighbor_attr', 'targets', 'positive_mask', 'is_weak', 'sampled_seed_index'):
        single[name] = batch[name][1:2]
    torch.testing.assert_close(model(single, shuffle_go=True), shuffled[1:2], atol=1e-6, rtol=1e-5)
    go = model.encode_go()
    edge = torch.tensor([[4, 5], [0, 2]])
    manual = torch.stack((edge[0], model.association_permutation[edge[1]]))
    torch.testing.assert_close(model._binary_go_mean(edge, go, 13, True),
                               model._binary_go_mean(manual, go, 13, False))


def _chain_batch():
    batch = _batch()
    batch['protein_x'] = torch.zeros(4, 6)
    batch['base_logits'] = torch.zeros(4, 8)
    batch['candidate_go'] = torch.empty(4, 0, dtype=torch.long)
    batch['candidate_attr'] = torch.empty(4, 0, 3)
    batch['anchor_x'] = torch.empty(0, 6)
    batch['anchor_go_edge'] = torch.empty(2, 0, dtype=torch.long)
    batch['neighbor_index'] = torch.empty(4, 0, dtype=torch.long)
    batch['neighbor_attr'] = torch.empty(4, 0, 3)
    batch['sampled_protein_x'] = torch.zeros(12, 6)
    batch['sampled_seed_index'] = torch.arange(4)
    batch['sampled_anchor_index'] = torch.empty(0, dtype=torch.long)
    batch['sampled_candidate_go'] = torch.empty(12, 0, dtype=torch.long)
    batch['sampled_candidate_attr'] = torch.empty(12, 0, 3)
    batch['sampled_pseudo_edge'] = torch.empty(2, 0, dtype=torch.long)
    batch['sampled_gold_edge'] = torch.tensor([(8 + row, col) for row in range(4)
                                             for col in range((row % 2) * 4, (row % 2 + 1) * 4)]).T
    batch['sampled_edge_index'] = torch.tensor([(8 + i, 4 + i) for i in range(4)] +
                                              [(4 + i, i) for i in range(4)]).T
    batch['sampled_edge_attr'] = torch.ones(8, 3)
    batch['sampled_edge_type'] = torch.tensor([0] * 4 + [3] * 4)
    batch['targets'] = torch.tensor([[1.] * 4 + [0.] * 4, [0.] * 4 + [1.] * 4] * 2)
    batch['positive_mask'] = batch['targets'].bool()
    return batch


def test_go_to_core2_to_core1_to_seed_is_learnable_and_pp_off_breaks_path():
    torch.manual_seed(844)
    model, batch = _model(), _chain_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    start = F.binary_cross_entropy_with_logits(model(batch), batch['targets']).item()
    for _ in range(120):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(batch), batch['targets'])
        loss.backward()
        optimizer.step()
    model.eval()
    full = F.binary_cross_entropy_with_logits(model(batch), batch['targets']).item()
    off = F.binary_cross_entropy_with_logits(model(batch, use_pp_context=False), batch['targets']).item()
    assert full < start * 0.5 and off > full + 0.1, (start, full, off)
    # There are no direct core labels/query votes or per-seed sequence cues.
    assert not model(batch, return_details=True)['core_vote'].any()
    assert torch.equal(model(batch, use_pp_context=False)[0], model(batch, use_pp_context=False)[1])


def test_chunk_backend_cached_go_and_activation_recomputation_match():
    torch.manual_seed(85)
    left, batch = _model(candidate_dropout=0.2), _sampled_batch()
    right = _model(candidate_dropout=0.2, activation_checkpointing=True)
    _open(left)
    right.load_state_dict(left.state_dict())
    torch.manual_seed(88)
    a = left(batch)
    F.binary_cross_entropy_with_logits(a, batch['targets']).backward()
    rng = torch.get_rng_state()
    torch.manual_seed(88)
    b = right(batch)
    F.binary_cross_entropy_with_logits(b, batch['targets']).backward()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    for (name, x), (_, y) in zip(left.named_parameters(), right.named_parameters()):
        if x.grad is None:
            assert y.grad is None, name
        else:
            torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5, msg=name)
    left.eval()
    before = left(batch)
    left.config = replace(left.config, go_chunk=1, attention_backend='math')
    torch.testing.assert_close(left(batch, go_encoding=left.encode_go()), before, atol=1e-6, rtol=1e-5)


def test_empty_graph_retains_small_fp32_correction_under_bf16():
    model, batch = _model(), _chain_batch()
    batch['sampled_edge_index'] = torch.empty(2, 0, dtype=torch.long)
    batch['sampled_edge_type'] = torch.empty(0, dtype=torch.long)
    batch['sampled_edge_attr'] = torch.empty(0, 3)
    batch['sampled_gold_edge'] = torch.empty(2, 0, dtype=torch.long)
    batch['base_logits'].fill_(2)
    with torch.no_grad():
        model.correction_decoder[-1].bias.fill_(0.000411)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = model(batch, return_details=True)
        F.binary_cross_entropy_with_logits(out['logits'], batch['targets']).backward()
    assert out['logits'].dtype == torch.float32
    assert torch.all(out['logits'] > 2.0004) and out['pp_edge_count'] == 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_malformed_graph_contract_rejected():
    with pytest.raises(ValueError, match='exactly two'):
        _model(sage_layers=1)
    model, batch = _model(), _sampled_batch()
    batch['sampled_edge_type'][0] = 5
    with pytest.raises(ValueError, match='relation type'):
        model(batch)


def test_edges_beyond_seed_two_hops_cannot_change_anchor_query_readout():
    # When another seed expands a shared depth-2 node, its new incoming edges
    # must not become an unintended third PP hop in this seed's core readout.
    torch.manual_seed(841)
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


def test_real_sampler_and_model_external_predictions_ignore_batch_partition(tmp_path):
    from test_full_task_data_v084 import data_v084, inference_fixture
    data = data_v084(tmp_path, holdout=1, variant='dynamic', dropout=.75)
    data.prepare_neighbors()
    input_dir = inference_fixture(tmp_path, data)
    ontology = _ontology()
    ontology['task_to_ontology'] = torch.arange(data.num_task_go)
    torch.manual_seed(842)
    model = FullTaskGraphModelV084(data.feature_dim, ontology, _model().config).eval()
    _open(model)
    for shuffle in (False, True):
        combined = model(data.inference_batch(input_dir, [0, 1]), shuffle_go=shuffle)
        separate = torch.cat([model(data.inference_batch(input_dir, [row]), shuffle_go=shuffle)
                              for row in (0, 1)])
        torch.testing.assert_close(combined, separate, atol=1e-6, rtol=1e-5)
