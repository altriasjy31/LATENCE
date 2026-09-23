"""Numerical and information-flow contracts for the GO-conditioned model."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nbs_pg.full_task_model_v081 import FullTaskGraphModelV081, FullTaskModelConfigV081
from test_full_task_model import _batch, _ontology


def _model(**kwargs):
    values = dict(hidden_dim=16, query_dim=8, ontology_layers=1,
                  decoder_hidden=16, go_chunk=3, candidate_dropout=0.0)
    values.update(kwargs)
    return FullTaskGraphModelV081(6, _ontology(), FullTaskModelConfigV081(**values))


def _loss(details, batch):
    target = batch["positive_mask"].float()
    return (F.binary_cross_entropy_with_logits(details["logits"], target) +
            F.binary_cross_entropy_with_logits(details["graph_logits"], target))


def test_aliases_preserve_columns_and_residual_starts_exactly_at_base():
    ontology = _ontology()
    ontology["task_to_ontology"] = torch.tensor([4, 1, 4, 2, 8, 5, 6, 7])
    model = FullTaskGraphModelV081(6, ontology, _model().config)
    batch = _batch()
    result = model(batch, return_details=True)
    assert result["logits"].shape == (4, 8)
    assert torch.equal(result["logits"], batch["base_logits"])
    assert not torch.equal(result["graph_logits"], torch.zeros_like(result["graph_logits"]))
    go = model.encode_go()
    assert torch.equal(go[0], go[2])
    graph = model._prepare_graph(batch, go, use_weak_go=True, use_core_go=True)
    assert graph[-2][0, 0, 0] == 1 and graph[-2][0, 2, 0] == 0
    assert graph[-1][2, 0] == 0 and graph[-1][2, 2] > 0


def test_auxiliary_objective_has_first_step_gradients_through_both_paths():
    torch.manual_seed(73)
    model, batch = _model(), _batch()
    _loss(model(batch, return_details=True), batch).backward()
    for prefix in ("box_center_encoder", "box_offset_encoder", "go_layers",
                   "protein_encoder", "anchor_update", "weak_edge_encoder",
                   "core_edge_encoder", "query", "weak_key", "weak_value",
                   "core_key", "core_value", "graph_output", "correction_decoder"):
        norms = [p.grad.norm() for name, p in model.named_parameters()
                 if name.startswith(prefix) and p.grad is not None]
        assert norms and max(norms) > 0, prefix
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_go_query_changes_which_neighbour_is_read():
    model = _model()
    query = torch.eye(8)[:2]
    key, value = query[None], query[None]
    prior = torch.zeros(1, 2)
    valid = torch.ones(1, 2, dtype=torch.bool)
    readout = model._attend(query, key, value, prior, valid)
    assert readout[0, 0, 0] > readout[0, 0, 1] + 0.5
    assert readout[0, 1, 1] > readout[0, 1, 0] + 0.5
    model.config = replace(model.config, query_conditioned=False)
    pooled = model._attend(query, key, value, prior, valid)
    assert torch.equal(pooled[:, 0], pooled[:, 1])


def test_target_labels_and_direct_base_cannot_enter_graph_score():
    torch.manual_seed(21)
    model, batch = _model().eval(), _batch()
    expected = model(batch, return_details=True)
    changed = dict(batch)
    changed["targets"] = torch.randn_like(batch["targets"])
    changed["positive_mask"] = ~batch["positive_mask"]
    changed["is_weak"] = ~batch["is_weak"]
    changed["expert_prob"] = torch.rand_like(batch["targets"])
    assert torch.equal(model(changed), expected["logits"])
    changed["base_logits"] = batch["base_logits"] + 7
    actual = model(changed, return_details=True)
    assert torch.equal(actual["graph_logits"], expected["graph_logits"])
    assert torch.equal(actual["logits"], changed["base_logits"])


def test_candidate_dropout_removes_aggregation_and_exact_identity_together():
    model, batch = _model(candidate_dropout=1.0), _batch()
    go = model.encode_go()
    graph = model._prepare_graph(batch, go, use_weak_go=True, use_core_go=True)
    assert not graph[4].any()
    assert not graph[-2].any()
    actual = model(batch, return_details=True)
    weak_off = model(batch, use_weak_go=False, return_details=True)
    assert torch.equal(actual["graph_logits"], weak_off["graph_logits"])
    model.eval()
    retained = model._prepare_graph(batch, model.encode_go(), use_weak_go=True, use_core_go=True)
    assert retained[4].all() and retained[-2].any()


def test_dropout_rng_is_reproducible_and_independent_of_supervision():
    model, batch = _model(candidate_dropout=0.5), _batch()
    torch.manual_seed(99)
    expected = model(batch, return_details=True)
    changed = dict(batch)
    changed["positive_mask"] = ~batch["positive_mask"]
    changed["targets"] = 1 - batch["targets"]
    torch.manual_seed(99)
    actual = model(changed, return_details=True)
    assert torch.equal(actual["graph_logits"], expected["graph_logits"])


def test_recomputed_chunks_match_outputs_gradients_and_dropout_rng():
    torch.manual_seed(51)
    first = _model(candidate_dropout=0.3, activation_checkpointing=False)
    second = _model(candidate_dropout=0.3, activation_checkpointing=True)
    torch.nn.init.normal_(first.correction_decoder[-1].weight, std=0.1)
    second.load_state_dict(first.state_dict())
    batch = _batch()
    torch.manual_seed(913)
    expected = first(batch, return_details=True)
    _loss(expected, batch).backward()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(913)
    actual = second(batch, return_details=True)
    _loss(actual, batch).backward()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-6)
    for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5, msg=name)


def test_sdpa_and_math_attention_agree_with_empty_neighbours():
    torch.manual_seed(19)
    model, batch = _model().eval(), _batch()
    batch["candidate_go"][0] = -1
    batch["neighbor_index"][1] = -1
    torch.nn.init.normal_(model.correction_decoder[-1].weight, std=0.1)
    first = model(batch, return_details=True)
    model.config = replace(model.config, attention_backend="math")
    second = model(batch, return_details=True)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=1e-6, rtol=1e-5)
    assert torch.isfinite(first["logits"]).all()
    assert torch.equal(first["routing"][0, :, 1], torch.zeros(8))
    assert torch.equal(first["routing"][1, :, 2], torch.zeros(8))


def test_no_anchor_no_candidate_is_finite_with_fp32_small_residual_under_amp():
    model, batch = _model(), _batch()
    batch["anchor_x"] = torch.empty(0, 6)
    batch["anchor_go_edge"] = torch.empty(2, 0, dtype=torch.long)
    batch["neighbor_index"] = torch.full((4, 2), -1)
    batch["candidate_go"] = torch.full((4, 2), -1)
    batch["base_logits"] = torch.full((4, 8), 2.0)
    with torch.no_grad():
        model.correction_decoder[-1].bias.fill_(0.000411)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        details = model(batch, return_details=True)
        loss = _loss(details, batch)
    loss.backward()
    assert details["logits"].dtype == torch.float32
    assert torch.all(details["logits"] > 2.0004)
    assert torch.equal(details["core_vote"], torch.zeros(4, 8))
    assert torch.equal(details["routing"][..., 0], torch.ones(4, 8))
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_cached_go_is_eval_only_and_go_chunk_does_not_change_prediction():
    model, batch = _model().eval(), _batch()
    torch.nn.init.normal_(model.correction_decoder[-1].weight, std=0.1)
    go = model.encode_go()
    expected = model(batch)
    torch.testing.assert_close(model(batch, go_encoding=go), expected)
    model.config = replace(model.config, go_chunk=1)
    torch.testing.assert_close(model(batch), expected, atol=1e-6, rtol=1e-5)
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        model(batch, go_encoding=go)
