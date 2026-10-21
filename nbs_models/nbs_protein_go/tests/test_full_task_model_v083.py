"""Graph-message, ablation, numerical and optimization contracts for v0.8.3."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nbs_pg.full_task_model_v083 import FullTaskGraphModelV083, FullTaskModelConfigV083
from test_full_task_model import _batch, _ontology


def _model(**kwargs):
    options = dict(hidden_dim=16, query_dim=8, decoder_hidden=16, ontology_layers=1,
                   go_chunk=3, candidate_dropout=0.0)
    options.update(kwargs)
    return FullTaskGraphModelV083(6, _ontology(), FullTaskModelConfigV083(**options))


def _pp_batch():
    batch = _batch()
    rng = torch.Generator().manual_seed(183)
    batch.update(pp_protein_x=torch.randn(6, 6, generator=rng),
                 pp_candidate_go=torch.tensor([[0, 2], [2, 4], [3, 5],
                                                [1, 7], [6, 4], [5, 2]]),
                 pp_candidate_attr=torch.rand(6, 2, 3, generator=rng),
                 pp_neighbor_index=torch.tensor([[[0, 1], [2, 3], [4, 5]],
                                                  [[2, 3], [0, 5], [1, 4]],
                                                  [[4, 5], [1, 3], [0, 2]],
                                                  [[0, 3], [4, 2], [1, 5]]]),
                 pp_neighbor_attr=torch.rand(4, 3, 2, 3, generator=rng))
    return batch


def _open_residual(model):
    with torch.no_grad():
        model.correction_decoder[-1].weight.normal_(std=0.2)


def test_base_identity_alias_columns_and_no_auxiliary_or_classification_head():
    model, batch = _model(), _pp_batch()
    result = model(batch, return_details=True)
    assert torch.equal(result["logits"], batch["base_logits"])
    assert result["graph_logits"] is None
    assert result["pp_edge_count"] == 24 and result["pp_message_norm"] > 0
    assert result["routing"].shape == (4, 8, 4)
    assert not hasattr(model, "graph_output")
    ontology = _ontology()
    ontology["task_to_ontology"] = torch.tensor([4, 1, 4, 2, 8, 5, 6, 7])
    aliased = FullTaskGraphModelV083(6, ontology, model.config)
    assert torch.equal(aliased.encode_go()[0], aliased.encode_go()[2])
    details = aliased(batch, return_details=True)
    assert details["logits"].shape == (4, 8)
    assert details["core_vote"][2, 0] == 0 and details["core_vote"][2, 2] > 0


def test_both_graph_paths_all_pp_relations_learn_after_output_initialization():
    torch.manual_seed(12)
    model, batch = _model(), _pp_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in range(2):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(batch), batch["targets"])
        loss.backward()
        if step:
            prefixes = ("box_center_encoder", "box_offset_encoder", "go_layers",
                        "protein_encoder", "anchor_update", "weak_edge_encoder",
                        "core_edge_encoder", "query", "weak_key", "weak_value",
                        "core_key", "core_value", "core_exact_value",
                        "pp_context_encoder", "correction_decoder")
            prefixes += tuple("pp_messages." + x for x in model.PP_RELATIONS)
            prefixes += tuple("pp_edge_encoders." + x for x in model.PP_RELATIONS)
            for prefix in prefixes:
                gradients = [p.grad.norm() for name, p in model.named_parameters()
                             if name.startswith(prefix) and p.grad is not None]
                assert gradients and max(gradients) > 0, prefix
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())
        optimizer.step()


def test_no_teacher_targets_or_role_can_affect_forward():
    model, batch = _model().eval(), _pp_batch()
    _open_residual(model)
    expected = model(batch)
    changed = dict(batch, targets=1 - batch["targets"],
                   positive_mask=~batch["positive_mask"], is_weak=~batch["is_weak"],
                   modelout=torch.ones(4, 8), teacher_prob=torch.rand(4, 8),
                   expert_prob=torch.rand(4, 8))
    assert torch.equal(model(changed), expected)


def test_exact_support_preserves_task_identity_even_when_go_embeddings_alias():
    ontology = _ontology()
    ontology["task_to_ontology"] = torch.tensor([0, 1, 0, 3, 4, 5, 6, 7])
    model = FullTaskGraphModelV083(6, ontology, _model().config).eval()
    batch = _pp_batch()
    # One anchor's GO=0 is an alias of GO=2: compressed GO mean is identical.
    batch["anchor_go_edge"] = torch.tensor([[0, 1, 2, 3], [0, 3, 5, 7]])
    changed = dict(batch, anchor_go_edge=batch["anchor_go_edge"].clone())
    changed["anchor_go_edge"][1, 0] = 2
    go = model.encode_go()
    a = model._prepare_graph_v083(batch, go, use_weak_go=True, use_core_go=True,
                                  use_pp_context=True, shuffle_go=False)
    b = model._prepare_graph_v083(changed, go, use_weak_go=True, use_core_go=True,
                                  use_pp_context=True, shuffle_go=False)
    assert torch.equal(a["core_value"], b["core_value"])
    before, after = model(batch, return_details=True), model(changed, return_details=True)
    assert before["core_support_mass"][0, 0] > 0
    assert after["core_support_mass"][0, 0] == 0
    assert after["core_support_mass"][0, 2] > 0


def test_ablation_flags_remove_each_message_and_graph_off_retains_trainable_own_path():
    torch.manual_seed(83)
    model, batch = _model().eval(), _pp_batch()
    _open_residual(model)
    full = model(batch)
    for flags in (dict(use_weak_go=False), dict(use_core_go=False),
                  dict(use_pp_context=False), dict(use_weak_go=False, use_core_go=False)):
        assert not torch.allclose(full, model(batch, **flags), atol=1e-7, rtol=0), flags
    off = model(batch, use_weak_go=False, use_core_go=False, return_details=True)
    assert not torch.equal(off["logits"], batch["base_logits"])
    assert not off["core_support_mass"].any() and not off["core_vote"].any()
    assert torch.equal(off["routing"][..., 0], torch.ones(4, 8))
    assert off["pp_edge_count"] == 0


def test_pp_confidence_zero_removes_edges_and_small_confidence_attenuates_message():
    model, batch = _model().eval(), _pp_batch()
    go = model.encode_go()
    for attrs in batch["pp_neighbor_attr"]:
        attrs[..., 0] = 1
    full, _, _ = model._pp_context(batch, go, 4, use_weak_go=True,
                                   use_pp_context=True, shuffle_go=False)
    batch["pp_neighbor_attr"][..., 0] = 1e-3
    small, _, _ = model._pp_context(batch, go, 4, use_weak_go=True,
                                    use_pp_context=True, shuffle_go=False)
    assert small.norm() < full.norm() * 0.005
    batch["pp_neighbor_attr"][..., 0] = 0
    zero, _, count = model._pp_context(batch, go, 4, use_weak_go=True,
                                       use_pp_context=True, shuffle_go=False)
    assert not zero.any() and count == 0


def test_association_shuffle_is_deterministic_batch_independent_and_keeps_outputs():
    model, batch = _model().eval(), _pp_batch()
    _open_residual(model)
    actual = model(batch, shuffle_go=True)
    assert not torch.allclose(actual, model(batch), atol=1e-7, rtol=0)
    assert torch.equal(actual, model(batch, shuffle_go=True))
    # Only target rows are selected; the shared anchor/context pools stay shared.
    select_keys = ("protein_x", "base_logits", "candidate_go", "candidate_attr",
                   "neighbor_index", "neighbor_attr", "targets", "positive_mask", "is_weak")
    single = dict(batch)
    for key in select_keys:
        single[key] = batch[key][1:2]
    torch.testing.assert_close(model(single, shuffle_go=True), actual[1:2],
                               atol=1e-6, rtol=1e-5)


def test_empty_context_candidates_anchors_and_fp32_residual_under_amp():
    model, batch = _model(), _pp_batch()
    batch["anchor_x"] = torch.empty(0, 6)
    batch["anchor_go_edge"] = torch.empty(2, 0, dtype=torch.long)
    batch["neighbor_index"] = torch.full((4, 0), -1)
    batch["neighbor_attr"] = torch.empty(4, 0, 3)
    batch["candidate_go"] = torch.full((4, 0), -1)
    batch["candidate_attr"] = torch.empty(4, 0, 3)
    batch["base_logits"] = torch.full((4, 8), 2.0)
    with torch.no_grad():
        model.correction_decoder[-1].bias.fill_(0.000411)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(batch, return_details=True)
        loss = F.binary_cross_entropy_with_logits(result["logits"], batch["targets"])
    loss.backward()
    assert result["logits"].dtype == torch.float32
    assert torch.all(result["logits"] > 2.0004)
    assert not result["core_support_mass"].any()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_chunk_recomputation_outputs_gradients_and_rng_match():
    torch.manual_seed(29)
    left = _model(candidate_dropout=0.2, activation_checkpointing=False)
    right = _model(candidate_dropout=0.2, activation_checkpointing=True)
    _open_residual(left)
    right.load_state_dict(left.state_dict())
    batch = _pp_batch()
    torch.manual_seed(199)
    a = left(batch, return_details=True)
    F.binary_cross_entropy_with_logits(a["logits"], batch["targets"]).backward()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(199)
    b = right(batch, return_details=True)
    F.binary_cross_entropy_with_logits(b["logits"], batch["targets"]).backward()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key in a:
        if a[key] is not None:
            torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)
    for (name, lp), (_, rp) in zip(left.named_parameters(), right.named_parameters()):
        assert lp.grad is not None and rp.grad is not None, name
        torch.testing.assert_close(lp.grad, rp.grad, atol=1e-6, rtol=1e-5, msg=name)


def test_chunk_size_attention_backend_and_cached_go_agree():
    model, batch = _model().eval(), _pp_batch()
    _open_residual(model)
    expected = model(batch)
    torch.testing.assert_close(model(batch, go_encoding=model.encode_go()), expected)
    model.config = replace(model.config, go_chunk=1, attention_backend="math")
    torch.testing.assert_close(model(batch), expected, atol=1e-6, rtol=1e-5)
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        model(batch, go_encoding=model.encode_go())


def test_graph_signal_can_be_learned_when_own_features_and_base_cannot_separate_targets():
    torch.manual_seed(813)
    model, batch = _model(activation_checkpointing=False), _pp_batch()
    # All target proteins and their base predictions are identical. Only their
    # distinct core neighbours identify which GO is positive. This cannot be
    # solved by memorizing the target protein projection or a global bias.
    batch["protein_x"].zero_()
    batch["base_logits"].fill_(-1)
    batch["candidate_go"].fill_(-1)
    batch["neighbor_index"] = torch.arange(4)[:, None]
    batch["neighbor_attr"] = torch.ones(4, 1, 3)
    batch["anchor_go_edge"] = torch.tensor([[0, 1, 2, 3], [0, 2, 4, 6]])
    batch["targets"] = torch.zeros(4, 8)
    batch["targets"][torch.arange(4), torch.arange(4) * 2] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = F.binary_cross_entropy_with_logits(model(batch), batch["targets"]).item()
    for _ in range(35):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(batch), batch["targets"])
        loss.backward()
        optimizer.step()
    after = F.binary_cross_entropy_with_logits(model(batch), batch["targets"]).item()
    assert after < before * 0.65
    graph_off = F.binary_cross_entropy_with_logits(
        model(batch, use_weak_go=False, use_core_go=False), batch["targets"]).item()
    assert graph_off > after + 0.1
