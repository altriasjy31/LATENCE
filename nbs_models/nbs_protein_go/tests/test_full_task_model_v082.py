"""Readout, information-flow and optimization contracts on small real graphs."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nbs_pg.full_task_model_v082 import FullTaskGraphModelV082, FullTaskModelConfigV082
from test_full_task_model import _batch, _ontology


def _model(variant="dual", **kwargs):
    options = dict(hidden_dim=16, query_dim=8, ontology_layers=1, decoder_hidden=16,
                   go_chunk=3, candidate_dropout=0.0, query_budget=4)
    options.update(kwargs)
    return FullTaskGraphModelV082(6, _ontology(), FullTaskModelConfigV082(**options),
                                 variant=variant)


def _objective(result, batch):
    target = batch["targets"]
    classification = F.binary_cross_entropy_with_logits(result["classification_logits"], target)
    mask = result["selected_mask"]
    if mask.any():
        classification = classification + F.binary_cross_entropy_with_logits(
            result["logits"][mask], target[mask])
    return classification


@pytest.mark.parametrize("variant", ["classifier", "dual"])
def test_identity_initialization_then_both_graph_paths_and_class_rows_learn(variant):
    torch.manual_seed(73)
    model, batch = _model(variant), _batch()
    initial = model(batch, return_details=True)
    assert torch.equal(initial["logits"], batch["base_logits"])
    assert model.classification_output.weight.shape == (8, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    for _ in range(3):
        optimizer.zero_grad()
        _objective(model(batch, return_details=True), batch).backward()
        optimizer.step()
    prefixes = ["classification_hidden", "classification_output", "protein_encoder",
                "box_center_encoder", "go_layers", "weak_value", "core_value",
                "weak_edge_encoder", "core_edge_encoder", "anchor_update"]
    if variant == "dual":
        prefixes.extend(["query", "weak_key", "core_key", "match_hidden", "match_output"])
    for prefix in prefixes:
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix)]
        assert gradients and all(x is not None and torch.isfinite(x).all() for x in gradients), prefix
        assert any(torch.count_nonzero(x) > 0 for x in gradients), prefix
    assert all(p.grad is not None for p in model.parameters())
    assert not torch.equal(model(batch), batch["base_logits"])


def test_dual_only_refines_selected_columns_and_decodes_only_budget_queries():
    torch.manual_seed(41)
    model, batch = _model().eval(), _batch()
    torch.nn.init.normal_(model.classification_output.weight, std=.1)
    torch.nn.init.normal_(model.match_output.weight, std=.1)
    calls = []
    hook = model.match_hidden.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape))
    result = model(batch, return_details=True)
    hook.remove()
    mask = result["selected_mask"]
    assert torch.equal(mask.sum(1), torch.full((4,), 4))
    assert result["selected_indices"].shape == (4, 4)
    assert calls == [torch.Size([4, 4, 34])]
    assert torch.equal(result["logits"][~mask], result["classification_logits"][~mask])
    assert torch.count_nonzero(result["logits"][mask] - result["classification_logits"][mask]) > 0
    assert torch.equal(result["match_delta_logits"][~mask], torch.zeros_like(result["logits"][~mask]))
    assert not torch.equal(result["classification_logits"][~mask], batch["base_logits"][~mask])


def test_aliases_have_independent_classification_rows_and_task_outputs():
    ontology = _ontology()
    ontology["task_to_ontology"] = torch.tensor([4, 1, 4, 2, 8, 5, 6, 7])
    model = FullTaskGraphModelV082(6, ontology, _model().config, variant="classifier")
    batch = _batch()
    with torch.no_grad():
        model.classification_output.bias[0] = -.4
        model.classification_output.bias[2] = .7
    go = model.encode_go()
    assert torch.equal(go[0], go[2])
    result = model(batch, return_details=True)
    assert result["logits"].shape == (4, 8)
    torch.testing.assert_close(result["delta_logits"][:, 0], torch.full((4,), -.4))
    torch.testing.assert_close(result["delta_logits"][:, 2], torch.full((4,), .7))
    assert result["selected_indices"].shape == (4, 0)


def test_selector_and_forward_cannot_read_supervision_or_teacher():
    torch.manual_seed(33)
    model, batch = _model().eval(), _batch()
    torch.nn.init.normal_(model.classification_output.weight, std=.1)
    torch.nn.init.normal_(model.match_output.weight, std=.1)
    expected = model(batch, return_details=True)
    changed = dict(batch, targets=1 - batch["targets"],
                   positive_mask=~batch["positive_mask"], is_weak=~batch["is_weak"],
                   teacher_prob=torch.rand_like(batch["targets"]),
                   expert_prob=torch.rand_like(batch["targets"]))
    actual = model(changed, return_details=True)
    for key in expected:
        if expected[key] is not None:
            assert torch.equal(actual[key], expected[key]), key


def test_eval_selection_independent_of_rng_and_batching_and_training_rng_restores():
    model, batch = _model(query_budget=3, selector_high_fraction=0,
                          selector_uncertain_fraction=0, selector_weak_fraction=0,
                          selector_core_fraction=0).eval(), _batch()
    first = model(batch, return_details=True)
    torch.rand(100)
    second = model(batch, return_details=True)
    assert torch.equal(first["selected_indices"], second["selected_indices"])
    one = dict(batch)
    for key in ("protein_x", "base_logits", "candidate_go", "candidate_attr",
                "neighbor_index", "neighbor_attr"):
        one[key] = batch[key][1:2]
    single = model(one, return_details=True)
    assert torch.equal(single["selected_indices"][0], first["selected_indices"][1])
    model.train()
    rng = torch.get_rng_state()
    expected = model(batch, return_details=True)
    torch.rand(200)
    torch.set_rng_state(rng)
    actual = model(batch, return_details=True)
    assert torch.equal(expected["selected_indices"], actual["selected_indices"])


def test_selector_sparse_evidence_fills_budget_without_duplicates_and_full_budget_is_all_go():
    model = _model(query_budget=7, selector_high_fraction=.1,
                   selector_uncertain_fraction=.1, selector_weak_fraction=.4,
                   selector_core_fraction=.4).eval()
    scores = torch.arange(8).float().expand(4, -1)
    pair = torch.zeros(4, 8, 4)
    pair[:, 7, :2] = 1
    indices, mask = model.select_queries(scores, pair, torch.zeros_like(scores))
    assert torch.equal(mask.sum(1), torch.full((4,), 7))
    assert all(len(row.unique()) == 7 for row in indices)
    assert mask[:, 7].all()
    model.config = replace(model.config, query_budget=300)
    _, mask = model.select_queries(scores, pair, torch.zeros_like(scores))
    assert mask.all()


def test_graph_ablations_remove_messages_and_are_not_cosmetic():
    torch.manual_seed(61)
    model, batch = _model().eval(), _batch()
    torch.nn.init.normal_(model.classification_output.weight, std=.1)
    torch.nn.init.normal_(model.match_output.weight, std=.1)
    full = model(batch, return_details=True)
    weak_off = model(batch, use_weak_go=False, return_details=True)
    core_off = model(batch, use_core_go=False, return_details=True)
    assert not torch.equal(full["classification_logits"], weak_off["classification_logits"])
    assert not torch.equal(full["classification_logits"], core_off["classification_logits"])
    assert not torch.equal(full["logits"], weak_off["logits"])
    assert not torch.equal(full["logits"], core_off["logits"])
    assert torch.equal(core_off["core_vote"], torch.zeros_like(core_off["core_vote"]))


def test_empty_neighbours_fp32_residual_and_math_sdpa_parity():
    torch.manual_seed(31)
    model, batch = _model().eval(), _batch()
    batch["candidate_go"][0] = -1
    batch["neighbor_index"][1] = -1
    torch.nn.init.normal_(model.match_output.weight, std=.1)
    first = model(batch, return_details=True)
    model.config = replace(model.config, attention_backend="math")
    second = model(batch, return_details=True)
    torch.testing.assert_close(first["logits"], second["logits"], atol=1e-6, rtol=1e-5)
    batch["anchor_x"] = torch.empty(0, 6)
    batch["anchor_go_edge"] = torch.empty(2, 0, dtype=torch.long)
    batch["neighbor_index"].fill_(-1)
    batch["candidate_go"].fill_(-1)
    batch["base_logits"].fill_(2)
    with torch.no_grad():
        model.classification_output.weight.zero_()
        model.classification_output.bias.fill_(.000411)
        model.match_output.weight.zero_()
        model.match_output.bias.zero_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(batch, return_details=True)
    assert result["logits"].dtype == torch.float32
    assert torch.isfinite(result["logits"]).all()
    assert torch.all(result["logits"] > 2.0004)


def test_checkpointed_matching_preserves_gradients_and_selector_rng():
    torch.manual_seed(23)
    first = _model(candidate_dropout=.3, activation_checkpointing=False)
    second = _model(candidate_dropout=.3, activation_checkpointing=True)
    torch.nn.init.normal_(first.classification_output.weight, std=.1)
    torch.nn.init.normal_(first.match_output.weight, std=.1)
    second.load_state_dict(first.state_dict())
    batch = _batch()
    torch.manual_seed(811)
    expected = first(batch, return_details=True)
    _objective(expected, batch).backward()
    rng = torch.get_rng_state()
    torch.manual_seed(811)
    actual = second(batch, return_details=True)
    _objective(actual, batch).backward()
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(expected["selected_indices"], actual["selected_indices"])
    torch.testing.assert_close(actual["logits"], expected["logits"])
    for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters()):
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5, msg=name)


@pytest.mark.parametrize("variant", ["classifier", "dual"])
def test_small_full_multilabel_problem_optimizes_both_heads(variant):
    torch.manual_seed(81)
    model, batch = _model(variant), _batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    initial = float(F.binary_cross_entropy_with_logits(model(batch), batch["targets"]))
    for _ in range(40):
        optimizer.zero_grad()
        _objective(model(batch, return_details=True), batch).backward()
        optimizer.step()
    model.eval()
    result = model(batch, return_details=True)
    classification = float(F.binary_cross_entropy_with_logits(result["classification_logits"], batch["targets"]))
    final = float(F.binary_cross_entropy_with_logits(result["logits"], batch["targets"]))
    assert classification < initial * .5
    assert final < initial * .5


def test_go_cache_eval_only_and_invalid_configuration_rejected():
    model, batch = _model().eval(), _batch()
    torch.testing.assert_close(model(batch), model(batch, go_encoding=model.encode_go()))
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        model(batch, go_encoding=model.encode_go())
    with pytest.raises(ValueError, match="fractions"):
        _model(selector_high_fraction=1)
