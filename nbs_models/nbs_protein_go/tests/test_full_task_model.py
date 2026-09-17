"""Small end-to-end contracts, not a claim about real LATENCE performance."""

from dataclasses import replace

import pytest
import torch

from nbs_pg.full_task_loss import FullTaskLossConfig, full_task_loss
from nbs_pg.full_task_model import FullTaskGraphModel, FullTaskModelConfig, mean_adjacency


def _ontology():
    generator = torch.Generator().manual_seed(17)
    return {
        "center": torch.randn(10, 6, generator=generator),
        "offset": torch.rand(10, 6, generator=generator) + 0.15,
        "task_to_ontology": torch.arange(8),
        "edges": {
            "is_a": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [8, 8, 8, 8, 9, 9, 9, 9]]),
            "has_child": torch.tensor([[8, 8, 8, 8, 9, 9, 9, 9], [0, 1, 2, 3, 4, 5, 6, 7]]),
        },
    }


def _model():
    return FullTaskGraphModel(
        6, _ontology(),
        FullTaskModelConfig(
            hidden_dim=16, query_dim=8, ontology_layers=1,
            decoder_hidden=16, go_chunk=3, dropout=0,
        ),
    )


def _batch():
    generator = torch.Generator().manual_seed(9)
    labels = torch.zeros(4, 8)
    labels[torch.arange(4), torch.tensor([0, 2, 4, 6])] = 1
    labels[torch.arange(4), torch.tensor([1, 3, 5, 7])] = 0.8
    return {
        "protein_x": torch.randn(4, 6, generator=generator),
        "base_logits": torch.randn(4, 8, generator=generator) * 0.2 - 1,
        "candidate_go": torch.tensor([[0, 4], [2, 0], [4, 6], [6, 2]]),
        "candidate_attr": torch.tensor([
            [[0.7, 0.9, 1.0], [0.4, 0.5, 0.5]],
            [[0.8, 0.8, 1.0], [0.3, 0.2, 0.5]],
            [[0.9, 0.6, 1.0], [0.2, 0.4, 0.5]],
            [[0.6, 0.7, 1.0], [0.5, 0.1, 0.5]],
        ]),
        "anchor_x": torch.randn(4, 6, generator=generator),
        "anchor_go_edge": torch.tensor([[0, 0, 1, 2, 3, 3], [1, 4, 3, 5, 7, 2]]),
        "neighbor_index": torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
        "neighbor_attr": torch.tensor([
            [[0.8, 0.4, 1.0], [0.3, 0.7, 0.5]],
            [[0.7, 0.6, 1.0], [0.4, 0.2, 0.5]],
            [[0.9, 0.8, 1.0], [0.2, 0.4, 0.5]],
            [[0.6, 0.5, 1.0], [0.5, 0.3, 0.5]],
        ]),
        "targets": labels,
        "positive_mask": labels > 0,
        "is_weak": torch.tensor([True, True, False, False]),
    }


def _objective(model, batch):
    return full_task_loss(
        model(batch), batch["base_logits"], batch["targets"],
        batch["positive_mask"], batch["is_weak"],
        FullTaskLossConfig(hard_pu_k=8, background_pu_k=0),
    )[0]


def test_canonical_aliases_share_geometry_without_merging_task_predictions_or_loss():
    torch.manual_seed(103)
    ontology = _ontology()
    # Nonconsecutive aliases and non-sorted source rows must preserve task order.
    ontology["task_to_ontology"] = torch.tensor([4, 1, 4, 2, 8, 5, 6, 7])
    model = FullTaskGraphModel(6, ontology, _model().config)
    batch = _batch()
    batch["base_logits"][0, 0] = -2
    batch["base_logits"][0, 2] = 2
    initial = model(batch, return_details=True)
    assert initial["logits"].shape == (4, 8)
    assert torch.equal(initial["logits"], batch["base_logits"])
    go = model.encode_go()
    assert go.shape == (8, model.config.hidden_dim)
    assert torch.equal(go[0], go[2])
    assert not torch.equal(go[0], go[1])

    # Aliases do not overwrite each other's candidate evidence or core votes.
    _, _, evidence, votes = model.encode_proteins(
        batch, go, use_weak_go=True, use_core_go=True)
    assert evidence[0, 0, 0] == 1 and evidence[0, 2, 0] == 0
    assert votes[2, 0] == 0 and votes[2, 2] > 0

    # Supervision remains per classifier column, including differing alias targets.
    logits = initial["logits"]
    logits.retain_grad()
    loss, _ = full_task_loss(logits, batch["base_logits"], batch["targets"],
                            batch["positive_mask"], batch["is_weak"],
                            FullTaskLossConfig(hard_pu_k=8, background_pu_k=0))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 2] > 0
    assert torch.isfinite(model.decoder[-1].weight.grad).all()
    assert torch.count_nonzero(model.decoder[-1].weight.grad) > 0


@pytest.mark.parametrize("mapping", [torch.tensor(0), torch.tensor([[0, 1]])])
def test_task_mapping_still_rejects_non_vector_shapes(mapping):
    ontology = _ontology()
    ontology["task_to_ontology"] = mapping
    with pytest.raises(ValueError, match="one-dimensional"):
        FullTaskGraphModel(6, ontology)


@pytest.mark.parametrize("mapping", [torch.empty(0, dtype=torch.long),
                                    torch.tensor([0, -1]), torch.tensor([0, 10])])
def test_task_mapping_still_rejects_empty_missing_or_out_of_range_rows(mapping):
    ontology = _ontology()
    ontology["task_to_ontology"] = mapping
    with pytest.raises(ValueError, match="outside ontology"):
        FullTaskGraphModel(6, ontology)


def test_single_zero_output_layer_starts_at_base_and_opens_both_graph_paths():
    torch.manual_seed(23)
    model, batch = _model(), _batch()
    initial = model(batch, return_details=True)
    assert torch.equal(initial["logits"], batch["base_logits"])
    assert torch.equal(initial["delta"], torch.zeros_like(initial["delta"]))
    assert initial["routing"].shape == (4, 8, 3)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    peaks = {name: 0.0 for name in (
        "box_center_encoder", "box_offset_encoder", "go_layers", "weak_edge_encoder",
        "core_edge_encoder", "weak_update", "anchor_update", "core_update", "query",
    )}
    for _ in range(5):
        optimizer.zero_grad()
        _objective(model, batch).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all(), name
                for prefix in peaks:
                    if name.startswith(prefix):
                        peaks[prefix] = max(peaks[prefix], float(parameter.grad.norm()))
        optimizer.step()
    assert all(value > 1e-9 for value in peaks.values()), peaks
    assert not torch.equal(model(batch), batch["base_logits"])


def test_forward_ignores_target_supervision_and_is_batch_and_chunk_invariant():
    torch.manual_seed(31)
    model, batch = _model(), _batch()
    # Make the residual nonzero so the contracts cover the actual decoder.
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    model.eval()
    expected = model(batch)
    changed = dict(batch)
    changed["targets"] = torch.rand_like(batch["targets"])
    changed["positive_mask"] = ~batch["positive_mask"]
    changed["is_weak"] = ~batch["is_weak"]
    assert torch.equal(model(changed), expected)

    row_fields = {
        "protein_x", "base_logits", "candidate_go", "candidate_attr", "neighbor_index",
        "neighbor_attr", "targets", "positive_mask", "is_weak",
    }
    pieces = []
    for start, end in ((0, 1), (1, 3), (3, 4)):
        piece = {key: value[start:end] if key in row_fields else value for key, value in batch.items()}
        pieces.append(model(piece))
    assert torch.allclose(torch.cat(pieces), expected, atol=1e-6, rtol=1e-6)
    model.config = replace(model.config, go_chunk=8)
    assert torch.allclose(model(batch), expected, atol=1e-6, rtol=1e-6)
    model.config = replace(model.config, go_chunk=1)
    assert torch.allclose(model(batch), expected, atol=1e-6, rtol=1e-6)
    cached = model.encode_go()
    assert torch.equal(model(batch, go_encoding=cached), model(batch))
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        model(batch, go_encoding=cached)


def test_relation_direction_delivers_only_to_receivers_and_normalizes_incoming_edges():
    adjacency = mean_adjacency(torch.tensor([[0, 2], [1, 1]]), 3)
    values = torch.tensor([[2.0, 4.0], [100.0, 200.0], [6.0, 8.0]])
    message = torch.sparse.mm(adjacency, values)
    assert torch.equal(message, torch.tensor([[0.0, 0.0], [4.0, 6.0], [0.0, 0.0]]))


def test_empty_neighborhoods_remain_finite_and_cannot_emit_core_votes():
    model, batch = _model(), _batch()
    torch.nn.init.normal_(model.decoder[-1].weight, std=0.1)
    batch["candidate_go"] = torch.full((4, 2), -1)
    batch["neighbor_index"] = torch.full((4, 2), -1)
    batch["anchor_x"] = torch.empty(0, 6)
    batch["anchor_go_edge"] = torch.empty(2, 0, dtype=torch.long)
    details = model(batch, return_details=True)
    assert torch.isfinite(details["logits"]).all()
    assert torch.equal(details["core_vote"], torch.zeros(4, 8))
    assert torch.equal(details["routing"][..., 0], torch.ones(4, 8))
    assert torch.equal(details["routing"][..., 1:], torch.zeros(4, 8, 2))
    _objective(model, batch).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_cpu_bfloat16_preserves_small_fp32_residual_and_finite_backward():
    model, batch = _model(), _batch()
    batch["base_logits"] = torch.full((4, 8), 2.0)
    with torch.no_grad():
        model.decoder[-1].bias.fill_(0.000411)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        details = model(batch, return_details=True)
        loss, _ = full_task_loss(
            details["logits"], batch["base_logits"], batch["targets"],
            batch["positive_mask"], batch["is_weak"],
        )
    assert details["logits"].dtype == torch.float32
    assert torch.all(details["delta"] > 0.0004)
    assert torch.all(details["delta"] < 0.00042)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_two_graph_routes_fit_distinct_positives_with_identical_protein_features():
    torch.manual_seed(5)
    model, batch = _model(), _batch()
    # All targets have the same protein features and the same first-stage scores.
    # Only their weak-GO identity and core-GO neighborhood carry the answer.
    batch["protein_x"] = batch["protein_x"][:1].expand(4, -1).clone()
    batch["anchor_x"] = batch["anchor_x"][:1].expand(4, -1).clone()
    batch["base_logits"] = torch.full((4, 8), -1.0)
    weak_positive = torch.tensor([0, 2, 4, 6])
    core_positive = torch.tensor([1, 3, 5, 7])
    rows = torch.arange(4)
    batch["candidate_go"] = weak_positive[:, None]
    batch["candidate_attr"] = torch.tensor([[[0.7, 0.8, 1.0]]]).expand(4, -1, -1).clone()
    batch["neighbor_index"] = rows[:, None]
    batch["neighbor_attr"] = torch.tensor([[[0.8, 0.4, 1.0]]]).expand(4, -1, -1).clone()
    batch["anchor_go_edge"] = torch.stack((rows, core_positive))
    batch["targets"] = torch.zeros(4, 8)
    batch["targets"][rows, weak_positive] = 1
    batch["targets"][rows, core_positive] = 1
    batch["positive_mask"] = batch["targets"] > 0
    batch["is_weak"] = torch.ones(4, dtype=torch.bool)
    initial_loss = _objective(model, batch).detach()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    for _ in range(100):
        optimizer.zero_grad()
        loss = _objective(model, batch)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        full = model(batch)
        weak_off = model(batch, use_weak_go=False)
        core_off = model(batch, use_core_go=False)
    assert loss < initial_loss * 0.15
    top_two = full.topk(2, dim=1).indices
    assert batch["positive_mask"].gather(1, top_two).all()
    assert (full[rows, weak_positive] > weak_off[rows, weak_positive] + 0.5).all()
    assert (full[rows, core_positive] > core_off[rows, core_positive] + 0.5).all()
    assert full[~batch["positive_mask"]].mean() < batch["base_logits"].mean()


def test_query_decoder_can_learn_positives_absent_from_exact_pair_evidence():
    torch.manual_seed(11)
    model, batch = _model(), _batch()
    rows = torch.arange(4)
    batch["protein_x"] = batch["protein_x"][:1].expand(4, -1).clone()
    batch["base_logits"] = torch.full((4, 8), -1.0)
    batch["candidate_go"] = rows[:, None]
    batch["candidate_attr"] = torch.tensor([[[0.7, 0.8, 1.0]]]).expand(4, -1, -1).clone()
    # All core neighborhoods are identical.  Only the compressed weak-GO
    # graph state can identify the correct, different GO query for each row.
    batch["neighbor_index"] = torch.zeros(4, 1, dtype=torch.long)
    batch["neighbor_attr"] = torch.tensor([[[0.8, 0.4, 1.0]]]).expand(4, -1, -1).clone()
    batch["anchor_go_edge"] = torch.tensor([[0], [7]])
    target_go = torch.tensor([2, 3, 0, 1])
    batch["targets"] = torch.zeros(4, 8)
    batch["targets"][rows, target_go] = 1
    batch["positive_mask"] = batch["targets"] > 0
    batch["is_weak"] = torch.ones(4, dtype=torch.bool)
    assert not (batch["candidate_go"] == target_go[:, None]).any()
    assert not (model(batch, return_details=True)["core_vote"][rows, target_go] > 0).any()
    initial_loss = _objective(model, batch).detach()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(120):
        optimizer.zero_grad()
        loss = _objective(model, batch)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        weak_off = model(batch, use_weak_go=False)
    assert loss < initial_loss * 0.2
    assert torch.equal(logits.argmax(1), target_go)
    assert (logits[rows, target_go] > weak_off[rows, target_go] + 0.5).all()
