from __future__ import annotations

import torch

from nbs_pg import (
    NBSConfig,
    NBSLossWeights,
    ProteinGONBSModel,
    ProteinGOQueryBatch,
    build_nbs_protein_go_heterodata,
    mask_candidate_evidence_edges,
    nbs_training_loss,
)


def make_toy_graph():
    g = torch.Generator().manual_seed(7)
    num_proteins, num_go, box_dim = 48, 15, 8
    protein_x = torch.randn(num_proteins, 32, generator=g)
    go_center = torch.randn(num_go, box_dim, generator=g)
    # Parents are deliberately larger than children in this toy hierarchy.
    go_offset = 0.05 + torch.rand(num_go, box_dim, generator=g)
    go_offset[0] += 1.0

    src = torch.arange(num_proteins)
    ppi = torch.stack([src, (src + 1) % num_proteins])
    similarity = torch.randint(0, num_proteins, (2, 80), generator=g)
    weak_to_core = torch.stack(
        [torch.arange(24, num_proteins), torch.arange(24, num_proteins) % 12]
    )
    true_pg = torch.stack(
        [torch.arange(24), torch.arange(24) % num_go]
    )
    candidate_pg = torch.stack(
        [torch.arange(24, num_proteins), (torch.arange(24, num_proteins) + 2) % num_go]
    )
    pseudo_pg = torch.stack(
        [torch.arange(24, num_proteins), torch.arange(24, num_proteins) % num_go]
    )
    pseudo_conf = torch.linspace(0.55, 0.95, pseudo_pg.size(1))
    go_is_a = torch.stack(
        [torch.arange(1, num_go), (torch.arange(1, num_go) - 1) // 2]
    )
    return build_nbs_protein_go_heterodata(
        protein_x,
        go_center,
        go_offset,
        go_is_a,
        ppi_edge_index=ppi,
        similarity_edge_index=similarity,
        weak_to_core_edge_index=weak_to_core,
        gold_protein_go_edge_index=true_pg,
        backbone_candidate_protein_go_edge_index=candidate_pg,
        pseudo_protein_go_edge_index=pseudo_pg,
        pseudo_annotation_confidence=pseudo_conf,
    )


def make_query():
    candidates = torch.arange(24, 48)
    seed = torch.tensor([0, 1, 2, 3])
    seed_query = torch.tensor([0, 0, 1, 1])
    query_go = torch.tensor([0, 1, 5, 6])
    go_query = torch.tensor([0, 0, 1, 1])
    labels = torch.zeros(2, candidates.numel())
    labels[0, torch.arange(candidates.numel()) % 5 == 0] = 1
    labels[1, torch.arange(candidates.numel()) % 7 == 0] = 1
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :16] = True  # remaining entries are unknown, not negative
    pseudo_mask = mask.clone()
    pseudo_mask[:, :4] = False
    confidence = torch.ones_like(labels)
    confidence[pseudo_mask] = 0.7
    base_logits = torch.randn_like(labels) * 0.2
    candidate_evidence = torch.sigmoid(base_logits + 0.1)
    return ProteinGOQueryBatch(
        seed_protein_index=seed,
        seed_query_index=seed_query,
        num_queries=2,
        query_go_index=query_go,
        go_query_index=go_query,
        candidate_protein_index=candidates,
        base_logits=base_logits,
        candidate_evidence=candidate_evidence,
        query_go_frequency=torch.tensor([0.02, 0.20]),
        labels=labels,
        mask=mask,
        confidence=confidence,
        pseudo_mask=pseudo_mask,
    )


def main() -> None:
    graph = make_toy_graph()
    query = make_query()
    # Candidate annotations must not be visible during candidate prediction.
    graph = mask_candidate_evidence_edges(
        graph,
        query.candidate_protein_index,
        query_go_index=query.query_go_index,
        gold_mode="all",
        pseudo_mode="all",
        candidate_mode="query_only",
    )
    config = NBSConfig(
        hidden_dim=64,
        num_layers=2,
        go_tower_layers=1,
        score_chunk_size=16,
        relation_aggr="gated_sum",
    )
    model = ProteinGONBSModel(
        config,
        protein_input_dim=graph["protein"].x.size(1),
        go_box_dim=graph["go"].center.size(1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(graph, query, return_aux=True)
        loss, parts = nbs_training_loss(
            output,
            weights=NBSLossWeights(anchor=0.1, pseudo=0.5),
        )
        loss.backward()
        optimizer.step()
        print(
            f"step={step} loss={loss.item():.4f} "
            f"graph_scale={output.auxiliary['graph_delta_scale'].item():.4f} "
            f"true={parts['true'].item():.4f} pseudo={parts['pseudo'].item():.4f}"
        )


if __name__ == "__main__":
    main()
