from __future__ import annotations

from pathlib import Path

import torch

from nbs_pg import NBSConfig, ProteinGONBSModel, mask_candidate_evidence_edges
from nbs_pg.training import (
    NBSFixedEpochTrainer,
    NBSFixedEpochTrainingConfig,
    NBSLocalBatch,
    NBSLossConfig,
    NBSRunComponents,
)

from toy_train import make_query, make_toy_graph


def main() -> None:
    graph = make_toy_graph()
    query = make_query()
    graph = mask_candidate_evidence_edges(
        graph,
        query.candidate_protein_index,
        query_go_index=query.query_go_index,
        gold_mode="all",
        pseudo_mode="all",
        candidate_mode="query_only",
    )
    model = ProteinGONBSModel(
        NBSConfig(hidden_dim=64, num_layers=2, go_tower_layers=1),
        protein_input_dim=graph["protein"].x.size(1),
        go_box_dim=graph["go"].center.size(1),
    )
    components = NBSRunComponents(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=2e-3),
        train_loader=[NBSLocalBatch(graph=graph, query=query)],
        loss_config=NBSLossConfig(primary="asl"),
        metadata={"example": "fixed_epoch_train"},
    )
    trainer = NBSFixedEpochTrainer(
        components,
        NBSFixedEpochTrainingConfig(
            epochs=5,
            save_epochs=(3, 5),
            output_dir=str(Path("outputs") / "toy_nbs_fixed_epoch"),
            amp=False,
            log_interval=1,
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.fit()


if __name__ == "__main__":
    main()
