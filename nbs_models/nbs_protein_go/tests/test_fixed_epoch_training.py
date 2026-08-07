from pathlib import Path

import torch
import torch.nn as nn

from nbs_pg.training import (
    NBSFixedEpochTrainer,
    NBSFixedEpochTrainingConfig,
    NBSLocalBatch,
    NBSLossConfig,
    NBSRunComponents,
)
from nbs_pg.types import NBSMatchOutput, ProteinGOQueryBatch


class _Matcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_delta_scale = nn.Parameter(torch.tensor(0.0))


class _DummyNBS(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.matcher = _Matcher()

    def forward(self, graph, query, global_go_cache=None, return_aux=False):
        del graph, global_go_cache
        base = query.base_logits
        logits = base + self.bias + torch.tanh(self.matcher.graph_delta_scale)
        auxiliary = None
        if return_aux:
            auxiliary = {
                "base_logits": base,
                "applied_graph_delta": logits - base,
                "source_weights": torch.ones(query.num_queries, 1, device=logits.device),
                "null_weight": torch.zeros(query.num_queries, device=logits.device),
            }
        return NBSMatchOutput(
            logits=logits,
            labels=query.labels,
            mask=query.mask,
            confidence=query.confidence,
            pseudo_mask=query.pseudo_mask,
            supervision_weight=query.supervision_weight,
            auxiliary=auxiliary,
        )


def _batch():
    labels = torch.tensor([[1.0, 0.0]])
    query = ProteinGOQueryBatch(
        seed_protein_index=torch.tensor([0]),
        seed_query_index=torch.tensor([0]),
        num_queries=1,
        candidate_protein_index=torch.tensor([0, 1]),
        base_logits=torch.zeros_like(labels),
        labels=labels,
        mask=torch.ones_like(labels, dtype=torch.bool),
        supervision_weight=torch.ones_like(labels),
    )
    return NBSLocalBatch(graph=None, query=query)


def test_fixed_epoch_training_saves_named_snapshots_without_best(tmp_path: Path):
    model = _DummyNBS()
    components = NBSRunComponents(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
        train_loader=[_batch(), _batch()],
        loss_config=NBSLossConfig(primary="bce"),
    )
    config = NBSFixedEpochTrainingConfig(
        epochs=3,
        save_epochs=(2, 3),
        output_dir=str(tmp_path),
        checkpoint_prefix="nbs",
        amp=False,
        log_interval=100,
    )
    trainer = NBSFixedEpochTrainer(components, config, device="cpu", logger=lambda _: None)
    history = trainer.fit()
    assert len(history) == 3
    assert (tmp_path / "nbs_epoch2.pt").exists()
    assert (tmp_path / "nbs_epoch3.pt").exists()
    assert not list(tmp_path.glob("*best*"))
    checkpoint = torch.load(tmp_path / "nbs_epoch3.pt", map_location="cpu", weights_only=False)
    assert checkpoint["selection_policy"]["validation_used"] is False
    assert checkpoint["selection_policy"]["early_stopping"] is False
    assert checkpoint["selection_policy"]["mode"] == "fixed_epoch_snapshots"


def test_validation_and_early_stopping_are_rejected():
    for kwargs in ({"validation_used": True}, {"early_stopping": True}):
        config = NBSFixedEpochTrainingConfig(epochs=1, save_epochs=(1,), **kwargs)
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid model-selection policy was accepted")


def test_save_interval_is_combined_with_named_and_final_epochs():
    config = NBSFixedEpochTrainingConfig(
        epochs=12,
        save_epochs=(7,),
        save_interval_epochs=5,
        save_final=True,
    )
    config.validate()
    assert config.epochs_to_save() == {5, 7, 10, 12}


def test_legacy_save_every_alias_is_supported_but_conflicts_are_rejected():
    config = NBSFixedEpochTrainingConfig.from_mapping(
        {"epochs": 10, "save_epochs": [], "save_every": 5}
    )
    assert config.save_interval_epochs == 5
    assert config.epochs_to_save() == {5, 10}
    try:
        NBSFixedEpochTrainingConfig.from_mapping(
            {
                "epochs": 10,
                "save_epochs": [],
                "save_every": 5,
                "save_interval_epochs": 10,
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting checkpoint interval aliases were accepted")
