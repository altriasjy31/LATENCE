from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from nbs_pg.distributed import NBSDistributedConfig, NBSDistributedContext
from nbs_pg.training import (
    NBSFixedEpochTrainer,
    NBSFixedEpochTrainingConfig,
    NBSLocalBatch,
    NBSRunComponents,
    seed_everything,
)
from nbs_pg.types import ProteinGOQueryBatch


class SmokeLoader:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.epoch = 1

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return 3

    def __iter__(self):
        for local_step in range(len(self)):
            global_index = self.rank + local_step * self.world_size
            value = float(self.epoch * 10 + global_index)
            query = ProteinGOQueryBatch(
                seed_protein_index=torch.zeros(1, dtype=torch.long),
                seed_query_index=torch.zeros(1, dtype=torch.long),
                num_queries=1,
            )
            yield NBSLocalBatch(graph=torch.tensor([[value]], dtype=torch.float32), query=query)


def smoke_loss(model, batch, _config):
    output = model(batch.graph)
    loss = (output - 1.0).pow(2).mean()
    return loss, {"total": loss.detach()}


def main():
    distributed_config = NBSDistributedConfig(enabled=True, backend="gloo", find_unused_parameters=False)
    context = NBSDistributedContext.from_environment(distributed_config, initialize=True)
    seed_everything(123)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    components = NBSRunComponents(
        model=model,
        optimizer=optimizer,
        train_loader=SmokeLoader(context.rank, context.world_size),
    )
    output = Path(os.environ["NBS_DDP_SMOKE_DIR"])
    trainer = NBSFixedEpochTrainer(
        components,
        NBSFixedEpochTrainingConfig(
            epochs=1,
            save_epochs=(1,),
            output_dir=str(output),
            checkpoint_prefix="smoke",
            amp=False,
            log_interval=100,
        ),
        device="cpu",
        distributed_context=context,
        distributed_config=distributed_config,
        forward_loss_fn=smoke_loss,
    )
    trainer.fit()
    context.barrier()
    if context.is_main_process:
        assert (output / "smoke_epoch1.pt").exists()
        assert (output / "training_history.json").exists()
    context.cleanup()


if __name__ == "__main__":
    main()
