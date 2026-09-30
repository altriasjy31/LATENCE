"""v084 sampling with labels hidden for the complete DDP supervision batch.

Feature/candidate/topology caches and external inference are unchanged.  The
caller supplies this update's global seed IDs; no exclusion set is retained on
the data object, so training cannot contaminate subsequent validation batches.
"""
from __future__ import annotations

import numpy as np
import torch

from .full_task_data_v084 import FullTaskDataV084


class FullTaskDataV087(FullTaskDataV084):
    SAMPLING_ARCHITECTURE_VERSION = "0.8.4"

    def data_contract(self):
        result = super().data_contract()
        result["v087_label_mask"] = (
            "global_DDP_supervision_seeds_when_provided_else_local_batch; "
            "all_holdout_labels_always_excluded")
        return result

    def batch(self, protein_ids, device="cpu", *, supervision_seed_ids=None):
        if supervision_seed_ids is None:
            # Exact old behavior is important for validation and legacy controls.
            return super().batch(protein_ids, device=device)
        seeds = np.asarray(supervision_seed_ids)
        if (seeds.ndim != 1 or not len(seeds) or seeds.dtype.kind not in "iu" or
                np.any(seeds < 0) or np.any(seeds >= self.registry.num_proteins)):
            raise ValueError("global supervision seed IDs must be a nonempty vector of valid integer protein IDs")
        seeds = seeds.astype(np.int64, copy=False)
        if len(np.unique(seeds)) != len(seeds):
            raise ValueError("global supervision seed IDs must be unique")
        roles = self.registry.role_code[seeds]
        allowed = np.isin(roles, [self.registry.role_to_code["core"],
                                  self.registry.role_to_code["weak"]])
        if not allowed.all():
            raise ValueError("global supervision seeds must have core or weak roles")
        local_ids = np.asarray(protein_ids, dtype=np.int64).reshape(-1)
        if not np.isin(local_ids, seeds).all():
            raise ValueError("global supervision seed IDs must include every local supervision seed")
        batch = super().batch(protein_ids, device=device)
        sampled_ids = batch["sampled_global_ids"].long()
        excluded = torch.as_tensor(np.array(seeds, copy=True), device=sampled_ids.device)

        def hide(key, source_ids):
            edge = batch[key]
            keep = ~torch.isin(source_ids[edge[0].long()], excluded)
            batch[key] = edge[:, keep]

        hide("sampled_gold_edge", sampled_ids)
        hide("sampled_pseudo_edge", sampled_ids)
        anchor_ids = sampled_ids[batch["sampled_anchor_index"].long()]
        hide("anchor_go_edge", anchor_ids)
        # V084 saves the original cosine graph for structural PU support before
        # replacing decoder anchors with its sampled first-hop anchors. Rebuild
        # precisely that original sorted unique anchor order, not the new order.
        loss_anchor_ids = self.anchor_core_ids[np.unique(self._neighbors[local_ids])]
        loss_anchor_ids = torch.as_tensor(np.array(loss_anchor_ids, copy=True),
                                          device=sampled_ids.device)
        if len(loss_anchor_ids) != len(batch["loss_anchor_x"]):
            raise ValueError("fixed loss-support anchor identity does not match its feature rows")
        hide("loss_anchor_go_edge", loss_anchor_ids)
        return batch
