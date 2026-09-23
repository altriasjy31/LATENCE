#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/msaprob.py

Dataset wrapper for attaching teacher probability rows to an existing
MSABinaryDataset. Designed for LATENCE exp_train pseudo-label training.

Rows in prob_path are assumed to follow:
    metadata[mode][task][namekey]

The wrapper aligns probability rows by protein name, not by integer index,
because the binary MSA index may drop proteins that are present in metadata.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


def load_pickle(path: Union[str, Path]):
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


class PseudoProbDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        metadata_file: Union[str, Path],
        mode: str,
        task: str,
        prob_path: Union[str, Path],
        num_classes: int,
        namekey: str = "proteins",
        return_prob_dtype: str = "float16",
    ):
        self.base = base_dataset
        self.metadata_file = str(metadata_file)
        self.mode = str(mode)
        self.task = str(task)
        self.prob_path = str(prob_path)
        self.num_classes = int(num_classes)
        self.namekey = str(namekey)
        self.return_prob_dtype = str(return_prob_dtype)

        meta = load_pickle(self.metadata_file)
        if self.mode not in meta:
            raise KeyError(f"metadata missing mode: {self.mode}")
        if self.task not in meta[self.mode]:
            raise KeyError(f"metadata missing task: {self.mode}/{self.task}")
        if self.namekey not in meta[self.mode][self.task]:
            raise KeyError(f"metadata missing key: {self.mode}/{self.task}/{self.namekey}")

        proteins_meta = [str(p) for p in meta[self.mode][self.task][self.namekey]]

        protein_to_meta_idx = {}
        duplicates = []
        for i, p in enumerate(proteins_meta):
            if p in protein_to_meta_idx:
                duplicates.append(p)
            else:
                protein_to_meta_idx[p] = i
        if duplicates:
            raise RuntimeError(
                f"Duplicate proteins found in metadata {self.mode}/{self.task}. "
                f"Examples: {duplicates[:10]}"
            )

        if not hasattr(base_dataset, "proteins"):
            raise RuntimeError(
                "PseudoProbDataset requires base_dataset.proteins for protein-name alignment."
            )

        base_proteins = [str(p) for p in base_dataset.proteins]
        prob_indices = []
        missing = []
        for p in base_proteins:
            if p not in protein_to_meta_idx:
                missing.append(p)
            else:
                prob_indices.append(protein_to_meta_idx[p])
        if missing:
            raise RuntimeError(
                f"{len(missing)} base dataset proteins are missing from metadata. "
                f"Examples: {missing[:10]}"
            )

        self.prob_indices = np.asarray(prob_indices, dtype=np.int64)

        prob = np.load(self.prob_path, mmap_mode="r")
        if prob.ndim != 2:
            raise ValueError(f"Pseudo prob matrix must be 2D, got shape={prob.shape}")
        if prob.shape[0] != len(proteins_meta):
            raise ValueError(
                "Pseudo prob row count does not match metadata proteins: "
                f"prob_rows={prob.shape[0]}, metadata_rows={len(proteins_meta)}"
            )
        if prob.shape[1] != self.num_classes:
            raise ValueError(
                "Pseudo prob class dimension mismatch: "
                f"prob_classes={prob.shape[1]}, num_classes={self.num_classes}"
            )
        self.prob_shape = tuple(prob.shape)
        self.prob_dtype = str(prob.dtype)
        del prob

        self._prob = None

    @property
    def prob(self):
        if self._prob is None:
            self._prob = np.load(self.prob_path, mmap_mode="r")
        return self._prob

    @property
    def sample_shard_ids(self):
        return self.base.sample_shard_ids

    @property
    def proteins(self):
        return self.base.proteins

    def __len__(self):
        return len(self.base)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_prob"] = None
        return state

    def __getitem__(self, idx):
        input_data, hard_y = self.base[idx]
        prob_idx = int(self.prob_indices[idx])

        if self.return_prob_dtype == "float32":
            arr = np.array(self.prob[prob_idx], dtype=np.float32, copy=True)
        else:
            arr = np.array(self.prob[prob_idx], dtype=np.float16, copy=True)

        prob = torch.from_numpy(arr)
        return input_data, {"hard_y": hard_y, "prob": prob}

class EvalExternalProbDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        external_prob_path,
        external_prob_ids=None,
        metadata_file=None,
        mode="ind_test",
        task=None,
        num_classes=None,
    ):
        self.base = base_dataset
        self.external_prob_path = str(external_prob_path)
        self.num_classes = int(num_classes)

        prob = np.load(self.external_prob_path, mmap_mode="r")

        if prob.ndim != 2:
            raise ValueError(f"external prob must be 2D, got {prob.shape}")

        if prob.shape[1] != self.num_classes:
            raise ValueError(
                f"external prob classes={prob.shape[1]}, "
                f"expected={self.num_classes}"
            )

        self.prob_shape = tuple(prob.shape)
        self.prob_dtype = str(prob.dtype)

        if external_prob_ids is None:
            if len(base_dataset) != prob.shape[0]:
                raise ValueError(
                    "external_prob_ids is None, so row order must match dataset. "
                    f"But len(dataset)={len(base_dataset)}, prob rows={prob.shape[0]}"
                )
            self.row_indices = np.arange(len(base_dataset), dtype=np.int64)

        else:
            with open(external_prob_ids, "r") as f:
                prob_ids = [line.strip().split()[0] for line in f if line.strip()]

            if len(prob_ids) != prob.shape[0]:
                raise ValueError(
                    f"external_prob_ids rows={len(prob_ids)}, "
                    f"prob rows={prob.shape[0]}"
                )

            id_to_row = {p: i for i, p in enumerate(prob_ids)}

            if not hasattr(base_dataset, "proteins"):
                raise RuntimeError(
                    "base_dataset must expose .proteins when external_prob_ids is provided"
                )

            row_indices = []
            missing = []

            for p in base_dataset.proteins:
                p = str(p)
                if p not in id_to_row:
                    missing.append(p)
                else:
                    row_indices.append(id_to_row[p])

            if missing:
                raise RuntimeError(
                    f"{len(missing)} proteins missing from external_prob_ids. "
                    f"Examples: {missing[:10]}"
                )

            self.row_indices = np.asarray(row_indices, dtype=np.int64)

        self._prob = None
        del prob

    @property
    def prob(self):
        if self._prob is None:
            self._prob = np.load(self.external_prob_path, mmap_mode="r")
        return self._prob

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, idx):
        input_data, y = self.base[idx]
        row = int(self.row_indices[idx])
        p = np.array(self.prob[row], dtype=np.float16, copy=True)
        p = torch.from_numpy(p)

        return input_data, {
            "target": y,
            "external_prob": p,
        }