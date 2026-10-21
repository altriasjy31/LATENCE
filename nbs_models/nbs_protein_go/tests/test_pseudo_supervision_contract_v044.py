from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.latence_graph_stores import ProteinRegistryStore, RoleLocalProteinGOCSRStore
from nbs_pg.local_loader import _validate_supervision_provenance


class _FakeGOStore:
    def __init__(self, proteins, probabilities=None, source_rank=None):
        self.num_go = 1
        self._proteins = np.asarray(proteins, dtype=np.int64)
        self.indptr = np.asarray([0, len(self._proteins)], dtype=np.int64)
        self._prob = None if probabilities is None else np.asarray(probabilities, dtype=np.float32)
        self._rank = None if source_rank is None else np.asarray(source_rank, dtype=np.uint16)

    def get(self, go_idx):
        assert go_idx == 0
        out = {"protein_idx": self._proteins.copy()}
        if self._prob is not None:
            out["probability"] = self._prob.copy()
        if self._rank is not None:
            out["source_rank"] = self._rank.copy()
        return out


class _FakeBase:
    def gather_matrix(self, proteins, gos):
        return np.zeros((len(gos), len(proteins)), dtype=np.float32)


def test_modelout_positive_is_never_sampled_as_unlabelled_even_when_pseudo_loss_off():
    gold = _FakeGOStore([0, 1])
    candidate = _FakeGOStore([1, 2, 3], source_rank=[0, 1, 2])
    pseudo = _FakeGOStore([2], probabilities=[0.8])
    cfg = NBSQueryEpisodeConfig(
        num_queries=1,
        support_per_query=1,
        gold_positive_per_query=1,
        hard_candidate_per_query=2,
        pseudo_positive_per_query=0,
        max_candidates=8,
    )
    sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=candidate,
        pseudo=pseudo,
        base_logits=_FakeBase(),
        train_go_counts=np.asarray([2.0]),
        config=cfg,
        seed=1,
    )
    episode = sampler.sample(seed=7)
    # Protein 2 is modelout-positive and must not be relabelled as sampled-unlabelled.
    assert 2 not in set(episode.candidate_protein_idx.tolist())
    assert 3 in set(episode.candidate_protein_idx.tolist())


def test_episode_rejects_nonpositive_pseudo_probability():
    gold = _FakeGOStore([0, 1])
    candidate = _FakeGOStore([2], source_rank=[0])
    pseudo = _FakeGOStore([2], probabilities=[0.49])
    cfg = NBSQueryEpisodeConfig(
        num_queries=1,
        support_per_query=1,
        gold_positive_per_query=1,
        hard_candidate_per_query=1,
        pseudo_positive_per_query=0,
        max_candidates=8,
    )
    sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=candidate,
        pseudo=pseudo,
        base_logits=_FakeBase(),
        train_go_counts=np.asarray([2.0]),
        config=cfg,
        seed=1,
    )
    with pytest.raises(ValueError, match="0.5 <= stored probability"):
        sampler.sample(seed=7)


def test_pseudo_message_topk_is_probability_ranked(tmp_path: Path):
    registry = tmp_path / "protein_registry.csv"
    registry.write_text(
        "protein_idx,protein_id,role,role_row_idx,dataset_mode\n"
        "0,P0,weak,0,exp_train\n",
        encoding="utf-8",
    )
    indptr = tmp_path / "indptr.npy"
    go = tmp_path / "go.npy"
    prob = tmp_path / "prob.npy"
    np.save(indptr, np.asarray([0, 3], dtype=np.int64))
    np.save(go, np.asarray([10, 20, 30], dtype=np.int32))
    np.save(prob, np.asarray([0.6, 0.95, 0.8], dtype=np.float16))

    store = RoleLocalProteinGOCSRStore(
        indptr,
        go,
        role="weak",
        registry=ProteinRegistryStore(registry),
        probability_path=prob,
    )
    edge, probability = store.gather([0], topk=2)
    assert edge[1].tolist() == [20, 30]
    assert np.allclose(probability, [0.95, 0.8], atol=1e-3)


def test_runtime_provenance_contract_accepts_gold_train_and_modelout_weak(tmp_path: Path):
    gold_dir = tmp_path / "gold_annotations"
    gold_dir.mkdir()
    edge = gold_dir / "gold_protein_go_edge_index.i32.npy"
    np.save(edge, np.empty((2, 0), dtype=np.int32))
    (gold_dir / "gold_annotations_manifest.json").write_text(
        json.dumps({
            "task": "bp",
            "role": "core",
            "mode": "train",
            "source": {"label_key": "prop_annotations"},
        }),
        encoding="utf-8",
    )
    inverted_manifest = tmp_path / "go_protein_inverted_index_manifest.json"
    inverted_manifest.write_text("{}", encoding="utf-8")
    weak_manifest_path = tmp_path / "weak_graph_predictions_manifest.json"
    weak_manifest_path.write_text("{}", encoding="utf-8")

    config = {
        "task": "bp",
        "supervision_contract": {
            "core_gold": {
                "role": "core",
                "dataset_mode": "train",
                "metadata_label_key": "prop_annotations",
            },
            "weak_pseudo": {
                "role": "weak",
                "dataset_mode": "exp_train",
                "prediction_key": "modelout::mix_expert_base_anchor::decoderprob::expert",
                "decoder_prob_source": "expert",
                "external_probability_used": True,
                "comparison": ">",
                "threshold": 0.5,
                "use_probability_as_soft_target": True,
                "negative_policy": "none",
                "forbid_exp_train_prop_annotations": True,
            },
        },
    }
    weak_manifest = {
        "roles": [
            {"role": "core", "dataset_mode": "train"},
            {"role": "weak", "dataset_mode": "exp_train"},
        ],
        "rare_definition": {"training_annotation_key": "prop_annotations"},
        "model_semantics": {
            "modelout": {
                "prediction_key": "modelout::mix_expert_base_anchor::decoderprob::expert",
                "decoder_prob_source": "expert",
                "external_probability_used": True,
            }
        },
        "weak_pseudo_targets": {
            "role": "weak",
            "comparison": ">",
            "threshold": 0.5,
            "edge_attr_columns": ["modelout_probability"],
        },
    }
    inverted = {
        "indices": {
            "pseudo": {"payloads": {"probability": "pseudo_probability.f16.npy"}},
            "gold": {"source_edge_index": str(edge)},
        }
    }
    _validate_supervision_provenance(
        config,
        weak_manifest=weak_manifest,
        weak_manifest_path=weak_manifest_path,
        inverted=inverted,
        inverted_manifest=inverted_manifest,
    )
