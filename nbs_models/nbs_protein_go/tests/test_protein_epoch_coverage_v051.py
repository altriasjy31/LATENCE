from __future__ import annotations

from pathlib import Path

import numpy as np

from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.latence_stores import GOProteinCSRStore, RoleAwareBaseLogitStore, RoleProbabilitySlice
from nbs_pg.local_loader import NBSLocalGraphSamplingConfig
from nbs_pg.training import _merge_ddp_protein_coverage


class _FakeContext:
    def __init__(self, extra_payload):
        self.extra_payload = extra_payload

    def all_gather_object(self, value):
        return [value, self.extra_payload]


def _save(path: Path, value: np.ndarray) -> Path:
    np.save(path, value)
    return path


def test_packed_ddp_protein_coverage_union():
    local = {
        "root": np.array([True, False, True, False, False], dtype=np.bool_),
        "local": np.array([True, True, True, False, False], dtype=np.bool_),
    }
    remote = {
        "root": np.packbits(
            np.array([False, True, False, True, False], dtype=np.bool_),
            bitorder="little",
        ).tobytes(),
        "local": np.packbits(
            np.array([False, False, False, True, True], dtype=np.bool_),
            bitorder="little",
        ).tobytes(),
    }
    counts = _merge_ddp_protein_coverage(_FakeContext(remote), local)
    assert counts == {"root": 4, "local": 5}


def test_epoch_unit_is_go_coverage_and_stage1_reference_is_diagnostic():
    config = NBSLocalGraphSamplingConfig.from_mapping(
        {
            "steps_per_epoch_per_rank": None,
            "coverage_cycles_per_epoch": 1.0,
            "epoch_unit": "eligible_go_coverage_cycle",
            "stage1_reference_global_steps_per_epoch": 4384,
        }
    )
    assert config.epoch_unit == "eligible_go_coverage_cycle"
    assert config.stage1_reference_global_steps_per_epoch == 4384


def test_episode_exposes_transient_root_and_target_ids(tmp_path: Path):
    gold_indptr = _save(tmp_path / "gold_indptr.npy", np.array([0, 3, 6], np.int64))
    gold_protein = _save(
        tmp_path / "gold_protein.npy", np.array([0, 1, 2, 3, 4, 5], np.int32)
    )
    candidate_indptr = _save(
        tmp_path / "candidate_indptr.npy", np.array([0, 2, 4], np.int64)
    )
    candidate_protein = _save(
        tmp_path / "candidate_protein.npy", np.array([6, 7, 8, 9], np.int32)
    )
    pseudo_indptr = _save(
        tmp_path / "pseudo_indptr.npy", np.array([0, 1, 2], np.int64)
    )
    pseudo_protein = _save(
        tmp_path / "pseudo_protein.npy", np.array([10, 11], np.int32)
    )
    pseudo_prob = _save(
        tmp_path / "pseudo_prob.npy", np.array([0.8, 0.9], np.float16)
    )
    probability_path = _save(
        tmp_path / "prob.npy", np.full((12, 2), 0.2, dtype=np.float16)
    )

    sampler = GOQueryEpisodeSampler(
        gold=GOProteinCSRStore(gold_indptr, gold_protein),
        candidate=GOProteinCSRStore(candidate_indptr, candidate_protein),
        pseudo=GOProteinCSRStore(
            pseudo_indptr,
            pseudo_protein,
            payload_paths={"probability": pseudo_prob},
        ),
        base_logits=RoleAwareBaseLogitStore(
            [RoleProbabilitySlice("all", 0, 12, str(probability_path))],
            num_go=2,
        ),
        train_go_counts=np.array([3, 3], dtype=np.float64),
        config=NBSQueryEpisodeConfig(
            num_queries=2,
            support_per_query=1,
            gold_positive_per_query=1,
            hard_candidate_per_query=1,
            pseudo_positive_per_query=1,
            max_candidates=8,
        ),
        seed=7,
    )
    episode = sampler.sample(seed=11)
    metadata = episode.metadata
    for key in (
        "support_protein_idx",
        "candidate_protein_idx",
        "root_protein_idx",
        "gold_target_protein_idx",
        "hard_target_protein_idx",
        "pseudo_target_protein_idx",
    ):
        assert isinstance(metadata[key], np.ndarray)
        assert metadata[key].dtype == np.int64
    assert np.intersect1d(
        metadata["support_protein_idx"], metadata["candidate_protein_idx"]
    ).size == 0
    assert np.array_equal(
        metadata["root_protein_idx"],
        np.unique(
            np.concatenate(
                [metadata["support_protein_idx"], metadata["candidate_protein_idx"]]
            )
        ),
    )
