"""Exact AP, frequency membership and report provenance without another forward."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "nbs_models/nbs_protein_go/nbs_pg/evaluation_report_v092.py"
spec = importlib.util.spec_from_file_location("_test_v092_report_module", path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture_data():
    rows = [[], [1], [0, 2, 2], [], [2, 5], []]
    ptr = np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64)
    gold = types.SimpleNamespace(indptr=ptr, go_idx=np.concatenate([np.asarray(row, np.int32) for row in rows]))
    return types.SimpleNamespace(core_ids=np.array([2, 4], np.int64), validation_ids=np.array([1], np.int64),
        weak_ids=np.array([0, 3], np.int64), num_task_go=6, stores=types.SimpleNamespace(gold_messages=gold))


class CoreFrequencyTests(unittest.TestCase):
    def test_global_rows_duplicate_memberships_and_alias_columns(self):
        data = fixture_data()
        expected = np.array([1, 0, 2, 0, 0, 1])
        np.testing.assert_array_equal(report.core_counts(data), expected)
        np.testing.assert_array_equal(data.core_ids, [2, 4])

    def test_holdout_duplicates_and_out_of_range_are_rejected(self):
        for ids in ([1, 2], [2, 2], [2, 99], [-1, 2]):
            data = fixture_data()
            data.core_ids = np.array(ids, np.int64)
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                report.core_counts(data)

    def test_bad_membership_column_and_noninteger_ids_are_rejected(self):
        data = fixture_data()
        data.stores.gold_messages.go_idx[1] = 6
        with self.assertRaises(ValueError):
            report.core_counts(data)
        data = fixture_data()
        data.core_ids = np.array([2., 4.])
        with self.assertRaises(ValueError):
            report.core_counts(data)


@unittest.skipUnless((ROOT / "scripts/nbs/diagnose_nbs_knn_v090.py").is_file(),
                     "requires installation in existing LATENCE runtime")
class ActualMetricReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = report._load_helpers(ROOT)

    def test_all_tied_AP_is_prevalence_not_trapezoidal_area(self):
        labels = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], bool)
        scores = np.full((2, 4), .5, np.float32)
        result = report.summarize_probabilities(labels, scores, np.array([1, 5, 9, 10]), self.helpers)
        self.assertEqual(result["metrics"]["NBS_final"]["micro_ap"], 25.)
        self.assertFalse(result["PR_AUC_computed_here"])
        self.assertEqual(result["num_GO_columns"], 4)

    def test_nested_rare_le5_bin_and_all_columns_are_retained(self):
        labels = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1]], bool)
        scores = np.where(labels, .9, .1).astype(np.float32)
        counts = np.array([0, 1, 5, 9, 10, 1000])
        result = report.summarize_probabilities(labels, {"NBS_final": scores, "backbone_base": scores}, counts, self.helpers)
        bins = {(row["train_core_count_min"], row["train_core_count_max"]): row for row in result["frequency_strata"]}
        self.assertEqual(bins[(0, 0)]["columns"], 1)
        self.assertEqual(bins[(1, 5)]["columns"], 2)
        self.assertEqual(bins[(1, 9)]["columns"], 3)
        self.assertEqual(result["metrics"]["NBS_final"]["micro_ap"], 100.)
        self.assertEqual(result["metrics"]["NBS_final"]["fmax"], 100.)
        self.assertEqual(result["delta_NBS_minus_backbone"], {"fmax": 0., "micro_ap": 0.})
        self.assertTrue(result["frequency_bins_overlap"])
        self.assertFalse(result["zero_shot_performance_claimed"])

    def test_nonfinite_shape_and_column_frequency_mismatch_are_rejected(self):
        labels = np.array([[1, 0, 1]], bool)
        for scores, counts in (([[.2, np.nan, .8]], np.array([1, 2, 3])),
                               ([[.2, .8]], np.array([1, 2, 3])),
                               ([[.2, .3, .8]], np.array([1, 2])),
                               ([[.2, .3, 1.1]], np.array([1, 2, 3]))):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                report.summarize_probabilities(labels, scores, counts, self.helpers)

    def test_write_report_uses_existing_arrays_and_metadata_ID_order(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, work = root / "inputs", root / "work"
            inputs.mkdir()
            (inputs / "ind_test_input_manifest.json").write_text(json.dumps({
                "cache_signature": {"representation_manifest_sha256": "fixture-representation"}}))
            (inputs / "protein_ids.txt").write_text("proteinA\nproteinB\nproteinC\n")
            np.save(inputs / "ind_test_repr.f16.npy", np.zeros((3, 2), np.float16))
            scores = np.array([[.9, .1, .1], [.1, .9, .1], [.1, .1, .9]], np.float32)
            np.save(inputs / "backbone_ind_test_prob.f16.npy", scores.astype(np.float16))
            metadata = root / "metadata.pkl"
            with metadata.open("wb") as stream:
                pickle.dump({"ind_test": {"bp": {"proteins": ["proteinC", "proteinA", "proteinB"],
                            "prop_annotations": [[2], [0], [1]]}}}, stream)
            checkpoint = root / "nbs_epoch5.pt"
            torch.save({"step": 42, "epoch": 5, "runner_version": "0.9.2"}, checkpoint)
            folder = work / "eval_step42/full"
            folder.mkdir(parents=True)
            prediction = folder / "nbs_ind_test_prob.f32.npy"
            np.save(prediction, scores)
            before = prediction.read_bytes()
            provenance = {"step": 42, "checkpoint_sha256": report._sha256(checkpoint),
                "output_probability_sha256": report._sha256(prediction),
                "input_manifest_sha256": report._sha256(inputs / "ind_test_input_manifest.json")}
            (folder / "nbs_full_task_prediction_manifest.json").write_text(json.dumps(provenance))
            gold = types.SimpleNamespace(indptr=np.array([0, 2]), go_idx=np.array([0, 1], np.int32))
            data = types.SimpleNamespace(core_ids=np.array([0]), validation_ids=np.array([], np.int64),
                num_task_go=3, stores=types.SimpleNamespace(gold_messages=gold),
                _signature={"representation_manifest_sha256": "fixture-representation"})
            # A model forward would fail: the report API deliberately receives
            # no model and must consume the exact already-exported arrays.
            args = types.SimpleNamespace(stage="evaluate", checkpoint=checkpoint, work_dir=work,
                input_dir=inputs, metadata_file=metadata, ablation="full")
            with patch.dict(self.helpers.NUM_CLASSES, {"bp": 3}), patch.object(report, "_load_helpers", return_value=self.helpers):
                result_path = report.write_extended_report(args, {"task": "bp"}, data)
            result = json.loads(result_path.read_text())
            self.assertEqual(result_path, folder / "metrics_v092/summary.json")
            self.assertEqual(result["metrics"]["NBS_final"]["micro_ap"], 100.)
            self.assertEqual(result["metrics"]["backbone_base"]["micro_ap"], 100.)
            self.assertEqual(result["checkpoint_step"], 42)
            self.assertEqual(prediction.read_bytes(), before)
            # Altering the exported prediction must not produce a fresh report
            # under the old manifest's provenance.
            np.save(prediction, scores * .5)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                report.write_extended_report(args, {"task": "bp"}, data)


if __name__ == "__main__":
    unittest.main()
