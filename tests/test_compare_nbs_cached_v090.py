"""Toy correctness checks only; no trained-model performance is inferred."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cached_comparison", ROOT / "scripts/nbs/compare_nbs_cached_v090.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CachedFmaxTests(unittest.TestCase):
    def test_native_curve_and_precision_at_recall_without_interpolation(self):
        labels = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        probabilities = np.array([[.9, .8], [.0, .6]], dtype=np.float64)
        got = MODULE.curve_statistics(labels, probabilities)
        self.assertAlmostEqual(got["standard_protein_fmax"], 600 / 7)
        self.assertAlmostEqual(got["standard_protein_precision_at_fmax"], 75.)
        self.assertAlmostEqual(got["standard_protein_recall_at_fmax"], 100.)
        self.assertEqual(got["precision_at_recall"], {"45": 100., "50": 100., "55": 75.})
        self.assertNotIn("curve", got)

    def test_unreachable_recall_is_null_and_exact_threshold_is_strict(self):
        labels = np.array([[1, 1]], dtype=np.uint8)
        probabilities = np.array([[.5, 0.]])
        got = MODULE.curve_statistics(labels, probabilities, save_curve=True)
        self.assertEqual(got["precision_at_recall"]["55"], None)
        self.assertEqual(got["curve"]["recall_percent"][500], 0.)
        self.assertEqual(got["curve"]["recall_percent"][499], 50.)
        self.assertEqual(len(got["curve"]["threshold"]), 1001)

    def test_shape_and_probability_guard(self):
        with self.assertRaisesRegex(ValueError, "equally shaped"):
            MODULE.curve_statistics(np.zeros((2, 3)), np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "finite"):
            MODULE.curve_statistics(np.zeros((1, 1)), np.array([[np.nan]]))

    def test_cache_hash_guard_rejects_tampering_or_missing_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "probability.npy"
            np.save(path, np.array([[.5]], dtype=np.float32))
            digest = MODULE.sha256(path)
            MODULE.verify_hash(path, digest, "cached probability")
            np.save(path, np.array([[.6]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                MODULE.verify_hash(path, digest, "cached probability")
            with self.assertRaisesRegex(ValueError, "missing valid source"):
                MODULE.verify_hash(path, None, "cached probability")

    def test_epoch_discovery_uses_manifest_not_epoch_times_48(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            full = work / "eval_step15/full"
            (full / "metrics_fmax_only").mkdir(parents=True)
            manifest = {"epoch": 25, "cumulative_completed_epochs": 25, "step": 15,
                        "phase_step_in_epoch": 3, "phase_plan": {"steps_per_epoch": 3},
                        "checkpoint_path": "/example/nbs_epoch25.pt"}
            (full / "nbs_full_task_prediction_manifest.json").write_text(json.dumps(manifest))
            (full / "metrics_fmax_only/nbs_w2s_comparison.json").write_text(
                json.dumps({"evaluation_scope": "standard_protein_fmax_only"}))
            path, found, _ = MODULE.find_cache(work, 25)
            self.assertEqual(path.parent.parent.name, "eval_step15")
            self.assertEqual(found["epoch"], 25)
            manifest["cumulative_completed_epochs"] = 24
            (full / "nbs_full_task_prediction_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "found 0"):
                MODULE.find_cache(work, 25)


if __name__ == "__main__":
    unittest.main()
