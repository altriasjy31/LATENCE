"""Closed-set composition and binding checks; no production PyTorch needed."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/nbs/eval_nbs_shared_head_v091.py"
spec = importlib.util.spec_from_file_location("shared_eval_tests", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class SharedEvaluationTests(unittest.TestCase):
    def test_residual_composition_preserves_old_classifier(self):
        rng = np.random.default_rng(42)
        h = rng.normal(size=(9, 7)).astype(np.float32)
        old_w = rng.normal(size=(11, 7)).astype(np.float32)
        old_b = rng.normal(size=11).astype(np.float32)
        q = rng.normal(size=(11, 7)).astype(np.float32)
        b = rng.normal(size=11).astype(np.float32)
        rw, rb = evaluation.compose_residuals(old_w, old_b, q, b)
        np.testing.assert_allclose(h @ (q + rw).T + b + rb, h @ old_w.T + old_b, atol=2e-6, rtol=2e-6)

    def test_invalid_head_dimensions_or_nonfinite_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluation.compose_residuals(np.zeros((3, 2)), np.zeros(3), np.zeros((2, 3)), np.zeros(2))
        q = np.zeros((3, 2)); q[0, 0] = np.nan
        with self.assertRaises(ValueError):
            evaluation.compose_residuals(np.zeros((3, 2)), np.zeros(3), q, np.zeros(3))

    def test_checkpoint_bindings_and_step_are_strict(self):
        config = {"source_epoch": 60, "head": {"hidden_dim": 7}, "train": {"steps": 500}}
        state = {"version": "0.9.1-shared-head-stage1", "config_sha256": "cfg", "cache_manifest_sha256": "cache", "source_checkpoint_sha256": "src",
                 "source_epoch": 60, "head_config": {"hidden_dim": 7}, "step": 500, "head_state": {},
                 "training_implementation": {"module.py": "digest"}}
        evaluation.validate_head_binding(state, config, "cfg", "cache", "src")
        for key in ("version", "config_sha256", "cache_manifest_sha256", "source_checkpoint_sha256", "source_epoch", "head_config", "step"):
            changed = dict(state); changed[key] = None
            with self.assertRaises(ValueError):
                evaluation.validate_head_binding(changed, config, "cfg", "cache", "src")

    def test_implementation_bytes_and_paths_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "scripts/nbs/module.py"
            path.parent.mkdir(parents=True); path.write_text("reviewed\n")
            records = {"scripts/nbs/module.py": evaluation.sha256(path)}
            evaluation.validate_training_implementation(records, root)
            path.write_text("changed\n")
            with self.assertRaises(ValueError):
                evaluation.validate_training_implementation(records, root)
            with self.assertRaises(ValueError):
                evaluation.validate_training_implementation({"../outside.py": "x"}, root)

    def test_probabilities_and_metrics_both_gate_composition(self):
        p = np.array([[.2, .8]], np.float32)
        metrics = {"fmax": 57.8, "micro_ap": 51.4}
        self.assertTrue(evaluation.compare_composed(p, p, metrics, metrics)["passed"])
        self.assertFalse(evaluation.compare_composed(p, p + .01, metrics, metrics)["passed"])
        changed = {"fmax": 57.5, "micro_ap": 51.4}
        self.assertFalse(evaluation.compare_composed(p, p, metrics, changed)["passed"])

    def test_core_frequency_counts_use_core_only_and_preserve_columns(self):
        arrays = {"label_indptr.i64.npy": np.array([0, 2, 3, 5]),
                  "is_weak.bool.npy": np.array([False, True, False]),
                  "label_go_idx.i32.npy": np.array([0, 2, 1, 1, 2])}
        np.testing.assert_array_equal(evaluation.core_counts_from_cache(arrays, 4), [1, 1, 2, 0])

    def test_cli_requires_explicit_inputs_but_help_needs_no_torch(self):
        args = evaluation.parser().parse_args(["--config", "config.json", "--head-checkpoint", "shared_head_step500.pt",
            "--input-dir", "inputs", "--metadata-file", "metadata.pkl", "--output-dir", "eval"])
        self.assertEqual(args.head_checkpoint.name, "shared_head_step500.pt")


if __name__ == "__main__":
    unittest.main()
