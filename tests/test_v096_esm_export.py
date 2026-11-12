"""Native-source exporter checks. Set ESM_SOURCE_DIR to esm_models/model.py's dir."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_esm_v096", ROOT / "scripts/nbs/export_esm_penultimate_v096.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
SOURCE = Path(os.environ.get("ESM_SOURCE_DIR", ROOT / "esm_models"))
try:
    import torch
except ImportError:
    torch = None


class InputContractTests(unittest.TestCase):
    def test_ids_order_and_rejections(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ids.npy"
            np.save(p, np.array(["P2", "P1"]))
            self.assertEqual(export.load_ids(p).tolist(), ["P2", "P1"])
            for values in [["P1", "P1"], ["../bad"], ["P 1"], []]:
                np.save(p, np.array(values, dtype=str))
                with self.assertRaises(ValueError):
                    export.load_ids(p)

    def test_layer_order_and_missing_validation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "A.npy"
            np.save(p, {"mean": {34: np.array([1., 2.]), 35: np.array([3., 4.]), 36: np.array([5., 6.])}})
            out = export.load_plm_batch(td, ["A"], [34, 35, 36], 2)
            np.testing.assert_array_equal(out, [[1, 2, 3, 4, 5, 6]])
            with self.assertRaises(ValueError):
                export.load_plm_batch(td, ["A"], [33, 35, 36], 2)

    def test_multiple_cache_dirs_reject_conflicting_ids(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a", Path(td) / "b"
            a.mkdir(); b.mkdir()
            raw = {"mean": {i: np.ones(2) for i in [34, 35, 36]}}
            np.save(a / "P1.npy", raw)
            np.save(b / "P2.npy", raw)
            actual = export.load_plm_batch([a, b], ["P1", "P2"], [34, 35, 36], 2)
            self.assertEqual(actual.shape, (2, 6))
            np.save(b / "P1.npy", raw)
            export.load_plm_batch([a, b], ["P1"], [34, 35, 36], 2)
            raw["mean"][34] = np.zeros(2)
            np.save(b / "P1.npy", raw)
            with self.assertRaises(ValueError):
                export.load_plm_batch([a, b], ["P1"], [34, 35, 36], 2)

    def test_latent_ensemble_requires_explicit_concat(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ["a.pt", "b.pt"]:
                (Path(td) / name).write_bytes(b"x")
            args = type("Args", (), {"checkpoint": None, "model_dir": td, "ensemble": "single"})()
            with self.assertRaises(ValueError):
                export.resolve_checkpoints(args)
            args.ensemble = "concat"
            self.assertEqual(len(export.resolve_checkpoints(args)), 2)


@unittest.skipUnless(torch is not None and (SOURCE / "model.py").exists(), "torch and original ESM_SOURCE_DIR required")
class NativeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_type, _ = export.import_source(SOURCE)

    def model(self, add_res=False, seed=42):
        torch.manual_seed(seed)
        return self.model_type(aspect="BPO", layer_list=[7, 5], embed_dim=4,
                               go_term_list=["GO:1", "GO:2", "GO:3"],
                               dropout=.3, activation="elu", seed=seed,
                               add_res=add_res).eval()

    def test_prelinear_exact_with_optional_residual(self):
        for residual in [False, True]:
            model = self.model(residual)
            x = torch.randn(4, 12)
            h, err = export.capture_penultimate(model, x, verify=True)
            fast_h, _ = export.capture_penultimate(model, x)
            self.assertEqual(h.shape, (4, 5))
            self.assertEqual(err, 0)
            torch.testing.assert_close(h, fast_h, atol=0, rtol=0)
            self.assertFalse(h.requires_grad)

    def test_strict_checkpoint_and_aspect(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.pt"
            self.model().save_config(p)
            export.load_model(self.model_type, p, "BP")
            with self.assertRaises(ValueError):
                export.load_model(self.model_type, p, "MF")
            config = torch.load(p, weights_only=True)
            config["state_dict"].pop("branch1.0.weight")
            torch.save(config, p)
            with self.assertRaises(RuntimeError):
                export.load_model(self.model_type, p, "BP")

    def test_native_numpy_config_containers(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "numpy_config.pt"
            for dtype in [str, bytes]:
                with self.subTest(go_dtype=dtype):
                    model = self.model(True)
                    model.layer_list = np.asarray(model.layer_list, dtype=np.int64)
                    model.go_term_list = np.asarray(model.go_term_list, dtype=dtype)
                    model.save_config(path)
                    loaded = export.load_model(self.model_type, path, "BP", trust_checkpoint=True)
                    self.assertEqual(loaded.layer_list, [7, 5])
                    self.assertEqual(loaded.go_term_list, ["GO:1", "GO:2", "GO:3"])
                    x = torch.randn(3, 12)
                    torch.testing.assert_close(model(x), loaded(x), atol=0, rtol=0)

    def setup_export(self, td, ensemble=False):
        td = Path(td)
        (td / "raw").mkdir()
        ids = np.array(["P3", "P1", "P2"])
        np.save(td / "ids.npy", ids)
        rng = np.random.default_rng(4)
        for pid in ids:
            np.save(td / "raw" / (pid + ".npy"), {"mean": {i: rng.normal(size=4).astype(np.float32) for i in [34, 35, 36]}})
        model = self.model(True)
        model.save_config(td / "m1.pt")
        paths = [str(td / "m1.pt")]
        if ensemble:
            self.model(False, 43).save_config(td / "m2.pt")
            paths.append(str(td / "m2.pt"))
        argv = ["--esm-source-dir", str(SOURCE), "--checkpoint"] + paths + [
            "--features-dir", str(td / "raw"), "--ids", str(td / "ids.npy"),
            "--output-dir", str(td / "out"), "--device", "cpu", "--batch-size", "2",
            "--dtype", "float32", "--allow-nonstandard-layers"]
        if ensemble:
            argv += ["--ensemble", "concat"]
        return export.build_parser().parse_args(argv), ids

    def test_end_to_end_native_and_identity(self):
        with tempfile.TemporaryDirectory() as td:
            args, ids = self.setup_export(td, ensemble=True)
            manifest = export.run(args)
            out = Path(args.output_dir)
            self.assertEqual(manifest["shape"], [3, 10])
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["schema"], export.FORMAT)
            self.assertEqual(manifest["training_provenance_status"], "unknown_requires_split_and_time_audit")
            np.testing.assert_array_equal(np.load(out / "protein_ids.npy"), ids)
            features = np.load(out / "features.npy")
            raw = export.load_plm_batch(args.features_dir, ids, [34, 35, 36], 4)
            expected = []
            for path in args.checkpoint:
                model = export.load_model(self.model_type, path, "BP")
                h, _ = export.capture_penultimate(model, torch.from_numpy(raw))
                expected.append(h.numpy())
            np.testing.assert_allclose(features, np.concatenate(expected, axis=1), atol=1e-6)
            for item in manifest["files"].values():
                self.assertEqual(item["sha256"], export.sha256_file(out / item["path"]))
            args.output_dir = str(Path(td) / "out2")
            np.save(Path(td) / "ids.npy", ids[:1])
            second = export.run(args)
            self.assertEqual(manifest["representation_identity"], second["representation_identity"])
            with self.assertRaises(FileExistsError):
                export.run(args)

    def test_float16_cache(self):
        with tempfile.TemporaryDirectory() as td:
            args, ids = self.setup_export(td)
            args.dtype = "float16"
            export.run(args)
            array = np.load(Path(args.output_dir) / "features.npy")
            self.assertEqual(array.dtype, np.float16)
            model = export.load_model(self.model_type, args.checkpoint[0], "BP")
            raw = export.load_plm_batch(args.features_dir, ids, [34, 35, 36], 4)
            hidden, _ = export.capture_penultimate(model, torch.from_numpy(raw))
            np.testing.assert_allclose(array.astype(np.float32), hidden.numpy(), atol=1e-3, rtol=1e-3)

    def test_failures_never_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as td:
            args, ids = self.setup_export(td)
            (Path(args.features_dir[0]) / (ids[-1] + ".npy")).unlink()
            with self.assertRaises(FileNotFoundError):
                export.run(args)
            self.assertFalse(Path(args.output_dir).exists())
            self.assertFalse(list(Path(td).glob("out.partial-*")))


if __name__ == "__main__":
    unittest.main()
