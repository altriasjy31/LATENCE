"""Cache identity, binary membership and global-exclusion invariants; no torch."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shared_cache_io", ROOT / "nbs_models/nbs_protein_go/nbs_pg/shared_head_io_v091.py")
io = importlib.util.module_from_spec(spec)
spec.loader.exec_module(io)


def arrays():
    return {
        "hidden.f32.npy": np.zeros((4, 2), np.float32),
        "protein_ids.i64.npy": np.arange(4, dtype=np.int64),
        "is_weak.bool.npy": np.array([1, 1, 0, 0], bool),
        "label_indptr.i64.npy": np.arange(5, dtype=np.int64),
        "label_go_idx.i32.npy": np.array([0, 1, 2, 0], np.int32),
        "geometry.f32.npy": np.zeros((3, 1024), np.float32),
        "task_to_ontology.i64.npy": np.array([10, 20, 30], np.int64),
        "original_weight.f32.npy": np.zeros((3, 2), np.float32),
        "original_bias.f32.npy": np.zeros(3, np.float32),
    }


class SharedCacheTests(unittest.TestCase):
    def test_source_header_is_epoch60_three_layer_v089_only(self):
        source = {"runner_version": "0.8.9.1", "epoch": 60,
                  "model_config": {"sage_layers": 3, "prediction_mode": "direct"}}
        io.validate_source_header(source)
        for patch in ({"epoch": 70}, {"runner_version": "0.9.0"},
                      {"model_config": {"sage_layers": 6, "prediction_mode": "direct"}},
                      {"model_config": {"sage_layers": 3, "prediction_mode": "shared_bilinear"}}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                io.validate_source_header({**source, **patch})
        with self.assertRaises(ValueError):
            io.validate_source_header(source, expected_epoch=70)

    def test_macro_populations_are_unique_and_exclusions_include_other_microbatches(self):
        groups = list(io.macro_id_groups(np.arange(11), np.arange(100, 105), 4, 2, 8))
        combined = np.concatenate([g[0] for g in groups])
        self.assertEqual(len(combined), len(np.unique(combined)))
        self.assertEqual(set(combined), set(range(11)) | set(range(100, 105)))
        for ids, weak in groups:
            self.assertEqual(ids.shape, weak.shape)
            np.testing.assert_array_equal(weak, ids < 100)
            for start in range(0, len(ids), 2):
                local = ids[start:start + 2]
                self.assertTrue(np.isin(local, ids).all())
                if len(ids) > 2:
                    self.assertGreater(len(ids), len(local))
        again = list(io.macro_id_groups(np.arange(11), np.arange(100, 105), 4, 2, 8))
        for a, b in zip(groups, again):
            np.testing.assert_array_equal(a[0], b[0])

    def test_duplicate_or_overlapping_role_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            list(io.macro_id_groups(np.array([1, 2]), np.array([2, 3]), 1, 1, 0))
        with self.assertRaises(ValueError):
            list(io.macro_id_groups(np.array([1, 1]), np.array([3]), 1, 1, 0))

    def test_binary_label_membership_uses_global_core_and_role_local_weak_rows(self):
        registry = types.SimpleNamespace(role_to_code={"core": 0, "weak": 1},
            role_code=np.array([2, 2, 0, 2, 2, 1]), role_row=np.array([0, 0, 99, 0, 0, 0]))
        gold = types.SimpleNamespace(indptr=np.array([0, 0, 0, 3, 3, 3, 3]), go_idx=np.array([2, 0, 2]))
        calls = []
        def weak_row(row):
            calls.append(row)
            return {"go_idx": np.array([1, 1, 2]), "probability": np.array([.6, .7, .9])}
        pseudo = types.SimpleNamespace(get_role_row=weak_row)
        data = types.SimpleNamespace(registry=registry, stores=types.SimpleNamespace(gold_messages=gold,
                pseudo_messages=pseudo), num_task_go=3)
        result = io.labels_for_ids(data, np.array([5, 2]))
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(result[1], [0, 2])
        self.assertEqual(calls, [0])

    def test_complete_cache_roundtrip_and_content_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            record = {}
            for name, value in arrays().items():
                np.save(folder / name, value)
                record[name] = {"sha256": io.sha256(folder / name), "shape": list(value.shape), "dtype": str(value.dtype)}
            io.write_json(folder / "cache_manifest.json", {"schema": io.SCHEMA, "complete": True, "files": record})
            _, loaded = io.load_cache(folder)
            self.assertTrue(all(isinstance(value, np.memmap) for value in loaded.values()))
            for name, value in arrays().items():
                np.testing.assert_array_equal(loaded[name], value)
            path = folder / "hidden.f32.npy"
            path.write_bytes(path.read_bytes()[:-1] + b"x")
            with self.assertRaisesRegex(ValueError, "changed"):
                io.load_cache(folder)

    def test_incomplete_cache_is_rejected_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            io.write_json(path / "cache_manifest.json", {"schema": io.SCHEMA, "complete": False})
            with self.assertRaisesRegex(ValueError, "complete"):
                io.load_cache(path)

    def test_cache_microbatches_keep_role_ratio_and_bound_size(self):
        ids=np.arange(12800);weak=ids<10240
        parts=io.split_cache_macro(ids,weak,6400)
        self.assertEqual(len(parts),2)
        for part in parts:
            self.assertEqual(len(part),6400)
            self.assertEqual(int((part<10240).sum()),5120)
        self.assertEqual(len(np.unique(np.concatenate(parts))),len(ids))

    def test_top16_candidate_counts_deduplicate_and_bin_confidence(self):
        stats=io.candidate_statistics()
        io.update_candidate_statistics(stats,np.array([[1,1,2,-1],[2,3,-1,-1]]),
            np.array([[.9,.8,.5,0],[1.,.1,0,0]]),[np.array([1,5]),np.array([2,3])])
        final=io.finish_candidate_statistics(stats)
        self.assertEqual(final['candidates'],4)
        self.assertEqual(final['correct'],3)
        self.assertEqual(final['micro_precision'],.75)
        self.assertEqual(final['correct_count_histogram'][1:3],[1,1])

    def test_empty_negative_only_rows_are_retained(self):
        data=arrays()
        data['label_indptr.i64.npy']=np.array([0,1,1,2,4],np.int64)
        io.validate_cache_arrays(data)

    def test_structural_corruption_rejected(self):
        for key, value in (
            ("protein_ids.i64.npy", np.array([0, 1, 1, 3], np.int64)),
            ("label_indptr.i64.npy", np.array([0, 1, 0, 2, 4], np.int64)),
            ("label_go_idx.i32.npy", np.array([0, 1, 2, 3], np.int32)),
            ("hidden.f32.npy", np.full((4, 2), np.nan, np.float32)),
            ("geometry.f32.npy", np.zeros((3, 512), np.float32)),
        ):
            data = arrays()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                io.validate_cache_arrays(data)


if __name__ == "__main__":
    unittest.main()
