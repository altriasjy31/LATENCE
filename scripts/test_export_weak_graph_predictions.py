#!/usr/bin/env python3
"""Regression tests for export_weak_graph_predictions_v2.py.

The full forward test requires the LATENCE PyTorch environment and real model
modules.  These tests intentionally cover the parts that can be validated
without a checkpoint: registry alignment, rare definitions and streamed .npy
publication.
"""

from __future__ import annotations

import csv
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from export_weak_graph_predictions_v2 import (
        GORegistry,
        RawArrayWriter,
        build_parser,
        load_go_registry,
        load_protein_registry,
        make_rare_mask,
        rare_first_selector_topk,
        restricted_selector_topk,
        shrink_dense_rows,
    )
except ImportError:
    # Formal deployment removes the internal discussion suffix.
    from export_weak_graph_predictions import (
        GORegistry,
        RawArrayWriter,
        build_parser,
        load_go_registry,
        load_protein_registry,
        make_rare_mask,
        rare_first_selector_topk,
        restricted_selector_topk,
        shrink_dense_rows,
    )


class ExportWeakGraphPredictionsTests(unittest.TestCase):
    def test_rare_first_selector_does_not_call_private_static_score_api(self) -> None:
        source = inspect.getsource(rare_first_selector_topk)
        self.assertNotIn("._make_static_score(", source)
        self.assertIn("sigmoid(base_for_selector.float())", source)

    def test_restricted_is_rare_first_compatibility_alias(self) -> None:
        self.assertIs(restricted_selector_topk, rare_first_selector_topk)

    def test_parser_defaults_to_rare_first(self) -> None:
        parser = build_parser()
        action = next(
            item for item in parser._actions if item.dest == "rare_selector_scope"
        )
        self.assertEqual(action.default, "rare_first")
        self.assertIn("model_topk_filter", action.choices)

    def test_registries_and_train_q33(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protein_path = root / "protein_registry.csv"
            with protein_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "protein_idx",
                        "protein_id",
                        "role",
                        "role_row_idx",
                        "feature_file",
                        "dataset_mode",
                    ]
                )
                writer.writerow([0, "P0", "core", 0, "core.npy", "train"])
                writer.writerow([1, "P1", "weak", 0, "weak.npy", "exp_train"])
            protein = load_protein_registry(protein_path)
            self.assertEqual(protein.role_ids["core"], ("P0",))
            self.assertEqual(protein.role_global_indices["weak"].tolist(), [1])

            go_path = root / "go_registry.tsv"
            go_path.write_text(
                "go_idx\tinput_go_id\tgo_id\tname\n"
                "0\tGO:0\tGO:0\tzero\n"
                "1\tGO:1\tGO:1\tone\n"
                "2\tGO:2\tGO:2\ttwo\n"
                "3\tGO:3\tGO:3\tthree\n",
                encoding="utf-8",
            )
            go = load_go_registry(go_path)
            counts = np.asarray([0, 1, 2, 100], dtype=np.float64)
            mask, metadata = make_rare_mask(
                counts,
                policy="train_q33",
                max_count=5,
                include_zero=False,
                registry=go,
                ids_path=None,
            )
            self.assertEqual(mask.tolist(), [False, True, False, False])
            self.assertEqual(metadata["num_rare_terms"], 1)

    def test_ids_file_matches_input_and_canonical_ids(self) -> None:
        registry = GORegistry(
            go_idx=np.arange(3, dtype=np.int32),
            input_go_ids=("GO:OLD", "GO:2", "GO:3"),
            canonical_go_ids=("GO:1", "GO:2", "GO:3"),
            names=("", "", ""),
        )
        with tempfile.TemporaryDirectory() as directory:
            ids_path = Path(directory) / "rare.txt"
            ids_path.write_text("GO:1\nGO:3\n", encoding="utf-8")
            mask, _ = make_rare_mask(
                np.asarray([10, 20, 30], dtype=np.float64),
                policy="ids_file",
                max_count=5,
                include_zero=False,
                registry=registry,
                ids_path=ids_path,
            )
            self.assertEqual(mask.tolist(), [True, False, True])

    def test_stream_writer_edge_transpose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RawArrayWriter(root / "edge.raw", np.int32, columns=2)
            writer.append(np.asarray([[1, 10], [2, 20]], dtype=np.int32))
            writer.append(np.asarray([[3, 30]], dtype=np.int32))
            output = root / "edge.i32.npy"
            writer.finalize(output, transpose_two_columns=True)
            actual = np.load(output)
            expected = np.asarray([[1, 2, 3], [10, 20, 30]], dtype=np.int32)
            np.testing.assert_array_equal(actual, expected)

    def test_shrink_dense_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prob.f16.npy"
            array = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(5, 3)
            )
            array[:] = np.arange(15, dtype=np.float16).reshape(5, 3)
            array.flush()
            del array
            shrink_dense_rows(path, 3)
            actual = np.load(path)
            self.assertEqual(actual.shape, (3, 3))
            np.testing.assert_array_equal(
                actual, np.arange(9, dtype=np.float16).reshape(3, 3)
            )


if __name__ == "__main__":
    unittest.main()
