#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("build_pp_edge_types.py")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="test_pp_edge_types_") as raw:
        root = Path(raw)
        features = root / "features"
        relations = root / "pp_relations"
        output = root / "compiled"
        features.mkdir()
        relations.mkdir()

        registry_rows = [
            (0, "A", "core", 0, "core.npy", "train"),
            (1, "B", "core", 1, "core.npy", "train"),
            (2, "C", "core", 2, "core.npy", "train"),
            (3, "W1", "weak", 0, "weak.npy", "exp_train"),
            (4, "W2", "weak", 1, "weak.npy", "exp_train"),
        ]
        with (features / "protein_registry.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
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
            writer.writerows(registry_rows)
        write_json(
            features / "representation_manifest.json",
            {"schema_version": 1, "task": "bp"},
        )

        core_neighbors = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        core_scores = np.asarray(
            [[0.90, 0.80], [0.85, 0.70], [0.75, 0.65]], dtype=np.float16
        )
        weak_neighbors = np.asarray([[1, 0], [2, 0]], dtype=np.int32)
        weak_scores = np.asarray([[0.80, 0.60], [0.90, 0.20]], dtype=np.float16)
        np.save(relations / "pp_core_core_neighbors.i32.npy", core_neighbors)
        np.save(relations / "pp_core_core_scores.f16.npy", core_scores)
        np.save(relations / "pp_weak_core_neighbors.i32.npy", weak_neighbors)
        np.save(relations / "pp_weak_core_scores.f16.npy", weak_scores)

        core_manifest = {
            "query_role": "core",
            "query_count": 3,
            "core_count": 3,
            "resolved_k": 2,
            "neighbors_file": "pp_core_core_neighbors.i32.npy",
            "scores_file": "pp_core_core_scores.f16.npy",
        }
        weak_manifest = {
            "query_role": "weak",
            "query_count": 2,
            "core_count": 3,
            "resolved_k": 2,
            "neighbors_file": "pp_weak_core_neighbors.i32.npy",
            "scores_file": "pp_weak_core_scores.f16.npy",
        }
        write_json(relations / "pp_core_core_manifest.json", core_manifest)
        write_json(relations / "pp_weak_core_manifest.json", weak_manifest)
        write_json(
            relations / "pp_relations_manifest.json",
            {
                "schema_version": 1,
                "task": "bp",
                "relations": [core_manifest, weak_manifest],
            },
        )

        (root / "ppi.tsv").write_text(
            "\n".join(
                [
                    "source\ttarget\tcombined_score",
                    "A\tW1\t800",
                    "W1\tA\t900",
                    "B\tC\t600",
                    "X\tA\t900",
                    "C\tW2\t700",
                    "C\tC\t999",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        command = [
            sys.executable,
            str(SCRIPT),
            "--feature-dir",
            str(features),
            "--pp-relations-dir",
            str(relations),
            "--ppi-tsv",
            str(root / "ppi.tsv"),
            "--output-dir",
            str(output),
            "--similar-k",
            "1",
            "--similar-mode",
            "mutual",
            "--similar-score-reduce",
            "min",
            "--weak-k",
            "2",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        ppi = np.load(output / "pp_ppi_edge_index.i32.npy")
        ppi_attr = np.load(output / "pp_ppi_edge_attr.f32.npy")
        assert ppi.shape == (2, 4), ppi
        assert set(map(tuple, ppi.T.tolist())) == {
            (0, 3),
            (3, 0),
            (2, 4),
            (4, 2),
        }
        a_w1 = np.flatnonzero((ppi[0] == 0) & (ppi[1] == 3))[0]
        assert np.isclose(ppi_attr[a_w1, 0], 0.9)

        similar = np.load(output / "pp_similar_to_edge_index.i32.npy")
        similar_attr = np.load(output / "pp_similar_to_edge_attr.f32.npy")
        assert set(map(tuple, similar.T.tolist())) == {(0, 1), (1, 0)}
        assert np.allclose(similar_attr[:, 0], 0.85, atol=1e-3)

        weak = np.load(output / "pp_weak_to_core_edge_index.i32.npy")
        assert set(map(tuple, weak.T.tolist())) == {
            (3, 1),
            (3, 0),
            (4, 2),
            (4, 0),
        }
        assert set(weak[0].tolist()) == {3, 4}
        assert set(weak[1].tolist()) <= {0, 1, 2}

        manifest = json.loads(
            (output / "pp_edge_types_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["num_proteins"] == 5
        assert manifest["schema_version"] == 2
        assert (
            manifest["builder"] == {
            "id": "build_pp_edge_types",
            "version": "2.1.0-top100-weak-to-core",
            "file": "build_pp_edge_types.py",}
        )
        assert (
            manifest["source_details"]["weak_to_core"]["graph_message_direction"]
            == "weak-to-core"
        )
        assert (
            manifest["routing_contract"]["weak_to_core"]["minimum_message_passing_layers"]
            == 2
        )
        assert (
            [item["relation"] for item in manifest["relations"]] == [
            "ppi",
            "similar_to",
            "weak_to_core",]
        )

        for mode, expected in (("directed", 3), ("union", 4), ("mutual", 2)):
            mode_output = root / f"compiled_{mode}"
            mode_command = command.copy()
            mode_command[mode_command.index(str(output))] = str(mode_output)
            mode_command[mode_command.index("mutual")] = mode
            mode_command[mode_command.index("ppi,similar_to,weak_to_core")
                         if "ppi,similar_to,weak_to_core" in mode_command
                         else len(mode_command):] = []
            mode_command.extend(["--relations", "similar_to"])
            subprocess.run(
                mode_command, check=True, capture_output=True, text=True
            )
            mode_edges = np.load(
                mode_output / "pp_similar_to_edge_index.i32.npy"
            )
            assert mode_edges.shape == (2, expected), (mode, mode_edges)
            if mode == "directed":
                assert set(map(tuple, mode_edges.T.tolist())) == {
                    (1, 0),
                    (0, 1),
                    (0, 2),
                }

        for direction, expected in (
            (
                "weak-to-core",
                {(3, 1), (3, 0), (4, 2), (4, 0)},
            ),
            (
                "core-to-weak",
                {(1, 3), (0, 3), (2, 4), (0, 4)},
            ),
            (
                "bidirectional",
                {
                    (3, 1),
                    (3, 0),
                    (4, 2),
                    (4, 0),
                    (1, 3),
                    (0, 3),
                    (2, 4),
                    (0, 4),
                },
            ),
        ):
            direction_output = root / f"compiled_weak_{direction}"
            direction_command = command.copy()
            direction_command[direction_command.index(str(output))] = str(
                direction_output
            )
            direction_command.extend(
                [
                    "--relations",
                    "weak_to_core",
                    "--weak-message-direction",
                    direction,
                    "--knn-write-chunk-rows",
                    "1",
                ]
            )
            subprocess.run(
                direction_command, check=True, capture_output=True, text=True
            )
            direction_edges = np.load(
                direction_output / "pp_weak_to_core_edge_index.i32.npy"
            )
            assert set(map(tuple, direction_edges.T.tolist())) == expected

        launcher_output = root / "launcher_output"
        launcher_env = os.environ.copy()
        launcher_env.update(
            {
                "PROJECT_ROOT": str(root),
                "BUILD_SCRIPT": str(SCRIPT),
                "TASK": "bp",
                "RUN_TAG": "synthetic",
                "EPOCH": "1",
                "FEATURE_DIR": str(features),
                "PP_RELATIONS_DIR": str(relations),
                "PPI_PATH": str(root / "ppi.tsv"),
                "OUTPUT_DIR": str(launcher_output),
                "RELATIONS": "similar_to,weak_to_core",
                "KNN_WRITE_CHUNK_ROWS": "1",
            }
        )
        launcher = Path(__file__).with_name("run_build_pp_edge_types.py")
        launcher_result = subprocess.run(
            [sys.executable, str(launcher)],
            check=True,
            capture_output=True,
            text=True,
            env=launcher_env,
        )
        assert "--similar-mode directed" in launcher_result.stdout
        assert "LAUNCHER=run_build_pp_edge_types.py" in launcher_result.stdout
        assert "BUILDER=build_pp_edge_types.py" in launcher_result.stdout
        assert "[Implementation] build_pp_edge_types" in launcher_result.stdout
        assert "--similar-k 100" in launcher_result.stdout
        assert "--similar-message-direction neighbor-to-query" in launcher_result.stdout
        assert "--weak-k 100" in launcher_result.stdout
        assert "--weak-message-direction weak-to-core" in launcher_result.stdout
        assert "--knn-write-chunk-rows 1" in launcher_result.stdout
        assert "--relations similar_to,weak_to_core" in launcher_result.stdout
        assert (
            launcher_output / "pp_weak_to_core_edge_index.i32.npy"
        ).is_file()
        launcher_similar = np.load(
            launcher_output / "pp_similar_to_edge_index.i32.npy"
        )
        assert set(map(tuple, launcher_similar.T.tolist())) == {
            (1, 0),
            (2, 0),
            (0, 1),
            (2, 1),
            (0, 2),
            (1, 2),
        }
        launcher_weak = np.load(
            launcher_output / "pp_weak_to_core_edge_index.i32.npy"
        )
        assert set(map(tuple, launcher_weak.T.tolist())) == {
            (3, 1),
            (3, 0),
            (4, 2),
            (4, 0),
        }

        # KMAX=200 input must compile to the default directed top-100 graph,
        # giving every core query exactly 100 incoming neighbor messages.
        top100_root = root / "top100"
        top100_features = top100_root / "features"
        top100_relations = top100_root / "pp_relations"
        top100_output = top100_root / "compiled"
        top100_features.mkdir(parents=True)
        top100_relations.mkdir()
        num_core = 205
        with (top100_features / "protein_registry.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "protein_idx",
                    "protein_id",
                    "role",
                    "role_row_idx",
                ]
            )
            for idx in range(num_core):
                writer.writerow([idx, f"P{idx}", "core", idx])
        write_json(
            top100_features / "representation_manifest.json",
            {"schema_version": 1, "task": "bp"},
        )
        neighbors = np.empty((num_core, 200), dtype=np.int32)
        for query in range(num_core):
            neighbors[query] = [
                (query + offset) % num_core for offset in range(1, 201)
            ]
        scores = np.broadcast_to(
            np.linspace(1.0, 0.5, 200, dtype=np.float32),
            neighbors.shape,
        ).astype(np.float16)
        np.save(
            top100_relations / "pp_core_core_neighbors.i32.npy", neighbors
        )
        np.save(top100_relations / "pp_core_core_scores.f16.npy", scores)
        top100_manifest = {
            "query_role": "core",
            "query_count": num_core,
            "core_count": num_core,
            "resolved_k": 200,
            "neighbors_file": "pp_core_core_neighbors.i32.npy",
            "scores_file": "pp_core_core_scores.f16.npy",
        }
        write_json(
            top100_relations / "pp_core_core_manifest.json",
            top100_manifest,
        )
        write_json(
            top100_relations / "pp_relations_manifest.json",
            {
                "schema_version": 1,
                "task": "bp",
                "relations": [top100_manifest],
            },
        )
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--feature-dir",
                str(top100_features),
                "--pp-relations-dir",
                str(top100_relations),
                "--ppi-tsv",
                str(root / "unused.tsv"),
                "--output-dir",
                str(top100_output),
                "--relations",
                "similar_to",
                "--knn-write-chunk-rows",
                "17",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        top100_edges = np.load(
            top100_output / "pp_similar_to_edge_index.i32.npy"
        )
        assert top100_edges.shape == (2, num_core * 100)
        assert np.all(top100_edges[0] != top100_edges[1])
        assert np.array_equal(
            np.bincount(top100_edges[1], minlength=num_core),
            np.full(num_core, 100),
        )
        top100_compiled_manifest = json.loads(
            (top100_output / "pp_edge_types_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        top100_details = top100_compiled_manifest["source_details"][
            "similar_to"
        ]
        assert top100_details["mode"] == "directed"
        assert top100_details["requested_k"] == 100
        assert top100_details["resolved_k"] == 100
        assert top100_details["graph_message_direction"] == "neighbor-to-query"
        print("build_pp_edge_types synthetic regression: PASS")


if __name__ == "__main__":
    main()