import csv
import json
from pathlib import Path

import numpy as np
import torch

from nbs_pg.boxsqel_manifest import (
    align_boxsqel_to_go_registry,
    load_boxsqel_training_contract,
    normalize_go_identifier,
)
from nbs_pg.latence_stores import GOBoxStore


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_normalize_go_identifier_accepts_compact_and_iri():
    assert normalize_go_identifier("GO:0008150") == "GO:0008150"
    assert normalize_go_identifier("GO_0008150") == "GO:0008150"
    assert (
        normalize_go_identifier("<http://purl.obolibrary.org/obo/GO_0008150>")
        == "GO:0008150"
    )
    assert normalize_go_identifier("owl:Thing") is None


def test_boxsqel_manifest_alignment_projects_to_classifier_space(tmp_path: Path):
    checkpoint = tmp_path / "box_best.pt"
    class_map = {
        "<http://purl.obolibrary.org/obo/GO_0008150>": 0,
        "GO_0000001": 1,
        "owl:Thing": 2,
    }
    raw = torch.tensor(
        [
            [1.0, 2.0, -0.5, 0.25],
            [3.0, 4.0, 0.75, -1.25],
            [9.0, 9.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    torch.save(
        {
            "classes": class_map,
            "model_state_dict": {"class_embeds.weight": raw},
        },
        checkpoint,
    )
    parser_report = tmp_path / "parser.json"
    parser_payload = {
        "input_file": "data/go.norm",
        "report": {
            "parsed_lines": 4,
            "expanded_axioms": 4,
            "unparsed_lines": 0,
        },
        "classes": 3,
        "relations": 1,
    }
    _write_json(parser_report, parser_payload)
    manifest = tmp_path / "manifest.json"
    manifest_payload = {
        "model": "BoxSquaredEL",
        "architecture": {
            "class_vector_width": 4,
            "relation_vector_width": 8,
        },
        "config": {"embedding_size": 2, "strict_parser": True},
        "parser": parser_payload,
        "training": {
            "best_epoch": 7,
            "last_epoch": 10,
            "nan_detected": False,
        },
        "artifacts": {
            "best_checkpoint": str(checkpoint),
            "final_checkpoint": str(checkpoint),
            "npy_directory": str(tmp_path / "npy"),
        },
    }
    _write_json(manifest, manifest_payload)
    registry = tmp_path / "go_registry.tsv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["go_idx", "input_go_id", "go_id"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow({"go_idx": 0, "input_go_id": "GO:0008150", "go_id": "GO:0008150"})
        writer.writerow({"go_idx": 1, "input_go_id": "GO:0000001", "go_id": "GO:0000001"})
        # An alt-ID classifier column canonicalized to GO:0008150 must reuse the
        # same source geometry without compressing the classifier index space.
        writer.writerow({"go_idx": 2, "input_go_id": "GO:1234567", "go_id": "GO:0008150"})

    contract = load_boxsqel_training_contract(
        manifest,
        parser_report_path=parser_report,
        artifact_selection="best",
    )
    assert contract.embedding_dim == 2
    assert contract.num_ontology_classes == 3
    assert contract.selected_epoch == 7

    output = align_boxsqel_to_go_registry(
        manifest,
        registry,
        tmp_path / "aligned",
        parser_report_path=parser_report,
        artifact_selection="best",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["classifier_go_terms"] == 3
    assert payload["matched_go_terms"] == 3
    center = np.load(output.parent / "go_box_center.f32.npy")
    offset = np.load(output.parent / "go_box_offset.f32.npy")
    stats = np.load(output.parent / "go_box_stats.f32.npy")
    source = np.load(output.parent / "go_box_source_row.i32.npy")
    assert center.shape == (3, 2)
    assert offset.shape == (3, 2)
    assert stats.shape == (3, 6)
    assert source.tolist() == [0, 1, 0]
    np.testing.assert_allclose(center[0], center[2])
    np.testing.assert_allclose(offset[0], [0.5, 0.25])
    assert np.all(offset > 0)

    store = GOBoxStore.from_alignment_manifest(output)
    gathered = store.gather(np.asarray([2, 1]))
    assert gathered["center"].shape == (2, 2)
    np.testing.assert_allclose(gathered["center"][0], center[0])
