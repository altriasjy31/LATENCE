from __future__ import annotations

import json

import torch

from nbs_pg.boxsqel_relations import (
    build_boxsqel_gg_relations,
    parse_boxsqel_normalized_axioms,
)


def _checkpoint(path):
    classes = {
        "GO:0000001": 0,
        "GO:0000002": 1,
        "GO:0000003": 2,
        "GO:0000004": 3,
    }
    torch.save(
        {
            "classes": classes,
            "model_state_dict": {
                "model.class_embeds.weight": torch.randn(4, 8),
            },
        },
        path,
    )
    return classes


def test_normalized_relations_use_boxsqel_class_rows(tmp_path):
    checkpoint = tmp_path / "model.pt"
    classes = _checkpoint(checkpoint)
    normalized = tmp_path / "go.norm"
    normalized.write_text(
        "GO_0000001 SubClassOf GO_0000002\n"
        "GO_0000001 and GO_0000002 SubClassOf GO_0000003\n"
        "BFO_0000050 some GO_0000002 SubClassOf GO_0000004\n"
        "GO_0000003 SubClassOf BFO_0000050 some GO_0000004\n",
        encoding="utf-8",
    )
    parsed = parse_boxsqel_normalized_axioms(normalized, classes)
    assert len(parsed.nf1) == 1
    assert len(parsed.nf2) == 1
    assert len(parsed.nf3) == 1
    assert len(parsed.nf4) == 1
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"report": {"nf1": 1, "nf2": 1, "nf3": 1, "nf4": 1}}))
    manifest = build_boxsqel_gg_relations(
        normalized,
        checkpoint,
        tmp_path / "out",
        parser_report_path=report,
        ontology_version="test",
    )
    payload = json.loads(manifest.read_text())
    assert payload["num_classes"] == 4
    assert payload["relations"]["is_a"]["shape"] == [1, 4]
    assert payload["relations"]["part_of"]["shape"] == [1, 4]
    assert payload["future_go_interface"]["task_classifier_indices_are_not_reordered"]


def test_role_axioms_accept_boxsqel_subpropertyof_colon_syntax(tmp_path):
    normalized = tmp_path / "go.norm"
    normalized.write_text(
        "BFO_0000066 o BFO_0000050 SubPropertyOf: BFO_0000066\n"
        "BFO_0000050 o BFO_0000050 SubPropertyOf: BFO_0000050\n"
        "RO_0002212 o RO_0002212 SubPropertyOf: RO_0002213\n"
        "RO_0002211 o RO_0002211 SubPropertyOf: RO_0002211\n"
        "RO_0002092 o RO_0002092 SubPropertyOf: RO_0002092\n"
        "RO_0002091 o RO_0002091 SubPropertyOf: RO_0002091\n"
        "RO_0002211 SubPropertyOf: RO_0002213\n"
        "RO_0002212 SubObjectPropertyOf: RO_0002213\n"
        "BFO_0000050 SubPropertyOf BFO_0000051\n",
        encoding="utf-8",
    )

    parsed = parse_boxsqel_normalized_axioms(normalized, {}, strict=True)

    assert len(parsed.role_chain) == 6
    assert len(parsed.role_inclusion) == 3
    assert parsed.unparsed == []
    assert len(parsed.relations) == 8


def test_role_axiom_counts_match_boxsqel_parser_report(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"classes": {}, "model_state_dict": {}}, checkpoint)
    normalized = tmp_path / "go.norm"
    normalized.write_text(
        "BFO_0000066 o BFO_0000050 SubPropertyOf: BFO_0000066\n"
        "BFO_0000050 o BFO_0000050 SubPropertyOf: BFO_0000050\n"
        "RO_0002212 o RO_0002212 SubPropertyOf: RO_0002213\n"
        "RO_0002211 o RO_0002211 SubPropertyOf: RO_0002211\n"
        "RO_0002092 o RO_0002092 SubPropertyOf: RO_0002092\n"
        "RO_0002091 o RO_0002091 SubPropertyOf: RO_0002091\n"
        "RO_0002211 SubPropertyOf: RO_0002213\n"
        "RO_0002212 SubPropertyOf: RO_0002213\n"
        "BFO_0000050 SubPropertyOf: BFO_0000051\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "report": {
                    "parsed_lines": 9,
                    "nf1": 0,
                    "nf2": 0,
                    "nf3": 0,
                    "nf4": 0,
                    "role_inclusion": 3,
                    "role_chain": 6,
                },
                "classes": 0,
                "relations": 8,
            }
        ),
        encoding="utf-8",
    )

    manifest = build_boxsqel_gg_relations(
        normalized,
        checkpoint,
        tmp_path / "out",
        parser_report_path=report,
        ontology_version="regression-test",
        strict=True,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["relations"]["role_inclusion"]["shape"] == [3, 3]
    assert payload["relations"]["role_chain"]["shape"] == [6, 4]
    assert payload["unparsed_lines"] == 0
