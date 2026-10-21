"""Four distinct references, task-column alignment and actual tiny evaluator runs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/nbs/eval_nbs_full_task_v081.py"
spec = importlib.util.spec_from_file_location("v081_eval", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def files(tmp_path):
    n, g = 3, 2903
    truth = np.zeros((n, g), dtype=bool)
    truth[0, [0, 1]], truth[1, [2]], truth[2, [3, 4]] = True, True, True
    metadata = {"ind_test": {"cc": {"proteins": ["a", "b", "c"], "prop_annotations": truth}},
                "train": {"cc": {"proteins": ["a", "b", "c"], "annotations": truth}}}
    with (tmp_path / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    rng = np.random.default_rng(81)
    for name, boost in (("nbs", .8), ("backbone", .5), ("expert", .3), ("modelout", .6)):
        array = rng.uniform(.001, .05, (n, g)).astype(np.float32)
        array[truth] += boost
        np.save(tmp_path / f"{name}.npy", array)
    (tmp_path / "proteins.txt").write_text("a\nb\nc\n")
    (tmp_path / "go.txt").write_text("".join(f"GO:{i:07d}\n" for i in range(g)))
    (tmp_path / "registry.tsv").write_text("go_idx\tinput_go_id\tgo_id\n" + "".join(
        f"{i}\tGO:{i:07d}\tGO:{i:07d}\n" for i in range(g)))
    (tmp_path / "input.json").write_text(json.dumps({"registries": {"go_registry": str(tmp_path / "registry.tsv")}}))
    return tmp_path


def arguments(path):
    return ["--task", "cc", "--metadata-file", str(path / "metadata.pkl"),
            "--nbs-prob", str(path / "nbs.npy"), "--backbone-prob", str(path / "backbone.npy"),
            "--protein-ids", str(path / "proteins.txt"), "--input-manifest", str(path / "input.json"),
            "--output-dir", str(path / "out"), "--metric-backend", "local_micro"]


def test_distinct_expert_and_modelout_arrays_run_real_evaluator(files):
    result = module.main(arguments(files) + ["--expert-prob", str(files / "expert.npy"),
                         "--modelout-prob", str(files / "modelout.npy"), "--references-aligned-to-input"])
    assert set(result["methods"]) == {"NBS_final", "backbone_base", "expert_prob", "stage1_modelout"}
    assert result["reference_comparison_complete"]
    assert not result["evaluation_goal_complete"]  # local_micro is deliberately diagnostic
    assert result["goal_status"] == "diagnostic_local_micro"
    assert result["reference_provenance"]["expert_prob"]["source_sha256"] != result["reference_provenance"]["stage1_modelout"]["source_sha256"]
    assert "auprc_micro_exact" in result["methods"]["NBS_final"]["micro"]
    assert set(result["methods"]["NBS_final"]["top_k"]) == {"10", "50", "100"}
    for reference, label in (("expert_prob", "E"), ("stage1_modelout", "M"), ("backbone_base", "B")):
        expected = result["methods"]["NBS_final"]["micro"]["auprc_micro_exact"] - result["methods"][reference]["micro"]["auprc_micro_exact"]
        assert result["deltas"][f"G_minus_{label}"]["micro"]["auprc_micro_exact"] == pytest.approx(expected)
    report = json.loads((files / "out/nbs_ind_test_metrics.json").read_text())
    assert report["prediction_sources"]["expert_prob"] != report["prediction_sources"]["stage1_modelout"]
    assert report["prediction_sources"]["modelout_reference"] is None
    assert (files / "out/nbs_w2s_comparison.tsv").is_file()


def test_absent_references_rejected_unless_explicit_diagnostics(files):
    with pytest.raises(ValueError, match="Missing reference"):
        module.main(arguments(files))
    result = module.main(arguments(files) + ["--allow-missing-references"])
    assert result["goal_status"] == "incomplete_references"
    assert not result["evaluation_goal_complete"]
    assert result["missing_methods"] == ["expert_prob", "stage1_modelout"]
    assert result["deltas"]["G_minus_E"]["status"] == "missing_reference"


def test_same_path_cannot_masquerade_as_expert_and_modelout(files):
    with pytest.raises(ValueError, match='independently identified'):
        module.main(arguments(files) + ['--expert-prob',str(files/'expert.npy'),
                                       '--modelout-prob',str(files/'expert.npy'),
                                       '--references-aligned-to-input'])


def test_silent_alignment_assumption_rejected(files):
    with pytest.raises(ValueError, match="provide protein IDs"):
        module.main(arguments(files) + ["--expert-prob", str(files / "expert.npy"), "--allow-missing-references"])


def test_mismatched_ids_rejected_even_with_declaration(files):
    (files / "wrong.txt").write_text("a\nb\nWRONG\n")
    with pytest.raises(ValueError, match="protein IDs differ"):
        module.main(arguments(files) + ["--expert-prob", str(files / "expert.npy"),
                    "--expert-protein-ids", str(files / "wrong.txt"),
                    "--references-aligned-to-input", "--allow-missing-references"])


def test_reference_rows_and_columns_reordered(files):
    original = np.load(files / "expert.npy")
    np.save(files / "reversed.npy", original[::-1, ::-1])
    (files / "reverse_proteins.txt").write_text("c\nb\na\n")
    (files / "reverse_go.txt").write_text("".join(f"GO:{i:07d}\n" for i in reversed(range(2903))))
    result = module.main(arguments(files) + ["--expert-prob", str(files / "reversed.npy"),
                         "--expert-protein-ids", str(files / "reverse_proteins.txt"),
                         "--expert-go-ids", str(files / "reverse_go.txt"), "--allow-missing-references"])
    contract = result["reference_provenance"]["expert_prob"]
    assert contract["rows_reordered"] and contract["columns_reordered"]
    np.testing.assert_array_equal(np.load(contract["aligned_probability"]), original)


def test_mismatched_go_ids_rejected(files):
    (files / "bad_go.txt").write_text("GO:WRONG\n" + "".join(f"GO:{i:07d}\n" for i in range(1, 2903)))
    with pytest.raises(ValueError, match="GO IDs differ"):
        module.main(arguments(files) + ["--expert-prob", str(files / "expert.npy"),
                    "--expert-protein-ids", str(files / "proteins.txt"), "--expert-go-ids", str(files / "bad_go.txt"),
                    "--allow-missing-references"])


def test_probability_logits_not_silently_accepted(files):
    array = np.load(files / "expert.npy")
    array[0, 0] = 8
    np.save(files / "expert.npy", array)
    with pytest.raises(ValueError, match="finite probabilities"):
        module.main(arguments(files) + ["--expert-prob", str(files / "expert.npy"),
                    "--references-aligned-to-input", "--allow-missing-references"])


def test_generated_reference_manifest_binds_outputs(files):
    legacy = module._legacy()
    generated = {"outputs": {key: {"sha256": legacy._sha256(files / filename)} for key, filename in (
        ("expert_prob", "expert.npy"), ("stage1_modelout", "modelout.npy"),
        ("protein_ids", "proteins.txt"), ("go_ids", "go.txt"))},
        "modelout_semantics": {"prediction_key": "modelout::mix_expert_base_anchor::decoderprob::expert"},
        "cache_signature": {"sources": {key: {"sha256": legacy._sha256(files / filename)} for key, filename in (
            ("input_manifest", "input.json"), ("cached_backbone", "backbone.npy"))}}}
    manifest = files / "stage1_reference_manifest.json"
    manifest.write_text(json.dumps(generated))
    argv = arguments(files) + ["--stage1-reference-manifest", str(manifest)]
    for source in ("expert", "modelout"):
        argv += ["--" + source + "-prob", str(files / (source + ".npy")),
                 "--" + source + "-protein-ids", str(files / "proteins.txt"),
                 "--" + source + "-go-ids", str(files / "go.txt")]
    result = module.main(argv)
    assert result["stage1_reference_generation"]["manifest"] == generated
    np.save(files / "modelout.npy", np.zeros((3, 2903), dtype=np.float32))
    with pytest.raises(ValueError, match="manifest does not match modelout_prob"):
        module.main(argv)
