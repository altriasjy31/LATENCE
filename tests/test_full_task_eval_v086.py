"""Verify the exported checkpoint/branch actually generated the evaluated array."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
previous = load("v081_eval_test_fixture", Path(__file__).with_name("test_full_task_eval_v081.py"))
module = load("v086_eval", ROOT / "scripts/nbs/eval_nbs_full_task_v086.py")
files = previous.files


def encoder_identity():
    return {"encoder_variant": "preln", "model_architecture_version": "0.8.6",
            "data_architecture_version": "0.8.4",
            "model_config": {"encoder_variant": "preln", "pp_edge_dropout": 0.0},
            "sampler_config": {"mode": "dynamic", "first_hop": 32}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arguments(path, branch="final"):
    checkpoint = path / "nbs_step600.pt"
    checkpoint.write_bytes(b"test checkpoint identity")
    manifest = {
        "version": "0.8.6", "runner_version": "0.8.6", "variant": "dynamic",
        "branch": branch, "step": 600, "ablation": "full",
        "num_task_go": 2903, "prediction_space": "complete_task_classifier_columns",
        "uses_expert_probability_in_nbs_forward": False,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "output_probability_sha256": sha(path / "nbs.npy"),
        "input_manifest_sha256": sha(path / "input.json"),
        "prediction_implementation": {"model.py": "a" * 64},
        **encoder_identity(),
    }
    (path / "prediction.json").write_text(json.dumps(manifest))
    return previous.arguments(path) + [
        "--prediction-manifest", str(path / "prediction.json"),
        "--expert-prob", str(path / "expert.npy"),
        "--modelout-prob", str(path / "modelout.npy"), "--references-aligned-to-input"]


def test_evaluation_binds_probabilities_manifest_checkpoint_and_branch(files):
    result = module.main(arguments(files))
    provenance = result["prediction_provenance"]
    assert result["evaluation_version"] == "0.8.6"
    assert provenance["probability_sha256"] == sha(files / "nbs.npy")
    assert provenance["prediction_manifest_sha256"] == sha(files / "prediction.json")
    assert provenance["input_manifest_sha256"] == sha(files / "input.json")
    assert provenance["checkpoint_sha256"] == sha(files / "nbs_step600.pt")
    assert provenance["checkpoint_file_verified"]
    assert (provenance["step"], provenance["variant"], provenance["branch"]) == (600, "dynamic", "final")
    assert {key: provenance[key] for key in encoder_identity()} == encoder_identity()
    assert set(result["deltas"]) == {"G_minus_E", "G_minus_M", "G_minus_B"}
    saved = json.loads((files / "out/nbs_ind_test_metrics.json").read_text())
    assert saved["w2s_comparison"]["prediction_provenance"] == provenance


@pytest.mark.parametrize("source,message", [
    ("nbs.npy", "prediction probability"),
    ("input.json", "prepared input"),
    ("nbs_step600.pt", "checkpoint file"),
])
def test_mismatched_source_rejected_before_metric_output(files, source, message):
    argv = arguments(files)
    path = files / source
    if path.suffix == ".npy":
        array = np.load(path)
        array[0, 0] = .25
        np.save(path, array)
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=message):
        module.main(argv)
    assert not (files / "out/nbs_ind_test_metrics.json").exists()


def test_classification_branch_rejected(files):
    with pytest.raises(ValueError, match="single final matching"):
        module.main(arguments(files, branch="classification"))


def test_stale_v081_manifest_rejected(files):
    argv = arguments(files)
    path = files / "prediction.json"
    data = json.loads(path.read_text())
    data["runner_version"] = "0.8.1"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not a v0.8.6 output"):
        module.main(argv)


def intervention_results(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"same checkpoint for all interventions")
    method = {"symbol": "G", "primary": {"Fmax": 50.0, "AUPRC": 40.0},
              "micro": {"auprc_micro_exact": 45.0},
              "top_k": {"10": {"precision": {"mean": 0.6}}}}
    reference = {"source_sha256": "e" * 64, "aligned_sha256": "e" * 64}
    for ablation in module.ABLATIONS:
        item = {"evaluation_version": "0.8.6", "task": "bp", "num_samples": 1800,
                "num_classes": 21312, "metric_contract": {"primary_backend": "stage1"},
                "reference_comparison_complete": True,
                "prediction_provenance": {"checkpoint_sha256": sha(checkpoint),
                    "step": 600, "variant": "dynamic", "branch": "final", "ablation": ablation,
                    "input_manifest_sha256": "a" * 64, "backbone_probability_sha256": "b" * 64,
                    "prediction_implementation": {"model": "m" * 64}},
                "methods": {"NBS_final": json.loads(json.dumps(method)),
                            "expert_prob": method, "stage1_modelout": method, "backbone_base": method},
                "reference_provenance": {"expert_prob": reference, "stage1_modelout": reference},
                "deltas": {"G_minus_E": {"status": "ok"}}}
        item["prediction_provenance"].update(encoder_identity())
        if ablation != "full":
            item["methods"]["NBS_final"]["primary"]["Fmax"] -= 2
        directory = tmp_path / "eval_step600" / ablation / "metrics"
        directory.mkdir(parents=True)
        (directory / "nbs_w2s_comparison.json").write_text(json.dumps(item))
    return checkpoint


def test_interventions_same_sources_have_descriptive_deltas(tmp_path):
    checkpoint = intervention_results(tmp_path)
    result = module.main(["--summarize-work-dir", str(tmp_path), "--summary-checkpoint", str(checkpoint)])
    assert result["full_minus_intervention"]["full_minus_graph_off"]["primary"]["Fmax"] == 2
    assert len(result["full_minus_intervention"]) == 5
    assert "Inference dependence only" in result["interpretation"]
    assert (tmp_path / "eval_step600/nbs_graph_effects.json").is_file()


@pytest.mark.parametrize("mismatch", ["checkpoint", "input", "reference", "metric", "implementation", "evaluator",
                                     "encoder", "model_config", "sampler_config", "contract", "sampling_inference"])
def test_intervention_summary_rejects_mixed_sources(tmp_path, mismatch):
    checkpoint = intervention_results(tmp_path)
    path = tmp_path / "eval_step600/core_off/metrics/nbs_w2s_comparison.json"
    item = json.loads(path.read_text())
    if mismatch == "checkpoint":
        item["prediction_provenance"]["checkpoint_sha256"] = "c" * 64
    elif mismatch == "input":
        item["prediction_provenance"]["input_manifest_sha256"] = "c" * 64
    elif mismatch == "reference":
        item["reference_provenance"]["expert_prob"]["source_sha256"] = "c" * 64
    elif mismatch == "implementation":
        item["prediction_provenance"]["prediction_implementation"]["model"] = "c" * 64
    elif mismatch == "evaluator":
        item["prediction_provenance"]["evaluator_sha256"] = "c" * 64
    elif mismatch == "encoder":
        item["prediction_provenance"]["encoder_variant"] = "legacy"
        item["prediction_provenance"]["model_config"]["encoder_variant"] = "legacy"
    elif mismatch == "model_config":
        item["prediction_provenance"]["model_config"]["pp_edge_dropout"] = 0.1
    elif mismatch == "sampler_config":
        item["prediction_provenance"]["sampler_config"]["first_hop"] = 64
    elif mismatch == "contract":
        item["prediction_provenance"]["contract"] = {"data": "another"}
    elif mismatch == "sampling_inference":
        item["prediction_provenance"]["sampling_inference"] = "stochastic"
    else:
        item["metric_contract"]["primary_backend"] = "local_micro"
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="differs"):
        module.summarize_ablations(tmp_path, checkpoint)
    assert not (tmp_path / "eval_step600/nbs_graph_effects.json").exists()


@pytest.mark.parametrize("variant", ["graph", "local", "no_graph", "dual"])
def test_prior_variant_names_rejected(files, variant):
    argv = arguments(files)
    path = files / "prediction.json"
    manifest = json.loads(path.read_text())
    manifest["variant"] = variant
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unknown v0.8.6 variant"):
        module.main(argv)


def test_fixed_variant_preserves_separate_four_baselines(files):
    argv = arguments(files)
    path = files / "prediction.json"
    manifest = json.loads(path.read_text())
    manifest["variant"] = "fixed"
    manifest["sampler_config"]["mode"] = "fixed"
    path.write_text(json.dumps(manifest))
    result = module.main(argv)
    assert result["prediction_provenance"]["variant"] == "fixed"
    assert {entry["symbol"] for entry in result["methods"].values()} == {"E", "M", "B", "G"}
    assert result["methods"]["expert_prob"] != result["methods"]["stage1_modelout"]


def audit_files(files):
    # Add source-bound sidecars as real prepared inputs have them.
    ip = files / "input.json"
    value = json.loads(ip.read_text())
    value.update(protein_ids=str(files / "proteins.txt"), protein_ids_file_sha256=sha(files / "proteins.txt"))
    value["registries"]["go_registry_sha256"] = sha(files / "registry.tsv")
    ip.write_text(json.dumps(value))
    argv = arguments(files) + ["--expert-protein-ids", str(files / "proteins.txt"),
        "--modelout-protein-ids", str(files / "proteins.txt"),
        "--expert-go-ids", str(files / "go.txt"), "--modelout-go-ids", str(files / "go.txt")]
    current = module.main(argv)
    # Model/checkpoint is deliberately no longer available for metric recomputation.
    (files / "nbs_step600.pt").unlink()
    return files / "out/nbs_w2s_comparison.json", current


def test_standardized_metrics_separate_and_cached_recompute_needs_no_checkpoint(files):
    comparison, current = audit_files(files)
    before = comparison.read_bytes()
    result = module.main(["--recompute-comparison", str(comparison), "--metadata-file", str(files / "metadata.pkl"),
                          "--output-dir", str(files / "audit")])
    assert result["checkpoint_loaded"] is False
    assert result["model_inference_run"] is False
    assert comparison.read_bytes() == before
    for name in current["methods"]:
        assert result["methods"][name]["primary"] == current["methods"][name]["primary"]
        assert result["methods"][name]["standardized"] == current["methods"][name]["standardized"]
        assert "standard_protein_fmax" in result["methods"][name]["standardized"]
        assert "standard_micro_pr_auc" in result["methods"][name]["standardized"]
    assert "standardized" in result["deltas"]["G_minus_E"]


@pytest.mark.parametrize("source", ["nbs.npy", "backbone.npy", "expert.npy", "modelout.npy", "proteins.txt", "registry.tsv", "metadata.pkl"])
def test_cached_audit_rejects_changed_sources(files, source):
    comparison, _ = audit_files(files)
    path = files / source
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        module.main(["--recompute-comparison", str(comparison), "--metadata-file", str(files / "metadata.pkl"),
                     "--output-dir", str(files / "audit")])
    assert not (files / "audit/metric_audit_v086.json").exists()


@pytest.mark.parametrize("old_version", ["0.8.4", "0.8.5"])
def test_prior_probabilities_remain_accepted_only_in_audit_mode(files, old_version):
    comparison, _ = audit_files(files)
    pred_path = files / "prediction.json"
    prediction = json.loads(pred_path.read_text())
    prediction["version"] = prediction["runner_version"] = old_version
    pred_path.write_text(json.dumps(prediction))
    value = json.loads(comparison.read_text())
    value["evaluation_version"] = old_version
    value["prediction_provenance"]["prediction_manifest_sha256"] = sha(pred_path)
    comparison.write_text(json.dumps(value))
    result = module.main(["--recompute-comparison", str(comparison), "--metadata-file", str(files / "metadata.pkl"),
                          "--output-dir", str(files / "audit")])
    assert result["evaluation_version"] == old_version
    assert result["audit_version"] == "0.8.6"


def test_cached_audit_rejects_unbound_gold_metadata(files):
    comparison, _ = audit_files(files)
    item = json.loads(comparison.read_text())
    item.pop("metadata_provenance")
    comparison.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="recorded metadata SHA256"):
        module.main(["--recompute-comparison", str(comparison), "--metadata-file", str(files / "metadata.pkl"),
                     "--output-dir", str(files / "audit")])


def test_standardized_delta_excludes_threshold_score_and_counts(files):
    _, result = audit_files(files)
    keys = result["deltas"]["G_minus_B"]["standardized"]
    assert "standard_micro_fmax_exact_min_included_score" not in keys
    assert not any("threshold" in key or "count" in key for key in keys)


@pytest.mark.parametrize('key', list(encoder_identity()))
def test_new_exports_require_explicit_encoder_sampler_identity(files, key):
    argv = arguments(files)
    manifest_path = files / 'prediction.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.pop(key)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='v0.8.6 prediction'):
        module.main(argv)
    assert not (files / 'out/nbs_ind_test_metrics.json').exists()


@pytest.mark.parametrize('key', ['encoder_variant', 'model_config', 'sampler_config'])
def test_cached_v086_audit_rejects_comparison_encoder_sampler_mismatch(files, key):
    comparison, _ = audit_files(files)
    value = json.loads(comparison.read_text())
    value['prediction_provenance'].pop(key)
    comparison.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='manifest/comparison identity differs'):
        module.main(['--recompute-comparison', str(comparison), '--metadata-file', str(files / 'metadata.pkl'),
                     '--output-dir', str(files / 'audit')])
    assert not (files / 'audit/metric_audit_v086.json').exists()
