"""Real train -> inductive graph -> probability -> four-baseline CLI integration."""
from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nbs_pg.latence_stores import RoleAwareBaseLogitStore, RoleProbabilitySlice
from scripts.nbs import train_nbs_full_task_v085 as runner
from scripts.nbs import eval_nbs_full_task_v085 as evaluator
from test_full_task_data_v080 import inference_fixture
from test_full_task_runner_v085 import _data, _config, _train


def test_real_inductive_export_metrics_and_verified_prediction_reuse(tmp_path, monkeypatch):
    data = _data(tmp_path / "data", "dynamic")
    width = 2903  # Production CC evaluation width, not a fake evaluator.
    data.num_task_go = data.stores.num_task_go = data.stores.gold_messages.num_go = width
    data.stores.task_to_ontology = np.arange(width, dtype=np.int64)
    data.stores.task_to_ontology[-1] = 0
    rng = np.random.default_rng(51)
    boxes = {"center": rng.normal(size=(width, 4)).astype(np.float32),
             "offset": rng.uniform(.1, 1, size=(width, 4)).astype(np.float32),
             "stats": np.zeros((width, 2), np.float32)}
    data.stores.full_boxes = SimpleNamespace(
        num_go=width, gather=lambda rows: {key: value[rows] for key, value in boxes.items()})
    for role, count in (("core", 4), ("weak", 2)):
        np.save(tmp_path / "data" / f"{role}_full_p.npy", np.full((count, width), .2, np.float32))
    data.stores.episode_sampler.base_logit_store = RoleAwareBaseLogitStore([
        RoleProbabilitySlice("core", 0, 4, str(tmp_path / "data/core_full_p.npy")),
        RoleProbabilitySlice("weak", 4, 6, str(tmp_path / "data/weak_full_p.npy")),
    ], num_go=width)
    config = _config(data, "dynamic")
    config["task"] = "cc"
    config["full_task"]["model"]["go_chunk"] = 512
    saved = _train(data, config, tmp_path / "run", 2)
    input_dir = inference_fixture(tmp_path / "data", data)
    probability_path = input_dir / "backbone_ind_test_prob.f16.npy"
    np.save(probability_path, np.full((2, width), .2, np.float16))
    manifest_path = input_dir / "ind_test_input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["base_probability"]["sha256"] = runner.sha256(probability_path)
    protein_ids = input_dir / "protein_ids.txt"
    protein_ids.write_text("p0\np1\n")
    go_ids = input_dir / "go_ids.txt"
    go_ids.write_text("".join(f"GO:{column:07d}\n" for column in range(width)))
    registry = input_dir / "go_registry.tsv"
    registry.write_text("go_idx\tinput_go_id\n" + "".join(
        f"{column}\tGO:{column:07d}\n" for column in range(width)))
    manifest["protein_ids"] = str(protein_ids)
    manifest["protein_ids_file_sha256"] = runner.sha256(protein_ids)
    manifest["registries"] = {"go_registry": str(registry),
                              "go_registry_sha256": runner.sha256(registry)}
    manifest_path.write_text(json.dumps(manifest))
    metadata_path = tmp_path / "metadata.pkl"
    with metadata_path.open("wb") as handle:
        pickle.dump({"ind_test": {"cc": {"proteins": ["p0", "p1"], "prop_annotations": [[0, 1], [2]]}},
                     "train": {"cc": {"proteins": ["c0", "c1"], "prop_annotations": [[0, 1], [2]]}}}, handle)
    expert = np.full((2, width), .01, np.float32)
    expert[0, :2] = [.9, .8]
    expert[1, 2] = .85
    modelout = expert.copy()
    modelout[0, 1] = .6
    np.save(tmp_path / "expert.npy", expert)
    np.save(tmp_path / "modelout.npy", modelout)
    args = SimpleNamespace(checkpoint=tmp_path / "run/latest.pt", input_dir=input_dir,
                           metadata_file=metadata_path, work_dir=tmp_path / "run",
                           ablation="full", metric_backend="local_micro", auprc_mode="exact",
                           expert_prob=tmp_path / "expert.npy", modelout_prob=tmp_path / "modelout.npy",
                           expert_protein_ids=protein_ids, modelout_protein_ids=protein_ids,
                           expert_go_ids=go_ids, modelout_go_ids=go_ids,
                           references_aligned_to_input=False, allow_missing_references=False)
    hashes = {}
    model, _ = runner.build_model(data.feature_dim, data.ontology("cpu"), saved["model_config"], "dynamic", "cpu")
    model.load_state_dict(saved["model"])
    model.eval()
    for mode in ("full", "pp_off"):
        args.ablation = mode
        runner.evaluate(args, config, data, torch.device("cpu"))
        output = tmp_path / f"run/eval_step2/{mode}"
        prediction_path = output / "nbs_ind_test_prob.f32.npy"
        predictions = np.load(prediction_path)
        assert predictions.shape == (2, width) and predictions.dtype == np.float32
        assert np.all(np.isfinite(predictions))
        with torch.no_grad():
            batch = data.inference_batch(input_dir, np.arange(2), "cpu")
            direct = model(batch, **runner.forward_flags("dynamic", mode)).sigmoid().numpy()
        np.testing.assert_array_equal(predictions, direct)
        metrics = json.loads((output / "metrics/nbs_ind_test_metrics.json").read_text())
        assert metrics["ranking_analysis"]["status"] == "ok"
        comparison = json.loads((output / "metrics/nbs_w2s_comparison.json").read_text())
        assert {value["symbol"] for value in comparison["methods"].values()} == {"E", "M", "B", "G"}
        assert comparison["reference_comparison_complete"] is True
        assert comparison["evaluation_version"] == "0.8.5"
        for method in comparison["methods"].values():
            scores = method["standardized"]
            assert {"standard_protein_fmax", "standard_micro_pr_auc", "standard_micro_ap"} <= scores.keys()
            assert all(np.isfinite(scores[key]) for key in (
                "standard_protein_fmax", "standard_micro_pr_auc", "standard_micro_ap"))
        # These references place all three gold labels ahead of every unknown.
        for name in ("expert_prob", "stage1_modelout"):
            scores = comparison["methods"][name]["standardized"]
            assert scores["standard_micro_ap"] == pytest.approx(100.)
            assert scores["standard_micro_pr_auc"] == pytest.approx(100.)
            assert scores["standard_protein_fmax"] == pytest.approx(100.)
        for name in ("G_minus_E", "G_minus_M", "G_minus_B"):
            assert "standard_micro_ap" in comparison["deltas"][name]["standardized"]
        provenance = comparison["prediction_provenance"]
        assert provenance["ablation"] == mode and provenance["branch"] == "final"
        assert provenance["checkpoint_file_verified"] is True
        assert provenance["probability_sha256"] == runner.sha256(prediction_path)
        assert provenance["input_manifest_sha256"] == runner.sha256(manifest_path)
        assert provenance["manifest"]["uses_modelout_probability_as_target"] is False
        assert provenance["manifest"]["num_task_go"] == width
        assert provenance["manifest"]["runner_version"] == "0.8.5"
        assert provenance["manifest"]["model_architecture_version"] == "0.8.4"
        hashes[mode] = provenance["probability_sha256"]
    assert hashes["full"] != hashes["pp_off"]
    args.ablation = "full"
    prediction_path = tmp_path / "run/eval_step2/full/nbs_ind_test_prob.f32.npy"
    mtime = prediction_path.stat().st_mtime_ns
    def forbidden_reprediction(*args, **kwargs):
        raise AssertionError("verified cached predictions must not run another model forward")
    commands = []
    with monkeypatch.context() as patch:
        patch.setattr(runner.FullTaskGraphModelV084, "forward", forbidden_reprediction)
        patch.setattr(runner.subprocess, "run", lambda command, **kwargs: commands.append(command))
        runner.evaluate(args, config, data, torch.device("cpu"))
    assert prediction_path.stat().st_mtime_ns == mtime
    assert len(commands) == 1
    assert "--prediction-manifest" in commands[0] and "--expert-prob" in commands[0]
    assert "--modelout-prob" in commands[0] and "--precision-k" in commands[0]

    comparison_path = tmp_path / "run/eval_step2/full/metrics/nbs_w2s_comparison.json"
    original = json.loads(comparison_path.read_text())
    comparison_hash = runner.sha256(comparison_path)
    audit_dir = tmp_path / "cached_audit"
    audit_args = ["--recompute-comparison", str(comparison_path), "--metadata-file",
                  str(metadata_path), "--output-dir", str(audit_dir)]
    with monkeypatch.context() as patch:
        # This is the explicit saved-array entry point. It must not even load a
        # checkpoint, unlike ordinary evaluation's verified forward reuse.
        patch.setattr(torch, "load", forbidden_reprediction)
        patch.setattr(runner, "build_model", forbidden_reprediction)
        patch.setattr(runner.FullTaskGraphModelV084, "forward", forbidden_reprediction)
        audit = evaluator.main(audit_args)
        assert audit["checkpoint_loaded"] is False
        assert audit["model_inference_run"] is False
        assert audit["source_comparison"]["sha256"] == comparison_hash
        for name, method in original["methods"].items():
            assert audit["methods"][name]["primary"] == method["primary"]
            assert audit["methods"][name]["standardized"] == method["standardized"]
        checked = audit["verified_audit_sources"]
        for name, path in (("NBS_final", prediction_path), ("expert_prob_source", args.expert_prob),
                           ("stage1_modelout_source", args.modelout_prob), ("go_registry", registry)):
            assert checked[name]["sha256"] == runner.sha256(path)
        assert json.loads((audit_dir / "metric_audit_v085.json").read_text()) == audit
        assert (audit_dir / "metric_audit_v085.tsv").is_file()
        assert runner.sha256(comparison_path) == comparison_hash
        assert prediction_path.stat().st_mtime_ns == mtime

        # A same-shape reference edit must fail instead of producing a result
        # with stale E/M identities from the saved comparison.
        altered = expert.copy()
        altered[0, 0] = .7
        np.save(args.expert_prob, altered)
        with pytest.raises(ValueError, match="expert_prob_source SHA256 mismatch"):
            evaluator.main(audit_args)
