"""Real train -> inductive graph -> probability -> four-baseline CLI integration."""
from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nbs_pg.latence_stores import RoleAwareBaseLogitStore, RoleProbabilitySlice
from scripts.nbs import train_nbs_full_task_v087 as runner
from scripts.nbs import eval_nbs_full_task_v087 as evaluator
from test_full_task_data_v080 import inference_fixture
from test_full_task_runner_v086 import _config as _old_config
from test_full_task_runner_v087 import _data


def _config(data, variant, prediction_mode):
    config = _old_config(data, variant)
    full = config["full_task"]
    for key in ("steps", "checkpoints", "warmup_steps"):
        full.pop(key, None)
    full.update(epochs=5, epoch_unit="weak", checkpoint_epochs=[1, 2, 3, 4, 5],
                scheduler=dict(name="epoch_warmup_hold_cosine", warmup_epochs=.1,
                               hold_until_epoch=3., min_lr=.001))
    full["model"].update(prediction_mode=prediction_mode, encoder_variant="legacy", pp_edge_dropout=0.,
                         direct_bias_init=-4., source_dropout_query_candidate=.2,
                         source_dropout_neighbor_candidate=.2, source_dropout_neighbor_pseudo=.2,
                         source_dropout_seed=8087)
    full["loss"].update(mining_source="current_base", anchor_weight=0. if prediction_mode == "direct" else .05)
    return config


def _train(data, config, directory, steps):
    args = SimpleNamespace(work_dir=directory, epochs=None, stop_step=steps, stop_epoch=None, resume=None)
    torch.manual_seed(config["full_task"]["seed"])
    runner.train(args, config, data, torch.device("cpu"), rank=0, world=1)
    return torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)



@pytest.mark.parametrize("prediction_mode", ["direct", "residual"])
def test_real_inductive_export_metrics_and_verified_prediction_reuse(tmp_path, monkeypatch, prediction_mode):
    data = _data(tmp_path / "data", "fixed")
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
    config = _config(data, "fixed", prediction_mode)
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
    args = SimpleNamespace(stage="export", checkpoint=tmp_path / "run/latest.pt", input_dir=input_dir,
                           metadata_file=metadata_path, work_dir=tmp_path / "run",
                           ablation="full", metric_backend="local_micro", auprc_mode="exact",
                           expert_prob=tmp_path / "expert.npy", modelout_prob=tmp_path / "modelout.npy",
                           expert_protein_ids=protein_ids, modelout_protein_ids=protein_ids,
                           expert_go_ids=go_ids, modelout_go_ids=go_ids,
                           references_aligned_to_input=False, allow_missing_references=False)
    # Export requires no E/M, metadata reading or evaluator invocation.
    with monkeypatch.context() as patch:
        def no_subprocess(*args, **kwargs):
            raise AssertionError("export may not evaluate or prepare references")
        patch.setattr(runner.subprocess, "run", no_subprocess)
        export_args = SimpleNamespace(stage="export", checkpoint=args.checkpoint, input_dir=input_dir,
                                      work_dir=args.work_dir, ablation="full", metadata_file=None)
        runner.evaluate(export_args, config, data, torch.device("cpu"))
    first_output = tmp_path / "run/eval_step2/full"
    assert not (first_output / "metrics").exists()
    initial_manifest = json.loads((first_output / "nbs_full_task_prediction_manifest.json").read_text())
    assert initial_manifest["prediction_mode"] == prediction_mode
    assert initial_manifest["source_flags"] == evaluator.expected_source_flags("full")
    args.stage = "evaluate"
    hashes = {}
    model, _ = runner.build_model(data.feature_dim, data.ontology("cpu"), saved["model_config"], "fixed", "cpu")
    model.load_state_dict(saved["model"])
    model.eval()
    for mode in ("full", "query_candidate_off"):
        args.ablation = mode
        runner.evaluate(args, config, data, torch.device("cpu"))
        output = tmp_path / f"run/eval_step2/{mode}"
        prediction_path = output / "nbs_ind_test_prob.f32.npy"
        predictions = np.load(prediction_path)
        assert predictions.shape == (2, width) and predictions.dtype == np.float32
        assert np.all(np.isfinite(predictions))
        with torch.no_grad():
            batch = data.inference_batch(input_dir, np.arange(2), "cpu")
            direct = model(batch, **runner.forward_flags("fixed", mode)).sigmoid().numpy()
        np.testing.assert_array_equal(predictions, direct)
        metrics = json.loads((output / "metrics/nbs_ind_test_metrics.json").read_text())
        assert metrics["ranking_analysis"]["status"] == "ok"
        comparison = json.loads((output / "metrics/nbs_w2s_comparison.json").read_text())
        assert {value["symbol"] for value in comparison["methods"].values()} == {"E", "M", "B", "G"}
        assert comparison["reference_comparison_complete"] is True
        assert comparison["evaluation_version"] == "0.8.7"
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
        assert provenance["manifest"]["runner_version"] == "0.8.7"
        assert provenance["manifest"]["model_architecture_version"] == "0.8.7"
        assert provenance["manifest"]["data_architecture_version"] == "0.8.4"
        assert provenance["manifest"]["encoder_variant"] == "legacy"
        assert provenance["manifest"]["model_config"]["pp_edge_dropout"] == 0.
        assert provenance["prediction_mode"] == prediction_mode
        assert provenance["source_flags"] == evaluator.expected_source_flags(mode)
        assert provenance["manifest"]["model_config"]["source_dropout_neighbor_pseudo"] == .2
        assert provenance["manifest"]["epoch"] == 1
        assert provenance["manifest"]["sampler_config"] == config["full_task"]["sampler"]
        hashes[mode] = provenance["probability_sha256"]
    assert hashes["full"] != hashes["query_candidate_off"]
    args.ablation = "full"
    prediction_path = tmp_path / "run/eval_step2/full/nbs_ind_test_prob.f32.npy"
    mtime = prediction_path.stat().st_mtime_ns
    def forbidden_reprediction(*args, **kwargs):
        raise AssertionError("verified cached predictions must not run another model forward")
    commands = []
    with monkeypatch.context() as patch:
        patch.setattr(runner.FullTaskGraphModelV087, "forward", forbidden_reprediction)
        patch.setattr(runner.subprocess, "run", lambda command, **kwargs: commands.append(command))
        runner.evaluate(args, config, data, torch.device("cpu"))
    assert prediction_path.stat().st_mtime_ns == mtime
    assert len(commands) == 1
    assert "--prediction-manifest" in commands[0] and "--expert-prob" in commands[0]
    assert "--modelout-prob" in commands[0] and "--precision-k" in commands[0]

    # Cached probabilities may not silently repair a tampered identity header.
    prediction_manifest = prediction_path.parent / "nbs_full_task_prediction_manifest.json"
    original_manifest_text = prediction_manifest.read_text()
    corruptions = {
        "prediction_mode": "residual" if prediction_mode == "direct" else "direct",
        "source_flags": evaluator.expected_source_flags("query_candidate_off"),
        "runner_version": "0.8.6", "model_architecture_version": "0.8.6",
        "data_architecture_version": "0.8.3",
    }
    for key, corrupt in corruptions.items():
        changed = json.loads(original_manifest_text)
        changed[key] = corrupt
        prediction_manifest.write_text(json.dumps(changed))
        with pytest.raises(FileExistsError, match="existing predictions differ"):
            runner.evaluate(args, config, data, torch.device("cpu"))
        assert json.loads(prediction_manifest.read_text())[key] == corrupt
        prediction_manifest.write_text(original_manifest_text)

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
        patch.setattr(runner.FullTaskGraphModelV087, "forward", forbidden_reprediction)
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
        assert json.loads((audit_dir / "metric_audit_v087.json").read_text()) == audit
        assert (audit_dir / "metric_audit_v087.tsv").is_file()
        assert runner.sha256(comparison_path) == comparison_hash
        assert prediction_path.stat().st_mtime_ns == mtime

        # A same-shape reference edit must fail instead of producing a result
        # with stale E/M identities from the saved comparison.
        altered = expert.copy()
        altered[0, 0] = .7
        np.save(args.expert_prob, altered)
        with pytest.raises(ValueError, match="expert_prob_source SHA256 mismatch"):
            evaluator.main(audit_args)


def test_v087_cached_audit_accepts_legacy_v086_without_loading_checkpoint(tmp_path, monkeypatch):
    # Use the real preceding evaluator to make a source-bound v086 comparison.
    from tests import test_full_task_eval_v086 as previous
    fixture = previous.files.__wrapped__(tmp_path)
    comparison, old = previous.audit_files(fixture)
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-array metric audit must not load a model or checkpoint")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", forbidden)
        result = evaluator.main(["--recompute-comparison", str(comparison), "--metadata-file",
                                 str(tmp_path / "metadata.pkl"), "--output-dir", str(tmp_path / "new_audit")])
    assert result["evaluation_version"] == "0.8.6"
    assert result["audit_version"] == "0.8.7"
    assert result["checkpoint_loaded"] is False
    for name in old["methods"]:
        assert result["methods"][name]["standardized"] == old["methods"][name]["standardized"]
