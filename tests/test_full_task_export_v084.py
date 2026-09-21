"""Real train -> inductive graph -> probability -> four-baseline CLI integration."""
from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from nbs_pg.latence_stores import RoleAwareBaseLogitStore, RoleProbabilitySlice
from scripts.nbs import train_nbs_full_task_v084 as runner
from test_full_task_data_v080 import inference_fixture
from test_full_task_runner_v084 import _data, _config, _train


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
    manifest_path.write_text(json.dumps(manifest))
    (input_dir / "protein_ids.txt").write_text("p0\np1\n")
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
                           references_aligned_to_input=True, allow_missing_references=False)
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
        provenance = comparison["prediction_provenance"]
        assert provenance["ablation"] == mode and provenance["branch"] == "final"
        assert provenance["checkpoint_file_verified"] is True
        assert provenance["probability_sha256"] == runner.sha256(prediction_path)
        assert provenance["input_manifest_sha256"] == runner.sha256(manifest_path)
        assert provenance["manifest"]["uses_modelout_probability_as_target"] is False
        assert provenance["manifest"]["num_task_go"] == width
        hashes[mode] = provenance["probability_sha256"]
    assert hashes["full"] != hashes["pp_off"]
    args.ablation = "full"
    prediction_path = tmp_path / "run/eval_step2/full/nbs_ind_test_prob.f32.npy"
    mtime = prediction_path.stat().st_mtime_ns
    def forbidden_reprediction(*args, **kwargs):
        raise AssertionError("verified cached predictions must not run another model forward")
    monkeypatch.setattr(runner.FullTaskGraphModelV084, "forward", forbidden_reprediction)
    commands = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: commands.append(command))
    runner.evaluate(args, config, data, torch.device("cpu"))
    assert prediction_path.stat().st_mtime_ns == mtime
    assert len(commands) == 1
    assert "--prediction-manifest" in commands[0] and "--expert-prob" in commands[0]
    assert "--modelout-prob" in commands[0] and "--precision-k" in commands[0]
