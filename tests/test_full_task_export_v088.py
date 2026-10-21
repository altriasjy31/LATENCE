"""Real CC-width CPU training, inductive export and the Fmax-only CLI handoff."""
from __future__ import annotations

import csv
import json
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from nbs_pg.full_task_data_v088 import FullTaskDataV088
from nbs_pg.latence_stores import RoleAwareBaseLogitStore, RoleProbabilitySlice
from scripts.nbs import eval_nbs_full_task_v088 as evaluator
from scripts.nbs import train_nbs_full_task_v088 as runner
from test_full_task_data_v080 import inference_fixture
from test_full_task_runner_v088 import _config, _data, _train


def test_cc_full_width_train_export_and_fmax_only_cli(tmp_path, monkeypatch):
    # Preserve real sparse training labels and graph pools; expand only the
    # classifier/ontology axis to the production CC evaluator width.
    data_dir = tmp_path / "data"
    data = _data(data_dir, "dynamic")
    assert isinstance(data, FullTaskDataV088)
    width = 2903
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
        np.save(data_dir / f"{role}_full_p.npy", np.full((count, width), .2, np.float32))
    data.stores.episode_sampler.base_logit_store = RoleAwareBaseLogitStore([
        RoleProbabilitySlice("core", 0, 4, str(data_dir / "core_full_p.npy")),
        RoleProbabilitySlice("weak", 4, 6, str(data_dir / "weak_full_p.npy")),
    ], num_go=width)
    config = _config(data)
    config["task"] = "cc"
    saved = _train(data, config, tmp_path / "run", stop_step=2)
    assert saved["step"] == 2 and saved["epoch"] == 1
    assert saved["model"]["classifier.weight"].shape == (width, 8)
    assert saved["model_config"]["sage_layers"] == 3
    assert all(row["full_go_per_protein"] == width for row in saved["history"])

    input_dir = inference_fixture(data_dir, data)
    base_path = input_dir / "backbone_ind_test_prob.f16.npy"
    np.save(base_path, np.full((2, width), .2, np.float16))
    manifest_path = input_dir / "ind_test_input_manifest.json"
    inputs = json.loads(manifest_path.read_text())
    inputs["base_probability"]["sha256"] = runner.sha256(base_path)
    protein_ids = input_dir / "protein_ids.txt"
    protein_ids.write_text("p0\np1\n")
    registry = input_dir / "go_registry.tsv"
    registry.write_text("go_idx\tinput_go_id\n" + "".join(
        f"{column}\tGO:{column:07d}\n" for column in range(width)))
    inputs["protein_ids"] = str(protein_ids)
    inputs["protein_ids_file_sha256"] = runner.sha256(protein_ids)
    inputs["registries"] = {"go_registry": str(registry),
                            "go_registry_sha256": runner.sha256(registry)}
    manifest_path.write_text(json.dumps(inputs))
    metadata_path = tmp_path / "metadata.pkl"
    # No training-metric resources or E/M arrays are needed by this evaluator.
    with metadata_path.open("wb") as handle:
        pickle.dump({"ind_test": {"cc": {"proteins": ["p0", "p1"],
                                          "prop_annotations": [[0, 1], [2]]}}}, handle)
    args = SimpleNamespace(stage="export", checkpoint=tmp_path / "run/latest.pt",
                           input_dir=input_dir, work_dir=tmp_path / "run", ablation="full",
                           metadata_file=None, metric_backend="stage1", auprc_mode="exact")

    # Export itself cannot reach the evaluator or require independent labels.
    with monkeypatch.context() as patch:
        def forbidden_subprocess(*args, **kwargs):
            raise AssertionError("export must not invoke metric/reference preparation")
        patch.setattr(runner.subprocess, "run", forbidden_subprocess)
        runner.evaluate(args, config, data, torch.device("cpu"))
    destination = tmp_path / "run/eval_step2/full"
    assert not (destination / "metrics").exists()
    probability_path = destination / "nbs_ind_test_prob.f32.npy"
    probabilities = np.load(probability_path)
    assert probabilities.shape == (2, width) and probabilities.dtype == np.float32
    assert np.isfinite(probabilities).all()

    model, _ = runner.build_model(data.feature_dim, data.ontology("cpu"),
                                  saved["model_config"], "dynamic", "cpu")
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        batch = data.inference_batch(input_dir, [0, 1], "cpu")
        assert "targets" not in batch and "positive_mask" not in batch
        assert len(batch["blocks"]) == 3
        direct = model(batch).sigmoid().numpy()
        partitioned = np.concatenate([
            model(data.inference_batch(input_dir, [row], "cpu")).sigmoid().numpy()
            for row in range(2)])
    np.testing.assert_array_equal(probabilities, direct)
    np.testing.assert_allclose(probabilities, partitioned, rtol=1e-6, atol=1e-7)

    # This executes the actual evaluator script as a subprocess. Only the
    # already exported full model is passed through the runner's CLI handoff.
    args.stage = "evaluate"
    args.metadata_file = metadata_path
    runner.evaluate(args, config, data, torch.device("cpu"))
    metric_dir = destination / "metrics"
    assert {path.name for path in metric_dir.iterdir()} == {
        "nbs_w2s_comparison.json", "nbs_w2s_comparison.tsv"}
    comparison = json.loads((metric_dir / "nbs_w2s_comparison.json").read_text())
    assert comparison["evaluation_version"] == "0.8.8"
    assert comparison["evaluation_scope"] == "standard_protein_fmax_only"
    assert set(comparison["methods"]) == {"NBS_final", "backbone_base"}
    assert {method["symbol"] for method in comparison["methods"].values()} == {"G", "B"}
    assert comparison["metric_contract"]["requested_legacy_backend"] == "stage1"
    assert comparison["metric_contract"]["primary_backend"] == "not_computed"
    assert set(comparison["not_computed"]) == {
        "historical_primary_fmax", "AP", "PR-AUC", "TopK", "bootstrap", "E", "M"}
    for method in comparison["methods"].values():
        assert method["primary"] == method["micro"] == method["top_k"] == {}
        scores = method["standardized"]
        assert scores["standard_num_proteins"] == 2
        assert scores["standard_num_go"] == width
        assert scores["standard_num_positive_labels"] == 3
        assert scores["standard_num_protein_fmax_eligible"] == 2
        assert scores["standard_protein_fmax_threshold_step"] == .001
        assert 0 <= scores["standard_protein_fmax"] <= 100
        assert not any("ap" in key or "pr_auc" in key for key in scores)
    with (metric_dir / "nbs_w2s_comparison.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {row["comparison"] for row in rows} == {"NBS_final", "backbone_base", "G_minus_B"}
    assert "standardized.standard_protein_fmax" in rows[0]
    assert "standardized.standard_num_positive_labels" in rows[0]

    provenance = comparison["prediction_provenance"]
    manifest = provenance["manifest"]
    assert provenance["checkpoint_file_verified"] is True
    assert provenance["probability_sha256"] == runner.sha256(probability_path)
    assert provenance["input_manifest_sha256"] == runner.sha256(manifest_path)
    assert provenance["ablation"] == "full" and provenance["branch"] == "final"
    assert provenance["source_flags"] == evaluator.expected_source_flags("full")
    for key in ("runner_version", "model_architecture_version", "data_architecture_version"):
        assert manifest[key] == "0.8.8"
    assert manifest["encoder_variant"] == "hetero_tuned"
    assert manifest["prediction_mode"] == "direct" and manifest["output_fusion"] == "none"
    assert manifest["num_task_go"] == width
    assert manifest["uses_dense_backbone_logits_in_forward"] is False
    assert manifest["uses_expert_probability_in_nbs_forward"] is False
    assert manifest["uses_modelout_probability_in_nbs_forward"] is False
