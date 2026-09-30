"""Exercise epoch training/export dispatch without checkpoints or GPU jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nbs/run_nbs_v087.sh"


def environment(tmp_path):
    recorder = tmp_path / "python-recorder"
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['NBS_TEST_ARGUMENT_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    recorder.chmod(0o755)
    log = tmp_path / "arguments.jsonl"
    env = dict(os.environ, PYTHON_BIN=str(recorder), NBS_TEST_ARGUMENT_LOG=str(log),
               WORK_ROOT=str(tmp_path / "work"), NUM_GPUS="1", ALLOW_MISSING_REFERENCES="1")
    for key in ("WORK_DIR", "CONFIG", "ABLATION", "CHECKPOINT", "RESUME", "STEPS", "EPOCHS",
                "STOP_STEP", "STOP_EPOCH", "CHECKPOINT_EPOCHS", "DEV_MANIFEST", "EXPERT_PROB", "MODELOUT_PROB"):
        env.pop(key, None)
    return env, log


def run(tmp_path, action, preset="direct", **overrides):
    env, log = environment(tmp_path)
    env.update(overrides)
    result = subprocess.run(["bash", str(SCRIPT), action, preset], env=env, capture_output=True, text=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, rows


def test_train_epoch_horizon_without_evaluation(tmp_path):
    result, rows = run(tmp_path, "train")
    assert result.returncode == 0, result.stderr
    assert [row[row.index("--stage") + 1] for row in rows] == ["prepare", "train"]
    train = rows[-1]
    assert train[train.index("--epochs") + 1] == "5"
    assert "--steps" not in train
    assert "--stop-step" not in train and "--stop-epoch" not in train
    assert train[train.index("--config") + 1].endswith("bp_full_task_v0.8.7_direct.json")
    assert not any("evaluate" in row or "prepare_nbs_stage1_references_v081.py" in " ".join(row) for row in rows)


def test_resume_stop_endpoints_and_spaces(tmp_path):
    result, rows = run(tmp_path, "train", "residual_epoch", CONFIG=str(tmp_path / "custom config.json"),
                       RESUME=str(tmp_path / "saved checkpoint.pt"), STOP_EPOCH="3", NUM_GPUS="2",
                       DEV_MANIFEST=str(tmp_path / "development manifest.json"))
    assert result.returncode == 0, result.stderr
    assert rows[-1][:2] == ["-m", "torch.distributed.run"]
    assert rows[-1][rows[-1].index("--stop-epoch") + 1] == "3"
    assert rows[-1][rows[-1].index("--resume") + 1] == str(tmp_path / "saved checkpoint.pt")
    assert rows[-1][rows[-1].index("--config") + 1] == str(tmp_path / "custom config.json")
    assert rows[-1][rows[-1].index("--development-manifest") + 1] == str(tmp_path / "development manifest.json")


def test_smoke_is_twenty_steps_separate_directory(tmp_path):
    result, rows = run(tmp_path, "smoke", "residual_epoch")
    assert result.returncode == 0, result.stderr
    assert rows[-1][rows[-1].index("--stop-step") + 1] == "20"
    assert rows[-1][rows[-1].index("--work-dir") + 1] == str(tmp_path / "work/smoke_residual_epoch")
    assert [row[row.index("--stage") + 1] for row in rows] == ["prepare", "train"]


@pytest.mark.parametrize("overrides", [{"STEPS": "4000"}, {"STOP_STEP": "100", "STOP_EPOCH": "1"}])
def test_ambiguous_or_legacy_stop_controls_rejected(tmp_path, overrides):
    result, rows = run(tmp_path, "train", **overrides)
    assert result.returncode == 2
    assert not rows


def test_export_does_not_evaluate_or_prepare_reference(tmp_path):
    checkpoint = tmp_path / "saved checkpoint.pt"
    checkpoint.touch()
    result, rows = run(tmp_path, "export", CHECKPOINT=str(checkpoint), ALLOW_MISSING_REFERENCES="0")
    assert result.returncode == 0, result.stderr
    assert len(rows) == 1
    assert rows[0][rows[0].index("--stage") + 1] == "export"
    assert "--expert-prob" not in rows[0]


def test_evaluate_reference_cache_and_checkpoint_paths(tmp_path):
    checkpoint = tmp_path / "saved checkpoint.pt"
    checkpoint.touch()
    result, rows = run(tmp_path, "evaluate", CHECKPOINT=str(checkpoint),
                       ALLOW_MISSING_REFERENCES="0", AUTO_REFERENCES="1",
                       STAGE1_CHECKPOINT=str(tmp_path / "relocated Stage1/checkpoint.pt"),
                       STAGE1_REFERENCE_DIR=str(tmp_path / "reference outputs"))
    assert result.returncode == 0, result.stderr
    assert len(rows) == 2
    assert rows[0][0] == "scripts/nbs/prepare_nbs_stage1_references_v081.py"
    assert rows[0][rows[0].index("--cache-policy") + 1] == "reuse"
    assert rows[1][rows[1].index("--checkpoint") + 1] == str(checkpoint)
    assert rows[1][rows[1].index("--expert-prob") + 1] == str(tmp_path / "reference outputs/expert_prob.f32.npy")


def test_effects_cover_legacy_and_fine_source_interventions(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    result, rows = run(tmp_path, "effects", CHECKPOINT=str(checkpoint))
    assert result.returncode == 0, result.stderr
    assert len(rows) == 10
    assert {row[row.index("--ablation") + 1] for row in rows[:-1]} == {
        "full", "weak_off", "core_off", "pp_off", "graph_off", "go_shuffle",
        "query_candidate_off", "neighbor_candidate_off", "neighbor_pseudo_off"}
    assert rows[-1] == ["scripts/nbs/eval_nbs_full_task_v087.py", "--summarize-work-dir",
                        str(tmp_path / "work/direct"), "--summary-checkpoint", str(checkpoint)]


def test_epoch_series_default_and_override(tmp_path):
    work = tmp_path / "work/direct"
    work.mkdir(parents=True)
    for epoch in (1, 3, 5):
        (work / f"nbs_epoch{epoch}.pt").touch()
    result, rows = run(tmp_path, "series")
    assert result.returncode == 0, result.stderr
    assert [row[row.index("--checkpoint") + 1] for row in rows] == [str(work / f"nbs_epoch{i}.pt") for i in (1, 3, 5)]


def test_missing_series_checkpoint_stops_before_reference_export(tmp_path):
    result, rows = run(tmp_path, "series", ALLOW_MISSING_REFERENCES="0")
    assert result.returncode == 2
    assert "Checkpoint not found" in result.stderr
    assert not rows


def test_original_reference_path_remains_default(tmp_path):
    env, log = environment(tmp_path)
    env.pop("STAGE1_CHECKPOINT", None)
    result = subprocess.run(["bash", str(SCRIPT), "references", "direct"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0][rows[0].index("--stage1-checkpoint") + 1] == (
        "/home/dataset-assist-0/datafile/latence-dataset/outputs/weak_exp_train_detr/"
        "bp_weak_detr_v3_expert_prob_warmstart340_to400/weak_detr_decoder_epoch100.pt")


def test_configs_only_change_the_declared_prediction_and_regularization():
    configs = {name: json.loads((ROOT / f"nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.7_{name}.json").read_text())
               for name in ("residual_epoch", "direct", "direct_source_dropout")}
    historical = json.loads((ROOT / "nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.6_legacy.json").read_text())
    for name, config in configs.items():
        full = config["full_task"]
        assert config["release_version"] == "0.8.7"
        assert config["stage"]["name"] == "nbs_v087_" + name
        assert full["sampler"] == historical["full_task"]["sampler"]
        assert full["model"]["encoder_variant"] == "legacy"
        assert full["epochs"] == 5 and full["epoch_unit"] == "weak"
        assert "steps" not in full and "warmup_steps" not in full
        assert full["learning_rate"] == 1e-4
        assert full["loss"]["mining_source"] == "current_base"
        assert full["loss"]["anchor_weight"] == (0.05 if name == "residual_epoch" else 0.0)
        assert full["model"]["prediction_mode"] == ("residual" if name == "residual_epoch" else "direct")
        for key in ("query_candidate", "neighbor_candidate", "neighbor_pseudo"):
            assert full["model"]["source_dropout_" + key] == (0.2 if name == "direct_source_dropout" else 0.0)
