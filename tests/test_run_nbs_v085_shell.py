"""Exercise shell dispatch without loading a checkpoint or starting a GPU job."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nbs/run_nbs_v085.sh"


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
    for key in ("WORK_DIR", "CONFIG", "BRANCHES", "ABLATION", "ABLATIONS", "CHECKPOINT", "RESUME", "STEPS"):
        env.pop(key, None)
    return env, log


def test_evaluate_defaults_single_final_and_diagnose_six_interventions(tmp_path):
    env, log = environment(tmp_path)
    work = tmp_path / "work/dynamic"
    work.mkdir(parents=True)
    (work / "latest.pt").touch()
    subprocess.run(["bash", str(SCRIPT), "evaluate", "dynamic"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0][rows[0].index("--ablation") + 1] == "full"
    assert "--branch" not in rows[0]
    log.unlink()
    subprocess.run(["bash", str(SCRIPT), "diagnose", "dynamic"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 7
    assert {row[row.index("--ablation") + 1] for row in rows[:-1]} == {
        "full", "weak_off", "core_off", "pp_off", "graph_off", "go_shuffle"}
    assert all(row[0] == "scripts/nbs/train_nbs_full_task_v085.py" for row in rows[:-1])
    assert rows[-1] == ["scripts/nbs/eval_nbs_full_task_v085.py", "--summarize-work-dir", str(work),
                        "--summary-checkpoint", str(work / "latest.pt")]


def test_pilot_uses_new_config_and_600_step_default(tmp_path):
    env, log = environment(tmp_path)
    subprocess.run(["bash", str(SCRIPT), "pilot", "dynamic"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row[row.index("--stage") + 1] for row in rows] == ["prepare", "train"]
    assert rows[-1][rows[-1].index("--steps") + 1] == "600"
    assert rows[-1][rows[-1].index("--config") + 1].endswith("bp_full_task_v0.8.5_dynamic.json")


def test_missing_series_checkpoint_stops_before_reference_export(tmp_path):
    env, log = environment(tmp_path)
    env["CHECKPOINT_STEPS"] = "600 2000 4000"
    result = subprocess.run(["bash", str(SCRIPT), "series", "dynamic"], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Checkpoint not found" in result.stderr
    assert not log.exists()


def test_checkpoint_and_reference_override_with_spaces_are_single_arguments(tmp_path):
    env, log = environment(tmp_path)
    work = tmp_path / "work/dynamic"
    work.mkdir(parents=True)
    checkpoint = work / "saved checkpoint.pt"
    checkpoint.touch()
    source_checkpoint = tmp_path / "relocated Stage1" / "decoder checkpoint.pt"
    env.update(CHECKPOINT=str(checkpoint), STAGE1_CHECKPOINT=str(source_checkpoint),
               STAGE1_REFERENCE_DIR=str(tmp_path / "reference outputs"),
               INPUT_DIR=str(tmp_path / "prepared inputs"), ALLOW_MISSING_REFERENCES="0")
    for key in ("EXPERT_PROB", "MODELOUT_PROB", "AUTO_REFERENCES"):
        env.pop(key, None)
    subprocess.run(["bash", str(SCRIPT), "evaluate", "dynamic"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0][0] == "scripts/nbs/prepare_nbs_stage1_references_v081.py"
    assert rows[0][rows[0].index("--stage1-checkpoint") + 1] == str(source_checkpoint)
    assert rows[0][rows[0].index("--input-dir") + 1] == str(tmp_path / "prepared inputs")
    assert rows[1][rows[1].index("--checkpoint") + 1] == str(checkpoint)
    assert rows[1][rows[1].index("--expert-prob") + 1] == str(tmp_path / "reference outputs/expert_prob.f32.npy")
    assert rows[1][rows[1].index("--modelout-prob") + 1] == str(tmp_path / "reference outputs/stage1_modelout.f32.npy")


def test_references_default_uses_original_weak_exp_train_detr_location(tmp_path):
    env, log = environment(tmp_path)
    env.pop("STAGE1_CHECKPOINT", None)
    subprocess.run(["bash", str(SCRIPT), "references", "constant"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0][rows[0].index("--stage1-checkpoint") + 1] == (
        "/home/dataset-assist-0/datafile/latence-dataset/outputs/weak_exp_train_detr/"
        "bp_weak_detr_v3_expert_prob_warmstart340_to400/weak_detr_decoder_epoch100.pt")


def test_training_resume_and_config_override_preserve_spaces(tmp_path):
    env, log = environment(tmp_path)
    env.update(CONFIG=str(tmp_path / "custom experiment.json"),
               RESUME=str(tmp_path / "saved checkpoint.pt"), STEPS="2000", NUM_GPUS="2")
    subprocess.run(["bash", str(SCRIPT), "train", "constant"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[-1][:2] == ["-m", "torch.distributed.run"]
    assert rows[-1][rows[-1].index("--steps") + 1] == "2000"
    assert rows[-1][rows[-1].index("--resume") + 1] == env["RESUME"]
    assert rows[-1][rows[-1].index("--config") + 1] == env["CONFIG"]


def test_smoke_uses_separate_directory_and_twenty_steps(tmp_path):
    env, log = environment(tmp_path)
    subprocess.run(["bash", str(SCRIPT), "smoke", "constant"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[-1][rows[-1].index("--steps") + 1] == "20"
    assert rows[-1][rows[-1].index("--work-dir") + 1] == str(tmp_path / "work/smoke_constant")


def test_presets_change_only_declared_optimization_or_alignment():
    configs = {name: json.loads((ROOT / f"nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.5_{name}.json").read_text())
               for name in ("constant", "schedule", "lowlr", "aligned", "dynamic")}
    for name, config in configs.items():
        full = config["full_task"]
        assert config["stage"]["name"] == "nbs_v084_" + full["variant"]
        assert full["sampler"]["cache_dir"].endswith("v084/shared_pools")
        assert full["model"] == configs["constant"]["full_task"]["model"]
        assert full["loss"] == configs["constant"]["full_task"]["loss"]
        assert full["scheduler"]["horizon_steps"] == 4000
    assert configs["constant"]["full_task"]["scheduler"]["name"] == "constant"
    assert configs["schedule"]["full_task"]["scheduler"]["min_lr"] == 3e-5
    assert configs["lowlr"]["full_task"]["learning_rate"] == 1e-4
    assert configs["aligned"]["full_task"]["sampler"]["native_dropout"] == 1
    assert configs["dynamic"]["full_task"]["sampler"]["mode"] == "dynamic"


def test_default_schedule_and_development_path_forwarding(tmp_path):
    env, log = environment(tmp_path)
    env["DEV_MANIFEST"] = str(tmp_path / "development manifest.json")
    subprocess.run(["bash", str(SCRIPT), "pilot"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[-1][rows[-1].index("--work-dir") + 1].endswith("/schedule")
    assert rows[-1][rows[-1].index("--development-manifest") + 1] == env["DEV_MANIFEST"]


def test_cached_recompute_never_trains_or_exports_references(tmp_path):
    env, log = environment(tmp_path)
    env.update(COMPARISON_PATH=str(tmp_path / "old comparison.json"), PATH_MAP="/old location=/new location")
    subprocess.run(["bash", str(SCRIPT), "recompute"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0][0] == "scripts/nbs/eval_nbs_full_task_v085.py"
    assert rows[0][rows[0].index("--recompute-comparison") + 1] == env["COMPARISON_PATH"]
    assert rows[0][rows[0].index("--path-map") + 1] == env["PATH_MAP"]
