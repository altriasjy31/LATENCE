"""Exercise shell dispatch without loading a checkpoint or starting a GPU job."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nbs/run_nbs_v082.sh"


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


def test_dual_evaluation_exports_both_branches_and_requested_ablations(tmp_path):
    env, log = environment(tmp_path)
    work = tmp_path / "work/dual"
    work.mkdir(parents=True)
    (work / "latest.pt").touch()
    env["ABLATIONS"] = "full weak_off core_off"
    subprocess.run(["bash", str(SCRIPT), "evaluate", "dual"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 6
    assert {(row[row.index("--ablation") + 1], row[row.index("--branch") + 1]) for row in rows} == {
        (ablation, branch) for ablation in ("full", "weak_off", "core_off") for branch in ("final", "classification")}
    assert all(row[0] == "scripts/nbs/train_nbs_full_task_v082.py" for row in rows)


def test_teacher_training_uses_new_config_and_4000_step_default(tmp_path):
    env, log = environment(tmp_path)
    subprocess.run(["bash", str(SCRIPT), "pilot", "teacher"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row[row.index("--stage") + 1] for row in rows] == ["prepare", "train"]
    assert rows[-1][rows[-1].index("--steps") + 1] == "4000"
    assert rows[-1][rows[-1].index("--config") + 1].endswith("bp_full_task_v0.8.2_teacher.json")


def test_missing_series_checkpoint_stops_before_reference_export(tmp_path):
    env, log = environment(tmp_path)
    env["CHECKPOINT_STEPS"] = "600 2000 4000"
    result = subprocess.run(["bash", str(SCRIPT), "series", "dual"], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Checkpoint not found" in result.stderr
    assert not log.exists()
