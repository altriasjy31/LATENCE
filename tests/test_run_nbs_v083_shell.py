"""Exercise shell dispatch without loading a checkpoint or starting a GPU job."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nbs/run_nbs_v083.sh"


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
    work = tmp_path / "work/graph"
    work.mkdir(parents=True)
    (work / "latest.pt").touch()
    subprocess.run(["bash", str(SCRIPT), "evaluate", "graph"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0][rows[0].index("--ablation") + 1] == "full"
    assert "--branch" not in rows[0]
    log.unlink()
    subprocess.run(["bash", str(SCRIPT), "diagnose", "graph"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 7
    assert {row[row.index("--ablation") + 1] for row in rows[:-1]} == {
        "full", "weak_off", "core_off", "pp_off", "graph_off", "go_shuffle"}
    assert all(row[0] == "scripts/nbs/train_nbs_full_task_v083.py" for row in rows[:-1])
    assert rows[-1] == ["scripts/nbs/eval_nbs_full_task_v083.py", "--summarize-work-dir", str(work),
                        "--summary-checkpoint", str(work / "latest.pt")]


def test_graph_training_uses_new_config_and_4000_step_default(tmp_path):
    env, log = environment(tmp_path)
    subprocess.run(["bash", str(SCRIPT), "pilot", "graph"], env=env, check=True, capture_output=True)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row[row.index("--stage") + 1] for row in rows] == ["prepare", "train"]
    assert rows[-1][rows[-1].index("--steps") + 1] == "4000"
    assert rows[-1][rows[-1].index("--config") + 1].endswith("bp_full_task_v0.8.3_graph.json")


def test_missing_series_checkpoint_stops_before_reference_export(tmp_path):
    env, log = environment(tmp_path)
    env["CHECKPOINT_STEPS"] = "600 2000 4000"
    result = subprocess.run(["bash", str(SCRIPT), "series", "graph"], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Checkpoint not found" in result.stderr
    assert not log.exists()
