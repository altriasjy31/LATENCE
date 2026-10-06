"""CLI dispatch is model-only; it never triggers a test evaluation during training."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nbs/run_nbs_v088.sh"


def run(tmp_path, action, **overrides):
    log = tmp_path / "args.jsonl"
    python = tmp_path / "recorder"
    python.write_text(f"#!{sys.executable}\nimport json,sys\nwith open({str(log)!r},'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')\n")
    python.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(python), WORK_ROOT=str(tmp_path / "work"))
    for key in ("NUM_GPUS", "WORK_DIR", "STEPS", "STOP_STEP", "STOP_EPOCH", "CHECKPOINT", "ABLATION", "RESUME", "CONFIG", "PRESET"):
        env.pop(key, None)
    env.update(overrides)
    result = subprocess.run(["bash", str(SCRIPT), action], env=env, text=True, capture_output=True)
    return result, [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_training_defaults_to_two_GPU_and_never_evaluates(tmp_path):
    result, rows = run(tmp_path, "train")
    assert result.returncode == 0, result.stderr
    assert [r[r.index("--stage") + 1] for r in rows] == ["prepare", "train"]
    assert rows[-1][:2] == ["-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in rows[-1]
    assert not any("--input-dir" in r for r in rows)


def test_smoke_is_isolated_twenty_step_train(tmp_path):
    result, rows = run(tmp_path, "smoke", NUM_GPUS="1")
    assert result.returncode == 0, result.stderr
    assert rows[-1][rows[-1].index("--stop-step") + 1] == "20"
    assert rows[-1][rows[-1].index("--work-dir") + 1].endswith("/smoke_hetero_tuned")


@pytest.mark.parametrize("action", ["effects", "series", "diagnose", "pilot"])
def test_ablations_and_sweeps_are_not_exposed(tmp_path, action):
    result, rows = run(tmp_path, action)
    assert result.returncode == 2 and not rows


def test_nonfull_environment_ablation_is_rejected(tmp_path):
    result, rows = run(tmp_path, "evaluate", ABLATION="core_off")
    assert result.returncode == 2 and not rows


def test_evaluation_without_reference_export(tmp_path):
    checkpoint = tmp_path / "checkpoint with spaces.pt"; checkpoint.touch()
    result, rows = run(tmp_path, "evaluate", CHECKPOINT=str(checkpoint))
    assert result.returncode == 0, result.stderr
    assert len(rows) == 1
    assert rows[0][rows[0].index("--stage") + 1] == "evaluate"
    assert rows[0][rows[0].index("--ablation") + 1] == "full"
    assert "--allow-missing-references" in rows[0]
    assert rows[0][rows[0].index("--metric-backend") + 1] == "stage1"
    assert "--expert-prob" not in rows[0]
