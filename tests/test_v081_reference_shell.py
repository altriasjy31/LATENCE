"""Verify reference creation happens before NBS and once per checkpoint series."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def run_shell(tmp_path):
    root = tmp_path / "project with spaces"
    scripts = root / "scripts/nbs"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts/nbs/run_nbs_v081.sh"
    (scripts / source.name).write_text(source.read_text())
    fake = tmp_path / "fake_python"
    fake.write_text(f"#!{sys.executable}\n" + "import json, os, sys\n"
                    "with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
    fake.chmod(0o755)
    log = tmp_path / "calls.jsonl"

    def run(action, **overrides):
        env = {key: value for key, value in os.environ.items() if not key.startswith(
            ("EXPERT_", "MODELOUT_", "STAGE1_", "EXTERNAL_", "ALLOW_MISSING_", "AUTO_REFERENCES", "REFERENCES_"))}
        env.update(PYTHON_BIN=str(fake), CALL_LOG=str(log), NUM_GPUS="1")
        env.update(overrides)
        result = subprocess.run(["bash", str(scripts / source.name), action, "graph"],
                                env=env, text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


def test_series_exports_once_before_three_nbs_evaluations(run_shell):
    result, calls = run_shell("series", EXTERNAL_PROB_PATH="/source with spaces/expert.prop.npy")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 4
    assert calls[0][0].endswith("prepare_nbs_stage1_references_v081.py")
    assert calls[0][calls[0].index("--external-prob-path") + 1] == "/source with spaces/expert.prop.npy"
    for call in calls[1:]:
        assert call[call.index("--stage") + 1] == "evaluate"
        assert call[call.index("--expert-prob") + 1].endswith("expert_prob.f32.npy")
        assert call[call.index("--modelout-prob") + 1].endswith("stage1_modelout.f32.npy")
        assert "--stage1-reference-manifest" in call
        assert "--references-aligned-to-input" not in call


def test_manual_pair_skips_generation(run_shell):
    result, calls = run_shell("evaluate", EXPERT_PROB="/e.npy", MODELOUT_PROB="/m.npy", REFERENCES_ALIGNED="1")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert calls[0][calls[0].index("--expert-prob") + 1] == "/e.npy"


def test_partial_manual_pair_fails_before_inference(run_shell):
    result, calls = run_shell("evaluate", EXPERT_PROB="/e.npy")
    assert result.returncode != 0 and not calls
    assert "unset both" in result.stderr


def test_pilot_does_not_read_independent_references(run_shell):
    result, calls = run_shell("pilot")
    assert result.returncode == 0, result.stderr
    assert [call[call.index("--stage") + 1] for call in calls] == ["prepare", "train"]


def test_references_only_does_not_load_nbs(run_shell):
    result, calls = run_shell("references")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and calls[0][0].endswith("prepare_nbs_stage1_references_v081.py")


def test_explicit_incomplete_diagnostics_does_not_export(run_shell):
    result, calls = run_shell("evaluate", ALLOW_MISSING_REFERENCES="1")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and "--allow-missing-references" in calls[0]
