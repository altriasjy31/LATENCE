"""Launcher commands and isolation, without importing torch or loading a model."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/nbs/run_nbs_v088_mb800.py"
spec = importlib.util.spec_from_file_location("mb800_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def project(tmp_path, monkeypatch):
    config = json.loads((SCRIPT.parents[2] / launcher.CONFIG_REL).read_text())
    config_path = tmp_path / launcher.CONFIG_REL
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config))
    runner = tmp_path / launcher.RUNNER_REL
    runner.parent.mkdir(parents=True)
    runner.write_text("# original runner stand-in: never executed by these unit tests\n")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    workdir = tmp_path / launcher.RUN_ROOT_REL / "hetero_tuned"
    return tmp_path, config, config_path, workdir


def add_bound_run(project, step=100, epoch=None):
    _, config, config_path, workdir = project
    workdir.mkdir(parents=True)
    marker = launcher.expected_marker(config, config_path.resolve(), workdir.resolve())
    (workdir / launcher.MARKER).write_text(json.dumps(marker))
    resolved = dict(config, runtime={"world_size": 1, "global_weak_batch": 640,
        "global_core_batch": 160, "accumulation_steps": 1, "steps_per_epoch": 765})
    (workdir / "resolved_config.json").write_text(json.dumps(resolved))
    (workdir / "latest.pt").write_bytes(b"checkpoint is never loaded by launcher")
    (workdir / "training_history.json").write_text(json.dumps([
        {"step": value, "seconds_per_step": 2., "elapsed_seconds": value * 2.,
         "steps_per_epoch": 765, "peak_allocated_gb": 18.}
        for value in range(5, step + 1, 5)]))
    if epoch:
        (workdir / f"nbs_epoch{epoch}.pt").write_bytes(b"completed epoch")
    return workdir


def test_probe_dry_run_is_plain_cuda_and_never_creates_workdir(project, capsys):
    _, _, _, workdir = project
    assert launcher.main(["probe", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not workdir.exists()
    assert [cmd[cmd.index("--stage") + 1] for cmd in plan["commands"]] == ["prepare", "train"]
    train = plan["commands"][1]
    assert train[train.index("--stop-step") + 1] == "20"
    assert "--resume" not in train and "torch.distributed.run" not in train
    assert train[train.index("--device") + 1] == "cuda:0"
    assert plan["environment"] == {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "WORLD_SIZE": "1",
        "RANK": "0", "LOCAL_RANK": "0", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def test_environment_clears_distributed_inheritance_and_sets_threads():
    env = launcher.child_environment(2, 3, {"PATH": "/bin", "WORLD_SIZE": "2",
        "RANK": "1", "MASTER_ADDR": "oldhost", "MASTER_PORT": "12345",
        "LOCAL_WORLD_SIZE": "2", "GROUP_RANK": "1", "TORCHELASTIC_RUN_ID": "old",
        "OMPI_COMM_WORLD_RANK": "1", "PMI_RANK": "1", "TORCHRUN_FOO": "x",
        "OMP_NUM_THREADS": "40", "MKL_NUM_THREADS": "40", "OPENBLAS_NUM_THREADS": "40"})
    assert env["WORLD_SIZE"] == "1" and env["RANK"] == env["LOCAL_RANK"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "2" and env["PATH"] == "/bin"
    assert not any(key.startswith(("TORCHELASTIC_", "TORCHRUN_", "OMPI_", "PMI_")) for key in env)
    assert "MASTER_PORT" not in env and "MASTER_ADDR" not in env and "GROUP_RANK" not in env
    assert {env[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")} == {"3"}


def test_optional_probe_100_steps_covers_learning_rate_warmup(project, capsys):
    launcher.main(["probe", "--probe-steps", "100", "--dry-run"])
    command = json.loads(capsys.readouterr().out)["commands"][1]
    assert command[command.index("--stop-step") + 1] == "100"
    assert not project[3].exists()


def test_probe_runs_prepare_then_train_without_shell_and_prints_status(project, monkeypatch, capsys):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["shell"] is False and kwargs["check"] is True
        assert kwargs["cwd"] == project[0]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.main(["probe"]) == 0
    assert len(calls) == 2
    marker = json.loads((project[3] / launcher.MARKER).read_text())
    assert marker["tooling_version"] == "0.8.8.2" and marker["model_version"] == "0.8.8"
    assert '"state": "no training history"' in capsys.readouterr().out


def test_train_resumes_only_latest_in_same_directory_with_fixed_epoch_horizon(project, capsys):
    workdir = add_bound_run(project)
    launcher.main(["train", "--stop-epoch", "5", "--dry-run"])
    train = json.loads(capsys.readouterr().out)["commands"][1]
    assert train[train.index("--resume") + 1] == str(workdir / "latest.pt")
    assert train[train.index("--stop-epoch") + 1] == "5"
    assert "--epochs" not in train  # Never shrink or rewrite the configured five-epoch schedule.
    assert "--stop-step" not in train


def test_existing_target_epoch_is_noop(project, monkeypatch, capsys):
    add_bound_run(project, epoch=1)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: pytest.fail("must not relaunch"))
    assert launcher.main(["train"]) == 0
    assert "already exists" in capsys.readouterr().out


@pytest.mark.parametrize("epoch", [1, 5])
def test_evaluate_uses_explicit_completed_epoch_and_original_metric_protocol(project, capsys, epoch):
    workdir = add_bound_run(project, epoch=epoch)
    launcher.main(["evaluate", "--epoch", str(epoch), "--dry-run",
                   "--input-dir", "input with spaces", "--metadata-file", "meta;literal.pkl"])
    command = json.loads(capsys.readouterr().out)["commands"][0]
    assert command[command.index("--checkpoint") + 1] == str(workdir / f"nbs_epoch{epoch}.pt")
    assert command[command.index("--metric-backend") + 1] == "stage1"
    assert command[command.index("--auprc-mode") + 1] == "exact"
    assert command[command.index("--input-dir") + 1] == str(project[0] / "input with spaces")
    assert command[command.index("--metadata-file") + 1] == str(project[0] / "meta;literal.pkl")


@pytest.mark.parametrize("arguments", [
    ["probe", "--physical-gpu", "0,1"], ["probe", "--ablation", "weak_off"],
    ["probe", "--resume", "/old/latest.pt"], ["train", "--stop-epoch", "2"],
    ["probe", "--stop-epoch", "1"], ["train", "--epoch", "1"],
    ["probe", "--stage", "train"], ["probe", "--threads", "0"],
    ["train", "--probe-steps", "100"],
])
def test_unsupported_overrides_rejected(project, arguments):
    with pytest.raises(SystemExit):
        launcher.main(arguments)
    assert not project[3].exists()


@pytest.mark.parametrize("key,value", [("weak_batch", 64), ("core_batch", 16),
    ("accumulation_steps", 4), ("epochs", 1), ("learning_rate", .003)])
def test_batch_contract_rejected_before_any_output(project, key, value):
    config = project[1]
    config["full_task"][key] = value
    project[2].write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        launcher.main(["probe", "--dry-run"])
    assert not project[3].exists()


def test_old_config_and_output_paths_rejected(project):
    old = project[0] / "outputs/latence_nbs_experiments/v088/hetero_tuned"
    with pytest.raises(SystemExit):
        launcher.main(["probe", "--work-dir", str(old), "--dry-run"])
    with pytest.raises(SystemExit):
        launcher.main(["probe", "--config", "nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.8_hetero_tuned.json"])
    assert not old.exists() and not project[3].exists()


@pytest.mark.parametrize("mutation", ["marker", "resolved", "runtime", "missing_latest", "external_symlink"])
def test_resume_refuses_mismatched_existing_run(project, mutation):
    workdir = add_bound_run(project)
    if mutation == "marker":
        (workdir / launcher.MARKER).unlink()
    elif mutation in ("resolved", "runtime"):
        path = workdir / "resolved_config.json"
        saved = json.loads(path.read_text())
        if mutation == "resolved":
            saved["full_task"]["seed"] += 1
        else:
            saved["runtime"]["world_size"] = 2
        path.write_text(json.dumps(saved))
    elif mutation == "missing_latest":
        (workdir / "latest.pt").unlink()
    else:
        (workdir / "latest.pt").unlink()
        (workdir / "latest.pt").symlink_to(project[0] / "old_checkpoint.pt")
    with pytest.raises(SystemExit):
        launcher.main(["train", "--dry-run"])


def test_probe_refuses_existing_training_and_eval_requires_epoch_checkpoint(project):
    add_bound_run(project)
    with pytest.raises(SystemExit):
        launcher.main(["probe", "--dry-run"])
    with pytest.raises(SystemExit):
        launcher.main(["evaluate", "--dry-run"])


def test_kernel_lock_is_nonblocking_and_released(project):
    with launcher.run_lock(project[3]):
        with pytest.raises(ValueError, match="Another launcher"):
            with launcher.run_lock(project[3]):
                pytest.fail("second process must not enter")
    with launcher.run_lock(project[3]):
        pass  # A lock file can remain; only the live kernel lock prevents launch.


def test_status_weights_actual_windows_without_elapsed_difference_on_resume(project):
    workdir = add_bound_run(project)
    path = workdir / "training_history.json"
    rows = json.loads(path.read_text())
    rows.append({"step": 103, "seconds_per_step": 4., "elapsed_seconds": 12.,
                 "steps_per_epoch": 765, "peak_allocated_gb": 16.})
    path.write_text(json.dumps(rows))
    report = launcher.status(workdir)
    assert report["measured_steps"] == 30
    assert report["seconds_per_step"] == pytest.approx((27 * 2 + 3 * 4) / 30)
    assert report["rank0_peak_allocated_gb"] == 18.
    assert report["remaining_training_hours_to_epoch1"] == pytest.approx((765 - 103) * 2.2 / 3600)
    assert report["estimated_epoch_speedup_vs_v088_epoch1"] == pytest.approx(127899.59018327296 / (2.2 * 765))
    assert "not an observed complete-epoch speedup" in report["speedup_scope"]
    assert "loading" in report["timing_scope"] and "evaluation" in report["timing_scope"]


def test_status_100_step_probe_uses_last_30_and_excludes_warmup(project):
    workdir = add_bound_run(project)
    path = workdir / "training_history.json"
    rows = json.loads(path.read_text())
    for row in rows[:2]:
        row["seconds_per_step"] = 1000.
    path.write_text(json.dumps(rows))
    report = launcher.status(workdir)
    assert report["measured_steps"] == 30 and report["seconds_per_step"] == 2.


def test_failed_child_stops_sequence_and_propagates_error(project, monkeypatch):
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(7, command)
    monkeypatch.setattr(launcher.subprocess, "run", fail)
    assert launcher.main(["probe"]) == 7
    assert len(calls) == 1  # A failed prepare must never start training.


@pytest.mark.parametrize("key,value", [("min_lr", .0003), ("warmup_epochs", .2),
                                       ("hold_until_epoch", 1.)])
def test_scheduler_contract_rejected_before_output(project, key, value):
    config = project[1]
    config["full_task"]["scheduler"][key] = value
    project[2].write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        launcher.main(["probe", "--dry-run"])
    assert not project[3].exists()
