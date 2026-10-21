"""Full-only evaluator preserves the established metric implementations."""
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = load("v087_eval_fixture", Path(__file__).with_name("test_full_task_eval_v087.py"))
module = load("v088_evaluator", ROOT / "scripts/nbs/eval_nbs_full_task_v088.py")
files = previous.files


def arguments(files):
    argv = previous.arguments(files)
    argv[argv.index("--metric-backend") + 1] = "stage1"
    manifest_path = files / "prediction.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(version="0.8.8", runner_version="0.8.8", model_architecture_version="0.8.8",
                    data_architecture_version="0.8.8", encoder_variant="hetero_tuned",
                    prediction_mode="direct", output_fusion="none", uses_dense_backbone_logits_in_forward=False)
    manifest["model_config"].update(encoder_variant="hetero_tuned", prediction_mode="direct", hidden_dim=320, sage_layers=3)
    manifest_path.write_text(json.dumps(manifest))
    for flag in ("--expert-prob", "--modelout-prob"):
        i = argv.index(flag)
        del argv[i:i+2]
    return argv


def test_full_model_evaluation_does_not_require_E_M(files, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Fmax-only must not invoke complete metrics or AP ranking")
    monkeypatch.setattr(module, "compute_standard_metrics", forbidden)
    import nbs_pg.full_task_metrics_v085 as kernels
    monkeypatch.setattr(kernels, "_exact_micro", forbidden)
    result = module.main(arguments(files))
    assert result["evaluation_version"] == "0.8.8"
    assert result["evaluation_scope"] == "standard_protein_fmax_only"
    assert result["metric_contract"]["primary_backend"] == "not_computed"
    assert all(not value["primary"] and not value["micro"] and not value["top_k"] for value in result["methods"].values())
    assert all("standard_micro_ap" not in value["standardized"] for value in result["methods"].values())
    assert set(result["methods"]) == {"NBS_final", "backbone_base"}
    assert result["prediction_provenance"]["output_fusion"] == "none"
    assert result["prediction_provenance"]["uses_dense_backbone_logits_in_forward"] is False
    assert result["prediction_provenance"]["source_flags"] == dict.fromkeys(module.SOURCE_KEYS, True)


def test_metric_functions_and_standardized_formula_are_unchanged():
    assert module.compute_standard_metrics is previous.module.compute_standard_metrics
    assert inspect.getsource(module._add_standardized) == inspect.getsource(previous.module._add_standardized)
    labels = np.asarray([[1, 0, 1], [0, 1, 0]], bool)
    probability = np.asarray([[.8, .5, .2], [.2, .7, .1]], np.float32)
    complete = previous.module.compute_standard_metrics(labels, probability)
    fmax_only = module.compute_fmax_only(labels, probability)
    assert fmax_only == {key: complete[key] for key in fmax_only}
    assert not any("micro" in key for key in fmax_only)
    assert module._legacy().__file__ == previous.module._legacy().__file__
    assert inspect.getsource(module._legacy()._load_stage1_metric_backend) == inspect.getsource(previous.module._legacy()._load_stage1_metric_backend)


@pytest.mark.parametrize("ablation", ["weak_off", "core_off", "pp_off", "query_candidate_off", "go_shuffle"])
def test_paused_interventions_are_rejected(files, ablation):
    argv = arguments(files)
    path = files / "prediction.json"
    data = json.loads(path.read_text()); data["ablation"] = ablation
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="full evaluation only"):
        module.main(argv)


@pytest.mark.parametrize("key,value", [("output_fusion", "B_plus_G"), ("uses_dense_backbone_logits_in_forward", True), ("prediction_mode", "residual")])
def test_model_fusion_is_rejected(files, key, value):
    argv = arguments(files)
    path = files / "prediction.json"
    data = json.loads(path.read_text()); data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.main(argv)


def test_sweep_and_cached_audit_cli_disabled():
    with pytest.raises(ValueError, match="paused"):
        module.main(["--summarize-work-dir", "unused"])


@pytest.mark.parametrize("kind", ["ties", "empty_gold", "no_positive"])
def test_fmax_only_is_exactly_old_kernel_on_edge_cases(kind):
    labels = np.asarray([[1, 0, 1], [0, 1, 0]], bool)
    scores = np.asarray([[.5, .5, 1.], [0., .5, .5]], np.float32)
    if kind == "empty_gold": labels[1] = False
    if kind == "no_positive": labels[:] = False
    full = previous.module.compute_standard_metrics(labels, scores)
    actual = module.compute_fmax_only(labels, scores)
    assert actual == {key: full[key] for key in actual}
