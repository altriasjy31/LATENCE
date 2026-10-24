"""Exercise shipped config -> explicit loader, plus dynamic-target export."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_local_loader_v088 import export
from nbs_pg import local_loader_v089, local_loader_v090
from scripts.nbs import train_nbs_full_task_v090 as runner
from scripts.nbs import predict_nbs_go_v090 as targets
from nbs_pg.go_registry_v090 import GORegistryV090
from test_full_task_runner_v090 import _data, _config, _train
from test_full_task_data_v080 import inference_fixture


def _v090(config):
    result = deepcopy(config)
    result.update(release_version="0.9.0", model_version="0.9.0")
    result["stage"]["name"] = "nbs_v090_shared_go"
    return result


def test_real_mmap_builder_consumes_named_release_without_changing_config(export):
    config, _ = export
    config = _v090(config)
    before = deepcopy(config)
    stores = local_loader_v090.build_latence_nbs_stores(config)
    assert config == before
    assert stores.num_task_go == 512 and stores.full_boxes.num_go == 513
    assert stores.gold_messages.gather([0]).shape == (2, 512)
    assert not hasattr(stores, "go_registry_v090")  # Lazy for closed-set runs.
    with pytest.raises(ValueError, match="stage=nbs_v089_long_phase"):
        local_loader_v089.build_latence_nbs_stores(config)


@pytest.mark.parametrize("preset", ["mb6400_single", "mb12800_ddp"])
def test_actual_shipped_config_goes_through_main_builder(tmp_path, export, monkeypatch, preset):
    fixture, _ = export
    shipped = json.loads((runner.ROOT / "nbs_models/nbs_protein_go/configs" /
        f"bp_full_task_v0.9.0_{preset}_ep100.json").read_text())
    shipped["data"] = fixture["data"]
    shipped["go_boxsqel"] = fixture["go_boxsqel"]
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(shipped))
    calls = []
    def data_boundary(config, *, stores):
        calls.append((config["release_version"], stores.num_task_go))
        return SimpleNamespace(prepare_neighbors=lambda **kwargs: {"prepared": True})
    monkeypatch.setattr(runner, "FullTaskData", data_boundary)
    monkeypatch.setattr("sys.argv", ["runner", "--stage", "prepare", "--config", str(config_file),
        "--work-dir", str(tmp_path / "work"), "--device", "cpu"])
    for name in ("WORLD_SIZE", "RANK"):
        monkeypatch.delenv(name, raising=False)
    runner.main()
    assert calls == [("0.9.0", 512)]


@pytest.mark.parametrize("key,value", [("release_version", "0.8.9.1"), ("model_version", "0.8.8")])
def test_forged_consumer_rejected_before_opening_data(export, key, value):
    config, _ = export
    config = _v090(config)
    config[key] = value
    with pytest.raises(ValueError, match=key):
        local_loader_v090.build_latence_nbs_stores(config)


def test_dynamic_go_export_reorders_columns_and_adds_existing_context_node(tmp_path):
    data = _data(tmp_path / "data")
    config = _config(data, weak_batch=2)
    _train(data, config, tmp_path / "train")
    checkpoint = tmp_path / "train/nbs_epoch2.pt"
    model, _ = targets.verified_model(checkpoint, config, data, torch.device("cpu"))
    inp = inference_fixture(tmp_path / "data", data)
    records = [{"go_id": f"GO:{row + 1:07d}", "ontology_row": row} for row in range(8)]
    task_ids = [f"GO:{row + 1:07d}" for row in data.task_to_ontology_rows]
    registry = GORegistryV090(records, task_ids, data.task_to_ontology_rows, num_ontology_rows=8)
    requested = ["GO:0000008", "GO:0000002", "GO:0000001", "GO:0000001"]
    output, columns = targets.export_targets(model, data, registry, requested, inp,
        tmp_path / "targets", device="cpu", batch_size=1, go_chunk_size=2)
    values = np.load(output)
    assert values.shape == (2, 4)
    with torch.no_grad():
        expected = model(data.inference_batch(inp, np.arange(2), "cpu"),
            target_ontology_ids=[7, 1, 0, 0], go_chunk_size=8).sigmoid().numpy()
    np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-6)
    assert np.array_equal(values[:, 2], values[:, 3])
    assert columns.read_text().splitlines()[1].endswith("\t7\t0")
    with pytest.raises(FileExistsError, match="empty output"):
        targets.export_targets(model, data, registry, requested, inp, tmp_path / "targets")
    with pytest.raises(ValueError, match="no supplied ontology geometry"):
        targets.export_targets(model, data, registry, ["GO:9999999"], inp, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
