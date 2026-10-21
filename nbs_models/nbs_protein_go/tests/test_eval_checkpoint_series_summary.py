from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "nbs"
    / "summarize_nbs_checkpoint_series.py"
)
SPEC = importlib.util.spec_from_file_location("nbs_eval_series_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_epochs_is_ordered_and_unique():
    assert MODULE.parse_epochs("1,2,2;4") == (1, 2, 4)


def test_summarize_checkpoint_series(tmp_path: Path):
    for epoch, score in ((1, 51.0), (2, 52.0)):
        epoch_dir = tmp_path / f"epoch{epoch}"
        prediction_dir = epoch_dir / "predictions"
        prediction_dir.mkdir(parents=True)
        (epoch_dir / "nbs_ind_test_metrics.json").write_text(
            json.dumps(
                {
                    "num_samples": 3,
                    "num_classes": 5,
                    "metrics": {
                        "NBS_final": {"Fmax": score},
                        "backbone_base": {"Fmax": 50.0},
                    },
                    "metric_deltas_vs_backbone": {
                        "NBS_final": {"Fmax": score - 50.0}
                    },
                    "candidate_analysis": {},
                    "diagnostics": {
                        "probability_delta": {"overall": {"mean": 0.01 * epoch}}
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (prediction_dir / "nbs_full_task_prediction_manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint": f"nbs_epoch{epoch}.pt",
                    "checkpoint_sha256": str(epoch) * 64,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    summary = MODULE.summarize(tmp_path, (1, 2), strict=True)
    assert summary["completed_epochs"] == [1, 2]
    assert not summary["missing_epochs"]
    primary = (tmp_path / "nbs_epoch_series_primary.tsv").read_text(
        encoding="utf-8"
    )
    assert "primary.NBS_final.Fmax" in primary
    assert "delta_vs_backbone.NBS_final.Fmax" in primary
    diagnostics = (tmp_path / "nbs_epoch_series_diagnostics.tsv").read_text(
        encoding="utf-8"
    )
    assert "diagnostics.probability_delta.overall.mean" in diagnostics
