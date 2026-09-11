#!/usr/bin/env python3
"""Summarize a predeclared series of NBS independent-test evaluations.

The evaluator writes one ``nbs_ind_test_metrics.json`` per checkpoint.  This
utility keeps the complete JSON payloads intact and produces two compact TSVs:

* primary Stage-1-compatible metrics and their NBS-minus-backbone deltas;
* selected candidate/query-scope/routing/delta diagnostics.

It deliberately does not choose a best epoch.  The independent test set is a
reporting/diagnostic set, not an early-stopping source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def parse_epochs(value: str) -> tuple[int, ...]:
    epochs: list[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        epoch = int(item)
        if epoch <= 0:
            raise ValueError("epochs must be positive integers")
        if epoch not in epochs:
            epochs.append(epoch)
    if not epochs:
        raise ValueError("at least one epoch is required")
    return tuple(epochs)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def flatten_numeric(
    value: Any,
    *,
    prefix: str = "",
    output: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    output = {} if output is None else output
    if is_number(value):
        output[prefix] = value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = str(key) if not prefix else f"{prefix}.{key}"
            flatten_numeric(child, prefix=child_prefix, output=output)
    return output


def selected_diagnostics(result: Mapping[str, Any]) -> dict[str, float | int]:
    selected: dict[str, float | int] = {}
    for key in (
        "candidate_analysis",
        "candidate_coverage",
        "query_scope_analysis",
        "rare_analysis",
        "confidence_intervals",
        "ranking_analysis",
        "diagnostics",
    ):
        if key in result:
            flatten_numeric(result[key], prefix=key, output=selected)
    return selected


def checkpoint_diagnostics(path: Path) -> dict[str, Any]:
    """Read learned residual scales and stored training metrics."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", {})
    suffixes = (
        "matcher.graph_delta_scale",
        "matcher.query_scale",
        "matcher.context_scale",
        "matcher.candidate_evidence_scale",
        "matcher.null_logit",
    )
    scalars: dict[str, float] = {}
    for suffix in suffixes:
        matches = [value for key, value in state.items() if str(key).endswith(suffix)]
        if len(matches) != 1 or int(matches[0].numel()) != 1:
            continue
        raw = float(matches[0].detach().float().item())
        short = suffix.rsplit(".", 1)[-1]
        scalars[f"{short}_raw"] = raw
        if short.endswith("scale"):
            scalars[f"{short}_tanh"] = math.tanh(raw)
    return {
        "checkpoint_epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "learned_scalars": scalars,
        "training_epoch_metrics": payload.get("epoch_metrics", {}),
    }


def write_table(path: Path, rows: list[dict[str, Any]], *, leading: Iterable[str]) -> None:
    leading = tuple(leading)
    keys = sorted({key for row in rows for key in row if key not in leading})
    columns = [*leading, *keys]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    series_root: Path,
    epochs: tuple[int, ...],
    *,
    strict: bool,
    checkpoint_dir: Path | None = None,
    checkpoint_pattern: str = "nbs_epoch{epoch}.pt",
) -> dict[str, Any]:
    primary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    missing: list[int] = []

    for epoch in epochs:
        epoch_dir = series_root / f"epoch{epoch}"
        metrics_path = epoch_dir / "nbs_ind_test_metrics.json"
        if not metrics_path.is_file():
            missing.append(epoch)
            continue
        result = load_json(metrics_path)
        manifest_path = epoch_dir / "predictions" / "nbs_full_task_prediction_manifest.json"
        manifest = load_json(manifest_path) if manifest_path.is_file() else {}
        checkpoint_info: dict[str, Any] = {}
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / checkpoint_pattern.format(epoch=epoch)
            if not checkpoint_path.is_file():
                if strict:
                    raise FileNotFoundError(checkpoint_path)
            else:
                checkpoint_info = checkpoint_diagnostics(checkpoint_path)
        common = {
            "epoch": epoch,
            "checkpoint": manifest.get("checkpoint", ""),
            "checkpoint_sha256": manifest.get("checkpoint_sha256", ""),
            "num_samples": result.get("num_samples", ""),
            "num_classes": result.get("num_classes", ""),
        }
        primary = dict(common)
        flatten_numeric(result.get("metrics", {}), prefix="primary", output=primary)
        flatten_numeric(
            result.get("metric_deltas_vs_backbone", {}),
            prefix="delta_vs_backbone",
            output=primary,
        )
        primary_rows.append(primary)

        diagnostics = dict(common)
        diagnostics.update(selected_diagnostics(result))
        flatten_numeric(
            checkpoint_info,
            prefix="checkpoint_diagnostics",
            output=diagnostics,
        )
        diagnostic_rows.append(diagnostics)
        records.append(
            {
                **common,
                "metrics_file": str(metrics_path),
                "primary": result.get("metrics", {}),
                "delta_vs_backbone": result.get("metric_deltas_vs_backbone", {}),
                "candidate_analysis": result.get("candidate_analysis", {}),
                "candidate_coverage": result.get("candidate_coverage", {}),
                "query_scope_analysis": result.get("query_scope_analysis", {}),
                "rare_analysis": result.get("rare_analysis", {}),
                "confidence_intervals": result.get("confidence_intervals", {}),
                "ranking_analysis": result.get("ranking_analysis", {}),
                "diagnostics": result.get("diagnostics", {}),
                "checkpoint_diagnostics": checkpoint_info,
            }
        )

    if strict and missing:
        raise FileNotFoundError(
            "missing independent-test metrics for epochs: "
            + ", ".join(str(value) for value in missing)
        )
    if not primary_rows:
        raise FileNotFoundError(f"no completed epoch metrics found below {series_root}")

    series_root.mkdir(parents=True, exist_ok=True)
    primary_path = series_root / "nbs_epoch_series_primary.tsv"
    diagnostics_path = series_root / "nbs_epoch_series_diagnostics.tsv"
    summary_path = series_root / "nbs_epoch_series_summary.json"
    write_table(
        primary_path,
        primary_rows,
        leading=("epoch", "checkpoint", "checkpoint_sha256", "num_samples", "num_classes"),
    )
    write_table(
        diagnostics_path,
        diagnostic_rows,
        leading=("epoch", "checkpoint", "checkpoint_sha256", "num_samples", "num_classes"),
    )
    summary = {
        "schema_version": 1,
        "contract": {
            "checkpoint_selection": "no best epoch is selected from ind_test",
            "comparison": "every NBS checkpoint is compared with the same Stage-1 backbone predictions",
            "input_reuse": "all epochs must use one validated inductive-input cache",
        },
        "requested_epochs": list(epochs),
        "completed_epochs": [int(row["epoch"]) for row in primary_rows],
        "missing_epochs": missing,
        "records": records,
        "artifacts": {
            "primary_tsv": str(primary_path),
            "diagnostics_tsv": str(diagnostics_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-root", type=Path, required=True)
    parser.add_argument("--epochs", required=True, help="comma-separated predeclared epochs")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-pattern", default="nbs_epoch{epoch}.pt")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    root = args.series_root.expanduser().resolve()
    summary = summarize(
        root,
        parse_epochs(args.epochs),
        strict=not bool(args.allow_missing),
        checkpoint_dir=(
            None
            if args.checkpoint_dir is None
            else args.checkpoint_dir.expanduser().resolve()
        ),
        checkpoint_pattern=str(args.checkpoint_pattern),
    )
    print(json.dumps(summary["artifacts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
