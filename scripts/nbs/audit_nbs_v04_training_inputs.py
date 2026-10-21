#!/usr/bin/env python3
"""Audit LATENCE NBS v0.4 training inputs.

NBS v0.4.4 supervision-contract fix:
- Retains the v0.4.3 TASK/RUN_TAG/EPOCH runtime overrides.
- Verifies core gold supervision is sourced from train.prop_annotations.
- Verifies weak supervision is expert-assisted modelout annotation membership
  (source comparison > 0.5) plus the retained modelout probability payload.
- Checks gold GO counts against the first-stage train annotation counts.
- Samples sparse pseudo probabilities against dense modelout probabilities.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "0.4.4"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_json(path: Path, label: str, failures: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        failures.append(f"missing {label}: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive error reporting
        failures.append(f"failed to read {label} {path}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        failures.append(f"{label} must contain a JSON object: {path}")
        return None
    return value


def _normalise_task(value: str) -> str:
    task = value.strip().lower()
    aliases = {
        "biological_process": "bp",
        "molecular_function": "mf",
        "cellular_component": "cc",
    }
    task = aliases.get(task, task)
    if task not in {"bp", "mf", "cc"}:
        raise ValueError(f"unsupported task {value!r}; expected bp, mf, or cc")
    return task


def _manifest_task(manifest: dict[str, Any]) -> str | None:
    for key in ("task", "metadata_task"):
        value = manifest.get(key)
        if not isinstance(value, str):
            continue
        try:
            return _normalise_task(value)
        except ValueError:
            continue
    return None


def _resolve_runtime_config(
    root: Path,
    template: dict[str, Any],
    *,
    task_override: str | None,
    run_tag_override: str | None,
    epoch_override: int | None,
    data_root_override: str | None,
    alignment_override: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a task-specific audit view from a shared config template."""
    config = copy.deepcopy(template)

    task = _normalise_task(task_override or str(config.get("task", "bp")))
    run_tag = run_tag_override or config.get("run_tag")
    if not isinstance(run_tag, str) or not run_tag.strip():
        raise ValueError("RUN_TAG/--run-tag is required")
    run_tag = run_tag.strip()

    epoch_raw = epoch_override if epoch_override is not None else config.get("data_epoch")
    if epoch_raw is None:
        raise ValueError("EPOCH/--epoch is required")
    epoch = int(epoch_raw)
    if epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {epoch}")

    task_root_rel = (
        Path(data_root_override)
        if data_root_override
        else Path("outputs/latence_nbs") / run_tag / f"epoch{epoch}" / task
    )
    alignment_rel = (
        Path(alignment_override)
        if alignment_override
        else task_root_rel
        / "nbs_indices"
        / "go_boxsqel_512"
        / "go_box_alignment_manifest.json"
    )

    config["task"] = task
    config["run_tag"] = run_tag
    config["data_epoch"] = epoch
    config.setdefault("data", {})["root"] = str(task_root_rel)
    config.setdefault("go_boxsqel", {})["alignment_manifest"] = str(alignment_rel)

    # This field is not used for input validation, but keeping it task-specific
    # avoids misleading resolved-config diagnostics.
    training = config.setdefault("training", {})
    current_output = str(training.get("output_dir", ""))
    if not current_output or "bp_nbs" in current_output:
        training["output_dir"] = f"outputs/latence_nbs_train/{task}_nbs_v0.4"

    resolved = {
        "task": task,
        "run_tag": run_tag,
        "epoch": epoch,
        "task_root": str(_resolve(root, task_root_rel)),
        "alignment_manifest": str(_resolve(root, alignment_rel)),
    }
    return config, resolved




def _resolve_manifest_data(base: Path, value: str | Path) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw.resolve()
    candidate = (base / raw).resolve()
    if candidate.exists():
        return candidate
    return (base / raw.name).resolve()


def _role_spec(manifest: dict[str, Any], role: str) -> dict[str, Any] | None:
    for item in manifest.get("roles", []):
        if isinstance(item, dict) and str(item.get("role")) == role:
            return item
    return None


def _chunked_bincount(values: np.ndarray, *, minlength: int, chunk: int = 5_000_000) -> np.ndarray:
    counts = np.zeros(int(minlength), dtype=np.int64)
    size = int(values.size)
    for start in range(0, size, int(chunk)):
        block = np.asarray(values[start : start + int(chunk)], dtype=np.int64)
        if block.size:
            counts += np.bincount(block, minlength=minlength)[:minlength]
    return counts


def _audit_supervision_contract(
    *,
    config: dict[str, Any],
    task: str,
    inverted_path: Path,
    inverted: dict[str, Any] | None,
    gold_spec: dict[str, Any] | None,
    weak_manifest_path: Path,
    weak_manifest: dict[str, Any] | None,
    failures: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "core_gold_source": None,
        "weak_pseudo_source": None,
        "pseudo_probability_range": None,
        "pseudo_dense_sample_max_abs_diff": None,
        "gold_counts_match_train_prop_annotations": False,
    }
    contract = config.get("supervision_contract")
    if not isinstance(contract, dict):
        failures.append("training config lacks supervision_contract")
        return summary
    core = contract.get("core_gold")
    weak = contract.get("weak_pseudo")
    if not isinstance(core, dict) or not isinstance(weak, dict):
        failures.append("supervision_contract must define core_gold and weak_pseudo")
        return summary
    if weak_manifest is None or inverted is None:
        return summary

    core_role = str(core.get("role", "core"))
    weak_role = str(weak.get("role", "weak"))
    expected_gold_key = str(core.get("metadata_label_key", "prop_annotations"))
    core_role_spec = _role_spec(weak_manifest, core_role)
    weak_role_spec = _role_spec(weak_manifest, weak_role)
    if core_role_spec is None:
        failures.append(f"weak-graph manifest lacks core role {core_role!r}")
    elif str(core_role_spec.get("dataset_mode")) != str(core.get("dataset_mode", "train")):
        failures.append("core role does not map to dataset_mode=train")
    if weak_role_spec is None:
        failures.append(f"weak-graph manifest lacks weak role {weak_role!r}")
    elif str(weak_role_spec.get("dataset_mode")) != str(weak.get("dataset_mode", "exp_train")):
        failures.append("weak role does not map to dataset_mode=exp_train")

    rare = weak_manifest.get("rare_definition", {})
    if str(rare.get("training_annotation_key")) != expected_gold_key:
        failures.append(
            "core gold annotation key mismatch: "
            f"manifest={rare.get('training_annotation_key')!r}, expected={expected_gold_key!r}"
        )

    # Gold provenance + exact GO-count consistency with train.prop_annotations.
    if gold_spec is None:
        failures.append("cannot verify core gold provenance because gold index is missing")
    else:
        source_edge_raw = gold_spec.get("source_edge_index")
        if not isinstance(source_edge_raw, str):
            failures.append("gold index lacks source_edge_index provenance")
        else:
            gold_edge_path = _resolve_manifest_data(inverted_path.parent, source_edge_raw)
            gold_manifest_path = gold_edge_path.parent / "gold_annotations_manifest.json"
            gold_manifest = _read_json(gold_manifest_path, "gold annotation provenance manifest", failures)
            if gold_manifest is not None:
                if _manifest_task(gold_manifest) != task:
                    failures.append("gold annotation task does not match requested task")
                if str(gold_manifest.get("role")) != core_role:
                    failures.append("gold annotation role is not core")
                if str(gold_manifest.get("mode")) != str(core.get("dataset_mode", "train")):
                    failures.append("gold annotation mode is not train")
                if str(gold_manifest.get("source", {}).get("label_key")) != expected_gold_key:
                    failures.append("gold annotation source is not train.prop_annotations")
                summary["core_gold_source"] = {
                    "role": gold_manifest.get("role"),
                    "mode": gold_manifest.get("mode"),
                    "label_key": gold_manifest.get("source", {}).get("label_key"),
                    "manifest": str(gold_manifest_path),
                }

            if gold_edge_path.is_file():
                try:
                    gold_edge = np.load(gold_edge_path, mmap_mode="r")
                    if gold_edge.ndim != 2 or gold_edge.shape[0] != 2:
                        failures.append("gold source edge_index must have shape [2,E]")
                    else:
                        if gold_edge.shape[1] and core_role_spec is not None:
                            lo = int(core_role_spec.get("global_protein_idx_min", -1))
                            hi = int(core_role_spec.get("global_protein_idx_max", -1))
                            pmin = int(np.min(gold_edge[0]))
                            pmax = int(np.max(gold_edge[0]))
                            if pmin < lo or pmax > hi:
                                failures.append(
                                    "gold Protein-GO edges contain proteins outside the core role: "
                                    f"edge_range=({pmin},{pmax}), core_range=({lo},{hi})"
                                )
                        train_counts_file = rare.get("train_counts_file")
                        if isinstance(train_counts_file, str):
                            train_counts_path = _resolve_manifest_data(weak_manifest_path.parent, train_counts_file)
                            if train_counts_path.is_file():
                                train_counts = np.asarray(np.load(train_counts_path, mmap_mode="r"), dtype=np.float64)
                                if train_counts.shape != (int(inverted.get("num_go", 0)),):
                                    failures.append("train GO counts do not align with task GO space")
                                elif not np.allclose(train_counts, np.rint(train_counts), atol=1e-8):
                                    failures.append("train GO counts are unexpectedly non-integral")
                                else:
                                    observed = _chunked_bincount(
                                        gold_edge[1], minlength=train_counts.size
                                    )
                                    expected = np.rint(train_counts).astype(np.int64)
                                    mismatch = np.flatnonzero(observed != expected)
                                    if mismatch.size:
                                        examples = mismatch[:10].tolist()
                                        failures.append(
                                            "gold GO counts disagree with train.prop_annotations counts: "
                                            f"mismatched_terms={mismatch.size}, examples={examples}"
                                        )
                                    else:
                                        summary["gold_counts_match_train_prop_annotations"] = True
                            else:
                                failures.append(f"missing train GO counts file: {train_counts_path}")
                except Exception as exc:
                    failures.append(
                        f"failed to audit gold source edge_index {gold_edge_path}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                failures.append(f"missing gold source edge_index: {gold_edge_path}")

    # Weak pseudo provenance must be expert-assisted modelout > 0.5 with the
    # original probability retained as the soft target.
    modelout = weak_manifest.get("model_semantics", {}).get("modelout", {})
    expected_prediction_key = str(
        weak.get("prediction_key", "modelout::mix_expert_base_anchor::decoderprob::expert")
    )
    if str(modelout.get("prediction_key")) != expected_prediction_key:
        failures.append("weak pseudo prediction_key is not the configured modelout output")
    if str(modelout.get("decoder_prob_source")) != str(weak.get("decoder_prob_source", "expert")):
        failures.append("weak modelout decoder_prob_source is not expert")
    if bool(modelout.get("external_probability_used")) is not bool(
        weak.get("external_probability_used", True)
    ):
        failures.append("weak modelout external_probability_used disagrees with contract")

    pseudo_targets = weak_manifest.get("weak_pseudo_targets", {})
    if str(pseudo_targets.get("role")) != weak_role:
        failures.append("weak pseudo target role is not weak")
    if str(pseudo_targets.get("comparison")) != str(weak.get("comparison", ">")):
        failures.append("weak pseudo comparison operator disagrees with contract")
    try:
        threshold = float(pseudo_targets.get("threshold"))
    except (TypeError, ValueError):
        threshold = float("nan")
    if threshold != float(weak.get("threshold", 0.5)):
        failures.append("weak pseudo threshold must be exactly 0.5")
    if list(pseudo_targets.get("edge_attr_columns", [])) != ["modelout_probability"]:
        failures.append("weak pseudo edge payload is not modelout_probability")
    if not bool(weak.get("use_probability_as_soft_target", True)):
        failures.append("weak supervision contract must retain modelout probability as soft target")
    if str(weak.get("negative_policy", "none")) != "none":
        failures.append("weak supervision contract must use negative_policy='none'")
    if not bool(weak.get("forbid_exp_train_prop_annotations", True)):
        failures.append("weak supervision contract must forbid exp_train.prop_annotations")

    summary["weak_pseudo_source"] = {
        "role": pseudo_targets.get("role"),
        "comparison": pseudo_targets.get("comparison"),
        "threshold": pseudo_targets.get("threshold"),
        "prediction_key": modelout.get("prediction_key"),
        "decoder_prob_source": modelout.get("decoder_prob_source"),
        "external_probability_used": modelout.get("external_probability_used"),
    }

    indptr_file = pseudo_targets.get("csr_indptr_file")
    indices_file = pseudo_targets.get("csr_indices_file")
    probability_file = pseudo_targets.get("csr_probability_file")
    if all(isinstance(value, str) for value in (indptr_file, indices_file, probability_file)):
        indptr_path = _resolve_manifest_data(weak_manifest_path.parent, str(indptr_file))
        indices_path = _resolve_manifest_data(weak_manifest_path.parent, str(indices_file))
        probability_path = _resolve_manifest_data(weak_manifest_path.parent, str(probability_file))
        try:
            indptr = np.load(indptr_path, mmap_mode="r")
            pseudo_go = np.load(indices_path, mmap_mode="r")
            pseudo_prob = np.load(probability_path, mmap_mode="r")
            if indptr.ndim != 1 or pseudo_go.ndim != 1 or pseudo_prob.ndim != 1:
                failures.append("weak pseudo CSR arrays must be one-dimensional")
            elif int(indptr[0]) != 0 or int(indptr[-1]) != pseudo_go.size or pseudo_prob.size != pseudo_go.size:
                failures.append("weak pseudo CSR arrays are misaligned")
            else:
                if pseudo_prob.size:
                    pmin = float(np.min(pseudo_prob))
                    pmax = float(np.max(pseudo_prob))
                    summary["pseudo_probability_range"] = [pmin, pmax]
                    if not np.all(np.isfinite(pseudo_prob)):
                        failures.append("weak pseudo probability contains non-finite values")
                    if pmin < 0.5 or pmax > 1.0:
                        failures.append(
                            f"weak pseudo stored probabilities must satisfy 0.5 <= p <= 1.0 (CSR membership records source modelout > 0.5 before float16 quantization), observed=({pmin},{pmax})"
                        )
                expected_nnz = int(pseudo_targets.get("nnz", pseudo_go.size))
                if pseudo_go.size != expected_nnz:
                    failures.append(
                        f"weak pseudo nnz mismatch: arrays={pseudo_go.size}, manifest={expected_nnz}"
                    )

                # Deterministically verify that the sparse probability payload is
                # the same first-stage modelout probability, not a transformed score.
                if weak_role_spec is not None and pseudo_prob.size:
                    dense_file = weak_role_spec.get("modelout_dense_file")
                    if isinstance(dense_file, str):
                        dense_path = _resolve_manifest_data(weak_manifest_path.parent, dense_file)
                        dense = np.load(dense_path, mmap_mode="r")
                        expected_shape = (
                            int(weak_role_spec.get("rows", 0)),
                            int(inverted.get("num_go", 0)),
                        )
                        if dense.shape != expected_shape:
                            failures.append(
                                f"modelout dense probability shape {dense.shape} != {expected_shape}"
                            )
                        else:
                            sample_n = min(4096, int(pseudo_prob.size))
                            rng = np.random.default_rng(3407)
                            flat = rng.choice(int(pseudo_prob.size), size=sample_n, replace=False)
                            rows = np.searchsorted(indptr, flat, side="right") - 1
                            gos = np.asarray(pseudo_go[flat], dtype=np.int64)
                            sparse_value = np.asarray(pseudo_prob[flat], dtype=np.float32)
                            dense_value = np.asarray(dense[rows, gos], dtype=np.float32)
                            max_diff = float(np.max(np.abs(sparse_value - dense_value), initial=0.0))
                            summary["pseudo_dense_sample_max_abs_diff"] = max_diff
                            if max_diff > 1e-3:
                                failures.append(
                                    "weak pseudo probability payload does not match dense modelout probability: "
                                    f"sample_max_abs_diff={max_diff}"
                                )
        except Exception as exc:
            failures.append(
                "failed to audit weak pseudo CSR/modelout probability: "
                f"{type(exc).__name__}: {exc}"
            )
    else:
        failures.append("weak pseudo target manifest lacks CSR probability files")

    pseudo_spec = inverted.get("indices", {}).get("pseudo")
    if isinstance(pseudo_spec, dict) and weak_role_spec is not None:
        protein_file = pseudo_spec.get("protein_idx")
        if isinstance(protein_file, str):
            protein_path = _resolve_manifest_data(inverted_path.parent, protein_file)
            try:
                pseudo_global = np.load(protein_path, mmap_mode="r")
                if pseudo_global.size:
                    lo = int(weak_role_spec.get("global_protein_idx_min", -1))
                    hi = int(weak_role_spec.get("global_protein_idx_max", -1))
                    pmin = int(np.min(pseudo_global))
                    pmax = int(np.max(pseudo_global))
                    if pmin < lo or pmax > hi:
                        failures.append(
                            "pseudo inverted index contains proteins outside weak role: "
                            f"edge_range=({pmin},{pmax}), weak_range=({lo},{hi})"
                        )
                if int(pseudo_spec.get("num_edges", pseudo_global.size)) != pseudo_global.size:
                    failures.append("pseudo inverted-index edge count disagrees with protein_idx array")
                if int(pseudo_targets.get("nnz", pseudo_global.size)) != pseudo_global.size:
                    failures.append("pseudo inverted-index edge count disagrees with weak pseudo nnz")
            except Exception as exc:
                failures.append(
                    f"failed to audit pseudo inverted-index protein roles: {type(exc).__name__}: {exc}"
                )

    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LATENCE NBS v0.4 training inputs")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True, help="Training config or shared config template")
    parser.add_argument("--task", default=None, help="Runtime task override: bp, mf, or cc")
    parser.add_argument("--run-tag", default=None, help="Runtime RUN_TAG override")
    parser.add_argument("--epoch", type=int, default=None, help="Runtime data epoch override")
    parser.add_argument("--data-root", default=None, help="Optional task-root override")
    parser.add_argument("--alignment-manifest", default=None, help="Optional task-to-ontology mapping override")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    config_path = _resolve(root, args.config)
    failures: list[str] = []

    template = _read_json(config_path, "training config template", failures)
    if template is None:
        report = {
            "schema_version": 2,
            "auditor_version": VERSION,
            "config_template": str(config_path),
            "failures": failures,
            "passed": False,
        }
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        print(text, end="")
        raise SystemExit(2)

    try:
        config, runtime = _resolve_runtime_config(
            root,
            template,
            task_override=args.task,
            run_tag_override=args.run_tag,
            epoch_override=args.epoch,
            data_root_override=args.data_root,
            alignment_override=args.alignment_manifest,
        )
    except (TypeError, ValueError) as exc:
        failures.append(f"invalid runtime task configuration: {exc}")
        report = {
            "schema_version": 2,
            "auditor_version": VERSION,
            "config_template": str(config_path),
            "failures": failures,
            "passed": False,
        }
        text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        print(text, end="")
        raise SystemExit(2)

    task = runtime["task"]
    data = config.get("data", {})
    task_root = Path(runtime["task_root"])
    if not task_root.is_dir():
        failures.append(f"missing task data root for {task}: {task_root}")

    inverted_path = task_root / str(data.get("go_protein_inverted_index_manifest", ""))
    inverted = _read_json(inverted_path, "GO->Protein inverted-index manifest", failures)
    gold_spec: dict[str, Any] | None = None
    if inverted is not None:
        gold_value = inverted.get("indices", {}).get("gold")
        gold_spec = gold_value if isinstance(gold_value, dict) else None
        if gold_spec is None:
            failures.append("GO->Protein inverted index lacks gold supervision")
        elif not isinstance(gold_spec.get("protein_major"), dict):
            failures.append("gold index lacks Protein->GO CSR required for weak->core->GO messages")
        if "candidate" not in inverted.get("indices", {}):
            failures.append("GO->Protein inverted index lacks candidate relation")

        source_manifest_raw = inverted.get("source_manifest")
        if isinstance(source_manifest_raw, str):
            source_manifest_path = Path(source_manifest_raw)
            if not source_manifest_path.is_absolute():
                source_manifest_path = (inverted_path.parent / source_manifest_path).resolve()
            source_manifest = _read_json(source_manifest_path, "inverted-index source manifest", failures)
            source_task = _manifest_task(source_manifest) if source_manifest else None
            if source_task is not None and source_task != task:
                failures.append(
                    f"inverted-index source task mismatch: requested={task}, manifest={source_task}"
                )

    weak_manifest_path = task_root / str(data.get("weak_graph_predictions_manifest", ""))
    weak_manifest = _read_json(weak_manifest_path, "weak-graph prediction manifest", failures)
    weak_task = _manifest_task(weak_manifest) if weak_manifest else None
    if weak_task is not None and weak_task != task:
        failures.append(f"weak-graph prediction task mismatch: requested={task}, manifest={weak_task}")

    supervision_summary = _audit_supervision_contract(
        config=config,
        task=task,
        inverted_path=inverted_path,
        inverted=inverted,
        gold_spec=gold_spec,
        weak_manifest_path=weak_manifest_path,
        weak_manifest=weak_manifest,
        failures=failures,
    )

    alignment_path = Path(runtime["alignment_manifest"])
    alignment = _read_json(alignment_path, "task-to-BoxSquaredEL alignment manifest", failures)
    source_row: np.ndarray | None = None
    if alignment is not None:
        arrays = alignment.get("arrays", {})
        source_spec = arrays.get("source_row") if isinstance(arrays, dict) else None
        source_file = source_spec.get("file") if isinstance(source_spec, dict) else None
        if not isinstance(source_file, str):
            failures.append("alignment manifest lacks arrays.source_row.file")
        else:
            source_path = alignment_path.parent / source_file
            if not source_path.is_file():
                failures.append(f"missing task-to-ontology source-row array: {source_path}")
            else:
                try:
                    source_row = np.load(source_path, mmap_mode="r")
                except Exception as exc:
                    failures.append(
                        f"failed to load task-to-ontology source-row array {source_path}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        alignment_task = _manifest_task(alignment)
        if alignment_task is not None and alignment_task != task:
            failures.append(f"alignment task mismatch: requested={task}, manifest={alignment_task}")

    full_path = _resolve(root, str(data.get("full_go_box_manifest", "")))
    full = _read_json(full_path, "full BoxSquaredEL ontology manifest", failures)
    full_classes = int(full.get("num_ontology_classes", 0)) if full else 0

    task_go_terms = int(inverted.get("num_go", 0)) if inverted else 0
    if source_row is not None:
        if source_row.size != task_go_terms:
            failures.append(
                "task-to-ontology mapping length differs from task GO space: "
                f"mapping={source_row.size}, task_go={task_go_terms}"
            )
        if source_row.size and full_classes > 0 and (
            int(source_row.min()) < 0 or int(source_row.max()) >= full_classes
        ):
            failures.append("task-to-ontology mapping leaves the BoxSquaredEL class space")

    gg_path = _resolve(root, str(data.get("boxsqel_gg_relations_manifest", "")))
    gg = _read_json(gg_path, "BoxSquaredEL G-G relation manifest", failures)
    if gg is not None and full is not None:
        if int(gg.get("num_classes", -1)) != full_classes:
            failures.append("full GO box and G-G relation class counts differ")
        gg_sha = gg.get("source", {}).get("boxsqel_checkpoint_sha256")
        full_sha = full.get("source", {}).get("checkpoint_sha256")
        if gg_sha != full_sha:
            failures.append("full GO box and G-G relations use different BoxSquaredEL checkpoints")

    pp_path = task_root / str(data.get("pp_sampling_indices_manifest", ""))
    pp = _read_json(pp_path, "P-P sampling-index manifest", failures)
    pp_axes: dict[str, Any] = {}
    if pp is not None:
        pp_task = _manifest_task(pp)
        if pp_task is not None and pp_task != task:
            failures.append(f"P-P sampling task mismatch: requested={task}, manifest={pp_task}")
        required_pp = {"ppi", "similar_to", "weak_to_core"}
        relations = pp.get("relations", {})
        if set(relations) != required_pp:
            failures.append("P-P sampling manifest does not contain exactly the three NBS relations")
        else:
            if int(relations["similar_to"].get("key_axis", -1)) != 1:
                failures.append(
                    "similar_to must be sampled by destination while preserving source->destination messages"
                )
            if int(relations["weak_to_core"].get("key_axis", -1)) != 0:
                failures.append("weak_to_core must be sampled by weak source")
        pp_axes = {name: spec.get("key_axis") for name, spec in relations.items()}

    training = config.get("training", {})
    if training.get("validation_used") or training.get("early_stopping"):
        failures.append("fixed-epoch training must not use validation or early stopping")
    local = config.get("local_sampling", {})
    if int(local.get("steps_per_epoch_per_rank", 0)) <= 0:
        failures.append("steps_per_epoch_per_rank must be positive for DDP")

    candidate_edges = 0
    if inverted is not None:
        candidate_edges = int(
            inverted.get("indices", {}).get("candidate", {}).get("num_edges", 0)
        )

    report = {
        "schema_version": 2,
        "auditor_version": VERSION,
        "config_template": str(config_path),
        "config_template_task": template.get("task"),
        "resolved": runtime,
        "task": task,
        "task_go_terms": task_go_terms,
        "full_boxsqel_classes": full_classes,
        "candidate_edges": candidate_edges,
        "gold_index_present": gold_spec is not None,
        "gold_protein_major_present": bool(
            gold_spec is not None and isinstance(gold_spec.get("protein_major"), dict)
        ),
        "supervision": supervision_summary,
        "pp_sampling_key_axis": pp_axes,
        "ddp": config.get("distributed", {}),
        "fixed_epoch": {
            "epochs": training.get("epochs"),
            "save_interval_epochs": training.get("save_interval_epochs"),
            "validation_used": training.get("validation_used"),
            "early_stopping": training.get("early_stopping"),
        },
        "failures": failures,
        "passed": not failures,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = _resolve(root, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
