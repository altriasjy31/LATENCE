from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import torch

from nbs_pg import NBSConfig, ProteinGONBSModel, load_boxsqel_training_contract
from nbs_pg.training import (
    NBSLossConfig,
    NBSRunComponents,
    freeze_go_geometry,
)



def _project_path(value: str) -> str:
    path = Path(value)
    return str(path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve())


def _resolve_boxsqel_contract(config: dict[str, Any]):
    spec = dict(config.get("go_boxsqel", {}))
    manifest = spec.get("manifest")
    if not manifest:
        raise ValueError("go_boxsqel.manifest is required for NBS training")
    parser_report = spec.get("parser_report")
    contract = load_boxsqel_training_contract(
        _project_path(str(manifest)),
        parser_report_path=(
            None if not parser_report else _project_path(str(parser_report))
        ),
        project_root=Path.cwd(),
        artifact_selection=str(spec.get("artifact_selection", "best")),
        require_artifacts=bool(spec.get("require_artifacts", True)),
    )
    expected = spec.get("expected_embedding_size")
    if expected is not None and int(expected) != contract.embedding_dim:
        raise ValueError(
            f"configured BoxSquaredEL dimension {expected} != manifest {contract.embedding_dim}"
        )
    return contract

def _load_callable(spec: str) -> Callable[[dict[str, Any]], Any]:
    if ":" not in spec:
        raise ValueError("train_loader_factory must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, function_name, None)
    if value is None or not callable(value):
        raise TypeError(f"train loader factory is not callable: {spec}")
    return value


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = config.get("optimizer", {})
    if optimizer_config.get("name", "adamw").lower() != "adamw":
        raise ValueError("the v0.4 reference component factory currently supports AdamW")
    base_lr = float(optimizer_config.get("lr", 1e-4))
    graph_scale_lr = float(optimizer_config.get("graph_delta_scale_lr", 5e-4))
    weight_decay = float(optimizer_config.get("weight_decay", 1e-4))
    graph_scale = model.matcher.graph_delta_scale
    graph_scale_id = id(graph_scale)
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) != graph_scale_id
    ]
    groups = [
        {"params": main_parameters, "lr": base_lr, "weight_decay": weight_decay},
        {"params": [graph_scale], "lr": graph_scale_lr, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups)


def build_components(config: dict[str, Any]) -> NBSRunComponents:
    """Reference fixed-epoch component factory.

    The project-specific ``train_loader_factory`` is responsible for combining
    GO-query episodes with direction-safe local graph materialization.  It must
    return a re-iterable loader of ``NBSLocalBatch`` objects.  Keeping this part
    outside the model package prevents the 281-million-edge graph from being
    accidentally loaded as one PyG object.
    """
    contract = _resolve_boxsqel_contract(config)
    box_spec = dict(config.get("go_boxsqel", {}))
    alignment_value = box_spec.get("alignment_manifest")
    alignment_path = None
    if alignment_value:
        alignment_path = Path(_project_path(str(alignment_value)))
        if bool(box_spec.get("require_alignment_manifest", True)) and not alignment_path.exists():
            raise FileNotFoundError(
                f"aligned GO box manifest not found: {alignment_path}. "
                "Run scripts/nbs/run_prepare_go_boxsqel_for_nbs.py first."
            )
    loader_spec = config.get("train_loader_factory")
    if not loader_spec:
        raise ValueError(
            "Set train_loader_factory='module:function' in the NBS config. "
            "The function receives the full config and must return an iterable "
            "of NBSLocalBatch objects built from LATENCE mmap/CSR stores."
        )
    loader = _load_callable(str(loader_spec))(config)
    model_kwargs = dict(config.get("model_inputs", {}))
    if model_kwargs.get("protein_input_dim") is None:
        raise ValueError("model_inputs must define protein_input_dim")
    configured_box_dim = model_kwargs.get("go_box_dim")
    if configured_box_dim is None:
        model_kwargs["go_box_dim"] = contract.embedding_dim
    elif int(configured_box_dim) != contract.embedding_dim:
        raise ValueError(
            f"model_inputs.go_box_dim={configured_box_dim} != "
            f"BoxSquaredEL embedding_size={contract.embedding_dim}"
        )
    model = ProteinGONBSModel(
        NBSConfig(**config.get("model", {})),
        protein_input_dim=int(model_kwargs["protein_input_dim"]),
        go_box_dim=int(model_kwargs["go_box_dim"]),
    )
    if bool(config.get("stage", {}).get("freeze_go_geometry", True)):
        freeze_go_geometry(model, True)
    optimizer = _build_optimizer(model, config)
    loss_config = NBSLossConfig.from_mapping(config.get("loss", {}))
    return NBSRunComponents(
        model=model,
        optimizer=optimizer,
        train_loader=loader,
        scheduler=None,
        loss_config=loss_config,
        global_go_graph=getattr(loader, "global_go_graph", None),
        metadata={
            "task": config.get("task"),
            "run_tag": config.get("run_tag"),
            "data_epoch": config.get("data_epoch"),
            "selection_policy": "fixed_epoch_snapshots",
            "loader": type(loader).__name__,
            "distributed_runtime": dict(config.get("_distributed_runtime", {})),
            "validation_used": False,
            "early_stopping": False,
            "boxsqel": contract.summary(),
            "go_box_alignment_manifest": (
                None if alignment_path is None else str(alignment_path)
            ),
        },
    )
