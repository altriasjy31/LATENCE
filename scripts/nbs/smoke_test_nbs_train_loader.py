#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _load_callable(spec: str) -> Callable[[dict[str, Any]], Any]:
    if ":" not in spec:
        raise ValueError("factory must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"not callable: {spec}")
    return function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize and audit one real LATENCE NBS local batch"
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "nbs_models" / "nbs_protein_go"))
    from nbs_pg.schema import (  # pylint: disable=import-outside-toplevel
        BACKBONE_CANDIDATE_GO_TO_PROTEIN,
        BACKBONE_CANDIDATE_PROTEIN_TO_GO,
        SIMILAR_TO,
        WEAK_TO_CORE,
    )

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_distributed_runtime"] = {
        "enabled": False,
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "backend": "none",
    }
    loader = _load_callable(config["train_loader_factory"])(config)
    loader.set_epoch(args.epoch)
    batch = next(iter(loader))
    graph = batch.graph
    graph.validate(raise_on_error=True)
    protein_global = np.asarray(graph["protein"].node_id.cpu(), dtype=np.int64)
    registry = loader.materializer.stores.registry

    def edge_count(edge_type: tuple[str, str, str]) -> int:
        return int(graph[edge_type].edge_index.shape[1])

    failures: list[str] = []
    weak = graph[WEAK_TO_CORE].edge_index.cpu().numpy()
    if weak.size:
        weak_source = protein_global[weak[0]]
        weak_destination = protein_global[weak[1]]
        if any(registry.role_of(int(value)) != "weak" for value in weak_source):
            failures.append("weak_to_core contains a non-weak message source")
        if any(registry.role_of(int(value)) != "core" for value in weak_destination):
            failures.append("weak_to_core contains a non-core message destination")
    similar = graph[SIMILAR_TO].edge_index.cpu().numpy()
    if similar.size:
        endpoints = protein_global[similar.reshape(-1)]
        if any(registry.role_of(int(value)) != "core" for value in endpoints):
            failures.append("similar_to contains a non-core endpoint")

    forward = graph[BACKBONE_CANDIDATE_PROTEIN_TO_GO].edge_index
    reverse = graph[BACKBONE_CANDIDATE_GO_TO_PROTEIN].edge_index
    candidates = batch.query.candidate_protein_index.cpu()
    queries = batch.query.query_go_index.cpu()
    for protein in candidates.tolist():
        for go in queries.tolist():
            if bool(((forward[0] == protein) & (forward[1] == go)).any()):
                failures.append("candidate query edge remains in forward relation")
                break
            if bool(((reverse[1] == protein) & (reverse[0] == go)).any()):
                failures.append("candidate query edge remains in reverse relation")
                break

    report = {
        "schema_version": 1,
        "config": str(config_path),
        "epoch": int(args.epoch),
        "query_shape": list(batch.query.base_logits.shape),
        "local_proteins": int(graph["protein"].num_nodes),
        "local_go_classes": int(graph["go"].num_nodes),
        "edge_counts": {
            "|".join(edge_type): edge_count(edge_type)
            for edge_type in graph.edge_types
        },
        "metadata": dict(batch.metadata),
        "full_go_cache_nodes": int(loader.global_go_graph["go"].num_nodes),
        "direction_safe": not failures,
        "failures": failures,
        "passed": not failures,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
