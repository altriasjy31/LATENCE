#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hetero_k_hop_closure_v2 import __version__, k_hop_closure_hetero


def main() -> None:
    try:
        import dgl
        import torch
    except ImportError:
        print("hetero_k_hop_closure_v2 regression: SKIP (PyTorch/DGL unavailable)")
        return

    assert __version__ == "2.1.0-weak-to-core-closure"

    # protein 0/1 are weak, protein 2/3 are core.
    # Intended paths:
    #   weak-0 -> core-2 -> GO-0
    #   weak-1 -> core-3 -> GO-1
    graph = dgl.heterograph(
        {
            ("protein", "weak_to_core", "protein"): (
                torch.tensor([0, 1]),
                torch.tensor([2, 3]),
            ),
            ("protein", "annotated_with", "go"): (
                torch.tensor([2, 3]),
                torch.tensor([0, 1]),
            ),
            ("go", "annotates", "protein"): (
                torch.tensor([0, 1]),
                torch.tensor([2, 3]),
            ),
        },
        num_nodes_dict={"protein": 4, "go": 2},
    )

    closure = k_hop_closure_hetero(
        graph,
        {"go": torch.tensor([0])},
        k=2,
        fanout=-1,
        edge_dir="in",
        allowed_etypes=["annotated_with", "weak_to_core"],
    )
    assert set(closure["go"].tolist()) == {0}
    assert set(closure["protein"].tolist()) == {0, 2}

    one_hop = k_hop_closure_hetero(
        graph,
        {"go": torch.tensor([0])},
        k=1,
        fanout=-1,
        edge_dir="in",
        allowed_etypes=["annotated_with", "weak_to_core"],
    )
    assert set(one_hop["protein"].tolist()) == {2}

    forward = k_hop_closure_hetero(
        graph,
        {"protein": torch.tensor([1])},
        k=2,
        fanout=-1,
        edge_dir="out",
        allowed_etypes=["weak_to_core", "annotated_with"],
    )
    assert set(forward["protein"].tolist()) == {1, 3}
    assert set(forward["go"].tolist()) == {1}

    print("hetero_k_hop_closure_v2 regression: PASS")


if __name__ == "__main__":
    main()