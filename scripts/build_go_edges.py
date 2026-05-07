#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build GO hierarchy edge tensors for cc / mf / bp.

Output files:
    out_dir/go_edges_cc.pt
    out_dir/go_edges_mf.pt
    out_dir/go_edges_bp.pt

Also output canonical GO term order:
    out_dir/go_terms_cc.txt
    out_dir/go_terms_mf.txt
    out_dir/go_terms_bp.txt

Each .pt file is a torch.LongTensor with shape [E, 2].
Each row is:
    [child_index, parent_index]

These indices must match the column order of logits.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------
# Make project_root importable when this script is executed from scripts/
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from msa_models.helper_functions.obo_parser import (  # noqa: E402
    ONTO_TYPES,
    save_go_edges_for_all_namespaces,
)


def project_path(path):
    """
    Resolve relative paths against project_root.

    Example:
        data/go.obo -> project_root/data/go.obo
    """
    if path is None:
        return None

    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build GO hierarchy edge tensors for cc/mf/bp."
    )

    parser.add_argument(
        "--obo",
        type=str,
        default="data/go.obo",
        help="Path to go.obo. Relative paths are resolved from project_root. "
             "Default: data/go.obo",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/go_edges",
        help="Output directory for go_edges_*.pt and go_terms_*.txt. "
             "Relative paths are resolved from project_root. "
             "Default: data/go_edges",
    )

    parser.add_argument(
        "--label-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing label order files. "
            "The script will look for terms_cc.txt / terms_mf.txt / terms_bp.txt. "
            "If absent, it will also accept go_terms_cc.txt / go_terms_mf.txt / go_terms_bp.txt. "
            "Do not use together with --label-cc/--label-mf/--label-bp."
        ),
    )

    parser.add_argument(
        "--label-cc",
        type=str,
        default=None,
        help="Optional label-order file for cellular component.",
    )

    parser.add_argument(
        "--label-mf",
        type=str,
        default=None,
        help="Optional label-order file for molecular function.",
    )

    parser.add_argument(
        "--label-bp",
        type=str,
        default=None,
        help="Optional label-order file for biological process.",
    )

    parser.add_argument(
        "--transitive",
        action="store_true",
        help=(
            "Use child -> all ancestors edges instead of only direct child -> parent edges. "
            "Recommended when your label set does not contain all intermediate GO parents."
        ),
    )

    parser.add_argument(
        "--with-rels",
        action="store_true",
        help=(
            "Pass with_rels=True to Ontology. "
            "Warning: in the current parser this treats all OBO relationships as parent edges. "
            "Usually you should leave this disabled."
        ),
    )

    parser.add_argument(
        "--include-root",
        action="store_true",
        help=(
            "Only used when no label files are provided. "
            "If set, include GO root terms GO:0005575 / GO:0003674 / GO:0008150."
        ),
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation of saved edge tensors.",
    )

    return parser.parse_args()


def collect_label_files(args):
    """
    Decide whether to use external label files.

    Return:
        None
            means automatically generate label terms from the OBO.

        dict
            {
                "cc": Path(...),
                "mf": Path(...),
                "bp": Path(...),
            }
    """
    specific_labels = {
        "cc": args.label_cc,
        "mf": args.label_mf,
        "bp": args.label_bp,
    }

    has_label_dir = args.label_dir is not None
    num_specific = sum(v is not None for v in specific_labels.values())

    if has_label_dir and num_specific > 0:
        raise ValueError(
            "Use either --label-dir or --label-cc/--label-mf/--label-bp, not both."
        )

    if has_label_dir:
        label_dir = project_path(args.label_dir)

        label_files = {}

        for ont_type in ONTO_TYPES:
            candidates = [
                label_dir / f"terms_{ont_type}.txt",
                label_dir / f"go_terms_{ont_type}.txt",
            ]

            found = [p for p in candidates if p.exists()]

            if len(found) == 0:
                raise FileNotFoundError(
                    f"Cannot find label file for {ont_type} in {label_dir}. "
                    f"Tried: {', '.join(str(p) for p in candidates)}"
                )

            # Prefer terms_xx.txt if both exist.
            label_files[ont_type] = found[0]

        return label_files

    if num_specific > 0:
        missing = [k for k, v in specific_labels.items() if v is None]
        if missing:
            raise ValueError(
                "If using explicit label files, all three must be provided. "
                f"Missing: {missing}"
            )

        return {
            "cc": project_path(args.label_cc),
            "mf": project_path(args.label_mf),
            "bp": project_path(args.label_bp),
        }

    # No external label order.
    # The helper will generate full namespace term lists from the OBO.
    return None


def validate_label_files(label_files):
    if label_files is None:
        return

    for ont_type, path in label_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Label file for {ont_type} does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Label path for {ont_type} is not a file: {path}")


def load_tensor_safely(path):
    """
    Compatibility wrapper for torch.load.

    Newer PyTorch supports weights_only=True.
    Older PyTorch may not.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_outputs(out_dir, stats_by_type):
    """
    Validate that saved edge tensors are compatible with:

        load_go_edges(path, device)

    Expected:
        tensor dtype: torch.long
        tensor shape: [E, 2]
        each row: [child_index, parent_index]
    """
    for ont_type in ONTO_TYPES:
        edge_path = out_dir / f"go_edges_{ont_type}.pt"

        if not edge_path.exists():
            raise FileNotFoundError(f"Missing output edge file: {edge_path}")

        edges = load_tensor_safely(edge_path)

        if not isinstance(edges, torch.Tensor):
            raise TypeError(f"{edge_path} is not a torch.Tensor.")

        if edges.dtype != torch.long:
            raise TypeError(f"{edge_path} dtype must be torch.long, got {edges.dtype}.")

        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(
                f"{edge_path} must have shape [E, 2], got {tuple(edges.shape)}."
            )

        num_labels = stats_by_type[ont_type]["num_labels"]

        if edges.numel() > 0:
            min_idx = int(edges.min().item())
            max_idx = int(edges.max().item())

            if min_idx < 0:
                raise ValueError(f"{edge_path} contains negative index: {min_idx}")

            if max_idx >= num_labels:
                raise ValueError(
                    f"{edge_path} contains index {max_idx}, "
                    f"but num_labels = {num_labels}."
                )


def print_stats(stats_by_type):
    print("\nGO edge construction summary")
    print("=" * 72)

    for ont_type in ONTO_TYPES:
        s = stats_by_type[ont_type]

        print(f"\n[{ont_type}]")
        print(f"  num_labels                 : {s['num_labels']}")
        print(f"  num_edges                  : {s['num_edges']}")
        print(f"  parent_links_seen          : {s['parent_links_seen']}")
        print(f"  skipped_not_in_labels      : {s['skipped_not_in_labels']}")
        print(f"  skipped_cross_namespace    : {s['skipped_cross_namespace']}")
        print(f"  skipped_missing_or_obsolete: {s['skipped_missing_or_obsolete']}")
        print(f"  transitive                 : {bool(s['transitive'])}")
        print(f"  edge_path                  : {s['edge_path']}")
        print(f"  term_path                  : {s['term_path']}")

        if s["num_edges"] == 0:
            print(
                "  WARNING: num_edges == 0. "
                "The hierarchy loss for this ontology will be zero."
            )


def main():
    args = parse_args()

    obo_path = project_path(args.obo)
    out_dir = project_path(args.out_dir)

    if not obo_path.exists():
        raise FileNotFoundError(f"go.obo not found: {obo_path}")

    if not obo_path.is_file():
        raise ValueError(f"--obo is not a file: {obo_path}")

    label_files = collect_label_files(args)
    validate_label_files(label_files)

    out_dir.mkdir(parents=True, exist_ok=True)

    stats_by_type = save_go_edges_for_all_namespaces(
        obo_path=obo_path,
        out_dir=out_dir,
        label_files=label_files,
        label_terms=None,
        with_rels=args.with_rels,
        transitive=args.transitive,
        include_root=args.include_root,
    )

    if not args.no_validate:
        validate_outputs(out_dir, stats_by_type)

    stats_path = out_dir / "go_edges_stats.json"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_by_type, f, indent=2, ensure_ascii=False)

    print_stats(stats_by_type)

    print("\nSaved stats:")
    print(f"  {stats_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()