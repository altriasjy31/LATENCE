#!/usr/bin/env python3
"""Bind external baseline inputs to an existing LATENCE unseen-GO test cohort.

Uses the manifest's full protein and GO axes, without selecting positive proteins.
Only sequences and optional sequence-verified InterPro features are exported.
GO labels are never copied into the external model input pickle.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import tempfile

import pandas as pd

from build_unseen_go_test_v095 import fasta, normalize_sequence
from eval_unseen_go_v095 import TEST_SCHEMA, checked_ids, read_manifest, sha256


def ipr_items(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if value.startswith(("[", "(", "{")):
            value = ast.literal_eval(value)
        else:
            value = re.split(r"[;,\s]+", value) if value else []
    if not hasattr(value, "__iter__") or isinstance(value, (bytes, dict)):
        raise ValueError(f"Expected a list of InterPro IDs, got {type(value).__name__}")
    values = sorted(set(str(item).strip() for item in value if str(item).strip()))
    invalid = [item for item in values if re.fullmatch(r"IPR\d{6}", item) is None]
    if invalid:
        raise ValueError(f"Not InterPro accession IDs: {invalid[:5]}")
    return values


def sequence_identity(ids, sequences):
    digest = hashlib.sha256()
    for protein in ids:
        digest.update((protein + "\t" + sequences[protein] + "\n").encode("utf-8"))
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--fasta", type=Path, required=True,
                        help="Original inference sequences, NOT a label-only reconstructed axis")
    parser.add_argument("--interpro-metadata", type=Path,
                        help="Trusted pickle DataFrame (or flat column dict) containing InterPro hits and sequences")
    parser.add_argument("--protein-field", default="proteins")
    parser.add_argument("--sequence-field", default="sequences")
    parser.add_argument("--interpro-field", default="interpros")
    parser.add_argument("--feature-provenance", default=None,
                        help="Release/source of optional InterPro features; required when supplying metadata")
    parser.add_argument("--allow-missing-interpro-rows", action="store_true",
                        help="Explicitly retain missing feature rows as [] and record them; never drops proteins")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.interpro_metadata is not None and not args.feature_provenance:
        parser.error("--feature-provenance is required with --interpro-metadata")
    if args.allow_missing_interpro_rows and args.interpro_metadata is None:
        parser.error("--allow-missing-interpro-rows requires --interpro-metadata")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"Choose a new input directory: {args.output_dir}")
    test = read_manifest(args.test_manifest, TEST_SCHEMA)
    if args.cohort not in test["cohorts"]:
        raise ValueError(f"Unknown cohort {args.cohort}; choices: {list(test['cohorts'])}")
    cohort = test["cohorts"][args.cohort]
    ids = checked_ids(cohort["protein_ids"], "full cohort protein axis")
    gos = checked_ids(test["candidate_go_ids"], "candidate GO axis")
    if any(re.fullmatch(r"GO:\d{7}", go) is None for go in gos):
        raise ValueError("Invalid candidate GO ID")
    sequences = fasta(args.fasta, ids)
    missing_seq = sorted(set(ids) - sequences.keys())
    if missing_seq:
        raise ValueError(f"{len(missing_seq)} cohort sequences missing: {missing_seq[:10]}")

    metadata = pd.DataFrame({"proteins": ids, "sequences": [sequences[p] for p in ids]})
    feature_sources, missing_ipr = {}, []
    if args.interpro_metadata is not None:
        # Input pickle is trusted user data. Never load remotely discovered pickles.
        source = pd.read_pickle(args.interpro_metadata)
        if isinstance(source, dict):
            source = pd.DataFrame(source)
        if not isinstance(source, pd.DataFrame):
            raise ValueError("InterPro metadata must be a DataFrame or flat column dictionary")
        for field in (args.protein_field, args.sequence_field, args.interpro_field):
            if field not in source:
                raise ValueError(f"InterPro metadata has no column {field!r}; available: {list(source.columns)}")
        target = source[source[args.protein_field].isin(set(ids))]
        duplicates = target[args.protein_field].duplicated(keep=False)
        if duplicates.any():
            raise ValueError(f"Ambiguous InterPro protein IDs: {target.loc[duplicates, args.protein_field].tolist()[:10]}")
        by_id = target.set_index(args.protein_field, drop=False)
        hits = []
        for protein in ids:
            if protein not in by_id.index:
                missing_ipr.append(protein)
                hits.append([])
                continue
            record = by_id.loc[protein]
            if normalize_sequence(record[args.sequence_field]) != sequences[protein]:
                raise ValueError(f"InterPro sequence mismatch for {protein}; do not join changed sequences by name")
            hits.append(ipr_items(record[args.interpro_field]))
        if missing_ipr and not args.allow_missing_interpro_rows:
            raise ValueError(f"{len(missing_ipr)} proteins absent from InterPro metadata, e.g. {missing_ipr[:10]}. "
                             "Use the full release or explicitly allow missing feature rows.")
        metadata["interpros"] = hits
        feature_sources = {"path": str(args.interpro_metadata.resolve()),
                           "sha256": sha256(args.interpro_metadata),
                           "provenance": args.feature_provenance,
                           "identity_policy": "exact protein ID plus normalized sequence equality",
                           "original_interpro_field": args.interpro_field,
                           "missing_rows_allowed": args.allow_missing_interpro_rows,
                           "missing_row_count": len(missing_ipr),
                           "proteins_with_interpro_hits": sum(bool(x) for x in hits)}

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".external_unseen_input.", dir=args.output_dir.parent) as tmp:
        stage = Path(tmp) / "output"
        stage.mkdir()
        (stage / "protein_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        (stage / "go_ids.txt").write_text("\n".join(gos) + "\n", encoding="utf-8")
        (stage / "sequences.fasta").write_text("".join(f">{p}\n{sequences[p]}\n" for p in ids), encoding="utf-8")
        metadata.to_pickle(stage / "metadata.pkl")
        (stage / "missing_interpro_protein_ids.txt").write_text(
            "".join(p + "\n" for p in missing_ipr), encoding="utf-8")
        manifest = {
            "schema": "external-unseen-go-input-v1", "complete": True,
            "namespace": test["namespace"], "cohort": args.cohort,
            "cohort_role": cohort.get("role"),
            "test_manifest_path": str(args.test_manifest.resolve()),
            "test_manifest_sha256": sha256(args.test_manifest),
            "go_obo_sha256": test.get("sources", {}).get("new_obo", {}).get("sha256"),
            "num_proteins": len(ids), "num_candidate_go": len(gos),
            "num_label_evaluable_proteins": len(cohort["evaluable_protein_ids"]),
            "input_fasta_path": str(args.fasta.resolve()), "input_fasta_sha256": sha256(args.fasta),
            "protein_sequence_axis_sha256": sequence_identity(ids, sequences),
            "files": {name: {"sha256": sha256(stage / name)} for name in
                      ("protein_ids.txt", "go_ids.txt", "sequences.fasta", "metadata.pkl")},
            "interpro_features": feature_sources or None,
            "labels_in_model_inputs": False,
            "protein_pool": "full cohort, including proteins without observed positive labels",
            "sequence_source_claim": "User-supplied original FASTA; identity to prior LATENCE features requires the original prepared FASTA",
        }
        (stage / "external_input_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(args.output_dir)
        stage.rename(args.output_dir)
    print(f"{args.cohort}: {len(ids)} proteins × {len(gos)} candidate GO; model input contains no GO labels")
    print(args.output_dir / "external_input_manifest.json")


if __name__ == "__main__":
    main()
