#!/usr/bin/env python3
"""Append a temporally unseen GO evaluation split to an existing unidata pickle.

The ontology extension fixes the candidate axis. The test annotations are read
after that axis has been validated. Use only trusted project pickle inputs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile

import pandas as pd


SCHEMA = "latence-unseen-go-test-v1"
SPLIT_KEY = "unseen_go_test"
GO_ID = re.compile(r"GO:[0-9]{7}")
CODED_GO = re.compile(r"(GO:[0-9]{7})\|([A-Za-z_]+)\b")
EXPERIMENTAL_CODES = set("EXP IDA IPI IMP IGI IEP HTP HDA HMP HGI HEP".split())
NAMESPACES = {"bp": "biological_process", "mf": "molecular_function", "cc": "cellular_component"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def unique_ids(values, where):
    if isinstance(values, (str, bytes, dict)):
        raise ValueError(f"{where}: expected a list of IDs")
    ids = list(values)
    if not ids or any(not isinstance(x, str) or not x or x != x.strip() for x in ids):
        raise ValueError(f"{where}: empty or invalid ID axis")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{where}: duplicated IDs")
    return ids


def check_artifact(path, record):
    if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"Ontology extension artifact differs from its manifest: {path}")


def extension_axis(directory, obo):
    path = directory / "extension_manifest.json"
    manifest = json_read(path)
    if (manifest.get("schema") != "latence-v095-anchored-new-go-v1"
            or manifest.get("complete") is not True or manifest.get("ontology_only") is not True):
        raise ValueError("Expected a complete ontology-only v095 extension")
    ns = manifest["task"]
    if ns not in NAMESPACES or manifest["namespaces"] != [ns]:
        raise ValueError("The benchmark needs a single-namespace extension matching the v095 task")
    if sha256(obo) != manifest["sources"]["new_obo"]["sha256"]:
        raise ValueError("The GO OBO differs from the OBO used for the extension")
    names = manifest["artifacts"]
    for name in ("new_go_ids.txt", "candidate_go.tsv"):
        check_artifact(directory / name, names[name])
    bundle = directory / "ontology_bundle.pt"
    bundle_verified = bundle.is_file()
    if bundle_verified:
        check_artifact(bundle, names[bundle.name])
    go_ids = unique_ids((directory / "new_go_ids.txt").read_text().splitlines(), "new_go_ids.txt")
    if any(not GO_ID.fullmatch(go) for go in go_ids) or len(go_ids) != manifest["new_rows"]:
        raise ValueError("Invalid new GO list or count")
    with (directory / "candidate_go.tsv").open(encoding="utf-8", newline="") as handle:
        fitted = [r for r in csv.DictReader(handle, delimiter="\t") if r["status"] == "fitted"]
    start = int(manifest["old_rows"])
    rows = [int(r["ontology_row"]) for r in fitted]
    if ([r["go_id"] for r in fitted] != go_ids
            or any(r["namespace"] != ns for r in fitted)
            or rows != list(range(start, start + len(go_ids)))):
        raise ValueError("Fitted GO identities or ontology rows differ from the declared append order")
    return manifest, go_ids, rows, bundle_verified


def go_aliases(path, candidate_ids, namespace, expected_version):
    """Resolve active canonical terms and unique active alt_ids in the supplied OBO."""
    opener = gzip.open if path.suffix == ".gz" else open
    records = {}
    current = None
    version = None
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("["):
                if current and current.get("id"):
                    key = current["id"]
                    if key in records:
                        raise ValueError(f"Duplicated OBO term: {key}")
                    records[key] = current
                current = {"alt_id": []} if line == "[Term]" else None
            elif line.startswith("data-version: "):
                version = line.split(": ", 1)[1]
            elif current is not None:
                key, sep, value = line.partition(": ")
                if sep and value and key == "alt_id":
                    current[key].append(value.split()[0])
                elif sep and value and key in {"id", "namespace", "is_obsolete"}:
                    current[key] = value.split()[0]
    if current and current.get("id"):
        if current["id"] in records:
            raise ValueError(f"Duplicated OBO term: {current['id']}")
        records[current["id"]] = current
    if version != expected_version:
        raise ValueError(f"OBO data-version mismatch: {version!r}, expected {expected_version!r}")
    owners = defaultdict(set)
    for go, term in records.items():
        for key in [go, *term["alt_id"]]:
            owners[key].add(go)
    candidates = set(candidate_ids)
    for go in candidates:
        term = records.get(go, {})
        if (term.get("namespace") != NAMESPACES[namespace]
                or term.get("is_obsolete") == "true" or owners[go] != {go}):
            raise ValueError(f"Candidate is not a unique active OBO term in {namespace}: {go}")
    return {key: go for key, assigned in owners.items()
            if len(assigned) == 1 and (go := next(iter(assigned))) in candidates}


def normalize_sequence(value):
    if not isinstance(value, str):
        return None
    seq = "".join(value.split()).upper().rstrip("*")
    return seq if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+", seq) else None


def fasta(path, protein_ids):
    ids = set(protein_ids)
    result = {}
    header = None
    chunks = []
    opener = gzip.open if path.suffix == ".gz" else open

    def save():
        if header is None:
            return
        hits = ids.intersection([header, *header.split("|")])
        if len(hits) > 1:
            raise ValueError(f"Ambiguous FASTA header on test axis: {header}")
        if hits:
            protein = hits.pop()
            sequence = normalize_sequence("".join(chunks))
            if protein in result or sequence is None:
                raise ValueError(f"Duplicate ID or invalid FASTA sequence: {protein}")
            result[protein] = sequence

    with opener(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                save()
                fields = line[1:].strip().split()
                if not fields:
                    raise ValueError(f"Empty FASTA header: {path}")
                header = fields[0]
                chunks = []
            elif line:
                if header is None:
                    raise ValueError(f"FASTA sequence precedes its header: {path}")
                chunks.append(line)
    save()
    if not result:
        raise ValueError(f"No FASTA IDs match the independent-test protein axis: {path}")
    return result


def independent_block(metadata, namespace):
    split = metadata.get("ind_test")
    if not isinstance(split, dict):
        raise ValueError("The original metadata has no ind_test mapping")
    for key in (NAMESPACES[namespace], namespace, namespace.upper()):
        block = split.get(key)
        if isinstance(block, dict) and "proteins" in block:
            return block
    raise ValueError(f"The original metadata has no {namespace} ind_test protein axis")


def expected_sequences(block, ids, fasta_path):
    raw = block.get("sequences", block.get("sequence"))
    if isinstance(raw, dict):
        values = {p: normalize_sequence(raw.get(p)) for p in ids}
    elif raw is not None and not isinstance(raw, (str, bytes)) and hasattr(raw, "__len__"):
        if len(raw) != len(ids):
            raise ValueError("Independent-test metadata has inconsistent sequence/protein counts")
        values = {p: normalize_sequence(seq) for p, seq in zip(ids, raw)}
    else:
        values = {}
    if fasta_path is not None:
        from_fasta = fasta(fasta_path, ids)
        for protein, sequence in from_fasta.items():
            if values.get(protein) not in (None, sequence):
                raise ValueError(f"Original metadata and FASTA disagree for {protein}")
        values.update(from_fasta)
    return values


def annotation_items(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith(("[", "(", "{")):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
        else:
            return [value] if value else []
    if hasattr(value, "__iter__") and not isinstance(value, (bytes, dict)):
        return [str(x) for x in value if x is not None]
    return []


def evidence(value, aliases):
    hits = defaultdict(lambda: defaultdict(set))
    coded = False
    for item in annotation_items(value):
        if re.search(r"\bNOT\b", item, flags=re.I):
            continue
        for raw_go, code in CODED_GO.findall(item):
            coded = True
            code = code.upper()
            if raw_go in aliases and code in EXPERIMENTAL_CODES:
                hits[aliases[raw_go]][raw_go].add(code)
    return hits, coded


def make_cohort(ids, audit, positives, role, identity_policy, input_sha=None):
    eligible = [item["protein_id"] for item in audit if item["evaluable"]]
    result = {
        "protein_ids": ids,
        "evaluable_protein_ids": eligible,
        "positives": positives,
        "role": role,
        "identity_policy": identity_policy,
        "identity_status_counts": dict(Counter(item["identity_status"] for item in audit)),
        "n_positive_proteins": len({item["protein_id"] for item in positives}),
        "n_positive_go_ids": len({item["go_id"] for item in positives}),
    }
    if input_sha is not None:
        result["input_manifest_sha256"] = input_sha
    return result


def bind_input(directory, ids, namespace):
    manifest_path = directory / "ind_test_input_manifest.json"
    manifest = json_read(manifest_path)
    proteins = directory / "protein_ids.txt"
    actual = unique_ids(proteins.read_text().splitlines(), str(proteins))
    if (set(actual) != set(ids) or manifest["num_proteins"] != len(actual)
            or manifest["task"] != namespace):
        raise ValueError(f"Prepared input IDs or task differ from this cohort: {directory}")
    digest = manifest.get("protein_ids_file_sha256")
    if digest is not None and sha256(proteins) != digest:
        raise ValueError(f"Prepared protein order hash differs from its manifest: {directory}")
    return sha256(manifest_path)


def ind_test_cohort(block, swiss, aliases, fasta_path, require_sequence, input_dir, namespace):
    ids = unique_ids(block["proteins"], "ind_test proteins")
    wanted = set(ids)
    names, accessions = defaultdict(list), defaultdict(set)
    for row, protein in enumerate(swiss["proteins"]):
        if protein in wanted:
            names[protein].append(row)
    fallback = wanted.difference(names)
    if fallback and "accessions" in swiss:
        for row, value in enumerate(swiss["accessions"]):
            if isinstance(value, str):
                for key in fallback.intersection(map(str.strip, value.split(";"))):
                    accessions[key].add(row)

    sequences = expected_sequences(block, ids, fasta_path)
    audit, positives = [], []
    n_coded = 0
    for protein in ids:
        name_rows = names.get(protein)
        candidates = name_rows or accessions.get(protein, ())
        method = "entry_name" if name_rows else "accession" if candidates else "unmatched"
        match = {"protein_id": protein, "match_method": method, "record_2025": "",
                 "identity_status": "ambiguous" if candidates else "unmatched", "evaluable": False}
        audit.append(match)
        if len(candidates) != 1:
            continue
        row = swiss.iloc[next(iter(candidates))]
        expected = sequences.get(protein)
        actual = normalize_sequence(row["sequences"])
        status = "sequence_unverified"
        if actual is None:
            status = "invalid_2025_sequence"
        elif expected is not None:
            status = "identical" if expected == actual else "sequence_mismatch"
        eligible = status == "identical" or (status == "sequence_unverified" and not require_sequence)
        match.update(record_2025=str(row["proteins"]), identity_status=status, evaluable=eligible)
        if not eligible:
            continue
        found, coded = evidence(row["annotations"], aliases)
        n_coded += int(coded)
        for go, raw_codes in sorted(found.items()):
            codes = {code for values in raw_codes.values() for code in values}
            positives.append({"protein_id": protein, "go_id": go, "evidence_codes": sorted(codes),
                              "raw_go_ids": sorted(raw_codes), "record_2025": str(row["proteins"])})
    if not n_coded and any(row["evaluable"] for row in audit):
        raise ValueError("No GO|CODE records found in the matched 2025 annotations")
    policy = ("exact_sequence_required" if require_sequence else
              "unique_entry_name_or_accession_with_sequence_check_when_available")
    input_sha = bind_input(input_dir, ids, namespace) if input_dir is not None else None
    cohort = make_cohort(ids, audit, positives, "primary_independent", policy, input_sha)
    return cohort, audit


def strict_flags(values):
    tokens = values.astype(str).str.strip().str.lower()
    true, false = {"true", "1", "1.0"}, {"false", "0", "0.0"}
    if not tokens.isin(true | false).all():
        raise ValueError(f"{values.name}: invalid boolean flags")
    return tokens.isin(true)


def dark_weak_cohort(future_path, index_path, swiss, aliases, input_dir, namespace):
    future = pd.read_csv(future_path, sep="\t", dtype=str, keep_default_na=False)
    index = pd.read_csv(index_path, sep="\t", dtype=str, keep_default_na=False)
    for frame in (future, index):
        if not frame["rid22"].str.fullmatch(r"[0-9]+").all():
            raise ValueError("Invalid rid22 in the dark-weak audit tables")
        frame["rid22"] = frame["rid22"].astype(int)
    if index["rid22"].duplicated().any() or index["protein_22"].duplicated().any():
        raise ValueError("Ambiguous historical dark-weak match index")
    strict = future.loc[strict_flags(future["strict_eligible"])].copy()
    if strict.empty or not strict["match_status"].eq("identical").all():
        raise ValueError("Expected strict identical-sequence rows in future_go_pairs.tsv")
    selected = strict[["rid22", "protein_22", "protein_25"]].drop_duplicates().sort_values("rid22")
    ids = unique_ids(selected["protein_22"], "dark-weak strict audited protein IDs")
    index = index.set_index("rid22")
    audit, raw_by_protein = [], {}
    for item in selected.itertuples(index=False):
        hit = index.loc[item.rid22]
        if (hit["protein_22"] != item.protein_22 or hit["match_status"] != "identical"
                or not re.fullmatch(r"[0-9]+(?:\.0+)?", hit["rid25"])):
            raise ValueError(f"Dark-weak audit identity mismatch: {item.protein_22}")
        row_id = int(hit["rid25"].split(".", 1)[0])
        if row_id >= len(swiss):
            raise ValueError(f"2025 positional row outside the original Swiss-Prot table: {row_id}")
        row = swiss.iloc[row_id]
        seq = normalize_sequence(row["sequences"])
        digest = hashlib.sha256(seq.encode("ascii")).hexdigest() if seq else None
        if row["proteins"] != item.protein_25 or digest != hit["sequence_sha256"]:
            raise ValueError(f"Dark-weak original 2025 pickle order or sequence differs: {item.protein_22}")
        raw_by_protein[item.protein_22] = evidence(row["annotations"], aliases)[0]
        audit.append({"protein_id": item.protein_22, "record_2025": item.protein_25,
                      "identity_status": "identical", "match_method": "rid25_and_sequence_sha256",
                      "evaluable": True})
    grouped = defaultdict(lambda: [set(), set()])
    for item in strict.loc[strict_flags(strict["is_strict_experimental"])].itertuples(index=False):
        go = aliases.get(item.go_id_2025)
        if go is None:
            continue
        raw_hits = raw_by_protein[item.protein_22]
        declared = set(item.evidence_codes_2025.split(";")) & EXPERIMENTAL_CODES
        confirmed = raw_hits.get(go, {}).get(item.go_id_2025, set()) & declared
        if not confirmed:
            raise ValueError(f"Dark-weak direct experimental evidence differs: {item.protein_22}, {go}")
        codes, raw_ids = grouped[(item.protein_22, go)]
        codes.update(confirmed)
        raw_ids.add(item.go_id_2025)
    by_id = {item["protein_id"]: item["record_2025"] for item in audit}
    order = {p: i for i, p in enumerate(ids)}
    positives = [{"protein_id": protein, "go_id": go, "evidence_codes": sorted(codes),
                  "raw_go_ids": sorted(raw_ids), "record_2025": by_id[protein]}
                 for (protein, go), (codes, raw_ids) in sorted(
                     grouped.items(), key=lambda item: (order[item[0][0]], item[0][1]))]
    input_sha = bind_input(input_dir, ids, namespace) if input_dir is not None else None
    policy = "strict_future_annotation_rows_with_verified_2025_positional_index_and_sequence_hash"
    return make_cohort(ids, audit, positives, "supplementary_dark_weak", policy, input_sha), audit


def write_result(out, metadata, manifest, audits):
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as work:
        stage = Path(work) / "result"
        stage.mkdir()
        merged = dict(metadata)
        merged[SPLIT_KEY] = manifest
        with (stage / "unidata_with_dark_weak_unseen_go.pkl").open("wb") as handle:
            pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
        json_write(stage / "test_manifest.json", manifest)
        for name, cohort in manifest["cohorts"].items():
            (stage / f"{name}_protein_ids.txt").write_text(
                "\n".join(cohort["protein_ids"]) + "\n", encoding="utf-8")
            audit_fields = ["protein_id", "record_2025", "identity_status", "match_method", "evaluable"]
            with (stage / f"{name}_identity_audit.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=audit_fields, delimiter="\t")
                writer.writeheader()
                writer.writerows(audits[name])
            positive_fields = ["protein_id", "go_id", "evidence_codes", "raw_go_ids", "record_2025"]
            with (stage / f"{name}_positives.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=positive_fields, delimiter="\t")
                writer.writeheader()
                for row in cohort["positives"]:
                    writer.writerow({**row, "evidence_codes": ";".join(row["evidence_codes"]),
                                     "raw_go_ids": ";".join(row["raw_go_ids"])})
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"Output directory already exists: {out}")
        os.rename(stage, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Original unidata pickle")
    parser.add_argument("--swiss25", type=Path, required=True, help="Original swissprot_2025_03.pkl")
    parser.add_argument("--new-obo", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--ind-test-input-dir", type=Path, help="Existing v095 inference input cache")
    parser.add_argument("--ind-test-fasta", type=Path, help="Original independent-test sequences")
    parser.add_argument("--require-sequence-match", action="store_true")
    parser.add_argument("--future-pairs", type=Path, help="Optional dark-weak future_go_pairs.tsv")
    parser.add_argument("--match-index", type=Path, help="Optional dark-weak cohort_match_index.tsv")
    parser.add_argument("--dark-weak-input-dir", type=Path, help="Optional prepared dark-weak inference cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.future_pairs) != bool(args.match_index) or (args.dark_weak_input_dir and not args.future_pairs):
        parser.error("Use --future-pairs and --match-index together for the optional dark-weak cohort")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    extension, go_ids, rows, bundle_verified = extension_axis(args.extension_dir, args.new_obo)
    namespace = extension["task"]
    aliases = go_aliases(args.new_obo, go_ids, namespace, extension["new_obo_version"])
    with args.metadata.open("rb") as handle:
        metadata = pickle.load(handle)
    if not isinstance(metadata, dict) or SPLIT_KEY in metadata:
        raise ValueError(f"Expected original metadata dict without an existing {SPLIT_KEY!r} key")
    swiss = pd.read_pickle(args.swiss25)
    required = {"proteins", "sequences", "annotations"}
    if not isinstance(swiss, pd.DataFrame) or not required.issubset(swiss.columns):
        raise ValueError(f"Expected 2025 Swiss-Prot columns: {sorted(required)}")
    ind_fasta = args.ind_test_fasta
    if ind_fasta is None and args.ind_test_input_dir:
        prepared_fasta = args.ind_test_input_dir / "normalized_sequences.fasta"
        if prepared_fasta.is_file():
            ind_fasta = prepared_fasta
    if ind_fasta is not None and ind_fasta.resolve() == args.swiss25.resolve():
        raise ValueError("Independent-test sequences must come from an independent FASTA")
    cohorts, audits = {}, {}
    cohorts["ind_test_exp"], audits["ind_test_exp"] = ind_test_cohort(
        independent_block(metadata, namespace), swiss, aliases, ind_fasta,
        args.require_sequence_match, args.ind_test_input_dir, namespace)
    if args.future_pairs:
        cohorts["dark_weak_future_exp"], audits["dark_weak_future_exp"] = dark_weak_cohort(
            args.future_pairs, args.match_index, swiss, aliases, args.dark_weak_input_dir, namespace)
    paths = {"metadata": args.metadata, "swiss25": args.swiss25, "new_obo": args.new_obo,
             "extension_manifest": args.extension_dir / "extension_manifest.json",
             "ind_test_fasta": ind_fasta, "future_pairs": args.future_pairs,
             "match_index": args.match_index}
    manifest = {
        "schema": SCHEMA, "complete": True, "namespace": namespace,
        "old_obo_version": extension["old_obo_version"], "new_obo_version": extension["new_obo_version"],
        "candidate_go_ids": go_ids, "candidate_ontology_rows": rows,
        "ontology_bundle_sha256": extension["artifacts"]["ontology_bundle.pt"]["sha256"],
        "ontology_bundle_file_verified_at_build": bundle_verified,
        "v095_resolved_config_sha256": extension["sources"]["v095_bindings"]["v095_config"]["sha256"],
        "cohorts": cohorts, "strict_experimental_codes": sorted(EXPERIMENTAL_CODES),
        "ground_truth_policy": "direct experimental GO, active canonical or unique alt_id; unobserved pairs are unlabeled",
        "stage1_exposure_audited": False,
        "sources": {key: {"path": str(path.resolve()), "sha256": sha256(path)}
                    for key, path in paths.items() if path is not None},
    }
    write_result(args.output_dir, metadata, manifest, audits)
    for name, item in cohorts.items():
        print(f"[unseen GO] {name}: {len(item['evaluable_protein_ids'])}/{len(item['protein_ids'])} eligible, "
              f"{item['n_positive_proteins']} proteins, {item['n_positive_go_ids']} GO, "
              f"{len(item['positives'])} pairs; {item['identity_status_counts']}")
    print(f"[unseen GO] {len(go_ids)} candidates; {args.output_dir / 'test_manifest.json'}")


if __name__ == "__main__":
    main()
