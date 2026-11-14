#!/usr/bin/env python3
"""Inventory unseen-GO evidence across complete protein axes before inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
from pathlib import Path
import pickle
import re
import tempfile

import pandas as pd

from build_unseen_go_test_v095 import (
    CODED_GO, EXPERIMENTAL_CODES, GO_ID, SCHEMA, SPLIT_KEY, annotation_items, bind_input,
    evidence, go_aliases, json_read, json_write, make_cohort, normalize_sequence, sha256, unique_ids,
)


IPR_ID = re.compile(r"\bIPR[0-9]{6,}\b")
MAPPING_ROW = re.compile(r"InterPro:(IPR[0-9]{6,})\b.*?;\s*(GO:[0-9]{7})")
DARK_AXIS = "dark_weak_population"
SWISS_AXIS = "swissprot_2025_population"


def source_ref(path, digest=None):
    return {"path": str(path.resolve()), "sha256": digest or sha256(path)}


def load_swiss(path, *, historical=False):
    table = pd.read_pickle(path)
    required = {"proteins", "sequences", "interpros", "accessions"}
    if not historical:
        required.add("annotations")
    if not isinstance(table, pd.DataFrame) or not required.issubset(table.columns):
        raise ValueError(f"{path}: expected Swiss-Prot columns {sorted(required)}")
    unique_ids(table["proteins"], str(path))
    if any(any(char.isspace() for char in protein) for protein in table["proteins"]):
        raise ValueError(f"{path}: protein IDs may not contain whitespace")
    return table.reset_index(drop=True)


def read_mapping(path, aliases):
    opener = gzip.open if path.suffix == ".gz" else open
    raw_pairs = set()
    mapping = defaultdict(lambda: defaultdict(set))
    version = None
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if line.lower().startswith("!version date:"):
                version = line.split(":", 1)[1].strip()
            if not line or line.startswith(("!", "#")):
                continue
            match = MAPPING_ROW.fullmatch(line)
            if match is None:
                raise ValueError(f"{path}:{number}: malformed InterPro2GO row")
            ipr, raw_go = match.groups()
            raw_pairs.add((ipr, raw_go))
            if raw_go in aliases:
                mapping[ipr][aliases[raw_go]].add(raw_go)
    if not raw_pairs:
        raise ValueError(f"{path}: no InterPro2GO data rows")
    meta = {
        **source_ref(path), "version_date": version, "direct_mapping_pairs": len(raw_pairs),
        "mapped_interpro_ids": len(mapping),
        "mapped_candidate_go": len({go for by_go in mapping.values() for go in by_go}),
        "candidate_mapping_pairs": sum(map(len, mapping.values())),
    }
    return mapping, meta


def ipr_hits(value):
    return set(IPR_ID.findall(str(value)))


def accessions(value):
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    return []


def sequence_axis(frame, label):
    axis = dict(zip(frame["proteins"], map(normalize_sequence, frame["sequences"])))
    invalid = [protein for protein, sequence in axis.items() if sequence is None]
    if invalid:
        raise ValueError(f"{label}: {len(invalid)} invalid sequences on the full inference axis: {invalid[:10]}")
    return axis


def match_index(path, old):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"rid22", "rid25", "protein_22", "match_status", "sequence_sha256"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: expected fields {sorted(required)}")
        records = list(reader)
    if any(re.fullmatch(r"[0-9]+", row["rid22"] or "") is None for row in records):
        raise ValueError(f"{path}: invalid rid22")
    records.sort(key=lambda row: int(row["rid22"]))
    if [int(row["rid22"]) for row in records] != list(range(len(records))):
        raise ValueError(f"{path}: rid22 must cover the complete ordered historical axis")
    names = unique_ids([row["protein_22"] for row in records], str(path))
    old = old.set_index("proteins", drop=False)
    missing = set(names).difference(old.index)
    if missing:
        raise ValueError(f"{len(missing)} indexed dark-weak proteins absent in the old pickle: {sorted(missing)[:10]}")
    old = old.loc[names].reset_index(drop=True)
    sequences = sequence_axis(old, "historical dark-weak")
    for row in records:
        protein = row["protein_22"]
        digest = hashlib.sha256(sequences[protein].encode("ascii")).hexdigest()
        if row["sequence_sha256"] != digest:
            raise ValueError(f"Original 2022 dark-weak sequence SHA256 differs for {protein}")
    return records, old, sequences


def same_identity(old_name, old_accessions, new_name, new_accessions):
    return old_name == new_name or bool(set(old_accessions).intersection(new_accessions))


def verified_indexed_matches(records, old, indexed, sequences):
    matched = {}
    sequence_verified = {}
    for row, accession_text in zip(records, old["accessions"]):
        if row["match_status"] != "identical":
            continue
        protein = row["protein_22"]
        rid25 = row["rid25"] or ""
        if re.fullmatch(r"[0-9]+(?:\.0+)?", rid25) is None:
            raise ValueError(f"Identical record has an invalid rid25 for {protein}")
        position = int(rid25.split(".", 1)[0])
        if position >= len(indexed):
            raise ValueError(f"rid25 is out of range for {protein}")
        hit = indexed.iloc[position]
        if (normalize_sequence(hit["sequences"]) != sequences[protein]
                or row.get("protein_25") not in (None, "", hit["proteins"])):
            raise ValueError(f"Original 2025 positional table or sequence differs for {protein}")
        if (row.get("entry_name_changed") or "").lower() in {"0", "0.0", "false"} and protein != hit["proteins"]:
            raise ValueError(f"Original 2025 positional table has a different entry name for {protein}")
        sequence_verified[protein] = position
        if same_identity(protein, accessions(accession_text), hit["proteins"], accessions(hit["accessions"])):
            matched[protein] = position
        elif row.get("match_method") not in (None, "", "sequence_only"):
            raise ValueError(f"Original 2025 positional table has no shared identity for {protein}")
    return matched, sequence_verified


def full_release_matches(old, indexed, full, verified, sequences):
    """Keep verified indexed matches and recover exact name/accession matches in the full release."""
    new_positions = dict(zip(full["proteins"], range(len(full))))
    omitted = set(indexed["proteins"]).difference(new_positions)
    if omitted:
        raise ValueError(f"The full 2025 release omits {len(omitted)} indexed entries: {sorted(omitted)[:10]}")
    for name, sequence in zip(indexed["proteins"], indexed["sequences"]):
        if normalize_sequence(sequence) != normalize_sequence(full["sequences"].iat[new_positions[name]]):
            raise ValueError(f"Indexed and full 2025 sequences differ for {name}")
    old_names = list(old["proteins"])
    old_accessions = {p: accessions(value) for p, value in zip(old_names, old["accessions"])}
    wanted_ac = {token for values in old_accessions.values() for token in values}
    by_accession = defaultdict(list)
    for pos, value in enumerate(full["accessions"]):
        for token in wanted_ac.intersection(accessions(value)):
            by_accession[token].append(pos)
    matched = {}
    methods = {}
    for protein in old_names:
        tokens = old_accessions[protein]
        if protein in verified:
            indexed_name = indexed["proteins"].iat[verified[protein]]
            pos = new_positions[indexed_name]
            matched[protein], methods[protein] = pos, "indexed_exact_in_full_release"
            continue
        options = {pos for token in tokens for pos in by_accession[token]}
        if protein in new_positions:
            options.add(new_positions[protein])
        scores = {}
        for pos in options:
            if normalize_sequence(full["sequences"].iat[pos]) != sequences[protein]:
                continue
            target = accessions(full["accessions"].iat[pos])
            scores[pos] = (
                full["proteins"].iat[pos] == protein,
                bool(tokens and target and tokens[0] == target[0]),
                len(set(tokens).intersection(target)),
            )
        if not scores:
            continue
        best = max(scores.values())
        choices = [pos for pos, score in scores.items() if score == best]
        if len(choices) == 1:
            matched[protein], methods[protein] = choices[0], "full_release_name_or_accession_exact"
        else:
            methods[protein] = "ambiguous_full_release_exact_match"
    return matched, methods


def experimental_pairs(protein, target, value, aliases):
    hits, _ = evidence(value, aliases)
    return [
        {"protein_id": protein, "go_id": go, "evidence_codes": sorted(
            {code for codes in by_raw.values() for code in codes}),
         "raw_go_ids": sorted(by_raw), "record_2025": target}
        for go, by_raw in sorted(hits.items())
    ]


def ipr_pairs(protein, target, hits, mapping, code):
    grouped = defaultdict(lambda: (set(), set()))
    for ipr in hits:
        for go, raw in mapping.get(ipr, {}).items():
            supporting_ipr, raw_ids = grouped[go]
            supporting_ipr.add(ipr)
            raw_ids.update(raw)
    return [
        {"protein_id": protein, "go_id": go, "evidence_codes": [code],
         "raw_go_ids": sorted(raw), "record_2025": target, "interpro_ids": sorted(iprs)}
        for go, (iprs, raw) in sorted(grouped.items())
    ]


def union_pairs(exp, ipr, axis):
    paired = {(row["protein_id"], row["go_id"]): row for row in exp}
    for row in ipr:
        key = (row["protein_id"], row["go_id"])
        if key not in paired:
            paired[key] = row
            continue
        other = paired[key]
        paired[key] = {**row, **{field: sorted(set(row[field]).union(other[field]))
                                 for field in ("raw_go_ids", "evidence_codes")}}
    order = {protein: pos for pos, protein in enumerate(axis)}
    return sorted(paired.values(), key=lambda row: (order[row["protein_id"]], row["go_id"]))


def evidence_view(ids, eligible, positives, axis, role, policy, evidence_type, input_hash):
    result = make_cohort(ids, [{"protein_id": protein, "evaluable": protein in eligible,
                                "identity_status": "verified" if protein in eligible else "unverified"}
                               for protein in ids], positives, role, policy, input_hash)
    result["population_axis"] = axis
    result["evidence_type"] = evidence_type
    return result


def count_ipr_coverage(values, mapped_iprs):
    covered = 0
    counts = Counter()
    for value in values:
        hits = ipr_hits(value)
        covered += bool(hits)
        counts.update(mapped_iprs.intersection(hits))
    return covered, counts


def annotation_coverage(values):
    raw_rows = experimental_rows = 0
    for value in values:
        codes = {code.upper() for item in annotation_items(value) if re.search(r"\bNOT\b", item, re.I) is None
                 for _, code in CODED_GO.findall(item)}
        raw_rows += bool(codes)
        experimental_rows += bool(codes.intersection(EXPERIMENTAL_CODES))
    return {"records_with_coded_go": raw_rows,
            "records_without_coded_go": len(values) - raw_rows,
            "records_with_strict_experimental_go": experimental_rows}


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ";".join(value) if isinstance(value, list) else value
                             for key, value in row.items()})


def write_axis(folder, name, sequences):
    with (folder / f"{name}_protein_ids.txt").open("w", encoding="utf-8") as ids, \
         (folder / f"{name}.fasta").open("w", encoding="utf-8") as fasta:
        for protein, sequence in sequences.items():
            ids.write(f"{protein}\n")
            fasta.write(f">{protein}\n{sequence}\n")
    with (folder / f"{name}_inputs.pkl").open("wb") as stream:
        pickle.dump({"proteins": list(sequences)}, stream, protocol=pickle.HIGHEST_PROTOCOL)


def build(args):
    test = json_read(args.test_manifest)
    if test.get("schema") != SCHEMA or test.get("complete") is not True:
        raise ValueError("Expected a completed original unseen-GO test manifest")
    paths = {"base_test_manifest": args.test_manifest, "new_obo": args.new_obo,
             "swiss22_dark": args.swiss22_dark, "swiss25": args.swiss25,
             "match_index": args.match_index, "swiss25_full": args.swiss25_full}
    sources = {key: source_ref(path) for key, path in paths.items() if path is not None}
    expected_obo = test.get("sources", {}).get("new_obo", {}).get("sha256")
    if expected_obo is not None and sources["new_obo"]["sha256"] != expected_obo:
        raise ValueError("new_obo differs from the original test manifest source")
    candidates = unique_ids(test["candidate_go_ids"], "unseen-GO candidate axis")
    if any(GO_ID.fullmatch(go) is None for go in candidates):
        raise ValueError("Unseen-GO candidate axis must contain GO IDs")
    aliases = go_aliases(args.new_obo, candidates, test["namespace"], test["new_obo_version"])
    new_map, new_meta = read_mapping(args.interpro2go_2025, aliases)
    old_map, old_meta = read_mapping(args.interpro2go_2022, aliases) if args.interpro2go_2022 else ({}, None)
    mapped_iprs = new_map.keys() | old_map.keys()
    old = load_swiss(args.swiss22_dark, historical=True)
    indexed = load_swiss(args.swiss25)
    historical_source_size = len(old)
    records, old, dark_sequences = match_index(args.match_index, old)
    dark_ids = list(dark_sequences)
    verified, sequence_verified = verified_indexed_matches(records, old, indexed, dark_sequences)
    full = load_swiss(args.swiss25_full) if args.swiss25_full else None
    if full is None:
        future, matched = indexed, verified
        methods = dict.fromkeys(verified, "indexed_name_or_accession_exact")
    else:
        future = full
        matched, methods = full_release_matches(old, indexed, full, verified, dark_sequences)
    dark_input_hash = bind_input(args.dark_input_dir, dark_ids, test["namespace"]) if args.dark_input_dir else None
    counts = {}
    for label, values in {
        "dark_2022": old["interpros"],
        "swiss25_indexed": indexed["interpros"],
        "dark_2025_indexed_identical": indexed["interpros"].iloc[list(sequence_verified.values())],
        "dark_2025_indexed_exact": indexed["interpros"].iloc[list(verified.values())],
        "dark_2025_resolved": future["interpros"].iloc[list(matched.values())],
    }.items():
        counts[label] = (len(values), *count_ipr_coverage(values, mapped_iprs))
    if full is not None:
        values = full["interpros"]
        counts["swiss25_full_release"] = (len(values), *count_ipr_coverage(values, mapped_iprs))

    future_exp, future_ipr, historical_new, historical_old, audit = [], [], [], [], []
    for row, hits in zip(records, old["interpros"]):
        protein = row["protein_22"]
        position = matched.get(protein)
        target = ""
        if position is not None:
            target = future["proteins"].iat[position]
            future_exp.extend(experimental_pairs(protein, target, future["annotations"].iat[position], aliases))
            future_ipr.extend(ipr_pairs(protein, target, ipr_hits(future["interpros"].iat[position]),
                                        new_map, "IPR2GO_2025"))
        old_hits = ipr_hits(hits)
        historical_new.extend(ipr_pairs(protein, "", old_hits, new_map, "IPR2GO_2025"))
        if old_meta is not None:
            historical_old.extend(ipr_pairs(protein, "", old_hits, old_map, "IPR2GO_2022"))
        audit.append({
            "protein_id": protein, "source_rid22": row["rid22"], "index_rid25": row["rid25"],
            "index_match_status": row["match_status"], "record_2025": target,
            "resolved_2025_row": position if position is not None else "",
            "resolved_match_method": methods.get(protein, "unresolved"),
            "sequence_sha256": row["sequence_sha256"],
        })

    cohorts = {}
    verified_eligible = set(matched)
    future_policy = "exact_sequence_and_entry_name_or_shared_accession"
    scope = "supplementary_dark_weak"
    dark_views = {
        "dark_weak_all_exp": (verified_eligible, future_exp, scope + "_experimental",
                              future_policy, "experimental_direct"),
        "dark_weak_all_ipr25": (verified_eligible, future_ipr, scope + "_interpro_proxy",
                                future_policy, "interpro2go_proxy"),
        "dark_weak_all_union": (verified_eligible, union_pairs(future_exp, future_ipr, dark_ids),
                                scope + "_mixed_evidence", future_policy, "mixed_evidence"),
        "dark_weak_history_ipr22_map25": (set(dark_ids), historical_new,
                                         scope + "_remapping_proxy", "verified_historical_sequence",
                                         "interpro2go_proxy"),
    }
    if old_meta is not None:
        dark_views["dark_weak_history_ipr22_map22"] = (
            set(dark_ids), historical_old, scope + "_historical_proxy", "verified_historical_sequence",
            "interpro2go_proxy",
        )
    for name, (eligible, positives, role, policy, evidence_type) in dark_views.items():
        cohorts[name] = evidence_view(
            dark_ids, eligible, positives, DARK_AXIS, role, policy, evidence_type, dark_input_hash,
        )

    axes = {DARK_AXIS: dark_sequences}
    if args.include_swiss25:
        swiss_sequences = sequence_axis(full, "full 2025 Swiss-Prot")
        axes[SWISS_AXIS] = swiss_sequences
        ids = list(swiss_sequences)
        swiss_input_hash = bind_input(args.swiss25_input_dir, ids, test["namespace"]) if args.swiss25_input_dir else None
        all_exp, all_ipr = [], []
        for protein, annotation, hits in full[["proteins", "annotations", "interpros"]].itertuples(index=False, name=None):
            all_exp.extend(experimental_pairs(protein, protein, annotation, aliases))
            all_ipr.extend(ipr_pairs(protein, protein, ipr_hits(hits), new_map, "IPR2GO_2025"))
        for suffix, pairs, evidence_type in (
            ("exp", all_exp, "experimental_direct"), ("ipr25", all_ipr, "interpro2go_proxy"),
        ):
            cohorts[f"swissprot_2025_all_{suffix}"] = evidence_view(
                ids, set(ids), pairs, SWISS_AXIS, "atlas_population_" + evidence_type,
                "native_release_entry_may_include_training_exposure", evidence_type, swiss_input_hash,
            )

    added_names = set(cohorts).intersection(test["cohorts"])
    if added_names:
        raise ValueError(f"The input already has population cohorts: {sorted(added_names)}")
    summary = [
        {"cohort": name, "role": view["role"], "evidence_type": view["evidence_type"],
         "population_axis": view["population_axis"],
         "proteins": len(view["protein_ids"]), "evaluable_proteins": len(view["evaluable_protein_ids"]),
         "positive_proteins": view["n_positive_proteins"], "positive_go": view["n_positive_go_ids"],
         "direct_positive_pairs": len(view["positives"])}
        for name, view in cohorts.items()
    ]
    ipr_report = [
        {"interpro_id": ipr, "candidate_go_2025_mapping": sorted(new_map.get(ipr, {})),
         "candidate_go_2022_mapping": sorted(old_map.get(ipr, {})),
         **{label: per_ipr[ipr] for label, (_, _, per_ipr) in counts.items()}}
        for ipr in sorted(mapped_iprs)
    ]
    inventory = {
        "schema": "latence-unseen-go-population-inventory-v1", "namespace": test["namespace"],
        "candidate_go_count": len(candidates), "sources": sources,
        "mapping_2025": new_meta, "mapping_2022": old_meta,
        "interpro_coverage": {label: {"protein_records": total, "records_with_any_ipr": annotated,
                                     "candidate_mapping_iprs_observed": len(per_ipr)}
                              for label, (total, annotated, per_ipr) in counts.items()},
        "indexed_2025_annotation_coverage": annotation_coverage(indexed["annotations"]),
        "full_2025_annotation_coverage": annotation_coverage(full["annotations"]) if full is not None else None,
        "dark_match_index_status_counts": dict(Counter(row["match_status"] for row in records)),
        "dark_indexed_sequence_verified": len(sequence_verified),
        "dark_indexed_identity_and_sequence_verified": len(verified),
        "dark_resolved_match_methods": dict(Counter(row["resolved_match_method"] for row in audit)),
        "dark_22_pickle_rows_outside_index": historical_source_size - len(old),
        "dark_resolved_unique_2025_entries": len(set(matched.values())),
        "cohorts": summary,
        "dark_population_source": "All indexed historical dark-weak proteins in rid22 order",
        "evidence_policy": "Experimental and InterPro-derived direct labels are separate views. Unobserved pairs are unlabeled.",
    }
    manifest = {**test, "cohorts": {**test["cohorts"], **cohorts}, "population_sources": sources,
                "population_mapping_2025": new_meta, "population_mapping_2022": old_meta}
    out = args.output_dir
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as work:
        folder = Path(work) / "result"
        folder.mkdir()
        json_write(folder / "test_manifest.json", manifest)
        json_write(folder / "inventory.json", inventory)
        write_tsv(folder / "cohort_inventory.tsv", summary, list(summary[0]))
        write_tsv(folder / "interpro2go_overlap.tsv", ipr_report,
                  ["interpro_id", "candidate_go_2025_mapping", "candidate_go_2022_mapping", *counts])
        write_tsv(folder / "dark_weak_match_audit.tsv", audit, list(audit[0]))
        for name, sequences in axes.items():
            write_axis(folder, name, sequences)
        for name, view in cohorts.items():
            write_tsv(folder / f"{name}_positives.tsv", view["positives"],
                      ["protein_id", "go_id", "evidence_codes", "raw_go_ids", "record_2025", "interpro_ids"])
        if args.metadata:
            with args.metadata.open("rb") as stream:
                metadata = pickle.load(stream)
            if not isinstance(metadata, dict) or (SPLIT_KEY in metadata and metadata[SPLIT_KEY] != test):
                raise ValueError("Metadata must be original unidata or contain the supplied original unseen test")
            metadata[SPLIT_KEY] = manifest
            with (folder / "unidata_with_dark_weak_unseen_go.pkl").open("wb") as stream:
                pickle.dump(metadata, stream, protocol=pickle.HIGHEST_PROTOCOL)
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"Output directory already exists: {out}")
        folder.rename(out)
    for row in summary:
        print(f"{row['cohort']}: {row['evaluable_proteins']:,}/{row['proteins']:,} evaluable, "
              f"{row['positive_proteins']:,} positive proteins, {row['positive_go']:,} GO, "
              f"{row['direct_positive_pairs']:,} direct positive pairs")
    print(out / "inventory.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--new-obo", type=Path, required=True)
    parser.add_argument("--swiss22-dark", type=Path, required=True)
    parser.add_argument("--swiss25", type=Path, required=True, help="Original pickle used to assign rid25")
    parser.add_argument("--swiss25-full", type=Path, help="Same release from DAT parsed with --keep-all")
    parser.add_argument("--match-index", type=Path, required=True)
    parser.add_argument("--interpro2go-2025", type=Path, required=True)
    parser.add_argument("--interpro2go-2022", type=Path)
    parser.add_argument("--include-swiss25", action="store_true", help="Export full-release evidence views and FASTA")
    parser.add_argument("--dark-input-dir", type=Path, help="Optional prepared v095 input cache")
    parser.add_argument("--swiss25-input-dir", type=Path, help="Optional prepared v095 input cache")
    parser.add_argument("--metadata", type=Path, help="Optional original or previously merged unidata pickle")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.include_swiss25 and args.swiss25_full is None:
        parser.error("--include-swiss25 requires --swiss25-full")
    if args.swiss25_input_dir and not args.include_swiss25:
        parser.error("--swiss25-input-dir requires --include-swiss25")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    build(args)


if __name__ == "__main__":
    main()
