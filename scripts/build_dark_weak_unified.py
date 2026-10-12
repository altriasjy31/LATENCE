#!/usr/bin/env python3
"""Build a versioned dark-weak benchmark joining experimental and InterPro2GO labels.

Reuses v1.0.1 build_dark_weak_test.py (ontology, table readers, historical
match index) and v0.1 probe_dark_weak_interpro2go.py (mapping reader).
Does not overwrite the existing experimental benchmark.

Output matrices use ALL original 2022 dark-weak cohort rows and the exact
LATENCE GO vocabulary order (including alias-equivalent columns). Never
interpret a zero in any label matrix as a validated negative.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import sparse

from build_dark_weak_test import (Ontology, NS, clean_seq, coded_go, read_table,
                                  load_go_vocab, file_sha256)
from probe_dark_weak_interpro2go import (read_mapping, normalize_mapping,
                                          ipr_ids, map_go)

MATRIX_TYPES = (
    'historical_ipr2go_direct',
    'future_ipr2go_all_direct',
    'future_ipr2go_delta_direct',
    'future_ipr2go_novel_direct',
    'future_ipr2go_novel_propagated',
    'future_experimental_swissprot_novel_direct',
    'future_experimental_crosssource_novel_direct',
    'future_union_novel_direct',
    'future_both_sources_novel_direct',
    # v0.2.1: explicit 2022 knowledge control and complete 2022→2025 change accounting.
    'historical_ipr2go_propagated',
    'historical_ipr2go_matched_direct',
    'historical_ipr2go_matched_propagated',
    'future_ipr2go_all_propagated',
    'future_ipr2go_retained_direct',
    'future_ipr2go_retained_propagated',
    'future_ipr2go_removed_direct',
    'future_ipr2go_delta_propagated',
    'future_ipr2go_removed_propagated',
    'cumulative_ipr2go_observed_direct',
)


def as_bool(series: pd.Series) -> pd.Series:
    """Read TSV booleans without treating literal 'False' as truthy."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.eq(1)
    vals = series.astype('string').str.strip().str.lower()
    bad = vals[~vals.isin(['true', 'false', '1', '0', '<na>'])]
    if not bad.empty:
        raise ValueError(f'Unexpected boolean value(s): {bad.iloc[:5].tolist()}')
    return vals.isin(('true', '1'))


def frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return '_No records._'
    columns = list(df.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    for row in df.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(v).replace('|', '\\|').replace('\n', ' ')
                                      for v in row) + ' |')
    return '\n'.join(lines)


def require_cols(df: pd.DataFrame, cols, name):
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f'{name}: missing columns {missing}')


def extract_exp_pairs(go_file: Path, ont: Ontology, known_rows: set[int]):
    """Reproduce the prior STRICT experimental + same-sequence Swiss-Prot novelty.

    Returns direct canonical GO sets for *the same 2022 row IDs*, independent of
    the InterPro version. Future experimental labels must not be re-inferred
    from the IPR2GO expansion.
    """
    if not go_file.is_file():
        raise FileNotFoundError(f'Missing prior benchmark pairs: {go_file}')
    df = pd.read_csv(go_file, sep='\t')
    require_cols(df, ['rid22', 'strict_eligible', 'is_strict_experimental',
                      'historically_unrecorded_identical_seq', 'canonical_go_2022'],
                 str(go_file))
    mask = (as_bool(df['strict_eligible']) & as_bool(df['is_strict_experimental'])
            & as_bool(df['historically_unrecorded_identical_seq']))
    result = defaultdict(set)
    for rid, value in df.loc[mask, ['rid22', 'canonical_go_2022']].itertuples(index=False, name=None):
        if pd.isna(value):
            continue
        canon = ont.canonical(str(value))
        if canon and int(rid) in known_rows:
            result[int(rid)].add(canon)
    return dict(result)


def group_canonical_columns(vocab: list[str], ont: Ontology) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for i, go in enumerate(vocab):
        canonical = ont.canonical(go)
        assert canonical is not None
        groups[canonical].append(i)
    return dict(groups)


def make_sparse_labels(protein_count, vocab_size, positives: dict[int, set[str]], columns):
    rr, cc = [], []
    unique_pairs = 0
    for rid, gos in positives.items():
        for go in gos:
            cols = columns.get(go, ())
            if cols:
                unique_pairs += 1
                rr.extend([rid] * len(cols))
                cc.extend(cols)
    x = sparse.csr_matrix((np.ones(len(rr), dtype=np.uint8), (rr, cc)),
                          shape=(protein_count, vocab_size), dtype=np.uint8)
    x.sum_duplicates()
    return x, unique_pairs


def summarize_matrix(x: sparse.csr_matrix, canonical_pairs: int, namespace: str, kind: str):
    return {'namespace': namespace, 'matrix': kind,
            'n_positive_proteins': int(x.getnnz(axis=1).astype(bool).sum()),
            'n_model_column_pairs': int(x.nnz),
            'n_canonical_pairs': int(canonical_pairs)}


def load_and_validate_inputs(args):
    if args.output_dir.resolve() == args.base_root.resolve():
        raise ValueError('The unified output directory must differ from the base benchmark directory')
    if args.output_dir.resolve() in args.base_root.resolve().parents:
        raise ValueError('Refusing to write over a parent of the original benchmark')
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f'Output directory is nonempty: {out}; use --overwrite')
        if not (out / 'unified_manifest.json').exists():
            raise FileExistsError(f'Refusing --overwrite without unified_manifest.json marker: {out}')
        # Only known output paths are permitted. Prevent destructive overwrite.
        permitted = {'unified_manifest.json', 'report.md', 'summary.json', 'tables', 'labels',
                     'unified_test_labeled.pkl', 'protein_ids.txt', 'test_protein_ids.txt',
                     'historical_test_protein_ids.txt', 'historical_test_labeled.pkl'}
        if {p.name for p in out.iterdir()} - permitted:
            raise FileExistsError(f'Unrecognized files in {out}; refusing overwrite')
    if out.exists() and any(out.iterdir()):
        # Operates ONLY on the marked unified subdirectory, never base_root.
        for dname in ('tables', 'labels'):
            if (out / dname).is_dir():
                shutil.rmtree(out / dname)
        for fname in ('summary.json', 'report.md', 'unified_manifest.json',
                      'unified_test_labeled.pkl', 'protein_ids.txt',
                      'test_protein_ids.txt', 'historical_test_protein_ids.txt',
                      'historical_test_labeled.pkl'):
            (out / fname).unlink(missing_ok=True)
    old = read_table(args.cohort_2022)
    full_old = read_table(args.uniprot_2022)
    future = read_table(args.uniprot_2025)
    matches = pd.read_csv(args.match_index, sep='\t')
    require_cols(old, ['proteins', 'sequences', 'interpros'], '2022 cohort')
    require_cols(full_old, ['proteins', 'sequences', 'interpros', 'annotations'], 'Full 2022 Swiss-Prot')
    require_cols(future, ['proteins', 'sequences', 'interpros'], '2025 Swiss-Prot')
    require_cols(matches, ['rid22', 'rid25', 'protein_22', 'match_status'], 'Historical match index')
    if old['proteins'].duplicated().any():
        raise ValueError('Duplicate 2022 names; cannot safely use existing match index')
    if matches['rid22'].duplicated().any():
        raise ValueError('Multiple 2025 matches for same 2022 rid22')
    if len(matches) != len(old) or set(matches['rid22']) != set(range(len(old))):
        raise ValueError('Match index must include every original 2022 cohort row')
    ordered = matches.sort_values('rid22').reset_index(drop=True)
    if any(ordered['protein_22'].astype(str).to_numpy() != old['proteins'].astype(str).to_numpy()):
        raise ValueError('2022 protein order does not agree with the original match index')
    bad = ordered.loc[ordered['match_status'].eq('identical') & ordered['rid25'].isna()]
    if len(bad):
        raise ValueError('Identical matches have missing 2025 rows')
    for _, row in ordered.loc[ordered['match_status'].eq('identical')].iterrows():
        j = int(row['rid25'])
        if j < 0 or j >= len(future):
            raise ValueError(f'Invalid rid25={j}')
    return old, full_old, future, ordered


def run(args):
    old, full_old, future, matches = load_and_validate_inputs(args)
    ont = Ontology(args.go_obo_2022, mode=args.propagation)
    m0_raw, m0_meta = read_mapping(args.mapping_2022)
    m1_raw, m1_meta = read_mapping(args.mapping_2025)
    m0, unmapped_22 = normalize_mapping(m0_raw, ont)
    m1, unmapped_25 = normalize_mapping(m1_raw, ont)
    assert len(old) == len(matches)

    # Require a version-locked main benchmark for joining existing experimental positives.
    exp_by_rid = extract_exp_pairs(
        args.base_root / 'tables' / 'future_go_pairs.tsv', ont,
        set(matches.loc[matches['match_status'].eq('identical'), 'rid22'].astype(int))
    )
    old_seq = [clean_seq(s) for s in old['sequences']]
    target_sequences = {s for s in old_seq if s is not None}

    # Temporal reference: all 2022 Swiss-Prot GO annotations and InterPro hits
    # on *exactly identical* sequences; not merely each target's own annotations.
    same_seq_go = defaultdict(set)
    same_seq_ipr = defaultdict(set)
    for seq, raw, ips in zip(full_old['sequences'], full_old['annotations'], full_old['interpros']):
        s = clean_seq(seq)
        if s is not None and s in target_sequences:
            same_seq_ipr[s].update(ipr_ids(ips))
            for raw_go in coded_go(raw):
                c = ont.canonical(raw_go)
                if c:
                    same_seq_go[s].add(c)
    hist_GO_closure = {s:ont.propagate(gs) for s,gs in same_seq_go.items()}
    hist_IPR_closure = {s:ont.propagate(map_go(ips,m0)) for s,ips in same_seq_ipr.items()}

    # Do not use future labels in any protein identity selection. Only trust the
    # historical builder's unique identity match and explicit exact-sequence check.
    labels = {kind: defaultdict(set) for kind in MATRIX_TYPES}
    ipr_delta_pairs = []
    temporal_status_pairs = []
    historical_pairs = []
    row_summary = []
    for rid, matched in enumerate(matches.itertuples(index=False)):
        assert int(matched.rid22) == rid
        seq = old_seq[rid]
        valid_future = matched.match_status == 'identical' and pd.notna(matched.rid25)
        i22 = ipr_ids(old.at[rid, 'interpros'])
        g22 = map_go(i22, m0)
        labels['historical_ipr2go_direct'][rid] = set(g22)
        labels['historical_ipr2go_propagated'][rid] = ont.propagate(g22)
        # This full historical baseline includes even the 79 sequence-revised
        # proteins; the matched-only version below supports fair 2022/2025 comparison.
        for go in sorted(g22):
            historical_pairs.append({
                'rid22':rid, 'protein_22':str(old.at[rid,'proteins']),
                'namespace':NS.get(ont.terms[go].namespace, 'unmapped'),
                'go_id_2022':go,
                'supporting_iprs_2022':';'.join(sorted(i for i in i22 if go in m0.get(i, set()))),
            })
        labels['cumulative_ipr2go_observed_direct'][rid] = set(g22)
        i25 = set()
        g25 = set()
        delta = set()
        novel = set()
        exp = set()
        new_matches = set()
        retired = set()
        retained_go = set()
        removed_go = set()
        if valid_future:
            j = int(matched.rid25)
            later = future.iloc[j]
            if seq is None or seq != clean_seq(later['sequences']):
                raise ValueError(f'Historical identity index disagrees with sequences at rid22={rid}')
            i25 = ipr_ids(later['interpros'])
            g25 = map_go(i25, m1)
            new_matches = i25 - i22
            retired = i22 - i25
            g01 = map_go(i22, m1)
            g10 = map_go(i25, m0)
            delta = g25 - g22
            retained_go = g22 & g25
            removed_go = g22 - g25
            # Temporal partition (canonical direct GO) must be exact:
            #   G22 = retained U removed; G25 = retained U added.
            assert not (retained_go & removed_go)
            assert not (retained_go & delta)
            assert not (removed_go & delta)
            assert len(retained_go) + len(removed_go) == len(g22)
            assert len(retained_go) + len(delta) == len(g25)
            known_go = hist_GO_closure.get(seq, set())
            known_ipr = hist_IPR_closure.get(seq, set())
            historical_knowledge = known_go | known_ipr | ont.propagate(g22)
            novel = {g for g in delta if g not in historical_knowledge}
            exp = exp_by_rid.get(rid, set())
            exp_cross = {g for g in exp if g not in known_ipr}
            labels['historical_ipr2go_matched_direct'][rid] = set(g22)
            labels['historical_ipr2go_matched_propagated'][rid] = ont.propagate(g22)
            labels['future_ipr2go_all_direct'][rid] = g25
            labels['future_ipr2go_all_propagated'][rid] = ont.propagate(g25)
            labels['future_ipr2go_retained_direct'][rid] = retained_go
            labels['future_ipr2go_retained_propagated'][rid] = (
                ont.propagate(g22) & ont.propagate(g25)
            )
            labels['future_ipr2go_removed_direct'][rid] = removed_go
            labels['future_ipr2go_delta_direct'][rid] = delta
            # Ontology-level novelty must compare the two propagated snapshots,
            # not simply propagate a direct difference (which can repeat old ancestors).
            labels['future_ipr2go_delta_propagated'][rid] = (
                ont.propagate(g25) - ont.propagate(g22)
            )
            labels['future_ipr2go_removed_propagated'][rid] = (
                ont.propagate(g22) - ont.propagate(g25)
            )
            labels['cumulative_ipr2go_observed_direct'][rid] = g22 | g25
            for go in sorted(g22 | g25):
                status = ('retained' if go in retained_go
                          else 'added' if go in delta else 'removed')
                temporal_status_pairs.append({
                    'rid22':rid, 'protein_22':str(old.at[rid,'proteins']),
                    'protein_25':str(later['proteins']),
                    'namespace':NS.get(ont.terms[go].namespace, 'unmapped'),
                    'go_id_2022':go, 'temporal_status':status,
                    'supporting_iprs_2022':';'.join(sorted(
                        i for i in i22 if go in m0.get(i, set()))),
                    'supporting_iprs_2025':';'.join(sorted(
                        i for i in i25 if go in m1.get(i, set()))),
                })
            labels['future_ipr2go_novel_direct'][rid] = novel
            labels['future_ipr2go_novel_propagated'][rid] = ont.propagate(novel) - historical_knowledge
            labels['future_experimental_swissprot_novel_direct'][rid] = exp
            labels['future_experimental_crosssource_novel_direct'][rid] = exp_cross
            labels['future_union_novel_direct'][rid] = novel | exp_cross
            labels['future_both_sources_novel_direct'][rid] = novel & exp_cross
            for g in sorted(delta):
                sources = {i for i in i25 if g in m1.get(i, set())}
                fresh_support = sources & new_matches
                old_support_new_mapping = {i for i in (sources & i22) if g not in m0.get(i, set())}
                if g in g01 and g in g10:
                    change_type = 'mapping_and_match_counterfactual'
                elif g in g01:
                    change_type = 'mapping_only_counterfactual'
                elif g in g10:
                    change_type = 'match_only_counterfactual'
                else:
                    change_type = 'joint_or_lost_match_counterfactual'
                namespace = NS.get(ont.terms[g].namespace, 'unmapped')
                ipr_delta_pairs.append({
                    'rid22':rid,'protein_22':str(old.at[rid,'proteins']),
                    'protein_25':str(later['proteins']),
                    'namespace':namespace,'go_id_2022':g,
                    'counterfactual_type':change_type,
                    'supporting_iprs_25':';'.join(sorted(sources)),
                    'newly_matched_support_iprs':';'.join(sorted(fresh_support)),
                    'old_iprs_with_new_mapping':';'.join(sorted(old_support_new_mapping)),
                    'not_implied_by_2022_self_ipr':g not in ont.propagate(g22),
                    'not_implied_by_2022_same_seq_go':g not in known_go,
                    'not_implied_by_2022_same_seq_ipr':g not in known_ipr,
                    'historical_novel_crosssource':g in novel,
                    'future_exp_swissprot_novel_support':g in exp,
                    'future_exp_crosssource_novel_support':g in exp_cross,
                })
        row_summary.append({
            'matrix_row':rid, 'rid22':rid, 'protein_22':old.at[rid,'proteins'],
            'rid25':int(matched.rid25) if pd.notna(matched.rid25) else '',
            'protein_25':future.iloc[int(matched.rid25)]['proteins'] if pd.notna(matched.rid25) else '',
            'match_status':str(matched.match_status),
            'sequence_sha256':hashlib.sha256(seq.encode('ascii')).hexdigest() if seq else '',
            'valid_future':valid_future,
            'n_iprs_22':len(i22),'n_iprs_25':len(i25),
            'n_added_iprs':len(new_matches),'n_retired_iprs':len(retired),
            'n_iprgo_22':len(g22),'n_iprgo_25':len(g25),
            'n_iprgo_delta':len(delta),'n_iprgo_crosssource_novel':len(novel),
            'n_iprgo_retained':len(retained_go),
            'n_iprgo_removed':len(removed_go),
            'n_iprgo_cumulative':len(labels['cumulative_ipr2go_observed_direct'][rid]),
            'n_exp_swissprot_novel':len(exp),
            'n_exp_crosssource_novel':len(exp_cross),
            'n_future_union_novel':len(novel | exp_cross),
        })

    # For unified annotation provenance, enumerate all benchmark-direct positives
    # (not the old 2022 baseline); keep the 2025 sources individually visible.
    unified_pairs = []
    for rid in range(len(old)):
        exp = labels['future_experimental_swissprot_novel_direct'].get(rid, set())
        exp_cross = labels['future_experimental_crosssource_novel_direct'].get(rid, set())
        ipr = labels['future_ipr2go_novel_direct'].get(rid, set())
        for go in sorted(exp | ipr):
            ns = NS.get(ont.terms[go].namespace, 'other')
            if ns not in ('bp','mf','cc'):
                continue
            unified_pairs.append({
                'rid22':rid,'protein_22':old.at[rid,'proteins'],
                'namespace':ns,'canonical_go_2022':go,
                'experimental_swissprot_novel':go in exp,
                'experimental_crosssource_novel':go in exp_cross,
                'interpro2go_crosssource_novel':go in ipr,
                'evaluation_union_label':go in (exp_cross | ipr),
                'both_sources':go in (exp_cross & ipr),
            })

    out = args.output_dir
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'labels').mkdir(parents=True, exist_ok=True)
    protein_rows = pd.DataFrame(row_summary)
    protein_rows.to_csv(out / 'tables' / 'full_dark_weak_cohort.tsv', sep='\t', index=False)
    pd.DataFrame(ipr_delta_pairs, columns=[
        'rid22','protein_22','protein_25','namespace','go_id_2022',
        'counterfactual_type','supporting_iprs_25','newly_matched_support_iprs',
        'old_iprs_with_new_mapping','not_implied_by_2022_self_ipr',
        'not_implied_by_2022_same_seq_go','not_implied_by_2022_same_seq_ipr',
        'historical_novel_crosssource',
        'future_exp_swissprot_novel_support','future_exp_crosssource_novel_support',
    ]).to_csv(out / 'tables' / 'ipr2go_temporal_delta_pairs.tsv', sep='\t', index=False)
    pd.DataFrame(unified_pairs, columns=[
        'rid22','protein_22','namespace','canonical_go_2022',
        'experimental_swissprot_novel','experimental_crosssource_novel',
        'interpro2go_crosssource_novel','evaluation_union_label','both_sources',
    ]).to_csv(out / 'tables' / 'unified_direct_positive_pairs.tsv', sep='\t', index=False)

    vocab_paths = {'bp':args.bp_terms,'mf':args.mf_terms,'cc':args.cc_terms}
    stats = []
    eligible = {'bp':set(), 'mf':set(), 'cc':set()}
    allowed_go = set()
    eval_rows_by_type = defaultdict(set)
    for ns, path in vocab_paths.items():
        if path is None:
            continue
        vocab = load_go_vocab(path, ont, ns)
        columns = group_canonical_columns(vocab, ont)
        allowed_go.update(columns)
        folder = out / 'labels' / ns
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'go_ids.txt').write_text(''.join(t+'\n' for t in vocab), encoding='utf-8')
        canonical_columns = [ont.canonical(t) for t in vocab]
        pd.DataFrame({'model_column':list(range(len(vocab))),
                      'model_go_id':vocab,
                      'canonical_go_id_2022':canonical_columns}).to_csv(
                        folder / 'vocabulary_mapping.tsv', sep='\t', index=False)
        for kind in MATRIX_TYPES:
            x, canonical_pairs = make_sparse_labels(len(old), len(vocab), labels[kind], columns)
            sparse.save_npz(folder / f'{kind}.npz', x, compressed=True)
            stats.append(summarize_matrix(x, canonical_pairs, ns, kind))
            if kind.startswith('future_') and kind.endswith('_direct'):
                eval_rows_by_type[kind].update(np.flatnonzero(x.getnnz(axis=1)).tolist())
            if kind in ('future_union_novel_direct', 'future_ipr2go_novel_direct',
                        'future_experimental_swissprot_novel_direct',
                        'future_experimental_crosssource_novel_direct',
                        'historical_ipr2go_direct', 'historical_ipr2go_matched_direct',
                        'future_ipr2go_retained_direct', 'future_ipr2go_removed_direct',
                        'future_ipr2go_delta_direct', 'cumulative_ipr2go_observed_direct'):
                positive_rows = np.flatnonzero(x.getnnz(axis=1)).astype(np.int32)
                np.save(folder / f'evaluable_rows_{kind}.npy', positive_rows)
                if kind == 'future_union_novel_direct':
                    eligible[ns] = set(positive_rows.tolist())
                    np.save(folder / 'evaluable_rows.npy', positive_rows)
        # Labels are full-cohort rows; explicit metadata prevents wrong indexing.
        protein_rows[['matrix_row','rid22','protein_22','rid25','protein_25',
                      'match_status','sequence_sha256','valid_future']].to_csv(
                        folder / 'row_metadata.tsv', sep='\t', index=False)

    # Audit 2022 GO observations and a disjoint retained/added/removed temporal
    # partition, with a per-pair model-vocabulary-coverage flag.
    hist_pairs_frame = pd.DataFrame(historical_pairs, columns=[
        'rid22','protein_22','namespace','go_id_2022','supporting_iprs_2022'])
    temporal_frame = pd.DataFrame(temporal_status_pairs, columns=[
        'rid22','protein_22','protein_25','namespace','go_id_2022',
        'temporal_status','supporting_iprs_2022','supporting_iprs_2025'])
    hist_pairs_frame['in_model_vocab'] = hist_pairs_frame['go_id_2022'].isin(allowed_go)
    temporal_frame['in_model_vocab'] = temporal_frame['go_id_2022'].isin(allowed_go)
    hist_pairs_frame.to_csv(out / 'tables' / 'ipr2go_2022_baseline_pairs.tsv',sep='\t',index=False)
    temporal_frame.to_csv(out / 'tables' / 'ipr2go_temporal_status_pairs.tsv',sep='\t',index=False)
    transition_stats = []
    for ns in ('bp','mf','cc'):
        raw_ns = temporal_frame.loc[temporal_frame['namespace'].eq(ns)]
        for status in ('retained','added','removed'):
            a = raw_ns.loc[raw_ns['temporal_status'].eq(status)]
            b = a.loc[a['in_model_vocab']]
            transition_stats.append({
                'namespace':ns,'temporal_status':status,
                'proteins':int(a['rid22'].nunique()),'canonical_pairs':int(len(a)),
                'unique_go':int(a['go_id_2022'].nunique()),
                'model_evaluable_proteins':int(b['rid22'].nunique()),
                'model_evaluable_canonical_pairs':int(len(b)),
            })
    transition_frame = pd.DataFrame(transition_stats)
    transition_frame.to_csv(out / 'tables' / 'ipr2go_transition_summary.tsv',sep='\t',index=False)

    # A compact pickle for direct inferencing of only rows with future
    # verifiable positives, while retaining the all-40k canonical row index.
    labeled_subset = []
    for rid in sorted(set().union(*eligible.values())):
        exp = labels['future_experimental_swissprot_novel_direct'].get(rid, set())
        exp_cross = labels['future_experimental_crosssource_novel_direct'].get(rid, set())
        ipr = labels['future_ipr2go_novel_direct'].get(rid, set())
        rawrow = old.iloc[rid]
        labeled_subset.append({
            'rid22':rid, 'proteins':rawrow['proteins'], 'accessions':rawrow.get('accessions', ''),
            'sequences':rawrow['sequences'], 'sequence_sha256':protein_rows.at[rid,'sequence_sha256'],
            'go_experimental_swissprot_novel':sorted(exp),
            'go_experimental_crosssource_novel':sorted(exp_cross),
            'go_ipr2go_crosssource_novel':sorted(ipr),
            'go_unified_evaluable':sorted((exp_cross | ipr) & allowed_go),
            'go_intersection_evaluable':sorted((exp_cross & ipr) & allowed_go),
        })
    pd.DataFrame(labeled_subset, columns=['rid22','proteins','accessions','sequences',
        'sequence_sha256','go_experimental_swissprot_novel',
        'go_experimental_crosssource_novel','go_ipr2go_crosssource_novel',
        'go_unified_evaluable','go_intersection_evaluable']).to_pickle(
            out / 'unified_test_labeled.pkl')
    (out / 'protein_ids.txt').write_text(''.join(str(v)+'\n' for v in old['proteins']),encoding='utf-8')
    (out / 'test_protein_ids.txt').write_text(''.join(str(v['proteins'])+'\n' for v in labeled_subset),encoding='utf-8')
    # 'historical_' does not begin with 'future_' and therefore needs explicit
    # registration for the separate 2022-baseline evaluation index.
    historical_rows = set().union(*[
        set(np.load(out / 'labels' / ns /
            'evaluable_rows_historical_ipr2go_direct.npy').tolist())
        for ns in ('bp','mf','cc')
        if (out / 'labels' / ns / 'evaluable_rows_historical_ipr2go_direct.npy').is_file()
    ])
    historical_list = sorted(historical_rows)
    (out / 'historical_test_protein_ids.txt').write_text(''.join(
        str(old.at[i, 'proteins']) + '\n' for i in historical_list), encoding='utf-8')
    historical_labeled = []
    for rid in historical_list:
        rawrow = old.iloc[rid]
        historical_labeled.append({
            'rid22':rid, 'proteins':str(rawrow['proteins']),
            'accessions':rawrow.get('accessions', ''),
            'sequences':rawrow['sequences'],
            'sequence_sha256':protein_rows.at[rid,'sequence_sha256'],
            'go_historical_ipr2go_direct':sorted(
                labels['historical_ipr2go_direct'].get(rid,set()) & allowed_go),
            'go_historical_ipr2go_propagated':sorted(
                labels['historical_ipr2go_propagated'].get(rid,set()) & allowed_go),
            'benchmark_role':'2022_observed_computational_baseline_not_future_blind',
        })
    pd.DataFrame(historical_labeled, columns=[
        'rid22','proteins','accessions','sequences','sequence_sha256',
        'go_historical_ipr2go_direct','go_historical_ipr2go_propagated',
        'benchmark_role']).to_pickle(out / 'historical_test_labeled.pkl')
    stat_frame = pd.DataFrame(stats)
    stat_frame.to_csv(out / 'tables' / 'matrix_summary.tsv', sep='\t', index=False)
    v2022 = sum(s == 'identical' for s in matches['match_status'])
    counterfactual_summary = (pd.DataFrame(ipr_delta_pairs)
                              .groupby(['namespace','counterfactual_type'])
                              .agg(proteins=('rid22','nunique'),
                                   positive_pairs=('go_id_2022','size'))
                              .reset_index() if ipr_delta_pairs else pd.DataFrame(
                                  columns=['namespace','counterfactual_type','proteins','positive_pairs']))
    counterfactual_summary.to_csv(out / 'tables' / 'counterfactual_summary.tsv', sep='\t', index=False)
    exprows = {i for i,g in labels['future_experimental_swissprot_novel_direct'].items() if g}
    expcrossrows = {i for i,g in labels['future_experimental_crosssource_novel_direct'].items() if g}
    iprrows = {i for i,g in labels['future_ipr2go_novel_direct'].items() if g}
    unionrows = iprrows | expcrossrows
    eval_exp = eval_rows_by_type['future_experimental_swissprot_novel_direct']
    eval_exp_cross = eval_rows_by_type['future_experimental_crosssource_novel_direct']
    eval_ipr = eval_rows_by_type['future_ipr2go_novel_direct']
    eval_union = eval_rows_by_type['future_union_novel_direct']
    result = {
        'n_dark_weak_cohort':len(old),
        'n_identical_matches':int(v2022),
        'n_historical_ipr2go_direct_proteins':int(sum(bool(g) for g in labels['historical_ipr2go_direct'].values())),
        'n_historical_ipr2go_model_evaluable_proteins':len(historical_rows),
        'n_historical_ipr2go_matched_direct_proteins':int(sum(bool(g) for g in labels['historical_ipr2go_matched_direct'].values())),
        'n_temporal_ipr2go_retained_proteins':int(sum(bool(g) for g in labels['future_ipr2go_retained_direct'].values())),
        'n_temporal_ipr2go_removed_proteins':int(sum(bool(g) for g in labels['future_ipr2go_removed_direct'].values())),
        'n_temporal_ipr2go_added_proteins':int(sum(bool(g) for g in labels['future_ipr2go_delta_direct'].values())),
        'n_temporal_ipr2go_retained_canonical_pairs':int(sum(len(g) for g in labels['future_ipr2go_retained_direct'].values())),
        'n_temporal_ipr2go_removed_canonical_pairs':int(sum(len(g) for g in labels['future_ipr2go_removed_direct'].values())),
        'n_temporal_ipr2go_added_canonical_pairs':int(sum(len(g) for g in labels['future_ipr2go_delta_direct'].values())),
        'n_has_ipr_2022':int((protein_rows['n_iprs_22'] > 0).sum()),
        'n_has_ipr_2025_exact':int((protein_rows.loc[protein_rows['valid_future'], 'n_iprs_25'] > 0).sum()),
        'n_no_ipr_2022_yes_2025':int(((protein_rows['n_iprs_22'] == 0) &
                                        (protein_rows['n_iprs_25'] > 0) &
                                         protein_rows['valid_future']).sum()),
        'n_new_ipr_2025_exact':int(((protein_rows['n_added_iprs'] > 0) & protein_rows['valid_future']).sum()),
        'n_future_ipr2go_delta_proteins':int(sum(bool(x) for x in labels['future_ipr2go_delta_direct'].values())),
        'n_future_ipr2go_delta_pairs':len(ipr_delta_pairs),
        'n_future_ipr2go_crosssource_novel_proteins':len(iprrows),
        'n_future_experimental_swissprot_novel_proteins':len(exprows),
        'n_future_experimental_crosssource_novel_proteins':len(expcrossrows),
        'n_future_union_novel_proteins':len(unionrows),
        'n_future_union_new_proteins_vs_experimental':len(iprrows - exprows),
        'n_future_crosssource_union_new_proteins_vs_crosssource_experimental':len(iprrows - expcrossrows),
        'n_future_overlap_proteins':len(iprrows & expcrossrows),
        'n_model_evaluable_experimental_swissprot_novel_proteins':len(eval_exp),
        'n_model_evaluable_experimental_crosssource_novel_proteins':len(eval_exp_cross),
        'n_model_evaluable_ipr2go_crosssource_novel_proteins':len(eval_ipr),
        'n_model_evaluable_union_novel_proteins':len(eval_union),
        'n_model_evaluable_new_ipr2go_proteins_beyond_experimental':len(eval_ipr - eval_exp),
        'n_model_evaluable_new_ipr2go_proteins_beyond_crosssource_experimental':len(eval_ipr - eval_exp_cross),
        'n_future_overlap_direct_pairs':sum(
            len(g & labels['future_experimental_crosssource_novel_direct'].get(i, set()))
            for i,g in labels['future_ipr2go_novel_direct'].items()),
        'n_distinct_identical_sequences':int(protein_rows.loc[protein_rows.valid_future, 'sequence_sha256'].nunique()),
        'unmapped_old_go_terms':len(unmapped_22),
        'unmapped_future_go_terms':len(unmapped_25),
        'mapping_meta_2022':m0_meta,
        'mapping_meta_2025':m1_meta,
        'dataset_interpretation':{
            'historical_ipr2go_direct':'2022 full dark-weak cohort computational positive baseline; NOT an independent future test',
            'historical_ipr2go_matched_direct':'2022 labels restricted to identical 2022/2025 sequences for fair time comparisons',
            'historical_ipr2go_matched_propagated':'ancestor-closed 2022 labels restricted to unchanged sequences',
            'historical_ipr2go_propagated':'2022 IPR2GO baseline plus selected ontology ancestor closure',
            'future_ipr2go_retained_direct':'same canonical GO inferred at both snapshots',
            'future_ipr2go_retained_propagated':'canonical GO implied by both propagated snapshot label sets',
            'future_ipr2go_removed_direct':'2022-inferred GO missing in 2025; REMOVED ANNOTATION, not a negative biological finding',
            'future_ipr2go_delta_propagated':'2025 propagated GO minus 2022 propagated GO on matched sequence-invariant proteins',
            'cumulative_ipr2go_observed_direct':'union of observed 2022/2025 GO; descriptive knowledge inventory, NOT a blind evaluation set',
            'future_ipr2go_all_direct':'2025 computational annotation pool, NOT necessarily novel',
            'future_ipr2go_delta_direct':'future IPR2GO newly obtainable direct GO (vs I22*M22)',
            'future_ipr2go_novel_direct':'future IPR2GO delta excluding 2022 GO and IPR2GO knowledge of same-sequence Swiss-Prot',
            'future_experimental_swissprot_novel_direct':'prior experimental benchmark labels, excluding historical same-sequence GO (not IPR2GO)',
            'future_experimental_crosssource_novel_direct':'prior experimental labels additionally excluding 2022 same-sequence IPR2GO implication',
            'future_union_novel_direct':'union of future IPR2GO crosssource novelty and experimental crosssource novelty',
        },
    }
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    manifest = {
        'script_version':'0.2.1', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'row_order':'exact original 2022 dark weak cohort / match_index rid22',
        'input_2022_cohort':str(args.cohort_2022),
        'input_2022_swissprot':str(args.uniprot_2022),
        'input_2025_swissprot':str(args.uniprot_2025),
        'base_benchmark':str(args.base_root),
        'ontology_2022':str(args.go_obo_2022),
        'propagation':args.propagation,
        'mapping_2022':m0_meta, 'mapping_2025':m1_meta,
        'mapping_2022_file':str(args.mapping_2022),
        'mapping_2025_file':str(args.mapping_2025),
        'file_hashes':{
            str(k):file_sha256(k) for k in [args.mapping_2022, args.mapping_2025, args.go_obo_2022]
        },
        'status_contract':'zeros are UNOBSERVED, not true negatives',
        'restrictions':[
            'UniProt .dat DR InterPro cross-references are ID-level and lack match coordinates',
            'InterPro2GO is computational transferred annotation, not independent experimental evidence',
            '2022 GO normalization excludes future-only GO from model-aligned matrices',
            'Original primary accessions not verified by this script',
            'Historical novelty is relative to same-sequence 2022 Swiss-Prot only, not full 2022 protein universe',
            'Transductive weak-protein membership and model data provenance require separate audit',
        ],
    }
    (out / 'unified_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    summary_show = stat_frame.loc[stat_frame['matrix'].isin([
        'historical_ipr2go_direct', 'future_ipr2go_all_direct',
        'future_ipr2go_delta_direct', 'future_ipr2go_novel_direct',
        'future_experimental_swissprot_novel_direct',
        'future_experimental_crosssource_novel_direct',
        'future_union_novel_direct', 'future_both_sources_novel_direct'])].copy()
    synopsis = [
        '# LATENCE Unified Historical Dark–Weak Benchmark (experimental + InterPro2GO)',
        '',
        f'- 2022 dark-weak universe: **{len(old):,}**',
        f'- Exact-sequence 2022/2025 matches: **{int(v2022):,}**',
        f'- Historical IPR entries (all 2022 dark weak): **{result["n_has_ipr_2022"]:,}**',
        f'- Historical 2022 IPR2GO-positive proteins: **{result["n_historical_ipr2go_direct_proteins"]:,}**',
        f'- Historical 2022 IPR2GO-positive model-evaluable proteins: **{len(historical_rows):,}**',
        f'- Future IPR entries (exact-sequence subset): **{result["n_has_ipr_2025_exact"]:,}**',
        f'- No historical IPR but future IPR exists: **{result["n_no_ipr_2022_yes_2025"]:,}**',
        f'- Proteins with 2025 IPR2GO delta: **{result["n_future_ipr2go_delta_proteins"]:,}**',
        f'- InterPro2GO cross-source historical-novel proteins: **{len(iprrows):,}**',
        f'- Prior experimental Swiss-Prot historical-novel proteins: **{len(exprows):,}**',
        f'- Experimental proteins remaining novel after 2022 IPR2GO control: **{len(expcrossrows):,}**',
        f'- Unified cross-source historical-novel proteins: **{len(unionrows):,}**',
        f'- New IPR2GO proteins beyond original experimental cohort (all 2022 GO): **{len(iprrows - exprows):,}**',
        f'- Model-vocabulary-evaluable union: **{len(eval_union):,}**',
        f'- Model-vocabulary-evaluable extra IPR2GO proteins beyond original experimental: **{len(eval_ipr - eval_exp):,}**',
        '',
        '## Source-separated GO matrix statistics',
        '',
        frame_to_markdown(summary_show),
        '',
        '## Matched-cohort InterPro2GO direct-label transitions',
        '',
        frame_to_markdown(transition_frame),
        '',
        'The identical-sequence-only transition partition is **2022 = retained + removed** '
        'and **2025 = retained + added** (per protein/canonical GO). Removed means a lost '
        'annotation, **NOT** evidence that the protein lacks that function.',
        '',
        '## Temporal IPR2GO source decomposition',
        '',
        frame_to_markdown(counterfactual_summary),
        '',
        '## Terminology',
        '- Experimental and IPR2GO are separately recorded; they are **not** interchangeable truth sources.',
        '- 2022 IPR2GO is a historical-knowledge positive-label baseline, NOT a prospective blind test; audit any training/teacher/pseudo use of this source.',
        '- The 2022 full-cohort and matched-only historical matrices are different; do not compare a 40,384-row count with a 40,305-row count.',
        '- `historical_test_protein_ids.txt` contains model-vocabulary-positive 2022 baseline proteins, not the newly validated 2025 test proteins.',
        '- Missing GO entries are unknown (open-world labels).',
        '- The original experimental 2022/2025 subset is preserved as an independent matrix.',
        '- Cross-source novelty additionally excludes IPR2GO evidence for 2022 identical-sequence Swiss-Prot records.',
        '- Temporal updates include both new IPR matching and new IPR-to-GO mappings.',
        '- The unified direct matrix is for label lookup and positive-label ranking; naive Fmax on all 40k rows is invalid.',
        '- For GO namespaces, see `tables/matrix_summary.tsv` and `labels/*/evaluable_rows.npy`.',
        '- Future-only GO is omitted from 2022-model-aligned matrices and reported as unmapped.',
    ]
    (out / 'report.md').write_text('\n'.join(synopsis)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k.startswith('n_')},indent=2))
    print(stat_frame.to_string(index=False))
    print('Historical IPR2GO transition summary (identical sequences only):')
    print(transition_frame.to_string(index=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort-2022',type=Path,required=True)
    p.add_argument('--uniprot-2022',type=Path,required=True)
    p.add_argument('--uniprot-2025',type=Path,required=True)
    p.add_argument('--match-index',type=Path,default=Path('data/dark_weak_test/tables/cohort_match_index.tsv'))
    p.add_argument('--base-root',type=Path,default=Path('data/dark_weak_test'))
    p.add_argument('--mapping-2022',type=Path,required=True)
    p.add_argument('--mapping-2025',type=Path,required=True)
    p.add_argument('--go-obo-2022',type=Path,required=True)
    p.add_argument('--bp-terms',type=Path)
    p.add_argument('--mf-terms',type=Path)
    p.add_argument('--cc-terms',type=Path)
    p.add_argument('--propagation',choices=('deepgozero','is_a_part_of','is_a'),default='deepgozero')
    p.add_argument('--output-dir',type=Path,default=Path('data/dark_weak_test/unified'))
    p.add_argument('--overwrite',action='store_true')
    run(p.parse_args())

if __name__ == '__main__':
    main()
