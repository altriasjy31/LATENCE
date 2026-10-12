#!/usr/bin/env python3
"""Historical InterPro2GO pilot: 2022 dark-weak vs 2025 UniProt snapshots.

Requires v1.0.1's build_dark_weak_test.py in ./scripts/ to reuse the
historical GO mapping and input-parser contract. This probe NEVER overwrites
the curated/experimental benchmark, and it never treats no GO as a negative.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import re

import pandas as pd

from build_dark_weak_test import Ontology, clean_seq, read_table, coded_go, EXPERIMENTAL_CODES, NS

IPR_RE = re.compile(r'IPR\d{6,}')
GO_RE = re.compile(r'GO:\d{7}')
MAPPING_RE = re.compile(r'^InterPro:(IPR\d+)\b.*?;\s*(GO:\d{7})\s*$')


def ipr_ids(value) -> set[str]:
    """Read DeepGOZero's list of IPR strings; handle list repr as well."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {ip for x in value for ip in IPR_RE.findall(str(x))}
    return set(IPR_RE.findall(str(value)))


def read_mapping(path: Path) -> tuple[dict[str, set[str]], dict]:
    opener = gzip.open if str(path).endswith('.gz') else open
    pairs = defaultdict(set)
    bad_lines = []
    headers = []
    n_data = 0
    with opener(path, 'rt', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            text = line.strip()
            if not text or text.startswith('!') or text.startswith('#'):
                if text.startswith('!'):
                    headers.append(text)
                continue
            n_data += 1
            m = MAPPING_RE.match(text)
            if not m:
                if len(bad_lines) < 6:
                    bad_lines.append(f'{ln}: {text[:180]}')
                continue
            pairs[m.group(1)].add(m.group(2))
    if n_data == 0 or bad_lines:
        raise ValueError(f'InterPro2GO input has malformed rows: {path}\n' + '\n'.join(bad_lines))
    return dict(pairs), {'file': str(path), 'n_interpro': len(pairs),
                         'n_direct_pairs': sum(map(len, pairs.values())),
                         'header': headers[:8]}


def normalize_mapping(mapping: dict[str, set[str]], ont: Ontology):
    # 2022 GO vocabulary is the comparison universe. Future-only GO is audited,
    # not silently assigned into the trained vocabulary.
    normal = {}
    unmapped = set()
    for ipr, gos in mapping.items():
        present = set()
        for go in gos:
            canonical = ont.canonical(go)
            if canonical is None:
                unmapped.add(go)
            else:
                present.add(canonical)
        normal[ipr] = present
    return normal, sorted(unmapped)


def map_go(iprs: set[str], mapping: dict[str, set[str]]) -> set[str]:
    return set().union(*(mapping.get(i, set()) for i in iprs)) if iprs else set()


def load_model_terms(paths: dict, ont: Ontology) -> dict[str, set[str]]:
    from build_dark_weak_test import load_go_vocab
    vocab = {}
    for ns, path in paths.items():
        if path:
            ids = load_go_vocab(path, ont, ns)
            vocab[ns] = {ont.canonical(i) for i in ids}
    return vocab


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'interpro2go_pairs.tsv').exists() and not args.overwrite:
        raise FileExistsError(f'Output exists: {out}; pass --overwrite for this probe subdirectory only')
    old = read_table(args.cohort_2022)
    full_old = read_table(args.uniprot_2022)
    future = read_table(args.uniprot_2025)
    match = pd.read_csv(args.match_index, sep='\t')
    for label, frame, cols in [
        ('cohort', old, ['proteins','sequences','interpros','annotations']),
        ('full 2022 reference', full_old, ['proteins','sequences','annotations']),
        ('future', future, ['proteins','sequences','interpros']),
        ('matches', match, ['protein_22','rid25','match_status'])]:
        missing = set(cols) - set(frame.columns)
        if missing:
            raise ValueError(f'{label} missing {sorted(missing)}')
    if old['proteins'].duplicated().any():
        raise ValueError('2022 cohort has duplicate protein entry names')
    if match['protein_22'].duplicated().any():
        raise ValueError('Match index has duplicate historical rows')

    ont = Ontology(args.go_obo_2022, mode=args.propagation)
    map0_raw, map0_meta = read_mapping(args.mapping_2022)
    map1_raw, map1_meta = read_mapping(args.mapping_2025)
    map0, bad0 = normalize_mapping(map0_raw, ont)
    map1, bad1 = normalize_mapping(map1_raw, ont)
    vocab = load_model_terms({'bp':args.bp_terms,'mf':args.mf_terms,'cc':args.cc_terms}, ont)
    old_by_name = old.set_index('proteins', drop=False)
    identical = match.loc[match['match_status'].eq('identical')].copy()
    missing_old = sorted(set(identical['protein_22']) - set(old_by_name.index))
    if missing_old:
        raise ValueError(f'Unresolved historical protein IDs: {missing_old[:10]}')
    if identical['rid25'].isna().any():
        raise ValueError('Identical match missing rid25')
    # Gather historical GO on all identical-sequence Swiss-Prot records. This
    # is intentionally the same historical evidence scope as main builder v1.
    target_sequences = {clean_seq(v) for v in old_by_name.loc[identical['protein_22'], 'sequences']}
    target_sequences.discard(None)
    historic_by_seq = defaultdict(set)
    for seq, annotations in zip(full_old['sequences'], full_old['annotations']):
        s0 = clean_seq(seq)
        if s0 in target_sequences:
            historic_by_seq[s0].update(
                c for g in coded_go(annotations) if (c := ont.canonical(g))
            )
    historic_closure_by_seq = {s:ont.propagate(v) for s,v in historic_by_seq.items()}

    # Only audit historical GO from the same protein, not a claim that all old
    # Swiss-Prot or TrEMBL evidence has been exhausted.
    pairs = []
    protein_rows = []
    counts = Counter()
    for row in identical.itertuples(index=False):
        historical = old_by_name.loc[row.protein_22]
        rid25 = int(row.rid25)
        if not 0 <= rid25 < len(future):
            raise IndexError(f'Invalid future row index {rid25}')
        later = future.iloc[rid25]
        seq0, seq1 = clean_seq(historical['sequences']), clean_seq(later['sequences'])
        if not seq0 or seq0 != seq1:
            raise ValueError(f'Match-index mismatch for {row.protein_22} at rid25={rid25}')
        i0, i1 = ipr_ids(historical['interpros']), ipr_ids(later['interpros'])
        g00 = map_go(i0, map0)
        g01 = map_go(i0, map1)  # mapping-only counterfactual
        g10 = map_go(i1, map0)  # match-only counterfactual
        g11 = map_go(i1, map1)
        delta = g11 - g00
        old_ipr_closure = ont.propagate(g00)
        history_go = historic_by_seq.get(seq0, set())
        history_closure = historic_closure_by_seq.get(seq0, set())
        new_exp_go = set()
        if 'annotations' in future.columns:
            new_exp_go = {c for g,codes in coded_go(later['annotations']).items()
                          if codes & EXPERIMENTAL_CODES
                          for c in [ont.canonical(g)] if c}
        protein_rows.append({
            'sequence_sha256': hashlib.sha256(seq0.encode()).hexdigest(),
            'protein_22':row.protein_22, 'protein_25':later['proteins'],
            'rid25':rid25, 'n_ipr_22':len(i0), 'n_ipr_25':len(i1),
            'n_go_22':len(g00), 'n_go_25':len(g11),
            'n_added_direct_go':len(delta),
            'n_added_not_implied_by_old_ipr':len(delta-old_ipr_closure),
            'n_new_ipr':len(i1-i0), 'n_retired_ipr':len(i0-i1),
            'n_old_go_annotations':len(history_go),
        })
        for g in sorted(delta):
            ns = NS.get(ont.terms[g].namespace)
            if not ns:
                continue
            contributors = {ipr for ipr in i1 if g in map1.get(ipr, set())}
            new_hits = contributors & (i1 - i0)
            old_hits = contributors & i0
            updated_mappings = {ipr for ipr in old_hits if g not in map0.get(ipr, set())}
            if g in g01 and g in g10:
                mode='mapping_and_match_counterfactual'
            elif g in g01:
                mode='mapping_only_counterfactual'
            elif g in g10:
                mode='match_only_counterfactual'
            else:
                mode='joint_or_lost_match_counterfactual'
            in_vocab = g in vocab[ns] if ns in vocab else None
            pairs.append({
                'protein_22':row.protein_22, 'protein_25':later['proteins'],
                'rid25':rid25, 'namespace':ns, 'go_id_2022':g,
                'supporting_ipr_2025':';'.join(sorted(contributors)),
                'new_matched_ipr':';'.join(sorted(new_hits)),
                'existing_ipr_new_mapping':';'.join(sorted(updated_mappings)),
                'counterfactual_type':mode,
                'new_vs_ipr22':True,
                'not_implied_by_2022_ipr2go':g not in old_ipr_closure,
                'not_in_2022_self_go_closure':g not in history_closure,
                'future_exp_same_go':g in new_exp_go,
                'in_model_vocab':in_vocab,
            })
            counts[ns] += 1
    pframe = pd.DataFrame(protein_rows)
    cols = ['protein_22','protein_25','rid25','namespace','go_id_2022',
            'supporting_ipr_2025','new_matched_ipr','existing_ipr_new_mapping',
            'counterfactual_type','new_vs_ipr22','not_implied_by_2022_ipr2go',
            'not_in_2022_self_go_closure',
            'future_exp_same_go','in_model_vocab']
    df = pd.DataFrame(pairs, columns=cols)
    pframe.to_csv(out / 'protein_summary.tsv', sep='\t', index=False)
    df.to_csv(out / 'interpro2go_pairs.tsv', sep='\t', index=False)
    stats = []
    for ns in ('bp','mf','cc'):
        subset = df[df['namespace'].eq(ns)]
        if subset.empty:
            stats.append({'namespace':ns,'proteins':0,'direct_pairs':0,'unique_go':0,
                          'vocab_pairs':0,'history_unrecorded_vocab_pairs':0,'history_unrecorded_vocab_proteins':0,
                          'future_experimental_supported_pairs':0})
            continue
        available = subset[subset['in_model_vocab'].eq(True)] if ns in vocab else subset.iloc[0:0]
        strict = available[available['not_in_2022_self_go_closure'] &
                           available['not_implied_by_2022_ipr2go']]
        stats.append({'namespace':ns,
                      'proteins':subset['protein_22'].nunique(),
                      'direct_pairs':len(subset),
                      'unique_go':subset['go_id_2022'].nunique(),
                      'vocab_pairs':len(available),
                      'history_unrecorded_vocab_pairs':len(strict),
                      'history_unrecorded_vocab_proteins':strict['protein_22'].nunique(),
                      'future_experimental_supported_pairs':int(subset['future_exp_same_go'].sum())})
    pd.DataFrame(stats).to_csv(out / 'namespace_summary.tsv', sep='\t', index=False)
    breakdown = df['counterfactual_type'].value_counts().to_dict() if not df.empty else {}
    manifest = {
        'task':'Historical InterPro2GO incremental annotation pilot',
        'description':'Pure automatic future mapping support; NOT independent experimental truth',
        'input_cohort':str(args.cohort_2022),'input_full_2022':str(args.uniprot_2022),
        'input_future':str(args.uniprot_2025),
        'input_matches':str(args.match_index),
        'mapping_2022':map0_meta,'mapping_2025':map1_meta,
        'go_ontology_2022':str(args.go_obo_2022),
        'match_status_included':'identical',
        'n_proteins_checked':len(pframe),
        'n_proteins_with_any_new_ipr2go':int((pframe['n_added_direct_go']>0).sum()),
        'n_added_go_pairs':len(df),
        'n_distinct_sequences_checked':int(pframe['sequence_sha256'].nunique()),
        'n_no_interpro_match_in_2022':int((pframe['n_ipr_22']==0).sum()),
        'n_no_interpro_match_in_2025':int((pframe['n_ipr_25']==0).sum()),
        'n_pairs_not_in_same_seq_historical_go':int(df['not_in_2022_self_go_closure'].sum()) if not df.empty else 0,
        'n_pairs_not_implied_by_2022_ipr2go':int(df['not_implied_by_2022_ipr2go'].sum()) if not df.empty else 0,
        'counterfactual_breakdown':breakdown,
        'unmapped_old_go_ids':bad0,
        'unmapped_new_go_ids':bad1,
        'historical_evidence_scope':'All identical-sequence 2022 Swiss-Prot entries',
        'vocab_go_order_modified':False,
        'warnings':['Not a ground-truth experimental benchmark',
                    'InterPro DR accessions in UniProt DAT lack match coordinates or confidence',
                    'Novelty check uses all identical-sequence 2022 Swiss-Prot annotations, not TrEMBL',
                    'No automatic label matrices are generated; audit candidate labels before formal evaluation']
    }
    (out / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('n_proteins_checked','n_proteins_with_any_new_ipr2go','n_added_go_pairs', 'n_no_interpro_match_in_2022', 'counterfactual_breakdown')},indent=2))
    print(pd.DataFrame(stats).to_string(index=False))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort-2022',type=Path,required=True)
    p.add_argument('--uniprot-2022',type=Path,required=True, help='Full historical Swiss-Prot table for identical-sequence GO evidence')
    p.add_argument('--uniprot-2025',type=Path,required=True)
    p.add_argument('--match-index',type=Path,required=True)
    p.add_argument('--mapping-2022',type=Path,required=True)
    p.add_argument('--mapping-2025',type=Path,required=True)
    p.add_argument('--go-obo-2022',type=Path,required=True)
    p.add_argument('--bp-terms',type=Path)
    p.add_argument('--mf-terms',type=Path)
    p.add_argument('--cc-terms',type=Path)
    p.add_argument('--propagation',choices=('deepgozero','is_a_part_of','is_a'),default='deepgozero')
    p.add_argument('--output-dir',type=Path,default=Path('data/dark_weak_test/interpro2go_probe'))
    p.add_argument('--overwrite',action='store_true')
    run(p.parse_args())

if __name__=='__main__':
    main()
