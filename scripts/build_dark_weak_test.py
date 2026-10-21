#!/usr/bin/env python3
"""Build the LATENCE historical dark-weak, future-GO test dataset.

Sequences and GO annotations are taken from the DeepGOZero-style pandas
DataFrames extracted from archived Swiss-Prot .dat.gz files. The script
NEVER chooses a historical/future identity match based on GO annotations.

The first item in the DataFrame 'accessions' field is not guaranteed to be
UniProt's primary accession: the published DeepGOZero uni2pandas.py parser
replaces the value for each AC line. Optional --dat-* inputs recover the
actual first AC token for an independently verified strict subset; their
sequence content is NOT used.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

# This is deliberately consistent with DeepGOZero utils.py, including TAS/IC.
LATENCE_CODES = frozenset('EXP IDA IPI IMP IGI IEP TAS IC HTP HDA HMP HGI HEP'.split())
# GO evidence types actually classified as experimental (including high throughput).
EXPERIMENTAL_CODES = frozenset('EXP IDA IPI IMP IGI IEP HTP HDA HMP HGI HEP'.split())
NS = {'biological_process': 'bp', 'molecular_function': 'mf',
      'cellular_component': 'cc'}
GO_RE = re.compile(r'GO:\d{7}')
RAW_ANNOT_RE = re.compile(r'(GO:\d{7})\|([A-Za-z_]+)')
AA_RE = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$')
ROOTS = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}


def clean_seq(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    seq = ''.join(value.split()).upper().rstrip('*')
    return seq if seq and AA_RE.fullmatch(seq) else None


def list_items(value: Any) -> list[str]:
    if value is None or (isinstance(value, (float, np.floating)) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value[0] in '[({' and value[-1] in '])}':
            try:
                result = ast.literal_eval(value)
                if isinstance(result, (list, tuple, set)):
                    return [str(v) for v in result]
            except (ValueError, SyntaxError):
                pass
        return [value]
    return []


def go_ids(value: Any) -> set[str]:
    return {go for item in list_items(value) for go in GO_RE.findall(item)}


def coded_go(value: Any) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for item in list_items(value):
        for go, code in RAW_ANNOT_RE.findall(item):
            out[go].add(code.upper())
    return dict(out)


def ac_tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return list(dict.fromkeys(x.strip() for x in value.split(';') if x.strip()))


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in ('.pkl', '.pickle'):
        df = pd.read_pickle(path)
    elif suffix == '.parquet':
        df = pd.read_parquet(path)
    elif suffix == '.csv':
        df = pd.read_csv(path)
    elif suffix in ('.tsv', '.txt'):
        df = pd.read_csv(path, sep='\t')
    else:
        raise ValueError(f'Unsupported input format {path}; use pkl/parquet/csv/tsv')
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Expected pandas DataFrame from {path}')
    return df.reset_index(drop=True).copy()


def check_required(df: pd.DataFrame, columns: list[str], label: str):
    missing = [x for x in columns if x not in df.columns]
    if missing:
        raise ValueError(f'{label} missing columns: {missing}')


def is_weak(row: Any, has_exp: bool, has_raw: bool) -> bool:
    if has_exp:
        return not go_ids(row.exp_annotations)
    if has_raw:
        codes = coded_go(row.annotations)
        return not any(bool(c & LATENCE_CODES) for c in codes.values())
    raise ValueError('Need exp_annotations or annotations to identify weak proteins')


def prepare_cohort(args, full22: pd.DataFrame | None, warnings: list[str]) -> pd.DataFrame:
    if args.cohort_2022 is not None:
        cohort = read_table(args.cohort_2022)
        provided_prefiltered = True
    elif full22 is not None:
        cohort = full22.copy()
        provided_prefiltered = False
    else:
        raise ValueError('Provide --cohort-2022 or --uniprot-2022')
    check_required(cohort, ['proteins', 'accessions', 'sequences'], '2022 cohort')
    if 'dark_matter' not in cohort.columns:
        if args.darkness_2022 is not None:
            dark = read_table(args.darkness_2022)
            check_required(dark, ['proteins', 'dark_matter'], 'darkness table')
            if dark['proteins'].duplicated().any():
                raise ValueError('Darkness table has duplicate entry names; cannot join safely')
            before = len(cohort)
            cohort = cohort.merge(dark[['proteins', 'dark_matter']], on='proteins',
                                  how='left', validate='many_to_one')
            n_na = int(cohort['dark_matter'].isna().sum())
            if n_na:
                warnings.append(f'{n_na}/{before} rows missing darkness classification; excluded')
        elif provided_prefiltered:
            warnings.append('--cohort-2022 has no dark_matter field; trusting it is already dark-filtered')
            cohort['dark_matter'] = True
        else:
            raise ValueError('Full 2022 table needs dark_matter or --darkness-2022')
    cohort = cohort.loc[cohort['dark_matter'].fillna(False).astype(bool)].copy()
    dark_input_count = len(cohort)
    if 'exp_annotations' not in cohort.columns and 'annotations' not in cohort.columns:
        raise ValueError('2022 cohort must contain exp_annotations or annotations')
    exp_col = 'exp_annotations' in cohort.columns
    raw_col = 'annotations' in cohort.columns
    cohort['_weak'] = [is_weak(r, exp_col, raw_col) for r in cohort.itertuples(index=False)]
    cohort = cohort.loc[cohort['_weak']].copy().reset_index(drop=True)
    cohort.insert(0, 'rid22', np.arange(len(cohort), dtype=np.int32))
    if cohort['proteins'].duplicated().any():
        raise ValueError('2022 cohort contains duplicate UniProt entry names; disambiguate source first')
    cohort['sequence_clean'] = cohort['sequences'].map(clean_seq)
    cohort['tokens'] = cohort['accessions'].map(ac_tokens)
    cohort.attrs['dark_input_count'] = dark_input_count
    cohort.attrs['dark_input_is_complete'] = not provided_prefiltered
    return cohort


@dataclass
class GOTerm:
    go_id: str
    name: str = ''
    namespace: str = ''
    parents: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    alt_ids: list[str] = field(default_factory=list)
    obsolete: bool = False
    replaced_by: list[str] = field(default_factory=list)
    consider: list[str] = field(default_factory=list)


class Ontology:
    def __init__(self, path: Path, mode: str = 'deepgozero'):
        self.path = str(path)
        self.mode = mode
        self.terms: dict[str, GOTerm] = {}
        self.alt: dict[str, str] = {}
        self._closure: dict[str, set[str]] = {}
        self._parse(path)

    def _parse(self, path: Path):
        current: GOTerm | None = None
        in_term = False

        def flush():
            if current is not None and current.go_id:
                self.terms[current.go_id] = current

        with path.open('r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if line == '[Term]':
                    flush()
                    current, in_term = GOTerm(''), True
                    continue
                if line.startswith('[') and line.endswith(']'):
                    flush()
                    current, in_term = None, False
                    continue
                if not in_term or not current or ': ' not in line:
                    continue
                key, value = line.split(': ', 1)
                if key == 'id':
                    current.go_id = value
                elif key == 'name':
                    current.name = value
                elif key == 'namespace':
                    current.namespace = value
                elif key == 'alt_id':
                    current.alt_ids.append(value)
                elif key == 'is_obsolete':
                    current.obsolete = value == 'true'
                elif key == 'is_a':
                    current.parents['is_a'].append(value.split(' ! ')[0])
                elif key == 'relationship':
                    parts = value.split()
                    if len(parts) >= 2 and parts[1].startswith('GO:'):
                        current.parents[parts[0]].append(parts[1])
                elif key == 'replaced_by':
                    current.replaced_by.append(value)
                elif key == 'consider':
                    current.consider.append(value)
            flush()
        for go, term in self.terms.items():
            for alt in term.alt_ids:
                if alt in self.alt and self.alt[alt] != go:
                    raise ValueError(f'GO alt_id collision: {alt}')
                self.alt[alt] = go

    def canonical(self, go: str) -> str | None:
        canonical = self.alt.get(go, go)
        term = self.terms.get(canonical)
        return canonical if term and not term.obsolete else None

    def classify(self, go: str, future: 'Ontology | None' = None) -> tuple[str, str | None, str]:
        old = self.canonical(go)
        if old:
            return ('mapped_alt_id' if old != go else 'mapped', old, '')
        t = self.terms.get(self.alt.get(go, go))
        if t and t.obsolete:
            return ('obsolete_2022', None, ','.join(t.replaced_by))
        if future:
            newer = future.canonical(go)
            if newer:
                return ('absent_2022_present_2025', None, newer)
            fterm = future.terms.get(future.alt.get(go, go))
            if fterm and fterm.obsolete:
                return ('obsolete_2025', None, ','.join(fterm.replaced_by))
        return ('unmapped_unknown', None, '')

    def closure(self, go: str) -> set[str]:
        go = self.canonical(go)
        if go is None:
            return set()
        if go in self._closure:
            return self._closure[go]
        seen = set()
        stack = [go]
        while stack:
            term_id = stack.pop()
            term_id = self.canonical(term_id)
            if not term_id or term_id in seen:
                continue
            seen.add(term_id)
            term = self.terms[term_id]
            for relation, target_ids in term.parents.items():
                if (self.mode == 'is_a' and relation != 'is_a') or (
                    self.mode == 'is_a_part_of' and relation not in {'is_a', 'part_of'}
                ):
                    continue
                stack.extend(target_ids)
        self._closure[go] = seen
        return seen

    def propagate(self, gos: set[str]) -> set[str]:
        out: set[str] = set()
        for x in gos:
            out.update(self.closure(x))
        return out


def build_pair_candidates(cohort: pd.DataFrame, df25: pd.DataFrame,
                          include_sequence_only: bool = False) -> dict[int, set[int]]:
    """Join by shared list accession or entry name, not future annotations.

    Sequence-only matches are opt-in: identical sequences may be present in
    different proteins or taxa and need explicit audit.
    """
    by_token = defaultdict(set)
    by_name = defaultdict(set)
    for i, (name, toks) in enumerate(zip(cohort['proteins'], cohort['tokens'])):
        by_name[str(name)].add(i)
        for token in toks:
            by_token[token].add(i)
    pairs: dict[int, set[int]] = defaultdict(set)
    for j, (name, token_string) in enumerate(zip(df25['proteins'], df25['accessions'])):
        ids = set(by_name.get(str(name), ()))
        for token in ac_tokens(token_string):
            ids.update(by_token.get(token, ()))
        for i in ids:
            pairs[i].add(j)
    if include_sequence_only:
        by_seq = defaultdict(set)
        for i, seq in enumerate(cohort['sequence_clean']):
            if seq:
                by_seq[seq].add(i)
        for j, seq in enumerate(df25['sequences']):
            s = clean_seq(seq)
            if s:
                for i in by_seq.get(s, ()):
                    if i not in pairs or not pairs[i]:
                        pairs[i].add(j)
    return pairs


def resolve_pairs(cohort: pd.DataFrame, df25: pd.DataFrame,
                  pairs: dict[int, set[int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_records = []
    ambiguous_pairs = []
    names25 = df25['proteins'].astype(str).tolist()
    seq25 = df25['sequences'].tolist()
    acc25 = df25['accessions'].tolist()
    raw25_tokens_cache: dict[int, set[str]] = {}
    for r in cohort.itertuples(index=False):
        i = int(r.rid22)
        candidates = pairs.get(i, set())
        old_tokens = set(r.tokens)
        scores = []
        for j in candidates:
            seq = clean_seq(seq25[j])
            name_match = str(r.proteins) == names25[j]
            if j not in raw25_tokens_cache:
                raw25_tokens_cache[j] = set(ac_tokens(acc25[j]))
            new_tokens = raw25_tokens_cache[j]
            overlap = len(old_tokens & new_tokens)
            first = bool(r.tokens) and bool(new_tokens) and bool(ac_tokens(acc25[j])) and r.tokens[0] == ac_tokens(acc25[j])[0]
            equal_seq = (r.sequence_clean is not None and seq is not None and
                         r.sequence_clean == seq)
            # No annotation-dependent evidence is involved in ranking.
            score = (int(equal_seq), int(name_match), int(first), overlap)
            scores.append((score, j, name_match, equal_seq, overlap, first))
        if not scores:
            match_records.append({'rid22': i, 'rid25': pd.NA, 'match_status': 'unmatched',
                                  'match_method': 'none', 'n_candidates': 0,
                                  'n_top_ties': 0, 'entry_name_changed': pd.NA,
                                  'n_shared_accessions': 0, 'first_listed_equal': False})
            continue
        top_score = max(x[0] for x in scores)
        best = sorted((x for x in scores if x[0] == top_score), key=lambda x: x[1])
        if len(best) != 1:
            for x in best:
                ambiguous_pairs.append({'rid22': i, 'rid25': x[1], 'score': str(x[0])})
            match_records.append({'rid22': i, 'rid25': pd.NA, 'match_status': 'ambiguous',
                                  'match_method': 'tie', 'n_candidates': len(scores),
                                  'n_top_ties': len(best), 'entry_name_changed': pd.NA,
                                  'n_shared_accessions': 0, 'first_listed_equal': False})
            continue
        _, j, name_match, equal_seq, overlap, first = best[0]
        if equal_seq:
            status = 'identical'
        elif r.sequence_clean is None or clean_seq(seq25[j]) is None:
            status = 'unresolved_sequence'
        else:
            status = 'changed'
        method = 'name_and_ac' if name_match and overlap else (
                 'name_only' if name_match else ('ac_and_seq' if overlap and equal_seq else (
                 'ac_only' if overlap else 'sequence_only')))
        match_records.append({'rid22': i, 'rid25': j, 'match_status': status,
                              'match_method': method, 'n_candidates': len(scores),
                              'n_top_ties': 1, 'entry_name_changed': not name_match,
                              'n_shared_accessions': overlap, 'first_listed_equal': bool(first)})
    return pd.DataFrame(match_records), pd.DataFrame(ambiguous_pairs,
                columns=['rid22', 'rid25', 'score'])


def read_raw_dat_primary(path: Path, target_names: set[str]) -> dict[str, str]:
    """Optional audit of original UniProt primary IDs; sequences still use pandas."""
    opener = gzip.open if str(path).endswith('.gz') else open
    output = {}
    current = None
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.startswith('ID   '):
                current = line[5:].split()[0]
            elif current in target_names and line.startswith('AC   '):
                if current not in output:
                    output[current] = line[5:].split(';', 1)[0].strip()
            elif line.startswith('//'):
                current = None
    return output


def evidence25_for(row: pd.Series, raw_col='annotations') -> tuple[set[str], set[str], dict[str, set[str]]]:
    raw = coded_go(row.get(raw_col))
    latence = go_ids(row.get('exp_annotations')) if 'exp_annotations' in row else {
        g for g, ev in raw.items() if ev & LATENCE_CODES}
    strict = {g for g, ev in raw.items() if ev & EXPERIMENTAL_CODES}
    return latence, strict, raw


def load_go_vocab(path: Path, ontology: Ontology, namespace: str) -> list[str]:
    if path.suffix.lower() == '.npy':
        terms = np.load(path, allow_pickle=False).tolist()
    elif path.suffix.lower() in ('.pkl', '.pickle'):
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.Series):
            terms = obj.to_list()
        elif isinstance(obj, (list, tuple, np.ndarray)):
            terms = list(obj)
        elif isinstance(obj, pd.DataFrame) and any(x in obj.columns for x in ('go_id', 'gos', 'terms')):
            terms = obj[next(x for x in ('go_id', 'gos', 'terms') if x in obj.columns)].tolist()
        elif isinstance(obj, dict) and any(x in obj for x in ('go_id', 'gos', 'terms')):
            terms = list(obj[next(x for x in ('go_id', 'gos', 'terms') if x in obj)])
        else:
            raise ValueError(f'Unsupported pickle vocabulary object in {path}')
    else:
        text = path.read_text(encoding='utf-8')
        terms = [m.group(0) for m in GO_RE.finditer(text)]
    ids = [str(v) for v in terms]
    if not ids:
        raise ValueError(f'No GO IDs in vocabulary: {path}')
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate GO IDs in vocabulary {path}')
    for term in ids:
        canon = ontology.canonical(term)
        if canon is None:
            raise ValueError(f'Vocabulary includes GO absent/obsolete in 2022 ontology: {term}')
        ns = NS.get(ontology.terms[canon].namespace)
        if ns != namespace:
            raise ValueError(f'Vocabulary {namespace} has {term} in {ns}')
    # Preserve the checkpoint's *original GO order and width*. GO alt_ids can
    # legitimately coexist with their canonical IDs in a legacy task vocabulary.
    # They are not repeated raw IDs, and removing one would misalign logits.
    # write_matrices() maps each canonical GO to ALL corresponding model columns.
    return ids


def write_tsv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def write_lines(values, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(str(x) + '\n' for x in values), encoding='utf-8')


def table_report(df: pd.DataFrame) -> str:
    if df.empty:
        return '_None._\n'
    cols = list(df.columns)
    def cell(v):
        return str(v).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for values in df.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(cell(v) for v in values) + ' |')
    return '\n'.join(lines) + '\n'


def write_matrices(proteins: pd.DataFrame, records: list[dict], vocab_paths: dict,
                   ont: Ontology, root: Path, warnings: list[str],
                   full_historical_evidence: bool) -> list[dict]:
    try:
        from scipy import sparse
    except ImportError:
        if any(vocab_paths.values()):
            raise RuntimeError('scipy is required when task vocabularies are supplied')
        return []
    stats = []
    if proteins.empty:
        return stats
    protein_ids = proteins['rid22'].astype(int).tolist()
    rows_by_id = {p: i for i, p in enumerate(protein_ids)}
    record_by_id = {r['rid22']: r for r in records}
    for namespace, vocab_path in vocab_paths.items():
        if vocab_path is None:
            continue
        vocab = load_go_vocab(vocab_path, ont, namespace)
        # One biological GO may occupy multiple model columns when an alt_id
        # and its canonical ID are BOTH present in the checkpoint vocabulary.
        # A simple dict comprehension would silently keep only the last index.
        canonical_to_columns: dict[str, list[int]] = defaultdict(list)
        for col, term in enumerate(vocab):
            canonical_to_columns[ont.canonical(term)].append(col)
        collisions = {term: cols for term, cols in canonical_to_columns.items()
                      if len(cols) > 1}
        duplicate_columns = sum(len(cols) - 1 for cols in collisions.values())
        output = root / 'labels' / namespace
        output.mkdir(parents=True, exist_ok=True)
        # Keep the exact original ordering and width, with no dedup/reordering.
        write_lines(vocab, output / 'go_ids.txt')
        vocab_audit = []
        for col, term in enumerate(vocab):
            canonical = ont.canonical(term)
            vocab_audit.append({
                'model_column': col,
                'model_go_id': term,
                'canonical_go_id_2022': canonical,
                'is_alt_id': term != canonical,
                'canonical_group_size': len(canonical_to_columns[canonical]),
            })
        write_tsv(pd.DataFrame(vocab_audit), output / 'vocabulary_mapping.tsv')
        write_tsv(pd.DataFrame(
            [r for r in vocab_audit if r['canonical_group_size'] > 1],
            columns=list(vocab_audit[0]),
        ), output / 'vocabulary_alias_collisions.tsv')
        if collisions:
            examples = '; '.join(
                f"{canonical} <- {','.join(vocab[i] for i in cols)}"
                for canonical, cols in list(collisions.items())[:5]
            )
            msg = (f'{namespace}: {len(collisions)} canonical GO alias collision groups '
                   f'({duplicate_columns} extra model columns). Original order/width '
                   f'preserved; labels copied into each alias-equivalent column. '
                   f'See labels/{namespace}/vocabulary_alias_collisions.tsv. '
                   f'Examples: {examples}')
            warnings.append(msg)
            print('[GO vocabulary] WARNING: ' + msg, flush=True)
        write_lines(proteins['protein_22'], output / 'protein_ids.txt')
        row_md = proteins[['rid22', 'protein_22', 'protein_25', 'sequence_sha256',
                           'first_listed_accession_22', 'raw_primary_22',
                           'raw_primary_25', 'raw_primary_equal']].copy()
        row_md.insert(0, 'matrix_row', np.arange(len(row_md)))
        write_tsv(row_md, output / 'row_metadata.tsv')
        sets = {'direct': [], 'propagated': [], 'historically_unrecorded_direct': [],
                'historically_unrecorded_propagated': [],
                'strict_experimental_direct': [], 'strict_experimental_propagated': []}
        for pid in protein_ids:
            r = record_by_id[pid]
            direct = set(r['future_mapped'])
            exp = set(r['future_experimental_mapped'])
            historic = set(r['historical_same_seq_closure'])
            new_direct = direct - historic
            sets['direct'].append(direct)
            sets['propagated'].append(ont.propagate(direct))
            sets['historically_unrecorded_direct'].append(new_direct)
            sets['historically_unrecorded_propagated'].append(ont.propagate(new_direct) - historic)
            sets['strict_experimental_direct'].append(exp)
            sets['strict_experimental_propagated'].append(ont.propagate(exp))
        for key, go_sets in sets.items():
            if key.startswith('historically_unrecorded') and not full_historical_evidence:
                continue  # Partial history must not masquerade as proven functional novelty.
            rs, cs = [], []
            unique_canonical_positive_pairs = 0
            for ridx, terms in enumerate(go_sets):
                for go in terms:
                    cols = canonical_to_columns.get(go, ())
                    if cols:
                        unique_canonical_positive_pairs += 1
                        for col in cols:
                            rs.append(ridx)
                            cs.append(col)
            matrix = sparse.csr_matrix(
                (np.ones(len(rs), dtype=np.uint8), (rs, cs)),
                shape=(len(proteins), len(vocab)), dtype=np.uint8
            )
            sparse.save_npz(output / f'{key}.npz', matrix, compressed=True)
            stats.append({'namespace': namespace, 'matrix': key,
                          'n_proteins': int((np.asarray(matrix.sum(axis=1)).ravel() > 0).sum()),
                          'n_positive_pairs': int(matrix.nnz),
                          'n_canonical_positive_pairs': int(unique_canonical_positive_pairs),
                          'n_model_terms': len(vocab),
                          'n_unique_canonical_terms': len(canonical_to_columns),
                          'n_alias_collision_groups': len(collisions),
                          'n_alias_duplicate_columns': duplicate_columns})
    return stats


def build(args: argparse.Namespace) -> dict:
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f'{output} is nonempty. Use --overwrite to replace generated files.')
        if not (output / 'manifest.json').exists():
            known_generated = {'tables', 'labels', 'summary.json', 'manifest.json',
                 'report.md', 'dark_weak_test_strict.pkl', 'strict_protein_ids.txt',
                 'all_candidate_protein_ids.txt'}
            unknown = {path.name for path in output.iterdir()} - known_generated
            if unknown:
                raise FileExistsError(f'Refusing to overwrite unrecognized files in {output}: {sorted(unknown)}')
        for sub in ('tables', 'labels'):
            if (output / sub).exists():
                shutil.rmtree(output / sub)
        for filename in ('manifest.json', 'summary.json', 'report.md',
                         'dark_weak_test_strict.pkl', 'strict_protein_ids.txt',
                         'all_candidate_protein_ids.txt'):
            (output / filename).unlink(missing_ok=True)
    (output / 'tables').mkdir(parents=True, exist_ok=True)
    (output / 'labels').mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    full22 = read_table(args.uniprot_2022) if args.uniprot_2022 else None
    if full22 is not None:
        check_required(full22, ['proteins', 'sequences'], 'full 2022 Swiss-Prot')
    df25 = read_table(args.uniprot_2025)
    check_required(df25, ['proteins', 'accessions', 'sequences'], '2025 Swiss-Prot')
    df25.insert(0, 'rid25', np.arange(len(df25), dtype=np.int32))
    cohort = prepare_cohort(args, full22, warnings)
    if args.label_policy == 'experimental' and 'annotations' not in df25.columns:
        raise ValueError('experimental policy requires raw 2025 annotations (GO|evidence)')
    if 'exp_annotations' not in df25.columns and 'annotations' not in df25.columns:
        raise ValueError('2025 table requires exp_annotations or annotations')
    if 'annotations' not in cohort.columns:
        warnings.append('Cohort lacks raw 2022 annotations; history-knowledge comparisons are incomplete')
    if full22 is None:
        warnings.append('No full 2022 Swiss-Prot supplied; same-sequence historical evidence is limited to the supplied cohort')
    elif 'annotations' not in full22.columns:
        warnings.append('Full 2022 Swiss-Prot lacks raw annotations; historical any-evidence checks are incomplete')
    historical_reference_complete = bool(full22 is not None and 'annotations' in full22.columns)
    if full22 is not None and len(cohort):
        if full22['proteins'].duplicated().any():
            historical_reference_complete = False
            warnings.append('Full 2022 reference contains duplicate entry names; historical novelty matrices withheld')
        else:
            reference_seq = full22.set_index('proteins')['sequences']
            missing_names = int((~cohort['proteins'].isin(reference_seq.index)).sum())
            seq_ref = cohort['proteins'].map(reference_seq).map(clean_seq)
            disagree = int((seq_ref.notna() & cohort['sequence_clean'].ne(seq_ref)).sum())
            if missing_names or disagree:
                historical_reference_complete = False
                warnings.append(f'Full 2022 reference misses {missing_names} cohort names and disagrees on {disagree} sequences; historical novelty matrices withheld')
    if 'annotations' not in df25.columns:
        warnings.append('2025 annotations lacks evidence codes; strict experimental labels cannot be audited')
    if args.enable_sequence_only:
        warnings.append('Sequence-only fallback enabled; such matches may conflate entries with identical sequence')

    ont = Ontology(args.go_obo_2022, mode=args.propagation)
    ont25 = Ontology(args.go_obo_2025) if args.go_obo_2025 else None
    # Fail fast for truly malformed vocabularies, before expensive 2022->2025
    # identity matching and historical GO analysis. Canonical alt_id collisions
    # are permitted and audited by write_matrices().
    for ns, path in {'bp': args.bp_terms, 'mf': args.mf_terms,
                     'cc': args.cc_terms}.items():
        if path is not None:
            load_go_vocab(path, ont, ns)
    print(f'[1/5] cohort={len(cohort):,}, future={len(df25):,}, GO terms={len(ont.terms):,}', flush=True)
    pairs = build_pair_candidates(cohort, df25, args.enable_sequence_only)
    matches, ambiguous_pairs = resolve_pairs(cohort, df25, pairs)
    write_tsv(ambiguous_pairs, output / 'tables' / 'ambiguous_match_pairs.tsv')
    cohort_base = cohort[['rid22', 'proteins', 'accessions', 'sequence_clean']].copy()
    cohort_base = cohort_base.rename(columns={'proteins': 'protein_22', 'accessions': 'accessions_22'})
    cohort_base['sequence_sha256'] = cohort_base['sequence_clean'].map(
        lambda s: hashlib.sha256(s.encode()).hexdigest() if s else '')
    matches = matches.merge(cohort_base.drop(columns=['sequence_clean']), on='rid22', validate='one_to_one')
    matched = matches.loc[matches['rid25'].notna()].copy()
    matched['rid25'] = matched['rid25'].astype(int)
    info25 = df25[['rid25', 'proteins', 'accessions']].rename(
        columns={'proteins': 'protein_25', 'accessions': 'accessions_25'})
    matched = matched.merge(info25, on='rid25', validate='many_to_one')
    matched['first_listed_accession_22'] = matched['accessions_22'].map(
        lambda x: ac_tokens(x)[0] if ac_tokens(x) else '')
    matched['first_listed_accession_25'] = matched['accessions_25'].map(
        lambda x: ac_tokens(x)[0] if ac_tokens(x) else '')

    if bool(args.dat_2022) != bool(args.dat_2025):
        raise ValueError('--dat-2022 and --dat-2025 must be provided together')
    if args.require_raw_primary and not args.dat_2022:
        raise ValueError('--require-raw-primary requires both --dat-* inputs')
    raw22 = raw25 = {}
    if args.dat_2022:
        print('[2/5] recovering optional original primary accessions from DAT records', flush=True)
        raw22 = read_raw_dat_primary(args.dat_2022, set(matched['protein_22']))
        raw25 = read_raw_dat_primary(args.dat_2025, set(matched['protein_25']))
    matched['raw_primary_22'] = matched['protein_22'].map(raw22).fillna('')
    matched['raw_primary_25'] = matched['protein_25'].map(raw25).fillna('')
    matched['raw_primary_equal'] = (matched['raw_primary_22'].ne('') &
                                    matched['raw_primary_22'].eq(matched['raw_primary_25']))

    def anno25(j):
        row = df25.iloc[int(j)]
        return evidence25_for(row)
    evidence25_cache = {}
    for j in matched['rid25'].unique():
        evidence25_cache[j] = anno25(j)
    matched['n_2025_latence_labels'] = matched['rid25'].map(
        lambda j: len(evidence25_cache[j][0]))
    matched['n_2025_strict_exp_labels'] = matched['rid25'].map(
        lambda j: len(evidence25_cache[j][1]))
    label_idx = 0 if args.label_policy == 'latence' else 1
    matched['n_2025_target_labels'] = matched['rid25'].map(
        lambda j: len(evidence25_cache[j][label_idx]))
    write_tsv(matches.drop(columns=['accessions_22']), output / 'tables' / 'cohort_match_index.tsv')

    selected = matched.loc[matched['n_2025_target_labels'] > 0].copy()
    selected['sequence_sha256'] = selected['sequence_sha256'].astype(str)
    selected['name_changed'] = selected['protein_22'].ne(selected['protein_25'])
    print(f'[3/5] future-labeled candidates={len(selected):,}; identical={int(selected.match_status.eq("identical").sum()):,}', flush=True)

    if args.dat_2022:
        strict_mask = selected['match_status'].eq('identical')
        if args.require_raw_primary:
            strict_mask &= selected['raw_primary_equal']
    else:
        strict_mask = selected['match_status'].eq('identical')
    selected['strict_eligible'] = strict_mask
    selected['exclusion_reason'] = np.where(strict_mask, '',
        np.where(selected['match_status'].eq('identical'), 'raw_primary_not_verified',
                 'sequence_' + selected['match_status'].astype(str)))

    # Count identical sequences in both snapshots; do not confuse with test-set duplicates.
    target_seqs = set(cohort.loc[cohort['rid22'].isin(selected['rid22']), 'sequence_clean'].dropna())
    seq25_counts = Counter()
    for seq in df25['sequences']:
        cleaned = clean_seq(seq)
        if cleaned in target_seqs:
            seq25_counts[cleaned] += 1
    selected['n_2025_same_sequence_entries'] = selected['rid22'].map(
        lambda i: seq25_counts.get(cohort.at[int(i), 'sequence_clean'], 0))
    selected['sequence_duplicate_2025'] = selected['n_2025_same_sequence_entries'] > 1
    if args.require_unique_2025_sequence:
        dup = selected['sequence_duplicate_2025'] & selected['strict_eligible']
        selected.loc[dup, 'strict_eligible'] = False
        selected.loc[dup, 'exclusion_reason'] = 'duplicate_sequence_2025'

    # For baseline knowledge, include ALL historical Swiss-Prot rows if available.
    history = full22 if full22 is not None else cohort
    history_has_annotations = 'annotations' in history.columns
    hist_codes_by_seq = defaultdict(lambda: defaultdict(set))
    hist_exp_by_seq = defaultdict(set)
    if history_has_annotations:
        target_seqs_clean = set(target_seqs)
        for seq, raw in zip(history['sequences'], history['annotations']):
            s = clean_seq(seq)
            if s in target_seqs_clean:
                for go, codes in coded_go(raw).items():
                    hist_codes_by_seq[s][go].update(codes)
    if 'exp_annotations' in history.columns:
        for seq, ann in zip(history['sequences'], history['exp_annotations']):
            s = clean_seq(seq)
            if s in target_seqs:
                hist_exp_by_seq[s].update(go_ids(ann))
    elif history_has_annotations:
        for seq, m in hist_codes_by_seq.items():
            hist_exp_by_seq[seq].update(g for g, ev in m.items() if ev & LATENCE_CODES)

    # Candidate-level label audit and ontology alignment.
    records: list[dict] = []
    go_pair_rows = []
    unmapped_rows = []
    for m in selected.itertuples(index=False):
        i, j = int(m.rid22), int(m.rid25)
        seq = cohort.at[i, 'sequence_clean']
        oldrow = cohort.iloc[i]
        new_lat, new_exp, raw25_coded = evidence25_cache[j]
        future_direct = new_lat if args.label_policy == 'latence' else new_exp
        old_codes = coded_go(oldrow.get('annotations'))
        old_exp_go = go_ids(oldrow.get('exp_annotations'))
        old_all_go = set(old_codes)
        same_codes = hist_codes_by_seq.get(seq, {})
        same_all = set(same_codes)
        same_exp = hist_exp_by_seq.get(seq, set())
        hist_same_mapped = {c for g in same_all if (c := ont.canonical(g))}
        hist_exp_mapped = {c for g in same_exp if (c := ont.canonical(g))}
        hist_self_mapped = {c for g in old_all_go if (c := ont.canonical(g))}
        same_closure = ont.propagate(hist_same_mapped)
        self_closure = ont.propagate(hist_self_mapped)
        exp_closure = ont.propagate(hist_exp_mapped)
        fut_mapped = {c for g in future_direct if (c := ont.canonical(g))}
        fut_exp_mapped = {c for g in new_exp if (c := ont.canonical(g))}
        records.append({
            'rid22': i, 'rid25': j, 'future_mapped': fut_mapped,
            'future_experimental_mapped': fut_exp_mapped,
            'historical_same_seq_closure': same_closure,
            'historical_self_closure': self_closure,
            'historical_same_seq_exp_closure': exp_closure,
        })
        for go in sorted(future_direct):
            map_status, canon, mapping_info = ont.classify(go, ont25)
            ns = NS.get(ont.terms[canon].namespace, 'other') if canon else 'unmapped'
            term = ont.terms[canon].name if canon else ''
            record = {
                'rid22': i, 'protein_22': m.protein_22,
                'protein_25': m.protein_25, 'match_status': m.match_status,
                'strict_eligible': bool(m.strict_eligible),
                'go_id_2025': go, 'canonical_go_2022': canon or '',
                'go_name_2022': term, 'namespace': ns,
                'ontology_mapping': map_status, 'ontology_detail': mapping_info,
                'evidence_codes_2025': ';'.join(sorted(raw25_coded.get(go, set()))),
                'is_strict_experimental': bool(go in new_exp),
                'historical_self_direct': go in old_all_go or bool(canon and canon in hist_self_mapped),
                'historical_self_implied': bool(canon and canon in self_closure),
                'historical_identical_seq_direct': go in same_all or bool(canon and canon in hist_same_mapped),
                'historical_identical_seq_implied': bool(canon and canon in same_closure),
                'historical_identical_seq_experimental_implied': bool(canon and canon in exp_closure),
                'historically_unrecorded_identical_seq': (bool(canon) and
                       historical_reference_complete and bool(canon not in same_closure)),
                'historical_reference_complete': historical_reference_complete,
            }
            go_pair_rows.append(record)
            if not canon:
                unmapped_rows.append(record.copy())
    go_pairs = pd.DataFrame(go_pair_rows)
    go_unmapped = pd.DataFrame(unmapped_rows, columns=go_pairs.columns)
    selected = selected.sort_values(['rid22', 'rid25']).reset_index(drop=True)
    write_tsv(selected.drop(columns=['strict_eligible']).assign(
        strict_eligible=selected['strict_eligible']), output / 'tables' / 'future_labeled_candidates.tsv')
    write_tsv(selected.loc[~selected['strict_eligible']], output / 'tables' / 'excluded_labeled_candidates.tsv')
    write_tsv(go_pairs, output / 'tables' / 'future_go_pairs.tsv')
    write_tsv(go_unmapped, output / 'tables' / 'unmapped_go_pairs.tsv')
    hist_evidence = Counter()
    if not go_pairs.empty:
        for text_codes in go_pairs.loc[go_pairs['strict_eligible'], 'evidence_codes_2025']:
            for code in str(text_codes).split(';'):
                if code.strip():
                    hist_evidence[code.strip()] += 1
        if not hist_evidence:
            hist_evidence['not_available'] = 0
    write_tsv(pd.DataFrame([{'evidence_code': code, 'strict_positive_pairs': count}
         for code, count in sorted(hist_evidence.items())],
         columns=['evidence_code', 'strict_positive_pairs']),
         output / 'tables' / '2025_evidence_code_summary.tsv')
    if not go_unmapped.empty:
        missing_agg = go_unmapped.groupby(['go_id_2025', 'ontology_mapping', 'ontology_detail'],
                                            dropna=False).agg(
                        strict_positive_pairs=('strict_eligible', 'sum'),
                        proteins=('rid22', 'nunique')).reset_index()
    else:
        missing_agg = pd.DataFrame(columns=['go_id_2025','ontology_mapping',
                                             'ontology_detail','strict_positive_pairs','proteins'])
    write_tsv(missing_agg, output / 'tables' / 'unmapped_go_summary.tsv')

    strict = selected.loc[selected['strict_eligible']].copy()
    strict[['rid22', 'protein_22', 'protein_25', 'sequence_sha256',
            'match_method', 'name_changed', 'raw_primary_equal',
            'n_2025_same_sequence_entries']].to_csv(
        output / 'tables' / 'strict_proteins.tsv', sep='\t', index=False)
    # Preserve historical protein ids and sequences plus NEW labels, avoiding
    # converting missing annotations to confirmed negatives.
    by_rid = {r['rid22']: r for r in records}
    dataset_rows = []
    for m in strict.itertuples(index=False):
        i, j = int(m.rid22), int(m.rid25)
        lat, exp, _ = evidence25_cache[j]
        rec = by_rid[i]
        dataset_rows.append({
            'proteins': m.protein_22,
            'proteins_2025': m.protein_25,
            'accessions': m.accessions_22,
            'sequences': cohort.at[i, 'sequence_clean'],
            'exp_annotations': sorted(lat),  # LATENCE project compatibility
            'strict_experimental_annotations': sorted(exp),
            'future_labels_selected': sorted(lat if args.label_policy == 'latence' else exp),
            'future_mapped_2022': sorted(rec['future_mapped']),
            'future_propagated_2022': sorted(ont.propagate(rec['future_mapped'])),
            'historically_unrecorded_direct': (sorted(
                rec['future_mapped'] - rec['historical_same_seq_closure'])
                if historical_reference_complete else None),
            'rid22': i, 'rid25': j,
        })
    pd.DataFrame(dataset_rows, columns=['proteins', 'proteins_2025', 'accessions',
        'sequences', 'exp_annotations', 'strict_experimental_annotations',
        'future_labels_selected', 'future_mapped_2022', 'future_propagated_2022',
        'historically_unrecorded_direct', 'rid22', 'rid25']).to_pickle(
            output / 'dark_weak_test_strict.pkl')
    write_lines(strict['protein_22'], output / 'strict_protein_ids.txt')
    write_lines(selected['protein_22'], output / 'all_candidate_protein_ids.txt')

    # Inspect all 2025 entries with identical sequences to the strict targets.
    strictly_targeted = set(cohort.loc[cohort['rid22'].isin(strict['rid22']), 'sequence_clean'])
    twin_raw = defaultdict(list)
    for j, (seq, name) in enumerate(zip(df25['sequences'], df25['proteins'])):
        s = clean_seq(seq)
        if s in strictly_targeted:
            twin_raw[s].append((j, str(name), go_ids(df25.at[j, 'exp_annotations'])
                                 if 'exp_annotations' in df25.columns else {
                                 g for g, codes in coded_go(df25.at[j, 'annotations']).items()
                                 if codes & LATENCE_CODES}))
    twin_rows = []
    for m in strict.itertuples(index=False):
        seq = cohort.at[int(m.rid22), 'sequence_clean']
        entries = twin_raw[seq]
        n_go_profiles = len({frozenset(go) for _, _, go in entries})
        twin_rows.append({'rid22': int(m.rid22), 'protein_22': m.protein_22,
                          'n_2025_same_sequence_entries': len(entries),
                          'n_2025_go_profiles': n_go_profiles,
                          '2025_names': ';'.join(sorted(x[1] for x in entries))})
    write_tsv(pd.DataFrame(twin_rows, columns=['rid22', 'protein_22',
            'n_2025_same_sequence_entries', 'n_2025_go_profiles', '2025_names']),
              output / 'tables' / 'identical_sequence_2025.tsv')

    vocab_paths = {'bp': args.bp_terms, 'mf': args.mf_terms, 'cc': args.cc_terms}
    matrix_stats = write_matrices(strict, records, vocab_paths, ont, output, warnings, historical_reference_complete)
    matrix_columns = ['namespace', 'matrix', 'n_proteins', 'n_positive_pairs',
        'n_canonical_positive_pairs', 'n_model_terms', 'n_unique_canonical_terms',
        'n_alias_collision_groups', 'n_alias_duplicate_columns']
    write_tsv(pd.DataFrame(matrix_stats, columns=matrix_columns),
          output / 'tables' / 'model_vocabulary_coverage.tsv')

    if not go_pairs.empty:
        ns_counts = go_pairs.loc[
            go_pairs['strict_eligible'] & go_pairs['namespace'].isin(['bp', 'mf', 'cc'])
        ].groupby('namespace').agg(
            proteins=('rid22', 'nunique'),
            distinct_direct_go=('canonical_go_2022', 'nunique'),
            direct_positive_pairs=('canonical_go_2022', 'size'),
            historical_novel_direct_pairs=('historically_unrecorded_identical_seq', 'sum')
        ).reset_index()
    else:
        ns_counts = pd.DataFrame(columns=['namespace', 'proteins', 'distinct_direct_go',
                                          'direct_positive_pairs', 'historical_novel_direct_pairs'])
    write_tsv(ns_counts, output / 'tables' / 'namespace_summary.tsv')
    data_summary = {
        'count_2022_dark_input': int(cohort.attrs.get('dark_input_count', len(cohort))),
        'dark_input_is_complete': bool(cohort.attrs.get('dark_input_is_complete', False)),
        'count_2022_dark_weak': int(len(cohort)),
        'count_matches_identical': int(matches['match_status'].eq('identical').sum()),
        'count_matches_changed': int(matches['match_status'].eq('changed').sum()),
        'count_matches_ambiguous': int(matches['match_status'].eq('ambiguous').sum()),
        'count_matches_unmatched': int(matches['match_status'].eq('unmatched').sum()),
        'count_2025_future_labeled_candidates': int(len(selected)),
        'count_future_labeled_identical_sequences': int(selected['match_status'].eq('identical').sum()),
        'count_strict_test_proteins': int(len(strict)),
        'count_future_labeled_sequence_changes': int(selected['match_status'].eq('changed').sum()),
        'count_future_labeled_renamed': int(selected['name_changed'].sum()),
        'count_strict_renamed': int(strict['name_changed'].sum()),
        'count_strict_2025_sequence_duplicated': int(strict['sequence_duplicate_2025'].sum()),
        'count_distinct_strict_sequences': int(strict['sequence_sha256'].nunique()),
        'count_unique_future_go_raw': int(go_pairs.loc[go_pairs['strict_eligible'], 'go_id_2025'].nunique()) if not go_pairs.empty else 0,
        'count_unmapped_go_pairs_strict': int(go_unmapped['strict_eligible'].sum()) if not go_unmapped.empty else 0,
        'count_unmapped_go_ids_strict': int(go_unmapped.loc[go_unmapped['strict_eligible'], 'go_id_2025'].nunique()) if not go_unmapped.empty else 0,
        'count_strict_experimental_positive_pairs': int(go_pairs.loc[
            go_pairs['strict_eligible'], 'is_strict_experimental'].sum()) if not go_pairs.empty else 0,
        'count_future_go_pairs_strict': int(go_pairs['strict_eligible'].sum()) if not go_pairs.empty else 0,
        'historical_novel_direct_pairs_strict': int(go_pairs.loc[
            go_pairs['strict_eligible'], 'historically_unrecorded_identical_seq'].sum()) if not go_pairs.empty else 0,
        'history_reference_complete': historical_reference_complete,
        'source_has_2025_raw_evidence': bool('annotations' in df25),
        'original_primary_verified': bool(args.dat_2022 and args.dat_2025),
        'label_policy': args.label_policy,
        'ontology_propagation': args.propagation,
    }
    input_files = {k: getattr(args, k) for k in ['cohort_2022', 'uniprot_2022',
                  'uniprot_2025', 'darkness_2022', 'go_obo_2022', 'go_obo_2025',
                  'bp_terms', 'mf_terms', 'cc_terms', 'dat_2022', 'dat_2025']}
    input_metadata = {k: {'size_bytes': v.stat().st_size,
                          'sha256': file_sha256(v) if args.hash_inputs else None}
                      for k, v in input_files.items() if v is not None}
    info = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'builder_version': '1.0.1',
            'inputs': {k: str(getattr(args, k)) if getattr(args, k) is not None else None
                       for k in ['cohort_2022', 'uniprot_2022', 'uniprot_2025',
                                 'darkness_2022', 'go_obo_2022', 'go_obo_2025',
                                 'bp_terms', 'mf_terms', 'cc_terms', 'dat_2022', 'dat_2025']},
            'input_file_metadata': input_metadata,
            'parameters': {'label_policy': args.label_policy,
                'propagation': args.propagation, 'enable_sequence_only': args.enable_sequence_only,
                'require_raw_primary': args.require_raw_primary,
                'require_unique_2025_sequence': args.require_unique_2025_sequence},
            'dataframe_size': {'source_2025': len(df25),
                               'full_2022': len(full22) if full22 is not None else None},
            'dataframe_columns': {'2022_cohort': list(cohort.columns), '2025': list(df25.columns)},
            'counts': data_summary, 'warnings': warnings}
    (output / 'summary.json').write_text(json.dumps(data_summary, indent=2, ensure_ascii=False) + '\n')
    (output / 'manifest.json').write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str) + '\n')
    report = f'''# LATENCE Historical Dark–Weak Test: build report

Generated (UTC): {info['created_utc']}

## Scope and evidence contract

- Dark classification is taken from the supplied `dark_matter` flag; this builder **does not recompute** the Nature (2023) darkness algorithm.
- Weak means 2022 `exp_annotations` is empty **under the original DeepGOZero/LATENCE definition**.
- Future positives are 2025 `{args.label_policy}` labels, not necessarily biological functions first discovered in 2025.
- Strict candidates require an identical 2022/2025 amino-acid sequence and unique *best-scoring* identity match. Duplicate sequences in 2025 are reported, not automatically deleted.
- Original UniProt primary accessions: {'recovered from .dat records' if args.dat_2022 else '**not verified**; first AC token in the DataFrame may be unreliable'}.
- All unmatched or unannotated GO cells are **unknown**, not confirmed negatives.
- 2022 ontology propagation mode: `{args.propagation}`.

## Cohort and test counts

| Statistic | Count |
| --- | ---: |
| 2022 dark proteins present in source | {data_summary['count_2022_dark_input']:,} |
| 2022 dark and weak proteins | {data_summary['count_2022_dark_weak']:,} |
| 2022→2025 exact sequence matches (entire cohort) | {data_summary['count_matches_identical']:,} |
| Future-labeled candidates (all sequence statuses) | {len(selected):,} |
| Future-labeled with identical sequences | {data_summary['count_future_labeled_identical_sequences']:,} |
| **Strict labeled test proteins** | **{len(strict):,}** |
| Future-labeled sequence changes | {data_summary['count_future_labeled_sequence_changes']:,} |
| Future-labeled renamed entries | {data_summary['count_future_labeled_renamed']:,} |
| Strict renamed entries | {data_summary['count_strict_renamed']:,} |
| Strict proteins also sharing sequence with another 2025 record | {data_summary['count_strict_2025_sequence_duplicated']:,} |
| Distinct strict sequences | {data_summary['count_distinct_strict_sequences']:,} |
| Direct 2025 label protein–GO pairs in strict cohort | {data_summary['count_future_go_pairs_strict']:,} |
| Direct strict experimental evidence pairs among selected labels | {data_summary['count_strict_experimental_positive_pairs']:,} |
| GO pairs not mappable into historical ontology | {data_summary['count_unmapped_go_pairs_strict']:,} |
| Distinct unmappable historical GO terms | {data_summary['count_unmapped_go_ids_strict']:,} |
| GO pairs not historically recorded for any identical-sequence 2022 Swiss-Prot record | {data_summary['historical_novel_direct_pairs_strict']:,} |

## GO categories (strict dataset; direct mapped labels)

{table_report(ns_counts)}

## Evidence codes among strict future-positive pairs

{table_report(pd.DataFrame([{'evidence_code': code, 'positive_pairs': count}
              for code, count in sorted(hist_evidence.items())]))}

## Model vocabulary alignment

{table_report(pd.DataFrame(matrix_stats, columns=matrix_columns))}

Aliased GO vocabulary columns remain in their **original trained-model order**. When a
canonical GO appears under multiple model IDs (e.g., `alt_id` and current ID), its
positive labels are written to **all corresponding columns**, not just the final
column. This preserves logits-to-label alignment but inflates column-wise GO counts
relative to **unique biological GO terms**. Inspect `labels/{{bp,mf,cc}}/vocabulary_mapping.tsv`,
`vocabulary_alias_collisions.tsv`, and canonical-vs-model pair counts above; do not
silently deduplicate trained output columns.

If model GO vocabularies are not provided, namespace annotation tables are generated but **evaluation matrices are not**. Vocabularies must preserve LATENCE's exact training/inference GO order.

## Files

- `tables/cohort_match_index.tsv`: each historical dark weak protein, including ambiguity/unmatched.
- `tables/future_labeled_candidates.tsv`: future-labeled entries and sequence identity audits.
- `tables/excluded_labeled_candidates.tsv`: excluded future-labeled entries and reasons.
- `tables/strict_proteins.tsv`: retained sequence-identical entries.
- `tables/future_go_pairs.tsv`: direct future evidence with old-ontology mapping and historical evidence flags.
- `tables/unmapped_go_pairs.tsv`: GO IDs absent/obsolete in the historical ontology.
- `tables/namespace_summary.tsv`: per-namespace counts.
- `tables/2025_evidence_code_summary.tsv`: evidence codes of future positive pairs.
- `tables/unmapped_go_summary.tsv`: unmapped term status and counts.
- `tables/identical_sequence_2025.tsv`: duplicate-sequence/GO profile audit.
- `dark_weak_test_strict.pkl`: sequence and historical IDs with future labels as Python lists.
- `labels/{{bp,mf,cc}}/*.npz`: optional model-order CSR matrices for direct, propagated, experimental-only and historical-unrecorded labels.
- `labels/{{bp,mf,cc}}/vocabulary_mapping.tsv`: original model column → historical canonical GO.
- `labels/{{bp,mf,cc}}/vocabulary_alias_collisions.tsv`: alias-equivalent model columns (may be empty).

## Interpretation / limitations

1. `EXP_CODES` in DeepGOZero includes TAS and IC. The LATENCE-compatible label count is **not** automatically the count of truly experimental validations. See `is_strict_experimental` and strict experimental matrices.
2. The supplied DataFrame `accessions` may contain only the **last AC line** due to the original DeepGOZero parser's overwrite behavior. This pipeline does not call its first token "primary"; optional `--dat-2022/--dat-2025` can verify original first-AC primary without reloading sequences.
3. "Historically unrecorded" only excludes GO implied by annotations on *exactly identical* 2022 Swiss-Prot sequences, using the configured historical ontology. It does not exclude homolog-based transfer, unpublished evidence, or all of TrEMBL. These matrices are withheld unless the full 2022 annotated reference covers the cohort.
4. GO roots, label order, evidence code policy, P–P graph visibility, pseudo-label construction and evaluation procedure must be matched to the LATENCE model to prevent time leakage.
5. The test set is **selected on future annotation availability** and is not an unbiased sample of all dark proteins. No performance was computed by this data-building script.

## Warnings

''' + ('\n'.join(f'- {x}' for x in warnings) if warnings else '- None.') + '\n'
    (output / 'report.md').write_text(report, encoding='utf-8')
    print(f'[4/5] strict={len(strict):,}; GO positive pairs={data_summary["count_future_go_pairs_strict"]:,}', flush=True)
    print(f'[5/5] wrote {output.resolve()}', flush=True)
    return data_summary


def make_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Build an auditable 2022→2025 dark weak LATENCE test set from pandas tables')
    p.add_argument('--cohort-2022', type=Path,
        help='Optional already filtered 2022 dark/weak DataFrame; if absent derive from --uniprot-2022')
    p.add_argument('--uniprot-2022', type=Path,
        help='Full 2022 Swiss-Prot DataFrame for dark selection and identical-sequence historical GO audit')
    p.add_argument('--uniprot-2025', type=Path, required=True,
        help='Full 2025 Swiss-Prot DataFrame from uni2pandas.py')
    p.add_argument('--darkness-2022', type=Path,
        help='2022 table with proteins+dark_matter if full Swiss-Prot lacks that column')
    p.add_argument('--go-obo-2022', type=Path, required=True)
    p.add_argument('--go-obo-2025', type=Path, help='Optional ontology for future-only/obsolete GO diagnosis')
    p.add_argument('--bp-terms', type=Path, help='EXACT ordered BP vocabulary used by the model')
    p.add_argument('--mf-terms', type=Path, help='EXACT ordered MF vocabulary used by the model')
    p.add_argument('--cc-terms', type=Path, help='EXACT ordered CC vocabulary used by the model')
    p.add_argument('--dat-2022', type=Path, help='Optional original uniprot_sprot.dat(.gz) for PRIMARY accession only')
    p.add_argument('--dat-2025', type=Path, help='Optional original uniprot_sprot.dat(.gz) for PRIMARY accession only')
    p.add_argument('--require-raw-primary', action='store_true',
                   help='Require same verified original primary accession for strict inclusion')
    p.add_argument('--require-unique-2025-sequence', action='store_true',
                   help='Restrict strict cohort to 2025 sequences represented by only one Swiss-Prot entry')
    p.add_argument('--label-policy', choices=('latence', 'experimental'), default='latence')
    p.add_argument('--propagation', choices=('deepgozero', 'is_a_part_of', 'is_a'), default='deepgozero')
    p.add_argument('--enable-sequence-only', action='store_true',
                   help='Use exact-sequence matching as last resort for previously unmatched 2022 proteins')
    p.add_argument('--output-dir', type=Path, default=Path('data/dark_weak_test'))
    p.add_argument('--hash-inputs', action='store_true',
                   help='Compute SHA256 of all input files; may be expensive for large Swiss-Prot archives')
    p.add_argument('--overwrite', action='store_true', help='Allow replacing outputs in nonempty output folder')
    return p


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.cohort_2022 is None and args.uniprot_2022 is None:
        parser.error('provide --cohort-2022 or --uniprot-2022')
    summary = build(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
