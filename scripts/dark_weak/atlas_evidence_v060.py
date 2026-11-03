"""Canonical relation accounting and full-vocabulary Atlas recovery metrics.

Evidence layers deliberately overlap: the future union is the set union of
experimental and InterPro2GO support. This module counts relations once while
preserving every supporting source. It never fuses model scores or treats an
unobserved relation as a validated negative.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd

SET_KEYS = ('history', 'future_union', 'future_exp', 'future_ipr')
MEMBERSHIP = {
    'history': 'in_history', 'future_union': 'in_future_union',
    'future_exp': 'in_future_exp', 'future_ipr': 'in_future_ipr',
}
RANK_BASE_COLUMNS = ['model', 'protein', 'canonical_index', 'go_id', 'score',
    'rank', 'rank_optimistic', 'rank_pessimistic', 'tie_size', 'score_is_zero',
    'reciprocal_rank', 'reciprocal_rank_lower', 'reciprocal_rank_upper',
    'conservative_reciprocal_rank', 'rank_vocabulary_size']
RECOVERY_COLUMNS = ['layer', 'k', 'hit_policy', 'target_model', 'reference_models', 'status',
    'n_layer_pairs', 'n_layer_proteins', 'n_target_recovered', 'n_reference_recovered',
    'n_shared_recovered', 'n_target_gained', 'n_target_lost',
    'n_target_gained_proteins', 'n_target_gained_go',
    'n_reference_recovered_proteins', 'n_target_recovered_proteins',
    'n_newly_recovered_proteins', 'target_recall', 'reference_union_recall',
    'historical_pairs', 'future_pairs', 'historical_plus_target_supported_future_pairs',
    'historical_plus_reference_supported_future_pairs']
EVIDENCE_COLUMNS = ['protein', 'canonical_index', 'go_id', *MEMBERSHIP.values(),
                    'future_support_source_count']


def _validate_inverse(inverse, ncanon=None):
    original = np.asarray(inverse)
    if original.ndim != 1 or original.dtype.kind not in 'iu':
        raise ValueError('inverse must be a one-dimensional integer vector')
    inverse = original.astype(np.int64, copy=False)
    if inverse.size and (inverse.min() < 0 or (ncanon is not None and inverse.max() >= ncanon)):
        raise ValueError('inverse contains an invalid canonical GO index')
    return inverse


def canonical_sets(raw_sets, inverse, root_col=None):
    """Convert raw index-list labels to distinct (protein, canonical GO) pairs.

    raw_sets has the four SET_KEYS; each value is {protein: raw_GO_indices}.
    Empty optional layers may be supplied as {}. root_col is a canonical index.
    """
    inverse = _validate_inverse(inverse)
    unknown = set(raw_sets) - set(SET_KEYS)
    if unknown:
        raise ValueError(f'Unknown evidence layer(s): {sorted(unknown)}')
    result, audit = {}, {}
    for name in SET_KEYS:
        pairs, raw_count, removed_root = set(), 0, 0
        group = raw_sets.get(name, {})
        for protein, indices in group.items():
            cols = np.asarray(list(indices))
            if cols.size and (cols.dtype.kind not in 'iu' or cols.ndim != 1):
                raise ValueError(f'{name}/{protein}: labels must be integer column indices')
            cols = np.unique(cols.astype(np.int64, copy=False))
            if cols.size and (cols.min() < 0 or cols.max() >= len(inverse)):
                raise ValueError(f'{name}/{protein}: label column outside model vocabulary')
            raw_count += len(cols)
            canonical = np.unique(inverse[cols])
            removed_root += int(root_col is not None and root_col in canonical)
            pairs.update((str(protein), int(c)) for c in canonical if c != root_col)
        result[name] = pairs
        audit[name] = {'raw_unique_column_pairs': int(raw_count),
                       'canonical_nonroot_pairs': len(pairs),
                       'canonical_nonroot_proteins': len({p for p, _ in pairs}),
                       'root_pairs_removed': removed_root}
    return result, audit


def evidence_membership(sets, canon, *, strict_union=True, strict_novel=False):
    """Preserve nested evidence provenance on the union of canonical relations.

    Future/history overlap is surfaced separately. strict_novel may be enabled
    for a benchmark promising historical novelty; the caller owns that claim.
    """
    unknown = set(sets) - set(SET_KEYS)
    if unknown:
        raise ValueError(f'Unknown evidence layer(s): {sorted(unknown)}')
    sets = {name: {(str(p), int(c)) for p, c in sets.get(name, set())}
            for name in SET_KEYS}
    all_pairs = set().union(*sets.values())
    if any(c < 0 or c >= len(canon) for _, c in all_pairs):
        raise ValueError('Evidence has an invalid canonical GO index')
    expected = sets['future_exp'] | sets['future_ipr']
    missing = expected - sets['future_union']
    extra = sets['future_union'] - expected
    overlap = sets['history'] & sets['future_union']
    audit = {
        'future_union_equals_source_union': not missing and not extra,
        'future_union_missing_source_pairs': len(missing),
        'future_union_extra_pairs': len(extra),
        'future_source_overlap_pairs': len(sets['future_exp'] & sets['future_ipr']),
        'future_source_overlap_proteins': len({p for p, _ in sets['future_exp']} &
                                             {p for p, _ in sets['future_ipr']}),
        'history_future_overlap_pairs': len(overlap),
        'history_future_overlap_examples': [[p, canon[c]] for p, c in sorted(overlap)[:10]],
        'canonical_relation_union_size': len(all_pairs),
    }
    if strict_union and (missing or extra):
        raise ValueError('Future union differs from experimental | InterPro2GO canonical relations: '
                         f'{len(missing)} missing, {len(extra)} extra')
    if strict_novel and overlap:
        raise ValueError(f'Strict future novelty violated: {len(overlap)} canonical relations overlap history')
    records = []
    for protein, index in sorted(all_pairs):
        pair = (protein, index)
        records.append({'protein': protein, 'canonical_index': index, 'go_id': canon[index],
                        **{MEMBERSHIP[name]: pair in sets[name] for name in SET_KEYS},
                        'future_support_source_count': int(pair in sets['future_exp']) +
                                                       int(pair in sets['future_ipr'])})
    frame = pd.DataFrame(records, columns=EVIDENCE_COLUMNS)
    for flag in MEMBERSHIP.values():
        frame[flag] = frame[flag].astype(bool)
    return frame, audit


def rank_columns(ks):
    columns = list(RANK_BASE_COLUMNS)
    for k in ks:
        columns += [f'hit_at_{k}', f'hit_at_{k}_lower', f'hit_at_{k}_upper',
                    f'hit_at_{k}_conservative']
    return columns


def rank_positive_pairs(raw_scores, raw_ids, canon, inverse, alias_sizes,
                        positive_pairs, *, model, ks, root_id):
    """Rank supplied positives against ALL canonical nonroot GO for each row.

    All ties keep their original canonical column order for legacy point
    estimates, alongside best/worst position bounds. A conservative recovery
    requires positive score AND a complete tied block within the top K.
    """
    scores = np.asarray(raw_scores)
    inverse = _validate_inverse(inverse, len(canon))
    sizes = np.asarray(alias_sizes)
    if scores.ndim != 2 or scores.shape[1] != len(inverse):
        raise ValueError('Probability matrix width differs from original GO column order')
    if len(canon) != len(set(canon)):
        raise ValueError('Canonical vocabulary contains duplicate identifiers')
    expected_sizes = np.bincount(inverse, minlength=len(canon))
    if sizes.shape != (len(canon),) or not np.array_equal(sizes, expected_sizes) or (sizes <= 0).any():
        raise ValueError('alias_sizes does not match canonical inverse mapping')
    ks = tuple(ks)
    if not ks or any(not isinstance(k, (int, np.integer)) or k <= 0 for k in ks) or len(ks) != len(set(ks)):
        raise ValueError('ks must contain distinct positive integers')
    if isinstance(raw_ids, Mapping):
        order = {str(name): int(index) for name, index in raw_ids.items()}
        if any(i < 0 or i >= scores.shape[0] for i in order.values()) or len(set(order.values())) != len(order):
            raise ValueError('Protein index map is not one-to-one within probability rows')
    else:
        ids = list(map(str, raw_ids))
        if len(ids) != scores.shape[0] or len(ids) != len(set(ids)):
            raise ValueError('Probability protein IDs are duplicated or misaligned')
        order = {name: i for i, name in enumerate(ids)}
    allowed = np.flatnonzero(np.asarray(canon) != root_id)
    if not len(allowed):
        raise ValueError('Ranking vocabulary has no nonroot GO')
    allowed_set = set(allowed.tolist())
    by_protein = {}
    for protein, index in set(positive_pairs):
        protein, index = str(protein), int(index)
        if index < 0 or index >= len(canon):
            raise ValueError('Positive canonical GO index outside vocabulary')
        if index in allowed_set:
            by_protein.setdefault(protein, set()).add(index)
    missing = set(by_protein) - set(order)
    if missing:
        raise ValueError(f'{model}: predictions missing {len(missing)} positive proteins, e.g. {sorted(missing)[:3]}')
    output = []
    for protein in sorted(by_protein):
        raw = np.asarray(scores[order[protein]], dtype=np.float64)
        if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
            raise ValueError(f'{model}/{protein}: probabilities must be finite in [0,1]')
        combined = np.bincount(inverse, weights=raw, minlength=len(canon)) / sizes
        p = combined[allowed]
        sorting = np.argsort(-p, kind='stable')
        stable = np.empty(len(allowed), dtype=np.int64)
        stable[sorting] = np.arange(1, len(allowed) + 1)
        sorted_p = np.sort(p)
        for index in sorted(by_protein[protein]):
            local = int(np.searchsorted(allowed, index))
            score = float(combined[index])
            best = int(1 + len(p) - np.searchsorted(sorted_p, score, side='right'))
            worst = int(len(p) - np.searchsorted(sorted_p, score, side='left'))
            row = {'model': model, 'protein': protein, 'canonical_index': index,
                   'go_id': canon[index], 'score': score, 'rank': int(stable[local]),
                   'rank_optimistic': best, 'rank_pessimistic': worst,
                   'tie_size': worst - best + 1, 'score_is_zero': score == 0,
                   'reciprocal_rank': 1. / stable[local],
                   'reciprocal_rank_lower': 1. / worst,
                   'reciprocal_rank_upper': 1. / best,
                   'conservative_reciprocal_rank': 1. / worst if score > 0 else 0.,
                   'rank_vocabulary_size': len(allowed)}
            for k in ks:
                row[f'hit_at_{k}'] = int(stable[local] <= k)
                row[f'hit_at_{k}_lower'] = int(worst <= k)
                row[f'hit_at_{k}_upper'] = int(best <= k)
                row[f'hit_at_{k}_conservative'] = int(score > 0 and worst <= k)
            output.append(row)
    return pd.DataFrame(output, columns=rank_columns(ks))


def summarize_ranks(frame, ks):
    """Positive-only rank metrics; fractions in [0,1], no false-negative claims."""
    out = {'n_pairs': int(len(frame)), 'n_proteins': int(frame.protein.nunique()),
           'n_go': int(frame.go_id.nunique()), 'status': 'ok' if len(frame) else 'no_positives'}
    if frame.empty:
        out.update(pairwise_mrr=None, pairwise_mrr_lower=None, pairwise_mrr_upper=None,
                   conservative_pairwise_mrr=None, mean_best_rr=None,
                   positive_zero_fraction=None, positive_tie_fraction=None)
    else:
        out.update(pairwise_mrr=float(frame.reciprocal_rank.mean()),
                   pairwise_mrr_lower=float(frame.reciprocal_rank_lower.mean()),
                   pairwise_mrr_upper=float(frame.reciprocal_rank_upper.mean()),
                   conservative_pairwise_mrr=float(frame.conservative_reciprocal_rank.mean()),
                   mean_best_rr=float(frame.groupby('protein').reciprocal_rank.max().mean()),
                   positive_zero_fraction=float(frame.score_is_zero.mean()),
                   positive_tie_fraction=float((frame.tie_size > 1).mean()))
    for k in ks:
        for suffix in ('', '_lower', '_upper', '_conservative'):
            hit = f'hit_at_{k}{suffix}'
            out[f'n_hits_at_{k}{suffix}'] = int(frame[hit].sum()) if len(frame) else 0
            out[f'micro_recall_at_{k}{suffix}'] = float(frame[hit].mean()) if len(frame) else None
            out[f'macro_recall_at_{k}{suffix}'] = float(frame.groupby('protein')[hit].mean().mean()) if len(frame) else None
        out[f'n_hit_proteins_at_{k}_conservative'] = int(
            frame.loc[frame[f'hit_at_{k}_conservative'] > 0, 'protein'].nunique()) if len(frame) else 0
    return out


def recovery_expansion(pair_ranks, evidence, *, ks,
                       stage1_models=('backbone', 'expert', 'modelout'), target_model='nbs',
                       hit_policies=('stable', 'conservative')):
    """Count evidence extension and target discoveries over a union of references.

    The reference union is a set of recovered known-positive relations, not
    probability fusion. Counts preserve source membership, including overlaps.
    Stable-column and conservative (positive-score complete ties) recovery
    policies are reported separately by default.
    """
    columns = list(RECOVERY_COLUMNS)
    if not hit_policies or set(hit_policies) - {'stable', 'conservative'}:
        raise ValueError('hit_policies must contain stable and/or conservative')
    available = set(pair_ranks.model.unique())
    if target_model not in available:
        return pd.DataFrame(columns=columns)
    references = tuple(m for m in stage1_models if m in available)
    missing_references = set(stage1_models) - available
    if missing_references:
        raise ValueError(f'Reference model(s) missing: {sorted(missing_references)}')
    if not references:
        raise ValueError('At least one Stage-1 reference is required')
    # All model rows must represent the same canonical positives. Otherwise a
    # missing prediction could silently masquerade as an unrecovered relation.
    pair_sets = {m: set(zip(g.protein, g.go_id)) for m, g in pair_ranks.groupby('model')}
    expected = pair_sets[target_model]
    if any(pair_sets[m] != expected for m in references):
        raise ValueError('Expansion comparison requires identical positive pairs for every model')
    histories = set(zip(evidence.loc[evidence.in_history, 'protein'], evidence.loc[evidence.in_history, 'go_id']))
    future = set(zip(evidence.loc[evidence.in_future_union, 'protein'], evidence.loc[evidence.in_future_union, 'go_id']))
    rows = []
    for layer, flag in MEMBERSHIP.items():
        layer_pairs = set(zip(evidence.loc[evidence[flag], 'protein'], evidence.loc[evidence[flag], 'go_id']))
        for k in ks:
            for policy in hit_policies:
                hitsets = {}
                for model in (*references, target_model):
                    sub = pair_ranks[(pair_ranks.model == model) & (pair_ranks[f'hit_at_{k}' + ('_conservative' if policy == 'conservative' else '')] > 0)]
                    hitsets[model] = set(zip(sub.protein, sub.go_id)) & layer_pairs
                target = hitsets[target_model]
                reference = set().union(*(hitsets[m] for m in references))
                gained, lost, shared = target - reference, reference - target, target & reference
                reference_proteins = {p for p, _ in reference}
                target_proteins = {p for p, _ in target}
                target_future = target & future
                reference_future = reference & future
                rows.append({'layer': layer, 'k': int(k), 'hit_policy': policy, 'target_model': target_model,
                    'reference_models': '|'.join(references), 'status': 'ok' if layer_pairs else 'no_positives',
                    'n_layer_pairs': len(layer_pairs), 'n_layer_proteins': len({p for p, _ in layer_pairs}),
                    'n_target_recovered': len(target), 'n_reference_recovered': len(reference),
                    'n_shared_recovered': len(shared), 'n_target_gained': len(gained), 'n_target_lost': len(lost),
                    'n_target_gained_proteins': len({p for p, _ in gained}),
                    'n_target_gained_go': len({g for _, g in gained}),
                    'n_reference_recovered_proteins': len(reference_proteins),
                    'n_target_recovered_proteins': len(target_proteins),
                    'n_newly_recovered_proteins': len(target_proteins - reference_proteins),
                    'target_recall': len(target)/len(layer_pairs) if layer_pairs else None,
                    'reference_union_recall': len(reference)/len(layer_pairs) if layer_pairs else None,
                    'historical_pairs': len(histories), 'future_pairs': len(future),
                    'historical_plus_target_supported_future_pairs': len(histories | target_future),
                    'historical_plus_reference_supported_future_pairs': len(histories | reference_future)})
    return pd.DataFrame(rows, columns=columns)
