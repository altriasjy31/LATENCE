"""Direct-evidence Top-K precision, recall and F1 with real output counts.

Unknown protein--GO pairs are not verified negatives. ``observed_*`` scores
measure agreement with the supplied canonical direct evidence, not biological
accuracy. All values are fractions in [0, 1]; K is a candidate budget, not a
score threshold and these F1 values are not threshold-maximized Fmax.

Candidates use exactly the Atlas conservative rule: positive canonical score
and pessimistic tied-group rank <= K, over ALL nonroot canonical GO. Alias
scores are float64 means, matching ``atlas_evidence_v060.rank_positive_pairs``.
Equal scores at a cutoff are excluded as one group, so returned counts can be
less than K. Precision divides by those actual returned counts.

Frequency masks filter candidates AFTER full-vocabulary ranking. They never
create an easier reranking problem. Every bin retains the entire layer protein
cohort. Macro precision averages over all cohort proteins with predictions,
including proteins without observed positives in that frequency bin; macro
recall and mean per-protein F1 average over bin-positive proteins, with no-hit
and abstaining positive proteins contributing zero. The harmonic mean of macro
precision and recall is separately named. Empty-gold recall/F1 are unavailable.

Memory is bounded by one GO score row plus small per-bin/per-K accumulators.
No prediction matrix, dense gold matrix or candidate-edge table is copied.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _protein_rows(raw_ids, nrows):
    if isinstance(raw_ids, Mapping):
        order = {}
        for name, index in raw_ids.items():
            if not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_)):
                raise ValueError('Protein row indices must be integers')
            name, index = str(name), int(index)
            if name in order:
                raise ValueError('Protein identifiers must be unique after string conversion')
            order[name] = index
        if len(set(order.values())) != len(order) or any(i < 0 or i >= nrows for i in order.values()):
            raise ValueError('Protein row mapping must be one-to-one within prediction rows')
    else:
        ids = list(map(str, raw_ids))
        if len(ids) != nrows or len(set(ids)) != len(ids):
            raise ValueError('Prediction protein IDs must be unique and aligned to score rows')
        order = {name: index for index, name in enumerate(ids)}
    return order


def direct_topk_metrics(raw_scores, raw_ids, canon, inverse, alias_sizes, pairs,
                        ks, root_id, frequency_bins=None, *, protein_ids=None):
    """Return one long-form row per frequency bin and K.

    ``raw_ids`` is a protein->prediction-row mapping or ordered ID sequence.
    ``pairs`` contains (protein, canonical_GO_index) direct positive relations.
    Pass ``protein_ids=view['ids']`` to preserve every protein in the evidence
    layer, including proteins whose canonical nonroot direct gold is empty.
    With no explicit cohort, proteins occurring in ``pairs`` define the cohort
    once, before splitting into bins. Extra prediction rows are never evaluated.

    ``frequency_bins`` maps bin names to canonical-GO boolean masks. ``all`` is
    always present. A supplied ``all`` mask must retain every nonroot GO.
    No model inference, calibration, ontology propagation or fusion is performed.
    """
    scores = np.asarray(raw_scores)
    canon = list(canon)
    inverse = np.asarray(inverse)
    sizes = np.asarray(alias_sizes)
    if len(canon) != len(set(canon)):
        raise ValueError('Canonical GO identifiers must be unique')
    if inverse.ndim != 1 or inverse.dtype.kind not in 'iu':
        raise ValueError('Canonical inverse mapping must be a one-dimensional integer vector')
    if scores.ndim != 2 or scores.shape[1] != len(inverse) or scores.dtype.kind not in 'biuf':
        raise ValueError('Prediction score matrix must be numeric and align to the original GO axis')
    if inverse.size and (inverse.min() < 0 or inverse.max() >= len(canon)):
        raise ValueError('Canonical inverse mapping contains an out-of-range index')
    inverse = inverse.astype(np.int64, copy=False)
    expected_sizes = np.bincount(inverse, minlength=len(canon))
    if sizes.shape != (len(canon),) or not np.array_equal(sizes, expected_sizes) or np.any(sizes <= 0):
        raise ValueError('Alias sizes do not match the canonical inverse mapping')
    allowed = np.flatnonzero(np.asarray(canon) != root_id)
    if not len(allowed):
        raise ValueError('Ranking vocabulary contains no nonroot GO')
    ks = tuple(ks)
    if not ks or len(set(ks)) != len(ks) or any(
            not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 1 for k in ks):
        raise ValueError('K values must be distinct positive integers')
    order = _protein_rows(raw_ids, scores.shape[0])
    positive_pairs = set()
    source_proteins = set()
    for protein, column in pairs:
        if (not isinstance(column, (int, np.integer)) or isinstance(column, (bool, np.bool_))
                or column < 0 or column >= len(canon)):
            raise ValueError('Direct evidence contains an invalid canonical GO index')
        protein, column = str(protein), int(column)
        source_proteins.add(protein)
        if canon[column] != root_id:
            positive_pairs.add((protein, column))
    if protein_ids is None:
        cohort = sorted(source_proteins)
    else:
        cohort = list(map(str, protein_ids))
        if len(cohort) != len(set(cohort)):
            raise ValueError('Layer protein cohort must contain distinct IDs')
        if source_proteins - set(cohort):
            raise ValueError('Direct evidence proteins occur outside the requested layer cohort')
    missing = set(cohort) - set(order)
    if missing:
        raise ValueError(f'Predictions missing {len(missing)} layer proteins: {sorted(missing)[:3]}')
    bins = {'all': np.ones(len(canon), dtype=bool)}
    for name, mask in (frequency_bins or {}).items():
        mask = np.asarray(mask)
        if mask.shape != (len(canon),) or mask.dtype.kind != 'b':
            raise ValueError('Frequency bins must be boolean masks on canonical GO columns')
        if str(name) == 'all' and not np.all(mask[allowed]):
            raise ValueError('The all frequency mask must retain the full nonroot vocabulary')
        if str(name) in bins and str(name) != 'all':
            raise ValueError('Frequency bin names must be unique after string conversion')
        bins[str(name)] = mask
    # Root terms never become candidates or observed direct positives.
    bins = {name: mask.copy() for name, mask in bins.items()}
    for mask in bins.values():
        mask[np.asarray(canon) == root_id] = False
    gold_by_protein = {}
    for protein, column in positive_pairs:
        gold_by_protein.setdefault(protein, []).append(column)
    stat = {(name, int(k)): dict(n_pairs=0, n_positive_proteins=0,
        n_predictions=0, n_candidate_proteins=0, n_hits=0, n_hit_proteins=0,
        precision_sum=0., recall_sum=0., protein_f1_sum=0.)
        for name in bins for k in ks}
    for protein in cohort:
        raw = np.asarray(scores[order[protein]], dtype=np.float64)
        if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
            raise ValueError(f'{protein}: probabilities must be finite and lie in [0,1]')
        combined = np.bincount(inverse, weights=raw, minlength=len(canon)) / sizes
        sorting = np.argsort(-combined[allowed], kind='stable')
        ranked = allowed[sorting]
        sorted_scores = combined[ranked]
        # A complete tied group can be returned only when its last rank <= K.
        ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True]) + 1
        ends = ends[sorted_scores[ends - 1] > 0]
        gold_mask = np.zeros(len(canon), dtype=bool)
        gold_mask[gold_by_protein.get(protein, [])] = True
        gold_counts = {name: int(np.count_nonzero(gold_mask & mask)) for name, mask in bins.items()}
        for k in ks:
            end_index = int(np.searchsorted(ends, k, side='right'))
            selected = ranked[:int(ends[end_index - 1])] if end_index else ranked[:0]
            for name, mask in bins.items():
                selected_bin = selected[mask[selected]]
                predicted = len(selected_bin)
                hits = int(np.count_nonzero(gold_mask[selected_bin]))
                positive = gold_counts[name]
                acc = stat[name, int(k)]
                acc['n_pairs'] += positive
                acc['n_positive_proteins'] += int(positive > 0)
                acc['n_predictions'] += predicted
                acc['n_candidate_proteins'] += int(predicted > 0)
                acc['n_hits'] += hits
                acc['n_hit_proteins'] += int(hits > 0)
                if predicted:
                    acc['precision_sum'] += hits / predicted
                if positive:
                    acc['recall_sum'] += hits / positive
                    acc['protein_f1_sum'] += 2 * hits / (predicted + positive)
    result = []
    for name, mask in bins.items():
        for k in ks:
            acc = stat[name, int(k)]
            hits, positive, predicted = acc['n_hits'], acc['n_pairs'], acc['n_predictions']
            eligible, active = acc['n_positive_proteins'], acc['n_candidate_proteins']
            micro_p = hits / predicted if predicted else None
            micro_r = hits / positive if positive else None
            macro_p = acc['precision_sum'] / active if active else None
            macro_r = acc['recall_sum'] / eligible if eligible else None
            # No gold means F1 is unavailable; positive gold with abstention is 0.
            mean_pr_f1 = None if not positive else (0. if not active or macro_p + macro_r == 0
                else 2 * macro_p * macro_r / (macro_p + macro_r))
            result.append(dict(frequency_bin=name, k=int(k), hit_policy='conservative',
                status='no_proteins' if not cohort else ('no_positives' if not positive else 'ok'),
                n_proteins=len(cohort), n_pairs=positive,
                n_rank_go=len(allowed), n_frequency_go=int(mask.sum()),
                n_positive_proteins=eligible, n_proteins_without_observed_positive=len(cohort)-eligible,
                n_predictions=predicted, n_candidate_proteins=active,
                n_abstaining_proteins=len(cohort)-active,
                n_hits=hits, n_hit_proteins=acc['n_hit_proteins'],
                candidate_protein_coverage=active/len(cohort) if cohort else None,
                abstention_fraction=(len(cohort)-active)/len(cohort) if cohort else None,
                observed_micro_precision=micro_p, observed_micro_recall=micro_r,
                observed_micro_f1=2*hits/(predicted+positive) if positive else None,
                observed_macro_precision=macro_p, observed_macro_recall=macro_r,
                observed_macro_f1=acc['protein_f1_sum']/eligible if eligible else None,
                observed_macro_f1_from_mean_pr=mean_pr_f1,
                prediction_scope='frequency_mask_after_full_vocabulary_topk',
                gold_scope='canonical_direct_evidence_only', score_units='fraction'))
    return result
