"""Protein row permutation control for unseen GO ranking metrics."""

from operator import index

import numpy as np


def _score_ranks(score_rows, columns):
    """Count strictly higher scores and equal scores for each requested column."""
    higher = np.empty((len(score_rows), len(columns)), dtype=np.intp)
    tied = np.empty_like(higher)
    for row, scores in enumerate(score_rows):
        sorted_scores = np.sort(scores)
        selected = scores[columns]
        left = np.searchsorted(sorted_scores, selected, side="left")
        right = np.searchsorted(sorted_scores, selected, side="right")
        higher[row] = len(scores) - right
        tied[row] = right - left
    return higher, tied


def _first_positive_groups(higher, tied, pair_columns, pairs_per_protein):
    """Find the earliest positive tie group for each recipient/donor pair."""
    n_proteins = len(pairs_per_protein)
    first_higher = np.empty((len(higher), n_proteins), dtype=np.intp)
    first_tied = np.empty_like(first_higher)
    first_positives = np.empty_like(first_higher)
    boundaries = np.r_[0, np.cumsum(pairs_per_protein)]
    for i in range(n_proteins):
        columns = pair_columns[boundaries[i]:boundaries[i + 1]]
        relevant_higher = higher[:, columns]
        best = relevant_higher.argmin(axis=1)
        first_higher[:, i] = relevant_higher[np.arange(len(higher)), best]
        first_tied[:, i] = tied[np.arange(len(higher)), columns[best]]
        first_positives[:, i] = np.count_nonzero(
            relevant_higher == first_higher[:, i, None], axis=1
        )
    return first_higher, first_tied, first_positives


def _expected_reciprocal_first_rank(higher, tied, positives, candidate_count):
    harmonic = np.r_[0.0, np.cumsum(1.0 / np.arange(1, candidate_count + 1))]
    reciprocal = (harmonic[higher + tied] - harmonic[higher]) / tied
    multi = positives > 1
    if multi.any():
        greater = higher[multi]
        count = tied[multi]
        hits = positives[multi]
        survival = np.ones(len(hits))
        expectation = np.zeros(len(hits))
        for offset in range(int(np.max(count - hits)) + 1):
            possible = offset <= count - hits
            hazard = np.divide(hits, count - offset, out=np.zeros(len(hits)), where=possible)
            expectation += survival * hazard / (greater + offset + 1)
            survival *= 1 - hazard
        reciprocal[multi] = expectation
    return reciprocal


def _hit_probability(higher, tied, positives, k):
    """Probability of at least one positive under uniform ordering within ties."""
    slots = np.clip(k - higher, 0, tied)
    probability = slots / tied
    multi = (positives > 1) & (slots > 0) & (slots < tied)
    if multi.any():
        count = tied[multi]
        hits = positives[multi]
        remaining = count - slots[multi]
        miss = np.ones(len(hits))
        for offset in range(int(hits.max())):
            active = offset < hits
            miss[active] *= np.maximum(remaining[active] - offset, 0) / (count[active] - offset)
        probability[multi] = 1.0 - miss
    return probability


def protein_row_permutation(score_rows, labels, positive_rows, top_k, n_permutations=1000, seed=0):
    """Shuffle whole score rows over the full cohort, keeping label rows fixed.

    score_rows has shape (all cohort proteins, candidate GOs). labels is a
    boolean matrix of observed positives for the rows indexed by positive_rows.
    Returns metric summaries and per-permutation arrays keyed by the same
    metric names. Percentiles describe the permutation null distribution.
    """
    scores = np.asarray(score_rows)
    truth = np.asarray(labels)
    positive_rows = np.asarray(positive_rows)
    ks = tuple(index(k) for k in top_k)
    n_permutations = index(n_permutations)
    seed = index(seed)
    if not ks or min(ks) < 1 or len(ks) != len(set(ks)):
        raise ValueError("top_k must contain unique positive integers")
    if n_permutations < 1 or seed < 0:
        raise ValueError("n_permutations must be positive and seed nonnegative")
    if scores.ndim != 2 or not all(scores.shape) or scores.dtype.kind not in "fiu":
        raise ValueError("score_rows must be a nonempty protein-by-GO numeric matrix")
    if not np.isfinite(scores).all():
        raise ValueError("score_rows must be finite")
    if truth.ndim != 2 or truth.shape[1] != scores.shape[1] or truth.dtype != np.bool_:
        raise ValueError("labels must be a boolean matrix on the score GO axis")
    if positive_rows.shape != (len(truth),) or (
        len(positive_rows) and positive_rows.dtype.kind not in "iu"
    ):
        raise ValueError("positive_rows must index one donor row per label row")
    if (np.any(positive_rows < 0) or np.any(positive_rows >= len(scores))
            or len(np.unique(positive_rows)) != len(positive_rows)):
        raise ValueError("positive_rows must contain distinct valid score row indices")
    if not len(truth):
        return {}, {}

    pairs_per_protein = truth.sum(axis=1)
    if np.any(pairs_per_protein == 0):
        raise ValueError("each label row must have an observed positive")
    pair_recipients, pair_gos = np.nonzero(truth)
    go_columns, pair_columns, pairs_per_go = np.unique(
        pair_gos, return_inverse=True, return_counts=True
    )
    higher, tied = _score_ranks(scores, go_columns)
    first_higher, first_tied, first_positives = _first_positive_groups(
        higher, tied, pair_columns, pairs_per_protein
    )

    rng = np.random.default_rng(seed)
    donor_rows = np.empty((n_permutations + 1, len(truth)), dtype=np.intp)
    donor_rows[0] = positive_rows
    for run in range(1, len(donor_rows)):
        donor_rows[run] = rng.permutation(len(scores))[positive_rows]
    sampled_pair_rows = donor_rows[:, pair_recipients]
    sampled_higher = higher[sampled_pair_rows, pair_columns]
    sampled_tied = tied[sampled_pair_rows, pair_columns]
    label_rows = np.arange(len(truth))
    candidate_count = scores.shape[1]
    reciprocal = _expected_reciprocal_first_rank(
        first_higher, first_tied, first_positives, candidate_count
    )
    trial_metrics = {
        "mrr_first_positive_expected": reciprocal[donor_rows, label_rows].mean(axis=1),
        "mean_positive_pair_midrank": (sampled_higher + (sampled_tied + 1) / 2).mean(axis=1),
    }
    for k in ks:
        effective = min(k, candidate_count)
        credit = np.clip(effective - sampled_higher, 0, sampled_tied) / sampled_tied
        trial_metrics[f"pair_recall@{k}"] = credit.mean(axis=1)
        trial_metrics[f"precision@{k}"] = credit.sum(axis=1) / (len(truth) * effective)
        trial_metrics[f"protein_recall@{k}"] = (
            credit / pairs_per_protein[pair_recipients]
        ).sum(axis=1) / len(truth)
        trial_metrics[f"go_macro_recall@{k}"] = (
            credit / pairs_per_go[pair_columns]
        ).sum(axis=1) / len(go_columns)
        hit = _hit_probability(first_higher, first_tied, first_positives, effective)
        trial_metrics[f"protein_hit_rate@{k}"] = hit[donor_rows, label_rows].mean(axis=1)

    summary = {}
    draws = {}
    for metric, values in trial_metrics.items():
        observed = float(values[0])
        null = values[1:]
        q025, q975 = np.quantile(null, (0.025, 0.975))
        # Keep mathematically equal metrics together at floating-point precision.
        if metric == "mean_positive_pair_midrank":
            tail = {"empirical_p_le": (1 + int(np.count_nonzero(null <= observed + 1e-12))) / (len(null) + 1)}
        else:
            tail = {"empirical_p_ge": (1 + int(np.count_nonzero(null >= observed - 1e-12))) / (len(null) + 1)}
        summary[metric] = {
            "observed": observed,
            "null_mean": float(null.mean()),
            "null_q025": float(q025),
            "null_q975": float(q975),
            **tail,
        }
        draws[metric] = null
    return summary, draws
