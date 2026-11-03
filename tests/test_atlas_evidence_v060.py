"""Scientific contracts for nested evidence and Atlas relation expansion."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('atlas_evidence_v060',
                ROOT/'scripts/dark_weak/atlas_evidence_v060.py')
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class EvidenceTests(unittest.TestCase):
    def test_alias_dedup_root_removal_and_source_membership(self):
        inverse = np.array([0, 1, 0, 2])
        raw = {'history': {'P': [0, 2]}, 'future_exp': {'A': [0, 2, 3]},
               'future_ipr': {'A': [0], 'B': [1]},
               'future_union': {'A': [0, 2, 3], 'B': [1]}}
        sets, counts = mod.canonical_sets(raw, inverse, root_col=2)
        self.assertEqual(sets['history'], {('P', 0)})
        self.assertEqual(sets['future_union'], {('A', 0), ('B', 1)})
        self.assertEqual(counts['future_exp']['raw_unique_column_pairs'], 3)
        self.assertEqual(counts['future_exp']['root_pairs_removed'], 1)
        table, audit = mod.evidence_membership(sets, ['GO:1', 'GO:2', 'GO:root'], strict_novel=True)
        self.assertEqual(len(table), 3)
        row = table[table.protein == 'A'].iloc[0]
        self.assertTrue(row.in_future_exp)
        self.assertTrue(row.in_future_ipr)
        self.assertEqual(row.future_support_source_count, 2)
        self.assertEqual(audit['future_source_overlap_pairs'], 1)
        self.assertEqual(audit['future_source_overlap_proteins'], 1)

    def test_union_mismatch_rejected_and_novelty_audited(self):
        sets = {'history': {('A', 0)}, 'future_union': {('A', 0)},
                'future_exp': {('A', 0)}, 'future_ipr': set()}
        _, audit = mod.evidence_membership(sets, ['GO:1'])
        self.assertEqual(audit['history_future_overlap_pairs'], 1)
        with self.assertRaisesRegex(ValueError, 'novelty'):
            mod.evidence_membership(sets, ['GO:1'], strict_novel=True)
        sets['future_union'] = set()
        with self.assertRaisesRegex(ValueError, 'union differs'):
            mod.evidence_membership(sets, ['GO:1'])

    def test_full_vocabulary_ranks_and_mean_aliases(self):
        canon = ['GO:1', 'GO:2', 'GO:3', 'GO:root']
        inverse = np.array([0, 1, 2, 3, 0])
        p = np.array([[.1, .8, .7, 1., .9]], dtype=np.float32)
        ranks = mod.rank_positive_pairs(p, ['A'], canon, inverse, [2, 1, 1, 1],
                   {('A', 0)}, model='model', ks=[1, 3], root_id='GO:root')
        self.assertAlmostEqual(ranks.iloc[0].score, .5, places=6)
        # Positives are not oracle filtered; the two negative/unknown scores
        # remain in the denominator and outrank this positive.
        self.assertEqual(ranks.iloc[0]['rank'], 3)
        self.assertEqual(ranks.iloc[0].rank_vocabulary_size, 3)
        self.assertEqual(ranks.iloc[0].hit_at_1_conservative, 0)
        self.assertEqual(ranks.iloc[0].hit_at_3_conservative, 1)

    def test_zero_and_nonzero_ties(self):
        canon = ['GO:1', 'GO:2', 'GO:3', 'GO:root']
        inverse = np.arange(4)
        rows = mod.rank_positive_pairs(np.array([[0, 0, 0, 1], [.4, .4, .1, 1]]),
            ['zero', 'tied'], canon, inverse, np.ones(4), {('zero', 0), ('tied', 0)},
            model='m', ks=[1, 2, 3], root_id='GO:root').set_index('protein')
        zero, tied = rows.loc['zero'], rows.loc['tied']
        self.assertEqual(zero['rank'], 1)
        self.assertEqual(zero.rank_pessimistic, 3)
        self.assertEqual(zero.hit_at_1, 1)
        self.assertEqual(zero.hit_at_3_lower, 1)
        self.assertEqual(zero.hit_at_3_conservative, 0)
        self.assertEqual(tied.hit_at_1, 1)
        self.assertEqual(tied.hit_at_1_conservative, 0)
        self.assertEqual(tied.hit_at_2_conservative, 1)

    def test_id_reordering_and_missing_predictions(self):
        canon = ['GO:1', 'GO:2']
        p = np.array([[.1, .9], [.9, .1]])
        pairs = {('A', 1), ('B', 0)}
        first = mod.rank_positive_pairs(p, ['A', 'B'], canon, [0, 1], [1, 1], pairs,
                    model='m', ks=[1], root_id='absent')
        second = mod.rank_positive_pairs(p[::-1], {'B': 0, 'A': 1}, canon, [0, 1], [1, 1], pairs,
                    model='m', ks=[1], root_id='absent')
        pd.testing.assert_frame_equal(first, second)
        with self.assertRaisesRegex(ValueError, 'missing'):
            mod.rank_positive_pairs(p[:1], ['A'], canon, [0, 1], [1, 1], pairs,
                    model='m', ks=[1], root_id='absent')

    def test_macro_recall_and_empty_summary(self):
        canon = ['GO:1', 'GO:2', 'GO:3']
        ranks = mod.rank_positive_pairs(np.array([[.9, .8, .1], [.1, .8, .9]]), ['A', 'B'],
                   canon, np.arange(3), np.ones(3), {('A', 0), ('A', 1), ('B', 0)},
                   model='m', ks=[1], root_id='absent')
        summary = mod.summarize_ranks(ranks, [1])
        self.assertAlmostEqual(summary['micro_recall_at_1_conservative'], 1/3)
        self.assertAlmostEqual(summary['macro_recall_at_1_conservative'], .25)
        self.assertEqual(summary['n_hit_proteins_at_1_conservative'], 1)
        empty = mod.summarize_ranks(ranks.iloc[:0], [1])
        self.assertIsNone(empty['micro_recall_at_1'])
        self.assertEqual(empty['n_hits_at_1'], 0)
        self.assertEqual(empty['status'], 'no_positives')

    def test_incremental_recovery_against_all_stage1_evidence(self):
        canon = ['GO:1', 'GO:2', 'GO:3']
        sets = {'history': {('H', 0)}, 'future_union': {('A', 0), ('A', 1), ('B', 2)},
                'future_exp': {('A', 0), ('B', 2)}, 'future_ipr': {('A', 1)}}
        evidence, _ = mod.evidence_membership(sets, canon)
        pairs = set().union(*sets.values())
        ids = ['H', 'A', 'B']
        matrices = {
            'backbone': [[.9, .2, .1], [.9, .8, .1], [.9, .2, .1]], # A0 only
            'expert': [[.9, .2, .1], [.7, .9, .1], [.9, .2, .1]], # A1 only
            'modelout': [[.9, .2, .1], [.9, .8, .1], [.9, .2, .1]],
            'nbs': [[.9, .2, .1], [.9, .8, .1], [.1, .2, .9]], # A0+B2; A1 lost
        }
        ranks = pd.concat([mod.rank_positive_pairs(np.asarray(p), ids, canon, np.arange(3),
                    np.ones(3), pairs, model=m, ks=[1], root_id='absent')
                    for m, p in matrices.items()], ignore_index=True)
        output = mod.recovery_expansion(ranks, evidence, ks=[1])
        future = output[output.layer == 'future_union'].iloc[0]
        self.assertEqual(future.n_reference_recovered, 2) # Set union, no score ensemble
        self.assertEqual(future.n_target_recovered, 2)
        self.assertEqual(future.n_shared_recovered, 1)
        self.assertEqual(future.n_target_gained, 1)
        self.assertEqual(future.n_target_lost, 1)
        self.assertEqual(future.n_target_gained_proteins, 1)
        self.assertEqual(future.n_newly_recovered_proteins, 1)
        self.assertEqual(future.historical_plus_target_supported_future_pairs, 3)
        # The historical H edge is counted once, though all models recover it.
        history = output[output.layer == 'history'].iloc[0]
        self.assertEqual(history.n_layer_pairs, 1)
        self.assertEqual(history.historical_pairs, 1)
        self.assertEqual(history.n_target_gained, 0)

    def test_unequal_model_positive_sets_rejected(self):
        canon = ['GO:1', 'GO:2']
        evidence, _ = mod.evidence_membership({'future_exp': {('A', 0)},
             'future_union': {('A', 0)}, 'future_ipr': set(), 'history': set()}, canon)
        a = mod.rank_positive_pairs([[.9, .1]], ['A'], canon, [0, 1], [1, 1], {('A', 0)},
                    model='backbone', ks=[1], root_id='absent')
        n = mod.rank_positive_pairs([[.9, .1]], ['A'], canon, [0, 1], [1, 1], {('A', 0), ('A', 1)},
                    model='nbs', ks=[1], root_id='absent')
        with self.assertRaisesRegex(ValueError, 'identical positive pairs'):
            mod.recovery_expansion(pd.concat([a, n]), evidence, ks=[1], stage1_models=['backbone'])

    def test_recovery_policies_and_missing_target(self):
        canon = ['GO:1', 'GO:2']
        evidence, _ = mod.evidence_membership({'future_exp': {('A', 0)},
             'future_union': {('A', 0)}, 'future_ipr': set(), 'history': set()}, canon)
        b = mod.rank_positive_pairs([[.1, .9]], ['A'], canon, [0, 1], [1, 1], {('A', 0)},
                    model='backbone', ks=[1], root_id='absent')
        n = mod.rank_positive_pairs([[0, 0]], ['A'], canon, [0, 1], [1, 1], {('A', 0)},
                    model='nbs', ks=[1], root_id='absent')
        table = mod.recovery_expansion(pd.concat([b, n]), evidence, ks=[1], stage1_models=['backbone'])
        future = table[table.layer == 'future_union'].set_index('hit_policy')
        self.assertEqual(future.loc['stable', 'n_target_gained'], 1)
        self.assertEqual(future.loc['conservative', 'n_target_gained'], 0)
        skipped = mod.recovery_expansion(b, evidence, ks=[1])
        self.assertTrue(skipped.empty)
        self.assertEqual(skipped.columns.tolist(), mod.RECOVERY_COLUMNS)
        with self.assertRaisesRegex(ValueError, 'Reference model'):
            mod.recovery_expansion(pd.concat([b, n]), evidence, ks=[1])


if __name__ == '__main__':
    unittest.main()
