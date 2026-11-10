"""Contracts for direct observed-support precision with honest denominators."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts/dark_weak'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = load('atlas_precision_v062', 'atlas_precision_v062.py')
old = load('atlas_evidence_for_precision_tests', 'atlas_evidence_v060.py')


class PrecisionTests(unittest.TestCase):
    def metrics(self, p, pairs, ks=(1,), *, ids=None, bins=None, proteins=None):
        p = np.asarray(p)
        n = p.shape[1]
        return mod.direct_topk_metrics(p, ids if ids is not None else ['A', 'B'][:len(p)],
            [f'GO:{i}' for i in range(n)], np.arange(n), np.ones(n), pairs,
            ks, 'absent', bins, protein_ids=proteins)

    def test_actual_conservative_denominator_and_zero_abstention(self):
        rows = self.metrics([[.8,.8,.2,0], [0,0,0,0]], {('A',0),('B',0)}, ks=(1,2,5))
        first, second, third = rows
        self.assertEqual(first['n_predictions'],0)
        self.assertIsNone(first['observed_micro_precision'])
        self.assertEqual(first['observed_micro_recall'],0)
        self.assertEqual(first['observed_micro_f1'],0)
        self.assertEqual(first['abstention_fraction'],1)
        self.assertEqual(second['n_predictions'],2)
        self.assertEqual(second['n_hits'],1)
        self.assertEqual(second['observed_micro_precision'],.5)
        self.assertEqual(second['observed_micro_recall'],.5)
        self.assertEqual(second['observed_micro_f1'],.5)
        self.assertEqual(second['candidate_protein_coverage'],.5)
        self.assertEqual(third['n_predictions'],3) # zero-score terms never returned
        self.assertAlmostEqual(third['observed_micro_precision'],1/3)

    def test_full_vocabulary_then_frequency_filter_and_entire_cohort(self):
        bins={'rare':np.array([True,False,False]), 'empty':np.zeros(3,dtype=bool)}
        rows=self.metrics([[.6,.9,.1],[.9,.1,.2]], {('A',0),('B',1)}, bins=bins)
        rare=next(r for r in rows if r['frequency_bin']=='rare')
        # A's rare positive remains below full-vocabulary Top1. B has no rare
        # observed gold, but its rare candidate must count in precision's denom.
        self.assertEqual(rare['n_proteins'],2)
        self.assertEqual(rare['n_positive_proteins'],1)
        self.assertEqual(rare['n_predictions'],1)
        self.assertEqual(rare['n_hits'],0)
        self.assertEqual(rare['observed_micro_precision'],0)
        self.assertEqual(rare['observed_macro_precision'],0)
        self.assertEqual(rare['observed_micro_recall'],0)
        empty=next(r for r in rows if r['frequency_bin']=='empty')
        self.assertEqual(empty['status'],'no_positives')
        self.assertIsNone(empty['observed_micro_precision'])
        self.assertIsNone(empty['observed_micro_recall'])
        self.assertIsNone(empty['observed_micro_f1'])

    def test_alias_mean_root_exclusion_and_recall_parity(self):
        p=np.array([[.1,.8,.7,1,.9],[0,0,0,1,0]],dtype=np.float32)
        ids={'A':0,'B':1}; canon=['GO:a','GO:b','GO:c','GO:root']
        inverse=np.array([0,1,2,3,0]); sizes=np.array([2,1,1,1]); pairs={('A',0),('B',0)}
        result=mod.direct_topk_metrics(p,ids,canon,inverse,sizes,pairs,[1,3,8],'GO:root')
        ranks=old.rank_positive_pairs(p,ids,canon,inverse,sizes,pairs,
            model='m',ks=[1,3,8],root_id='GO:root')
        summary=old.summarize_ranks(ranks,[1,3,8])
        for row in result:
            self.assertAlmostEqual(row['observed_micro_recall'],summary[f'micro_recall_at_{row["k"]}_conservative'])
        self.assertEqual(result[0]['n_hits'],0)
        self.assertEqual(result[1]['n_predictions'],3)
        self.assertEqual(result[1]['n_rank_go'],3)
        self.assertEqual(result[2]['n_predictions'],3)

    def test_explicit_layer_cohort_not_prediction_union(self):
        p=[[.9,.1],[.9,.1],[.9,.1]]
        rows=self.metrics(p,{('A',0)},ids=['A','B','C'],proteins=['A','B'])
        self.assertEqual(rows[0]['n_proteins'],2)
        self.assertEqual(rows[0]['n_predictions'],2)
        self.assertEqual(rows[0]['observed_micro_precision'],.5)
        self.assertEqual(rows[0]['observed_macro_precision'],.5)
        self.assertEqual(rows[0]['observed_micro_recall'],1)
        self.assertEqual(rows[0]['observed_macro_recall'],1)
        self.assertEqual(rows[0]['observed_macro_f1'],1)
        self.assertAlmostEqual(rows[0]['observed_macro_f1_from_mean_pr'],2/3)

    def test_no_positive_bin_retains_predictions_but_f1_unavailable(self):
        rows=self.metrics([[.8,.2]],set(),ids=['A'],proteins=['A'])
        row=rows[0]
        self.assertEqual(row['n_predictions'],1)
        self.assertEqual(row['observed_micro_precision'],0)
        self.assertIsNone(row['observed_micro_recall'])
        self.assertIsNone(row['observed_micro_f1'])
        self.assertIsNone(row['observed_macro_f1'])

    def test_empty_cohort_and_invalid_inputs(self):
        rows=self.metrics([[.8,.2]],set(),ids=['A'])
        self.assertEqual(rows[0]['status'],'no_proteins')
        self.assertIsNone(rows[0]['candidate_protein_coverage'])
        with self.assertRaisesRegex(ValueError,'missing'):
            self.metrics([[.8,.2]],{('B',0)},ids=['A'])
        with self.assertRaisesRegex(ValueError,'outside'):
            self.metrics([[.8,.2]],{('A',0)},ids=['A'],proteins=[])
        with self.assertRaisesRegex(ValueError,'finite'):
            self.metrics([[np.nan,.2]],{('A',0)},ids=['A'])
        with self.assertRaisesRegex(ValueError,'all frequency mask'):
            self.metrics([[.8,.2]],{('A',0)},ids=['A'],bins={'all':np.array([True,False])})
        with self.assertRaisesRegex(ValueError,'positive integers'):
            self.metrics([[.8,.2]],{('A',0)},ids=['A'],ks=(0,))

    def test_memmap_and_permuted_prediction_rows(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'scores.npy'
            np.save(path,np.array([[.9,.1],[.1,.9]],dtype=np.float32))
            scores=np.load(path,mmap_mode='r')
            a=mod.direct_topk_metrics(scores,{'B':0,'A':1},['GO:0','GO:1'],[0,1],[1,1],
                {('A',1),('B',0)},[1],'absent')
            self.assertEqual(a[0]['n_hits'],2)
            self.assertEqual(a[0]['observed_micro_precision'],1)


if __name__=='__main__':
    unittest.main()
