"""Regression tests for Stage-1 shard order and versioned MSA view."""
from __future__ import annotations

import argparse
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.dark_weak.stage1_alignment import ProteinRowPlacement
from scripts.dark_weak.prepare_msa_index_view import make_view
from scripts.dark_weak.prepare_inference_split import build as prepare_split
from scripts.dark_weak.align_historical_probabilities import align


class InputRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def create_msa_indices(self):
        a=self.root/'first';a.mkdir()
        b=self.root/'second';b.mkdir()
        # The first index is in Z,A order, and A/Z share a shard. B is in another.
        np.arange(16,dtype=np.uint8).tofile(a/'shard0.bin')
        np.arange(16,24,dtype=np.uint8).tofile(b/'shard1.bin')
        first={'shards':['shard0.bin'],'proteins':['Z','A'],'shard_ids':[0,0],
               'offsets':[0,8],'nseqs':[2,2],'seqlens':[4,4]}
        second={'shards':['shard1.bin'],'proteins':['B'],'shard_ids':[0],
                'offsets':[0],'nseqs':[2],'seqlens':[4]}
        for folder,name,source in ((a,'index.pkl',first),(b,'index.pkl',second)):
            with (folder/name).open('wb') as f:pickle.dump(source,f)
        return a/'index.pkl',b/'index.pkl'

    def test_view_preserves_requested_order_and_shards(self):
        first,second=self.create_msa_indices()
        targets=self.root/'target.txt';targets.write_text('A\nB\nZ\n')
        output=self.root/'view'/'index.pkl'
        report=make_view(targets,[first,second],output)
        with output.open('rb') as f:index=pickle.load(f)
        self.assertEqual(index['proteins'],['A','B','Z'])
        self.assertEqual(list(index['shard_ids']),[0,1,0])
        self.assertEqual(list(index['offsets']),[8,0,0])
        self.assertEqual(report['num_proteins'],3)
        self.assertEqual(report['source_cutoff_verified'],False)

    def test_missing_protein_is_error(self):
        first,second=self.create_msa_indices()
        targets=self.root/'target.txt';targets.write_text('A\nMISSING\n')
        with self.assertRaisesRegex(ValueError,'Missing 1/2'):
            make_view(targets,[first,second],self.root/'view'/'index.pkl')

    def test_duplicate_across_sources_requires_policy(self):
        first,_=self.create_msa_indices()
        targets=self.root/'target.txt';targets.write_text('A\n')
        with self.assertRaisesRegex(ValueError,'multiple MSA indices'):
            make_view(targets,[first,first],self.root/'view'/'index.pkl')
        result=make_view(targets,[first,first],self.root/'view2'/'index.pkl',duplicate_policy='first')
        self.assertEqual(result['n_collisions'],1)

    def test_scatter_writer_recovers_order_from_shard_iteration(self):
        placement=ProteinRowPlacement(['A','B','Z'])
        out=np.zeros(3,dtype=np.float32)
        ids=['A','Z','B']  # Typical shard-grouped, shuffle=False iteration.
        scores=np.asarray([11,33,22],dtype=np.float32)
        out[placement.locate_batch(ids)]=scores
        placement.assert_complete()
        np.testing.assert_equal(out,[11,22,33])
        with self.assertRaisesRegex(ValueError,'multiple times'):
            placement.locate_batch(['A'])

    def test_unseen_protein_is_error(self):
        placement=ProteinRowPlacement(['A','B'])
        placement.locate_batch(['B'])
        with self.assertRaisesRegex(ValueError,'missing 1 proteins'):
            placement.assert_complete()

    def test_input_pickle_excludes_future_label_splits(self):
        ns=('biological_process','molecular_function','cellular_component')
        sample={'proteins':['A'],'annotations':[[6]],'prop_annotations':[[0,6]]}
        metadata={'train':{k:sample for k in ns},
            'dark_weak_test_direct':{k:sample for k in ns},
            'dark_weak_test_2022_ipr2go_matched_direct':{k:sample for k in ns}}
        source=self.root/'metadata.pkl'
        with source.open('wb') as f:pickle.dump(metadata,f)
        (self.root/'unified'/'tables').mkdir(parents=True)
        pd.DataFrame({'protein_22':['A']}).to_csv(
            self.root/'unified'/'tables'/'full_dark_weak_cohort.tsv',sep='\t',index=False)
        args=argparse.Namespace(metadata=source,unified_root=self.root/'unified',
            output=self.root/'inference.pkl',id_dir=self.root/'ids',split='dark_weak_inference',
            source_splits=['dark_weak_test_direct','dark_weak_test_2022_ipr2go_matched_direct'],
            full_cohort=False,overwrite=False,replace_split=False,retain_benchmark_splits=False)
        manifest=prepare_split(args)
        with args.output.open('rb') as f:opened=pickle.load(f)
        self.assertNotIn('dark_weak_test_direct',opened)
        self.assertEqual(opened['train'],metadata['train'])
        self.assertEqual(opened['dark_weak_inference']['biological_process']['prop_annotations'],[[]])
        self.assertEqual(len(manifest['removed_benchmark_splits']),2)

    def test_teacher_probability_rows_aligned_by_historical_metadata(self):
        source=self.root/'teacher.npy'
        arr=np.zeros((3,21312),dtype=np.float16)
        arr[:,0]=[.3,.7,.9]  # Source order B,A,C.
        np.save(source,arr)
        meta={'exp_train':{'biological_process':{'proteins':['B','A','C']}}}
        metadata=self.root/'meta.pkl'
        with metadata.open('wb') as f:pickle.dump(meta,f)
        target=self.root/'desired.txt';target.write_text('A\nC\n')
        out=self.root/'aligned.npy'
        manifest=align(probabilities=source,target_ids=target,
                       output_probabilities=out,output_ids=self.root/'ids.txt',
                       task='bp',source_metadata=metadata,source_split='exp_train',block_size=1)
        got=np.load(out)
        np.testing.assert_allclose(got[:,0],[.7,.9],atol=.0003)
        self.assertEqual((self.root/'ids.txt').read_text(),'A\nC\n')
        self.assertFalse(manifest['historical_source_time_cutoff_verified'])

    def test_native_msabin_reads_view_if_installed(self):
        try:
            from experiments.msabin import MSABinaryDataset
        except ImportError:
            self.skipTest('MSABinaryDataset not installed in this isolated environment')
        first,second=self.create_msa_indices()
        targets=self.root/'target.txt';targets.write_text('A\nB\nZ\n')
        output=self.root/'view'/'index.pkl'
        make_view(targets,[first,second],output)
        ns='biological_process'
        meta={'dark_weak_inference':{ns:{'proteins':['A','B','Z'],
              'prop_annotations':[[],[],[]]}}}
        source=self.root/'meta.pkl'
        with source.open('wb') as f:pickle.dump(meta,f)
        ds=MSABinaryDataset(index_file=str(output),metadata_file=str(source),
            mode='dark_weak_inference',task=ns,num_classes=12,topk=2,max_len=4,
            need_proteins=True,read_mode='full',sample_strategy='head',
            shuffle_rows_at_getitem=False)
        self.assertEqual(ds.proteins,['A','B','Z'])
        (name,msa),y=ds[0]
        self.assertEqual(name,'A')
        self.assertEqual(tuple(msa.shape),(2,4))
        np.testing.assert_equal(msa.numpy().ravel(),np.arange(8,16))
        self.assertEqual(int(y.sum()),0)

if __name__=='__main__':unittest.main()
