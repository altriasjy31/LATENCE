"""Synthetic preparation tests, preserving old splits and label provenance."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/nbs/prepare_deepgo_training_from_unidata.py'
spec = importlib.util.spec_from_file_location('prepare_training', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrepareTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.args = Namespace(metadata=str(self.root/'unidata.pkl'), protein_table=str(self.root/'swiss22.pkl'),
                              terms_file=str(self.root/'terms.txt'), output_dir=str(self.root/'out'),
                              namespace='bp', label_field='prop_annotations', label_format='go-ids',
                              label_representation='lists', train_split='train', valid_split=None)
        self.terms = ['GO:0000001', 'GO:0000002', 'GO:0000003']
        Path(self.args.terms_file).write_text('\n'.join(self.terms)+'\n')
        self.metadata = {'train': {'biological_process': {
            'proteins': ['B', 'A'], 'sequences': ['DDD', 'AAA'],
            'prop_annotations': [['GO:0000002'], ['GO:0000001', 'GO:0099999']]}},
            'val': {'bp': {'proteins': ['C'], 'prop_annotations': [['GO:0000003']]}},
            'ind_test': {'bp': {'proteins': ['X'], 'prop_annotations': [['GO:0099999']]}}}
        self.table = pd.DataFrame({'proteins':['A','B','C'], 'sequences':['AAA','DDD','CCC'],
                                   'interpros':[['IPR000001'],[],['IPR000002']],
                                   'prop_annotations':[['GO:0000003'],['GO:0099999'],['GO:0099999']]})
        self.write_inputs()

    def tearDown(self):
        self.tmp.cleanup()

    def write_inputs(self):
        pd.to_pickle(self.metadata, self.args.metadata)
        self.table.to_pickle(self.args.protein_table)

    def test_go_labels_and_exact_split_are_preserved(self):
        module.run(self.args)
        train = pd.read_pickle(Path(self.args.output_dir)/'train.pkl')
        valid = pd.read_pickle(Path(self.args.output_dir)/'valid.pkl')
        self.assertEqual(train.proteins.tolist(), ['B', 'A'])
        self.assertEqual(train.prop_annotations.tolist(), [['GO:0000002'], ['GO:0000001']])
        self.assertEqual(train.interpros.tolist(), [[], ['IPR000001']])
        self.assertEqual(valid.proteins.tolist(), ['C'])
        manifest = json.loads((Path(self.args.output_dir)/'prepare_training_manifest.json').read_text())
        self.assertEqual(manifest['split_inventory']['train']['excluded_nonvocabulary_pairs'], 1)
        self.assertTrue(manifest['split_inventory']['train']['sequence_verified_against_metadata'])
        self.assertFalse(manifest['split_inventory']['valid']['sequence_verified_against_metadata'])
        self.assertEqual(list(train), ['proteins','sequences','interpros','prop_annotations'])

    def test_explicit_index_labels(self):
        self.args.label_format = 'indices'
        self.metadata['train']['biological_process']['prop_annotations'] = [[1], [0, 2]]
        self.metadata['val']['bp']['prop_annotations'] = [[2]]
        self.write_inputs()
        module.run(self.args)
        frame = pd.read_pickle(Path(self.args.output_dir)/'train.pkl')
        self.assertEqual(frame.prop_annotations.tolist(), [['GO:0000002'], ['GO:0000001','GO:0000003']])

    def test_wrong_format_and_out_of_range_fail(self):
        self.metadata['train']['biological_process']['prop_annotations'] = [[1], [3]]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'expected GO:'):
            module.run(self.args)
        self.args.label_format = 'indices'
        with self.assertRaisesRegex(ValueError, 'outside supplied terms'):
            module.run(self.args)
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_binary_matrix_requires_explicit_representation(self):
        self.args.label_format = 'indices'
        self.metadata['train']['biological_process']['prop_annotations'] = np.array([[0,1,0],[1,0,0]])
        self.metadata['val']['bp']['prop_annotations'] = np.array([[0,0,1]])
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'ambiguous binary'):
            module.run(self.args)
        self.args.label_representation = 'multihot'
        module.run(self.args)
        frame = pd.read_pickle(Path(self.args.output_dir)/'train.pkl')
        self.assertEqual(frame.prop_annotations.tolist(), [['GO:0000002'], ['GO:0000001']])

    def test_sparse_multihot(self):
        try:
            from scipy.sparse import csr_matrix
        except ImportError:
            self.skipTest('SciPy not installed')
        self.args.label_format = 'indices'
        self.args.label_representation = 'multihot'
        self.metadata['train']['biological_process']['prop_annotations'] = csr_matrix([[0,1,0],[1,0,0]])
        self.metadata['val']['bp']['prop_annotations'] = csr_matrix([[0,0,1]])
        self.write_inputs()
        module.run(self.args)
        frame = pd.read_pickle(Path(self.args.output_dir)/'train.pkl')
        self.assertEqual(frame.prop_annotations.tolist(), [['GO:0000002'], ['GO:0000001']])

    def test_ambiguous_validation_split(self):
        self.metadata['valid'] = self.metadata['val']
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'Specify --valid-split'):
            module.run(self.args)
        self.args.valid_split = 'val'
        module.run(self.args)

    def test_missing_protein_does_not_drop_rows(self):
        self.table = self.table.iloc[:2]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'absent from old table'):
            module.run(self.args)
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_sequence_mismatch_and_split_overlap(self):
        self.metadata['train']['biological_process']['sequences'][0] = 'EEE'
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'sequence mismatch'):
            module.run(self.args)
        del self.metadata['train']['biological_process']['sequences']
        self.metadata['val']['bp']['proteins'] = ['A']
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, 'overlap in proteins'):
            module.run(self.args)


if __name__ == '__main__':
    unittest.main()
