"""CPU data/registry contract checks; optional torch checks when installed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'scripts/nbs'))
import export_deepgo_unseen_scores as ex
import train_deepgo_unseen_baseline as tr


class DeepGOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Pass original repo location when running from another unpack directory.
        import os
        cls.repo = Path(os.environ.get('DEEPGO_SOURCE', str(BASE.parent/'source/deepgo2')))
        if not (cls.repo/'deepgo/data.py').exists():
            raise unittest.SkipTest('Set DEEPGO_SOURCE to the supplied deepgo2 directory')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Path(self.tmp.name)
        self.inp = self.p/'input'; self.inp.mkdir()
        (self.inp/'protein_ids.txt').write_text('P1\nP2\n')
        (self.inp/'go_ids.txt').write_text('GO:0000002\nGO:0000003\n')
        pd.DataFrame({'proteins':['P2','P1'], 'sequences':['CCC','AAA'], 'interpros':[[],['IPR1']]}).to_pickle(self.inp/'metadata.pkl')
        (self.inp/'sequences.fasta').write_text('>P1\nAAA\n>P2\nCCC\n')
        self.bind_inputs()
        self.terms = self.p/'terms.pkl'
        pd.DataFrame({'gos':['GO:0000001']}).to_pickle(self.terms)
        self.norm = self.p/'go.norm'
        self.norm.write_text('GO_0000002 SubClassOf GO_0000001\nGO_0000002 SubClassOf RO_0000001 some GO_0000001\n')

    def bind_inputs(self):
        m = {'schema':'external-unseen-go-input-v1', 'complete':True, 'files':{n:{'sha256':ex.sha256(self.inp/n)} for n in ['protein_ids.txt','go_ids.txt','sequences.fasta','metadata.pkl']}}
        (self.inp/'external_input_manifest.json').write_text(json.dumps(m))

    def tearDown(self):
        self.tmp.cleanup()

    def test_registry_preserves_native_order(self):
        terms, nfs, rels, zeros, full = ex.registry(self.repo, self.terms, self.norm)
        self.assertEqual(full, {'GO:0000001': 0, 'GO:0000002': 1})
        self.assertEqual(rels, {'RO:0000001': 0})
        self.assertEqual(nfs[0], [(1,0)])

    def test_fixed_ipr_encoder_and_axis_order(self):
        df = ex.load_metadata(self.inp, ['P1','P2'])
        np.testing.assert_array_equal(ex.ipr_batch(df, [0,1], {'IPR1':0,'IPR2':1}), [[1,0],[0,0]])

    def test_missing_only_exports_zero_columns(self):
        (self.inp/'go_ids.txt').write_text('GO:0000003\nGO:0000001\n')
        self.bind_inputs()
        args = ex.parser().parse_args(['--repo-dir',str(self.repo),'--input-dir',str(self.inp),'--terms-file',str(self.terms),'--normal-forms',str(self.norm),'--output-dir',str(self.p/'out')])
        ex.run(args)
        m = json.loads((self.p/'out/external_prediction_manifest.json').read_text())
        self.assertEqual(m['schema'], 'external-unseen-go-prediction-v1')
        self.assertEqual(m['status'], 'no_supported_go')
        self.assertEqual(np.load(self.p/'out/probabilities.npy').shape, (2,0))
        cov = pd.read_csv(self.p/'out/go_coverage.tsv', sep='\t')
        self.assertEqual(list(cov.registry_role), ['absent_from_checkpoint_registry', 'supervised_class'])

    def test_target_annotations_removed_all_annotation_fields(self):
        df = pd.DataFrame({'proteins':['P'],'prop_annotations':[['GO:0000001','GO:0000002']], 'exp_annotations':[['GO:0000002']], 'interpros':[['IPR1']]})
        clean, counts = tr.clean_labels(df, {'GO:0000002'}, 'prop_annotations')
        self.assertEqual(clean.prop_annotations[0], ['GO:0000001'])
        self.assertEqual(clean.exp_annotations[0], [])
        self.assertEqual(counts, {'prop_annotations':1,'exp_annotations':1})

    def test_holdout_excludes_sequence_duplicates_with_other_names(self):
        df = pd.DataFrame({'proteins':['OTHER','TRAIN'], 'sequences':[' A A A*','WWW']})
        keep, counts = tr.exclude_holdouts(df, [self.inp], 'sequences')
        self.assertEqual(keep.proteins.tolist(), ['TRAIN'])
        self.assertEqual(counts['excluded_by_exact_sequence'], 1)

    def test_training_audit_runs_without_torch(self):
        for name, prots, seqs in [('train', ['T1','T2'], ['DDD','EEE']), ('valid', ['V1','V2'], ['FFF','GGG'])]:
            pd.DataFrame({'proteins':prots, 'sequences':seqs, 'interpros':[['IPR1'],[]],
                          'prop_annotations':[['GO:0000001','GO:0000002'],[]]}).to_pickle(self.p/(name+'.pkl'))
        (self.p/'candidates.txt').write_text('GO:0000002\n')
        args = tr.parser().parse_args(['--repo-dir',str(self.repo),'--train-pickle',str(self.p/'train.pkl'),
            '--valid-pickle',str(self.p/'valid.pkl'),'--terms-file',str(self.terms),'--candidate-go-ids',str(self.p/'candidates.txt'),
            '--normal-forms',str(self.norm),'--output-dir',str(self.p/'trained'),'--audit-only','--exclude-input-dir',str(self.inp)])
        args.feature_column = 'interpros'
        tr.run(args)
        m = json.loads((self.p/'trained/training_manifest.json').read_text())
        self.assertEqual(m['n_supported_candidate_go'],1)
        self.assertEqual(m['removed_candidate_annotations']['train'],{'prop_annotations':1})

    def test_coded_labels_are_rejected(self):
        for labels in [['GO:0000001|EXP'], [0], 'GO:0000001', float('nan')]:
            path = self.p/'bad_labels.pkl'
            pd.DataFrame({'proteins':['P'], 'interpros':[[]], 'prop_annotations':[labels]}).to_pickle(path)
            with self.assertRaises(ValueError):
                tr.read_frame(path, 'proteins', 'interpros', 'prop_annotations')

    def test_no_old_positive_training_split_is_rejected(self):
        for name, proteins in [('train',['T']), ('valid',['V'])]:
            pd.DataFrame({'proteins':proteins, 'interpros':[['IPR1']],
                          'prop_annotations':[['GO:0000002']]}).to_pickle(self.p/(name+'.pkl'))
        (self.p/'candidates.txt').write_text('GO:0000002\n')
        args = tr.parser().parse_args(['--repo-dir',str(self.repo),'--train-pickle',str(self.p/'train.pkl'),
            '--valid-pickle',str(self.p/'valid.pkl'),'--terms-file',str(self.terms),'--candidate-go-ids',str(self.p/'candidates.txt'),
            '--normal-forms',str(self.norm),'--output-dir',str(self.p/'empty_train'),'--audit-only'])
        args.feature_column = 'interpros'
        with self.assertRaisesRegex(ValueError, 'no old-task positive'):
            tr.run(args)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
    def test_full_export_with_synthetic_checkpoint(self):
        import torch
        cp = self.p/'test.th'
        model = ex.native_model_class(self.repo)(2,1,1,1,'cpu',hidden_dim=4,embed_dim=4)
        model.eval()
        torch.save(model.state_dict(),cp)
        iprs = self.p/'interpros.pkl'
        pd.DataFrame({'interpros':['IPR1','IPR2']}).to_pickle(iprs)
        args = ex.parser().parse_args(['--repo-dir',str(self.repo),'--input-dir',str(self.inp),
            '--terms-file',str(self.terms),'--normal-forms',str(self.norm),'--checkpoint',str(cp),
            '--interpros-file',str(iprs),'--output-dir',str(self.p/'full_export')])
        args.feature_column = 'interpros'
        ex.run(args)
        actual = np.load(self.p/'full_export/probabilities.npy')
        model.all_gos = torch.tensor([1])
        with torch.no_grad():
            expected = model(torch.tensor([[1.,0.],[0.,0.]])).numpy()
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
        self.assertEqual(actual.shape,(2,1))
        m = json.loads((self.p/'full_export/external_prediction_manifest.json').read_text())
        self.assertEqual(m['status'],'scored')
        self.assertEqual(m['missing_go_ids'],['GO:0000003'])
        self.assertEqual(m['sha256']['probabilities.npy'],ex.sha256(self.p/'full_export/probabilities.npy'))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
    def test_native_forward_arbitrary_class_rows(self):
        import torch
        model = ex.native_model_class(self.repo)(2,1,1,1,'cpu',hidden_dim=4,embed_dim=4)
        model.eval()
        x = torch.tensor([[1.,0.],[0.,0.]])
        original = model(x)
        model.all_gos = torch.tensor([0,1])
        actual = model(x)
        self.assertTrue(torch.allclose(original, actual[:,:1]))
        with torch.no_grad():
            expected = torch.sigmoid(model.net(x) @ (model.go_embed.weight + model.rel_embed.weight[-1]).T + model.go_rad.weight.abs().T)
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
