"""Original source adapter checks; native PyTorch tests activate on the server."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/nbs'))
import export_deepgozero_original_scores as ex
import train_deepgozero_original_baseline as tr
from eval_external_unseen_go_v095 import load_external


class OriginalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(os.environ.get('DEEPGOZERO_SOURCE',
            str(ROOT.parents[1] / 'deepgozero_original_work/source/deepgozero')))
        if not (cls.repo / 'deepgozero.py').is_file():
            raise unittest.SkipTest('Set DEEPGOZERO_SOURCE to the original uploaded repository')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = Path(self.tmp.name)
        self.inp = self.p / 'inputs'
        self.inp.mkdir()
        (self.inp / 'protein_ids.txt').write_text('P1\nP2\n')
        (self.inp / 'go_ids.txt').write_text('GO:0000002\nGO:0000003\n')
        (self.inp / 'sequences.fasta').write_text('>P1\nAAA\n>P2\nCCC\n')
        pd.DataFrame({'proteins': ['P2', 'P1'], 'sequences': ['CCC', 'AAA'],
                      'interpros': [[], ['IPR000001']]}).to_pickle(self.inp / 'metadata.pkl')
        self.bind()
        self.terms = self.p / 'terms.pkl'
        self.iprs = self.p / 'interpros.pkl'
        pd.DataFrame({'gos': ['GO:0000001']}).to_pickle(self.terms)
        pd.DataFrame({'interpros': ['IPR000001', 'IPR000002']}).to_pickle(self.iprs)
        self.norm = self.p / 'go.norm'
        self.norm.write_text('GO_0000002 SubClassOf GO_0000001\n'
                             'GO_0000002 SubClassOf RO_0000001 some GO_0000001\n')

    def tearDown(self):
        self.tmp.cleanup()

    def bind(self):
        manifest = {'schema': 'external-unseen-go-input-v1', 'complete': True,
            'files': {name: {'sha256': ex.sha256(self.inp / name)} for name in
                ('protein_ids.txt', 'go_ids.txt', 'sequences.fasta', 'metadata.pkl')}}
        (self.inp / 'external_input_manifest.json').write_text(json.dumps(manifest))

    def args(self, output, *extra):
        return ex.parser().parse_args(['--repo-dir', str(self.repo), '--input-dir', str(self.inp),
            '--terms-file', str(self.terms), '--normal-forms', str(self.norm),
            '--output-dir', str(self.p / output), *map(str, extra)])

    def test_native_registry_order_and_relation(self):
        terms, nfs, rels, zeros, full = ex.registry(self.repo, self.terms, self.norm)
        self.assertEqual(full, {'GO:0000001': 0, 'GO:0000002': 1})
        self.assertEqual(rels, {'RO:0000001': 0})
        self.assertEqual(nfs[3], [(1, 0, 0)])

    def test_norm_typo_correction_is_explicit_and_local(self):
        nodes, path = ex.source_nodes(self.repo, ['Residual', 'MLPBlock', 'DGELModel'])
        before = ex.sha256(path)
        correction = ex.FixMLPNorm('batchnorm1d')
        tree = correction.visit(ast.Module(body=nodes, type_ignores=[]))
        self.assertEqual(correction.count, 1)
        self.assertIn('nn.BatchNorm1d', ast.unparse(tree))
        self.assertEqual(ex.sha256(path), before)
        self.assertIn('predict_zero', ast.unparse(tree))

    def test_infer_normalization_from_state_buffers(self):
        state = {'net.0.layer_norm.weight': [], 'net.0.layer_norm.bias': []}
        self.assertEqual(ex.infer_mlp_norm(state), 'layernorm')
        state['net.0.layer_norm.running_mean'] = []
        self.assertEqual(ex.infer_mlp_norm(state), 'batchnorm1d')
        with self.assertRaisesRegex(ValueError, 'buffers imply'):
            ex.infer_mlp_norm(state, 'layernorm')

    def test_zero_coverage_exports_evaluable_diagnostic(self):
        (self.inp / 'go_ids.txt').write_text('GO:0000003\nGO:0000001\n')
        self.bind()
        ex.run(self.args('zero'))
        model = load_external(self.p / 'zero', ['P1', 'P2'], ['GO:0000003', 'GO:0000001'])
        self.assertEqual(model['scores'].shape, (2, 0))
        self.assertEqual(model['manifest']['status'], 'no_supported_go')
        reasons = pd.read_csv(self.p / 'zero/go_coverage.tsv', sep='\t').reason.tolist()
        self.assertEqual(reasons, ['absent_from_checkpoint_registry', 'supervised_class'])

    def test_definition_gate_is_logical_not_natural_language(self):
        defs = self.p / 'definitions_go.txt'
        defs.write_text('GO_0000003: GO_0000001 and RO_0000001 some GO_0000001\n')
        ex.run(self.args('definition_gate', '--candidate-policy', 'definitions', '--definitions-file', defs))
        cov = pd.read_csv(self.p / 'definition_gate/go_coverage.tsv', sep='\t')
        self.assertEqual(cov.iloc[0].reason, 'excluded_by_native_definition_policy')
        self.assertEqual(np.load(self.p / 'definition_gate/probabilities.npy').shape, (2, 0))

    def test_audit_partial_registry_without_torch(self):
        ex.run(self.args('audit', '--audit-only'))
        payload = json.loads((self.p / 'audit/registry_audit.json').read_text())
        self.assertEqual(payload['supported_go_count'], 1)
        self.assertFalse((self.p / 'audit/probabilities.npy').exists())

    def test_singleton_batches_merged_without_dropping(self):
        batches = list(tr.training_batches(np.arange(5), 2, True))
        self.assertEqual(list(map(len, batches)), [2, 3])
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(5))
        self.assertEqual([len(x) for x in tr.training_batches(np.arange(3), 1, False)], [1, 1, 1])
        with self.assertRaises(ValueError):
            list(tr.training_batches(np.arange(1), 2, True))

    def test_normalization_provenance_guard(self):
        unverified, path = tr.normalization_provenance(self.norm)
        self.assertEqual(unverified['status'], 'unverified_legacy_normal_forms')
        sidecar = self.p / 'normalization_manifest.json'
        sidecar.write_text(json.dumps({'schema': 'original-deepgozero-normalization-v1',
            'complete': False, 'mode': 'native', 'status': 'native_diagnostic_unsafe'}))
        with self.assertRaisesRegex(ValueError, 'native Groovy diagnostic'):
            tr.normalization_provenance(self.norm)
        data = {'schema': 'original-deepgozero-normalization-v1', 'complete': True,
            'mode': 'corrected', 'status': 'normalized', 'backend': 'java-source-file',
            'go_norm_sha256': ex.sha256(self.norm)}
        sidecar.write_text(json.dumps(data))
        verified, path = tr.normalization_provenance(self.norm)
        self.assertEqual(verified['status'], 'verified_corrected_normalization')
        self.norm.write_text(self.norm.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            tr.normalization_provenance(self.norm)

    def test_training_audit_original_identity(self):
        for split, proteins, seqs in [('train', ['T1','T2','T3'], ['DDD','EEE','FFF']),
                                      ('valid', ['V1','V2'], ['GGG','HHH'])]:
            pd.DataFrame({'proteins': proteins, 'sequences': seqs,
                'interpros': [['IPR000001']] * len(proteins),
                'prop_annotations': [['GO:0000001', 'GO:0000002']] * len(proteins)}).to_pickle(self.p / f'{split}.pkl')
        candidates = self.p / 'candidates.txt'
        candidates.write_text('GO:0000002\n')
        args = tr.parser().parse_args(['--repo-dir', str(self.repo), '--train-pickle', str(self.p / 'train.pkl'),
            '--valid-pickle', str(self.p / 'valid.pkl'), '--terms-file', str(self.terms),
            '--candidate-go-ids', str(candidates), '--normal-forms', str(self.norm),
            '--mlp-norm', 'batchnorm1d', '--exclude-input-dir', str(self.inp), '--audit-only',
            '--output-dir', str(self.p / 'training_audit')])
        args.feature_column = 'interpros'
        tr.run(args)
        manifest = json.loads((self.p / 'training_audit/training_manifest.json').read_text())
        self.assertEqual(manifest['schema'], 'original-deepgozero-label-zero-shot-training-v1')
        self.assertEqual(manifest['embed_dim'], 1024)
        self.assertEqual(manifest['mlp_norm'], 'batchnorm1d')
        self.assertEqual(manifest['removed_candidate_annotations']['train'], {'prop_annotations': 3})

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable')
    def test_native_predict_zero_and_export(self):
        import torch
        for norm in ('batchnorm1d', 'layernorm'):
            cls = ex.original_model_class(self.repo, norm)
            model = cls(2, 1, 1, 1, 'cpu', hidden_dim=4, embed_dim=4).eval()
            cp = self.p / (norm + '.th')
            torch.save(model.state_dict(), cp)
            x = torch.tensor([[1., 0.], [0., 0.]])
            with torch.no_grad():
                expected = model.predict_zero(x, torch.tensor([1])).numpy()
                seen = model.predict_zero(x, torch.tensor([0]))
                self.assertTrue(torch.allclose(seen, model(x)))
            ex.run(self.args(norm, '--checkpoint', cp, '--interpros-file', self.iprs))
            actual = np.load(self.p / norm / 'probabilities.npy')
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
            exported = load_external(self.p / norm, ['P1','P2'], ['GO:0000002', 'GO:0000003'])
            self.assertEqual(exported['manifest']['checkpoints'][0]['mlp_norm'], norm)


if __name__ == '__main__':
    unittest.main()
