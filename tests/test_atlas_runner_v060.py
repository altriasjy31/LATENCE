"""Synthetic four-view end-to-end tests; no model inference or GPU required."""
from __future__ import annotations
import contextlib
import importlib.util
import io
import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('atlas_runner_v060', ROOT/'scripts/dark_weak/eval_atlas_v060.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture(root):
    unified = root/'unified'
    metadata, config = {}, {}
    for ns, namespace in mod.ranking.NS.items():
        folder = unified/'labels'/ns
        folder.mkdir(parents=True)
        gos = [f'GO:{i:07}' for i in range(1, 7)] + [mod.ranking.ROOT_IDS[ns], 'GO:0000008']
        can = gos.copy(); can[-1] = gos[0]
        (folder/'go_ids.txt').write_text('\n'.join(gos)+'\n')
        pd.DataFrame({'model_column': range(8), 'model_go_id': gos,
                      'canonical_go_id_2022': can}).to_csv(folder/'vocabulary_mapping.tsv', sep='\t', index=False)
        direct = {'future_exp': {'A': [2]}, 'future_ipr': {'B': [0, 7], 'C': [5]},
                  'future_union': {'A': [2], 'B': [0, 7], 'C': [5]},
                  'history': {'A': [1], 'B': [3], 'D': [4]}}
        for layer, entries in direct.items():
            metadata.setdefault(mod.SPLITS[layer], {})[namespace] = {
                'proteins': list(entries), 'annotations': list(entries.values()),
                'prop_annotations': [sorted(set(cols) | {6}) for cols in entries.values()]}
        metadata.setdefault('train', {})[namespace] = {
            'proteins': ['T1', 'T2', 'T3'], 'prop_annotations': [[0, 1, 6], [0, 2, 6], [3, 4, 5, 6]]}
        pred = root/'prediction'/ns
        pred.mkdir(parents=True)
        (pred/'ids.txt').write_text('A\nB\nC\nD\n')
        (pred/'gos.txt').write_text('\n'.join(gos)+'\n')
        config[ns] = {}
        for name in ('backbone', 'expert', 'modelout', 'nbs'):
            values = np.full((4, 8), .1, np.float32)
            values[:, 6] = .99
            if name == 'nbs':
                values[0, 2] = .9
                values[1, [0, 7]] = .9
                values[2, 5] = .9
            np.save(pred/f'{name}.npy', values)
            config[ns][name] = {'probability': str(pred/f'{name}.npy'),
                               'protein_ids': str(pred/'ids.txt'), 'go_ids': str(pred/'gos.txt')}
    table = unified/'tables'
    table.mkdir()
    pd.DataFrame({'protein_22': ['A', 'B', 'C', 'D'],
                  'sequence_sha256': ['hA', 'hB', 'hC', 'hD']}).to_csv(table/'full_dark_weak_cohort.tsv', sep='\t', index=False)
    mp, cp = root/'metadata.pkl', root/'config.json'
    mp.write_bytes(pickle.dumps(metadata)); cp.write_text(json.dumps(config))
    return mp, cp, unified, metadata, config


def options(mp, cp, unified, out, *extra):
    return mod.parser().parse_args(['--metadata', str(mp), '--prediction-config', str(cp),
        '--unified-root', str(unified), '--classification', 'none', '--bootstrap', '3',
        '--ks', '1', '3', '--out', str(out), *extra])


def quiet(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.run(args)


class AtlasRunnerTests(unittest.TestCase):
    def test_complete_nested_suite_and_new_relations(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mp, cp, unified, metadata, config = fixture(root)
            before = mod.common.sha256(mp)
            inventory, audit = quiet(options(mp, cp, unified, root/'out', '--train-metadata', str(mp)))
            self.assertEqual(len(inventory), 12)
            self.assertEqual(audit['unique_counts']['future_union']['unique_proteins_across_namespaces'], 3)
            self.assertEqual(audit['unique_counts']['future_union']['raw_direct_pairs'], 12)
            self.assertEqual(audit['unique_counts']['future_union']['canonical_direct_pairs_without_root'], 9)
            self.assertEqual(mod.common.sha256(mp), before)
            evidence = pd.read_csv(root/'out/atlas_observed_relations.tsv', sep='\t')
            self.assertEqual(len(evidence), 18)
            recovery = pd.read_csv(root/'out/stage2_incremental_recovery.tsv', sep='\t')
            future = recovery[recovery.layer.eq('future_union') & recovery.k.eq(1) & recovery.hit_policy.eq('conservative')]
            self.assertTrue((future.n_target_gained == 3).all())
            self.assertTrue((future.n_newly_recovered_proteins == 3).all())
            details = pd.read_csv(root/'out/stage2_incremental_pairs.tsv', sep='\t')
            self.assertEqual(set(details.state), {'gained'})
            self.assertTrue(details.novel_vs_history.all())
            metrics = pd.read_csv(root/'out/atlas_metrics.tsv', sep='\t')
            self.assertTrue({'rare', 'rare_le_5', 'medium', 'common', 'all'}.issubset(set(metrics.frequency_bin)))
            with self.assertRaises(FileExistsError):
                quiet(options(mp, cp, unified, root/'out'))

    def test_inventory_without_probabilities(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mp, cp, unified, _, _ = fixture(root)
            args = mod.parser().parse_args(['--metadata', str(mp), '--unified-root', str(unified),
                '--inventory-only', '--out', str(root/'out')])
            inventory, audit = quiet(args)
            self.assertEqual(len(inventory), 12)
            self.assertTrue(audit['inventory_only'])
            self.assertTrue(pd.read_csv(root/'out/prediction_coverage.tsv', sep='\t').empty)

    def test_missing_history_prediction_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mp, cp, unified, metadata, config = fixture(root)
            for ns in config:
                ids = Path(config[ns]['backbone']['protein_ids'])
                ids.write_text('A\nB\nC\n')
                for model in config[ns]:
                    p = Path(config[ns][model]['probability'])
                    np.save(p, np.load(p)[:3])
            with self.assertRaisesRegex(ValueError, 'required proteins missing'):
                quiet(options(mp, cp, unified, root/'out'))
            self.assertFalse((root/'out').exists())

    def test_union_and_historical_novelty_protection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mp, cp, unified, metadata, config = fixture(root)
            metadata[mod.SPLITS['history']]['biological_process']['annotations'][0] = [2]
            metadata[mod.SPLITS['history']]['biological_process']['prop_annotations'][0] = [2, 6]
            mp.write_bytes(pickle.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'novelty violated'):
                quiet(options(mp, cp, unified, root/'out'))

    def test_source_input_hash_and_missing_reference_checks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mp, cp, unified, metadata, config = fixture(root)
            config['bp']['backbone']['protein_ids_sha256'] = '0' * 64
            cp.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'protein_ids_sha256 differs'):
                quiet(options(mp, cp, unified, root/'hash'))
            config['bp']['backbone'].pop('protein_ids_sha256')
            config['bp'].pop('expert'); cp.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'all Stage-1 references'):
                quiet(options(mp, cp, unified, root/'refs'))

    def test_optional_future_classification_native_and_stage1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mp, cp, unified, metadata, config = fixture(root)
            quiet(options(mp, cp, unified, root/'out', '--namespaces', 'bp',
                          '--classification', 'future', '--skip-stage1-official'))
            classification = pd.read_csv(root/'out/classification_metrics.tsv', sep='\t')
            self.assertEqual(len(classification), 12)
            self.assertEqual(set(classification.evidence_layer), {'future_union', 'future_exp', 'future_ipr'})
            self.assertIn('stage1_hist__fmax_micro', classification)
            self.assertIn('standard_micro_ap', classification)
            self.assertEqual(set(classification.label_field), {'prop_annotations'})

    def test_stage1_only_allows_empty_target_table_with_correct_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); mp, cp, unified, metadata, config = fixture(root)
            for ns in config:
                config[ns].pop('nbs')
            cp.write_text(json.dumps(config))
            quiet(options(mp, cp, unified, root/'out'))
            recovery = pd.read_csv(root/'out/stage2_incremental_recovery.tsv', sep='\t')
            self.assertTrue(recovery.empty)
            self.assertIn('hit_policy', recovery)
            self.assertIn('layer', recovery)


if __name__ == '__main__':
    unittest.main()
