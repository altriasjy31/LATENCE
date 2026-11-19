#!/usr/bin/env python3
"""Full-shape synthetic check: 40,384 proteins × 265 GO, 9 pairs / 7 GO.

Random fixtures only; this is NOT a biological result or real model inference.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/nbs'))
from eval_external_unseen_go_v095 import main


def export(folder, proteins, gos, matrix):
    folder.mkdir()
    (folder / 'protein_ids.txt').write_text('\n'.join(proteins) + '\n')
    (folder / 'go_ids.txt').write_text('\n'.join(gos) + '\n')
    np.save(folder / 'probabilities.npy', matrix)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    (folder / 'external_prediction_manifest.json').write_text(json.dumps({
        'schema': 'external-unseen-go-prediction-v1', 'complete': True,
        'model': 'synthetic_shape_fixture', 'protein_ids_file': 'protein_ids.txt',
        'go_ids_file': 'go_ids.txt', 'probability_file': 'probabilities.npy', 'sha256': hashes}))


def run():
    start = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        proteins = [f'P{i:06d}' for i in range(40384)]
        gos = [f'GO:{i + 1000000:07d}' for i in range(265)]
        positive_go = [0, 1, 2, 3, 4, 5, 6, 0, 1]
        pairs = [dict(protein_id=proteins[i*100], go_id=gos[g]) for i, g in enumerate(positive_go)]
        test = {'schema': 'latence-unseen-go-test-v1', 'complete': True, 'namespace': 'bp',
                'candidate_go_ids': gos, 'cohorts': {'dark_weak_all_exp': {
                    'protein_ids': proteins, 'evaluable_protein_ids': proteins[:40305], 'positives': pairs}}}
        test_path = root / 'test.json'
        test_path.write_text(json.dumps(test))
        # Many exact ties deliberately exercise tie credit at population scale.
        scores = (np.random.default_rng(3).integers(0, 101, (len(proteins), len(gos)), dtype=np.int16)
                  .astype(np.float32) / 100)
        export(root / 'a', proteins, gos, scores)
        export(root / 'b', proteins[::-1], gos[::-1], scores[::-1, ::-1])
        main(['--test-manifest', str(test_path), '--cohort', 'dark_weak_all_exp',
              '--predictions-dir', str(root / 'a'), '--reference-predictions-dir', str(root / 'b'),
              '--output-dir', str(root / 'evaluation')])
        result = json.loads((root / 'evaluation/metrics.json').read_text())
        left, right = result['scopes']['external_common'], result['scopes']['reference_common']
        assert left == right, 'Axis permutations changed metrics'
        assert left['counts']['cohort_protein_count'] == 40384
        assert left['counts']['identity_evaluable_protein_count'] == 40305
        assert left['GO_to_protein']['n_proteins_ranked_per_go'] == 40384
        assert left['GO_to_protein']['n_observed_positive_pairs'] == 9
        assert left['GO_to_protein']['observed_positive_go_count'] == 7
        assert left['counts']['candidate_go_count'] == 265
        table = (root / 'evaluation/paired_GO_to_protein.tsv').read_text().splitlines()
        assert len(table) == 10  # header + nine pairs
    print(f'PASS population shape 40384 x 265, nine pairs, seven GO, reordered axes and ties ({time.monotonic()-start:.2f}s)')


if __name__ == '__main__':
    run()
