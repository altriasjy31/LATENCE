#!/usr/bin/env python3
"""Deterministic adapter-evaluation tests; no model weights are required."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/nbs'))
from eval_external_unseen_go_v095 import main, load_external, evaluate, read_test
from eval_unseen_go_v095 import ranking_metrics, exact_threshold_metrics

G = ['GO:1000001', 'GO:1000002', 'GO:1000003']
IDS = ['A', 'B', 'C', 'D']


def export(folder, ids, gos, scores, **extra):
    folder.mkdir()
    (folder / 'protein_ids.txt').write_text('\n'.join(ids) + '\n')
    (folder / 'go_ids.txt').write_text(('\n'.join(gos) + '\n') if gos else '')
    np.save(folder / 'probabilities.npy', np.array(scores, dtype=np.float32).reshape(len(ids), len(gos)))
    manifest = dict(schema='external-unseen-go-prediction-v1', complete=True,
                    protein_ids_file='protein_ids.txt', go_ids_file='go_ids.txt',
                    probability_file='probabilities.npy', model='synthetic fixture',
                    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}, **extra)
    (folder / 'external_prediction_manifest.json').write_text(json.dumps(manifest))


def export_latence_fixture(folder, ids, gos, scores):
    """Create valid synthetic v095 provenance for checking the native loader branch."""
    folder.mkdir()
    protein_path = folder / 'protein_ids.txt'
    protein_path.write_text('\n'.join(ids) + '\n')
    np.save(folder / 'probabilities.npy', np.asarray(scores, dtype=np.float32))
    rows = [3, 4, 5]
    targets = [dict(output_column=i, requested_go_id=g, canonical_go_id=g, ontology_row=row,
                    task_columns=[], seen_task_columns=[], stage2_supervised=False,
                    readout='single_shared_GO_scorer', zero_shot_scope='Stage2_unsupervised_target_only')
               for i, (g, row) in enumerate(zip(gos, rows))]
    with (folder / 'targets.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(targets)
    payload = dict(schema='nbs-v095-requested-go-prediction-v1', complete=True,
                   protein_ids_file='protein_ids.txt', probability_file='probabilities.npy',
                   target_go_file='targets.tsv', num_proteins=len(ids), num_target_GO=len(gos),
                   targets=targets, output_probability_sha256=hashlib.sha256((folder / 'probabilities.npy').read_bytes()).hexdigest(),
                   identity=dict(ontology_bundle_sha256='a' * 64, requested_go_ids=gos,
                                 target_ontology_rows=rows,
                                 protein_ids_sha256=hashlib.sha256(protein_path.read_bytes()).hexdigest()),
                   ontology_extension=dict(schema='nbs-v095-anchored-ontology-replacement-v1',
                                           old_coordinate_anchors_verified=True,
                                           new_trainable_parameter_rows_added=False, num_ontology_go=6))
    (folder / 'prediction_manifest.json').write_text(json.dumps(payload))


def expect_error(call, substring):
    try:
        call()
    except (ValueError, FileExistsError) as error:
        assert substring in str(error), str(error)
    else:
        raise AssertionError(f'Expected failure: {substring}')


def check(folder):
    test = dict(schema='latence-unseen-go-test-v1', complete=True, namespace='bp',
                candidate_go_ids=G, candidate_ontology_rows=[3, 4, 5],
                ontology_bundle_sha256='a' * 64, cohorts={'ind_test_exp': dict(
                protein_ids=IDS, evaluable_protein_ids=IDS[:3], positives=[
                    dict(protein_id='A', go_id=G[0]), dict(protein_id='A', go_id=G[2]),
                    dict(protein_id='B', go_id=G[1])])})
    path = folder / 'test.json'
    path.write_text(json.dumps(test))
    # Exact tie of A/C for GO1, reversed protein order, reversed GO order.
    external = folder / 'external'
    export(external, ['D', 'C', 'B', 'A'], [G[1], G[0]],
           [[.9, .8], [.2, .4], [.6, .2], [.4, .4]])
    ref = folder / 'reference'
    export(ref, IDS, G, [[.7, .3, .9], [.2, .6, .1], [.4, .2, .1], [.3, .1, .2]])
    out = folder / 'eval'
    main(['--test-manifest', str(path), '--cohort', 'ind_test_exp',
          '--predictions-dir', str(external), '--reference-predictions-dir', str(ref),
          '--top-k', '1', '2', '--retrieval-top-k', '1', '2', '--output-dir', str(out)])
    result = json.loads((out / 'metrics.json').read_text())
    ext = result['scopes']['external_supported']
    assert result['models']['external']['protein_rows_reordered'] is True
    assert result['models']['external']['missing_go_ids'] == [G[2]]
    assert ext['counts']['excluded_positive_pairs_due_to_GO_coverage'] == 1
    assert ext['counts']['retained_observed_positive_proteins'] == 2
    assert ext['counts']['cohort_protein_count'] == 4
    assert ext['counts']['identity_evaluable_protein_count'] == 3
    assert ext['counts']['retrieval_protein_count'] == 4
    assert ext['protein_to_GO']['ranking']['pair_recall@1'] == .75
    assert ext['protein_to_GO']['ranking']['precision_all_evaluable_rows_proxy@1'] == .5
    assert ext['protein_to_GO']['ranking']['protein_macro_f1_observed_proxy@1'] == .75
    # GO1: D first, tied A/C occupy positions 2..3; GO2: D first, B second.
    assert ext['GO_to_protein']['pair_observed_recall@1'] == 0
    assert ext['GO_to_protein']['pair_observed_recall@2'] == .75
    assert ext['GO_to_protein']['go_macro_observed_ap_proxy'] == (1 / 3 + 1 / 2) / 2
    assert result['common_candidate_go_ids'] == G[:2]
    assert result['scopes']['reference_supported']['counts']['retained_observed_positive_pairs'] == 3
    assert result['scopes']['reference_common']['counts']['retained_observed_positive_pairs'] == 2
    assert len((out / 'paired_GO_to_protein.tsv').read_text().splitlines()) == 3
    # Reuse metric kernels exactly, independent of axis permutation.
    positives = {'A': {G[0]: {}}, 'B': {G[1]: {}}}
    scores = np.array([[.4, .4], [.2, .6]], dtype=np.float32)
    truth = np.array([[True, False], [False, True]])
    expected = ranking_metrics(G[:2], IDS[:2], scores, truth, positives, [1, 2], 3)[0]
    for key, value in expected.items():
        assert ext['protein_to_GO']['ranking'][key] == value
    assert ext['protein_to_GO']['classification_positive_proteins'] == exact_threshold_metrics(scores, truth)
    # Native reference is read through the strict v095 provenance validator.
    native = folder / 'native_reference'
    export_latence_fixture(native, IDS, G, [[.7, .3, .9], [.2, .6, .1], [.4, .2, .1], [.3, .1, .2]])
    native_out = folder / 'native_eval'
    main(['--test-manifest', str(path), '--predictions-dir', str(external),
          '--reference-predictions-dir', str(native), '--top-k', '1', '2',
          '--retrieval-top-k', '1', '2', '--output-dir', str(native_out)])
    nr = json.loads((native_out / 'metrics.json').read_text())
    assert nr['models']['reference']['kind'] == 'latence-v095'
    assert nr['scopes']['reference_common'] == result['scopes']['reference_common']
    # Empty support remains diagnostic and never becomes zero performance.
    empty = folder / 'empty'
    export(empty, IDS, [], np.empty((4, 0)))
    empty_out = folder / 'empty_eval'
    main(['--test-manifest', str(path), '--predictions-dir', str(empty),
          '--reference-predictions-dir', str(ref), '--output-dir', str(empty_out)])
    zero = json.loads((empty_out / 'metrics.json').read_text())['scopes']['external_supported']
    assert zero['status'] == 'no_supported_GO_metrics_undefined'
    assert zero['protein_to_GO']['ranking'] is None and zero['GO_to_protein'] is None
    assert zero['counts']['excluded_positive_pairs_due_to_GO_coverage'] == 3
    # Observed-label-empty cohort preserves full ranking pool and undefined recalls.
    test['cohorts']['ind_test_exp']['positives'] = []
    path.write_text(json.dumps(test))
    empty_labels = folder / 'empty_labels'
    main(['--test-manifest', str(path), '--predictions-dir', str(external), '--output-dir', str(empty_labels)])
    scope = json.loads((empty_labels / 'metrics.json').read_text())['scopes']['external_supported']
    assert scope['status'] == 'no_observed_positive_metrics_undefined'
    assert scope['GO_to_protein']['n_proteins_ranked_per_go'] == 4
    assert scope['GO_to_protein']['pair_observed_recall@10'] is None
    # No silent protein-pool reduction and no stale/synthetic data acceptance.
    incomplete = folder / 'incomplete'
    export(incomplete, IDS[:3], G[:2], [[.1, .2]] * 3)
    expect_error(lambda: load_external(incomplete, IDS, G), '1 missing')
    bound = folder / 'wrong_bound'
    export(bound, IDS, G[:2], [[.1, .2]] * 4, test_manifest_sha256='0' * 64)
    expect_error(lambda: load_external(bound, IDS, G, test_path=path), 'binding differs')
    nonfinite = folder / 'nonfinite'
    export(nonfinite, IDS, G[:2], [[.1, .2], [.2, np.nan], [.1, .2], [.1, .2]])
    expect_error(lambda: load_external(nonfinite, IDS, G), 'finite')
    # An evidence-independent score export may serve other views of the same full axis.
    view_test = json.loads(json.dumps(test))
    view_test['cohorts']['dark_weak_all_exp'] = dict(protein_ids=IDS,
        evaluable_protein_ids=IDS[:3], positives=[dict(protein_id='A', go_id=G[0])])
    view_test['cohorts']['dark_weak_all_union'] = dict(protein_ids=IDS[::-1],
        evaluable_protein_ids=IDS[::-1], positives=[dict(protein_id='A', go_id=G[0]),
                                                 dict(protein_id='B', go_id=G[1])])
    view_test['cohorts']['different_axis'] = dict(protein_ids=IDS[:3],
        evaluable_protein_ids=IDS[:3], positives=[])
    view_path = folder / 'views.json'
    view_path.write_text(json.dumps(view_test))
    view_hash = hashlib.sha256(view_path.read_bytes()).hexdigest()
    bound_views = folder / 'views_export'
    export(bound_views, IDS, G[:2], [[.8, .2], [.1, .9], [.3, .2], [.1, .4]],
           test_manifest_sha256=view_hash, cohort='dark_weak_all_exp',
           namespace='bp', requested_go_ids=G)
    view_out = folder / 'view_eval'
    main(['--test-manifest', str(view_path), '--cohort', 'dark_weak_all_union',
          '--predictions-dir', str(bound_views), '--output-dir', str(view_out)])
    vr = json.loads((view_out / 'metrics.json').read_text())
    assert vr['scopes']['external_supported']['counts']['retained_observed_positive_pairs'] == 2
    assert vr['models']['external']['prediction_provenance']['cohort'] == 'dark_weak_all_exp'
    expect_error(lambda: load_external(bound_views, IDS[:3], G, test_path=view_path,
                                        cohort_name='different_axis'), '1 extra')
    wrong_origin = folder / 'wrong_origin'
    export(wrong_origin, IDS, G[:2], [[.1, .2]] * 4, test_manifest_sha256=view_hash,
           cohort='different_axis', namespace='bp', requested_go_ids=G)
    expect_error(lambda: load_external(wrong_origin, IDS, G, test_path=view_path,
                                        cohort_name='dark_weak_all_exp'), 'origin cohort has a different')
    unknown_origin = folder / 'unknown_origin'
    export(unknown_origin, IDS, G[:2], [[.1, .2]] * 4, test_manifest_sha256=view_hash,
           cohort='unknown', namespace='bp', requested_go_ids=G)
    expect_error(lambda: load_external(unknown_origin, IDS, G, test_path=view_path,
                                        cohort_name='dark_weak_all_exp'), 'origin cohort is absent')
    wrong_namespace = folder / 'wrong_namespace'
    export(wrong_namespace, IDS, G[:2], [[.1, .2]] * 4, namespace='mf')
    expect_error(lambda: load_external(wrong_namespace, IDS, G, test_path=view_path), 'namespace differs')
    partial_request = folder / 'partial_request'
    export(partial_request, IDS, G[:2], [[.1, .2]] * 4, requested_go_ids=G[:2])
    expect_error(lambda: load_external(partial_request, IDS, G), 'requested GO axis differs')
    (external / 'probabilities.npy').write_bytes(b'corrupted')
    expect_error(lambda: load_external(external, IDS, G), 'SHA256 mismatch')
    expect_error(lambda: main(['--test-manifest', str(path), '--predictions-dir', str(ref),
                              '--output-dir', str(out)]), 'new output')


def check_real_deepgo_zero_support_export(folder):
    """Exercise the actual exporter/reader boundary without weights or PyTorch."""
    import os
    import pandas as pd
    import export_deepgo_unseen_scores as deepgo
    repo = Path(os.environ.get('DEEPGO_SOURCE', str(ROOT.parent / 'source/deepgo2')))
    if not (repo / 'deepgo/data.py').exists():
        print('DeepGO integration: SKIP (set DEEPGO_SOURCE to attached source)')
        return
    inputs = folder / 'prepared_deepgo'
    inputs.mkdir()
    (inputs / 'protein_ids.txt').write_text('P1\nP2\n')
    (inputs / 'go_ids.txt').write_text('GO:0000003\nGO:0000001\n')
    (inputs / 'sequences.fasta').write_text('>P1\nAAA\n>P2\nCCC\n')
    pd.DataFrame(dict(proteins=['P1', 'P2'], sequences=['AAA', 'CCC'], interpros=[[], []])).to_pickle(inputs / 'metadata.pkl')
    im = dict(schema='external-unseen-go-input-v1', complete=True,
              files={p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs.iterdir()})
    (inputs / 'external_input_manifest.json').write_text(json.dumps(im))
    terms = folder / 'terms.pkl'
    pd.DataFrame(dict(gos=['GO:0000001'])).to_pickle(terms)
    norm = folder / 'go.norm'
    norm.write_text('GO_0000002 SubClassOf GO_0000001\n')
    output = folder / 'actual_deepgo_output'
    args = deepgo.parser().parse_args(['--repo-dir', str(repo), '--input-dir', str(inputs),
           '--terms-file', str(terms), '--normal-forms', str(norm), '--output-dir', str(output)])
    deepgo.run(args)
    loaded = load_external(output, ['P1', 'P2'], ['GO:0000003', 'GO:0000001'])
    assert loaded['gos'] == [] and loaded['scores'].shape == (2, 0)
    print('DeepGO exporter -> external evaluator zero-support integration: PASS')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as temp:
        check(Path(temp))
        check_real_deepgo_zero_support_export(Path(temp))
    print('external unseen evaluation synthetic checks: PASS')
