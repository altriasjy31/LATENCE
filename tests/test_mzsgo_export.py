"""Lightweight adapter tests; optional genuine-model CPU test when torch exists.
Run from the package/project root:
python tests/test_mzsgo_export.py --repo-dir /path/to/MZSGO-src
"""
import argparse
import importlib.util
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'nbs'))
import export_mzsgo_unseen_scores as e


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo-dir', type=Path, required=True)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fasta = tmp / 'test.fasta'
        fasta.write_text('>p2 anything\nACD\n>p1\nEFG\n')
        assert e.read_fasta(fasta, ['p1', 'p2']) == ['EFG', 'ACD']
        try:
            e.read_fasta(fasta, ['p1'])
            raise AssertionError('missing FASTA record must fail')
        except ValueError:
            pass
        obo = tmp / 'go.obo'
        obo.write_text('[Term]\nid: GO:0000001\nname: test\nnamespace: biological_process\ndef: "definition" []\n\n[Term]\nid: GO:0000002\nname: old\nnamespace: biological_process\nis_obsolete: true\n[Typedef]\nid: part_of\n')
        gos, text, coverage = e.choose_terms(e.read_obo(obo), ['GO:0000002', 'GO:0000001', 'GO:0000003'], 'bp', 'all')
        assert gos == ['GO:0000001'] and text == ['test: definition']
        assert [r for _, r in coverage] == ['obsolete', 'supported', 'not_main_id_in_obo']
        prepared = tmp / 'prepared'
        prepared.mkdir()
        for name, content in {'protein_ids.txt': 'p1\n', 'go_ids.txt': 'GO:0000001\n', 'sequences.fasta': '>p1\nEFG\n', 'metadata.pkl': 'fake fixture bytes'}.items():
            (prepared / name).write_text(content)
        manifest = {'schema': 'external-unseen-go-input-v1', 'complete': True, 'namespace': 'bp', 'go_obo_sha256': e.sha256(obo), 'files': {name: {'sha256': e.sha256(prepared / name)} for name in ('protein_ids.txt', 'go_ids.txt', 'sequences.fasta', 'metadata.pkl')}}
        (prepared / 'external_input_manifest.json').write_text(json.dumps(manifest))
        assert e.validate_input_manifest(prepared, 'bp', obo)['namespace'] == 'bp'
        (prepared / 'go_ids.txt').write_text('GO:0000002\n')
        try:
            e.validate_input_manifest(prepared, 'bp', obo)
            raise AssertionError('Changed input GO axis accepted')
        except ValueError:
            pass
        local_nlp = tmp / 'nlp'
        local_nlp.mkdir()
        (local_nlp / 'model.safetensors').write_bytes(b'first weights')
        (local_nlp / 'config.json').write_text('{}')
        before = e.nlp_asset_identity(str(local_nlp), None)
        (local_nlp / 'model.safetensors').write_bytes(b'other weights')
        assert e.nlp_asset_identity(str(local_nlp), None) != before
        try:
            e.nlp_asset_identity('remote/model', 'main')
            raise AssertionError('Mutable NLP revision accepted')
        except ValueError:
            pass
        assert e.parse_domains('IPR000001;IPR000002;IPR000001') == ['IPR000001', 'IPR000002']
        try:
            e.parse_domains(['GO:0000001'])
            raise AssertionError('GO labels accepted as protein domains')
        except ValueError:
            pass
        dom = tmp / 'domain.pkl'
        with dom.open('wb') as f:
            pickle.dump({'IPR000001': {'embedding': np.array([1, 3], dtype=np.float32)}, 'IPR000002': {'embedding': np.array([3, 7], dtype=np.float32)}}, f)
        features, counts = e.domain_features([['IPR000001', 'IPR000002'], [], ['IPR999999']], dom, 2)
        np.testing.assert_array_equal(features, [[2, 5], [0, 0], [0, 0]])
        assert counts == [2, 0, 0]
        cache = tmp / 'cache'
        cache.mkdir()
        array, done, done_path = e.cache_array(cache, 'x', {'sequence_hash': 'abc'}, (3, 2))
        array[0] = [1, 2]
        done[0] = True
        e.save_done(array, done, done_path)
        array2, done2, _ = e.cache_array(cache, 'x', {'sequence_hash': 'abc'}, (3, 2))
        assert done2.tolist() == [True, False, False]
        np.testing.assert_array_equal(array2[0], [1, 2])
        try:
            e.cache_array(cache, 'x', {'sequence_hash': 'changed'}, (3, 2))
            raise AssertionError('stale cache accepted')
        except ValueError:
            pass
        print('PASS: axes, input integrity, immutable NLP provenance, GO support, domain mean/zero, cache resume')
        if importlib.util.find_spec('torch') is None:
            print('SKIP: actual CustomModel checkpoint loading and chunk parity (PyTorch unavailable)')
            return
        import torch
        spec = importlib.util.spec_from_file_location('_test_original_mzsgo', args.repo_dir / 'model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(13)
        original = module.CustomModel(esm_dim=4, nlp_dim=3, domain_size=2, hidden_dim=8).eval()
        ckpt = tmp / 'model.pt'
        torch.save({'state_dict': {'module.' + k: v for k, v in original.state_dict().items()}}, ckpt)
        loaded, dimensions = e.load_custom_model(args.repo_dir, ckpt, 'cpu')
        assert dimensions == {'esm_dim': 4, 'nlp_dim': 3, 'domain_size': 2, 'hidden_dim': 8}
        rng = np.random.default_rng(7)
        x = rng.normal(size=(5, 4)).astype(np.float32)
        d = rng.normal(size=(5, 2)).astype(np.float32)
        g = rng.normal(size=(7, 3)).astype(np.float32)
        output = np.empty((5, 7), dtype=np.float32)
        e.score_blocks(loaded, x, d, g, 2, 3, output, 'cpu')
        with torch.no_grad():
            xp = torch.tensor(x)[:, None, :].expand(-1, 7, -1).reshape(-1, 4)
            dp = torch.tensor(d)[:, None, :].expand(-1, 7, -1).reshape(-1, 2)
            reference = original(xp, dp, torch.tensor(g), 5).sigmoid().numpy()
        np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-7)
        # Native predict_single scores one protein across all requested labels.
        for i in range(5):
            with torch.no_grad():
                single = original(torch.tensor(x[i:i+1]).expand(7, -1), torch.tensor(d[i:i+1]).expand(7, -1), torch.tensor(g), 1).sigmoid().numpy()[0]
            np.testing.assert_allclose(output[i], single, rtol=1e-5, atol=1e-7)
        print('PASS: actual source CustomModel strict checkpoint load and chunked/single/full score parity')


if __name__ == '__main__':
    main()
