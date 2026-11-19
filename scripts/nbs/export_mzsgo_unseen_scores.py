#!/usr/bin/env python3
"""Export unthresholded MZSGO scores on fixed LATENCE protein and GO axes.

Imports only the supplied repository's CustomModel; does not use its live
InterPro downloads, label parser, test filtering, or thresholded JSON output.
Frozen ESM/Qwen features are cached by input hashes. No training is performed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import pickle
import re
from pathlib import Path

import numpy as np

NAMESPACES = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def validate_input_manifest(input_dir, namespace, go_obo):
    path = input_dir / 'external_input_manifest.json'
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'external-unseen-go-input-v1' or manifest.get('complete') is not True:
        raise ValueError('Prepared input manifest is incomplete or has the wrong schema')
    if manifest.get('namespace') != namespace:
        raise ValueError('Requested namespace differs from prepared input namespace')
    for name in ('protein_ids.txt', 'go_ids.txt', 'sequences.fasta', 'metadata.pkl'):
        expected = manifest.get('files', {}).get(name, {}).get('sha256')
        if not expected or sha256(input_dir / name) != expected:
            raise ValueError(f'Prepared input integrity failed: {name}')
    expected_obo = manifest.get('go_obo_sha256')
    if expected_obo is None:
        test_path = Path(manifest.get('test_manifest_path', ''))
        if test_path.is_file():
            if sha256(test_path) != manifest.get('test_manifest_sha256'):
                raise ValueError('Original test manifest changed after preparing inputs')
            test = json.loads(test_path.read_text())
            expected_obo = test.get('sources', {}).get('new_obo', {}).get('sha256')
    if not expected_obo:
        raise ValueError('Input has no bound GO ontology checksum; rerun prepare_external_unseen_inputs.py')
    if sha256(go_obo) != expected_obo:
        raise ValueError('--go-obo differs from the ontology used to define this test')
    return manifest


def nlp_asset_identity(model, revision):
    path = Path(model)
    if path.is_dir():
        # Bind actual local bytes, not just a reusable directory name.
        files = sorted(p for p in path.rglob('*') if p.is_file()
                       and p.suffix in {'.json', '.safetensors', '.bin', '.model', '.txt', '.tiktoken'}
                       and '.cache' not in p.relative_to(path).parts)
        if not files or not any(p.suffix in {'.safetensors', '.bin'} for p in files):
            raise ValueError('Local NLP folder has no model weight files')
        return {'local_files_sha256': {str(p.relative_to(path)): sha256(p) for p in files}}
    if not revision or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Remote NLP features require an immutable 40-character --nlp-revision')
    return {'hf_repository': model, 'revision': revision}


def read_axis(path, go=False):
    ids = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate IDs in {path}')
    if go and any(not re.fullmatch(r'GO:\d{7}', x) for x in ids):
        raise ValueError(f'GO IDs must be canonical GO:nnnnnnn in {path}')
    return ids


def read_fasta(path, ids):
    records, key = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0]
            if key in records:
                raise ValueError(f'Duplicate FASTA ID {key}')
            records[key] = ''
        elif line.strip():
            if key is None:
                raise ValueError('FASTA sequence precedes a header')
            records[key] += line.strip().upper()
    if set(records) != set(ids):
        raise ValueError('FASTA and protein_ids.txt must have exactly the same protein set')
    if any(not records[x] for x in ids):
        raise ValueError('Empty protein sequences are unsupported')
    return [records[x] for x in ids]


def read_obo(path):
    terms, term = {}, None
    def finish():
        if term and 'id' in term:
            if term['id'] in terms:
                raise ValueError('Duplicate main GO ID ' + term['id'])
            terms[term['id']] = term.copy()
    for line in Path(path).read_text().splitlines():
        if line == '[Term]':
            finish()
            term = {}
        elif line.startswith('['):
            finish()
            term = None
        elif term is not None and ': ' in line:
            k, v = line.split(': ', 1)
            if k in {'id', 'name', 'namespace', 'def', 'is_obsolete'}:
                term[k] = v
    finish()
    return terms


def choose_terms(terms, requested, namespace, text_mode):
    supported, descriptions, coverage = [], [], []
    for go in requested:
        term = terms.get(go)
        reason = 'supported'
        text = ''
        if term is None:
            reason = 'not_main_id_in_obo'
        elif term.get('is_obsolete') == 'true':
            reason = 'obsolete'
        elif term.get('namespace') != NAMESPACES[namespace]:
            reason = 'different_namespace'
        else:
            name = term.get('name', '').strip()
            match = re.search(r'"(.*?)"', term.get('def', ''))
            definition = match.group(1) if match else ''
            text = name if text_mode == 'name' else definition if text_mode == 'def' else ': '.join(x for x in (name, definition) if x)
            if not text:
                reason = 'no_requested_text'
        coverage.append((go, reason))
        if reason == 'supported':
            supported.append(go)
            descriptions.append(text)
    return supported, descriptions, coverage


def parse_domains(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        value = re.split(r'[;,\s]+', value.strip()) if value.strip() else []
    if not isinstance(value, (list, tuple, set, np.ndarray)):
        raise ValueError(f'Expected an IPR ID collection, got {type(value).__name__}')
    vals = sorted(set(str(x).strip() for x in value if str(x).strip()))
    if any(not re.fullmatch(r'IPR\d{6}', x) for x in vals):
        raise ValueError(f'Expected raw IPR IDs, not GO annotations or domain descriptions: {vals[:4]}')
    return vals


def load_domains(args, ids):
    if args.domain_mode == 'none':
        return [[] for _ in ids], {'mode': 'explicit_no_domain_ablation'}
    if not args.domain_embeddings or not Path(args.domain_embeddings).is_file():
        raise ValueError('--domain-embeddings must point to the original MZSGO Qwen domain embedding dictionary')
    if args.domain_mode == 'metadata':
        import pandas as pd
        path = args.input_dir / 'metadata.pkl'
        df = pd.read_pickle(path)
        for col in (args.protein_field, args.interpro_field):
            if col not in df:
                raise ValueError(f'{path} missing {col}; supply the correct field or explicitly use --domain-mode none')
        if df[args.protein_field].astype(str).duplicated().any():
            raise ValueError('Duplicate protein metadata IDs')
        mapping = dict(zip(df[args.protein_field].astype(str), df[args.interpro_field]))
    else:
        if not args.domain_tsv:
            raise ValueError('--domain-mode tsv requires --domain-tsv (protein<TAB>IPR...; optional header protein/interpros)')
        path = Path(args.domain_tsv)
        mapping = {}
        for i, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            parts = line.split('\t')
            if i == 0 and parts[0] in {'protein', 'proteins', 'protein_id'}:
                continue
            if len(parts) != 2:
                raise ValueError(f'{path}:{i+1} requires exactly 2 tab-separated columns')
            if parts[0] in mapping:
                raise ValueError(f'Duplicate domain record: {parts[0]}')
            mapping[parts[0]] = parts[1]
    missing = [x for x in ids if x not in mapping]
    if missing:
        raise ValueError(f'{len(missing)} proteins absent from domain table. Add explicit empty IPR records; no implicit missing-as-empty. Examples: {missing[:4]}')
    return [parse_domains(mapping[x]) for x in ids], {'mode': args.domain_mode, 'path': str(path.resolve()), 'sha256': sha256(path)}


def load_custom_model(repo_dir, checkpoint, device):
    import torch
    path = Path(repo_dir) / 'model.py'
    spec = importlib.util.spec_from_file_location('_latence_mzsgo_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location='cpu')
    if isinstance(state, dict):
        state = state.get('model_state_dict', state.get('state_dict', state))
    if not isinstance(state, dict):
        raise ValueError('Expected a MZSGO state_dict checkpoint')
    if state and all(k.startswith('module.') for k in state):
        state = {k[len('module.'):]: v for k, v in state.items()}
    try:
        dims = {'esm_dim': int(state['esm_proj.0.weight'].shape[1]),
                'nlp_dim': int(state['nlp_proj.0.weight'].shape[1]),
                'domain_size': int(state['domain_proj.0.weight'].shape[1]),
                'hidden_dim': int(state['esm_proj.0.weight'].shape[0])}
    except (KeyError, AttributeError) as e:
        raise ValueError('Checkpoint does not match supplied MZSGO CustomModel') from e
    model = module.CustomModel(**dims)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, dims


def cache_array(cache_dir, name, expected, shape):
    """Resume completed rows only when provenance and array shape match exactly."""
    path, manifest = cache_dir / f'{name}.npy', cache_dir / f'{name}.json'
    done_path = cache_dir / f'{name}_done.npy'
    expected = dict(expected, shape=list(shape), dtype='float32')
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError(f'Feature cache provenance mismatch at {manifest}; use a new --cache-dir')
        arr = np.load(path, mmap_mode='r+')
        done = np.load(done_path)
        if arr.shape != shape or arr.dtype != np.float32 or done.shape != (shape[0],):
            raise ValueError(f'Invalid feature cache shape at {path}')
        if not np.isfinite(arr[done]).all():
            raise ValueError(f'Nonfinite completed feature rows at {path}')
    else:
        if path.exists() or done_path.exists():
            raise ValueError(f'Unbound partial cache at {path}; remove it or use a new --cache-dir')
        arr = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
        done = np.zeros(shape[0], dtype=bool)
        arr[:] = np.nan
        arr.flush()
        np.save(done_path, done)
        write_json(manifest, expected)
    return arr, done, done_path


def save_done(arr, done, done_path):
    arr.flush()
    temp = done_path.with_suffix('.tmp.npy')
    np.save(temp, done)
    temp.replace(done_path)


def text_embeddings(args, descriptions, dim, cache_dir, provenance):
    import torch
    array, done, done_path = cache_array(cache_dir, 'go_embeddings', provenance, (len(descriptions), dim))
    if done.all():
        return array
    from transformers import AutoModel, AutoTokenizer
    kw = {'revision': args.nlp_revision} if args.nlp_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(args.nlp_model, **kw)
    model = AutoModel.from_pretrained(args.nlp_model, **kw).to(args.device).eval()
    if int(model.config.hidden_size) != dim:
        raise ValueError(f'NLP model hidden_size {model.config.hidden_size} differs from checkpoint {dim}')
    with torch.inference_mode():
        for i, text in enumerate(descriptions):
            if done[i]:
                continue
            if args.text_encoding == 'source-test':
                # Matches go_embed.list_embedding: 1024 chars, 512-char chunks,
                # token-weighted mean, including each chunk's special tokens.
                text = text[:1024]
                vectors = []
                for start in range(0, len(text), 512):
                    encoded = tokenizer(text[start:start + 512], return_tensors='pt')
                    encoded = {k: v.to(args.device) for k, v in encoded.items()}
                    vectors.append(model(**encoded).last_hidden_state[0])
                value = torch.cat(vectors, dim=0).mean(dim=0)
            else:
                # Matches predict_util -> encode_go_description_single.
                encoded = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
                encoded = {k: v.to(args.device) for k, v in encoded.items()}
                hidden = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask']).last_hidden_state
                mask = encoded['attention_mask'].unsqueeze(-1).float()
                value = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9))[0]
            val = value.float().cpu().numpy()
            if val.shape != (dim,) or not np.isfinite(val).all():
                raise ValueError(f'Invalid text embedding row {i}')
            array[i] = val
            done[i] = True
            if (i + 1) % 25 == 0:
                save_done(array, done, done_path)
                print(f'[MZSGO] GO features {int(done.sum())}/{len(done)}', flush=True)
    save_done(array, done, done_path)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return array


def sequence_embeddings(args, sequences, dim, cache_dir, provenance):
    import torch
    array, done, done_path = cache_array(cache_dir, 'esm_embeddings', provenance, (len(sequences), dim))
    if done.all():
        return array
    import esm
    if args.esm_checkpoint:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.esm_checkpoint)
    else:
        loader = getattr(esm.pretrained, args.esm_model, None)
        if loader is None:
            raise ValueError('fair-esm 2 is required with esm.pretrained.' + args.esm_model)
        model, alphabet = loader()
    model = model.to(args.device).eval()
    if int(model.embed_dim) != dim:
        raise ValueError(f'ESM embed_dim {model.embed_dim} differs from checkpoint {dim}')
    converter = alphabet.get_batch_converter()
    with torch.inference_mode():
        for i, sequence in enumerate(sequences):
            if done[i]:
                continue
            seq = sequence[:args.max_seq_length] if args.max_seq_length else sequence
            _, _, tokens = converter([('protein', seq)])
            outputs = model(tokens.to(args.device), repr_layers=[model.num_layers], return_contacts=False)
            value = outputs['representations'][model.num_layers][0, 1:1 + len(seq)].mean(0)
            val = value.float().cpu().numpy()
            if val.shape != (dim,) or not np.isfinite(val).all():
                raise ValueError(f'Invalid sequence embedding row {i}')
            array[i] = val
            done[i] = True
            if (i + 1) % 100 == 0:
                save_done(array, done, done_path)
                print(f'[MZSGO] ESM features {int(done.sum())}/{len(done)}', flush=True)
    save_done(array, done, done_path)
    del model, alphabet, converter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return array


def domain_features(domains, path, dim):
    features = np.zeros((len(domains), dim), dtype=np.float32)
    counts = []
    dictionary = {}
    if path:
        with Path(path).open('rb') as f:
            dictionary = pickle.load(f)
        if not isinstance(dictionary, dict) or not dictionary:
            raise ValueError('Domain embeddings must be a non-empty MZSGO dictionary')
    for i, iprs in enumerate(domains):
        valid = []
        for ipr in iprs:
            if ipr not in dictionary:
                continue
            entry = dictionary[ipr]
            if not isinstance(entry, dict) or 'embedding' not in entry:
                raise ValueError(f'Unexpected MZSGO domain cache record for {ipr}')
            vector = np.asarray(entry['embedding'], dtype=np.float32)
            if vector.shape != (dim,) or not np.isfinite(vector).all():
                raise ValueError(f'Domain vector dimension/nonfinite error for {ipr}')
            valid.append(vector)
        if valid:
            features[i] = np.asarray(valid).mean(0)
        counts.append(len(valid))
    return features, counts


def score_blocks(model, esm, domain, go, batch_size, go_batch_size, output, device):
    import torch
    n, m = len(esm), len(go)
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            x = torch.as_tensor(np.array(esm[start:stop]), device=device)
            d = torch.as_tensor(np.array(domain[start:stop]), device=device)
            for j in range(0, m, go_batch_size):
                end = min(j + go_batch_size, m)
                g = torch.as_tensor(np.array(go[j:end]), device=device)
                k, b = end-j, stop-start
                xp = x[:, None, :].expand(-1, k, -1).reshape(-1, x.shape[1])
                dp = d[:, None, :].expand(-1, k, -1).reshape(-1, d.shape[1])
                scores = torch.sigmoid(model(xp, dp, g, b)).float().cpu().numpy()
                if scores.shape != (b, k) or not np.isfinite(scores).all():
                    raise ValueError('MZSGO returned invalid scores')
                output[start:stop, j:end] = scores
            if hasattr(output, 'flush'):
                output.flush()
            if start % (batch_size * 100) == 0:
                print(f'[MZSGO] scores {stop}/{n}', flush=True)


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--go-obo', type=Path, required=True)
    p.add_argument('--namespace', choices=NAMESPACES, default='bp')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--domain-mode', choices=['metadata', 'tsv', 'none'], required=True)
    p.add_argument('--domain-embeddings', type=Path)
    p.add_argument('--domain-tsv', type=Path)
    p.add_argument('--protein-field', default='proteins')
    p.add_argument('--interpro-field', default='interpros')
    p.add_argument('--nlp-model', default='Qwen/Qwen3-Embedding-4B', help='Original model ID or downloaded local folder')
    p.add_argument('--nlp-revision', default='5cf2132abc99cad020ac570b19d031efec650f2b', help='Immutable NLP model commit (default is Qwen 4B snapshot named in supplied source config)')
    p.add_argument('--esm-model', default='esm2_t33_650M_UR50D')
    p.add_argument('--esm-checkpoint', help='Local fair-esm .pt (matching contact-regression file may be required)')
    p.add_argument('--text-mode', choices=['all', 'name', 'def'], default='all')
    p.add_argument('--text-encoding', choices=['source-test', 'source-predict'], default='source-test')
    p.add_argument('--max-seq-length', type=int, default=2048, help='Original training/test truncation; 0 matches unlimited native FASTA predictor')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--go-batch-size', type=int, default=128)
    p.add_argument('--training-go-ids', type=Path, help='Exact checkpoint supervised GO set; enables measured seen-GO coverage audit')
    p.add_argument('--training-protein-ids', type=Path, help='Exact checkpoint training proteins; exact ID overlap only, not homology exclusion')
    return p


def main():
    args = make_parser().parse_args()
    if args.batch_size < 1 or args.go_batch_size < 1 or args.max_seq_length < 0:
        raise ValueError('Positive batch sizes and nonnegative max-seq-length required')
    for path in (args.repo_dir / 'model.py', args.checkpoint, args.go_obo, args.input_dir / 'protein_ids.txt', args.input_dir / 'go_ids.txt', args.input_dir / 'sequences.fasta'):
        if not path.is_file():
            raise FileNotFoundError(path)
    input_manifest = validate_input_manifest(args.input_dir, args.namespace, args.go_obo)
    ids = read_axis(args.input_dir / 'protein_ids.txt')
    requested = read_axis(args.input_dir / 'go_ids.txt', go=True)
    if not ids:
        raise ValueError('Empty protein axis')
    sequences = read_fasta(args.input_dir / 'sequences.fasta', ids)
    gos, descriptions, coverage = choose_terms(read_obo(args.go_obo), requested, args.namespace, args.text_mode)
    domains, domain_meta = load_domains(args, ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_manifest = args.output_dir / 'external_prediction_manifest.json'
    if final_manifest.exists():
        raise FileExistsError(f'Completed output exists: {final_manifest}; use a new --output-dir')
    cache = args.cache_dir or args.output_dir / 'features'
    cache.mkdir(parents=True, exist_ok=True)
    import torch
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly pass --device cpu if intended')
    # Keep lightweight scoring model on CPU during the two larger encoder phases.
    model, dims = load_custom_model(args.repo_dir, args.checkpoint, 'cpu')
    go_prov = {'descriptions_sha256': digest_json(list(zip(gos, descriptions))), 'obo_sha256': sha256(args.go_obo), 'model': args.nlp_model, 'revision': args.nlp_revision, 'asset_identity': nlp_asset_identity(args.nlp_model, args.nlp_revision), 'encoding': args.text_encoding, 'text_mode': args.text_mode}
    esm_prov = {'sequence_axis_sha256': digest_json(list(zip(ids, sequences))), 'model': args.esm_model, 'checkpoint_sha256': sha256(args.esm_checkpoint) if args.esm_checkpoint else None, 'max_seq_length': args.max_seq_length, 'pooling': 'residue_mean'}
    if gos:
        go = text_embeddings(args, descriptions, dims['nlp_dim'], cache, go_prov)
        esm_features = sequence_embeddings(args, sequences, dims['esm_dim'], cache, esm_prov)
        dfeatures, counts = domain_features(domains, args.domain_embeddings if args.domain_mode != 'none' else None, dims['domain_size'])
        temp = args.output_dir / 'probabilities.partial.npy'
        output = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float32, shape=(len(ids), len(gos)))
        output[:] = np.nan
        score_blocks(model.to(args.device).eval(), esm_features, dfeatures, go, args.batch_size, args.go_batch_size, output, args.device)
        output.flush()
        del output
        temp.replace(args.output_dir / 'probabilities.npy')
    else:
        counts = [0] * len(ids)
        np.save(args.output_dir / 'probabilities.npy', np.empty((len(ids), 0), dtype=np.float32))
    (args.output_dir / 'protein_ids.txt').write_text('\n'.join(ids) + '\n')
    (args.output_dir / 'go_ids.txt').write_text('\n'.join(gos) + ('\n' if gos else ''))
    (args.output_dir / 'go_coverage.tsv').write_text('go_id\tsupport\n' + ''.join(f'{g}\t{r}\n' for g, r in coverage))
    (args.output_dir / 'feature_coverage.tsv').write_text('protein\tsequence_length\tencoded_length\tn_ipr\tn_ipr_in_embedding_cache\n' + ''.join(f'{p}\t{len(seq)}\t{min(len(seq), args.max_seq_length) if args.max_seq_length else len(seq)}\t{len(ds)}\t{n}\n' for p, seq, ds, n in zip(ids, sequences, domains, counts)))
    train_gos = set(read_axis(args.training_go_ids, go=True)) if args.training_go_ids else None
    train_proteins = set(read_axis(args.training_protein_ids)) if args.training_protein_ids else None
    paths = ['protein_ids.txt', 'go_ids.txt', 'probabilities.npy']
    manifest = {'schema': 'external-unseen-go-prediction-v1', 'complete': True, 'model': 'MZSGO' if args.domain_mode != 'none' else 'MZSGO_no_domain_ablation', 'protein_ids_file': paths[0], 'go_ids_file': paths[1], 'probability_file': paths[2], 'sha256': {x: sha256(args.output_dir / x) for x in paths}, 'shape': [len(ids), len(gos)], 'namespace': args.namespace, 'score_space': 'raw_sigmoid_probability', 'hierarchy_propagated': False, 'thresholded': False, 'requested_go_ids': requested, 'unsupported_go_ids': [g for g, reason in coverage if reason != 'supported'], 'checkpoint': {'path': str(args.checkpoint.resolve()), 'sha256': sha256(args.checkpoint)}, 'source_model_sha256': sha256(args.repo_dir / 'model.py'), 'dimensions': dims, 'input_manifest_sha256': sha256(args.input_dir / 'external_input_manifest.json'), 'test_manifest_sha256': input_manifest['test_manifest_sha256'], 'cohort': input_manifest['cohort'], 'go_features': go_prov, 'sequence_features': esm_prov, 'domain_features': dict(domain_meta, cache_sha256=sha256(args.domain_embeddings) if args.domain_mode != 'none' else None, proteins_without_supported_ipr=sum(n == 0 for n in counts)), 'training_overlap_audit': {'status': 'provided_axes_exact_id_only' if train_gos is not None and train_proteins is not None else 'unverified_checkpoint_training_axes', 'requested_go_seen_in_training': sorted(set(requested) & train_gos) if train_gos is not None else None, 'protein_ids_seen_in_training': sorted(set(ids) & train_proteins) if train_proteins is not None else None, 'training_go_axis_sha256': sha256(args.training_go_ids) if args.training_go_ids else None, 'training_protein_axis_sha256': sha256(args.training_protein_ids) if args.training_protein_ids else None}, 'software': {'torch': torch.__version__, 'numpy': np.__version__}, 'protocol': 'External checkpoint transfer; unseen relative to LATENCE does not imply unseen relative to MZSGO without checkpoint training-axis audit.'}
    write_json(final_manifest, manifest)
    print(f'[MZSGO] complete: {len(ids)} proteins x {len(gos)}/{len(requested)} GO -> {args.output_dir}')


if __name__ == '__main__':
    main()
