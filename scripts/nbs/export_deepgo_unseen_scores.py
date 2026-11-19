#!/usr/bin/env python3
"""Export dense zero-shot scores from the supplied deepgo2 (DeepGO-SE) source.

Only classes already fitted in a checkpoint's ORIGINAL terms/normal-forms
registry can be scored. Do not replace go.norm with a newer file at inference.
The InterPro variant is the repository's DeepGOZero variant, not a claim of
bitwise reproduction of the separate 2022 DeepGOZero repository.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ids(path, allow_empty=False):
    vals = [v.strip() for v in Path(path).read_text().splitlines() if v.strip()]
    if len(set(vals)) != len(vals) or (not vals and not allow_empty):
        raise ValueError(f'Empty or duplicate IDs: {path}')
    return vals


def vocab(path, field):
    p = Path(path)
    if p.suffix in ('.pkl', '.pickle'):
        obj = pd.read_pickle(p)
        if isinstance(obj, pd.DataFrame):
            vals = obj[field].tolist()
        elif isinstance(obj, pd.Series):
            vals = obj.tolist()
        else:
            vals = list(obj)
        vals = [str(v) for v in vals]
    else:
        vals = ids(p)
    if not vals or len(set(vals)) != len(vals):
        raise ValueError(f'Invalid vocabulary: {path}')
    return vals


def source_function(repo, relative, name):
    """Load the exact source definition, avoiding unrelated optional DGL imports."""
    p = Path(repo) / relative
    tree = ast.parse(p.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f'Expected exactly one {name} in {p}')
    return ast.Module(body=nodes, type_ignores=[]), p


def registry(repo, terms_file, normal_forms):
    terms = vocab(terms_file, 'gos')
    tree, p = source_function(repo, 'deepgo/data.py', 'load_normal_forms')
    env = {}
    exec(compile(tree, str(p), 'exec'), env)
    forms = env['load_normal_forms'](str(normal_forms), {g: i for i, g in enumerate(terms)})
    if len(forms) != 6:
        raise ValueError('Unsupported load_normal_forms return format')
    *nfs, rels, zeros = forms
    full = {g: i for i, g in enumerate(terms)}
    if set(zeros) & set(full):
        raise ValueError('Overlapping supervised/zero registry')
    full.update(zeros)
    if sorted(full.values()) != list(range(len(full))):
        raise ValueError('Noncontiguous checkpoint registry')
    return terms, nfs, rels, zeros, full


def native_model_class(repo):
    import torch
    base_path = Path(repo) / 'deepgo/base.py'
    spec = importlib.util.spec_from_file_location('_latence_deepgo_native_base', base_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    tree, p = source_function(repo, 'deepgo/models.py', 'DeepGOModel')
    env = dict(th=torch, nn=torch.nn, BaseDeepGOModel=base.BaseDeepGOModel,
               Residual=base.Residual, MLPBlock=base.MLPBlock)
    exec(compile(tree, str(p), 'exec'), env)
    return env['DeepGOModel']


def load_metadata(input_dir, protein_ids):
    df = pd.read_pickle(Path(input_dir) / 'metadata.pkl')
    if 'proteins' not in df.columns:
        raise ValueError('metadata.pkl must contain proteins')
    df = df.copy()
    df['proteins'] = df.proteins.astype(str)
    if df.proteins.duplicated().any():
        raise ValueError('Duplicate metadata protein IDs')
    df = df.set_index('proteins', drop=False)
    missing = set(protein_ids) - set(df.index)
    if missing:
        raise ValueError(f'{len(missing)} protein IDs absent from metadata')
    return df.loc[protein_ids].reset_index(drop=True)


def ipr_values(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        raise ValueError('InterPro field must be a list/set of IDs, not a string')
    return list(value)


def ipr_batch(df, rows, iprs, field='interpros'):
    out = np.zeros((len(rows), len(iprs)), dtype=np.float32)
    for i, row in enumerate(rows):
        for term in ipr_values(df.iloc[int(row)][field]):
            if term in iprs:
                out[i, iprs[term]] = 1.0  # Fix missing assignment in uploaded data.py.
    return out


def feature_source(args, df, protein_ids, expected_dim):
    if args.feature_kind == 'interpros':
        if not args.interpros_file:
            raise ValueError('--interpros-file is required for the InterPro model')
        iprs = {g: i for i, g in enumerate(vocab(args.interpros_file, 'interpros'))}
        if len(iprs) != expected_dim or args.feature_column not in df:
            raise ValueError('IPR vocabulary length / metadata feature column mismatch')
        counts = [sum(t in iprs for t in set(ipr_values(v))) for v in df[args.feature_column]]
        report = {'kind': 'interpros', 'dimension': len(iprs), 'zero_covered_feature_proteins': int(sum(c == 0 for c in counts)),
                  'note': 'All protein rows retained; zero-feature rows may produce ties.',
                  'interpros_sha256': sha256(args.interpros_file)}
        return lambda rows: ipr_batch(df, rows, iprs, args.feature_column), report
    if expected_dim != 2560:
        raise ValueError('Native ESM2-3B feature dimension must equal 2560')
    if args.feature_column in df.columns:
        arr = np.asarray(df[args.feature_column].tolist(), dtype=np.float32)
        origin = 'metadata column ' + args.feature_column
    else:
        import torch
        if args.esm_cache:
            cache = Path(args.esm_cache)
            obj = torch.load(cache, map_location='cpu', weights_only=False)
            cids = [str(v) for v in obj['proteins']]
            features = obj['data']
            if len(set(cids)) != len(cids) or set(cids) != set(protein_ids):
                raise ValueError('ESM cache protein axis mismatch')
            order = {g: i for i, g in enumerate(cids)}
            arr = np.asarray(features.cpu() if hasattr(features, 'cpu') else features, dtype=np.float32)[[order[g] for g in protein_ids]]
            origin = str(cache.resolve())
        else:
            p = Path(args.repo_dir) / 'deepgo/extract_esm.py'
            spec = importlib.util.spec_from_file_location('_latence_native_extract_esm', p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # No implicit disk cache reuse. Native extractor may reorder rows by length.
            cids, features = mod.extract_esm(str(Path(args.input_dir) / 'sequences.fasta'),
                model_location=args.esm_model, device=args.device, out_file=None)
            if len(set(cids)) != len(cids) or set(cids) != set(protein_ids):
                raise ValueError('Native ESM extractor changed the protein axis')
            order = {g: i for i, g in enumerate(cids)}
            arr = features.cpu().numpy()[[order[g] for g in protein_ids]]
            origin = 'native extractor, layer 36 mean pooling, first 1022 residues'
    if arr.shape != (len(protein_ids), expected_dim) or not np.isfinite(arr).all():
        raise ValueError('Invalid ESM features')
    return lambda rows: arr[rows], {'kind': 'esm2_t36_3B_UR50D', 'dimension': 2560,
        'origin': origin, 'cache_sha256': sha256(args.esm_cache) if args.esm_cache else None, 'sequence_binding': 'external cache/metadata must correspond to these exact sequences' if args.esm_cache or args.feature_column in df else 'computed from sequences.fasta'}


def save_manifest(out, manifest):
    names = ['protein_ids.txt', 'go_ids.txt', 'probabilities.npy']
    manifest.update(schema='external-unseen-go-prediction-v1', complete=True,
                    protein_ids_file=names[0], go_ids_file=names[1], probability_file=names[2],
                    sha256={n: sha256(out / n) for n in names})
    (out / 'external_prediction_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def input_provenance(input_dir):
    p = Path(input_dir) / 'external_input_manifest.json'
    manifest = json.loads(p.read_text())
    if manifest.get('schema') != 'external-unseen-go-input-v1' or manifest.get('complete') is not True:
        raise ValueError('Use prepare_external_unseen_inputs.py to create complete inputs')
    for name in ['protein_ids.txt', 'go_ids.txt', 'sequences.fasta', 'metadata.pkl']:
        expected = manifest.get('files', {}).get(name, {}).get('sha256')
        if not expected or sha256(Path(input_dir)/name) != expected:
            raise ValueError(f'Input artifact hash mismatch: {name}')
    return dict(input_manifest_sha256=sha256(p), **{k: manifest.get(k) for k in
        ['namespace', 'cohort', 'test_manifest_sha256', 'protein_sequence_axis_sha256', 'interpro_features']})


def run(args):
    provenance = input_provenance(args.input_dir)
    out = Path(args.output_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Choose a new output directory: {out}')
    out.mkdir(parents=True)
    proteins = ids(Path(args.input_dir) / 'protein_ids.txt')
    requested = ids(Path(args.input_dir) / 'go_ids.txt')
    terms, nfs, rels, zeros, full = registry(args.repo_dir, args.terms_file, args.normal_forms)
    supported, coverage = [], []
    for g in requested:
        role = 'zero_class' if g in zeros else 'supervised_class' if g in full else 'absent_from_checkpoint_registry'
        ok = role == 'zero_class' or (role == 'supervised_class' and args.allow_supervised_targets)
        if ok:
            supported.append(g)
        coverage.append({'go_id': g, 'supported': ok, 'registry_role': role, 'checkpoint_index': full.get(g, '')})
    pd.DataFrame(coverage).to_csv(out / 'go_coverage.tsv', sep='\t', index=False)
    (out / 'protein_ids.txt').write_text('\n'.join(proteins) + '\n')
    (out / 'go_ids.txt').write_text(''.join(g + '\n' for g in supported))
    manifest = {**provenance, 'model': 'deepgo2 repository / DeepGOZero-' + args.feature_kind,
        'requested_go_ids': requested, 'requested_go_count': len(requested), 'supported_go_count': len(supported),
        'missing_go_ids': [g for g in requested if g not in supported],
        'coverage_file': 'go_coverage.tsv', 'n_proteins': len(proteins),
        'allow_supervised_targets': args.allow_supervised_targets,
        'registry': {'terms_file': str(Path(args.terms_file).resolve()), 'terms_sha256': sha256(args.terms_file),
                     'normal_forms_file': str(Path(args.normal_forms).resolve()), 'normal_forms_sha256': sha256(args.normal_forms)},
        'source_sha256': {n: sha256(Path(args.repo_dir) / n) for n in ['deepgo/base.py', 'deepgo/models.py', 'deepgo/data.py']},
        'readout': 'native sigmoid dot product with hasFunction relation and absolute class radius; no threshold/rounding/ancestor propagation'}
    if args.audit_only or not supported:
        if supported:
            (out / 'registry_audit.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(f'Registry audit: {len(supported)}/{len(requested)} supported. No inference requested.')
            return
        np.save(out / 'probabilities.npy', np.empty((len(proteins), 0), dtype=np.float32))
        manifest['status'] = 'no_supported_go'
        save_manifest(out, manifest)
        print('No supported zero-shot GO rows. Coverage-only empty-column export; no performance claim.')
        return
    if not args.checkpoint:
        raise ValueError('--checkpoint required for scoring (repeat for an equal-weight ensemble)')
    import torch
    cls = native_model_class(args.repo_dir)
    checkpoint_entries = []
    state = torch.load(args.checkpoint[0], map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    dim = int(state['go_embed.weight'].shape[1])
    input_dim = int(state['net.0.linear.weight'].shape[1])
    df = load_metadata(args.input_dir, proteins)
    feature_fn, feature_report = feature_source(args, df, proteins, input_dim)
    pred = np.zeros((len(proteins), len(supported)), dtype=np.float32)
    for cp in args.checkpoint:
        meta_path = Path(str(cp) + '.json')
        cm = None
        if meta_path.exists():
            cm = json.loads(meta_path.read_text())
            checks = [('terms_sha256', sha256(args.terms_file)), ('normal_forms_sha256', sha256(args.normal_forms))]
            if args.feature_kind == 'interpros':
                checks.append(('interpros_sha256', sha256(args.interpros_file)))
            if cm.get('feature_kind') != args.feature_kind:
                raise ValueError(f'Checkpoint feature kind mismatch: {cp}')
            for key, actual in checks:
                if cm.get(key) != actual:
                    raise ValueError(f'Checkpoint sidecar {key} mismatch: {cp}')
        current = torch.load(cp, map_location='cpu', weights_only=True)
        if 'state_dict' in current:
            current = current['state_dict']
        net = cls(input_dim, len(terms), len(zeros), len(rels), args.device,
                  hidden_dim=dim, embed_dim=dim).to(args.device)
        net.load_state_dict(current, strict=True)
        # Native forward uses these indices without batch-normalizing class centers.
        net.all_gos = torch.tensor([full[g] for g in supported], dtype=torch.long, device=args.device)
        net.eval()
        with torch.no_grad():
            for start in range(0, len(proteins), args.batch_size):
                rows = np.arange(start, min(start + args.batch_size, len(proteins)))
                x = torch.from_numpy(feature_fn(rows)).to(args.device)
                pred[rows] += net(x).cpu().numpy() / len(args.checkpoint)
        checkpoint_entries.append({'path': str(Path(cp).resolve()), 'sha256': sha256(cp),
                                   'registry_hash_sidecar_verified': meta_path.exists(),
                                   'sidecar_sha256': sha256(meta_path) if cm is not None else None,
                                   'training_provenance': {k: cm.get(k) for k in ['schema', 'model', 'protocol', 'feature_kind', 'nf4_radius_policy', 'best_epoch', 'seed', 'n_train', 'n_valid', 'supervised_positive_pairs', 'holdout_exclusion']} if cm is not None else None})
    if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
        raise ValueError('Invalid predicted probabilities')
    np.save(out / 'probabilities.npy', pred)
    manifest.update(status='scored', feature_report=feature_report, checkpoints=checkpoint_entries,
        registry_binding='sidecar hashes verified when present; native checkpoints without sidecars require original training terms and normal forms in original order')
    save_manifest(out, manifest)
    print(f'Exported {pred.shape} unthresholded probabilities to {out}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-dir', required=True)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--terms-file', required=True)
    p.add_argument('--normal-forms', required=True, help='EXACT original checkpoint go.norm/go-plus.norm, with original line order')
    p.add_argument('--checkpoint', action='append')
    p.add_argument('--feature-kind', choices=['interpros', 'esm2'], default='interpros')
    p.add_argument('--feature-column', help='Defaults to feature-kind')
    p.add_argument('--interpros-file')
    p.add_argument('--esm-cache', help='Native extract_esm torch cache with proteins and data; must match sequences')
    p.add_argument('--esm-model', default='esm2_t36_3B_UR50D')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cpu')
    p.add_argument('--allow-supervised-targets', action='store_true', help='Non-zero-shot diagnostic only; excluded by default')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--output-dir', required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    args.feature_column = args.feature_column or args.feature_kind
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    run(args)
