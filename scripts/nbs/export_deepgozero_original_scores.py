#!/usr/bin/env python3
"""Export dense requested-GO scores using the ORIGINAL DeepGOZero DGELModel.

The supplied source contains nn.BatchNorm, which is not a PyTorch class.
For inference, auto mode infers BatchNorm1d versus LayerNorm from checkpoint
buffers and records the chosen compatibility correction. No ontology row is
invented: terms and normal forms must be the originals used for that checkpoint.
"""
from __future__ import annotations
import argparse
import ast
import json
import math
from pathlib import Path
import re
import numpy as np
import pandas as pd

from export_deepgo_unseen_scores import (
    ids, vocab, sha256, load_metadata, ipr_batch, ipr_values,
    input_provenance, save_manifest,
)


def source_nodes(repo, names):
    path = Path(repo) / 'deepgozero.py'
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if {node.name for node in nodes} != set(names) or len(nodes) != len(names):
        raise ValueError(f'{path}: cannot find exactly the expected original DeepGOZero definitions {names}')
    return nodes, path


def registry(repo, terms_file, normal_forms):
    terms = vocab(terms_file, 'gos')
    if any(re.fullmatch(r'GO:\d{7}', go) is None for go in terms):
        raise ValueError('Supervised terms must be canonical GO IDs')
    nodes, path = source_nodes(repo, ['load_normal_forms'])
    env = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    *nfs, rels, zeros = env['load_normal_forms'](str(normal_forms), dict(zip(terms, range(len(terms)))))
    full = {go: i for i, go in enumerate(terms)}
    if set(full) & set(zeros):
        raise ValueError('Overlapping supervised/zero registry')
    full.update(zeros)
    if sorted(full.values()) != list(range(len(full))):
        raise ValueError('Noncontiguous ontology registry')
    return terms, nfs, rels, zeros, full


class FixMLPNorm(ast.NodeTransformer):
    def __init__(self, selected):
        self.selected = selected
        self.count = 0

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'nn' and node.attr == 'BatchNorm':
            self.count += 1
            node.attr = {'batchnorm1d': 'BatchNorm1d', 'layernorm': 'LayerNorm'}[self.selected]
        return node


def original_model_class(repo, mlp_norm):
    if mlp_norm not in ('batchnorm1d', 'layernorm'):
        raise ValueError('Select an explicit valid MLP normalization for model construction')
    import torch
    nodes, path = source_nodes(repo, ['Residual', 'MLPBlock', 'DGELModel'])
    tree = ast.Module(body=nodes, type_ignores=[])
    correction = FixMLPNorm(mlp_norm)
    tree = ast.fix_missing_locations(correction.visit(tree))
    if correction.count != 1:
        raise ValueError(f'Expected one nn.BatchNorm compatibility correction, found {correction.count}; source differs from audited attachment')
    env = dict(th=torch, nn=torch.nn, math=math)
    exec(compile(tree, str(path), 'exec'), env)
    return env['DGELModel']


def checkpoint_state(torch, path):
    try:
        obj = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
        obj = obj.get('state_dict', obj.get('model_state_dict', obj))
    if not isinstance(obj, dict) or not obj:
        raise ValueError('Expected a nonempty DGELModel state_dict')
    if all(key.startswith('module.') for key in obj):
        obj = {key[7:]: val for key, val in obj.items()}
    return obj


def infer_mlp_norm(state, requested='auto'):
    prefix = 'net.0.layer_norm.'
    if prefix + 'weight' not in state or prefix + 'bias' not in state:
        raise ValueError('Checkpoint lacks expected original MLP normalization parameters')
    found = 'batchnorm1d' if prefix + 'running_mean' in state else 'layernorm'
    if requested not in ('auto', found):
        raise ValueError(f'Checkpoint normalization buffers imply {found}, but --mlp-norm={requested}')
    return found


def definition_ids(repo, path):
    if path is None:
        return None
    src = Path(repo) / 'utils.py'
    nodes = [n for n in ast.parse(src.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == 'get_goplus_defs']
    if len(nodes) != 1:
        raise ValueError('Missing native get_goplus_defs')
    env = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(src), 'exec'), env)
    return set(env['get_goplus_defs'](str(path)))


def run(args):
    provenance = input_provenance(args.input_dir)
    proteins = ids(Path(args.input_dir) / 'protein_ids.txt')
    requested = ids(Path(args.input_dir) / 'go_ids.txt')
    if any(re.fullmatch(r'GO:\d{7}', go) is None for go in requested):
        raise ValueError('Candidate GO axis must contain canonical IDs')
    terms, nfs, relations, zeros, full = registry(args.repo_dir, args.terms_file, args.normal_forms)
    defined = definition_ids(args.repo_dir, args.definitions_file)
    if args.candidate_policy == 'definitions' and defined is None:
        raise ValueError('--candidate-policy definitions requires the ORIGINAL --definitions-file')
    supported, rows = [], []
    for go in requested:
        role = 'zero_class' if go in zeros else 'supervised_class' if go in full else 'absent_from_checkpoint_registry'
        ok = role == 'zero_class' or (role == 'supervised_class' and args.allow_supervised_targets)
        if ok and args.candidate_policy == 'definitions' and go not in defined:
            ok, reason = False, 'excluded_by_native_definition_policy'
        else:
            reason = 'supported' if ok else role
        if ok:
            supported.append(go)
        rows.append(dict(go_id=go, supported=ok, reason=reason, registry_role=role,
                         ontology_row=full.get(go), native_definition_eligible=None if defined is None else go in defined))
    out = Path(args.output_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Choose a new output directory: {out}')
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / 'go_coverage.tsv', sep='\t', index=False)
    (out / 'protein_ids.txt').write_text('\n'.join(proteins) + '\n')
    (out / 'go_ids.txt').write_text(''.join(go + '\n' for go in supported))
    manifest = {**provenance, 'model': 'original_DeepGOZero_DGELModel',
        'requested_go_ids': requested, 'supported_go_count': len(supported),
        'unsupported_go_ids': [go for go in requested if go not in supported],
        'candidate_policy': args.candidate_policy, 'allow_supervised_targets': args.allow_supervised_targets,
        'normal_forms_sha256': sha256(args.normal_forms), 'terms_sha256': sha256(args.terms_file),
        'definitions_sha256': sha256(args.definitions_file) if args.definitions_file else None,
        'source_sha256': sha256(Path(args.repo_dir) / 'deepgozero.py'),
        'readout': 'original DGELModel.predict_zero; raw sigmoid(dot(MLP(x), GO+hasFunction) + abs(radius))',
        'thresholded': False, 'hierarchy_propagated': False,
        'scope_note': 'Registry policy evaluates all scoreable requested classes; native public predictor additionally restricts classes to definitions_go.txt. Neither policy adds new untrained rows.',
        'strict_zero_shot_claimed': False}
    if args.audit_only or not supported:
        if supported:
            (out / 'registry_audit.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(f'Original DeepGOZero registry audit: {len(supported)}/{len(requested)} supported; inference not run')
        else:
            np.save(out / 'probabilities.npy', np.empty((len(proteins), 0), dtype=np.float32))
            manifest['status'] = 'no_supported_go'
            save_manifest(out, manifest)
            print('No trained zero-class rows support requested GO; coverage-only export, no numeric performance claim')
        return
    if not args.checkpoint or not args.interpros_file:
        raise ValueError('Scoring requires --checkpoint and ORIGINAL --interpros-file')
    import torch
    df = load_metadata(args.input_dir, proteins)
    if args.interpro_field not in df:
        raise ValueError('Prepared metadata lacks the InterPro feature column')
    iprs = {g: i for i, g in enumerate(vocab(args.interpros_file, 'interpros'))}
    features = [ipr_values(v) for v in df[args.interpro_field]]
    if any(re.fullmatch(r'IPR\d{6}', v) is None for row in features for v in row):
        raise ValueError('Expected raw InterPro accession IDs')
    pred = np.zeros((len(proteins), len(supported)), dtype=np.float32)
    cp_records = []
    for cp in args.checkpoint:
        state = checkpoint_state(torch, cp)
        normalization = infer_mlp_norm(state, args.mlp_norm)
        dim = int(state['go_embed.weight'].shape[1])
        hidden_dim, input_dim = map(int, state['net.0.linear.weight'].shape)
        if input_dim != len(iprs) or hidden_dim != dim:
            raise ValueError('Original checkpoint input/hidden/GO dimensions do not match the supplied vocabulary')
        meta_path = Path(str(cp) + '.json')
        cm = None
        if meta_path.is_file():
            cm = json.loads(meta_path.read_text())
            for key, expected in [('terms_sha256', sha256(args.terms_file)),
                                  ('normal_forms_sha256', sha256(args.normal_forms)),
                                  ('interpros_sha256', sha256(args.interpros_file)),
                                  ('mlp_norm', normalization)]:
                if cm.get(key) != expected:
                    raise ValueError(f'Checkpoint sidecar mismatch: {key}')
        cls = original_model_class(args.repo_dir, normalization)
        net = cls(len(iprs), len(terms), len(zeros), len(relations), args.device,
                  hidden_dim=hidden_dim, embed_dim=dim).to(args.device)
        net.load_state_dict(state, strict=True)
        net.eval()
        go_rows = torch.tensor([full[g] for g in supported], dtype=torch.long, device=args.device)
        with torch.inference_mode():
            for start in range(0, len(proteins), args.batch_size):
                indexes = np.arange(start, min(start + args.batch_size, len(proteins)))
                x = torch.from_numpy(ipr_batch(df, indexes, iprs, args.interpro_field)).to(args.device)
                pred[indexes] += net.predict_zero(x, go_rows).cpu().numpy() / len(args.checkpoint)
        cp_records.append({'path': str(Path(cp).resolve()), 'sha256': sha256(cp),
            'mlp_norm': normalization, 'nn_BatchNorm_compatibility_correction': normalization,
            'registry_sidecar_verified': cm is not None,
            'training_protocol': None if cm is None else {k: cm.get(k) for k in
                ('protocol', 'nf4_radius_policy', 'mlp_norm', 'seed', 'best_epoch', 'holdout_exclusion', 'normalization_source')},
            'sidecar_sha256': sha256(meta_path) if cm is not None else None})
    if not np.isfinite(pred).all() or np.any(pred < 0) or np.any(pred > 1):
        raise ValueError('Invalid predicted probabilities')
    np.save(out / 'probabilities.npy', pred)
    manifest.update(status='scored', checkpoints=cp_records,
        interpros_sha256=sha256(args.interpros_file), shape=list(pred.shape),
        zero_covered_interpro_proteins=sum(not any(v in iprs for v in row) for row in features),
        registry_binding='Original terms/norm/IPR order mandatory; sidecar checks enforced for newly trained checkpoints',
        software={'torch': torch.__version__, 'numpy': np.__version__})
    save_manifest(out, manifest)
    print(f'Original DeepGOZero: exported {pred.shape} dense scores to {out}')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-dir', required=True)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--terms-file', required=True)
    p.add_argument('--normal-forms', required=True, help='EXACT original checkpoint normal-forms file and line order')
    p.add_argument('--interpros-file')
    p.add_argument('--checkpoint', action='append', help='Repeat for same-registry equal-weight ensemble')
    p.add_argument('--interpro-field', default='interpros')
    p.add_argument('--mlp-norm', choices=['auto', 'batchnorm1d', 'layernorm'], default='auto')
    p.add_argument('--candidate-policy', choices=['registry', 'definitions'], default='registry')
    p.add_argument('--definitions-file', help='Native definitions_go.txt, required only for definitions policy')
    p.add_argument('--allow-supervised-targets', action='store_true', help='Marked non-zero-shot diagnostic only')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.batch_size < 1:
        raise ValueError('Positive batch size required')
    run(args)
