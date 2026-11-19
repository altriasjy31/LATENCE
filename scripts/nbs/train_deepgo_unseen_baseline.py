#!/usr/bin/env python3
"""Train the supplied deepgo2 DeepGOZero variant with label-zero-shot targets.

This trains ONLY an external baseline. New GO classes occur only in supplied
normalized ontology axioms, never in supervised output columns. It is a
controlled adaptation, not reproduction of original paper datasets or weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import types
import shutil
import numpy as np
import pandas as pd
from export_deepgo_unseen_scores import ids, vocab, registry, native_model_class, sha256, ipr_values, ipr_batch


def read_frame(path, id_field, feature_field, label_field):
    df = pd.read_pickle(path).copy()
    for key in [id_field, feature_field, label_field]:
        if key not in df:
            raise ValueError(f'{path}: missing column {key}')
    df['proteins'] = df[id_field].astype(str)
    if df.proteins.duplicated().any():
        raise ValueError(f'{path}: duplicate proteins')
    for protein, values in zip(df.proteins, df[label_field]):
        if not isinstance(values, (list, set, tuple, np.ndarray)):
            raise ValueError(f'{path}: {protein}/{label_field} must be a collection of canonical GO IDs; use the training preparation program for indices or coded annotations')
        if isinstance(values, np.ndarray) and values.ndim != 1:
            raise ValueError(f'{path}: {protein}/{label_field} must be a one-dimensional GO collection')
        invalid = [v for v in values if not isinstance(v, str) or re.fullmatch(r'GO:\d{7}', v) is None]
        if invalid:
            raise ValueError(f'{path}: {protein}/{label_field} has noncanonical labels {invalid[:3]}; expected GO: followed by seven digits')
    return df


def clean_labels(df, candidates, label_field):
    df = df.copy()
    counts = {}
    for key in df.columns:
        if key != label_field and 'annot' not in key.lower() and 'label' not in key.lower():
            continue
        removed = 0
        values = []
        for v in df[key]:
            if isinstance(v, (list, set, tuple, np.ndarray)):
                removed += sum(str(x) in candidates for x in v)
                values.append([x for x in v if str(x) not in candidates])
            else:
                values.append(v)
        df[key] = values
        if removed:
            counts[key] = removed
    return df, counts


def seqhash(sequence):
    if not isinstance(sequence, str):
        raise ValueError('Missing sequence for holdout check')
    normalized = ''.join(sequence.split()).upper().rstrip('*')
    if not re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWYBXZJUO]+', normalized):
        raise ValueError('Invalid sequence for holdout check')
    return hashlib.sha256(normalized.encode()).hexdigest()


def exclude_holdouts(df, directories, sequence_field):
    proteins, hashes = set(), set()
    for d in directories:
        proteins.update(ids(Path(d) / 'protein_ids.txt'))
        hdf = pd.read_pickle(Path(d) / 'metadata.pkl')
        if 'sequences' not in hdf:
            raise ValueError('Holdout metadata lacks sequences; cannot check sequence leakage')
        hashes.update(seqhash(v) for v in hdf.sequences)
    if not directories:
        return df, {'excluded_by_id': 0, 'excluded_by_exact_sequence': 0}
    if sequence_field not in df:
        raise ValueError('Training data lacks sequence field required for holdout exclusion')
    by_id = df.proteins.isin(proteins)
    by_seq = df[sequence_field].map(seqhash).isin(hashes)
    return df.loc[~(by_id | by_seq)].reset_index(drop=True), {
        'excluded_by_id': int(by_id.sum()), 'excluded_by_exact_sequence': int(by_seq.sum()),
        'excluded_union': int((by_id | by_seq).sum())}


def run(args):
    out = Path(args.output_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'Choose a new training/audit output directory: {out}')
    out.mkdir(parents=True)
    candidates = set(ids(args.candidate_go_ids))
    terms = vocab(args.terms_file, 'gos')
    overlap = candidates.intersection(terms)
    if overlap:
        raise ValueError(f'{len(overlap)} candidate GO in supervised terms. Supply old terms excluding ALL candidate GO; do not train with zero-valued candidate columns.')
    terms_file = out / 'terms.pkl'
    norm_file = out / 'go.norm'
    pd.DataFrame({'gos': terms}).to_pickle(terms_file)
    shutil.copyfile(args.normal_forms, norm_file)
    terms, nfs, relations, zeros, full = registry(args.repo_dir, terms_file, norm_file)
    missing_go = sorted(candidates - set(zeros))
    pd.DataFrame([{'go_id': g, 'in_zero_registry': g in zeros} for g in sorted(candidates)]).to_csv(out / 'candidate_coverage.tsv', sep='\t', index=False)
    if missing_go and not args.allow_partial_go:
        raise ValueError(f'{len(missing_go)} candidates absent from normalized axioms. See candidate_coverage.tsv; use matching new GO OWL axioms, or explicitly --allow-partial-go.')
    train = read_frame(args.train_pickle, args.id_field, args.feature_column, args.label_field)
    valid = read_frame(args.valid_pickle, args.id_field, args.feature_column, args.label_field)
    train, tr_excl = exclude_holdouts(train, args.exclude_input_dir or [], args.sequence_field)
    valid, va_excl = exclude_holdouts(valid, args.exclude_input_dir or [], args.sequence_field)
    if not len(train) or not len(valid):
        raise ValueError('Empty train/validation split after holdout exclusion')
    if set(train.proteins).intersection(valid.proteins):
        raise ValueError('Training and validation protein IDs overlap')
    if args.sequence_field in train and args.sequence_field in valid:
        if set(train[args.sequence_field].map(seqhash)).intersection(valid[args.sequence_field].map(seqhash)):
            raise ValueError('Exact sequences overlap training and validation splits')
    train, tr_removed = clean_labels(train, candidates, args.label_field)
    valid, va_removed = clean_labels(valid, candidates, args.label_field)
    if args.feature_kind == 'interpros':
        iprs = sorted({str(g) for values in train[args.feature_column] for g in ipr_values(values)})
        if not iprs:
            raise ValueError('Empty training InterPro vocabulary')
        ipr_map = {v: i for i, v in enumerate(iprs)}
        pd.DataFrame({'interpros': iprs}).to_pickle(out / 'interpros.pkl')
        feature_length = len(iprs)
        def feature_batch(df, rows):
            return ipr_batch(df, rows, ipr_map, args.feature_column)
    else:
        feature_length = 2560
        def feature_batch(df, rows):
            arr = np.asarray(df.iloc[rows][args.feature_column].tolist(), dtype=np.float32)
            if arr.shape != (len(rows), 2560) or not np.isfinite(arr).all():
                raise ValueError('Expected finite native ESM2-3B 2560-dimensional features')
            return arr
    term_map = {v: i for i, v in enumerate(terms)}
    supervised_positive_pairs = {}
    for split, df in [('train', train), ('validation', valid)]:
        count = sum(len(set(values).intersection(term_map)) for values in df[args.label_field])
        if count == 0:
            raise ValueError(f'{split}: no old-task positive labels remain after filtering. Check label encoding, vocabulary and split selection; empty-label individual proteins are allowed.')
        supervised_positive_pairs[split] = count
    def labels_batch(df, rows):
        labels = np.zeros((len(rows), len(terms)), dtype=np.float32)
        for k, row in enumerate(rows):
            for g in ipr_values(df.iloc[int(row)][args.label_field]):
                if g in term_map:
                    labels[k, term_map[g]] = 1.0
        return labels
    manifest = {'schema': 'deepgo2-label-zero-shot-training-v1',
        'model': 'DeepGOZero variant in supplied DeepGO-SE/deepgo2 repository',
        'protocol': 'new ontology classes available to EL loss; no candidate class in supervised label loss',
        'feature_kind': args.feature_kind, 'feature_column': args.feature_column,
        'input_dim': feature_length, 'embed_dim': args.embed_dim, 'seed': args.seed,
        'interpros_sha256': sha256(out / 'interpros.pkl') if args.feature_kind == 'interpros' else None,
        'nf4_radius_policy': args.nf4_radius_policy,
        'source_note': 'Uploaded source nf4_loss indexes class radius with relation index; native policy preserves this behavior, class-radius policy corrects it explicitly.',
        'terms_sha256': sha256(terms_file), 'normal_forms_sha256': sha256(norm_file),
        'n_train': len(train), 'n_valid': len(valid), 'n_supervised_terms': len(terms),
        'n_candidate_go': len(candidates), 'n_supported_candidate_go': len(candidates)-len(missing_go),
        'candidate_ids_sha256': sha256(args.candidate_go_ids),
        'removed_candidate_annotations': {'train': tr_removed, 'validation': va_removed},
        'supervised_positive_pairs': supervised_positive_pairs,
        'holdout_exclusion': {'input_dirs': args.exclude_input_dir or [], 'train': tr_excl, 'validation': va_excl},
        'training_inputs': {str(Path(p).resolve()): sha256(p) for p in [args.train_pickle, args.valid_pickle]},
        'source_sha256': {n: sha256(Path(args.repo_dir) / n) for n in ['deepgo/base.py', 'deepgo/models.py', 'deepgo/data.py']},
        'args': vars(args)}
    (out / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.audit_only:
        print(json.dumps({k: manifest[k] for k in ['n_train', 'n_valid', 'n_supervised_terms', 'n_supported_candidate_go', 'removed_candidate_annotations', 'holdout_exclusion']}, indent=2))
        return
    import torch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    for i, nf in enumerate(nfs):
        if (i == 0 and len(nf) < 2) or len(nf) == 1:
            raise ValueError(f'NF{i+1} needs at least two rows when nonempty for native BatchNorm training')
    forms = tuple(torch.tensor(nf, dtype=torch.long, device=args.device).reshape(-1, width)
                  for nf, width in zip(nfs, [2, 3, 3, 3]))
    cls = native_model_class(args.repo_dir)
    model = cls(feature_length, len(terms), len(zeros), len(relations), args.device,
                hidden_dim=args.embed_dim, embed_dim=args.embed_dim).to(args.device)
    if args.nf4_radius_policy == 'class-radius':
        def corrected_nf4(self, data):
            c = self.go_norm(self.go_embed(data[:, 0]))
            d = self.go_norm(self.go_embed(data[:, 2]))
            r = self.rel_embed(data[:, 1])
            rc = self.go_rad(data[:, 0]).abs()
            rd = self.go_rad(data[:, 2]).abs()
            return torch.relu(torch.linalg.norm(c - (d + r), dim=1, keepdim=True) - rc - rd - self.margin).mean()
        model.nf4_loss = types.MethodType(corrected_nf4, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)
    best = float('inf'); history = []
    cp = out / 'deepgozero.th'
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(train))
        loss_sum, count = 0., 0
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start+args.batch_size]
            x = torch.from_numpy(feature_batch(train, rows)).to(args.device)
            y = torch.from_numpy(labels_batch(train, rows)).to(args.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.nn.functional.binary_cross_entropy(pred, y)
            el = model.el_loss(forms)
            total = loss + el
            if not torch.isfinite(total):
                raise ValueError(f'Non-finite training loss epoch {epoch}')
            total.backward(); optimizer.step()
            loss_sum += float(total.detach()) * len(rows); count += len(rows)
        model.eval(); valid_sum = 0.
        with torch.no_grad():
            for start in range(0, len(valid), args.batch_size):
                rows = np.arange(start, min(start+args.batch_size, len(valid)))
                x = torch.from_numpy(feature_batch(valid, rows)).to(args.device)
                y = torch.from_numpy(labels_batch(valid, rows)).to(args.device)
                valid_sum += float(torch.nn.functional.binary_cross_entropy(model(x), y)) * len(rows)
        valid_loss = valid_sum / len(valid)
        if not np.isfinite(valid_loss):
            raise ValueError('Non-finite validation loss')
        if valid_loss < best:
            best = valid_loss
            torch.save(model.state_dict(), cp)
            meta = dict(manifest, best_epoch=epoch, best_valid_bce=best)
            Path(str(cp)+'.json').write_text(json.dumps(meta, indent=2) + '\n')
        history.append({'epoch': epoch, 'train_total_loss': loss_sum/count, 'valid_seen_bce': valid_loss, 'best_valid_seen_bce': best})
        pd.DataFrame(history).to_csv(out/'training_history.tsv', sep='\t', index=False)
        print(history[-1], flush=True)
        scheduler.step()
    print(f'Saved {cp}. Score candidates with export_deepgo_unseen_scores.py and this output terms.pkl/go.norm/interpros.pkl.')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-dir', required=True)
    p.add_argument('--train-pickle', required=True)
    p.add_argument('--valid-pickle', required=True)
    p.add_argument('--terms-file', required=True, help='Fixed old supervised GO vocabulary, no candidate IDs')
    p.add_argument('--candidate-go-ids', required=True)
    p.add_argument('--normal-forms', required=True, help='NEW ontology normalized OWL axioms, preserving logical definitions')
    p.add_argument('--feature-kind', choices=['interpros', 'esm2'], default='interpros')
    p.add_argument('--feature-column')
    p.add_argument('--id-field', default='proteins')
    p.add_argument('--sequence-field', default='sequences')
    p.add_argument('--label-field', default='prop_annotations')
    p.add_argument('--exclude-input-dir', action='append', help='Prepared independent test input; exclude matching IDs AND exact sequences from train/valid')
    p.add_argument('--allow-partial-go', action='store_true')
    p.add_argument('--nf4-radius-policy', choices=['native', 'class-radius'], default='native',
                   help='native preserves uploaded NF4 relation-index radius; class-radius is an explicitly corrected variant')
    p.add_argument('--embed-dim', type=int, default=2560)
    p.add_argument('--batch-size', type=int, default=37)
    p.add_argument('--epochs', type=int, default=128)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--output-dir', required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    args.feature_column = args.feature_column or args.feature_kind
    if min(args.embed_dim, args.batch_size, args.epochs) < 1:
        raise ValueError('Dimensions/batch size/epochs must be positive')
    run(args)
