#!/usr/bin/env python3
"""Adapt v0.2.1 unified dark-weak GO matrices to LATENCE's nested metadata pickle.

Existing metadata split keys remain unchanged. Added splits have exactly:
    metadata[split][ontology_name] = {
       'proteins': [2022 entry names],
       'prop_annotations': [[zero-based model GO column indices], ...],
    }

Every split stores DIRECT positive model GO indices in 'annotations' and
those labels' full ancestor-closure in 'prop_annotations' (including self).
The *_direct splits select proteins with a direct positive in that namespace,
while unsuffixed splits select proteins with propagated positives. Field
semantics NEVER change based on split names. The historical GO vocabulary
restricts representable ancestor indices to the trained model's GO columns.
The positive-only ranking evaluator uses 'annotations' for *_direct views.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import pickle
import re
import sys
import numpy as np
import pandas as pd
from scipy import sparse

NS = {'bp':'biological_process', 'mf':'molecular_function', 'cc':'cellular_component'}
GO_RE = re.compile(r'GO:\d{7}')

# Each view has two row-selection panels, but the annotations/prop_annotations
# FIELD CONTRACT never changes. '_direct' affects row selection/evaluation only.
VIEW_TO_SOURCE = {
    'dark_weak_test': 'future_union_novel_direct',
    'dark_weak_test_exp': 'future_experimental_crosssource_novel_direct',
    'dark_weak_test_ipr2go': 'future_ipr2go_novel_direct',
    'dark_weak_test_2022_ipr2go': 'historical_ipr2go_direct',
    'dark_weak_test_2022_ipr2go_matched': 'historical_ipr2go_matched_direct',
    'dark_weak_test_2025_ipr2go_all': 'future_ipr2go_all_direct',
    'dark_weak_test_2025_ipr2go_added': 'future_ipr2go_delta_direct',
    'dark_weak_test_2025_ipr2go_retained': 'future_ipr2go_retained_direct',
    'dark_weak_test_2025_ipr2go_removed': 'future_ipr2go_removed_direct',
}
PROPAGATED_SOURCES = {
    'dark_weak_test_ipr2go': 'future_ipr2go_novel_propagated',
    'dark_weak_test_2022_ipr2go': 'historical_ipr2go_propagated',
    'dark_weak_test_2022_ipr2go_matched': 'historical_ipr2go_matched_propagated',
    'dark_weak_test_2025_ipr2go_all': 'future_ipr2go_all_propagated',
    'dark_weak_test_2025_ipr2go_added': 'future_ipr2go_delta_propagated',
    'dark_weak_test_2025_ipr2go_retained': 'future_ipr2go_retained_propagated',
    'dark_weak_test_2025_ipr2go_removed': 'future_ipr2go_removed_propagated',
}
GROUPS = {
    'future': ['dark_weak_test','dark_weak_test_exp','dark_weak_test_ipr2go'],
    'history': ['dark_weak_test_2022_ipr2go','dark_weak_test_2022_ipr2go_matched'],
    'diagnostic': ['dark_weak_test_2025_ipr2go_all','dark_weak_test_2025_ipr2go_added',
                   'dark_weak_test_2025_ipr2go_retained','dark_weak_test_2025_ipr2go_removed'],
}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(4*1024*1024), b''):
            h.update(buf)
    return h.hexdigest()


def read_go_list(path: Path):
    values = GO_RE.findall(path.read_text(encoding='utf-8'))
    if not values or len(values) != len(set(values)):
        raise ValueError(f'GO vocabulary malformed: {path}')
    return values


def load_matrices(root: Path, sources: set[str], expected_go: dict[str,Path]):
    matrices = {}
    goids = {}
    canonids = {}
    nrows = None
    for ns in NS:
        folder = root / 'labels' / ns
        gofile = folder / 'go_ids.txt'
        gos = read_go_list(gofile)
        if ns in expected_go and expected_go[ns]:
            trusted = read_go_list(expected_go[ns])
            if gos != trusted:
                raise ValueError(f'{ns} GO order differs from actual LATENCE training vocabulary: {expected_go[ns]}')
        goids[ns] = gos
        frame = pd.read_csv(folder/'vocabulary_mapping.tsv', sep='\t')
        if not {'model_column','model_go_id','canonical_go_id_2022'}.issubset(frame.columns):
            raise ValueError(f'Incomplete vocabulary_mapping.tsv for {ns}')
        frame = frame.sort_values('model_column')
        if frame.model_column.tolist() != list(range(len(gos))) or frame.model_go_id.tolist() != gos:
            raise ValueError(f'{ns} GO vocabulary mapping and columns are misaligned')
        canonids[ns] = frame.canonical_go_id_2022.astype(str).tolist()
        for key in sources:
            file = folder / f'{key}.npz'
            if not file.is_file():
                raise FileNotFoundError(f'Required matrix missing: {file}')
            mat = sparse.load_npz(file).tocsr()
            if mat.shape[1] != len(gos):
                raise ValueError(f'Incorrect number of GO columns for {ns}/{key}')
            if nrows is None:
                nrows = mat.shape[0]
            if mat.shape[0] != nrows:
                raise ValueError(f'Inconsistent protein row count in {file}')
            mat.sum_duplicates()
            if mat.nnz and (mat.data <= 0).any():
                raise ValueError(f'Nonpositive sparse label entries in {file}')
            mat.data[:] = 1
            matrices[ns,key] = mat.astype(np.uint8)
    return matrices, goids, canonids, nrows


def load_ontology(obo: Path, expected_mode: str):
    # Reuse precisely the ontology logic of the historical builder, not a new one.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_dark_weak_test import Ontology
    return Ontology(obo, mode=expected_mode)


def load_raw_future_direct(root: Path, view: str, nrows: int, ont):
    """Read all direct GO IDs, including terms outside the checkpoint vocabulary.

    The unified v0.2.1 provenance tables retain the canonical direct terms
    before conversion to sparse model-column arrays. This is essential:
    an off-model GO can still have ancestors *inside* the model vocabulary.
    """
    future_flags={
        'dark_weak_test':'evaluation_union_label',
        'dark_weak_test_exp':'experimental_crosssource_novel',
        'dark_weak_test_ipr2go':'interpro2go_crosssource_novel',
    }
    temporal_status={
        'dark_weak_test_2025_ipr2go_all':{'retained','added'},
        'dark_weak_test_2025_ipr2go_added':{'added'},
        'dark_weak_test_2025_ipr2go_retained':{'retained'},
        'dark_weak_test_2025_ipr2go_removed':{'removed'},
    }
    if view in future_flags:
        flag=future_flags[view]
        file=root/'tables'/'unified_direct_positive_pairs.tsv'
        frame=pd.read_csv(file,sep='\t')
        required={'rid22','canonical_go_2022',flag}
        if not required.issubset(frame):
            raise ValueError(f'Missing {required-set(frame.columns)} in {file}')
        yes=frame[flag].astype(str).str.lower().isin(('true','1'))
        frame=frame.loc[yes,['rid22','canonical_go_2022']]
        idcol='canonical_go_2022'
    elif view in temporal_status:
        file=root/'tables'/'ipr2go_temporal_status_pairs.tsv'
        frame=pd.read_csv(file,sep='\t')
        required={'rid22','go_id_2022','temporal_status'}
        if not required.issubset(frame):
            raise ValueError(f'Missing {required-set(frame.columns)} in {file}')
        frame=frame.loc[frame['temporal_status'].isin(temporal_status[view]),
                        ['rid22','go_id_2022']]
        idcol='go_id_2022'
    elif view in {'dark_weak_test_2022_ipr2go',
                  'dark_weak_test_2022_ipr2go_matched'}:
        file=root/'tables'/'ipr2go_2022_baseline_pairs.tsv'
        frame=pd.read_csv(file,sep='\t')
        required={'rid22','go_id_2022'}
        if not required.issubset(frame):
            raise ValueError(f'Missing {required-set(frame.columns)} in {file}')
        frame=frame[['rid22','go_id_2022']]
        if view.endswith('_matched'):
            matched=pd.read_csv(root/'tables'/'full_dark_weak_cohort.tsv',
                                sep='\t',usecols=['valid_future'])
            if len(matched)!=nrows:
                raise ValueError('Historical matched flags misaligned with cohort')
            valid=matched.valid_future.astype(str).str.lower().isin(('true','1')).to_numpy()
            frame=frame.loc[frame['rid22'].astype(int).map(lambda i: 0<=i<nrows and valid[i])]
        idcol='go_id_2022'
    else:
        return None
    direct_by_row=defaultdict(set)
    for row,go in frame[['rid22',idcol]].itertuples(index=False,name=None):
        if pd.isna(go):
            continue
        row=int(row)
        if row<0 or row>=nrows:
            raise IndexError(f'GO evidence rid22={row} outside historical cohort')
        canon=ont.canonical(str(go))
        if canon is None:
            raise ValueError(f'GO in evidence table not in historical ontology: {go} ({file})')
        direct_by_row[row].add(canon)
    return dict(direct_by_row)


def propagate_all_namespaces(direct: dict[str,sparse.csr_matrix], canonids, ont,
                             raw_all_go:dict[int,set[str]]|None=None):
    """Cross-namespace propagation of direct GO using historical ontology."""
    nrows = next(iter(direct.values())).shape[0]
    colmaps = {}
    for ns in NS:
        colmap = defaultdict(list)
        for idx, c in enumerate(canonids[ns]):
            colmap[c].append(idx)
        colmaps[ns] = colmap
    by_ns = {ns:([],[]) for ns in NS}
    for r in range(nrows):
        direct_set = set()
        for ns in NS:
            x = direct[ns]
            direct_set.update(canonids[ns][int(c)] for c in x.indices[x.indptr[r]:x.indptr[r+1]])
        if raw_all_go is not None:
            direct_set.update(raw_all_go.get(r,set()))
        if not direct_set:
            continue
        closure = ont.propagate(direct_set)
        for ns in NS:
            rlist, clist = by_ns[ns]
            for c in closure:
                for j in colmaps[ns].get(c, ()):
                    rlist.append(r)
                    clist.append(j)
    result = {}
    for ns in NS:
        rr,cc = by_ns[ns]
        shape = direct[ns].shape
        y = sparse.csr_matrix((np.ones(len(rr),dtype=np.uint8),(rr,cc)),shape=shape)
        y.sum_duplicates()
        if y.nnz:
            y.data[:]=1
        result[ns] = y
    return result


def dict_split(prop_mats:dict[str,sparse.csr_matrix],
               direct_mats:dict[str,sparse.csr_matrix], ids:list[str],
               *, select:str='prop'):
    """Export GO labels with invariant field semantics.

    annotations: original/direct model-column GO IDs (integer indices).
    prop_annotations: self + ALL ontology ancestors representable in the
       immutable model GO vocabulary, irrespective of 'select' or split name.

    select='direct' chooses rows having a direct positive in this namespace;
    select='prop' chooses rows having an ancestor-closed positive. Both keep
    the corresponding direct and propagated GO labels aligned row-for-row.
    """
    if select not in ('direct','prop'):
        raise ValueError(f'Invalid row selection {select}')
    out={}
    for ns, namespace in NS.items():
        p=prop_mats[ns].tocsr()
        d=direct_mats[ns].tocsr()
        if p.shape != d.shape or p.shape[0] != len(ids):
            raise ValueError(f'{ns}: inconsistent direct/propagated matrix dimensions')
        # All direct GO terms must appear in the complete propagated closure.
        missing=d-d.multiply(p)
        missing.eliminate_zeros()
        if missing.nnz:
            raise AssertionError(f'{ns}: {missing.nnz} direct labels absent from prop_annotations')
        select_mat = d if select=='direct' else p
        rows=np.flatnonzero(np.diff(select_mat.indptr)>0)
        out[namespace] = {
            'proteins':[ids[i] for i in rows],
            'annotations':[d.indices[d.indptr[i]:d.indptr[i+1]].astype(int).tolist()
                           for i in rows],
            'prop_annotations':[p.indices[p.indptr[i]:p.indptr[i+1]].astype(int).tolist()
                                for i in rows],
        }
    return out


def build(args):
    root = args.unified_root
    manifest_file = root/'unified_manifest.json'
    if not manifest_file.exists():
        raise FileNotFoundError(f'Missing unified dataset manifest: {manifest_file}')
    source_manifest = json.loads(manifest_file.read_text())
    mode = source_manifest.get('propagation')
    if mode not in ('deepgozero','is_a_part_of','is_a'):
        raise ValueError(f'Missing/unsupported GO propagation mode in unified manifest: {mode}')
    if Path(source_manifest.get('ontology_2022','')).name != args.go_obo_2022.name:
        raise ValueError('GO ontology filename differs from unified build; check ontology version')
    ontology_hashes=[v for k,v in source_manifest.get('file_hashes',{}).items()
                     if Path(k).name==args.go_obo_2022.name]
    if ontology_hashes and sha256(args.go_obo_2022) not in ontology_hashes:
        raise ValueError('2022 GO ontology SHA256 differs from unified benchmark manifest')
    if Path(args.metadata).resolve() == Path(args.output).resolve():
        raise ValueError('Refusing to overwrite original LATENCE metadata pickle in place')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'Output already exists, provide --overwrite: {args.output}')
    if args.metadata.exists():
        with args.metadata.open('rb') as f:
            metadata = pickle.load(f)  # Trusted local project pickle ONLY
        if not isinstance(metadata, dict):
            raise TypeError('Original LATENCE metadata must be a dictionary')
    elif args.create_empty:
        metadata={}
    else:
        raise FileNotFoundError(f'Missing LATENCE original metadata pickle: {args.metadata}')
    selected = list(dict.fromkeys(v for g in args.groups for v in GROUPS[g]))
    required = {VIEW_TO_SOURCE[v] for v in selected}
    required.update(PROPAGATED_SOURCES[v] for v in selected if v in PROPAGATED_SOURCES)
    checkpoint_vocab = {'bp':args.bp_terms,'mf':args.mf_terms,'cc':args.cc_terms}
    mats,goids,canonids,nrows = load_matrices(root,required,checkpoint_vocab)
    cohortfile = root/'protein_ids.txt'
    ids = [x.strip() for x in cohortfile.read_text().splitlines() if x.strip()]
    if len(ids)!=nrows or len(ids)!=len(set(ids)):
        raise ValueError('Missing/duplicate proteins or protein count mismatch in unified protein_ids.txt')
    if args.groups and any(g in ('future','diagnostic') for g in args.groups):
        if not (root/'tables'/'full_dark_weak_cohort.tsv').exists():
            raise FileNotFoundError('Missing full_dark_weak_cohort.tsv to audit unchanged sequences')
        frame = pd.read_csv(root/'tables'/'full_dark_weak_cohort.tsv',sep='\t')
        if 'valid_future' not in frame.columns or len(frame)!=nrows:
            raise ValueError('Invalid future identity audit table')
        valid = frame.valid_future.astype(str).str.lower().eq('true').to_numpy()
        for source in required:
            if source.startswith('future_'):
                for ns in NS:
                    rows = np.flatnonzero(np.diff(mats[ns,source].indptr)>0)
                    if len(rows) and not valid[rows].all():
                        raise ValueError(f'{source}/{ns} has positives on non-identical 2022/2025 protein')
    ont = load_ontology(args.go_obo_2022,mode)
    changes = {}
    views = {}
    for view in selected:
        key = VIEW_TO_SOURCE[view]
        direct = {ns:mats[ns,key] for ns in NS}
        # The historical-v0.2.1 matrices named *_propagated can store
        # ontology-level *differences*, not the full closure of their direct
        # labels. NEVER use them as a label source for prop_annotations.
        # Instead, read the complete per-view direct GO IDs from provenance
        # tables (including off-vocabulary direct terms), compute true ontology
        # closure, and only then project into immutable model GO columns.
        raw_direct=load_raw_future_direct(root,view,nrows,ont)
        if raw_direct is None:
            raise ValueError(f'No raw direct GO source defined for {view}')
        prop=propagate_all_namespaces(direct,canonids,ont,raw_direct)
        for ns in NS:
            diff=direct[ns]-direct[ns].multiply(prop[ns])
            diff.eliminate_zeros()
            if diff.nnz:
                raise AssertionError(f'{view}/{ns}: full ancestor closure omits direct GO terms')

        for suffix, select in (('', 'prop'),('_direct','direct')):
            name=view+suffix
            if name in metadata and not args.replace_existing_splits:
                raise KeyError(f'Original metadata already has split {name}; refuse replacement')
            data=dict_split(prop,direct,ids,select=select)
            views[name]=data
            counts={ns:{'proteins':len(data[namespace]['proteins']),
                        'direct_model_columns':sum(len(x) for x in data[namespace]['annotations']),
                        'propagated_model_columns':sum(len(x) for x in data[namespace]['prop_annotations']),
                        'positive_model_columns':sum(len(x) for x in data[namespace]['prop_annotations'])}
                    for ns,namespace in NS.items()}
            changes[name] = {
                'source_matrix':key,
                'propagation':'full ancestor closure of raw direct GO labels, then projected to model GO vocabulary',
                'row_selection':select,
                'label_semantics':{'annotations':'direct_model_go_indices',
                                   'prop_annotations':'self_and_all_model_representable_ancestors'},
                'future_label_time_held_out':view in ('dark_weak_test','dark_weak_test_exp','dark_weak_test_ipr2go'),
                'protein_level_inductive':'not_assessed',
                'negative_labels_valid':False,
                'counts':counts,
            }
    metadata.update(views)
    # Per-namespace exact row lists for the model-side inference/export adapter.
    idroot=args.manifest.parent/'eval_split_ids'
    for view,data in views.items():
        directory=idroot/view
        directory.mkdir(parents=True,exist_ok=True)
        for ns,namespace in NS.items():
            (directory/f'{ns}_protein_ids.txt').write_text(
                ''.join(str(p)+'\n' for p in data[namespace]['proteins']),encoding='utf-8')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as f:
        pickle.dump(metadata,f,protocol=pickle.HIGHEST_PROTOCOL)
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    record={
        'version':'0.3.1','original_metadata':str(args.metadata),
        'original_metadata_sha256':sha256(args.metadata) if args.metadata.exists() else None,
        'output_metadata':str(args.output),'output_sha256':sha256(args.output),
        'unified_manifest_sha256':sha256(manifest_file),
        '2022_ontology':str(args.go_obo_2022),'2022_ontology_sha256':sha256(args.go_obo_2022),
        'namespace_names':NS,'protein_ids_file':str(cohortfile),'protein_id_source':'2022 UniProt entry name',
        'split_id_lists':str(idroot),
        'historical_ontology_mode':mode,'all_cohort_rows':nrows,
        'split_name_contract':'all splits store direct in annotations, complete ancestor closure in prop_annotations; *_direct selects directly-positive rows',
        'propagated_novelty_warning':'Ancestor closure may include GO already known in 2022; use annotations for historical-novel direct evaluation. Raw future GO outside model vocab is included where provenance tables provide it.',
        'aliases':'Alias-equivalent GO model columns remain present, in checkpoint order.',
        'notes':'Removed GO tags are descriptive and must not be converted to true negative examples.',
        'splits':changes,
    }
    args.manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'new_splits':{n:{ns:v['counts'][ns]['proteins'] for ns in NS}
                                            for n,v in changes.items()},
                      'output':str(args.output)},ensure_ascii=False,indent=2))
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,default=Path('data/unidata_with_exp_train_pseudo.pkl'))
    parser.add_argument('--create-empty',action='store_true', help='Tests only; no original train/test/ind_test saved')
    parser.add_argument('--unified-root',type=Path,default=Path('data/dark_weak_test/unified'))
    parser.add_argument('--go-obo-2022',type=Path,default=Path('data/go.obo'))
    parser.add_argument('--bp-terms',type=Path,default=Path('data/go_terms_bp.txt'))
    parser.add_argument('--mf-terms',type=Path,default=Path('data/go_terms_mf.txt'))
    parser.add_argument('--cc-terms',type=Path,default=Path('data/go_terms_cc.txt'))
    parser.add_argument('--groups',nargs='+',choices=GROUPS,default=['future','history','diagnostic'])
    parser.add_argument('--output',type=Path,default=Path('data/unidata_with_dark_weak_eval.pkl'))
    parser.add_argument('--manifest',type=Path,default=Path('data/dark_weak_test/eval_metadata_manifest.json'))
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--replace-existing-splits',action='store_true')
    args=parser.parse_args(argv)
    build(args)

if __name__=='__main__':
    main()
