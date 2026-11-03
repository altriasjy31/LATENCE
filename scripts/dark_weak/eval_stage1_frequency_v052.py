#!/usr/bin/env python3
"""Read-only GO frequency strata analysis for a completed Dark–Weak Stage-1 suite.

No network/model inference, no checkpoint selection, no modification of the
original run_complete_evaluation.py reports. Uses the identical *native* v085
Fmax/AuPRC/AP kernels on stratified GO columns, while evaluating future direct
GO ranks in the full canonical GO vocabulary, not oracle-filtered columns.

Frequency counts use unique historical TRAIN proteins per canonical GO term;
this fixes alias (alt_id) duplication and prevents 2025 test labels entering
frequency definitions. Quantile formulas match Stage-1 make_frequency_bins,
though canonicalization can change group composition for aliased GO terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / 'scripts'
for item in (PROJECT, SCRIPTS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import eval_dark_weak_predictions as ranking
from dark_weak.native_v085_metrics import native_imports, native_protocol, unified_v085_metrics
from utils.util_functions import make_frequency_bins

SCHEMA = 'latence-stage1-dark-weak-frequency-v052'
CLASSIC_NAMES = ('zero_train','rare','rare_le_5','medium','common','common_ge_50')
FIXED_GROUPS = [('zero',0,0),('1-5',1,5),('6-9',6,9),
                ('10-99',10,99),('100-999',100,999),('1000+',1000,None)]
DEFAULT_K = (10,50,100)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def np_unique_idx(value, width, label):
    a = np.asarray(value, dtype=np.int64).reshape(-1)
    if a.size and (a.min() < 0 or a.max() >= width):
        raise ValueError(f'{label}: GO index outside [0,{width})')
    return np.unique(a)


def read_vocabulary(root, ns, expected_hash=None):
    folder = root / 'labels' / ns
    path = folder / 'go_ids.txt'
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError('Frozen GO vocabulary SHA256 differs from the original complete manifest')
    gos = ranking.read_go_list(path)
    vocab = pd.read_csv(folder/'vocabulary_mapping.tsv', sep='\t').sort_values('model_column')
    if vocab['model_column'].tolist() != list(range(len(gos))) or vocab['model_go_id'].astype(str).tolist() != gos:
        raise ValueError('vocabulary_mapping.tsv differs from original GO classifier column order')
    canonical = vocab.canonical_go_id_2022.astype(str).tolist()
    if any(not x.startswith('GO:') for x in canonical):
        raise ValueError('Invalid canonical GO identifier')
    unique = list(dict.fromkeys(canonical))
    index = {go:i for i,go in enumerate(unique)}
    inverse = np.asarray([index[g] for g in canonical], dtype=np.int32)
    alias_sizes = np.bincount(inverse, minlength=len(unique)).astype(np.int32)
    return gos, unique, inverse, alias_sizes


def get_train_counts(path, ns, split, field, ncols, inverse, ncanon):
    with Path(path).open('rb') as f:
        data = pickle.load(f)  # Trust only the original local LATENCE metadata.
    if split not in data or ranking.NS[ns] not in data[split]:
        raise KeyError(f'Training metadata lacks {split}/{ranking.NS[ns]}')
    group = data[split][ranking.NS[ns]]
    ids = list(map(str, group['proteins']))
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate training protein IDs: frequency must count distinct proteins')
    ann = group[field]
    if len(ids) != len(ann):
        raise ValueError('Train protein and label array length mismatch')
    raw = np.zeros(ncols, dtype=np.int64)
    canon = np.zeros(ncanon, dtype=np.int64)
    for protein, row in zip(ids, ann):
        cols = np_unique_idx(row, ncols, f'train/{protein}')
        if cols.size:
            raw[cols] += 1
            canon[np.unique(inverse[cols])] += 1
    return raw, canon, len(ids)


def classic_bins(counts):
    src = make_frequency_bins(counts)
    out = {name: np.asarray(src.get(name,np.zeros(len(counts), dtype=bool)), dtype=bool)
           for name in CLASSIC_NAMES}
    pos = counts[counts > 0]
    quantiles = np.quantile(pos, [0.33, 0.66]).tolist() if len(pos) else [None, None]
    return out, {'q33':quantiles[0], 'q66':quantiles[1],
                 'semantics':'Original Stage-1 quantile rules applied to canonicalized historical positive-protein counts',
                 'overlap':'rare_le_5 and common_ge_50 are diagnostic intersections, not disjoint partitions'}


def fixed_bins(counts):
    result = {}
    for name, low, high in FIXED_GROUPS:
        result[name] = (counts>=low) & ((counts<=high) if high is not None else np.ones(len(counts), bool))
    return result


def load_v094_core_frequency(manifest_path, gos, inverse, ncanon):
    mp = Path(manifest_path).resolve(strict=True)
    data = read_json(mp)
    if data.get('schema') != 'nbs-weak-dataset-v094-r1' or data.get('complete') is not True:
        raise ValueError('Expected complete native v094 unified_dataset_manifest.json')
    if int(data.get('num_task_go',-1)) != len(gos):
        raise ValueError('Native v094 core frequency GO dimension mismatch')
    fname = data['files']['core_counts']
    fpath = mp.parent / fname
    if sha256(fpath) != data['hashes'][fname]:
        raise ValueError('v094 core-count file differs from its native hashed manifest')
    registry = data.get('sources',{}).get('task_registry', {})
    if not registry or not registry.get('path') or not registry.get('sha256'):
        raise ValueError('v094 native source missing hashed GO registry provenance')
    rpath = Path(registry['path'])
    if sha256(rpath) != registry['sha256']:
        raise ValueError('v094 native GO registry source changed since construction')
    # The GO registry must match the primary classifier column order, not just width.
    reg = pd.read_csv(rpath, sep='\t')
    possible = ('input_go_id','model_go_id','go_id')
    idcol = next((x for x in possible if x in reg.columns), None)
    indexcol = next((x for x in ('go_idx','model_column','index') if x in reg.columns), None)
    if not idcol or not indexcol:
        raise ValueError(f'Cannot verify native GO registry order: {list(reg.columns)}')
    reg = reg.sort_values(indexcol)
    if reg[indexcol].tolist() != list(range(len(gos))) or reg[idcol].astype(str).tolist() != gos:
        raise ValueError('v094 core frequency original GO columns are NOT aligned with Stage-1 GO IDs')
    source_counts = np.asarray(np.load(fpath, allow_pickle=False), dtype=np.int64)
    if source_counts.shape != (len(gos),) or (source_counts<0).any():
        raise ValueError('Invalid v094 native core count vector')
    # v094 core label CSR stores per-column counts. Canonical alias columns share
    # an ontology entity; max is an explicit proxy, not re-counted unique proteins.
    canonical = np.zeros(ncanon, np.int64)
    np.maximum.at(canonical, inverse, source_counts)
    return canonical, source_counts, {'unified_manifest':str(mp), 'sha256':sha256(mp),
              'native_core_counts_sha256':sha256(fpath),'go_registry_sha256':registry['sha256'],
              'canonical_alias_policy':'maximum raw core count among canonical alias-equivalent columns; not exact union'}


def load_reference_manifest(path, root, verify):
    p = Path(path).resolve(strict=True)
    manifest=read_json(p)
    if manifest.get('namespace') not in ranking.NS:
        raise ValueError('Original manifest task must be bp/mf/cc')
    if (manifest.get('metric_scope')!='unified' or manifest.get('metric_label_view')!='propagated'
            or manifest.get('ranking_label_view')!='direct'):
        raise ValueError('Expected original complete suite with propagated metric gold and direct rank gold')
    if manifest.get('protocol',{}).get('model_fusion')!='none':
        raise ValueError('This extension expects unmodified Stage-1 per-model probability matrices')
    if manifest.get('native_kernels',{}).get('v085_native_standardized',{}).get('metric_schema')!='0.8.5-standard-metrics-v1':
        raise ValueError('Original Stage-1 result does not use v085 standardized metrics')
    vocab=ranking.read_go_list(root/'labels'/manifest['namespace']/'go_ids.txt')
    if verify and sha256(root/'labels'/manifest['namespace']/'go_ids.txt')!=manifest.get('vocabulary_sha256'):
        raise ValueError('GO vocabulary hash differs from original Stage-1 score manifest')
    original=manifest['input_files']
    result={}
    for model, x in original.items():
        pp=Path(x['probability']).resolve(strict=True)
        idp=Path(x['protein_ids']).resolve(strict=True)
        gp=Path(x['go_ids']).resolve(strict=True)
        if verify and sha256(pp)!=x['probability_sha256']:
            raise ValueError(f'{model}: prediction differs from original score manifest')
        if ranking.read_go_list(gp)!=vocab:
            raise ValueError(f'{model}: GO vocabulary order differs')
        arr=ranking.load_prediction(pp,len(vocab))
        ids=ranking.load_protein_order(idp,arr.shape[0])
        result[model]=(arr,ids,pp,idp,gp)
    md=Path(manifest['metadata']).resolve(strict=True)
    if verify and sha256(md)!=manifest.get('metadata_sha256'):
        raise ValueError('Future annotation metadata changed since original evaluation')
    if 'backbone' not in result:
        raise ValueError('Existing complete suite must contain backbone reference')
    return manifest, md, result


def load_future_case(meta, split, ns, width):
    if split not in meta or ranking.NS[ns] not in meta[split]:
        raise KeyError(f'Missing future split: {split}/{ns}')
    d=meta[split][ranking.NS[ns]]
    ids=[str(x) for x in d['proteins']]
    if len(ids)!=len(set(ids)) or len(d['annotations'])!=len(ids) or len(d['prop_annotations'])!=len(ids):
        raise ValueError('Future case protein IDs or label dimensions invalid')
    direct=[]
    propagated=np.zeros((len(ids),width),np.uint8)
    for i,(a,pa) in enumerate(zip(d['annotations'],d['prop_annotations'])):
        da=np_unique_idx(a,width,f'{split}/{ids[i]} direct')
        full=np_unique_idx(pa,width,f'{split}/{ids[i]} propagated')
        if not np.isin(da,full).all():
            raise ValueError(f'{split}/{ids[i]}: propagated labels must include direct GO')
        direct.append(da)
        propagated[i,full]=1
    return ids, direct, propagated


def global_rank_records(arr, indices, direct, names, gos, canon, inverse, alias_sizes, root_id, group_masks, group_frequencies, ns, split, model, seqhash, ks):
    root_col = canon.index(root_id) if root_id in canon else -1
    allowed=np.asarray([i for i in range(len(canon)) if i!=root_col],dtype=np.int32)
    scores_denom=alias_sizes.astype(np.float64)
    out=[]
    for pi, (protein, direct_cols) in enumerate(zip(names,direct)):
        pos_canon=np.unique(inverse[direct_cols])
        pos_canon=pos_canon[pos_canon!=root_col]
        if pos_canon.size==0:
            continue
        raw=np.asarray(arr[indices[pi]],np.float64)
        if not np.isfinite(raw).all() or (raw<0).any() or (raw>1).any():
            raise ValueError(f'{model}/{protein}: invalid probability range')
        combined=np.bincount(inverse,weights=raw,minlength=len(canon))/scores_denom
        subset=combined[allowed]
        local_pos=np.searchsorted(allowed,pos_canon)
        ranks=ranking.positive_ranks(subset,local_pos)
        for c, rank in zip(pos_canon,ranks):
            for family, bins in group_masks.items():
                for group, m in bins.items():
                    if m[c]:
                        out.append({'split':split,'family':family,'bin':group,'model':model,
                                    'protein':protein,'sequence_hash':seqhash.get(protein,protein),
                                    'go_id':canon[int(c)],'train_frequency':int(group_frequencies[family][int(c)]),
                                    'rank':int(rank),'rank_percentile':float(1-(int(rank)-1)/max(1,len(allowed)-1)),
                                    'reciprocal_rank':1./int(rank),
                                    **{f'hit_at_{k}':int(rank<=k) for k in ks}})
    return out



def rank_summary(df, ks):
    if df.empty:
        return {'n_direct_pairs':0,'n_direct_proteins':0,'n_direct_GO':0, 'pairwise_mrr':None,
                'mean_best_rr':None,**{f'micro_recall_at_{k}':None for k in ks},
                **{f'macro_recall_at_{k}':None for k in ks}}
    result={'n_direct_pairs':int(len(df)), 'n_direct_proteins':int(df.protein.nunique()),
            'n_direct_GO':int(df.go_id.nunique()),
            'pairwise_mrr':float(df.reciprocal_rank.mean()),
            'mean_best_rr':float(df.groupby('protein').reciprocal_rank.max().mean())}
    for k in ks:
        result[f'micro_recall_at_{k}']=float(df[f'hit_at_{k}'].mean())
        result[f'macro_recall_at_{k}']=float(df.groupby('protein')[f'hit_at_{k}'].mean().mean())
    return result


def boot_pair_delta(model_rows, base_rows, ks, reps=500, seed=6061):
    # Both models have the same direct future-positive GO pairs.
    cols=['protein','go_id','sequence_hash']
    a=model_rows.sort_values(cols).reset_index(drop=True)
    b=base_rows.sort_values(cols).reset_index(drop=True)
    if not a[cols].equals(b[cols]):
        raise ValueError('Paired rank bootstrap requires identical protein/GO/sequence pairs')
    clusters, inv=np.unique(a.sequence_hash.to_numpy(),return_inverse=True)
    n=len(clusters)
    rng=np.random.default_rng(seed)
    outputs={}
    for metric, col in [('pairwise_mrr','reciprocal_rank'),
                        *[(f'micro_recall_at_{k}',f'hit_at_{k}') for k in ks]]:
        diffs=a[col].to_numpy(dtype=float)-b[col].to_numpy(dtype=float)
        sums=np.bincount(inv,weights=diffs,minlength=n)
        ns=np.bincount(inv,minlength=n)
        if reps > 0:
            draws=rng.integers(0,n,size=(reps,n))
            vals=sums[draws].sum(axis=1)/ns[draws].sum(axis=1)
            ci=np.quantile(vals,[.025,.975]).tolist()
        else:
            ci=[None,None]
        outputs[metric]={'delta':float(diffs.mean()),'ci95':ci,
                         'bootstrap_unit':'unique sequence SHA256 or protein fallback',
                         'n_clusters':n,'repetitions':reps}
    return outputs


def run(args):
    mp=Path(args.complete_manifest).resolve(strict=True)
    out=Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {out}')
    unified=Path(args.unified_root).resolve(strict=True)
    base, eval_md, models=load_reference_manifest(mp,unified,not args.skip_hash_verify)
    ns=base['namespace']
    gos, canon, inverse, alias_sizes=read_vocabulary(unified,ns,base.get('vocabulary_sha256') if not args.skip_hash_verify else None)
    ncols=len(gos)
    train_path=Path(args.train_metadata).resolve(strict=True)
    raw, historical_counts,ntrain=get_train_counts(train_path,ns,args.train_split,args.train_label_field,ncols,inverse,len(canon))
    # Freeze canonical frequency bands across all three Stage-1 models & test cases.
    cbins, cmeta=classic_bins(historical_counts)
    families={'historical_classic':cbins,
              'historical_fixed':fixed_bins(historical_counts)}
    meta_freq={'historical_classic':historical_counts,
               'historical_fixed':historical_counts}
    freq_meta={'historical_classic':cmeta,
               'historical_fixed':{'intervals':FIXED_GROUPS,
                   'semantics':'v094-style fixed intervals applied to the same Stage-1 historical train counts'}}
    v094info=None
    if args.v094_unified_manifest:
        v094count, v094raw, v094info=load_v094_core_frequency(
            args.v094_unified_manifest,gos,inverse,len(canon))
        families['v094_native_core_fixed']=fixed_bins(v094count)
        meta_freq['v094_native_core_fixed']=v094count
        freq_meta['v094_native_core_fixed']={'intervals':FIXED_GROUPS,
            'semantics':'Official v094 training core counts (GO alias max reconciliation); cross-stage diagnostic'}
    chosen={f:families[f] for f in args.families if f in families}
    unknown=set(args.families)-set(families)
    if unknown:
        raise ValueError(f'Requested frequency families unavailable: {unknown}; supply --v094-unified-manifest if needed')
    if not chosen:
        raise ValueError('At least one frequency family is required')
    for model, (a, orders, *_) in models.items():
        if a.shape[1]!=ncols:
            raise ValueError(f'{model}: GO width mismatch')
    cohort=unified/'tables'/'full_dark_weak_cohort.tsv'
    seqhash={}
    if cohort.is_file():
        temp=pd.read_csv(cohort,sep='\t',usecols=['protein_22','sequence_sha256'])
        seqhash=dict(zip(temp.protein_22.astype(str),temp.sequence_sha256.astype(str)))
    with eval_md.open('rb') as f:
        test_meta=pickle.load(f)  # Trust local LATENCE pickle only.
    native=native_imports(PROJECT) if args.metric_scope=='all' else None
    cases=args.splits or base['cases']
    rows=[]; pairs=[]; deltas=[]; class_rows=[]
    for family,bins in chosen.items():
        freq=meta_freq[family]
        for label, cmask in bins.items():
            colmask=cmask[inverse]
            class_rows.append({'family':family,'bin':label,
                'n_canonical_GO':int(cmask.sum()),'n_model_columns':int(colmask.sum()),
                'n_training_positive_GO':int(np.count_nonzero(freq[cmask])),
                'training_freq_min':int(freq[cmask].min()) if cmask.any() else None,
                'training_freq_max':int(freq[cmask].max()) if cmask.any() else None})
    for split in cases:
        print(f'[frequency] {split}: loading frozen predictions',flush=True)
        names,direct,y=load_future_case(test_meta,split,ns,ncols)
        if len(names)==0:
            raise ValueError(f'Empty future cohort: {split}')
        prepared={}
        for model,(array,order,*_) in models.items():
            missing=[name for name in names if name not in order]
            if missing:
                raise ValueError(f'{model}/{split} missing {len(missing)} protein IDs, e.g. {missing[:3]}')
            p=np.asarray(array[[order[name] for name in names]],dtype=np.float32)
            if not np.isfinite(p).all() or (p<0).any() or (p>1).any():
                raise ValueError(f'{model}/{split}: probability outside [0,1]')
            prepared[model]=p
            pair_rows=global_rank_records(p,np.arange(len(names)),direct,names,gos,canon,inverse,
                                          alias_sizes,ranking.ROOT_IDS[ns],chosen,meta_freq,ns,split,model,seqhash,args.ks)
            pairs.extend(pair_rows)
        for family,bins in chosen.items():
            for binname, canon_mask in bins.items():
                colmask=canon_mask[inverse]
                ybin=y[:,colmask]
                npositives=int(ybin.sum()); has_gold=int((ybin.sum(axis=1)>0).sum())
                nterm=int(canon_mask.sum())
                for model,p in prepared.items():
                    entry={'split':split,'family':family,'bin':binname,'model':model,
                           'num_rows':len(names),'n_canonical_GO':nterm,
                           'n_model_columns':int(colmask.sum()),
                           'n_propagated_positive_columns':npositives,
                           'n_propagated_positive_proteins':has_gold,
                           'observed_prevalence_percent':(100*npositives/ybin.size if ybin.size else None),
                           'Fmax_percent':None,'AuPRC_percent':None,'AP_percent':None,
                           'Fmax_threshold':None,'status':'no_positives' if not npositives else 'ok'}
                    if args.metric_scope=='all' and npositives and colmask.any():
                        if ybin.size>args.max_metric_cells and not args.allow_large_metrics:
                            raise MemoryError(f'{split}/{binname} contains {ybin.size:,} metric cells. '
                                'Use --allow-large-metrics only after checking RAM.')
                        stats=unified_v085_metrics(ybin,p[:,colmask],native,verify_fmax=not args.skip_fmax_parity)
                        entry.update(Fmax_percent=stats['Fmax_percent'],
                                     AuPRC_percent=stats['AuPRC_percent'],AP_percent=stats['AP_percent'],
                                     Fmax_threshold=stats['standard_protein_fmax_threshold'])
                    rows.append(entry)
        del prepared,y
    pair_df=pd.DataFrame(pairs)
    if pair_df.empty:
        raise ValueError('No future direct GO positive pairs were available for rank analysis')
    metrics_df=pd.DataFrame(rows)
    rank_rows=[]
    for (split,family,binname,model),g in pair_df.groupby(['split','family','bin','model'],sort=False):
        rank_rows.append({'split':split,'family':family,'bin':binname,'model':model,
                          **rank_summary(g,args.ks)})
    rank_df=pd.DataFrame(rank_rows)
    # Include empty ranking strata in the tabular output rather than silently drop them.
    base_rows=metrics_df[['split','family','bin','model']].drop_duplicates()
    rank_df=base_rows.merge(rank_df,on=['split','family','bin','model'],how='left')
    rank_df[['n_direct_pairs','n_direct_proteins','n_direct_GO']]=rank_df[
        ['n_direct_pairs','n_direct_proteins','n_direct_GO']].fillna(0).astype(int)
    for split in cases:
        for family,bins in chosen.items():
            for group in bins:
                g=pair_df[(pair_df.split==split)&(pair_df.family==family)&(pair_df.bin==group)]
                if g.empty:
                    continue
                baseline=g[g.model=='backbone']
                for model in models:
                    if model=='backbone':continue
                    other=g[g.model==model]
                    result=boot_pair_delta(other,baseline,args.ks,args.bootstrap,args.seed)
                    for metric,rec in result.items():
                        deltas.append({'split':split,'family':family,'bin':group,'model':model,
                                       'reference':'backbone','metric':metric,**rec})
    # Every model shares the same columns / proteins for paired Fmax/AUPRC/AP deltas.
    pivot=metrics_df.pivot(index=['split','family','bin'],columns='model',values=['Fmax_percent','AuPRC_percent','AP_percent'])
    metric_deltas=[]
    for (split,family,b),v in pivot.iterrows():
        for model in models:
            if model=='backbone':continue
            for metric in ('Fmax_percent','AuPRC_percent','AP_percent'):
                score=v.get((metric,model)); ref=v.get((metric,'backbone'))
                metric_deltas.append({'split':split,'family':family,'bin':b,
                                      'model':model,'reference':'backbone', 'metric':metric,
                                      'delta_percentage_points':(float(score-ref) if pd.notna(score) and pd.notna(ref) else None)})
    summary=metrics_df.merge(rank_df,on=['split','family','bin','model'],how='left')
    out.mkdir(parents=True,exist_ok=False)
    summary.to_csv(out/'frequency_metrics.tsv',sep='\t',index=False)
    pd.DataFrame(class_rows).to_csv(out/'frequency_groups.tsv',sep='\t',index=False)
    pair_df.to_csv(out/'direct_positive_global_ranks.tsv',sep='\t',index=False)
    pd.DataFrame(deltas).to_csv(out/'paired_rank_bootstrap_deltas.tsv',sep='\t',index=False)
    pd.DataFrame(metric_deltas).to_csv(out/'metric_deltas_vs_backbone.tsv',sep='\t',index=False)
    columns={'model_column':np.arange(ncols),'model_go_id':gos,
             'canonical_go_id_2022':[canon[i] for i in inverse],
             'historical_train_raw_column_count':raw,
             'historical_train_canonical_unique_protein_count':historical_counts[inverse]}
    for f,bins in chosen.items():
        columns[f'{f}_train_count']=meta_freq[f][inverse]
        for b,m in bins.items():columns[f'{f}__{b}']=m[inverse]
    pd.DataFrame(columns).to_csv(out/'go_frequency_lookup.tsv',sep='\t',index=False)
    base_metrics=mp.parent/f'{ns}_complete_metrics.tsv'
    if base_metrics.is_file():
        old=pd.read_csv(base_metrics,sep='\t')
        old.to_csv(out/'original_unstratified_metrics_copy.tsv',sep='\t',index=False)
        print('[original overall]')
        fields=[c for c in ['split','model','num_rows','Fmax_percent','AuPRC_percent','AP_percent'] if c in old]
        print(old[fields].to_string(index=False),flush=True)
    audit={'schema':SCHEMA,'complete_manifest':str(mp),'complete_manifest_sha256':sha256(mp),
           'original_unstratified_metrics_path':str(base_metrics) if base_metrics.is_file() else None,
           'future_eval_metadata':str(eval_md), 'future_eval_metadata_sha256':sha256(eval_md),
           'train_metadata':str(train_path),'train_metadata_sha256':sha256(train_path),
           'train_split':args.train_split,'train_label_field':args.train_label_field,
           'training_proteins':ntrain,'num_model_GO_columns':ncols,'num_canonical_GO':len(canon),
           'num_GO_alias_extra_columns':ncols-len(canon),
           'n_column_freq_different_from_canonical_freq':int(np.sum(raw!=historical_counts[inverse])),
           'frequency_groups':freq_meta,'family_names':list(chosen),
           'model_probability_sha256':{model:base['input_files'][model]['probability_sha256'] for model in models},
           'native_metric_protocol':native_protocol(native) if native else None,
           'scope':'Stratified v085 GO-restricted metrics and full-vocabulary future-direct-GO ranking',
           'label_policy':{'metrics':'prop_annotations full 2022 ancestor closure',
                           'ranking':'annotations direct future GO only',
                           'frequencies':'historical training positive GO per distinct protein, canonical alias union',
                           'zero_shot':'zero_train means no historical *direct* training positive; NOT strict model-wide GO zero-shot',
                           'AP_AuPRC':'Observed GO unlabeled positions treated as negatives for proxy metrics; not validated true negatives',
                           'Fmax':'per-bin optimum tuned on future labels (descriptive, not deployable threshold)',
                           'ranking_space':'ALL canonical GO, root excluded; bin filters POSITIVES after global ranking',
                           'alias_prediction':'mean of original per-column scores per canonical GO',
                           'case_overlap':'Experimental/IPR2GO/Union overlap; DO NOT treat as independent samples',
                           'R5':'classic Stage-1 quantile rules applied after GO alias canonicalization'},
           'v094_native_frequency_provenance':v094info,
           'cases':cases,'ks':list(args.ks),'bootstrap_reps':args.bootstrap,
           'score_scaling':{'Fmax_percent':'0..100','AuPRC_percent':'0..100',
                            'AP_percent':'0..100','rank_recall_and_MRR':'0..1'},
           'original_reports_modified':False}
    (out/'frequency_manifest.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(f'[frequency] Completed: {out}',flush=True)
    return summary,audit


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--complete-manifest',type=Path,required=True,
                    help='Existing outputs/dark_weak_eval/stage1_full_bp_epoch100/bp_complete_manifest.json')
    ap.add_argument('--unified-root',type=Path,default=Path('data/dark_weak_test/unified'))
    ap.add_argument('--train-metadata',type=Path,default=Path('data/unidata_with_exp_train_pseudo.pkl'))
    ap.add_argument('--train-split',default='train')
    ap.add_argument('--train-label-field',choices=('annotations','prop_annotations'),default='annotations')
    ap.add_argument('--v094-unified-manifest',type=Path,default=None)
    ap.add_argument('--families',nargs='+',default=['historical_classic','historical_fixed'],
                    choices=('historical_classic','historical_fixed','v094_native_core_fixed'))
    ap.add_argument('--splits',nargs='+',default=None)
    ap.add_argument('--metric-scope',choices=('all','ranking_only'),default='all')
    ap.add_argument('--ks',nargs='+',type=int,default=list(DEFAULT_K))
    ap.add_argument('--bootstrap',type=int,default=300)
    ap.add_argument('--seed',type=int,default=6061)
    ap.add_argument('--skip-fmax-parity',action='store_true')
    ap.add_argument('--skip-hash-verify',action='store_true',help='Only for a copy of files whose paths moved; never use for publication')
    ap.add_argument('--max-metric-cells',type=int,default=80_000_000)
    ap.add_argument('--allow-large-metrics',action='store_true')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv)
    if not args.ks or any(k<1 for k in args.ks) or len(args.ks)!=len(set(args.ks)):
        ap.error('--ks must be distinct positive integers')
    if args.bootstrap<0:ap.error('--bootstrap must be >= 0')
    if len(args.families)!=len(set(args.families)):ap.error('Duplicate --families')
    run(args)


if __name__ == '__main__':
    main()
