#!/usr/bin/env python3
"""Open-world, positive-only ranking evaluation on LATENCE dark-weak metadata.

Accepts pre-exported full-GO arrays from any LATENCE predictor (Backbone,
Expert, Modelout, NBS). DOES NOT run model forward passes or claim to reproduce
a project-specific Fmax/AUPRC backend. The index/order identity contract is
strictly checked for every model.

Example:
 python scripts/eval_dark_weak_predictions.py \
   --metadata data/unidata_with_dark_weak_eval.pkl \
   --unified-root data/dark_weak_test/unified \
   --split dark_weak_test_exp_direct --namespace bp \
   --prediction backbone=outputs/backbone_bp.npy \
   --prediction-ids backbone=outputs/bp_protein_ids.txt \
   --prediction-go-ids backbone=data/go_terms_bp.txt \
   --prediction nbs=outputs/nbs_bp.npy \
   --prediction-ids nbs=outputs/bp_protein_ids.txt \
   --prediction-go-ids nbs=data/go_terms_bp.txt \
   --reference backbone --out data/dark_weak_test/evaluations/exp_bp
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import json
import pickle
import re
import warnings
import numpy as np
import pandas as pd

GO_RE = re.compile(r'GO:\d{7}')
NS = {'bp':'biological_process','mf':'molecular_function','cc':'cellular_component'}
ROOT_IDS = {'bp':'GO:0008150','mf':'GO:0003674','cc':'GO:0005575'}


def parse_kv(entries:list[str], what:str):
    out = {}
    for e in entries:
        if '=' not in e:
            raise ValueError(f'{what} must be MODEL=PATH, received {e!r}')
        name, val = e.split('=',1)
        name, val = name.strip(), val.strip()
        if not name or not val or name in out:
            raise ValueError(f'{what}: blank or duplicate MODEL in {e!r}')
        out[name] = Path(val)
    return out


def read_go_list(path:Path):
    terms=GO_RE.findall(path.read_text(encoding='utf-8'))
    if not terms or len(terms)!=len(set(terms)):
        raise ValueError(f'Malformed GO list {path}')
    return terms


def load_case(metadata_path:Path,root:Path,split:str,ns:str,remove_roots=True,label_view='auto'):
    with metadata_path.open('rb') as f:
        metadata = pickle.load(f)  # trusted LATENCE pickle only
    if not isinstance(metadata,dict) or split not in metadata:
        raise KeyError(f'No {split!r} in {metadata_path}')
    label_dict=metadata[split][NS[ns]]
    proteins = list(label_dict['proteins'])
    # Field invariant: prop_annotations ALWAYS stores the full ancestor closure.
    # The '_direct' suffix requests directly supported functions, not a
    # redefinition of the prop_annotations field.
    if label_view not in ('auto','direct','propagated'):
        raise ValueError(f'Unknown label_view={label_view}')
    label_field = ('annotations' if (split.endswith('_direct') if label_view=='auto'
                    else label_view=='direct') else 'prop_annotations')
    if label_field not in label_dict:
        raise KeyError(
            f'Missing {label_field} for {split}/{NS[ns]}. '
            'Rebuild dark-weak metadata with v0.3.1; v0.3.0 used prop_annotations '
            'for direct-only labels and violated the LATENCE field contract.'
        )
    annots = list(label_dict[label_field])
    if len(proteins)!=len(annots) or len(set(proteins))!=len(proteins):
        raise ValueError('Duplicate protein or protein/label list length mismatch')
    folder=root/'labels'/ns
    gos=read_go_list(folder/'go_ids.txt')
    table=pd.read_csv(folder/'vocabulary_mapping.tsv',sep='\t').sort_values('model_column')
    if (table['model_column'].to_list()!=list(range(len(gos)))
        or table['model_go_id'].tolist()!=gos):
        raise ValueError('GO mapping columns do not match model GO order')
    canon=table.canonical_go_id_2022.astype(str).tolist()
    canon_unique=list(dict.fromkeys(canon))
    lut={v:i for i,v in enumerate(canon_unique)}
    col_to_canon=np.array([lut[g] for g in canon],dtype=np.int32)
    counts=np.bincount(col_to_canon,minlength=len(canon_unique)).astype(np.float64)
    root=ROOT_IDS[ns]
    excluded={lut[root]} if remove_roots and root in lut else set()
    acceptable=[i for i in range(len(canon_unique)) if i not in excluded]
    display_indices={c:j for j,c in enumerate(acceptable)}
    labels=[]
    names=[]
    for protein,indices in zip(proteins,annots):
        if not isinstance(indices,(list,tuple,np.ndarray)):
            raise TypeError(f'{split}/{NS[ns]} annotations must be list of GO integer indices')
        ints = np.asarray(indices,dtype=np.int64)
        if len(ints) and (ints.min()<0 or ints.max()>=len(gos)):
            raise IndexError(f'GO label indices outside [0,{len(gos)}) for {protein}')
        # Alias-equivalent model output columns count as one canonical GO label.
        cset=sorted(set(col_to_canon[ints].tolist())-excluded)
        if not cset:
            continue
        names.append(protein)
        labels.append(np.array([display_indices[i] for i in cset],dtype=np.int32))
    if not names:
        raise ValueError(f'No positive labels for {split}/{NS[ns]} after root filtering')
    return names,labels,gos,col_to_canon,counts,np.asarray(acceptable,dtype=np.int32),canon_unique


def load_prediction(path:Path, expected_width:int):
    if path.suffix!='.npy':
        raise ValueError(f'Only .npy arrays (memmap-compatible) supported: {path}')
    arr=np.load(path,mmap_mode='r',allow_pickle=False)
    if arr.ndim!=2 or arr.shape[1]!=expected_width or not np.issubdtype(arr.dtype,np.number):
        raise ValueError(f'Invalid prediction matrix {path}: shape={arr.shape}, expected width {expected_width}')
    return arr


def load_protein_order(path:Path,rows:int):
    protein_ids=[x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(protein_ids)!=rows or len(set(protein_ids))!=rows:
        raise ValueError(f'Prediction ID list must have {rows} unique IDs: {path}')
    return {v:i for i,v in enumerate(protein_ids)}


def collapse_canonical(raw, inverse, counts, accepted):
    """Pre-specified mean over alias-equivalent model columns; no label-peeking."""
    if not np.isfinite(raw).all():
        raise ValueError('Predictions contain NaN or infinity')
    combined=np.bincount(inverse,weights=raw,minlength=len(counts))/counts
    return combined[accepted]


def positive_ranks(scores,positive):
    """1-based ranks, stable column order for ties, vectorized when few positives."""
    n=len(scores)
    if len(positive)<=3:
        # The tie-breaker must match stable descending argsort.
        return np.array([1+np.count_nonzero(scores>scores[i])+
                         np.count_nonzero(scores[:i]==scores[i])
                         for i in positive],dtype=np.int32)
    order=np.argsort(-scores,kind='stable')
    ranks=np.empty(n,dtype=np.int32)
    ranks[order]=np.arange(1,n+1,dtype=np.int32)
    return ranks[positive]


def summarize_per_protein(df:pd.DataFrame,ks:list[int]):
    out={'n_proteins':len(df),'n_confirmed_positive_pairs':int(df['n_positive'].sum()),
         'mean_best_positive_reciprocal_rank':float(df['best_rr'].mean()),
         'pairwise_mrr':float(df['rr_sum'].sum()/df['n_positive'].sum()),
         'mean_positive_rank_percentile':float(df['rank_percentile_sum'].sum()/df['n_positive'].sum())}
    for k in ks:
        out[f'macro_recall_at_{k}']=float(df[f'recall_at_{k}'].mean())
        out[f'micro_recall_at_{k}']=float(df[f'hits_at_{k}'].sum()/df['n_positive'].sum())
    return out


def compute_rows(model, pred_ids, names, labs, inverse, counts, accepted, ks,
                 seqhash_by_protein=None, save_ranks=False):
    missing=[v for v in names if v not in pred_ids]
    if missing:
        raise KeyError(f'Predictions missing {len(missing)} required proteins: {missing[:10]}')
    result=[]
    pairs=[]
    candidate_n=len(accepted)
    for name, positives in zip(names,labs):
        raw=np.asarray(model[pred_ids[name]],dtype=np.float64)
        score=collapse_canonical(raw,inverse,counts,accepted)
        ranks=positive_ranks(score,positives)
        rr=1.0/ranks.astype(np.float64)
        # 1.0 = best rank; 0.0 = worst rank.
        ranks_norm=1.0-(ranks.astype(np.float64)-1)/max(1,candidate_n-1)
        row={'protein':name,
             'sequence_sha256':(seqhash_by_protein or {}).get(name,name),
             'n_positive':len(positives),
             'rr_sum':float(rr.sum()),'best_rr':float(rr.max()),
             'rank_percentile_sum':float(ranks_norm.sum())}
        for k in ks:
            nhits=int((ranks<=k).sum())
            row[f'hits_at_{k}']=nhits
            row[f'recall_at_{k}']=nhits/len(positives)
        result.append(row)
        if save_ranks:
            for col,rank in zip(positives,ranks):
                pairs.append({'protein':name,'canonical_candidate_col':int(col),
                              'rank':int(rank),'rank_percentile':float(1-(rank-1)/max(candidate_n-1,1))})
    return pd.DataFrame(result),pd.DataFrame(pairs)


def paired_bootstrap(model_df:pd.DataFrame,base_df:pd.DataFrame,ks:list[int],reps=500,seed=6061):
    if model_df.protein.tolist()!=base_df.protein.tolist():
        raise ValueError('Cannot bootstrap two models evaluated on different protein orders')
    # The sequence hash is a natural resampling cluster; same sequence in
    # different UniProt entries is not treated as independent evidence.
    clusters = model_df.sequence_sha256.astype(str).to_numpy()
    _,inv=np.unique(clusters,return_inverse=True)
    nclusters=int(inv.max()+1)
    sizes=np.bincount(inv)
    rng=np.random.default_rng(seed)
    result={}
    keys=[f'recall_at_{k}' for k in ks]+['best_rr']
    for key in keys:
        diff=model_df[key].to_numpy()-base_df[key].to_numpy()
        sums=np.bincount(inv,weights=diff,minlength=nclusters)
        point=float(diff.mean())
        boots=np.empty(reps,dtype=np.float64)
        for i in range(reps):
            sample=rng.integers(0,nclusters,size=nclusters)
            boots[i]=sums[sample].sum()/sizes[sample].sum()
        result[f'delta_{key}']={'estimate':point,'ci95_low':float(np.quantile(boots,.025)),
                                 'ci95_high':float(np.quantile(boots,.975)),
                                 'bootstrap_unit':'2022 sequence SHA256',
                                 'bootstrap_replicates':reps}
    return result


def run(args):
    predictions=parse_kv(args.prediction,'--prediction')
    ids=parse_kv(args.prediction_ids,'--prediction-ids')
    goids=parse_kv(args.prediction_go_ids,'--prediction-go-ids')
    if set(predictions)!=set(ids) or (not args.assume_go_order and set(predictions)!=set(goids)):
        raise ValueError('Every model must supply prediction array, protein IDs, and GO IDs (unless --assume-go-order)')
    if args.reference is not None and args.reference not in predictions:
        raise ValueError(f'--reference {args.reference} missing from models')
    if 'removed' in args.split and not args.allow_descriptive_labels:
        raise ValueError('Removed IPR2GO mappings are NOT biological negatives. Use --allow-descriptive-labels for descriptive positive-label ranking only.')
    root=args.unified_root
    names,labs,termids,inverse,counts,accepted,canon_names=load_case(
        args.metadata,root,args.split,args.namespace,args.exclude_roots,
        getattr(args,'label_view','auto'))
    hashes={}
    cohort_path=root/'tables'/'full_dark_weak_cohort.tsv'
    if cohort_path.exists():
        cohort=pd.read_csv(cohort_path,sep='\t',usecols=['protein_22','sequence_sha256'])
        hashes=dict(zip(cohort.protein_22.astype(str),cohort.sequence_sha256.astype(str)))
    if not hashes:
        warnings.warn('No sequence hash available: bootstrap will group by protein name, less conservative.')
    arrays={}
    indices={}
    for mod,path in predictions.items():
        arrays[mod]=load_prediction(path,len(termids))
        indices[mod]=load_protein_order(ids[mod],arrays[mod].shape[0])
        if not args.assume_go_order:
            other=read_go_list(goids[mod])
            if termids!=other:
                raise ValueError(f'GO column order differs from LATENCE source order for {mod}')
    results={}
    frames={}
    all_pairs=[]
    args.out.mkdir(parents=True,exist_ok=True)
    for mod,arr in arrays.items():
        print(f'Ranking {mod}: {len(names)} positive proteins in {args.namespace}/{args.split}')
        df,pairs=compute_rows(arr,indices[mod],names,labs,inverse,counts,accepted,args.ks,
                              hashes, args.save_pair_ranks)
        df.insert(0,'model',mod)
        frames[mod]=df
        results[mod]=summarize_per_protein(df,args.ks)
        if args.save_pair_ranks and len(pairs):
            pairs.insert(0,'model',mod)
            # Expand GO index in the canonical-collapsed subset to GO strings.
            pairs['canonical_go_id']=[canon_names[accepted[int(c)]]
                                      for c in pairs.canonical_candidate_col]
            all_pairs.append(pairs)
    result_rows=[{'model':k,'split':args.split,'namespace':args.namespace,**v} for k,v in results.items()]
    pd.DataFrame(result_rows).to_csv(args.out/'metrics.tsv',sep='\t',index=False)
    pd.concat(frames.values(),ignore_index=True).to_csv(args.out/'per_protein.tsv',sep='\t',index=False)
    if all_pairs:
        pd.concat(all_pairs,ignore_index=True).to_csv(args.out/'per_positive_ranks.tsv',sep='\t',index=False)
    comparison={}
    if args.reference is not None:
        for mod in predictions:
            if mod!=args.reference:
                comparison[mod]=paired_bootstrap(frames[mod],frames[args.reference],args.ks,args.bootstrap,args.seed)
    output={
        'split':args.split,'namespace':args.namespace,
        'candidate_go_count':int(len(accepted)),
        'canonical_go_alias_policy':'mean of alias-equivalent model columns',
        'protein_list_alignment':'explicit per-model protein IDs, error on missing ID',
        'go_list_alignment':'explicit full GO list' if not args.assume_go_order else 'assumed unchecked model order',
        'excluded_ontology_roots':bool(args.exclude_roots),
        'open_world_warning':'Unannotated GO positions UNKNOWN, not negatives. No Fmax/AUPRC calculated.',
        'model_metrics':results,'paired_deltas_vs_reference':comparison,
        'model_inputs':{k:{'array':str(v),'ids':str(ids[k]),'go_ids':str(goids.get(k,'UNVERIFIED'))}
                        for k,v in predictions.items()},
    }
    (args.out/'metrics.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(pd.DataFrame(result_rows).to_string(index=False))
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,default=Path('data/unidata_with_dark_weak_eval.pkl'))
    p.add_argument('--unified-root',type=Path,default=Path('data/dark_weak_test/unified'))
    p.add_argument('--split',required=True,help='First-level metadata key, preferably ending _direct for future novel GO')
    p.add_argument('--label-view',choices=['auto','direct','propagated'],default='auto',
                   help='Choose direct annotations versus full ancestor closure explicitly')
    p.add_argument('--namespace',choices=NS,required=True)
    p.add_argument('--prediction',action='append',required=True,help='MODEL=/abs/or/relative/prediction.npy; repeat')
    p.add_argument('--prediction-ids',action='append',required=True,help='MODEL=protein_ids.txt; repeat')
    p.add_argument('--prediction-go-ids',action='append',default=[],help='MODEL=go_ids.txt; repeat')
    p.add_argument('--assume-go-order',action='store_true',help='Unsafe escape hatch; records that GO order was not verified')
    p.add_argument('--reference',help='Model name, e.g. backbone; paired Δ and sequence-cluster bootstrap')
    p.add_argument('--ks',type=int,nargs='+',default=[10,50,100])
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--seed',type=int,default=6061)
    p.add_argument('--out',type=Path,default=Path('data/dark_weak_test/evaluations/default'))
    p.add_argument('--save-pair-ranks',action='store_true')
    p.add_argument('--allow-descriptive-labels',action='store_true')
    p.add_argument('--include-ontology-roots',dest='exclude_roots',action='store_false')
    p.set_defaults(exclude_roots=True)
    args=p.parse_args(argv)
    if len(set(args.ks))!=len(args.ks) or any(k<=0 for k in args.ks):
        p.error('--ks must be distinct positive integers')
    if args.bootstrap<=0:
        p.error('--bootstrap must be positive')
    run(args)

if __name__=='__main__':
    main()
