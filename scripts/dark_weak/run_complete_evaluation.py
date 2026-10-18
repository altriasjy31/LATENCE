#!/usr/bin/env python3
"""Unified Dark–Weak benchmark evaluation from *frozen predictions*.

Runs existing open-world ranking plus the ORIGINAL v085 standardized metric
kernel for protein-centric Fmax AND PR-AUC, for Stage-1 and NBS predictions.
The obsolete Stage-1 micro-Fmax / micro-AP is optional diagnostic only. Never calls model inference or tunes checkpoints.

All supplied model predictions need explicit, valid protein and GO IDs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
SCRIPTS=PROJECT/'scripts'
sys.path.insert(0,str(SCRIPTS))
import eval_dark_weak_predictions as ranking
from dark_weak.native_v085_metrics import native_imports, unified_v085_metrics, native_protocol

NS=ranking.NS
PANELS={
 'future':['dark_weak_test_exp_direct','dark_weak_test_ipr2go_direct','dark_weak_test_direct'],
 'history':['dark_weak_test_2022_ipr2go_matched_direct',
            'dark_weak_test_2025_ipr2go_all_direct',
            'dark_weak_test_2025_ipr2go_added_direct',
            'dark_weak_test_2025_ipr2go_retained_direct'],
}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def imports_for_metrics(include_stage1=False):
    """Import frozen project-native v085 metrics; Stage-1 micro is optional."""
    kernels = native_imports(PROJECT)
    legacy = None
    if include_stage1:
        try:
            from experiments.eval_weak_ind_test_detr_diagnostics import compute_metric_pack
        except Exception as exc:
            raise ImportError('Optional Stage-1 micro diagnostic source missing at '
                'experiments/eval_weak_ind_test_detr_diagnostics.py') from exc
        legacy = compute_metric_pack
    return legacy, kernels


def standardized_fmax(labels, prob, kernels):
    """Deprecated compatibility helper; not used for primary evaluation.

    Older v0.4.0 tests passed the legacy (validate, thresholds, curve, defs,
    sha) tuple, whereas v0.4.2+ uses named native kernels. This helper now
    safely accepts either; all official Fmax/AuPRC/AP calls instead use
    unified_v085_metrics().
    """
    if isinstance(kernels, dict):
        validate = kernels['validate']
        thresholds = kernels['thresholds']
        curve = kernels['protein_curve']
    else:
        validate, thresholds, curve, _, _ = kernels
    y, p = validate(labels, prob)
    grid = thresholds(.001)
    f1, precision, recall, eligible = curve(y, p, grid)
    best = int(np.argmax(f1))
    return dict(standard_protein_fmax=100 * float(f1[best]),
                standard_protein_fmax_threshold=float(grid[best]),
                standard_protein_precision_at_fmax=100 * float(precision[best]),
                standard_protein_recall_at_fmax=100 * float(recall[best]),
                standard_num_protein_fmax_eligible=int(eligible),
                standard_num_proteins=int(y.shape[0]),
                standard_num_go=int(y.shape[1]),
                standard_num_positive_labels=int(y.sum()),
                standard_protein_fmax_threshold_step=.001)


def load_labels(meta,split,ns,view,width):
    d=meta[split][NS[ns]]
    names=[str(x) for x in d['proteins']]
    if len(names)!=len(set(names)):raise ValueError(f'Duplicate protein IDs in {split}/{ns}')
    key='annotations' if view=='direct' or (view=='auto' and split.endswith('_direct')) else 'prop_annotations'
    ann=d[key]
    if len(ann)!=len(names):raise ValueError(f'{split}/{ns} label count differs from protein count')
    yy=np.zeros((len(names),width),dtype=np.uint8)
    for i,row in enumerate(ann):
        cols=np.asarray(row,dtype=np.int64)
        if len(cols) and (np.min(cols)<0 or np.max(cols)>=width):
            raise IndexError(f'{split}/{ns}/{names[i]} invalid GO index')
        yy[i,cols]=1
    return names,yy,key


def prepare_model_inputs(preds,ids,gos,expected_go):
    if set(preds)!=set(ids) or set(preds)!=set(gos):
        raise ValueError('All models require array, protein IDs, and original ordered GO IDs')
    catalog={}
    for key,p in preds.items():
        mat=ranking.load_prediction(p,len(expected_go))
        order=ranking.load_protein_order(ids[key],mat.shape[0])
        if ranking.read_go_list(gos[key])!=expected_go:
            raise ValueError(f'{key}: GO column order is not the unchanged model output order')
        catalog[key]=(mat,order)
    return catalog


def verify_native_nbs_adapters(specs, predictions, idfiles, gofiles, vocab):
    """Bind an NBS probability to its unchanged native export and phase provenance.

    Optional for Stage-1-only evaluation; strongly recommended for v089.
    It does NOT certify biological temporal independence or absence of training
    identity overlap; that requires the separate actual graph audit.
    """
    result={}
    for model, path in specs.items():
        if model not in predictions:
            raise ValueError(f'Native adapter for {model} has no matching prediction input')
        adapter=json.loads(Path(path).read_text())
        version=adapter.get('runner_version')
        if version not in ('0.8.8','0.8.9.1'):
            raise ValueError(f'{model}: unknown native NBS adapter runner_version {version!r}')
        if Path(adapter['probability']).resolve()!=Path(predictions[model]).resolve():
            raise ValueError(f'{model}: adapter probability path differs from supplied prediction')
        if sha256(predictions[model])!=adapter.get('prediction_sha256'):
            raise ValueError(f'{model}: NBS probability hash differs from adapter')
        if Path(adapter['protein_ids']).read_bytes()!=Path(idfiles[model]).read_bytes():
            raise ValueError(f'{model}: original NBS prediction protein-ID order differs')
        if ranking.read_go_list(gofiles[model])!=vocab:
            raise ValueError(f'{model}: original NBS GO order differs')
        ppath=Path(adapter['native_prediction_manifest'])
        manifest=json.loads(ppath.read_text())
        if manifest.get('runner_version')!=version or manifest.get('version')!=version:
            raise ValueError(f'{model}: native prediction version differs from adapter')
        if manifest.get('output_probability_sha256')!=adapter.get('prediction_sha256'):
            raise ValueError(f'{model}: native output probability digest differs')
        if (manifest.get('model_architecture_version')!='0.8.8'
                or manifest.get('data_architecture_version')!='0.8.8'
                or manifest.get('num_task_go')!=len(vocab)
                or manifest.get('prediction_space')!='complete_task_classifier_columns'):
            raise ValueError(f'{model}: native model/GO contract differs')
        if (manifest.get('output_fusion')!='none'
                or manifest.get('uses_expert_probability_in_nbs_forward') is not False
                or manifest.get('uses_modelout_probability_in_nbs_forward') is not False):
            raise ValueError(f'{model}: NBS forward unexpectedly uses E/M or fusion')
        if manifest.get('checkpoint_sha256')!=adapter.get('checkpoint_sha256'):
            raise ValueError(f'{model}: native checkpoint identity differs')
        cpath=Path(adapter.get('checkpoint_path') or manifest.get('checkpoint_path',''))
        if not cpath.is_file() or sha256(cpath)!=adapter['checkpoint_sha256']:
            raise ValueError(f'{model}: native NBS checkpoint is missing or altered')
        ipath=Path(adapter.get('native_input_manifest',''))
        if not ipath.is_file() or sha256(ipath)!=adapter.get('prepared_input_manifest_sha256'):
            raise ValueError(f'{model}: native input manifest is missing or altered')
        if manifest.get('input_manifest_sha256')!=adapter.get('prepared_input_manifest_sha256'):
            raise ValueError(f'{model}: native prepared-input SHA differs')
        if version=='0.8.9.1':
            # The original v089 validator carries the phase/lifetime semantics;
            # do not reconstruct cumulative epochs from phase-local steps.
            from scripts.nbs.eval_nbs_full_task_v089 import _phase_identity
            calculated=_phase_identity(manifest)
            if calculated!=adapter.get('phase_identity'):
                raise ValueError(f'{model}: v089 phase identity differs from audited adapter')
        result[model]={'runner_version':version,'model_architecture_version':'0.8.8',
                       'native_prediction_manifest':str(ppath.resolve()),
                       'native_prediction_manifest_sha256':sha256(ppath),
                       'checkpoint_sha256':adapter['checkpoint_sha256'],
                       'checkpoint_path':str(cpath.resolve()),
                       'input_manifest_sha256':adapter['prepared_input_manifest_sha256'],
                       'phase_identity':adapter.get('phase_identity'),
                       'graph_audit':adapter.get('graph_audit'),
                       'caution':'Provenance verified; 2022 seen-weak overlap remains transductive without graph/sequence audit.'}
    return result


def run(args):
    predictions=ranking.parse_kv(args.prediction,'--prediction')
    idfiles=ranking.parse_kv(args.prediction_ids,'--prediction-ids')
    gofiles=ranking.parse_kv(args.prediction_go_ids,'--prediction-go-ids')
    vocab=ranking.read_go_list(args.unified_root/'labels'/args.namespace/'go_ids.txt')
    models=prepare_model_inputs(predictions,idfiles,gofiles,vocab)
    native_specs=ranking.parse_kv(getattr(args,'native_nbs_adapter',[]) or [], '--native-nbs-adapter')
    native_verification=verify_native_nbs_adapters(native_specs,predictions,idfiles,gofiles,vocab)
    source_dtypes = {name: str(arr.dtype) for name, (arr, _) in models.items()}
    mixed_dtype = len(set(source_dtypes.values())) > 1
    if mixed_dtype:
        message = ('Model prediction matrices have mixed storage dtypes '
                   f'{source_dtypes}. Casting fp16 to fp32 does NOT undo ties; '
                   'v085 PR-AUC/AP use exact grouped-score ranking. '
                   'Prefer re-exporting Stage-1 float32 to match v088 float32.')
        if getattr(args, 'strict_prediction_dtype', False):
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning)
    if args.reference not in models:raise ValueError(f'Reference model {args.reference!r} not in predictions')
    with args.metadata.open('rb') as f:meta=pickle.load(f)
    cases=([args.split] if args.split else
           (PANELS['future']+PANELS['history'] if args.panel=='all' else PANELS[args.panel]))
    # Preserve --label-view for old callers; new primary metric defaults to
    # all directly-supported GO + historical ancestors (prop_annotations).
    compat=getattr(args,'label_view',None)
    metric_view=getattr(args,'metric_label_view',None) or ('propagated' if compat in (None,'auto') else compat)
    rank_view=getattr(args,'ranking_label_view',None) or ('direct' if compat in (None,'auto') else compat)
    if metric_view not in ('direct','propagated') or rank_view not in ('direct','propagated'):
        raise ValueError('Both metric and ranking views must be direct or propagated')
    scope={'both':'unified_plus_legacy_micro','v088':'unified','stage1':'stage1_legacy_micro'}.get(
        args.metric_scope,args.metric_scope)
    valid_scopes={'unified','unified_plus_legacy_micro','stage1_legacy_micro','ranking_only'}
    if scope not in valid_scopes:raise ValueError(f'Invalid metric_scope: {scope}')
    use_standard=scope in ('unified','unified_plus_legacy_micro')
    use_legacy=scope in ('unified_plus_legacy_micro','stage1_legacy_micro')
    if scope!='ranking_only':
        stage1_kernel, std_kernel=imports_for_metrics(include_stage1=use_legacy)
        implementations={'v085_native_standardized':native_protocol(std_kernel)}
        if use_legacy:
            implementations['stage1_micro_diagnostics']={
                'implementation':'experiments.eval_weak_ind_test_detr_diagnostics.compute_metric_pack',
                'threshold_step':args.stage1_threshold_step,
                'legacy_ap_mode':args.auprc_mode,
                'warning':'Micro-Fmax and micro average precision are not the primary metrics',
            }
    else:
        stage1_kernel=std_kernel=None
        implementations={}
    reports=[]
    out=args.out
    for split in cases:
        print(f'\n=== {args.namespace} {split} ===',flush=True)
        folder=out/args.namespace/split
        ranking_args=argparse.Namespace(metadata=args.metadata,unified_root=args.unified_root,
            split=split,namespace=args.namespace,
            prediction=args.prediction,prediction_ids=args.prediction_ids,
            prediction_go_ids=args.prediction_go_ids,
            assume_go_order=False,reference=args.reference,ks=args.ks,
            bootstrap=args.bootstrap,seed=args.seed,out=folder/'ranking',
            save_pair_ranks=args.save_pair_ranks,
            allow_descriptive_labels=args.allow_descriptive_labels,
            exclude_roots=not args.include_roots,label_view=rank_view)
        ranking_result=ranking.run(ranking_args)
        names,labels,field=load_labels(meta,split,args.namespace,metric_view,len(vocab))
        if len(names)==0 or labels.sum()==0:
            raise ValueError(f'No observed positive labels in {split}/{args.namespace}')
        suite={'split':split,'namespace':args.namespace,'metric_label_field':field,
               'ranking_label_field':'annotations' if rank_view=='direct' else 'prop_annotations',
               'num_rows':len(names),'num_positive_proteins':int((labels.sum(axis=1)>0).sum()),
               'num_positive_model_columns':int(labels.sum()),
               'ranking':ranking_result['model_metrics'],
               'closed_world_proxy_warning':('Unannotated GO positions are UNKNOWN, not established negatives. '
                   'Standard Fmax/PR-AUC are proxy scores under sparse observed annotation.'),
               'metrics':{}}
        for model,(array,idmap) in models.items():
            missing=[name for name in names if name not in idmap]
            if missing:raise ValueError(f'{model}: missing {len(missing)} evaluation proteins, first {missing[:3]}')
            if scope=='ranking_only':continue
            if labels.size>args.max_metric_cells and not args.allow_large_metrics:
                raise MemoryError(f'Metric scoring needs {labels.size:,} [protein, GO] cells. '
                    f'Use --allow-large-metrics to explicitly accept the memory cost, '
                    f'or --metric-scope ranking_only.')
            # All predictors share identical target row/GO order; no GO alias
            # deduplication, no probability postprocessing, no test labels in forward.
            p=np.asarray(array[[idmap[name] for name in names]],dtype=np.float32)
            if not np.isfinite(p).all() or np.min(p)<0 or np.max(p)>1:
                raise ValueError(f'{model}: standardized metrics require finite probabilities in [0,1]')
            result={}
            if use_standard:
                result['v085_standardized']=unified_v085_metrics(
                    labels,p,std_kernel,verify_fmax=not getattr(args,'skip_fmax_parity',False))
            if use_legacy:
                result['stage1_diagnostic_micro']=stage1_kernel(labels,p,
                    threshold_step=args.stage1_threshold_step,auprc_mode=args.auprc_mode,
                    compute_sample_fmax=False,no_empty_labels=False,
                    no_zero_classes=False,metrics_are_percent=True)
            suite['metrics'][model]=result
            del p
        if scope!='ranking_only':
            for model in suite['metrics']:
                if model==args.reference:continue
                suite['metrics'][model]['delta_vs_reference']={}
                current=suite['metrics'][model]
                baseline=suite['metrics'][args.reference]
                for met,fields in (
                    ('v085_standardized',['Fmax_percent','AuPRC_percent','AP_percent']),
                    ('stage1_diagnostic_micro',['fmax_micro','auprc_micro_exact','auprc_micro_hist'])
                ):
                    if met not in current or met not in baseline:continue
                    suite['metrics'][model]['delta_vs_reference'][met]={
                        key:float(current[met][key])-float(baseline[met][key])
                        for key in fields if key in current[met] and key in baseline[met]}
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'complete_metrics.json').write_text(json.dumps(suite,indent=2,ensure_ascii=False)+'\n')
        for key,data in suite['metrics'].items():
            entry={'split':split,'namespace':args.namespace,'model':key,
                'num_rows':suite['num_rows'],'metric_label_field':field,
                'ranking_label_field':suite['ranking_label_field']}
            native=data.get('v085_standardized')
            if native:
                entry['Fmax_percent']=native['Fmax_percent']
                entry['AuPRC_percent']=native['AuPRC_percent']
                entry['AP_percent']=native['AP_percent']
                entry['observed_positive_prevalence_percent']=native['observed_positive_prevalence_percent']
                entry['AuPRC_native_key']=native['AuPRC_native_key']
                entry['AP_native_key']=native['AP_native_key']
                entry['source_probability_dtype']=source_dtypes[key]
                entry['Fmax_threshold']=native.get('standard_protein_fmax_threshold',float('nan'))
                for k,v in native.items():
                    if isinstance(v,(int,float)) and not isinstance(v,bool):
                        entry['v085.'+k]=v
            diag=data.get('stage1_diagnostic_micro')
            if diag:
                entry.update({'stage1_diagnostic.'+k:v for k,v in diag.items()
                              if isinstance(v,(int,float)) and not isinstance(v,bool)})
            reports.append(entry)
    out.mkdir(parents=True,exist_ok=True)
    if reports:
        pd.DataFrame(reports).to_csv(out/f'{args.namespace}_complete_metrics.tsv',sep='\t',index=False)
    summary={'panel':args.panel,'explicit_split':args.split,'namespace':args.namespace,
             'metric_label_view':metric_view,'ranking_label_view':rank_view,
             'metric_scope':scope,'cases':cases,
             'model_probability_storage_dtypes':source_dtypes,
             'mixed_model_storage_dtypes':mixed_dtype,
             'reference':args.reference,'native_kernels':implementations,
             'native_nbs_provenance':native_verification,
             'metadata':str(args.metadata.resolve()),'metadata_sha256':sha256(args.metadata),
             'vocabulary_sha256':sha256(args.unified_root/'labels'/args.namespace/'go_ids.txt'),
             'input_files':{name:{'probability':str(path.resolve()),'probability_sha256':sha256(path),
                 'protein_ids':str(idfiles[name].resolve()),'go_ids':str(gofiles[name].resolve())}
                for name,path in predictions.items()},
             'protocol':{'checkpoint_selection':'frozen_preselected','future_labels_in_forward':False,
                         'ranking':'canonical-positive-only; default direct annotations',
                         'primary_metrics':'v085 protein-centric Fmax + trapezoidal micro PR-AUC; 0.001 threshold',
                         'required_diagnostic':'non-interpolated v085 micro-AP and observed label prevalence',
                         'pr_auc_warning':'all-tied scores give (1+prevalence)/2 trapezoidal PR-AUC but AP=prevalence',
                         'primary_gold':'prop_annotations full ancestor closure by default',
                         'optional_micro':'Stage-1 micro Fmax and AP (not comparable to standardized)',
                         'model_fusion':'none','GO_alias':'raw original classifier columns for Fmax/AuPRC'}}
    (out/f'{args.namespace}_complete_manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(f'[Unified evaluation] {out}')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--namespace',choices=NS,required=True)
    p.add_argument('--metadata',type=Path,default=Path('data/unidata_with_dark_weak_eval.pkl'))
    p.add_argument('--unified-root',type=Path,default=Path('data/dark_weak_test/unified'))
    p.add_argument('--panel',choices=('future','history','all'),default='future')
    p.add_argument('--split',default=None,help='Optional single split, overrides panel')
    p.add_argument('--label-view',choices=('auto','direct','propagated'),default='auto',help='Deprecated compatibility override for both views')
    p.add_argument('--metric-label-view',choices=('direct','propagated'),default=None,help='Primary Fmax/AuPRC gold; default propagated')
    p.add_argument('--ranking-label-view',choices=('direct','propagated'),default=None,help='Positive-only ranking gold; default direct')
    p.add_argument('--prediction',action='append',required=True)
    p.add_argument('--prediction-ids',action='append',required=True)
    p.add_argument('--prediction-go-ids',action='append',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--native-nbs-adapter',action='append',default=[],
                   help='Optional NBS export provenance: MODEL=/path/to/nbs_v089_adapter.json; '
                        'verify checkpoint, input and phase lineage before scoring')
    p.add_argument('--ks',nargs='+',type=int,default=[10,50,100])
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=6061)
    p.add_argument('--save-pair-ranks',action='store_true')
    p.add_argument('--allow-descriptive-labels',action='store_true')
    p.add_argument('--include-roots',action='store_true')
    p.add_argument('--metric-scope',choices=('unified','unified_plus_legacy_micro','stage1_legacy_micro','ranking_only','both','stage1','v088'),default='unified')
    p.add_argument('--skip-fmax-parity',action='store_true',help='Skip independent original v088 Fmax cross-check to reduce compute time')
    p.add_argument('--strict-prediction-dtype',action='store_true',
                   help='Refuse mixed prediction storage dtypes (fp16 and fp32 alter score ties in AP/PR-AUC)')
    p.add_argument('--auprc-mode',choices=('exact','hist','none'),default='exact',help='Optional legacy Stage-1 AP mode only; unified v085 PR-AUC is always native')
    p.add_argument('--stage1-threshold-step',type=float,default=.01)
    p.add_argument('--max-metric-cells',type=int,default=80000000)
    p.add_argument('--allow-large-metrics',action='store_true')
    p.add_argument('--out',type=Path,default=Path('data/dark_weak_test/evaluations/complete'))
    args=p.parse_args(argv)
    if args.bootstrap<1 or not args.ks or min(args.ks)<1:p.error('bootstrap and K must be positive')
    # The original Stage1 and NBS kernels live in different project directories.
    nbs_path=PROJECT/'nbs_models'/'nbs_protein_go'
    sys.path.insert(0,str(nbs_path))
    run(args)

if __name__=='__main__':main()
