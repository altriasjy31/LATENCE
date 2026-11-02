#!/usr/bin/env python3
"""Fit exploratory frequency-group B thresholds on Stage1-seen training core.

This is explicitly in-sample calibration, not independent validation. No weak
gold, held-out label targets, or independent-test gold enters threshold fitting.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT/'nbs_models/nbs_protein_go/nbs_pg/unified_weak_dataset_v093.py'
spec = importlib.util.spec_from_file_location('_v093_backbone_calibration_dataset', MODULE)
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)
SCHEMA = 'nbs-backbone-group-thresholds-v093-r1'


def threshold_grid():
    return np.arange(1001, dtype=np.float64)/1000.


def histogram_counts(probability, positive, thresholds=None):
    """Accumulate scores into bins preserving strict probability > threshold.

    searchsorted(side=left) means a value equal to a grid threshold does not
    contribute to that threshold. Both scores and the grid use float64; scores
    merely retain their original stored f16/f32 value on conversion.
    """
    thresholds = threshold_grid() if thresholds is None else np.asarray(thresholds, np.float64)
    p, y = np.asarray(probability, np.float64), np.asarray(positive, bool)
    if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Calibration probabilities and binary gold must align and be finite in [0,1]')
    bins = np.searchsorted(thresholds, p.ravel(), side='left')
    histogram = np.bincount(bins, minlength=len(thresholds)+1).astype(np.int64)
    positive_histogram = np.bincount(bins[y.ravel()], minlength=len(thresholds)+1).astype(np.int64)
    return histogram, positive_histogram


def curve_from_histograms(histogram, positive_histogram):
    histogram, positive_histogram = np.asarray(histogram, np.int64), np.asarray(positive_histogram, np.int64)
    if histogram.ndim != 1 or histogram.shape != positive_histogram.shape or len(histogram) < 2:
        raise ValueError('Matching calibration histograms are required')
    predicted = np.cumsum(histogram[::-1], dtype=np.int64)[::-1][1:]
    tp = np.cumsum(positive_histogram[::-1], dtype=np.int64)[::-1][1:]
    gold = int(positive_histogram.sum())
    precision = np.divide(tp, predicted, out=np.zeros(len(tp), np.float64), where=predicted > 0)
    recall = tp/gold if gold else np.zeros(len(tp), np.float64)
    f1 = np.divide(2.*tp, predicted+gold, out=np.zeros(len(tp), np.float64), where=(predicted+gold) > 0)
    return {'predicted': predicted, 'tp': tp, 'gold_positive_pairs': gold,
            'precision': precision, 'recall': recall, 'f1': f1}


def _atomic_json(path, value):
    path=Path(path)
    descriptor, name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def calibrate(args, project_root=ROOT):
    started=time.monotonic()
    if args.low_frequency_max not in (0,5,9,99,999):
        raise ValueError('Calibration low-frequency-max must be a group boundary: 0,5,9,99,999')
    root=Path(project_root).resolve()
    work=dataset.resolve(root,args.source_work_dir)
    config_path=work/'resolved_config.json'
    config=dataset.read_json(config_path)
    if config['full_task'].get('label_split','all_seen') not in (None,'all_seen'):
        raise ValueError('Calibration starts from the original all_seen source configuration')
    data_root=dataset.resolve(root,config['data']['root'])
    weak_path=dataset.resolve(data_root,config['data']['weak_graph_predictions_manifest'])
    inverted_path=dataset.resolve(data_root,config['data']['go_protein_inverted_index_manifest'])
    weak,inverted=dataset.read_json(weak_path),dataset.read_json(inverted_path)
    if weak.get('task') != config.get('task'):
        raise ValueError('Calibration source task differs from configuration')
    registry_path=data_root/'features/protein_registry.csv'
    records,roles,role_rows,role_ids=dataset.load_registry(registry_path)
    expected_registry_hash=weak.get('protein_registry',{}).get('sha256')
    if expected_registry_hash is not None and dataset.sha256(registry_path)!=expected_registry_hash:
        raise ValueError('Source exporter registry hash differs from current registry')
    num_go=int(weak['go_registry']['num_terms'])
    ptr,go,ptr_path,go_path=dataset.read_gold(inverted['indices']['gold']['protein_major'],
                                           inverted_path.parent,len(roles),num_go)
    training,validation=dataset.training_core_split(role_ids['core'],ptr,config['full_task'])
    counts=dataset.count_gold(ptr,go,training,num_go)
    mapping,go_registry,go_sources=dataset._load_go_mapping(root,config,weak,weak_path,num_go)
    seen,heldout=dataset.make_seen_mask(counts,mapping,args.label_holdout_fraction,args.seed)
    groups=dataset.frequency_groups(counts)
    record=next((r for r in weak.get('roles',[]) if r.get('role')=='core'),None)
    if not record or not record.get('backbone_dense_file'):
        raise FileNotFoundError('Source export has no saved core backbone probabilities')
    probability_path=dataset.resolve(weak_path.parent,record['backbone_dense_file'])
    probability=np.load(probability_path,mmap_mode='r',allow_pickle=False)
    if probability.shape != (len(role_ids['core']),num_go) or probability.dtype.kind!='f':
        raise ValueError('Core backbone matrix must preserve role-local rows and task columns')
    if record.get('rows',len(role_ids['core'])) != len(role_ids['core']):
        raise ValueError('Exporter core row count differs from protein registry')
    if records[0].get('protein_id') is not None and record.get('protein_ids_sha256'):
        import hashlib
        digest=hashlib.sha256()
        for protein in role_ids['core']:
            digest.update(records[int(protein)]['protein_id'].encode());digest.update(b'\n')
        if digest.hexdigest()!=record['protein_ids_sha256']:
            raise ValueError('Core exporter protein-ID order differs from registry')
    source_paths={'source_config':config_path,'weak_manifest':weak_path,'inverted_manifest':inverted_path,
                  'protein_registry':registry_path,'gold_indptr':ptr_path,'gold_go_idx':go_path,
                  'core_backbone_probability':probability_path,**go_sources}
    print('[calibration] hashing training inputs',flush=True)
    identity={'task':config['task'],'sources':{name:dataset.file_identity(path) for name,path in source_paths.items()},
              'training_core_ids_sha256':dataset.array_sha256(training),
              'validation_core_ids_sha256':dataset.array_sha256(validation),
              'core_counts_sha256':dataset.array_sha256(counts),
              'seen_task_mask_content_sha256_int64':dataset.array_sha256(seen),
              'heldout_ontology_rows_sha256':dataset.array_sha256(heldout),
              'label_holdout_fraction':args.label_holdout_fraction,'label_holdout_seed':args.seed,
              'low_frequency_max':args.low_frequency_max,
              'criterion':'pooled_micro_f1','threshold_step':.001,'comparison':'>',
              'threshold_dtype':'float64','calibrator_sha256':dataset.sha256(__file__),
              'dataset_module_sha256':dataset.sha256(MODULE)}
    output=dataset.resolve(root,args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_name(output.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists():
            existing=dataset.read_json(output)
            if existing.get('schema')!=SCHEMA or existing.get('complete') is not True or existing.get('identity')!=identity:
                raise ValueError('Existing calibration has a different source/split identity; use a new --output')
            print('[calibration] identical complete threshold policy reused: '+str(output),flush=True)
            return existing
        grid=threshold_grid()
        hist=np.zeros((len(dataset.GROUPS),len(grid)+1),np.int64)
        positive_hist=np.zeros_like(hist)
        masks=[(groups==i)&seen for i in range(len(dataset.GROUPS))]
        for start in range(0,len(training),args.chunk_rows):
            ids=training[start:start+args.chunk_rows]
            p=np.asarray(probability[role_rows[ids]],np.float64)
            dataset._validate_prob_block(p,(len(ids),num_go))
            y=np.zeros((len(ids),num_go),bool)
            for row,protein in enumerate(ids):
                labels=np.asarray(go[int(ptr[protein]):int(ptr[protein+1])],np.int64)
                y[row,labels]=True
            for i,mask in enumerate(masks):
                if mask.any():
                    h,hp=histogram_counts(p[:,mask],y[:,mask],grid)
                    hist[i]+=h;positive_hist[i]+=hp
            stop=start+len(ids)
            if start==0 or stop==len(training) or start//4096 != stop//4096:
                print(f'[calibration] {stop}/{len(training)} training-core proteins',flush=True)
        thresholds={name:.5 for name,_,_ in dataset.GROUPS}
        report=[]
        for i,(name,lower,upper) in enumerate(dataset.GROUPS):
            curve=curve_from_histograms(hist[i],positive_hist[i])
            backbone_selected=bool(i in (1,2,3,4) and upper <= args.low_frequency_max)
            fit=bool(backbone_selected and curve['gold_positive_pairs']>0)
            best=int(np.argmax(curve['f1'])) if fit else 500
            if fit:thresholds[name]=float(grid[best])
            report.append({'frequency_bin':name,'core_count_min':lower,'core_count_max':upper,
                           'all_task_columns':int((groups==i).sum()),'seen_fitted_columns':int(masks[i].sum()),
                           'threshold':float(grid[best]),'threshold_fitted':fit,
                           'status':'fitted_in_sample' if fit else ('fixed_modelout_threshold' if not backbone_selected and i>0 else 'no_seen_gold_positive_pairs'),
                           'pooled_gold_positive_pairs':curve['gold_positive_pairs'],
                           'backbone_metrics_at_reported_threshold':{
                               'true_positive_pairs':int(curve['tp'][best]),'predicted_pairs':int(curve['predicted'][best]),
                               'precision':float(curve['precision'][best]),'recall':float(curve['recall'][best]),
                               'micro_f1':float(curve['f1'][best])}})
        result={'schema':SCHEMA,'complete':True,'task':config['task'],'thresholds':thresholds,
                'metric_unit':'fraction; thresholds use probability units',
                'independently_calibrated':False,'identity':identity,
                'provenance':{'calibration_scope':'Stage1_seen_training_core_in_sample',
                    'criterion':'frequency_group_pooled_micro_f1','comparison':'>','threshold_dtype':'float64',
                    'threshold_grid':{'min':0.,'max':1.,'step':.001,'count':1001},
                    'tie_breaking':'lowest threshold attaining maximum grid F1',
                    'num_training_core_proteins':len(training),'num_excluded_validation_core_proteins':len(validation),
                    'num_seen_task_columns':int(seen.sum()),'num_excluded_task_columns':int((~seen).sum()),
                    'label_holdout_fraction':args.label_holdout_fraction,'label_holdout_seed':args.seed,
                    'low_frequency_max':args.low_frequency_max,
                    'independent_test_labels_used':False,'heldout_label_targets_used_for_threshold_fitting':False,
                    'upstream_stage1_has_seen_core':True,'strict_pipeline_zero_shot':False,
                    'modelout_source_group_thresholds':'fixed 0.5; no modelout core calibration performed',
                    'claim':'Exploratory in-sample training thresholds; not generalization-optimal or independently calibrated.'},
                'groups':report}
        _atomic_json(output,result)
        print(f'[calibration complete] {output} ({time.monotonic()-started:.1f}s)',flush=True)
        return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-work-dir',type=Path,default=ROOT/'outputs/latence_nbs_experiments/v092/geometry_single12800_scratch')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/latence_nbs_experiments/v093/backbone_core_thresholds.json')
    p.add_argument('--label-holdout-fraction',type=float,default=0.)
    p.add_argument('--low-frequency-max',type=int,default=999)
    p.add_argument('--seed',type=int,default=1337)
    p.add_argument('--chunk-rows',type=int,default=256)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.chunk_rows<1 or args.seed<0:
        raise SystemExit('chunk-rows must be positive and seed nonnegative')
    calibrate(args)


if __name__=='__main__':main()
