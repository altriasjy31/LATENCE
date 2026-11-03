#!/usr/bin/env python3
"""Auditably combine native v085 and both historical Stage-1 metric paths.

Only metric function ASTs are loaded from Stage-1 sources: model/inference
imports are deliberately never executed. Original function bodies are unchanged.
All scores use percent, except probability thresholds and optional IC Smin.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

LEGACY_FUNCTIONS = ('fmax_from_histograms', 'exact_micro_auprc',
                    'filter_metric_arrays', 'compute_metric_pack')
NATIVE_NUMERIC_FIELDS = (
    'standard_num_proteins', 'standard_num_go', 'standard_num_positive_labels',
    'standard_num_protein_fmax_eligible', 'standard_num_empty_gold_proteins',
    'standard_protein_fmax', 'standard_protein_fmax_threshold',
    'standard_protein_precision_at_fmax', 'standard_protein_recall_at_fmax',
    'standard_protein_fmax_threshold_step', 'standard_protein_fmax_grid_0p01',
    'standard_protein_fmax_grid_0p01_threshold', 'standard_micro_ap',
    'standard_micro_pr_auc', 'standard_micro_fmax_exact',
    'standard_micro_fmax_exact_threshold',
    'standard_micro_fmax_exact_min_included_score',
)
LEGACY_FIELDS = ('fmax_micro', 'threshold_micro', 'precision_micro',
                 'recall_micro', 'auprc_micro_hist', 'auprc_micro_exact')
SAMPLE_FIELDS = ('fmax_sample', 'threshold_sample', 'precision_sample', 'recall_sample')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _function_namespace(path, names, globals_):
    """Compile selected original function nodes with deferred annotations."""
    path = Path(path)
    source = path.read_text(encoding='utf-8')
    parsed = ast.parse(source, filename=str(path))
    selected = {n.name: n for n in parsed.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names}
    missing = set(names) - selected.keys()
    if missing:
        raise RuntimeError(f'{path}: required original metric functions missing: {sorted(missing)}')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.Module(body=[future] + [selected[n] for n in names], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(globals_)
    exec(compile(module, str(path), 'exec'), namespace)
    metadata = dict(path=str(path.resolve()), sha256=_sha(path),
                    function_sha256={n: hashlib.sha256(
                        ast.get_source_segment(source, selected[n]).encode('utf-8')).hexdigest()
                        for n in names}, extraction='original top-level function ASTs unchanged')
    return namespace, metadata


class OriginalOfficial:
    """Lazy torch-backed original helper; allows native-only use without torch."""
    def __init__(self, function, namespace):
        self.function = function
        self.namespace = namespace

    def __call__(self, y, p, *, ic=None):
        try:
            import torch
        except ImportError as exc:
            raise ImportError('Original Stage-1 helper requires torch. Use '
                              '--skip-stage1-official for a native/histogram-only run.') from exc
        self.namespace.update(torch=torch, th=torch, math=math)
        yy = torch.from_numpy(np.array(y, dtype=np.float32, copy=True))
        pp = torch.from_numpy(np.array(p, dtype=np.float32, copy=True))
        ii = None if ic is None else torch.from_numpy(np.array(ic, dtype=np.float32, copy=True))
        return self.function(yy, pp, threshold=True, auprc=True,
                             smin=ii is not None, icary=ii,
                             no_empty_labels=False, no_zero_classes=False)


def load_kernels(project_root, *, official=True, official_source=None):
    """Return native/legacy/official kernels and audit metadata for one project."""
    project = Path(project_root).resolve()
    adapter = project / 'scripts/dark_weak/native_v085_metrics.py'
    if not adapter.is_file():
        raise FileNotFoundError(f'Existing native v085 adapter required: {adapter}')
    name = '_latence_native_adapter_v054_' + hashlib.sha256(str(adapter).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Import the exact metric file directly. Importing nbs_pg's package __init__
    # would otherwise pull graph/torch training dependencies into CPU evaluation.
    native_path = project/'nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py'
    if not native_path.is_file():
        raise FileNotFoundError(f'Existing unmodified native metric source required: {native_path}')
    native_spec = importlib.util.spec_from_file_location(name+'_kernel', native_path)
    native_module = importlib.util.module_from_spec(native_spec)
    native_spec.loader.exec_module(native_module)
    native = dict(compute=native_module.compute_standard_metrics,
                  definitions=native_module.metric_definitions,
                  sha=native_module.metrics_implementation_sha256,
                  validate=native_module._validate, thresholds=native_module._thresholds,
                  protein_curve=native_module._protein_curve)
    module.validate_metric_contract(native)
    native['unified_v085_metrics'] = module.unified_v085_metrics
    native['native_protocol'] = module.native_protocol
    legacy_path = project / 'experiments/eval_weak_ind_test_detr_diagnostics.py'
    namespace, legacy_meta = _function_namespace(legacy_path, LEGACY_FUNCTIONS,
                                                dict(np=np, Dict=Dict, Optional=Optional, Tuple=Tuple))
    result = dict(native=native, legacy=namespace['compute_metric_pack'], official=None,
                  sources=dict(stage1_hist=legacy_meta,
                               native_adapter=dict(path=str(adapter), sha256=_sha(adapter)),
                               native_kernel=dict(path=str(native_path), sha256=_sha(native_path))))
    if official:
        if official_source is None:
            candidates = [project/'helper_functions/helper.py', project/'helper.py',
                          Path(__file__).with_name('stage1_evalperf_original_v054.py')]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError('Stage-1 original evalperf_torch source missing')
        else:
            path = Path(official_source).resolve()
        official_ns, meta = _function_namespace(path, ('evalperf_torch',), dict(math=math))
        result['official'] = OriginalOfficial(official_ns['evalperf_torch'], official_ns)
        result['sources']['stage1_official'] = meta
    return result


def _validate_arrays(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 2 or y.shape != p.shape:
        raise ValueError('labels and probabilities must be equally shaped 2-D arrays')
    if y.dtype.kind not in 'biuf' or p.dtype.kind not in 'biuf':
        raise ValueError('labels and probabilities must be real numeric arrays')
    if np.any((y != 0) & (y != 1)):
        raise ValueError('gold labels must be binary 0/1')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('probabilities must be finite and lie in [0,1]')
    return y, p


def _diagnostics(y, p):
    positives = p[y != 0]
    return dict(score_zero_fraction=float(np.mean(p == 0)) if p.size else None,
                score_one_fraction=float(np.mean(p == 1)) if p.size else None,
                score_all_scores_equal=bool(np.all(p == p.flat[0])) if p.size else None,
                score_per_row_all_equal_fraction=float(np.mean(np.all(p == p[:, :1], axis=1)))
                    if p.shape[0] and p.shape[1] else None,
                score_positive_zero_fraction=float(np.mean(positives == 0)) if positives.size else None,
                score_min=float(np.min(p)) if p.size else None,
                score_max=float(np.max(p)) if p.size else None)


def _empty_native(y, p):
    """No zero-width cohort can be scored; stable unavailable fields, not zeros."""
    out = {name: None for name in NATIVE_NUMERIC_FIELDS}
    out.update(standard_num_proteins=int(y.shape[0]), standard_num_go=int(y.shape[1]),
               standard_num_positive_labels=int(y.sum()),
               standard_num_protein_fmax_eligible=int(np.count_nonzero(y.sum(axis=1))),
               standard_num_empty_gold_proteins=int(np.count_nonzero(y.sum(axis=1) == 0)),
               Fmax_percent=None, AuPRC_percent=None, AP_percent=None,
               observed_positive_prevalence_percent=None,
               AuPRC_native_key='standard_micro_pr_auc', AP_native_key='standard_micro_ap',
               evaluation_backend='v085_native_standardized', standard_threshold_step_used=.001)
    return out


def comprehensive_metrics(y, p, *, native, legacy, official=None,
                          threshold_step=.01, compute_sample_fmax=True,
                          no_empty_labels=False, no_zero_classes=False,
                          verify_fmax=True, ic=None):
    """Preserve all native outputs and add explicitly prefixed Stage-1 outputs.

    Optional Stage-1 row/column filters affect both historical paths only;
    v085 always retains the complete supplied arrays. IC must be an explicit
    historical GO vector aligned with input columns; it is never estimated here.
    """
    y, p = _validate_arrays(y, p)
    step = float(threshold_step)
    if (not np.isfinite(step) or not 0 < step <= 1
            or not np.isclose(round(1/step) * step, 1, rtol=0, atol=1e-12)):
        raise ValueError('Stage-1 threshold_step must divide 1 exactly')
    if ic is not None:
        ic = np.asarray(ic, dtype=np.float32)
        if ic.ndim != 1 or len(ic) != y.shape[1] or not np.isfinite(ic).all() or np.any(ic < 0):
            raise ValueError('IC must be a finite nonnegative vector aligned to supplied GO columns')
    if y.size:
        out = native['unified_v085_metrics'](y, p, native, verify_fmax=verify_fmax)
    else:
        out = _empty_native(y, p)
    out.update(_diagnostics(y, p))
    out.update(metric_has_positive_labels=bool(y.sum()),
               metric_empty_axis=not bool(y.size),
               stage1_hist__threshold_step=step)
    fields = LEGACY_FIELDS + (SAMPLE_FIELDS if compute_sample_fmax else ())
    if y.size:
        yy, pp = y, p
        if no_empty_labels:
            rows = yy.sum(axis=1) > 0
            yy, pp = yy[rows], pp[rows]
        columns = np.ones(y.shape[1], dtype=bool)
        if no_zero_classes:
            columns = yy.sum(axis=0) > 0
            yy, pp = yy[:, columns], pp[:, columns]
    else:
        yy, pp = y, p
        columns = np.ones(y.shape[1], dtype=bool)
    out.update(stage1_hist__num_proteins=int(yy.shape[0]), stage1_hist__num_go=int(yy.shape[1]),
               stage1_hist__num_positives=int(yy.sum()))
    if yy.size:
        hist = legacy(yy, pp, threshold_step=step, auprc_mode='exact',
                      compute_sample_fmax=compute_sample_fmax,
                      no_empty_labels=False, no_zero_classes=False, metrics_are_percent=True)
        # Exact AP is optional in original source, but here its absence is explicit.
        for name in fields:
            out['stage1_hist__' + name] = hist.get(name)
    else:
        out.update({'stage1_hist__' + name: None for name in fields})
    if official is not None:
        official_fields = ('fmax', 'threshold', 'auprc') + (('smin',) if ic is not None else ())
        if yy.size:
            values = official(yy, pp, ic=None if ic is None else ic[columns])
            out.update({'stage1_official__'+name: values.get(name) for name in official_fields})
        else:
            out.update({'stage1_official__'+name: None for name in official_fields})
    return out


def metric_protocol(kernels, *, threshold_step=.01, compute_sample_fmax=True,
                    no_empty_labels=False, no_zero_classes=False):
    """Serializable exact provenance and definitions; score fields remain explicit."""
    return dict(schema='latence-comprehensive-metrics-v0.5.4', sources=kernels['sources'],
                native=kernels['native']['native_protocol'](kernels['native']),
                stage1_hist=dict(enabled=True, prefix='stage1_hist__',
                    threshold_step=float(threshold_step), probability_dtype='float32',
                    implementation='original compute_metric_pack + fmax_from_histograms ASTs',
                    threshold_rule='floor(float32 probability / step), reverse cumulative histogram bins',
                    threshold_endpoint='bin zero includes zero-probability predictions',
                    fmax='pooled micro F1 on histogram grid; earliest maximizing bin',
                    auprc_micro_hist='histogram approximate non-interpolated micro AP (not trapezoidal AUC)',
                    auprc_micro_exact='sklearn average_precision_score; optional availability explicit',
                    compute_sample_fmax=bool(compute_sample_fmax),
                    fmax_sample='mean precision over predicted rows; mean recall over positive-gold rows',
                    no_empty_labels=bool(no_empty_labels), no_zero_classes=bool(no_zero_classes)),
                stage1_official=dict(enabled=kernels['official'] is not None,
                    prefix='stage1_official__', threshold_step=.01,
                    implementation='unchanged original evalperf_torch AST on CPU float32 torch tensors',
                    threshold_rule='strict score > t/100; t=0..100',
                    auprc='threshold-grid micro PR points sorted by recall then torch.trapz; not histogram AP',
                    smin='only with explicit nonnegative historical IC vector; same original helper; unscaled IC units',
                    no_empty_labels=bool(no_empty_labels), no_zero_classes=bool(no_zero_classes)),
                units=('scores in percent; grid thresholds [0,1]; native exact-micro threshold '
                       'may be the representable value below zero; fractions [0,1]; optional Smin in IC units'),
                no_positive_labels='native standardized scores zero; has-positive flag false; original legacy conventions retained',
                empty_axis='score values unavailable (None); counts and empty-axis flag retained',
                score_diagnostics='all supplied rows and columns, before optional Stage-1 filters')
