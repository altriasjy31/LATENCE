#!/usr/bin/env python3
"""Verified, unchanged v085 Fmax / trapezoidal PR-AUC / AP metric adapter.

Uses the original project nbs_pg.full_task_metrics_v085; never copies or
monkeypatches its metric implementation. Pin the exact schema and key names
now that the original source has been inspected. All reported scores are %.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

REQUIRED_SCHEMA = '0.8.5-standard-metrics-v1'
REQUIRED_FIELDS = (
    'standard_protein_fmax', 'standard_protein_fmax_threshold',
    'standard_protein_precision_at_fmax', 'standard_protein_recall_at_fmax',
    'standard_micro_pr_auc', 'standard_micro_ap',
    'standard_micro_fmax_exact', 'standard_num_positive_labels',
    'standard_num_proteins', 'standard_num_go',
)


def native_imports(project_root: Path | None = None):
    if project_root is not None:
        model_dir = Path(project_root) / 'nbs_models' / 'nbs_protein_go'
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
    try:
        from nbs_pg.full_task_metrics_v085 import (
            compute_standard_metrics, metric_definitions,
            metrics_implementation_sha256, _validate, _thresholds, _protein_curve,
        )
    except ImportError as exc:
        raise ImportError('The unmodified nbs_pg/full_task_metrics_v085.py is required. '
                          'Do not substitute Stage-1 compute_metric_pack.') from exc
    kernels = dict(compute=compute_standard_metrics, definitions=metric_definitions,
                   sha=metrics_implementation_sha256, validate=_validate,
                   thresholds=_thresholds, protein_curve=_protein_curve)
    validate_metric_contract(kernels)
    return kernels


def validate_metric_contract(kernels: Mapping[str, Any]) -> dict:
    defs = kernels['definitions']()
    if not isinstance(defs, dict) or defs.get('schema') != REQUIRED_SCHEMA:
        raise RuntimeError('Wrong v085 metric schema; required '
                           f'{REQUIRED_SCHEMA}, observed {getattr(defs, "get", lambda *_: None)("schema")}')
    desc = str(defs.get('micro_pr_auc', '')).lower()
    if 'trapezoid' not in desc or 'precision=1' not in desc:
        raise RuntimeError('v085 micro PR-AUC definition changed; review before evaluating')
    if 'non-interpolated' not in str(defs.get('micro_ap', '')).lower():
        raise RuntimeError('v085 non-interpolated AP definition changed')
    return defs


def v088_original_fmax(y_true: np.ndarray, probs: np.ndarray, kernels: Mapping[str, Any]):
    """Exactly mirrors the original v088 Fmax-only path (without AP/PR-AUC)."""
    y, p = kernels['validate'](y_true, probs)
    thresholds = kernels['thresholds'](.001)
    f1, precision, recall, eligible = kernels['protein_curve'](y, p, thresholds)
    best = int(np.argmax(f1))
    return dict(standard_protein_fmax=100 * float(f1[best]),
                standard_protein_fmax_threshold=float(thresholds[best]),
                standard_protein_precision_at_fmax=100 * float(precision[best]),
                standard_protein_recall_at_fmax=100 * float(recall[best]),
                standard_num_proteins=int(y.shape[0]), standard_num_go=int(y.shape[1]),
                standard_num_positive_labels=int(y.sum()),
                standard_num_protein_fmax_eligible=int(eligible),
                standard_protein_fmax_threshold_step=.001)


def unified_v085_metrics(y_true: np.ndarray, probs: np.ndarray,
                         kernels: Mapping[str, Any], *, verify_fmax=True):
    """Native v085 outputs plus fixed reporting aliases, preserving every field.

    Fmax_percent = standard_protein_fmax (protein-centric)
    AuPRC_percent = standard_micro_pr_auc (trapezoidal micro PR area)
    AP_percent = standard_micro_ap (non-interpolated micro average precision)
    """
    validate_metric_contract(kernels)
    native = kernels['compute'](y_true, probs, threshold_step=.001)
    if not isinstance(native, dict):
        raise TypeError('compute_standard_metrics must return a dict')
    missing = set(REQUIRED_FIELDS) - set(native)
    if missing:
        raise KeyError(f'Original v085 output missing required fields: {sorted(missing)}')
    fmax, auprc, ap = (float(native[key]) for key in
                       ('standard_protein_fmax', 'standard_micro_pr_auc', 'standard_micro_ap'))
    if any(not math.isfinite(v) or v < 0 or v > 100.000001 for v in (fmax, auprc, ap)):
        raise ValueError('Original v085 Fmax/AuPRC/AP must be finite percentages [0,100]')
    y = np.asarray(y_true)
    if (int(native['standard_num_proteins']) != y.shape[0]
            or int(native['standard_num_go']) != y.shape[1]
            or int(native['standard_num_positive_labels']) != int(y.sum())):
        raise ValueError('v085 metric sample/GO/positive counts disagree with the gold matrix')
    if verify_fmax:
        legacy = v088_original_fmax(y_true, probs, kernels)
        for key in ('standard_protein_fmax', 'standard_protein_fmax_threshold',
                    'standard_protein_precision_at_fmax', 'standard_protein_recall_at_fmax'):
            if not np.isclose(float(native[key]), float(legacy[key]), atol=1e-8, rtol=0):
                raise AssertionError(f'Original v088 Fmax-only parity failed for {key}: '
                                     f'new={native[key]}, old={legacy[key]}')
    result = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in native.items()}
    result.update(Fmax_percent=fmax, AuPRC_percent=auprc, AP_percent=ap,
                  AuPRC_native_key='standard_micro_pr_auc',
                  AP_native_key='standard_micro_ap',
                  observed_positive_prevalence_percent=100 * int(y.sum()) / y.size,
                  evaluation_backend='v085_native_standardized',
                  standard_threshold_step_used=.001)
    return result


def native_protocol(kernels):
    defs = validate_metric_contract(kernels)
    return dict(implementation='nbs_pg.full_task_metrics_v085.compute_standard_metrics',
                implementation_sha256=kernels['sha'](), metric_definitions=defs,
                metric_schema=REQUIRED_SCHEMA, threshold_step=.001,
                Fmax='standard_protein_fmax: protein-centric Fmax with > threshold',
                AuPRC='standard_micro_pr_auc: MICRO TRAPEZOIDAL PR-AUC, not AP',
                AP='standard_micro_ap: non-interpolated MICRO average precision',
                caution=('The v085 trapezoidal PR-AUC can be high under ties/sparse '
                         'positives: at positive prevalence q and identical scores, '
                         'PR-AUC=(1+q)/2 while AP=q. Report AP and prevalence.'),
                units='percent except thresholds',
                ontology_prediction_propagation=False,
                go_axis='complete original classifier columns including aliases')
