from copy import deepcopy
import math
import functools as ft
import typing as T
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Sequence, Union
import numpy as np
import numpy.typing as nT
import torch
import torch as th
import torch.nn as nn
import torch.nn.parallel as dp
import sklearn.metrics as M
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from functools import partial, reduce

from scipy.stats import wilcoxon, mannwhitneyu
import helper_functions.prc as prc

Tensor = torch.Tensor
NDarray = nT.NDArray
ndarray = np.ndarray

def AuPRC_torch(targs: Tensor, preds: Tensor):
    pr, rc, _ = prc.prc_torch(targs.flatten(), preds.flatten())
    idx = torch.argsort(rc)
    pr, rc = pr[idx], rc[idx]
    return 100 * torch.trapz(pr, rc)

def fmax_torch(targs: Tensor, preds: Tensor):
    n = 100
    
    fmax_score = torch.as_tensor(0.)
    for t in range(n+1):
        threshold = t / n

        pred_bi = torch.where(preds > threshold,1,0)
        # score = M.f1_score(targs.flatten(), pred_bi.flatten())

        # MCM = M.multilabel_confusion_matrix(targs, pred_bi)
        tp = (pred_bi * targs).sum(1)
        tp_and_fp = pred_bi.sum(1)
        tp_and_fn = targs.sum(1)

        idx = torch.where(tp_and_fp != 0)
        tp_and_fp = tp_and_fp[idx]

        # control zero division
        precision = (tp[idx[0]] / tp_and_fp 
        if tp_and_fp.size(0) != 0 
        else torch.as_tensor(0.0)).mean()

        idx = torch.where(tp_and_fn != 0)
        tp_and_fn = tp_and_fn[idx]
        recall = (tp[idx] / tp_and_fn
        if tp_and_fn.size(0) != 0
        else torch.as_tensor(0.0)).mean()

        if (denom := precision + recall) == 0.0: 
            denom = torch.as_tensor(1.)
        if (score := 2 * precision * recall / denom) > fmax_score: 
            fmax_score = score
    return fmax_score * 100

class PReprot(T.TypedDict):
    fmax: float
    threshold: float
    smin: float
    auprc: float

def evalperf_torch(targs: th.Tensor, 
                   preds: th.Tensor,
                   threshold: bool=False,
                   smin: bool=False,
                   auprc: bool=False,
                   no_empty_labels: bool=False,
                   no_zero_classes: bool=False,
                   icary: th.Tensor | None=None,
                   ):
    n = 100
    if no_empty_labels:
        idx = torch.where(targs.sum(1))[0]
        targs = targs[idx]
        preds = preds[idx]
    
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        preds = preds[:, idx]
    
    assert not smin or isinstance(icary, torch.Tensor), \
    "when smin is true, icary must be not none"

    report: PReprot
    report = {"fmax": 0., "threshold": 0., "smin": -1.,
              "auprc": 0.}
    mi, ru = 0., 0.
    prs = th.zeros(n+1)
    rcs = th.zeros(n+1)
    for i, t in enumerate(range(n+1)):
        thres = t / n

        pred_mask = preds > thres
        targ_mask = targs > 0
        pred_bi = torch.where(pred_mask, 1, 0)
        tpM = pred_bi * targs
        fnM = targs * torch.where(~pred_mask, 1, 0)
        fpM = torch.where(~targ_mask, 1, 0) * pred_bi

        tp_sum = tpM.sum().item()
        pred_sum = pred_bi.sum().item()
        true_sum = targs.sum().item()

        # control zero division
        precision = tp_sum / pred_sum if pred_sum != 0.0 else 0.0
        recall = tp_sum / true_sum if true_sum != 0.0 else 0.0

        # fmax
        denom = precision + recall
        if denom == 0.0: denom = 1
        score = 100 * 2 * precision * recall / denom
        if score > report["fmax"]:
            report["fmax"] = score
            report["threshold"] = thres

        # mi, ru, smin
        if icary is not None:
            mi = (fpM * icary).sum(1).mean().item()
            ru = (fnM * icary).sum(1).mean().item()
        smin_score = math.sqrt(ru*ru+mi*mi)
        if report["smin"] < 0 or smin_score < report["smin"]:
            report["smin"] = smin_score

        # precision, recall
        prs[i] = precision
        rcs[i] = recall

    sorted_index = torch.argsort(rcs)
    report["auprc"] = (torch.trapz(prs[sorted_index],
                                  rcs[sorted_index]) * 100).item()
    keys = ["fmax"]
    if threshold: keys.append("threshold")
    if smin: keys.append("smin")
    if auprc: keys.append("auprc")
    return {k: report[k] for k in keys}

def ROC_score(targs: np.ndarray, preds: np.ndarray,):
    """
    class-centric rco-auc socre
    """

    idx = targs.sum(0) > 0
    targs = targs[:, idx]
    preds = preds[:, idx]

    def single_class(y_true: np.ndarray, 
                     y_score: np.ndarray):
        fpr, tpr, _ = M.roc_curve(y_true.ravel(), y_score.ravel())
        return np.trapz(tpr, fpr)
    
    return 100*np.mean([single_class(targs[:, i], preds[:, i])
                        for i in range(targs.shape[1])])


def AuPRC_score(targs: np.ndarray, preds: np.ndarray,
                no_zero_classes: bool = False):
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        preds = preds[:, idx]
    pr_ary, rc_ary, _ = precision_recall_curve(targs.flatten(), preds.flatten())
    AuPRC_value = auc(rc_ary, pr_ary)
    return 100 * AuPRC_value

def MCC_score(targs: np.ndarray, preds: np.ndarray):
    """
    class centric MCC score
    """
    idx = targs.sum(0) > 0
    targs = targs[:, idx]
    preds = preds[:, idx]

    def single_class(y_true: np.ndarray,
                     y_score: np.ndarray):
        return M.matthews_corrcoef(y_true.ravel(), 
                                   y_score.ravel())

    return 100*np.mean([single_class(targs[:, i], preds[:, i])
                        for i in range(targs.shape[1])])

def fmax_score(targs: ndarray, preds: ndarray,
               no_empty_labels: bool = True,
               no_zero_classes: bool = False,
               need_threshold: bool =False):
    n = 100
    if no_empty_labels:
        idx = np.where(targs.sum(1))[0]
        targs = targs[idx]
        preds = preds[idx]
    
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        preds = preds[:, idx]
    fmax_value = 0.
    best_thres = 0.
    for t in range(n+1):
        thres = t / n
        preds_label = np.where(preds > thres, 1, 0)
        true_positive = np.where(np.logical_and(preds_label, targs), 1, 0)
        tp: ndarray = true_positive.sum(axis=1)
        tp_and_fp: ndarray = preds_label.sum(axis=1)
        tp_and_fn: ndarray = targs.sum(axis=1)

        idx = np.where(tp_and_fp != 0)[0]
        tp_and_fp1: ndarray = tp_and_fp[idx]
        tp1: ndarray = tp[idx]
        avgprs = tp1 / tp_and_fp1 if tp_and_fp1.shape[0] != 0 else np.array([0.])

        # when no_empty_label is False
        idx = np.where(tp_and_fn != 0)[0]
        tp_and_fn1: ndarray = tp_and_fn[idx]
        tp1 = tp[idx]
        avgrcs = tp1 / tp_and_fn1 if tp_and_fn1.shape[0] != 0 else np.array([0.])

        avgpr = np.mean(avgprs)
        avgrc = np.mean(avgrcs)
        fscore = (2 * avgpr * avgrc) / (avgpr + avgrc) if avgpr != 0 or avgrc != 0 else 0.
        if fscore > fmax_value:
            fmax_value = fscore
            best_thres = thres
    # fmax_value = reduce(_calculate_fmax, np.arange(n+1))
    # fmax_value = max(map(_calculate_fmax, np.arange(n+1)))

    if need_threshold:
        return fmax_value * 100, best_thres
    else:
        return fmax_value * 100

def fmax_sklearn(targs: ndarray, preds: ndarray):
    """
    """
    n = 100
    # return max(f1_at(x/n) for x in range(n+1)) * 100
    targs = targs.astype(int)
    fmax_score = 0.
    for t in range(n+1):
        threshold = t / n

        pred_bi = np.where(preds > threshold,1,0)
        # score = M.f1_score(targs.flatten(), pred_bi.flatten())

        # MCM = M.multilabel_confusion_matrix(targs, pred_bi)
        tp_sum = np.logical_and(targs, pred_bi).sum()
        pred_sum = pred_bi.sum()
        true_sum = targs.sum()

        # control zero division
        precision = tp_sum / pred_sum if pred_sum != 0.0 else 0.0
        recall = tp_sum / true_sum if true_sum != 0.0 else 0.0

        denom = precision + recall
        if denom == 0.0: denom = 1
        if (score := 2 * precision * recall / denom) > fmax_score: 
            fmax_score = score
    return fmax_score * 100

def simple_prf_divide(numerator: NDarray, denominator: NDarray):
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = np.true_divide(numerator, denominator)

    if not np.any(mask):
        return result
    result[mask] = 0.0
    return result 

def fmax_tp(tp: np.ndarray, no_empty_labels: bool = True):
    """
    tp: 2 x 1 x n_classes
    """
    t, p = tp # both are 1 x n_classes
    return fmax_score(t,p, no_empty_labels)

def AuPRC_tp(tp: ndarray):
    t, p = tp
    return AuPRC_score(t,p)

def fmax_pvalue(pvaluefunc: Callable[[ndarray,ndarray, str], Any]):
    def _wrapper(targs_preds0, targs_preds1, 
                 alternative: str = "two_sided",
                 no_empty_labels:bool=False):
        vfmax_score = np.vectorize(fmax_tp,signature="(i,j,k),()->()")
        fm_ary0 = vfmax_score(targs_preds0, no_empty_labels)
        fm_ary1 = vfmax_score(targs_preds1, no_empty_labels)

        _, pvalue = pvaluefunc(fm_ary0, fm_ary1, alternative)
        return pvalue
    return _wrapper

def AuPRC_pvalue(pvaluefunc: Callable[[ndarray,ndarray, str], Any]):
    def _wrapper(targs_preds0, targs_preds1,
                 alternative: str = "two_sided"):
        vAuPRC_score = np.vectorize(AuPRC_tp, signature="(i,j,k)->()")
        
        au_ary0 = vAuPRC_score(targs_preds0)
        au_ary1 = vAuPRC_score(targs_preds1)

        _, pvalue = pvaluefunc(au_ary0, au_ary1, alternative)
        return pvalue
    return _wrapper

@fmax_pvalue
def fmax_wilcoxon(m0: ndarray,m1: ndarray, alternative: str = "two_sided"):
    return wilcoxon(m0,m1, alternative=alternative)

@AuPRC_pvalue
def AuPRC_wilcoxon(m0: ndarray, m1: ndarray, alternative: str = "two_sided"):
    return wilcoxon(m0, m1, alternative=alternative)

@fmax_pvalue
def fmax_mannwhitneyu(m0: ndarray, m1: ndarray, alternative: str = "two_sided"):
    return mannwhitneyu(m0, m1, alternative=alternative)

@AuPRC_pvalue
def AuPRC_mannwhitneyu(m0: ndarray, m1:ndarray, alternative: str = "two_sided"):
    return mannwhitneyu(m0, m1, alternative=alternative)

def eval_performance(targs: np.ndarray, preds: np.ndarray,
                     threshold: bool = False,
                     smin: bool = False,
                     auprc: bool = False,
                     no_empty_labels: bool = False,
                     no_zero_classes: bool = False,
                     icary: np.ndarray | None = None,
                     ):
    n = 100
    if no_empty_labels:
        idx = np.where(targs.sum(1))[0]
        targs = targs[idx]
        preds = preds[idx]
    
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        preds = preds[:, idx]
    
    assert not smin or isinstance(icary, np.ndarray), \
    "when smin is true, icary must be not none"

    report: PReprot
    report = {"fmax": 0., "threshold": 0., "smin": -1.,
              "auprc": 0.}
    mi, ru = 0., 0.
    prs = np.zeros(n+1)
    rcs = np.zeros(n+1)
    for i, t in enumerate(range(n+1)):
        thres = t / n
        # preds_label = np.where(preds > thres, 1, 0)
        # true_positive = np.where(np.logical_and(preds_label, targs), 1, 0)
        pred_mask = preds > thres
        targ_mask = targs > 0
        pred_bi = np.where(pred_mask, 1, 0)
        tpM = pred_bi * targs
        fnM = targs * np.where(~pred_mask, 1, 0) # 1 * (0 -> 1), true but false
        fpM = np.where(~targ_mask, 1, 0) * pred_bi # (0 -> 1) * 1, false but true

        tp_sum = tpM.sum()
        pred_sum = pred_bi.sum()
        true_sum = targs.sum()

        # control zero division
        precision = tp_sum / pred_sum if pred_sum != 0.0 else 0.0
        recall = tp_sum / true_sum if true_sum != 0.0 else 0.0

        # fmax
        denom = precision + recall
        if denom == 0.0: denom = 1
        if (score := 100 * 2 * precision * recall / denom) > report["fmax"]: 
            report["fmax"] = score
            report["threshold"] = thres

        # mi, ru, smin
        if icary is not None:
            mi = (fpM * icary).sum(1).mean()
            ru = (fnM * icary).sum(1).mean()
        smin_score = np.sqrt(ru*ru+mi*mi)
        if report["smin"] < 0 or smin_score < report["smin"]:
            report["smin"] = smin_score

        # precision, recall
        prs[i] = precision
        rcs[i] = recall

    sorted_index = np.argsort(rcs)
    report["auprc"] = np.trapz(prs[sorted_index],
                               rcs[sorted_index]) * 100
    keys = ["fmax"]
    if threshold: keys.append("threshold")
    if smin: keys.append("smin")
    if auprc: keys.append("auprc")
    return {k: report[k] for k in keys}


def load_by_module_name(cpu_model: nn.Module,
                        gpu_model: nn.Module,
                        r: None, 
                        module_name: str):
    """
    Using:
    partial_loading = partial(load_by_module_name, cpu_model, gpu_model)
    reduce(partial_loading, module_names, None)
    """
    gpu_module = gpu_model.get_submodule(module_name)
    if isinstance(gpu_module, dp.DataParallel):
        gpu_module = gpu_module.module
    else:
        pass
    
    cpu_model.get_submodule(module_name).load_state_dict(gpu_module.state_dict())
    
    return r
    
def loading_with_cpu(cpu_model: nn.Module, gpu_model: nn.Module, weights: OrderedDict,
                     module_names: List[str]):
    """
    loading weights that trained on gpu into cpu
    """
    gpu_model.load_state_dict(weights)
    partial_loading = partial(load_by_module_name, cpu_model, gpu_model)
    reduce(partial_loading, module_names, None)

def check_model_weights(model0: nn.Module, model1: nn.Module):
    return all([torch.equal(w0,w1) 
                for w0, w1 in zip(model0.state_dict().values(),
                                  model1.state_dict().values())])

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = (self.ema if self.ema is not None 
                             else self.val) * 0.99 + self.val * 0.01


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


if __name__ == "__main__":
    """
    """
    targs = np.array([[0, 1, 1, 0], [1, 0, 1, 0]])
    preds = np.array([[0.2, 0.3, 0.3, 0.5], [0.2, 0.3, 0.5, 0.3]])
    f0 = fmax_score(targs, preds)
    a0 = AuPRC_score(targs, preds) 
    print("fmax {:.4f}, AuPRC {:.4f}".format(f0,a0))
