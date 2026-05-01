import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple, Optional

def _binary_clf_curve(
    preds: Tensor,
    target: Tensor,
    pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """adapted from https://github.com/scikit-learn/scikit- learn/blob/master/sklearn/metrics/_ranking.py."""

    # remove class dimension if necessary
    if preds.ndim > target.ndim:
        preds = preds[:, 0]
    desc_score_indices = torch.argsort(preds, descending=True)

    preds = preds[desc_score_indices]
    target = target[desc_score_indices]

    weight = 1.0

    # pred typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
    threshold_idxs = F.pad(distinct_value_indices, [0, 1], value=float(target.size(0) - 1))
    target = (target == pos_label).to(torch.long)
    tps = torch.cumsum(target * weight, dim=0)[threshold_idxs]

    fps = 1 + threshold_idxs - tps

    return fps, tps, preds[threshold_idxs]

def _precision_recall_curve_compute_single_class(
    preds: Tensor,
    target: Tensor,
    pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes precision-recall pairs for single class inputs.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        pos_label: integer determining the positive class.
        sample_weights: sample weights for each data point
    """

    fps, tps, thresholds = _binary_clf_curve(
        preds=preds, target=target, pos_label=pos_label
    )
    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # stop when full recall attained and reverse the outputs so recall is decreasing
    last_ind = torch.where(tps == tps[-1])[0][0]
    # sl = slice(0, last_ind.item() + 1)
    sl = torch.arange(0, last_ind.item()+1)

    # need to call reversed explicitly, since including that to slice would
    # introduce negative strides that are not yet supported in pytorch
    # precision = torch.cat([reversed(precision[sl]), torch.ones(1, dtype=precision.dtype, device=precision.device)])

    # recall = torch.cat([reversed(recall[sl]), torch.zeros(1, dtype=recall.dtype, device=recall.device)])
    precision = torch.cat([precision[sl].flip([0]), torch.ones(1, dtype=precision.dtype, device=precision.device)])

    recall = torch.cat([recall[sl].flip([0]), torch.zeros(1, dtype=recall.dtype, device=recall.device)])

    # thresholds = reversed(thresholds[sl]).detach().clone()
    thresholds = thresholds[sl].flip([0]).detach().clone()

    return precision, recall, thresholds

def _precision_recall_curve_update(
    preds: Tensor,
    target: Tensor,
) -> Tuple[Tensor, Tensor, int]:
    """Updates and returns variables required to compute the precision-recall pairs for different thresholds.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_classes: integer with number of classes for multi-label and multiclass problems.
            Should be set to ``None`` for binary problems
        pos_label: integer determining the positive class. Default is ``None``
            which for binary problem is translate to 1. For multiclass problems
            this argument should not be set as we iteratively change it in the
            range [0,num_classes-1]
    """

    if len(preds.shape) == len(target.shape):
        pos_label = 1

        # binary problem
        preds = preds.flatten()
        target = target.flatten()

    else:
        raise ValueError("preds and target must have same number of dimensions, or one additional dimension for preds")

    return preds, target, pos_label

def prc_torch(target: Tensor, preds: Tensor):
    preds, target, pos_label = _precision_recall_curve_update(preds, target)
    return _precision_recall_curve_compute_single_class(preds, target, pos_label)