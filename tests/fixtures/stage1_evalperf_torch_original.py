# Isolated user-provided helper.py function snapshot for metric regression tests.
# Original helper SHA256: 70135074954cf10a084c75e215908f3fa106595d6bbf4c24e4cc3f2e62b7f4ef
# Only this function is AST-loaded, so unavailable helper_functions imports are not needed.
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
