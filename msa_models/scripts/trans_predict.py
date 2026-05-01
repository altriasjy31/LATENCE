import typing as T
import json
import pickle
import pathlib as P
import itertools as it
import functools as ft
import operator as opr
import collections as clt
import os
import sys
from dataclasses import dataclass

prj_root = P.Path(__file__).parent.parent
if (p := str(prj_root)) not in sys.path:
    sys.path.append(p)

import helper_functions.helper as H
import helper_functions.obo_parser as gp

import numpy as np
import pandas as pd

def propagate(tp_tensor: np.ndarray | T.Tuple[np.ndarray, np.ndarray],
              go_rels: gp.Ontology,
              terms_df: pd.DataFrame,
              top_k: int = 100,
              min_score: float = 0.5
              ):
    targs, preds = tp_tensor
    assert isinstance(targs, np.ndarray) and isinstance(preds, np.ndarray)

    sorted_index = np.argsort(preds, 1)
    top_k_indices = sorted_index[:, :top_k]

    mask = preds >= min_score
    rows = np.arange(preds.shape[0])[:, None]
    mask[rows, top_k_indices] = False

    preds[~mask] = 0.

    # propagate on the go graph
    assert hasattr(terms_df, "gos")
    term_idx = {k: i for i, k in enumerate(terms_df.gos)}
    colmask = np.zeros(preds.shape[1], dtype=np.bool_)
    for i in np.arange(preds.shape[0]):
        pred_annots = terms_df.loc[mask[i]]
        pred_scores = preds[i, mask[i]]
        colmask.fill(False)
        for go_id, score in zip(pred_annots, pred_scores):
            supgo_set = go_rels.get_anchestors(go_id)
            supgoids = [idx for t in supgo_set 
                        if (idx := term_idx.get(t)) is not None]
            colmask[supgoids] = True
            preds[i, np.logical_and(colmask, ~mask[i])] = score
            update_mask = np.logical_and(colmask, mask[i])
            preds[i, update_mask] = np.maximum(preds[i, update_mask], score)
            # for sup_goid in supgoids:
            #     if mask[i, sup_goid]: # has score not zero
            #         preds[i, sup_goid] = max(preds[i, sup_goid], score)
            #     else:
            #         preds[i, sup_goid] = score

@dataclass
class ProgramArgs:
    pred_path: P.Path
    gofile_path: P.Path
    termfile_path: P.Path
    top_k: float
    min_score: float

def main(pred_path: P.Path,
         gofile_path: P.Path,
         termfile_path: P.Path,
         top_k: int,
         min_score: float):
    tp_tensor = np.load(pred_path)

    go_rels = gp.Ontology(gofile_path, with_rels=True)

    terms_df = pd.read_json(termfile_path)

    propagate(tp_tensor, go_rels, terms_df,
              top_k, min_score)
    
    fname = pred_path.name
    saving_path = pred_path.parent.joinpath(f"trans-{fname.removesuffix('.npy')}.npy")
    np.save(saving_path, tp_tensor)

if __name__ == "__main__":
    import argparse as A

    parser = A.ArgumentParser()
    parser.add_argument("pred_path", type=P.Path)
    parser.add_argument("gofile_path", type=P.Path)
    parser.add_argument("termfile_path", type=P.Path)

    parser.add_argument("-k", "--top-k", type=int, 
                        default=100)
    parser.add_argument("-s", "--min-score", type=float,
                        default=0.5)
    
    opt = ProgramArgs(**vars(parser.parse_args()))

    main(opt.pred_path, opt.gofile_path, 
         opt.termfile_path, opt.top_k,
         opt.min_score)