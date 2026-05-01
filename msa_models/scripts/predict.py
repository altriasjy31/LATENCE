import typing as T
import pickle
from functools import partial, reduce
import itertools
import operator
import subprocess
from argparse import ArgumentParser, Namespace
import re

from torch import Tensor

import sys
import os

prj_dir = os.path.dirname(os.path.dirname(__file__))
if not prj_dir in sys.path:
    sys.path.append(prj_dir)
from models import Arch
from models.utils import record_generator
from calculate_msa import cal_jackhmmer, cal_reformat
from experiments.preprocess.alignparser import extract_from_a3m, padding_msa
from experiments.go.build_som import building
import helper_functions.obo_parser as OP

import pandas as pd
import torch
import torch.cuda.amp as amp
from torch.nn import Module
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO import write
from typing import Hashable, Iterable, List, Dict, Optional, Tuple, Union
import re

model_dict = {
    "gendis": Arch
}

ont2int = {v: k for k, v in enumerate(["cc","mf","bp"])}

Sig = torch.nn.Sigmoid()

device_pattern = re.compile(r"\d+(,\d+)*|\d?|-1")

## calculate multiple sequence alignment (msa)
## return the result filepaths
## the msa files' sufix is a3m
def write_fasta(seq_dir: str):
    def _writing(r: List[str], record: SeqRecord):
        fname = f"{record.name}.fa"
        fpath = os.path.join(seq_dir, fname)
        write(record, fpath, "fasta")
        r.append(record.name)
        return r
    return _writing

def calculate_msa(test_fapath: str, dbfilepath: str, hhlibpath: str):
    """
    test_fapath: test sequence path
    """

    test_dir = os.path.dirname(os.path.abspath(test_fapath))
    # evalue = incE = incdomE = 10
    iterations = 5
    evalue = 1e-8

    # create seq_dir, msa_dir, out_dir (jackhmmer print information)
    seq_dir, msa_dir, output_dir = [os.path.join(test_dir, x)
                                    for x in ["seq", "msa", "output"]]
    try:
        os.mkdir(seq_dir)
        os.mkdir(msa_dir)
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    # read sequence
    rgen = record_generator(test_fapath)
    # divide into different file
    access_lst: T.List[str] = reduce(write_fasta(seq_dir), rgen, [])

    # call jackhmmer
    cal_jackhmmer(access_lst, dbfilepath, seq_dir, output_dir, msa_dir,
                  iterations=iterations,
                  evalue=evalue)
    
    # call reformat
    a3m_dir = os.path.join(test_dir, "a3m")
    try:
        os.mkdir(a3m_dir)
    except FileExistsError:
        pass
    cal_reformat(access_lst, msa_dir, a3m_dir, hhlib=hhlibpath)

    return access_lst, a3m_dir

## build TSMF (load trained weight)
def load_state_dict(r: None, e: Tuple[Module, Dict]):
    model, state_dict = e
    model.load_state_dict(state_dict)
    return r

def build_tsmf(config_dir: str, weight_paths: List[Dict], device: Union[List[int], int, None]):
    task_lst = ["cco", "mfo", "bpo"]

    assert len(task_lst) == len(weight_paths), \
        "the number of weight_paths is not equal to the task_lst"

    cco_configpath, mfo_configpath, bpo_configpath = \
        [os.path.join(config_dir, f"{x}.yml") for x in task_lst]
    cco_model, mfo_model, bpo_model = \
        [building(p, device=device) for p in [cco_configpath, mfo_configpath, bpo_configpath]]

    reduce(load_state_dict, 
           zip([cco_model, mfo_model, bpo_model], 
               weight_paths), None) 
     
    return cco_model, mfo_model, bpo_model

## protein function prediction
def pred_add(r: List[Tensor], e: List[Tensor]):
    """
    r: [cco_pred, mfo_pred, bpo_pred]
    e: [cco_pred, mfo_pred, bpo_pred]
    """
    return [a+b for a, b in zip(r, e)]

def model_forward(input: Tensor, model: Module):
    model.eval()
    p: Tensor
    with torch.no_grad():
        if torch.cuda.is_available():
            with amp.autocast_mode.autocast():
                p = Sig(model(input)).cpu()
        else:
            p = Sig(model(input)).cpu()
        p.detach_()
        model.zero_grad()
    return p 

def pred_generator(cco_model: Module, 
                   mfo_model: Module, 
                   bpo_model: Module,
                   msa_buffer_size: int,
                   MAXLEN: int,
                   top_k: int,
                   n: int,
                   msa_path: str):
    for _ in range(n):
        msa = extract_from_a3m(msa_path, msa_buffer_size, top_k)
        msa = padding_msa(msa, MAXLEN, top_k).unsqueeze_(dim=0)
        yield [model_forward(msa, m) for m in [cco_model, mfo_model, bpo_model]]

def function_prediction(cco_model: Module, mfo_model: Module, bpo_model: Module,
                        access_lst: List[str], a3m_dir: str,
                        msa_buffer_size: int = 10000, 
                        MAXLEN: int = 2000, top_k: int = 40):
    """
    Return:

    predictions: n_queries x [n_cco_classes, n_mfo_classes, n_bpo_classes]
    """
    msa_paths = [os.path.join(a3m_dir, f"{x}.a3m")
                 for x in access_lst]
    
    n = 5
    partial_generator = partial(pred_generator,
                                cco_model, mfo_model, bpo_model,
                                msa_buffer_size, MAXLEN, top_k,
                                n)

    # [pred1, pred2, ..., pred_i, ...]
    # pred_i: [cco_pred, mfo_pred, bpo_pred]
    return [[x / n for x in reduce(pred_add,
                                   partial_generator(p))] 
            for p in msa_paths]

def formating_single_task(pred: Tensor, p: Tuple[str, List[str]], 
                          ont: OP.Ontology) -> List[Tuple[str, str, float, str]]:
    k, go_ordered = p
    plist: List[float] = pred.numpy().tolist()[0]
    return [(goname, k, score, ont.ont.get(goname, {}).get("name", "")) 
            for goname, score in zip(go_ordered, plist)]

# formating the predictions
def formating_single_prediction(preds: List[Tensor], 
                                task_go_ordered: Dict[str, List[str]],
                                ont: OP.Ontology):
    """
    Return: List[Tuple[str, str, float]]
    """
    
    # return [dict(zip(l, p))
    #         for p, l in zip(pred, task_go_ordered)]
    _single_task = partial(formating_single_task, ont=ont)
    return itertools.starmap(_single_task,
                             zip(preds, task_go_ordered.items()))

def formating(preds_lst: List[List[Tensor]], 
              task_go_ordered: Dict[str, List[str]],
              ont: OP.Ontology):
    """
    preds: [[cco_pred, mfo_pred, bpo_pred] ...]
    

    Return: Iterable[Iterable[List[Tuple[str, str, float, str]]]]
    """

    partial_formation = partial(formating_single_prediction, 
                                task_go_ordered=task_go_ordered,
                                ont=ont)
    return map(partial_formation, preds_lst)

def predicting(test_fapath: str, dbfilepath: str, hhlibpath: str, # calculate_msa
               config_dir: str, weights: List[Dict], # build_tsmf
               msa_buffer_size: int, MAXLEN: int, top_k: int, # function_prediction
               task_go_ordered: Dict[str, List[str]], # formating
               ont: OP.Ontology, # obo file parsed result
               device: Union[List[int], int, None] # device
               ):
    """
    # calulate_msa options
    test_fapath: str
    dbfilepath: str
    hhlibpath: str

    # build_tsmf options
    config_dir: str
    weights: List[Dict]

    # function_prediction options
    msa_buffer_size: int
    MAXLEN: int
    top_k: int

    # formating options
    task_go_ordered: Dict[str, List[str]]

    Return:
    formated_prediction: Iterable[Iterable[List[Tuple[str, str, float]]]]
    """
    access_lst, a3m_dir = calculate_msa(test_fapath, dbfilepath, hhlibpath)
    cco_model, mfo_model, bpo_model = build_tsmf(config_dir, weights, device=device)
    query_preds = function_prediction(cco_model, mfo_model, bpo_model,
                                      access_lst, a3m_dir, 
                                      msa_buffer_size, MAXLEN, top_k)
    return access_lst, formating(query_preds, task_go_ordered, ont)

def build_output_tuple(ac_name: str, go_name: str, k: str, score: float, term_name: float):
    return (ac_name, go_name, k, score, term_name)

def output_as_tuple(ac_name: str, preds: Iterable[List[Tuple[str, str, float]]]):
    partial_build = partial(build_output_tuple, ac_name)
    return itertools.starmap(partial_build,
                             itertools.chain.from_iterable(preds))

def saving_as_tsv(formated_preds: Iterable[Iterable[List[Tuple[str, str, float, str]]]], 
                  access_lst: List[str],
                  saving_path: str):
    """
    Return: zip(tuple)

    a row is composed of name: str, term: str, ontology: str, score: float, description
    """
    pred_tuple_lst = itertools.starmap(output_as_tuple, zip(access_lst, formated_preds))
    # pred_df = pd.DataFrame([p for iplst in pred_tuple_lst for p in iplst],
    pred_df = pd.DataFrame(itertools.chain.from_iterable(pred_tuple_lst),
                           columns=["name", "term", "ontology", "score", "description"])
    seqid_df = pd.DataFrame(enumerate(access_lst), columns=["id", "name"])
    pred_df = seqid_df.join(pred_df.set_index("name"), on="name")

    pred_df.to_csv(saving_path, sep="\t", index=False)

    return zip(pred_df["id"], pred_df["name"], 
               pred_df["term"], pred_df["ontology"], 
               pred_df["score"], pred_df["description"])

def row2key(row: Tuple):
    """
    row: Tuple[Label, Series[id, name, term, ontology, score]]
    """
    seqid: int = row[0]
    ontid = ont2int.get(row[3], len(ont2int))
    score: float = row[4]
    
    return seqid, ontid, -score # high score is first


def sorted_with_score(rows: Iterable[Tuple[Hashable, pd.Series]]):
    """
    rows: Iterable[Tuple[Label, Series[id, name, term, ontology, score]]]
    """
    
    return sorted(rows, key=row2key)

def parsing_device(device: Optional[str]):
    if device is None: return device

    gpu_ids: T.Union[T.List[int], int, None]
    matched = device_pattern.fullmatch(device)
    if matched is None:
        raise ValueError("device should be 0,1,2 or 0 or -1")
    else:
        if device.find(",") != -1:
            gpu_ids = [int(x) for x in device.strip().split(",")]
        else:
            gpu_ids = int(device)

    return gpu_ids

def main():
    """
    Return: Iterable[Tuple[Label, Series[id, name, term, ontology, score]]]
    """

    parser = ArgumentParser()

    parser.add_argument("fapath")
    parser.add_argument("dbpath")
    parser.add_argument("hhlib")
    parser.add_argument("configdir")
    parser.add_argument("weightpaths", nargs=3,
                        help="cco mfo bpo weight paths")
    parser.add_argument("golabelpath", help="task_go_ordered_lst pkl file path")
    parser.add_argument("geneontology", help="the path of geneontology file in obo format")
    parser.add_argument("saving_path", help="prediction result saving path")
    parser.add_argument("--msa-buffer-size", dest="msa_buffer_size",
                        type=int, default=10000,
                        help="the maximum buffer size to read sequence" + \
                             "from MSA for sampling")
    parser.add_argument("--MAXLEN", type=int, default=2000,
                        help="the maximum length for sequence")
    parser.add_argument("--top-k", dest="top_k", type=int,
                        default=40,
                        help="the maximum sampling size for input")
    parser.add_argument("--gpu-ids", dest="gpu_ids", type=str, 
                        help="set the device, e.g. 0,1 or 0, where -1 means cpu")
    
    opt = parser.parse_args()

    gpu_ids = parsing_device(opt.gpu_ids)
    if gpu_ids is None:
        device = torch.device("cpu")
    elif isinstance(gpu_ids, List):
        device = torch.device(f"cuda:{min(gpu_ids)}")
    else:
        device = torch.device(f"cuda:{gpu_ids}") if gpu_ids != -1 else torch.device("cpu")

    weights: List[Dict]
    weights = [torch.load(p, map_location=device) 
               for p in opt.weightpaths]
    with open(opt.golabelpath, "rb") as h:
        task2gosorted: Dict[str, List] = pickle.load(h)
    
    ont = OP.Ontology(opt.geneontology, True)
    access_lst, formated_preds = predicting(opt.fapath, opt.dbpath, opt.hhlib,
                                            opt.configdir, weights,
                                            opt.msa_buffer_size, opt.MAXLEN, opt.top_k,
                                            task2gosorted, ont,
                                            device=gpu_ids)
    if opt.saving_path is not None:
        iter_rows = saving_as_tsv(formated_preds, access_lst, opt.saving_path)
    else: iter_rows = None
    
    return iter_rows

if __name__ == "__main__":
    main()