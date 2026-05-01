import typing as T
from typing import Any, Tuple 
from argparse import ArgumentParser
import collections.abc as abc
import itertools as it
import torch

from Bio.SeqIO import parse
from Bio.SeqRecord import SeqRecord
from typing import Generator
import yaml
import argparse as argp

def to_np(x):
    return x.cpu().detach().numpy()

def parsing(parser : ArgumentParser):
    opt = parser.parse_args()
    if opt.config is not None:
        with open(opt.config, "r") as h:
            config_dict: T.Dict = yaml.safe_load(h)
        assert "base" in config_dict, \
            "we firstly get the base configuration filepath from the value of 'base' key, " +\
                "then, read the common option from the base configuration file"
        with open(config_dict["base"], "r") as h:
            base_dict: T.Dict = yaml.safe_load(h)
        option_dict = opt.__dict__
        option_dict.update(base_dict)
        option_dict.update(config_dict) # set the num classes of config_dict and update unique options
        return argp.Namespace(**option_dict)

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt

def record_generator(fasta_path: str) -> Generator[SeqRecord, None, None]:
    with open(fasta_path, "r") as h:
        for record in parse(h, "fasta"):
            yield record


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, abc.Iterable):
        return tuple(x)
    return tuple(it.repeat(x, n))