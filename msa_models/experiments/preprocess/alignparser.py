import typing as T
from typing import Tuple, Optional, Iterator, Any, List, Iterable, Set
import numpy as np
from Bio import SeqIO as sio
from Bio.SeqRecord import SeqRecord
import re
import os
import sys
from itertools import islice, takewhile, dropwhile
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor

alphabet_size = 21
alphabet = np.array(list("-ARNDCQEGHILKMFPSTWYV"), dtype='|S1').view(np.uint8)
pattern = re.compile(r"[a-z]")

ndarray = np.ndarray

# size_of_msa = 100
def remove_lowercase(s: str):
    return pattern.sub("", s)


def fa2list(record: SeqRecord):
    s = remove_lowercase(str(record.seq))
    slst = list(s)
    return slst


def keep_same_size(size: Optional[int] = None):
    last_size = size

    def _check(s: List[str]):
        nonlocal last_size
        flag = len(s) == last_size if last_size is not None else True
        last_size = len(s)
        return flag

    return _check


def seqrecord2numpy(seqrecord: Iterable[SeqRecord]):
    # seq_lst = [fa2list(r) for r in seqrecord]
    # seq_lst = list(map(fa2list, seqrecord))
    seq_map = map(fa2list, seqrecord)
    seq_lst: T.List[T.List[str]] = list(filter(keep_same_size(None), seq_map))
    seq_mat = np.array(seq_lst, dtype="|S1").view(np.uint8)
    return seq_mat


# @jit(nopython=True)
def encoding(seq_mat: np.ndarray):
    for i in range(alphabet_size):
        seq_mat = np.where(seq_mat == alphabet[i], np.uint8(i), seq_mat)

    seq_mat = np.where(seq_mat > 20, 0, seq_mat)
    return seq_mat


def fasta_generator(fa_path: str):
    with open(fa_path, "r") as h:
        yield from sio.parse(h, "fasta")


def parsing_a3m_by_range(a3m_path: str, select_range: Optional[Tuple[int, int]] = None,
                         start: Optional[int] = None, end: Optional[int] = None,
                         size_of_msa: int = 100,
                         previous_seq_record: Optional[Iterable[SeqRecord]] = None,
                         only_seq_record: bool = False):
    # seq_record = sio.parse(a3m_path, "fasta") if previous_seq_record is None else previous_seq_record
    seq_record = fasta_generator(a3m_path) if previous_seq_record is None else previous_seq_record
    # print("get size of seq_record {}".format(sys.getsizeof(seq_record)))
    if select_range is not None:
        st, ed = select_range
    else:
        st = start if start is not None else 0
        ed = end if end is not None else st + size_of_msa

    assert type(st) == int and type(ed) == int, \
        "the select range is not Integer type"

    # seqrecord_slice = islice(seq_record, st, ed, 1)
    seqrecord_slice = takewhile(count_k_from_generator(ed - st),
                                dropwhile(count_k_from_generator(st),
                                          seq_record))
    # print("get size of seqrecord_slice {}".format(sys.getsizeof(seqrecord_slice)))
    if only_seq_record:
        return seqrecord_slice, seq_record
    else:
        seq_mat = seqrecord2numpy(seqrecord_slice)
        seq_mat = encoding(seq_mat)

        return seq_mat


def parsing_a3m(a3m_path, top_k=-1):
    seq_record = sio.parse(a3m_path, "fasta")
    if top_k != -1:
        # seqrecord_slice = islice(seq_record, 0, top_k)
        seqrecord_slice = takewhile(count_k_from_generator(top_k),
                                    seq_record)
        seq_mat = seqrecord2numpy(seqrecord_slice)
    else:
        seq_mat = seqrecord2numpy(seq_record)

    # for i in range(alphabet.shape[0]):
    #     seq_mat = np.where(seq_mat == alphabet[i], i, seq_mat)
    seq_mat = encoding(seq_mat)

    return seq_mat


def transform_a3m2mat_mpi(accessid, msa_dir, mat_dir, top_k=256):
    a3m_fmt = "{}.a3m"
    mat_fmt = "{}.npy"

    a3m_filepath = os.path.join(msa_dir, a3m_fmt.format(accessid))
    mat_filepath = os.path.join(mat_dir, mat_fmt.format(accessid))

    seq_mat = parsing_a3m(a3m_filepath, top_k)

    np.save(mat_filepath, seq_mat)

def extract_from_a3m(msa_path : str, msa_buffer_size: int = 10000, top_k: int = 40):
    """
    ensure the size of sequence extracted from file is larger than top_k
    """
    seqrecord_gen, seq_record = parsing_a3m_by_range(msa_path, start=0,
                                            size_of_msa=msa_buffer_size,
                                            only_seq_record=True)
    def seqrecord2mat(seqrecord : Iterable[SeqRecord]):
        records = list(seqrecord)
        r_size = len(records)
        indices = np.random.permutation(r_size)
        sel_indices = indices[top_k]
        mask = np.array([False] * msa_buffer_size, dtype=bool)
        mask[sel_indices] = True
        seqfilter: T.Iterable[SeqRecord] = filter(select_from_generator_by_index(mask.tolist()), records)
        seq_mat = seqrecord2numpy(seqfilter)
        seq_mat = encoding(seq_mat)
        return seq_mat

    try:
        seq_mat = seqrecord2mat(seqrecord_gen)
        # print(msa_path, seq_mat.shape)
    except RuntimeError:
        # print("capture stopiteration exception")
        seq_mat = seqrecord2mat(seqrecord_gen)
    except StopIteration:
        # print("capture stopiteration exception")
        seq_mat = seqrecord2mat(seqrecord_gen)

    return seq_mat
 
def padding_msa(msa, MAXLEN: int = 2000, top_k: int = 40) -> Tensor:
    msa = torch.tensor(msa, dtype=torch.int64)
    c, h = msa.shape
    pad_c = 0
    pad_h = 0
    if h < MAXLEN:
        pad_h = MAXLEN - h
    
    if c < top_k:
        pad_c = top_k - c
    
    if h < MAXLEN or c < top_k:
        msa = F.pad(msa, (0, pad_h, 0, pad_c))

    # msa: k x MAXLEN
    return msa[:top_k, :MAXLEN]


# @lru_cache(maxsize=512)
def count_k_from_generator(k: int):
    sum = 0

    def _count(x: Any):
        nonlocal sum
        sum += 1
        return sum <= k

    return _count


def select_from_generator_by_index(indices: List[bool]):
    i = -1

    def _select(x: Any):
        nonlocal i
        i += 1
        return indices[i]

    return _select


def output2dict(filepath: str):
    parsed_flag = False
    parsed_pattern = re.compile(r"^(:?Scores for complete sequences)")
    buffer_pattern = re.compile(r"^(:?Domain annotation for each sequence)")
    empty_pattern = re.compile(r"No hits")
    end_pattern = re.compile(r"^//")
    contents: List[str] = []
    with open(filepath, "r") as h:
        for line in h:
            line = line.strip()
            if len(line) == 0:
                continue
            matched = parsed_pattern.search(line)
            if not parsed_flag and matched is not None:
                parsed_flag = True
                contents = []

            if parsed_flag:
                matched = empty_pattern.search(line)
                if matched is not None:
                    parsed_flag = False
                    continue
                matched = buffer_pattern.search(line)
                if matched is not None:
                    parsed_flag = False
                else:
                    contents.append(line)

            matched = end_pattern.search(line)
            if matched is not None:
                break

    for line in contents[4:]:
        line = re.sub(r"^\+\s+", "", line, count=1)
        elst = line.split()
        evalue, score, bias = elst[:3]
        hit_name = elst[-1]

        yield hit_name, {"evalue": evalue,
                         "score": float(score),
                         "bias": float(bias)}


def transform_out2dict_mpi(accessid: str, outdir: str):
    outputFilePath = os.path.join(outdir, f"{accessid}.out")
    return accessid, dict(output2dict(outputFilePath))


def transform_out2dict_for_map(p: Tuple[str, str]):
    accessid, outdir = p
    outputFilePath = os.path.join(outdir, f"{accessid}.out")
    return accessid, dict(output2dict(outputFilePath))


def transform_out2dict_shared_memory(accessid: str, outdir: str, ac2res):
    outputFilePath = os.path.join(outdir, f"{accessid}.out")
    ac2res[accessid] = dict(output2dict(outputFilePath))


if __name__ == "__main__":
    import time
    from torch.nn import functional as F
    import torch

    st = time.time()
    a3m_path = "/mnt/exhome/Databases/GOA-S2F/CAFA/CAFA3/training/a3m_uniref/A0A086F3E3.a3m"
    # mat_path = "/mnt/exhome/Databases/GOA-S2F/CAFA/CAFA3/target/mat_uniref/T96060018291.npy"

    # seq_mat = parsing_a3m(a3m_path, size_of_msa)
    seq_mat = parsing_a3m_by_range(a3m_path, start=0)
    # np.save(mat_path, seq_mat)
    ed = time.time()
    print(f"consumed {ed - st}s")
    seq_onehot: torch.Tensor
    seq_onehot = F.one_hot(torch.tensor(seq_mat, dtype=torch.int64), 21)
    print(seq_onehot.shape)
