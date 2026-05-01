# preprocess the sequence fasta format data

import Bio.SeqIO as sio
from Bio.SeqRecord import SeqRecord
import os

from functools import reduce
import re

def fasta2generator(fafilepath):
    with open(fafilepath, "r") as h:
        for r in sio.parse(h, "fasta"):
            yield r


def split_into_single(fafilepath : str, savingdir : str):
    name_format = "{}.fa"
    def perform_saving(resu, record : SeqRecord):
        seqid = record.id
        savingpath = os.path.join(savingdir, name_format.format(seqid))
        with open(savingpath, "w") as h:
            sio.write(record, h, "fasta")
    
        return resu
    
    reduce(perform_saving, fasta2generator(fafilepath), None)

def read_by_byte(filepath):
    with open(filepath, "r") as h:
        while h.readable():
            yield h.read(1)

# Lines beginning with a hash # symbol will be treated as commentary lines in HHsearch/HHblits.
def a3m2generator(a3mfilepath):
    byte_gen = read_by_byte(a3mfilepath)
    block = ""

    for c in byte_gen:
        if c != "\x00":
            block += c
        else:
            yield block
            block = ""
    
    yield block

def a3m2pairwise(a3mfilepath):
    block_gen = a3m2generator(a3mfilepath)
    name_format = re.compile(r">\w+\|(?P<id>\w+)\|")
    def build_pair(block_str):
        resu = name_format.search(block_str)
        if resu is not None:
            seqid = resu.group("id")
            return seqid, block_str
        else:
            return "", block_str
    
    return map(build_pair, block_gen)

def split_a3m(a3mfilepath, saving_dir):
    saving_name_format = "{}_a3m.a3m"
    def _saving_by_condition(idset):
        seqid_block_map = a3m2pairwise(a3mfilepath)
        seqid_block_filter = filter(lambda p: p[0] in idset, seqid_block_map)
        
    
        def perform_saving(resu, e):
            seqid, block = e
            filepath = os.path.join(
                saving_dir, 
                saving_name_format.format(seqid)
            )
            with open(filepath, "w") as h:
                h.write(block)
            return resu
        
        reduce(perform_saving, seqid_block_filter)
    return _saving_by_condition
    
