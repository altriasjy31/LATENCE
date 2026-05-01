from optparse import Option
import sys
import os
from typing import List, Optional

prj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if not prj_dir in sys.path:
    sys.path.append(prj_dir)

from experiments.preprocess.alignment import jackhmmer_align_mpi, msa_reformat_mpi

from multiprocessing import Pool
import pickle
from argparse import ArgumentParser, Namespace

# cpu is 2 in default 
# iterations is 3 not 5
def cal_jackhmmer(access_lst: List[str], dbfile: str, # positional arguments
                  seqdir: str, outdir: str,msadir: str,
                  iterations: int = 3, ofile: Optional[str] = None, # options
                  evalue: Optional[float] = None, 
                  incE: Optional[float] = None, 
                  incdomE: Optional[float] = None,
                  tscore: Optional[float] = None, 
                  incT: Optional[float] = None, 
                  incdomT: Optional[float] = None,
                  idflag: bool = False,idcutoff: float = 0.95,threads: int = 2,
                  workers: int = 5):

    args_lst = [
        (x, seqdir,dbfile,
            outdir,msadir,ofile,iterations,
                evalue, incE, incdomE,
                tscore,incT,incdomT,
                idflag, idcutoff, threads) \
            for x in access_lst
        ]

    with Pool(processes=workers) as pool:
        pool.starmap(
            jackhmmer_align_mpi,
            args_lst
        )

def cal_reformat(access_lst: List[str], indir: str, outdir:str, # positional arguments
                 # options
                 infmt: str = "sto", outfmt: str = "a3m", 
                 hhlib: str = "", 
                 textw: int=1000, 
                 gap: str="-", 
                 workers: int=5):
    ifilenamefmt = "{}." + f"{infmt}"
    ofilenamefmt = "{}." + f"{outfmt}"
    def get_needed(acid):
        ifilepath = os.path.join(indir, ifilenamefmt.format(acid))
        ofilepath = os.path.join(outdir, ofilenamefmt.format(acid))

        cond = [
            os.path.exists(ifilepath) and os.path.getsize(ifilepath) > 0,
            not os.path.exists(ofilepath)
        ]

        return all(cond)

    ac_filter = filter(get_needed, access_lst)
        
    
    args_lst = [
        (x, indir, outdir, infmt, outfmt, hhlib, 
         textw, gap) \
            for x in ac_filter
    ]

    with Pool(processes=workers) as pool:
        pool.starmap(msa_reformat_mpi, args_lst)