import os
import pathlib
import sys

prj_path = os.path.dirname(os.path.dirname(__file__))
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import logging
import shutil
import subprocess
import typing as T
from pathlib import Path

import experiments.search as search
import utils.batch as batch

logger = logging.getLogger(__name__)

def mmseqs_preprocess(query: pathlib.Path,
                      base: pathlib.Path,
                      mmseqs: pathlib.Path):
    queries, _ = batch.get_queries(query, None)

    queries_unique = []
    for job_number, (raw_jobname, query_sequences, a3m_lines) in enumerate(queries):
        # remove duplicates before searching
        query_sequences = (
            [query_sequences] if isinstance(query_sequences, str) else query_sequences
        )
        query_seqs_unique = []
        for x in query_sequences:
            if x not in query_seqs_unique:
                query_seqs_unique.append(x)
        query_seqs_cardinality = [0] * len(query_seqs_unique)
        for seq in query_sequences:
            seq_idx = query_seqs_unique.index(seq)
            query_seqs_cardinality[seq_idx] += 1

        queries_unique.append([raw_jobname, query_seqs_unique, query_seqs_cardinality])

    base.mkdir(exist_ok=True, parents=True)
    query_file = base.joinpath("query.fas")
    with query_file.open("w") as f:
        for job_number, (
            raw_jobname,
            query_sequences,
            query_seqs_cardinality,
        ) in enumerate(queries_unique):
            for seq in query_sequences:
                f.write(f">{raw_jobname}\n{seq}\n")

    search.run_mmseqs(
        mmseqs,
        ["createdb", query_file, base.joinpath("qdb"), "--shuffle", "0"],
    )
    with base.joinpath("qdb.lookup").open("w") as f:
        id = 0
        file_number = 0
        for job_number, (
            raw_jobname,
            query_sequences,
            query_seqs_cardinality,
        ) in enumerate(queries_unique):
            for seq in query_sequences:
                f.write(f"{id}\t{raw_jobname}\t{file_number}\n")
                id += 1
            file_number += 1
    
    return query_file

def mmseqs_end_process(query_file: pathlib.Path,
                       base: pathlib.Path,
                       mmseqs: pathlib.Path):
    query_file.unlink()
    search.run_mmseqs(mmseqs, ["rmdb", base.joinpath("qdb")])
    search.run_mmseqs(mmseqs, ["rmdb", base.joinpath("qdb_h")])

def mmseqs_simple_search(
    dbbase: Path,
    base: Path,
    uniref_db: Path = Path("uniref30_2103_db"),
    mmseqs: Path = Path("mmseqs"),
    s: float = 5.7,
    e: float = 1e-4,
    db_load_mode: int = 2,
    threads: int = 32,
    max_accept: int = 10000,
    max_seqs: int = 10000,
    diff: int = 10000,
    filter_msa: int = 1,
):
    """
    Run mmseqs with a local mmseqs database set

    uniref_db: uniprot db (UniRef30)
    
    """

    used_dbs = [uniref_db]

    for db in used_dbs:
        if not dbbase.joinpath(f"{db}.dbtype").is_file():
            raise FileNotFoundError(f"Database {db} does not exist")
        if (
            not dbbase.joinpath(f"{db}.idx").is_file()
            and not dbbase.joinpath(f"{db}.idx.index").is_file()
        ):
            logger.info("Search does not use index")
            db_load_mode = 0
            dbSuffix1 = "_seq"
            dbSuffix2 = "_aln"
        else:
            dbSuffix1 = ".idx"
            dbSuffix2 = ".idx"

    search_base: T.List[T.Union[str, Path]]
    search_base = ["search", 
                       base.joinpath("qdb"), 
                       dbbase.joinpath(uniref_db), 
                       base.joinpath("res"), 
                       base.joinpath("tmp"), 
                       "--threads", str(threads)]
    search_param: T.List[T.Union[str, Path]]
    search_param = [
        "--num-iterations", "3",
        "--db-load-mode", str(db_load_mode),
        "-a", "-s", str(s),
        "-e", str(e),
        "--max-seqs", str(max_seqs),
        "--max-accept", str(max_accept),
    ]
    search.run_mmseqs(mmseqs, search_base + search_param)
    search.run_mmseqs(
        mmseqs,
        [
            "result2msa",
            base.joinpath("qdb"), dbbase.joinpath(uniref_db),
            base.joinpath("res"),
            base.joinpath("uniref.a3m"),
            "--msa-format-mode", "6",
            "--db-load-mode", str(db_load_mode),
            "--threads", str(threads),
            "--filter-msa", str(filter_msa),
            "--diff", str(diff),
        ],
    )
    
    mmseqs_head: T.List[T.Union[str, Path]] = [mmseqs]
    subprocess.run(mmseqs_head+ ["rmdb", base.joinpath("res")])
    search.run_mmseqs(mmseqs, ["mvdb", base.joinpath("uniref.a3m"), base.joinpath("final.a3m")])

    shutil.copyfile(base.joinpath("qdb.lookup"), base.joinpath("final.a3m.lookup"))
    search.run_mmseqs(mmseqs, 
                      ["unpackdb", 
                       base.joinpath("final.a3m"), 
                       base.joinpath("."), 
                       "--unpack-name-mode", "1", 
                       "--unpack-suffix", ".a3m"])
    
    search.run_mmseqs(mmseqs, ["rmdb", base.joinpath("final.a3m")])
    search.run_mmseqs(mmseqs, ["rmdb", base.joinpath("uniref.a3m")])
    search.run_mmseqs(mmseqs, ["rmdb", base.joinpath("res")])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        type=Path,
        help="fasta files with the queries.",
    )
    parser.add_argument(
        "dbbase",
        type=Path,
        help="The path to the database and indices you downloaded and created with setup_databases.sh",
    )
    parser.add_argument(
        "base", type=Path, help="Directory for the results (and intermediate files)"
    )
    parser.add_argument(
        "-s",
        type=float,
        default=5.7,
        help="mmseqs sensitivity. Lowering this will result in a much faster search but possibly sparser msas",
    )
    parser.add_argument(
        "-e",
        type=float,
        default=1e-4,
        help="mmseqs evalue. Lowering this will result in a smaller number of sequence in multiple sequence results"
    )
    # dbs are uniref, templates and environmental
    # We normally don't use templates
    parser.add_argument(
        "--db1", type=Path, default=Path("uniref30_2103_db"), help="UniRef database"
    )

    # control the msa size
    parser.add_argument(
        "--max-accept",
        type=int,
        default=10000,
        help="Max alignments accepted per query in the align stage. "
             "This is the main knob for limiting the number of sequences "
             "that can end up in the final a3m (e.g. 8000, 16000).",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=10000,
        help="Max candidates kept by the prefilter per query. "
             "Should be >= --max-accept, otherwise it becomes the real bottleneck.",
    )
    parser.add_argument(
        "--diff",
        type=int,
        default=10000,
        help="result2msa MSA diversification: keep at least this many seqs per "
             "50-column block. Increase together with --max-accept to avoid "
             "the a3m being shrunk by the default filter (1000).",
    )
    parser.add_argument(
        "--filter-msa",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether result2msa filters the MSA (1) or not (0). "
             "Set 0 if you want the a3m size to follow --max-accept more closely.",
    )

    parser.add_argument(
        "--mmseqs",
        type=Path,
        default=Path("mmseqs"),
        help="Location of the mmseqs binary",
    )

    parser.add_argument("--db-load-mode", type=int, default=2)
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()
    if args.max_seqs < args.max_accept:
        logger.warning(
            "--max-seqs (%d) < --max-accept (%d); "
            "prefilter will cap the results. Raising --max-seqs to --max-accept.",
            args.max_seqs, args.max_accept,
        )
        args.max_seqs = args.max_accept
    if args.diff < args.max_accept:
        logger.warning(
            "--diff (%d) < --max-accept (%d); "
            "result2msa will likely shrink the a3m. Consider raising --diff.",
            args.diff, args.max_accept,
        )

    query_file = mmseqs_preprocess(args.query, args.base, args.mmseqs)

    mmseqs_simple_search(
        dbbase=args.dbbase,
        base=args.base,
        uniref_db=args.db1,
        mmseqs=args.mmseqs,
        s=args.s,
        e=args.e,
        db_load_mode=args.db_load_mode,
        threads=args.threads,
        max_accept=args.max_accept,
        max_seqs=args.max_seqs,
        diff=args.diff,
        filter_msa=args.filter_msa,
    )
    
    mmseqs_end_process(query_file, args.base, args.mmseqs)

if __name__ == "__main__":
    main()