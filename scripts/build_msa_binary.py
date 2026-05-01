#!/usr/bin/env python3
import os
import sys
import string
import pathlib
import argparse
import pickle
import shutil
import hashlib
import typing as T
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# 如果脚本放在项目 scripts/ 目录下，这样一般能找到 helper_functions
prj_path = str(pathlib.Path(__file__).parent.parent)
if prj_path not in sys.path:
    sys.path.append(prj_path)

try:
    import msa_models.helper_functions.constants as C
    DEFAULT_ALPHABET = C.trR_ALPHABET
except Exception:
    DEFAULT_ALPHABET = None


NOLOWER = str.maketrans("", "", string.ascii_lowercase)


def build_mapping(alphabet: str) -> np.ndarray:
    mapping = np.zeros(256, dtype=np.uint8)
    arr = np.frombuffer(alphabet.encode("ascii"), dtype=np.uint8)
    mapping[arr] = np.arange(len(alphabet), dtype=np.uint8)
    return mapping


def stable_uint64(s: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
        byteorder="little",
        signed=False,
    )


def iter_a3m_sequences(a3m_file: str):
    """
    流式读取 A3M/FASTA 序列。

    支持序列换行，不假设每条序列只占一行。
    """
    seq_parts = []

    with open(a3m_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if seq_parts:
                    yield "".join(seq_parts)
                    seq_parts = []
            else:
                if "\x00" not in line:
                    seq_parts.append(line)

        if seq_parts:
            yield "".join(seq_parts)


def encode_a3m_file(
    a3m_file: str,
    mapping: np.ndarray,
    max_msa_size: int = -1,
    store_max_len: int = -1,
    shuffle_rows: bool = False,
    seed: int = 1,
    protein_id: str = "",
) -> T.Optional[np.ndarray]:
    """
    把单个 A3M 文件编码成 uint8[nseq, seqlen]。

    注意：
    1. 删除 A3M 小写 insertion。
    2. 只保留与 query sequence 等长的序列。
    3. query sequence 保持第 0 行。
    4. token id 使用 C.trR_ALPHABET 对应的映射。
    """

    query_len = None
    used_len = None
    nseq = 0

    # 用 bytearray 避免保存大量 Python string 列表
    buf = bytearray()

    for raw_seq in iter_a3m_sequences(a3m_file):
        seq = raw_seq.translate(NOLOWER).strip()

        if not seq:
            continue

        if query_len is None:
            query_len = len(seq)

            if store_max_len is not None and store_max_len > 0:
                used_len = min(query_len, store_max_len)
            else:
                used_len = query_len

        # 保持和原 encoder 一致：只保留与 query 等长的序列
        if len(seq) != query_len:
            continue

        seq = seq[:used_len]

        try:
            b = seq.encode("ascii")
        except UnicodeEncodeError:
            b = seq.encode("ascii", errors="ignore")

        # 非 ASCII 字符导致长度变化时，跳过该序列
        if len(b) != used_len:
            continue

        buf.extend(b)
        nseq += 1

        if max_msa_size != -1 and nseq >= max_msa_size:
            break

    if nseq == 0 or used_len is None or used_len == 0:
        return None

    chars = np.frombuffer(buf, dtype=np.uint8).reshape(nseq, used_len)
    arr = mapping[chars]

    if shuffle_rows and arr.shape[0] > 2:
        rng_seed = stable_uint64(f"{seed}:{protein_id}")
        rng = np.random.default_rng(rng_seed)

        perm = rng.permutation(arr.shape[0] - 1) + 1

        order = np.empty(arr.shape[0], dtype=np.int64)
        order[0] = 0
        order[1:] = perm

        arr = arr[order]

    return np.ascontiguousarray(arr, dtype=np.uint8)


def scan_msa_dir(
    msa_dir: str,
    msa_format: str = "a3m",
    recursive: bool = False,
    duplicate_policy: str = "error",
):
    """
    扫描 MSA 目录。

    duplicate_policy:
        error:
            protein id 重复时报错。
        first:
            重复时保留第一个。
        relative:
            用相对路径去掉后缀作为 protein id。
            适合 recursive=True 且不同子目录可能有相同文件名的情况。
    """
    assert duplicate_policy in ["error", "first", "relative"]

    root = Path(msa_dir)
    suffix = "." + msa_format.lstrip(".")

    pattern = f"**/*{suffix}" if recursive else f"*{suffix}"
    paths = sorted(root.glob(pattern))

    records = []
    seen = set()

    for p in paths:
        if not p.is_file():
            continue

        if duplicate_policy == "relative":
            rel = p.relative_to(root).as_posix()
            if not rel.endswith(suffix):
                continue
            protein = rel[: -len(suffix)]
        else:
            name = p.name
            if not name.endswith(suffix):
                continue
            protein = name[: -len(suffix)]

        if protein in seen:
            if duplicate_policy == "error":
                raise RuntimeError(
                    f"Duplicate protein id found: {protein}. "
                    f"Use --duplicate-policy first or relative if intended."
                )
            elif duplicate_policy == "first":
                continue

        seen.add(protein)
        records.append((protein, str(p)))

    return records


class BinaryShardWriter:
    """
    连续写入 uint8 MSA 矩阵到 shard 文件。
    """

    def __init__(
        self,
        out_dir: str,
        max_shard_bytes: int,
        prefix: str,
    ):
        self.out_dir = out_dir
        self.shard_dir = os.path.join(out_dir, "shards")
        os.makedirs(self.shard_dir, exist_ok=True)

        self.max_shard_bytes = int(max_shard_bytes)
        self.prefix = prefix

        self.part = 0
        self.fh = None
        self.current_size = 0
        self.shards: T.List[str] = []

    def _open_new(self):
        self.close()

        name = f"{self.prefix}_{self.part:06d}.bin"
        rel_path = os.path.join("shards", name)
        abs_path = os.path.join(self.out_dir, rel_path)

        self.fh = open(abs_path, "wb", buffering=1024 * 1024)
        self.current_size = 0
        self.part += 1
        self.shards.append(rel_path)

    def write(self, arr: np.ndarray):
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        nbytes = int(arr.nbytes)

        if self.fh is None:
            self._open_new()
        elif self.current_size > 0 and self.current_size + nbytes > self.max_shard_bytes:
            self._open_new()

        local_shard_id = len(self.shards) - 1
        offset = self.current_size

        # 避免 arr.tobytes() 的额外复制
        self.fh.write(memoryview(arr).cast("B"))
        self.current_size += nbytes

        return local_shard_id, offset

    def close(self):
        if self.fh is not None:
            self.fh.flush()
            self.fh.close()
            self.fh = None


def build_worker(
    worker_id: int,
    records: T.List[T.Tuple[str, str]],
    msa_dir: str,
    out_dir: str,
    alphabet: str,
    max_msa_size: int,
    store_max_len: int,
    max_shard_bytes: int,
    shuffle_rows: bool,
    seed: int,
    min_seqs: int,
    report_every: int,
):
    mapping = build_mapping(alphabet)

    writer = BinaryShardWriter(
        out_dir=out_dir,
        max_shard_bytes=max_shard_bytes,
        prefix=f"shard_w{worker_id:03d}",
    )

    proteins = []
    source_relpaths = []
    shard_ids = []
    offsets = []
    nseqs = []
    seqlens = []

    skipped_empty = 0
    skipped_error = 0
    error_examples = []

    for j, (protein, msa_path) in enumerate(records, start=1):
        try:
            arr = encode_a3m_file(
                a3m_file=msa_path,
                mapping=mapping,
                max_msa_size=max_msa_size,
                store_max_len=store_max_len,
                shuffle_rows=shuffle_rows,
                seed=seed,
                protein_id=protein,
            )
        except TypeError:
            # 兼容函数参数名写错时不会进入这里；保留无意义。
            raise
        except Exception as e:
            skipped_error += 1
            if len(error_examples) < 20:
                error_examples.append((protein, msa_path, repr(e)))
            continue

        if arr is None or arr.shape[0] < min_seqs:
            skipped_empty += 1
            continue

        local_sid, offset = writer.write(arr)

        proteins.append(protein)
        source_relpaths.append(os.path.relpath(msa_path, msa_dir))
        shard_ids.append(local_sid)
        offsets.append(offset)
        nseqs.append(arr.shape[0])
        seqlens.append(arr.shape[1])

        if report_every > 0 and j % report_every == 0:
            print(
                f"[worker {worker_id}] processed {j}/{len(records)}, "
                f"kept={len(proteins)}",
                flush=True,
            )

    writer.close()

    return {
        "worker_id": worker_id,
        "shards": writer.shards,
        "proteins": proteins,
        "source_relpaths": source_relpaths,
        "shard_ids": shard_ids,
        "offsets": offsets,
        "nseqs": nseqs,
        "seqlens": seqlens,
        "skipped_empty": skipped_empty,
        "skipped_error": skipped_error,
        "error_examples": error_examples,
    }


def split_records_round_robin(records, n_chunks: int):
    chunks = [records[i::n_chunks] for i in range(n_chunks)]
    return [c for c in chunks if len(c) > 0]


def build_binary(args):
    if args.alphabet is None:
        raise RuntimeError(
            "No alphabet found. Either make helper_functions.constants.C.trR_ALPHABET "
            "available or pass --alphabet explicitly."
        )

    if os.path.exists(args.out_dir):
        if args.overwrite:
            shutil.rmtree(args.out_dir)
        elif os.listdir(args.out_dir):
            raise RuntimeError(
                f"Output directory already exists and is not empty: {args.out_dir}. "
                f"Use --overwrite if intended."
            )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "shards"), exist_ok=True)

    records = scan_msa_dir(
        msa_dir=args.msa_dir,
        msa_format=args.msa_format,
        recursive=args.recursive,
        duplicate_policy=args.duplicate_policy,
    )

    if args.limit and args.limit > 0:
        records = records[: args.limit]

    if len(records) == 0:
        raise RuntimeError(f"No .{args.msa_format} files found in {args.msa_dir}")

    print(f"Found MSA files: {len(records)}")
    print(f"Output dir: {args.out_dir}")
    print(f"max_msa_size: {args.max_msa_size}")
    print(f"store_max_len: {args.store_max_len}")
    print(f"shuffle_rows: {args.shuffle_rows}")

    max_shard_bytes = int(args.max_shard_gb * 1024 ** 3)

    num_workers = max(1, int(args.num_workers))
    num_workers = min(num_workers, len(records))

    chunks = split_records_round_robin(records, num_workers)

    results = []

    if num_workers == 1:
        res = build_worker(
            worker_id=0,
            records=chunks[0],
            msa_dir=args.msa_dir,
            out_dir=args.out_dir,
            alphabet=args.alphabet,
            max_msa_size=args.max_msa_size,
            store_max_len=args.store_max_len,
            max_shard_bytes=max_shard_bytes,
            shuffle_rows=args.shuffle_rows,
            seed=args.seed,
            min_seqs=args.min_seqs,
            report_every=args.report_every,
        )
        results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = []

            for wid, chunk in enumerate(chunks):
                fut = ex.submit(
                    build_worker,
                    wid,
                    chunk,
                    args.msa_dir,
                    args.out_dir,
                    args.alphabet,
                    args.max_msa_size,
                    args.store_max_len,
                    max_shard_bytes,
                    args.shuffle_rows,
                    args.seed,
                    args.min_seqs,
                    args.report_every,
                )
                futures.append(fut)

            for fut in as_completed(futures):
                res = fut.result()
                print(
                    f"[worker {res['worker_id']}] done: "
                    f"kept={len(res['proteins'])}, "
                    f"empty={res['skipped_empty']}, "
                    f"errors={res['skipped_error']}",
                    flush=True,
                )
                results.append(res)

    results.sort(key=lambda x: x["worker_id"])

    global_shards = []
    proteins = []
    source_relpaths = []
    shard_ids = []
    offsets = []
    nseqs = []
    seqlens = []

    total_empty = 0
    total_errors = 0
    error_examples = []

    for res in results:
        base_sid = len(global_shards)

        global_shards.extend(res["shards"])
        proteins.extend(res["proteins"])
        source_relpaths.extend(res["source_relpaths"])

        shard_ids.extend([base_sid + int(x) for x in res["shard_ids"]])
        offsets.extend([int(x) for x in res["offsets"]])
        nseqs.extend([int(x) for x in res["nseqs"]])
        seqlens.extend([int(x) for x in res["seqlens"]])

        total_empty += res["skipped_empty"]
        total_errors += res["skipped_error"]
        error_examples.extend(res["error_examples"])

    if len(proteins) == 0:
        raise RuntimeError("No valid MSA was written.")

    shard_ids_np = np.asarray(shard_ids, dtype=np.int32)
    offsets_np = np.asarray(offsets, dtype=np.int64)
    nseqs_np = np.asarray(nseqs, dtype=np.int32)
    seqlens_np = np.asarray(seqlens, dtype=np.int32)

    total_tokens = int(np.sum(nseqs_np.astype(np.int64) * seqlens_np.astype(np.int64)))

    index = {
        "version": 2,
        "format": "raw_uint8_row_major_msa_shards",
        "alphabet": args.alphabet,
        "dtype": "uint8",

        "msa_dir": os.path.abspath(args.msa_dir),
        "msa_format": args.msa_format,

        "max_msa_size": args.max_msa_size,
        "store_max_len": args.store_max_len,
        "shuffle_rows": args.shuffle_rows,

        "shards": global_shards,
        "proteins": proteins,
        "source_relpaths": source_relpaths,

        "shard_ids": shard_ids_np,
        "offsets": offsets_np,
        "nseqs": nseqs_np,
        "seqlens": seqlens_np,

        "stats": {
            "num_samples": len(proteins),
            "num_shards": len(global_shards),
            "total_tokens": total_tokens,
            "total_bytes_uint8": total_tokens,
            "skipped_empty": total_empty,
            "skipped_error": total_errors,
            "error_examples": error_examples[:50],
        },
    }

    index_path = os.path.join(args.out_dir, "index.pkl")

    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print("Binary MSA build finished.")
    print(f"Index file: {index_path}")
    print(f"Samples written: {len(proteins)}")
    print(f"Shards written: {len(global_shards)}")
    print(f"Total uint8 tokens: {total_tokens:,}")
    print(f"Approx binary size: {total_tokens / 1024 ** 3:.2f} GB")
    print(f"Skipped empty: {total_empty}")
    print(f"Skipped errors: {total_errors}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("msa_dir", help="包含 .a3m 文件的目录")
    parser.add_argument("out_dir", help="输出二进制 MSA shard 的目录")

    parser.add_argument("--msa-format", default="a3m")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--duplicate-policy",
        choices=["error", "first", "relative"],
        default="error",
    )

    parser.add_argument(
        "--alphabet",
        default=DEFAULT_ALPHABET,
        help="token alphabet；默认使用 helper_functions.constants.C.trR_ALPHABET",
    )

    parser.add_argument(
        "--max-msa-size",
        type=int,
        default=-1,
        help="每个 MSA 文件最多保存多少条序列；-1 表示保存全部",
    )

    parser.add_argument(
        "--store-max-len",
        type=int,
        default=-1,
        help=(
            "二进制中最多保存多少列 residue position；"
            "如果模型永远只看前 1000 位，建议设为 1000。"
        ),
    )

    parser.add_argument(
        "--max-shard-gb",
        type=float,
        default=4.0,
        help="每个 shard 文件的目标最大大小，单位 GB",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="预处理进程数；慢盘上不要盲目开太大",
    )

    parser.add_argument(
        "--shuffle-rows",
        action="store_true",
        help="预处理阶段固定打乱 query 以外的 MSA 行",
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-seqs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    build_binary(args)


if __name__ == "__main__":
    main()