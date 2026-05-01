import os
import math
import pickle
import random
import typing as T
from collections import OrderedDict, defaultdict

import numpy as np
import torch as th
import torch.utils.data as td


def load_metadata_file(path: str):
    if path.endswith(".pt"):
        import torch

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

    else:
        raise ValueError(f"Unsupported metadata file: {path}")


def normalize_labels(x) -> T.List[int]:
    if x is None:
        return []

    if isinstance(x, (int, np.integer)):
        return [int(x)]

    return [int(v) for v in list(x)]


class TensorLRUCache:
    """
    按字节限制的 worker-local LRU cache。

    注意：
    DataLoader 多 worker 时，每个 worker 都有自己的 cache。
    总 cache 上限大致是：

        num_workers * cache_max_bytes
    """

    def __init__(self, max_bytes: int = 0):
        self.max_bytes = int(max_bytes)
        self.cur_bytes = 0
        self.cache = OrderedDict()

    @staticmethod
    def tensor_nbytes(x: th.Tensor) -> int:
        return int(x.numel() * x.element_size())

    def get(self, key):
        if self.max_bytes <= 0:
            return None

        if key not in self.cache:
            return None

        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key, value: th.Tensor):
        if self.max_bytes <= 0:
            return

        size = self.tensor_nbytes(value)

        if size > self.max_bytes:
            return

        if key in self.cache:
            old = self.cache.pop(key)
            self.cur_bytes -= self.tensor_nbytes(old)

        self.cache[key] = value
        self.cur_bytes += size

        while self.cur_bytes > self.max_bytes:
            _, old = self.cache.popitem(last=False)
            self.cur_bytes -= self.tensor_nbytes(old)

    def clear(self):
        self.cache.clear()
        self.cur_bytes = 0


class MSABinaryDataset(td.Dataset):
    """
    从 build_msa_binary.py 生成的二进制 shard 中读取 MSA。

    返回:
        f1d: uint8 tensor, shape [msa_buffer_size, max_len]
        y:   bool/float tensor, shape [num_classes]，如果 return_labels=True
    """

    def __init__(
        self,
        index_file: str,
        root_dir: T.Optional[str] = None,
    
        metadata_file: T.Optional[str] = None,
        mode: T.Optional[str] = None,
        task: T.Optional[str] = None,
        namekey: str = "proteins",
        labelkey: str = "prop_annotations",
        missing_action: str = "skip",
    
        num_classes: int = 40000,
    
        # 新增：显式 topk
        topk: T.Optional[int] = None,
    
        # 保留旧参数，作为 topk 的 fallback
        msa_buffer_size: int = 2000,
    
        max_len: int = 1000,
        need_proteins: bool = False,
        return_labels: T.Optional[bool] = None,
        label_dtype: th.dtype = th.bool,
    
        read_mode: str = "full",
        sample_strategy: str = "random",
        shuffle_rows_at_getitem: bool = True,
    
        pad_token: int = 0,
        cache_max_bytes: int = 0,
        max_open_files: int = 64,
    
        # 新增：采样随机性相关
        sample_seed: T.Optional[int] = None,
        avoid_last_sample: bool = True,
        avoid_last_sample_tries: int = 8,
        last_sample_cache_size: int = 4096,
    ):
        """
        read_mode:
            "full":
                读取完整 MSA 矩阵，然后在内存中采样行。
                优点：大块连续读，通常稳定。
                缺点：极深 MSA 会读取较多数据。

            "rows":
                只读取采样到的行。
                优点：理论读取字节少。
                缺点：会产生很多小随机读；msa_buffer_size=2000 时可能反而慢。

            "block":
                只在 sample_strategy="block" 时有效。
                读取 query 行 + 一个连续 row block。
                如果预处理时用了 --shuffle-rows，这通常是一个速度/随机性的折中。

        sample_strategy:
            "random":
                query + 随机采样的 MSA 行。

            "block":
                query + 一个随机连续 block。
                建议和 build 阶段的 --shuffle-rows 配合使用。

            "head":
                直接取前 msa_buffer_size 行。
                速度快，但随机性最弱。

        missing_action:
            "skip":
                metadata 中有 protein 但 binary index 中没有时跳过。

            "error":
                遇到缺失 protein 直接报错。
        """
        assert missing_action in ["skip", "error"]
        assert read_mode in ["full", "rows", "block"]
        assert sample_strategy in ["random", "block", "head"]

        self.index_file = index_file

        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(index_file))

        self.root_dir = root_dir

        with open(index_file, "rb") as f:
            index = pickle.load(f)

        self.index = index

        self.shard_rel_paths = list(index["shards"])
        self.shard_paths = [
            os.path.join(self.root_dir, p) for p in self.shard_rel_paths
        ]

        self.all_proteins = list(index["proteins"])
        self.shard_ids = np.asarray(index["shard_ids"], dtype=np.int32)
        self.offsets = np.asarray(index["offsets"], dtype=np.int64)
        self.nseqs = np.asarray(index["nseqs"], dtype=np.int32)
        self.seqlens = np.asarray(index["seqlens"], dtype=np.int32)

        if len(set(self.all_proteins)) != len(self.all_proteins):
            raise RuntimeError(
                "Duplicate protein names found in binary index. "
                "Rebuild with --duplicate-policy relative or fix input names."
            )

        protein_to_record = {
            p: i for i, p in enumerate(self.all_proteins)
        }

        sample_record_indices = []
        sample_proteins = []
        sample_labels = None

        if metadata_file is not None:
            if mode is None or task is None:
                raise ValueError(
                    "When metadata_file is given, mode and task must also be given."
                )

            meta = load_metadata_file(metadata_file)
            sub = meta[mode][task]

            proteins = list(sub[namekey])

            if labelkey in sub:
                labels = list(sub[labelkey])
            else:
                labels = [[] for _ in proteins]

            if len(proteins) != len(labels):
                raise RuntimeError(
                    f"metadata proteins/labels length mismatch: "
                    f"{len(proteins)} vs {len(labels)}"
                )

            sample_labels = []
            missing = []

            for p, y in zip(proteins, labels):
                p = str(p)

                rid = protein_to_record.get(p)

                if rid is None:
                    missing.append(p)
                    continue

                sample_record_indices.append(rid)
                sample_proteins.append(p)
                sample_labels.append(normalize_labels(y))

            if missing:
                msg = (
                    f"{len(missing)} proteins in metadata are missing from binary MSA index. "
                    f"Examples: {missing[:10]}"
                )

                if missing_action == "error":
                    raise RuntimeError(msg)
                else:
                    print("[MSABinaryDataset]", msg)

        else:
            sample_record_indices = list(range(len(self.all_proteins)))
            sample_proteins = self.all_proteins[:]
            sample_labels = None

        self.record_indices = np.asarray(sample_record_indices, dtype=np.int64)
        self.proteins = sample_proteins

        self.num_classes = int(num_classes)

        if topk is None:
            topk = msa_buffer_size
        
        if topk is None or int(topk) <= 0:
            raise ValueError(
                f"topk must be a positive integer, got {topk}. "
                f"Do not use topk=-1 if you want fixed batch shape."
            )
        
        self.topk = int(topk)
        
        # 为兼容旧代码，仍然保留这个属性
        self.msa_buffer_size = self.topk
        
        if max_len is None or int(max_len) <= 0:
            raise ValueError(
                f"max_len must be a positive integer for fixed-shape batching, got {max_len}."
            )
        
        self.max_len = int(max_len)
        
        self.sample_seed = sample_seed
        self.avoid_last_sample = bool(avoid_last_sample)
        self.avoid_last_sample_tries = int(avoid_last_sample_tries)
        self.last_sample_cache_size = int(last_sample_cache_size)
        
        self._py_rng = None
        self._rng_key = None
        self._last_row_sigs = OrderedDict()
        self.need_proteins = bool(need_proteins)

        if return_labels is None:
            self.return_labels = sample_labels is not None
        else:
            self.return_labels = bool(return_labels)

        if self.return_labels and sample_labels is None:
            raise ValueError(
                "return_labels=True but no metadata_file was provided."
            )

        self.label_dtype = label_dtype

        if sample_labels is not None:
            label_indices = []
            label_indptr = [0]

            for y in sample_labels:
                y = normalize_labels(y)
                label_indices.extend(y)
                label_indptr.append(len(label_indices))

            self.label_indices = np.asarray(label_indices, dtype=np.int32)
            self.label_indptr = np.asarray(label_indptr, dtype=np.int64)
        else:
            self.label_indices = None
            self.label_indptr = None

        self.read_mode = read_mode
        self.sample_strategy = sample_strategy
        self.shuffle_rows_at_getitem = bool(shuffle_rows_at_getitem)

        self.pad_token = int(pad_token)

        self.cache_max_bytes = int(cache_max_bytes)
        self._msa_cache = TensorLRUCache(self.cache_max_bytes)

        self.max_open_files = int(max_open_files)
        self._fd_cache = OrderedDict()

        print(
            f"Loaded MSABinaryDataset: samples={len(self.proteins)}, "
            f"binary_records={len(self.all_proteins)}, "
            f"shards={len(self.shard_paths)}, "
            f"msa_buffer_size={self.msa_buffer_size}, "
            f"max_len={self.max_len}, "
            f"read_mode={self.read_mode}, "
            f"sample_strategy={self.sample_strategy}, "
            f"return_labels={self.return_labels}, "
            f"cache={self.cache_max_bytes / 1024 ** 3:.2f} GB per worker"
        )

    @property
    def sample_shard_ids(self):
        return self.shard_ids[self.record_indices]

    def __len__(self):
        return len(self.record_indices)

    def __getstate__(self):
        """
        DataLoader 多进程时，不把父进程 fd/cache/rng pickle 到 worker。
        每个 worker 独立维护 fd/cache/rng。
        """
        state = self.__dict__.copy()
    
        state["_fd_cache"] = OrderedDict()
        state["_msa_cache"] = TensorLRUCache(self.cache_max_bytes)
    
        state["_py_rng"] = None
        state["_rng_key"] = None
        state["_last_row_sigs"] = OrderedDict()
    
        return state

    def close(self):
        for _, fd in self._fd_cache.items():
            try:
                os.close(fd)
            except OSError:
                pass

        self._fd_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_fd(self, shard_id: int):
        shard_id = int(shard_id)

        if shard_id in self._fd_cache:
            fd = self._fd_cache.pop(shard_id)
            self._fd_cache[shard_id] = fd
            return fd

        path = self.shard_paths[shard_id]
        fd = os.open(path, os.O_RDONLY)

        self._fd_cache[shard_id] = fd

        while len(self._fd_cache) > self.max_open_files:
            _, old_fd = self._fd_cache.popitem(last=False)

            try:
                os.close(old_fd)
            except OSError:
                pass

        return fd

    def _get_rng(self) -> random.Random:
        """
        返回当前进程 / 当前 worker 独立的 Python RNG。
    
        目的：
            1. 每个 DataLoader worker 的采样流不同；
            2. 同一个 worker 内，每次 __getitem__ 调用都会推进 RNG 状态；
            3. persistent_workers=True 时，worker 不销毁，RNG 状态会持续推进。
        """
        worker_info = td.get_worker_info()
    
        # DDP 情况下，最好把 rank 混入 seed，避免不同 rank 的采样流完全一致。
        rank = int(os.environ.get("RANK", "0"))
    
        if worker_info is None:
            # num_workers=0，单进程读取
            if self._py_rng is None:
                if self.sample_seed is None:
                    seed = int.from_bytes(os.urandom(8), byteorder="little", signed=False)
                else:
                    seed = int(self.sample_seed)
    
                seed = (seed + 1000003 * rank) & ((1 << 64) - 1)
                self._py_rng = random.Random(seed)
                self._rng_key = (os.getpid(), -1, seed)
    
            return self._py_rng
    
        else:
            seed = int(worker_info.seed)
    
            if self.sample_seed is not None:
                seed ^= int(self.sample_seed) & ((1 << 64) - 1)
    
            seed = (seed + 1000003 * rank) & ((1 << 64) - 1)
    
            key = (os.getpid(), int(worker_info.id), seed)
    
            if self._py_rng is None or self._rng_key != key:
                self._py_rng = random.Random(seed)
                self._rng_key = key
    
            return self._py_rng

    @staticmethod
    def _pread_to_bytearray(fd: int, nbytes: int, offset: int) -> bytearray:
        nbytes = int(nbytes)
        offset = int(offset)

        buf = bytearray(nbytes)

        if nbytes == 0:
            return buf

        if hasattr(os, "preadv"):
            view = memoryview(buf)
            total = 0

            while total < nbytes:
                n = os.preadv(fd, [view[total:]], offset + total)

                if n == 0:
                    break

                total += n

            if total != nbytes:
                raise IOError(f"Short preadv: expected {nbytes}, got {total}")

            return buf

        chunks = []
        total = 0

        while total < nbytes:
            data = os.pread(fd, nbytes - total, offset + total)

            if not data:
                break

            chunks.append(data)
            total += len(data)

        if total != nbytes:
            raise IOError(f"Short pread: expected {nbytes}, got {total}")

        buf[:] = b"".join(chunks)
        return buf

    def _global_record_id(self, sample_index: int) -> int:
        return int(self.record_indices[sample_index])

    def _read_full_msa_tensor(self, record_id: int) -> th.Tensor:
        cached = self._msa_cache.get(record_id)

        if cached is not None:
            return cached

        shard_id = int(self.shard_ids[record_id])
        offset = int(self.offsets[record_id])
        nseq = int(self.nseqs[record_id])
        seqlen = int(self.seqlens[record_id])

        nbytes = nseq * seqlen

        fd = self._get_fd(shard_id)
        buf = self._pread_to_bytearray(fd, nbytes, offset)

        arr = np.frombuffer(buf, dtype=np.uint8).reshape(nseq, seqlen)
        tensor = th.from_numpy(arr)

        self._msa_cache.put(record_id, tensor)

        return tensor

    def _read_rows_msa_tensor(
        self,
        record_id: int,
        rows: T.List[int],
    ) -> th.Tensor:
        shard_id = int(self.shard_ids[record_id])
        base_offset = int(self.offsets[record_id])
        seqlen = int(self.seqlens[record_id])

        fd = self._get_fd(shard_id)

        out = np.empty((len(rows), seqlen), dtype=np.uint8)

        for j, r in enumerate(rows):
            row_offset = base_offset + int(r) * seqlen
            data = os.pread(fd, seqlen, row_offset)

            if len(data) != seqlen:
                raise IOError(
                    f"Short row read: expected {seqlen}, got {len(data)}"
                )

            out[j] = np.frombuffer(data, dtype=np.uint8)

        return th.from_numpy(out)

    def _read_query_plus_block(
        self,
        record_id: int,
        start_row: int,
        count_rest: int,
    ) -> th.Tensor:
        """
        读取 query row + 连续 block。

        返回:
            uint8[1 + count_rest, seqlen]
        """
        shard_id = int(self.shard_ids[record_id])
        base_offset = int(self.offsets[record_id])
        seqlen = int(self.seqlens[record_id])

        fd = self._get_fd(shard_id)

        out = np.empty((1 + count_rest, seqlen), dtype=np.uint8)

        # query row
        q = os.pread(fd, seqlen, base_offset)

        if len(q) != seqlen:
            raise IOError(f"Short query read: expected {seqlen}, got {len(q)}")

        out[0] = np.frombuffer(q, dtype=np.uint8)

        if count_rest > 0:
            block_offset = base_offset + int(start_row) * seqlen
            block_bytes = count_rest * seqlen

            buf = self._pread_to_bytearray(fd, block_bytes, block_offset)
            block = np.frombuffer(buf, dtype=np.uint8).reshape(count_rest, seqlen)
            out[1:] = block

        return th.from_numpy(out)

    def _sample_rows(self, nseq: int, k: int) -> T.List[int]:
        """
        返回要读取的 row indices。

        query 始终为第 0 行。
        """
        nseq = int(nseq)
        k = int(k)

        if nseq <= 0:
            return []

        k = min(k, nseq)

        if k <= 0:
            return []

        if k >= nseq:
            if (
                self.shuffle_rows_at_getitem
                and self.sample_strategy == "random"
                and nseq > 1
            ):
                rest = th.randperm(nseq - 1).add(1).tolist()
                return [0] + rest

            return list(range(nseq))

        if k == 1:
            return [0]

        if self.sample_strategy == "head":
            return list(range(k))

        if self.sample_strategy == "block":
            # query + 连续 block
            start = random.randint(1, nseq - (k - 1))
            return [0] + list(range(start, start + k - 1))

        # default: random
        sampled = random.sample(range(1, nseq), k - 1)
        return [0] + sampled

    def _make_label(self, sample_index: int) -> th.Tensor:
        y = th.zeros(self.num_classes, dtype=self.label_dtype)

        assert self.label_indices is not None
        assert self.label_indptr is not None

        start = int(self.label_indptr[sample_index])
        end = int(self.label_indptr[sample_index + 1])

        labels = self.label_indices[start:end]

        if labels.size > 0:
            labels = labels[(labels >= 0) & (labels < self.num_classes)]

            if labels.size > 0:
                labels_t = th.from_numpy(labels.astype(np.int64, copy=False))
                y[labels_t] = 1

        return y

    def _pad_or_crop_msa(
        self,
        msa: th.Tensor,
        target_msa_size: int,
        target_len: int,
    ) -> th.Tensor:
        """
        输入:
            msa: uint8[nseq, seqlen]

        输出:
            uint8[target_msa_size, target_len]
        """
        if msa.size(0) == target_msa_size and msa.size(1) == target_len:
            return msa.contiguous()

        out = msa.new_full(
            (target_msa_size, target_len),
            fill_value=self.pad_token,
        )

        h = min(msa.size(0), target_msa_size)
        w = min(msa.size(1), target_len)

        if h > 0 and w > 0:
            out[:h, :w].copy_(msa[:h, :w])

        return out

    def _rows_signature(self, rows: T.List[int]) -> int:
        """
        用紧凑 signature 记录上一次采样，避免保存完整 topk 行号带来的内存开销。
        """
        return hash(tuple(rows))
    
    
    def _remember_last_rows(self, record_id: int, rows: T.List[int]):
        if not self.avoid_last_sample:
            return
    
        self._last_row_sigs[int(record_id)] = self._rows_signature(rows)
    
        while len(self._last_row_sigs) > self.last_sample_cache_size:
            self._last_row_sigs.popitem(last=False)
    
    
    def _sample_topk_rows(self, record_id: int, nseq: int) -> T.List[int]:
        """
        固定 topk 采样逻辑。
    
        返回:
            rows: 实际需要读取的 MSA 行号，长度为 min(nseq, topk)
    
        约定:
            rows[0] 永远是 0，也就是 query sequence。
    
        输出 shape 的固定性不在这里完成，而是在 _pad_or_crop_msa() 中完成：
            actual rows <= topk
            final f1d shape == [topk, max_len]
        """
        nseq = int(nseq)
    
        if nseq <= 0:
            return []
    
        topk = int(self.topk)
        k = min(nseq, topk)
    
        if k <= 1:
            rows = [0]
            self._remember_last_rows(record_id, rows)
            return rows
    
        rng = self._get_rng()
    
        def draw_once() -> T.List[int]:
            # 1. 不推荐用于训练增强：总是取前 k 行
            if self.sample_strategy == "head":
                return list(range(k))
    
            # 2. block 采样：query + 一个连续 block
            #    如果预处理时使用了 --shuffle-rows，这个 block 在生物学上近似是随机子集。
            if self.sample_strategy == "block":
                if k >= nseq:
                    rest = list(range(1, nseq))
    
                    if self.shuffle_rows_at_getitem:
                        rng.shuffle(rest)
    
                    return [0] + rest
    
                # query 占 1 行，剩余 k - 1 行从 [1, nseq - 1] 里取连续 block
                max_start = nseq - (k - 1)
                start = rng.randint(1, max_start)
    
                return [0] + list(range(start, start + k - 1))
    
            # 3. 默认 random 采样：query + 随机 topk - 1 条
            if k >= nseq:
                rest = list(range(1, nseq))
    
                if self.shuffle_rows_at_getitem:
                    rng.shuffle(rest)
    
                return [0] + rest
    
            sampled = rng.sample(range(1, nseq), k - 1)
            return [0] + sampled
    
        rows = draw_once()
    
        # 避免同一个 worker 内、同一个 record 连续采样出完全相同的 rows。
        # 注意：这不是全局严格去重；严格全局去重需要维护巨大历史，通常不值得。
        if self.avoid_last_sample and self.sample_strategy != "head":
            last_sig = self._last_row_sigs.get(int(record_id))
            cur_sig = self._rows_signature(rows)
    
            # 判断理论上是否有变化空间
            if self.sample_strategy == "random":
                can_vary = nseq > k
            elif self.sample_strategy == "block":
                # block 起点数量为 nseq - k + 1
                can_vary = nseq > k
            else:
                can_vary = False
    
            if can_vary:
                tries = 0
    
                while cur_sig == last_sig and tries < self.avoid_last_sample_tries:
                    rows = draw_once()
                    cur_sig = self._rows_signature(rows)
                    tries += 1
    
        self._remember_last_rows(record_id, rows)
        return rows

    @staticmethod
    def _is_identity_rows(rows: T.List[int], nseq: int) -> bool:
        if len(rows) != int(nseq):
            return False
    
        for i, r in enumerate(rows):
            if int(r) != i:
                return False
    
        return True
    
    
    @staticmethod
    def _is_query_plus_contiguous_block(rows: T.List[int]) -> bool:
        """
        判断 rows 是否形如:
            [0, s, s+1, s+2, ..., s+m-1]
    
        这种情况下可以用一次 query read + 一次连续 block read。
        """
        if len(rows) <= 1:
            return False
    
        if rows[0] != 0:
            return False
    
        start = rows[1]
    
        if start < 1:
            return False
    
        for j, r in enumerate(rows[1:]):
            if r != start + j:
                return False
    
        return True

    def __getitem__(self, sample_index: int):
        record_id = self._global_record_id(sample_index)
        protein = self.proteins[sample_index]
    
        nseq = int(self.nseqs[record_id])
        seqlen = int(self.seqlens[record_id])
    
        target_msa_size = int(self.topk)
        target_len = int(self.max_len)
    
        # 核心：每次 __getitem__ 都重新采样 topk 行号
        rows = self._sample_topk_rows(record_id=record_id, nseq=nseq)
    
        actual_k = len(rows)
    
        if actual_k == 0:
            # 理论上 build 阶段 min_seqs>=1 时不会发生
            msa = th.empty((0, seqlen), dtype=th.uint8)
    
        else:
            # fast block path:
            # query + contiguous block 可以避免读取完整 MSA。
            use_block_read = (
                self.read_mode == "block"
                and self.sample_strategy == "block"
                and self._is_query_plus_contiguous_block(rows)
            )
    
            if use_block_read:
                msa = self._read_query_plus_block(
                    record_id=record_id,
                    start_row=rows[1],
                    count_rest=actual_k - 1,
                )
    
            elif (
                self.read_mode == "rows"
                and actual_k < nseq
            ):
                # 只读取采样到的 topk 行。
                # 注意：topk=2000 时会产生较多小随机读，是否更快需要实测。
                msa = self._read_rows_msa_tensor(record_id, rows)
    
            else:
                # 默认路径：读取完整 MSA，然后 index_select topk 行。
                msa = self._read_full_msa_tensor(record_id)
    
                if not self._is_identity_rows(rows, nseq):
                    idx = th.as_tensor(rows, dtype=th.long)
                    msa = msa.index_select(0, idx)
    
        # 关键：无论 actual_k 是多少，最终输出固定为 [topk, max_len]
        f1d = self._pad_or_crop_msa(
            msa,
            target_msa_size=target_msa_size,
            target_len=target_len,
        )
    
        if self.return_labels:
            y = self._make_label(sample_index)
    
            if self.need_proteins:
                return (protein, f1d), y
            else:
                return f1d, y
    
        else:
            if self.need_proteins:
                return protein, f1d
            else:
                return f1d


class ShardShuffleBatchSampler(td.Sampler):
    """
    让 batch 尽量来自同一个 shard，以减少随机读。

    用法:
        sampler = ShardShuffleBatchSampler(
            dataset.sample_shard_ids,
            batch_size=40,
            shuffle=True,
            seed=1,
        )

        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            ...
        )

    注意：
        使用 batch_sampler 时，DataLoader 不要再传 batch_size 和 shuffle。
    """

    def __init__(
        self,
        shard_ids,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 1,
    ):
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        groups = defaultdict(list)

        for i, sid in enumerate(shard_ids):
            groups[int(sid)].append(i)

        self.groups = dict(groups)
        self.shards = list(self.groups.keys())

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        shards = self.shards[:]

        if self.shuffle:
            rng.shuffle(shards)

        for sid in shards:
            indices = self.groups[sid][:]

            if self.shuffle:
                rng.shuffle(indices)

            batch = []

            for idx in indices:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            if batch and not self.drop_last:
                yield batch

    def __len__(self):
        n = 0

        for indices in self.groups.values():
            m = len(indices)

            if self.drop_last:
                n += m // self.batch_size
            else:
                n += math.ceil(m / self.batch_size)

        return n


def collate_pad_variable_msa(batch, pad_token: int = 0):
    """
    当 msa_buffer_size=-1 或 max_len=-1 时，不同样本 shape 可能不同。
    这个 collate_fn 会按 batch 内最大 shape padding。
    """
    has_label = isinstance(batch[0], tuple) and len(batch[0]) == 2

    if has_label:
        xs = [item[0] for item in batch]
        ys = [item[1] for item in batch]
    else:
        xs = batch
        ys = None

    has_protein = isinstance(xs[0], tuple)

    if has_protein:
        proteins = [x[0] for x in xs]
        msas = [x[1] for x in xs]
    else:
        proteins = None
        msas = xs

    bsz = len(msas)
    max_msa = max(x.size(0) for x in msas)
    max_len = max(x.size(1) for x in msas)

    out = msas[0].new_full(
        (bsz, max_msa, max_len),
        fill_value=pad_token,
    )

    for i, x in enumerate(msas):
        out[i, : x.size(0), : x.size(1)].copy_(x)

    if has_protein:
        x_out = (proteins, out)
    else:
        x_out = out

    if ys is not None:
        return x_out, th.stack(ys, dim=0)
    else:
        return x_out