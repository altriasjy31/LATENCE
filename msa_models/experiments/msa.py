import os
import random
import sys
import pathlib
prj_path = str(pathlib.Path(__file__).parent.parent)
if prj_path not in sys.path:
   sys.path.append(prj_path)

import helper_functions.constants as C

import pickle
import torch
import torch as th
import torch.nn as nn
import torch.utils.data as td
import torch.utils.data.dataset as D
import torch.nn.functional as F
import typing as T
from collections import OrderedDict

import string
import re
import itertools as it
import more_itertools as mit
import numpy as np
import Bio.SeqIO as sio
import Bio.SeqRecord as R

DATA = T.Dict[str, T.Dict[str, T.Dict[str, T.List[T.Union[T.List[int], str]]]]]
NOLOWER = str.maketrans('','', string.ascii_lowercase)
SPILTER = re.compile(">[^>\n]+\n")
MAPPING = np.zeros(256,np.uint8) # unsigned int8, 0-255
alphabet = np.frombuffer(C.trR_ALPHABET.encode(), np.uint8)
index_alphabet = np.arange(len(C.trR_ALPHABET))
MAPPING[alphabet] = index_alphabet


class MSADataset(D.Dataset):
  namekey = "proteins"
  labelkey = "prop_annotations"
  num_classes = 40000
  msa_format = "a3m"

  def __init__(self, 
               dataset: str, msa_dir: str,
               mode: str, task: str, 
               num_classes: int, 
               msa_buffer_size: int,
               max_len: int,
               cutoff: float = 0.8,
               eps: float = 1e-9,
               need_proteins: bool = False,
               msa_file_cache_size: int = 0,
               msa_cache_dtype: str = "uint8",
               **kwargs):
    """
    msa_buffer_size:
        Number of MSA sequences returned per sample. Original meaning; changing is not recommended.

    msa_file_cache_size:
        MSA file cache size, measured in “number of files”.
        0  means no caching.
        >0 means each DataLoader worker caches the most recently accessed N MSA files.
        -1 means caching all accessed MSA files, which may consume a large amount of memory.

    msa_cache_dtype:
        "uint8": Cache uint8-encoded results, saves memory, but needs to be converted to long in __getitem__.
        "long":  Cache long tensors, faster, but memory usage is about 8 times that of uint8.
    """

    if dataset.endswith(".pt"):
        data = torch.load(dataset)
    elif dataset.endswith(".pkl"):
        with open(dataset, "rb") as h:
            data = pickle.load(h)
    else:
        raise NotImplementedError("not imp")

    assert msa_cache_dtype in ["uint8", "long"]

    self.data = data
    self.mode = mode
    self.task = task
    self.msa_dir = msa_dir

    self.num_classes = num_classes
    self.msa_buffer_size = msa_buffer_size
    self.max_len = max_len
    self.cutoff = cutoff
    self.eps = eps
    self.need_proteins = need_proteins

    self.msa_max_size = kwargs.get("msa_max_size", -1)

    # 新增：MSA 文件缓存参数
    self.msa_file_cache_size = int(msa_file_cache_size)
    self.msa_cache_dtype = msa_cache_dtype
    self._msa_cache = OrderedDict()

    self.targets = self._load_data(data, mode, task, msa_dir)
    self.len = len(self.targets["msa"])

    print(f"Loaded {mode}-set, length: {self.len}")
    print(
        f"MSA cache: size={self.msa_file_cache_size}, "
        f"dtype={self.msa_cache_dtype}"
    )

  def __len__(self):
     return self.len

  def reset_mode(self, mode, task=None, msa_dir=None):
    self.mode = mode

    if task is not None:
        self.task = task

    if msa_dir is not None:
        self.msa_dir = msa_dir

    self.clear_msa_cache()

    self.targets = self._load_data(
        self.data,
        self.mode,
        self.task,
        self.msa_dir
    )

    self.len = len(self.targets["msa"])
    print(f"Loaded {self.mode}-set, length: {self.len}")

  def __getstate__(self):
    """
    防止 DataLoader 使用 spawn 多进程时，把父进程中的缓存一起 pickle 到 worker。
    每个 worker 应该维护自己的 cache。
    """
    state = self.__dict__.copy()
    state["_msa_cache"] = OrderedDict()
    return state

  def clear_msa_cache(self):
    self._msa_cache.clear()

  def _load_msa_tensor_uncached(self, msa_path: str) -> th.Tensor:
    """
    从磁盘读取并编码 MSA。
    这个函数本身不做缓存。
    """
    a3m_lines = load_from(msa_path, self.msa_max_size)
    msa_np = encoder(a3m_lines)

    # encoder 返回的是 np.uint8
    msa_np = np.ascontiguousarray(msa_np)
    msa = th.from_numpy(msa_np)

    if self.msa_cache_dtype == "long":
        msa = msa.long()
    else:
        msa = msa.to(th.uint8)

    return msa

  def _get_msa_tensor(self, msa_path: str) -> th.Tensor:
    """
    带 LRU 逻辑的 MSA 获取函数。
    """
    # 0 表示完全不缓存
    if self.msa_file_cache_size == 0:
        return self._load_msa_tensor_uncached(msa_path)

    cache = self._msa_cache

    # cache hit
    if msa_path in cache:
        msa = cache.pop(msa_path)
        cache[msa_path] = msa
        return msa

    # cache miss
    msa = self._load_msa_tensor_uncached(msa_path)
    cache[msa_path] = msa

    # msa_file_cache_size > 0 时启用 LRU 淘汰
    # msa_file_cache_size == -1 时不淘汰
    if self.msa_file_cache_size > 0:
        while len(cache) > self.msa_file_cache_size:
            cache.popitem(last=False)

    return msa

  def _load_data(self,
                data: DATA,
                mode: str,
                task: str,
                msa_dir: str,
                msa_format: T.Optional[str] = None):
    """
    只保留有 MSA 文件的样本，并且同步过滤 proteins / labels / 其他等长字段。
    """

    if msa_format is None:
       msa_format = self.msa_format

    subdata = data[mode][task]
    proteins = subdata[self.namekey]
    n = len(proteins)

    kept_indices = []
    msa_paths = []

    for i, x in enumerate(proteins):
        p = os.path.join(msa_dir, f"{x}.{msa_format}")
        if os.path.exists(p):
            kept_indices.append(i)
            msa_paths.append(p)

    new_subdata = {}

    for k, v in subdata.items():
        if isinstance(v, list) and len(v) == n:
            new_subdata[k] = [v[i] for i in kept_indices]
        else:
            new_subdata[k] = v

    new_subdata["msa"] = msa_paths

    return new_subdata

  def get(self, key, index):
      if self.targets.get(key) is not None:
          assert index < len(self.targets[key])
          return self.targets[key][index]
      elif key == self.labelkey:
          return []

  def __getitem__(self, index):
    proteins = self.get(self.namekey, index)
    msa_path = self.get("msa", index)
    assert isinstance(msa_path, str)

    # 新增：从缓存读取，缓存 miss 时才从磁盘读取
    msa = self._get_msa_tensor(msa_path).long()

    current_size = msa.size(0)

    # shuffle，但保持 query sequence 不变
    if current_size > 1:
        idx = th.randperm(current_size - 1) + 1
        idx = th.cat([th.zeros(1, dtype=th.long), idx])
        msa = msa.index_select(0, idx)

    # 控制返回的 MSA 序列数
    if self.msa_buffer_size == -1:
        target_msa_size = current_size
    else:
        target_msa_size = self.msa_buffer_size

    current_len = msa.size(1)

    pad_len = max(0, self.max_len - current_len)
    pad_size = max(0, target_msa_size - current_size)

    f1d = F.pad(msa, (0, pad_len, 0, pad_size))
    f1d = f1d[:target_msa_size, :self.max_len]

    y = self.get(self.labelkey, index)
    assert isinstance(y, T.List)

    # 不建议复用 self.classes，因为它是类变量，且在多 worker 下容易让逻辑变复杂
    labels = th.zeros(self.num_classes, dtype=th.int)

    if len(y) > 0:
        # 保持原来 self.classes[:self.num_classes] 的语义：
        # 超过 num_classes 的 label 不使用
        y = [yy for yy in y if 0 <= yy < self.num_classes]
        if len(y) > 0:
            labels[y] = 1

    if not self.need_proteins:
        return f1d, labels
    else:
        return (proteins, f1d), labels

def bufferedAlignReader(a3m_file: str, max_size: int = 10000):
  max_lines = max_size * 2
  with open(a3m_file, "r") as h:
    for i, line in enumerate(h, start=1):
      if i > max_lines:
        break
      yield line

def get_sequence(seqrecord: R.SeqRecord):
   return str(seqrecord.seq)

def load_from(a3m_file: str, max_size: int = 10000):
  """
  max_size: maximum sequence amount in each file
  """

  # with open(a3m_file, "r") as h:
    # return list(mit.take(2*max_size, h)) if max_size != -1 else list(h)
  if max_size != -1:
    return list(bufferedAlignReader(a3m_file, max_size))
  else:
    with open(a3m_file, "r") as h:
      return h.readlines()

def sequence2array(seq: str):
  return np.frombuffer(seq.encode(), dtype=np.uint8)

def build_array(seqs: T.List[str], shuffle: bool = False):
  nlen = len(seqs[0])
  seqarys = [sequence2array(x) for x in seqs
             if len(x) == nlen]
  if shuffle: random.shuffle(seqarys)
  return seqarys

def encoder(a3m_lines: T.List[str]) -> T.List[np.ndarray]:
  """
  each elements denote all the lines from a a3m file
  0, 2, 4, 6, ... is the sequence names
  1, 3, 5, 7, ... is the sequence content
  """
  # seqs = [line.translate(NOLOWER).rstrip()
  #         for line in a3m_lines
  #         if not (line.startswith(">") or "\x00" in line)]
  seqs = [a3m_lines[i].translate(NOLOWER).rstrip()
          for i in range(1, len(a3m_lines),2)]
  # m = np.array([sequence2array(x) for x in seqs])
  m = np.array(build_array(seqs))
  return MAPPING[m]

class MSAEncoder(nn.Module):
    """
    """
    num_embeddings = 21
    def __init__(self, embedding_dim : int, 
                 encoding_strategy : str = "emb_plus_one_hot"):
        super(MSAEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.encoding_strategy = encoding_strategy
        self.emb_layer = nn.Embedding(num_embeddings=self.num_embeddings, 
                                  embedding_dim=self.embedding_dim)

        assert encoding_strategy in ["one_hot", "emb", 
                                     "emb_plus_one_hot"],\
            f"the encoding strategy {encoding_strategy} is not implemented"
        
    
    def forward(self, input : th.Tensor) -> th.Tensor:
        x: th.Tensor
        if self.encoding_strategy == "one_hot":
            x = F.one_hot(input, num_classes=self.embedding_dim)
        else:
            x = self.emb_layer(input)

        if self.encoding_strategy == "emb_plus_one_hot":
            return x + F.one_hot(input, num_classes=self.embedding_dim)
        else:
            return x.float()
        
class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: T.Optional[int] = 1):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        assert isinstance(self.padding_idx, int)
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

if __name__ == "__main__":
  import time 
  import torch.multiprocessing as mp
  import argparse as P

  mp.set_sharing_strategy("file_system")
  parser = P.ArgumentParser()

  parser.add_argument("dataset")
  parser.add_argument("msa")

  parser.add_argument("-m", "--mode")
  parser.add_argument("-t", "--task")
  parser.add_argument("-c", "--num-classes", type=int)
  parser.add_argument("-s", "--msa-buffer-size", type=int, default=2000)
  parser.add_argument("-l", "--MAXLEN",type=int, default=1000)
  parser.add_argument("-b", "--batch-size", type=int, default=32)
  parser.add_argument("-e", "--epochs", type=int, default=1)
  parser.add_argument("--max-msa-size", type=int, default=-1)
  parser.add_argument("--num-workers", type=int, default=10)
  parser.add_argument("--not-shuffle", action="store_true")
  parser.add_argument(
      "--msa-file-cache-size",
      type=int,
      default=0,
      help="每个 DataLoader worker 缓存的 MSA 文件数；0=不缓存，-1=缓存访问过的全部"
  )

  parser.add_argument(
      "--msa-cache-dtype",
      type=str,
      default="uint8",
      choices=["uint8", "long"],
      help="uint8 更省内存；long 更快但占用约 8 倍内存"
  )

  opt = parser.parse_args()
  print(opt)

  msa_dataset = MSADataset(
      opt.dataset,
      opt.msa,
      opt.mode,
      opt.task,
      opt.num_classes,
      opt.msa_buffer_size,
      opt.MAXLEN,
      msa_max_size=opt.max_msa_size,
      msa_file_cache_size=opt.msa_file_cache_size,
      msa_cache_dtype=opt.msa_cache_dtype,
  )

  loader_kwargs = dict(
      dataset=msa_dataset,
      batch_size=opt.batch_size,
      num_workers=opt.num_workers,
      shuffle=not opt.not_shuffle,
      pin_memory=True,
  )

  if opt.num_workers > 0:
      loader_kwargs["persistent_workers"] = opt.msa_file_cache_size != 0
      loader_kwargs["prefetch_factor"] = 2

  msa_loader = td.DataLoader(**loader_kwargs)
  
  Epochs = opt.epochs
  for epoch in range(Epochs):
    st = time.time()
    for i, (X, y) in enumerate(msa_loader):
      ed = time.time()
      print(f"Epoch [{epoch+1}/{Epochs}]: consumed {ed-st}s for item {i}")
      print(f"X shape {X.shape}, y shape {y.shape}")
      print(np.where(y)[0].shape)
      st = time.time()