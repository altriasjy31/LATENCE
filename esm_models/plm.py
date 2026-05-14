import os
import re
import shutil
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from Bio import SeqIO
from tqdm import tqdm
import torch
from esm import FastaBatchedDataset, pretrained
import esm
import numpy as np

from settings import settings_dict as settings


_FASTA_LINE_WIDTH = 60


def _write_fasta_record(handle, label: str, sequence: str) -> None:
    """Write one FASTA record using only record.id as the label.

    This matches the original implementation, where output files are named
    exactly as <record.id>.npy and descriptions after whitespace are discarded.
    """
    handle.write(f">{label}\n")
    for i in range(0, len(sequence), _FASTA_LINE_WIDTH):
        handle.write(sequence[i:i + _FASTA_LINE_WIDTH] + "\n")


def _atomic_np_save(path: str, obj: Any, allow_pickle: bool = True) -> None:
    """Atomically save a .npy file to avoid half-written cache files.

    The temporary file deliberately does not end with .npy so that cache scans
    do not confuse abandoned temporary files with completed features.
    """
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as fh:
        np.save(fh, obj, allow_pickle=allow_pickle)
    os.replace(tmp_path, path)


def _cast_array(arr: Any, save_dtype: str) -> np.ndarray:
    arr = np.asarray(arr)
    if save_dtype and save_dtype.lower() != "auto":
        arr = arr.astype(np.dtype(save_dtype), copy=False)
    return arr


def _load_esm_model(model_path: Optional[str], model_name: Optional[str]):
    if model_path and os.path.exists(model_path):
        return pretrained.load_model_and_alphabet(model_path)
    if model_name:
        if model_name == "esm2_t33_650M_UR50D":
            return esm.pretrained.esm2_t33_650M_UR50D()
        if model_name == "esm2_t36_3B_UR50D":
            return esm.pretrained.esm2_t36_3B_UR50D()
        raise NotImplementedError(f"model {model_name} is not implemented")
    raise ValueError("model_path or model_name must be provided")


def _esm_extract_worker(config: Dict[str, Any]) -> Dict[str, Any]:
    """Spawn-safe worker entry point for one FASTA shard on one CUDA device."""
    rank = int(config["rank"])
    gpu_id = int(config["gpu_id"])

    torch.cuda.set_device(gpu_id)

    embedder = PlmEmbed(
        fasta_file=config["fasta_file"],
        working_dir=config["working_dir"],
        model_name=config["model_name"],
        model_path=config["model_path"],
        use_gpu=True,
        repr_layers=config["repr_layers"],
        include=config["include"],
        cache_dir=config["output_dir"],
    )
    embedder.esm_extract(
        fasta_file=config["fasta_file"],
        repr_layers=config["repr_layers"],
        model_path=config["model_path"],
        model_name=config["model_name"],
        use_gpu=True,
        truncate=config["truncate"],
        include=config["include"],
        batch_size=config["batch_size"],
        output_dir=config["output_dir"],
        overwrite=True,  # shards are already filtered in the parent process
        device_id=gpu_id,
        loader_workers=config["loader_workers"],
        save_dtype=config["save_dtype"],
        esm_fp16=config["esm_fp16"],
        allow_tf32=config["allow_tf32"],
        progress_position=rank,
        progress_disable=config["progress_disable"],
    )
    return {
        "rank": rank,
        "gpu_id": gpu_id,
        "count": int(config["count"]),
        "token_load": int(config["token_load"]),
        "fasta_file": config["fasta_file"],
    }


class PlmEmbed:

    def __init__(
        self,
        fasta_file: str,
        working_dir: str,
        model_name: str = "esm2_t36_3B_UR50D",
        model_path: str = settings['esm3b_path'],
        use_gpu: bool = True,
        repr_layers: list = [34, 35, 36],
        include: list = ["mean"],
        cache_dir: str = None,
    ):
        working_dir = os.path.abspath(working_dir)
        os.makedirs(working_dir, exist_ok=True)

        if cache_dir is None:
            cache_dir = os.path.join(working_dir, "embed_feature")
        os.makedirs(cache_dir, exist_ok=True)

        self.working_dir = working_dir
        self.cache_dir = os.path.abspath(cache_dir)
        self.use_gpu = use_gpu
        self.fasta_file = os.path.abspath(fasta_file)
        # Kept for compatibility with the original misspelled attribute.
        self.filltered_fasta_file = os.path.join(working_dir, 'filtered.fasta')
        self.filtered_fasta_file = self.filltered_fasta_file
        self.repr_layers = repr_layers
        self.model_path = model_path
        self.model_name = model_name
        self.include = include

    def parse_fasta(self, fasta_file=None) -> dict:
        '''
        parse fasta file

        args:
            fasta_file: fasta file path
        return:
            fasta_dict: fasta dictionary {id: sequence}
        '''
        if fasta_file is None:
            fasta_file = self.fasta_file

        fasta_dict = {}
        duplicated = []
        for record in SeqIO.parse(fasta_file, 'fasta'):
            rid = str(record.id)
            if rid in fasta_dict:
                duplicated.append(rid)
            fasta_dict[rid] = str(record.seq)

        if duplicated:
            examples = ', '.join(duplicated[:10])
            raise ValueError(f'Duplicate FASTA IDs found, e.g. {examples}. Please make IDs unique first.')
        return fasta_dict

    @staticmethod
    def _processed_ids(cache_dir: str) -> set:
        processed_ids = set()
        if not os.path.exists(cache_dir):
            return processed_ids
        for entry in os.scandir(cache_dir):
            if not entry.is_file():
                continue
            file_name = entry.name
            # Only completed feature files are counted. Atomic-save temporary
            # files are intentionally ignored.
            if file_name.endswith('.npy') and '.tmp.' not in file_name:
                processed_ids.add(file_name[:-4])
        return processed_ids

    @staticmethod
    def _resolve_cuda_devices(devices: Optional[Union[str, Sequence[int]]] = None) -> List[int]:
        if not torch.cuda.is_available():
            return []
        n_visible = torch.cuda.device_count()
        if devices is None:
            return list(range(n_visible))
        if isinstance(devices, str):
            text = devices.strip().lower()
            if text in {'', 'all', 'auto'}:
                return list(range(n_visible))
            ids = [int(x) for x in re.split(r'[,:;\s]+', text) if x != '']
        else:
            ids = [int(x) for x in devices]

        invalid = [i for i in ids if i < 0 or i >= n_visible]
        if invalid:
            raise ValueError(
                f'Invalid CUDA device IDs {invalid}. Only {n_visible} visible device(s) are available. '
                'When CUDA_VISIBLE_DEVICES is set, use local visible IDs, e.g. 0,1.'
            )
        # Keep user order while removing accidental duplicates.
        unique_ids = []
        for i in ids:
            if i not in unique_ids:
                unique_ids.append(i)
        return unique_ids

    def filter_fasta(self, fasta_file=None, cache_dir=None, filltered_fasta_file=None):
        """
        Only keep FASTA sequences that are not already present in the feature
        cache directory.
        """
        if fasta_file is None:
            fasta_file = self.fasta_file
        if cache_dir is None:
            cache_dir = self.cache_dir
        if filltered_fasta_file is None:
            filltered_fasta_file = self.filltered_fasta_file

        os.makedirs(os.path.dirname(os.path.abspath(filltered_fasta_file)), exist_ok=True)
        processed_ids = self._processed_ids(cache_dir)
        filtered_fasta_dict = {}
        seen = set()
        duplicated = []

        with open(filltered_fasta_file, 'w') as f:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                rid = str(record.id)
                if rid in seen:
                    duplicated.append(rid)
                    continue
                seen.add(rid)
                if rid in processed_ids:
                    continue
                seq = str(record.seq)
                filtered_fasta_dict[rid] = seq
                _write_fasta_record(f, rid, seq)

        if duplicated:
            examples = ', '.join(duplicated[:10])
            raise ValueError(f'Duplicate FASTA IDs found, e.g. {examples}. Please make IDs unique first.')

        return filtered_fasta_dict, filltered_fasta_file

    def _write_filtered_shards(
        self,
        fasta_file: str,
        cache_dir: str,
        shard_dir: str,
        num_shards: int,
        overwrite: bool = False,
        truncate_tokens: int = 1022,
    ) -> List[Dict[str, Any]]:
        """Filter already-cached entries and split remaining FASTA by token load."""
        if num_shards <= 0:
            raise ValueError('num_shards must be positive')

        os.makedirs(shard_dir, exist_ok=True)
        processed_ids = set() if overwrite else self._processed_ids(cache_dir)

        shard_paths = [os.path.join(shard_dir, f'shard_{i:04d}.fasta') for i in range(num_shards)]
        handles = [open(path, 'w') for path in shard_paths]
        counts = [0 for _ in range(num_shards)]
        token_loads = [0 for _ in range(num_shards)]
        seen = set()
        duplicated = []

        try:
            for record in tqdm(SeqIO.parse(fasta_file, 'fasta'), desc='split FASTA for ESM2 GPUs', ascii=' >='):
                rid = str(record.id)
                if rid in seen:
                    duplicated.append(rid)
                    continue
                seen.add(rid)
                if rid in processed_ids:
                    continue

                seq = str(record.seq)
                shard_idx = min(range(num_shards), key=lambda i: token_loads[i])
                _write_fasta_record(handles[shard_idx], rid, seq)
                counts[shard_idx] += 1
                token_loads[shard_idx] += min(len(seq), truncate_tokens) + 1
        finally:
            for h in handles:
                h.close()

        if duplicated:
            examples = ', '.join(duplicated[:10])
            raise ValueError(f'Duplicate FASTA IDs found, e.g. {examples}. Please make IDs unique first.')

        return [
            {
                'path': shard_paths[i],
                'count': counts[i],
                'token_load': token_loads[i],
            }
            for i in range(num_shards)
        ]

    def extract(
        self,
        fasta_file: str = None,
        repr_layers: list = [34, 35, 36],
        model_path: str = None,
        model_name: str = None,
        use_gpu: bool = True,
        truncate: bool = True,
        include: list = ["mean", "per_tok", "bos", "contacts"],
        batch_size: int = 4096,
        output_dir: str = None,
        overwrite: bool = False,
        model_type: str = "esm",
        # New multi-GPU / throughput options.
        multi_gpu: bool = False,
        devices: Optional[Union[str, Sequence[int]]] = None,
        workers_per_gpu: int = 1,
        shard_dir: str = None,
        keep_shards: bool = False,
        loader_workers: int = 0,
        save_dtype: str = "float32",
        esm_fp16: bool = False,
        allow_tf32: bool = False,
        progress_disable: bool = False,
    ) -> None:

        if model_type != "esm":
            raise NotImplementedError(f"model type {model_type} is not implemented")

        if multi_gpu:
            self.esm_extract_multi_gpu(
                fasta_file=fasta_file,
                repr_layers=repr_layers,
                model_path=model_path,
                model_name=model_name,
                use_gpu=use_gpu,
                truncate=truncate,
                include=include,
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
                devices=devices,
                workers_per_gpu=workers_per_gpu,
                shard_dir=shard_dir,
                keep_shards=keep_shards,
                loader_workers=loader_workers,
                save_dtype=save_dtype,
                esm_fp16=esm_fp16,
                allow_tf32=allow_tf32,
                progress_disable=progress_disable,
            )
        else:
            self.esm_extract(
                fasta_file=fasta_file,
                repr_layers=repr_layers,
                model_path=model_path,
                model_name=model_name,
                use_gpu=use_gpu,
                truncate=truncate,
                include=include,
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
                device_id=None,
                loader_workers=loader_workers,
                save_dtype=save_dtype,
                esm_fp16=esm_fp16,
                allow_tf32=allow_tf32,
                progress_disable=progress_disable,
            )

    def esm_extract_multi_gpu(
        self,
        fasta_file: str = None,
        repr_layers: list = [34, 35, 36],
        model_path: str = None,
        model_name: str = None,
        use_gpu: bool = True,
        truncate: bool = True,
        include: list = ["mean", "per_tok", "bos", "contacts"],
        batch_size: int = 4096,
        output_dir: str = None,
        overwrite: bool = False,
        devices: Optional[Union[str, Sequence[int]]] = None,
        workers_per_gpu: int = 1,
        shard_dir: str = None,
        keep_shards: bool = False,
        loader_workers: int = 0,
        save_dtype: str = "float32",
        esm_fp16: bool = False,
        allow_tf32: bool = False,
        progress_disable: bool = False,
    ) -> None:
        if output_dir is None:
            output_dir = self.cache_dir
        if fasta_file is None:
            fasta_file = self.fasta_file
        if model_name is None:
            model_name = self.model_name
        if model_path is None:
            model_path = self.model_path
        if workers_per_gpu <= 0:
            raise ValueError('workers_per_gpu must be positive')

        os.makedirs(output_dir, exist_ok=True)

        if not use_gpu or not torch.cuda.is_available():
            print('Multi-GPU embedding requested, but CUDA is unavailable or disabled; falling back to single-process CPU extraction.')
            return self.esm_extract(
                fasta_file=fasta_file,
                repr_layers=repr_layers,
                model_path=model_path,
                model_name=model_name,
                use_gpu=False,
                truncate=truncate,
                include=include,
                batch_size=batch_size,
                output_dir=output_dir,
                overwrite=overwrite,
                loader_workers=loader_workers,
                save_dtype=save_dtype,
                esm_fp16=False,
                allow_tf32=allow_tf32,
                progress_disable=progress_disable,
            )

        device_ids = self._resolve_cuda_devices(devices)
        if len(device_ids) == 0:
            raise RuntimeError('No CUDA devices are visible.')

        worker_devices = []
        for gpu_id in device_ids:
            worker_devices.extend([gpu_id] * workers_per_gpu)

        base_shard_dir = os.path.abspath(shard_dir or os.path.join(self.working_dir, '.esm2go_embed_shards'))
        run_shard_dir = os.path.join(base_shard_dir, f'run_{os.getpid()}')
        if os.path.exists(run_shard_dir):
            shutil.rmtree(run_shard_dir)
        os.makedirs(run_shard_dir, exist_ok=True)

        shard_infos = self._write_filtered_shards(
            fasta_file=fasta_file,
            cache_dir=output_dir,
            shard_dir=run_shard_dir,
            num_shards=len(worker_devices),
            overwrite=overwrite,
            truncate_tokens=1022 if truncate else 10 ** 9,
        )
        total_to_process = sum(info['count'] for info in shard_infos)
        if total_to_process == 0:
            print('### esm_extract_multi_gpu ###')
            print('All sequences are already processed in the feature cache directory. Return.')
            if not keep_shards:
                shutil.rmtree(run_shard_dir, ignore_errors=True)
            return

        active_configs = []
        for rank, info in enumerate(shard_infos):
            if info['count'] == 0:
                continue
            active_configs.append({
                'rank': len(active_configs),
                'gpu_id': worker_devices[rank],
                'fasta_file': info['path'],
                'count': info['count'],
                'token_load': info['token_load'],
                'working_dir': self.working_dir,
                'repr_layers': repr_layers,
                'model_path': model_path,
                'model_name': model_name,
                'truncate': truncate,
                'include': include,
                'batch_size': batch_size,
                'output_dir': output_dir,
                'loader_workers': loader_workers,
                'save_dtype': save_dtype,
                'esm_fp16': esm_fp16,
                'allow_tf32': allow_tf32,
                'progress_disable': progress_disable,
            })

        print(
            f'Extracting ESM2 features on {len(device_ids)} GPU(s), '
            f'{len(active_configs)} worker process(es), {total_to_process} unprocessed sequence(s).'
        )
        print(f'Visible CUDA device IDs used for ESM2: {device_ids}')

        ctx = mp.get_context('spawn')
        try:
            with ProcessPoolExecutor(max_workers=len(active_configs), mp_context=ctx) as executor:
                futures = [executor.submit(_esm_extract_worker, cfg) for cfg in active_configs]
                for future in as_completed(futures):
                    result = future.result()
                    print(
                        f"Finished ESM2 shard rank={result['rank']} on cuda:{result['gpu_id']} "
                        f"({result['count']} sequences, token_load={result['token_load']})."
                    )
        finally:
            if not keep_shards:
                shutil.rmtree(run_shard_dir, ignore_errors=True)

    def esm_extract(
        self,
        fasta_file: str = None,
        repr_layers: list = [34, 35, 36],
        model_path: str = None,
        model_name: str = None,
        use_gpu: bool = True,
        truncate: bool = True,
        include: list = ["mean", "per_tok", "bos", "contacts"],
        batch_size: int = 4096,
        output_dir: str = None,
        overwrite: bool = False,
        # New single-worker options.
        device_id: Optional[int] = None,
        loader_workers: int = 0,
        save_dtype: str = "float32",
        esm_fp16: bool = False,
        allow_tf32: bool = False,
        progress_position: int = 0,
        progress_disable: bool = False,
    ) -> None:

        if output_dir is None:
            output_dir = self.cache_dir
        if fasta_file is None:
            fasta_file = self.fasta_file
        if model_name is None:
            model_name = self.model_name
        if model_path is None:
            model_path = self.model_path
        include = list(include or [])
        os.makedirs(output_dir, exist_ok=True)

        # Filter FASTA only in the parent/single-worker path. Multi-GPU workers
        # receive already-filtered shard FASTAs and pass overwrite=True.
        if not overwrite:
            _, fasta_file = self.filter_fasta(fasta_file, output_dir)

        if os.path.getsize(fasta_file) == 0:
            print(f'### {self.extract.__name__} ###')
            print('All sequences are already processed in the default cache directory. Return.')
            return

        if allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model, alphabet = _load_esm_model(model_path, model_name)
        model.eval()

        if torch.cuda.is_available() and use_gpu:
            if device_id is None:
                device_id = 0
            device = torch.device(f'cuda:{int(device_id)}')
            torch.cuda.set_device(device)
            if esm_fp16:
                model = model.half()
            model = model.to(device)
            print(f'Transferred ESM2 model to {device}')
        else:
            device = torch.device('cpu')
            if esm_fp16:
                print('esm_fp16 was requested but CUDA is not used; keeping ESM2 model in float32 on CPU.')
            print('Using CPU')

        dataset = FastaBatchedDataset.from_file(fasta_file)
        batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(),
            batch_sampler=batches,
            num_workers=loader_workers,
            pin_memory=(device.type == 'cuda'),
        )

        assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
        repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

        pooled_only = set(include).issubset({'mean', 'sum', 'max', 'min', 'bos'})

        with torch.inference_mode():
            iterator = tqdm(
                enumerate(data_loader),
                total=len(batches),
                desc=f'Extracting ESM2 features {device}',
                ascii=' >=',
                position=progress_position,
                disable=progress_disable,
            )
            for batch_idx, (labels, strs, toks) in iterator:
                if truncate:
                    toks = toks[:, :1022]
                if device.type == 'cuda':
                    toks = toks.to(device=device, non_blocking=True)

                out = model(toks, repr_layers=repr_layers, return_contacts=('contacts' in include))

                if pooled_only:
                    effective_lengths = [
                        max(0, min(len(s), out['representations'][repr_layers[0]].shape[1] - 1))
                        for s in strs
                    ]
                    length_tensor = torch.as_tensor(effective_lengths, device=device, dtype=torch.long)
                    length_for_div = length_tensor.clamp_min(1).to(dtype=torch.float32).unsqueeze(1)
                    max_len = int(out['representations'][repr_layers[0]].shape[1] - 1)
                    mask = torch.arange(max_len, device=device).unsqueeze(0) < length_tensor.unsqueeze(1)
                    mask3 = mask.unsqueeze(-1)

                    pooled: Dict[str, Dict[int, np.ndarray]] = {k: {} for k in include}
                    for layer, rep in out['representations'].items():
                        if 'bos' in include:
                            pooled['bos'][layer] = _cast_array(rep[:, 0, :].detach().cpu().numpy(), save_dtype)

                        need_token_pool = any(k in include for k in ('mean', 'sum', 'max', 'min'))
                        if need_token_pool:
                            token_rep = rep[:, 1:1 + max_len, :]
                            # Pool in float32 for numerical stability if the ESM model
                            # was explicitly converted to half precision.
                            if token_rep.dtype in (torch.float16, torch.bfloat16):
                                token_calc = token_rep.float()
                            else:
                                token_calc = token_rep

                            if 'sum' in include or 'mean' in include:
                                sums = (token_calc * mask3).sum(dim=1)
                                if 'sum' in include:
                                    pooled['sum'][layer] = _cast_array(sums.detach().cpu().numpy(), save_dtype)
                                if 'mean' in include:
                                    means = sums / length_for_div
                                    pooled['mean'][layer] = _cast_array(means.detach().cpu().numpy(), save_dtype)

                            if 'max' in include:
                                neg_inf = torch.finfo(token_calc.dtype).min
                                max_vals = token_calc.masked_fill(~mask3, neg_inf).max(dim=1).values
                                pooled['max'][layer] = _cast_array(max_vals.detach().cpu().numpy(), save_dtype)

                            if 'min' in include:
                                pos_inf = torch.finfo(token_calc.dtype).max
                                min_vals = token_calc.masked_fill(~mask3, pos_inf).min(dim=1).values
                                pooled['min'][layer] = _cast_array(min_vals.detach().cpu().numpy(), save_dtype)

                    for i, label in enumerate(labels):
                        out_file = os.path.join(output_dir, f"{label}.npy")
                        if (not overwrite) and os.path.exists(out_file):
                            continue
                        result = {"name": label}
                        for key in include:
                            result[key] = {layer: pooled[key][layer][i] for layer in repr_layers}
                        _atomic_np_save(out_file, result, allow_pickle=True)

                else:
                    # Compatibility fallback for per-token/contact outputs. This
                    # path preserves the original output structure but is slower
                    # and transfers full token representations to CPU.
                    representations = {
                        layer: t.detach().to(device='cpu') for layer, t in out['representations'].items()
                    }
                    if 'contacts' in include:
                        contacts = out['contacts'].detach().to(device='cpu')

                    for i, label in enumerate(labels):
                        out_file = os.path.join(output_dir, f"{label}.npy")
                        if (not overwrite) and os.path.exists(out_file):
                            continue

                        result = {"name": label}
                        seq_len = min(len(strs[i]), representations[repr_layers[0]].shape[1] - 1)

                        if 'per_tok' in include:
                            result['per_tok'] = {
                                layer: _cast_array(t[i, 1:seq_len + 1].clone().numpy(), save_dtype)
                                for layer, t in representations.items()
                            }

                        if 'mean' in include:
                            result['mean'] = {
                                layer: _cast_array(t[i, 1:seq_len + 1].mean(0).clone().numpy(), save_dtype)
                                for layer, t in representations.items()
                            }

                        if 'bos' in include:
                            result['bos'] = {
                                layer: _cast_array(t[i, 0].clone().numpy(), save_dtype)
                                for layer, t in representations.items()
                            }

                        if 'contacts' in include:
                            result['contacts'] = _cast_array(
                                contacts[i, :seq_len, :seq_len].clone().numpy(),
                                save_dtype,
                            )

                        if 'sum' in include:
                            result['sum'] = {
                                layer: _cast_array(t[i, 1:seq_len + 1].sum(0).clone().numpy(), save_dtype)
                                for layer, t in representations.items()
                            }

                        if 'max' in include:
                            result['max'] = {
                                layer: _cast_array(t[i, 1:seq_len + 1].max(0).values.clone().numpy(), save_dtype)
                                for layer, t in representations.items()
                            }

                        if 'min' in include:
                            result['min'] = {
                                layer: _cast_array(t[i, 1:seq_len + 1].min(0).values.clone().numpy(), save_dtype)
                                for layer, t in representations.items()
                            }

                        _atomic_np_save(out_file, result, allow_pickle=True)

                del out
                if device.type == 'cuda' and (batch_idx + 1) % 64 == 0:
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_file', type=str, help='path of fasta file')
    parser.add_argument('workdir', type=str, help='path of working directory')
    parser.add_argument('-mn', '--model_name', type=str, help='name of model', default="esm2_t36_3B_UR50D")
    parser.add_argument('-mp', '--model_path', type=str, help='path of model', default=None)
    parser.add_argument('-c', '--cache_dir', type=str, help='path of cache directory', default=None)
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('--multi_gpu', action='store_true', help='split FASTA and extract ESM2 features on multiple visible CUDA devices')
    parser.add_argument('--devices', type=str, default='all', help='CUDA visible device IDs for multi-GPU extraction, e.g. all or 0,1,2,3')
    parser.add_argument('--workers_per_gpu', type=int, default=1, help='number of ESM2 worker processes per GPU; keep 1 for esm2_t36_3B unless you have ample VRAM')
    parser.add_argument('--batch_size', type=int, default=4096, help='ESM token batch size per worker')
    parser.add_argument('--loader_workers', type=int, default=0, help='DataLoader workers per ESM2 process')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing feature files')
    parser.add_argument('--keep_shards', action='store_true', help='keep temporary FASTA shards for debugging')
    parser.add_argument('--save_dtype', type=str, default='float32', choices=['float32', 'float16', 'auto'], help='dtype used for saved feature arrays')
    parser.add_argument('--esm_fp16', action='store_true', help='run ESM2 in float16 on CUDA; faster/lower memory but features are not bitwise identical to float32')
    parser.add_argument('--allow_tf32', action='store_true', help='allow TF32 matmul on CUDA; faster on Ampere+ but not bitwise identical')
    parser.add_argument('--include', type=str, nargs='+', default=['mean'], choices=['mean', 'per_tok', 'bos', 'contacts', 'sum', 'max', 'min'], help='which representations to return')
    parser.add_argument('--repr_layers', type=int, nargs='+', default=[-3, -2, -1], help='which layers to extract; default [-3, -2, -1]')
    args = parser.parse_args()

    plm = PlmEmbed(
        args.fasta_file,
        args.workdir,
        model_name=args.model_name,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        use_gpu=args.use_gpu,
        include=args.include,
        repr_layers=args.repr_layers,
    )
    plm.extract(
        fasta_file=plm.fasta_file,
        repr_layers=plm.repr_layers,
        model_path=plm.model_path,
        model_name=plm.model_name,
        use_gpu=plm.use_gpu,
        include=plm.include,
        output_dir=plm.cache_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        multi_gpu=args.multi_gpu,
        devices=args.devices,
        workers_per_gpu=args.workers_per_gpu,
        keep_shards=args.keep_shards,
        loader_workers=args.loader_workers,
        save_dtype=args.save_dtype,
        esm_fp16=args.esm_fp16,
        allow_tf32=args.allow_tf32,
    )
