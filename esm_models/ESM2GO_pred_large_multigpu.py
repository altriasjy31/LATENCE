import os
import argparse
import csv
import gc
import json

import torch
from Bio import SeqIO
import numpy as np
import scipy.sparse as ssp
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import ESM2GODataset, ESM2ResNet
from model_utils import Predictor
from obo_tools import ObOTools
from plm import PlmEmbed
from settings import settings_dict as settings

# the following package is from local
import obo_tools
oboTools = obo_tools.ObOTools(
    go_obo=settings['obo_file'],
    obo_pkl=settings['obo_pkl_file']
)


class ESM2GO_pipeline:
    def __init__(
        self,
        working_dir: str,
        fasta_file: str,
        pred_batch_size: int = 512,
        pred_chunk_size: int = 1024,
        device: str = 'cuda',
        top_terms: int = 500,
        aspects: list = ['BPO', 'CCO', 'MFO'],
        cache_dir: str = None,
        save_numpy: bool = True,
        save_topk_numpy: bool = True,
        numpy_dir: str = None,
        write_tsv: bool = True,
        score_threshold: float = 0.001,
        back_prop: bool = False,
        ## the following parameters should be fixed if you want to use the pretrained model
        repr_layers: list = [34, 35, 36],
        embed_batch_size: int = 4096,
        embed_multi_gpu: bool = False,
        embed_devices: str = 'all',
        embed_workers_per_gpu: int = 1,
        embed_loader_workers: int = 0,
        embed_save_dtype: str = 'float32',
        embed_keep_shards: bool = False,
        esm_fp16: bool = False,
        esm_allow_tf32: bool = False,
        embed_model_name: str = "esm2_t36_3B_UR50D",
        embed_model_path: str = settings['esm3b_path'],
        include: list = ['mean'],
        model_dir: str = settings['MODEL_CHECKPOINT_DIR'],
    ) -> None:
        if pred_batch_size <= 0:
            raise ValueError('pred_batch_size must be positive')
        if pred_chunk_size <= 0:
            raise ValueError('pred_chunk_size must be positive')
        if top_terms <= 0:
            raise ValueError('top_terms must be positive')
        if embed_workers_per_gpu <= 0:
            raise ValueError('embed_workers_per_gpu must be positive')
        if embed_loader_workers < 0:
            raise ValueError('embed_loader_workers must be non-negative')

        self.working_dir = os.path.abspath(working_dir)
        self.fasta_file = os.path.abspath(fasta_file)
        self.pred_batch_size = pred_batch_size
        self.pred_chunk_size = pred_chunk_size

        if not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.top_terms = top_terms
        self.aspects = aspects
        self.cache_dir = cache_dir

        self.save_numpy = save_numpy
        self.save_topk_numpy = save_topk_numpy
        self.numpy_dir = os.path.abspath(numpy_dir) if numpy_dir else os.path.join(self.working_dir, 'ESM2GO_numpy')
        self.write_tsv = write_tsv
        self.score_threshold = score_threshold
        self.back_prop = back_prop

        self.repr_layers = repr_layers
        self.embed_model_name = embed_model_name
        self.embed_model_path = embed_model_path
        self.include = include
        self.embed_batch_size = embed_batch_size
        self.embed_multi_gpu = embed_multi_gpu
        self.embed_devices = embed_devices
        self.embed_workers_per_gpu = embed_workers_per_gpu
        self.embed_loader_workers = embed_loader_workers
        self.embed_save_dtype = embed_save_dtype
        self.embed_keep_shards = embed_keep_shards
        self.esm_fp16 = esm_fp16
        self.esm_allow_tf32 = esm_allow_tf32

        self.model_dir = os.path.abspath(model_dir)
        self.result_file = os.path.join(self.working_dir, 'ESM2GO.tsv')

        os.makedirs(self.working_dir, exist_ok=True)
        if self.save_numpy or self.save_topk_numpy:
            os.makedirs(self.numpy_dir, exist_ok=True)

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
        for record in SeqIO.parse(fasta_file, 'fasta'):
            fasta_dict[record.id] = str(record.seq)
        return fasta_dict

    def parse_fasta_ids(self, fasta_file=None) -> np.ndarray:
        '''Return FASTA IDs in input order. Duplicate IDs are rejected because
        embedding cache files and output rows cannot be mapped unambiguously.
        '''
        if fasta_file is None:
            fasta_file = self.fasta_file

        ids = []
        seen = set()
        duplicated = []
        for record in SeqIO.parse(fasta_file, 'fasta'):
            rid = str(record.id)
            if rid in seen:
                duplicated.append(rid)
            seen.add(rid)
            ids.append(rid)

        if duplicated:
            examples = ', '.join(duplicated[:10])
            raise ValueError(f'Duplicate FASTA IDs found, e.g. {examples}. Please make EntryID unique first.')
        if len(ids) == 0:
            raise ValueError(f'No FASTA records found in {fasta_file}')

        return np.asarray(ids, dtype=str)

    def get_embed_features(self):
        Embed = PlmEmbed(
            fasta_file=self.fasta_file,
            working_dir=self.working_dir,
            repr_layers=self.repr_layers,
            model_name=self.embed_model_name,
            model_path=self.embed_model_path,
            use_gpu=('cuda' in self.device),
            include=self.include,
            cache_dir=self.cache_dir,
        )
        print('Extracting ESM2 embedding features')
        Embed.extract(
            fasta_file=self.fasta_file,
            model_name=self.embed_model_name,
            model_path=self.embed_model_path,
            use_gpu=('cuda' in self.device),
            repr_layers=self.repr_layers,
            include=self.include,
            batch_size=self.embed_batch_size,
            model_type='esm',
            multi_gpu=self.embed_multi_gpu,
            devices=self.embed_devices,
            workers_per_gpu=self.embed_workers_per_gpu,
            loader_workers=self.embed_loader_workers,
            save_dtype=self.embed_save_dtype,
            keep_shards=self.embed_keep_shards,
            esm_fp16=self.esm_fp16,
            allow_tf32=self.esm_allow_tf32,
        )
        feature_dir = Embed.cache_dir
        return feature_dir

    def create_name_npy(self):
        names = self.parse_fasta_ids(self.fasta_file)
        name_npy_path = os.path.join(self.working_dir, 'names.npy')
        np.save(name_npy_path, names)
        return name_npy_path

    @staticmethod
    def _normalize_ids(ids):
        normalized = []
        for x in list(ids):
            if isinstance(x, bytes):
                normalized.append(x.decode())
            else:
                normalized.append(str(x))
        return normalized

    @staticmethod
    def _to_numpy_float32(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return x

    def _align_predictions_to_expected_order(self, protein_ids, y_preds, expected_names):
        got = self._normalize_ids(protein_ids)
        expected = self._normalize_ids(expected_names)

        if len(got) != len(expected):
            raise ValueError(f'Predictor returned {len(got)} proteins, but expected {len(expected)} proteins.')

        if got == expected:
            return y_preds

        pos = {pid: i for i, pid in enumerate(got)}
        missing = [pid for pid in expected if pid not in pos]
        if missing:
            examples = ', '.join(missing[:10])
            raise ValueError(f'Predictor output is missing expected protein IDs, e.g. {examples}.')

        order = np.asarray([pos[pid] for pid in expected], dtype=np.int64)
        return y_preds[order]

    def _make_predict_loader(self, feature_dir: str, names_npy_path: str):
        predict_dataset = ESM2GODataset(
            features_dir=feature_dir,
            names_npy=names_npy_path,
            repr_layers=self.repr_layers,
            labels_npy=None,  # inference only
        )
        return DataLoader(
            predict_dataset,
            batch_size=self.pred_batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _predict_names_chunk(self, predictor: Predictor, feature_dir: str, names_chunk: np.ndarray, aspect: str):
        '''Run the official Predictor on a small name shard.

        This preserves compatibility with the local ESM2GODataset/Predictor
        implementation while preventing Predictor.predict() from materializing
        all proteins at once.
        '''
        tmp_names_path = os.path.join(self.working_dir, f'.ESM2_names_{aspect}_{os.getpid()}.npy')
        np.save(tmp_names_path, np.asarray(names_chunk, dtype=str))
        try:
            predict_loader = self._make_predict_loader(feature_dir, tmp_names_path)
            predictor.update_loader(predict_loader)
            protein_ids, y_preds = predictor.predict()
            y_preds = self._to_numpy_float32(y_preds)
            y_preds = self._align_predictions_to_expected_order(protein_ids, y_preds, names_chunk)
            return y_preds
        finally:
            try:
                os.remove(tmp_names_path)
            except FileNotFoundError:
                pass

    def _update_predictor_model(self, predictor: Predictor, model, child_matrix, child_matrix_path: str):
        '''Update Predictor without loading child_matrix unless back_prop needs it.

        The official snippet sets predictor.back_prop = False before predict(); in
        that case child_matrix is not used. Some Predictor implementations still
        expect a matrix-shaped object in update_model(), so an empty placeholder is
        attempted first. If that is rejected, we fall back to the real sparse file.
        '''
        predictor.back_prop = self.back_prop
        matrix_for_update = child_matrix
        if matrix_for_update is None and not self.back_prop:
            matrix_for_update = np.empty((0, 0), dtype=np.float32)

        try:
            predictor.update_model(model, matrix_for_update)
            predictor.back_prop = self.back_prop
            return child_matrix
        except Exception:
            if child_matrix is None and os.path.exists(child_matrix_path):
                print('Predictor.update_model rejected an empty child_matrix; loading real child_matrix_ssp.npz.')
                child_matrix = ssp.load_npz(child_matrix_path).toarray()
                predictor.update_model(model, child_matrix)
                predictor.back_prop = self.back_prop
                return child_matrix
            raise

    def _open_float16_npy_memmap(self, path: str, shape):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return np.lib.format.open_memmap(
            path,
            mode='w+',
            dtype=np.float16,
            shape=shape,
            version=(2, 0),
        )

    def _write_outputs_from_memmap(
        self,
        score_mm,
        names,
        term_list,
        term_name_list,
        aspect: str,
        result_file: str,
        top_scores_path: str = None,
        top_indices_path: str = None,
    ):
        '''Write compact top-k numpy outputs and/or TSV from a dense score memmap.

        top_scores_path: float16 .npy, shape = (n_proteins, top_k)
        top_indices_path: int32 .npy, shape = (n_proteins, top_k), indices into term_list; -1 means empty.
        '''
        n_terms = len(term_list)
        top_k = min(self.top_terms, n_terms)
        save_topk = top_scores_path is not None and top_indices_path is not None

        top_scores_mm = None
        top_indices_mm = None
        if save_topk:
            top_scores_mm = np.lib.format.open_memmap(
                top_scores_path,
                mode='w+',
                dtype=np.float16,
                shape=(len(names), top_k),
                version=(2, 0),
            )
            top_indices_mm = np.lib.format.open_memmap(
                top_indices_path,
                mode='w+',
                dtype=np.int32,
                shape=(len(names), top_k),
                version=(2, 0),
            )

        tsv_handle = None
        writer = None
        if self.write_tsv:
            tsv_handle = open(result_file, 'a', newline='')
            writer = csv.writer(tsv_handle, delimiter='\t')

        try:
            for start in tqdm(
                range(0, len(names), self.pred_chunk_size),
                desc=f'write {aspect} top-{top_k} outputs',
                ascii=' >=',
            ):
                end = min(start + self.pred_chunk_size, len(names))
                block = score_mm[start:end]
                names_block = names[start:end]

                if save_topk:
                    top_scores_block = np.full((end - start, top_k), np.nan, dtype=np.float16)
                    top_indices_block = np.full((end - start, top_k), -1, dtype=np.int32)

                for local_i, entry_id in enumerate(names_block):
                    row = block[local_i].astype(np.float32, copy=False)
                    if top_k < n_terms:
                        idx = np.argpartition(row, -top_k)[-top_k:]
                    else:
                        idx = np.arange(n_terms, dtype=np.int64)

                    idx = idx[row[idx] > self.score_threshold]
                    if idx.size > 0:
                        idx = idx[np.argsort(row[idx])[::-1]]

                    if save_topk and idx.size > 0:
                        m = min(idx.size, top_k)
                        idx_m = idx[:m]
                        top_indices_block[local_i, :m] = idx_m.astype(np.int32, copy=False)
                        top_scores_block[local_i, :m] = row[idx_m].astype(np.float16, copy=False)

                    if writer is not None and idx.size > 0:
                        entry_id = str(entry_id)
                        for j in idx:
                            writer.writerow([
                                entry_id,
                                str(term_list[j]),
                                f'{float(row[j]):.6g}',
                                aspect,
                                term_name_list[j],
                            ])

                if save_topk:
                    top_scores_mm[start:end] = top_scores_block
                    top_indices_mm[start:end] = top_indices_block
        finally:
            if tsv_handle is not None:
                tsv_handle.close()
            if top_scores_mm is not None:
                top_scores_mm.flush()
                del top_scores_mm
            if top_indices_mm is not None:
                top_indices_mm.flush()
                del top_indices_mm

    @staticmethod
    def _jsonable_seed(seed):
        if seed is None:
            return None
        try:
            return int(seed)
        except Exception:
            return str(seed)

    def predict(self, feature_dir: str):
        name_npy_path = self.create_name_npy()
        names = np.load(name_npy_path, allow_pickle=False)
        n_proteins = len(names)

        if self.write_tsv:
            columns = ['EntryID', 'term', 'score', 'aspect', 'go_term_name']
            with open(self.result_file, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(columns)

        # A placeholder avoids loading the dense GO child matrix when back_prop=False.
        predictor = Predictor(
            model=None,
            PredictLoader=None,
            device=self.device,
            child_matrix=np.empty((0, 0), dtype=np.float32),
            back_prop=self.back_prop,
        )

        seeds_dict = {}
        for aspect in self.aspects:
            aspect_model_dir = os.path.join(self.model_dir, aspect)
            if not os.path.exists(aspect_model_dir):
                print(f'No model found for {aspect}')
                continue

            models = sorted(
                os.path.join(aspect_model_dir, model)
                for model in os.listdir(aspect_model_dir)
                if model.endswith('.pt')
            )
            if len(models) == 0:
                print(f'No model found in {aspect_model_dir}')
                continue

            probe_model = ESM2ResNet.load_config(models[0])
            term_list = list(probe_model.go_term_list)
            del probe_model
            n_terms = len(term_list)

            print(f'Predicting {aspect}: {n_proteins} proteins x {n_terms} GO terms, {len(models)} model(s)')

            term_name_list = [oboTools.goID2name(str(t)) or '' for t in term_list]
            seeds_dict[aspect] = {}

            if self.save_numpy:
                score_npy_path = os.path.join(self.numpy_dir, f'{aspect}.scores.float16.npy')
            else:
                score_npy_path = os.path.join(self.working_dir, f'.{aspect}.scores.tmp.float16.npy')

            if self.save_numpy or self.save_topk_numpy:
                terms_npy_path = os.path.join(self.numpy_dir, f'{aspect}.terms.npy')
                np.save(terms_npy_path, np.asarray(term_list, dtype=str))
            else:
                terms_npy_path = None

            meta_json_path = os.path.join(self.numpy_dir, f'{aspect}.scores.float16.json') if (self.save_numpy or self.save_topk_numpy) else None
            top_scores_npy_path = os.path.join(self.numpy_dir, f'{aspect}.top{self.top_terms}.scores.float16.npy') if self.save_topk_numpy else None
            top_indices_npy_path = os.path.join(self.numpy_dir, f'{aspect}.top{self.top_terms}.indices.int32.npy') if self.save_topk_numpy else None

            score_mm = self._open_float16_npy_memmap(
                score_npy_path,
                shape=(n_proteins, n_terms),
            )

            child_matrix_path = os.path.join(aspect_model_dir, 'child_matrix_ssp.npz')
            child_matrix = None
            if self.back_prop:
                child_matrix = ssp.load_npz(child_matrix_path).toarray()

            for model_idx, model_path in enumerate(tqdm(models, desc=f'generate {aspect} prediction', ascii=' >=')):
                model = ESM2ResNet.load_config(model_path)
                if list(model.go_term_list) != term_list:
                    raise ValueError(f'go_term_list mismatch in model {model_path}')

                seed = getattr(model, 'seed', None)
                seeds_dict[aspect][model_path] = self._jsonable_seed(seed)

                model = model.to(self.device)
                child_matrix = self._update_predictor_model(predictor, model, child_matrix, child_matrix_path)

                chunk_iter = range(0, n_proteins, self.pred_chunk_size)
                for chunk_no, start in enumerate(tqdm(chunk_iter, desc=f'{aspect} chunks', leave=False, ascii=' >=')):
                    end = min(start + self.pred_chunk_size, n_proteins)
                    names_chunk = names[start:end]
                    y_preds = self._predict_names_chunk(predictor, feature_dir, names_chunk, aspect)

                    expected_shape = (end - start, n_terms)
                    if y_preds.shape != expected_shape:
                        raise ValueError(
                            f'Prediction shape mismatch for {aspect}, {model_path}, rows {start}:{end}. '
                            f'Expected {expected_shape}, got {y_preds.shape}.'
                        )

                    if model_idx == 0:
                        score_mm[start:end] = y_preds.astype(np.float16, copy=False)
                    else:
                        # Running average: avg_k = (avg_{k-1} * k + pred_k) / (k + 1)
                        avg = score_mm[start:end].astype(np.float32, copy=True)
                        avg *= float(model_idx)
                        avg += y_preds
                        avg /= float(model_idx + 1)
                        score_mm[start:end] = avg.astype(np.float16, copy=False)

                    del y_preds
                    if (chunk_no + 1) % 32 == 0:
                        score_mm.flush()

                score_mm.flush()
                del model
                gc.collect()
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()

            if self.write_tsv or self.save_topk_numpy:
                self._write_outputs_from_memmap(
                    score_mm=score_mm,
                    names=names,
                    term_list=term_list,
                    term_name_list=term_name_list,
                    aspect=aspect,
                    result_file=self.result_file,
                    top_scores_path=top_scores_npy_path,
                    top_indices_path=top_indices_npy_path,
                )

            score_mm.flush()
            del score_mm

            if self.save_numpy or self.save_topk_numpy:
                meta = {
                    'aspect': aspect,
                    'dtype': 'float16',
                    'shape': [int(n_proteins), int(n_terms)],
                    'dense_score_npy': score_npy_path if self.save_numpy else None,
                    'top_scores_npy': top_scores_npy_path,
                    'top_indices_npy': top_indices_npy_path,
                    'protein_names_npy': name_npy_path,
                    'terms_npy': terms_npy_path,
                    'row_order': 'Rows are in protein_names_npy order.',
                    'column_order': 'Columns are in terms_npy order.',
                    'ensemble_models': models,
                    'model_seeds': seeds_dict[aspect],
                    'score_threshold_for_tsv': self.score_threshold,
                    'top_terms_for_tsv': self.top_terms,
                }
                with open(meta_json_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                if self.save_numpy:
                    print(f'Saved {aspect} dense float16 numpy scores to {score_npy_path}')
                if self.save_topk_numpy:
                    print(f'Saved {aspect} compact top-k numpy scores to {top_scores_npy_path} and {top_indices_npy_path}')

            if not self.save_numpy and os.path.exists(score_npy_path):
                os.remove(score_npy_path)

        seeds_path = os.path.join(self.working_dir, 'ESM2GO_model_seeds.json')
        with open(seeds_path, 'w') as f:
            json.dump(seeds_dict, f, indent=2)

    def parent_propagation(self, df: pd.DataFrame):
        '''
        propagate the prediction to the parent terms
        df.columns = ['EntryID', 'term', 'score']
        '''
        df_dict = df.groupby('EntryID').apply(lambda x: x.set_index('term')['score'].to_dict()).to_dict()

        result_dict = {}
        for EntryID, term_score in tqdm(df_dict.items(), desc='propagate prediction', ascii=' >='):
            result_dict[EntryID] = oboTools.backprop_cscore(term_score, min_cscore=0.001)

        rows = []
        for EntryID, terms_scores in result_dict.items():
            for term, score in terms_scores.items():
                rows.append({'EntryID': EntryID, 'term': term, 'score': score})

        result_df = pd.DataFrame(rows)
        return result_df

    def main(self):
        feature_dir = self.get_embed_features()
        self.predict(feature_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # example usage: python ESM2GO_pred_large.py -w example -f example/example.fasta --use_gpu
    parser.add_argument('-w', '--working_dir', type=str, help='working directory', required=True)
    parser.add_argument('-f', '--fasta_file', type=str, help='fasta file', required=True)
    parser.add_argument('-top', '--top_terms', type=int, help='number of top terms to be kept in the TSV prediction', default=500)
    parser.add_argument('-m', '--model_dir', type=str, help='model directory', default=settings['MODEL_CHECKPOINT_DIR'])
    parser.add_argument('--esm_path', type=str, help='esm model path', default=settings['esm3b_path'])
    parser.add_argument('--use_gpu', action='store_true', help='use gpu')
    parser.add_argument('--aspect', type=str, nargs='+', default=['BPO', 'CCO', 'MFO'], choices=['BPO', 'CCO', 'MFO'], help='aspects of model to predict')
    parser.add_argument('--cache_dir', type=str, help='cache directory', default=None)

    parser.add_argument('--pred_batch_size', type=int, default=512, help='DataLoader batch size used by Predictor')
    parser.add_argument('--pred_chunk_size', type=int, default=1024, help='number of proteins materialized per Predictor.predict() call')
    parser.add_argument('--embed_batch_size', type=int, default=4096, help='ESM token batch size per ESM2 worker process')
    parser.add_argument('--multi_gpu_embed', action='store_true', help='split FASTA and extract ESM2 features on multiple visible CUDA devices')
    parser.add_argument('--embed_devices', type=str, default='all', help='CUDA visible device IDs for ESM2 extraction, e.g. all or 0,1,2,3')
    parser.add_argument('--embed_workers_per_gpu', type=int, default=1, help='number of ESM2 worker processes per GPU; keep 1 for esm2_t36_3B unless VRAM is ample')
    parser.add_argument('--embed_loader_workers', type=int, default=0, help='DataLoader workers inside each ESM2 process')
    parser.add_argument('--embed_save_dtype', type=str, default='float32', choices=['float32', 'float16', 'auto'], help='dtype used for saved ESM2 feature arrays')
    parser.add_argument('--keep_embed_shards', action='store_true', help='keep temporary ESM2 FASTA shards for debugging')
    parser.add_argument('--esm_fp16', action='store_true', help='run ESM2 in float16 on CUDA; faster/lower memory but features are not bitwise identical to float32')
    parser.add_argument('--esm_allow_tf32', action='store_true', help='allow TF32 matmul on CUDA; faster on Ampere+ but not bitwise identical')
    parser.add_argument('--numpy_dir', type=str, default=None, help='directory for float16 numpy score matrices')
    parser.add_argument('--score_threshold', type=float, default=0.001, help='minimum score kept in TSV output')
    parser.add_argument('--no_save_numpy', action='store_true', help='do not keep per-aspect dense float16 .npy score matrices')
    parser.add_argument('--no_topk_numpy', action='store_true', help='do not save compact top-k numpy arrays')
    parser.add_argument('--no_tsv', action='store_true', help='do not write ESM2GO.tsv top-k output')
    parser.add_argument('--back_prop', action='store_true', help='enable Predictor back-propagation with child_matrix_ssp.npz; disabled by default')

    args = parser.parse_args()
    working_dir = os.path.abspath(args.working_dir)
    fasta_file = os.path.abspath(args.fasta_file)
    model_dir = os.path.abspath(args.model_dir)
    esm_path = os.path.abspath(args.esm_path)
    device = 'cuda' if args.use_gpu else 'cpu'

    ESM2GO_pipeline(
        working_dir=working_dir,
        fasta_file=fasta_file,
        device=device,
        top_terms=args.top_terms,
        aspects=args.aspect,
        model_dir=model_dir,
        embed_model_path=esm_path,
        cache_dir=args.cache_dir,
        pred_batch_size=args.pred_batch_size,
        pred_chunk_size=args.pred_chunk_size,
        embed_batch_size=args.embed_batch_size,
        embed_multi_gpu=args.multi_gpu_embed,
        embed_devices=args.embed_devices,
        embed_workers_per_gpu=args.embed_workers_per_gpu,
        embed_loader_workers=args.embed_loader_workers,
        embed_save_dtype=args.embed_save_dtype,
        embed_keep_shards=args.keep_embed_shards,
        esm_fp16=args.esm_fp16,
        esm_allow_tf32=args.esm_allow_tf32,
        save_numpy=not args.no_save_numpy,
        save_topk_numpy=not args.no_topk_numpy,
        numpy_dir=args.numpy_dir,
        write_tsv=not args.no_tsv,
        score_threshold=args.score_threshold,
        back_prop=args.back_prop,
    ).main()
