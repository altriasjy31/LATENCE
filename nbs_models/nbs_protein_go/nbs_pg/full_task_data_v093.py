"""Explicit derived weak supervision and rare-aware candidate views for v093.

The original audited loader is used only to construct the immutable feature,
ontology and protein graph stores. Its old weak CSR is then replaced by the
versioned derived CSR. The new contract records both the bootstrap source and
the actual replacement bytes; no old manifest is rewritten or impersonated.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .full_task_data import FullTaskData
from .full_task_data_v090 import FullTaskDataV090
from .full_task_sampler_fast import FastBlockSamplerMixin, _csr_rows
from .latence_graph_stores import RoleLocalProteinGOCSRStore
from .local_loader_v090 import build_latence_nbs_stores as _build_bootstrap_stores
from .unified_weak_dataset_v093 import load_unified_manifest, choose_candidate_rows, _load_go_mapping


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _resolve(base, value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (Path(base) / value).resolve()


def _manifest_path(config):
    value = config.get('data', {}).get('unified_dataset_manifest')
    if not value:
        raise ValueError('v093 requires data.unified_dataset_manifest')
    return Path(value).resolve()


def build_latence_nbs_stores_v093(config):
    path = _manifest_path(config)
    manifest = load_unified_manifest(path, verify=True)
    source = manifest['source_config']
    source_path = _resolve(path.parent, manifest.get('files', {}).get('source_config_snapshot', source['path']))
    if _sha256(source_path) != source['sha256']:
        raise ValueError('Original bootstrap configuration changed after dataset construction')
    original = json.loads(source_path.read_text(encoding='utf-8'))
    if original.get('task') != config.get('task'):
        raise ValueError('Derived dataset and requested task differ')
    for name in ('weak_manifest', 'inverted_manifest', 'protein_registry', 'gold_indptr',
                 'gold_go_idx', 'task_registry', 'full_ontology_manifest', 'alignment_manifest',
                 'backbone_weak'):
        identity = manifest.get('sources', {}).get(name)
        if identity is not None and _sha256(_resolve(path.parent, identity['path'])) != identity['sha256']:
            raise ValueError('Original training input changed after dataset construction: ' + name)
    # This is a compatibility bootstrap, not the declared new supervision.
    # Its source teacher contract is checked by the original loader before
    # replacing that CSR with the independently hash-bound derived data.
    stores = _build_bootstrap_stores(original)
    if stores.num_task_go != int(manifest['num_task_go']):
        raise ValueError('Derived dataset changes the immutable task GO axis')
    weak_path = _resolve(original['data']['root'], original['data']['weak_graph_predictions_manifest'])
    weak = json.loads(weak_path.read_text(encoding='utf-8'))
    project_root = Path(__file__).resolve().parents[3]
    mapping, go_registry, _ = _load_go_mapping(project_root, original, weak, weak_path, stores.num_task_go)
    if not np.array_equal(mapping, stores.task_to_ontology):
        raise ValueError('v093 GO identities differ from the original aligned ontology mapping')
    if go_registry.contract() != manifest.get('GO_registry_contract'):
        raise ValueError('Derived dataset GO registry policy differs; rebuild with the installed v093 builder')
    stores.go_registry_v090 = go_registry  # Compatibility attribute consumed by existing contracts.
    spec = manifest['pseudo']
    stores.pseudo_messages = RoleLocalProteinGOCSRStore(
        _resolve(path.parent, spec['indptr']), _resolve(path.parent, spec['go_idx']),
        role='weak', registry=stores.registry,
        probability_path=_resolve(path.parent, spec['probability']))
    # The episode API is not used by full-task training, but keep its direct
    # per-protein accessor aligned for inspection. Its old GO-inverted pseudo
    # index remains a bootstrap-only store and is never read by this adapter.
    stores.episode_sampler.pseudo_by_protein = stores.pseudo_messages
    stores._unified_manifest_path = path
    stores._unified_manifest = manifest
    return stores


class FullTaskDataV093(FastBlockSamplerMixin, FullTaskDataV090):
    VERSION = '0.9.3'

    def __init__(self, config, stores=None):
        path = _manifest_path(config)
        manifest = load_unified_manifest(path, verify=True)
        self.unified_manifest_path, self.unified_manifest = path, manifest
        self.core_counts = np.load(_resolve(path.parent, manifest['files']['core_counts']), mmap_mode='r', allow_pickle=False)
        self._unified_seen = np.asarray(np.load(_resolve(path.parent, manifest['files']['seen_task_mask']), allow_pickle=False), dtype=bool)
        self._cached_candidate_go = np.load(_resolve(path.parent, manifest['candidates']['go']), mmap_mode='r', allow_pickle=False)
        self._cached_candidate_attr = np.load(_resolve(path.parent, manifest['candidates']['attr']), mmap_mode='r', allow_pickle=False)
        stores = build_latence_nbs_stores_v093(config) if stores is None else stores
        if getattr(stores, '_unified_manifest_path', path) != path:
            raise ValueError('Supplied stores belong to another derived dataset')
        super().__init__(config, stores=stores)
        if self.core_counts.shape != (self.num_task_go,) or self.core_counts.dtype.kind not in 'iu' or (self.core_counts < 0).any():
            raise ValueError('Derived core counts leave the task axis')
        expected_mapping = np.load(_resolve(path.parent, manifest['files']['task_to_ontology']), allow_pickle=False)
        if not np.array_equal(expected_mapping, self.task_to_ontology_rows):
            raise ValueError('Derived canonical mapping differs from actual ontology stores')
        k, n = int(manifest['candidates']['topk']), self.registry.num_proteins
        if self.candidate_topk != k or self._cached_candidate_go.shape != (n, k) or self._cached_candidate_attr.shape != (n, k, 3):
            raise ValueError('Derived candidate cache does not match sampler topK/protein rows')
        for name, actual in [('training_core_ids', self.core_ids), ('validation_ids', self.validation_ids), ('training_weak_ids', self.weak_ids)]:
            expected = np.load(_resolve(path.parent, manifest['files'][name]), allow_pickle=False)
            if not np.array_equal(actual, expected):
                raise ValueError('Derived dataset training population differs: ' + name)

    def _load_label_split(self, selection):
        mask = self._unified_seen.copy()
        if mask.shape != (self.num_task_go,) or not mask.any():
            raise ValueError('Derived seen mask must retain supervised GO columns')
        for canonical in np.unique(self.task_to_ontology_rows):
            group = mask[self.task_to_ontology_rows == canonical]
            if group.any() and not group.all():
                raise ValueError('Derived seen mask splits a canonical alias group')
        return mask, {'schema': 'nbs-v093-derived-label-split-v1',
            'scope': 'stage2_label_holdout_or_natural_zero',
            'seen_task_columns': np.flatnonzero(mask).tolist(),
            'num_task_go': self.num_task_go,
            'manifest_sha256': _sha256(self.unified_manifest_path),
            'strict_pipeline_zero_shot': False,
            'claim': 'Stage2 unseen only; upstream Stage1 and teachers may have supervised these GO'}

    def data_contract(self):
        result = super().data_contract()
        result['v093_unified_dataset'] = {
            'manifest_sha256': _sha256(self.unified_manifest_path),
            'schema': self.unified_manifest['schema'],
            'derived_files': deepcopy(self.unified_manifest['hashes']),
            'policies': deepcopy(self.unified_manifest.get('policies', {})),
            'bootstrap_source_config': deepcopy(self.unified_manifest['source_config']),
            'old_weak_manifest_is_bootstrap_only': True,
            'actual_weak_supervision': 'derived binary membership and selected raw teacher probabilities',
            'old_GO_inverted_pseudo_index_used': False,
            'candidate_contract': 'cached rare-aware global rows; same policy applied to inference queries'}
        return result

    def _candidate_rows(self, ids):
        ids = np.asarray(ids, np.int64)
        if ids.ndim != 1 or (ids < 0).any() or (ids >= len(self._cached_candidate_go)).any():
            raise ValueError('Candidate cache requires valid global protein rows')
        return np.asarray(self._cached_candidate_go[ids], np.int64), np.asarray(self._cached_candidate_attr[ids], np.float32)

    def _truncate_candidates(self, go, attr):
        go, attr = np.asarray(go), np.asarray(attr)
        if go.ndim != 2 or attr.shape != (*go.shape, 3) or not np.isfinite(attr).all():
            raise ValueError('Candidate axes must be [protein,K] and [protein,K,3]')
        if go.shape[1] == self.candidate_topk:
            # A prepared quota selection must not be globally reranked/truncated.
            valid = (go >= 0) & (go < self.num_task_go)
            safe = np.where(valid, go, 0)
            valid &= self.seen_task_mask[safe]
            return np.where(valid, go, -1).astype(np.int64), np.where(valid[..., None], attr, 0).astype(np.float32)
        return choose_candidate_rows(go, attr, self.core_counts, self.seen_task_mask,
            mode=self.unified_manifest['candidates']['mode'], topk=self.candidate_topk)

    def _small_candidates(self, ids):
        return self._truncate_candidates(*self._candidate_rows(ids))

    def _label_edges(self, nodes, excluded):
        gold, pseudo = super()._label_edges(nodes, excluded)
        return gold[:, self.seen_task_mask[gold[1]]], pseudo[:, self.seen_task_mask[pseudo[1]]]

    def _filter_label_view(self, batch):
        output = super()._filter_label_view(batch)
        for key in ('anchor_go_edge', 'loss_anchor_go_edge', 'gold_edge', 'pseudo_edge'):
            if key in output:
                edge = output[key]
                if edge.ndim != 2 or edge.shape[0] != 2:
                    raise ValueError('Legacy label evidence must have shape [2,E]: ' + key)
                seen = torch.as_tensor(self.seen_task_mask.copy(), device=edge.device)
                output[key] = edge[:, seen[edge[1].long()]]
        return output

    def _target_arrays(self, ids):
        """Global core versus role-local weak addressing, with no rethresholding."""
        weak = self.registry.role_code[ids] == self.registry.role_to_code['weak']
        positive = np.zeros((len(ids), self.num_task_go), bool)
        soft = np.zeros((len(ids), self.num_task_go), np.float32)
        for positions, store, rows in (
                (np.flatnonzero(~weak), self.stores.gold_messages, ids[~weak]),
                (np.flatnonzero(weak), self.stores.pseudo_messages, np.asarray(self.registry.role_row[ids[weak]], np.int64))):
            local, go, edge_positions = _gather_rows(store.indptr, store.go_idx, rows)
            if len(go):
                allowed = self.seen_task_mask[go]
                local, go, edge_positions = local[allowed], go[allowed], edge_positions[allowed]
                target_rows = positions[local]
                positive[target_rows, go] = True
                if store is self.stores.pseudo_messages:
                    probability = np.asarray(store.probability[edge_positions], np.float32)
                    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                        raise ValueError('Derived weak teacher probabilities must lie in [0,1]')
                    soft[target_rows, go] = probability
        return weak, positive, soft

    def batch(self, protein_ids, device='cpu', *, supervision_seed_ids=None):
        local = self._valid_ids(protein_ids, 'v093 local queries')
        global_ids = local if supervision_seed_ids is None else self._valid_ids(supervision_seed_ids, 'v093 macro supervision seeds')
        if not np.isin(local, global_ids).all():
            raise ValueError('Macro supervision seeds must contain every local query')
        if self._neighbors is None and not self._load_neighbors():
            raise RuntimeError('Core neighbor cache absent; run prepare')
        neighbors = np.asarray(self._neighbors[local], np.int64)
        if np.any(self.anchor_core_ids[neighbors] == local[:, None]):
            raise ValueError('Core target self-edge would leak gold supervision')
        base = self.stores.episode_sampler.base_logit_store.gather_matrix(local,
            np.arange(self.num_task_go, dtype=np.int64)).T.copy()
        candidate, attr = self._candidate_rows(local)
        result = self._graph(self.stores.features.gather(local), base, candidate, attr,
            neighbors, np.asarray(self._neighbor_attrs[local]), device)
        weak, positive, soft = self._target_arrays(local)
        result.update(targets=torch.as_tensor(positive.astype(np.float32), device=device),
            positive_mask=torch.as_tensor(positive, device=device),
            weak_soft_targets=torch.as_tensor(soft, device=device),
            is_weak=torch.as_tensor(weak, device=device),
            core_counts=torch.as_tensor(np.array(self.core_counts), dtype=torch.long, device=device),
            seen_task_mask=torch.as_tensor(self.seen_task_mask.copy(), device=device))
        self._require_pools()
        result = self._attach_blocks(result, -local - 1,
            (self._retrieval_neighbors[local], self._retrieval_attrs[local]), global_ids, device)
        return self._filter_label_view(result)

    def inference_batch(self, input_dir, row_ids, device='cpu'):
        context = (self._sampling_step, self._sampling_rank, self._sampling_training)
        self.set_sampling_context(0, 0, False)
        try:
            result = FullTaskData.inference_batch(self, input_dir, row_ids, device=device)
            required_pool = min(self.num_task_go, int(getattr(self, 'config', {}).get('stage', {}).get('protein_go_candidate_topk', 512)))
            if result['candidate_go'].shape[1] < required_pool:
                raise ValueError('Inference cache has already truncated the candidate pool; rebuild full-task candidate inputs before applying v093 quotas')
            arrays = self._inference[str(Path(input_dir).resolve())]
            rows = np.asarray(row_ids, np.int64).reshape(-1)
            if len(np.unique(rows)) != len(rows):
                raise ValueError('External query rows must be unique')
            result = self._attach_blocks(result, -rows - 1,
                (arrays['v084_neighbors'][rows], arrays['v084_attrs'][rows]), (), device)
            return self._filter_label_view(result)
        finally:
            self.set_sampling_context(*context)


def _gather_rows(ptr, labels, rows):
    rows = np.asarray(rows, np.int64)
    if not len(rows):
        return (np.empty(0, np.int64),) * 3
    starts = np.asarray(ptr[rows], np.int64)
    degree = np.asarray(ptr[rows + 1], np.int64) - starts
    total = int(degree.sum())
    if not total:
        return (np.empty(0, np.int64),) * 3
    offsets = np.repeat(starts - np.cumsum(degree) + degree, degree) + np.arange(total)
    return np.repeat(np.arange(len(rows)), degree), np.asarray(labels[offsets], np.int64), offsets
