"""Add frozen ESM features after the unchanged v095 graph has been sampled."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
import torch

from .esm_feature_store_v096 import (ESMPenultimateStoreV096, checked_ids,
                                    read_registry_protein_ids, sha256_file)
from .full_task_data_v094 import FullTaskDataV094, build_latence_nbs_stores_v094


def build_latence_nbs_stores_v096(config):
    # ESM changes neither cosine retrieval nor biological graph edges.
    return build_latence_nbs_stores_v094(config)


class FullTaskDataV096(FullTaskDataV094):
    VERSION = '0.9.6'

    def __init__(self, config, stores=None):
        spec = config.get('data', {})
        if not spec.get('esm_features_manifest') or not spec.get('esm_features_manifest_sha256'):
            raise ValueError('v096 needs a frozen data.esm_features_manifest and its sha256')
        esm = ESMPenultimateStoreV096(spec['esm_features_manifest'],
                                    expected_sha256=spec['esm_features_manifest_sha256'])
        if config.get('full_task', {}).get('model', {}).get('esm_dim') != esm.feature_dim:
            raise ValueError('Configured model.esm_dim differs from the feature cache')
        super().__init__(config, stores=stores)
        self.esm = esm
        self.registry_protein_ids = read_registry_protein_ids(self.registry.path, self.registry.num_proteins)
        # All registry roles must be covered, including nodes only seen as
        # neighbors and held-out core rows. No training/testing label is read.
        self._registry_esm_rows = esm.rows_for(self.registry_protein_ids)
        self._esm_inference = {}

    def data_contract(self):
        result = super().data_contract()
        result['v096_esm_features'] = {
            **self.esm.identity(),
            'registry_protein_ids_sha256': sha256_file(self.registry.path),
            'alignment': 'exact_protein_id_lookup_for_all_registry_roles_and_query_ghosts',
            'missing_policy': 'error', 'frozen': True,
            'retrieval_space': 'unchanged_Stage1_only',
            'classifier_logits_used_as_features': False,
            'strict_pipeline_zero_shot': False}
        return result

    def _inference_esm(self, input_dir):
        directory = Path(input_dir).resolve(strict=True)
        key = str(directory)
        ids_path = directory / 'protein_ids.txt'
        supplemental = directory / 'esm_penultimate' / 'manifest.json'
        if key not in self._esm_inference:
            names = checked_ids(np.asarray(ids_path.read_text(encoding='utf-8').splitlines(), dtype=str),
                                name='inference protein IDs')
            count = len(np.load(directory / 'ind_test_repr.f16.npy', mmap_mode='r', allow_pickle=False))
            if count != len(names):
                raise ValueError('Inference protein IDs do not match the original representation rows')
            store = ESMPenultimateStoreV096(supplemental) if supplemental.is_file() else self.esm
            if (store.feature_dim != self.esm.feature_dim or
                    store.identity()['representation_identity'] != self.esm.identity()['representation_identity']):
                raise ValueError('Inference ESM checkpoint/layers/ensemble/source code differs from training')
            mapped = store.rows_for(names)
            self._esm_inference[key] = {'store': store, 'rows': mapped,
                'protein_ids_sha256': sha256_file(ids_path), 'uses_supplemental_cache': supplemental.is_file()}
        entry = self._esm_inference[key]
        if (sha256_file(ids_path) != entry['protein_ids_sha256'] or
                supplemental.is_file() != entry['uses_supplemental_cache']):
            raise ValueError('Inference IDs or ESM cache selection changed while in use')
        entry['store'].assert_unchanged()
        return entry

    def inference_esm_identity(self, input_dir):
        entry = self._inference_esm(input_dir)
        return {'cache': entry['store'].identity(),
                'protein_ids_sha256': entry['protein_ids_sha256'],
                'uses_supplemental_cache': entry['uses_supplemental_cache'],
                'alignment': 'exact_protein_id_lookup'}

    def _attach_esm(self, batch, *, external=None):
        ids = batch['sampled_global_ids'].detach().cpu().numpy()
        if ids.ndim != 1 or ids.dtype.kind not in 'iu':
            raise ValueError('Sampled protein identities must be one-dimensional integers')
        known, ghosts = np.flatnonzero(ids >= 0), np.flatnonzero(ids < 0)
        if len(known) and np.any(ids[known] >= self.registry.num_proteins):
            raise ValueError('Sampled support global ID exceeds protein registry')
        values = np.empty((len(ids), self.esm.feature_dim), dtype=np.float32)
        if len(known):
            values[known] = self.esm.gather_rows(self._registry_esm_rows[ids[known]])
        # Training ghost -g-1 denotes GLOBAL registry g; inference ghost -r-1
        # denotes EXTERNAL input row r. These axes must never be interchanged.
        ghost_rows = -ids[ghosts] - 1
        mapping = self._registry_esm_rows if external is None else external['rows']
        source = self.esm if external is None else external['store']
        if len(ghost_rows) and np.any(ghost_rows >= len(mapping)):
            raise ValueError('Sampled query ghost leaves its protein identity axis')
        if len(ghosts):
            values[ghosts] = source.gather_rows(mapping[ghost_rows])
        batch['sampled_esm_x'] = torch.as_tensor(values, device=batch['sampled_protein_x'].device)
        if batch['sampled_esm_x'].shape[0] != batch['sampled_protein_x'].shape[0]:
            raise ValueError('ESM features do not cover every sampled protein node')
        return batch

    def batch(self, protein_ids, device='cpu', *, supervision_seed_ids=None):
        result = super().batch(protein_ids, device=device, supervision_seed_ids=supervision_seed_ids)
        return self._attach_esm(result)

    def inference_batch(self, input_dir, row_ids, device='cpu'):
        external = self._inference_esm(input_dir)
        result = super().inference_batch(input_dir, row_ids, device=device)
        return self._attach_esm(result, external=external)
