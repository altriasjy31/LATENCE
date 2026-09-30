"""Global DDP label masking over the actual v084 graph/cosine fixture."""
import numpy as np
import pytest
import torch

from nbs_pg.full_task_data_v087 import FullTaskDataV087
from nbs_pg.full_task_evidence_v081 import StructuralSupportConfig, structural_core_support
from test_full_task_data_v084 import data_v084, inference_fixture


def _data(tmp_path, **kwargs):
    old = data_v084(tmp_path, **kwargs)
    old.prepare_neighbors()
    new = FullTaskDataV087(old.config, stores=old.stores)
    new.ensure_prepared()
    return old, new


def _equal_batch(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], dict):
            _equal_batch(left[key], right[key])
        else:
            assert torch.equal(left[key], right[key]), key


def _sources(data, batch, local_ids):
    sampled = batch['sampled_global_ids']
    anchors = sampled[batch['sampled_anchor_index']]
    loss_anchors = torch.from_numpy(np.array(data.anchor_core_ids[
        np.unique(data._neighbors[np.asarray(local_ids)])], copy=True))
    return {'sampled_gold_edge': sampled, 'sampled_pseudo_edge': sampled,
            'anchor_go_edge': anchors, 'loss_anchor_go_edge': loss_anchors}


def test_default_and_explicit_local_masks_reproduce_v084(tmp_path):
    old, new = _data(tmp_path)
    expected = old.batch([4, 0])
    _equal_batch(expected, new.batch([4, 0]))
    _equal_batch(expected, new.batch([4, 0], supervision_seed_ids=[4, 0]))
    old_contract, new_contract = old.data_contract(), new.data_contract()
    assert new_contract.pop('v087_label_mask').startswith('global_DDP_supervision_seeds')
    assert old_contract == new_contract
    assert old._pool_signature == new._pool_signature


def test_remote_core_and_weak_seed_labels_are_removed_from_all_incidence(tmp_path):
    _, data = _data(tmp_path)
    local, global_seeds = [4], [4, 0, 5]
    before = data.batch(local)
    after = data.batch(local, supervision_seed_ids=global_seeds)
    sources = _sources(data, before, local)
    for key, ids in sources.items():
        original_labels = ids[before[key][0]]
        # The fixture truly contains remote-query labels in every tested input.
        assert torch.isin(original_labels, torch.tensor([0, 5])).any(), key
        remaining_labels = ids[after[key][0]]
        assert not torch.isin(remaining_labels, torch.tensor(global_seeds)).any(), key
        keep = ~torch.isin(original_labels, torch.tensor(global_seeds))
        assert torch.equal(after[key], before[key][:, keep]), key
    for key, value in before.items():
        if key not in sources:
            if isinstance(value, dict):
                _equal_batch(value, after[key])
            else:
                assert torch.equal(value, after[key]), key
    # In particular, B candidates, topology and every node feature survive.
    for key in ('candidate_go', 'candidate_attr', 'sampled_candidate_go',
                'sampled_candidate_attr', 'sampled_edge_index', 'sampled_protein_x'):
        assert torch.equal(before[key], after[key])


def test_global_mask_reaches_fixed_pu_support_not_just_forward(tmp_path):
    _, data = _data(tmp_path)
    before = data.batch([4])
    after = data.batch([4], supervision_seed_ids=[4, 0])
    config = StructuralSupportConfig(min_similarity=0., min_neighbors=1,
                                      min_vote_fraction=0.)
    def support(batch):
        graph = {k: batch['loss_'+k] for k in
                 ('anchor_x', 'anchor_go_edge', 'neighbor_index', 'neighbor_attr')}
        return structural_core_support(graph, data.num_task_go, config)
    initial, masked = support(before), support(after)
    # GO 1 belongs only to remote core seed 0 among these cosine neighbors.
    assert initial[0, 1] > 0 and masked[0, 1] == 0


def test_global_mask_has_no_persistent_state_or_external_effect(tmp_path):
    old, data = _data(tmp_path, variant='dynamic', holdout=1)
    old.set_sampling_context(7, 1, True)
    data.set_sampling_context(7, 1, True)
    reference = data.batch([4])
    data.batch([4], supervision_seed_ids=[4, 0, 5])
    _equal_batch(reference, data.batch([4]))
    _equal_batch(old.batch([0]), data.batch([0]))
    input_dir = inference_fixture(tmp_path, data)
    _equal_batch(old.inference_batch(input_dir, [0, 1]),
                 data.inference_batch(input_dir, [0, 1]))
    assert (data._sampling_step, data._sampling_rank, data._sampling_training) == (7, 1, True)


@pytest.mark.parametrize('global_ids', [[4, 4], [0, 5], [-1, 4], [4, 6],
    [4., 0.], [[4, 0]], [], [True, False]])
def test_invalid_global_seed_contract_is_rejected(tmp_path, global_ids):
    _, data = _data(tmp_path)
    with pytest.raises(ValueError, match='global supervision'):
        data.batch([4], supervision_seed_ids=global_ids)


def test_non_training_registry_role_is_rejected(tmp_path):
    _, data = _data(tmp_path)
    data.registry.role_code = np.array(data.registry.role_code, copy=True)
    data.registry.role_code[5] = 99
    with pytest.raises(ValueError, match='core or weak'):
        data.batch([4], supervision_seed_ids=[4, 5])
