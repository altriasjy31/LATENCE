import torch
from nbs_pg.full_task_evidence_v081 import (
    StructuralSupportConfig, structural_core_support, alias_pu_exclusions,
)


def _batch():
    return {"anchor_x": torch.zeros(3, 2),
            "anchor_go_edge": torch.tensor([[0, 0, 1, 2], [0, 0, 0, 1]]),
            "neighbor_index": torch.tensor([[0, 1, 2], [0, 0, -1]]),
            "neighbor_attr": torch.tensor([[[.9,0,0],[.8,0,0],[.6,0,0]],
                                           [[.9,0,0],[.9,0,0],[1.,0,0]]])}


def test_independent_neighbors_support_without_duplicate_votes():
    batch = _batch()
    result = structural_core_support(batch, 4)
    assert torch.isclose(result[0, 0], torch.tensor(1.7/2.3))
    assert result[0, 1] == 0 and torch.count_nonzero(result[1]) == 0
    batch.update(targets=torch.ones(2,4), positive_mask=torch.ones(2,4).bool(),
                 expert_prob=torch.ones(2,4), base_logits=torch.randn(2,4))
    assert torch.equal(result, structural_core_support(batch,4))
    assert not result.requires_grad


def test_empty_or_low_similarity_has_no_support():
    batch = _batch()
    assert torch.count_nonzero(structural_core_support(batch,4,StructuralSupportConfig(min_similarity=.99))) == 0
    batch['anchor_x'] = torch.zeros(0,2)
    batch['anchor_go_edge'] = torch.zeros(2,0).long()
    assert torch.equal(structural_core_support(batch,4),torch.zeros(2,4))


def test_aliases_excluded_without_adding_or_losing_positive_columns():
    positive = torch.tensor([[True,False,False,False], [False,True,False,False]])
    before = positive.clone()
    result = alias_pu_exclusions(positive,torch.tensor([0,1,0,2]))
    assert torch.equal(result,torch.tensor([[False,False,True,False],[False,False,False,False]]))
    assert torch.equal(positive,before)


def test_corrupt_inductive_neighbor_cache_is_rebuilt(tmp_path):
    import numpy as np
    from test_full_task_data_v080 import fixture_data, inference_fixture
    data = fixture_data(tmp_path,holdout=1)
    data.prepare_neighbors(device='cpu')
    inputs = inference_fixture(tmp_path,data)
    expected = data.inference_batch(inputs,[0,1])['neighbor_index'].clone()
    cached = next(data.cache_dir.glob('inductive_core_neighbors_*.npz'))
    with np.load(cached) as f:
        payload = dict(f)
    payload['neighbors'] = np.zeros_like(payload['neighbors'])
    # Stored digest still describes the original graph.
    np.savez(cached,**payload)
    data._inference.clear()
    result = data.inference_batch(inputs,[0,1])['neighbor_index']
    assert torch.equal(result,expected)
