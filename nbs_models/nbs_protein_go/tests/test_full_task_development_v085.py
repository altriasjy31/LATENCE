from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from nbs_pg.full_task_development_v085 import DevelopmentSetV085, build_contract, sha256


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))


def _fixture(tmp_path):
    inp = tmp_path / 'inputs'; inp.mkdir()
    refs = tmp_path / 'references'; refs.mkdir()
    (inp / 'protein_ids.txt').write_text('dev_a\ndev_b\n')
    go = tmp_path / 'go_registry.tsv'
    go.write_text('go_idx\tinput_go_id\n0\tGO:A\n1\tGO:B\n2\tGO:C\n')
    base = np.array([[.8, .1, .2], [.1, .7, .2]], np.float32)
    for name, values in {
        'backbone_ind_test_prob.f16.npy': base,
        'ind_test_repr.f16.npy': np.eye(2, dtype=np.float32),
        'candidate_go_index.i32.npy': np.array([[0], [1]], np.int32),
        'candidate_edge_attr.f32.npy': np.ones((2, 1, 3), np.float32),
    }.items():
        np.save(inp / name, values)
    manifest = {'num_proteins': 2, 'protein_ids_file_sha256': sha256(inp / 'protein_ids.txt'),
        'cache_signature': {'num_classes': 3, 'go_registry_sha256': sha256(go)},
        'registries': {'go_registry': str(go)},
        'base_probability': {'sha256': sha256(inp / 'backbone_ind_test_prob.f16.npy')},
        'representation': {'sha256': sha256(inp / 'ind_test_repr.f16.npy')},
        'candidate_evidence': {'selector_scope': 'full_task', 'expert_probability_used': False,
            'label_hint_used': False, 'go_index_sha256': sha256(inp / 'candidate_go_index.i32.npy'),
            'edge_attr_sha256': sha256(inp / 'candidate_edge_attr.f32.npy')}}
    _write_json(inp / 'ind_test_input_manifest.json', manifest)
    outputs = {}
    for key, name, value in [('expert_prob', 'expert.npy', base + .01),
                              ('stage1_modelout', 'modelout.npy', base + .02)]:
        np.save(refs / name, value)
        outputs[key] = {'path': str(refs / name), 'sha256': sha256(refs / name)}
    for key, name, value in [('protein_ids', 'protein_ids.txt', 'dev_a\ndev_b\n'),
                              ('go_ids', 'go_ids.txt', 'GO:A\nGO:B\nGO:C\n')]:
        (refs / name).write_text(value)
        outputs[key] = {'path': str(refs / name), 'sha256': sha256(refs / name)}
    _write_json(refs / 'stage1_reference_manifest.json', {'outputs': outputs,
        'cache_signature': {'sources': {
            'input_manifest': {'sha256': sha256(inp / 'ind_test_input_manifest.json')},
            'cached_backbone': {'sha256': sha256(inp / 'backbone_ind_test_prob.f16.npy')}}},
        'semantics': {'labels_consumed_by_model': False, 'ind_test_label_boost': False}})
    # Labels deliberately use reversed rows and columns, as real source files may.
    np.save(tmp_path / 'labels.npy', np.array([[0, 1, 0], [0, 0, 1]], np.uint8))
    for name, value in [('label_protein_ids', 'dev_b\ndev_a\n'),
                        ('label_go_ids', 'GO:C\nGO:B\nGO:A\n'),
                        ('stage1_training_ids', 'core_a\nweak_a\n'),
                        ('final_test_ids', 'final_a\nfinal_b\n')]:
        (tmp_path / f'{name}.txt').write_text(value)
    registry = tmp_path / 'proteins.csv'
    registry.write_text('protein_idx,protein_id,role,role_row_idx\n0,core_a,core,0\n1,weak_a,weak,0\n')
    data = FakeData(inp, registry, base)
    args = dict(input_dir=inp, reference_dir=refs, labels=tmp_path / 'labels.npy',
        **{name: tmp_path / f'{name}.txt' for name in (
            'label_protein_ids', 'label_go_ids', 'stage1_training_ids', 'final_test_ids')})
    return args, data


class FakeData:
    def __init__(self, inp, registry, base):
        self.registry = SimpleNamespace(path=registry, num_proteins=2)
        self.num_task_go = 3
        self.input_dir, self.base = inp, base
        self._sampling_step, self._sampling_rank, self._sampling_training = 8, 1, True
        self.seen = []

    def set_sampling_context(self, step=0, rank=0, training=False):
        self._sampling_step, self._sampling_rank, self._sampling_training = step, rank, training

    def inference_batch(self, input_dir, row_ids, device='cpu'):
        assert input_dir == self.input_dir
        assert not self._sampling_training
        self.seen.extend(row_ids.tolist())
        return {'base_logits': torch.logit(torch.tensor(self.base[row_ids], device=device))}


class FakeModel(torch.nn.Module):
    def encode_go(self):
        return None

    def forward(self, batch, go_encoding=None, **flags):
        assert set(batch) == {'base_logits'}
        assert not self.training
        assert flags == {'use_pp': True}
        # Deliberately consume RNG to verify evaluation restores training state.
        torch.rand(1)
        return batch['base_logits']


def _load(args, data, tmp_path):
    manifest = tmp_path / 'development.json'
    _write_json(manifest, build_contract(**args))
    return DevelopmentSetV085.load(manifest, data), manifest


def test_reorders_gold_by_original_protein_and_go_ids(tmp_path):
    args, data = _fixture(tmp_path)
    dev, _ = _load(args, data, tmp_path)
    np.testing.assert_array_equal(dev.labels, [[1, 0, 0], [0, 1, 0]])
    assert set(dev.references) == {'B', 'E', 'M'}
    assert dev.contract['training_registry_sha256'] == sha256(data.registry.path)


@pytest.mark.parametrize('key', ['stage1_training_ids', 'final_test_ids'])
def test_rejects_overlap_even_before_training(tmp_path, key):
    args, _ = _fixture(tmp_path)
    args[key].write_text('dev_a\n')
    with pytest.raises(ValueError, match='overlaps'):
        build_contract(**args)


def test_runtime_rejects_any_registry_role_overlap(tmp_path):
    args, data = _fixture(tmp_path)
    data.registry.path.write_text('protein_idx,protein_id,role\n0,dev_b,weak\n1,core_a,core\n')
    with pytest.raises(ValueError, match='Stage-2 registry'):
        _load(args, data, tmp_path)


@pytest.mark.parametrize('key', ['labels', 'stage1_training_ids', 'final_test_ids'])
def test_source_mutations_invalidate_immutable_contract(tmp_path, key):
    args, data = _fixture(tmp_path)
    _, manifest = _load(args, data, tmp_path)
    if key == 'labels':
        np.save(args[key], np.array([[1, 0, 0], [0, 1, 0]], np.uint8))
    else:
        args[key].write_text(args[key].read_text() + 'new_protein\n')
    with pytest.raises(ValueError, match='changed'):
        DevelopmentSetV085.load(manifest, data)


def test_input_and_reference_hashes_verified(tmp_path):
    args, _ = _fixture(tmp_path)
    np.save(args['input_dir'] / 'candidate_go_index.i32.npy', np.array([[1], [2]], np.int32))
    with pytest.raises(ValueError, match='SHA256'):
        build_contract(**args)


def test_rejects_misaligned_reference_ids(tmp_path):
    args, _ = _fixture(tmp_path)
    path = args['reference_dir'] / 'protein_ids.txt'
    path.write_text('dev_b\ndev_a\n')
    manifest = args['reference_dir'] / 'stage1_reference_manifest.json'
    value = json.loads(manifest.read_text())
    value['outputs']['protein_ids']['sha256'] = sha256(path)
    _write_json(manifest, value)
    with pytest.raises(ValueError, match='order must equal'):
        build_contract(**args)


def test_rejects_missing_reference_or_nonbinary_gold(tmp_path):
    args, _ = _fixture(tmp_path)
    np.save(args['labels'], np.ones((2, 3), np.float32) * .5)
    with pytest.raises(ValueError, match='binary'):
        build_contract(**args)


def test_evaluate_restores_state_and_reports_four_methods_without_label_inputs(tmp_path):
    args, data = _fixture(tmp_path)
    dev, _ = _load(args, data, tmp_path)
    model = FakeModel()
    before_rng = torch.get_rng_state().clone()
    result = dev.evaluate(model, data, 'cpu', 1, forward_flags={'use_pp': True})
    assert model.training
    assert (data._sampling_step, data._sampling_rank, data._sampling_training) == (8, 1, True)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert data.seen == [0, 1]
    assert set(result['methods']) == {'B', 'E', 'M', 'G'}
    assert result['deltas']['G_minus_B']['standard_micro_pr_auc'] == pytest.approx(0)
    assert result['selection_score'] == result['methods']['G']['standard_micro_pr_auc']
    assert result['selection_guard']


def test_evaluate_restores_on_error(tmp_path):
    args, data = _fixture(tmp_path)
    dev, _ = _load(args, data, tmp_path)
    model = FakeModel().eval()
    def fail(*args, **kwargs):
        raise RuntimeError('inference failed')
    data.inference_batch = fail
    with pytest.raises(RuntimeError, match='failed'):
        dev.evaluate(model, data, 'cpu', 1)
    assert not model.training
    assert data._sampling_training


def test_cli_immutable_output_and_repeat(tmp_path):
    from scripts.nbs.prepare_nbs_development_v085 import main
    args, _ = _fixture(tmp_path)
    output = tmp_path / 'development.json'
    argv = [item for key, value in args.items() for item in ('--' + key.replace('_', '-'), str(value))]
    main(argv + ['--output', str(output)])
    first = output.read_bytes()
    main(argv + ['--output', str(output)])
    assert output.read_bytes() == first
    args['final_test_ids'].write_text('other_final_id\n')
    with pytest.raises(FileExistsError):
        main(argv + ['--output', str(output)])
