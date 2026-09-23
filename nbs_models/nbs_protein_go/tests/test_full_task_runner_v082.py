"""Real mmap/optimizer/metric integration for all v0.8.2 experiment arms."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nbs_pg.full_task_data_v082 import FullTaskDataV082
from nbs_pg.full_task_loss_v082 import FullTaskLossConfigV082
from nbs_pg.latence_stores import RoleAwareBaseLogitStore, RoleProbabilitySlice
from scripts.nbs import train_nbs_full_task_v082 as runner
from test_full_task_data_v082 import fixture_teacher_data
from test_full_task_data_v080 import inference_fixture


def _data(tmp_path, *, width=6):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config, stores, manifest_path, teacher_path = fixture_teacher_data(tmp_path, construct=False)
    config['full_task']['holdout_core_count'] = 1
    if width != 6:
        stores.num_task_go = stores.gold_messages.num_go = width
        probability = np.full((2, width), .02, np.float32)
        probability[:, :6] = np.load(teacher_path)
        np.save(teacher_path, probability)
        go_path = tmp_path / 'gg_relations/go_registry.tsv'
        go_path.write_text('go_idx\tinput_go_id\n' + ''.join(
            f'{i}\tGO:{i:07d}\n' for i in range(width)))
        manifest = json.loads(manifest_path.read_text())
        manifest['go_registry'].update(num_terms=width, sha256=runner.sha256(go_path))
        manifest_path.write_text(json.dumps(manifest))
        for role, count in (('core', 4), ('weak', 2)):
            np.save(tmp_path / f'{role}_full_p.npy', np.full((count, width), .2, np.float32))
        stores.episode_sampler.base_logit_store = RoleAwareBaseLogitStore([
            RoleProbabilitySlice('core', 0, 4, str(tmp_path / 'core_full_p.npy')),
            RoleProbabilitySlice('weak', 4, 6, str(tmp_path / 'weak_full_p.npy')),
        ], num_go=width)
    ontology_width = max(8, width)
    rng = np.random.default_rng(34)
    boxes = {'center': rng.normal(size=(ontology_width, 4)).astype(np.float32),
             'offset': rng.uniform(.1, 1, size=(ontology_width, 4)).astype(np.float32),
             'stats': np.zeros((ontology_width, 2), np.float32)}
    stores.full_boxes = SimpleNamespace(
        num_go=ontology_width, gather=lambda rows: {k: v[rows] for k, v in boxes.items()})
    stores.task_to_ontology = np.arange(width, dtype=np.int64)
    stores.task_to_ontology[2] = 0  # Output columns stay distinct despite ontology aliases.
    edges = np.array([[0, 6], [1, 6], [2, 7], [3, 7], [4, 6], [5, 7]], dtype=np.int64)
    stores.go_relations = {'is_a': SimpleNamespace(edge=edges),
                           'has_child': SimpleNamespace(edge=edges[:, ::-1])}
    data = FullTaskDataV082(config, stores=stores)
    data.prepare_neighbors(device='cpu', query_batch_size=2)
    return data


def _config(data, variant='dual'):
    config = copy.deepcopy(data.config)
    config['task'] = 'cc'
    config['full_task'].update(
        variant=variant, learning_rate=.01, weight_decay=0., seed=77,
        weak_batch=1, core_batch=1, warmup_steps=1, steps=4,
        checkpoints=[2, 4], checkpoint_interval=0, log_every=1, eval_batch=2,
        model=dict(hidden_dim=16, query_dim=8, decoder_hidden=8, ontology_layers=1,
                   go_chunk=3, dropout=.2, candidate_dropout=.15,
                   activation_checkpointing=True, attention_backend='math'),
        loss=asdict(FullTaskLossConfigV082(hard_pu_k=1, background_pu_k=1)))
    if variant != 'teacher':
        config['full_task']['model']['query_budget'] = 3
    return config


def _train(data, config, directory, steps, resume=None):
    args = SimpleNamespace(work_dir=Path(directory), steps=steps,
                           resume=None if resume is None else Path(resume))
    torch.manual_seed(config['full_task']['seed'])
    runner.train(args, config, data, torch.device('cpu'), rank=0, world=1)
    return torch.load(Path(directory) / 'latest.pt', map_location='cpu', weights_only=False)


def _assert_equal_tree(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_equal_tree(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_equal_tree(left, right)
    else:
        assert first == second


def test_dual_resume_preserves_teacher_updates_selection_rng_and_optimizer(tmp_path):
    data = _data(tmp_path / 'data')
    config = _config(data)
    complete = _train(data, copy.deepcopy(config), tmp_path / 'complete', 4)
    first = _train(data, copy.deepcopy(config), tmp_path / 'resumed', 2)
    resumed = _train(data, copy.deepcopy(config), tmp_path / 'resumed', 4,
                     tmp_path / 'resumed/latest.pt')
    assert first['step'] == 2 and resumed['step'] == complete['step'] == 4
    assert first['runner_version'] == '0.8.2'
    for name in ('classification_output.weight', 'match_output.weight'):
        assert torch.count_nonzero(first['model'][name]) > 0
    _assert_equal_tree(complete['model'], resumed['model'])
    _assert_equal_tree(complete['optimizer'], resumed['optimizer'])
    assert torch.equal(complete['rng'][0]['cpu'], resumed['rng'][0]['cpu'])
    assert complete['weak_cycle']['cursor'] == resumed['weak_cycle']['cursor']
    assert [x['loss'] for x in complete['history']] == [x['loss'] for x in resumed['history']]
    assert all(x['full_go_per_protein'] == 6 for x in resumed['history'])
    assert all(np.isfinite(x['grad_norm']) and x['grad_norm'] > 0 for x in resumed['history'])
    assert resumed['history'][-1]['absolute_logit_delta'] > 0
    assert resumed['history'][-1]['contrib_refine'] > 0
    validation = json.loads((tmp_path / 'resumed/validation_history.json').read_text())
    assert [x['step'] for x in validation] == [0, 2, 4]
    assert all(name in validation[-1] for name in ('full', 'classification', 'weak_off', 'core_off'))
    assert (tmp_path / 'resumed/best_core_holdout.pt').is_file()


@pytest.mark.parametrize('variant,head', [('teacher', 'correction_decoder.2.weight'),
                                         ('classifier', 'classification_output.weight')])
def test_single_head_controls_train_real_dense_teacher(tmp_path, variant, head):
    data = _data(tmp_path / 'data')
    saved = _train(data, _config(data, variant), tmp_path / variant, 2)
    assert saved['variant'] == variant
    assert torch.count_nonzero(saved['model'][head]) > 0
    assert saved['history'][-1]['contrib_refine'] == 0
    assert saved['contract']['data']['teacher_v082']['probability_sha256'] == runner.sha256(data.teacher_path)


@pytest.mark.parametrize('change,error', [
    ('old_version', 'needs a fresh run'), ('teacher', 'same data, core split'),
    ('learning_rate', 'preserve batch, RNG'), ('query_budget', 'identical model/loss'),
])
def test_resume_rejects_changed_training_contract(tmp_path, change, error):
    data = _data(tmp_path / 'data')
    config = _config(data)
    saved = _train(data, config, tmp_path / 'run', 2)
    checkpoint = tmp_path / 'run/latest.pt'
    if change == 'old_version':
        saved['runner_version'] = '0.8.1'
        torch.save(saved, checkpoint)
    elif change == 'teacher':
        mutable = np.load(data.teacher_path, mmap_mode='r+')
        mutable[0, 5] += .1  # Dense below-threshold teacher, absent from sparse positives.
        mutable.flush()
        data = FullTaskDataV082(data.config, stores=data.stores)
    elif change == 'learning_rate':
        config['full_task']['learning_rate'] *= 2
    else:
        config['full_task']['model']['query_budget'] += 1
    with pytest.raises(ValueError, match=error):
        _train(data, config, tmp_path / 'run', 4, checkpoint)


def test_real_inductive_exports_bind_final_and_classification_to_checkpoint(tmp_path, monkeypatch):
    width = 2903  # Production CC evaluator's full vocabulary; retain alias columns.
    data = _data(tmp_path / 'data', width=width)
    config = _config(data)
    config['full_task']['model']['go_chunk'] = 512
    _train(data, config, tmp_path / 'run', 2)
    input_dir = inference_fixture(tmp_path / 'data', data)
    probability_path = input_dir / 'backbone_ind_test_prob.f16.npy'
    np.save(probability_path, np.full((2, width), .2, np.float16))
    input_manifest_path = input_dir / 'ind_test_input_manifest.json'
    manifest = json.loads(input_manifest_path.read_text())
    manifest['base_probability']['sha256'] = runner.sha256(probability_path)
    manifest['cache_signature']['go_registry_sha256'] = data._go_registry_sha256
    input_manifest_path.write_text(json.dumps(manifest))
    (input_dir / 'protein_ids.txt').write_text('p0\np1\n')
    metadata_path = tmp_path / 'metadata.pkl'
    with metadata_path.open('wb') as handle:
        pickle.dump({'ind_test': {'cc': {'proteins': ['p0', 'p1'], 'prop_annotations': [[0, 1], [2]]}},
                     'train': {'cc': {'proteins': ['c0', 'c1'], 'prop_annotations': [[0, 1], [2]]}}}, handle)
    for name, probability in (('expert', .8), ('modelout', .85)):
        reference = np.full((2, width), .2, np.float32)
        reference[0, [0, 1]] = probability
        reference[1, 2] = probability
        np.save(tmp_path / f'{name}.npy', reference)
    args = SimpleNamespace(checkpoint=tmp_path / 'run/latest.pt', input_dir=input_dir,
                           metadata_file=metadata_path, work_dir=tmp_path / 'run',
                           ablation='full', branch='final', metric_backend='local_micro',
                           expert_prob=tmp_path / 'expert.npy', modelout_prob=tmp_path / 'modelout.npy',
                           references_aligned_to_input=True, allow_missing_references=False)
    summaries = {}
    for branch, suffix in (('final', ''), ('classification', 'classification')):
        args.branch = branch
        runner.evaluate(args, config, data, torch.device('cpu'))
        output = tmp_path / 'run/eval_step2/full' / suffix
        prediction = np.load(output / 'nbs_ind_test_prob.f32.npy')
        assert prediction.shape == (2, width) and prediction.dtype == np.float32
        assert np.isfinite(prediction).all()
        summary = json.loads((output / 'metrics/nbs_w2s_comparison.json').read_text())
        summaries[branch] = summary
        provenance = summary['prediction_provenance']
        assert provenance['branch'] == branch and provenance['step'] == 2
        assert provenance['checkpoint_file_verified'] is True
        assert provenance['checkpoint_sha256'] == runner.sha256(args.checkpoint)
        assert provenance['probability_sha256'] == runner.sha256(output / 'nbs_ind_test_prob.f32.npy')
        assert summary['reference_comparison_complete'] is True
        assert summary['evaluation_goal_complete'] is False  # Local micro diagnostic only.
        assert (output / 'metrics/nbs_primary_comparison.tsv').is_file()
    assert summaries['final']['methods']['NBS_final']['symbol'] == 'G'
    assert summaries['classification']['methods']['NBS_final']['symbol'] == 'C'
    assert summaries['final']['prediction_provenance']['probability_path'] != summaries['classification']['prediction_provenance']['probability_path']
    args.branch = 'final'
    output = tmp_path / 'run/eval_step2/full'
    mtime = (output / 'nbs_ind_test_prob.f32.npy').stat().st_mtime_ns
    def forbidden_reprediction(*args, **kwargs):
        raise AssertionError('validated existing predictions must be reused')
    monkeypatch.setattr(runner.FullTaskGraphModelV082, 'forward', forbidden_reprediction)
    runner.evaluate(args, config, data, torch.device('cpu'))
    assert (output / 'nbs_ind_test_prob.f32.npy').stat().st_mtime_ns == mtime
    monkeypatch.setattr(runner, 'prediction_implementation', lambda variant: {'changed': 'source'})
    with pytest.raises(FileExistsError, match='existing predictions differ'):
        runner.evaluate(args, config, data, torch.device('cpu'))


def _ddp_worker(rank, world, init_file, directory, data, config):
    torch.set_num_threads(1)
    torch.manual_seed(config['full_task']['seed'])
    dist.init_process_group('gloo', rank=rank, world_size=world, init_method=f'file://{init_file}')
    try:
        runner.train(SimpleNamespace(work_dir=Path(directory), steps=2, resume=None),
                     config, data, torch.device('cpu'), rank=rank, world=world)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='CPU DDP requires Gloo')
def test_two_rank_dual_saves_each_selector_rng(tmp_path):
    data = _data(tmp_path / 'data')
    config = _config(data)
    try:
        mp.start_processes(_ddp_worker, args=(2, str(tmp_path / 'rendezvous'),
                                            str(tmp_path / 'ddp'), data, config),
                           nprocs=2, join=True, start_method='fork')
    except mp.ProcessRaisedException as exc:
        if 'gloo/transport/tcp/device.cc' in str(exc) and 'Operation not permitted' in str(exc):
            pytest.skip('Execution environment denies Gloo TCP sockets; run DDP on training host')
        raise
    saved = torch.load(tmp_path / 'ddp/latest.pt', map_location='cpu', weights_only=False)
    assert saved['world_size'] == 2 and saved['step'] == 2
    assert len(saved['rng']) == 2
    assert not torch.equal(saved['rng'][0]['cpu'], saved['rng'][1]['cpu'])
    assert torch.count_nonzero(saved['model']['match_output.weight']) > 0
    assert len(saved['history']) == 2
