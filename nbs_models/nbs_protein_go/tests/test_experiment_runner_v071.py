from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/nbs'))
from run_nbs_v071_experiments import VARIANTS, clean_environment, make_training_config
from analyze_nbs_v071_experiments import analyze_smoke, evaluation_rows
from nbs_pg.config import NBSConfig
from nbs_pg.episode import NBSQueryEpisodeConfig
from nbs_pg.training import NBSFixedEpochTrainingConfig, NBSSchedulerConfig


def test_stale_environment_cannot_override_experiment():
    env = clean_environment({'NBS_RESUME': 'epoch20.pt', 'NBS_MAX_STEPS_PER_EPOCH': '20',
        'NBS_NUM_QUERIES': '64', 'NBS_ONECYCLE_PCT_START': '.05',
        'NBS_EVAL_LIMIT_PROTEINS': '2', 'STAGE1_CHECKPOINT': 'base.pt', 'CUDA_VISIBLE_DEVICES': '0,1'})
    assert not any(k.startswith('NBS_') for k in env)
    assert env['STAGE1_CHECKPOINT'] == 'base.pt'
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1'


def test_smoke_and_probe_have_independent_epoch_scheduler_contracts(tmp_path: Path):
    for variant in VARIANTS:
        for phase in ('smoke', 'probe'):
            config = make_training_config(ROOT, variant, phase, tmp_path / phase, 3)
            NBSConfig(**config['model']).validate()
            ep = NBSQueryEpisodeConfig(**config['episode'])
            ep.validate()
            NBSFixedEpochTrainingConfig.from_mapping(config['training']).validate()
            NBSSchedulerConfig.from_mapping(config['scheduler']).validate()
            assert config['model']['candidate_evidence_residual_scale_init'] != 0
            assert ep.candidate_evidence_scope == 'decoded_all'
            if phase == 'smoke':
                assert config['training']['epochs'] == 1
                assert config['training']['save_epochs'] == [1]
                assert config['training']['max_steps_per_epoch'] == 20
                assert config['scheduler']['name'] == 'none'
            else:
                assert config['training']['max_steps_per_epoch'] is None
                assert config['training']['epochs'] == 3
                assert config['scheduler']['name'] == 'onecycle'
            if variant != 'correctness_control':
                assert ep.num_queries == 144
                assert ep.max_candidates >= ep.supervision_candidate_upper_bound
                assert ep.weak_primary_hard_query_slots_per_episode == 16


def test_analysis_never_calls_fmax_only_increase_a_ranking_success():
    record = {'epoch': 20, 'primary': {'NBS_final': {'fmax': 55.7, 'auprc': 28.1}, 'backbone_base': {'fmax': 55.6, 'auprc': 32.0}},
              'ranking_analysis': {'paired_delta_NBS_minus_backbone': {str(k): {'precision_at_k': {'mean': -.1}, 'recall_at_k': {'mean': -.2}} for k in (10, 50, 100)}}}
    row = evaluation_rows({'records': [record]}, 'fixed_epoch20')[0]
    assert row['delta_fmax_pp'] > 0
    assert row['interpretation'] == 'ranking_not_fully_recovered_vs_backbone'


def test_missing_gradient_logs_fail_smoke_gate(tmp_path: Path):
    config = make_training_config(ROOT, 'sampling_loss', 'smoke', tmp_path, 3)
    (tmp_path / 'resolved_config.json').write_text(json.dumps(config))
    (tmp_path / 'training_history.json').write_text(json.dumps([{'steps_per_rank': 20, 'total': .1, 'world_size': 2}]))
    result = analyze_smoke(tmp_path)
    assert result['status'] == 'needs_attention'
    assert result['checks']['rank0_gradient_log_present'] is False
    assert result['checks']['rank1_gradient_log_present'] is False


def test_training_gradient_probe_records_before_clipping(tmp_path: Path):
    import torch
    from nbs_pg.training import NBSFixedEpochTrainer, NBSRunComponents, NBSLossConfig
    from test_fixed_epoch_training import _DummyNBS, _batch
    model = _DummyNBS()
    with torch.no_grad():
        model.bias.fill_(.4)
    components = NBSRunComponents(model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.1),
        train_loader=[_batch(), _batch(), _batch()], loss_config=NBSLossConfig(primary='bce'))
    config = NBSFixedEpochTrainingConfig(epochs=1, save_epochs=(1,), output_dir=str(tmp_path),
        amp=False, progress_bar=False, gradient_probe_steps=2, grad_clip=.000001)
    trainer = NBSFixedEpochTrainer(components, config, device='cpu', logger=lambda _: None)
    trainer.fit()
    probe = json.loads((tmp_path / 'gradient_probe_rank0.json').read_text())
    assert [r['global_step'] for r in probe['records']] == [1, 2]
    assert probe['gradient_stage'] == 'unscaled_before_clipping'
    assert probe['records'][0]['parameters']['matcher.graph_delta_scale']['grad_l2'] > config.grad_clip
