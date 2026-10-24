"""CPU phase, state-transfer, fixed-metric and real training regression checks.

Synthetic fixtures establish contracts, never production Fmax or GPU throughput.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.nbs import train_nbs_full_task_v090 as runner
from scripts.nbs import eval_nbs_full_task_v090 as evaluator
from scripts.nbs import check_nbs_v090_acceptance as gate
from nbs_pg.full_task_data_v090 import FullTaskDataV090
from nbs_pg.full_task_loss_v090 import FullTaskLossConfigV090
from nbs_pg.full_task_model_v090 import FullTaskModelConfigV090
from nbs_pg.full_task_phase_v090 import PhasePlanV090, resume_contract, validate_resume_contract
from test_full_task_data_v083 import data_v083
from nbs_pg.full_task_data_v088 import FullTaskDataV088, RELATIONS


def _assert_state_equal(left, right, path="state"):
    if torch.is_tensor(left):
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_state_equal(left[key], right[key], path + "." + str(key))
    elif isinstance(left, (list,tuple)):
        assert len(left) == len(right), path
        for i,(a,b) in enumerate(zip(left,right)):
            _assert_state_equal(a,b,path+f"[{i}]")
    else:
        assert left == right, path


def _data(tmp_path, variant="dynamic"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    old = data_v083(tmp_path, holdout=1, enabled=False)
    cfg = copy.deepcopy(old.config)
    cfg['release_version'] = cfg['model_version'] = '0.9.0'
    cfg.setdefault('stage',{})['name'] = 'nbs_v090_shared_go'
    cfg['full_task'].update(variant=variant, sampler=dict(
        mode=variant,seed=8084,native_pool_per_relation=3,retrieval_pool_size=2,
        candidate_topk=1,stable_fraction=.5,scan_block_size=3,
        cache_dir=str(tmp_path/'v084_pools'),max_nodes=100,
        fanouts=[{name:1 for name in RELATIONS},{name:1 for name in RELATIONS},{'cosine':2}]))
    rng=np.random.default_rng(34)
    boxes={'center':rng.normal(size=(8,4)).astype(np.float32),
           'offset':rng.uniform(.1,1,size=(8,4)).astype(np.float32),'stats':np.zeros((8,2),np.float32)}
    old.stores.full_boxes=SimpleNamespace(num_go=8,gather=lambda rows:{key:value[rows] for key,value in boxes.items()})
    old.stores.task_to_ontology=np.array([0,1,0,3,4,5],np.int64)
    edge=np.array([[0,6],[1,6],[2,7],[3,7],[4,6],[5,7]],np.int64)
    old.stores.go_relations={'is_a':SimpleNamespace(edge=edge),'has_child':SimpleNamespace(edge=edge[:,::-1])}
    result = FullTaskDataV090(cfg, stores=old.stores)
    result.prepare_neighbors(device='cpu',query_batch_size=2)
    return result


def _config(data, weak_batch=1):
    cfg=copy.deepcopy(data.config)
    cfg['full_task'].update(learning_rate=.01,weight_decay=0.,grad_clip=5.,seed=77,
        weak_batch=weak_batch,core_batch=2,accumulation_steps=2,epochs=2,epoch_unit='weak',
        scheduler=dict(name='epoch_warmup_hold_cosine',min_lr=.001,warmup_epochs=.1,hold_until_epoch=1.),
        log_every=1,eval_batch=2,checkpoint_every_epochs=1,monitor_core=False,
        loss=asdict(FullTaskLossConfigV090()),
        model=asdict(FullTaskModelConfigV090(hidden_dim=8,sage_layers=3,
            ontology_layers=1,input_dropout=.1,dropout=.2)))
    return cfg


def _train(data, config, directory, *, stop_step=None, resume=None, encoder_from=None):
    args = SimpleNamespace(work_dir=Path(directory), stop_step=stop_step, stop_epoch=None,
                           epochs=None, resume=Path(resume) if resume else None,
                           encoder_from=Path(encoder_from) if encoder_from else None)
    torch.manual_seed(config['full_task']['seed'])
    runner.train(args, config, data, torch.device('cpu'), rank=0, world=1)
    return torch.load(Path(directory)/'latest.pt', map_location='cpu', weights_only=False)


@pytest.mark.parametrize('weak_batch,variant', [(1,'dynamic'),(2,'fixed'),(4,'dynamic')])
def test_real_training_exact_resume_and_loss_decomposition(tmp_path, weak_batch, variant):
    data = _data(tmp_path/'data', variant)
    config = _config(data, weak_batch)
    complete = _train(data, config, tmp_path/'complete')
    first = _train(data, config, tmp_path/'resumed', stop_step=1)
    resumed = _train(data, config, tmp_path/'resumed', resume=tmp_path/'resumed/latest.pt')
    assert resumed['runner_version'] == resumed['model_architecture_version'] == '0.9.0'
    assert resumed['data_architecture_version'] == '0.8.8'
    assert resumed['phase_origin']['origin_epochs'] == 0
    assert resumed['phase_origin']['encoder_transfer'] is None
    assert 'classifier.weight' not in resumed['model']
    assert 'scorer.protein_projection.weight' in resumed['model']
    for key in ('model','optimizer','scheduler','rng','epoch_stream','epoch_plan'):
        _assert_state_equal(complete[key], resumed[key], key)
    for key in ('loss','grad_norm','contrib_positive','contrib_negative','contrib_core_positive',
                'contrib_core_negative','contrib_weak_positive','contrib_weak_negative'):
        assert [row[key] for row in complete['history']] == [row[key] for row in resumed['history']]
    for row in resumed['history']:
        assert row['loss'] == pytest.approx(row['contrib_positive']+row['contrib_negative'], rel=1e-6)
        assert row['loss'] == pytest.approx(sum(row[f'contrib_{r}_{s}'] for r in ('core','weak') for s in ('positive','negative')), rel=1e-6)
    assert first['step'] == 1


def test_phase_cannot_claim_old_head_exposure_and_resume_contract_frozen():
    values=dict(origin_epochs=0,target_epochs=100,origin_weak_exposures=0,
                origin_core_exposures=0,origin_optimizer_updates=0,
                steps_per_epoch=3,n_weak=10,n_core=8,global_weak_batch=4,global_core_batch=2)
    phase=PhasePlanV090(**values)
    assert phase.progress(3)['lifetime_weak_exposures'] == 10
    assert phase.progress(3)['lifetime_core_exposures'] == 6
    with pytest.raises(ValueError,match='start at zero'):
        PhasePlanV090(**{**values,'origin_epochs':1,'origin_weak_exposures':10,
                        'origin_core_exposures':6,'origin_optimizer_updates':3})
    cfg=dict(weak_batch=4,core_batch=2,epochs=100,epoch_unit='weak',checkpoint_every_epochs=5)
    contract=resume_contract(cfg,phase,1)
    validate_resume_contract(copy.deepcopy(contract),contract)
    changed=copy.deepcopy(contract);changed['full_task']['checkpoint_every_epochs']=10
    with pytest.raises(ValueError,match='resume requires identical'):
        validate_resume_contract(changed,contract)


def test_metric_reuses_original_kernel_and_new_identity_rejects_old_mode():
    from scripts.nbs.eval_nbs_full_task_v089 import compute_fmax_only as original
    y=np.array([[1,0,1],[0,1,0],[0,0,0]],bool)
    p=np.array([[.5,.4,.3],[.2,.9,.1],[.4,.4,.4]],np.float32)
    assert evaluator.compute_fmax_only(y,p) == original(y,p)
    m=dict(model_architecture_version='0.9.0',data_architecture_version='0.8.8',
           encoder_variant='hetero_tuned',prediction_mode='shared_bilinear',
           model_config=dict(encoder_variant='hetero_tuned',prediction_mode='shared_bilinear'),
           sampler_config={'mode':'dynamic'},variant='dynamic',
           uses_dense_backbone_logits_in_forward=False,output_fusion='none')
    evaluator._encoder_identity(m)
    with pytest.raises(ValueError,match='shared_bilinear'):
        evaluator._encoder_identity({**m,'prediction_mode':'direct'})


def test_gate_keeps_original_baseline_and_protocol_and_no_untrained_claim():
    baseline=Path(__file__).resolve().parents[3]/'BASELINE_CONTRACT_v088.json'
    baseline=json.loads(baseline.read_text())
    result=gate.check_reports(baseline)
    assert result['status']=='未验证' and result['new_Fmax'] is None
    assert result['effective_threshold']==pytest.approx(60.72314936447243)
    assert gate.BASELINE_COMPARISON_SHA256 == 'ad34e5baff69e092974ed42aa4e236046c5af89a4a3d2ee81c4291be8a3fc04e'


def test_real_old_checkpoint_encoder_only_starts_new_head_phase_at_zero(tmp_path):
    data=_data(tmp_path/'data')
    config=_config(data)
    old_config=copy.deepcopy(config)
    old_config['release_version']=old_config['model_version']='0.8.8'
    old_config['stage']['name']='nbs_v088_hetero_tuned'
    old_config['full_task']['model']={key:value for key,value in old_config['full_task']['model'].items()
        if key not in {'jk_mode','edge_dropout','scorer_initial_bias','inference_go_chunk_size'}}
    old_config['full_task']['model']['prediction_mode']='direct'
    old_config['full_task'].update(checkpoint_epochs=[1,2],checkpoint_interval=0)
    old_data=FullTaskDataV088(old_config,stores=data.stores)
    old_data.ensure_prepared()
    old_args=SimpleNamespace(work_dir=tmp_path/'old',stop_step=None,stop_epoch=None,epochs=None,resume=None)
    torch.manual_seed(old_config['full_task']['seed'])
    runner.source_runner.legacy_runner.train(old_args,old_config,old_data,torch.device('cpu'),0,1)
    source=torch.load(tmp_path/'old/nbs_epoch2.pt',weights_only=False)
    fresh=_train(data,config,tmp_path/'new',stop_step=1,encoder_from=tmp_path/'old/nbs_epoch2.pt')
    transfer=fresh['phase_origin']['encoder_transfer']
    assert transfer['source_encoder_epochs']==2
    assert transfer['source_encoder_optimizer_updates']==source['step']
    assert transfer['old_head_transferred'] is False
    assert fresh['lifetime_optimizer_updates']==1
    assert fresh['phase_origin']['origin_epochs']==0
    assert fresh['phase_origin']['origin_weak_exposures']==0
    assert fresh['phase_origin']['initialization']=='fresh'
    assert fresh['prior_metadata']['per_go_frequency_prior'] is False
    # Resume must keep the separate encoder lineage while continuing only its new schedule.
    resumed=_train(data,config,tmp_path/'new',resume=tmp_path/'new/latest.pt')
    assert resumed['phase_origin']==fresh['phase_origin']
    assert resumed['step']==4


def test_encoder_transfer_rejects_heldout_label_initialization(tmp_path):
    data=_data(tmp_path/'data')
    data.seen_task_mask=np.array([False,True,False,True,True,True])
    with pytest.raises(ValueError,match='forbidden with unseen'):
        runner.initialize_encoder_from({},tmp_path/'old.pt','a'*64,{},data,None,None,None,{},None,None,'dynamic')


def test_real_export_reuses_source_bound_probabilities_and_fixed_epoch(tmp_path,monkeypatch):
    from test_full_task_data_v080 import inference_fixture
    data=_data(tmp_path/'data')
    config=_config(data,weak_batch=2)
    saved=_train(data,config,tmp_path/'train')
    input_dir=inference_fixture(tmp_path/'data',data)
    args=SimpleNamespace(stage='export',checkpoint=tmp_path/'train/nbs_epoch2.pt',
                         input_dir=input_dir,work_dir=tmp_path/'predictions',
                         ablation='full',metadata_file=None)
    def no_metrics(*args,**kwargs):
        raise AssertionError('export cannot launch metrics')
    monkeypatch.setattr(runner.subprocess,'run',no_metrics)
    runner.evaluate(args,config,data,torch.device('cpu'))
    directory=args.work_dir/f"eval_step{saved['step']}"/'full'
    path=directory/'nbs_ind_test_prob.f32.npy'
    assert np.load(path).shape==(2,6)
    manifest=json.loads((directory/'nbs_full_task_prediction_manifest.json').read_text())
    evaluator._encoder_identity(manifest)
    evaluator._phase_identity(manifest)
    digest=runner.sha256(path)
    runner.evaluate(args,config,data,torch.device('cpu'))
    assert runner.sha256(path)==digest
    args.checkpoint=tmp_path/'train/latest.pt'
    with pytest.raises(ValueError,match='fixed completed-epoch'):
        runner.evaluate(args,config,data,torch.device('cpu'))


def _synthetic_gate_report():
    # Artificial scores exercise the gate, and never claim trained-model quality.
    base=json.loads((Path(__file__).resolve().parents[3]/'BASELINE_CONTRACT_v088.json').read_text())
    new=copy.deepcopy(base)
    phase=PhasePlanV090(origin_epochs=0,target_epochs=100,origin_weak_exposures=0,
        origin_core_exposures=0,origin_optimizer_updates=0,steps_per_epoch=10,
        n_weak=100,n_core=50,global_weak_batch=10,global_core_batch=2)
    origin=dict(initialization='fresh',origin_epochs=0,origin_weak_exposures=0,
        origin_core_exposures=0,origin_optimizer_updates=0,source_checkpoint=None,
        source_checkpoint_sha256=None,source_runner_version=None,source_phase_step=0,
        source_world_size=0,source_optimizer_state_reused=False,source_rng_state_reused=False,
        new_training_seed=77,encoder_transfer=None)
    fields=runner._phase_fields(phase,10)
    source_flags=runner.source_flags('full')
    before=base['prediction_provenance']
    m=dict(version='0.9.0',runner_version='0.9.0',model_architecture_version='0.9.0',
        data_architecture_version='0.8.8',variant='dynamic',encoder_variant='hetero_tuned',
        prediction_mode='shared_bilinear',output_fusion='none',uses_dense_backbone_logits_in_forward=False,
        uses_expert_probability_in_nbs_forward=False,uses_modelout_probability_in_nbs_forward=False,
        uses_modelout_probability_as_target=False,ablation='full',branch='final',source_flags=source_flags,
        model_config={'encoder_variant':'hetero_tuned','prediction_mode':'shared_bilinear'},
        sampler_config={'mode':'dynamic'},phase_plan=asdict(phase),phase_origin=origin,
        checkpoint_sha256='a'*64,output_probability_sha256='b'*64,
        input_manifest_sha256=before['input_manifest_sha256'],num_task_go=21312,
        prediction_space='complete_task_classifier_columns',step=10,**fields)
    provenance={**before,**{key:m[key] for key in evaluator.ENCODER_IDENTITY_KEYS},
        **evaluator._phase_identity(m), 'manifest':m,'step':10,
        'probability_sha256':'b'*64,'prediction_manifest_sha256':'c'*64,'checkpoint_sha256':'a'*64,
        'ablation':'full','branch':'final','source_flags':source_flags,
        'contract':{'data':{'v090_label_split':{'scope':'closed_set_all_seen'}}}}
    new.update(evaluation_version='0.9.0',prediction_provenance=provenance,
        evaluation_scope='standard_protein_fmax_only',metric_contract={'primary_backend':'not_computed','evaluation_scope':'standard_protein_fmax_only'})
    new['methods']['NBS_final']['standardized']=copy.deepcopy(base['methods']['backbone_base']['standardized'])
    new['methods']['NBS_final']['standardized']['standard_protein_fmax']=61.
    return base,new


def test_strict_new_gate_accepts_synthetic_boundary_and_rejects_contract_drift():
    base,new=_synthetic_gate_report()
    result=gate.check_reports(base,new)
    assert result['status']=='达标' and result['fixed_conditions_verified']
    for field,value,message in [('prediction_mode','direct','shared_bilinear'),
                                ('uses_dense_backbone_logits_in_forward',True,'fusion')]:
        changed=copy.deepcopy(new);changed['prediction_provenance'][field]=value
        with pytest.raises(ValueError,match=message):
            gate.check_reports(base,changed)
    changed=copy.deepcopy(new)
    changed['prediction_provenance']['contract']['data']['v090_label_split']['scope']='stage2_label_holdout'
    with pytest.raises(ValueError,match='all-seen'):
        gate.check_reports(base,changed)
    changed=copy.deepcopy(new);changed['methods']['NBS_final']['standardized']['standard_protein_fmax']=57.
    assert gate.check_reports(base,changed)['status']=='未达标'


def _gloo_worker(rank, data, config, directory, rendezvous, stop_step, resume):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(rendezvous),rank=rank,world_size=2)
    try:
        torch.manual_seed(config['full_task']['seed'])
        args=SimpleNamespace(work_dir=Path(directory),stop_step=stop_step,stop_epoch=None,epochs=None,
                             resume=Path(resume) if resume else None,encoder_from=None)
        runner.train(args,copy.deepcopy(config),data,torch.device('cpu'),rank,2)
    finally:
        dist.destroy_process_group()


def test_real_two_process_gloo_training_empty_weak_rank_and_exact_resume(tmp_path):
    # fork preserves read-only mmap fixtures without rewriting paths/mtime in each
    # rank; no CUDA state is initialized anywhere in this CPU-only test.
    import torch.multiprocessing as mp
    data=_data(tmp_path/'data')
    config=_config(data,weak_batch=2)
    config['full_task'].update(core_batch=1,accumulation_steps=1,expected_world_size=2)
    torch.set_num_threads(1)
    runs=[('complete',None,None),('resumed',1,None),('resumed',None,tmp_path/'resumed/latest.pt')]
    for i,(directory,stop,resume) in enumerate(runs):
        try:
            mp.start_processes(_gloo_worker,args=(data,config,tmp_path/directory,tmp_path/f'gloo{i}',stop,resume),
                               nprocs=2,join=True,start_method='fork')
        except mp.ProcessRaisedException as exc:
            # This precise error was observed after attempting real two-rank
            # startup in the restricted runtime. Other DDP failures still fail.
            if 'gloo/transport/tcp/device.cc' in str(exc) and 'Operation not permitted' in str(exc):
                pytest.skip('Observed runtime denial of Gloo TCP device creation (Operation not permitted)')
            raise
    complete=torch.load(tmp_path/'complete/latest.pt',weights_only=False)
    resumed=torch.load(tmp_path/'resumed/latest.pt',weights_only=False)
    assert complete['world_size']==2 and complete['step']==2
    for key in ('model','optimizer','scheduler','rng','epoch_stream','epoch_plan'):
        _assert_state_equal(complete[key],resumed[key],key)
    for row in resumed['history']:
        assert np.isfinite(row['loss'])
        assert row['loss']==pytest.approx(sum(row[f'contrib_{r}_{s}'] for r in ('core','weak') for s in ('positive','negative')),rel=1e-6)


@pytest.mark.parametrize('score,exit_code,status',[(61.,0,'达标'),(57.,1,'未达标')])
def test_checker_cli_exit_semantics_and_evaluation_route(tmp_path,score,exit_code,status):
    import subprocess
    import sys
    base,new=_synthetic_gate_report()
    new['methods']['NBS_final']['standardized']['standard_protein_fmax']=score
    comparison=tmp_path/'comparison.json';comparison.write_text(json.dumps(new))
    baseline=Path(__file__).resolve().parents[3]/'BASELINE_CONTRACT_v088.json'
    output=tmp_path/'acceptance.json'
    command=[sys.executable,str(runner.ROOT/'scripts/nbs/check_nbs_v090_acceptance.py'),
             '--baseline-comparison',str(baseline),'--comparison',str(comparison),'--output',str(output)]
    completed=subprocess.run(command,cwd=runner.ROOT,capture_output=True,text=True)
    assert completed.returncode==exit_code,completed.stderr
    assert json.loads(output.read_text())['status']==status
    # Idempotent repeated evaluation succeeds for both verified outcomes.
    assert runner.run_acceptance_gate(command,output,comparison,baseline)['status']==status
    # Contract drift must not be hidden by a previous valid acceptance file.
    new['prediction_provenance']['output_fusion']='bad'
    comparison.write_text(json.dumps(new))
    with pytest.raises(RuntimeError,match='stale, invalid'):
        runner.run_acceptance_gate(command,output,comparison,baseline)


def test_checker_cli_missing_comparison_is_unverified(tmp_path):
    import subprocess
    import sys
    baseline=Path(__file__).resolve().parents[3]/'BASELINE_CONTRACT_v088.json'
    output=tmp_path/'acceptance.json'
    result=subprocess.run([sys.executable,str(runner.ROOT/'scripts/nbs/check_nbs_v090_acceptance.py'),
        '--baseline-comparison',str(baseline),'--output',str(output)],cwd=runner.ROOT,capture_output=True,text=True)
    assert result.returncode==2
    assert json.loads(output.read_text())['status']=='未验证'


def test_loss_uses_model_seen_mask_even_if_batch_mask_missing(tmp_path):
    data=_data(tmp_path/'data')
    config=_config(data)
    seen=np.array([False,True,False,True,True,True])
    model,_=runner.build_model(data.feature_dim,data.ontology('cpu'),config['full_task']['model'],
                               'dynamic','cpu',seen)
    batch=data.batch([4,0],'cpu',supervision_seed_ids=np.array([4,0]))
    batch['positive_mask'][:,~torch.as_tensor(seen)]=False
    batch['targets']=batch['positive_mask'].float()
    batch=model.sanitize_evidence(batch)
    batch.pop('seen_task_mask')
    _,loss,parts=runner.training_objective(model,batch,FullTaskLossConfigV090(),global_role_counts=(1,1))
    assert torch.isfinite(loss) and parts['seen_go_columns'].item()==4
    batch['seen_task_mask']=torch.ones(6,dtype=torch.bool)
    with pytest.raises(ValueError,match='immutable model'):
        runner.training_objective(model,batch,FullTaskLossConfigV090(),global_role_counts=(1,1))
