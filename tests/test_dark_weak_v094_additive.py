"""No-GPU regression for additive v094 Dark-Weak integration.

Runs against the REAL installed v085 metric kernel and original v043 ranker.
Native model forward/checkpoint decoding must be exercised on the server.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'scripts'/'dark_weak'


def load(path, key):
    import sys
    sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module

native = load(SCRIPTS/'v094_native_adapter.py','v094_addon_native')
graph = load(SCRIPTS/'v094_graph_audit.py','v094_addon_graph')
wrapper = load(SCRIPTS/'v094_unified_score.py','v094_addon_score')
indtest = load(SCRIPTS/'v094_ind_test_supplement.py','v094_addon_indtest')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def fixture(directory, mode='dark_weak_inference'):
    root = Path(directory)
    inp = root/'input'; inp.mkdir()
    work = root/'export'
    ids = ['PROTEIN_2022_A', 'PROTEIN_2022_B']
    ids_file = inp/'protein_ids.txt'; ids_file.write_text('\n'.join(ids)+'\n')
    g = 21312
    goids = [f'GO:{i:07d}' for i in range(1, g+1)]
    goids_file = root/'go_ids.txt'; goids_file.write_text('\n'.join(goids)+'\n')
    registry = root/'go_registry.tsv'
    with registry.open('w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f, delimiter='\t')
        wr.writerow(['go_idx', 'input_go_id'])
        wr.writerows((i,go_id) for i,go_id in enumerate(goids))
    baseline = np.full((2,g), .1, np.float16)
    baseline[0,11] = .25; baseline[1,12] = .26
    np.save(inp/'ind_test_repr.f16.npy',np.ones((2,4),np.float16))
    np.save(inp/'backbone_ind_test_prob.f16.npy',baseline)
    np.save(inp/'candidate_go_index.i32.npy',np.tile(np.arange(512,dtype=np.int32),(2,1)))
    np.save(inp/'candidate_edge_attr.f32.npy',np.ones((2,512,3),np.float32))
    np.save(inp/'test_core_neighbors.i32.npy',np.array([[0,1],[1,0]],np.int32))
    np.save(inp/'test_core_edge_attr.f32.npy',np.ones((2,2,3),np.float32))
    signature = {'pp_topk':2}
    manifest = {
        'task':'bp', 'mode':mode, 'num_proteins':2,
        'protein_ids_file_sha256':native.sha256(ids_file),
        'cache_signature':signature,
        'representation':{'sha256':native.sha256(inp/'ind_test_repr.f16.npy')},
        'base_probability':{'sha256':native.sha256(inp/'backbone_ind_test_prob.f16.npy')},
        'candidate_evidence':{'selector_scope':'full_task','fixed_k':512,
                              'expert_probability_used':False,'label_hint_used':False,
                              'go_index_sha256':native.sha256(inp/'candidate_go_index.i32.npy'),
                              'edge_attr_sha256':native.sha256(inp/'candidate_edge_attr.f32.npy')},
        'external_pp':{'neighbor_index_sha256':native.sha256(inp/'test_core_neighbors.i32.npy'),
                       'edge_attr_sha256':native.sha256(inp/'test_core_edge_attr.f32.npy')},
        'registries':{'go_registry':str(registry), 'go_registry_sha256':native.sha256(registry)},
        'leakage_contract':dict.fromkeys(native.FORBIDDEN_INPUTS,False),
    }
    write_json(inp/'ind_test_input_manifest.json', manifest)
    config = {
        'release_version':'0.9.4', 'model_version':'0.9.4', 'task':'bp',
        'stage':{'name':'nbs_v094_weak_teacher','protein_go_candidate_topk':512},
        'full_task':{
            'experiment_revision':native.REVISION,
            'weak_target_mode':'backbone_binary_plus_modelout_dense_soft',
            'architecture_extension':'v094_single_shared_head_raw_geometry_dense_teacher',
            'model':{'output_strategy':'shared_direct','prediction_mode':'shared_bilinear'},
        }, 'v094_experiment':{'strict_pipeline_zero_shot':False},
    }
    cfg = root/'config.json'; write_json(cfg,config)
    checkpoint = root/'nbs_epoch5.pt'; checkpoint.write_bytes(b'synthetic checkpoint identity')
    pred_dir=work/'eval_step12'/'full'; pred_dir.mkdir(parents=True)
    predicted = np.full((2,g), .05, np.float32)
    predicted[0,11] = .91; predicted[1,12] = .92
    predpath = pred_dir/'nbs_ind_test_prob.f32.npy'; np.save(predpath,predicted)
    prediction_manifest={
       'schema':native.NATIVE_SCHEMA, 'complete':True, 'experiment_revision':native.REVISION,
       'task':'bp', 'checkpoint_sha256':native.sha256(checkpoint),
       'input_manifest_sha256':native.sha256(inp/'ind_test_input_manifest.json'),
       'num_task_go':g, 'output_fusion':'single_shared_GO_scorer',
       'uses_dense_backbone_logits_in_forward':False,
       'uses_expert_probability_in_nbs_forward':False,
       'zero_shot_scope':'Stage2-only; reused upstream representations may encode held-out GO',
       'model_config':config['full_task']['model'], 'prediction_implementation':{'model.py':'hash'},
       'step':12,'epoch':5,'probability_path':str(predpath),
       'output_probability_sha256':native.sha256(predpath),
    }
    pred_manifest=pred_dir/'nbs_full_task_prediction_manifest.json';write_json(pred_manifest,prediction_manifest)
    args = SimpleNamespace(task='bp',config=cfg,checkpoint=checkpoint,
        input_dir=inp,expected_protein_ids=ids_file,go_ids=goids_file,
        expected_mode=mode,expected_epoch=5,work_dir=work,
        train_runner=ROOT/'scripts/nbs/train_nbs_full_task_v094.py',device='cpu',
        graph_audit_json=None,allow_self_support=False,
        out=root/'adapter.json',overwrite_adapter=False)
    return locals()


class V094AddonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = fixture(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_input_config_and_attached_native_prediction(self):
        f=self.f
        native.validate_config(f['config'],'bp')
        prepared=native.validate_input(f['config'],f['args'])
        self.assertEqual(prepared['candidate_pool_size'],512)
        a=native.make_adapter(f['args'],execute_native=False)
        self.assertEqual(a['epoch'],5)
        self.assertEqual(a['output_fusion'],'single_shared_GO_scorer')
        self.assertFalse(a['graph_audit']['strict_inductive_certified'])
        result=wrapper.verify_v094_adapter(f['args'].out,
            {'model':'nbs','namespace':'bp','probability':f['predpath'].resolve()},
            {'nbs':f['ids_file'].resolve()},{'nbs':f['goids_file'].resolve()},f['goids'])
        self.assertEqual(result['epoch'],5)

    def test_bad_scope_and_missing_shard_hash_rejected(self):
        f=self.f
        path=f['inp']/'ind_test_input_manifest.json'
        manifest=native.json_file(path)
        manifest['candidate_evidence']['selector_scope']='rare_first'
        write_json(path,manifest)
        with self.assertRaisesRegex(ValueError,'full-task candidate'):
            native.validate_input(f['config'],f['args'])

    def test_candidate_teacher_or_label_hint_rejected(self):
        f=self.f
        path=f['inp']/'ind_test_input_manifest.json'
        for flag in ('expert_probability_used', 'label_hint_used'):
            manifest=native.json_file(path)
            manifest['candidate_evidence'][flag]=True
            write_json(path, manifest)
            with self.assertRaisesRegex(ValueError,'exclude Expert probabilities'):
                native.validate_input(f['config'],f['args'])
            manifest['candidate_evidence'][flag]=False
            write_json(path, manifest)

    def test_wrong_model_output_semantics_rejected(self):
        f=self.f
        manifest=native.json_file(f['pred_manifest'])
        manifest['output_fusion']='none'
        write_json(f['pred_manifest'],manifest)
        with self.assertRaisesRegex(ValueError,'scorer or zero-shot'):
            native.make_adapter(f['args'],execute_native=False)

    def test_pred_file_modified_rejected(self):
        f=self.f
        arr=np.load(f['predpath'])
        arr[0,2]=.7;np.save(f['predpath'],arr)
        with self.assertRaisesRegex(ValueError,'bytes changed'):
            native.make_adapter(f['args'],execute_native=False)

    def test_do_not_write_into_native_training_directory(self):
        f=self.f
        (f['work']/'latest.pt').write_bytes(b'actual training directory')
        with self.assertRaisesRegex(ValueError,'training directory'):
            native.make_adapter(f['args'],execute_native=False)

    def test_graph_query_ghost_and_self_support(self):
        query=np.array([0,1]); blocks=[{'dst_global_ids':np.array([-1,-2]),
            'edge_index':np.array([[2,2],[0,1]]), 'edge_type':np.array([0,0])}]
        batch={'sampled_global_ids':np.array([-1,-2,5]),
               'sampled_seed_index':np.array([0,1]),
               'blocks':blocks,'_audit_cosine_relation_index':0}
        got=graph.graph_schema_checks(query,batch,np.array([5,-1]))
        self.assertEqual(got['batch_self_support_hits'],1)
        bad=dict(batch);bad['base_logits']=np.zeros((2,1))
        with self.assertRaisesRegex(ValueError,'information entered'):
            graph.graph_schema_checks(query,bad)
        bad=dict(batch);bad['_audit_cosine_relation_index']=4
        with self.assertRaisesRegex(ValueError,'noncosine'):
            graph.graph_schema_checks(query,bad)

    def test_graph_identity_discovery_only_if_explicit_column(self):
        f=self.f
        registry=f['root']/'registry.csv'
        registry.write_text('protein_idx,role,role_row_idx\n0,weak,0\n')
        self.assertIsNone(graph.discover_original_ids(registry,['A']))
        registry.write_text('protein_idx,protein_id\n0,A\n1,B\n')
        np.testing.assert_array_equal(graph.discover_original_ids(registry,['A','C']),[0,-1])

    def test_v085_ind_test_supplementary_direct_real_kernel(self):
        f=self.f
        meta={'ind_test':{'biological_process':{
            'proteins':f['ids'],'prop_annotations':[[11],[12]],
            'annotations':[[11],[12]]}}}
        metadata=f['root']/'gold.pkl'
        with metadata.open('wb') as h:pickle.dump(meta,h)
        input_manifest=native.json_file(f['inp']/'ind_test_input_manifest.json')
        input_manifest['mode']='ind_test'
        write_json(f['inp']/'ind_test_input_manifest.json',input_manifest)
        pred=native.json_file(f['pred_manifest'])
        pred['input_manifest_sha256']=native.sha256(f['inp']/'ind_test_input_manifest.json')
        write_json(f['pred_manifest'],pred)
        output=f['root']/'native_scores.json'
        args=SimpleNamespace(config=f['cfg'],task='bp',metadata=metadata,split='ind_test',
            checkpoint=f['checkpoint'],prediction_manifest=f['pred_manifest'],
            input_dir=f['inp'],go_ids=f['goids_file'],expected_epoch=5,
            reference=[],reference_ids=[],reference_go_ids=[],out=output,overwrite=False)
        result=indtest.run(args)
        self.assertGreaterEqual(result['models']['NBS_final']['Fmax_percent'],result['models']['backbone_base']['Fmax_percent'])
        self.assertIn('AuPRC_percent',result['models']['NBS_final'])
        self.assertIn('AP_percent',result['models']['NBS_final'])
        self.assertEqual(result['native_metrics_parity'],'not_available_native_export_only')
        self.assertTrue(output.is_file())

    def test_v094_unified_score_end_to_end_on_original_ranker(self):
        f=self.f
        adapter=native.make_adapter(f['args'],execute_native=False)
        metadata=f['root']/'gold.pkl'
        data={'dark_weak_test_direct':{'biological_process':{
            'proteins':f['ids'],'annotations':[[11],[12]],
            'prop_annotations':[[11],[12]]}}}
        with metadata.open('wb') as h:pickle.dump(data,h)
        folder=f['root']/'benchmark/labels/bp';folder.mkdir(parents=True)
        (folder/'go_ids.txt').write_text('\n'.join(f['goids'])+'\n')
        table=folder/'vocabulary_mapping.tsv'
        with table.open('w') as handle:
            handle.write('model_column\tmodel_go_id\tcanonical_go_id_2022\n')
            for i,g in enumerate(f['goids']):handle.write(f'{i}\t{g}\t{g}\n')
        output=f['root']/'score'
        args=['--native-v094-adapter',f'nbs={f["args"].out}',
              '--namespace','bp','--split','dark_weak_test_direct',
              '--unified-root',str(f['root']/'benchmark'), '--metadata',str(metadata),
              '--prediction',f'nbs={f["predpath"]}',
              '--prediction-ids',f'nbs={f["ids_file"]}',
              '--prediction-go-ids',f'nbs={f["goids_file"]}',
              '--prediction',f'backbone={f["inp"]/"backbone_ind_test_prob.f16.npy"}',
              '--prediction-ids',f'backbone={f["ids_file"]}',
              '--prediction-go-ids',f'backbone={f["goids_file"]}',
              '--reference','backbone','--bootstrap','3','--ks','1','10',
              '--out',str(output)]
        result=wrapper.run(args)
        self.assertEqual(result['native_v094']['nbs']['training_revision'],native.REVISION)
        metric=json.loads((output/'bp/dark_weak_test_direct/complete_metrics.json').read_text())
        self.assertIn('AuPRC_percent',metric['metrics']['nbs']['v085_standardized'])
        self.assertTrue((output/'bp_v094_provenance.json').is_file())
        self.assertTrue((output/'bp_complete_manifest.json').is_file())


if __name__=='__main__':
    unittest.main()
