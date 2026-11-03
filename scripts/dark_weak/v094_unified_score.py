#!/usr/bin/env python3
"""Safe v094 provenance layer around the EXISTING Dark–Weak v085 evaluator.

All Stage-1/NBS metrics are computed by the existing
scripts/dark_weak/run_complete_evaluation.py.  This additive wrapper verifies
v094 NBS prediction lineage, then writes a separate provenance sidecar; it does
NOT change any original evaluation or acceptance code or native reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT/'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT/'scripts'))

from scripts.dark_weak import v094_native_adapter as native
from scripts.dark_weak import run_complete_evaluation as original
from scripts import eval_dark_weak_predictions as ranking


def parse_kv(values, what):
    result = {}
    for value in values:
        if '=' not in value:
            raise ValueError(f'{what} requires MODEL=PATH, got {value!r}')
        name, path = value.split('=', 1)
        if not name or not path or name in result:
            raise ValueError(f'{what}: invalid or duplicate model {name!r}')
        result[name] = Path(path).resolve(strict=True)
    return result


def verify_v094_adapter(adapter_path: Path, supplied: dict, ids: dict, goids: dict,
                        vocab: list[str], require_graph_audit=False) -> dict:
    adapter_path = Path(adapter_path).resolve(strict=True)
    a = native.json_file(adapter_path)
    name = supplied['model']
    if a.get('schema') != native.SCHEMA or a.get('runner_version') != '0.9.4':
        raise ValueError('The native adapter is not a v094 weak-teacher adapter')
    if a.get('experiment_revision') != native.REVISION or a.get('task') != supplied['namespace']:
        raise ValueError('v094 adapter experiment revision/task differs from request')
    if Path(a['probability']).resolve() != supplied['probability'] or native.sha256(supplied['probability']) != a['probability_sha256']:
        raise ValueError('v094 adapter probability does not match scored matrix')
    if Path(a['protein_ids']).resolve() != ids[name] or native.sha256(ids[name]) != a['protein_ids_sha256']:
        raise ValueError('v094 adapter protein IDs differ from scored ID file')
    if Path(a['go_ids']).resolve() != goids[name] or native.sha256(goids[name]) != a['go_ids_sha256']:
        raise ValueError('v094 adapter GO ID file differs from scored GO columns')
    if native.read_ids(goids[name]) != vocab:
        raise ValueError('v094 model GO columns differ from canonical immutable task model columns')
    cpath = Path(a['config']).resolve(strict=True)
    if native.sha256(cpath) != a['config_sha256']:
        raise ValueError('v094 training configuration changed after export')
    cfg = native.json_file(cpath)
    native.validate_config(cfg, supplied['namespace'])
    ckp = Path(a['checkpoint']).resolve(strict=True)
    if native.sha256(ckp) != a['checkpoint_sha256']:
        raise ValueError('Native v094 checkpoint changed')
    inp = Path(a['prepared_input_manifest']).resolve(strict=True)
    if native.sha256(inp) != a['prepared_input_manifest_sha256']:
        raise ValueError('Native prepared input manifest changed')
    pred = Path(a['native_prediction_manifest']).resolve(strict=True)
    if native.sha256(pred) != a['native_prediction_manifest_sha256']:
        raise ValueError('Native v094 prediction manifest changed')
    m = native.json_file(pred)
    if (m.get('schema') != native.NATIVE_SCHEMA or m.get('complete') is not True
            or m.get('checkpoint_sha256') != a['checkpoint_sha256']
            or m.get('input_manifest_sha256') != a['prepared_input_manifest_sha256']
            or m.get('output_probability_sha256') != a['probability_sha256']
            or m.get('experiment_revision') != native.REVISION
            or m.get('output_fusion') != 'single_shared_GO_scorer'
            or m.get('uses_dense_backbone_logits_in_forward') is not False
            or m.get('uses_expert_probability_in_nbs_forward') is not False
            or m.get('zero_shot_scope') != 'Stage2-only; reused upstream representations may encode held-out GO'
            or m.get('epoch') != a['epoch'] or m.get('step') != a['step']
            or m.get('num_task_go') != len(vocab)):
        raise ValueError('Native prediction no longer agrees with v094 model semantics/identity')
    if not isinstance(m.get('prediction_implementation'), dict) or not m['prediction_implementation']:
        raise ValueError('Missing native v094 prediction implementation hashes')
    original_input_dir = Path(inp).parent
    if Path(a['protein_ids']).resolve() != (original_input_dir/'protein_ids.txt').resolve():
        raise ValueError('Native prediction protein order is not the prepared-input order')
    graph = a.get('graph_audit', {})
    if require_graph_audit:
        if graph.get('status') in (None, 'not_run') or graph.get('audited_rows', 0) != len(native.read_ids(ids[name])):
            raise ValueError('A complete native graph audit is required by --require-graph-audit')
        if graph.get('original_global_ids_provided') is not True:
            raise ValueError('Strict graph-audit gate requires original global protein IDs')
        if graph.get('batch_self_support_hits'):
            raise ValueError('Cannot pass strict graph-audit gate with self-support')
    if graph.get('audit_path'):
        auditpath = Path(graph['audit_path']).resolve(strict=True)
        if native.sha256(auditpath) != graph.get('audit_sha256'):
            raise ValueError('Native v094 graph audit file changed after adapter generation')
    return {'schema': a['schema'], 'adapter': str(adapter_path),
            'adapter_sha256': native.sha256(adapter_path), 'checkpoint_sha256': a['checkpoint_sha256'],
            'epoch': a['epoch'], 'step': a['step'], 'training_revision': a['experiment_revision'],
            'native_prediction_manifest': str(pred),
            'native_prediction_manifest_sha256': a['native_prediction_manifest_sha256'],
            'input_manifest_sha256': a['prepared_input_manifest_sha256'],
            'prediction_sha256': a['probability_sha256'],
            'graph_audit': graph,
            'protocol': ('Retrospective historical weak-to-strong recovery; '
                         'training-seen weak protein status must be separately reported; '
                         'no strict protein-inductive or full-pipeline GO zero-shot claim')}


def run(argv=None):
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('--native-v094-adapter', action='append', required=True,
                   help='MODEL=/path/to/v094_adapter.json; one per NBS v094 predictor')
    p.add_argument('--require-graph-audit', action='store_true')
    p.add_argument('--overwrite-unified-report', action='store_true')
    a, remaining = p.parse_known_args(argv)
    q = argparse.ArgumentParser(add_help=False)
    q.add_argument('--namespace', choices=('bp','mf','cc'), required=True)
    q.add_argument('--prediction', action='append', required=True)
    q.add_argument('--prediction-ids', action='append', required=True)
    q.add_argument('--prediction-go-ids', action='append', required=True)
    q.add_argument('--unified-root', type=Path, default=Path('data/dark_weak_test/unified'))
    q.add_argument('--out', type=Path, default=Path('data/dark_weak_test/evaluations/complete'))
    parsed, _ = q.parse_known_args(remaining)
    preds = parse_kv(parsed.prediction, '--prediction')
    ids = parse_kv(parsed.prediction_ids, '--prediction-ids')
    gos = parse_kv(parsed.prediction_go_ids, '--prediction-go-ids')
    adapters = parse_kv(a.native_v094_adapter, '--native-v094-adapter')
    if not adapters.keys() <= preds.keys() or set(preds) != set(ids) or set(preds) != set(gos):
        raise ValueError('Every model needs exactly one probability, protein ID and GO ID file')
    if not adapters:
        raise ValueError('Expected at least one v094 native adapter')
    vocab = ranking.read_go_list(parsed.unified_root/'labels'/parsed.namespace/'go_ids.txt')
    verified = {}
    for model, adapter_path in adapters.items():
        verified[model] = verify_v094_adapter(
            adapter_path,
            {'model': model, 'namespace': parsed.namespace, 'probability': preds[model]},
            ids, gos, vocab, require_graph_audit=a.require_graph_audit)
    output = parsed.out.resolve()
    primary = output/f'{parsed.namespace}_complete_manifest.json'
    sidecar = output/f'{parsed.namespace}_v094_provenance.json'
    if (primary.exists() or sidecar.exists()) and not a.overwrite_unified_report:
        raise FileExistsError(f'Target evaluation already exists: {primary} or {sidecar}; use a NEW --out or --overwrite-unified-report')
    # We intentionally DO NOT pass v094 to the v088/v089 native adapter CLI,
    # whose exact 0.8.x architecture/output-fusion checks must stay unchanged.
    print('[v094 provenance] verified native checkpoint, input, logits and full GO order', flush=True)
    original.main(remaining)
    if not primary.is_file():
        raise FileNotFoundError(f'Legacy unified scoring did not create {primary}')
    sidecar_payload = {
        'schema': 'latence-dark-weak-v094-unified-provenance-v1',
        'native_v094': verified,
        'namespace': parsed.namespace,
        'existing_unified_scoring_manifest': str(primary),
        'existing_unified_scoring_manifest_sha256': native.sha256(primary),
        'metrics_implementation': 'nbs_pg.full_task_metrics_v085.compute_standard_metrics',
        'data_use': ('Full Dark–Weak eval pickle labels used ONLY after native prediction export; '
                     'GO ancestor closure used for Fmax/AuPRC/AP; direct GO used for recall/ranking'),
        'native_training_and_evaluation_files_modified': False,
    }
    native.atomic_json(sidecar, sidecar_payload, overwrite=a.overwrite_unified_report)
    print(f'[v094 Dark–Weak provenance] {sidecar}', flush=True)
    return sidecar_payload


if __name__ == '__main__':
    run(sys.argv[1:])
