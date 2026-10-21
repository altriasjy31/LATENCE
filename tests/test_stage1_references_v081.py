"""Reference forward parity, ID alignment, cache integrity and label isolation."""
from __future__ import annotations

import ast
import argparse
import importlib.util
import json
import math
from pathlib import Path
import pickle
import shutil
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nbs/prepare_nbs_stage1_references_v081.py"
spec = importlib.util.spec_from_file_location("stage1_references_v081_tested", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
PRODUCTION_LOAD_RUNTIME = module.load_runtime


def real_query_decoder():
    """Execute the actual decoder definitions without importing GPU/MSA tooling."""
    path = ROOT / "experiments/weak_exp_train_detr.py"
    source = ast.parse(path.read_text())
    wanted = {"prob_to_logit", "build_anchor_logits_from_prob", "safe_gather_2d", "build_backbone_memory",
              "LightweightCandidateSelector", "TrainableOntologyQueryEmbedding", "ExpertGuidedOntologyQueryDecoder"}
    nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    namespace = dict(torch=torch, nn=nn, F=F, math=math, Optional=Optional, Dict=Dict, Tuple=Tuple,
                     List=List, _PROB_EPS=1e-6)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    decoder = namespace["ExpertGuidedOntologyQueryDecoder"](
        num_classes=3, classifier_dim=4, memory_dim=4, query_dim=4, num_heads=1,
        num_layers=1, ffn_dim=8, dropout=0, topk=2, delta_max=0.8,
        query_decoder_logit_base_mode="mix_expert_base_anchor", expert_base_mix_alpha=0.8,
        anchor_delta_gate_init=0.2, use_learnable_selector=False,
        selector_use_protein_term_affinity=False, memory_mode="pooled")
    with torch.no_grad():
        decoder.delta_head[-1].bias.fill_(0.7)
    return decoder.eval()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 3, bias=False)
        self.query_decoder = real_query_decoder()

    def backbone(self, encoded, permute_dims=None, return_embedding=False):
        index = encoded[:, 0].float()
        logits = torch.stack([index * 0.2 - 0.3, index * -0.1 + 0.6, index * 0.05 - 1.2], dim=-1)
        hidden = torch.stack([index, index + 0.2, index * 0.3, index * -0.4], dim=-1)
        return logits, hidden


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    project = tmp_path / "project"
    inputs = project / "inputs"
    inputs.mkdir(parents=True)
    (project / "scripts/nbs").mkdir(parents=True)
    for rel in ("scripts/nbs/prepare_nbs_ind_test_inputs.py", "scripts/export_weak_graph_predictions.py"):
        shutil.copyfile(ROOT / rel, project / rel)
    (project / "experiments").mkdir()
    (project / "experiments/eval_weak_ind_test_detr_diagnostics.py").write_text("# fake dependency\n")
    (project / "data/ind_MSA_bin").mkdir(parents=True)
    (project / "data/external_probs/esm2_3b").mkdir(parents=True)
    checkpoint = project / "stage1.pt"
    checkpoint.write_bytes(b"test checkpoint identity")
    msa = project / "data/ind_MSA_bin/index.pkl"
    msa.write_bytes(pickle.dumps({"proteins": ["a", "b", "c"]}))
    metadata = project / "metadata.pkl"
    metadata.write_bytes(pickle.dumps({"ind_test": {"bp": {"proteins": ["a", "b", "c"],
                       "prop_annotations": ["REAL GOLD MUST NOT ENTER DATASET"] * 3}}}))
    proteins = ["c", "a", "b"]
    (inputs / "protein_ids.txt").write_text("\n".join(proteins) + "\n")
    registry = project / "go_registry.tsv"
    # Canonical aliases are valid, while original task columns remain distinct.
    registry.write_text("go_idx\tinput_go_id\tgo_id\n0\tGO:A\tGO:A\n1\tGO:B\tGO:A\n2\tGO:C\tGO:C\n")
    expert = np.asarray([[.9, .123456, .7], [.2, .8, .3], [.4, .1, .95]], dtype=np.float32)
    external = project / "data/external_probs/esm2_3b/bp_predictions.prop.float16.npy"
    np.save(external, expert)
    model = TinyModel()
    cached_base = torch.sigmoid(model.backbone(torch.tensor([[2], [0], [1]]))[0]).numpy()
    np.save(inputs / "backbone_ind_test_prob.f16.npy", cached_base.astype(np.float16))
    weak = project / "weak_manifest.json"
    semantics = {"prediction_key": "modelout::mix_expert_base_anchor::decoderprob::expert",
                 "query_decoder_logit_base_mode": "mix_expert_base_anchor", "expert_base_mix_alpha": .8,
                 "decoder_prob_source": "expert", "topk_source": "external_topk", "external_prob_blend_alpha": 1.0,
                 "label_hint_used": False}
    weak.write_text(json.dumps({"checkpoint": {"path": str(checkpoint), "sha256": module.sha256(checkpoint)},
                               "model_semantics": {"modelout": semantics}}))
    manifest = {"task": "bp", "num_proteins": 3, "sequence_encoding": {"mode": "metadata_msa_binary"},
                "cache_signature": {"num_classes": 3, "stage1_checkpoint_sha256": module.sha256(checkpoint),
                                    "msa_index_sha256": module.sha256(msa), "weak_graph_manifest_sha256": module.sha256(weak)},
                "registries": {"weak_graph_manifest": str(weak), "go_registry": str(registry)}}
    (inputs / "ind_test_input_manifest.json").write_text(json.dumps(manifest))
    options = SimpleNamespace(permute_dims=[0, 3, 2, 1], query_decoder_topk=2)
    state = {"loads": 0, "fail": False, "dataset": None}

    def build_dataset(opt, **kwargs):
        payload = pickle.loads(Path(opt.file_address).read_bytes())
        block = payload["ind_test"]["biological_process"]
        assert block["prop_annotations"] == [[], [], []]
        assert Path(opt.working_address) == msa
        assert kwargs == {"mode": "ind_test", "task": "biological_process", "need_proteins": True}

        class Dataset:
            return_labels = True

            def __len__(self):
                return 3

        state["dataset"] = Dataset()
        return state["dataset"]

    def make_loader(dataset, **kwargs):
        assert dataset.return_labels is False
        # Deliberately differ from both metadata and independent input order.
        return [(["b", "c"], torch.tensor([[1], [2]])), (["a"], torch.tensor([[0]]))]

    def runtime(args, context):
        state["loads"] += 1
        if state["fail"]:
            raise RuntimeError("controlled inference failure")
        return {"torch": torch, "device": torch.device("cpu"), "model": model,
                "model_args": options, "opt": SimpleNamespace(),
                "exp": SimpleNamespace(build_msa_dataset=build_dataset, make_loader=make_loader,
                                       set_model_proteins=lambda m, ps: None),
                "runtime_source_files": {"test_runtime": module.source_record(SCRIPT)},
                "checkpoint_load": {"missing_query_decoder": [], "unexpected_query_decoder": []},
                "config_sources": [], "resolved_arg_sources": {}, "configuration_conflicts": {}}

    monkeypatch.setattr(module, "load_runtime", runtime)
    argv = ["--project-root", str(project), "--input-dir", str(inputs), "--metadata-file", str(metadata),
            "--device", "cpu", "--no-amp", "--batch-size", "2"]
    return SimpleNamespace(project=project, inputs=inputs, expert=expert, external=external,
                           model=model, argv=argv, state=state, manifest=manifest, checkpoint=checkpoint)


def test_exact_stage1_modelout_keeps_anchor_gate_and_original_query_topk(fixture):
    result = module.main(fixture.argv)
    outputs = result["outputs"]
    expert = np.load(outputs["expert_prob"]["path"])
    np.testing.assert_array_equal(expert, fixture.expert[[2, 0, 1]].astype(np.float16).astype(np.float32))
    base_logits, hidden = fixture.model.backbone(torch.tensor([[2], [0], [1]]))
    with torch.no_grad():
        qout = fixture.model.query_decoder(h=hidden, base_logits=base_logits,
            classifier_weight=fixture.model.classifier.weight, external_prob=torch.from_numpy(expert),
            has_external_prob=torch.ones(3, dtype=torch.bool), topk_source="external_topk",
            external_prob_blend_alpha=1.0, y_hint=None)
    expected = torch.sigmoid(qout["logits"].float()).numpy()
    actual = np.load(outputs["stage1_modelout"]["path"])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    naive = base_logits.clone().scatter_add_(1, qout["topk_idx"], qout["delta"])
    assert not np.allclose(actual, torch.sigmoid(naive).numpy())
    assert not np.allclose(actual, expert)
    assert qout["topk_idx"].shape[1] == 2
    assert result["semantics"]["labels_consumed_by_model"] is False
    assert result["backbone_parity"]["pairs_above_atol"] == 0
    assert result["cache_signature"]["external_column_identity_independently_verified"] is False
    assert Path(outputs["go_ids"]["path"]).read_text().splitlines() == ["GO:A", "GO:B", "GO:C"]
    assert not (Path(outputs["expert_prob"]["path"]).parent / "msa_selection.pkl").exists()


def test_reuse_before_runtime_and_output_tampering_forces_rebuild(fixture):
    result = module.main(fixture.argv)
    assert fixture.state["loads"] == 1
    module.main(fixture.argv + ["--cache-policy", "require"])
    assert fixture.state["loads"] == 1
    path = Path(result["outputs"]["stage1_modelout"]["path"])
    np.save(path, np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(RuntimeError, match="Compatible Stage-1"):
        module.main(fixture.argv + ["--cache-policy", "require"])
    module.main(fixture.argv)
    assert fixture.state["loads"] == 2


def test_source_change_invalidates_cache_and_failed_refresh_preserves_previous(fixture):
    result = module.main(fixture.argv)
    output_dir = Path(result["outputs"]["expert_prob"]["path"]).parent
    hashes = {path.name: module.sha256(path) for path in output_dir.iterdir()}
    fixture.state["fail"] = True
    with pytest.raises(RuntimeError, match="controlled"):
        module.main(fixture.argv + ["--cache-policy", "refresh"])
    assert {path.name: module.sha256(path) for path in output_dir.iterdir()} == hashes
    changed = fixture.expert.copy()
    changed[0, 0] = .91
    np.save(fixture.external, changed)
    with pytest.raises(RuntimeError, match="Compatible Stage-1"):
        module.main(fixture.argv + ["--cache-policy", "require"])
    assert fixture.state["loads"] == 2  # failed refresh only; require never loads model


@pytest.mark.parametrize("already_owned", [False, True])
def test_output_override_never_deletes_unrelated_files(fixture, already_owned):
    output = fixture.inputs / "stage1_references_v081"
    if already_owned:
        module.main(fixture.argv)
    else:
        output.mkdir()
    valuable = output / "user_checkpoint.pt"
    valuable.write_bytes(b"user checkpoint must be preserved")
    with pytest.raises(ValueError, match="non-owned/non-reference"):
        module.main(fixture.argv + ["--cache-policy", "refresh"])
    assert valuable.read_bytes() == b"user checkpoint must be preserved"
    assert fixture.state["loads"] == int(already_owned)


def test_checkpoint_mismatch_rejected_before_model_load(fixture):
    fixture.checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="Stage-1 checkpoint SHA256"):
        module.main(fixture.argv)
    assert fixture.state["loads"] == 0


def test_default_external_missing_does_not_guess_another_probability(fixture):
    fixture.external.rename(fixture.external.with_name("bp_predictions.float16.npy"))
    with pytest.raises(FileNotFoundError, match="no fallback"):
        module.main(fixture.argv)
    assert fixture.state["loads"] == 0


def test_universe_external_matrix_explicit_row_and_go_mapping(fixture):
    ordered = np.vstack([np.zeros(3), fixture.expert[1], fixture.expert[2], fixture.expert[0]])[:, ::-1]
    np.save(fixture.external, ordered)
    with pytest.raises(ValueError, match="universe-sized"):
        module.main(fixture.argv)
    row_ids, go_ids = fixture.project / "external_ids.txt", fixture.project / "external_go.txt"
    row_ids.write_text("extra\nb\nc\na\n")
    go_ids.write_text("GO:C\nGO:B\nGO:A\n")
    result = module.main(fixture.argv + ["--external-protein-ids", str(row_ids), "--external-go-ids", str(go_ids)])
    np.testing.assert_array_equal(np.load(result["outputs"]["expert_prob"]["path"]),
                                  fixture.expert[[2, 0, 1]].astype(np.float16).astype(np.float32))
    assert result["cache_signature"]["external_column_identity_independently_verified"] is True


def test_incompatible_backbone_is_not_replaced(fixture):
    path = fixture.inputs / "backbone_ind_test_prob.f16.npy"
    np.save(path, np.zeros((3, 3), dtype=np.float16))
    before = module.sha256(path)
    with pytest.raises(ValueError, match="Recomputed backbone differs"):
        module.main(fixture.argv)
    assert module.sha256(path) == before
    assert not (fixture.inputs / "stage1_references_v081").exists()


def test_backbone_only_checkpoint_rejected_by_production_loader():
    exporter = module.import_file("_reference_test_real_exporter", ROOT / "scripts/export_weak_graph_predictions.py")
    with pytest.raises(RuntimeError, match="no query_decoder state"):
        exporter.load_checkpoint_strict(TinyModel(), {"backbone": {}}, allow_partial=False)


def test_wrong_modelout_variant_is_never_silently_substituted(fixture):
    with pytest.raises(ValueError, match="Unsupported modelout key"):
        module.main(fixture.argv + ["--modelout-key", "modelout::mix_expert_base_anchor::decoderprob::mix_exp_base_a0.8"])


def test_numpy_byte_ids_decode_without_changing_protein_names(tmp_path):
    path = tmp_path / "ids.npy"
    np.save(path, np.asarray([b"a", b"b"]))
    assert module.ids_from_file(path) == ["a", "b"]


def test_production_loader_restores_checkpoint_args_and_uses_independent_index(fixture, monkeypatch):
    """Real exporter parser/config merging/loading with a tiny stand-in backbone."""
    tree = ast.parse((ROOT / "experiments/eval_weak_ind_test_detr_diagnostics.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"str2bool", "build_argparser"}]
    namespace = {"argparse": argparse, "Any": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual_stage1_cli", "exec"), namespace)
    diagnostic = SimpleNamespace(build_argparser=namespace["build_argparser"])

    class Model(nn.Module):
        def __init__(self, opt, args):
            super().__init__()
            assert opt.mode == "ind_test" and opt.shuffle is False
            assert args.allow_eval_label_boost is False
            self.backbone = nn.Linear(4, 3)
            self.classifier = self.backbone
            self.query_decoder = real_query_decoder()

    config = fixture.project / "model_config.pkl"
    config.write_bytes(b"model configuration identity")
    config_args = {"model_config": str(config), "file_address": "old_train.pkl", "working_address": "old_training_MSA/index.pkl",
                   "num_classes": 3, "top_k": 8, "max_len": 64, "query_decoder_topk": 2,
                   "query_decoder_logit_base_mode": "mix_expert_base_anchor", "expert_base_mix_alpha": .8}
    payload_model = Model(SimpleNamespace(mode="ind_test", shuffle=False), SimpleNamespace(allow_eval_label_boost=False))
    payload = {"backbone": payload_model.backbone.state_dict(), "query_decoder": payload_model.query_decoder.state_dict(),
               "model_args": config_args}
    torch.save(payload, fixture.checkpoint)
    # Adjacent args.json may be older; checkpoint architecture must win.
    (fixture.project / "args.json").write_text(json.dumps({**config_args, "expert_base_mix_alpha": .5, "query_decoder_topk": 99}))
    fake_weak = SimpleNamespace(__file__=__file__, build_weak_opt_from_config=lambda args: SimpleNamespace(), WeakMSAGOWithDETRDecoder=Model)
    fake_exp = SimpleNamespace(__file__=__file__, set_seed=torch.manual_seed)
    monkeypatch.setitem(sys.modules, "experiments.weak_exp_train_detr", fake_weak)
    monkeypatch.setitem(sys.modules, "experiments.exp_train", fake_exp)
    real_import = module.import_file
    monkeypatch.setattr(module, "import_file", lambda name, path: diagnostic if name == "_nbs_reference_stage1_diagnostic" else real_import(name, path))
    args = module.build_parser().parse_args(fixture.argv)
    args.project_root = fixture.project
    args.stage1_checkpoint = fixture.checkpoint
    args.diagnostic_script = fixture.project / "experiments/eval_weak_ind_test_detr_diagnostics.py"
    args.msa_index = fixture.project / "data/ind_MSA_bin/index.pkl"
    args.output_dir = fixture.inputs / "refs"
    context = {"go_path": fixture.project / "go_registry.tsv", "gos": ["a", "b", "c"],
               "contract": {"modelout_key": "modelout::mix_expert_base_anchor::decoderprob::expert"},
               "semantics": {"expert_base_mix_alpha": .8}, "args_path": fixture.project / "args.json",
               "input_manifest": {"cache_signature": {"stage1_model_config_sha256": module.sha256(config)}}}
    result = PRODUCTION_LOAD_RUNTIME(args, context)
    assert result["model_args"].working_address == str(args.msa_index)
    assert result["model_args"].expert_base_mix_alpha == .8
    assert result["model_args"].query_decoder_topk == 2
    assert result["checkpoint_load"]["missing_query_decoder"] == []
    assert result["configuration_conflicts"]["query_decoder_topk"] == {"args_json": 99, "checkpoint_used": 2}
    assert result["model_args"].mode == "ind_test"
    config.write_bytes(b"modified configuration")
    with pytest.raises(ValueError, match="Stage-1 model config SHA256"):
        PRODUCTION_LOAD_RUNTIME(args, context)
