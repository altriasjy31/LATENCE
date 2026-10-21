"""The real v088 loader accepts its named consumer without weakening provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from nbs_pg import local_loader, local_loader_v087, local_loader_v088


def shipped():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/bp_full_task_v0.8.8_hetero_tuned.json").read_text())


@pytest.fixture
def export(tmp_path):
    """Tiny real mmap export: two proteins and the shipped top-512 contract."""
    def array(name, value, dtype=np.int64):
        path = tmp_path / (name + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(value, dtype=dtype))
        return str(path)

    def manifest(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return str(path)

    config = shipped()
    (tmp_path / "features").mkdir()
    (tmp_path / "features/protein_registry.csv").write_text(
        "protein_idx,role,role_row_idx\n0,core,0\n1,weak,0\n")
    feature_roles = []
    roles = []
    for index, (role, mode) in enumerate((("core", "train"), ("weak", "exp_train"))):
        feature_roles.append({"role": role, "count": 1,
                              "feature_file": array(role + "_x", [[1., 2., 3., 4.]], np.float32)})
        roles.append({"role": role, "dataset_mode": mode,
                      "global_protein_idx_min": index, "global_protein_idx_max": index,
                      "backbone_dense_file": array(role + "_b", np.full((1, 512), .3), np.float32)})
    representation = manifest("features/representation_manifest.json",
                              {"feature_dim": 4, "roles": feature_roles})
    candidate_edge = array("candidate_edges", np.stack([np.repeat([0, 1], 512), np.tile(np.arange(512), 2)]))
    candidate_attr = array("candidate_attrs", np.tile([[.3, .5, 1.]], (1024, 1)), np.float32)
    weak = {
        "roles": roles, "go_registry": {"num_terms": 512},
        "rare_definition": {"training_annotation_key": "prop_annotations",
                            "train_counts_file": array("train_counts", np.ones(512))},
        "model_semantics": {"modelout": {
            "prediction_key": "modelout::mix_expert_base_anchor::decoderprob::expert",
            "decoder_prob_source": "expert", "external_probability_used": True},
            "protein_go_selector": {"scope": "full_task"}},
        "weak_pseudo_targets": {
            "role": "weak", "comparison": ">", "threshold": .5,
            "edge_attr_columns": ["modelout_probability"],
            "csr_indptr_file": array("pseudo_protein_indptr", [0, 512]),
            "csr_indices_file": array("pseudo_protein_go", np.arange(512)),
            "csr_probability_file": array("pseudo_protein_prob", np.full(512, .8), np.float32)},
        "backbone_candidate_edges": {"edge_index_file": candidate_edge,
                                     "edge_attr_file": candidate_attr},
    }
    weak_path = manifest("weak.json", weak)
    gold_manifest = {"task": "bp", "role": "core", "mode": "train",
                     "source": {"label_key": "prop_annotations"}}
    manifest("gold/gold_annotations_manifest.json", gold_manifest)
    gold_edge = array("gold/edge", np.stack([np.zeros(512), np.arange(512)]))
    inverted = {"source_manifest": weak_path, "indices": {
        "candidate": {"fixed_degree": 512, "source_protein_start": 0,
                      "indptr": array("candidate_indptr", np.arange(513) * 2),
                      "protein_idx": array("candidate_protein", np.tile([0, 1], 512))},
        "pseudo": {"indptr": array("pseudo_indptr", np.arange(513)),
                   "protein_idx": array("pseudo_protein", np.ones(512)),
                   "payloads": {"probability": array("pseudo_probability", np.full(512, .8), np.float32)}},
        "gold": {"source_edge_index": gold_edge,
                 "indptr": array("gold_indptr", np.arange(513)),
                 "protein_idx": array("gold_protein", np.zeros(512)),
                 "protein_major": {"num_go": 512,
                                   "indptr": array("gold_protein_indptr", [0, 512, 512]),
                                   "go_idx": array("gold_protein_go", np.arange(512))}},
    }}
    inverted_path = manifest("inverted.json", inverted)
    alignment = manifest("alignment.json", {"arrays": {"source_row": {
        "file": array("task_to_ontology", np.arange(512))}}})
    gg = {"num_classes": 513, "relations": {}}
    for name in ("is_a", "has_child", "part_of", "has_part"):
        pairs = [[0, 512]] if name == "is_a" else np.empty((0, 2))
        gg["relations"][name] = {"file": array(name, pairs)}
    gg_path = manifest("gg.json", gg)
    boxes = manifest("boxes.json", {"arrays": {
        name: {"file": array(name, np.ones((513, 4)), np.float32)}
        for name in ("center", "offset", "stats")}})
    config["data"].update({"root": str(tmp_path), "representation_manifest": representation,
        "go_protein_inverted_index_manifest": inverted_path,
        "weak_graph_predictions_manifest": weak_path,
        "pp_sampling_indices_manifest": manifest("sampling.json", {"relations": {}}),
        "full_go_box_manifest": boxes, "boxsqel_gg_relations_manifest": gg_path})
    config["go_boxsqel"]["alignment_manifest"] = alignment
    sources = dict(weak_manifest=weak, weak_manifest_path=Path(weak_path),
                   inverted=inverted, inverted_manifest=Path(inverted_path))
    return config, sources


def test_real_v088_builder_preserves_store_data_and_config(export):
    config, sources = export
    before = copy.deepcopy(config)
    stores = local_loader_v088.build_latence_nbs_stores(config)
    assert config == before
    assert stores.num_task_go == 512
    assert stores.registry.num_proteins == 2
    assert stores.full_boxes.num_go == 513
    np.testing.assert_array_equal(stores.gold_messages.gather([0]),
                                  np.stack([np.zeros(512), np.arange(512)]))
    np.testing.assert_allclose(stores.pseudo_messages.get_role_row(0)["probability"], .8)
    edge, attr = stores.candidate_messages.gather([0, 1])
    assert edge.shape == (2, 1024)
    np.testing.assert_allclose(attr[:, 0], .3)
    # The old APIs remain untouched; no temporary stage renaming or monkeypatch.
    with pytest.raises(ValueError, match="outside explicit v083/v084"):
        local_loader._validate_supervision_provenance(config, **sources)
    with pytest.raises(ValueError, match="outside explicit v083/v084/v087"):
        local_loader_v087._validate_supervision_provenance(config, **sources)


@pytest.mark.parametrize("path,value,match", [
    (("stage", "name"), "nbs_v087_direct", "stage=nbs_v088_hetero_tuned"),
    (("stage", "name"), "nbs_v088_other", "stage=nbs_v088_hetero_tuned"),
    (("release_version",), "0.8.7", "release_version"),
    (("full_task", "weak_target_mode"), "soft", "binary_membership"),
    (("supervision_contract", "core_gold", "role"), "weak", "core_gold.role"),
    (("supervision_contract", "core_gold", "dataset_mode"), "exp_train", "core_gold.dataset_mode"),
    (("supervision_contract", "core_gold", "source"), "modelout", "core_gold.source"),
    (("supervision_contract", "core_gold", "metadata_label_key"), "annotations", "metadata_label_key"),
    (("supervision_contract", "weak_pseudo", "source"), "metadata", "weak_pseudo.source"),
    (("supervision_contract", "weak_pseudo", "dataset_mode"), "train", "weak_pseudo.dataset_mode"),
    (("supervision_contract", "weak_pseudo", "use_probability_as_soft_target"), True, "use_probability"),
    (("supervision_contract", "weak_pseudo", "threshold"), .4, "threshold"),
    (("supervision_contract", "weak_pseudo", "comparison"), ">=", "comparison"),
    (("supervision_contract", "weak_pseudo", "negative_policy"), "metadata", "negative_policy"),
    (("supervision_contract", "weak_pseudo", "forbid_exp_train_prop_annotations"), False, "forbid_exp_train"),
])
def test_wrong_consumer_or_label_contract_rejected_before_data_open(path, value, match):
    config = shipped()  # Its production paths do not exist in this test environment.
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        local_loader_v088.build_latence_nbs_stores(config)


@pytest.mark.parametrize("which,path,value,match", [
    ("weak_manifest", ("roles", 0, "dataset_mode"), "ind_test", "core role"),
    ("weak_manifest", ("roles", 1, "dataset_mode"), "train", "weak role"),
    ("weak_manifest", ("rare_definition", "training_annotation_key"), "annotations", "annotation key"),
    ("weak_manifest", ("model_semantics", "modelout", "prediction_key"), "B", "configured modelout"),
    ("weak_manifest", ("model_semantics", "modelout", "decoder_prob_source"), "base", "not expert"),
    ("weak_manifest", ("model_semantics", "modelout", "external_probability_used"), False, "external-probability"),
    ("weak_manifest", ("weak_pseudo_targets", "threshold"), .4, "threshold"),
    ("weak_manifest", ("weak_pseudo_targets", "comparison"), ">=", "comparison"),
    ("weak_manifest", ("weak_pseudo_targets", "edge_attr_columns"), ["base_probability"], "payload"),
    ("weak_manifest", ("model_semantics", "protein_go_selector", "scope"), "rare_first", "scope"),
    ("inverted", ("indices", "candidate", "fixed_degree"), 16, "fixed degree"),
    ("inverted", ("source_manifest",), "different_weak.json", "different weak"),
    ("inverted", ("indices", "pseudo", "payloads"), {}, "probability payload"),
    ("inverted", ("indices", "gold", "source_edge_index"), None, "source_edge_index"),
])
def test_existing_source_provenance_checks_remain_live(export, which, path, value, match):
    config, sources = export
    changed = copy.deepcopy(sources)
    target = changed[which]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        local_loader_v088._validate_supervision_provenance(config, **changed)


@pytest.mark.parametrize("key,value,match", [("task", "mf", "task"),
    ("role", "weak", "role"), ("mode", "exp_train", "mode"),
    ("source", {"label_key": "annotations"}, "train.prop_annotations")])
def test_gold_manifest_provenance_is_still_checked(export, key, value, match):
    config, sources = export
    path = Path(sources["inverted"]["indices"]["gold"]["source_edge_index"]).parent / "gold_annotations_manifest.json"
    payload = json.loads(path.read_text())
    payload[key] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        local_loader_v088._validate_supervision_provenance(config, **sources)
