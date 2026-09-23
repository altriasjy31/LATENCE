"""Explicit v086 entry point for the unchanged audited v084 store format.

The old loader binds binary weak targets to named v083/v084 consumers. This
additive module extends only that version allowlist to v086 and preserves all
source, column, gold and pseudo provenance checks. It does not rewrite a config
or monkeypatch the old module, so concurrent v085 runs remain unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .local_loader import (
    LatenceNBSStores, ProteinRegistryStore, RoleAwareProteinFeatureStore,
    GlobalProteinGOCSRStore, RoleLocalProteinGOCSRStore, RoleProbabilitySlice,
    RoleAwareBaseLogitStore, FixedDegreeCandidateAttributeStore,
    FixedDegreeProteinGOStore, NBSQueryEpisodeConfig, GOQueryEpisodeSampler,
    EdgeOffsetCSRStore, FullGOBoxStore, DirectGORelationStore,
    build_ontology_space_summary, resolve_data_path, _load_json, _resolve_index_spec,
)


def _validate_supervision_provenance(
    config: Mapping[str, Any],
    *,
    weak_manifest: Mapping[str, Any],
    weak_manifest_path: Path,
    inverted: Mapping[str, Any],
    inverted_manifest: Path,
) -> None:
    """Enforce the LATENCE stage-2 supervision source contract at runtime.

    Core supervision must come from ``train.prop_annotations``.  Weak-set
    supervision must come exclusively from first-stage expert-assisted
    ``modelout`` annotations with probability > 0.5; ``exp_train`` metadata
    labels are never accepted as NBS weak supervision.
    """
    contract = config.get("supervision_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(
            "training config lacks supervision_contract; NBS refuses to infer "
            "core/weak label provenance implicitly"
        )
    core = contract.get("core_gold")
    weak = contract.get("weak_pseudo")
    if not isinstance(core, Mapping) or not isinstance(weak, Mapping):
        raise ValueError("supervision_contract must define core_gold and weak_pseudo")

    role_specs = {str(item.get("role")): item for item in weak_manifest.get("roles", [])}
    core_role = str(core.get("role", "core"))
    weak_role = str(weak.get("role", "weak"))
    if core_role not in role_specs or weak_role not in role_specs:
        raise ValueError("weak-graph manifest does not contain configured core/weak roles")
    if str(role_specs[core_role].get("dataset_mode")) != str(core.get("dataset_mode", "train")):
        raise ValueError("core role is not aligned with dataset_mode=train")
    if str(role_specs[weak_role].get("dataset_mode")) != str(weak.get("dataset_mode", "exp_train")):
        raise ValueError("weak role is not aligned with dataset_mode=exp_train")

    expected_gold_key = str(core.get("metadata_label_key", "prop_annotations"))
    rare = weak_manifest.get("rare_definition", {})
    if str(rare.get("training_annotation_key")) != expected_gold_key:
        raise ValueError(
            "first-stage train annotation key does not match NBS core gold contract: "
            f"manifest={rare.get('training_annotation_key')!r}, expected={expected_gold_key!r}"
        )

    semantics = weak_manifest.get("model_semantics", {}).get("modelout", {})
    expected_prediction_key = str(
        weak.get("prediction_key", "modelout::mix_expert_base_anchor::decoderprob::expert")
    )
    if str(semantics.get("prediction_key")) != expected_prediction_key:
        raise ValueError("weak pseudo supervision is not sourced from the configured modelout prediction")
    if str(semantics.get("decoder_prob_source")) != str(weak.get("decoder_prob_source", "expert")):
        raise ValueError("weak modelout decoder probability source is not expert")
    if bool(semantics.get("external_probability_used")) is not bool(
        weak.get("external_probability_used", True)
    ):
        raise ValueError("weak modelout external-probability provenance disagrees with contract")

    pseudo_targets = weak_manifest.get("weak_pseudo_targets", {})
    if str(pseudo_targets.get("role")) != weak_role:
        raise ValueError("weak pseudo target role is not 'weak'")
    if str(pseudo_targets.get("comparison")) != str(weak.get("comparison", ">")):
        raise ValueError("weak pseudo comparison operator disagrees with supervision contract")
    if float(pseudo_targets.get("threshold", float("nan"))) != float(weak.get("threshold", 0.5)):
        raise ValueError("weak pseudo threshold must be exactly 0.5")
    if list(pseudo_targets.get("edge_attr_columns", [])) != ["modelout_probability"]:
        raise ValueError("weak pseudo edge payload must be modelout_probability")
    if not bool(weak.get("use_probability_as_soft_target", True)):
        # v083/v084/v086 retain the same audited >0.5 CSR membership/provenance;
        # their dedicated full-task consumers convert selected targets to one.
        # Older loaders still require probability-valued weak targets.
        binary_full_task = (
            str(config.get("stage", {}).get("name", "")).startswith(("nbs_v083_", "nbs_v084_", "nbs_v086_"))
            and config.get("full_task", {}).get("weak_target_mode") == "binary_membership"
        )
        if not binary_full_task:
            raise ValueError("NBS weak supervision requires modelout probability as a soft target outside explicit v083/v084/v086 binary_membership mode")
    if str(weak.get("negative_policy", "none")) != "none":
        raise ValueError("NBS weak supervision contract requires negative_policy='none'")
    if not bool(weak.get("forbid_exp_train_prop_annotations", True)):
        raise ValueError("NBS must forbid exp_train.prop_annotations as weak supervision")

    stage = config.get("stage", {})
    expected_candidate_scope = stage.get("protein_go_candidate_scope")
    expected_candidate_topk = stage.get("protein_go_candidate_topk")
    selector_semantics = weak_manifest.get("model_semantics", {}).get(
        "protein_go_selector"
    )
    if selector_semantics is None:
        selector_semantics = weak_manifest.get("model_semantics", {}).get(
            "rare_selector", {}
        )
    observed_candidate_scope = str(
        selector_semantics.get("scope", "rare_first")
    )
    if observed_candidate_scope == "restricted":
        observed_candidate_scope = "rare_first"
    if (
        expected_candidate_scope is not None
        and observed_candidate_scope != str(expected_candidate_scope)
    ):
        raise ValueError(
            "Protein-GO candidate selector scope disagrees with the training "
            f"contract: manifest={observed_candidate_scope!r}, "
            f"expected={expected_candidate_scope!r}"
        )

    indices = inverted.get("indices", {})
    candidate_spec = indices.get("candidate")
    if not isinstance(candidate_spec, Mapping):
        raise ValueError("GO->Protein inverted index lacks candidate supervision")
    if (
        expected_candidate_topk is not None
        and int(candidate_spec.get("fixed_degree", -1))
        != int(expected_candidate_topk)
    ):
        raise ValueError(
            "Protein-GO candidate fixed degree disagrees with the training "
            f"contract: index={candidate_spec.get('fixed_degree')!r}, "
            f"expected={expected_candidate_topk!r}"
        )
    inverted_source = inverted.get("source_manifest")
    if isinstance(inverted_source, str):
        inverted_source_path = resolve_data_path(
            inverted_manifest.parent, inverted_source
        )
        if inverted_source_path.resolve() != weak_manifest_path.resolve():
            raise ValueError(
                "GO->Protein inverted index was built from a different weak "
                "graph manifest"
            )
    pseudo_spec = indices.get("pseudo")
    if not isinstance(pseudo_spec, Mapping):
        raise ValueError("GO->Protein inverted index lacks weak pseudo supervision")
    if "probability" not in dict(pseudo_spec.get("payloads", {})):
        raise ValueError("pseudo inverted index lacks modelout probability payload")

    gold_spec = indices.get("gold")
    if not isinstance(gold_spec, Mapping):
        raise ValueError("GO->Protein inverted index lacks core gold supervision")
    source_edge = gold_spec.get("source_edge_index")
    if not isinstance(source_edge, str):
        raise ValueError("gold index does not record source_edge_index provenance")
    gold_edge_path = resolve_data_path(inverted_manifest.parent, source_edge)
    gold_manifest_path = gold_edge_path.parent / "gold_annotations_manifest.json"
    if not gold_manifest_path.is_file():
        raise FileNotFoundError(
            f"gold provenance manifest is missing: {gold_manifest_path}; "
            "re-export gold edges with export_gold_protein_go_edges.py"
        )
    gold_manifest = _load_json(gold_manifest_path)
    if str(gold_manifest.get("task")) != str(config.get("task")):
        raise ValueError("gold annotation manifest task disagrees with training task")
    if str(gold_manifest.get("role")) != core_role:
        raise ValueError("gold annotation manifest role is not core")
    if str(gold_manifest.get("mode")) != str(core.get("dataset_mode", "train")):
        raise ValueError("gold annotation manifest mode is not train")
    if str(gold_manifest.get("source", {}).get("label_key")) != expected_gold_key:
        raise ValueError("gold annotation manifest is not sourced from train.prop_annotations")


def build_latence_nbs_stores(config: Mapping[str, Any]) -> LatenceNBSStores:
    if not str(config.get("stage", {}).get("name", "")).startswith("nbs_v086_"):
        raise ValueError("v086 store loader requires an explicit nbs_v086_ stage")
    if config.get("release_version") != "0.8.6":
        raise ValueError("v086 store loader requires release_version=0.8.6")
    if config.get("full_task", {}).get("weak_target_mode") != "binary_membership":
        raise ValueError("v086 requires weak_target_mode=binary_membership")
    if config.get("supervision_contract", {}).get("weak_pseudo", {}).get("use_probability_as_soft_target") is not False:
        raise ValueError("v086 requires binary membership, not weak probability targets")
    data = dict(config["data"])
    root = Path(data["root"]).resolve()
    registry = ProteinRegistryStore(root / "features/protein_registry.csv")
    representation_manifest = root / data.get("representation_manifest", "features/representation_manifest.json")
    features = RoleAwareProteinFeatureStore(representation_manifest, registry)

    inverted_manifest = root / data["go_protein_inverted_index_manifest"]
    inverted = _load_json(inverted_manifest)
    indices = inverted["indices"]
    if "gold" not in indices:
        raise ValueError(
            "GO->Protein inverted index lacks gold supervision. Rebuild it with "
            "--gold-edge-index before starting NBS training."
        )
    gold_spec = indices["gold"]
    gold = _resolve_index_spec(inverted_manifest, gold_spec)
    protein_major_gold = gold_spec.get("protein_major")
    if not isinstance(protein_major_gold, Mapping):
        raise ValueError(
            "gold inverted index lacks Protein->GO CSR required for weak->core->GO "
            "messages. Re-run run_build_go_protein_inverted_index.py with "
            "GOLD_EDGE_INDEX and MERGE_EXISTING=1."
        )
    gold_messages = GlobalProteinGOCSRStore(
        resolve_data_path(inverted_manifest.parent, protein_major_gold["indptr"]),
        resolve_data_path(inverted_manifest.parent, protein_major_gold["go_idx"]),
        num_go=int(protein_major_gold["num_go"]),
    )
    candidate = _resolve_index_spec(inverted_manifest, indices["candidate"])
    pseudo = _resolve_index_spec(inverted_manifest, indices["pseudo"]) if "pseudo" in indices else None

    weak_manifest_path = root / data["weak_graph_predictions_manifest"]
    weak_manifest = _load_json(weak_manifest_path)
    _validate_supervision_provenance(
        config,
        weak_manifest=weak_manifest,
        weak_manifest_path=weak_manifest_path,
        inverted=inverted,
        inverted_manifest=inverted_manifest,
    )
    num_task_go = int(weak_manifest["go_registry"]["num_terms"])
    train_counts = np.load(resolve_data_path(weak_manifest_path.parent, weak_manifest["rare_definition"]["train_counts_file"]), mmap_mode="r")
    pseudo_messages = None
    if "weak_pseudo_targets" in weak_manifest:
        spec = weak_manifest["weak_pseudo_targets"]
        pseudo_messages = RoleLocalProteinGOCSRStore(
            resolve_data_path(weak_manifest_path.parent, spec["csr_indptr_file"]),
            resolve_data_path(weak_manifest_path.parent, spec["csr_indices_file"]),
            role=str(spec["role"]),
            registry=registry,
            probability_path=resolve_data_path(
                weak_manifest_path.parent, spec["csr_probability_file"]
            ),
        )
    role_slices = []
    weak_role_start: Optional[int] = None
    weak_role_end: Optional[int] = None
    for role in weak_manifest["roles"]:
        role_name = str(role["role"])
        global_start = int(role["global_protein_idx_min"])
        global_end = int(role["global_protein_idx_max"]) + 1
        role_slices.append(RoleProbabilitySlice(
            role=role_name,
            global_start=global_start,
            global_end=global_end,
            probability_path=str(resolve_data_path(weak_manifest_path.parent, role["backbone_dense_file"])),
        ))
        if role_name == "weak":
            weak_role_start = global_start
            weak_role_end = global_end
    base_logits = RoleAwareBaseLogitStore(role_slices, num_go=num_task_go)

    alignment_path = Path(config["go_boxsqel"]["alignment_manifest"])
    if not alignment_path.is_absolute():
        alignment_path = Path.cwd() / alignment_path
    alignment = _load_json(alignment_path)
    task_to_ontology = np.load(alignment_path.parent / alignment["arrays"]["source_row"]["file"], mmap_mode="r")

    candidate_spec = indices["candidate"]
    candidate_source = weak_manifest.get("backbone_candidate_edges")
    if candidate_source is None:
        candidate_source = weak_manifest["backbone_rare_edges"]
    candidate_attributes = FixedDegreeCandidateAttributeStore(
        resolve_data_path(weak_manifest_path.parent, candidate_source["edge_attr_file"]),
        fixed_degree=int(candidate_spec["fixed_degree"]),
        source_protein_start=int(candidate_spec["source_protein_start"] or 0),
    )
    candidate_messages = FixedDegreeProteinGOStore(
        resolve_data_path(
            weak_manifest_path.parent, candidate_source["edge_index_file"]
        ),
        resolve_data_path(
            weak_manifest_path.parent, candidate_source["edge_attr_file"]
        ),
        fixed_degree=int(candidate_spec["fixed_degree"]),
        source_protein_start=int(candidate_spec["source_protein_start"] or 0),
    )

    # Build task-level direct is_a pairs from the full BoxSquaredEL ontology.
    # This lets the query sampler deliberately co-sample child/parent rows so
    # the query-axis hierarchy objective is not almost always empty.
    gg_manifest_path = Path(data["boxsqel_gg_relations_manifest"])
    if not gg_manifest_path.is_absolute():
        gg_manifest_path = Path.cwd() / gg_manifest_path
    gg = _load_json(gg_manifest_path)
    full_is_a_path = gg_manifest_path.parent / gg["relations"]["is_a"]["file"]
    full_is_a = np.load(full_is_a_path, mmap_mode="r")
    ontology_to_task: dict[int, int] = {}
    for task_go_idx, ontology_go_idx in enumerate(np.asarray(task_to_ontology, dtype=np.int64).tolist()):
        ontology_to_task.setdefault(int(ontology_go_idx), int(task_go_idx))
    task_hierarchy_pairs: list[tuple[int, int]] = []
    for child_ontology, parent_ontology in np.asarray(full_is_a[:, :2], dtype=np.int64).tolist():
        child_task = ontology_to_task.get(int(child_ontology))
        parent_task = ontology_to_task.get(int(parent_ontology))
        if child_task is not None and parent_task is not None and child_task != parent_task:
            task_hierarchy_pairs.append((child_task, parent_task))
    hierarchy_pairs = (
        np.asarray(task_hierarchy_pairs, dtype=np.int64).T
        if task_hierarchy_pairs
        else np.empty((2, 0), dtype=np.int64)
    )

    episode_cfg = NBSQueryEpisodeConfig(**dict(config.get("episode", {})))
    pseudo_active_role_rows = np.empty(0, dtype=np.int64)
    if pseudo_messages is not None and episode_cfg.weak_focus_queries_per_episode > 0:
        gold_degree = np.diff(np.asarray(gold.indptr, dtype=np.int64))
        pseudo_degree = (
            np.diff(np.asarray(pseudo.indptr, dtype=np.int64))
            if pseudo is not None
            else np.zeros(num_task_go, dtype=np.int64)
        )
        if episode_cfg.gold_support_policy == "fixed":
            eligible_mask = gold_degree >= (
                int(episode_cfg.support_per_query)
                + int(episode_cfg.gold_positive_per_query)
            )
        else:
            eligible_mask = gold_degree >= 1
            if episode_cfg.singleton_requires_pseudo:
                singleton = gold_degree == 1
                eligible_mask[singleton] &= pseudo_degree[singleton] > 0
        pseudo_active_role_rows = pseudo_messages.active_role_rows_for_go_mask(
            eligible_mask
        )

    episode_sampler = GOQueryEpisodeSampler(
        gold=gold,
        candidate=candidate,
        pseudo=pseudo,
        pseudo_by_protein=pseudo_messages,
        pseudo_active_role_rows=pseudo_active_role_rows,
        hierarchy_pairs=hierarchy_pairs,
        base_logits=base_logits,
        train_go_counts=np.asarray(train_counts),
        task_to_ontology_go=np.asarray(task_to_ontology),
        candidate_attributes=candidate_attributes,
        candidate_by_protein=candidate_messages,
        config=episode_cfg,
        seed=int(config.get("training", {}).get("seed", 3407)),
        weak_global_start=weak_role_start,
        weak_global_end=weak_role_end,
    )

    sampling_manifest_path = root / data["pp_sampling_indices_manifest"]
    sampling = _load_json(sampling_manifest_path)
    pp: dict[str, EdgeOffsetCSRStore] = {}
    for name, spec in sampling["relations"].items():
        pp[name] = EdgeOffsetCSRStore(
            indptr_path=resolve_data_path(sampling_manifest_path.parent, spec["indptr"]),
            edge_offset_path=resolve_data_path(sampling_manifest_path.parent, spec["edge_offset"]),
            edge_index_path=resolve_data_path(sampling_manifest_path.parent, spec["edge_index"]),
            edge_attr_path=resolve_data_path(sampling_manifest_path.parent, spec["edge_attr"]),
            key_axis=int(spec["key_axis"]),
        )

    full_manifest_path = Path(data["full_go_box_manifest"])
    if not full_manifest_path.is_absolute():
        full_manifest_path = Path.cwd() / full_manifest_path
    full_boxes = FullGOBoxStore.from_manifest(full_manifest_path)
    if task_to_ontology.shape != (num_task_go,):
        raise ValueError("task-to-ontology mapping does not align with classifier GO columns")
    if task_to_ontology.size and (
        int(np.min(task_to_ontology)) < 0
        or int(np.max(task_to_ontology)) >= full_boxes.num_go
    ):
        raise IndexError("task-to-ontology mapping leaves the full BoxSquaredEL class space")
    if gold_messages.num_proteins != registry.num_proteins:
        raise ValueError("gold Protein->GO CSR does not align with protein registry")
    if int(gg.get("num_classes", -1)) != full_boxes.num_go:
        raise ValueError("BoxSquaredEL G-G node space differs from full GO box rows")
    go_relations = {
        name: DirectGORelationStore(
            gg_manifest_path.parent / gg["relations"][name]["file"],
            num_go=full_boxes.num_go,
        )
        for name in ("is_a", "has_child", "part_of", "has_part")
    }

    # Three distinct GO spaces coexist in NBS and must never be conflated:
    # (1) immutable task classifier columns, (2) directly supervised task
    # queries, and (3) the complete BoxSquaredEL ontology used as semantic
    # context.  Build immutable masks once so local-batch and epoch diagnostics
    # can prove that non-task/context-only ontology nodes are actually exposed.
    task_to_ontology_arr = np.asarray(task_to_ontology, dtype=np.int64)
    (
        task_ontology_mask,
        eligible_ontology_mask,
        context_only_task_ontology_mask,
        ontology_summary,
    ) = build_ontology_space_summary(
        task_to_ontology_arr,
        np.asarray(episode_sampler.eligible_go, dtype=np.int64),
        full_ontology_rows=int(full_boxes.num_go),
        relation_edge_counts={
            name: int(store.edge.shape[0]) for name, store in go_relations.items()
        },
    )
    relation_partition_edges: dict[str, dict[str, int]] = {}
    for name, store in go_relations.items():
        edge = np.asarray(store.edge, dtype=np.int64)
        if edge.size == 0:
            relation_partition_edges[name] = {
                "task_to_task": 0,
                "task_to_non_task": 0,
                "non_task_to_task": 0,
                "non_task_to_non_task": 0,
            }
            continue
        src_task = task_ontology_mask[edge[:, 0]]
        dst_task = task_ontology_mask[edge[:, 1]]
        relation_partition_edges[name] = {
            "task_to_task": int(np.count_nonzero(src_task & dst_task)),
            "task_to_non_task": int(np.count_nonzero(src_task & ~dst_task)),
            "non_task_to_task": int(np.count_nonzero(~src_task & dst_task)),
            "non_task_to_non_task": int(np.count_nonzero(~src_task & ~dst_task)),
        }
    ontology_summary["relation_partition_edges"] = relation_partition_edges
    # Count only canonical forward relations here; inverse stores duplicate the
    # same structural evidence in the opposite message direction.
    ontology_summary["canonical_task_non_task_cross_edges"] = int(sum(
        relation_partition_edges.get(name, {}).get("task_to_non_task", 0)
        + relation_partition_edges.get(name, {}).get("non_task_to_task", 0)
        for name in ("is_a", "part_of")
    ))
    return LatenceNBSStores(
        registry=registry,
        features=features,
        episode_sampler=episode_sampler,
        candidate_messages=candidate_messages,
        gold_messages=gold_messages,
        pseudo_messages=pseudo_messages,
        pp=pp,
        full_boxes=full_boxes,
        go_relations=go_relations,
        task_to_ontology=task_to_ontology_arr,
        task_ontology_mask=task_ontology_mask,
        eligible_ontology_mask=eligible_ontology_mask,
        context_only_task_ontology_mask=context_only_task_ontology_mask,
        ontology_summary=ontology_summary,
        feature_dim=features.feature_dim,
        num_task_go=num_task_go,
    )
