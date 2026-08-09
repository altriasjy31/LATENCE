from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import numpy as np
import torch

try:
    from .data import PYG_AVAILABLE, build_nbs_protein_go_heterodata, mask_candidate_evidence_edges
except ModuleNotFoundError as exc:  # lightweight index/DDP tests without PyG
    if exc.name != "torch_geometric":
        raise
    PYG_AVAILABLE = False
    build_nbs_protein_go_heterodata = None  # type: ignore[assignment]
    mask_candidate_evidence_edges = None  # type: ignore[assignment]
from .episode import GOQueryEpisodeSampler, NBSGlobalEpisode, NBSQueryEpisodeConfig
from .latence_graph_stores import (
    DirectGORelationStore,
    EdgeOffsetCSRStore,
    FixedDegreeProteinGOStore,
    FullGOBoxStore,
    GlobalProteinGOCSRStore,
    ProteinRegistryStore,
    RoleAwareProteinFeatureStore,
    RoleLocalProteinGOCSRStore,
    resolve_data_path,
)
from .latence_stores import (
    FixedDegreeCandidateAttributeStore,
    GOProteinCSRStore,
    RoleAwareBaseLogitStore,
    RoleProbabilitySlice,
)
from .training import NBSLocalBatch


@dataclass
class NBSLocalGraphSamplingConfig:
    # Explicit synchronized steps per epoch.  ``None`` lets the loader derive
    # a world-size-invariant epoch length from GO coverage.
    steps_per_epoch_per_rank: Optional[int] = 1000
    coverage_cycles_per_epoch: float = 1.0
    # Protein-major epoch targets.  In ``protein_major_with_go_floor`` mode
    # the loader repeats GO cycles until weak pseudo targets and core gold
    # support/targets each reach the requested equivalent-pass budget.
    weak_pseudo_equivalent_passes_per_epoch: float = 0.0
    core_gold_equivalent_passes_per_epoch: float = 0.0
    # Hybrid epoch planning targets unique weak proteins directly.  The
    # efficiency factor is conservative because some weak-focus GO pools may
    # not provide a full target block after filtering and DDP partitioning.
    weak_unique_coverage_target_per_epoch: float = 0.0
    weak_focus_planning_efficiency: float = 0.85
    # NBS epochs are defined by GO-query coverage, not by the number of
    # first-stage protein mini-batches.  The optional stage-1 reference is
    # diagnostic only and never changes the resolved loader length.
    epoch_unit: str = "eligible_go_coverage_cycle"
    stage1_reference_global_steps_per_epoch: Optional[int] = None
    pp_hops: int = 2
    ppi_fanouts: tuple[int, ...] = (8, 4)
    similar_to_fanouts: tuple[int, ...] = (8, 4)
    weak_to_core_fanouts: tuple[int, ...] = (16, 0)
    candidate_message_topk: int = 32
    pseudo_message_topk: int = 32
    gold_message_topk: Optional[int] = 64
    go_hops: int = 1
    go_is_a_fanout: int = 16
    go_has_child_fanout: int = 16
    go_part_of_fanout: int = 8
    go_has_part_fanout: int = 8
    include_pseudo_messages: bool = False
    base_seed: int = 3407

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "NBSLocalGraphSamplingConfig":
        raw = dict(value or {})
        for name in ("ppi_fanouts", "similar_to_fanouts", "weak_to_core_fanouts"):
            if name in raw:
                raw[name] = tuple(int(x) for x in raw[name])
        if "gold_message_topk" in raw and raw["gold_message_topk"] is not None:
            raw["gold_message_topk"] = int(raw["gold_message_topk"])
        if "steps_per_epoch_per_rank" in raw and raw["steps_per_epoch_per_rank"] is not None:
            raw["steps_per_epoch_per_rank"] = int(raw["steps_per_epoch_per_rank"])
        if "coverage_cycles_per_epoch" in raw:
            raw["coverage_cycles_per_epoch"] = float(raw["coverage_cycles_per_epoch"])
        for name in (
            "weak_pseudo_equivalent_passes_per_epoch",
            "core_gold_equivalent_passes_per_epoch",
            "weak_unique_coverage_target_per_epoch",
            "weak_focus_planning_efficiency",
        ):
            if name in raw:
                raw[name] = float(raw[name])
        if (
            "stage1_reference_global_steps_per_epoch" in raw
            and raw["stage1_reference_global_steps_per_epoch"] is not None
        ):
            raw["stage1_reference_global_steps_per_epoch"] = int(
                raw["stage1_reference_global_steps_per_epoch"]
            )
        result = cls(**raw)
        result.validate()
        return result

    def validate(self) -> None:
        if self.steps_per_epoch_per_rank is not None and self.steps_per_epoch_per_rank <= 0:
            raise ValueError("steps_per_epoch_per_rank must be positive or None")
        if self.coverage_cycles_per_epoch <= 0:
            raise ValueError("coverage_cycles_per_epoch must be positive")
        if self.weak_pseudo_equivalent_passes_per_epoch < 0:
            raise ValueError("weak_pseudo_equivalent_passes_per_epoch cannot be negative")
        if self.core_gold_equivalent_passes_per_epoch < 0:
            raise ValueError("core_gold_equivalent_passes_per_epoch cannot be negative")
        if not 0.0 <= self.weak_unique_coverage_target_per_epoch <= 1.0:
            raise ValueError("weak_unique_coverage_target_per_epoch must lie in [0,1]")
        if not 0.0 < self.weak_focus_planning_efficiency <= 1.0:
            raise ValueError("weak_focus_planning_efficiency must lie in (0,1]")
        if self.epoch_unit not in {
            "eligible_go_coverage_cycle",
            "protein_major_with_go_floor",
            "hybrid_go_weak_coverage",
        }:
            raise ValueError(
                "epoch_unit must be eligible_go_coverage_cycle, "
                "protein_major_with_go_floor or hybrid_go_weak_coverage"
            )
        if self.epoch_unit == "protein_major_with_go_floor":
            if self.weak_pseudo_equivalent_passes_per_epoch <= 0:
                raise ValueError("protein-major epochs require a positive weak pseudo pass target")
            if self.core_gold_equivalent_passes_per_epoch <= 0:
                raise ValueError("protein-major epochs require a positive core gold pass target")
        if self.epoch_unit == "hybrid_go_weak_coverage":
            if self.weak_unique_coverage_target_per_epoch <= 0:
                raise ValueError("hybrid epochs require a positive unique weak coverage target")
            if self.core_gold_equivalent_passes_per_epoch <= 0:
                raise ValueError("hybrid epochs require a positive core gold pass target")
        if (
            self.stage1_reference_global_steps_per_epoch is not None
            and self.stage1_reference_global_steps_per_epoch <= 0
        ):
            raise ValueError(
                "stage1_reference_global_steps_per_epoch must be positive or None"
            )
        if self.pp_hops <= 0 or self.go_hops < 0:
            raise ValueError("invalid P-P/GO hop count")
        if self.gold_message_topk is not None and self.gold_message_topk <= 0:
            raise ValueError("gold_message_topk must be positive or None")
        if self.candidate_message_topk <= 0:
            raise ValueError("candidate_message_topk must be positive")
        if self.pseudo_message_topk <= 0:
            raise ValueError("pseudo_message_topk must be positive")
        for name in ("ppi_fanouts", "similar_to_fanouts", "weak_to_core_fanouts"):
            values = getattr(self, name)
            if len(values) < self.pp_hops:
                raise ValueError(f"{name} must provide at least pp_hops entries")
            if any(value < 0 for value in values):
                raise ValueError(f"{name} cannot contain negative fanouts")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_index_spec(manifest_path: Path, spec: Mapping[str, Any]) -> GOProteinCSRStore:
    payload_paths = {
        name: resolve_data_path(manifest_path.parent, value)
        for name, value in dict(spec.get("payloads", {})).items()
    }
    return GOProteinCSRStore(
        resolve_data_path(manifest_path.parent, spec["indptr"]),
        resolve_data_path(manifest_path.parent, spec["protein_idx"]),
        payload_paths=payload_paths,
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
        raise ValueError("NBS weak supervision requires modelout probability as a soft target")
    if str(weak.get("negative_policy", "none")) != "none":
        raise ValueError("NBS weak supervision contract requires negative_policy='none'")
    if not bool(weak.get("forbid_exp_train_prop_annotations", True)):
        raise ValueError("NBS must forbid exp_train.prop_annotations as weak supervision")

    indices = inverted.get("indices", {})
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

def _dedupe_edges(edge: np.ndarray, attr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if edge.shape[1] == 0:
        return edge.astype(np.int64, copy=False), attr.astype(np.float32, copy=False)
    key = edge[0].astype(np.int64) * (int(edge[1].max(initial=0)) + 1) + edge[1].astype(np.int64)
    order = np.lexsort((-attr[:, 0], key))
    sorted_key = key[order]
    keep = np.ones(order.size, dtype=bool)
    keep[1:] = sorted_key[1:] != sorted_key[:-1]
    chosen = order[keep]
    return edge[:, chosen].astype(np.int64, copy=False), attr[chosen].astype(np.float32, copy=False)


def _map_edge(edge: np.ndarray, source_nodes: np.ndarray, destination_nodes: Optional[np.ndarray] = None) -> np.ndarray:
    destination_nodes = source_nodes if destination_nodes is None else destination_nodes
    if edge.shape[1] == 0:
        return np.empty((2, 0), dtype=np.int64)
    source = np.searchsorted(source_nodes, edge[0])
    destination = np.searchsorted(destination_nodes, edge[1])
    if np.any(source_nodes[source] != edge[0]) or np.any(destination_nodes[destination] != edge[1]):
        raise RuntimeError("global/local edge mapping failed")
    return np.stack([source, destination], axis=0).astype(np.int64)


@dataclass
class LatenceNBSStores:
    registry: ProteinRegistryStore
    features: RoleAwareProteinFeatureStore
    episode_sampler: GOQueryEpisodeSampler
    candidate_messages: FixedDegreeProteinGOStore
    gold_messages: GlobalProteinGOCSRStore
    pseudo_messages: Optional[RoleLocalProteinGOCSRStore]
    pp: dict[str, EdgeOffsetCSRStore]
    full_boxes: FullGOBoxStore
    go_relations: dict[str, DirectGORelationStore]
    task_to_ontology: np.ndarray
    feature_dim: int
    num_task_go: int


class LatenceNBSLocalGraphMaterializer:
    def __init__(self, stores: LatenceNBSStores, config: NBSLocalGraphSamplingConfig) -> None:
        self.stores = stores
        self.config = config

    def _sample_pp(
        self,
        roots: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
        known = np.unique(roots.astype(np.int64))
        frontier = known.copy()
        relation_edges: dict[str, list[np.ndarray]] = {name: [] for name in self.stores.pp}
        relation_attrs: dict[str, list[np.ndarray]] = {name: [] for name in self.stores.pp}
        fanout_map = {
            "ppi": self.config.ppi_fanouts,
            "similar_to": self.config.similar_to_fanouts,
            "weak_to_core": self.config.weak_to_core_fanouts,
        }
        for hop in range(self.config.pp_hops):
            next_nodes: list[np.ndarray] = []
            for name, store in self.stores.pp.items():
                fanout = fanout_map[name][hop]
                edge, attr = store.sample(frontier, fanout, rng=rng)
                if edge.shape[1]:
                    relation_edges[name].append(edge)
                    relation_attrs[name].append(attr)
                    next_nodes.extend([edge[0], edge[1]])
            if not next_nodes:
                break
            combined = np.unique(np.concatenate(next_nodes))
            frontier = np.setdiff1d(combined, known, assume_unique=False)
            known = np.unique(np.concatenate([known, combined]))
            if frontier.size == 0:
                break
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in self.stores.pp:
            if relation_edges[name]:
                result[name] = (
                    np.concatenate(relation_edges[name], axis=1),
                    np.concatenate(relation_attrs[name], axis=0),
                )
            else:
                result[name] = (np.empty((2, 0), np.int64), np.empty((0, 3), np.float32))
        return known, result

    def _sample_go(
        self,
        initial: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        known = np.unique(initial.astype(np.int64))
        frontier = known.copy()
        is_a_edges: list[np.ndarray] = []
        part_edges: list[np.ndarray] = []
        fanouts = {
            "is_a": self.config.go_is_a_fanout,
            "has_child": self.config.go_has_child_fanout,
            "part_of": self.config.go_part_of_fanout,
            "has_part": self.config.go_has_part_fanout,
        }
        for _hop in range(self.config.go_hops):
            next_nodes: list[np.ndarray] = []
            for name, store in self.stores.go_relations.items():
                rows = store.sample(frontier, fanouts[name], rng=rng)
                if not rows.size:
                    continue
                next_nodes.extend([rows[:, 0], rows[:, 1]])
                if name == "is_a":
                    is_a_edges.append(rows)
                elif name == "has_child":
                    is_a_edges.append(rows[:, [1, 0]])
                elif name == "part_of":
                    part_edges.append(rows)
                elif name == "has_part":
                    part_edges.append(rows[:, [1, 0]])
            if not next_nodes:
                break
            combined = np.unique(np.concatenate(next_nodes))
            frontier = np.setdiff1d(combined, known, assume_unique=False)
            known = np.unique(np.concatenate([known, combined]))
            if frontier.size == 0:
                break
        is_a = np.unique(np.concatenate(is_a_edges, axis=0), axis=0) if is_a_edges else np.empty((0, 2), np.int64)
        part = np.unique(np.concatenate(part_edges, axis=0), axis=0) if part_edges else np.empty((0, 2), np.int64)
        return known, is_a, part

    def materialize(self, episode: NBSGlobalEpisode, *, seed: int) -> NBSLocalBatch:
        if not PYG_AVAILABLE or build_nbs_protein_go_heterodata is None or mask_candidate_evidence_edges is None:
            raise ModuleNotFoundError("torch_geometric is required for local graph materialization")
        rng = np.random.default_rng(int(seed))
        roots = np.unique(np.concatenate([episode.seed_protein_idx, episode.candidate_protein_idx]))
        protein_nodes, pp = self._sample_pp(roots, rng)

        # Message evidence remains source-major on disk and is only sliced for
        # local proteins.  This is the critical boundary preventing the 281M
        # candidate relation from becoming a monolithic PyG graph.
        candidate_task_edge, candidate_attr = self.stores.candidate_messages.gather(
            protein_nodes, topk=self.config.candidate_message_topk
        )
        pseudo_task_edge = np.empty((2, 0), np.int64)
        pseudo_prob = np.empty(0, np.float32)
        if self.config.include_pseudo_messages and self.stores.pseudo_messages is not None:
            pseudo_task_edge, pseudo_prob = self.stores.pseudo_messages.gather(
                protein_nodes, topk=self.config.pseudo_message_topk
            )

        # Materialize true annotations for every locally sampled non-candidate
        # protein.  This preserves the intended weak->core->GO route without
        # loading the complete gold graph.  Explicit support edges are unioned
        # afterwards so a configurable top-k cap can never remove the query
        # support relation itself.
        gold_message_proteins = np.setdiff1d(
            protein_nodes, episode.candidate_protein_idx, assume_unique=False
        )
        gold_task_edge = self.stores.gold_messages.gather(
            gold_message_proteins, topk=self.config.gold_message_topk
        )
        support_task_go = episode.query_go_idx[episode.seed_query_idx]
        support_edge = np.stack([episode.seed_protein_idx, support_task_go], axis=0)
        gold_task_edge = np.unique(
            np.concatenate([gold_task_edge, support_edge], axis=1), axis=1
        )

        def to_ontology(edge: np.ndarray) -> np.ndarray:
            if edge.shape[1] == 0:
                return edge.copy()
            mapped = edge.copy()
            mapped[1] = self.stores.task_to_ontology[edge[1]]
            return mapped

        gold_edge = to_ontology(gold_task_edge)
        candidate_edge = to_ontology(candidate_task_edge)
        pseudo_edge = to_ontology(pseudo_task_edge)
        go_initial = np.unique(
            np.concatenate(
                [
                    episode.query_ontology_go_idx,
                    gold_edge[1],
                    candidate_edge[1],
                    pseudo_edge[1],
                ]
            )
        )
        go_nodes, is_a_global_rows, part_global_rows = self._sample_go(go_initial, rng)

        protein_nodes = np.unique(
            np.concatenate(
                [
                    protein_nodes,
                    gold_edge[0],
                    candidate_edge[0],
                    pseudo_edge[0],
                ]
            )
        )
        protein_x = self.stores.features.gather(protein_nodes)
        boxes = self.stores.full_boxes.gather(go_nodes)

        ppi_edge, ppi_attr = pp["ppi"]
        similar_edge, similar_attr = pp["similar_to"]
        weak_edge, weak_attr = pp["weak_to_core"]
        ppi_local = _map_edge(ppi_edge, protein_nodes)
        similar_local = _map_edge(similar_edge, protein_nodes)
        weak_local = _map_edge(weak_edge, protein_nodes)
        gold_local = _map_edge(gold_edge, protein_nodes, go_nodes)
        candidate_edge, candidate_attr = _dedupe_edges(candidate_edge, candidate_attr)
        candidate_local = _map_edge(candidate_edge, protein_nodes, go_nodes)
        pseudo_attr = np.stack(
            [pseudo_prob, np.zeros_like(pseudo_prob), np.ones_like(pseudo_prob)], axis=1
        ).astype(np.float32) if pseudo_prob.size else np.empty((0, 3), np.float32)
        pseudo_edge, pseudo_attr = _dedupe_edges(pseudo_edge, pseudo_attr)
        pseudo_local = _map_edge(pseudo_edge, protein_nodes, go_nodes)
        is_a_local = _map_edge(is_a_global_rows.T, go_nodes) if is_a_global_rows.size else np.empty((2, 0), np.int64)
        part_local = _map_edge(part_global_rows.T, go_nodes) if part_global_rows.size else np.empty((2, 0), np.int64)
        topology_is_a = np.ones((is_a_local.shape[1], 2), dtype=np.float32)
        topology_part = np.ones((part_local.shape[1], 2), dtype=np.float32)

        graph = build_nbs_protein_go_heterodata(
            torch.as_tensor(protein_x, dtype=torch.float32),
            torch.as_tensor(boxes["center"], dtype=torch.float32),
            torch.as_tensor(boxes["offset"], dtype=torch.float32),
            torch.as_tensor(is_a_local, dtype=torch.long),
            go_part_of_edge_index=torch.as_tensor(part_local, dtype=torch.long),
            ppi_edge_index=torch.as_tensor(ppi_local, dtype=torch.long),
            similarity_edge_index=torch.as_tensor(similar_local, dtype=torch.long),
            weak_to_core_edge_index=torch.as_tensor(weak_local, dtype=torch.long),
            gold_protein_go_edge_index=torch.as_tensor(gold_local, dtype=torch.long),
            backbone_candidate_protein_go_edge_index=torch.as_tensor(candidate_local, dtype=torch.long),
            pseudo_protein_go_edge_index=torch.as_tensor(pseudo_local, dtype=torch.long),
            ppi_edge_attr=torch.as_tensor(ppi_attr, dtype=torch.float32),
            similarity_edge_attr=torch.as_tensor(similar_attr, dtype=torch.float32),
            weak_to_core_edge_attr=torch.as_tensor(weak_attr, dtype=torch.float32),
            backbone_candidate_edge_attr=torch.as_tensor(candidate_attr, dtype=torch.float32),
            pseudo_annotation_edge_attr=torch.as_tensor(pseudo_attr, dtype=torch.float32),
            go_is_a_topology=torch.as_tensor(topology_is_a, dtype=torch.float32),
            go_part_of_topology=torch.as_tensor(topology_part, dtype=torch.float32),
            go_stats=torch.as_tensor(boxes["stats"], dtype=torch.float32),
            protein_ids=torch.as_tensor(protein_nodes, dtype=torch.long),
            go_ids=torch.as_tensor(go_nodes, dtype=torch.long),
        )

        seed_local = torch.as_tensor(np.searchsorted(protein_nodes, episode.seed_protein_idx), dtype=torch.long)
        candidate_local_idx = torch.as_tensor(np.searchsorted(protein_nodes, episode.candidate_protein_idx), dtype=torch.long)
        query_go_local = torch.as_tensor(np.searchsorted(go_nodes, episode.query_ontology_go_idx), dtype=torch.long)
        graph = mask_candidate_evidence_edges(
            graph,
            candidate_local_idx,
            query_go_index=query_go_local,
            gold_mode="all",
            pseudo_mode="all",
            candidate_mode="query_only",
            inplace=True,
        )
        query = episode.to_query_batch(
            seed_protein_local=seed_local,
            candidate_protein_local=candidate_local_idx,
            query_go_local=query_go_local,
        )

        query_position = {int(value): row for row, value in enumerate(episode.query_ontology_go_idx.tolist())}
        hierarchy: list[tuple[int, int]] = []
        for child, parent in is_a_global_rows.tolist():
            if child in query_position and parent in query_position:
                hierarchy.append((query_position[child], query_position[parent]))
        hierarchy_edges = None
        if hierarchy:
            hierarchy_edges = torch.as_tensor(np.asarray(hierarchy, np.int64).T, dtype=torch.long)

        root_ids = np.asarray(episode.metadata.get("root_protein_idx", np.empty(0, np.int64)), dtype=np.int64)
        role_to_code = getattr(self.stores.registry, "role_to_code", {})
        role_code = getattr(self.stores.registry, "role_code", None)
        core_code = role_to_code.get("core") if isinstance(role_to_code, Mapping) else None
        weak_code = role_to_code.get("weak") if isinstance(role_to_code, Mapping) else None
        def role_subset(values: np.ndarray, code: Optional[int]) -> np.ndarray:
            if code is None or role_code is None or values.size == 0:
                return np.empty(0, dtype=np.int64)
            return values[np.asarray(role_code)[values] == int(code)].astype(np.int64, copy=False)

        return NBSLocalBatch(
            graph=graph,
            query=query,
            hierarchy_edges=hierarchy_edges,
            metadata={
                **dict(episode.metadata),
                "global_seed": int(seed),
                "protein_nodes": int(protein_nodes.size),
                # Transient global node IDs support exact epoch-level node
                # coverage diagnostics.  They remain on CPU and are not saved
                # in checkpoints or graph artifacts.
                "local_protein_idx": protein_nodes.astype(np.int64),
                "root_core_protein_idx": role_subset(root_ids, core_code),
                "root_weak_protein_idx": role_subset(root_ids, weak_code),
                "local_core_protein_idx": role_subset(protein_nodes, core_code),
                "local_weak_protein_idx": role_subset(protein_nodes, weak_code),
                "go_nodes": int(go_nodes.size),
                "candidate_edges_materialized": int(candidate_local.shape[1]),
                "candidate_edges_global_total": int(
                    getattr(self.stores.candidate_messages, "num_edges", -1)
                ),
                "hierarchy_query_edges": 0 if hierarchy_edges is None else int(hierarchy_edges.shape[1]),
                "direction_safe": True,
            },
        )

    def build_global_go_graph(self):
        if not PYG_AVAILABLE or build_nbs_protein_go_heterodata is None:
            raise ModuleNotFoundError("torch_geometric is required for the full GO cache graph")
        go_nodes = np.arange(self.stores.full_boxes.num_go, dtype=np.int64)
        boxes = self.stores.full_boxes.gather(go_nodes)
        # Direct relation arrays are small enough to materialize once per rank;
        # protein and candidate relations remain excluded.
        def all_edges(name: str) -> np.ndarray:
            store = self.stores.go_relations[name]
            return store.edge
        is_a = all_edges("is_a")
        part = all_edges("part_of")
        return build_nbs_protein_go_heterodata(
            torch.empty((0, self.stores.feature_dim), dtype=torch.float32),
            torch.as_tensor(boxes["center"], dtype=torch.float32),
            torch.as_tensor(boxes["offset"], dtype=torch.float32),
            torch.as_tensor(is_a.T, dtype=torch.long),
            go_part_of_edge_index=torch.as_tensor(part.T, dtype=torch.long),
            go_is_a_topology=torch.ones((is_a.shape[0], 2), dtype=torch.float32),
            go_part_of_topology=torch.ones((part.shape[0], 2), dtype=torch.float32),
            go_stats=torch.as_tensor(boxes["stats"], dtype=torch.float32),
            protein_ids=torch.empty(0, dtype=torch.long),
            go_ids=torch.arange(self.stores.full_boxes.num_go, dtype=torch.long),
        )


class LatenceNBSLocalBatchLoader:
    """Rank-sharded, deterministic, re-iterable NBS loader."""

    def __init__(
        self,
        sampler: GOQueryEpisodeSampler,
        materializer: LatenceNBSLocalGraphMaterializer,
        *,
        rank: int,
        world_size: int,
        steps_per_epoch_per_rank: int,
        base_seed: int,
        coverage_plan: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.sampler = sampler
        self.materializer = materializer
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.steps = int(steps_per_epoch_per_rank)
        self.base_seed = int(base_seed)
        self.coverage_plan = dict(coverage_plan or {})
        sample_parameters = inspect.signature(self.sampler.sample).parameters
        self._sampler_supports_episode_context = (
            "epoch" in sample_parameters and "global_episode" in sample_parameters
        )
        self.epoch = 1
        self.global_go_graph = materializer.build_global_go_graph()
        if self.steps <= 0 or self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid DDP loader shard")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[NBSLocalBatch]:
        for local_step in range(self.steps):
            global_episode = self.rank + local_step * self.world_size
            seed = (
                self.base_seed
                + self.epoch * 1_000_003
                + global_episode * 97
            )
            if self._sampler_supports_episode_context:
                kwargs = {
                    "seed": seed,
                    "epoch": self.epoch,
                    "global_episode": global_episode,
                }
                sample_parameters = inspect.signature(self.sampler.sample).parameters
                if "rank" in sample_parameters:
                    kwargs["rank"] = self.rank
                if "world_size" in sample_parameters:
                    kwargs["world_size"] = self.world_size
                episode = self.sampler.sample(**kwargs)
            else:
                episode = self.sampler.sample(seed=seed)
            yield self.materializer.materialize(episode, seed=seed + 31)


def build_latence_nbs_stores(config: Mapping[str, Any]) -> LatenceNBSStores:
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
    candidate_source = weak_manifest["backbone_rare_edges"]
    candidate_attributes = FixedDegreeCandidateAttributeStore(
        resolve_data_path(weak_manifest_path.parent, candidate_source["edge_attr_file"]),
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
        config=episode_cfg,
        seed=int(config.get("training", {}).get("seed", 3407)),
        weak_global_start=weak_role_start,
        weak_global_end=weak_role_end,
    )

    candidate_messages = FixedDegreeProteinGOStore(
        resolve_data_path(weak_manifest_path.parent, candidate_source["edge_index_file"]),
        resolve_data_path(weak_manifest_path.parent, candidate_source["edge_attr_file"]),
        fixed_degree=int(candidate_spec["fixed_degree"]),
        source_protein_start=int(candidate_spec["source_protein_start"] or 0),
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
        task_to_ontology=np.asarray(task_to_ontology, dtype=np.int64),
        feature_dim=features.feature_dim,
        num_task_go=num_task_go,
    )


def resolve_epoch_cycle_requirements(
    *,
    epoch_unit: str,
    go_cycles_floor: float,
    weak_pseudo_pass_target: float,
    core_gold_pass_target: float,
    pseudo_active_weak: int,
    predicted_pseudo_pairs_per_go_cycle: int,
    core_count: int,
    predicted_core_occurrences_per_go_cycle: int,
) -> dict[str, float]:
    """Resolve GO-cycle repetitions from protein-major supervision targets.

    ``weak_pseudo_pass_target`` and ``core_gold_pass_target`` are occurrence-
    equivalent passes, not promises of unique-node coverage.  Runtime bitset
    diagnostics report the corresponding unique coverage after each epoch.
    """
    go_cycles = float(go_cycles_floor)
    if go_cycles <= 0:
        raise ValueError("go_cycles_floor must be positive")
    weak_cycles = 0.0
    core_cycles = 0.0
    if epoch_unit == "protein_major_with_go_floor":
        if weak_pseudo_pass_target <= 0 or core_gold_pass_target <= 0:
            raise ValueError("protein-major pass targets must be positive")
        if pseudo_active_weak <= 0 or predicted_pseudo_pairs_per_go_cycle <= 0:
            raise ValueError("weak pseudo pass planning requires non-zero pseudo supervision")
        if core_count <= 0 or predicted_core_occurrences_per_go_cycle <= 0:
            raise ValueError("core gold pass planning requires non-zero core supervision")
        weak_cycles = float(
            pseudo_active_weak * weak_pseudo_pass_target
            / predicted_pseudo_pairs_per_go_cycle
        )
        core_cycles = float(
            core_count * core_gold_pass_target
            / predicted_core_occurrences_per_go_cycle
        )
    elif epoch_unit != "eligible_go_coverage_cycle":
        raise ValueError(f"unsupported epoch_unit: {epoch_unit}")
    resolved = max(go_cycles, weak_cycles, core_cycles)
    return {
        "go_cycles_required": go_cycles,
        "weak_pseudo_cycles_required": weak_cycles,
        "core_gold_cycles_required": core_cycles,
        "resolved_cycles": resolved,
    }


def resolve_hybrid_epoch_requirements(
    *,
    go_cycles_floor: float,
    eligible_go_count: int,
    coverage_slots_per_episode: int,
    world_size: int,
    weak_unique_coverage_target: float,
    pseudo_eligible_active_weak: int,
    weak_focus_queries_per_episode: int,
    weak_focus_targets_per_query: int,
    weak_focus_planning_efficiency: float,
    core_gold_pass_target: float,
    core_count: int,
    predicted_core_occurrences_per_go_cycle: int,
) -> dict[str, float | int]:
    """Plan a hybrid epoch from GO coverage and unique weak exposure.

    Unlike the occurrence-equivalent v0.5.2 planner, this contract reserves
    explicit weak-first GO-query slots and converts a desired unique weak
    coverage fraction directly into synchronized optimizer steps.
    """
    if eligible_go_count <= 0:
        raise ValueError("eligible_go_count must be positive")
    if coverage_slots_per_episode <= 0 or world_size <= 0:
        raise ValueError("coverage slots and world_size must be positive")
    if not 0.0 < weak_unique_coverage_target <= 1.0:
        raise ValueError("weak_unique_coverage_target must lie in (0,1]")
    if pseudo_eligible_active_weak <= 0:
        raise ValueError("hybrid weak planning requires eligible pseudo-active proteins")
    if weak_focus_queries_per_episode <= 0 or weak_focus_targets_per_query <= 0:
        raise ValueError("hybrid weak planning requires positive weak-focus capacity")
    if not 0.0 < weak_focus_planning_efficiency <= 1.0:
        raise ValueError("weak_focus_planning_efficiency must lie in (0,1]")
    if core_gold_pass_target <= 0 or core_count <= 0:
        raise ValueError("hybrid core planning requires positive targets/counts")
    if predicted_core_occurrences_per_go_cycle <= 0:
        raise ValueError("predicted core occurrences per GO cycle must be positive")

    global_coverage_slots = int(coverage_slots_per_episode) * int(world_size)
    go_steps = int(np.ceil(
        float(eligible_go_count) * float(go_cycles_floor) / global_coverage_slots
    ))
    focus_capacity_per_global_step = (
        int(weak_focus_queries_per_episode)
        * int(weak_focus_targets_per_query)
        * int(world_size)
    )
    weak_unique_target_count = int(np.ceil(
        int(pseudo_eligible_active_weak) * float(weak_unique_coverage_target)
    ))
    effective_focus_capacity = max(
        1.0,
        float(focus_capacity_per_global_step)
        * float(weak_focus_planning_efficiency),
    )
    weak_steps = int(np.ceil(weak_unique_target_count / effective_focus_capacity))

    core_occurrences_per_global_step = (
        float(predicted_core_occurrences_per_go_cycle)
        * global_coverage_slots
        / float(eligible_go_count)
    )
    core_steps = int(np.ceil(
        float(core_count) * float(core_gold_pass_target)
        / max(core_occurrences_per_global_step, 1e-12)
    ))
    resolved_steps = max(go_steps, weak_steps, core_steps)
    estimated_go_cycles = (
        resolved_steps * global_coverage_slots / float(eligible_go_count)
    )
    return {
        "go_steps_required": int(go_steps),
        "weak_unique_steps_required": int(weak_steps),
        "core_steps_required": int(core_steps),
        "resolved_steps": int(resolved_steps),
        "weak_unique_target_count": int(weak_unique_target_count),
        "focus_capacity_per_global_step": int(focus_capacity_per_global_step),
        "effective_focus_capacity_per_global_step": float(effective_focus_capacity),
        "estimated_go_cycles": float(estimated_go_cycles),
    }


def build_latence_nbs_train_loader(config: Mapping[str, Any]) -> LatenceNBSLocalBatchLoader:
    runtime = dict(config.get("_distributed_runtime", {}))
    rank = int(runtime.get("rank", 0))
    world_size = int(runtime.get("world_size", 1))
    sampling_cfg = NBSLocalGraphSamplingConfig.from_mapping(config.get("local_sampling"))
    stores = build_latence_nbs_stores(config)
    materializer = LatenceNBSLocalGraphMaterializer(stores, sampling_cfg)
    explicit_steps = sampling_cfg.steps_per_epoch_per_rank
    coverage_slots = int(stores.episode_sampler.coverage_slots_per_episode)
    eligible_go_count = int(stores.episode_sampler.eligible_go.size)
    role_counts = {
        role: int(indices.size)
        for role, indices in stores.registry.role_global_indices.items()
    }
    pseudo_active_weak = 0
    if stores.pseudo_messages is not None:
        pseudo_active_weak = int(
            np.count_nonzero(np.diff(stores.pseudo_messages.indptr) > 0)
        )

    sampler = stores.episode_sampler
    eligible = sampler.eligible_go
    pseudo_per_go_cycle = 0
    if sampler.pseudo is not None and sampler.config.pseudo_positive_per_query > 0:
        pseudo_per_go_cycle = int(np.sum(np.minimum(
            sampler.pseudo_degree[eligible],
            int(sampler.config.pseudo_positive_per_query),
        )))
    core_occurrences_per_go_cycle = 0
    for go_idx in eligible.tolist():
        support_count, positive_count = sampler._gold_requirements(int(go_idx))
        core_occurrences_per_go_cycle += int(support_count + positive_count)

    pseudo_eligible_active_weak = int(sampler.pseudo_active_role_rows.size)
    hybrid_plan: dict[str, float | int] = {}
    if sampling_cfg.epoch_unit == "hybrid_go_weak_coverage":
        if sampler.config.weak_focus_queries_per_episode <= 0:
            raise ValueError(
                "hybrid_go_weak_coverage requires episode.weak_focus_queries_per_episode > 0"
            )
        hybrid_plan = resolve_hybrid_epoch_requirements(
            go_cycles_floor=float(sampling_cfg.coverage_cycles_per_epoch),
            eligible_go_count=int(eligible_go_count),
            coverage_slots_per_episode=int(coverage_slots),
            world_size=int(world_size),
            weak_unique_coverage_target=float(
                sampling_cfg.weak_unique_coverage_target_per_epoch
            ),
            pseudo_eligible_active_weak=int(pseudo_eligible_active_weak),
            weak_focus_queries_per_episode=int(
                sampler.config.weak_focus_queries_per_episode
            ),
            weak_focus_targets_per_query=int(
                sampler.config.weak_focus_targets_per_query
            ),
            weak_focus_planning_efficiency=float(
                sampling_cfg.weak_focus_planning_efficiency
            ),
            core_gold_pass_target=float(
                sampling_cfg.core_gold_equivalent_passes_per_epoch
            ),
            core_count=int(role_counts.get("core", 0)),
            predicted_core_occurrences_per_go_cycle=int(
                core_occurrences_per_go_cycle
            ),
        )
        resolved_steps = (
            int(explicit_steps)
            if explicit_steps is not None
            else int(hybrid_plan["resolved_steps"])
        )
        resolved_cycles = float(
            resolved_steps * coverage_slots * world_size / max(1, eligible_go_count)
        )
        go_cycles_required = float(sampling_cfg.coverage_cycles_per_epoch)
        weak_cycles_required = 0.0
        core_cycles_required = float(
            int(hybrid_plan["core_steps_required"])
            * coverage_slots
            * world_size
            / max(1, eligible_go_count)
        )
    else:
        cycle_plan = resolve_epoch_cycle_requirements(
            epoch_unit=sampling_cfg.epoch_unit,
            go_cycles_floor=float(sampling_cfg.coverage_cycles_per_epoch),
            weak_pseudo_pass_target=float(
                sampling_cfg.weak_pseudo_equivalent_passes_per_epoch
            ),
            core_gold_pass_target=float(
                sampling_cfg.core_gold_equivalent_passes_per_epoch
            ),
            pseudo_active_weak=int(pseudo_active_weak),
            predicted_pseudo_pairs_per_go_cycle=int(pseudo_per_go_cycle),
            core_count=int(role_counts.get("core", 0)),
            predicted_core_occurrences_per_go_cycle=int(
                core_occurrences_per_go_cycle
            ),
        )
        go_cycles_required = float(cycle_plan["go_cycles_required"])
        weak_cycles_required = float(cycle_plan["weak_pseudo_cycles_required"])
        core_cycles_required = float(cycle_plan["core_gold_cycles_required"])
        resolved_cycles = float(cycle_plan["resolved_cycles"])
        if explicit_steps is None:
            global_slots_per_step = max(1, coverage_slots * world_size)
            resolved_steps = int(np.ceil(
                eligible_go_count * resolved_cycles / global_slots_per_step
            ))
        else:
            resolved_steps = int(explicit_steps)

    coverage_plan = {
        "epoch_unit": str(sampling_cfg.epoch_unit),
        "eligible_go_count": eligible_go_count,
        "eligibility_summary": dict(stores.episode_sampler.eligibility_summary),
        "coverage_slots_per_episode": coverage_slots,
        "weak_focus_queries_per_episode": int(
            sampler.config.weak_focus_queries_per_episode
        ),
        "weak_focus_targets_per_query": int(
            sampler.config.weak_focus_targets_per_query
        ),
        "world_size": int(world_size),
        "coverage_cycles_per_epoch": float(sampling_cfg.coverage_cycles_per_epoch),
        "resolved_coverage_cycles_per_epoch": float(resolved_cycles),
        "go_cycles_required": float(go_cycles_required),
        "weak_pseudo_cycles_required": float(weak_cycles_required),
        "core_gold_cycles_required": float(core_cycles_required),
        "weak_pseudo_equivalent_passes_target": float(
            sampling_cfg.weak_pseudo_equivalent_passes_per_epoch
        ),
        "core_gold_equivalent_passes_target": float(
            sampling_cfg.core_gold_equivalent_passes_per_epoch
        ),
        "weak_unique_coverage_target": float(
            sampling_cfg.weak_unique_coverage_target_per_epoch
        ),
        "weak_focus_planning_efficiency": float(
            sampling_cfg.weak_focus_planning_efficiency
        ),
        "predicted_pseudo_pairs_per_go_cycle": int(pseudo_per_go_cycle),
        "pseudo_sampling_mode": str(sampler.config.pseudo_sampling_mode),
        "pseudo_unique_targeting": bool(
            sampler.config.pseudo_sampling_mode == "go_cyclic_unique"
        ),
        "predicted_core_gold_occurrences_per_go_cycle": int(
            core_occurrences_per_go_cycle
        ),
        "resolved_steps_per_epoch_per_rank": int(resolved_steps),
        "estimated_base_cycle_coverage": float(
            resolved_steps * coverage_slots * world_size / max(1, eligible_go_count)
        ),
        "hybrid_plan": dict(hybrid_plan),
        "protein_universe": {
            "total": int(stores.registry.num_proteins),
            "role_counts": role_counts,
            "pseudo_active_weak": int(pseudo_active_weak),
            "pseudo_eligible_active_weak": int(pseudo_eligible_active_weak),
        },
        "stage1_reference": {
            "global_steps_per_epoch": sampling_cfg.stage1_reference_global_steps_per_epoch,
            "used_to_resolve_nbs_steps": False,
            "reason": (
                "stage 1 iterates protein mini-batches, whereas NBS uses "
                "GO coverage plus explicit weak-first query slots"
            ),
        },
    }
    return LatenceNBSLocalBatchLoader(
        stores.episode_sampler,
        materializer,
        rank=rank,
        world_size=world_size,
        steps_per_epoch_per_rank=resolved_steps,
        base_seed=sampling_cfg.base_seed,
        coverage_plan=coverage_plan,
    )
