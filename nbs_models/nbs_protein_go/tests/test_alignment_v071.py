from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from nbs_pg.candidate_evidence import align_candidate_evidence
from nbs_pg.config import NBSConfig
from nbs_pg.episode import GOQueryEpisodeSampler, NBSQueryEpisodeConfig
from nbs_pg.inference import ExternalCandidateEvidenceStore
from nbs_pg.latence_graph_stores import FixedDegreeProteinGOStore, RoleLocalProteinGOCSRStore
from nbs_pg.latence_stores import GOProteinCSRStore, RoleAwareBaseLogitStore, RoleProbabilitySlice
from nbs_pg.matcher import NBSGatedDeltaAttnRes
from nbs_pg.types import NBSNeighborhoodHierarchy, NBSQueryCondition
from test_hybrid_weak_focus_v054 import _registry, _save


def test_evidence_is_order_invariant_and_shared_by_training_and_inference(tmp_path: Path):
    go = np.array([[2, 0, 4], [1, 2, 4]], dtype=np.int32)
    attr = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 20
    attr[:, :, 2] = [1, .5, 1 / 3]
    edges = np.stack([np.repeat([0, 1], 3), go.reshape(-1)])
    train = FixedDegreeProteinGOStore(
        _save(tmp_path / 'edges.npy', edges),
        _save(tmp_path / 'attrs.npy', attr.reshape(-1, 3)),
        fixed_degree=3, source_protein_start=0,
    )
    external = ExternalCandidateEvidenceStore(_save(tmp_path / 'go.npy', go), tmp_path / 'attrs.npy')
    p, q = np.array([1, 0]), np.array([4, 3, 2, 0])
    a, b = train.gather_matrix(p, q, chunk_size=1), external.gather(p, q)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[1], np.zeros((2, 3)))
    np.testing.assert_array_equal(a[3, 1], attr[0, 1])
    perm = np.array([5, 1, 3, 0, 4, 2])
    np.testing.assert_array_equal(a, align_candidate_evidence(p, q, edges[:, perm], attr.reshape(-1, 3)[perm]))


def make_sampler(tmp_path: Path, *, complete: bool = True, num_queries: int = 8):
    num_go, core, weak = 8, 24, 4
    registry = _registry(tmp_path / 'registry.csv', core=core, weak=weak)
    candidate_go = np.tile(np.arange(6, dtype=np.int32), (core + weak, 1))
    candidate_attr = np.zeros((core + weak, 6, 3), dtype=np.float32)
    candidate_attr[..., 0] = .6
    candidate_attr[..., 1] = .7
    candidate_attr[..., 2] = 1 / np.arange(1, 7)
    edges = np.stack([np.repeat(np.arange(core + weak), 6), candidate_go.reshape(-1)])
    forward = FixedDegreeProteinGOStore(
        _save(tmp_path / 'candidate_edges.npy', edges),
        _save(tmp_path / 'candidate_attr.npy', candidate_attr.reshape(-1, 3)),
        fixed_degree=6, source_protein_start=0,
    )
    cand_rows = [np.arange(core + weak) if g < 6 else np.empty(0, dtype=int) for g in range(num_go)]
    cand_ptr = np.cumsum([0] + [len(x) for x in cand_rows])
    pseudo_rows = [np.arange(core, core + weak) if g < 4 else np.empty(0, dtype=int) for g in range(num_go)]
    pseudo_ptr = np.cumsum([0] + [len(x) for x in pseudo_rows])
    probabilities = np.repeat([.95, .85, .75, .65], weak).astype(np.float16)
    pseudo_forward = RoleLocalProteinGOCSRStore(
        _save(tmp_path / 'p_ptr.npy', np.arange(0, weak * 4 + 1, 4, dtype=np.int64)),
        _save(tmp_path / 'p_go.npy', np.tile(np.arange(4, dtype=np.int32), weak)),
        role='weak', registry=registry,
        probability_path=_save(tmp_path / 'p_prob.npy', np.tile(np.array([.95, .85, .75, .65], dtype=np.float16), weak)),
    )
    config = NBSQueryEpisodeConfig(
        num_queries=num_queries, support_per_query=1, gold_positive_per_query=1,
        hard_candidate_per_query=1, pseudo_positive_per_query=2,
        pseudo_sampling_mode='go_cyclic_unique', query_sampling_mode='shuffled_cycle',
        weak_primary_query_bank_mode='greedy_multi', weak_primary_query_source='pseudo_candidate_intersection',
        weak_primary_proteins_per_episode=4, weak_focus_queries_per_episode=4,
        weak_focus_targets_per_query=2, weak_primary_positive_go_per_protein=2,
        weak_primary_rotate_go_across_epochs=True,
        weak_primary_hard_negative_go_per_protein=2, weak_primary_background_go_per_protein=2,
        weak_primary_hard_query_slots_per_episode=2,
        weak_primary_complete_query_positives=complete,
        candidate_evidence_scope='decoded_all', max_candidates=48,
        require_full_supervision_retention=True,
    )
    return GOQueryEpisodeSampler(
        gold=GOProteinCSRStore(_save(tmp_path / 'g_ptr.npy', np.arange(0, core + 1, 3, dtype=np.int64)), _save(tmp_path / 'g_id.npy', np.arange(core, dtype=np.int32))),
        candidate=GOProteinCSRStore(_save(tmp_path / 'c_ptr.npy', cand_ptr), _save(tmp_path / 'c_id.npy', np.concatenate(cand_rows).astype(np.int32))),
        pseudo=GOProteinCSRStore(_save(tmp_path / 's_ptr.npy', pseudo_ptr), _save(tmp_path / 's_id.npy', np.concatenate(pseudo_rows).astype(np.int32)), payload_paths={'probability': _save(tmp_path / 's_prob.npy', probabilities)}),
        candidate_by_protein=forward, pseudo_by_protein=pseudo_forward,
        pseudo_active_role_rows=np.arange(weak),
        base_logits=RoleAwareBaseLogitStore([RoleProbabilitySlice('all', 0, core + weak, str(_save(tmp_path / 'base.npy', np.full((core + weak, num_go), .01, dtype=np.float16))))], num_go=num_go),
        train_go_counts=np.full(num_go, 3.), config=config, seed=3407,
        weak_global_start=core, weak_global_end=core + weak,
    )


def test_full_query_positives_hard_budget_and_label_independent_evidence(tmp_path: Path):
    sampler = make_sampler(tmp_path)
    episode = sampler.sample(seed=51, epoch=1, global_episode=0)
    assert sampler.coverage_slots_per_episode == 2
    assert set(episode.metadata['weak_primary_hard_query_go_idx']) == {4, 5}
    for protein in episode.metadata['weak_primary_anchor_protein_idx']:
        column = np.flatnonzero(episode.candidate_protein_idx == protein)[0]
        for row, go in enumerate(episode.query_go_idx):
            assert episode.mask[row, column]
            assert episode.weak_primary_mask[row, column]
            assert bool(episode.pseudo_mask[row, column]) == (go < 4)
            assert bool(episode.candidate_evidence[row, column, 2] > 0) == (go < 6)
        assert int(episode.pseudo_mask[:, column].sum()) == 4
    assert episode.metadata['decoded_cross_pseudo_pairs'] > 0
    assert episode.metadata['weak_primary_hard_go_per_anchor_min'] == 2
    assert episode.metadata['candidate_evidence_positive_pairs'] > 0
    assert episode.metadata['candidate_evidence_unknown_pairs'] > 0
    assert episode.metadata['weak_primary_anchor_count'] == 4
    assert sampler.weak_primary_progress(epoch=1, rank=0, world_size=1)['remaining'] == 0
    np.testing.assert_array_equal(episode.candidate_evidence, sampler.candidate_by_protein.gather_matrix(episode.candidate_protein_idx, episode.query_go_idx))


def test_rotation_changes_assignments_and_hard_queries_never_use_pseudo(tmp_path: Path):
    sampler = make_sampler(tmp_path, complete=False)
    assignments = []
    for epoch in range(1, 5):
        sampler.sample(seed=51, epoch=epoch, global_episode=0)
        assignments.append({p: set(v) for p, v in sampler._weak_primary_anchor_assignments_current.items()})
        for go in sampler._weak_primary_hard_query_current:
            assert go >= 4
    assert any(len(set.union(*(a[p] for a in assignments))) > 2 for p in assignments[0])


def test_double_zero_branch_is_dead_and_nonzero_scale_starts_residual():
    evidence = torch.tensor([[[.7, .5, .5], [.2, -.5, .25]]])
    for scale in (0., .1):
        torch.manual_seed(7)
        matcher = NBSGatedDeltaAttnRes(NBSConfig(hidden_dim=8, num_layers=1, candidate_evidence_residual_scale_init=scale), num_sources=1)
        out = matcher._encode_candidate_evidence(evidence, torch.full((1, 2), .4))
        torch.testing.assert_close(out, evidence[..., 2])
        out.sum().backward()
        norm = float(matcher.candidate_evidence_residual[-1].weight.grad.norm())
        assert (norm > 0) == (scale != 0)
        # The scale starts learning only after the zero output layer moves.
        assert float(matcher.candidate_evidence_scale.grad) == 0


def test_cold_start_full_matcher_gradient_chain_opens_after_optimizer_steps():
    torch.manual_seed(19)
    matcher = NBSGatedDeltaAttnRes(NBSConfig(hidden_dim=8, num_layers=1,
        candidate_evidence_residual_scale_init=.1, graph_delta_scale_init=0.), num_sources=1)
    hierarchy = NBSNeighborhoodHierarchy(final_context=torch.randn(4, 8), source_contexts=torch.randn(1, 4, 8), source_names=('source',))
    base = torch.randn(2, 4)
    condition = NBSQueryCondition(base_query=torch.randn(2, 8), seed_index=torch.tensor([0, 1]),
        seed_query_index=torch.tensor([0, 1]), num_queries=2, candidate_index=None,
        base_logits=base, candidate_evidence=torch.tensor([[[.7, .6, .5]] * 4] * 2),
        expert_prob=None, query_go_frequency=torch.ones(2), labels=None, mask=None,
        confidence=None, pseudo_mask=None)
    torch.testing.assert_close(matcher(hierarchy, condition).logits, base, rtol=0, atol=0)
    optimizer = torch.optim.SGD(matcher.parameters(), lr=.1)
    seen = {'graph': False, 'residual': False, 'scale': False}
    labels = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    for _ in range(12):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(matcher(hierarchy, condition).logits, labels)
        loss.backward()
        for name, parameter in [('graph', matcher.graph_delta_scale), ('residual', matcher.candidate_evidence_residual[-1].weight), ('scale', matcher.candidate_evidence_scale)]:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            seen[name] |= bool(parameter.grad.abs().sum() > 0)
        optimizer.step()
    assert all(seen.values()), seen


def test_candidate_only_export_preserves_base_and_diagnostic_arrays(tmp_path: Path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from nbs_pg.inference import export_full_task_probabilities, FullTaskInferenceConfig
    from nbs_pg.types import ProteinGOQueryBatch
    # Stub only graph construction/model scoring, exercise the real chunked exporter.
    class Model:
        config = SimpleNamespace(candidate_evidence_dim=3)
        def eval(self): return self
        def encode_graph(self, *args, **kwargs): return None
        def score_external_candidates(self, encoded, query, x, **kwargs):
            delta = torch.ones_like(query.base_logits)
            return SimpleNamespace(logits=query.base_logits + delta,
                auxiliary={'applied_graph_delta': delta, 'delta_gate': delta * .5})
    template = ProteinGOQueryBatch(seed_protein_index=torch.empty(0, dtype=torch.long), seed_query_index=torch.empty(0, dtype=torch.long), num_queries=3)
    materializer = SimpleNamespace(materialize=lambda *a, **kw: SimpleNamespace(graph=SimpleNamespace(to=lambda d: None), query=template))
    stores = SimpleNamespace(num_task_go=3, episode_sampler=SimpleNamespace(train_go_counts=np.ones(3)))
    base = np.array([[0., .2, 1.], [.3, 0., .5]], dtype=np.float16)
    evidence = ExternalCandidateEvidenceStore(_save(tmp_path / 'e_go.npy', np.array([[1], [0]], dtype=np.int32)),
        _save(tmp_path / 'e_attr.npy', np.array([[[.2, .2, 1.]], [[.3, .3, 1.]]], dtype=np.float32)))
    with patch('nbs_pg.inference.build_support_episode', return_value=None):
        out = export_full_task_probabilities(model=Model(), stores=stores, materializer=materializer, global_go_cache=None,
            external_repr=np.zeros((2, 4)), base_values=base, output_path=tmp_path / 'prob.npy', device='cpu',
            config=FullTaskInferenceConfig(go_chunk_size=2, protein_batch_size=1, preserve_base_outside_candidates=True),
            candidate_evidence_store=evidence, logit_delta_output_path=tmp_path / 'delta.npy', delta_gate_output_path=tmp_path / 'gate.npy')
    selected = np.array([[False, True, False], [True, False, False]])
    np.testing.assert_array_equal(out[~selected], base[~selected])
    assert np.all(out[selected] > base[selected])
    assert np.all(np.load(tmp_path / 'delta.npy')[~selected] == 0)
    assert np.all(np.load(tmp_path / 'gate.npy')[~selected] == 0)


def test_audit_reports_complete_positive_alignment_and_actual_hard_pairs(tmp_path: Path):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/nbs'))
    from audit_nbs_protein_go_batch_alignment import _row_for_anchor, _summary
    sampler = make_sampler(tmp_path)
    episode = sampler.sample(seed=51, epoch=1, global_episode=0)
    stores = SimpleNamespace(episode_sampler=sampler, registry=sampler.pseudo_by_protein.registry)
    rows = [_row_for_anchor(stores, episode, epoch=1, episode_index=0, protein=int(p)) for p in episode.metadata['weak_primary_anchor_protein_idx']]
    assert all(r['query_to_loss_positive_gap'] == 0 for r in rows)
    assert all(r['hard_pu_go'] == 2 for r in rows)
    assert all(r['background_pu_go'] == 2 for r in rows)
    result = _summary(rows, [], config_path=tmp_path / 'config.json')
    assert result['anchor_rates']['all_query_positives_supervised'] == 1
    assert result['cross_epoch_repeated_proteins'] == 0


def test_v060_inference_load_only_allows_absent_disabled_go_branch():
    from nbs_pg.model import ProteinGONBSModel
    from nbs_pg.checkpoint_compat import load_inference_state
    torch.manual_seed(31)
    model = ProteinGONBSModel(NBSConfig(hidden_dim=8, num_layers=1, use_go_residual_query=False,
        candidate_evidence_residual_scale_init=0.), protein_input_dim=4, go_box_dim=3)
    full = {k: v.clone() for k, v in model.state_dict().items()}
    legacy = {k: v for k, v in full.items() if not k.startswith('query_encoder.go_residual_')}
    result = load_inference_state(model, legacy)
    assert result['mode'] == 'disabled_go_residual_branch_only'
    for k, v in legacy.items():
        torch.testing.assert_close(model.state_dict()[k], v, atol=0, rtol=0)
    assert float(model.matcher.candidate_evidence_scale) == 0
    broken = dict(legacy)
    del broken['matcher.graph_delta_scale']
    try:
        load_inference_state(model, broken)
    except RuntimeError:
        pass
    else:
        raise AssertionError('an active missing parameter was silently accepted')
    model.config.use_go_residual_query = True
    try:
        load_inference_state(model, legacy)
    except RuntimeError:
        pass
    else:
        raise AssertionError('enabled GO residual parameters cannot be invented')
