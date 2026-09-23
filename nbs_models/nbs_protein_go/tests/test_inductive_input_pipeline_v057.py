from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_script(name: str, relative: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_pickle_without_sequences_selects_msa_binary_mode(tmp_path: Path) -> None:
    prepare = load_script(
        "_prepare_nbs_ind_test_inputs_test",
        "scripts/nbs/prepare_nbs_ind_test_inputs.py",
    )
    path = tmp_path / "metadata.pkl"
    payload = {
        "ind_test": {
            "biological_process": {
                "proteins": ["P2", "P1"],
                "prop_annotations": [[1], [0]],
            }
        }
    }
    path.write_bytes(pickle.dumps(payload))
    ids, sequences = prepare.parse_pickle(
        path,
        mode="ind_test",
        task="bp",
        protein_key=None,
        sequence_key=None,
    )
    assert ids == ["P2", "P1"]
    assert sequences is None


def test_fasta_and_pickle_are_aligned_to_metadata_order(tmp_path: Path) -> None:
    prepare = load_script(
        "_prepare_nbs_ind_test_inputs_align_test",
        "scripts/nbs/prepare_nbs_ind_test_inputs.py",
    )
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">P1\nACD\n>P2 description\nEFG\n", encoding="utf-8")
    metadata = tmp_path / "metadata.pkl"
    metadata.write_bytes(
        pickle.dumps(
            {
                "ind_test": {
                    "bp": {
                        "proteins": ["P2", "P1"],
                        "prop_annotations": [[1], [0]],
                    }
                }
            }
        )
    )
    args = type(
        "Args",
        (),
        {
            "fasta": fasta,
            "metadata_file": metadata,
            "mode": "ind_test",
            "task": "bp",
            "pickle_protein_key": None,
            "pickle_sequence_key": None,
        },
    )()
    records, _ = prepare.resolve_records(args)
    assert records == [("P2", "EFG"), ("P1", "ACD")]


def test_singleton_encoding_uses_only_query_row() -> None:
    prepare = load_script(
        "_prepare_nbs_ind_test_inputs_encoding_test",
        "scripts/nbs/prepare_nbs_ind_test_inputs.py",
    )
    encoded, truncated, unknown = prepare.encode_singleton_batch(
        [("P1", "ACDX")],
        top_k=3,
        max_len=3,
        alphabet="-ACDX",
        unknown_policy="x",
    )
    assert encoded.shape == (1, 3, 3)
    assert encoded[0, 0].tolist() == [1, 2, 3]
    assert np.count_nonzero(encoded[0, 1:]) == 0
    assert truncated == 1
    assert unknown == 0


def test_external_pp_encoding_is_batch_isolated() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from nbs_pg import NBSConfig, ProteinGONBSModel

    torch.manual_seed(7)
    model = ProteinGONBSModel(
        NBSConfig(
            hidden_dim=8,
            num_layers=2,
            input_dropout=0.0,
            relation_dropout=0.0,
            residual_dropout=0.0,
        ),
        protein_input_dim=4,
        go_box_dim=3,
    ).eval()
    protein = torch.randn(2, 4)
    neighbor = torch.randn(2, 3, 4)
    attr = torch.rand(2, 3, 3)
    attr[:, :, 0] = attr[:, :, 0].clamp(0.1, 1.0)

    together = model.encode_external_protein_candidates(
        protein,
        neighbor_x=neighbor,
        neighbor_edge_attr=attr,
        neighbor_fanouts=(3, 2),
    )
    separate = [
        model.encode_external_protein_candidates(
            protein[row : row + 1],
            neighbor_x=neighbor[row : row + 1],
            neighbor_edge_attr=attr[row : row + 1],
            neighbor_fanouts=(3, 2),
        )
        for row in range(2)
    ]
    expected = torch.cat([value.final_context for value in separate], dim=0)
    assert torch.allclose(together.final_context, expected, atol=1e-6, rtol=1e-6)
    assert any("similar_to" in name for name in together.source_names)
    for source_idx, name in enumerate(together.source_names):
        if "similar_to" not in name:
            assert torch.count_nonzero(together.source_contexts[source_idx]) == 0


def test_inductive_api_contract_accepts_the_complete_r5_modules() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    exporter = load_script(
        "_export_nbs_full_task_predictions_contract_test",
        "scripts/nbs/export_nbs_full_task_predictions.py",
    )
    import nbs_pg.inference as inference_api
    from nbs_pg import ProteinGONBSModel
    from nbs_pg.matcher import NBSGatedDeltaAttnRes

    assert (
        exporter.validate_inductive_inference_api(
            inference_api, ProteinGONBSModel, NBSGatedDeltaAttnRes
        )
        == 5
    )


def test_inductive_api_contract_rejects_new_exporter_with_old_library() -> None:
    pytest.importorskip("torch")
    exporter = load_script(
        "_export_nbs_full_task_predictions_mixed_contract_test",
        "scripts/nbs/export_nbs_full_task_predictions.py",
    )

    class OldModel:
        def score_external_candidates(self, encoded, query, candidate_x):
            del encoded, query, candidate_x

    class OldMatcher:
        def forward(self, hierarchy, condition, return_aux=False):
            del hierarchy, condition, return_aux

    old_inference = SimpleNamespace(
        ExternalCandidateEvidenceStore=object,
        FullTaskInferenceConfig=object,
        export_full_task_probabilities=lambda: None,
    )
    with pytest.raises(RuntimeError, match="newer exporter.*older nbs_pg"):
        exporter.validate_inductive_inference_api(
            old_inference, OldModel, OldMatcher
        )


def test_stage1_metric_adapter_preserves_the_official_result() -> None:
    pytest.importorskip("torch")
    evaluator_script = load_script(
        "_eval_nbs_ind_test_predictions_metric_test",
        "scripts/nbs/eval_nbs_ind_test_predictions.py",
    )
    captured = {}

    def fake_evalperf(**kwargs):
        captured.update(kwargs)
        return {"fmax": 55.5, "auprc": 44.0, "threshold": 0.37}

    labels = np.asarray([[1, 0], [0, 1]], dtype=np.bool_)
    probabilities = np.asarray([[0.8, 0.1], [0.2, 0.7]], dtype=np.float32)
    result = evaluator_script._compute_stage1_metric_pack(
        fake_evalperf, labels, probabilities
    )
    assert result == {"fmax": 55.5, "auprc": 44.0, "threshold": 0.37}
    assert captured["threshold"] is True
    assert captured["auprc"] is True
    assert captured["no_empty_labels"] is False
    assert captured["no_zero_classes"] is False
