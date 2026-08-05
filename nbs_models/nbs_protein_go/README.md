# NBS protein–GO

**NBS (Neighborhood–BoxSquare)** models the LATENCE weak-to-strong transition as
functional neighbourhoods being organized inside nested GO boxes.  The boxes
are implemented with **BoxSquaredEL**; the model name is broader than the
specific box-embedding method.

```text
weak evolutionary / PPI / prediction neighbourhood
        -> GO-conditioned neighbourhood query
        -> nested BoxSquaredEL ontology geometry
        -> gated graph residual over first-stage base logits
        -> stronger functional neighbourhood
```

The package is designed for the second stage of LATENCE.  It is not a renamed
HGAT and does not merge all graph evidence into one generic relation.

## Project location

The intended LATENCE layout is:

```text
latence-project/
├── nbs_models/
│   └── nbs_protein_go/
│       ├── nbs_pg/
│       ├── tests/
│       ├── examples/
│       └── pyproject.toml
└── scripts/
    └── nbs/
        ├── build_go_protein_inverted_index.py
        └── run_build_go_protein_inverted_index.py
```

The second-stage data root is expected to be `outputs/latence_nbs/`.

## Core design

```text
Protein / GO evidence-aware heterogeneous backbone
    -> additive per-layer, per-relation neighbourhood sources
    -> NBSNeighborhoodHierarchy
    -> GO-query-conditioned NBS-GDAR routing
    -> base logits + gated graph delta
```

GO nodes retain separate `center`, `log-offset`, static box anchor and propagated
context channels.  The final graph branch starts exactly at the first-stage
base logits because `graph_delta_scale=0` at initialization.

## Evidence-separated schema

Protein relations:

```text
protein --ppi-----------> protein
protein --similar_to----> protein
protein --weak_to_core--> protein
```

Protein–GO evidence:

```text
protein --gold_annotated_with-------> GO
GO      --has_gold_annotation-------> protein

protein --backbone_rare_candidate---> GO
GO      --candidate_of--------------> protein

protein --pseudo_annotated_with-----> GO
GO      --has_pseudo_annotation-----> protein
```

GO ontology:

```text
GO child  --is_a-------> GO parent
GO parent --has_child-> GO child
GO part   --part_of----> GO whole
GO whole  --has_part---> GO part
```

The default edge dimensions match the current LATENCE manifests:

- P–P: `[confidence, source_score, reciprocal_rank]`;
- gold/pseudo annotation: three provenance-aware columns;
- backbone candidate: `[backbone_probability, selector_score, reciprocal_rank]`;
- `is_a/has_child`: 8 BoxSquaredEL features plus
  `[inverse_hop_weight, is_direct]`;
- `part_of/has_part`: `[inverse_hop_weight, is_direct]` without pretending that
  partonomy is identical to class inclusion.

## Hundred-million-scale candidate evidence

For the current BP export, the backbone relation contains **281,457,664 edges**:

```text
549,722 proteins × 512 rare-GO candidates
```

This relation is a data-scale contribution of LATENCE and remains a formal NBS
evidence source.  It must not be loaded as one GPU-resident `HeteroData` edge
store.  Production training keeps it in mmap storage and materializes only
batch-local candidate edges.

## GO→Protein inverted index

NBS is GO-query oriented while the exported candidate relation is
protein-major.  The accompanying builder performs two streaming passes and
creates GO-major CSR without globally sorting all 281 million edges in memory.

Direct BP run:

```bash
python scripts/nbs/run_build_go_protein_inverted_index.py
```

Configurable CLI:

```bash
python scripts/nbs/build_go_protein_inverted_index.py \
  --project-root /path/to/latence-project \
  --weak-graph-manifest outputs/latence_nbs/<run>/epoch100/bp/weak_graph_predictions/weak_graph_predictions_manifest.json \
  --protein-registry outputs/latence_nbs/<run>/epoch100/bp/features/protein_registry.csv \
  --output-dir outputs/latence_nbs/<run>/epoch100/bp/nbs_indices/go_protein
```

Candidate outputs use compact source ranks after verifying the fixed-512,
protein-major contract:

```text
candidate_go_indptr.i64.npy
candidate_protein_idx.i32.npy
candidate_source_rank.u16.npy
```

The original candidate edge attributes are retrieved by:

```text
source_row = (global_protein_idx - source_protein_start) * 512 + source_rank
```

Pseudo outputs are also inverted from weak protein-major CSR and converted from
role-local rows to global protein indices.

## Leakage control

Use relation-specific masking:

```python
graph = mask_candidate_evidence_edges(
    graph,
    candidate_protein_index,
    query_go_index=query_go_index,
    gold_mode="all",
    pseudo_mode="all",
    candidate_mode="query_only",
)
```

Gold and pseudo annotations are removed for candidate proteins.  Only the
currently scored candidate–query-GO backbone edges are removed, so unrelated
first-stage candidate evidence can remain as context.

## Expert-free NBS inference

The default candidate delta gate is `student_candidate` and uses:

```text
base probability
backbone candidate evidence
NBS graph score
GO training frequency
```

Expert probability is retained only through the explicit
`delta_gate_feature_mode="legacy_expert"` ablation.  Modelout remains a pseudo
supervision source and strong reference, not a required NBS inference input.

## Tests

Pure PyTorch/NumPy tests:

```bash
cd nbs_models/nbs_protein_go
PYTHONPATH=. pytest -q \
  tests/test_box_geometry.py \
  tests/test_go_box_encoder.py \
  tests/test_matcher_equivalence.py \
  tests/test_residual_refinement.py \
  tests/test_global_go_fallback.py \
  tests/test_inverted_index.py \
  tests/test_student_gate.py
```

Real PyG integration tests, to be run on the LATENCE server:

```bash
PYTHONPATH=. pytest -q \
  tests/test_source_additivity.py \
  tests/test_data_leakage.py
```
