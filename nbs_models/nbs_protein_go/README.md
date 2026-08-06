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

## v0.3 fixed-epoch training

NBS training now follows the first LATENCE stage and does **not** use a
validation set for checkpoint selection.  A run always proceeds to its configured
final epoch and saves explicitly named snapshots, for example:

```text
nbs_epoch100.pt
nbs_epoch150.pt
```

There is no `best.pt`, validation metric monitor, patience, or early stopping.
The snapshots are evaluated independently after training.  Checkpoint metadata
records:

```text
selection_policy = fixed_epoch_snapshots
validation_used = false
early_stopping = false
```

The reference BP schedule is in:

```text
configs/bp_fixed_epoch_v0.3.json
```

Validate the policy without constructing the full graph:

```bash
python scripts/nbs/run_train_nbs_fixed_epochs.py
# with NBS_VALIDATE_CONFIG_ONLY=1
```

or directly:

```bash
python scripts/nbs/train_nbs_fixed_epochs.py \
  --config nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.3.json \
  --validate-config-only
```

Production training uses a project-specific component factory:

```bash
export NBS_COMPONENT_FACTORY=experiments.nbs.latence_nbs_components:build_components
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

The component factory builds the model, optimizer and a re-iterable loader of
`NBSLocalBatch` objects.  The loader must materialize only batch-local graph
relations from mmap/CSR stores; it must not construct one global PyG object for
the 281,457,664 candidate edges.

### Training modules

```text
nbs_pg/training.py
    fixed-epoch trainer, atomic checkpoints, resume and training history

nbs_pg/latence_stores.py
    GO-major CSR, fixed-512 candidate attributes, role-aware base logits,
    and GO box mmap stores

nbs_pg/episode.py
    GO-query episode sampling with episode-wide support/candidate separation,
    gold positives, hard candidates, optional pseudo positives and
    sampled-unlabelled supervision weights

experiments/nbs/latence_nbs_components.py
    reference component factory and AdamW parameter groups
```

### Correct NBS hierarchy axis

NBS scores have shape:

```text
[GO query, candidate protein]
```

The hierarchy loss therefore constrains child and parent **query rows**.  Use
`query_hierarchy_violation_loss()` or `hierarchy_go_axis=0`.  The old
`[protein, GO]` column convention remains available through `go_axis=1` for
compatibility.

### Three-column candidate evidence

The gate accepts the exported relation attributes:

```text
[backbone_probability, selector_score, reciprocal_rank]
```

At initialization, candidate evidence equals reciprocal rank.  A bounded,
zero-initialized residual can subsequently learn from all three columns.  This
keeps candidate ranking evidence distinct from the separate base-probability
channel.

### Open-world supervision weights

`ProteinGOQueryBatch.supervision_weight` applies to both gold and pseudo
positions.  It supports low-weight sampled-unlabelled candidates without
turning every unknown protein–GO pair into a hard negative.  Pseudo confidence
is multiplied only for positions marked by `pseudo_mask`.

The episode sampler's "held-out gold positive" is only a support/query split
inside the **training episode**.  It is still part of the training objective and
must not be interpreted as a validation split or used for model selection.

## BoxSquaredEL training contract and GO-index projection

The supplied GO geometry run uses the following verified contract:

```text
model                         BoxSquaredEL
embedding dimension           512
ontology classes              44,919
relations                     8
classifier GO columns         21,312 (BP)
class vector layout           [center(512), abs(offset)(512)]
selected geometry epoch       1000
strict parser unparsed lines  0
```

Reference copies of the source configuration are included under:

```text
configs/go_boxsqel/go_boxsqel_manifest_512.json
configs/go_boxsqel/go_boxsqel_parser_report_512.json
```

The ontology embedding cannot be indexed directly by BP classifier columns,
because it contains all 44,919 parsed ontology classes.  Before NBS training,
project the geometry through the immutable `go_registry.tsv`:

```bash
python scripts/nbs/run_prepare_go_boxsqel_for_nbs.py
```

Default output:

```text
outputs/latence_nbs/<RUN_TAG>/epoch100/bp/nbs_indices/go_boxsqel_512/
├── go_box_center.f32.npy
├── go_box_offset.f32.npy
├── go_box_stats.f32.npy
├── go_box_source_row.i32.npy
└── go_box_alignment_manifest.json
```

The projection reads the BoxSquaredEL checkpoint class mapping, normalizes GO
IRIs and compact identifiers, preserves all 21,312 classifier indices, and
replicates the same geometry for canonical/alt-ID duplicate columns.  Strict
mode fails if any classifier GO term lacks BoxSquaredEL geometry.

The BP training template now fixes:

```text
model_inputs.go_box_dim = 512
```

and records the BoxSquaredEL manifest, parser report, artifact selection, and
alignment manifest under `go_boxsqel`.

## Periodic checkpoint interval

In addition to named snapshots such as epochs 100 and 150, training supports:

```json
"save_interval_epochs": 10
```

The actual checkpoint epochs are the union of:

```text
multiples of save_interval_epochs
+ explicitly listed save_epochs
+ final epoch when save_final=true
```

For a 150-epoch run with interval 10, checkpoints are written at epochs
10, 20, ..., 150.  To save every five epochs without editing JSON:

```bash
NBS_SAVE_INTERVAL_EPOCHS=5 \
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

or directly:

```bash
python scripts/nbs/train_nbs_fixed_epochs.py \
  --config nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.3.json \
  --save-interval-epochs 5
```

The former `save_every` JSON field is accepted only as a compatibility alias;
new configurations should use `save_interval_epochs`.
