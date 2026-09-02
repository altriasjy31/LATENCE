# LATENCE NBS protein–GO v0.4

**NBS (Neighborhood–BoxSquare)** models the LATENCE weak-to-strong transition as
protein functional neighbourhoods being organized inside nested ontology
boxes. Box geometry is implemented by **BoxSquaredEL**.

```text
first-stage protein representation / base logits
        +
PPI / similar_to / weak_to_core neighbourhoods
        +
gold / backbone-candidate / pseudo Protein–GO evidence
        +
full BoxSquaredEL ontology geometry and direct G–G relations
        ↓
GO-conditioned NBS query and gated graph residual
        ↓
stronger functional neighbourhood prediction
```

NBS is not a renamed HGAT. Evidence provenance, relation direction, candidate
leakage control and the first-stage base-logit anchor are explicit contracts.

## Project layout

```text
latence-project/
├── nbs_models/
│   └── nbs_protein_go/
│       ├── nbs_pg/
│       ├── configs/
│       ├── tests/
│       └── examples/
├── experiments/
│   └── nbs/
│       ├── latence_nbs_components.py
│       └── latence_nbs_loader.py
└── scripts/
    └── nbs/
```

Second-stage data and training outputs use:

```text
outputs/latence_nbs/
outputs/latence_nbs_train/
```

## Evidence-separated graph schema

Protein relations preserve their upstream message direction:

```text
protein --ppi-----------> protein
protein --similar_to----> protein       # neighbour -> core query
protein --weak_to_core--> protein       # weak -> core
```

Protein–GO evidence remains separated:

```text
protein --gold_annotated_with-------> GO
GO      --has_gold_annotation-------> protein

protein --backbone_rare_candidate---> GO
GO      --candidate_of--------------> protein

protein --pseudo_annotated_with-----> GO
GO      --has_pseudo_annotation-----> protein
```

GO relations:

```text
GO class --is_a-------> GO superclass
GO class --has_child--> GO subclass
GO part  --part_of----> GO whole
GO whole --has_part---> GO part
```

## Hundred-million-edge candidate relation

The current BP export contains:

```text
549,722 proteins × 512 rare-GO candidates
= 281,457,664 backbone candidate edges
```

This full relation remains a formal NBS data asset. It is memory-mapped and
indexed on disk; it is never converted into one global PyG `HeteroData` object.
Only the candidate blocks required by one local episode are materialized.

## Required data preparation

### 1. GO→Protein and Protein→GO annotation indices

The already generated candidate and pseudo GO-major indices can be retained.
Gold supervision requires two views:

```text
GO -> Protein CSR       query/support and positive sampling
Protein -> GO CSR       local gold messages for sampled core proteins
```

To add gold indices without rebuilding the 281 million candidate relation:

```bash
export GOLD_EDGE_INDEX=/path/to/gold_protein_go_edge_index.i32.npy
export GOLD_ONLY=1
export MERGE_EXISTING=1
python scripts/nbs/run_build_go_protein_inverted_index.py
```

The existing manifest is updated atomically and candidate/pseudo entries are
preserved. Gold Protein→GO CSR is required to realize the intended:

```text
weak protein -> core protein -> GO
```

message route.

### 2. Direction-safe P–P sampling indices

```bash
python scripts/nbs/run_build_pp_sampling_indices.py
```

Sampling keys are relation-specific while returned messages retain original
orientation:

```text
ppi          keyed by source
similar_to   keyed by destination, message remains neighbour -> query
weak_to_core keyed by weak source, message remains weak -> core
```

### 3. Full BoxSquaredEL class space and G–G relations

The NBS ontology node space is the complete BoxSquaredEL checkpoint class map:

```text
44,919 classes
512-dimensional center
512-dimensional offset
```

It is not the BP/MF/CC classifier subset and is not rebuilt from `go.obo`
indices. Both geometry and G–G relation endpoints use the immutable
BoxSquaredEL checkpoint row.

```bash
python scripts/nbs/run_prepare_boxsqel_full_ontology_for_nbs.py
python scripts/nbs/run_build_boxsqel_gg_relations.py
```

Or run the three v0.4 preparation steps together:

```bash
python scripts/nbs/run_prepare_nbs_v04_training_inputs.py
```

The G–G builder parses the same normalized `go.norm` used to train
BoxSquaredEL, validates counts against its parser report, and emits:

```text
direct NF1 is_a / has_child message edges
direct NF4 part_of / has_part message edges
all NF1/NF2/NF3/NF4 arrays
role inclusion and role chain arrays
```

Closure is deliberately excluded from default GO message passing. Other
normal forms are retained for later relation-specific or hypergraph studies.

Task labels remain in the immutable first-stage classifier order. The existing
BP/MF/CC alignment manifests map each task column into the full 44,919-class
space, including canonical/alt-ID duplicate handling.

### 4. Audit

```bash
python scripts/nbs/run_audit_nbs_v04_training_inputs.py
```

The audit checks:

- candidate, gold and gold Protein→GO indices;
- task-label to full-ontology mapping;
- full box and G–G checkpoint consistency;
- P–P sampling key axes;
- fixed-epoch/no-validation policy;
- DDP-local step contract.

## Production train loader

The configured factory is:

```text
experiments.nbs.latence_nbs_loader:build_train_loader
```

It constructs a rank-sharded `LatenceNBSLocalBatchLoader`. Every episode:

1. samples task GO queries and episode-disjoint support proteins;
2. samples held-out gold positives, hard backbone candidates and optional
   pseudo positives from GO-major CSR;
3. retrieves base logits and three-column candidate evidence from mmap;
4. expands relation-specific P–P neighbourhoods with bounded fanout;
5. materializes gold annotations for sampled non-candidate proteins;
6. slices only local backbone candidate and optional pseudo edges;
7. maps task GO columns into the full BoxSquaredEL ontology;
8. samples direct G–G neighbourhoods;
9. converts global IDs to local PyG IDs;
10. removes candidate evidence in both Protein→GO and GO→Protein directions;
11. returns one `NBSLocalBatch`.

The full candidate relation never becomes a global PyG edge store.

### Local loader smoke test

Run one real batch before training:

```bash
python scripts/nbs/run_smoke_test_nbs_train_loader.py
```

It validates PyG structure, relation directions, query/candidate masking, local
node counts and the 44,919-class global GO graph.

## Single-node multi-GPU DDP

v0.4 uses one process per GPU with `DistributedDataParallel`; it does not use
`DataParallel`.

```bash
export NBS_NUM_GPUS=4
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

Equivalent explicit launch:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  scripts/nbs/train_nbs_fixed_epochs.py \
  --config nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.4.json
```

DDP contracts:

- every rank has the same number of local steps;
- global episode IDs are interleaved by rank, so rank samples are disjoint and
  reproducible;
- model initialization is identical before DDP synchronization;
- rank-specific randomness is applied only after model synchronization;
- `find_unused_parameters=true` is the conservative default for sparse
  relation activation;
- only rank 0 writes checkpoints, pointers, history and primary logs;
- epoch losses are reduced across ranks;
- RNG state is gathered per rank for checkpoint resume;
- the full GO cache is built after DDP parameter synchronization and refreshed
  after checkpoint restore.

A short production DDP smoke run can be launched without editing JSON:

```bash
export NBS_NUM_GPUS=2
export NBS_EPOCHS=1
export NBS_SAVE_EPOCHS=1
export NBS_MAX_STEPS_PER_EPOCH=1
export NBS_OUTPUT_DIR=outputs/latence_nbs_train/ddp_smoke
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

## Fixed-epoch training policy

Training does not use a validation set, best-checkpoint selection or early
stopping. It runs to the configured final epoch and saves fixed snapshots.

```json
{
  "epochs": 150,
  "save_epochs": [100, 150],
  "save_interval_epochs": 10,
  "validation_used": false,
  "early_stopping": false
}
```

The saved epochs are the union of:

```text
multiples of save_interval_epochs
+ explicit save_epochs
+ final epoch
```

To save every five epochs:

```bash
NBS_SAVE_INTERVAL_EPOCHS=5 \
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

Checkpoint metadata explicitly records fixed-epoch selection and distributed
world size. No `best.pt` is produced.

## Default BP stage

`configs/bp_fixed_epoch_v0.4.json` starts with a conservative gold-only stage:

```text
2 NBS layers
GO geometry frozen
pseudo loss disabled
pseudo message edges disabled
expert probability absent from forward
query hierarchy loss disabled initially
```

Candidate evidence uses:

```text
[backbone_probability, selector_score, reciprocal_rank]
```

The initial candidate encoder equals reciprocal-rank evidence and learns a
bounded residual from the other two fields. Final logits start exactly at the
first-stage base logits because graph delta scale is zero initialized.

## v0.6 weak-primary and full-task candidates

v0.6 changes the sampling unit without changing the immutable Stage-1 task GO
columns.  `weak_primary_exhaustive` owns a deterministic no-replacement queue
of active weak proteins on every DDP rank.  Each weak anchor induces one of its
modelout-positive GO labels, preferentially from the intersection with its
full-task top-512 backbone/selector edges; anchors sharing a GO can occupy the
same query.  Empty intersections fall back to a modelout-positive label, so an
unverified candidate edge is never promoted to a positive target.
Core gold support, hierarchy pairs and shuffled GO-floor queries are sampled
around these induced queries.  Only forced weak anchors consume the queue;
ordinary GO-major pseudo fillers do not count toward epoch completion.

Protein--GO candidate construction now defaults to `full_task`: the frozen
Stage-1 static prefilter and learned selector operate on every task GO column,
then retain top-512.  `rare_first` remains available for controlled ablation.
New manifests publish `backbone_candidate_edges` and record the selector scope;
readers retain compatibility with historical `backbone_rare_edges` manifests.

## v0.6.1 evaluation

v0.6.1 does not change v0.6.0 sampling, model weights or checkpoint semantics.
It upgrades the independent-test evaluator with sequence-identity, MSA-Neff,
Foldseek and EXP/IDA slices; candidate/query-channel reliability diagrams and
ECE; paired protein-bootstrap diagnostics; precision@k/recall@k after `k` is
predeclared; student–3B-expert error correlation; and a strict external-method
comparison manifest. See the project-root `NBS_EVALUATION_v0.6.1.md`.

## Leakage protection

The local materializer excludes candidate proteins from gold message lookup and
then applies relation-specific bidirectional masking:

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

Gold and pseudo relations are removed for candidate proteins. Backbone
candidate edges are removed only for the currently scored query GO set, so
unrelated first-stage functional context remains available.

## Independent-test inference

The v0.5.6.1 inference revision supports two external-protein modes:

- feature-only: the original external input projection with zero relation
  sources;
- isolated inductive P-P: retrieve core neighbours for each test protein and
  aggregate `core -> test` messages through the trained `similar_to` operator.

The latter never inserts test proteins into the training graph and never emits
test-to-test edges. `ExternalPPNeighborhoodStore` consumes role-local core
indices and the standard three-column P-P edge attributes. Full-task scoring
uses the frozen per-layer `similar_to` fanouts and applies the same trained
hierarchy-adapter projection used for proteins encoded inside the training
graph. It still visits every classifier column; query width and candidate top-K remain
mini-batch/sparse-evidence policies only.

The support-graph GO block size is frozen by the resolved inference config. It
is not exposed as an unconstrained memory-only knob because changing the block
changes the jointly sampled support graph.

See `NBS_INDUCTIVE_INFERENCE.md` at the project root for FASTA/pickle input,
cache, temporary deployment workspace, and diagnostic-evaluation commands.

## Tests

Pure PyTorch/NumPy tests:

```bash
cd nbs_models/nbs_protein_go
PYTHONPATH=. pytest -q tests \
  --ignore=tests/test_source_additivity.py \
  --ignore=tests/test_data_leakage.py
```

Real PyG tests on the LATENCE server:

```bash
PYTHONPATH=. pytest -q \
  tests/test_source_additivity.py \
  tests/test_data_leakage.py \
  tests/test_real_local_batch_v04.py
```

CPU two-process DDP framework smoke:

```bash
NBS_DDP_SMOKE_DIR=/tmp/nbs_ddp_smoke \
PYTHONPATH=. python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  examples/ddp_fixed_epoch_smoke.py
```

## Future GO versions

v0.4 leaves a versioned interface but does not run future-GO experiments yet.
For a later GO release:

```text
future go.obo
-> the same normalization pipeline
-> future go.norm
-> future BoxSquaredEL checkpoint and class map
-> full class-row box export
-> normalized G–G rebuild
-> task-label-to-future-class mapping
```

The protein universe and task classifier columns may remain fixed while the
ontology version changes. Manifests record ontology version and source hashes,
so future experiments can separate ontology evolution from new protein
annotations.
