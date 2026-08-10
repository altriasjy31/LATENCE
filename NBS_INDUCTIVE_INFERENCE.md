# NBS independent-test inference and diagnostics

This revision adds a complete, leakage-safe independent-test path without
changing the v0.5.6 scientific training configuration or checkpoint weights.

## Input modes

The preparation program accepts either:

1. a project pickle containing ordered `ind_test` protein IDs. If sequences are
   absent, the configured Stage-1 `working_address` / `--msa-index` is read and
   the original MSA binary is used;
2. a FASTA file. Each sequence is encoded as a query-only singleton MSA and
   padded to the original Stage-1 `top_k 脳 max_len` input shape;
3. FASTA plus pickle. The pickle defines evaluation order and FASTA sequences
   are aligned strictly by protein ID.

For the formal independent-test experiment, prefer the original MSA-binary
mode when those alignments exist. Singleton-MSA is a deployment fallback and
must be reported as a distinct input-information condition.

Labels in the pickle are never passed to Stage-1, candidate selection, P-P
retrieval, NBS graph construction, or NBS forward. They are read only by the
final evaluator.

## End-to-end launcher

```bash
export TASK=bp
export LATENCE_PROJECT_ROOT=/path/to/latence-project
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.5.6.json
export NBS_CHECKPOINT=outputs/latence_nbs_train/bp_nbs_v056_formal/nbs_epoch100.pt
export STAGE1_CHECKPOINT=outputs/weak_exp_train_detr/.../weak_detr_decoder_epoch100.pt
export METADATA_FILE=data/unidata_with_exp_train_pseudo.pkl
export NBS_DATA_ROOT=outputs/latence_nbs/.../epoch100/bp
export NBS_EVAL_OUTPUT_DIR=outputs/latence_nbs_eval/bp_epoch100

python scripts/nbs/run_eval_nbs_ind_test_predictions.py
```

For FASTA-only inference, also set:

```bash
export NBS_IND_TEST_FASTA=/path/to/query.fasta
```

With `NBS_RUN_EVALUATION=0`, an arbitrary service FASTA is not forced to match
the project's default `ind_test` metadata. Set `METADATA_FILE` explicitly only
when its protein IDs should be used as an alignment contract.

For a relocated MSA binary or Stage-1 model config:

```bash
export STAGE1_MSA_INDEX=/path/to/ind_MSA_bin/index.pkl
export STAGE1_MODEL_CONFIG=/path/to/model_config.pkl
```

The default experiment workspace is
`$NBS_EVAL_OUTPUT_DIR/inductive_inputs`. Its manifest is checked before Stage-1
inference; matching artifacts are reused only after shape and SHA-256 checks,
while a checkpoint, registry, input order, candidate-K, or P-P-K change
invalidates the cache.

## Temporary deployment workspace

For service-style requests:

```bash
export NBS_USE_TMP_WORKSPACE=1
export NBS_TMP_ROOT=/tmp
export NBS_RUN_EVALUATION=0
python scripts/nbs/run_eval_nbs_ind_test_predictions.py
```

The prepared representation, base probability, sparse candidates and P-P
neighbours are stored in a unique temporary directory and removed after the
final full-task prediction is written. Set `NBS_KEEP_TMP_WORKSPACE=1` only for
debugging. A fixed reusable workspace can be selected with
`NBS_IND_TEST_WORK_DIR`.

## Generated inductive inputs

`prepare_nbs_ind_test_inputs.py` writes:

- `protein_ids.txt`;
- `ind_test_repr.f16.npy`, `[N,2048]` for the current BP model;
- `backbone_ind_test_prob.f16.npy`, `[N,num_task_GO]`;
- `candidate_go_index.i32.npy`, `[N,K_candidate]`;
- `candidate_edge_attr.f32.npy`, containing backbone probability, trained
  selector score and reciprocal rank;
- `test_core_neighbors.i32.npy`, `[N,K_pp]` in the role-local core space;
- `test_core_edge_attr.f32.npy`, containing confidence, cosine score and
  reciprocal rank;
- `ind_test_input_manifest.json` with ordered-ID, checkpoint, registry and
  leakage contracts.

The Stage-1 checkpoint must match both the representation manifest and the
weak-graph manifest used for NBS training. A mismatch is a hard failure.

## External P-P semantics

FAISS retrieval is `test -> core`, but message passing is `core -> test` through
the already-trained `similar_to` relation operator. Test proteins are encoded
independently: no test-to-test edge or batch-level interaction is created.
The number of neighbours consumed at each layer comes from the frozen
`local_sampling.similar_to_fanouts` training configuration (8 then 4 for the
current BP v0.5.6 run); the preparation stage stores their maximum.

Two inference modes remain available:

- `inductive_pp_feature_candidate` (formal default): representation +
  student-only candidate evidence + isolated core-neighbour messages;
- `inductive_feature_candidate`: the v0.5.3-compatible feature-only ablation,
  with all external relation-source residuals set to zero.

`go_chunk_size` is frozen to the resolved config value (256 for the current BP
run). It defines a shared sampled support-graph block and is therefore a
scientific inference parameter, not a free memory knob. The exporter rejects a
different CLI value instead of silently changing predictions. Protein batch
size remains a throughput setting and must not change a protein's result.

## Evaluation output

`nbs_ind_test_metrics.json` reports:

- overall Fmax and micro-AUPRC for `NBS_final`, `backbone_base`, and optional
  `modelout_reference`;
- zero/rare/medium/common, `rare_le_5`, and `common_ge_50` results using the
  exact Stage-1 training-count bin definition;
- candidate versus non-candidate positions;
- eligible direct-query versus context-only GO columns;
- probability delta and applied logit-delta distributions, including sign and
  quantiles, split by true labels, frequency, candidate status and query scope;
- delta-gate, routing source-weight and null-weight summaries;
- protein-order and manifest alignment contracts.

`modelout_reference` may contain expert information and is never an admissible
NBS forward input. It remains a strong reference only.