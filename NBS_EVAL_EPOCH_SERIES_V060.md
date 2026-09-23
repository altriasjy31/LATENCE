# NBS v0.6.0 checkpoint-series independent-test evaluation

This wrapper evaluates a predeclared checkpoint series with one immutable
inductive-input cache.  It does not change model, sampling, scoring or metric
semantics.

## Required contract

Use the exact Stage-1 checkpoint, metadata and MSA index used by the NBS data
artifacts.  Formal runs must keep both external test-to-core P-P messages and
top-512 candidate evidence enabled.

```bash
export TASK=bp
export NBS_CHECKPOINT_DIR=outputs/latence_nbs_train/bp_nbs_v060_perfopt_tailfix_fullrestore_formal
export NBS_EVAL_EPOCHS=1,2,3,4
export NBS_EVAL_SERIES_ROOT=outputs/latence_nbs_eval/bp_nbs_v060_perfopt_epoch1_4

# The wrapper automatically prefers NBS_CHECKPOINT_DIR/resolved_config.json.
# Set this only when an explicit path is required.
export NBS_TRAIN_CONFIG="$NBS_CHECKPOINT_DIR/resolved_config.json"

export STAGE1_CHECKPOINT=/absolute/path/to/weak_detr_decoder_epoch100.pt
export METADATA_FILE=data/unidata_with_exp_train_pseudo.pkl
export STAGE1_MSA_INDEX=data/ind_MSA_bin/index.pkl

export NBS_EVAL_DEVICE=cuda:0
export STAGE1_EVAL_DEVICE=cuda:0
export NBS_EVAL_MIN_FREE_GPU_GB=30
export STAGE1_EVAL_MIN_FREE_GPU_GB=30

export NBS_EVAL_USE_EXTERNAL_PP=1
export NBS_EVAL_USE_CANDIDATE_EVIDENCE=1
export NBS_SAVE_INFERENCE_DIAGNOSTICS=1
export NBS_METRIC_BACKEND=stage1
export NBS_EVAL_BOOTSTRAP_REPLICATES=1000
export NBS_EVAL_PRECISION_K=10,50,100

python scripts/nbs/run_eval_nbs_checkpoint_series.py
```

Do not set `NBS_EVAL_LIMIT_PROTEINS` for a formal evaluation.  A smoke run is a
separate protocol and must use `NBS_RUN_EVALUATION=0`.

If training occupies every GPU, wait for a clean epoch checkpoint and stop the
training process before evaluation.  The GPU preflight intentionally fails
rather than competing with training and producing an OOM.

## Shared cache and outputs

The first epoch creates or validates:

```text
<series-root>/_shared_inductive_inputs/
```

Later epochs use `cache-policy=require`; any Stage-1 checkpoint, protein order,
candidate contract, MSA index or P-P retrieval change becomes a hard error.

Each epoch writes:

```text
epoch1/predictions/nbs_full_task_prob.f16.npy
epoch1/predictions/nbs_applied_logit_delta.f16.npy
epoch1/predictions/nbs_delta_gate.f16.npy
epoch1/predictions/nbs_routing_source_weights.f16.npy
epoch1/predictions/nbs_routing_null_weight.f16.npy
epoch1/nbs_ind_test_metrics.json
epoch1/nbs_primary_comparison.tsv
```

The series root writes:

```text
nbs_epoch_series_protocol.json
nbs_epoch_series_primary.tsv
nbs_epoch_series_diagnostics.tsv
nbs_epoch_series_summary.json
```

`nbs_epoch_series_primary.tsv` is the first table to inspect.  It contains the
official Stage-1 metric backend for NBS and the common Stage-1 backbone, plus
NBS-minus-backbone deltas.  The diagnostics table retains candidate/non-
candidate, eligible/context-only, rare-frequency, calibration, applied-delta,
delta-gate and routing summaries. It also reads each checkpoint's stored
training metrics and reports raw/`tanh` values for `graph_delta_scale`,
`query_scale`, `context_scale`, `candidate_evidence_scale`, plus `null_logit`.

The wrapper never identifies a best epoch.  If this independent test set is
used to redesign the model or choose a checkpoint, it has become a development
set; a fresh locked temporal/independent set is then required for the final
claim.
