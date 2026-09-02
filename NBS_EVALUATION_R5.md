# NBS v0.5.6 completed-training evaluation plan (inductive-r5)

## 1. What r5 fixes

The r3 launcher/exporter enabled isolated independent-test P--P inference, but
the small r3 patch archive did not contain the matching `nbs_pg/inference.py`
and `nbs_pg/model.py`. Applying only that archive to an older project produced
exactly this mixed deployment:

```
new exporter + old inference/model
```

The resulting `ExternalPPNeighborhoodStore` import failure is independent of
the checkpoint, input arrays, and CUDA memory. r4 shipped the missing modules,
but its first successful forward exposed a second, independent defect:
`condition.seed_index` belongs to the sampled training-support graph, whereas
the external scoring hierarchy contains only the current independent-test
protein batch. Treating these as one protein index space produced
`seed_index contains indices outside [0, 2)` in the two-protein smoke test.

r5 keeps two explicit hierarchies in the matcher:

- the sampled training-support hierarchy supplies support-seed contexts and
  determines query/source routing weights;
- the isolated external-protein hierarchy supplies candidate contexts and is
  the object actually scored.

This is not an index clipping fix: clipping or remapping support indices into
the two external rows would bind GO queries to arbitrary test proteins and
make results depend on test-batch composition. r5 ships `inference.py`,
`model.py`, `matcher.py`, exporter and launcher as one contract and checks API
version 5 before Stage-1 preparation. No checkpoint parameters are added; the
trained `similar_to` layers are reused for isolated `core -> test` messages, so
the existing epoch-150 checkpoint remains strictly loadable.

r5 retains r4's separation of metric meanings:

- primary overall Fmax/AUPRC: the exact project Stage-1
  `msa_models/helper_functions/helper.py::evalperf_torch` backend;
- masked/candidate/delta diagnostics: explicitly named flattened-position
  micro metrics. These are useful for attribution but are not substituted for
  the official Stage-1 metric in the primary comparison.

## 2. Frozen primary comparison

Register the comparison before reading independent-test results:

- primary NBS checkpoint: `nbs_epoch150.pt` (the fixed training endpoint);
- primary baseline: the Stage-1 `backbone_ind_test_prob.f16.npy` generated from
  the same ordered proteins and Stage-1 checkpoint;
- primary input condition: original independent-test MSA binary when available;
  singleton FASTA-MSA is a separate deployment condition;
- primary inference: external P--P on, candidate evidence on;
- primary metric backend: `stage1`;
- full task space: BP 21,312 columns, not candidate-only columns;
- expert/modelout probability: never an NBS forward input. If supplied to the
  evaluator it is a non-deployable reference only.

Do not choose epoch 150 because it scores best on `ind_test`. It is primary
because it is the predeclared end of the fixed-epoch training run. Earlier
checkpoints may be reported as a descriptive stability curve, but must not be
used to select a winner on this test set.

## 3. Install and API preflight

Extract the r5 patch at the project root, preserving paths. The first launcher
line expected after environment validation is:

```
[NBS inference API preflight] version=5 status=ok
```

The export process repeats the check before CUDA and prints:

```
[NBS inference API] version=5 status=ok
```

If either check fails, stop. Do not remove `ExternalPPNeighborhoodStore` and do
not use a feature-only run as the formal replacement.

## 4. Two-protein end-to-end smoke test

Use a new smoke workspace. In particular, do not reuse the earlier
`bp_epoch30/inductive_inputs` directory because its manifest may describe only
the two limited rows.

For the specific r4 run that reached the matcher error, the new
`bp_epoch150_smoke2/inductive_inputs` artifacts are already complete and valid.
After installing r5, change `NBS_INPUT_CACHE_POLICY` from `refresh` to `require`
and rerun the launcher. This skips the successful Stage-1 preparation while
strictly checking its cache signature. The exporter opens its outputs with
overwrite mode, so the partial r4 prediction arrays do not need manual deletion.

```bash
export TASK=bp
export LATENCE_PROJECT_ROOT=/home/dataset-local/data_local/shaojiangyi/latence-project
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.5.6.json
export NBS_CHECKPOINT=/home/dataset-assist-0/datafile/latence-dataset/outputs/latence_nbs_train/bp_nbs_v056_formal/nbs_epoch150.pt
export STAGE1_CHECKPOINT=/absolute/path/to/the/frozen/stage1/checkpoint.pt
export METADATA_FILE=$LATENCE_PROJECT_ROOT/data/unidata_with_exp_train_pseudo.pkl
export NBS_DATA_ROOT=/home/dataset-assist-0/datafile/latence-dataset/outputs/latence_nbs/bp_weak_detr_v3_expert_prob_warmstart340_to400/epoch100/bp

export NBS_EVAL_OUTPUT_DIR=/home/dataset-assist-0/datafile/latence-dataset/outputs/latence_nbs_eval/bp_epoch150_smoke2
export NBS_IND_TEST_WORK_DIR=$NBS_EVAL_OUTPUT_DIR/inductive_inputs
export NBS_PRED_OUTPUT_DIR=$NBS_EVAL_OUTPUT_DIR/predictions
export NBS_INPUT_CACHE_POLICY=refresh
export NBS_RUN_EVALUATION=0
export NBS_EVAL_LIMIT_PROTEINS=2
export NBS_SAVE_INFERENCE_DIAGNOSTICS=1
export NBS_EVAL_USE_EXTERNAL_PP=1
export NBS_EVAL_USE_CANDIDATE_EVIDENCE=1

python scripts/nbs/run_eval_nbs_ind_test_predictions.py
```

Smoke acceptance criteria:

1. both API checks report version 5;
2. `nbs_full_task_prob.f16.npy` has shape `[2, 21312]`, finite values, and
   probabilities in `[0,1]`;
3. the prediction manifest reports
   `inference_mode=inductive_pp_feature_candidate`,
   `external_pp.enabled=true`, `test_to_test_edges=false`,
   `candidate_evidence=sparse_fixed_k`, and
   `uses_expert_probability_in_nbs_forward=false`;
   `routing_contract.index_spaces_separate=true`;
4. diagnostic arrays have the same two protein rows and no NaN/Inf.

As a numerical batching check, repeat the same smoke prediction with
`NBS_EVAL_PROTEIN_BATCH=1` in a second prediction directory and compare it with
batch size 2. Since outputs are stored as float16, the expected maximum stored
probability difference is zero or at most one float16 rounding unit. A larger
or structured difference indicates cross-protein interaction or batching drift.

## 5. Full epoch-150 run

Remove the limit and use a dedicated full input cache. On its first construction
use `refresh`; later checkpoint/ablation runs should use `require` so a mismatch
fails rather than silently regenerating inputs.

```bash
unset NBS_EVAL_LIMIT_PROTEINS
export NBS_RUN_EVALUATION=1
export NBS_METRIC_BACKEND=stage1
export NBS_AUPRC_MODE=hist
export NBS_EVAL_USE_EXTERNAL_PP=1
export NBS_EVAL_USE_CANDIDATE_EVIDENCE=1

export NBS_IND_TEST_WORK_DIR=/home/dataset-assist-0/datafile/latence-dataset/outputs/latence_nbs_eval/bp_shared_full_inputs
export NBS_INPUT_CACHE_POLICY=refresh
export NBS_EVAL_OUTPUT_DIR=/home/dataset-assist-0/datafile/latence-dataset/outputs/latence_nbs_eval/bp_epoch150_full
export NBS_PRED_OUTPUT_DIR=$NBS_EVAL_OUTPUT_DIR/predictions

python scripts/nbs/run_eval_nbs_ind_test_predictions.py
```

After the first successful full run, set:

```bash
export NBS_INPUT_CACHE_POLICY=require
```

The full-run acceptance contract is:

- input, output, and metadata protein ID hashes agree;
- expected full independent-test row count is present (1,800 for the current BP
  split, if that is the frozen metadata version);
- prediction width is 21,312 and every column was scored;
- checkpoint, GO registry, representation manifest, weak-graph manifest, and
  input manifest hashes pass;
- no independent-test label enters representation, candidate selection,
  P--P retrieval, graph construction, or forward inference;
- all probabilities and diagnostics are finite.

## 6. What to report

The primary table uses `nbs_ind_test_metrics.json -> metrics`:

| Row | Role | Interpretation |
|---|---|---|
| `backbone_base` | primary baseline | Stage-1 probability before NBS refinement |
| `NBS_final` | primary model | epoch 150, P--P on, candidate evidence on |
| `modelout_reference` | optional reference | may contain expert information; never call it a deployable baseline |

Report absolute deltas `NBS_final - backbone_base` for official Fmax and AUPRC,
including the official threshold if returned by `evalperf_torch`.

Then report diagnostic slices, always with their micro-metric label:

- train-frequency bins: zero, rare, medium, common, `rare_le_5`, `common_ge_50`;
- candidate versus non-candidate positions;
- eligible direct-query versus context-only GO columns;
- probability delta and applied logit-delta on positive versus unlabelled
  positions;
- delta-gate, routing source weights, and null-routing weight.

Interpretation checks:

- an overall gain with a large rare-class loss suggests overcorrection toward
  common labels;
- gains confined to candidate positions suggest the candidate evidence channel
  dominates and full-task generalization is weak;
- nearly zero applied logit delta/gate means NBS has collapsed toward the
  backbone even if routing looks nonzero;
- large signed deltas on unlabelled positions with worse AUPRC suggest excessive
  graph propagation;
- near-unit null routing across most GO terms suggests support/context is being
  ignored; near-zero null weight with unstable deltas suggests overreliance on
  graph sources.

## 7. Post-hoc attribution runs

Use the same full input directory with `NBS_INPUT_CACHE_POLICY=require`, but a
different evaluation and prediction directory for every run:

| Run | `NBS_EVAL_USE_EXTERNAL_PP` | `NBS_EVAL_USE_CANDIDATE_EVIDENCE` | Question |
|---|---:|---:|---|
| full | 1 | 1 | predeclared primary result |
| no-PP | 0 | 1 | contribution of isolated core-neighbour messages |
| no-candidate | 1 | 0 | contribution of sparse Stage-1 candidate evidence |
| feature-only | 0 | 0 | residual NBS behavior from representation/base/GO support only |

These are attribution diagnostics, not replacement trained-model ablations:
channels were removed only at inference time. If a channel-removal result is
important enough for a causal architecture claim, retrain the corresponding
model with that channel absent.

## 8. Optional checkpoint stability curve

After the epoch-150 result is frozen, evaluate a small predeclared set such as
epochs 30, 60, 100, 120, and 150 with the same `require`d full input cache and
separate output directories. Use this only to diagnose convergence and late
drift. Do not select the maximum-scoring epoch as the reported model unless a
separate validation set—not `ind_test`—defined that selection rule in advance.
