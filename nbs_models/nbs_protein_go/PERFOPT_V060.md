# LATENCE NBS v0.6.0 perfopt

Release name: `latence_nbs_v060_perfopt_20260821`

This package accelerates the existing BP weak-primary, full-task top-512
training path without changing its supervision or inference contract.  The
original `bp_fixed_epoch_v0.6.0.json` remains available; the optimized formal
configuration is:

`nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.6.0_perfopt.json`

The archive has one versioned top-level directory.  From the project root,
apply the packaged source snapshot with:

```bash
tar -xzf /path/to/latence_nbs_v060_perfopt_20260821.tar.gz \
  --strip-components=1 -C "$PWD"
```

If the server worktree contains uncommitted edits, save or diff them before
extracting because files included in the snapshot are replaced in place.

## Preserved contracts

- Protein-GO top-512 remains sparse graph evidence, not an output-vocabulary
  restriction.  BP inference still produces all 21,312 task probabilities.
- Weak pseudo positives still require `modelout probability > 0.5`.
- Values at or below the threshold are not pseudo negatives; unknown pairs
  remain masked.
- Full sampled gold, hard, pseudo and weak-primary supervision retention stays
  mandatory (`max_candidates=3136`, `require_full_supervision_retention=true`).
- Expert probability is not introduced as an NBS forward input.

## Performance changes

1. `WeightedSAGEConv` projects each source node once for dense edge relations,
   then gathers the projected states by edge.  Sparse relations retain the
   original edge-first path.  This is mathematically equivalent to projecting
   every repeated source occurrence.
2. Fixed-degree Protein-GO mmap gather is vectorized across proteins.
3. Candidate canonicalization deduplicates inside known source-major blocks,
   with an automatic fallback to the generic global implementation if the
   fixed-degree invariant is not satisfied.
4. `prefetch_batches=1` materializes batch n+1 on one CPU worker while batch n
   is running on the accelerator.  Sampling remains serialized and preserves
   episode order and seed derivation.
5. Training history and the epoch summary now expose CPU phase timings:
   `cpu_mat`, `cpu_cand`, `cpu_loc`, and `cpu_graph`, with the full fields kept
   as `avg_*_seconds` in `training_history.json`.

The source projection reordering is algebraically identical but, as with any
change in GEMM shape under BF16, last-bit floating-point differences are
possible.  Sampling, supervision arrays, edge selection and loss definitions
are unchanged.

## Server regression tests

```bash
/opt/conda/envs/pytorch2.4/bin/python -m pytest -q \
  nbs_models/nbs_protein_go/tests/test_hybrid_weak_focus_v054.py \
  nbs_models/nbs_protein_go/tests/test_inverted_index.py \
  nbs_models/nbs_protein_go/tests/test_ddp_episode_sharding_v04.py \
  nbs_models/nbs_protein_go/tests/test_performance_paths_v060.py \
  scripts/test_export_weak_graph_predictions.py
```

Then materialize a real optimized batch:

```bash
/opt/conda/envs/pytorch2.4/bin/python \
  scripts/nbs/smoke_test_nbs_train_loader.py \
  --project-root "$PWD" \
  --config nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.6.0_perfopt.json \
  --epoch 1 \
  --output outputs/nbs_v060_perfopt_loader_smoke.json
```

## Recommended 20-step comparison probe

Run this only as a timing/invariant probe, using a new output directory:

```bash
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.6.0_perfopt.json
export NBS_NUM_GPUS=2
export NBS_EPOCHS=1
export NBS_MAX_STEPS_PER_EPOCH=20
export NBS_WEAK_PRIMARY_PROTEINS_PER_EPISODE=96
export NBS_WEAK_PRIMARY_QUERY_SOURCE=pseudo_candidate_intersection
export NBS_PREFETCH_BATCHES=1
export NBS_FRESH_START=1
export NBS_OUTPUT_DIR=outputs/latence_nbs_train/bp_nbs_v060_perfopt_20step_probe
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

Compare `elapsed_seconds / steps_per_rank` with the previous 20-step probe.
The following correctness indicators must remain unchanged or valid:

- `cand_keep=100%`, `hard_keep=100%`, `wft_keep=100%`;
- `rowpart=ok`, `wpq=100%`, `wft_fill=100%`;
- `candidate_dropped_by_cap=0` and `full_supervision_retained=true`;
- no reduction of candidate, pseudo or gold edge counts caused by the optimized
  gather/dedupe paths.

Use `cpu_mat`, `cpu_cand`, `cpu_loc`, and `cpu_graph` to identify any remaining
CPU bottleneck.  Because prefetch overlaps CPU and GPU work, `cpu_mat` is a
cost measurement, not necessarily time added directly to each visible step.

## Formal 20-epoch run

Only start the formal run after the regression suite, real-batch smoke test and
20-step comparison probe pass:

```bash
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.6.0_perfopt.json
export NBS_NUM_GPUS=2
export NBS_EPOCHS=20
export NBS_SAVE_INTERVAL_EPOCHS=1
export NBS_WEAK_PRIMARY_PROTEINS_PER_EPISODE=96
export NBS_WEAK_PRIMARY_QUERY_SOURCE=pseudo_candidate_intersection
export NBS_PREFETCH_BATCHES=1
export NBS_FRESH_START=1
export NBS_OUTPUT_DIR=outputs/latence_nbs_train/bp_nbs_v060_perfopt_formal
unset NBS_MAX_STEPS_PER_EPOCH
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

The startup plan must report `steps/rank=2549` for the current two-rank data
state.  If it reports `optimizer_steps/epoch=20`, a stale
`NBS_MAX_STEPS_PER_EPOCH` is still active and the run is only a probe.

For a prefetch-only rollback, set `NBS_PREFETCH_BATCHES=0`.  To return to the
complete baseline, point `NBS_TRAIN_CONFIG` back to
`bp_fixed_epoch_v0.6.0.json` and use the corresponding baseline source tree.
