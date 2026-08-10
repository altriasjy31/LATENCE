# NBS v0.5.4 deployment note

This is an incremental update over v0.5.3.

## Recommended 100-step probe

```bash
export NBS_NUM_GPUS=2
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.5.4.json
export NBS_FRESH_START=1
export NBS_EPOCHS=1
export NBS_SAVE_EPOCHS=1
export NBS_MAX_STEPS_PER_EPOCH=100
export NBS_OUTPUT_DIR=outputs/latence_nbs_train/bp_v054_hybrid_weak_probe
unset NBS_RESUME

python scripts/nbs/run_train_nbs_fixed_epochs.py
```

Inspect:

```text
wfq, wfa, wft, wft_fill, wft_keep
pweak, pweak_all, puniq_eff
qcov, hard_keep, cand_keep
peak_mem, elapsed
```

## Full hybrid epoch

A probe checkpoint must not be resumed into a full OneCycle run because the scheduler total-step contract differs.

```bash
export NBS_FRESH_START=1
unset NBS_MAX_STEPS_PER_EPOCH
export NBS_OUTPUT_DIR=outputs/latence_nbs_train/bp_v054_hybrid_weak_full_epoch

python scripts/nbs/run_train_nbs_fixed_epochs.py
```

The startup plan prints the exact GO, weak and core step requirements. The resolved step count depends on the actual number of pseudo-eligible active weak proteins.

## Formal 150-epoch run

After the probe and full-epoch diagnostics pass:

```bash
export NBS_FRESH_START=1
unset NBS_EPOCHS
unset NBS_SAVE_EPOCHS
unset NBS_MAX_STEPS_PER_EPOCH
export NBS_OUTPUT_DIR=outputs/latence_nbs_train/bp_nbs_v054_formal

python scripts/nbs/run_train_nbs_fixed_epochs.py
```

Do not resume a v0.5.3 checkpoint: the episode layout, epoch length and OneCycle total-step contract changed.
