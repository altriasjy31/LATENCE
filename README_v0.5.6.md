# NBS v0.5.6 quick usage

v0.5.6 adds conservative background-PU supervision without changing the graph
or candidate union.

Formal BP defaults:

```text
Q=64
C=1536
hard PU/query=16, weight=0.2
background PU/query=16, weight=0.05, base_prob<=0.05
pseudo/query=32
```

Recommended final smoke before a 150-epoch run:

```bash
export NBS_NUM_GPUS=2
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.5.6.json
export NBS_FRESH_START=1
export NBS_EPOCHS=1
export NBS_SAVE_EPOCHS=1
export NBS_MAX_STEPS_PER_EPOCH=327
export NBS_OUTPUT_DIR=outputs/latence_nbs_train/bp_v056_background_pu_probe
unset NBS_RESUME
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

Primary acceptance diagnostics:

```text
negrows ~ 100%
posonly ~ 0%
bg_fill high (preferably >=80-90%)
qcov ~ 100%
pweak remains ~90%+
hard_keep >=95%
cand_keep >=95%
```

Do not interpret background PU as biological ground-truth negatives. Unknown
pairs outside the selected low-base background remain masked and contribute no
ASL gradient.
