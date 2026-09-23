# LATENCE NBS v0.6.0 perfopt tail-fix full restore

Release name: `latence_nbs_v060_perfopt_tailfix_fullrestore_20260901`

This archive is a complete, internally consistent source restore based on the
validated `latence_nbs_v060_perfopt_20260821` snapshot plus the R2 weak-primary
tail-exhaustion patch.

Use it when a previous overlay left `nbs_pg/__init__.py`, `training.py`, the
training launcher or JSON configurations at incompatible revisions.  In
particular, this snapshot restores all OneCycle scheduler symbols required by
the package and launcher:

- `NBSSchedulerConfig`
- `build_nbs_scheduler`
- `resolve_scheduler_step_plan`
- `validate_scheduler_resume_contract`

It also retains the v0.6.0 perfopt paths and the weak-primary tail fix.  It
does not include datasets, generated indices, outputs or checkpoints.

## Install

From the project root:

```bash
tar -xzf /path/to/latence_nbs_v060_perfopt_tailfix_fullrestore_20260901.tar.gz \
  --strip-components=1 -C "$PWD"
```

Use the matching base perfopt JSON.  Do not use a later tqdmopt JSON containing
`progress_postfix_mode` with this restored baseline:

```bash
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.6.0_perfopt.json
```

The progress bar uses the validated baseline single-line presentation.  Model,
sampling, optimizer and scheduler semantics remain those of the perfopt run.

## Preflight

```bash
/opt/conda/envs/pytorch2.4/bin/python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, str(Path("nbs_models/nbs_protein_go").resolve()))
from nbs_pg.training import (
    NBSSchedulerConfig,
    NBSFixedEpochTrainingConfig,
    build_nbs_scheduler,
    resolve_scheduler_step_plan,
)
print("NBS training imports: OK")
PY
```

Then validate the selected JSON without constructing the model/data:

```bash
export NBS_VALIDATE_CONFIG_ONLY=1
python scripts/nbs/run_train_nbs_fixed_epochs.py
unset NBS_VALIDATE_CONFIG_ONLY
```

The formal two-GPU startup must report:

```text
num_gpus=2
steps/rank=2549
optimizer_steps/epoch=2549
total_optimizer_steps=50980
```
