# LATENCE Dark–Weak Stage-1 task normalization hotfix v0.5.1

## Scope

**Only** `scripts/dark_weak/export_stage1_predictions.py` is modified. No Stage-1 model,
NBS v094 code, native metrics, checkpoint, graph cache, or Dark–Weak labels are touched.
The new test file is additive.

The actual `experiments.exp_train.normalize_task('bp')` returns `biological_process`.
Previously, `export()` indexed a shorthand-only width dict using this long name,
causing `KeyError: 'biological_process'` at line 54. A second metadata expression
would have failed after merely changing the width dict.

The fix keeps **two distinct names**:
- `task_short = 'bp'` for GO width (21312) and `prediction_manifest.json.task`;
- `task = 'biological_process'` for metadata, `MSABinaryDataset`, `PseudoProbDataset`,
  and the existing Stage-1 option builder.

It adds `prediction_manifest.json.task_namespace = 'biological_process'` for traceability.
The GO columns, protein ordering, query decoder, probabilities, weak-label policy,
and native v085 metric code are unchanged.

## Install on server from LATENCE project root

```bash
cp scripts/dark_weak/export_stage1_predictions.py \
   scripts/dark_weak/export_stage1_predictions.py.before_v051.bak
unzip -o /PATH/TO/LATENCE_stage1_dark_weak_task_fix_v0.5.1.zip -d .
python -m py_compile scripts/dark_weak/export_stage1_predictions.py
python -m unittest discover -s tests -p 'test_dark_weak_stage1_task_alias_v051.py' -v
```

The patch archive has **no** versioned training files or native evaluators.
If the server exporter has been independently modified since v094, inspect
`CHANGE_stage1_task_alias_v051.diff` and apply only its small task-alias hunks.

## Rerun

Use your previous Stage-1 export command without changing `--task bp` or the
frozen checkpoint and existing metadata/Expert inputs. The prior failure occurs
before output files are written. If the output directory already contains files
from a later failed attempt, use a NEW versioned `--output_dir`, or inspect and
explicitly opt for `--overwrite`.

## Validation

The added regression tests verify BP/MF/CC alias resolution against the actual
`experiments.exp_train.TASKS` dictionary, metadata/MSA full namespaces, and an
in-memory fake Stage-1 full-axis export (21312 GO) with a deliberately permuted
loader order. They do not claim to validate real GPU checkpoint inference.
