# LATENCE NBS v0.5.5 quick notes

Use the new formal BP configuration:

```bash
export NBS_NUM_GPUS=2
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.5.5.json
export NBS_FRESH_START=1
unset NBS_MAX_STEPS_PER_EPOCH
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

Startup now prints two separate plans:

```text
NBS query coverage plan: ... eligible_go=...
NBS ontology context plan: task_labels=..., task_ontology_rows=...,
eligible_task_queries=..., context_only_task_rows=..., full_ontology=...,
non_task_context_rows=..., global_cache=..., cross_task_context_edges=...
```

The first line is the supervised task-query space.  The second is the ontology
context space.  They are intentionally different.

Epoch logs add:

```text
ont_local=.../44919(...)
ont_non_task=.../...(...)
task_ctx=.../...(%)
```

`qcov` is not an ontology coverage metric; it is direct supervised query
coverage.
