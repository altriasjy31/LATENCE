# NBS v0.5.5 — ontology-space diagnostics and calibrated hybrid epoch planner

## Why this release exists

Training logs reported `eligible_go=18300` for BP while the immutable first-stage
classifier contains 21,312 columns and the full BoxSquaredEL ontology contains
44,919 class rows.  These are three different index spaces:

1. task output labels (BP: 21,312 classifier columns),
2. directly supervised NBS task queries (BP: 18,300 under the current strict
   singleton/pseudo contract), and
3. full BoxSquaredEL ontology context (44,919 rows).

v0.5.5 makes that distinction explicit and audits actual full-ontology exposure.

## Supervised-query eligibility diagnostics

`GOQueryEpisodeSampler.eligibility_summary` now records:

- total task labels;
- zero-gold labels;
- singleton labels with/without pseudo support;
- total ineligible labels and the exact exclusion reason.

For the current BP data, 21,312 - 18,300 = 3,012, matching the 3,012 singleton
terms that lack a `modelout > 0.5` weak pseudo-positive.

## Full ontology context diagnostics

The train loader now explicitly distinguishes:

- task classifier columns;
- unique task-mapped ontology rows;
- eligible query ontology rows;
- context-only task ontology rows;
- full BoxSquaredEL ontology rows;
- ontology rows not mapped by the current task.

It also records direct G-G edge partitions (`task->task`, `task->non-task`,
`non-task->task`, `non-task->non-task`) for each relation.  Canonical
`is_a + part_of` task/non-task cross edges are reported at startup.

The full GO cache still contains every BoxSquaredEL row.  Local sampled graphs
continue to expand direct `is_a/has_child/part_of/has_part` neighbours in the
full ontology row space.

## Epoch-level ontology exposure

Every local batch now exposes transient global ontology row IDs for diagnostics.
DDP merges compact bitmaps and reports:

- `ont_local`: unique full-ontology rows that entered sampled local graphs;
- `ont_non_task`: unique ontology rows outside the task vocabulary that entered
  local graphs;
- `task_ctx`: task ontology rows excluded from direct supervised querying but
  still observed as graph context.

These metrics are separate from `qcov`, which remains supervised task-query
coverage only.

## Hybrid epoch planner correction

v0.5.4 estimated weak coverage from the explicit weak-focus block only and
resolved ~558 BP steps.  Real runs showed that ordinary coverage/hierarchy
queries contribute a comparable amount of pseudo supervision.

v0.5.5 estimates pseudo capacity from all query slots:

- mean baseline pseudo positives per eligible GO;
- non-focus query count;
- explicit weak-focus target capacity;
- DDP world size;
- conservative unique-weak efficiency.

With the supplied BP statistics (`Q=64`, 2 GPUs, 70% weak target, efficiency
0.60), the planner resolves:

- GO floor: 286 steps;
- weak unique target: ~322 steps;
- core exposure: 327 steps;
- final epoch length: 327 steps/rank.

This matches the empirical 327-step probe much better than the old 558-step
estimate.

## Compatibility

The older `resolve_hybrid_epoch_requirements` call contract remains accepted;
when the new all-query pseudo-capacity inputs are absent, it falls back to the
v0.5.4 weak-focus-only planner.
