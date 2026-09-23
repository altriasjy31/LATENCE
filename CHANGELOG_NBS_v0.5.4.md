# LATENCE NBS v0.5.4 — Hybrid GO coverage and weak-first supervision

## Motivation

The v0.5.3 `go_cyclic_unique` sampler substantially improved weak-protein coverage:

- 100 synchronized steps: 74,250 unique weak pseudo targets (15.2% of all weak proteins).
- 200 synchronized steps: 120,790 unique weak pseudo targets (24.7%).

However, unique efficiency decreased as training progressed because GO-major scheduling still spent many query slots on GO terms whose pseudo pools overlapped the same multi-label weak proteins. Linear extrapolation from 200 steps therefore does not guarantee 70% unique weak coverage.

## Hybrid query schedule

v0.5.4 keeps the NBS model GO-query conditioned, but partitions each Q=64 episode into:

- 16 hierarchy query rows (8 child-parent pairs),
- 16 weak-focus GO query rows selected through unseen weak proteins,
- 32 shuffled-cycle GO coverage rows.

A weak-focus row is constructed as:

1. take an epoch-unseen, rank-owned weak protein from the protein-major pseudo CSR;
2. choose one eligible GO annotation for that protein using modelout probability, pseudo-pool capacity and GO specificity;
3. construct the normal NBS GO query with core support and BoxSquaredEL context;
4. force the weak anchor into the pseudo-positive set;
5. fill the remaining pseudo-positive quota with `go_cyclic_unique` sampling.

The architecture remains GO-query based; only the selection of some GO rows is weak-first.

## Epoch planning

New epoch unit:

```text
hybrid_go_weak_coverage
```

The number of synchronized steps is the maximum of:

- one complete eligible-GO coverage floor;
- a configurable unique weak-protein target;
- one equivalent core gold support/target pass.

Formal BP defaults:

```text
Q = 64
C = 1536
weak_focus_queries_per_episode = 16
weak_focus_targets_per_query = 32
weak_unique_coverage_target_per_epoch = 0.70
weak_focus_planning_efficiency = 0.60
```

The 0.60 planning efficiency is deliberately conservative relative to the declining v0.5.3 unique/pair efficiency. The exact denominator is the number of weak proteins that have at least one modelout-positive annotation in the eligible GO query space, not blindly all weak proteins.

## DDP behavior

- Weak-focus anchors are selected from rank-owned role-local weak rows.
- All focus anchors are pre-marked before ordinary pseudo filling, preventing one focus anchor from being consumed by another query in the same episode.
- Existing `go_cyclic_unique` rank-aware fallback remains available for filling each GO pseudo pool.

## Diagnostics

New batch/epoch fields include:

```text
wfq       realized/requested weak-focus GO rows
wfa       newly reserved weak-focus anchors
wft       retained weak-focus pseudo pairs
wft_fill  selected focus targets / configured focus capacity
wft_keep  retained focus targets / selected focus targets
puniq_eff unique pseudo-target weak proteins / pseudo-pair occurrences
```

The epoch log reports `pweak` against the eligible pseudo-active weak denominator and keeps `pweak_all` against the complete weak set.

## Files changed

- `nbs_pg/episode.py`
- `nbs_pg/latence_graph_stores.py`
- `nbs_pg/local_loader.py`
- `nbs_pg/training.py`
- `scripts/nbs/train_nbs_fixed_epochs.py`
- `scripts/nbs/run_train_nbs_fixed_epochs.py`
- `configs/bp_fixed_epoch_v0.5.4.json`
- `tests/test_hybrid_weak_focus_v054.py`
- `pyproject.toml`

## Validation

- Python compileall: passed.
- Non-PyG regression suite: 70 passed.
- Production PyG/DDP tests must be rerun in the LATENCE server environment.
