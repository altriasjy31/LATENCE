# NBS migration

## Conceptual name

NBS now denotes **Neighborhood–BoxSquare**.  BoxSquaredEL is the concrete GO
geometry used to place functional neighbourhoods into nested boxes.

## Project paths

```text
nn_graph_model/nbs_protein_go  -> nbs_models/nbs_protein_go
outputs/latence_nn_pp          -> outputs/latence_nbs
```

Only LATENCE project fields and paths should be renamed.  Standard PyTorch names
such as `torch.nn` and `nn.Module` must remain unchanged.

## Model and schema changes

| Previous prototype | Updated NBS | Main change |
|---|---|---|
| true/pseudo only | gold/candidate/pseudo relation pairs | first-stage top-512 candidate evidence is first-class |
| P–P edge dims passed manually | PPI/similar/weak defaults are all 3 | matches actual manifests |
| 8-dimensional GO edge feature | 8 box + 2 topology for `is_a` | closure hop and directness retained |
| box inclusion reused implicitly | `part_of` uses separate topology features | partonomy is not subclass inclusion |
| bidirectional NeighborLoader default | directional default | preserves `weak_to_core` semantics |
| candidate gate could read expert | student-candidate gate by default | formal NBS inference is expert-free |
| one masking mode | relation-specific masking | gold/pseudo all; candidate query-only |
| no production reverse index | streaming GO→Protein CSR | supports 281,457,664 candidate edges |

Compatibility aliases for `TRUE_*` remain, but resolve to the new gold relation
names rather than recreating the obsolete schema.

## v0.3 training migration

Previous training plans referenced validation-based checkpoint selection.  NBS
v0.3 instead follows the first-stage LATENCE convention:

```text
validation loader / best checkpoint / patience
    -> fixed final epoch and named epoch snapshots
```

The reference BP configuration uses 150 total epochs and retains epochs 100 and
150.  Evaluation code should consume these snapshots independently; it must not
feed evaluation metrics back into the training loop.

Additional API changes:

| v0.2 | v0.3 |
|---|---|
| scalar candidate evidence | scalar or full `[probability, selector, rank]` evidence |
| hierarchy assumed GO columns | explicit `go_axis`; NBS defaults to query rows |
| confidence mainly pseudo-specific | `supervision_weight` for all selected positions |
| no trainer | fixed-epoch trainer with atomic snapshots and resume |
| no episode/store layer | mmap stores and GO-query episode sampler |

## v0.3.1 configuration migration

Use:

```json
"save_interval_epochs": 10
```

instead of the older `save_every` field.  The old field is still read for
compatibility but must not conflict with the new one.

Add the `go_boxsqel` configuration block and set `model_inputs.go_box_dim=512`.
Run `scripts/nbs/run_prepare_go_boxsqel_for_nbs.py` once before constructing the
training loader so ontology-class rows are projected into classifier GO order.

## v0.4 training integration

v0.4 replaces the placeholder loader contract with a production mmap/CSR local
loader and adds single-node DDP. Existing candidate/pseudo inverted indices are
retained. Add gold Protein-major CSR incrementally with `GOLD_ONLY=1` and
`MERGE_EXISTING=1`.

Task-projected BoxSquaredEL files remain required for task-to-full mapping, but
GO message passing now uses a separate full class-row ontology export and
normalized G–G manifest. Do not point v0.4 training at the old BP/MF/CC-only
`gg_relations_manifest.json`.
