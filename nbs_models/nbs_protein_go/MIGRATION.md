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
