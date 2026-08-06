# Validation status for NBS v0.3

Completed in the artifact environment:

- Python syntax compilation for package, examples, tests and NBS scripts;
- all pure PyTorch/NumPy tests in the v0.3 suite;
- fixed-epoch snapshot policy and checkpoint metadata;
- rejection of validation-set selection and early stopping;
- query-axis GO hierarchy consistency;
- three-column candidate evidence initialization;
- open-world supervision weights;
- GO-query episode support/candidate disjointness;
- BoxSquaredEL geometry, residual initialization, global GO fallback,
  inverted-index construction and expert-free student gate.

The artifact environment does not contain `torch_geometric`, so these retained
real-PyG tests must be rerun after deployment:

```bash
PYTHONPATH=. pytest -q \
  tests/test_source_additivity.py \
  tests/test_data_leakage.py
```

The production 281,457,664-edge files are not bundled.  The already-built
GO→Protein inverted index remains the source for GO-query candidate sampling.

## BoxSquaredEL alignment and checkpoint interval

```bash
PYTHONPATH=. pytest -q \
  tests/test_boxsqel_manifest.py \
  tests/test_fixed_epoch_training.py
```

These tests verify GO identifier normalization, checkpoint class-map alignment,
canonical/alt-ID geometry replication, aligned mmap loading, periodic epoch
unions, and the legacy `save_every` compatibility alias.
