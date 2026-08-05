# Validation status

Completed in the current environment:

- Python syntax compilation for package, examples, tests and NBS scripts;
- 9 pure PyTorch/NumPy tests passed:
  - box inclusion and calibration;
  - center/log-offset GO encoding;
  - factorized context-score equivalence;
  - exact base-logit initialization;
  - global GO-cache fallback;
  - fixed-K edge-index GO inversion;
  - protein-major CSR GO inversion with probability preservation;
  - default delta gate independence from expert probability.

The environment does not contain `torch_geometric`, so the real PyG tests were
not executed here:

```text
tests/test_source_additivity.py
tests/test_data_leakage.py
```

They must pass in the LATENCE server environment before full BP training.

The 281,457,664-edge production candidate file was not included in the uploaded
artifact, so the index builder was verified on representative small arrays.  A
full run must additionally confirm the manifest-declared fixed-512,
protein-major ordering; the builder aborts if that contract is violated.
