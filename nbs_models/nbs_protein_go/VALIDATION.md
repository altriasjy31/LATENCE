# Validation status for NBS v0.4

Validated in the artifact build environment:

- Python compilation for package, experiments and scripts;
- 29 pure PyTorch/NumPy tests passed; one real-PyG local-batch test was skipped because PyG is unavailable;
- two-process CPU/Gloo DDP training, metric reduction and rank-0 checkpoint
  writing;
- direction-keyed P-P CSR construction;
- GO-major and Protein-major annotation index construction;
- normalized BoxSquaredEL relation parsing on synthetic contracts;
- rank-disjoint/reproducible episode generation;
- local materializer contract with mocked PyG construction.

The build environment does not provide `torch_geometric` or the production
LATENCE arrays. The following must be rerun on the server:

```bash
PYTHONPATH=. pytest -q \
  tests/test_source_additivity.py \
  tests/test_data_leakage.py \
  tests/test_real_local_batch_v04.py

python scripts/nbs/run_audit_nbs_v04_training_inputs.py
python scripts/nbs/run_smoke_test_nbs_train_loader.py
```

Then perform the one-step two-GPU production smoke described in README.
