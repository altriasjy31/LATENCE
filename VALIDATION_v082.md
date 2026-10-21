# v0.8.2 validation

Environment: Python 3.11, PyTorch 2.4.1 CPU, single-thread BLAS/OpenMP.
Six selected test files: 61 passed, 1 skipped.

The tests cover:
- Real mmap dense teacher loading, immutable protein/GO order and source checks.
- All weak rows eligible, including no thresholded pseudo-positive.
- Full soft teacher target gradients, all-21,312-column strata, explicit weak/core balance.
- Both weak→GO and weak→core→GO paths, independent task alias output columns.
- Full classification gradients and matching attention restricted to the query budget.
- Exactly unchanged C outside the selected refinement set.
- No teacher/gold dependency in selection or graph forward; deterministic evaluation.
- FP32 residual/loss behavior under CPU BF16 autocast.
- Meaningful optimization of a small multi-label fixture.
- Exact equality of model, optimizer and torch RNG between continuous/resumed dual training.
- Old-version, changed dense teacher, LR and query-budget resume rejection.
- Real 2-protein/2,903-GO inductive export and existing metric CLI, with distinct E/M references.
- Separate final/classification output identities and bound checkpoint/input/probability hashes.
- Verified prediction-cache reuse, rejected source changes, and shell branch/ablation routing.

The two-rank CPU Gloo test is skipped only because this environment rejects
TCP communication with "Operation not permitted". CUDA/NCCL was not tested.
No access to the user's full BP dataset, pretrained Stage1 runtime or GPU:
this is implementation validation, not evidence of improved biological metrics.

Release gates:
- Python source compilation and bash syntax checks.
- Installer applied to an isolated full-project copy, then applied again.
- Backup behavior verified with an intentionally changed destination file.
- Packaged source hashes verified against package_files.json.
- No data, model checkpoints, or existing experiment outputs included.
