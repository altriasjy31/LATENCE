# v0.8.7 validation record

Test environment: Python 3.12, PyTorch 2.4.1+cpu, NumPy/SciPy, pytest 9.1.1. Local dependency installation was isolated; dependency binaries are not part of this release.

## Results

| Check | Result |
|---|---|
| New model | 20 passed |
| Global supervision seed data mask | 13 passed |
| Exact epochs and fixed schedule | 36 passed |
| Loss parity / global role normalization | 26 passed |
| Training and exact CPU resume | 9 passed |
| Export and real-array evaluation integration | 3 passed |
| Evaluator and shell/config contracts | 69 passed |
| Existing v086 model regression | 18 passed |
| Combined above | **194 passed** |
| Safe installer / packaging | **15 passed** |
| Actual two-process CPU Gloo training / resume | **1 skipped: runtime denied Gloo TCP device initialization** |

Only explicit Gloo device initialization permission denial is eligible for that skip. Training exceptions, numerical failures and timeouts fail the test. No actual multi-process gradient synchronization has been validated here. Separate deterministic loss tests verify that averaged rank gradients equal full-batch gradients, including ranks with no weak queries. This mathematical check does not replace a successful distributed run.

The combined CPU run reported 11 PyTorch deprecation warnings for its own activation-checkpoint CPU autocast context. No model/test failures occurred.

## What was exercised

- Residual mode with source dropout disabled matches v086 initialization, forward values, gradients and RNG.
- Direct output is unchanged when dense backbone logits are replaced, including by NaN, with candidate inputs held fixed; both relation-SAGE layers and GO matching receive gradients.
- Source masks are shared between query candidate injection and decoder evidence; independent source streams do not consume global RNG; activation checkpoint recomputation preserves masks.
- Weak IDs appear exactly once per epoch, with no padded/dropped tail. Core stream does not reset at weak boundaries and avoids duplicates within a global batch.
- Stream and scheduler restore exact state. Model/optimizer/Torch RNG match uninterrupted runs for residual, direct and source-dropout configurations.
- All global supervision seeds are excluded from sampled gold/pseudo, decoder gold anchors and fixed PU support. Candidate features, topology and external inference remain intact.
- New loss matches the old objective/gradient/RNG under default mining and single-rank normalization. Global-role weighted loss is correct for unequal rank splits.
- Real mmap fixtures and 2,903-column prediction arrays exercise train/export/evaluate/recompute. Export runs without E/M references or metadata. Cached output reuse checks source flags, prediction mode and version identity; tampering is rejected.
- v086 cached metric recomputation remains available through its historical identity rules.
- Installer tests reconstruct pristine v085+v086, verify dependency checksums, non-overwrite behavior, check-only, idempotence, path/symlink rejection, rollback and repeatable ZIP content.

## Run on the target server

From the project root, with the existing project dependencies and pytest installed:

```bash
PYTHONPATH=.:nbs_models/nbs_protein_go:nbs_models/nbs_protein_go/tests \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q \
  nbs_models/nbs_protein_go/tests/*_v087.py \
  tests/*_v087.py \
  nbs_models/nbs_protein_go/tests/test_full_task_model_v086.py
```

The expected non-distributed count is 194. The additional Gloo test must pass on a server permitting local distributed sockets before claiming CPU multi-process coverage. The README's 2-GPU 20-step smoke then checks the actual CUDA/NCCL environment and full graph tensors.

## Boundaries

No full graph training, GPU memory benchmark, NCCL execution, model-quality gain, or complete 5-epoch result is claimed. The weak epoch length 3,823 and total 19,115 updates are calculated from the uploaded population of 489,222 weak proteins and the configured 2 × 64 weak batch; runtime recomputes them from the loaded population.

All original v085+v086 project file bytes are preserved in the working merged baseline. The release installer additionally checks 76 required prior-version files against hashes from the original uploaded v086 package. This is an incremental release, not a self-contained source tree or data package.
