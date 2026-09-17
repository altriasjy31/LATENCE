# Validation v0.8.3

43 targeted tests passed on Python 3.11 / CPU PyTorch 2.4.1.

- 10 data tests: actual default config provenance validation, binary CSR membership;
  incoming native PP direction; deterministic deduplication/top-k across scan
  blocks; hash/split cache checks; cosine cache reuse; absent context; same
  train/inference context; no neighbor gold/pseudo payload.
- 11 model tests: all PP relation and weak/core gradients; exact GO incidence
  survives mean/ontology-alias ambiguity; all output columns; empty contexts;
  FP32 residual/loss under AMP; activation-checkpoint and chunk parity;
  GO shuffle reproducible independent of batch; weak/core/PP removal;
  useful optimization when only graph neighborhoods distinguish samples.
- 8 runner tests: actual mmap+PP batches; exact continuous vs resumed model and RNG;
  trainable graph/local/no_graph variants; no_graph not forced to B; old
  checkpoint rejection; changed implementation/base rejection; binary target
  guard; real 2,903-GO metric CLI and prediction export with source binding.
- 14 evaluator/shell tests: single-output E/M/B/G comparisons; probability/input/
  checkpoint binding; six-intervention summary with mismatch rejection;
  shell dispatch, defaults and missing checkpoint handling.

Release checks: syntax compilation, bash -n, checksummed ZIP; installation into
an isolated project; local-loader patch preserving unrelated local edits;
automatic backup; repeated installation; packaged tests on installed project.

CUDA/NCCL, full BP production data and two-GPU speed were not tested here.
Graph-learning toy tests establish implementation behavior, not real-task
performance improvement. The no_graph control retains GO ontology encoding;
it removes protein-neighborhood/candidate evidence, not every ontology prior.
