# v0.5.6.1 metrics hotfix

Deploy over v0.5.6. No retraining or data regeneration is required.

The hotfix only corrects epoch-level `negrows/posonly/mixed` aggregation under DDP. Background-PU sampling and all scientific training semantics are unchanged.

After deployment a short 10-step probe is sufficient. Expected behavior for Q=64 is approximately:

- `negrows` close to 100%
- `posonly` close to 0%
- `rowpart=ok`
- `bg_fill` remains near the v0.5.6 value
- `np_ratio` remains near the v0.5.6 value
