# v0.4.4 — v0.8.9.1 compatibility (incremental overlay)

- Added version-correct `scripts/dark_weak/export_nbs_v089_predictions.py`, requiring `release_version=0.8.9.1`, `model_version=0.8.8`, `stage=nbs_v089_long_phase`, and native v089 checkpoint/input/output SHA256, GO vocabulary, no-fusion and phase identity checks. Original v088 adapter preserved.
- Added `scripts/dark_weak/audit_v089_graph_inputs.py` using **v088 model/data class** plus **v089 store loader**, including runtime-version identity and actual ghost-graph checks.
- Based on **original server-uploaded v089 evaluator**, added a `evaluate_standard_only` path with v085 protein-Fmax, trapezoidal micro PR-AUC, and micro-AP, plus optional E/M reference-only scoring. Retained original `evaluate_fmax_only` unmodified, exposed via `--fmax-only`. Kept all original v089 `phase_plan`/`phase_origin` provenance and SHA checks. Existing train runner automatically calls the new default.
- Extended `scripts/dark_weak/run_complete_evaluation.py` with optional `--native-nbs-adapter MODEL=/path.json` to cryptographically bind output, checkpoint, input manifests, protein order, GO columns, and v089 lifetime counters to score reports. Stage-1 and v088 users are unaffected.
- Added 5 v089 synthetic tests (including native Fmax-only parity, PR-AUC/AP, lifetime-identity tampering, checkpoint tampering, adapter integrity and legacy-config rejection). Existing suite: 31 pass, 1 skip.
- Explicitly did **not** change any v088/v089 training, checkpoint, model, store-loader, phase-implementation or v085 metrics files.
