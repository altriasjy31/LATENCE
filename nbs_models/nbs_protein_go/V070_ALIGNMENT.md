# NBS v0.7.0: protein--GO alignment update

This snapshot is based on the complete v0.6.0 perfopt + tail-fix restore.  It
keeps the OneCycle scheduler, prefetch path, weak-primary exhaustive queue and
tail-drain invariants, then changes the query/supervision contract.

## What changed

1. **CPU protein--GO alignment audit**

   `scripts/nbs/audit_nbs_protein_go_batch_alignment.py` reports, for every
   sampled weak anchor, eligible pseudo GO count, pseudo/candidate intersection,
   query hits, actual pseudo-loss hits, controlled negatives, unknown decoded
   positions and effective positive/negative weights.  It works with both the
   v0.6 and v0.7 configs and never materializes a PyG graph.

2. **Multi-positive protein-primary query bank**

   `weak_primary_positive_go_per_protein` replaces the implicit one-protein /
   one-GO assignment when set above one.  A greedy b-matching first guarantees
   one positive per selected weak protein, then allocates additional positives
   while sharing GO rows and respecting query, ontology and per-row capacity.
   `weak_primary_rotate_go_across_epochs` keeps the strongest label stable and
   rotates the remaining ranked labels across epochs.

3. **Protein-column PU supervision**

   Primary weak proteins receive explicit candidate-disagreement hard PU and
   low-base background PU query positions.  `weak_primary_mask` identifies the
   resulting column objective.  `protein_column_asl` normalizes within each
   protein and then across proteins, so proteins with more sampled GO terms do
   not dominate proteins with fewer labels.

4. **Decoded-unknown base preservation**

   `loss.base_anchor_scope=unknown_decoded` anchors every decoded unknown
   position to the first-stage base posterior.  Unknown positions remain
   excluded from label ASL and are never converted into hard negatives.

5. **Optional isolated inductive target encoding**

   `local_sampling.candidate_context_mode=isolated_similar_to_core` removes all
   supervised target proteins from the support graph.  Each target is scored by
   the external inference encoder from its own representation and deterministic
   core `similar_to` neighbours.  Variable neighbour counts are supported by an
   explicit mask; zero-neighbour targets use the feature-only path.

6. **GO tower residual query input**

   The full ontology cache now preserves per-layer GO tower deltas.  When
   `model.use_go_residual_query=true`, the query encoder attends over those
   deltas and fuses them into the GO query.  The protein GNN and GDAR source
   routing equations are unchanged.

## Configurations

- `bp_fixed_epoch_v0.7.0_sampling_loss_probe.json`
  - Q=128; 96 weak-primary query-bank slots;
  - four positive GO targets per weak protein;
  - four hard PU plus four background PU targets per weak protein;
  - dual-axis loss and GO residual query enabled;
  - preserves v0.6 joint-local candidate encoding to isolate sampling/loss.

- `bp_fixed_epoch_v0.7.0_inductive_aligned_probe.json`
  - identical sampling, query and loss settings;
  - additionally enables isolated `similar_to`-core candidate encoding.

Both are three-epoch probe configs.  Do not resume a v0.6 optimizer/checkpoint
into them: v0.7 adds trainable query modules and changes the batch contract.

## Recommended execution order

Run from the project root.

### 1. Audit the current v0.6 sampler

```bash
python scripts/nbs/audit_nbs_protein_go_batch_alignment.py \
  --config nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.6.0_perfopt.json \
  --output-dir outputs/latence_nbs_audit/bp_v060_alignment \
  --epochs 1 \
  --episodes-per-epoch 64 \
  --world-size 2 \
  --rank 0
```

Repeat with `--rank 1` for a rank-specific DDP audit.  The primary decision
fields are `loss_positive_at_least_2`, median `loss_recall`, effective
negative/positive weight ratio and cross-epoch unique loss GO count.

### 2. Import and config validation

```bash
python - <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path("nbs_models/nbs_protein_go").resolve()))
from nbs_pg import __version__
from nbs_pg.training import NBSLossConfig
from nbs_pg.episode import NBSQueryEpisodeConfig

assert __version__ == "0.7.0"
print("NBS v0.7 imports: OK")
PY
```

```bash
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.7.0_sampling_loss_probe.json
export NBS_NUM_GPUS=1
export NBS_VALIDATE_CONFIG_ONLY=1
export NBS_FRESH_START=1
unset NBS_RESUME
python scripts/nbs/run_train_nbs_fixed_epochs.py
unset NBS_VALIDATE_CONFIG_ONLY
```

### 3. Twenty-step semantic smoke test

```bash
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.7.0_sampling_loss_probe.json
export NBS_NUM_GPUS=2
export NBS_EPOCHS=1
export NBS_SAVE_EPOCHS=1
export NBS_MAX_STEPS_PER_EPOCH=20
export NBS_FRESH_START=1
export NBS_OUTPUT_DIR=outputs/latence_nbs_train/bp_v070_sampling_loss_smoke20
unset NBS_RESUME
python scripts/nbs/run_train_nbs_fixed_epochs.py
```

Expected semantic diagnostics:

- `wpa` close to 96 (tail batches may be smaller);
- `wpp` substantially above 96 and ideally close to 384;
- `wpn` above zero;
- `protein_column_asl` and `contrib_protein_column_asl` finite/non-zero;
- `weak_primary_multi_positive_anchor_rate` high;
- `full_supervision_retained=1`, `candidate_dropped_by_cap=0`;
- `null_collapse=0` and finite GO residual attention.

### 4. Isolated candidate ablation

Use the same smoke command with:

```bash
export NBS_TRAIN_CONFIG=nbs_models/nbs_protein_go/configs/bp_fixed_epoch_v0.7.0_inductive_aligned_probe.json
export NBS_OUTPUT_DIR=outputs/latence_nbs_train/bp_v070_inductive_aligned_smoke20
```

Check that `isolated_candidate_count` equals the candidate union size and that
`isolated_candidate_with_core_neighbor` is non-zero.  Feature-only rows are
allowed unless `isolated_candidate_require_core_neighbor=true` is explicitly
set for a strict audit.

## Interpretation boundary

The sampling/loss probe should be compared with v0.6 first.  Only after it is
stable should the isolated candidate configuration be used to measure the
training--inference alignment effect.  This separation prevents a recall gain
from being confused with a context-path change.
