# LATENCE Dark–Weak Stage-1 frequency diagnostics v0.5.2

**Read-only additive extension** to `LATENCE_dark_weak_eval_v0.4.3` / `v0.5.0` / Stage-1 task hotfix `v0.5.1`. Includes **new files only**:

- `scripts/dark_weak/eval_stage1_frequency_v052.py`
- `tests/test_dark_weak_stage1_frequency_v052.py`
- `README_STAGE1_FREQUENCY_V052.md`

Does **not** replace `run_complete_evaluation.py`, `eval_nbs_full_task_v088.py`, `v094` training/evaluator, original result JSON/TSV, checkpoint, or prediction arrays.

## 0. IMPORTANT: what was actually supplied

The uploaded `stage1_full_bp_epoch100_bp_complete_manifest.json` holds **protocol and SHA256 provenance only**. There are **no numerical Fmax/AuPRC/AP results in this manifest**. You must use the server-side `stage1_full_bp_epoch100/bp_complete_metrics.tsv` or per-split `complete_metrics.json` for those results. This extension reads the original manifest to find all already-exported frozen prediction arrays, then computes the stratified results.

## 1. Prerequisites

Must run **inside the existing LATENCE project** with:

- v0.4.3 `scripts/dark_weak/native_v085_metrics.py`, `scripts/eval_dark_weak_predictions.py`
- Original untouched `nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py`
- Original `utils/util_functions.py` (for Stage-1 frequency-quantile definitions)
- Original consolidated Dark–Weak metadata with `annotations` and `prop_annotations`
- Original historical 2022 training metadata (not 2025 test labels): `data/unidata_with_exp_train_pseudo.pkl`
- Complete Stage-1 evaluation manifest and prediction .npy / protein_ids / go_ids matching stored hashes.

```
unzip -n LATENCE_dark_weak_eval_v0.5.2_stage1_frequency_additive.zip -d .
python -m unittest discover -s tests -p 'test_dark_weak_stage1_frequency_v052.py' -v
```

## 2. Full evaluation from existing probabilities (no model inference)

```
export STAGE1_COMPLETE=outputs/dark_weak_eval/stage1_full_bp_epoch100/bp_complete_manifest.json
export STAGE1_FREQ_OUT=outputs/dark_weak_eval/stage1_full_bp_epoch100_frequency_v052

python scripts/dark_weak/eval_stage1_frequency_v052.py \
  --complete-manifest "$STAGE1_COMPLETE" \
  --unified-root data/dark_weak_test/unified \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --train-split train \
  --train-label-field annotations \
  --families historical_classic historical_fixed \
  --metric-scope all \
  --ks 10 50 100 \
  --bootstrap 500 \
  --out "$STAGE1_FREQ_OUT"
```

A quick **ranking-only probe**, no v085 per-bin Fmax/AP computation:

```
python scripts/dark_weak/eval_stage1_frequency_v052.py \
  --complete-manifest "$STAGE1_COMPLETE" \
  --unified-root data/dark_weak_test/unified \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --families historical_classic historical_fixed \
  --metric-scope ranking_only \
  --bootstrap 50 \
  --out outputs/dark_weak_eval/stage1_frequency_rank_probe_v052
```

Each output directory must be new; existing paths are refused to protect previous results. Check the on-server `bp_complete_metrics.tsv` from your first evaluation separately; this new script additionally *copies* that table (if present) into its output folder and prints its values.

## 3. Optional comparison using actual v094 core frequencies

v094 uses its own core positive counts, which **are not necessarily identical** to historical Stage-1 `train.annotations` counts. Use an optional third family explicitly, not as a silent replacement:

```
export V094_CONFIG=outputs/latence_nbs_experiments/v094/weak_teacher_single12800/resolved_config.json
export V094_UNIFIED=$(python - <<'PY'
import json, os
print(json.load(open(os.environ['V094_CONFIG']))['data']['unified_dataset_manifest'])
PY
)

python scripts/dark_weak/eval_stage1_frequency_v052.py \
  --complete-manifest "$STAGE1_COMPLETE" \
  --unified-root data/dark_weak_test/unified \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --v094-unified-manifest "$V094_UNIFIED" \
  --families historical_classic historical_fixed v094_native_core_fixed \
  --metric-scope all \
  --bootstrap 500 \
  --out outputs/dark_weak_eval/stage1_bp_frequency_with_v094_core_v052
```

This option validates v094's native schema, core-count SHA256, GO registry SHA256 **and every classifier column identity**. If incompatible, fails rather than silently applying wrong counts.

## 4. Frequency strata

All frequency groups are frozen **before** future evaluation, using unique historical *training proteins* with direct-positive GO term annotations, grouping alias-equivalent model output columns into **canonical 2022 GO terms**.

`historical_classic` follows the *formula* of original Stage-1 `make_frequency_bins`: `zero_train=0`, `rare=(0,q33]`, `rare_le_5=1..5`, `medium=(q33,q66]`, `common=(q66,+inf)`, `common_ge_50>=50`. Quantiles are taken over positive-frequency canonical GO terms. This is intentionally **alias-aware**, so not necessarily byte-identical to legacy per-output-column bin assignment for aliased GO IDs. `rare_le_5` is **not guaranteed** to be a subset of `rare`.

`historical_fixed` applies the six non-overlapping native v094-style *bounds* to the same Stage-1 historical frequencies: `zero`, `1-5`, `6-9`, `10-99`, `100-999`, `1000+`.

`v094_native_core_fixed` is optional; it takes **actual v094 training-core counts**, not pseudo weak labels. Canonical alias counts are approximated by the `max` across aliases (v094's raw core count arrays do not retain the per-protein list needed for an exact alias union). Every alias remains an original column in Fmax/AuPRC/AP.

## 5. Scores and limitations

- `Fmax_percent`, `AuPRC_percent`, `AP_percent`: original native v085 kernel, **restricted to GO columns of that frequency bin**. Fmax optimizes a new threshold for each bin *using test labels*, so it is descriptive / an upper bound on unknown deployable performance; not a calibrated deployment threshold. Micro AP/PR-AUC treat unannotated positions as negatives for the observed-label proxy.
- `micro_recall_at_10/50/100`, `macro_recall_at_*`, `pairwise_mrr`, `mean_best_rr`: original direct **future** GO positives are assigned to a historical frequency bin, but ranked among **all canonical GO terms** (excluding the GO namespace root); no oracle removal of out-of-bin candidates. Canonical alias predictions are averaged, matching original Dark–Weak ranking evaluation.
- Every bin has counts of unique direct-positive pairs, proteins, GO terms, model columns and observed propagated positives. Empty bins are reported as **NA**, not invented zero performance.
- `paired_rank_bootstrap_deltas.tsv` uses the original sequence SHA256 clusters where available and a protein-ID fallback; it reports paired recall/MRR delta CIs for Expert and Modelout relative to Backbone. Very small bins have unstable uncertainty. CIs are **not** computed for Fmax or AP.
- `zero_train` denotes *no direct historical training positives in that GO term*. It **does not prove** model-wide zero-shot, since ontology ancestors, other labels and the Expert source may still expose that term's semantics.
- Experimental/IPR2GO/Union cohorts overlap and must not be treated as three independent confirmations.
- Full-GO summary scores **cannot be compared numerically as if they used the same GO dimensionality as within-bin Fmax/AuPRC/AP**. Compare models within the same bin and split.
- The training metadata is assumed to have the same original GO index axis as Stage-1, as in the upstream frozen LATENCE metadata; verify provenance externally if this metadata has been regenerated. This script strictly validates the frozen prediction GO ordering and canonical mapping, but the train pickle does not independently encode a GO axis hash.

## 6. Generated outputs

- `frequency_metrics.tsv`: one row per split × frequency family × bin × model; includes GO-restricted v085 Fmax/AuPRC/AP and **global** direct-label recall/MRR.
- `frequency_groups.tsv`: number of canonical GO terms, original classifier columns and historical training frequency range in each bin.
- `direct_positive_global_ranks.tsv`: each direct future `(protein, canonical GO)` positive pair, its full-vocabulary rank, and its frequency bin. This supports biological case study selection.
- `paired_rank_bootstrap_deltas.tsv`: paired confidence intervals for global rank recall/MRR relative to Backbone.
- `metric_deltas_vs_backbone.tsv`: descriptive Fmax/AuPRC/AP differences in percentage points.
- `go_frequency_lookup.tsv`: immutable raw GO column index, canonical GO ID, raw and canonical frequency counts and group masks.
- `frequency_manifest.json`: source/hashes and scientific definitions.
- `original_unstratified_metrics_copy.tsv`: safe copy of previously calculated overall metrics, if available (not recomputed).

## 7. Primary interpretative readout

Run:

```
python - <<'PY'
import pandas as pd
root='outputs/dark_weak_eval/stage1_full_bp_epoch100_frequency_v052'
x=pd.read_csv(root+'/frequency_metrics.tsv',sep='\t')
x=x[(x.split=='dark_weak_test_direct') & (x.family=='historical_classic')]
print(x[['bin','model','n_canonical_GO','n_direct_proteins','n_direct_pairs',
         'Fmax_percent','AP_percent','micro_recall_at_10','micro_recall_at_50',
         'pairwise_mrr']].to_string(index=False))
print('\n--- Delta vs backbone ---')
y=pd.read_csv(root+'/metric_deltas_vs_backbone.tsv',sep='\t')
print(y[(y.split=='dark_weak_test_direct') & (y.family=='historical_classic')].to_string(index=False))
PY
```

**For research analysis, upload** the existing `bp_complete_metrics.tsv` (overall), the new `frequency_metrics.tsv`, `paired_rank_bootstrap_deltas.tsv`, `frequency_groups.tsv`, and `frequency_manifest.json`. They are sufficient for a substantive Stage-1 result analysis without uploading enormous `.npy` matrices.
