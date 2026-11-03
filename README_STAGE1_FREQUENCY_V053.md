# LATENCE Dark–Weak Stage-1 frequency diagnostics v0.5.3 (metadata hotfix)

**Additive patch, not a replacement for NBS v094 or the frozen Stage-1 evaluation.** Only a new frequency evaluator and its test are installed; `eval_stage1_frequency_v052.py`, all prior JSON/TSV reports, pretrained models, v094 model/training, and prediction matrices remain untouched.

## Bug in v0.5.2

The original LATENCE `unidata_with_exp_train_pseudo.pkl` historical `train/<GO namespace>` usually has `proteins` and `prop_annotations`, but **not** `annotations`. v0.5.2 defaulted to `--train-label-field annotations`, causing `KeyError: 'annotations'`. Original Stage-1 `exp_train.py` `compute_ic_from_dataset()` uses `data[mode][task]['prop_annotations']` when counting historical training labels.

**Immediate fix without installing this patch:** rerun `eval_stage1_frequency_v052.py` with `--train-label-field prop_annotations` (in a new output directory).

**This patch:** provides `scripts/dark_weak/eval_stage1_frequency_v053.py`, changes CLI default to `prop_annotations`, has a targeted diagnostic if a requested field is unavailable, and records the actual count source in its output manifest. It never silently substitutes one label definition for another.

### Scientific interpretation

Training frequency of `prop_annotations` counts *propagated* historical positive protein–GO associations (direct plus ancestors). Accordingly, `rare`, `rare_le_5`, `medium`, `common`, and `zero_train` refer to **training-propagated label frequency**, not historical **direct-only** GO frequency. The latter cannot be recovered reliably from propagated labels alone. Current test ranking continues to use **future direct annotations**; test Fmax/AuPRC/AP continue to use future propagated annotations. Neither test label field is used to define the training frequency groups.

`historical_classic` uses the Stage-1 quantile formulas on canonicalized historical counts; `historical_fixed` uses fixed 0,1–5,6–9,10–99,100–999,1000+ bands. Per-frequency-bin native v085 Fmax/AuPRC/AP have a restricted GO denominator and should **not** be compared directly with full-vocabulary metrics. Future-direct Recall@K and MRR are evaluated against the **full** GO ranking. `zero_train` is *not* strict model-wide zero-shot.

## Install and test

From the LATENCE project root:

```bash
unzip -n LATENCE_dark_weak_stage1_frequency_hotfix_v0.5.3.zip -d .
python -m unittest discover -s tests -p 'test_dark_weak_stage1_frequency_v053.py' -v
```

## Quick ranking-only check (CPU)

```bash
export STAGE1_COMPLETE=outputs/dark_weak_eval/stage1_full_bp_epoch100/bp_complete_manifest.json

python scripts/dark_weak/eval_stage1_frequency_v053.py \
  --complete-manifest "$STAGE1_COMPLETE" \
  --unified-root data/dark_weak_test/unified \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --train-split train \
  --train-label-field prop_annotations \
  --families historical_classic historical_fixed \
  --metric-scope ranking_only \
  --ks 10 50 100 \
  --bootstrap 50 \
  --out outputs/dark_weak_eval/stage1_bp_frequency_rank_probe_v053
```

## Full stratified native v085 evaluation (CPU, potentially RAM-intensive)

```bash
python scripts/dark_weak/eval_stage1_frequency_v053.py \
  --complete-manifest "$STAGE1_COMPLETE" \
  --unified-root data/dark_weak_test/unified \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --train-split train \
  --train-label-field prop_annotations \
  --families historical_classic historical_fixed \
  --metric-scope all \
  --ks 10 50 100 \
  --bootstrap 500 \
  --out outputs/dark_weak_eval/stage1_bp_frequency_v053
```

If the output directory already exists, select a new one. Existing artifacts are deliberately not overwritten. No GPU inference or model training is involved.

## Check the output label semantics

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/dark_weak_eval/stage1_bp_frequency_v053/frequency_manifest.json')
d=json.loads(p.read_text())
print('schema:', d['schema'])
print('train_split:', d['train_split'])
print('train_label_field:', d['train_label_field'])
print('frequency definition:', d['label_policy']['frequencies'])
print('zero_train:', d['label_policy']['zero_shot'])
PY
```

Expected `train_label_field: prop_annotations`. The output includes `frequency_metrics.tsv`, `frequency_groups.tsv`, `paired_rank_bootstrap_deltas.tsv`, `metric_deltas_vs_backbone.tsv`, `direct_positive_global_ranks.tsv`, and `frequency_manifest.json`.

## Verification scope

Eight synthetic-data regression tests passed, including (1) train metadata with only propagated labels, (2) helpful failure if explicit direct labels are missing, (3) v053 default flag, (4) original manifest unchanged, (5) native v085 metrics and full-vocabulary rank protocol, and (6) v094 native core-frequency integration. No real server-side Stage-1 prediction arrays were available in this local test runtime, so actual stratified numerical results still require the server execution.
