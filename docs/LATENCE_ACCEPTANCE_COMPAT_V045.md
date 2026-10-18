# LATENCE v088 acceptance compatibility fix (v0.4.5)

## Root cause

`check_nbs_v088_acceptance.py` is bound to the **original Fmax-only** report schema:

```json
{"schema_version": 3, "evaluation_scope": "standard_protein_fmax_only"}
```

The v0.4.3 evaluation update changed the default output schema to version 4 and the scope to `standard_protein_fmax_and_v085_pr_auc`. These are different report contracts. Do not manually edit either key: the checker may additionally inspect method fields, metric contract, and provenance.

## Files in this package

- `scripts/nbs/eval_nbs_full_task_v088.py` — **BYTE-IDENTICAL** to the v088 Fmax-only evaluator supplied before the v0.4.2/v0.4.3 upgrades; SHA256: `524dff8942070a4f615b743059e84cc5e62859da6c04fe886abae8d9d0748145`. The original v088 acceptance code consumes its output.
- `scripts/nbs/eval_nbs_v085_supplementary.py` — **new**, independent scorer using the installed original `nbs_pg/full_task_metrics_v085.py` to report `standard_protein_fmax`, `standard_micro_pr_auc`, and `standard_micro_ap` after the original comparison has passed its shape/provenance validations. This output is stored alongside the acceptance JSON, not in it. It verifies unchanged Fmax, checkpoint/metadata/GO registry/array hashes, and original report schema.
- `optional_restore_v089/eval_nbs_full_task_v089.py` — original v089 Fmax-only evaluator supplied by the user, for OPTIONAL future restoration if `check_nbs_v089_acceptance.py` has the same contractual constraint. **Not installed automatically.**
- `tests/test_v045_acceptance_split.py` — synthetic tests.

The package does not modify `full_task_metrics_v085.py`, model/trainer files, or checkpoints. Previous Dark–Weak `run_complete_evaluation.py` remains available and retains its unified metrics.

## Fast recovery: existing epoch20 v088 predictions (NO GPU inference required)

From the LATENCE project root:

```bash
# 0. Set to your actual paths. Shown defaults are from the original v088 launcher.
export PYTHON_BIN="${PYTHON_BIN:-python}"
export WORK_DIR="${WORK_DIR:-outputs/latence_nbs_experiments/v088/hetero_tuned}"
export INPUT_DIR="${INPUT_DIR:-outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2}"
export METADATA_FILE="${METADATA_FILE:-/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl}"

# 1. Back up the PATCHED evaluator and its old JSON results (if present).
cp scripts/nbs/eval_nbs_full_task_v088.py scripts/nbs/eval_nbs_full_task_v088.py.pre_v045.bak
cp "$WORK_DIR/eval_step1920/full/metrics/nbs_w2s_comparison.json" \
   "$WORK_DIR/eval_step1920/full/metrics/nbs_w2s_comparison.v043_backup.json"

# 2. Install package (the original v088 evaluator is restored).
unzip -o LATENCE_acceptance_compat_v0.4.5.zip -d .
sha256sum scripts/nbs/eval_nbs_full_task_v088.py
# Must be 524dff8942070a4f615b743059e84cc5e62859da6c04fe886abae8d9d0748145

# 3. Re-evaluate the SAVED prediction file directly: no NBS forward or checkpoint load.
"$PYTHON_BIN" scripts/nbs/eval_nbs_full_task_v088.py \
  --task bp \
  --metadata-file "$METADATA_FILE" \
  --nbs-prob "$WORK_DIR/eval_step1920/full/nbs_ind_test_prob.f32.npy" \
  --backbone-prob "$INPUT_DIR/backbone_ind_test_prob.f16.npy" \
  --protein-ids "$INPUT_DIR/protein_ids.txt" \
  --candidate-go-index "$INPUT_DIR/candidate_go_index.i32.npy" \
  --input-manifest "$INPUT_DIR/ind_test_input_manifest.json" \
  --prediction-manifest "$WORK_DIR/eval_step1920/full/nbs_full_task_prediction_manifest.json" \
  --output-dir "$WORK_DIR/eval_step1920/full/metrics" \
  --metric-backend stage1

# 4. Verify the original acceptance scope.
"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['WORK_DIR']) / 'eval_step1920/full/metrics/nbs_w2s_comparison.json'
r = json.loads(p.read_text())
print('schema_version =', r['schema_version'])
print('evaluation_scope =', r['evaluation_scope'])
print('G Fmax =', r['methods']['NBS_final']['standardized']['standard_protein_fmax'])
print('B Fmax =', r['methods']['backbone_base']['standardized']['standard_protein_fmax'])
assert r['schema_version'] == 3
assert r['evaluation_scope'] == 'standard_protein_fmax_only'
PY

# 5. Original acceptance checker — now consumes an original-schema report.
"$PYTHON_BIN" scripts/nbs/check_nbs_v088_acceptance.py \
  --baseline-comparison BASELINE_CONTRACT_v088.json \
  --comparison "$WORK_DIR/eval_step1920/full/metrics/nbs_w2s_comparison.json" \
  --output "$WORK_DIR/acceptance_epoch20.json"
```

If the direct native command reports absent files, inspect `INPUT_DIR` and `WORK_DIR` before proceeding. **Do not pass** the v0.4.3 JSON as an acceptance report or manually re-label it as Fmax-only.

You can also run `bash scripts/nbs/run_nbs_v088.sh evaluate` after restoring the original evaluator. The v088 trainer validates the cached prediction and then reruns the original metrics, but this requires loading the model/data; direct scoring is lighter.

## Independent supplementary Fmax / AuPRC / AP (after original acceptance)

```bash
"$PYTHON_BIN" scripts/nbs/eval_nbs_v085_supplementary.py \
  --comparison "$WORK_DIR/eval_step1920/full/metrics/nbs_w2s_comparison.json"
```

This writes:

```text
$WORK_DIR/eval_step1920/full/metrics/nbs_supplementary_v085.json
```

while preserving the `nbs_w2s_comparison.json` bytes. The supplementary scorer recomputes Fmax to ensure bit-level **value consistency** (absolute tolerance `1e-9`) with the legacy report; reports original-v085 micro PR-AUC (trapezoidal) and micro AP separately; refuses changed inputs, metadata, GO registry, or checkpoint identity.

To inspect:

```python
import json, os
from pathlib import Path
p = Path(os.environ['WORK_DIR']) / 'eval_step1920/full/metrics/nbs_supplementary_v085.json'
d = json.loads(p.read_text())
for name in ('NBS_final', 'backbone_base'):
    s = d['methods'][name]
    print(name, 'Fmax', s['standard_protein_fmax'],
          'AuPRC', s['standard_micro_pr_auc'], 'AP', s['standard_micro_ap'])
print('G-B:', d['G_minus_B'])
```

## v089

The v0.4.4 patch likewise changed the default `eval_nbs_full_task_v089.py` scope. **If an independent v089 acceptance checker requires the old Fmax-only schema**, restore the original v089 evaluator separately:

```bash
cp scripts/nbs/eval_nbs_full_task_v089.py scripts/nbs/eval_nbs_full_task_v089.py.pre_v045.bak
cp optional_restore_v089/eval_nbs_full_task_v089.py scripts/nbs/eval_nbs_full_task_v089.py
```

Run its native evaluation to regenerate an original-schema report, then use the SAME supplementary scorer. This preserves v089 phase and lifetime checks in the original implementation. No changes to `train_nbs_full_task_v089.py` or its trained checkpoints.

## Validation and limitations

- Local synthetic tests verify preservation of original file SHA, supplementary no-overwrite behavior, Fmax agreement, metadata/prediction hashes, schema rejection, and row ID alignment.
- The actual `check_nbs_v088_acceptance.py` / baseline contract and server-only checkpoints were not uploaded here. Consequently, this package has not been run on the real acceptance checker. It restores the exact native evaluator that checker was designed to consume, instead of approximating its JSON format.
- A passing *report-schema check* does **not** imply that epoch20 satisfies any numerical acceptance target. Once the checker executes, examine its reported performance verdict separately.
- Dark–Weak panel evaluation still belongs in `scripts/dark_weak/run_complete_evaluation.py` using the original standardized metric kernel. Never overwrite an original acceptance JSON with a multi-metric report.
