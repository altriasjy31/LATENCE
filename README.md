# LATENCE Dark–Weak Unified Evaluation v0.4.4 — NBS 0.8.9.1 compatibility

This is an **incremental overlay on v0.4.3**, not a replacement for the LATENCE model, training code, original v085 metric implementation, or existing Dark–Weak label builders. The original v088 estimator remains installed and unchanged by this overlay. Changes have been audited against the user-provided versions of `full_task_phase_v089.py`, `local_loader_v089.py`, `eval_nbs_full_task_v089.py`, `train_nbs_full_task_v089.py`.

## 1. The decisive version distinction

| Item | v088 | New v089 long-phase |
| --- | --- | --- |
| Runner/checkpoint version | `0.8.8` | **`0.8.9.1`** |
| Graph/model architecture version | `0.8.8` | **still `0.8.8`** |
| Encoder | `FullTaskGraphModelV088` | `FullTaskGraphModelV088` |
| Native data reader | `FullTaskDataV088` | `FullTaskDataV088` |
| Source store loader | `local_loader_v088` | **`local_loader_v089`** |
| Native eval script | `eval_nbs_full_task_v088.py` | **`eval_nbs_full_task_v089.py`** |
| Phase lineage | none | `phase_plan`, `phase_origin`, cumulative epoch and lifetime exposure/update counts |
| Metric kernel | `full_task_metrics_v085` | **unchanged** `full_task_metrics_v085` |

The old `scripts/dark_weak/export_nbs_v088_predictions.py` does **not** accept v089 manifest/checkpoint/version provenance. Never bypass native SHA or phase checks to make it run.

## 2. Contents and safety

Only the following installed project files are added/changed:

```
scripts/nbs/eval_nbs_full_task_v089.py               # ADD/UPDATE ONLY THE v089 EVALUATOR
scripts/dark_weak/export_nbs_v089_predictions.py    # NEW version-correct native exporter bridge
scripts/dark_weak/audit_v089_graph_inputs.py         # NEW version-correct ghost graph audit
scripts/dark_weak/run_complete_evaluation.py        # UPDATE shared v085 evaluator; optional v089 provenance binding
tests/test_dark_weak_v044_v089.py                     # NEW native v089 synthetic regression
```

**Do not modify**: `train_nbs_full_task_v089.py`, `train_nbs_full_task_v088.py`, `full_task_model_v088.py`, `full_task_data_v088.py`, `local_loader_v089.py`, `full_task_phase_v089.py`, `full_task_metrics_v085.py`, existing checkpoints, stored weak pseudo labels or graph caches. The trainer explicitly hashes many of these files; changing them can invalidate checkpoint/resume/export. The v089 evaluator is NOT part of its `prediction_implementation()` hash set.

The new `eval_nbs_full_task_v089.py` is based on the **original uploaded v089 evaluator**, keeping `_phase_identity`, `_prediction_provenance`, `_encoder_identity`, `_validate_source_flags` and original `evaluate_fmax_only()` intact. Default evaluation now also computes AuPRC/AP; `--fmax-only` explicitly uses the unchanged original calculation and summary shape.

## 3. Install into an existing v0.4.3 project

Run at LATENCE project root:

```bash
cp scripts/nbs/eval_nbs_full_task_v089.py \
   scripts/nbs/eval_nbs_full_task_v089.py.backup
unzip -o LATENCE_dark_weak_eval_v0.4.4_v089_patch.zip -d .
```

Do **not** install this patch over an earlier baseline without v0.4.3, because it reuses the v0.4.3 `scripts/dark_weak/native_v085_metrics.py`, the Dark–Weak metadata, inference and ranking tools.

Regression:

```bash
python -m unittest discover -s tests -p 'test_dark_weak_v044_v089.py' -v
```

The test suite constructs an isolated synthetic v089 project tree, including the original v085 metric kernel, and verifies v089 standardized scores, parity with Fmax-only, tamper rejection for phase counters and checkpoint SHA, adapter-score provenance, and rejection of an old v088 configuration. Server tests require the original files `full_task_metrics_v085.py` and `full_task_phase_v089.py` in the expected native `nbs_pg/` location.

## 4. Standardized v089 `ind_test`: native Fmax + AuPRC + AP

Run the same v089 native evaluation command already used by your server, for example:

```bash
python scripts/nbs/train_nbs_full_task_v089.py \
  --stage evaluate \
  --config /ACTUAL/v089_phase_config.json \
  --work-dir /ACTUAL/v089_phase_work_dir \
  --checkpoint /ACTUAL/v089_phase_work_dir/nbs_epoch25.pt \
  --input-dir /ACTUAL/VALID_ind_test_prepared_inputs \
  --metadata-file data/unidata_with_exp_train_pseudo.pkl \
  --device cuda:0
```

`eval_nbs_full_task_v089.py` now reports, for `NBS_final` and `backbone_base` by default:

- `standard_protein_fmax`: **protein-centric Fmax (%)**, thresholds 0.000, 0.001, ..., 1.000 and strict `probability > threshold`;
- `standard_micro_pr_auc`: **exact tied-score grouped trapezoidal MICRO PR-AUC (%)**, not AP;
- `standard_micro_ap`: **non-interpolated MICRO average precision (%)**.

When supplied, independently identified Expert and Stage-1 Modelout references are **evaluation-only**; neither enters the standalone NBS forward pass. Results go to the native `eval_step<PHASE_STEP>/full/metrics/nbs_w2s_comparison.json` and `...tsv`. The manifest also includes `phase_identity` and verified checkpoint and input SHA256 identity.

To reproduce the old Fmax-only evaluator, run the native evaluator directly with its original arguments and `--fmax-only`. The existing `train_nbs_full_task_v089.py --stage evaluate` launcher does not forward `--fmax-only`; it now uses the enhanced default (without changing forward).

### Do not infer cumulative training progress from `step`

`eval_step600` means **phase-local** step 600. The lifetime completed weak epoch is `manifest.epoch` / `manifest.cumulative_completed_epochs` (the latter counts fully completed epochs). A new phase may start from a v088 checkpoint with reinitialized optimizer and RNG, so it must be described using `phase_origin` and `phase_plan`. The patch preserves the original `PhasePlanV089` validator.

## 5. Dark–Weak v089 native graph input audit

After producing **genuine new native Dark–Weak input caches** using the existing v089-compatible preparation pipeline (not by renaming an unrelated ind_test input folder), run:

```bash
python scripts/dark_weak/audit_v089_graph_inputs.py \
  --config /ACTUAL/v089_phase_config.json \
  --input-dir /ACTUAL/dark_weak_native_input_dir \
  --target-original-global-ids /ACTUAL/target_original_registry_global_ids.npy \
  --all-proteins \
  --device cpu \
  --out outputs/dark_weak_eval/v089_graph_audit.json
```

`FullTaskDataV088` is deliberately retained; the loader is `local_loader_v089`. The audit checks real sampled negative-ID ghost queries, cosine-only ghost edges, the absence of future labels, and optionally the target's **original training-registry node ID** appearing as a support node. Without original global IDs it does not certify target-self isolation. Other historical weak graph membership and pseudo-supervision also mean that this is a **seen-weak / transductive history-recovery** protocol, not automatically a protein-level zero-shot test.

## 6. Export the v089 checkpoint on Dark–Weak (when native caches exist)

```bash
python scripts/dark_weak/export_nbs_v089_predictions.py \
  --task bp \
  --config /ACTUAL/v089_phase_config.json \
  --checkpoint /ACTUAL/v089_phase_work_dir/nbs_epoch25.pt \
  --work-dir outputs/dark_weak_eval/v089_native_export \
  --input-dir /ACTUAL/dark_weak_native_input_dir \
  --expected-protein-ids data/dark_weak_test/eval_inference_ids/bp_protein_ids.txt \
  --go-ids data/go_terms_bp.txt \
  --graph-audit-json outputs/dark_weak_eval/v089_graph_audit.json \
  --out outputs/dark_weak_eval/v089_adapter.json
```

The adapter **calls the unchanged native v089 `train_nbs_full_task_v089.py --stage export`**. It does not implement graph sampling or logits itself. It then validates native checkpoint/output/input SHA256 hashes, complete classifier GO width, original GO column order, all-forward-source flags, **full v089 `PhasePlanV089` identity**, and `0.8.8` model architecture. It writes the original native probability `.npy` path plus metadata in `v089_adapter.json`.

A synthetic MSA/index view is not sufficient to create the GNN graph input caches: the project-native inherited input preparation code is still required, along with frozen target representation, Backbone scores, candidates and retrieval neighbors. The exporter deliberately fails if those native inputs are not present.

## 7. Unified Dark–Weak Stage-1 / v089 scoring

The original v0.4.3 scorer already accepts any correctly ordered complete GO probability `.npy`; it does not depend on NBS checkpoint version. v0.4.4 adds an optional **verifiable native adapter binding**:

```bash
python scripts/dark_weak/run_complete_evaluation.py \
  --namespace bp --panel future \
  --metadata data/unidata_with_dark_weak_eval.pkl \
  --unified-root data/dark_weak_test/unified \
  --metric-label-view propagated \
  --ranking-label-view direct \
  --prediction backbone=/ACTUAL/backbone_bp.f32.npy \
  --prediction-ids backbone=/ACTUAL/target_ids.txt \
  --prediction-go-ids backbone=data/go_terms_bp.txt \
  --prediction nbs=/ACTUAL/nbs_ind_test_prob.f32.npy \
  --prediction-ids nbs=/ACTUAL/protein_ids.txt \
  --prediction-go-ids nbs=data/go_terms_bp.txt \
  --native-nbs-adapter nbs=outputs/dark_weak_eval/v089_adapter.json \
  --reference backbone \
  --metric-scope unified \
  --bootstrap 1000 \
  --out outputs/dark_weak_eval/unified_v089
```

The optional `--native-nbs-adapter` binding refuses changed checkpoint bytes, predicted probability bytes, input manifest bytes, output phase identity and mismatched prediction/ID/GO paths. The verified v089 phase/provenance then appears in the unified manifest. For v088, continue to use the original `export_nbs_v088_predictions.py`; it remains untouched.

The Fmax/AuPRC/AP **all** invoke `nbs_pg.full_task_metrics_v085.compute_standard_metrics`. `prop_annotations` is complete GO ancestor closure; the direct GO `annotations` is kept separately for ranking. All GO classifier columns (including historical aliases) retain original model order. Unknown labels are **not known negatives**, making these summary metrics limited-label proxies especially for future novel functions. Trapezoidal PR-AUC under complete score ties can be much larger than AP; report both.

For BP, the 2022 rule-derived historical InterPro2GO panel has ~6,501 labeled proteins and thus ~138 million protein–GO score cells; this exceeds the v0.4.3 default 80M safety limit. `--panel history` on BP requires `--allow-large-metrics` with sufficient RAM, or separate restricted cohorts / `--metric-scope ranking_only` first. The original v085 exact micro rank sorting will allocate a large index array in addition to dense scores.

## 8. Testing status and strict boundaries

- Tested: 31 local unit/regression tests passed, 1 environment-dependent test skipped. The added 5 v089 tests run native v085 metrics on a genuine BP-width synthetic matrix and compare Fmax with the unchanged original v089 Fmax-only mode. Phase tampering, checkpoint tampering and forged adapter lineage are rejected.
- Not tested here: CUDA runs on actual v089 checkpoint, full production GNN feature/input caches, all Dark–Weak proteins, live GPU graph audits.
- Source identity required: the original v089 files and the original v085 metric kernel must remain exactly as installed for any pretrained checkpoint whose implementation hashes bind them.

