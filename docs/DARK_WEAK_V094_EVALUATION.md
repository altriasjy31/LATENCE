# LATENCE NBS v0.9.4 — additive Dark–Weak / independent-test unified evaluation

## 0. Scope and freeze policy

This is an **additive adapter** for the supplied `latence_nbs_v094_mysrc.zip`. It does **not** replace or edit:

- `scripts/nbs/run_nbs_v094.py`, `train_nbs_full_task_v094.py`, `predict_nbs_go_v094.py`;
- `nbs_pg/full_task_model_v094.py`, `full_task_data_v094.py`, `full_task_metrics_v085.py`;
- original `ind_test` reporting `metrics_v094/summary.json`, earlier v088/v089 acceptances, checkpoints or training state;
- existing `scripts/dark_weak/run_complete_evaluation.py`, Stage-1 exporter or Dark–Weak metadata.

Use **new output directories**. The only new files installed by this ZIP have names `v094_*` plus this documentation and one test module. If the package is applied to an otherwise different LATENCE checkout, verify availability and semantics of the existing Dark–Weak v0.4.3+ scripts before running.

All primary scores use the **unmodified** v085 `compute_standard_metrics`: `standard_protein_fmax` (protein-centric, strict `>`, grid step 0.001); `standard_micro_pr_auc` (micro trapezoidal PR-AUC); `standard_micro_ap` (micro non-interpolated AP). **AP and PR-AUC are not interchangeable**. All classifier columns and GO aliases remain in original order. For future annotation panels, primary labels are `prop_annotations` (direct GO plus complete representable ancestors); ranking uses direct positive `annotations`. Unobserved labels are unknown, not confirmed negatives.

## 1. Install (additive only)

At LATENCE project root:

```bash
unzip -l /path/LATENCE_dark_weak_eval_v0.5.0_v094_additive.zip
unzip -n /path/LATENCE_dark_weak_eval_v0.5.0_v094_additive.zip -d .
python -m unittest discover -s tests -p 'test_dark_weak_v094_additive.py' -v
```

`unzip -n` deliberately skips any conflicting file. For an upgrade of a previous copy of **only these v094 adapter files**, compare the files and replace only intended adapter paths, never original NBS code.

## 2. Original v094 `ind_test` remains unchanged

Your original command remains the acceptance/standard validation path:

```bash
# Use original native configuration, checkpoint, work directory and input-dir.
python scripts/nbs/run_nbs_v094.py evaluate --help
```

Do NOT change the original `metrics_v094/summary.json`. For v085 Fmax/AuPRC/AP in an additional JSON, use a **finished native `--stage evaluate` output**:

```bash
python scripts/dark_weak/v094_ind_test_supplement.py \
  --task bp \
  --config /PATH/TO/v094/resolved_config.json \
  --metadata data/unidata_with_exp_train_pseudo.pkl \
  --split ind_test \
  --checkpoint /PATH/TO/v094/nbs_epoch20.pt \
  --input-dir /PATH/TO/ORIGINAL_IND_TEST_INPUT_DIR \
  --prediction-manifest /PATH/TO/v094/eval_stepNNN/full/nbs_full_task_prediction_manifest.json \
  --go-ids /PATH/TO/ORIGINAL_ORDERED_GO_IDS.txt \
  --expected-epoch 20 \
  --out outputs/dark_weak_eval/v094_ind_test_stepNNN_v085.json
```

Here `NNN` is the checkpoint's **actual saved local step** (not guessed from cumulative epoch). This supplement reads checkpoint and input identity, reuses saved probabilities, and computes v085 metrics; if the original native summary exists, Fmax/AP parity is checked. E/M can be added as evaluation-only references via triplets of `--reference E=... --reference-ids E=... --reference-go-ids E=...` and likewise `M`.

## 3. Build label-free Dark–Weak inference targets (existing tools)

This step is only required once per frozen benchmark version. Preserve original labels and historical 2022 provenance. If already built, verify path and skip.

```bash
python scripts/dark_weak/prepare_inference_split.py \
  --metadata data/unidata_with_dark_weak_eval.pkl \
  --unified-root data/dark_weak_test/unified \
  --source-splits dark_weak_test_direct dark_weak_test_exp_direct dark_weak_test_ipr2go_direct \
  --output data/unidata_with_dark_weak_inference.pkl \
  --id-dir data/dark_weak_test/eval_inference_ids
```

For the **first prospective run**, the explicit `--source-splits` selects the smaller future-positive cohort; the original builder defaults to including a larger historical InterPro2GO group, which is unnecessary for this first GPU smoke test. If you later want complete historical coverage, create a separate, versioned inference metadata/output directory and use the appropriate historical splits.

The new `dark_weak_inference` metadata split has **empty** `annotations` and `prop_annotations`, and the label-bearing `dark_weak_test*` splits are removed **from the inference pickle only**. Evaluate against the original `data/unidata_with_dark_weak_eval.pkl` later. This is critical to avoid passing 2025 future GO labels into inference.

Create an MSA index **view** over existing 2022-compatible binary MSA shards, with filenames corresponding to the exact original training/weak history. Confirm your server's actual source-index paths:

```bash
python scripts/dark_weak/prepare_msa_index_view.py \
  --target-ids data/dark_weak_test/eval_inference_ids/bp_protein_ids.txt \
  --source-index /PATH/TO/HISTORICAL_WEAK_MSA/index.pkl \
  --source-index /PATH/TO/HISTORICAL_CORE_MSA/index.pkl \
  --out-index data/dark_weak_test/msa_v094_bp/index.pkl
```

It references existing binary shards without copying the MSA. If proteins are missing, it **fails**, rather than silently shrinking the test population. If a protein occurs in several sources, verify which shard corresponds to the correct historical sequence; use the explicit `--duplicate-policy first` only after checking source priority. The view code records `source_cutoff_verified=false`; the researcher must independently audit the **MSA search database temporal cutoff** before claiming entirely 2022-frozen evidence.

## 4. Use the unmodified native v094 preparation code

**Do not create a parallel v094 input generator.** The supplied repository's `prepare_nbs_ind_test_inputs.py` already supports an arbitrary `--mode`, a metadata pickle and a custom MSA index. It writes `ind_test_*` **file names** as a fixed format, which does not imply the contents are the original `ind_test` proteins.

```bash
python scripts/nbs/prepare_nbs_ind_test_inputs.py \
  --task bp \
  --mode dark_weak_inference \
  --metadata-file data/unidata_with_dark_weak_inference.pkl \
  --msa-index data/dark_weak_test/msa_v094_bp/index.pkl \
  --stage1-checkpoint /PATH/TO/FROZEN_STAGE1_CHECKPOINT.pt \
  --data-root /PATH/TO/V094_NBS_DATA_ROOT \
  --candidate-selector-scope full_task \
  --candidate-topk 512 \
  --pp-topk 8 \
  --device cuda:0 \
  --output-dir outputs/dark_weak_eval/v094_inputs_bp
```

Use `--data-root` = the `data.root` actually recorded in the **v094 checkpoint-matching config**; override the suggested `--pp-topk 8` with a number satisfying the actual graph sampler's `core_neighbors`/retrieval requirement. The builder generates: `protein_ids.txt`, `ind_test_repr.f16.npy`, `backbone_ind_test_prob.f16.npy`, `candidate_go_index.i32.npy`, `candidate_edge_attr.f32.npy`, `test_core_neighbors.i32.npy`, `test_core_edge_attr.f32.npy`, `ind_test_input_manifest.json`. It records no consumption of GO target labels and rejects wrong sequence/GO/index alignments.

If a previous output directory has cached inputs, its signature must match; use a new output directory rather than overwriting another run's cache. Stage-1 `Backbone` from this preparation can be used as a comparison reference after audit of clipping/output identity. Stage-1 Expert and Modelout must be exported separately with the existing `scripts/dark_weak/export_stage1_predictions.py`, using **exactly the same** 2022 target proteins and frozen GO order.

## 5. Audit graph inputs (strongly recommended)

```bash
python scripts/dark_weak/v094_graph_audit.py \
  --task bp \
  --config /PATH/TO/v094/resolved_config.json \
  --input-dir outputs/dark_weak_eval/v094_inputs_bp \
  --all-proteins \
  --batch-size 4 \
  --device cpu \
  --out outputs/dark_weak_eval/v094_graph_audit_bp.json
```

Optionally pass `--original-global-ids /PATH/TO/original_global_ids.i64.npy`: a vector aligned to `protein_ids.txt`, with original pre-existing NBS global protein IDs or `-1` for absent IDs. If the protein registry CSV has explicit `protein_id,protein_idx` columns, the script can derive these IDs; otherwise it records unknown and **does not claim identity isolation**. If an original weak node appears anywhere in a batch's union of graph support nodes, the audit flags a conservative risk. This is not proof of a self-loop to an individual query. The score adapter rejects a flagged audit by default; `--allow-self-support` is allowed only for explicitly **transductive/descriptive** research, never a strict inductive generalization claim.

All v094 graph audit/native export processes take the same advisory lock used by v092/v094 evaluation: `outputs/latence_nbs_experiments/v092/v092_evaluate.lock`. Do not run them concurrently with native evaluation or prepare jobs that share caches. This audit does not modify model/graph/weights; native preparers may create derived cache files according to their existing behavior.

## 6. Export NBS v094 predictions — native inference, isolated output

The adapter calls **unmodified** `scripts/nbs/train_nbs_full_task_v094.py --stage export` and then validates the original native prediction manifest and probability SHA256. It has no custom model forward and cannot be pointed at a training directory as its evaluation `--work-dir`.

```bash
python scripts/dark_weak/v094_native_adapter.py export \
  --task bp \
  --config /PATH/TO/v094/resolved_config.json \
  --checkpoint /PATH/TO/v094/nbs_epoch20.pt \
  --input-dir outputs/dark_weak_eval/v094_inputs_bp \
  --expected-protein-ids data/dark_weak_test/eval_inference_ids/bp_protein_ids.txt \
  --go-ids /PATH/TO/ORIGINAL_ORDERED_GO_IDS.txt \
  --expected-epoch 20 \
  --device cuda:0 \
  --graph-audit-json outputs/dark_weak_eval/v094_graph_audit_bp.json \
  --work-dir outputs/dark_weak_eval/v094_native_export_epoch20 \
  --out outputs/dark_weak_eval/v094_adapter_epoch20.json
```

If the original graph audit flagged historical self-support and you are deliberately measuring **seen-weak retrospective recovery**, add `--allow-self-support`, mark the protocol as descriptive/transductive, and never describe this as a new-protein induction experiment. For existing v094 exported probabilities in this **same isolated work directory** use the action `attach` instead of `export`; this avoids GPU work and does not rewrite the native report. Paths and checkpoint epochs shown are illustrative.

The adapter rejects predictions with a wrong model architecture/revision, different checkpoint, altered file hash, changed GO order, altered input MSA/graph manifest, out-of-range probabilities, or a different historical target set.

## 7. Score using the existing Dark–Weak v085 metric implementation

The wrapper verifies the v094 provenance, then delegates to **unchanged** `scripts/dark_weak/run_complete_evaluation.py`; that scorer handles matching 2022 cohort rows, multi-evidence split filters, full GO ancestor labels, optional independent ranking, and v085 metrics.

```bash
python scripts/dark_weak/v094_unified_score.py \
  --native-v094-adapter nbs=outputs/dark_weak_eval/v094_adapter_epoch20.json \
  --namespace bp \
  --panel future \
  --metadata data/unidata_with_dark_weak_eval.pkl \
  --unified-root data/dark_weak_test/unified \
  --metric-label-view propagated \
  --ranking-label-view direct \
  --prediction nbs=outputs/dark_weak_eval/v094_native_export_epoch20/eval_stepNNN/full/nbs_ind_test_prob.f32.npy \
  --prediction-ids nbs=outputs/dark_weak_eval/v094_inputs_bp/protein_ids.txt \
  --prediction-go-ids nbs=/PATH/TO/ORIGINAL_ORDERED_GO_IDS.txt \
  --prediction backbone=outputs/dark_weak_eval/v094_inputs_bp/backbone_ind_test_prob.f16.npy \
  --prediction-ids backbone=outputs/dark_weak_eval/v094_inputs_bp/protein_ids.txt \
  --prediction-go-ids backbone=/PATH/TO/ORIGINAL_ORDERED_GO_IDS.txt \
  --reference backbone \
  --bootstrap 1000 \
  --out outputs/dark_weak_eval/v094_unified_bp_epoch20
```

`NNN` is the original native checkpoint's saved `step` and must be read from the adapter JSON; use the actual file path from `adapter.probability`. E/M predictions may be appended using the same `--prediction / --prediction-ids / --prediction-go-ids` triplets and explicit frozen references. If ALL target original global IDs were resolved, no batch support hits, and complete graph audit was run, you may additionally request `--require-graph-audit`; this only checks the graph identity audit, **not** whether the weak protein appeared during training.

Output: unchanged standard benchmark `bp_complete_metrics.tsv`, `bp_complete_manifest.json`, per-evidence split `complete_metrics.json`; plus a new independent `bp_v094_provenance.json`. Existing directories are protected; use a **new `--out` per checkpoint or benchmark protocol**. The default `future` panel includes experimental, InterPro2GO and union novel labels. For large `history` panels, estimate N*G and RAM first (native v085 micro metrics sort all pairs); the original scorer enforces a metric-cell cap and requires an explicit override rather than silently switching to approximate scores.

## 8. Scientific interpretation and limitations

1. DarkWeak test contains historically sequenced and potentially NBS-weak-training-seen proteins; they are **not necessarily new protein independent test cases**. Score is historical recovery of annotations added later.
2. `prop_annotations` retains the complete 2022-ontology ancestor closure for each direct GO; `annotations` remains direct positives. Ancestor GO may have been historically known and should not be claimed as a newly discovered function.
3. The `InterPro2GO` source is automatically transferred annotation (IEA-like), not an independent experimental confirmation. Report experimental and transferred evidence separately.
4. Frozen 2022 sequences alone do not ensure strictly 2022-frozen MSA search database, retrieval evidence, or representation weights. This adapter explicitly records that MSA cutoff was **not independently verified**.
5. v094 `Stage2-only` unseen GO evaluation is **not full-pipeline zero-shot**. Stage-1 upstream representations may already encode held-out GO evidence, and the fixed GO output axis does not imply future-introduced GO inference.
6. Native v085 trapezoidal PR-AUC can be inflated by ties and sparse positives; report exact micro AP and observed positive prevalence alongside Fmax and PR-AUC.
7. The uploaded source's unrelated `tests/test_v094_runtime.py` fixture omits `probability_clip` in a mock `base_logit_store`. The original full v094 test suite has 1 fixture failure on that path; adding `probability_clip=1e-6` to the **isolated test fixture** makes the native v094 unit tests pass. The delivered addon does NOT change this native test fixture or any production file.

## 9. Validation status

- New additive code: 10 Python unittest cases (no-GPU) covering native schema/digests, wrong GO or output semantics, graph ghost identity and self-support warning, actual v085 Fmax/PR-AUC/AP from full-width synthetic BP probabilities, and `ind_test` supplementary comparisons.
- Original uploaded v094 test modules: 41 tests; one fails due to missing mock `probability_clip`; 41/41 pass after an **isolated, non-delivered** fixture-only adjustment.
- **Not run** on the user's trained GPU checkpoint, historical MSA files, or complete NBS input-cache. Real model predictions and quantitative benchmark results are not yet available.
