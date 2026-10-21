#!/usr/bin/env bash
# Run from any directory. All experiment settings can be overridden here or
# through the environment; the JSON config remains the model/loss definition.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_ROOT"
ACTION="${1:-pilot}"
VARIANT="${2:-${VARIANT:-graph}}"
case "$VARIANT" in graph|local|no_graph) ;; *) echo 'Variant must be graph, local or no_graph' >&2; exit 2;; esac
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.3_${VARIANT}.json}"
WORK_ROOT="${WORK_ROOT:-outputs/latence_nbs_experiments/v083}"
if [[ "$ACTION" == smoke ]]; then
  WORK_DIR="${WORK_DIR:-$WORK_ROOT/smoke_${VARIANT}}"
else
  WORK_DIR="${WORK_DIR:-$WORK_ROOT/$VARIANT}"
fi
NUM_GPUS="${NUM_GPUS:-2}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-4000}"
INPUT_DIR="${INPUT_DIR:-outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2}"
METADATA_FILE="${METADATA_FILE:-/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl}"
ABLATION="${ABLATION:-full}"
METRIC_BACKEND="${METRIC_BACKEND:-stage1}"
AUPRC_MODE="${AUPRC_MODE:-exact}"
STAGE1_REFERENCE_DIR="${STAGE1_REFERENCE_DIR:-$INPUT_DIR/stage1_references_v081}"
REFERENCES_READY=0
RUNNER="scripts/nbs/train_nbs_full_task_v083.py"
COMMON=(--config "$CONFIG" --work-dir "$WORK_DIR" --device "$DEVICE")

prepare_references() {
  local extra=()
  # EXTERNAL_PROB_PATH is the original Stage1 source; EXPERT_PROB is an
  # already-exported evaluation reference. They have different row contracts.
  local external="${STAGE1_EXTERNAL_PROB_PATH:-${EXTERNAL_PROB_PATH:-}}"
  if [[ -n "$external" ]]; then extra+=(--external-prob-path "$external"); fi
  if [[ -n "${STAGE1_CHECKPOINT:-}" ]]; then extra+=(--stage1-checkpoint "$STAGE1_CHECKPOINT"); fi
  if [[ -n "${STAGE1_EXTERNAL_PROTEIN_IDS:-}" ]]; then extra+=(--external-protein-ids "$STAGE1_EXTERNAL_PROTEIN_IDS"); fi
  if [[ -n "${STAGE1_EXTERNAL_GO_IDS:-}" ]]; then extra+=(--external-go-ids "$STAGE1_EXTERNAL_GO_IDS"); fi
  if [[ -n "${STAGE1_MSA_INDEX:-}" ]]; then extra+=(--msa-index "$STAGE1_MSA_INDEX"); fi
  if [[ -n "${STAGE1_MODEL_CONFIG:-}" ]]; then extra+=(--stage1-model-config "$STAGE1_MODEL_CONFIG"); fi
  if [[ -n "${STAGE1_TRAIN_ARGS_JSON:-}" ]]; then extra+=(--train-args-json "$STAGE1_TRAIN_ARGS_JSON"); fi
  if [[ -n "${STAGE1_DIAGNOSTIC_SCRIPT:-}" ]]; then extra+=(--diagnostic-script "$STAGE1_DIAGNOSTIC_SCRIPT"); fi
  if [[ "${STAGE1_NO_AMP:-0}" == 1 ]]; then extra+=(--no-amp); fi
  "$PYTHON_BIN" scripts/nbs/prepare_nbs_stage1_references_v081.py \
    --project-root "$PROJECT_ROOT" --task "${TASK:-bp}" --input-dir "$INPUT_DIR" \
    --metadata-file "$METADATA_FILE" --output-dir "$STAGE1_REFERENCE_DIR" \
    --device "${STAGE1_EVAL_DEVICE:-$DEVICE}" --batch-size "${STAGE1_EVAL_BATCH_SIZE:-8}" \
    --min-free-gpu-gb "${STAGE1_MIN_FREE_GPU_GB:-8}" \
    --amp-dtype "${STAGE1_AMP_DTYPE:-bfloat16}" \
    --cache-policy "${STAGE1_REFERENCE_CACHE_POLICY:-reuse}" "${extra[@]}"
  EXPERT_PROB="$STAGE1_REFERENCE_DIR/expert_prob.f32.npy"
  MODELOUT_PROB="$STAGE1_REFERENCE_DIR/stage1_modelout.f32.npy"
  EXPERT_PROTEIN_IDS="$STAGE1_REFERENCE_DIR/protein_ids.txt"
  MODELOUT_PROTEIN_IDS="$EXPERT_PROTEIN_IDS"
  EXPERT_GO_IDS="$STAGE1_REFERENCE_DIR/go_ids.txt"
  MODELOUT_GO_IDS="$EXPERT_GO_IDS"
  STAGE1_REFERENCE_MANIFEST="$STAGE1_REFERENCE_DIR/stage1_reference_manifest.json"
  REFERENCES_READY=1
}

ensure_references() {
  if [[ "$REFERENCES_READY" == 1 ]]; then return; fi
  if [[ -n "${EXPERT_PROB:-}" && -n "${MODELOUT_PROB:-}" ]]; then
    REFERENCES_READY=1
    return
  fi
  if [[ "${ALLOW_MISSING_REFERENCES:-0}" == 1 ]]; then return; fi
  if [[ -n "${EXPERT_PROB:-}" || -n "${MODELOUT_PROB:-}" ]]; then
    echo 'Supply both EXPERT_PROB and MODELOUT_PROB, or unset both for automatic Stage1 reference export.' >&2
    return 2
  fi
  if [[ "${AUTO_REFERENCES:-1}" != 1 ]]; then
    echo 'AUTO_REFERENCES=0: supply EXPERT_PROB and MODELOUT_PROB, or enable automatic export.' >&2
    return 2
  fi
  prepare_references
}

run_training() {
  "$PYTHON_BIN" "$RUNNER" --stage prepare "${COMMON[@]}" --backend torch
  local train_steps="$STEPS"
  if [[ "$ACTION" == smoke ]]; then train_steps="${SMOKE_STEPS:-20}"; fi
  local extra=(--steps "$train_steps")
  if [[ -n "${RESUME:-}" ]]; then extra+=(--resume "$RESUME"); fi
  if [[ "$NUM_GPUS" == 1 ]]; then
    "$PYTHON_BIN" "$RUNNER" --stage train "${COMMON[@]}" "${extra[@]}"
  else
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
      "$RUNNER" --stage train "${COMMON[@]}" "${extra[@]}"
  fi
}

run_evaluation() {
  local checkpoint="$1"
  [[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; return 2; }
  # The Stage1 process exits and releases GPU memory before NBS is loaded.
  ensure_references
  local extra=()
  if [[ -n "${EXPERT_PROB:-}" ]]; then extra+=(--expert-prob "$EXPERT_PROB"); fi
  if [[ -n "${MODELOUT_PROB:-}" ]]; then extra+=(--modelout-prob "$MODELOUT_PROB"); fi
  if [[ -n "${EXPERT_PROTEIN_IDS:-}" ]]; then extra+=(--expert-protein-ids "$EXPERT_PROTEIN_IDS"); fi
  if [[ -n "${MODELOUT_PROTEIN_IDS:-}" ]]; then extra+=(--modelout-protein-ids "$MODELOUT_PROTEIN_IDS"); fi
  if [[ -n "${EXPERT_GO_IDS:-}" ]]; then extra+=(--expert-go-ids "$EXPERT_GO_IDS"); fi
  if [[ -n "${MODELOUT_GO_IDS:-}" ]]; then extra+=(--modelout-go-ids "$MODELOUT_GO_IDS"); fi
  if [[ -n "${STAGE1_REFERENCE_MANIFEST:-}" ]]; then extra+=(--stage1-reference-manifest "$STAGE1_REFERENCE_MANIFEST"); fi
  if [[ "${REFERENCES_ALIGNED:-0}" == 1 ]]; then extra+=(--references-aligned-to-input); fi
  if [[ "${ALLOW_MISSING_REFERENCES:-0}" == 1 ]]; then
    extra+=(--allow-missing-references)
  elif [[ -z "${EXPERT_PROB:-}" || -z "${MODELOUT_PROB:-}" ]]; then
    echo 'Evaluation needs distinct EXPERT_PROB and MODELOUT_PROB. For incomplete diagnostics only, set ALLOW_MISSING_REFERENCES=1.' >&2
    return 2
  fi
  "$PYTHON_BIN" "$RUNNER" --stage evaluate "${COMMON[@]}" --checkpoint "$checkpoint" \
    --input-dir "$INPUT_DIR" --metadata-file "$METADATA_FILE" --ablation "$ABLATION" \
    --metric-backend "$METRIC_BACKEND" --auprc-mode "$AUPRC_MODE" "${extra[@]}"
}

case "$ACTION" in
  references) prepare_references ;;
  prepare) "$PYTHON_BIN" "$RUNNER" --stage prepare "${COMMON[@]}" --backend torch ;;
  smoke|pilot|train) run_training ;;
  evaluate) run_evaluation "${CHECKPOINT:-$WORK_DIR/latest.pt}" ;;
  diagnose)
    checkpoint="${CHECKPOINT:-$WORK_DIR/latest.pt}"
    for ABLATION in full weak_off core_off pp_off graph_off go_shuffle; do
      run_evaluation "$checkpoint"
    done
    "$PYTHON_BIN" scripts/nbs/eval_nbs_full_task_v083.py \
      --summarize-work-dir "$WORK_DIR" --summary-checkpoint "$checkpoint"
    ;;
  series)
    read -r -a checkpoint_steps <<< "${CHECKPOINT_STEPS:-600 2000 4000}"
    for step in "${checkpoint_steps[@]}"; do
      [[ "$step" =~ ^[0-9]+$ ]] || { echo 'CHECKPOINT_STEPS must contain integers separated by spaces' >&2; exit 2; }
      run_evaluation "$WORK_DIR/nbs_step${step}.pt"
    done
    ;;
  *) echo 'Usage: bash scripts/nbs/run_nbs_v083.sh {prepare|references|smoke|pilot|train|evaluate|series|diagnose} {graph|local|no_graph}' >&2; exit 2 ;;
esac
