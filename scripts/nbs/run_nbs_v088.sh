#!/usr/bin/env bash
# Single-model v088 run; independent evaluation is always explicitly requested.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_ROOT"
ACTION="${1:-train}"
PRESET="${2:-${PRESET:-hetero_tuned}}"
[[ "$PRESET" == hetero_tuned ]] || { echo 'v088 has one preset: hetero_tuned' >&2; exit 2; }
case "$ACTION" in prepare|smoke|train|export|evaluate) ;; *) echo 'Usage: run_nbs_v088.sh {prepare|smoke|train|export|evaluate} [hetero_tuned]; ablations/series are paused' >&2; exit 2;; esac
[[ "${ABLATION:-full}" == full ]] || { echo 'v088 permits ABLATION=full only' >&2; exit 2; }
[[ -z "${STEPS:-}" ]] || { echo 'Use STOP_STEP as a cumulative early endpoint, not STEPS' >&2; exit 2; }
[[ -z "${STOP_STEP:-}" || -z "${STOP_EPOCH:-}" ]] || { echo 'STOP_STEP and STOP_EPOCH are mutually exclusive' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.8_hetero_tuned.json}"
WORK_ROOT="${WORK_ROOT:-outputs/latence_nbs_experiments/v088}"
if [[ "$ACTION" == smoke ]]; then WORK_DIR="${WORK_DIR:-$WORK_ROOT/smoke_hetero_tuned}"; else WORK_DIR="${WORK_DIR:-$WORK_ROOT/hetero_tuned}"; fi
NUM_GPUS="${NUM_GPUS:-2}"
DEVICE="${DEVICE:-cuda:0}"
INPUT_DIR="${INPUT_DIR:-outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2}"
METADATA_FILE="${METADATA_FILE:-/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl}"
RUNNER="scripts/nbs/train_nbs_full_task_v088.py"
COMMON=(--config "$CONFIG" --work-dir "$WORK_DIR" --device "$DEVICE")
if [[ -n "${DEV_MANIFEST:-}" ]]; then COMMON+=(--development-manifest "$DEV_MANIFEST"); fi
prepare() { "$PYTHON_BIN" "$RUNNER" --stage prepare "${COMMON[@]}" --backend torch; }
case "$ACTION" in
  prepare) prepare ;;
  smoke|train)
    extra=()
    if [[ "$ACTION" == smoke ]]; then
      [[ -z "${RESUME:-}" ]] || { echo 'smoke starts fresh in its isolated folder' >&2; exit 2; }
      extra+=(--stop-step "${SMOKE_STEPS:-20}")
    else
      if [[ -n "${STOP_STEP:-}" ]]; then extra+=(--stop-step "$STOP_STEP"); fi
      if [[ -n "${STOP_EPOCH:-}" ]]; then extra+=(--stop-epoch "$STOP_EPOCH"); fi
      if [[ -n "${RESUME:-}" ]]; then extra+=(--resume "$RESUME"); fi
    fi
    prepare
    if [[ "$NUM_GPUS" == 1 ]]; then
      "$PYTHON_BIN" "$RUNNER" --stage train "${COMMON[@]}" "${extra[@]}"
    else
      "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
        "$RUNNER" --stage train "${COMMON[@]}" "${extra[@]}"
    fi
    ;;
  export|evaluate)
    checkpoint="${CHECKPOINT:-$WORK_DIR/latest.pt}"
    [[ -f "$checkpoint" ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }
    extra=()
    if [[ "$ACTION" == evaluate ]]; then
      extra+=(--metric-backend "${METRIC_BACKEND:-stage1}" --auprc-mode "${AUPRC_MODE:-exact}" --allow-missing-references)
      for name in EXPERT_PROB MODELOUT_PROB EXPERT_PROTEIN_IDS MODELOUT_PROTEIN_IDS EXPERT_GO_IDS MODELOUT_GO_IDS STAGE1_REFERENCE_MANIFEST; do
        if [[ -n "${!name:-}" ]]; then key="${name,,}"; extra+=("--${key//_/-}" "${!name}"); fi
      done
      if [[ "${REFERENCES_ALIGNED:-0}" == 1 ]]; then extra+=(--references-aligned-to-input); fi
    fi
    "$PYTHON_BIN" "$RUNNER" --stage "$ACTION" "${COMMON[@]}" --checkpoint "$checkpoint" \
      --input-dir "$INPUT_DIR" --metadata-file "$METADATA_FILE" --ablation full "${extra[@]}"
    ;;
esac
