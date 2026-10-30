#!/usr/bin/env bash
# Frozen v089 epoch60 -> cached representations -> new shared head.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd -- "$PROJECT_ROOT"
[[ "$#" -le 1 ]] || { echo 'Usage: run_nbs_shared_head_v091.sh {cache|smoke|train|convert|evaluate|joint|evaluate-joint}' >&2; exit 2; }
ACTION="${1:-train}"
case "$ACTION" in cache|smoke|train|convert|evaluate|joint|evaluate-joint) ;; *) echo "Unsupported stage: $ACTION" >&2; exit 2 ;; esac
[[ "${NUM_GPUS:-1}" == 1 ]] || { echo 'This frozen-encoder experiment uses one GPU, never torchrun.' >&2; exit 2; }
[[ "${DEVICE:-cuda:0}" == cuda:0 ]] || { echo 'Use DEVICE=cuda:0 and select the physical GPU with CUDA_VISIBLE_DEVICES.' >&2; exit 2; }
[[ -z "${INIT_FROM:-}${ENCODER_FROM:-}${STEPS:-}${STOP_EPOCH:-}${CHECKPOINT:-}${PRESET:-}" ]] || {
  echo 'Unset old-run INIT_FROM/ENCODER_FROM/STEPS/STOP_EPOCH/CHECKPOINT/PRESET; source epoch60 is pinned in the research config.' >&2; exit 2;
}
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-nbs_models/nbs_protein_go/configs/bp_shared_head_v091_from_v089_ep60.json}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[^,[:space:]]+$ && "$CUDA_VISIBLE_DEVICES" != -1 ]] || {
  echo 'Select exactly one GPU index or GPU UUID.' >&2; exit 2;
}
if [[ "$ACTION" == smoke ]]; then
  WORK_DIR="${WORK_DIR:-outputs/latence_nbs_experiments/v091/from_v089_ep60_shared_head_smoke}"
else
  WORK_DIR="${WORK_DIR:-outputs/latence_nbs_experiments/v091/from_v089_ep60_shared_head}"
fi
CHILD_ENV=(env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK
  -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT
  -u TORCHELASTIC_RUN_ID -u TORCHELASTIC_RESTART_COUNT -u TORCHELASTIC_MAX_RESTARTS
  -u TORCHELASTIC_ERROR_FILE -u TORCHELASTIC_USE_AGENT_STORE
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}" "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}"
  "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}")
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
"${CHILD_ENV[@]}" "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as stream:
    cfg = json.load(stream)
if cfg.get('schema') != 'nbs-shared-head-v091' or type(cfg.get('source_epoch')) is not int or cfg['source_epoch'] != 60:
    raise SystemExit('Use the dedicated shared-head research config sourced from v089 epoch60.')
PY
case "$ACTION" in
  cache)
    [[ -z "${RESUME:-}${STOP_STEP:-}${HEAD_CHECKPOINT:-}" ]] || { echo 'cache does not use training resume/stop/head-checkpoint.' >&2; exit 2; }
    # Separate launcher lock; cache builder also protects its data directory.
    mkdir -p -- outputs/latence_nbs_experiments/v091
    exec 9>outputs/latence_nbs_experiments/v091/.shared_ep60_cache_launcher.lock
    flock -n 9 || { echo 'Another epoch60 cache launcher is running.' >&2; exit 2; }
    extra=()
    if [[ -n "${CACHE_QUERY_BATCH:-}" ]]; then extra+=(--query-batch "$CACHE_QUERY_BATCH"); fi
    "${CHILD_ENV[@]}" "$PYTHON_BIN" "$SCRIPT_DIR/cache_nbs_shared_head_v091.py" --config "$CONFIG" --device "$DEVICE" "${extra[@]}" ;;
  smoke|train)
    [[ -z "${HEAD_CHECKPOINT:-}" ]] || { echo 'HEAD_CHECKPOINT is only used for evaluation.' >&2; exit 2; }
    extra=()
    if [[ "$ACTION" == smoke ]]; then
      [[ -z "${RESUME:-}${STOP_STEP:-}" ]] || { echo 'smoke is a new isolated head run; use SMOKE_STEPS.' >&2; exit 2; }
      extra+=(--stop-step "${SMOKE_STEPS:-3}")
    else
      if [[ -n "${RESUME:-}" ]]; then extra+=(--resume "$RESUME"); fi
      if [[ -n "${STOP_STEP:-}" ]]; then extra+=(--stop-step "$STOP_STEP"); fi
    fi
    # Keep the lock outside the new directory: the trainer intentionally
    # rejects any pre-existing contents for a fresh phase.
    WORK_DIR="${WORK_DIR%/}"
    mkdir -p -- "$(dirname -- "$WORK_DIR")"
    exec 9>"${WORK_DIR}.launcher.lock"
    flock -n 9 || { echo 'Another shared-head launcher is writing this WORK_DIR.' >&2; exit 2; }
    "${CHILD_ENV[@]}" "$PYTHON_BIN" "$SCRIPT_DIR/train_nbs_shared_head_v091.py" \
      --config "$CONFIG" --work-dir "$WORK_DIR" --device "$DEVICE" "${extra[@]}" ;;
  convert)
    [[ -z "${RESUME:-}${STOP_STEP:-}" ]] || { echo 'convert does not use RESUME/STOP_STEP.' >&2; exit 2; }
    [[ -n "${HEAD_CHECKPOINT:-}" && -f "$HEAD_CHECKPOINT" ]] || { echo 'Set HEAD_CHECKPOINT to shared_head_step<number>.pt.' >&2; exit 2; }
    CONVERT_DIR="${CONVERT_DIR:-$WORK_DIR/converted_$(basename -- "$HEAD_CHECKPOINT" .pt)}"
    mkdir -p -- "$(dirname -- "$CONVERT_DIR")"
    exec 9>"${CONVERT_DIR}.launcher.lock"
    flock -n 9 || { echo 'Another conversion writes this directory.' >&2; exit 2; }
    "${CHILD_ENV[@]}" "$PYTHON_BIN" "$SCRIPT_DIR/convert_nbs_shared_head_v091.py" \
      --config "$CONFIG" --head-checkpoint "$HEAD_CHECKPOINT" --device "$DEVICE" --output-dir "$CONVERT_DIR" ;;
  joint)
    [[ -z "${HEAD_CHECKPOINT:-}" ]] || { echo 'joint requires HYBRID_INIT or RESUME, not HEAD_CHECKPOINT.' >&2; exit 2; }
    JOINT_WORK_DIR="${JOINT_WORK_DIR:-outputs/latence_nbs_experiments/v091/from_v089_ep60_joint}"
    extra=()
    if [[ -n "${RESUME:-}" ]]; then
      [[ -z "${HYBRID_INIT:-}" ]] || { echo 'Use only RESUME when continuing joint.' >&2; exit 2; }
      extra+=(--resume "$RESUME")
    else
      [[ -n "${HYBRID_INIT:-}" && -f "$HYBRID_INIT" ]] || { echo 'Set HYBRID_INIT to checked hybrid_init.pt.' >&2; exit 2; }
      extra+=(--initialization "$HYBRID_INIT")
    fi
    if [[ -n "${STOP_STEP:-}" ]]; then extra+=(--stop-step "$STOP_STEP"); fi
    mkdir -p -- "$(dirname -- "$JOINT_WORK_DIR")"
    exec 9>"${JOINT_WORK_DIR}.launcher.lock"
    flock -n 9 || { echo 'Another joint writer is running.' >&2; exit 2; }
    "${CHILD_ENV[@]}" "$PYTHON_BIN" "$SCRIPT_DIR/train_nbs_hybrid_v091.py" \
      --config "$CONFIG" --work-dir "$JOINT_WORK_DIR" --device "$DEVICE" "${extra[@]}" ;;
  evaluate|evaluate-joint)
    [[ -z "${RESUME:-}${STOP_STEP:-}" ]] || { echo 'evaluate does not use RESUME/STOP_STEP.' >&2; exit 2; }
    extra=()
    if [[ "$ACTION" == evaluate ]]; then
      [[ -n "${HEAD_CHECKPOINT:-}" && -f "$HEAD_CHECKPOINT" ]] || { echo 'Set HEAD_CHECKPOINT to shared_head_step<number>.pt.' >&2; exit 2; }
      extra+=(--head-checkpoint "$HEAD_CHECKPOINT")
      EVAL_DIR="${EVAL_DIR:-$WORK_DIR/eval_$(basename -- "$HEAD_CHECKPOINT" .pt)}"
    else
      [[ -n "${JOINT_CHECKPOINT:-}" && -f "$JOINT_CHECKPOINT" ]] || { echo 'Set JOINT_CHECKPOINT to a fixed hybrid_step<number>.pt or hybrid_init.pt.' >&2; exit 2; }
      extra+=(--hybrid-checkpoint "$JOINT_CHECKPOINT")
      EVAL_DIR="${EVAL_DIR:-$(dirname -- "$JOINT_CHECKPOINT")/eval_$(basename -- "$JOINT_CHECKPOINT" .pt)}"
    fi
    mkdir -p -- "$(dirname -- "$EVAL_DIR")"
    exec 9>"${EVAL_DIR}.launcher.lock"
    flock -n 9 || { echo 'Another evaluation writes this directory.' >&2; exit 2; }
    "${CHILD_ENV[@]}" "$PYTHON_BIN" "$SCRIPT_DIR/eval_nbs_shared_head_v091.py" \
      --config "$CONFIG" "${extra[@]}" --device "$DEVICE" \
      --input-dir "${INPUT_DIR:-outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2}" \
      --metadata-file "${METADATA_FILE:-/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl}" \
      --output-dir "$EVAL_DIR" ;;
esac
