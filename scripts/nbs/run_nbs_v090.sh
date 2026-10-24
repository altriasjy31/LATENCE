#!/usr/bin/env bash
# v0.9.0 phase launcher: per-rank 6400 queries, fixed full-task Fmax evaluation.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_ROOT"
ACTION="${1:-train}"
PRESET="${2:-${PRESET:-ddp}}"
case "$ACTION" in
  prepare|smoke|train|export|evaluate) ;;
  *) echo 'Usage: run_nbs_v090.sh {prepare|smoke|train|export|evaluate} [single|ddp]' >&2; exit 2 ;;
esac
case "$PRESET" in
  single) EXPECTED_GPUS=1; DEFAULT_CONFIG=nbs_models/nbs_protein_go/configs/bp_full_task_v0.9.0_mb6400_single_ep100.json ;;
  ddp) EXPECTED_GPUS=2; DEFAULT_CONFIG=nbs_models/nbs_protein_go/configs/bp_full_task_v0.9.0_mb12800_ddp_ep100.json ;;
  *) echo 'v090 presets are single and ddp' >&2; exit 2 ;;
esac
[[ "${ABLATION:-full}" == full ]] || { echo 'v090 permits ABLATION=full only; separate ablations remain paused' >&2; exit 2; }
[[ -z "${INIT_FROM:-}" ]] || { echo "Use ENCODER_FROM for explicit encoder-only migration; INIT_FROM is not supported" >&2; exit 2; }
[[ -z "${STEPS:-}" ]] || { echo 'Use STOP_STEP for a phase-local optimizer-step endpoint; STEPS is unsupported' >&2; exit 2; }
[[ -z "${STOP_STEP:-}" || -z "${STOP_EPOCH:-}" ]] || { echo 'STOP_STEP and STOP_EPOCH are mutually exclusive' >&2; exit 2; }
[[ -z "${ENCODER_FROM:-}" || -z "${RESUME:-}" ]] || { echo 'ENCODER_FROM and RESUME are mutually exclusive: new phase versus strict same-phase resume' >&2; exit 2; }
for name in STOP_STEP STOP_EPOCH; do
  if [[ -n "${!name:-}" ]]; then
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
  fi
done
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"
NUM_GPUS="${NUM_GPUS:-$EXPECTED_GPUS}"
[[ "$NUM_GPUS" == "$EXPECTED_GPUS" ]] || { echo "Preset $PRESET requires NUM_GPUS=$EXPECTED_GPUS" >&2; exit 2; }
DEVICE="${DEVICE:-cuda:0}"
WORK_ROOT="${WORK_ROOT:-outputs/latence_nbs_experiments/v090}"
if [[ "$ACTION" == smoke ]]; then
  WORK_DIR="${WORK_DIR:-$WORK_ROOT/${PRESET}_smoke}"
else
  WORK_DIR="${WORK_DIR:-$WORK_ROOT/$PRESET}"
fi
INPUT_DIR="${INPUT_DIR:-outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2}"
METADATA_FILE="${METADATA_FILE:-/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl}"
RUNNER=scripts/nbs/train_nbs_full_task_v090.py

# A shell already used for torchrun must not accidentally initialize DDP during
# prepare/export/evaluate or direct single-GPU training. Changes apply to children.
CHILD_ENV=(env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK
  -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT
  -u TORCHELASTIC_RUN_ID -u TORCHELASTIC_RESTART_COUNT -u TORCHELASTIC_MAX_RESTARTS
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
  "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}"
  "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}")

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
"${CHILD_ENV[@]}" "$PYTHON_BIN" - "$CONFIG" "$PRESET" <<'PY'
import json
import sys

path, preset = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
ft = config.get("full_task", {})
checks = {
    "release_version": (config.get("release_version"), "0.9.0"),
    "prediction_mode": (ft.get("model", {}).get("prediction_mode"), "shared_bilinear"),
    "full_task.weak_batch": (ft.get("weak_batch"), 5120),
    "full_task.core_batch": (ft.get("core_batch"), 1280),
    "full_task.accumulation_steps": (ft.get("accumulation_steps"), 1),
    "full_task.epochs": (ft.get("epochs"), 100),
    "full_task.expected_world_size": (ft.get("expected_world_size"), 1 if preset == "single" else 2),
}
for name, (actual, expected) in checks.items():
    if actual != expected:
        raise SystemExit(f"{path}: {name} must be {expected!r}, got {actual!r}")
PY

COMMON=(--config "$CONFIG" --work-dir "$WORK_DIR" --device "$DEVICE")
if [[ -n "${DEV_MANIFEST:-}" ]]; then COMMON+=(--development-manifest "$DEV_MANIFEST"); fi
prepare() { "${CHILD_ENV[@]}" "$PYTHON_BIN" "$RUNNER" --stage prepare "${COMMON[@]}" --backend torch; }

case "$ACTION" in
  prepare)
    [[ -z "${ENCODER_FROM:-}${RESUME:-}${STOP_STEP:-}${STOP_EPOCH:-}" ]] || { echo 'prepare does not use initialization, resume or stop endpoints' >&2; exit 2; }
    prepare ;;
  smoke|train)
    extra=()
    if [[ "$ACTION" == smoke ]]; then
      [[ -z "${RESUME:-}${STOP_STEP:-}${STOP_EPOCH:-}" ]] || { echo 'smoke is an isolated new run; use SMOKE_STEPS, optionally ENCODER_FROM' >&2; exit 2; }
      [[ "${SMOKE_STEPS:-20}" =~ ^[1-9][0-9]*$ ]] || { echo 'SMOKE_STEPS must be a positive integer' >&2; exit 2; }
      [[ ! -f "$WORK_DIR/latest.pt" && ! -f "$WORK_DIR/phase_manifest.json" && ! -f "$WORK_DIR/training_history.json" ]] || { echo 'smoke work directory already has training outputs; choose a new WORK_DIR' >&2; exit 2; }
      extra+=(--stop-step "${SMOKE_STEPS:-20}")
    else
      if [[ -n "${STOP_STEP:-}" ]]; then extra+=(--stop-step "$STOP_STEP"); fi
      if [[ -n "${STOP_EPOCH:-}" ]]; then extra+=(--stop-epoch "$STOP_EPOCH"); fi
      if [[ -n "${RESUME:-}" ]]; then
        [[ -f "$RESUME" ]] || { echo "Resume checkpoint not found: $RESUME" >&2; exit 2; }
        extra+=(--resume "$RESUME")
      elif [[ -f "$WORK_DIR/latest.pt" || -f "$WORK_DIR/phase_manifest.json" || -f "$WORK_DIR/training_history.json" ]]; then
        echo 'Training outputs already exist: use RESUME in the same phase or a new WORK_DIR for ENCODER_FROM' >&2
        exit 2
      fi
    fi
    if [[ -n "${ENCODER_FROM:-}" ]]; then
      [[ -f "$ENCODER_FROM" ]] || { echo "Initialization checkpoint not found: $ENCODER_FROM" >&2; exit 2; }
      extra+=(--encoder-from "$ENCODER_FROM")
    fi
    mkdir -p -- "$WORK_DIR"
    command -v flock >/dev/null || { echo 'flock is required to prevent concurrent writes to the same WORK_DIR' >&2; exit 2; }
    exec 9>"$WORK_DIR/.launcher_train.lock"
    flock -n 9 || { echo "Another launcher is writing $WORK_DIR" >&2; exit 2; }
    prepare
    if [[ "$NUM_GPUS" == 1 ]]; then
      "${CHILD_ENV[@]}" "$PYTHON_BIN" "$RUNNER" --stage train "${COMMON[@]}" "${extra[@]}"
    else
      "${CHILD_ENV[@]}" "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
        "$RUNNER" --stage train "${COMMON[@]}" "${extra[@]}"
    fi ;;
  export|evaluate)
    [[ -z "${ENCODER_FROM:-}${RESUME:-}${STOP_STEP:-}${STOP_EPOCH:-}" ]] || { echo 'export/evaluate require CHECKPOINT, not initialization, resume or stop endpoints' >&2; exit 2; }
    [[ -n "${CHECKPOINT:-}" ]] || { echo 'Set CHECKPOINT to an explicit fixed nbs_epoch*.pt checkpoint' >&2; exit 2; }
    [[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
    [[ "$(basename -- "$CHECKPOINT")" =~ ^nbs_epoch[0-9]+\.pt$ ]] || { echo 'CHECKPOINT must be a fixed nbs_epoch<number>.pt file, not latest.pt or a monitor checkpoint' >&2; exit 2; }
    extra=()
    if [[ "$ACTION" == evaluate ]]; then
      [[ "${METRIC_BACKEND:-stage1}" == stage1 && "${AUPRC_MODE:-exact}" == exact ]] || { echo 'Acceptance requires METRIC_BACKEND=stage1 and AUPRC_MODE=exact' >&2; exit 2; }
      extra+=(--metric-backend stage1 --auprc-mode exact --baseline-comparison "${BASELINE_COMPARISON:-BASELINE_CONTRACT_v088.json}")
    fi
    "${CHILD_ENV[@]}" "$PYTHON_BIN" "$RUNNER" --stage "$ACTION" "${COMMON[@]}" --checkpoint "$CHECKPOINT" \
      --input-dir "$INPUT_DIR" --metadata-file "$METADATA_FILE" --ablation full "${extra[@]}" ;;
esac
