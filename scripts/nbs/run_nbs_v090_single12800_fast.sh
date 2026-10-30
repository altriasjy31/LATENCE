#!/usr/bin/env bash
# Independent P0 launcher based on run_nbs_v090_single12800.sh: uses a separate fast work root and the vectorized sampler.
# Additive v0.9.0 launcher: one GPU, one direct 12800-query optimizer batch.
# Six-layer sampling support lives in FullTaskDataV090; old launchers stay intact.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd -- "$PROJECT_ROOT"
[[ "$#" -le 1 ]] || { echo 'Usage: run_nbs_v090_single12800_fast.sh {prepare|smoke|train|export|evaluate}; no single/ddp preset argument' >&2; exit 2; }
ACTION="${1:-train}"
case "$ACTION" in
  prepare|smoke|train|export|evaluate) ;;
  *) echo 'Usage: run_nbs_v090_single12800_fast.sh {prepare|smoke|train|export|evaluate}' >&2; exit 2 ;;
esac
[[ -z "${PRESET:-}" ]] || { echo 'This launcher has one fixed single-GPU preset; unset PRESET' >&2; exit 2; }
[[ "${ABLATION:-full}" == full ]] || { echo 'v090 permits ABLATION=full only; separate ablations remain paused' >&2; exit 2; }
[[ -z "${INIT_FROM:-}" ]] || { echo 'INIT_FROM is unsupported; start this six-layer model fresh. ENCODER_FROM is only for a strictly compatible encoder.' >&2; exit 2; }
[[ -z "${STEPS:-}" ]] || { echo 'Use STOP_STEP for a phase-local optimizer-step endpoint; STEPS is unsupported' >&2; exit 2; }
[[ -z "${STOP_STEP:-}" || -z "${STOP_EPOCH:-}" ]] || { echo 'STOP_STEP and STOP_EPOCH are mutually exclusive' >&2; exit 2; }
[[ -z "${ENCODER_FROM:-}" || -z "${RESUME:-}" ]] || { echo 'ENCODER_FROM and RESUME are mutually exclusive: new phase versus strict same-phase resume' >&2; exit 2; }
for name in STOP_STEP STOP_EPOCH; do
  if [[ -n "${!name:-}" ]]; then
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
  fi
done
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-nbs_models/nbs_protein_go/configs/bp_full_task_v0.9.0_mb12800_single_ep100_6layer_lr001_weak.json}"
NUM_GPUS="${NUM_GPUS:-1}"
[[ "$NUM_GPUS" == 1 ]] || { echo 'This launcher requires NUM_GPUS=1 and never launches torchrun' >&2; exit 2; }
DEVICE="${DEVICE:-cuda:0}"
[[ "$DEVICE" == cuda:0 ]] || { echo 'Use DEVICE=cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES' >&2; exit 2; }
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[^,[:space:]]+$ && "$CUDA_VISIBLE_DEVICES" != -1 ]] || { echo 'Set CUDA_VISIBLE_DEVICES to exactly one physical GPU index or GPU UUID' >&2; exit 2; }
WORK_ROOT="${WORK_ROOT:-outputs/latence_nbs_experiments/v090_fast}"
if [[ "$ACTION" == smoke ]]; then
  WORK_DIR="${WORK_DIR:-$WORK_ROOT/mb12800_single_6layer_lr001_weak_smoke}"
else
  WORK_DIR="${WORK_DIR:-$WORK_ROOT/mb12800_single_6layer_lr001_weak_ep100}"
fi
INPUT_DIR="${INPUT_DIR:-outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2}"
METADATA_FILE="${METADATA_FILE:-/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl}"
RUNNER="$SCRIPT_DIR/train_nbs_full_task_v090_fast.py"

# Single-process children must not inherit a previous torchrun rank/store context.
CHILD_ENV=(env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK
  -u ROLE_RANK -u ROLE_WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT
  -u TORCHELASTIC_RUN_ID -u TORCHELASTIC_RESTART_COUNT -u TORCHELASTIC_MAX_RESTARTS
  -u TORCHELASTIC_ERROR_FILE -u TORCHELASTIC_USE_AGENT_STORE
  "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
  "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}"
  "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}")

[[ -f "$RUNNER" ]] || { echo "Fast runner not found: $RUNNER" >&2; exit 2; }
[[ -f "$PROJECT_ROOT/scripts/nbs/train_nbs_full_task_v090.py" ]] || { echo "Original scripts/nbs runner is required; preserve the original files" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
"${CHILD_ENV[@]}" "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
ft = config.get("full_task", {})
model = ft.get("model", {})
checks = {
    "release_version": (config.get("release_version"), "0.9.0"),
    "model_version": (config.get("model_version"), "0.9.0"),
    "prediction_mode": (model.get("prediction_mode"), "shared_bilinear"),
    "full_task.weak_batch": (ft.get("weak_batch"), 10240),
    "full_task.core_batch": (ft.get("core_batch"), 2560),
    "full_task.epochs": (ft.get("epochs"), 100),
    "full_task.expected_world_size": (ft.get("expected_world_size"), 1),
    "full_task.model.sage_layers": (model.get("sage_layers"), 6),
}
for name, (actual, expected) in checks.items():
    if actual != expected or (isinstance(expected, int) and type(actual) is not int):
        raise SystemExit(f"{path}: {name} must be {expected!r}, got {actual!r}")
accumulation = ft.get("accumulation_steps")
if type(accumulation) is not int or accumulation not in (1, 2):
    raise SystemExit(f"{path}: full_task.accumulation_steps must be 1 or 2, got {accumulation!r}")
fanouts = ft.get("sampler", {}).get("fanouts")
if not isinstance(fanouts, list) or len(fanouts) != 6:
    raise SystemExit(f"{path}: sampler.fanouts must contain six outer-to-inner relation mappings")
relations = {"ppi", "similar_to", "weak_to_core", "core_to_weak", "cosine"}
for layer, budgets in enumerate(fanouts):
    if not isinstance(budgets, dict) or not budgets or set(budgets) - relations:
        raise SystemExit(f"{path}: fanouts[{layer}] has invalid relation names or is not a mapping")
    if any(type(v) is not int or v < 0 for v in budgets.values()):
        raise SystemExit(f"{path}: fanouts[{layer}] budgets must be nonnegative integers")
    if not any(budgets.values()):
        raise SystemExit(f"{path}: fanouts[{layer}] must retain at least one relation")
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
      [[ "${SMOKE_STEPS:-3}" =~ ^[1-9][0-9]*$ ]] || { echo 'SMOKE_STEPS must be a positive integer' >&2; exit 2; }
      [[ ! -f "$WORK_DIR/latest.pt" && ! -f "$WORK_DIR/phase_manifest.json" && ! -f "$WORK_DIR/training_history.json" ]] || { echo 'smoke work directory already has training outputs; choose a new WORK_DIR' >&2; exit 2; }
      extra+=(--stop-step "${SMOKE_STEPS:-3}")
    else
      if [[ -n "${STOP_STEP:-}" ]]; then extra+=(--stop-step "$STOP_STEP"); fi
      if [[ -n "${STOP_EPOCH:-}" ]]; then extra+=(--stop-epoch "$STOP_EPOCH"); fi
      if [[ -n "${RESUME:-}" ]]; then
        [[ -f "$RESUME" ]] || { echo "Resume checkpoint not found: $RESUME" >&2; exit 2; }
        extra+=(--resume "$RESUME")
      elif [[ -f "$WORK_DIR/latest.pt" || -f "$WORK_DIR/phase_manifest.json" || -f "$WORK_DIR/training_history.json" ]]; then
        echo 'Training outputs already exist: use RESUME in the same phase or a new WORK_DIR' >&2
        exit 2
      fi
    fi
    if [[ -n "${ENCODER_FROM:-}" ]]; then
      [[ -f "$ENCODER_FROM" ]] || { echo "Initialization checkpoint not found: $ENCODER_FROM" >&2; exit 2; }
      echo 'ENCODER_FROM requires identical encoder depth and shapes; three-layer v089 checkpoints are incompatible with this six-layer preset. Fresh training is recommended.' >&2
      extra+=(--encoder-from "$ENCODER_FROM")
    fi
    mkdir -p -- "$WORK_DIR"
    command -v flock >/dev/null || { echo 'flock is required to prevent concurrent writes to the same WORK_DIR' >&2; exit 2; }
    exec 9>"$WORK_DIR/.launcher_train.lock"
    flock -n 9 || { echo "Another launcher is writing $WORK_DIR" >&2; exit 2; }
    prepare
    "${CHILD_ENV[@]}" "$PYTHON_BIN" "$RUNNER" --stage train "${COMMON[@]}" "${extra[@]}" ;;
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
