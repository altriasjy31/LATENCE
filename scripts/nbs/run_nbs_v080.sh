#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. All defaults are relative to this project's root.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.0.json}"
WORK_DIR="${WORK_DIR:-outputs/latence_nbs_experiments/v080/main}"
NUM_GPUS="${NUM_GPUS:-2}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-600}"
INPUT_DIR="${INPUT_DIR:-outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2}"
METADATA_FILE="${METADATA_FILE:-/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl}"
CHECKPOINT="${CHECKPOINT:-$WORK_DIR/best.pt}"
ABLATION="${ABLATION:-full}"
RUNNER="scripts/nbs/train_nbs_full_task.py"
COMMON=(--config "$CONFIG" --work-dir "$WORK_DIR" --device "$DEVICE")

case "${1:-pilot}" in
  prepare)
    "$PYTHON_BIN" "$RUNNER" --stage prepare "${COMMON[@]}" --backend torch
    ;;
  pilot|train)
    # One-time retrieval is reused by subsequent runs with the same core split.
    "$PYTHON_BIN" "$RUNNER" --stage prepare "${COMMON[@]}" --backend torch
    EXTRA=(--steps "$STEPS")
    if [[ -n "${RESUME:-}" ]]; then EXTRA+=(--resume "$RESUME"); fi
    if [[ "$NUM_GPUS" == "1" ]]; then
      "$PYTHON_BIN" "$RUNNER" --stage train "${COMMON[@]}" "${EXTRA[@]}"
    else
      "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
        "$RUNNER" --stage train "${COMMON[@]}" "${EXTRA[@]}"
    fi
    ;;
  evaluate)
    "$PYTHON_BIN" "$RUNNER" --stage evaluate "${COMMON[@]}" \
      --checkpoint "$CHECKPOINT" --input-dir "$INPUT_DIR" --metadata-file "$METADATA_FILE" \
      --ablation "$ABLATION" --metric-backend stage1
    ;;
  *)
    echo 'Usage: bash scripts/nbs/run_nbs_v080.sh {pilot|prepare|train|evaluate}' >&2
    exit 2
    ;;
esac
