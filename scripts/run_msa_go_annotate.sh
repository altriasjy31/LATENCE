#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dataset-local/data_local/shaojiangyi/latence-project

FILE_ADDRESS=${ROOT}/data/dataset_with_exp_train.pkl
WORKING_DIR=${ROOT}/data/sprot_2204_MSA
OUTPUT_DATASET_PKL=${ROOT}/data/dataset_with_exp_train_pseudo.pkl
PROB_OUTPUT_DIR=${ROOT}/data/outputs/msa_teacher_25_10_22_probs
MODEL_SAVING=${ROOT}/data/msa_models/checkpoints

TRAINED_MODEL_CC=cc_msa_model_rank1.pt
TRAINED_MODEL_MF=mf_msa_model_rank1.pt
TRAINED_MODEL_BP=bp_msa_model_rank1.pt

MODEL_CONFIG_CC=${ROOT}/data/msa_models/configs/model_opts/cc_msa_model_config.pkl
MODEL_CONFIG_MF=${ROOT}/data/msa_models/configs/model_opts/mf_msa_model_config.pkl
MODEL_CONFIG_BP=${ROOT}/data/msa_models/configs/model_opts/bp_msa_model_config.pkl

NUM_CLASSES_CC=2903
NUM_CLASSES_MF=7038
NUM_CLASSES_BP=21312

cd "${ROOT}"

python scripts/msa_go_annotate.py \
  --file_address "${FILE_ADDRESS}" \
  --working_dir "${WORKING_DIR}" \
  --output_dataset_pkl "${OUTPUT_DATASET_PKL}" \
  --prob_output_dir "${PROB_OUTPUT_DIR}" \
  --model_saving "${MODEL_SAVING}" \
  --trained_model_by_task "cc=${TRAINED_MODEL_CC},mf=${TRAINED_MODEL_MF},bp=${TRAINED_MODEL_BP}" \
  --num_classes_by_task "cc=${NUM_CLASSES_CC},mf=${NUM_CLASSES_MF},bp=${NUM_CLASSES_BP}" \
  --model_config_by_task "cc=${MODEL_CONFIG_CC},mf=${MODEL_CONFIG_MF},bp=${MODEL_CONFIG_BP}" \
  --thresholds "cc=0.5,mf=0.5,bp=0.5" \
  --batch_size 32 \
  --dataloader_num_workers 4 \
  --device "cuda:1" \
  --pin_memory