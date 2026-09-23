#!/usr/bin/env bash

# export SPROT_MSA_DIR=/home/dataset-local/data_local/shaojiangyi/latence-project/data/sprot_2204_MSA
# export SPROT_BIN_DIR=/home/dataset-local/data_local/shaojiangyi/latence-project/data/sprot_2204_MSA_bin

# export IND_MSA_DIR=/home/dataset-local/data_local/shaojiangyi/latence-project/data/ind_MSA
# export IND_BIN_DIR=/home/dataset-local/data_local/shaojiangyi/latence-project/data/ind_MSA_bin

set -euo pipefail

export MSA_DATASET=all
export MSA_NUM_WORKERS=8
export MSA_STORE_MAX_LEN=2048
export MSA_MAX_SHARD_GB=4
export MSA_SHUFFLE_ROWS=true
export MSA_OVERWRITE=true

python run_build_msa_binary.py "$@"