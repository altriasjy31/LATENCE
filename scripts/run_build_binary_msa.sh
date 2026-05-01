#!/usr/bin/env bash

SPROT_MSA_DIR=/home/dataset-local/data_local/shaojiangyi/latence-project/data/sprot_2204_MSA
SPROT_BIN_DIR=/home/dataset-local/data_local/shaojiangyi/latence-project/data/sprot_2204_MSA_bin

IND_MSA_DIR=/home/dataset-local/data_local/shaojiangyi/latence-project/data/ind_MSA
IND_BIN_DIR=/home/dataset-local/data_local/shaojiangyi/latence-project/data/ind_MSA_bin

python build_msa_binary.py \
  ${SPROT_MSA_DIR} \
  ${SPROT_BIN_DIR} \
  --msa-format a3m \
  --max-msa-size -1 \
  --store-max-len 2048 \
  --max-shard-gb 4 \
  --num-workers 8 \
  --shuffle-rows \
  --overwrite