#!/bin/bash
# -------------------------------------------------- #
GPU_ID=$1                                              #
# -------------------------------------------------- #
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export WORLD_SIZE=1
export LOCAL_RANK=$GPU_ID

export HCCL_WHITELIST_DISABLE=1
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1

DATADIR=/disk1/home/Databases/GOA-S2F/data-netgo
DATE=24-12-20
ONTOLOGY=(cc mf bp)
ONT=${ONTOLOGY[2]}

python scripts/construct_gendis.py \
        -c configs/training_netgo-v1/${ONT}-${DATE}.yml \
        $DATADIR/dataset_state_dict.pkl \
        $DATADIR/filt-MSAs/ \
        $DATADIR/training/trained_model/tmp-v1-${DATE}-1/${ONT}/