#!/bin/bash

# -------------------------------------------------- #
GPUS=$1                                              #
# -------------------------------------------------- #

export WORLD_SIZE=$((${GPUS}<8?${GPUS}:8))
# if [ $GPUS -lt 2 ]; then export WORLD_SIZE=$GPUS; else export WORLD_SIZE=2; fi


export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1

DATADIR=/data0/Databases/GOA-S2F/data-netgo
# DATE=24-12-20
# DATE=25-03-26
# DATE=25-04-02
DATE=25-07-05
ONTOLOGY=(cc mf bp)
IDX=$((${2}<3?${2}:2))
ONT=${ONTOLOGY[${IDX}]}

torchrun --standalone --nnodes=1 \
        --nproc_per_node=$WORLD_SIZE \
        scripts/construct_gendis.py -c configs/training_netgo-v1/${ONT}-${DATE}.yml \
        $DATADIR/dataset_state_dict.pkl \
        $DATADIR/filt-MSAs/ \
        $DATADIR/training/trained_model/ng-v1-${DATE}/${ONT}/