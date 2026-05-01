#!/bin/bash

# -------------------------------------------------- #
GPUS=$1                                              #
# -------------------------------------------------- #

export WORLD_SIZE=$((${GPUS}<8?${GPUS}:8))

# DATADIR=/data0/Databases/GOA-S2F/data-sp
DATADIR=/home/dataset-assist-0/datafile/shaojiangyi/latence-dataset/
# DATE=24-12-20
# DATE=25-03-26
# DATE=25-04-02
# DATE=25-07-05
# DATE=25-10-06
DATE=25-10-22
ONTOLOGY=(cc mf bp)
IDX=$((${2}<3?${2}:2))
ONT=${ONTOLOGY[${IDX}]}

torchrun --standalone --nnodes=1 \
        --nproc_per_node=$WORLD_SIZE \
        scripts/construct_gendis.py -c configs/evaluating_msa-v1/${ONT}-${DATE}-ind.yml \
        $DATADIR/dataset_state_ind_dict.pkl \
        $DATADIR/ind_MSA/ \
        $DATADIR/msa_models/sp-v1-${DATE}/${ONT}/

# torchrun --standalone --nnodes=1 \
#         --nproc_per_node=$WORLD_SIZE \
#         scripts/average_performance.py -c configs/evaluating_netgo-v1/${ONT}-${DATE}-avg.yml \
#         -n 5 \
#         $DATADIR/dataset_state_dict.pkl \
#         $DATADIR/filt-MSAs/ \
#         $DATADIR/training/trained_model/ng-v1-${DATE}/${ONT}/