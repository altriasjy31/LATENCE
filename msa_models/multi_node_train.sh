#!/bin/bash
# multi_node_train.sh

# Configuration
NNODES=2                        # Total number of nodes
NODE_RANK=$1                    # Current node rank (pass as argument)
GPUS_PER_NODE=8                 # GPUs per node
MASTER_ADDR="10.3.14.8"     # IP address of the main node
MASTER_PORT="60010"             # Communication port

# NCCL configuration
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0  # Set this to your network interface name
# export NCCL_PORT_MIN=33000
# export NCCL_PORT_MAX=34000
# sudo echo $NCCL_PORT_MIN $NCCL_PORT_MAX > /proc/sys/net/ipv4/ip_local_port_range
export NCCL_IB_DISABLE=1        # Try disabling InfiniBand if not using it

# Application specific variables
DATADIR=/data0/Databases/GOA-S2F/data-netgo
# DATE=24-12-20
# DATE=25-03-26
# DATE=25-04-02
DATE=25-07-05
ONTOLOGY=(cc mf bp)
IDX=$((${2}<3?${2}:2))
ONT=${ONTOLOGY[${IDX}]}

echo "Starting node $NODE_RANK of $NNODES with $GPUS_PER_NODE GPUs per node"
echo "Master node: $MASTER_IP:$MASTER_PORT"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"

# Launch distributed training
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/construct_gendis.py -c configs/training_netgo-v1/${ONT}-${DATE}.yml \
    $DATADIR/dataset_state_dict.pkl \
    $DATADIR/filt-MSAs \
    $DATADIR/training/trained_model/ng-v1-${DATE}/${ONT}
