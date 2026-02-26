#!/bin/bash
# Multi-node training on RunPod Instant Cluster.
# Uses HSDP: dp_shard=8 within node, dp_replicate=NUM_NODES across nodes.
#
# RunPod provides: MASTER_ADDR, MASTER_PORT, NUM_NODES, NODE_RANK, NUM_TRAINERS
#
# Usage (run on EACH pod):
#   bash ada_dion/scripts/run_multi_node.sh muon
#   bash ada_dion/scripts/run_multi_node.sh dion
#
# Or with explicit env vars:
#   MASTER_ADDR=10.0.0.1 NUM_NODES=2 NODE_RANK=0 \
#       bash ada_dion/scripts/run_multi_node.sh muon

set -e

OPTIMIZER=${1:?Usage: $0 <muon|dion|dion2|adamw> [extra_args...]}
shift
EXTRA_ARGS="$@"

# RunPod environment variables
NUM_NODES=${NUM_NODES:-2}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:?"MASTER_ADDR must be set for multi-node"}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${NUM_TRAINERS:-8}
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
STEPS=${STEPS:-10000}

# HSDP: shard within node, replicate across nodes
DP_SHARD=$GPUS_PER_NODE
DP_REPLICATE=$NUM_NODES

# Map optimizer to config function
case "$OPTIMIZER" in
    muon)   CONFIG="llama3_160m_muon" ;;
    dion)   CONFIG="llama3_160m_dion" ;;
    dion2)  CONFIG="llama3_160m_dion2" ;;
    adamw)  CONFIG="llama3_160m_adamw" ;;
    *)
        echo "Unknown optimizer: $OPTIMIZER"
        exit 1
        ;;
esac

echo "=== Multi-node training (HSDP) ==="
echo "Optimizer: $OPTIMIZER | Config: $CONFIG"
echo "Nodes: $NUM_NODES | GPUs/node: $GPUS_PER_NODE | Total: $TOTAL_GPUS"
echo "Node rank: $NODE_RANK | Master: $MASTER_ADDR:$MASTER_PORT"
echo "HSDP: dp_shard=$DP_SHARD, dp_replicate=$DP_REPLICATE"
echo "Steps: $STEPS"

# NCCL configuration for RunPod
export NCCL_SOCKET_IFNAME=ens1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# Memory and logging
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion}"
export WANDB_RUN_NAME="${OPTIMIZER}_${TOTAL_GPUS}gpu_hsdp_${STEPS}steps"

torchrun \
    --nnodes="$NUM_NODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m torchtitan.train \
    --module ada_dion.integration.config_registry \
    --config "$CONFIG" \
    --training.steps "$STEPS" \
    --parallelism.data_parallel_shard_degree "$DP_SHARD" \
    --parallelism.data_parallel_replicate_degree "$DP_REPLICATE" \
    $EXTRA_ARGS
