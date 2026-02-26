#!/bin/bash
# Single-node training on 8 H100 GPUs.
#
# Usage:
#   bash ada_dion/scripts/run_single_node.sh muon        # Muon optimizer
#   bash ada_dion/scripts/run_single_node.sh dion         # Dion optimizer
#   bash ada_dion/scripts/run_single_node.sh dion2        # Dion2 optimizer
#   bash ada_dion/scripts/run_single_node.sh adamw        # AdamW baseline
#   STEPS=500 NGPU=4 bash ada_dion/scripts/run_single_node.sh muon

set -e

OPTIMIZER=${1:?Usage: $0 <muon|dion|dion2|adamw> [extra_args...]}
shift
EXTRA_ARGS="$@"

NGPU=${NGPU:-8}
STEPS=${STEPS:-10000}

# Map optimizer to config function
case "$OPTIMIZER" in
    muon)   CONFIG="llama3_160m_muon" ;;
    dion)   CONFIG="llama3_160m_dion" ;;
    dion2)  CONFIG="llama3_160m_dion2" ;;
    adamw)  CONFIG="llama3_160m_adamw" ;;
    *)
        echo "Unknown optimizer: $OPTIMIZER"
        echo "Options: muon, dion, dion2, adamw"
        exit 1
        ;;
esac

echo "=== Single-node training ==="
echo "Optimizer: $OPTIMIZER"
echo "Config: $CONFIG"
echo "GPUs: $NGPU"
echo "Steps: $STEPS"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion}"
export WANDB_RUN_NAME="${OPTIMIZER}_${NGPU}gpu_${STEPS}steps"

torchrun \
    --nproc_per_node="$NGPU" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    --module ada_dion.integration.config_registry \
    --config "$CONFIG" \
    --training.steps "$STEPS" \
    --parallelism.data_parallel_shard_degree "$NGPU" \
    --parallelism.data_parallel_replicate_degree 1 \
    $EXTRA_ARGS
