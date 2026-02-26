#!/bin/bash
# Hyperparameter sweep runner.
# Runs all sweep configs sequentially on a single node.
#
# Usage:
#   bash ada_dion/scripts/run_sweep.sh
#   STEPS=2000 NGPU=4 bash ada_dion/scripts/run_sweep.sh

set -e

NGPU=${NGPU:-8}
STEPS=${STEPS:-5000}

export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-sweep}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=== Hyperparameter Sweep ==="
echo "GPUs: $NGPU | Steps: $STEPS"

run_config() {
    local name=$1
    local config=$2
    shift 2
    local overrides="$@"

    echo ""
    echo "=========================================="
    echo "  Running: $name"
    echo "=========================================="
    export WANDB_RUN_NAME="$name"

    torchrun \
        --nproc_per_node="$NGPU" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:0" \
        -m torchtitan.train \
        --module ada_dion.integration.config_registry \
        --config "$config" \
        --training.steps "$STEPS" \
        --parallelism.data_parallel_shard_degree "$NGPU" \
        $overrides || echo "FAILED: $name"
}

# AdamW sweep: lr x 3
for lr in 1e-4 3e-4 1e-3; do
    run_config "adamw_lr${lr}" llama3_160m_adamw --optimizer.lr "$lr"
done

# Muon sweep: lr x 3
for lr in 0.005 0.02 0.05; do
    run_config "muon_lr${lr}" llama3_160m_muon --optimizer.lr "$lr"
done

# Dion sweep: lr x 3, rank_frac x 3
for lr in 0.005 0.02 0.05; do
    for rf in 0.1 0.25 0.5; do
        run_config "dion_lr${lr}_rf${rf}" llama3_160m_dion \
            --optimizer.lr "$lr" --optimizer.rank_frac "$rf"
    done
done

# Dion2 sweep: lr x 3, alpha x 3
for lr in 0.005 0.02 0.05; do
    for alpha in 0.1 0.25 0.5; do
        run_config "dion2_lr${lr}_a${alpha}" llama3_160m_dion2 \
            --optimizer.lr "$lr" --optimizer.alpha "$alpha"
    done
done

echo ""
echo "=== Sweep complete ==="
