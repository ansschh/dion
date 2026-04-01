#!/bin/bash
# Part A Phase 1: Collect snapshots from 500-step Dion baseline run.
# Saves grad + momentum per layer every 50 steps.
# Usage: bash ada_dion/scripts/part_a_collect.sh

set -e
cd /workspace/ada-dion

NGPU=8
STEPS=500
FREQ=50
MODULE="ada_dion.integration.config_registry"
TOKENIZER="./assets/hf/Meta-Llama-3.1-8B"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=disabled

echo "=== Part A Phase 1: Collecting snapshots ==="
echo "  Steps: $STEPS, Freq: $FREQ, GPUs: $NGPU"

# Run Dion baseline with snapshot collection enabled
# We use a custom training script that hooks into the optimizer
torchrun \
  --nproc_per_node=$NGPU \
  --rdzv_backend=c10d \
  --rdzv_endpoint="localhost:0" \
  --local-ranks-filter 0 \
  --role rank \
  --tee 3 \
  -m torchtitan.train \
  --module "$MODULE" \
  --config llama3_320m_adadion \
  --training.steps $STEPS \
  --training.local-batch-size 32 \
  --training.global-batch-size 256 \
  --training.seq-len 2048 \
  --hf-assets-path "$TOKENIZER" \
  --parallelism.data-parallel-shard-degree $NGPU \
  --parallelism.data-parallel-replicate-degree 1 \
  --compile.enable \
  --activation-checkpoint.mode selective \
  --activation-checkpoint.selective-ac-option 2 \
  --metrics.log-freq 10 \
  --optimizer.no-use-anchor \
  --optimizer.no-use-adaptive-scalar \
  --optimizer.no-rank-normalize 2>&1 | tee logs/part_a_collect.log

echo "=== Collection done ==="
