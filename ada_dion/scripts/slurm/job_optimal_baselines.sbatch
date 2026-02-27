#!/bin/bash
# ============================================================
# SLURM: Optimal HP baselines via job array (4 optimizers)
#
# Runs each optimizer with best HPs from sweep:
#   AdamW:  lr=1e-3
#   Muon:   lr=0.005
#   Dion:   lr=0.005, rank_frac=0.5
#   Dion2:  lr=0.005, alpha=0.5
#
# Usage:
#   sbatch job_optimal_baselines.sbatch
# ============================================================

#SBATCH --job-name=adadion-optimal
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/optimal_%A_%a.out
#SBATCH --error=logs/optimal_%A_%a.err

set -e

source /resnick/home/atiwari2/ada-dion/ada_dion/scripts/slurm/env.sh

NGPU=4
STEPS=10000
MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-optimal}"

# Optimal configs from HP sweep
CONFIGS=(
    "adamw llama3_160m_adamw --optimizer.lr 1e-3"
    "muon llama3_160m_muon --optimizer.lr 0.005"
    "dion llama3_160m_dion --optimizer.lr 0.005 --optimizer.rank_frac 0.5"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.005 --optimizer.alpha 0.5"
)

CONFIG_LINE="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r OPT_NAME CONFIG_FN EXTRA_ARGS <<< "$CONFIG_LINE"

export WANDB_RUN_NAME="optimal_${OPT_NAME}_${SLURM_ARRAY_JOB_ID}"

echo "============================================"
echo "  Optimal Baseline: $OPT_NAME"
echo "  Job: $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "  Config: $CONFIG_FN"
echo "  Extra: $EXTRA_ARGS"
echo "  GPUs: $NGPU | Steps: $STEPS"
echo "============================================"

torchrun \
    --nproc_per_node=$NGPU \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:$MASTER_PORT" \
    -m torchtitan.train \
    --module ada_dion \
    --config "$CONFIG_FN" \
    --training.steps "$STEPS" \
    --metrics.log_freq 1 \
    --parallelism.data_parallel_shard_degree $NGPU \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1 \
    $EXTRA_ARGS

echo "Optimal baseline $OPT_NAME done at $(date)"
