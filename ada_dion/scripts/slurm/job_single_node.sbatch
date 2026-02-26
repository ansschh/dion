#!/bin/bash
# ============================================================
# SLURM: Single-node training on Caltech HPC (4x H200)
#
# Usage:
#   sbatch job_single_node.sbatch muon
#   sbatch job_single_node.sbatch adamw
#   STEPS=500 sbatch job_single_node.sbatch dion
# ============================================================

#SBATCH --job-name=adadion
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

set -e

OPTIMIZER=${1:-muon}
STEPS=${STEPS:-10000}
NGPU=4

echo "============================================"
echo "  Job: $SLURM_JOB_ID on $(hostname)"
echo "  Optimizer: $OPTIMIZER | GPUs: $NGPU | Steps: $STEPS"
echo "============================================"

# Environment
module load python/3.11.6-gcc-13.2.0-fh6i4o3
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e
source "$HOME/envs/adadion/bin/activate"

cd "$HOME/ada-dion"
mkdir -p logs

# NCCL
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

# W&B (optional)
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-caltech}"
export WANDB_RUN_NAME="${OPTIMIZER}_${NGPU}gpu_${STEPS}steps_${SLURM_JOB_ID}"

# Map optimizer to config
case "$OPTIMIZER" in
    adamw) CONFIG_FN="llama3_160m_adamw" ;;
    muon)  CONFIG_FN="llama3_160m_muon" ;;
    dion)  CONFIG_FN="llama3_160m_dion" ;;
    dion2) CONFIG_FN="llama3_160m_dion2" ;;
    *)     echo "Unknown: $OPTIMIZER"; exit 1 ;;
esac

torchrun \
    --nproc_per_node=$NGPU \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:$MASTER_PORT" \
    -m torchtitan.train \
    --module ada_dion.integration.config_registry \
    --config "$CONFIG_FN" \
    --training.steps "$STEPS" \
    --parallelism.data_parallel_shard_degree $NGPU \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1

echo "Done at $(date)"
