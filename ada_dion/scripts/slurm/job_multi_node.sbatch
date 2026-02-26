#!/bin/bash
# ============================================================
# SLURM: Multi-node training (HSDP: shard within node, replicate across)
#
# Usage:
#   sbatch job_multi_node.sbatch muon         # 2 nodes
#   NNODES=4 sbatch job_multi_node.sbatch dion  # 4 nodes
# ============================================================

# ------ SLURM CONFIG (edit for your cluster) ------
#SBATCH --job-name=adadion-multi
#SBATCH --partition=gpu                # <-- CHANGE to your GPU partition
#SBATCH --nodes=2                      # <-- CHANGE: 2 or 4 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4             # <-- CHANGE to GPUs per node (4 or 8)
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
# ---------------------------------------------------

set -e

OPTIMIZER=${1:-muon}
STEPS=${STEPS:-10000}
NGPU_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
NNODES=${SLURM_NNODES:-2}
WORLD_SIZE=$(( NNODES * NGPU_PER_NODE ))

echo "============================================"
echo "  Multi-node Job: $SLURM_JOB_ID"
echo "  Nodes: $NNODES ($SLURM_NODELIST)"
echo "  GPUs/node: $NGPU_PER_NODE"
echo "  World size: $WORLD_SIZE"
echo "  Optimizer: $OPTIMIZER"
echo "  Steps: $STEPS"
echo "============================================"

# ------ Environment setup ------
# Uncomment/edit for your cluster:
# module load cuda/12.4
# module load anaconda3
# eval "$(conda shell.bash hook)"
# conda activate adadion

# ------ Work directory ------
WORK_DIR="${SCRATCH:-$HOME}/ada-dion"
cd "$WORK_DIR"

# ------ NCCL / distributed settings ------
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Get master address from first node in allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

export MASTER_ADDR
export MASTER_PORT

# W&B
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-caltech}"
export WANDB_RUN_NAME="${OPTIMIZER}_${WORLD_SIZE}gpu_hsdp_${SLURM_JOB_ID}"

mkdir -p logs profiles

# ------ Map optimizer to config ------
case "$OPTIMIZER" in
    adamw) CONFIG_FN="llama3_160m_adamw" ;;
    muon)  CONFIG_FN="llama3_160m_muon" ;;
    dion)  CONFIG_FN="llama3_160m_dion" ;;
    dion2) CONFIG_FN="llama3_160m_dion2" ;;
    *)
        echo "Unknown optimizer: $OPTIMIZER"
        exit 1
        ;;
esac

# ------ HSDP config: shard within node, replicate across nodes ------
DP_SHARD=$NGPU_PER_NODE
DP_REPLICATE=$NNODES

echo "Parallelism: dp_shard=$DP_SHARD, dp_replicate=$DP_REPLICATE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# ------ Launch with srun (SLURM's MPI-like launcher) ------
# srun launches one task per node, torchrun spawns GPU workers per node
srun --kill-on-bad-exit=1 \
    torchrun \
        --nproc_per_node=$NGPU_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_backend=c10d \
        --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        -m torchtitan.train \
        --module ada_dion.integration.config_registry \
        --config "$CONFIG_FN" \
        --training.steps "$STEPS" \
        --parallelism.data_parallel_shard_degree "$DP_SHARD" \
        --parallelism.data_parallel_replicate_degree "$DP_REPLICATE" \
        --parallelism.tensor_parallel_degree 1

echo "Multi-node job $SLURM_JOB_ID finished at $(date)"
