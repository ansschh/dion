#!/bin/bash
# ============================================================
# SLURM: Single-node training (1 node, all GPUs)
#
# Usage:
#   sbatch job_single_node.sbatch muon        # Run Muon
#   sbatch job_single_node.sbatch adamw       # Run AdamW
#   sbatch job_single_node.sbatch dion        # Run Dion
#   sbatch job_single_node.sbatch dion2       # Run Dion2
#
# Or override defaults:
#   STEPS=500 sbatch job_single_node.sbatch muon
# ============================================================

# ------ SLURM CONFIG (edit for your cluster) ------
#SBATCH --job-name=adadion
#SBATCH --partition=gpu                # <-- CHANGE to your GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4             # <-- CHANGE to GPUs per node (4 or 8)
#SBATCH --cpus-per-task=32            # <-- CHANGE: ~8 CPUs per GPU is good
#SBATCH --mem=256G                    # <-- CHANGE if needed
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
# ---------------------------------------------------

set -e

# Parse optimizer from command line arg
OPTIMIZER=${1:-muon}
STEPS=${STEPS:-10000}
NGPU=${SLURM_GPUS_PER_NODE:-4}

echo "============================================"
echo "  Job: $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  Optimizer: $OPTIMIZER"
echo "  GPUs: $NGPU"
echo "  Steps: $STEPS"
echo "============================================"

# ------ Environment setup ------
# Uncomment/edit for your cluster:
# module load cuda/12.4
# module load anaconda3
# eval "$(conda shell.bash hook)"
# conda activate adadion

# Or if using a venv:
# source ~/envs/adadion/bin/activate

# ------ Work directory ------
WORK_DIR="${SCRATCH:-$HOME}/ada-dion"
cd "$WORK_DIR"

# ------ NCCL settings ------
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

# W&B (optional)
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-caltech}"
export WANDB_RUN_NAME="${OPTIMIZER}_${NGPU}gpu_${STEPS}steps_${SLURM_JOB_ID}"

# ------ Create log dir ------
mkdir -p logs profiles

# ------ Map optimizer to config ------
case "$OPTIMIZER" in
    adamw) CONFIG_FN="llama3_160m_adamw" ;;
    muon)  CONFIG_FN="llama3_160m_muon" ;;
    dion)  CONFIG_FN="llama3_160m_dion" ;;
    dion2) CONFIG_FN="llama3_160m_dion2" ;;
    *)
        echo "Unknown optimizer: $OPTIMIZER"
        echo "Options: adamw, muon, dion, dion2"
        exit 1
        ;;
esac

# ------ Launch training ------
echo "Launching torchrun with $NGPU GPUs..."

torchrun \
    --nproc_per_node=$NGPU \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:$MASTER_PORT" \
    -m torchtitan.train \
    --module ada_dion.integration.config_registry \
    --config "$CONFIG_FN" \
    --training.steps "$STEPS" \
    --parallelism.data_parallel_shard_degree "$NGPU" \
    --parallelism.data_parallel_replicate_degree 1 \
    --parallelism.tensor_parallel_degree 1

echo "Job $SLURM_JOB_ID finished at $(date)"
