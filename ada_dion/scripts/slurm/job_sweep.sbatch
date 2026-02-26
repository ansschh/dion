#!/bin/bash
# ============================================================
# SLURM: Hyperparameter sweep using SLURM job arrays.
# Launches 24 runs (one per array task) on single nodes.
#
# Usage:
#   sbatch job_sweep.sbatch
# ============================================================

# ------ SLURM CONFIG ------
#SBATCH --job-name=adadion-sweep
#SBATCH --partition=gpu                # <-- CHANGE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4             # <-- CHANGE
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --array=0-23                   # 24 sweep configs
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
# --------------------------

set -e

# Uncomment for your cluster:
# module load cuda/12.4
# module load anaconda3
# eval "$(conda shell.bash hook)"
# conda activate adadion

WORK_DIR="${SCRATCH:-$HOME}/ada-dion"
cd "$WORK_DIR"

NGPU=${SLURM_GPUS_PER_NODE:-4}
STEPS=5000
MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-sweep}"

mkdir -p logs

# ------ Sweep grid (24 configs) ------
# Index: optimizer, config_fn, extra_args
SWEEP_CONFIGS=(
    # AdamW: 3 LRs
    "adamw llama3_160m_adamw --optimizer.lr 1e-4"
    "adamw llama3_160m_adamw --optimizer.lr 3e-4"
    "adamw llama3_160m_adamw --optimizer.lr 1e-3"
    # Muon: 3 LRs
    "muon llama3_160m_muon --optimizer.lr 0.005"
    "muon llama3_160m_muon --optimizer.lr 0.02"
    "muon llama3_160m_muon --optimizer.lr 0.05"
    # Dion: 3 LRs x 3 rank_fracs = 9
    "dion llama3_160m_dion --optimizer.lr 0.005 --optimizer.rank_frac 0.1"
    "dion llama3_160m_dion --optimizer.lr 0.005 --optimizer.rank_frac 0.25"
    "dion llama3_160m_dion --optimizer.lr 0.005 --optimizer.rank_frac 0.5"
    "dion llama3_160m_dion --optimizer.lr 0.02 --optimizer.rank_frac 0.1"
    "dion llama3_160m_dion --optimizer.lr 0.02 --optimizer.rank_frac 0.25"
    "dion llama3_160m_dion --optimizer.lr 0.02 --optimizer.rank_frac 0.5"
    "dion llama3_160m_dion --optimizer.lr 0.05 --optimizer.rank_frac 0.1"
    "dion llama3_160m_dion --optimizer.lr 0.05 --optimizer.rank_frac 0.25"
    "dion llama3_160m_dion --optimizer.lr 0.05 --optimizer.rank_frac 0.5"
    # Dion2: 3 LRs x 3 alphas = 9
    "dion2 llama3_160m_dion2 --optimizer.lr 0.005 --optimizer.alpha 0.1"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.005 --optimizer.alpha 0.25"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.005 --optimizer.alpha 0.5"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.02 --optimizer.alpha 0.1"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.02 --optimizer.alpha 0.25"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.02 --optimizer.alpha 0.5"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.05 --optimizer.alpha 0.1"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.05 --optimizer.alpha 0.25"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.05 --optimizer.alpha 0.5"
)

# Get this task's config
CONFIG_LINE="${SWEEP_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r OPT_NAME CONFIG_FN EXTRA_ARGS <<< "$CONFIG_LINE"

export WANDB_RUN_NAME="sweep_${OPT_NAME}_task${SLURM_ARRAY_TASK_ID}"

echo "============================================"
echo "  Sweep task $SLURM_ARRAY_TASK_ID / 23"
echo "  Optimizer: $OPT_NAME"
echo "  Config: $CONFIG_FN"
echo "  Extra: $EXTRA_ARGS"
echo "============================================"

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
    --parallelism.tensor_parallel_degree 1 \
    $EXTRA_ARGS

echo "Sweep task $SLURM_ARRAY_TASK_ID finished at $(date)"
