#!/bin/bash
# ============================================================
# SLURM: HP sweep via job array (24 configs, 1 node x 4 H200 each)
#
# Usage:
#   sbatch job_sweep.sbatch
# ============================================================

#SBATCH --job-name=adadion-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err

set -e

# Environment (module load may fail on some compute nodes, venv has everything)
module load python/3.11.6-gcc-13.2.0-fh6i4o3 2>/dev/null || true
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e 2>/dev/null || true
source "$HOME/envs/adadion/bin/activate"

cd "$HOME/ada-dion"
mkdir -p logs

NGPU=4
STEPS=5000
MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-sweep}"

# 24 sweep configs
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

CONFIG_LINE="${SWEEP_CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r OPT_NAME CONFIG_FN EXTRA_ARGS <<< "$CONFIG_LINE"

export WANDB_RUN_NAME="sweep_${OPT_NAME}_task${SLURM_ARRAY_TASK_ID}"

echo "Sweep $SLURM_ARRAY_TASK_ID/23: $OPT_NAME | $EXTRA_ARGS"

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
    --parallelism.tensor_parallel_degree 1 \
    $EXTRA_ARGS

echo "Sweep task $SLURM_ARRAY_TASK_ID done at $(date)"
