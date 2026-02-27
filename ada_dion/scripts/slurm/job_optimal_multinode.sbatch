#!/bin/bash
# ============================================================
# SLURM: Optimal HP multi-node HSDP (2 nodes x 4 H200 = 8 GPUs)
#
# Usage:
#   sbatch job_optimal_multinode.sbatch
# ============================================================

#SBATCH --job-name=adadion-opt-multi
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/optimal_multi_%A_%a.out
#SBATCH --error=logs/optimal_multi_%A_%a.err

set -e

NGPU_PER_NODE=4
NNODES=$SLURM_NNODES
WORLD_SIZE=$(( NNODES * NGPU_PER_NODE ))
DP_SHARD=$NGPU_PER_NODE
DP_REPLICATE=$NNODES
STEPS=10000

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))

# Optimal configs from HP sweep
CONFIGS=(
    "adamw llama3_160m_adamw --optimizer.lr 1e-3"
    "muon llama3_160m_muon --optimizer.lr 0.005"
    "dion llama3_160m_dion --optimizer.lr 0.005 --optimizer.rank_frac 0.5"
    "dion2 llama3_160m_dion2 --optimizer.lr 0.005 --optimizer.alpha 0.5"
)

CONFIG_LINE="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r OPT_NAME CONFIG_FN EXTRA_ARGS <<< "$CONFIG_LINE"

echo "============================================"
echo "  Optimal Multi-Node: $OPT_NAME"
echo "  Nodes: $NNODES ($SLURM_NODELIST)"
echo "  World size: $WORLD_SIZE GPUs"
echo "  HSDP: shard=$DP_SHARD, replicate=$DP_REPLICATE"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "============================================"

source /resnick/home/atiwari2/ada-dion/ada_dion/scripts/slurm/env.sh

export MASTER_ADDR MASTER_PORT
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-optimal}"
export WANDB_RUN_NAME="optimal_${OPT_NAME}_${WORLD_SIZE}gpu_hsdp_${SLURM_ARRAY_JOB_ID}"

REPO=/resnick/home/atiwari2/ada-dion
VENV=/resnick/home/atiwari2/envs/adadion

srun --kill-on-bad-exit=1 bash -c "
    source $VENV/bin/activate
    SITE_PKGS=\$(python3 -c \"import site; print(site.getsitepackages()[0])\")
    export PYTHONPATH=\"\$SITE_PKGS:\$PYTHONPATH\"
    cd $REPO/torchtitan

    torchrun \
        --nproc_per_node=$NGPU_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m torchtitan.train \
        --module ada_dion \
        --config $CONFIG_FN \
        --training.steps $STEPS \
        --metrics.log_freq 1 \
        --parallelism.data_parallel_shard_degree $DP_SHARD \
        --parallelism.data_parallel_replicate_degree $DP_REPLICATE \
        --parallelism.tensor_parallel_degree 1 \
        $EXTRA_ARGS
"

echo "Optimal multi-node $OPT_NAME done at $(date)"
