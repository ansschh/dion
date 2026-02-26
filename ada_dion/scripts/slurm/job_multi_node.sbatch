#!/bin/bash
# ============================================================
# SLURM: Multi-node HSDP on Caltech HPC (N nodes x 4x H200)
#
# Usage:
#   sbatch job_multi_node.sbatch muon              # 2 nodes = 8 GPUs
#   sbatch --nodes=4 job_multi_node.sbatch dion    # 4 nodes = 16 GPUs
# ============================================================

#SBATCH --job-name=adadion-multi
#SBATCH --partition=gpu
#SBATCH --nodes=2
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
NGPU_PER_NODE=4
NNODES=$SLURM_NNODES
WORLD_SIZE=$(( NNODES * NGPU_PER_NODE ))

# HSDP: shard within node, replicate across nodes
DP_SHARD=$NGPU_PER_NODE
DP_REPLICATE=$NNODES

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))

echo "============================================"
echo "  Multi-node: $SLURM_JOB_ID"
echo "  Nodes: $NNODES ($SLURM_NODELIST)"
echo "  World size: $WORLD_SIZE GPUs"
echo "  HSDP: shard=$DP_SHARD, replicate=$DP_REPLICATE"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  Optimizer: $OPTIMIZER | Steps: $STEPS"
echo "============================================"

# Environment (module load may fail on some compute nodes, venv has everything)
module load python/3.11.6-gcc-13.2.0-fh6i4o3 2>/dev/null || true
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e 2>/dev/null || true
source "$HOME/envs/adadion/bin/activate"

cd "$HOME/ada-dion"
mkdir -p logs

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MASTER_ADDR MASTER_PORT
export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-caltech}"
export WANDB_RUN_NAME="${OPTIMIZER}_${WORLD_SIZE}gpu_hsdp_${SLURM_JOB_ID}"

case "$OPTIMIZER" in
    adamw) CONFIG_FN="llama3_160m_adamw" ;;
    muon)  CONFIG_FN="llama3_160m_muon" ;;
    dion)  CONFIG_FN="llama3_160m_dion" ;;
    dion2) CONFIG_FN="llama3_160m_dion2" ;;
    *)     echo "Unknown: $OPTIMIZER"; exit 1 ;;
esac

srun --kill-on-bad-exit=1 bash -c "
    module load python/3.11.6-gcc-13.2.0-fh6i4o3 2>/dev/null || true
    module load cuda/12.2.1-gcc-11.3.1-sdqrj2e 2>/dev/null || true
    source $HOME/envs/adadion/bin/activate

    torchrun \
        --nproc_per_node=$NGPU_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m torchtitan.train \
        --module ada_dion.integration.config_registry \
        --config $CONFIG_FN \
        --training.steps $STEPS \
        --parallelism.data_parallel_shard_degree $DP_SHARD \
        --parallelism.data_parallel_replicate_degree $DP_REPLICATE \
        --parallelism.tensor_parallel_degree 1
"

echo "Done at $(date)"
