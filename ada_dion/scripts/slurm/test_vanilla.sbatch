#!/bin/bash
#SBATCH --job-name=vanilla
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --output=logs/vanilla_%j.out
#SBATCH --error=logs/vanilla_%j.err

source /resnick/home/atiwari2/ada-dion/ada_dion/scripts/slurm/env.sh

# Diagnostic output (appears in .out)
python3 -c "import torchtitan; print('torchtitan:', torchtitan.__file__)"
python3 -c "import os; print('cwd:', os.getcwd())"
python3 -c "import os; print('tokenizer:', os.path.isdir('./tests/assets/tokenizer'))"
python3 -c "import os; print('c4_test:', os.path.isdir('./tests/assets/c4_test'))"

export MASTER_PORT=29500

torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:$MASTER_PORT \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_debugmodel \
    --training.steps 5

echo "Done at $(date)"
