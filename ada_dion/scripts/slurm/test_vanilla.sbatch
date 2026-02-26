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

source /resnick/home/atiwari2/envs/adadion/bin/activate
mkdir -p /resnick/home/atiwari2/ada-dion/logs
cd /resnick/home/atiwari2/ada-dion/logs

torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 -m torchtitan.train --module llama3 --config llama3_debugmodel --training.steps 5

echo "Done at $(date)"
