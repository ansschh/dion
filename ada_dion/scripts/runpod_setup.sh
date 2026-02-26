#!/bin/bash
# RunPod environment setup for ada-dion benchmarks
# Run this on each pod in the cluster.
#
# Prerequisites: RunPod Instant Cluster with H100 80GB pods
# Expected env vars set by RunPod: MASTER_ADDR, MASTER_PORT, NUM_NODES, NODE_RANK

set -e

echo "=== ada-dion RunPod Setup ==="
echo "Node: ${NODE_RANK:-0} / ${NUM_NODES:-1}"
echo "Master: ${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500}"

# ---- 1. Install PyTorch (if not pre-installed) ----
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || true

# ---- 2. Clone and install ada-dion ----
cd /workspace
if [ ! -d "ada-dion" ]; then
    git clone --recurse-submodules https://github.com/YOUR_USERNAME/ada-dion.git
fi
cd ada-dion

# Install TorchTitan in editable mode
pip install --quiet -e torchtitan/

# Install ada-dion
pip install --quiet -e ".[full,dev]"

# ---- 3. Install experiment dependencies ----
pip install --quiet wandb tensorboard datasets

# ---- 4. Download tokenizer assets ----
# For LLaMA3, we need the tokenizer. Use the test tokenizer for now.
# For real experiments, download: python torchtitan/scripts/download_hf_assets.py \
#     --repo_id meta-llama/Llama-3.1-8B --assets tokenizer
echo "Using test tokenizer from torchtitan/tests/assets/tokenizer"

# ---- 5. Configure NCCL for RunPod ----
export NCCL_SOCKET_IFNAME=ens1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- 6. W&B setup ----
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
    echo "W&B authenticated."
else
    echo "WARNING: WANDB_API_KEY not set. Set it for experiment logging."
fi

echo "=== Setup complete ==="
echo "Run experiments with: bash ada_dion/scripts/run_single_node.sh muon"
