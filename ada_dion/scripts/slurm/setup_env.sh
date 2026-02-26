#!/bin/bash
# ============================================================
# One-time setup on Caltech HPC: clone repo, create conda env, install deps.
# Run this from the login node ONCE.
#
# Usage: bash setup_env.sh
# ============================================================

set -e

# ------ CONFIG (edit these) ------
WORK_DIR="${SCRATCH:-$HOME}/ada-dion"
CONDA_ENV_NAME="adadion"
REPO_URL="https://github.com/ansschh/dion.git"
# ---------------------------------

echo "=== ada-dion Caltech HPC Setup ==="
echo "Work dir: $WORK_DIR"
echo "Conda env: $CONDA_ENV_NAME"

# 1. Load modules (uncomment/edit as needed for your cluster)
# module load cuda/12.4
# module load anaconda3
# module load python/3.11

# 2. Clone repo
if [ ! -d "$WORK_DIR" ]; then
    echo "Cloning repo..."
    git clone --recurse-submodules "$REPO_URL" "$WORK_DIR"
else
    echo "Repo already exists at $WORK_DIR, pulling latest..."
    cd "$WORK_DIR" && git pull && git submodule update --init --recursive
fi
cd "$WORK_DIR"

# 3. Create conda environment (if conda is available)
if command -v conda &>/dev/null; then
    if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
        echo "Creating conda env: $CONDA_ENV_NAME"
        conda create -n "$CONDA_ENV_NAME" python=3.11 -y
    fi
    echo "Activating conda env..."
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
else
    echo "WARNING: conda not found. Using system python."
    echo "You may need to: module load anaconda3"
fi

# 4. Install PyTorch with CUDA
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || \
    pip install --quiet torch torchvision

# 5. Install TorchTitan
pip install --quiet -e torchtitan/

# 6. Install ada-dion
pip install --quiet -e ".[full,dev]"

# 7. Verify
python -c "
import torch
import ada_dion
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'ada-dion {ada_dion.__version__}')
print('Setup OK!')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate $CONDA_ENV_NAME"
echo "Submit jobs from: $WORK_DIR"
