#!/bin/bash
# ============================================================
# One-time setup on Caltech HPC (Resnick cluster)
# Run from login node: bash setup_env.sh
# ============================================================

set -e

REPO_URL="https://github.com/ansschh/dion.git"

# Use home since no scratch env var is set
WORK_DIR="$HOME/ada-dion"
VENV_DIR="$HOME/envs/adadion"

echo "=== ada-dion Caltech HPC Setup ==="
echo "Work dir: $WORK_DIR"
echo "Venv: $VENV_DIR"

# 1. Load modules
module load python/3.11.6-gcc-13.2.0-fh6i4o3
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e
echo "Loaded python 3.11.6 + cuda 12.2.1"

# 2. Clone repo
if [ ! -d "$WORK_DIR" ]; then
    echo "Cloning repo..."
    git clone --recurse-submodules "$REPO_URL" "$WORK_DIR"
else
    echo "Repo exists, pulling latest..."
    cd "$WORK_DIR" && git pull && git submodule update --init --recursive
fi
cd "$WORK_DIR"

# 3. Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "Using python: $(which python3)"

# 4. Install PyTorch with CUDA 12.6 wheels (compatible with cluster's 12.2 driver)
# cu126 wheels require CUDA driver >=12.x and provide PyTorch 2.7+ (needed by TorchTitan)
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 5. Install TorchTitan
pip install --quiet -e torchtitan/

# 6. Install ada-dion
pip install --quiet -e ".[full,dev]"

# 7. Create log/output dirs
mkdir -p logs profiles checkpoints

# 8. Verify
python3 -c "
import torch
import ada_dion
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'ada-dion {ada_dion.__version__}')
print('Setup OK!')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To use in future sessions, run:"
echo "  module load python/3.11.6-gcc-13.2.0-fh6i4o3"
echo "  module load cuda/12.2.1-gcc-11.3.1-sdqrj2e"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $WORK_DIR"
