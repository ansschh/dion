#!/bin/bash
# ============================================================
# Common environment setup for all SLURM jobs on Caltech HPC.
# Source this from sbatch scripts: source "$(dirname "$0")/env.sh"
# ============================================================

REPO=/resnick/home/atiwari2/ada-dion
VENV=/resnick/home/atiwari2/envs/adadion

# Activate venv (module load not needed — venv has everything)
module load python/3.11.6-gcc-13.2.0-fh6i4o3 2>/dev/null || true
module load cuda/12.2.1-gcc-11.3.1-sdqrj2e 2>/dev/null || true
source "$VENV/bin/activate"

# Fix namespace collision: the torchtitan/ submodule directory shadows
# the installed torchtitan package when cwd is the repo root.
# Force installed packages to take priority via PYTHONPATH.
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="$SITE_PKGS:$PYTHONPATH"

# cd to torchtitan/ so TorchTitan's relative paths resolve correctly:
#   ./tests/assets/tokenizer  (tokenizer)
#   tests/assets/c4_test      (debug dataset)
mkdir -p "$REPO/logs"
cd "$REPO/torchtitan"

# NCCL + CUDA
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
