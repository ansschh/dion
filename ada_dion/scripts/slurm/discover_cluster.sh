#!/bin/bash
# ============================================================
# Run this FIRST on the Caltech HPC to discover what's available.
# Usage: bash discover_cluster.sh
# Then paste the output back to me so I can tune the SLURM configs.
# ============================================================

echo "============================================"
echo "  Cluster Discovery Report"
echo "============================================"
echo ""

echo "--- 1. Hostname & OS ---"
hostname
uname -a
echo ""

echo "--- 2. Available Partitions & GPUs ---"
sinfo -o "%20P %10a %5D %20G %10l" 2>/dev/null || echo "sinfo not available"
echo ""

echo "--- 3. GPU Types (detailed) ---"
sinfo -o "%20P %20G %10D %20f" 2>/dev/null || echo "sinfo not available"
echo ""

echo "--- 4. GPUs per node ---"
sinfo -N -o "%20N %10G %10c %10m" 2>/dev/null | head -20 || echo "sinfo not available"
echo ""

echo "--- 5. Available Modules (PyTorch/CUDA/Anaconda) ---"
module avail 2>&1 | grep -iE "pytorch|torch|cuda|anaconda|conda|python|nccl" || echo "No matching modules found"
echo ""

echo "--- 6. Existing conda envs ---"
conda env list 2>/dev/null || echo "conda not available"
echo ""

echo "--- 7. Python & PyTorch ---"
which python3 2>/dev/null && python3 --version 2>/dev/null
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not importable from default python"
echo ""

echo "--- 8. NVIDIA GPUs on login node (if any) ---"
nvidia-smi -L 2>/dev/null || echo "No GPUs on login node (normal)"
echo ""

echo "--- 9. Scratch / workspace directories ---"
echo "HOME=$HOME"
ls -d /scratch/$USER 2>/dev/null && echo "Scratch: /scratch/$USER"
ls -d /central/scratch/$USER 2>/dev/null && echo "Central scratch: /central/scratch/$USER"
ls -d $SCRATCH 2>/dev/null && echo "SCRATCH=$SCRATCH"
echo ""

echo "--- 10. Quota ---"
quota -s 2>/dev/null | head -10 || echo "quota command not available"
echo ""

echo "============================================"
echo "  Copy everything above and paste it back"
echo "============================================"
