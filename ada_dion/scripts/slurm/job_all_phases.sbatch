#!/bin/bash
# ============================================================
# SLURM: Submit all experiment phases as dependent jobs.
#
# Usage:
#   bash job_all_phases.sh
#
# This is a LAUNCHER (not a sbatch script) — run it from the login node.
# It submits Phase 1, 2, 3 jobs with dependencies so they run in order.
# ============================================================

set -e

WORK_DIR="${SCRATCH:-$HOME}/ada-dion"
cd "$WORK_DIR"
mkdir -p logs

echo "=== Submitting all experiment phases ==="

# ------ Phase 1: Performance characterization (500 steps, single node) ------
echo ""
echo "--- Phase 1: Performance Characterization ---"
PHASE1_JOBS=""
for opt in adamw muon dion dion2; do
    JOB_ID=$(sbatch --parsable \
        --job-name="p1_${opt}" \
        --export="ALL,STEPS=500" \
        ada_dion/scripts/slurm/job_single_node.sbatch "$opt")
    echo "  Submitted $opt: job $JOB_ID"
    PHASE1_JOBS="${PHASE1_JOBS}:${JOB_ID}"
done

# ------ Phase 2: Convergence (10k steps, 3 seeds) ------
echo ""
echo "--- Phase 2: Convergence (after Phase 1 completes) ---"
PHASE2_JOBS=""
for seed in 42 123 7; do
    for opt in adamw muon dion dion2; do
        JOB_ID=$(sbatch --parsable \
            --dependency=afterany${PHASE1_JOBS} \
            --job-name="p2_${opt}_s${seed}" \
            --export="ALL,STEPS=10000" \
            ada_dion/scripts/slurm/job_single_node.sbatch "$opt")
        echo "  Submitted $opt seed=$seed: job $JOB_ID"
        PHASE2_JOBS="${PHASE2_JOBS}:${JOB_ID}"
    done
done

# ------ Phase 3: Dion2 ablations (5k steps) ------
echo ""
echo "--- Phase 3: Dion2 Ablations (after Phase 1 completes) ---"
# Alpha sweep
for alpha in 0.25 0.5 1.0; do
    JOB_ID=$(sbatch --parsable \
        --dependency=afterany${PHASE1_JOBS} \
        --job-name="p3_a${alpha}" \
        --export="ALL,STEPS=5000" \
        ada_dion/scripts/slurm/job_single_node.sbatch dion2)
    echo "  Submitted Dion2 alpha=$alpha: job $JOB_ID"
done

# Selection method comparison
for sel in top_l1 random; do
    JOB_ID=$(sbatch --parsable \
        --dependency=afterany${PHASE1_JOBS} \
        --job-name="p3_sel_${sel}" \
        --export="ALL,STEPS=5000" \
        ada_dion/scripts/slurm/job_single_node.sbatch dion2)
    echo "  Submitted Dion2 selection=$sel: job $JOB_ID"
done

echo ""
echo "=== All phases submitted ==="
echo "Monitor with: squeue -u $USER"
echo "Cancel all with: scancel -u $USER"
