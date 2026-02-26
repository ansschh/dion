#!/bin/bash
# Full experiment matrix from the benchmark blueprint.
#
# Phase 1: Performance characterization (500 steps, 8 + 16 GPUs)
# Phase 2: Convergence vs wallclock (10k steps, 16 GPUs HSDP, 2-3 seeds)
# Phase 3: Ablations (Dion2 alpha sweep, selection method comparison)
#
# Usage:
#   bash ada_dion/scripts/run_all_experiments.sh phase1    # Only Phase 1
#   bash ada_dion/scripts/run_all_experiments.sh phase2    # Only Phase 2
#   bash ada_dion/scripts/run_all_experiments.sh phase3    # Only Phase 3
#   bash ada_dion/scripts/run_all_experiments.sh all       # Everything

set -e

PHASE=${1:-all}

export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-benchmark}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OPTIMIZERS="adamw muon dion dion2"

# ============================================================
# Phase 1: Performance Characterization (500 steps)
# ============================================================
run_phase1() {
    echo "============================================================"
    echo "  PHASE 1: Performance Characterization"
    echo "============================================================"

    # Single node (8 GPUs, FSDP)
    for opt in $OPTIMIZERS; do
        echo "--- Phase 1: $opt, 8 GPUs, FSDP ---"
        STEPS=500 NGPU=8 bash ada_dion/scripts/run_single_node.sh "$opt" \
            --profiling.enable_profiling true \
            --profiling.save_traces_folder "./profiles/phase1_${opt}_8gpu"
    done

    # Multi-node (16 GPUs, HSDP) — requires 2 nodes
    if [ "${NUM_NODES:-1}" -ge 2 ]; then
        for opt in $OPTIMIZERS; do
            echo "--- Phase 1: $opt, 16 GPUs, HSDP ---"
            STEPS=500 bash ada_dion/scripts/run_multi_node.sh "$opt" \
                --profiling.enable_profiling true \
                --profiling.save_traces_folder "./profiles/phase1_${opt}_16gpu"
        done
    else
        echo "Skipping 16-GPU experiments (need NUM_NODES >= 2)"
    fi
}

# ============================================================
# Phase 2: Convergence (10k steps, best config only, 2-3 seeds)
# ============================================================
run_phase2() {
    echo "============================================================"
    echo "  PHASE 2: Convergence vs Wallclock"
    echo "============================================================"

    for seed in 42 123 7; do
        for opt in $OPTIMIZERS; do
            echo "--- Phase 2: $opt, seed=$seed, 16 GPUs, HSDP ---"
            export WANDB_RUN_NAME="convergence_${opt}_seed${seed}"

            if [ "${NUM_NODES:-1}" -ge 2 ]; then
                STEPS=10000 bash ada_dion/scripts/run_multi_node.sh "$opt" \
                    --debug.seed "$seed" \
                    --validator.freq 200 \
                    --validator.steps 50
            else
                # Fallback: single-node FSDP
                STEPS=10000 NGPU=8 bash ada_dion/scripts/run_single_node.sh "$opt" \
                    --debug.seed "$seed" \
                    --validator.freq 200 \
                    --validator.steps 50
            fi
        done
    done
}

# ============================================================
# Phase 3: Ablations (Dion2 only)
# ============================================================
run_phase3() {
    echo "============================================================"
    echo "  PHASE 3: Dion2 Ablations"
    echo "============================================================"

    # Alpha sweep
    for alpha in 0.25 0.5 1.0; do
        echo "--- Phase 3: Dion2 alpha=$alpha, top_l1 ---"
        export WANDB_RUN_NAME="ablation_dion2_alpha${alpha}_topl1"
        STEPS=5000 NGPU=8 bash ada_dion/scripts/run_single_node.sh dion2 \
            --optimizer.alpha "$alpha" \
            --optimizer.selection top_l1
    done

    # Selection method comparison
    for sel in top_l1 random; do
        echo "--- Phase 3: Dion2 alpha=0.25, selection=$sel ---"
        export WANDB_RUN_NAME="ablation_dion2_sel_${sel}"
        STEPS=5000 NGPU=8 bash ada_dion/scripts/run_single_node.sh dion2 \
            --optimizer.alpha 0.25 \
            --optimizer.selection "$sel"
    done
}

# ============================================================
# Dispatch
# ============================================================
case "$PHASE" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    phase3) run_phase3 ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        echo "=== All experiments complete ==="
        ;;
    *)
        echo "Usage: $0 [phase1|phase2|phase3|all]"
        exit 1
        ;;
esac
