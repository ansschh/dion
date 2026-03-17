#!/bin/bash
# Run all experiment phases on RunPod (4x A100 80GB, no SLURM)
# Usage: nohup bash ada_dion/scripts/run_runpod_all.sh > runpod_all.log 2>&1 &

set -e
cd /workspace/ada-dion

export NGPU=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="${WANDB_MODE:-disabled}"

LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "=============================================="
echo "  AdaDion Full Experiment Pipeline (RunPod)"
echo "  GPUs: $NGPU x A100-80GB"
echo "  Started: $(date)"
echo "=============================================="

# ── Phase 1: Performance characterization (500 steps) ──
echo ""
echo "=== PHASE 1: Performance Characterization (500 steps) ==="
for opt in adamw muon dion dion2 adadion; do
    echo ""
    echo "--- Phase 1: $opt (500 steps) ---"
    STEPS=500 NGPU=$NGPU bash ada_dion/scripts/run_single_node.sh $opt \
        > "$LOGDIR/p1_${opt}.out" 2> "$LOGDIR/p1_${opt}.err" \
        && echo "  $opt: DONE" \
        || echo "  $opt: FAILED (check $LOGDIR/p1_${opt}.err)"
done
echo ""
echo "Phase 1 complete: $(date)"

# Check for Phase 1 errors
P1_ERRORS=0
for opt in adamw muon dion dion2 adadion; do
    if grep -q "Error\|FAILED\|Traceback" "$LOGDIR/p1_${opt}.err" 2>/dev/null; then
        echo "ERROR in Phase 1 $opt! Aborting."
        tail -20 "$LOGDIR/p1_${opt}.err"
        P1_ERRORS=1
    fi
done
if [ "$P1_ERRORS" -eq 1 ]; then
    echo "Phase 1 had errors. Stopping."
    exit 1
fi
echo "Phase 1 clean. Proceeding to Phase 2."

# ── Phase 2: Convergence analysis (10k steps, 3 seeds) ──
echo ""
echo "=== PHASE 2: Convergence Analysis (10000 steps, 3 seeds) ==="
for seed in 42 123 7; do
    for opt in adamw muon dion dion2 adadion; do
        echo ""
        echo "--- Phase 2: $opt seed=$seed (10000 steps) ---"
        STEPS=10000 NGPU=$NGPU bash ada_dion/scripts/run_single_node.sh $opt \
            --training.seed "$seed" \
            --metrics.log_freq 1 \
            > "$LOGDIR/p2_${opt}_s${seed}.out" 2> "$LOGDIR/p2_${opt}_s${seed}.err" \
            && echo "  $opt seed=$seed: DONE" \
            || echo "  $opt seed=$seed: FAILED (check $LOGDIR/p2_${opt}_s${seed}.err)"
    done
done
echo ""
echo "Phase 2 complete: $(date)"

# ── Phase 3: Dion2 Ablations (5k steps) ──
echo ""
echo "=== PHASE 3: Dion2 Ablations (5000 steps) ==="

# Alpha sweep
for alpha in 0.25 0.5 1.0; do
    echo ""
    echo "--- Phase 3: dion2 alpha=$alpha (5000 steps) ---"
    STEPS=5000 NGPU=$NGPU bash ada_dion/scripts/run_single_node.sh dion2 \
        --optimizer.fraction "$alpha" \
        > "$LOGDIR/p3_dion2_a${alpha}.out" 2> "$LOGDIR/p3_dion2_a${alpha}.err" \
        && echo "  dion2 alpha=$alpha: DONE" \
        || echo "  dion2 alpha=$alpha: FAILED"
done

# Selection method: top_l1 vs random
for sel in top_l1 random; do
    echo ""
    echo "--- Phase 3: dion2 selection=$sel (5000 steps) ---"
    STEPS=5000 NGPU=$NGPU bash ada_dion/scripts/run_single_node.sh dion2 \
        --optimizer.selection_method "$sel" \
        > "$LOGDIR/p3_dion2_sel_${sel}.out" 2> "$LOGDIR/p3_dion2_sel_${sel}.err" \
        && echo "  dion2 selection=$sel: DONE" \
        || echo "  dion2 selection=$sel: FAILED"
done

echo ""
echo "=============================================="
echo "  ALL PHASES COMPLETE"
echo "  Finished: $(date)"
echo "  Logs: $LOGDIR/"
echo "=============================================="
echo ""
echo "Summary:"
echo "  Phase 1: 5 runs x 500 steps"
echo "  Phase 2: 15 runs x 10000 steps (5 opts x 3 seeds)"
echo "  Phase 3: 5 runs x 5000 steps (ablations)"
echo "  Total: 25 runs"
