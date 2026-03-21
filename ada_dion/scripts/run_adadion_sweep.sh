#!/bin/bash
# AdaDion HP sweep on RunPod (4 GPUs)
# Total configs: 4 lr x 3 init_rank x 2 adaptive_rank x 2 mu = 48
# Usage: nohup bash ada_dion/scripts/run_adadion_sweep.sh > adadion_sweep.log 2>&1 &

set -e
cd /workspace/ada-dion

export NGPU=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set}"
export WANDB_MODE=online
export WANDB_PROJECT="ada-dion-sweep"

LOGDIR="logs/adadion_sweep"
mkdir -p "$LOGDIR"

STEPS=2000
RUN_IDX=0
TOTAL=48

echo "=============================================="
echo "  AdaDion HP Sweep (RunPod, ${NGPU} GPUs)"
echo "  Total configs: $TOTAL"
echo "  Steps per run: $STEPS"
echo "  W&B project:   $WANDB_PROJECT"
echo "  Started:        $(date)"
echo "=============================================="

for LR in 0.005 0.01 0.02 0.05; do
  for INIT_RANK in 32 64 128; do
    for ADAPTIVE in true false; do
      for MU in 0.9 0.95; do
        RUN_IDX=$((RUN_IDX + 1))
        RUN_NAME="adadion_lr${LR}_r${INIT_RANK}_adapt${ADAPTIVE}_mu${MU}"

        echo ""
        echo "--- [$RUN_IDX/$TOTAL] $RUN_NAME ---"

        torchrun \
          --nproc_per_node=$NGPU \
          --rdzv_backend=c10d \
          --rdzv_endpoint="localhost:0" \
          -m torchtitan.train \
          --module ada_dion.integration.config_registry \
          --config llama3_160m_adadion \
          --training.steps "$STEPS" \
          --optimizer.lr "$LR" \
          --optimizer.init_rank "$INIT_RANK" \
          --optimizer.adaptive_rank "$ADAPTIVE" \
          --optimizer.mu "$MU" \
          --metrics.enable_wandb true \
          --metrics.wandb_project "$WANDB_PROJECT" \
          --metrics.wandb_run_name "$RUN_NAME" \
          > "$LOGDIR/${RUN_NAME}.out" 2> "$LOGDIR/${RUN_NAME}.err" \
          && echo "  $RUN_NAME: DONE" \
          || echo "  $RUN_NAME: FAILED (check $LOGDIR/${RUN_NAME}.err)"
      done
    done
  done
done

echo ""
echo "=============================================="
echo "  AdaDion HP Sweep COMPLETE"
echo "  Finished: $(date)"
echo "  Logs:     $LOGDIR/"
echo "  Runs:     $RUN_IDX / $TOTAL"
echo "=============================================="
