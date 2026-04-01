#!/bin/bash
# Part B: Medium-horizon training comparison.
# 3 optimizers x 2 seeds x 2000 steps = 6 runs on LLaMA3 320M / C4.
# With validation enabled, full W&B logging, per-layer diagnostics.
#
# Usage: WANDB_API_KEY=xxx bash ada_dion/scripts/part_b_training.sh

set -e
cd /workspace/ada-dion

NGPU=8
STEPS=2000
MODULE="ada_dion.integration.config_registry"
TOKENIZER="./assets/hf/Meta-Llama-3.1-8B"
LOGDIR="logs/part_b"
INIT_RANK=64

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set}"
export WANDB_PROJECT="gsdion-validation"
export CC=gcc CXX=g++

mkdir -p "$LOGDIR"

echo "============================================================"
echo "  Part B: Medium-Horizon Training Comparison"
echo "  3 optimizers x 2 seeds x 2000 steps = 6 runs"
echo "  Model: LLaMA3 320M, Dataset: C4, GPUs: $NGPU"
echo "  W&B: $WANDB_PROJECT"
echo "  Started: $(date)"
echo "============================================================"

RUN_IDX=0
FAILED=0

# 3 optimizers: Dion baseline, NormDion, NormDion+CS+RMS
# For NormDion variants, LR is scaled by sqrt(rank)
# Dion LR=0.02, NormDion LR=0.02*sqrt(64)=0.16

CONFIGS=(
  # name|lr|rank_normalize|use_adaptive_scalar|use_rms_matching
  "dion_baseline|0.02|false|false|false"
  "normdion|0.16|true|false|false"
  "normdion_cs_rms|0.16|true|true|true"
)

SEEDS=(42 123)

for cfg_str in "${CONFIGS[@]}"; do
  IFS='|' read -r CFG_NAME CFG_LR DO_RNORM DO_SCALAR DO_RMS <<< "$cfg_str"
  for SEED in "${SEEDS[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    RUN_NAME="${CFG_NAME}_s${SEED}"
    RUN_LOG="$LOGDIR/$RUN_NAME"
    mkdir -p "$RUN_LOG"

    echo ""
    echo "--- [$RUN_IDX/6] $RUN_NAME (lr=$CFG_LR) ---"
    export WANDB_RUN_NAME="$RUN_NAME"

    # build flags
    RNORM_FLAG="--optimizer.no-rank-normalize"
    [ "$DO_RNORM" = "true" ] && RNORM_FLAG="--optimizer.rank-normalize"

    SCALAR_FLAG="--optimizer.no-use-adaptive-scalar"
    [ "$DO_SCALAR" = "true" ] && SCALAR_FLAG="--optimizer.use-adaptive-scalar"

    RMS_FLAG="--optimizer.no-use-rms-matching"
    [ "$DO_RMS" = "true" ] && RMS_FLAG="--optimizer.use-rms-matching"

    CMD=(
      torchrun
      --nproc_per_node=$NGPU
      --rdzv_backend=c10d
      --rdzv_endpoint="localhost:0"
      --local-ranks-filter 0
      --role rank
      --tee 3
      -m torchtitan.train
      --module "$MODULE"
      --config llama3_320m_adadion
      --training.steps $STEPS
      --training.local-batch-size 32
      --training.global-batch-size 256
      --training.seq-len 2048
      --hf-assets-path "$TOKENIZER"
      --optimizer.lr "$CFG_LR"
      --optimizer.init-rank $INIT_RANK
      $RNORM_FLAG
      $SCALAR_FLAG
      $RMS_FLAG
      --optimizer.no-use-trust-region
      --optimizer.no-use-residual-rank
      --optimizer.no-use-anchor
      --parallelism.data-parallel-shard-degree $NGPU
      --parallelism.data-parallel-replicate-degree 1
      --compile.enable
      --activation-checkpoint.mode selective
      --metrics.enable-wandb
      --metrics.enable-tensorboard
      --metrics.log-freq 10
    )

    if "${CMD[@]}" 2>&1 | tee "$RUN_LOG/train.log"; then
      echo "  [DONE] $RUN_NAME"
    else
      echo "  [FAIL] $RUN_NAME (exit $?)"
      FAILED=$((FAILED + 1))
    fi
  done
done

echo ""
echo "============================================================"
echo "  Part B COMPLETE"
echo "  Finished: $(date)"
echo "  Runs: $RUN_IDX / 6"
echo "  Failed: $FAILED"
echo "============================================================"
