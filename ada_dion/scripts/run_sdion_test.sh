#!/bin/bash
# SDion Test A + B on LLaMA3 320M / C4
#
# Test A: Plain Dion with drift/tail logging (2 seeds)
#   → Measures headroom: how often do drift/tail spike?
#
# Test B: 4-way screening (2 seeds each)
#   1. Dion baseline (enable_skip=false, all off)
#   2. Skip-only Dion (skip enabled, no consensus/anchor)
#   3. Skip + consensus
#   4. Skip + consensus + decaying anchor
#
# Total: 8 runs x 2000 steps

set -e
cd /workspace/ada-dion

NGPU=8
STEPS=2000
MODULE="ada_dion.integration.config_registry"
TOKENIZER="./assets/hf/Meta-Llama-3.1-8B"
LOGDIR="logs/sdion"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set}"
export WANDB_PROJECT="sdion-test"
export CC=gcc CXX=g++

mkdir -p "$LOGDIR"

echo "============================================================"
echo "  SDion Test A+B: Skip-Dion Screening"
echo "  4 configs x 2 seeds = 8 runs, $STEPS steps each"
echo "  Started: $(date)"
echo "============================================================"

RUN_IDX=0; FAILED=0
SEEDS=(42 123)

# All configs use SDion via the AdaDion name in config_registry
# SDion defaults: enable_skip=True, enable_consensus=True,
# enable_anchor=True, enable_recovery=True
# We override via CLI flags

# Config array: name|enable_skip|enable_consensus|enable_anchor
CONFIGS=(
  "dion_baseline|false|false|false"
  "skip_only|true|false|false"
  "skip_consensus|true|true|false"
  "skip_cons_anchor|true|true|true"
)

for cfg_str in "${CONFIGS[@]}"; do
  IFS='|' read -r CFG_NAME DO_SKIP DO_CONS DO_ANC <<< "$cfg_str"
  for SEED in "${SEEDS[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    RUN_NAME="${CFG_NAME}_s${SEED}"
    RUN_LOG="$LOGDIR/$RUN_NAME"
    mkdir -p "$RUN_LOG"

    echo ""
    echo "--- [$RUN_IDX/8] $RUN_NAME ---"
    export WANDB_RUN_NAME="$RUN_NAME"

    SKIP_FLAG="--optimizer.no-enable-skip"
    [ "$DO_SKIP" = "true" ] && SKIP_FLAG="--optimizer.enable-skip"

    CONS_FLAG="--optimizer.no-enable-consensus"
    [ "$DO_CONS" = "true" ] && CONS_FLAG="--optimizer.enable-consensus"

    ANC_FLAG="--optimizer.no-enable-anchor"
    [ "$DO_ANC" = "true" ] && ANC_FLAG="--optimizer.enable-anchor"

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
      --optimizer.lr 0.02
      --optimizer.init-rank 64
      $SKIP_FLAG
      $CONS_FLAG
      $ANC_FLAG
      --optimizer.enable-recovery
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
echo "  COMPLETE"
echo "  Runs: $RUN_IDX / 8, Failed: $FAILED"
echo "  Finished: $(date)"
echo "============================================================"
