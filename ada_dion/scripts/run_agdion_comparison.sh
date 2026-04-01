#!/bin/bash
# 3-candidate comparison on LLaMA3 320M / C4:
#   A: Dion baseline
#   B: Dion + Anchor
#   C: Dion + Anchor + Gap + Event-triggered refresh
# 2 seeds each = 6 runs, ~2000 steps, ~1h total

set -e
cd /workspace/ada-dion

NGPU=8
STEPS=2000
MODULE="ada_dion.integration.config_registry"
TOKENIZER="./assets/hf/Meta-Llama-3.1-8B"
LOGDIR="logs/agdion"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set}"
export WANDB_PROJECT="agdion-comparison"
export CC=gcc CXX=g++

mkdir -p "$LOGDIR"

echo "============================================================"
echo "  AGDion 3-Candidate Comparison"
echo "  A: Dion baseline"
echo "  B: Dion + Anchor"
echo "  C: Dion + Anchor + Gap + Event refresh"
echo "  2 seeds x 3 configs = 6 runs, $STEPS steps each"
echo "  Started: $(date)"
echo "============================================================"

RUN_IDX=0; FAILED=0

# Configs: name|use_anchor|anchor_alpha|use_gap|use_event_refresh
CONFIGS=(
  "dion_baseline|false|0|false|false"
  "dion_anchor|true|0.1|false|false"
  "dion_anchor_gap_ev|true|0.1|true|true"
)

SEEDS=(42 123)

for cfg_str in "${CONFIGS[@]}"; do
  IFS='|' read -r CFG_NAME DO_ANC ANC_ALPHA DO_GAP DO_EVREF <<< "$cfg_str"
  for SEED in "${SEEDS[@]}"; do
    RUN_IDX=$((RUN_IDX + 1))
    RUN_NAME="${CFG_NAME}_s${SEED}"
    RUN_LOG="$LOGDIR/$RUN_NAME"
    mkdir -p "$RUN_LOG"

    echo ""
    echo "--- [$RUN_IDX/6] $RUN_NAME ---"
    export WANDB_RUN_NAME="$RUN_NAME"

    ANC_FLAG="--optimizer.no-use-anchor"
    [ "$DO_ANC" = "true" ] && ANC_FLAG="--optimizer.use-anchor --optimizer.anchor-alpha $ANC_ALPHA"

    # gap and event refresh are not in the CLI config (AGDion defaults)
    # We control them via the config: use_gap and use_event_refresh default to True
    # For baseline/anchor-only, we need to disable them

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
      $ANC_FLAG
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
echo "  Runs: $RUN_IDX / 6, Failed: $FAILED"
echo "  Finished: $(date)"
echo "============================================================"
