#!/bin/bash
# AdaDion 320M HP sweep — adapted from Tatsu's run.sh
# LLaMA3 320M, C4 dataset, 2048 seq len, FSDP, 8 GPUs
# Global batch 256, local batch 32 (8 per micro-step × 4 grad accum)
# Token budget 3.2B → 6104 steps
#
# Usage: WANDB_API_KEY=xxx nohup bash ada_dion/scripts/run_320m_sweep.sh > sweep_320m.log 2>&1 &

set -e
cd /workspace/ada-dion

# ============================================================
# Environment
# ============================================================
NGPU=8
SEQ_LEN=2048
# Matches Tatsu's setup exactly: dim=768, 18 layers, MHA
# local_batch=32 fits on A100-80GB with dim=768 model
LOCAL_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=256  # 32 * 8 = 256, no grad accum needed
TOKEN_BUDGET=3200000000
TOKENS_PER_STEP=$((GLOBAL_BATCH_SIZE * SEQ_LEN))  # 524288
STEPS=$(( (TOKEN_BUDGET + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))  # 6104
SEED=0
MODULE="ada_dion.integration.config_registry"
TOKENIZER_PATH="./assets/hf/Meta-Llama-3.1-8B"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY must be set}"
export WANDB_PROJECT="ada-dion-320m-sweep"
export CC=gcc CXX=g++

LOGDIR="logs/sweep_320m"
mkdir -p "$LOGDIR"

echo "============================================================"
echo "  AdaDion 320M HP Sweep"
echo "============================================================"
echo "  Model:       LLaMA3 320M"
echo "  GPUs:        $NGPU"
echo "  Batch:       $GLOBAL_BATCH_SIZE (local: $LOCAL_BATCH_SIZE)"
echo "  Seq len:     $SEQ_LEN"
echo "  Steps:       $STEPS (3.2B tokens)"
echo "  W&B:         $WANDB_PROJECT"
echo "  Started:     $(date)"
echo "============================================================"

# ============================================================
# Sweep configs
# ============================================================
RUN_IDX=0
FAILED=0

# LR × init_rank × adaptive × mu = 4×3×2×2 = 48 configs
for LR in 0.005 0.01 0.02 0.05; do
  for INIT_RANK in 32 64 128; do
    for ADAPTIVE in true false; do
      for MU in 0.9 0.95; do
        RUN_IDX=$((RUN_IDX + 1))
        RUN_NAME="adadion_320m_lr${LR}_r${INIT_RANK}_adapt${ADAPTIVE}_mu${MU}"
        RUN_LOG_DIR="$LOGDIR/$RUN_NAME"
        mkdir -p "$RUN_LOG_DIR"

        echo ""
        echo "--- [$RUN_IDX/48] $RUN_NAME ---"

        export WANDB_RUN_NAME="$RUN_NAME"

        # Build adaptive-rank flag
        if [ "$ADAPTIVE" = "true" ]; then
            ADAPT_FLAG="--optimizer.adaptive-rank"
        else
            ADAPT_FLAG="--optimizer.no-adaptive-rank"
        fi

        CMD=(
            torchrun
            --nproc_per_node="$NGPU"
            --rdzv_backend=c10d
            --rdzv_endpoint="localhost:0"
            --local-ranks-filter 0
            --role rank
            --tee 3
            -m torchtitan.train
            --module "$MODULE"
            --config llama3_320m_adadion
            --training.steps "$STEPS"
            --training.local-batch-size $LOCAL_BATCH_SIZE
            --training.global-batch-size $GLOBAL_BATCH_SIZE
            --training.seq-len "$SEQ_LEN"
            --hf-assets-path "$TOKENIZER_PATH"
            --optimizer.lr "$LR"
            --optimizer.init-rank "$INIT_RANK"
            $ADAPT_FLAG
            --optimizer.mu "$MU"
            --parallelism.data-parallel-shard-degree "$NGPU"
            --parallelism.data-parallel-replicate-degree 1
            --metrics.enable-wandb
            --metrics.enable-tensorboard
            --metrics.log-freq 10
            --activation-checkpoint.mode selective
            --activation-checkpoint.selective-ac-option 2
            --validator.freq 100
            --validator.steps 20
        )

        if "${CMD[@]}" 2>&1 | tee "$RUN_LOG_DIR/train.log"; then
            echo "  [DONE] $RUN_NAME"
        else
            echo "  [FAIL] $RUN_NAME (exit $?)"
            FAILED=$((FAILED + 1))
        fi
      done
    done
  done
done

echo ""
echo "============================================================"
echo "  Sweep COMPLETE"
echo "  Finished:   $(date)"
echo "  Total:      $RUN_IDX / 48"
echo "  Failed:     $FAILED"
echo "============================================================"
