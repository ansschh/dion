#!/bin/bash
# Local validation using fake_backend (no GPU required).
# Tests that configs parse correctly and optimizer logic runs without errors.
#
# Usage:
#   bash ada_dion/scripts/run_local.sh              # test all optimizers
#   bash ada_dion/scripts/run_local.sh muon          # test specific optimizer

set -e

OPTIMIZER=${1:-"all"}
STEPS=${STEPS:-5}

echo "=== Local validation (fake_backend) ==="

run_test() {
    local config=$1
    echo ""
    echo "--- Testing config: $config ---"
    NGPU=4 COMM_MODE="fake_backend" \
    python -m torchtitan.train \
        --module ada_dion.integration.config_registry \
        --config "$config" \
        --training.steps "$STEPS" \
        --metrics.enable_tensorboard false \
        --metrics.enable_wandb false
    echo "--- $config: OK ---"
}

if [ "$OPTIMIZER" = "all" ]; then
    run_test llama3_debug_muon
    run_test llama3_debug_dion
    run_test llama3_debug_dion2
    echo ""
    echo "=== All configs validated successfully ==="
elif [ "$OPTIMIZER" = "muon" ]; then
    run_test llama3_debug_muon
elif [ "$OPTIMIZER" = "dion" ]; then
    run_test llama3_debug_dion
elif [ "$OPTIMIZER" = "dion2" ]; then
    run_test llama3_debug_dion2
else
    echo "Unknown optimizer: $OPTIMIZER"
    echo "Usage: $0 [all|muon|dion|dion2]"
    exit 1
fi
