#!/bin/bash
# Patch TorchTitan to recognize ada_dion as an experiment module.
# Run once after cloning / installing.
set -e

REPO_DIR="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
TT_EXP="$REPO_DIR/torchtitan/torchtitan/experiments"
SHIM_SRC="$REPO_DIR/ada_dion/integration/torchtitan_shim"

if [ ! -d "$TT_EXP" ]; then
    echo "ERROR: torchtitan experiments dir not found at $TT_EXP"
    exit 1
fi

# 1. Copy shim files
mkdir -p "$TT_EXP/ada_dion"
cp "$SHIM_SRC/__init__.py" "$TT_EXP/ada_dion/__init__.py"
cp "$SHIM_SRC/config_registry.py" "$TT_EXP/ada_dion/config_registry.py"

# 2. Register in _supported_experiments if not already present
if ! grep -q '"ada_dion"' "$TT_EXP/__init__.py"; then
    sed -i 's/"ft.llama3",/"ft.llama3",\n        "ada_dion",/' "$TT_EXP/__init__.py"
    echo "Registered ada_dion in _supported_experiments"
else
    echo "ada_dion already registered"
fi

echo "TorchTitan patched successfully."
