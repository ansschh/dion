"""
Hyperparameter sweep driver.

Generates a structured small sweep per optimizer:
  - AdamW: lr (3 values)
  - Muon: lr (3 values)
  - Dion: lr (3 values) x rank_frac (3 values) = 9 configs
  - Dion2: lr (3 values) x alpha (3 values) = 9 configs

Total: 3 + 3 + 9 + 9 = 24 runs.
Each run: 5000 steps on 8 GPUs FSDP.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any


@dataclass
class SweepConfig:
    """A single sweep run configuration."""
    optimizer: str
    config_fn: str
    overrides: dict[str, Any]
    name: str


# ======================================================================
# Sweep grids
# ======================================================================

SWEEP_GRIDS = {
    "adamw": {
        "lr": [1e-4, 3e-4, 1e-3],
    },
    "muon": {
        "lr": [0.005, 0.02, 0.05],
    },
    "dion": {
        "lr": [0.005, 0.02, 0.05],
        "rank_frac": [0.1, 0.25, 0.5],
    },
    "dion2": {
        "lr": [0.005, 0.02, 0.05],
        "alpha": [0.1, 0.25, 0.5],
    },
}

CONFIG_FN_MAP = {
    "adamw": "llama3_160m_adamw",
    "muon": "llama3_160m_muon",
    "dion": "llama3_160m_dion",
    "dion2": "llama3_160m_dion2",
}

# Map sweep param names to CLI override paths
PARAM_CLI_MAP = {
    "lr": "optimizer.lr",
    "rank_frac": "optimizer.rank_frac",
    "alpha": "optimizer.alpha",
}


def generate_sweep_configs(
    steps: int = 5000,
    num_gpus: int = 8,
) -> list[SweepConfig]:
    """Generate all sweep run configurations."""
    configs = []

    for opt_name, grid in SWEEP_GRIDS.items():
        param_names = list(grid.keys())
        param_values = list(grid.values())

        for combo in itertools.product(*param_values):
            kwargs = dict(zip(param_names, combo))

            # Build name
            parts = [opt_name]
            for k, v in kwargs.items():
                parts.append(f"{k}{v}")
            name = "_".join(parts)

            # Build CLI overrides
            overrides = {
                "training.steps": steps,
                f"parallelism.data_parallel_shard_degree": num_gpus,
            }
            for k, v in kwargs.items():
                cli_key = PARAM_CLI_MAP.get(k, f"optimizer.{k}")
                overrides[cli_key] = v

            configs.append(SweepConfig(
                optimizer=opt_name,
                config_fn=CONFIG_FN_MAP[opt_name],
                overrides=overrides,
                name=name,
            ))

    return configs


def sweep_config_to_cli(config: SweepConfig, num_gpus: int = 8) -> str:
    """Convert a SweepConfig to a torchrun CLI command."""
    parts = [
        f"torchrun --nproc_per_node={num_gpus}",
        '--rdzv_backend=c10d --rdzv_endpoint="localhost:0"',
        "-m torchtitan.train",
        "--module ada_dion.integration.config_registry",
        f"--config {config.config_fn}",
    ]

    for k, v in config.overrides.items():
        parts.append(f"--{k} {v}")

    return " \\\n    ".join(parts)


def print_sweep_matrix(steps: int = 5000, num_gpus: int = 8):
    """Print all sweep commands."""
    configs = generate_sweep_configs(steps=steps, num_gpus=num_gpus)

    print(f"Total sweep runs: {len(configs)}")
    print(f"Steps per run: {steps}")
    print(f"GPUs: {num_gpus}")
    print()

    for opt_name in SWEEP_GRIDS:
        opt_configs = [c for c in configs if c.optimizer == opt_name]
        print(f"\n{'=' * 60}")
        print(f"  {opt_name.upper()} ({len(opt_configs)} runs)")
        print(f"{'=' * 60}")
        for c in opt_configs:
            print(f"\n# {c.name}")
            print(sweep_config_to_cli(c, num_gpus))


def generate_sweep_shell_script(
    steps: int = 5000,
    num_gpus: int = 8,
) -> str:
    """Generate a bash script that runs all sweep configs sequentially."""
    configs = generate_sweep_configs(steps=steps, num_gpus=num_gpus)

    lines = [
        "#!/bin/bash",
        "set -e",
        "",
        f"# Hyperparameter sweep: {len(configs)} runs",
        f"# Steps per run: {steps}",
        f"# GPUs: {num_gpus}",
        "",
        'export WANDB_PROJECT="${WANDB_PROJECT:-ada-dion-sweep}"',
        "",
    ]

    for config in configs:
        lines.append(f"echo '=== Running: {config.name} ==='")
        lines.append(f"export WANDB_RUN_NAME='{config.name}'")
        lines.append(sweep_config_to_cli(config, num_gpus))
        lines.append("")

    lines.append("echo 'Sweep complete!'")
    return "\n".join(lines)


if __name__ == "__main__":
    print_sweep_matrix()
