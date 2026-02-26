"""
Tier C: Full LLM-scale experiment configurations.

Generates CLI commands for running the full experiment matrix on RunPod clusters.

C1. Pure FSDP2 (dp_shard=world_size, dp_replicate=1)
    - 8, 16, 32 GPUs
    - Shows: optimizer compute overhead + comm from orthonormalization

C2. HSDP (dp_shard=8 within node, dp_replicate=N across nodes)
    - 16 GPUs (2 nodes): dp_shard=8, dp_replicate=2
    - 32 GPUs (4 nodes): dp_shard=8, dp_replicate=4
    - Shows: Dion's low-rank sync advantage over full gradient all-reduce
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    """A single experiment run configuration."""
    name: str
    optimizer: str
    num_gpus: int
    num_nodes: int
    dp_shard: int
    dp_replicate: int
    tp_degree: int
    steps: int
    config_fn: str
    extra_args: dict


# All optimizer x config function mappings
OPTIMIZER_CONFIGS = {
    "adamw": "llama3_160m_adamw",
    "muon": "llama3_160m_muon",
    "dion": "llama3_160m_dion",
    "dion2": "llama3_160m_dion2",
}

OPTIMIZERS = list(OPTIMIZER_CONFIGS.keys())


def generate_c1_fsdp_experiments(
    steps: int = 10000,
    gpu_counts: list[int] | None = None,
) -> list[ExperimentConfig]:
    """
    C1: Pure FSDP2 experiments.
    dp_shard=world_size, dp_replicate=1, tp=1
    """
    if gpu_counts is None:
        gpu_counts = [8, 16, 32]

    experiments = []
    for num_gpus in gpu_counts:
        num_nodes = max(1, num_gpus // 8)
        for opt_name in OPTIMIZERS:
            experiments.append(ExperimentConfig(
                name=f"c1_fsdp_{opt_name}_{num_gpus}gpu",
                optimizer=opt_name,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                dp_shard=num_gpus,
                dp_replicate=1,
                tp_degree=1,
                steps=steps,
                config_fn=OPTIMIZER_CONFIGS[opt_name],
                extra_args={},
            ))

    return experiments


def generate_c2_hsdp_experiments(
    steps: int = 10000,
    gpu_counts: list[int] | None = None,
) -> list[ExperimentConfig]:
    """
    C2: HSDP experiments.
    dp_shard=8 (within node), dp_replicate=num_nodes (across nodes)
    """
    if gpu_counts is None:
        gpu_counts = [16, 32]

    experiments = []
    for num_gpus in gpu_counts:
        num_nodes = num_gpus // 8
        for opt_name in OPTIMIZERS:
            experiments.append(ExperimentConfig(
                name=f"c2_hsdp_{opt_name}_{num_gpus}gpu",
                optimizer=opt_name,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                dp_shard=8,
                dp_replicate=num_nodes,
                tp_degree=1,
                steps=steps,
                config_fn=OPTIMIZER_CONFIGS[opt_name],
                extra_args={},
            ))

    return experiments


def generate_performance_characterization(
    steps: int = 500,
) -> list[ExperimentConfig]:
    """
    Phase 1 from the blueprint: Performance characterization.
    Short runs (500 steps) to measure throughput + comm breakdown.
    """
    experiments = []

    # 8 GPUs (1 node), FSDP
    for opt_name in OPTIMIZERS:
        experiments.append(ExperimentConfig(
            name=f"perf_fsdp_{opt_name}_8gpu",
            optimizer=opt_name,
            num_gpus=8,
            num_nodes=1,
            dp_shard=8,
            dp_replicate=1,
            tp_degree=1,
            steps=steps,
            config_fn=OPTIMIZER_CONFIGS[opt_name],
            extra_args={"profiling.enable_profiling": True},
        ))

    # 16 GPUs (2 nodes), HSDP
    for opt_name in OPTIMIZERS:
        experiments.append(ExperimentConfig(
            name=f"perf_hsdp_{opt_name}_16gpu",
            optimizer=opt_name,
            num_gpus=16,
            num_nodes=2,
            dp_shard=8,
            dp_replicate=2,
            tp_degree=1,
            steps=steps,
            config_fn=OPTIMIZER_CONFIGS[opt_name],
            extra_args={"profiling.enable_profiling": True},
        ))

    return experiments


def experiment_to_cli(exp: ExperimentConfig) -> str:
    """Convert an ExperimentConfig to a torchrun CLI command."""
    gpus_per_node = exp.num_gpus // exp.num_nodes

    parts = [
        "torchrun",
        f"--nproc_per_node={gpus_per_node}",
        f"--nnodes={exp.num_nodes}",
    ]

    if exp.num_nodes > 1:
        parts.extend([
            "--node_rank=$NODE_RANK",
            "--master_addr=$MASTER_ADDR",
            "--master_port=$MASTER_PORT",
            '--rdzv_backend=c10d',
            '--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT"',
        ])
    else:
        parts.extend([
            '--rdzv_backend=c10d',
            '--rdzv_endpoint="localhost:0"',
        ])

    parts.extend([
        "-m torchtitan.train",
        "--module ada_dion.integration.config_registry",
        f"--config {exp.config_fn}",
        f"--training.steps {exp.steps}",
        f"--parallelism.data_parallel_shard_degree {exp.dp_shard}",
        f"--parallelism.data_parallel_replicate_degree {exp.dp_replicate}",
        f"--parallelism.tensor_parallel_degree {exp.tp_degree}",
    ])

    for k, v in exp.extra_args.items():
        parts.append(f"--{k} {v}")

    return " \\\n    ".join(parts)


def print_experiment_matrix():
    """Print the full experiment matrix as CLI commands."""
    sections = [
        ("Phase 1: Performance Characterization (500 steps)",
         generate_performance_characterization()),
        ("Phase 2: C1 Pure FSDP2 (10k steps)",
         generate_c1_fsdp_experiments()),
        ("Phase 2: C2 HSDP (10k steps)",
         generate_c2_hsdp_experiments()),
    ]

    for title, experiments in sections:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
        for exp in experiments:
            print(f"\n# {exp.name}")
            print(experiment_to_cli(exp))


if __name__ == "__main__":
    print_experiment_matrix()
