"""
Tier B: Microbenchmarks — isolate optimizer overhead cleanly.

B1. Optimizer-only step timing (synthetic tensors)
B2. Full step profiler trace setup (for use with TorchTitan's profiling)
"""
from __future__ import annotations

import json
import sys
import time
from typing import Optional

import torch
import torch.nn as nn

from ..optim.muon import Muon
from ..optim.dion import Dion
from ..optim.dion2 import Dion2


# ======================================================================
# B1: Optimizer-only step benchmark
# ======================================================================

def benchmark_optimizer_step(
    device: str = "cuda",
    shapes: Optional[list[tuple[int, int]]] = None,
    warmup_steps: int = 10,
    measure_steps: int = 100,
    dion_rank_frac: float = 0.25,
    dion2_alpha: float = 0.25,
) -> dict:
    """
    Time optimizer.step() in isolation for each optimizer.

    Creates random 2D tensors of typical LLM shapes, runs warmup steps,
    then measures step time over measure_steps.

    Returns dict with timing stats per optimizer per shape.
    """
    if shapes is None:
        shapes = [
            (4096, 4096),     # square (typical attention)
            (4096, 11008),    # MLP up-projection
            (11008, 4096),    # MLP down-projection
        ]

    optimizers_config = [
        ("AdamW", torch.optim.AdamW, {"lr": 3e-4, "weight_decay": 0.1}),
        ("Muon", Muon, {"lr": 0.02, "mu": 0.95, "ns_steps": 5}),
        ("Dion", Dion, {"lr": 0.02, "rank_frac": dion_rank_frac, "beta": 0.05}),
        ("Dion2", Dion2, {"lr": 0.02, "alpha": dion2_alpha, "selection": "top_l1", "mu": 0.95}),
    ]

    results = {}

    for shape in shapes:
        m, n = shape
        shape_key = f"{m}x{n}"
        results[shape_key] = {}

        for opt_name, opt_cls, opt_kwargs in optimizers_config:
            # Create parameter
            p = nn.Parameter(torch.randn(m, n, device=device, dtype=torch.bfloat16))
            optimizer = opt_cls([p], **opt_kwargs)

            # Warmup
            for _ in range(warmup_steps):
                p.grad = torch.randn_like(p) * 0.01
                optimizer.step()
                optimizer.zero_grad()

            # Synchronize before measuring
            if device == "cuda":
                torch.cuda.synchronize()

            # Measure
            times_ms = []
            for _ in range(measure_steps):
                p.grad = torch.randn_like(p) * 0.01

                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                optimizer.step()

                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                times_ms.append((t1 - t0) * 1000)
                optimizer.zero_grad()

            times_ms.sort()
            results[shape_key][opt_name] = {
                "mean_ms": sum(times_ms) / len(times_ms),
                "median_ms": times_ms[len(times_ms) // 2],
                "p90_ms": times_ms[int(0.9 * len(times_ms))],
                "min_ms": times_ms[0],
                "max_ms": times_ms[-1],
            }

            # Cleanup
            del p, optimizer
            if device == "cuda":
                torch.cuda.empty_cache()

    return results


# ======================================================================
# B2: Full step profiler trace configuration
# ======================================================================

def get_profiling_config(
    warmup_steps: int = 5,
    active_steps: int = 20,
) -> dict:
    """
    Return profiling configuration to use with TorchTitan's ProfilingConfig.

    The user should set these in the Trainer.Config:
        profiling=ProfilingConfig(
            enable_profiling=True,
            save_traces_folder="./profiles",
            profile_freq=active_steps,
        )

    After profiling, use parse_profiler_trace() to extract metrics.
    """
    return {
        "torchtitan_config": {
            "profiling.enable_profiling": True,
            "profiling.save_traces_folder": "./profiles",
            "profiling.profile_freq": warmup_steps + active_steps,
        },
        "instructions": [
            f"Run with --profiling.enable_profiling true "
            f"--profiling.save_traces_folder ./profiles "
            f"--profiling.profile_freq {warmup_steps + active_steps}",
            "Collect Chrome trace from ./profiles/",
            "Use parse_profiler_trace() to extract NCCL collective breakdown",
        ],
        "metrics_to_extract": [
            "time_forward_ms",
            "time_backward_ms",
            "time_optimizer_ms",
            "time_nccl_total_ms",
            "nccl_allreduce_count",
            "nccl_allgather_count",
            "nccl_reducescatter_count",
            "nccl_bytes_total",
        ],
    }


def parse_profiler_trace(trace_path: str) -> dict:
    """
    Parse a Chrome trace JSON exported by PyTorch profiler.

    Extracts:
      - Time in NCCL collectives (all-reduce, all-gather, reduce-scatter)
      - Per-collective call counts
      - Inferred bytes per collective

    Args:
        trace_path: Path to the Chrome trace JSON file.

    Returns:
        Dict with aggregated metrics.
    """
    with open(trace_path) as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])

    nccl_collectives = {
        "allreduce": {"count": 0, "total_us": 0},
        "allgather": {"count": 0, "total_us": 0},
        "reducescatter": {"count": 0, "total_us": 0},
        "other": {"count": 0, "total_us": 0},
    }

    forward_us = 0
    backward_us = 0
    optimizer_us = 0

    for event in events:
        name = event.get("name", "").lower()
        dur = event.get("dur", 0)

        # NCCL collectives
        if "nccl" in name:
            if "allreduce" in name or "all_reduce" in name:
                nccl_collectives["allreduce"]["count"] += 1
                nccl_collectives["allreduce"]["total_us"] += dur
            elif "allgather" in name or "all_gather" in name:
                nccl_collectives["allgather"]["count"] += 1
                nccl_collectives["allgather"]["total_us"] += dur
            elif "reducescatter" in name or "reduce_scatter" in name:
                nccl_collectives["reducescatter"]["count"] += 1
                nccl_collectives["reducescatter"]["total_us"] += dur
            else:
                nccl_collectives["other"]["count"] += 1
                nccl_collectives["other"]["total_us"] += dur

    # Convert to ms
    result = {}
    total_nccl_ms = 0
    for coll_type, stats in nccl_collectives.items():
        ms = stats["total_us"] / 1000
        result[f"nccl_{coll_type}_count"] = stats["count"]
        result[f"nccl_{coll_type}_ms"] = ms
        total_nccl_ms += ms
    result["nccl_total_ms"] = total_nccl_ms

    return result


# ======================================================================
# Runner
# ======================================================================

def run_all_tier_b(device: str = "cuda") -> dict:
    """Run all Tier B benchmarks."""
    results = {}

    print("=" * 60)
    print("Tier B1: Optimizer-only step timing")
    print("=" * 60)
    b1 = benchmark_optimizer_step(device=device)
    results["B1_step_timing"] = b1

    for shape_key, opt_results in b1.items():
        print(f"\n  Shape: {shape_key}")
        for opt_name, timing in opt_results.items():
            print(
                f"    {opt_name:8s}: "
                f"mean={timing['mean_ms']:.3f}ms  "
                f"p50={timing['median_ms']:.3f}ms  "
                f"p90={timing['p90_ms']:.3f}ms"
            )

    print()
    print("=" * 60)
    print("Tier B2: Profiler trace configuration")
    print("=" * 60)
    b2 = get_profiling_config()
    results["B2_profiler_config"] = b2
    for instruction in b2["instructions"]:
        print(f"  {instruction}")

    return results


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    results = run_all_tier_b(device=device)
    print("\n" + json.dumps(results, indent=2, default=str))
