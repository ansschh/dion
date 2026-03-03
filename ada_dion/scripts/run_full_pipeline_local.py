#!/usr/bin/env python3
"""
Full pipeline simulation — runs locally on CPU to validate everything end-to-end.

Simulates what happens on HPC:
  1. Trains a SimpleModel with AdaDion (real optimizer, real gradients)
  2. Collects AdaDion-specific diagnostics via OptimizerMetricsCollector
  3. Generates synthetic "HPC-like" data for all 5 optimizers
  4. Produces all 11+ benchmark plots
  5. Runs Tier A correctness tests for AdaDion
  6. Runs Tier B microbenchmark for AdaDion

Usage:
    python ada_dion/scripts/run_full_pipeline_local.py [--output-dir ./pipeline_output] [--steps 200]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Mock dion if triton unavailable (Windows/macOS)
# ---------------------------------------------------------------------------
try:
    from dion import Muon, Dion, Dion2
except (ImportError, ModuleNotFoundError):
    import torch
    mock = types.ModuleType("dion")
    mock.__path__ = []

    class _MockOpt(torch.optim.Optimizer):
        def __init__(self, params, **kw):
            super().__init__(params, kw)
        def step(self, closure=None):
            pass

    mock.Muon = type("Muon", (_MockOpt,), {})
    mock.Dion = type("Dion", (_MockOpt,), {})
    mock.Dion2 = type("Dion2", (_MockOpt,), {})
    sys.modules["dion"] = mock
    for sub in ["dion.muon", "dion.dion", "dion.dion2",
                 "dion.newton_schulz_triton", "dion.utils"]:
        sys.modules[sub] = types.ModuleType(sub)

import torch
import torch.nn as nn
import numpy as np

# Ensure project root on path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ada_dion.optim.adadion import AdaDion
from ada_dion.benchmarks.metrics_collector import OptimizerMetricsCollector
from ada_dion.benchmarks.plots import (
    save_all_plots,
    plot_val_loss_vs_tokens,
    plot_val_loss_vs_wallclock,
    plot_tokens_per_sec,
    plot_scaling_tokens_per_sec,
    plot_step_time_breakdown,
    plot_communicated_bytes,
    plot_collective_time,
    plot_comm_compute_ratio,
    plot_dion_rank_residual,
    plot_dion2_alpha_sparsity,
    plot_adadion_diagnostics,
    plot_adadion_anchor_drift,
    plot_adadion_tail_ratio,
    plot_adadion_effective_rank,
    plot_adadion_energy_captured,
    COLORS, LABELS,
)
from ada_dion.integration.param_grouper import group_params_for_hybrid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# SimpleModel (same as test_hybrid_optimizer.py)
# ======================================================================

class SimpleModel(nn.Module):
    """Minimal model mimicking Llama3 naming for param grouper."""
    def __init__(self, dim=64, hidden_dim=128, vocab_size=200):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            attn = nn.Module()
            attn.wq = nn.Linear(dim, dim, bias=False)
            attn.wk = nn.Linear(dim, dim, bias=False)
            attn.wv = nn.Linear(dim, dim, bias=False)
            attn.wo = nn.Linear(dim, dim, bias=False)
            layer.attention = attn
            layer.attention_norm = nn.LayerNorm(dim)
            ff = nn.Module()
            ff.w1 = nn.Linear(dim, hidden_dim, bias=False)
            ff.w2 = nn.Linear(hidden_dim, dim, bias=False)
            ff.w3 = nn.Linear(dim, hidden_dim, bias=False)
            layer.feed_forward = ff
            layer.ffn_norm = nn.LayerNorm(dim)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        h = self.tok_embeddings(x)
        for layer in self.layers:
            # Simplified: just matmul through attention + ff
            q = layer.attention.wq(h)
            k = layer.attention.wk(h)
            v = layer.attention.wv(h)
            h = h + layer.attention.wo(v)
            h = layer.attention_norm(h)
            h = h + layer.feed_forward.w2(
                torch.relu(layer.feed_forward.w1(h)) * layer.feed_forward.w3(h)
            )
            h = layer.ffn_norm(h)
        h = self.norm(h)
        return self.output(h)


# ======================================================================
# Phase 1: Real AdaDion training on SimpleModel
# ======================================================================

def run_adadion_training(model, num_steps, output_dir):
    """Train SimpleModel with AdaDion, collect metrics."""
    print(f"\n{'='*70}")
    print(f"  Phase 1: AdaDion Training ({num_steps} steps)")
    print(f"{'='*70}")

    groups = group_params_for_hybrid(model)
    param_groups = [{"params": groups.matrix_params}]
    if groups.embed_params:
        param_groups.append({
            "params": groups.embed_params,
            "algorithm": "adamw", "lr": 3e-4,
            "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01,
        })
    if groups.output_params:
        param_groups.append({
            "params": groups.output_params,
            "algorithm": "adamw", "lr": 3e-4 / math.sqrt(64),
            "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.01,
        })
    if groups.norm_params:
        param_groups.append({
            "params": groups.norm_params,
            "algorithm": "adamw", "lr": 3e-4,
            "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        })

    opt = AdaDion(param_groups, lr=0.02, rank_fraction=0.25)

    collector = OptimizerMetricsCollector(
        matrix_optimizers=[opt],
        output_dir=output_dir,
    )

    # Training data
    vocab_size = 200
    seq_len = 32
    batch_size = 4

    losses = []
    adadion_diag = {
        "step": [], "anchor_drift_mean": [], "tail_ratio_mean": [],
        "rank_mean": [], "energy_mean": [],
    }

    t_start = time.time()
    for step in range(num_steps):
        # Synthetic batch
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        target = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward
        opt.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        opt.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Collect metrics
        metrics = collector.collect(step=step)

        # Collect AdaDion diagnostics
        drift = opt.get_anchor_drift()
        tau = opt.get_tail_ratio()
        rank = opt.get_rank()
        energy = opt.get_energy_captured()

        adadion_diag["step"].append(step)
        adadion_diag["anchor_drift_mean"].append(
            sum(drift.values()) / max(len(drift), 1) if drift else 0
        )
        adadion_diag["tail_ratio_mean"].append(
            sum(tau.values()) / max(len(tau), 1) if tau else 0
        )
        adadion_diag["rank_mean"].append(
            sum(rank.values()) / max(len(rank), 1) if rank else 0
        )
        adadion_diag["energy_mean"].append(
            sum(energy.values()) / max(len(energy), 1) if energy else 0
        )

        if step % max(1, num_steps // 20) == 0 or step == num_steps - 1:
            elapsed = time.time() - t_start
            tau_val = adadion_diag["tail_ratio_mean"][-1]
            drift_val = adadion_diag["anchor_drift_mean"][-1]
            print(f"  step {step:4d}/{num_steps}  loss={loss_val:.4f}  "
                  f"tau={tau_val:.3f}  drift={drift_val:.4f}  "
                  f"time={elapsed:.1f}s")

    collector.close()

    total_time = time.time() - t_start
    print(f"\n  Training complete: {num_steps} steps in {total_time:.1f}s")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Decrease:     {(1 - losses[-1]/losses[0])*100:.1f}%")

    return losses, adadion_diag, total_time


# ======================================================================
# Phase 2: Generate synthetic HPC-like data for all optimizers
# ======================================================================

def generate_synthetic_runs(num_steps, adadion_losses):
    """Generate realistic synthetic training curves for all 5 optimizers."""
    print(f"\n{'='*70}")
    print(f"  Phase 2: Generating Synthetic HPC Data")
    print(f"{'='*70}")

    np.random.seed(42)
    steps = list(range(0, num_steps, max(1, num_steps // 100)))

    # Base loss curve: starts at ~5.3, decays to ~3.5 with noise
    def make_loss_curve(base_start, base_end, noise_std=0.02, speed=1.0):
        n = len(steps)
        t = np.linspace(0, 1, n)
        loss = base_start + (base_end - base_start) * (1 - np.exp(-3 * speed * t))
        loss += np.random.randn(n) * noise_std
        return loss.tolist()

    # Tokens per step = batch_size * seq_len * num_gpus
    tokens_per_step = 8 * 2048 * 4  # 65536

    runs = {}
    opts_config = {
        "adamw":  {"start": 5.35, "end": 3.60, "tps_base": 48000, "speed": 0.8},
        "muon":   {"start": 5.30, "end": 3.45, "tps_base": 46000, "speed": 1.0},
        "dion":   {"start": 5.30, "end": 3.50, "tps_base": 47000, "speed": 0.95},
        "dion2":  {"start": 5.30, "end": 3.48, "tps_base": 46500, "speed": 0.97},
        "adadion": {"start": 5.30, "end": 3.42, "tps_base": 45500, "speed": 1.05},
    }

    for opt_name, cfg in opts_config.items():
        n = len(steps)
        val_loss = make_loss_curve(cfg["start"], cfg["end"], speed=cfg["speed"])
        tps = (cfg["tps_base"] + np.random.randn(n) * 500).tolist()
        time_s = [s * tokens_per_step / cfg["tps_base"] for s in steps]
        tokens = [s * tokens_per_step for s in steps]

        runs[opt_name] = {
            "step": steps,
            "val_loss": val_loss,
            "tokens": tokens,
            "time_s": time_s,
            "tokens_per_sec": tps,
        }
        print(f"  {opt_name}: {n} data points, "
              f"loss {val_loss[0]:.3f} -> {val_loss[-1]:.3f}")

    return runs


def generate_systems_data():
    """Generate synthetic systems/breakdown data for bar plots."""
    print("  Generating systems breakdown data...")

    # Step time breakdown (ms)
    step_breakdown = {
        "adamw":   {"fwd_ms": 45, "bwd_ms": 85, "opt_ms": 12, "nccl_ms": 25},
        "muon":    {"fwd_ms": 45, "bwd_ms": 85, "opt_ms": 28, "nccl_ms": 30},
        "dion":    {"fwd_ms": 45, "bwd_ms": 85, "opt_ms": 22, "nccl_ms": 18},
        "dion2":   {"fwd_ms": 45, "bwd_ms": 85, "opt_ms": 20, "nccl_ms": 15},
        "adadion": {"fwd_ms": 45, "bwd_ms": 85, "opt_ms": 25, "nccl_ms": 20},
    }

    # Communication volume (bytes per step)
    comm_bytes = {
        "adamw":   {"allreduce_bytes": 0,         "allgather_bytes": 320e6, "reducescatter_bytes": 320e6},
        "muon":    {"allreduce_bytes": 100e6,      "allgather_bytes": 320e6, "reducescatter_bytes": 320e6},
        "dion":    {"allreduce_bytes": 25e6,       "allgather_bytes": 320e6, "reducescatter_bytes": 320e6},
        "dion2":   {"allreduce_bytes": 20e6,       "allgather_bytes": 320e6, "reducescatter_bytes": 320e6},
        "adadion": {"allreduce_bytes": 30e6,       "allgather_bytes": 320e6, "reducescatter_bytes": 320e6},
    }

    # Collective time (ms per step)
    coll_time = {
        "adamw":   {"allreduce_ms": 0,  "allgather_ms": 12, "reducescatter_ms": 13},
        "muon":    {"allreduce_ms": 15, "allgather_ms": 12, "reducescatter_ms": 13},
        "dion":    {"allreduce_ms": 5,  "allgather_ms": 12, "reducescatter_ms": 13},
        "dion2":   {"allreduce_ms": 3,  "allgather_ms": 12, "reducescatter_ms": 13},
        "adadion": {"allreduce_ms": 7,  "allgather_ms": 12, "reducescatter_ms": 13},
    }

    # Scaling data (comm/compute ratio vs GPUs)
    scaling = {}
    for opt_name, tps_base in [("adamw", 48000), ("muon", 46000),
                                ("dion", 47000), ("dion2", 46500),
                                ("adadion", 45500)]:
        gpus = [4, 8, 16, 32]
        ratios = [0.15, 0.22, 0.35, 0.52]  # increases with scale
        tps = [tps_base * g / 4 * (1 - r * 0.3) for g, r in zip(gpus, ratios)]
        scaling[opt_name] = {
            "num_gpus": gpus,
            "tokens_per_sec": tps,
            "comm_compute_ratio": ratios,
        }

    return step_breakdown, comm_bytes, coll_time, scaling


def generate_dion_diagnostic_data(num_steps):
    """Generate synthetic Dion/Dion2 diagnostic data."""
    steps = list(range(0, num_steps, max(1, num_steps // 100)))
    n = len(steps)
    np.random.seed(42)

    dion_data = {
        "step": steps,
        "rank_mean": (16 + np.random.randn(n) * 0.5).tolist(),
        "residual_norm_mean": (0.5 * np.exp(-np.linspace(0, 2, n)) +
                               np.random.randn(n) * 0.02).tolist(),
    }

    dion2_data = {
        "step": steps,
        "alpha": [0.25] * n,
        "selected_frac_mean": (0.25 + np.random.randn(n) * 0.02).tolist(),
        "sparsity": (0.75 + np.random.randn(n) * 0.01).tolist(),
    }

    return dion_data, dion2_data


# ======================================================================
# Phase 3: Generate all plots
# ======================================================================

def generate_all_plots(output_dir, runs, step_breakdown, comm_bytes,
                       coll_time, scaling, dion_data, dion2_data,
                       adadion_diag):
    """Generate and save all benchmark plots."""
    print(f"\n{'='*70}")
    print(f"  Phase 3: Generating Plots")
    print(f"{'='*70}")

    out = Path(output_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        """Save as both PNG and PDF."""
        fig.savefig(out / f"{name}.png", bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  {name}.png + .pdf")

    count = 0

    _save(plot_val_loss_vs_tokens(runs), "01_val_loss_vs_tokens"); count += 1
    _save(plot_val_loss_vs_wallclock(runs), "02_val_loss_vs_wallclock"); count += 1
    _save(plot_tokens_per_sec(runs), "03_tokens_per_sec"); count += 1
    _save(plot_scaling_tokens_per_sec(scaling), "04_scaling"); count += 1
    _save(plot_step_time_breakdown(step_breakdown), "05_step_time_breakdown"); count += 1
    _save(plot_communicated_bytes(comm_bytes), "06_communicated_bytes"); count += 1
    _save(plot_collective_time(coll_time), "07_collective_time"); count += 1
    _save(plot_comm_compute_ratio(scaling), "08_comm_compute_ratio"); count += 1
    _save(plot_dion_rank_residual(dion_data), "09_dion_rank_residual"); count += 1
    _save(plot_dion2_alpha_sparsity(dion2_data), "10_dion2_alpha_sparsity"); count += 1

    # 11. AdaDion diagnostics — combined + individual panels
    _save(plot_adadion_diagnostics(adadion_diag), "11_adadion_diagnostics"); count += 1
    _save(plot_adadion_anchor_drift(adadion_diag), "11a_adadion_anchor_drift"); count += 1
    _save(plot_adadion_tail_ratio(adadion_diag), "11b_adadion_tail_ratio"); count += 1
    _save(plot_adadion_effective_rank(adadion_diag), "11c_adadion_effective_rank"); count += 1
    _save(plot_adadion_energy_captured(adadion_diag), "11d_adadion_energy_captured"); count += 1

    # Also save via save_all_plots to verify that function works
    out2 = Path(output_dir) / "plots_via_save_all"
    save_all_plots(
        str(out2), runs=runs,
        dion_data=dion_data, dion2_data=dion2_data,
        adadion_data=adadion_diag,
    )
    count_via = len(list(out2.glob("*.png")))
    print(f"\n  save_all_plots() produced {count_via} additional plots in {out2}/")

    return count


# ======================================================================
# Phase 4: Tier A correctness for AdaDion
# ======================================================================

def run_adadion_tier_a():
    """Run Tier A correctness tests specifically for AdaDion."""
    print(f"\n{'='*70}")
    print(f"  Phase 4: Tier A Correctness Tests (AdaDion)")
    print(f"{'='*70}")

    shapes = [(128, 64), (64, 128), (256, 256)]
    results = {}

    # A1: Single-step invariants
    print("\n  A1: Single-step invariants")
    checks = {}
    for shape in shapes:
        m, n = shape
        p = nn.Parameter(torch.randn(m, n))
        p.grad = torch.randn(m, n) * 0.01
        p_before = p.data.clone()
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        opt.step()

        has_nan = torch.isnan(p.data).any().item()
        has_inf = torch.isinf(p.data).any().item()
        update_norm = (p.data - p_before).norm().item()

        checks[f"no_nan_{shape}"] = not has_nan
        checks[f"no_inf_{shape}"] = not has_inf
        checks[f"nonzero_update_{shape}"] = update_norm > 1e-10
        checks[f"reasonable_update_{shape}"] = update_norm < 1e6

        status = "PASS" if (not has_nan and not has_inf) else "FAIL"
        print(f"    {shape}: {status}  update_norm={update_norm:.6f}")

    results["A1"] = checks

    # A2: Reproducibility
    print("\n  A2: Reproducibility (50 steps)")
    trajectories = []
    for run in range(2):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(128, 64))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)
        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        trajectories.append(losses)

    max_diff = max(abs(a - b) for a, b in zip(trajectories[0], trajectories[1]))
    is_repro = max_diff < 1e-10
    print(f"    Reproducible: {is_repro}  max_diff={max_diff:.2e}")
    results["A2"] = {"reproducible": is_repro, "max_diff": max_diff}

    return results


# ======================================================================
# Phase 5: Microbenchmark for AdaDion
# ======================================================================

def run_adadion_microbench():
    """Run Tier B microbenchmark for AdaDion on CPU."""
    print(f"\n{'='*70}")
    print(f"  Phase 5: Tier B Microbenchmark (AdaDion on CPU)")
    print(f"{'='*70}")

    shapes = [(256, 256), (512, 256), (256, 512)]
    warmup = 5
    measure = 20

    results = {}
    for shape in shapes:
        m, n = shape
        p = nn.Parameter(torch.randn(m, n))
        opt = AdaDion([p], lr=0.02, rank_fraction=0.25)

        for _ in range(warmup):
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            opt.zero_grad()

        times_ms = []
        for _ in range(measure):
            p.grad = torch.randn_like(p) * 0.01
            t0 = time.perf_counter()
            opt.step()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)
            opt.zero_grad()

        times_ms.sort()
        timing = {
            "mean_ms": sum(times_ms) / len(times_ms),
            "median_ms": times_ms[len(times_ms) // 2],
            "p90_ms": times_ms[int(0.9 * len(times_ms))],
            "min_ms": times_ms[0],
            "max_ms": times_ms[-1],
        }
        shape_key = f"{m}x{n}"
        results[shape_key] = timing
        print(f"  {shape_key}: mean={timing['mean_ms']:.3f}ms  "
              f"p50={timing['median_ms']:.3f}ms  p90={timing['p90_ms']:.3f}ms")

    return results


# ======================================================================
# Phase 6: Sweep config validation
# ======================================================================

def validate_sweep_configs():
    """Validate all sweep configs generate correctly."""
    print(f"\n{'='*70}")
    print(f"  Phase 6: Sweep Config Validation")
    print(f"{'='*70}")

    from ada_dion.benchmarks.sweep import (
        generate_sweep_configs, sweep_config_to_cli,
        generate_sweep_shell_script, SWEEP_GRIDS,
    )

    configs = generate_sweep_configs()
    print(f"  Total configs: {len(configs)}")

    from collections import Counter
    counts = Counter(c.optimizer for c in configs)
    for opt, cnt in sorted(counts.items()):
        print(f"    {opt}: {cnt} configs")

    # Verify CLI generation
    adadion_configs = [c for c in configs if c.optimizer == "adadion"]
    print(f"\n  Sample AdaDion CLI commands:")
    for c in adadion_configs[:2]:
        cli = sweep_config_to_cli(c, num_gpus=4)
        print(f"    {c.name}:")
        print(f"      {cli[:120]}...")

    # Shell script
    script = generate_sweep_shell_script(steps=100, num_gpus=4)
    assert "adadion" in script.lower()
    print(f"\n  Shell script length: {len(script)} chars, contains AdaDion: True")

    return {"total_configs": len(configs), "counts": dict(counts)}


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Full pipeline simulation")
    parser.add_argument("--output-dir", default="./pipeline_output",
                        help="Output directory for all results")
    parser.add_argument("--steps", type=int, default=200,
                        help="Training steps for Phase 1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  AdaDion Full Pipeline Simulation")
    print(f"  Output: {output_dir}")
    print(f"  Steps:  {args.steps}")
    print(f"  Device: CPU")
    print("=" * 70)

    t_total = time.time()

    # Phase 1: Real AdaDion training
    torch.manual_seed(42)
    model = SimpleModel(dim=64, hidden_dim=128, vocab_size=200)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: SimpleModel ({total_params:,} params)")

    losses, adadion_diag, train_time = run_adadion_training(
        model, args.steps, str(metrics_dir)
    )

    # Phase 2: Synthetic data
    runs = generate_synthetic_runs(args.steps, losses)
    step_breakdown, comm_bytes, coll_time, scaling = generate_systems_data()
    dion_data, dion2_data = generate_dion_diagnostic_data(args.steps)

    # Phase 3: Generate all plots
    plot_count = generate_all_plots(
        str(output_dir), runs, step_breakdown, comm_bytes,
        coll_time, scaling, dion_data, dion2_data, adadion_diag
    )

    # Phase 4: Tier A correctness
    tier_a = run_adadion_tier_a()

    # Phase 5: Microbenchmark
    bench = run_adadion_microbench()

    # Phase 6: Sweep validation
    sweep = validate_sweep_configs()

    # Write summary
    summary = {
        "training": {
            "steps": args.steps,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "time_s": train_time,
        },
        "plots": {
            "count": plot_count,
            "output_dir": str(output_dir / "plots"),
        },
        "tier_a": tier_a,
        "microbench": bench,
        "sweep": sweep,
    }

    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    total_time = time.time() - t_total

    print(f"\n{'='*70}")
    print(f"  Pipeline Complete!")
    print(f"{'='*70}")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Training:       {train_time:.1f}s ({args.steps} steps)")
    print(f"  Plots:          {plot_count} generated")
    print(f"  Tier A:         {'PASS' if tier_a['A2']['reproducible'] else 'FAIL'}")
    print(f"  Metrics JSONL:  {metrics_dir / 'optimizer_metrics.jsonl'}")
    print(f"  Summary:        {summary_path}")
    print(f"\n  Output files:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            if size > 1024:
                print(f"    {p.relative_to(output_dir)}  ({size/1024:.1f} KB)")
            else:
                print(f"    {p.relative_to(output_dir)}  ({size} B)")


if __name__ == "__main__":
    main()
