#!/usr/bin/env python3
"""
Parse SLURM log files from TorchTitan runs and generate benchmark plots.

Usage:
    python generate_plots.py --logs-dir ~/ada-dion/logs --output-dir ~/ada-dion/plots

Expects log files named:
    - {jobid}_adadion.out       (Phase 1 baselines, Phase 3 multi-node)
    - sweep_{arrayjobid}_{taskid}.out  (Phase 2 HP sweep)
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Add project root to path so we can import ada_dion ──────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ada_dion.benchmarks.plots import (
    COLORS, LABELS, _setup_style, _finalize_ax, _place_legend,
    plot_val_loss_vs_tokens,
    plot_val_loss_vs_wallclock,
    plot_tokens_per_sec,
    plot_scaling_tokens_per_sec,
)

# ── Log parsing ─────────────────────────────────────────────────────

STEP_RE = re.compile(
    r"\[titan\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\d+.*"
    r"step:\s*(\d+)\s+"
    r"loss:\s*([\d.]+)\s+"
    r"grad_norm:\s*([\d.]+)\s+"
    r"memory:\s*([\d.]+)GiB.*"
    r"tps:\s*([\d,]+)\s+"
    r"tflops:\s*([\d.]+)\s+"
    r"mfu:\s*([\d.]+)%"
)

OPTIMIZER_RE = re.compile(r"Optimizer:\s*(\w+)")
SWEEP_RE = re.compile(r"Sweep\s+(\d+)/\d+:\s+(\w+)\s+\|(.+)")


def parse_log(path: Path) -> dict:
    """Parse a TorchTitan SLURM log file into structured data."""
    text = path.read_text(errors="replace")

    # Determine optimizer name
    m = OPTIMIZER_RE.search(text)
    opt_name = m.group(1) if m else None

    # Check for sweep config
    m = SWEEP_RE.search(text)
    sweep_info = None
    if m:
        sweep_info = {
            "task_id": int(m.group(1)),
            "optimizer": m.group(2),
            "extra_args": m.group(3).strip(),
        }
        opt_name = opt_name or m.group(2)

    # Parse step lines (deduplicate — each rank prints the same line)
    seen_steps = {}
    for m in STEP_RE.finditer(text):
        timestamp_str = m.group(1)
        step = int(m.group(2))
        if step in seen_steps:
            continue
        seen_steps[step] = {
            "timestamp": datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S"),
            "step": step,
            "loss": float(m.group(3)),
            "grad_norm": float(m.group(4)),
            "memory_gib": float(m.group(5)),
            "tps": int(m.group(6).replace(",", "")),
            "tflops": float(m.group(7)),
            "mfu": float(m.group(8)),
        }

    records = [seen_steps[s] for s in sorted(seen_steps)]

    # Compute wall-clock seconds relative to first step
    if records:
        t0 = records[0]["timestamp"]
        for r in records:
            r["time_s"] = (r["timestamp"] - t0).total_seconds()

    return {
        "optimizer": opt_name,
        "sweep_info": sweep_info,
        "records": records,
    }


def records_to_run(records: list[dict], seq_len: int = 2048,
                   local_batch: int = 8, num_gpus: int = 4) -> dict:
    """Convert parsed records into the dict format expected by plots.py."""
    tokens_per_step = local_batch * seq_len * num_gpus
    return {
        "step": [r["step"] for r in records],
        "val_loss": [r["loss"] for r in records],
        "tokens": [r["step"] * tokens_per_step for r in records],
        "time_s": [r["time_s"] for r in records],
        "tokens_per_sec": [r["tps"] for r in records],
        "memory_gib": [r["memory_gib"] for r in records],
        "mfu": [r["mfu"] for r in records],
        "grad_norm": [r["grad_norm"] for r in records],
    }


# ── Custom plots for data we have ──────────────────────────────────

def plot_sweep_results(sweep_data: list[dict]) -> plt.Figure:
    """HP sweep: final loss per config, grouped by optimizer."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    # Group by optimizer
    by_opt = {}
    for d in sweep_data:
        info = d["sweep_info"]
        opt = info["optimizer"]
        if opt not in by_opt:
            by_opt[opt] = []
        final_loss = d["records"][-1]["loss"] if d["records"] else float("nan")
        by_opt[opt].append((info["extra_args"], final_loss))

    # Plot as grouped bars
    all_labels = []
    all_vals = []
    all_colors = []
    for opt in ["adamw", "muon", "dion", "dion2"]:
        if opt not in by_opt:
            continue
        configs = by_opt[opt]
        for args, loss in configs:
            # Shorten label
            short = args.replace("--optimizer.", "").replace(" ", "=")
            label = f"{LABELS.get(opt, opt)}\n{short}"
            all_labels.append(label)
            all_vals.append(loss)
            all_colors.append(COLORS.get(opt, "gray"))

    x = np.arange(len(all_labels))
    bars = ax.bar(x, all_vals, color=all_colors, zorder=3, edgecolor="white",
                  linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, all_vals):
        if val < 5:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Final Loss @ 5k steps")
    ax.set_title("Hyperparameter Sweep Results")
    _finalize_ax(ax)
    fig.tight_layout()
    return fig


def plot_loss_comparison_bar(baselines: dict, multi_node: dict) -> plt.Figure:
    """Side-by-side bar chart: final loss for 4 GPU vs 8 GPU."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    opts = ["adamw", "muon", "dion", "dion2"]
    opts_present = [o for o in opts if o in baselines and o in multi_node]

    x = np.arange(len(opts_present))
    width = 0.35

    vals_4gpu = []
    vals_8gpu = []
    for opt in opts_present:
        bl = baselines[opt]
        mn = multi_node[opt]
        vals_4gpu.append(bl[-1]["loss"] if bl else float("nan"))
        vals_8gpu.append(mn[-1]["loss"] if mn else float("nan"))

    bars1 = ax.bar(x - width / 2, vals_4gpu, width, label="4 GPU (FSDP)",
                   color="#4c78a8", zorder=3, edgecolor="white")
    bars2 = ax.bar(x + width / 2, vals_8gpu, width, label="8 GPU (HSDP)",
                   color="#f58518", zorder=3, edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(o, o) for o in opts_present])
    ax.set_ylabel("Final Loss @ 10k steps")
    ax.set_title("Single-Node vs Multi-Node Final Loss")
    _finalize_ax(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_throughput_bar(baselines: dict, multi_node: dict) -> plt.Figure:
    """Side-by-side bar: steady-state TPS for 4 GPU vs 8 GPU."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    opts = ["adamw", "muon", "dion", "dion2"]
    opts_present = [o for o in opts if o in baselines and o in multi_node]

    x = np.arange(len(opts_present))
    width = 0.35

    def median_tps(records):
        # Use median of last 20% of steps for steady-state
        if not records:
            return 0
        n = max(1, len(records) // 5)
        return float(np.median([r["tps"] for r in records[-n:]]))

    vals_4gpu = [median_tps(baselines[o]) for o in opts_present]
    vals_8gpu = [median_tps(multi_node[o]) for o in opts_present]

    bars1 = ax.bar(x - width / 2, vals_4gpu, width, label="4 GPU (FSDP)",
                   color="#4c78a8", zorder=3, edgecolor="white")
    bars2 = ax.bar(x + width / 2, vals_8gpu, width, label="8 GPU (HSDP)",
                   color="#f58518", zorder=3, edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 500,
                    f"{h:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(o, o) for o in opts_present])
    ax.set_ylabel("Tokens/sec (steady-state median)")
    ax.set_title("Throughput: Single-Node vs Multi-Node")
    _finalize_ax(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_memory_bar(baselines: dict) -> plt.Figure:
    """Bar chart of peak memory usage per optimizer."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    opts = ["adamw", "muon", "dion", "dion2"]
    opts_present = [o for o in opts if o in baselines and baselines[o]]

    vals = []
    colors = []
    for opt in opts_present:
        peak = max(r["memory_gib"] for r in baselines[opt])
        vals.append(peak)
        colors.append(COLORS.get(opt, "gray"))

    x = np.arange(len(opts_present))
    bars = ax.bar(x, vals, color=colors, zorder=3, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.1f} GiB", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(o, o) for o in opts_present])
    ax.set_ylabel("Peak Memory (GiB)")
    ax.set_title("Peak GPU Memory Usage (4x H200)")
    ax.axhline(y=139.81, color="gray", linestyle="--", alpha=0.3, label="H200 capacity")
    _finalize_ax(ax)
    fig.tight_layout()
    return fig


def plot_mfu_over_time(baselines: dict) -> plt.Figure:
    """MFU over training steps for each optimizer."""
    _setup_style()
    fig, ax = plt.subplots()

    for opt_name, records in baselines.items():
        if not records:
            continue
        color = COLORS.get(opt_name, "gray")
        label = LABELS.get(opt_name, opt_name)
        steps = [r["step"] for r in records]
        mfu = [r["mfu"] for r in records]
        ax.plot(steps, mfu, color=color, label=label, linewidth=1.5, alpha=0.8, zorder=3)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("MFU (%)")
    ax.set_title("Model FLOPS Utilization Over Training")
    _finalize_ax(ax)
    _place_legend(ax, loc="lower right")
    fig.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots from SLURM logs")
    parser.add_argument("--logs-dir", required=True, help="Directory containing SLURM .out files")
    parser.add_argument("--output-dir", default="plots", help="Directory to save plot PNGs")

    # Job IDs
    parser.add_argument("--baseline-ids", nargs=4, required=True,
                        help="4 job IDs for adamw muon dion dion2 baselines")
    parser.add_argument("--sweep-id", required=True,
                        help="Array job ID for HP sweep")
    parser.add_argument("--multinode-ids", nargs=4, required=True,
                        help="4 job IDs for adamw muon dion dion2 multi-node")
    args = parser.parse_args()

    logs = Path(args.logs_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    opt_order = ["adamw", "muon", "dion", "dion2"]

    # ── Parse Phase 1: Baselines ──
    print("Parsing Phase 1: Baselines...")
    baselines = {}
    for job_id, opt in zip(args.baseline_ids, opt_order):
        log_path = logs / f"{job_id}_adadion.out"
        if not log_path.exists():
            print(f"  WARNING: {log_path} not found, skipping {opt}")
            continue
        data = parse_log(log_path)
        baselines[opt] = data["records"]
        print(f"  {opt}: {len(data['records'])} step records")

    # ── Parse Phase 2: Sweep ──
    print("Parsing Phase 2: HP Sweep...")
    sweep_data = []
    for task_id in range(24):
        log_path = logs / f"sweep_{args.sweep_id}_{task_id}.out"
        if not log_path.exists():
            print(f"  WARNING: {log_path} not found, skipping task {task_id}")
            continue
        data = parse_log(log_path)
        if data["sweep_info"]:
            sweep_data.append(data)
            final = data["records"][-1]["loss"] if data["records"] else "N/A"
            print(f"  task {task_id} ({data['sweep_info']['optimizer']}): final_loss={final}")

    # ── Parse Phase 3: Multi-node ──
    print("Parsing Phase 3: Multi-node...")
    multi_node = {}
    for job_id, opt in zip(args.multinode_ids, opt_order):
        log_path = logs / f"{job_id}_adadion-multi.out"
        if not log_path.exists():
            print(f"  WARNING: {log_path} not found, skipping {opt}")
            continue
        data = parse_log(log_path)
        multi_node[opt] = data["records"]
        print(f"  {opt}: {len(data['records'])} step records")

    # ── Generate plots ──
    print(f"\nGenerating plots to {output}...")
    plot_count = 0

    # Plot 1: Loss vs tokens (baselines)
    if baselines:
        runs = {opt: records_to_run(records, num_gpus=4)
                for opt, records in baselines.items() if records}
        fig = plot_val_loss_vs_tokens(runs)
        fig.savefig(output / "01_loss_vs_tokens.png", bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print("  01_loss_vs_tokens.png")

    # Plot 2: Loss vs wall-clock
    if baselines:
        runs = {opt: records_to_run(records, num_gpus=4)
                for opt, records in baselines.items() if records}
        fig = plot_val_loss_vs_wallclock(runs)
        fig.savefig(output / "02_loss_vs_wallclock.png", bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print("  02_loss_vs_wallclock.png")

    # Plot 3: Tokens/sec vs step
    if baselines:
        runs = {opt: records_to_run(records, num_gpus=4)
                for opt, records in baselines.items() if records}
        fig = plot_tokens_per_sec(runs)
        fig.savefig(output / "03_tokens_per_sec.png", bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print("  03_tokens_per_sec.png")

    # Plot 4: Scaling (4 GPU vs 8 GPU)
    if baselines and multi_node:
        scaling_runs = {}
        for opt in opt_order:
            if opt in baselines and opt in multi_node and baselines[opt] and multi_node[opt]:
                bl_records = baselines[opt]
                mn_records = multi_node[opt]
                # Median TPS from last 20%
                n_bl = max(1, len(bl_records) // 5)
                n_mn = max(1, len(mn_records) // 5)
                tps_4 = float(np.median([r["tps"] for r in bl_records[-n_bl:]]))
                tps_8 = float(np.median([r["tps"] for r in mn_records[-n_mn:]]))
                scaling_runs[opt] = {
                    "num_gpus": [4, 8],
                    "tokens_per_sec": [tps_4, tps_8],
                }
        if scaling_runs:
            fig = plot_scaling_tokens_per_sec(scaling_runs)
            fig.savefig(output / "04_scaling.png", bbox_inches="tight")
            plt.close(fig)
            plot_count += 1
            print("  04_scaling.png")

    # Plot 5: HP sweep results
    if sweep_data:
        fig = plot_sweep_results(sweep_data)
        fig.savefig(output / "05_sweep_results.png", bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print("  05_sweep_results.png")

    # Plot 6: Final loss comparison (4 GPU vs 8 GPU)
    if baselines and multi_node:
        fig = plot_loss_comparison_bar(baselines, multi_node)
        fig.savefig(output / "06_loss_4gpu_vs_8gpu.png", bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print("  06_loss_4gpu_vs_8gpu.png")

    # Plot 7: Throughput comparison (4 GPU vs 8 GPU)
    if baselines and multi_node:
        fig = plot_throughput_bar(baselines, multi_node)
        fig.savefig(output / "07_throughput_4gpu_vs_8gpu.png", bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print("  07_throughput_4gpu_vs_8gpu.png")

    # Plot 8: Memory usage
    if baselines:
        fig = plot_memory_bar(baselines)
        fig.savefig(output / "08_memory_usage.png", bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print("  08_memory_usage.png")

    # Plot 9: MFU over training
    if baselines:
        fig = plot_mfu_over_time(baselines)
        fig.savefig(output / "09_mfu_over_time.png", bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print("  09_mfu_over_time.png")

    # Plot 10: Multi-node loss curves
    if multi_node:
        _setup_style()
        fig, ax = plt.subplots()
        for opt_name, records in multi_node.items():
            if not records:
                continue
            run = records_to_run(records, num_gpus=8)
            color = COLORS.get(opt_name, "gray")
            label = LABELS.get(opt_name, opt_name)
            ax.plot(run["tokens"], run["val_loss"], color=color, label=label,
                    linewidth=2, zorder=3)
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Loss")
        ax.set_title("Multi-Node HSDP (8 GPU): Loss vs Tokens")
        _finalize_ax(ax)
        _place_legend(ax, loc="upper right")
        fig.tight_layout()
        fig.savefig(output / "10_multinode_loss_vs_tokens.png", bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print("  10_multinode_loss_vs_tokens.png")

    print(f"\nDone! {plot_count} plots saved to {output}/")


if __name__ == "__main__":
    main()
