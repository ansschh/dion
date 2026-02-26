"""
Benchmark plotting module — all 11 required plots.

Reads from W&B runs, TensorBoard event files, or JSON lines logs.
Each function returns a matplotlib Figure that can be saved or displayed.

Plots:
  1. Val loss vs tokens
  2. Val loss vs wall-clock time
  3. Tokens/sec vs step
  4. Scaling: tokens/sec vs #GPUs
  5. Step time breakdown (stacked bar)
  6. Total communicated bytes/step
  7. Collective time/step
  8. Comm/compute ratio vs world size
  9. Dion: rank + residual norm vs training step
  10. Dion2: alpha + sparsity vs training step
  11. (stub for AdaDion)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Color palette for optimizers
COLORS = {
    "adamw": "#1f77b4",
    "muon": "#ff7f0e",
    "dion": "#2ca02c",
    "dion2": "#d62728",
    "adadion": "#9467bd",
}

LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
    "dion": "Dion",
    "dion2": "Dion2",
    "adadion": "AdaDion",
}


def _setup_style():
    """Apply consistent plot styling."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })


def _place_legend(ax, loc="best"):
    """Place legend outside the grid area if there's risk of overlap."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(handles, labels, loc=loc, frameon=True, fancybox=False)


def _finalize_ax(ax):
    """Apply common axis formatting: grid behind data, clean spines."""
    ax.set_axisbelow(True)  # Grid lines behind data
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ======================================================================
# Core training plots
# ======================================================================

def plot_val_loss_vs_tokens(
    runs: dict[str, dict],
) -> plt.Figure:
    """
    Plot 1: Validation loss vs tokens consumed.

    Args:
        runs: Dict of {optimizer_name: {"tokens": [...], "val_loss": [...]}}
    """
    _setup_style()
    fig, ax = plt.subplots()

    for opt_name, data in runs.items():
        color = COLORS.get(opt_name, "gray")
        label = LABELS.get(opt_name, opt_name)
        ax.plot(data["tokens"], data["val_loss"], color=color, label=label,
                linewidth=2, zorder=3)

    ax.set_xlabel("Tokens")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss vs Tokens")
    _finalize_ax(ax)
    _place_legend(ax, loc="upper right")
    fig.tight_layout()
    return fig


def plot_val_loss_vs_wallclock(
    runs: dict[str, dict],
) -> plt.Figure:
    """
    Plot 2: Validation loss vs wall-clock time.

    Args:
        runs: Dict of {optimizer_name: {"time_s": [...], "val_loss": [...]}}
    """
    _setup_style()
    fig, ax = plt.subplots()

    for opt_name, data in runs.items():
        color = COLORS.get(opt_name, "gray")
        label = LABELS.get(opt_name, opt_name)
        time_h = [t / 3600 for t in data["time_s"]]
        ax.plot(time_h, data["val_loss"], color=color, label=label,
                linewidth=2, zorder=3)

    ax.set_xlabel("Wall-clock Time (hours)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss vs Wall-clock Time")
    _finalize_ax(ax)
    _place_legend(ax, loc="upper right")
    fig.tight_layout()
    return fig


def plot_tokens_per_sec(
    runs: dict[str, dict],
) -> plt.Figure:
    """
    Plot 3: Tokens/sec vs training step.

    Args:
        runs: Dict of {optimizer_name: {"step": [...], "tokens_per_sec": [...]}}
    """
    _setup_style()
    fig, ax = plt.subplots()

    for opt_name, data in runs.items():
        color = COLORS.get(opt_name, "gray")
        label = LABELS.get(opt_name, opt_name)
        ax.plot(data["step"], data["tokens_per_sec"], color=color, label=label,
                linewidth=1.5, alpha=0.8, zorder=3)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput Stability")
    _finalize_ax(ax)
    _place_legend(ax, loc="lower right")
    fig.tight_layout()
    return fig


def plot_scaling_tokens_per_sec(
    runs: dict[str, dict],
) -> plt.Figure:
    """
    Plot 4: Tokens/sec vs number of GPUs (scaling plot).

    Args:
        runs: Dict of {optimizer_name: {"num_gpus": [...], "tokens_per_sec": [...]}}
    """
    _setup_style()
    fig, ax = plt.subplots()

    for opt_name, data in runs.items():
        color = COLORS.get(opt_name, "gray")
        label = LABELS.get(opt_name, opt_name)
        ax.plot(data["num_gpus"], data["tokens_per_sec"], "o-",
                color=color, label=label, linewidth=2, markersize=8, zorder=3)

    # Ideal linear scaling reference
    if runs:
        first_run = next(iter(runs.values()))
        if len(first_run["num_gpus"]) > 0:
            base_gpus = first_run["num_gpus"][0]
            base_tps = first_run["tokens_per_sec"][0]
            ideal_gpus = first_run["num_gpus"]
            ideal_tps = [base_tps * (g / base_gpus) for g in ideal_gpus]
            ax.plot(ideal_gpus, ideal_tps, "--", color="gray",
                    label="Ideal linear", linewidth=1, alpha=0.5, zorder=2)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Scaling Efficiency")
    _finalize_ax(ax)
    _place_legend(ax, loc="upper left")
    fig.tight_layout()
    return fig


# ======================================================================
# Systems / breakdown plots
# ======================================================================

def plot_step_time_breakdown(
    runs: dict[str, dict],
) -> plt.Figure:
    """
    Plot 5: Step time breakdown (stacked bar: forward, backward, optimizer, NCCL).

    Args:
        runs: Dict of {optimizer_name: {"fwd_ms": float, "bwd_ms": float,
                                         "opt_ms": float, "nccl_ms": float}}
    """
    _setup_style()
    fig, ax = plt.subplots()

    opt_names = list(runs.keys())
    x = np.arange(len(opt_names))
    width = 0.6

    components = ["fwd_ms", "bwd_ms", "opt_ms", "nccl_ms"]
    comp_labels = ["Forward", "Backward", "Optimizer", "NCCL"]
    comp_colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2"]

    bottom = np.zeros(len(opt_names))
    for comp, comp_label, comp_color in zip(components, comp_labels, comp_colors):
        values = [runs[opt_name].get(comp, 0) for opt_name in opt_names]
        ax.bar(x, values, width, bottom=bottom, label=comp_label,
               color=comp_color, zorder=3, edgecolor="white", linewidth=0.5)
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in opt_names])
    ax.set_ylabel("Time (ms)")
    ax.set_title("Step Time Breakdown")
    _finalize_ax(ax)
    # Legend outside plot for bar charts to avoid covering bars
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    return fig


def plot_communicated_bytes(
    runs: dict[str, dict],
) -> plt.Figure:
    """
    Plot 6: Total communicated bytes per step by collective type.

    Args:
        runs: Dict of {optimizer_name: {"allreduce_bytes": int,
                                         "allgather_bytes": int,
                                         "reducescatter_bytes": int}}
    """
    _setup_style()
    fig, ax = plt.subplots()

    opt_names = list(runs.keys())
    x = np.arange(len(opt_names))
    width = 0.6

    collectives = ["allreduce_bytes", "allgather_bytes", "reducescatter_bytes"]
    coll_labels = ["AllReduce", "AllGather", "ReduceScatter"]
    coll_colors = ["#e45756", "#4c78a8", "#f58518"]

    bottom = np.zeros(len(opt_names))
    for coll, coll_label, coll_color in zip(collectives, coll_labels, coll_colors):
        values = [runs[opt_name].get(coll, 0) / 1e6 for opt_name in opt_names]
        ax.bar(x, values, width, bottom=bottom, label=coll_label,
               color=coll_color, zorder=3, edgecolor="white", linewidth=0.5)
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in opt_names])
    ax.set_ylabel("Communicated (MB/step)")
    ax.set_title("Communication Volume per Step")
    _finalize_ax(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    return fig


def plot_collective_time(
    runs: dict[str, dict],
) -> plt.Figure:
    """
    Plot 7: Collective time per step by type.

    Args:
        runs: Dict of {optimizer_name: {"allreduce_ms": float, ...}}
    """
    _setup_style()
    fig, ax = plt.subplots()

    opt_names = list(runs.keys())
    x = np.arange(len(opt_names))
    width = 0.6

    collectives = ["allreduce_ms", "allgather_ms", "reducescatter_ms"]
    coll_labels = ["AllReduce", "AllGather", "ReduceScatter"]
    coll_colors = ["#e45756", "#4c78a8", "#f58518"]

    bottom = np.zeros(len(opt_names))
    for coll, coll_label, coll_color in zip(collectives, coll_labels, coll_colors):
        values = [runs[opt_name].get(coll, 0) for opt_name in opt_names]
        ax.bar(x, values, width, bottom=bottom, label=coll_label,
               color=coll_color, zorder=3, edgecolor="white", linewidth=0.5)
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in opt_names])
    ax.set_ylabel("Time (ms/step)")
    ax.set_title("Collective Time per Step")
    _finalize_ax(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    return fig


def plot_comm_compute_ratio(
    runs: dict[str, dict],
) -> plt.Figure:
    """
    Plot 8: Communication/compute ratio vs world size.

    Args:
        runs: Dict of {optimizer_name: {"num_gpus": [...], "comm_compute_ratio": [...]}}
    """
    _setup_style()
    fig, ax = plt.subplots()

    for opt_name, data in runs.items():
        color = COLORS.get(opt_name, "gray")
        label = LABELS.get(opt_name, opt_name)
        ax.plot(data["num_gpus"], data["comm_compute_ratio"], "o-",
                color=color, label=label, linewidth=2, markersize=8, zorder=3)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Comm / Compute Ratio")
    ax.set_title("Communication Overhead vs Scale")
    _finalize_ax(ax)
    _place_legend(ax, loc="upper left")
    fig.tight_layout()
    return fig


# ======================================================================
# Optimizer-specific diagnostic plots
# ======================================================================

def plot_dion_rank_residual(
    data: dict,
) -> plt.Figure:
    """
    Plot 9: Dion rank + residual norm vs training step.

    Args:
        data: {"step": [...], "rank_mean": [...], "residual_norm_mean": [...]}
    """
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = data["step"]

    ax1.plot(steps, data["rank_mean"], color=COLORS["dion"], linewidth=1.5, zorder=3)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Rank (mean across layers)")
    ax1.set_title("Dion: Effective Rank")
    _finalize_ax(ax1)

    ax2.plot(steps, data["residual_norm_mean"], color=COLORS["dion"], linewidth=1.5, zorder=3)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("||M||_F (mean)")
    ax2.set_title("Dion: Residual Norm (after error feedback)")
    _finalize_ax(ax2)

    fig.tight_layout()
    return fig


def plot_dion2_alpha_sparsity(
    data: dict,
) -> plt.Figure:
    """
    Plot 10: Dion2 alpha/sparsity vs training step.

    Args:
        data: {"step": [...], "alpha": [...], "sparsity": [...], "selected_frac_mean": [...]}
    """
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = data["step"]

    ax1.plot(steps, data["selected_frac_mean"], color=COLORS["dion2"],
             linewidth=1.5, zorder=3, label="Actual")
    target_alpha = data["alpha"][0] if data["alpha"] else 0.25
    ax1.axhline(y=target_alpha, color="gray", linestyle="--", alpha=0.5,
                label=f"Target ({target_alpha})", zorder=2)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Selected Fraction")
    ax1.set_title("Dion2: Selected Row/Col Fraction")
    _finalize_ax(ax1)
    _place_legend(ax1, loc="upper right")

    ax2.plot(steps, data["sparsity"], color=COLORS["dion2"], linewidth=1.5, zorder=3)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Sparsity (1 - fraction updated)")
    ax2.set_title("Dion2: Update Sparsity")
    _finalize_ax(ax2)

    fig.tight_layout()
    return fig


def plot_adadion_stub() -> plt.Figure:
    """Plot 11: Stub for future AdaDion effective-rank estimate plot."""
    _setup_style()
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "AdaDion plot\n(not yet implemented)",
            ha="center", va="center", fontsize=16, color="gray",
            transform=ax.transAxes)
    ax.set_title("AdaDion: Effective Rank Estimate vs Chosen r/alpha")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ======================================================================
# Utility: load data from JSONL logs
# ======================================================================

def load_metrics_jsonl(path: str) -> list[dict]:
    """Load metrics from a JSON lines file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_all_plots(
    output_dir: str,
    runs: Optional[dict] = None,
    dion_data: Optional[dict] = None,
    dion2_data: Optional[dict] = None,
):
    """Save all available plots to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if runs:
        # Core plots
        if all("tokens" in r and "val_loss" in r for r in runs.values()):
            fig = plot_val_loss_vs_tokens(runs)
            fig.savefig(output_path / "01_val_loss_vs_tokens.png", bbox_inches="tight")
            plt.close(fig)

        if all("time_s" in r and "val_loss" in r for r in runs.values()):
            fig = plot_val_loss_vs_wallclock(runs)
            fig.savefig(output_path / "02_val_loss_vs_wallclock.png", bbox_inches="tight")
            plt.close(fig)

        if all("step" in r and "tokens_per_sec" in r for r in runs.values()):
            fig = plot_tokens_per_sec(runs)
            fig.savefig(output_path / "03_tokens_per_sec.png", bbox_inches="tight")
            plt.close(fig)

        if all("num_gpus" in r and "tokens_per_sec" in r for r in runs.values()):
            fig = plot_scaling_tokens_per_sec(runs)
            fig.savefig(output_path / "04_scaling.png", bbox_inches="tight")
            plt.close(fig)

        if all("fwd_ms" in r for r in runs.values()):
            fig = plot_step_time_breakdown(runs)
            fig.savefig(output_path / "05_step_time_breakdown.png", bbox_inches="tight")
            plt.close(fig)

        if all("allreduce_bytes" in r for r in runs.values()):
            fig = plot_communicated_bytes(runs)
            fig.savefig(output_path / "06_communicated_bytes.png", bbox_inches="tight")
            plt.close(fig)

        if all("allreduce_ms" in r for r in runs.values()):
            fig = plot_collective_time(runs)
            fig.savefig(output_path / "07_collective_time.png", bbox_inches="tight")
            plt.close(fig)

        if all("num_gpus" in r and "comm_compute_ratio" in r for r in runs.values()):
            fig = plot_comm_compute_ratio(runs)
            fig.savefig(output_path / "08_comm_compute_ratio.png", bbox_inches="tight")
            plt.close(fig)

    # Optimizer-specific plots
    if dion_data:
        fig = plot_dion_rank_residual(dion_data)
        fig.savefig(output_path / "09_dion_rank_residual.png", bbox_inches="tight")
        plt.close(fig)

    if dion2_data:
        fig = plot_dion2_alpha_sparsity(dion2_data)
        fig.savefig(output_path / "10_dion2_alpha_sparsity.png", bbox_inches="tight")
        plt.close(fig)

    # Stub
    fig = plot_adadion_stub()
    fig.savefig(output_path / "11_adadion_stub.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {output_path}")
