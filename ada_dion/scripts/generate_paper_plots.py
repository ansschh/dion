#!/usr/bin/env python3
"""
Generate publication-quality benchmark plots from SLURM logs.

Produces notebook-style 2x3 grid plots with EMA smoothing, log scales,
and the same visual conventions as the reference IFT6166 notebook.

Usage:
    python generate_paper_plots.py --logs-dir logs --output-dir paper/figures \
        --baseline-ids ID1 ID2 ID3 ID4 \
        --sweep-id SWEEP_ARRAY_ID \
        --multinode-ids ID1 ID2 ID3 ID4 \
        [--optimal-ids ID1 ID2 ID3 ID4] \
        [--optimal-multi-ids ID1 ID2 ID3 ID4]
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
import matplotlib.ticker as ticker
import numpy as np

# ── Styling ─────────────────────────────────────────────────────────

COLORS = {
    "adamw": "#1f77b4",
    "muon": "#ff7f0e",
    "dion": "#2ca02c",
    "dion2": "#d62728",
}
LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
    "dion": "Dion (low-rank)",
    "dion2": "Dion2 (α-fraction)",
}
LINESTYLES = {
    "adamw": "--",
    "muon": "-",
    "dion": "-",
    "dion2": "-",
}
LINEWIDTHS = {
    "adamw": 2.0,
    "muon": 2.0,
    "dion": 2.0,
    "dion2": 2.0,
}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def smooth(values, weight=0.95):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


# ── Log parsing ─────────────────────────────────────────────────────

OPTIMIZER_RE = re.compile(r"Optimizer:\s*(\w+)")
SWEEP_RE = re.compile(r"Sweep\s+(\d+)/\d+:\s+(\w+)\s+\|(.+)")

FIELD_RES = {
    "step": re.compile(r"step:\s*(\d+)"),
    "loss": re.compile(r"loss:\s*([\d.]+)"),
    "grad_norm": re.compile(r"grad_norm:\s*([\d.]+)"),
    "memory_gib": re.compile(r"memory:\s*([\d.]+)GiB"),
    "tps": re.compile(r"tps:\s*([\d,]+)"),
    "tflops": re.compile(r"tflops:\s*([\d.]+)"),
    "mfu": re.compile(r"mfu:\s*([\d.]+)%"),
}
TIMESTAMP_RE = re.compile(r"\[titan\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")

    m = OPTIMIZER_RE.search(text)
    opt_name = m.group(1) if m else None

    m = SWEEP_RE.search(text)
    sweep_info = None
    if m:
        sweep_info = {
            "task_id": int(m.group(1)),
            "optimizer": m.group(2),
            "extra_args": m.group(3).strip(),
        }
        opt_name = opt_name or m.group(2)

    seen_steps = {}
    for line in text.splitlines():
        if "step:" not in line or "loss:" not in line:
            continue
        if "[titan]" not in line and "INFO" not in line:
            continue

        step_m = FIELD_RES["step"].search(line)
        if not step_m:
            continue
        step = int(step_m.group(1))
        if step in seen_steps:
            continue

        ts_m = TIMESTAMP_RE.search(line)
        timestamp = None
        if ts_m:
            timestamp = datetime.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S")

        record = {"step": step, "timestamp": timestamp}
        for field, regex in FIELD_RES.items():
            if field == "step":
                continue
            fm = regex.search(line)
            if fm:
                val = fm.group(1)
                if field == "tps":
                    record[field] = int(val.replace(",", ""))
                else:
                    record[field] = float(val)

        seen_steps[step] = record

    records = [seen_steps[s] for s in sorted(seen_steps)]

    if records and records[0].get("timestamp"):
        t0 = records[0]["timestamp"]
        for r in records:
            r["time_s"] = (r["timestamp"] - t0).total_seconds() if r.get("timestamp") else 0.0

    return {"optimizer": opt_name, "sweep_info": sweep_info, "records": records}


def parse_phase(logs_dir, job_ids, opt_order, suffix="_adadion.out"):
    result = {}
    for job_id, opt in zip(job_ids, opt_order):
        path = logs_dir / f"{job_id}{suffix}"
        if not path.exists():
            # Try alternate suffixes
            for alt in ["_adadion.out", "_adadion-multi.out",
                        "_adadion-optimal.out", "_adadion-opt-multi.out"]:
                alt_path = logs_dir / f"{job_id}{alt}"
                if alt_path.exists():
                    path = alt_path
                    break
        if not path.exists():
            print(f"  WARNING: no log for {opt} (job {job_id})")
            continue
        data = parse_log(path)
        result[opt] = data["records"]
        n = len(data["records"])
        final = data["records"][-1].get("loss", "?") if n > 0 else "N/A"
        print(f"  {opt}: {n} records, final loss={final}")
    return result


# ── Plot functions ──────────────────────────────────────────────────

def plot_main_grid(baselines, title_suffix="", tokens_per_step=65536):
    """
    Notebook-style 2x3 grid:
      (0,0) Training Loss (log)     (0,1) Training Loss (linear, zoomed)  (0,2) Grad Norm
      (1,0) Tokens/sec              (1,1) MFU (%)                         (1,2) Memory (GiB)
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    for opt, records in baselines.items():
        if not records:
            continue
        steps = [r["step"] for r in records]
        losses = [r.get("loss", 0) for r in records]
        grad_norms = [r.get("grad_norm", 0) for r in records]
        tps_vals = [r.get("tps", 0) for r in records]
        mfu_vals = [r.get("mfu", 0) for r in records]
        mem_vals = [r.get("memory_gib", 0) for r in records]

        color = COLORS.get(opt, "gray")
        label = LABELS.get(opt, opt)
        ls = LINESTYLES.get(opt, "-")
        lw = LINEWIDTHS.get(opt, 1.5)

        # (0,0) Training Loss — log scale, smoothed
        sm_loss = smooth(losses, 0.95)
        axes[0, 0].plot(steps, np.maximum(sm_loss, 1e-6), color=color,
                        label=label, ls=ls, lw=lw)

        # (0,1) Training Loss — linear, last 80% zoomed
        axes[0, 1].plot(steps, smooth(losses, 0.98), color=color,
                        label=label, ls=ls, lw=lw)

        # (0,2) Grad Norm — log scale
        sm_gn = smooth(grad_norms, 0.9)
        axes[0, 2].plot(steps, np.maximum(sm_gn, 1e-8), color=color,
                        label=label, ls=ls, lw=lw)

        # (1,0) Tokens/sec
        sm_tps = smooth(tps_vals, 0.9)
        axes[1, 0].plot(steps, sm_tps, color=color, label=label, ls=ls, lw=lw)

        # (1,1) MFU
        sm_mfu = smooth(mfu_vals, 0.9)
        axes[1, 1].plot(steps, sm_mfu, color=color, label=label, ls=ls, lw=lw)

        # (1,2) Memory
        axes[1, 2].plot(steps, mem_vals, color=color, label=label, ls=ls, lw=lw)

    # Formatting
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Training Loss (log)")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].set_title("Training Loss (zoomed)")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Loss")
    if baselines:
        # Zoom to last 80%
        any_records = next(iter(baselines.values()))
        if any_records:
            max_step = any_records[-1]["step"]
            axes[0, 1].set_xlim(max_step * 0.2, max_step)
            # Auto-scale y to visible range
            all_final = [r[-1].get("loss", 1) for r in baselines.values() if r]
            if all_final:
                y_max = max(all_final) * 3
                axes[0, 1].set_ylim(0, min(y_max, 2.0))

    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Gradient Norm (log)")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("‖∇L‖")

    axes[1, 0].set_title("Throughput")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Tokens/sec")
    axes[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    axes[1, 1].set_title("Model FLOPS Utilization")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("MFU (%)")

    axes[1, 2].set_title("GPU Memory")
    axes[1, 2].set_xlabel("Step")
    axes[1, 2].set_ylabel("Memory (GiB)")

    for ax in axes.flat:
        ax.set_axisbelow(True)

    # Shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=13,
               frameon=True, fancybox=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Training Metrics{title_suffix}", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def plot_loss_vs_tokens(baselines, tokens_per_step=65536, title=""):
    """Loss vs tokens consumed, log-scale y."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for opt, records in baselines.items():
        if not records:
            continue
        tokens = [r["step"] * tokens_per_step for r in records]
        losses = smooth([r.get("loss", 0) for r in records], 0.95)
        ax.plot(tokens, np.maximum(losses, 1e-6), color=COLORS.get(opt, "gray"),
                label=LABELS.get(opt, opt), ls=LINESTYLES.get(opt, "-"),
                lw=LINEWIDTHS.get(opt, 2))

    ax.set_yscale("log")
    ax.set_xlabel("Tokens Consumed", fontsize=13)
    ax.set_ylabel("Training Loss", fontsize=13)
    ax.set_title(title or "Training Loss vs Tokens", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_loss_vs_wallclock(baselines, title=""):
    """Loss vs wall-clock time in minutes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for opt, records in baselines.items():
        if not records:
            continue
        time_min = [r.get("time_s", 0) / 60 for r in records]
        losses = smooth([r.get("loss", 0) for r in records], 0.95)
        ax.plot(time_min, np.maximum(losses, 1e-6), color=COLORS.get(opt, "gray"),
                label=LABELS.get(opt, opt), ls=LINESTYLES.get(opt, "-"),
                lw=LINEWIDTHS.get(opt, 2))

    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock Time (minutes)", fontsize=13)
    ax.set_ylabel("Training Loss", fontsize=13)
    ax.set_title(title or "Training Loss vs Wall-clock Time", fontsize=14)
    ax.legend(fontsize=12)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_scaling_comparison(baselines_4gpu, baselines_8gpu):
    """Side-by-side: final loss + throughput for 4 vs 8 GPUs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    opts = ["adamw", "muon", "dion", "dion2"]
    opts_present = [o for o in opts if o in baselines_4gpu and o in baselines_8gpu
                    and baselines_4gpu[o] and baselines_8gpu[o]]

    x = np.arange(len(opts_present))
    w = 0.35

    # Final loss
    loss_4 = [baselines_4gpu[o][-1].get("loss", 0) for o in opts_present]
    loss_8 = [baselines_8gpu[o][-1].get("loss", 0) for o in opts_present]
    b1 = ax1.bar(x - w/2, loss_4, w, label="4 GPU (FSDP)", color="#4c78a8", zorder=3)
    b2 = ax1.bar(x + w/2, loss_8, w, label="8 GPU (HSDP)", color="#f58518", zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS.get(o, o) for o in opts_present], fontsize=10)
    ax1.set_ylabel("Final Loss @ 10k steps")
    ax1.set_title("Final Loss: FSDP vs HSDP")
    ax1.legend()
    ax1.set_axisbelow(True)

    # Throughput
    def med_tps(records):
        if not records:
            return 0
        n = max(1, len(records) // 5)
        return float(np.median([r.get("tps", 0) for r in records[-n:]]))

    tps_4 = [med_tps(baselines_4gpu[o]) for o in opts_present]
    tps_8 = [med_tps(baselines_8gpu[o]) for o in opts_present]
    b1 = ax2.bar(x - w/2, tps_4, w, label="4 GPU (FSDP)", color="#4c78a8", zorder=3)
    b2 = ax2.bar(x + w/2, tps_8, w, label="8 GPU (HSDP)", color="#f58518", zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 500,
                     f"{h:,.0f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS.get(o, o) for o in opts_present], fontsize=10)
    ax2.set_ylabel("Tokens/sec (median)")
    ax2.set_title("Throughput: FSDP vs HSDP")
    ax2.legend()
    ax2.set_axisbelow(True)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    return fig


def plot_sweep_heatmaps(sweep_data):
    """Heatmaps for Dion and Dion2 sweep results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Extract Dion sweep
    lrs_dion = [0.005, 0.02, 0.05]
    rfs = [0.1, 0.25, 0.5]
    dion_grid = np.full((3, 3), np.nan)
    for d in sweep_data:
        info = d["sweep_info"]
        if info["optimizer"] != "dion" or not d["records"]:
            continue
        args = info["extra_args"]
        lr_m = re.search(r"lr\s+([\d.]+)", args)
        rf_m = re.search(r"rank_frac\s+([\d.]+)", args)
        if lr_m and rf_m:
            lr, rf = float(lr_m.group(1)), float(rf_m.group(1))
            if lr in lrs_dion and rf in rfs:
                i, j = lrs_dion.index(lr), rfs.index(rf)
                dion_grid[i, j] = d["records"][-1]["loss"]

    im1 = ax1.imshow(dion_grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=3)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([str(r) for r in rfs])
    ax1.set_yticks(range(3))
    ax1.set_yticklabels([str(lr) for lr in lrs_dion])
    ax1.set_xlabel("rank_frac")
    ax1.set_ylabel("Learning Rate")
    ax1.set_title("Dion: Final Loss @ 5k steps")
    for i in range(3):
        for j in range(3):
            v = dion_grid[i, j]
            if not np.isnan(v):
                ax1.text(j, i, f"{v:.3f}", ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color="white" if v > 1.5 else "black")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Extract Dion2 sweep
    alphas = [0.1, 0.25, 0.5]
    dion2_grid = np.full((3, 3), np.nan)
    for d in sweep_data:
        info = d["sweep_info"]
        if info["optimizer"] != "dion2" or not d["records"]:
            continue
        args = info["extra_args"]
        lr_m = re.search(r"lr\s+([\d.]+)", args)
        a_m = re.search(r"alpha\s+([\d.]+)", args)
        if lr_m and a_m:
            lr, alpha = float(lr_m.group(1)), float(a_m.group(1))
            if lr in lrs_dion and alpha in alphas:
                i, j = lrs_dion.index(lr), alphas.index(alpha)
                dion2_grid[i, j] = d["records"][-1]["loss"]

    im2 = ax2.imshow(dion2_grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=3)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels([str(a) for a in alphas])
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([str(lr) for lr in lrs_dion])
    ax2.set_xlabel("alpha")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Dion2: Final Loss @ 5k steps")
    for i in range(3):
        for j in range(3):
            v = dion2_grid[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:.3f}", ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color="white" if v > 1.5 else "black")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Hyperparameter Sweep: Final Loss", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_summary_table(baselines, title=""):
    """Summary table as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.axis("off")

    opts = ["adamw", "muon", "dion", "dion2"]
    opts_present = [o for o in opts if o in baselines and baselines[o]]

    headers = ["Optimizer", "Final Loss", "Avg TPS", "Peak Mem (GiB)", "MFU (%)", "Total Time"]
    rows = []
    for opt in opts_present:
        records = baselines[opt]
        n = max(1, len(records) // 10)
        final_loss = f"{records[-1].get('loss', 0):.5f}"
        avg_tps = f"{np.median([r.get('tps', 0) for r in records[-n:]]):,.0f}"
        peak_mem = f"{max(r.get('memory_gib', 0) for r in records):.1f}"
        avg_mfu = f"{np.median([r.get('mfu', 0) for r in records[-n:]]):.1f}"
        total_time_s = records[-1].get("time_s", 0) if records[-1].get("time_s") else 0
        total_time = f"{total_time_s/60:.1f} min"
        rows.append([LABELS.get(opt, opt), final_loss, avg_tps, peak_mem, avg_mfu, total_time])

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", colColours=["#e6e6e6"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Color the optimizer column
    for i, opt in enumerate(opts_present):
        table[i+1, 0].set_facecolor(COLORS.get(opt, "gray") + "30")

    ax.set_title(title or "Summary of Results", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--baseline-ids", nargs=4, required=True,
                        help="adamw muon dion dion2 baseline job IDs")
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--multinode-ids", nargs=4, required=True,
                        help="adamw muon dion dion2 multi-node job IDs")
    parser.add_argument("--optimal-ids", nargs=4, default=None,
                        help="adamw muon dion dion2 optimal baseline job IDs")
    parser.add_argument("--optimal-multi-ids", nargs=4, default=None,
                        help="adamw muon dion dion2 optimal multi-node job IDs")
    args = parser.parse_args()

    logs = Path(args.logs_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    setup_style()
    opt_order = ["adamw", "muon", "dion", "dion2"]
    tps_4gpu = 8 * 2048 * 4  # local_batch * seq_len * num_gpus

    # ── Parse all data ──
    print("=== Parsing Phase 1: Default HP Baselines ===")
    baselines = parse_phase(logs, args.baseline_ids, opt_order)

    print("\n=== Parsing Phase 2: HP Sweep ===")
    sweep_data = []
    for task_id in range(24):
        path = logs / f"sweep_{args.sweep_id}_{task_id}.out"
        if not path.exists():
            continue
        data = parse_log(path)
        if data["sweep_info"]:
            sweep_data.append(data)
    print(f"  {len(sweep_data)} sweep configs parsed")

    print("\n=== Parsing Phase 3: Multi-node ===")
    multi_node = parse_phase(logs, args.multinode_ids, opt_order, "_adadion-multi.out")

    optimal = None
    if args.optimal_ids:
        print("\n=== Parsing Optimal Baselines ===")
        optimal = parse_phase(logs, args.optimal_ids, opt_order)

    optimal_multi = None
    if args.optimal_multi_ids:
        print("\n=== Parsing Optimal Multi-node ===")
        optimal_multi = parse_phase(logs, args.optimal_multi_ids, opt_order)

    # ── Generate plots ──
    print(f"\n=== Generating plots to {output} ===")
    count = 0

    # Fig 1: Main 2x3 grid — default baselines
    if baselines:
        fig = plot_main_grid(baselines, " — Default HPs (4× H200, 10k steps)")
        fig.savefig(output / "fig1_main_grid_default.pdf", bbox_inches="tight")
        fig.savefig(output / "fig1_main_grid_default.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        count += 1
        print(f"  fig1_main_grid_default.pdf")

    # Fig 2: Main 2x3 grid — optimal baselines (if available)
    if optimal:
        fig = plot_main_grid(optimal, " — Optimal HPs (4× H200, 10k steps)")
        fig.savefig(output / "fig2_main_grid_optimal.pdf", bbox_inches="tight")
        fig.savefig(output / "fig2_main_grid_optimal.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        count += 1
        print(f"  fig2_main_grid_optimal.pdf")

    # Fig 3: Loss vs tokens
    data_for_loss = optimal if optimal else baselines
    if data_for_loss:
        fig = plot_loss_vs_tokens(data_for_loss, tps_4gpu,
                                  "Training Loss vs Tokens (4× H200)")
        fig.savefig(output / "fig3_loss_vs_tokens.pdf", bbox_inches="tight")
        fig.savefig(output / "fig3_loss_vs_tokens.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        count += 1
        print(f"  fig3_loss_vs_tokens.pdf")

    # Fig 4: Loss vs wall-clock
    if data_for_loss:
        fig = plot_loss_vs_wallclock(data_for_loss,
                                     "Training Loss vs Wall-clock (4× H200)")
        fig.savefig(output / "fig4_loss_vs_wallclock.pdf", bbox_inches="tight")
        fig.savefig(output / "fig4_loss_vs_wallclock.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        count += 1
        print(f"  fig4_loss_vs_wallclock.pdf")

    # Fig 5: Sweep heatmaps
    if sweep_data:
        fig = plot_sweep_heatmaps(sweep_data)
        fig.savefig(output / "fig5_sweep_heatmaps.pdf", bbox_inches="tight")
        fig.savefig(output / "fig5_sweep_heatmaps.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        count += 1
        print(f"  fig5_sweep_heatmaps.pdf")

    # Fig 6: Scaling comparison
    bl_for_scaling = optimal if optimal else baselines
    mn_for_scaling = optimal_multi if optimal_multi else multi_node
    if bl_for_scaling and mn_for_scaling:
        fig = plot_scaling_comparison(bl_for_scaling, mn_for_scaling)
        fig.savefig(output / "fig6_scaling_comparison.pdf", bbox_inches="tight")
        fig.savefig(output / "fig6_scaling_comparison.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        count += 1
        print(f"  fig6_scaling_comparison.pdf")

    # Fig 7: Multi-node 2x3 grid
    mn_data = optimal_multi if optimal_multi else multi_node
    if mn_data:
        fig = plot_main_grid(mn_data, " — HSDP (2×4 H200, 10k steps)")
        fig.savefig(output / "fig7_main_grid_multinode.pdf", bbox_inches="tight")
        fig.savefig(output / "fig7_main_grid_multinode.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        count += 1
        print(f"  fig7_main_grid_multinode.pdf")

    # Fig 8: Summary table
    tbl_data = optimal if optimal else baselines
    if tbl_data:
        fig = plot_summary_table(tbl_data, "Single-Node Results (4× H200, 10k steps)")
        fig.savefig(output / "fig8_summary_table.pdf", bbox_inches="tight")
        fig.savefig(output / "fig8_summary_table.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        count += 1
        print(f"  fig8_summary_table.pdf")

    if mn_data:
        fig = plot_summary_table(mn_data, "Multi-Node Results (8 GPU HSDP, 10k steps)")
        fig.savefig(output / "fig9_summary_table_multi.pdf", bbox_inches="tight")
        fig.savefig(output / "fig9_summary_table_multi.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        count += 1
        print(f"  fig9_summary_table_multi.pdf")

    print(f"\nDone! {count} figures saved to {output}/")


if __name__ == "__main__":
    main()
