"""
CIFAR-10 Benchmark for AdaDion V2.

Compares: AdamW, Dion-equivalent, AdaDionV2 (adaptive rank)
Model: ResNet-18 (CIFAR-10 variant)

Logs everything to W&B + generates local plots with Poppins font.

Usage:
  python benchmarks/cifar10_benchmark.py --epochs 50
  torchrun --nproc_per_node=N benchmarks/cifar10_benchmark.py --epochs 50
"""
import argparse, json, math, os, sys, time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Plotting setup — Poppins, minimal, no bold
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Try to use Poppins, fall back to sans-serif
_poppins = None
for f in fm.findSystemFonts():
    if "poppins" in f.lower() or "Poppins" in f:
        _poppins = f; break

plt.rcParams.update({
    "font.family": "Poppins" if _poppins else "sans-serif",
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLORS = {
    "adamw": "#4C72B0",
    "muon": "#C44E52",
    "dion": "#DD8452",
    "dion2": "#8172B3",
    "dion_equiv": "#937860",
    "adadion_v2": "#55A868",
}

# ---------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------
def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_dataloaders(batch_size, world_size, rank):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10("./data", train=False, transform=test_tf)
    sampler = DistributedSampler(train_ds, world_size, rank) if world_size > 1 else None
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None),
                          sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=512, num_workers=4, pin_memory=True)
    return train_dl, test_dl, sampler

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(device):
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def _split_params(model):
    """Split into matrix (2D) and scalar (1D/bias) params."""
    mat, scalar = [], []
    for p in model.parameters():
        (mat if p.ndim >= 2 and min(p.shape) >= 2 else scalar).append(p)
    return mat, scalar

def create_optimizer(name, model, lr, wd, rank_fraction=0.5, adaptive=False):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    mat, scalar = _split_params(model)

    if name in ("muon", "dion", "dion2"):
        from dion import Muon, Dion, Dion2
        scalar_group = {
            "params": scalar, "algorithm": "adamw",
            "lr": lr * 0.1, "weight_decay": wd,
            "betas": (0.9, 0.95), "eps": 1e-8,
        }
        groups = [{"params": mat}, scalar_group] if scalar else [{"params": mat}]

        if name == "muon":
            return Muon(groups, lr=lr, mu=0.95, weight_decay=wd)
        elif name == "dion":
            return Dion(groups, lr=lr, rank_fraction=rank_fraction, weight_decay=wd)
        elif name == "dion2":
            return Dion2(groups, lr=lr, fraction=rank_fraction, ef_decay=0.95, weight_decay=wd)

    # adadion_v2 or dion_equiv
    from ada_dion.optim.adadion_v2 import AdaDionV2
    groups = [{"params": mat}]
    if scalar:
        groups.append({"params": scalar, "algorithm": "adamw",
                        "lr": lr * 0.1, "weight_decay": wd,
                        "betas": (0.9, 0.95), "eps": 1e-8})

    return AdaDionV2(
        groups, lr=lr, mu=0.95, rank_fraction_max=0.7 if adaptive else rank_fraction,
        weight_decay=wd, adaptive_rank=adaptive,
        init_rank_fraction=rank_fraction, erank_ema_beta=0.5,
        rank_scale=1.5, rank_min=8, rank_quantize=4,
        rank_step_up=8, rank_step_down=4, use_quality_control=adaptive,
    )

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct = total = 0; loss_sum = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        lo = model(x)
        loss_sum += nn.functional.cross_entropy(lo, y, reduction="sum").item()
        correct += (lo.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return loss_sum / total, correct / total

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_one(cfg, rank, local_rank, world_size, wandb_run=None):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    train_dl, test_dl, sampler = get_dataloaders(cfg["batch_size"], world_size, rank)

    torch.manual_seed(cfg["seed"])
    model = build_model(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    opt = create_optimizer(cfg["optimizer"], model, cfg["lr"], cfg["weight_decay"],
                           cfg.get("rank_fraction", 0.5), cfg.get("adaptive_rank", False))

    total_steps = cfg["epochs"] * len(train_dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    log = defaultdict(list)
    step = 0
    t0 = time.time()

    for epoch in range(cfg["epochs"]):
        if sampler: sampler.set_epoch(epoch)
        epoch_loss = 0; epoch_count = 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            lo = model(x)
            loss = nn.functional.cross_entropy(lo, y)
            loss.backward()

            # grad norm before step
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9).item()

            # memory
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
            else:
                mem_mb = 0

            opt.step()
            opt.zero_grad()
            sched.step()
            step += 1
            epoch_loss += loss.item() * y.size(0)
            epoch_count += y.size(0)

            # per-step logging
            if step % cfg["log_freq"] == 0 and is_main:
                row = {
                    "step": step, "epoch": epoch,
                    "train_loss": loss.item(),
                    "lr": sched.get_last_lr()[0],
                    "grad_norm": gnorm,
                    "mem_mb": mem_mb,
                    "elapsed": time.time() - t0,
                    "throughput": step * cfg["batch_size"] / (time.time() - t0),
                }
                # adaptive rank diagnostics
                if hasattr(opt, "get_rank"):
                    ranks = opt.get_rank()
                    if ranks:
                        vals = list(ranks.values())
                        row["rank_mean"] = sum(vals) / len(vals)
                        row["rank_min"] = min(vals)
                        row["rank_max"] = max(vals)
                if hasattr(opt, "get_effective_rank"):
                    eranks = opt.get_effective_rank()
                    if eranks:
                        vals = list(eranks.values())
                        row["erank_mean"] = sum(vals) / len(vals)
                if hasattr(opt, "get_aerr"):
                    aerrs = opt.get_aerr()
                    if aerrs:
                        vals = list(aerrs.values())
                        row["aerr_mean"] = sum(vals) / len(vals)

                for k, v in row.items():
                    log[k].append(v)

                if wandb_run:
                    wandb_run.log({f"{cfg['name']}/{k}": v for k, v in row.items()}, step=step)

        # epoch eval
        val_loss, val_acc = evaluate(model.module if world_size > 1 else model, test_dl, device)
        train_loss_avg = epoch_loss / epoch_count

        if is_main:
            erow = {
                "epoch": epoch, "step": step,
                "train_loss_epoch": train_loss_avg,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_ppl": math.exp(min(val_loss, 20)),
                "elapsed": time.time() - t0,
            }
            for k, v in erow.items():
                log[f"eval/{k}"].append(v)

            if wandb_run:
                wandb_run.log({f"{cfg['name']}/{k}": v for k, v in erow.items()}, step=step)

            print(f"  [{cfg['name']}] ep {epoch+1}/{cfg['epochs']}"
                  f"  train={train_loss_avg:.4f}  val={val_loss:.4f}"
                  f"  acc={val_acc:.4f}  {time.time()-t0:.0f}s")

    return dict(log)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def smooth(vals, w=0.9):
    s = []; last = vals[0]
    for v in vals:
        last = w * last + (1 - w) * v
        s.append(last)
    return s

def plot_all(all_logs, outdir):
    os.makedirs(outdir, exist_ok=True)

    names = list(all_logs.keys())

    # 1. Training loss vs step
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        L = all_logs[n]
        if "step" in L and "train_loss" in L:
            ax.plot(L["step"], smooth(L["train_loss"]), color=COLORS.get(n, "gray"), label=n, linewidth=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Training Loss")
    ax.legend(frameon=False); fig.savefig(f"{outdir}/train_loss.png"); plt.close()

    # 2. Validation loss vs epoch
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        L = all_logs[n]
        if "eval/epoch" in L and "eval/val_loss" in L:
            ax.plot(L["eval/epoch"], L["eval/val_loss"], color=COLORS.get(n, "gray"),
                    label=n, linewidth=1.2, marker="o", markersize=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Validation Loss")
    ax.legend(frameon=False); fig.savefig(f"{outdir}/val_loss.png"); plt.close()

    # 3. Validation accuracy vs epoch
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        L = all_logs[n]
        if "eval/epoch" in L and "eval/val_acc" in L:
            ax.plot(L["eval/epoch"], [a * 100 for a in L["eval/val_acc"]],
                    color=COLORS.get(n, "gray"), label=n, linewidth=1.2, marker="o", markersize=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)"); ax.set_title("Validation Accuracy")
    ax.legend(frameon=False); fig.savefig(f"{outdir}/val_acc.png"); plt.close()

    # 4. Learning rate schedule
    fig, ax = plt.subplots(figsize=(7, 3))
    for n in names:
        L = all_logs[n]
        if "step" in L and "lr" in L:
            ax.plot(L["step"], L["lr"], color=COLORS.get(n, "gray"), label=n, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("lr"); ax.set_title("Learning Rate")
    ax.legend(frameon=False); fig.savefig(f"{outdir}/lr.png"); plt.close()

    # 5. Gradient norm vs step
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in names:
        L = all_logs[n]
        if "step" in L and "grad_norm" in L:
            ax.plot(L["step"], smooth(L["grad_norm"]), color=COLORS.get(n, "gray"),
                    label=n, linewidth=1, alpha=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("grad norm"); ax.set_title("Gradient Norm")
    ax.set_yscale("log"); ax.legend(frameon=False)
    fig.savefig(f"{outdir}/grad_norm.png"); plt.close()

    # 6. Memory usage
    fig, ax = plt.subplots(figsize=(7, 3))
    for n in names:
        L = all_logs[n]
        if "step" in L and "mem_mb" in L:
            ax.plot(L["step"], L["mem_mb"], color=COLORS.get(n, "gray"), label=n, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("peak memory (MB)"); ax.set_title("GPU Memory")
    ax.legend(frameon=False); fig.savefig(f"{outdir}/memory.png"); plt.close()

    # 7. Throughput
    fig, ax = plt.subplots(figsize=(7, 3))
    for n in names:
        L = all_logs[n]
        if "step" in L and "throughput" in L:
            ax.plot(L["step"], L["throughput"], color=COLORS.get(n, "gray"), label=n, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("samples/sec"); ax.set_title("Throughput")
    ax.legend(frameon=False); fig.savefig(f"{outdir}/throughput.png"); plt.close()

    # 8. Rank (adaptive only)
    has_rank = any("rank_mean" in all_logs[n] for n in names)
    if has_rank:
        fig, ax = plt.subplots(figsize=(7, 4))
        for n in names:
            L = all_logs[n]
            if "step" in L and "rank_mean" in L:
                ax.plot(L["step"], L["rank_mean"], color=COLORS.get(n, "gray"), label=f"{n} mean", linewidth=1.2)
                if "rank_min" in L:
                    ax.fill_between(L["step"], L["rank_min"], L["rank_max"],
                                    color=COLORS.get(n, "gray"), alpha=0.15)
        ax.set_xlabel("step"); ax.set_ylabel("rank"); ax.set_title("Rank")
        ax.legend(frameon=False); fig.savefig(f"{outdir}/rank.png"); plt.close()

    # 9. Effective rank
    has_erank = any("erank_mean" in all_logs[n] for n in names)
    if has_erank:
        fig, ax = plt.subplots(figsize=(7, 4))
        for n in names:
            L = all_logs[n]
            if "step" in L and "erank_mean" in L:
                ax.plot(L["step"], L["erank_mean"], color=COLORS.get(n, "gray"), label=n, linewidth=1.2)
        ax.set_xlabel("step"); ax.set_ylabel("effective rank"); ax.set_title("Effective Rank (EMA)")
        ax.legend(frameon=False); fig.savefig(f"{outdir}/erank.png"); plt.close()

    # 10. Approximation error
    has_aerr = any("aerr_mean" in all_logs[n] for n in names)
    if has_aerr:
        fig, ax = plt.subplots(figsize=(7, 4))
        for n in names:
            L = all_logs[n]
            if "step" in L and "aerr_mean" in L:
                ax.plot(L["step"], smooth(L["aerr_mean"]), color=COLORS.get(n, "gray"), label=n, linewidth=1.2)
        ax.set_xlabel("step"); ax.set_ylabel("approx error"); ax.set_title("Approximation Error")
        ax.legend(frameon=False); fig.savefig(f"{outdir}/aerr.png"); plt.close()

    # 11. Combined summary: 2x2 panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for n in names:
        L = all_logs[n]
        if "step" in L and "train_loss" in L:
            ax.plot(L["step"], smooth(L["train_loss"]), color=COLORS.get(n, "gray"), label=n, linewidth=1.2)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Training Loss")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for n in names:
        L = all_logs[n]
        if "eval/epoch" in L and "eval/val_acc" in L:
            ax.plot(L["eval/epoch"], [a * 100 for a in L["eval/val_acc"]],
                    color=COLORS.get(n, "gray"), label=n, linewidth=1.2, marker="o", markersize=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)"); ax.set_title("Validation Accuracy")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for n in names:
        L = all_logs[n]
        if "step" in L and "grad_norm" in L:
            ax.plot(L["step"], smooth(L["grad_norm"]), color=COLORS.get(n, "gray"), label=n, linewidth=1, alpha=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("grad norm"); ax.set_title("Gradient Norm")
    ax.set_yscale("log"); ax.legend(frameon=False)

    ax = axes[1, 1]
    for n in names:
        L = all_logs[n]
        if "step" in L and "throughput" in L:
            ax.plot(L["step"], L["throughput"], color=COLORS.get(n, "gray"), label=n, linewidth=1)
    ax.set_xlabel("step"); ax.set_ylabel("samples/sec"); ax.set_title("Throughput")
    ax.legend(frameon=False)

    fig.tight_layout(pad=2.0)
    fig.savefig(f"{outdir}/summary_panel.png"); plt.close()

    print(f"  Plots saved to {outdir}/")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="benchmarks/cifar10_results")
    parser.add_argument("--optimizers", type=str, default="adamw,muon,dion,dion2,dion_equiv,adadion_v2")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cifar10-adadion-v2")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)

    wandb_run = None
    if args.wandb and is_main:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    if is_main:
        os.makedirs(args.outdir, exist_ok=True)
        print("=" * 60)
        print("  CIFAR-10 Benchmark: AdaDion V2")
        print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}")
        print(f"  GPUs: {world_size}  Seed: {args.seed}")
        print("=" * 60)

    opt_names = [o.strip() for o in args.optimizers.split(",")]
    all_logs = {}

    LR_MAP = {"adamw": args.lr, "muon": 0.02, "dion": 0.02, "dion2": 0.02,
              "dion_equiv": 0.02, "adadion_v2": 0.02}

    for opt_name in opt_names:
        cfg = {
            "name": opt_name, "optimizer": opt_name,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": LR_MAP.get(opt_name, 0.02),
            "weight_decay": args.wd, "seed": args.seed,
            "log_freq": args.log_freq, "rank_fraction": 0.5,
            "adaptive_rank": (opt_name == "adadion_v2"),
        }

        if is_main:
            print(f"\n--- {opt_name} (lr={cfg['lr']}) ---")

        log = run_one(cfg, rank, local_rank, world_size, wandb_run)
        all_logs[opt_name] = log

        if is_main:
            with open(os.path.join(args.outdir, f"{opt_name}.json"), "w") as f:
                json.dump(log, f, indent=2)

    # summary + plots
    if is_main:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"  {'Optimizer':<20s} {'Val Loss':>10s} {'Val Acc':>10s} {'Time':>8s}")
        for n in opt_names:
            L = all_logs[n]
            if "eval/val_loss" in L and L["eval/val_loss"]:
                vl = L["eval/val_loss"][-1]
                va = L["eval/val_acc"][-1]
                t = L["eval/elapsed"][-1]
                print(f"  {n:<20s} {vl:10.4f} {va*100:9.2f}% {t:7.0f}s")

        plot_all(all_logs, args.outdir)

        with open(os.path.join(args.outdir, "summary.json"), "w") as f:
            json.dump({n: {k: v for k, v in L.items() if "eval" in k} for n, L in all_logs.items()}, f, indent=2)

    if wandb_run:
        # log final plots as W&B artifacts
        import wandb
        for fn in Path(args.outdir).glob("*.png"):
            wandb_run.log({fn.stem: wandb.Image(str(fn))})
        wandb_run.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
