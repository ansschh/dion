"""
CIFAR-10 Benchmark for AdaDion V2.

Compares: AdamW, Dion-equivalent (AdaDionV2 adaptive=off), AdaDionV2 (adaptive=on)
Model: ResNet-18 (good mix of 2D conv/linear layers)
Metrics: train loss, val loss, val accuracy, per-step time, rank stats

Usage:
  # Single GPU:
  python benchmarks/cifar10_benchmark.py --epochs 50

  # Multi-GPU:
  torchrun --nproc_per_node=N benchmarks/cifar10_benchmark.py --epochs 50
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_distributed():
    """Initialize distributed if launched with torchrun."""
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

    train_sampler = DistributedSampler(train_ds, world_size, rank) if world_size > 1 else None
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                          sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=256, num_workers=2, pin_memory=True)

    return train_dl, test_dl, train_sampler


def build_model(device):
    """ResNet-18 for CIFAR-10 (modified first layer for 32x32 input)."""
    model = models.resnet18(num_classes=10)
    # CIFAR-10: replace first conv (7x7, stride 2) with 3x3, stride 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # remove maxpool for 32x32
    return model.to(device)


def create_optimizer(name, model, lr, weight_decay, rank_fraction=0.5, adaptive_rank=False, device_mesh=None):
    """Create optimizer by name."""
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif name in ("dion_equiv", "adadion_v2"):
        from ada_dion.optim.adadion_v2 import AdaDionV2

        # Separate matrix (2D) and scalar (1D/bias) params
        matrix_params = []
        scalar_params = []
        for n, p in model.named_parameters():
            if p.ndim >= 2 and min(p.shape) >= 2:
                matrix_params.append(p)
            else:
                scalar_params.append(p)

        param_groups = [
            {"params": matrix_params},
            {"params": scalar_params, "algorithm": "adamw", "lr": lr * 0.1,
             "weight_decay": weight_decay, "betas": (0.9, 0.95), "eps": 1e-8},
        ]

        return AdaDionV2(
            param_groups,
            lr=lr,
            mu=0.95,
            rank_fraction_max=rank_fraction if name == "dion_equiv" else 0.7,
            weight_decay=weight_decay,
            adaptive_rank=(name == "adadion_v2" and adaptive_rank),
            init_rank_fraction=rank_fraction,
            erank_ema_beta=0.5,
            rank_scale=1.5,
            rank_min=8,
            rank_quantize=4,
            rank_step_up=8,
            rank_step_down=4,
            use_quality_control=adaptive_rank,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def evaluate(model, test_dl, device):
    model.eval()
    correct = total = 0
    loss_sum = 0
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return loss_sum / total, correct / total


def run_experiment(config, rank, local_rank, world_size):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    # data
    train_dl, test_dl, train_sampler = get_dataloaders(config["batch_size"], world_size, rank)

    # model
    torch.manual_seed(config["seed"])
    model = build_model(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # optimizer
    opt = create_optimizer(
        config["optimizer"], model, config["lr"], config["weight_decay"],
        config.get("rank_fraction", 0.5), config.get("adaptive_rank", False),
    )

    # scheduler
    total_steps = config["epochs"] * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # training loop
    results = {"config": config, "train": [], "eval": []}
    step = 0
    t_start = time.time()

    for epoch in range(config["epochs"]):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()

            opt.step()
            opt.zero_grad()
            scheduler.step()
            step += 1

            if step % config["log_freq"] == 0 and is_main:
                results["train"].append({
                    "step": step, "epoch": epoch,
                    "train_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed": time.time() - t_start,
                })

                # adaptive rank diagnostics
                if hasattr(opt, "get_rank"):
                    ranks = opt.get_rank()
                    if ranks:
                        results["train"][-1]["rank_mean"] = sum(ranks.values()) / len(ranks)

        # eval at end of each epoch
        val_loss, val_acc = evaluate(model.module if world_size > 1 else model, test_dl, device)
        if is_main:
            results["eval"].append({
                "epoch": epoch, "step": step,
                "val_loss": val_loss, "val_acc": val_acc,
                "elapsed": time.time() - t_start,
            })
            print(f"  [{config['name']}] epoch {epoch+1}/{config['epochs']}"
                  f"  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}"
                  f"  val_acc={val_acc:.4f}  {time.time()-t_start:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--outdir", type=str, default="benchmarks/cifar10_results")
    parser.add_argument("--optimizers", type=str, default="adamw,dion_equiv,adadion_v2",
                        help="comma-separated: adamw, dion_equiv, adadion_v2")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)

    if is_main:
        os.makedirs(args.outdir, exist_ok=True)
        print("=" * 60)
        print("  CIFAR-10 Benchmark: AdaDion V2")
        print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
        print(f"  GPUs: {world_size}, Seed: {args.seed}")
        print("=" * 60)

    opt_names = [o.strip() for o in args.optimizers.split(",")]

    configs = []
    for opt_name in opt_names:
        cfg = {
            "name": opt_name,
            "optimizer": opt_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr if opt_name == "adamw" else 0.02,
            "weight_decay": args.wd,
            "seed": args.seed,
            "log_freq": args.log_freq,
            "rank_fraction": 0.5,
            "adaptive_rank": (opt_name == "adadion_v2"),
        }
        configs.append(cfg)

    all_results = []
    for cfg in configs:
        if is_main:
            print(f"\n--- {cfg['name']} ---")
        res = run_experiment(cfg, rank, local_rank, world_size)
        all_results.append(res)

        if is_main:
            with open(os.path.join(args.outdir, f"{cfg['name']}.json"), "w") as f:
                json.dump(res, f, indent=2)

    # summary
    if is_main:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"  {'Optimizer':<20s} {'Val Loss':>10s} {'Val Acc':>10s} {'Time':>8s}")
        for r in all_results:
            if r["eval"]:
                last = r["eval"][-1]
                print(f"  {r['config']['name']:<20s} {last['val_loss']:10.4f} "
                      f"{last['val_acc']:10.4f} {last['elapsed']:7.1f}s")

        with open(os.path.join(args.outdir, "summary.json"), "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "train"} for r in all_results], f, indent=2)

    cleanup_distributed()


if __name__ == "__main__":
    main()
