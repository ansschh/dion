"""
GSDion ablation comparison on FashionMNIST.

Runs the 6-step ablation ladder from the advisor's recommendation:
  1. Dion (fixed rank, baseline)
  2. Dion + rank normalization
  3. + adaptive scalar
  4. + trust-region clipping
  5. + residual-driven rank (full GSDion)
  6. + MuonEq pre-balance

Logs every metric of interest per step to CSV and optionally W&B.
"""
import argparse, csv, importlib.util, json, math, os, sys, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# load optimizers directly to avoid dion import issues on CPU
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

gsdion_mod = _load("gsdion", "ada_dion/optim/gsdion.py")
GSDion = gsdion_mod.GSDion

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, 10, bias=False)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

# ---------------------------------------------------------------------------
# Configs — ablation ladder
# ---------------------------------------------------------------------------
# Advisor's recommended ablation order — isolates each mechanism
CONFIGS = {
    # 1. Dion baseline (fixed rank, no modifications)
    "1_dion_baseline": dict(
        rank_normalize=False, use_adaptive_scalar=False,
        use_rms_matching=False, use_trust_region=False,
        use_residual_rank=False, use_pre_balance=False, use_anchor=False,
    ),
    # 2. Dion + rank norm + lr*sqrt(r) to compensate
    #    (tests whether rank norm itself is neutral when LR compensated)
    "2_ranknorm_lrcomp": dict(
        rank_normalize=True, use_adaptive_scalar=False,
        use_rms_matching=False, use_trust_region=False,
        use_residual_rank=False, use_pre_balance=False, use_anchor=False,
        # LR override handled in run_single via lr_mult
    ),
    # 3. Dion + rank norm + c_t (RMS matching) only
    "3_ranknorm_rms": dict(
        rank_normalize=True, use_adaptive_scalar=False,
        use_rms_matching=True, use_trust_region=False,
        use_residual_rank=False, use_pre_balance=False, use_anchor=False,
    ),
    # 4. Dion + rank norm + bounded s_t only
    "4_ranknorm_scalar": dict(
        rank_normalize=True, use_adaptive_scalar=True,
        use_rms_matching=False, use_trust_region=False,
        use_residual_rank=False, use_pre_balance=False, use_anchor=False,
    ),
    # 5. Dion + rank norm + c_t + s_t
    "5_ranknorm_rms_scalar": dict(
        rank_normalize=True, use_adaptive_scalar=True,
        use_rms_matching=True, use_trust_region=False,
        use_residual_rank=False, use_pre_balance=False, use_anchor=False,
    ),
    # 6. + trust region
    "6_ranknorm_rms_scalar_tr": dict(
        rank_normalize=True, use_adaptive_scalar=True,
        use_rms_matching=True, use_trust_region=True,
        use_residual_rank=False, use_pre_balance=False, use_anchor=False,
    ),
    # 7. Full GSDion: + residual rank
    "7_full_gsdion": dict(
        rank_normalize=True, use_adaptive_scalar=True,
        use_rms_matching=True, use_trust_region=True,
        use_residual_rank=True, use_pre_balance=False, use_anchor=False,
    ),
}

# LR multiplier for configs with rank_normalize (compensate sqrt(r) shrinkage)
LR_MULT = {
    "2_ranknorm_lrcomp": None,  # set dynamically based on init_rank
}

# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    correct = total = 0; loss_sum = 0
    for x, y in dl:
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    avg_loss = loss_sum / total
    return avg_loss, correct / total, math.exp(min(avg_loss, 20))

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(config_name, config_kw, args, train_dl, test_dl, wandb_run=None):
    torch.manual_seed(args.seed)
    model = MLP(hidden=args.hidden)
    # LR compensation for rank-normalized configs
    run_lr = args.lr
    if config_name in LR_MULT:
        run_lr = args.lr * math.sqrt(args.init_rank)
        print(f"  [LR compensated: {args.lr} * sqrt({args.init_rank}) = {run_lr:.4f}]")
    opt = GSDion(model.parameters(), lr=run_lr, init_rank=args.init_rank, **config_kw)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    log_rows = []
    step = 0
    t0 = time.time()

    for epoch in range(999):
        for x, y in train_dl:
            if step >= args.steps:
                break
            logits = model(x)
            train_loss = nn.functional.cross_entropy(logits, y)
            train_loss.backward()

            # grad norm before step
            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.float().pow(2).sum().item()
            grad_norm = math.sqrt(grad_norm)

            opt.step()
            opt.zero_grad()
            scheduler.step()
            step += 1

            # per-step metrics
            row = {
                "step": step,
                "train_loss": train_loss.item(),
                "train_ppl": math.exp(min(train_loss.item(), 20)),
                "grad_norm": grad_norm,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - t0,
            }

            # optimizer diagnostics
            diag = opt.get_diagnostics()
            for k, v in diag.items():
                row[k] = v

            ranks = opt.get_rank()
            if ranks:
                row["rank"] = list(ranks.values())[0]
                row["rank_mean"] = sum(ranks.values()) / len(ranks)

            drifts = opt.get_anchor_drift()
            if drifts:
                row["anchor_drift"] = sum(drifts.values()) / len(drifts)

            # validation
            if step % args.eval_freq == 0 or step == args.steps:
                val_loss, val_acc, val_ppl = evaluate(model, test_dl)
                row["val_loss"] = val_loss
                row["val_acc"] = val_acc
                row["val_ppl"] = val_ppl

                elapsed = time.time() - t0
                print(f"  [{config_name}] step {step:4d}/{args.steps}"
                      f"  train={train_loss.item():.4f}"
                      f"  val={val_loss:.4f}  acc={val_acc:.4f}  ppl={val_ppl:.1f}"
                      f"  r={row.get('rank','?')}"
                      f"  q={row.get('p0/q_t','?')}"
                      f"  s={row.get('p0/s_t','?')}"
                      f"  scale={row.get('p0/scale','?')}"
                      f"  {elapsed:.1f}s")

            # W&B
            if wandb_run is not None:
                wandb_run.log({f"{config_name}/{k}": v for k, v in row.items()}, step=step)

            log_rows.append(row)
        if step >= args.steps:
            break

    val_loss, val_acc, val_ppl = evaluate(model, test_dl)
    return {
        "config": config_name,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "final_val_ppl": val_ppl,
        "total_time": time.time() - t0,
        "rows": log_rows,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--init-rank", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="gsdion_results")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gsdion-ablation")
    parser.add_argument("--configs", type=str, default="all",
                        help="comma-separated config names or 'all'")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # data
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST("./data", train=False, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=1024)

    # select configs
    if args.configs == "all":
        selected = list(CONFIGS.items())
    else:
        names = [n.strip() for n in args.configs.split(",")]
        selected = [(n, CONFIGS[n]) for n in names if n in CONFIGS]

    # W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    print("=" * 72)
    print(f"  GSDion Ablation Comparison")
    print(f"  Steps: {args.steps}  LR: {args.lr}  Rank: {args.init_rank}")
    print(f"  Hidden: {args.hidden}  Batch: {args.batch_size}  Seed: {args.seed}")
    print(f"  Configs: {len(selected)}")
    print("=" * 72)

    results = []
    for name, kw in selected:
        print(f"\n--- {name} ---")
        res = run_single(name, kw, args, train_dl, test_dl, wandb_run)
        results.append(res)

        # save per-run CSV
        csv_path = os.path.join(args.outdir, f"{name}.csv")
        if res["rows"]:
            keys = list(res["rows"][0].keys())
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(res["rows"])

    # summary
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'Config':<30s} {'Val Loss':>10s} {'Val Acc':>10s} {'Val PPL':>10s} {'Time':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        print(f"  {r['config']:<30s} {r['final_val_loss']:10.4f} {r['final_val_acc']:10.4f} "
              f"{r['final_val_ppl']:10.1f} {r['total_time']:7.1f}s")

    # save summary JSON
    summary = [{k: v for k, v in r.items() if k != "rows"} for r in results]
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run:
        wandb_run.finish()

    print(f"\nResults saved to {args.outdir}/")


if __name__ == "__main__":
    main()
