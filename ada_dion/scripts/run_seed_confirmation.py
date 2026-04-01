"""
Seed confirmation: Dion baseline vs Normalized Dion over 5 seeds.
Also logs update-norm ratio and cosine similarity between the two.
"""
import importlib.util, torch, torch.nn as nn, math, json, os, time, csv
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

spec = importlib.util.spec_from_file_location("gsdion", "ada_dion/optim/gsdion.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
GSDion = mod.GSDion

class MLP(nn.Module):
    def __init__(self, h=1024):
        super().__init__()
        self.fc1 = nn.Linear(784, h, bias=False)
        self.fc2 = nn.Linear(h, h, bias=False)
        self.fc3 = nn.Linear(h, 10, bias=False)
    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x.view(x.size(0), -1))))))

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    c = t = 0; ls = 0
    for x, y in dl:
        lo = model(x)
        ls += nn.functional.cross_entropy(lo, y, reduction="sum").item()
        c += (lo.argmax(1) == y).sum().item(); t += y.size(0)
    model.train()
    return ls / t, c / t

def run_pair(seed, train_dl, test_dl, steps=500, init_rank=64, lr=0.02):
    """Run Dion baseline and Normalized Dion with same seed, logging cosine sim."""
    results = {}

    for name, kw, run_lr in [
        ("dion", dict(rank_normalize=False, use_adaptive_scalar=False,
                      use_rms_matching=False, use_trust_region=False,
                      use_residual_rank=False), lr),
        ("normdion", dict(rank_normalize=True, use_adaptive_scalar=False,
                          use_rms_matching=False, use_trust_region=False,
                          use_residual_rank=False), lr * math.sqrt(init_rank)),
    ]:
        torch.manual_seed(seed)
        model = MLP()
        opt = GSDion(model.parameters(), lr=run_lr, init_rank=init_rank, **kw)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

        rows = []
        step = 0
        for ep in range(50):
            for x, y in train_dl:
                if step >= steps: break
                lo = model(x)
                loss = nn.functional.cross_entropy(lo, y)
                loss.backward()
                opt.step(); opt.zero_grad(); sched.step()
                step += 1
                if step % 50 == 0:
                    vl, va = evaluate(model, test_dl)
                    diag = opt.get_diagnostics()
                    rows.append({
                        "step": step,
                        "train_loss": loss.item(),
                        "val_loss": vl, "val_acc": va,
                        "dw_rms": diag.get("p0/dw_rms", 0),
                        "dw_fro": diag.get("p0/dw_fro", 0),
                        "grad_rms": diag.get("p0/grad_rms", 0),
                    })
            if step >= steps: break

        vl, va = evaluate(model, test_dl)
        results[name] = {"val_loss": vl, "val_acc": va, "rows": rows, "model": model}

    # cosine similarity between final updates (run one more step each)
    torch.manual_seed(seed + 99999)
    x = torch.randn(256, 784)
    y = torch.randint(0, 10, (256,))

    cosines = []
    for layer_name in ["fc1", "fc2", "fc3"]:
        m1 = results["dion"]["model"]
        m2 = results["normdion"]["model"]
        p1 = getattr(m1, layer_name).weight
        p2 = getattr(m2, layer_name).weight
        # use weight difference as proxy for accumulated update direction
        diff1 = p1.data.flatten()
        diff2 = p2.data.flatten()
        cos = torch.nn.functional.cosine_similarity(diff1.unsqueeze(0), diff2.unsqueeze(0)).item()
        cosines.append(cos)

    results["cosine_fc1"] = cosines[0]
    results["cosine_fc2"] = cosines[1]
    results["cosine_fc3"] = cosines[2]
    results["cosine_mean"] = sum(cosines) / len(cosines)

    return results


def main():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.286,), (0.353,))])
    train_ds = datasets.FashionMNIST("./data", train=True, download=True, transform=tf)
    test_ds = datasets.FashionMNIST("./data", train=False, transform=tf)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=1024)

    seeds = [42, 123, 7, 2024, 314]
    all_results = []

    print("=" * 72)
    print("  Seed Confirmation: Dion vs Normalized Dion (5 seeds)")
    print("  Steps: 500, LR: 0.02 (normdion: 0.02*sqrt(64)=0.16), Rank: 64")
    print("=" * 72)

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()
        res = run_pair(seed, train_dl, test_dl)
        elapsed = time.time() - t0
        d_vl = res["dion"]["val_loss"]
        d_va = res["dion"]["val_acc"]
        n_vl = res["normdion"]["val_loss"]
        n_va = res["normdion"]["val_acc"]
        cos_mean = res["cosine_mean"]
        print(f"  Dion:     val_loss={d_vl:.4f}  val_acc={d_va:.4f}")
        print(f"  NormDion: val_loss={n_vl:.4f}  val_acc={n_va:.4f}")
        print(f"  Cosine(weight dirs): fc1={res['cosine_fc1']:.4f} fc2={res['cosine_fc2']:.4f} fc3={res['cosine_fc3']:.4f} mean={cos_mean:.4f}")
        print(f"  Delta: val_acc {(n_va - d_va)*100:+.2f}pp  val_loss {n_vl - d_vl:+.4f}  ({elapsed:.1f}s)")
        all_results.append({
            "seed": seed,
            "dion_val_loss": d_vl, "dion_val_acc": d_va,
            "normdion_val_loss": n_vl, "normdion_val_acc": n_va,
            "cosine_fc1": res["cosine_fc1"],
            "cosine_fc2": res["cosine_fc2"],
            "cosine_fc3": res["cosine_fc3"],
            "cosine_mean": cos_mean,
        })

    # summary
    print("\n" + "=" * 72)
    print("  SUMMARY (5 seeds)")
    print("=" * 72)
    d_accs = [r["dion_val_acc"] for r in all_results]
    n_accs = [r["normdion_val_acc"] for r in all_results]
    d_loss = [r["dion_val_loss"] for r in all_results]
    n_loss = [r["normdion_val_loss"] for r in all_results]
    cosines = [r["cosine_mean"] for r in all_results]

    def stats(vals):
        m = sum(vals) / len(vals)
        std = math.sqrt(sum((v - m)**2 for v in vals) / len(vals))
        return m, std

    da_m, da_s = stats(d_accs)
    na_m, na_s = stats(n_accs)
    dl_m, dl_s = stats(d_loss)
    nl_m, nl_s = stats(n_loss)
    c_m, c_s = stats(cosines)

    print(f"  {'Metric':<25s} {'Dion':>15s} {'NormDion':>15s} {'Delta':>12s}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")
    print(f"  {'Val Acc':<25s} {da_m:.4f}±{da_s:.4f} {na_m:.4f}±{na_s:.4f} {(na_m-da_m)*100:+.2f}pp")
    print(f"  {'Val Loss':<25s} {dl_m:.4f}±{dl_s:.4f} {nl_m:.4f}±{nl_s:.4f} {nl_m-dl_m:+.4f}")
    print(f"  {'Cosine(weight dirs)':<25s} {'':>15s} {c_m:.4f}±{c_s:.4f}")
    print()

    # individual seed results
    print(f"  {'Seed':>6s}  {'Dion Acc':>10s}  {'Norm Acc':>10s}  {'Δpp':>8s}  {'Cosine':>8s}")
    for r in all_results:
        d = (r["normdion_val_acc"] - r["dion_val_acc"]) * 100
        print(f"  {r['seed']:6d}  {r['dion_val_acc']:10.4f}  {r['normdion_val_acc']:10.4f}  {d:+7.2f}  {r['cosine_mean']:8.4f}")

    os.makedirs("gsdion_results", exist_ok=True)
    with open("gsdion_results/seed_confirmation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to gsdion_results/seed_confirmation.json")


if __name__ == "__main__":
    main()
