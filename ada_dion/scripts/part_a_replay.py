#!/usr/bin/env python3
"""
Part A: Trace-and-Replay Study.

Phase 1: Run 500 steps of Dion baseline on LLaMA3 320M / C4, saving
         per-layer snapshots (grad, momentum, loss) every K steps.
Phase 2: Offline replay 4 optimizer formulas on the saved tensors.
         Log update norm, RMS, cosine-to-Dion, c_t*s_t, latency.

Usage (on pod):
  python ada_dion/scripts/part_a_replay.py \
    --snapshot-steps 500 --snapshot-freq 50 \
    --outdir replay_results
"""
import argparse, json, math, os, pickle, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms(t, eps=1e-12):
    return t.float().pow(2).mean().sqrt().clamp(min=eps).item()


def _col_norm(V, eps=1e-12):
    norms = torch.linalg.vector_norm(V.float(), dim=0).clamp(min=eps)
    return V / norms.to(V.dtype)


def dion_update(M, grad, Qbuf, r, mu=0.95):
    """One Dion step on a single tensor. Returns (D_raw, M_new, Qbuf, approx)."""
    dt = M.dtype
    M_new = mu * M + grad
    B = M_new + grad
    m, n = M.shape
    tr = m > n

    if not tr:
        Qp = Qbuf[:n, :r]
        Ph = B @ Qp
        Pf = Ph.float()
        gram = Pf.T @ Pf
        gram.diagonal().add_(1e-6)
        try:
            L = torch.linalg.cholesky(gram)
            Po = torch.linalg.solve_triangular(L, Pf.T, upper=False).T
        except torch.linalg.LinAlgError:
            Po, _ = torch.linalg.qr(Pf)
        Pd = Po.to(dt)
        R = (B.float().T @ Po).to(dt)
        approx = Pd @ R.T
        Qnew = _col_norm(R)
        D_raw = Pd @ Qnew.T
        Qbuf[:n, :r].copy_(Qnew)
    else:
        Qp = Qbuf[:m, :r]
        Phf = B.float().T @ Qp.float()
        Po, _ = torch.linalg.qr(Phf, mode="reduced")
        Pd = Po.to(dt)
        R = B @ Pd
        approx = R @ Pd.T
        Qnew = _col_norm(R)
        D_raw = Qnew @ Pd.T
        Qbuf[:m, :r].copy_(Qnew)

    M_ef = B - (1.0 - mu) * approx
    return D_raw, M_ef, Qbuf, approx


def replay_step(M, grad, Qbuf_orig, r, variant, state, mu=0.95, lr=0.02):
    """Replay one optimizer variant on saved (grad, M). Returns update + diagnostics."""
    Qbuf = Qbuf_orig.clone()
    m, n = M.shape
    dmin, dmax = min(m, n), max(m, n)
    ss = math.sqrt(dmin / dmax)

    t0 = time.perf_counter()
    D_raw, M_new, Qbuf_out, approx = dion_update(M, grad, Qbuf, r, mu)
    base_time = time.perf_counter() - t0

    # rank normalize
    if variant in ("normdion", "normdion_cs", "normdion_cs_rms"):
        D = D_raw / math.sqrt(r)
        eff_lr = lr * math.sqrt(r)
    else:
        D = D_raw
        eff_lr = lr

    # centered scalar
    s_t = 1.0
    if variant in ("normdion_cs", "normdion_cs_rms"):
        g_rms = _rms(grad)
        log_g = math.log(g_rms + 1e-12)
        if "log_ema" not in state:
            state["log_ema"] = log_g
        else:
            state["log_ema"] = 0.99 * state["log_ema"] + 0.01 * log_g
        dev = log_g - state["log_ema"]
        s_raw = math.exp(-0.3 * dev)
        s_t = max(1.0 / 1.2, min(1.2, s_raw))

    # RMS matching
    c_t = 1.0
    if variant in ("normdion_cs_rms",):
        rms_M = _rms(M)
        rms_D = _rms(D)
        c_t = rms_M / (rms_D + 1e-12)

    update = D * (eff_lr * ss * c_t * s_t)
    elapsed = time.perf_counter() - t0

    return {
        "update": update,
        "D_raw": D_raw,
        "update_fro": update.float().norm().item(),
        "update_rms": _rms(update),
        "D_fro": D.float().norm().item(),
        "s_t": s_t,
        "c_t": c_t,
        "cs_product": c_t * s_t,
        "eff_lr": eff_lr,
        "latency_ms": elapsed * 1000,
        "M_new": M_new,
        "Qbuf_out": Qbuf_out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", default="replay_snapshots")
    parser.add_argument("--outdir", default="replay_results")
    parser.add_argument("--ranks", default="64,128", help="comma-sep ranks")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.95)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ranks = [int(r) for r in args.ranks.split(",")]
    variants = ["dion", "normdion", "normdion_cs", "normdion_cs_rms"]

    snap_dir = args.snapshot_dir
    if not os.path.isdir(snap_dir):
        print(f"ERROR: snapshot dir {snap_dir} not found. Run Phase 1 first.")
        print("Phase 1 command:")
        print("  python ada_dion/scripts/part_a_collect.py --steps 500 --freq 50")
        sys.exit(1)

    snap_files = sorted([f for f in os.listdir(snap_dir) if f.endswith(".pt")])
    print(f"Found {len(snap_files)} snapshots in {snap_dir}")

    all_results = []

    for snap_file in snap_files:
        snap = torch.load(os.path.join(snap_dir, snap_file), weights_only=False)
        step = snap["step"]
        loss = snap["loss"]
        layers = snap["layers"]  # list of {name, grad, M, shape}

        print(f"\n--- Step {step}, loss={loss:.4f}, {len(layers)} layers ---")

        for rank in ranks:
            # first compute Dion baseline for cosine reference
            dion_updates = {}
            states = {v: {} for v in variants}

            for li, layer in enumerate(layers):
                name = layer["name"]
                grad = layer["grad"]
                M = layer["M"]
                m, n = M.shape
                dmin = min(m, n)

                if rank > dmin:
                    continue

                # init Qbuf deterministically
                gen = torch.Generator(device=M.device)
                gen.manual_seed(0x4753 + n * 1000 + rank)
                tr = m > n
                d0 = m if tr else n
                bc = min(dmin, 256)
                Qbuf_init = F.normalize(torch.randn(d0, bc, device=M.device, dtype=M.dtype, generator=gen), dim=0)

                row = {"step": step, "layer": name, "rank": rank, "loss": loss,
                       "grad_rms": _rms(grad), "M_rms": _rms(M),
                       "shape": f"{m}x{n}"}

                for vi, variant in enumerate(variants):
                    res = replay_step(M, grad, Qbuf_init.clone(), rank, variant,
                                      states[variant].setdefault(name, {}),
                                      mu=args.mu, lr=args.lr)

                    if variant == "dion":
                        dion_updates[name] = res["update"]

                    # cosine to Dion
                    if variant != "dion" and name in dion_updates:
                        cos = F.cosine_similarity(
                            res["update"].flatten().unsqueeze(0).float(),
                            dion_updates[name].flatten().unsqueeze(0).float()
                        ).item()
                    else:
                        cos = 1.0

                    # scalar product with momentum
                    sp = (M.float() * res["update"].float()).sum().item()

                    row[f"{variant}/update_fro"] = res["update_fro"]
                    row[f"{variant}/update_rms"] = res["update_rms"]
                    row[f"{variant}/s_t"] = res["s_t"]
                    row[f"{variant}/c_t"] = res["c_t"]
                    row[f"{variant}/cs"] = res["cs_product"]
                    row[f"{variant}/cos_to_dion"] = cos
                    row[f"{variant}/scalar_product"] = sp
                    row[f"{variant}/latency_ms"] = res["latency_ms"]

                # update norm ratio
                if "dion/update_rms" in row and "normdion_cs_rms/update_rms" in row:
                    row["rms_ratio_new_vs_dion"] = row["normdion_cs_rms/update_rms"] / (row["dion/update_rms"] + 1e-12)

                all_results.append(row)

                if li < 3:  # print first 3 layers
                    cos_cs = row.get("normdion_cs_rms/cos_to_dion", "?")
                    s = row.get("normdion_cs_rms/s_t", "?")
                    c = row.get("normdion_cs_rms/c_t", "?")
                    ratio = row.get("rms_ratio_new_vs_dion", "?")
                    print(f"  {name:<30s} r={rank}  cos={cos_cs:.4f}  s_t={s:.4f}  c_t={c:.4f}  rms_ratio={ratio:.4f}")

    # save
    out_path = os.path.join(args.outdir, "replay_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} entries to {out_path}")

    # summary statistics
    print("\n" + "=" * 72)
    print("  REPLAY SUMMARY")
    print("=" * 72)

    for rank in ranks:
        entries = [r for r in all_results if r["rank"] == rank]
        if not entries:
            continue
        print(f"\n  Rank {rank}:")
        for variant in variants:
            cos_key = f"{variant}/cos_to_dion"
            s_key = f"{variant}/s_t"
            rms_key = f"{variant}/update_rms"
            vals_cos = [r[cos_key] for r in entries if cos_key in r]
            vals_s = [r[s_key] for r in entries if s_key in r]
            vals_rms = [r[rms_key] for r in entries if rms_key in r]
            if vals_cos:
                mc = sum(vals_cos) / len(vals_cos)
                ms = sum(vals_s) / len(vals_s)
                mr = sum(vals_rms) / len(vals_rms)
                print(f"    {variant:<25s}  cos={mc:.4f}  s_t={ms:.4f}  update_rms={mr:.6f}")


if __name__ == "__main__":
    main()
