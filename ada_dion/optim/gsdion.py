"""
GSDion — Guarded Scale-Dion.

Dion's low-rank power iteration with safeguards addressing the implicit
step-size bug (||D_t||_F = sqrt(r_t)):

  1. Rank normalization:  D_t / sqrt(r_t)  →  ||D_t||_F = 1
  2. Adaptive scalar:     bias-corrected, bounded relative scalar around 1
       v_t = beta2 * v_{t-1} + (1-beta2) * RMS(G)^2
       v_hat = v_t / (1 - beta2^t)             (bias correction)
       s_raw = 1 / sqrt(v_hat + eps)
       s_t = clip(s_raw / s_ref, s_min, s_max)  (bounded around 1)
  3. RMS matching + trust-region:
       c_t = RMS(M) / (RMS(D_norm) + eps)
       q_t = ||M - proj||_F / (||M||_F + eps)   (residual quality)
       alpha_t = min(1, tau / (q_t + eps))
  4. Residual-driven rank with hysteresis
  5. Periodic refresh
  6. Optional MuonEq pre-balance
  7. Anchor (H2)

Each feature independently toggleable. FSDP2/DTensor compatible.

Key diagnostic: the full scalar m_t = lr * ss * alpha_t * c_t * s_t and
||delta_W||_F are logged per layer per step.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def _col_norm(V, eps=1e-12):
    norms = torch.linalg.vector_norm(V.float(), dim=0).clamp(min=eps)
    return V / norms.to(V.dtype)


def _col_norm_dist(V, pg, eps=1e-12):
    sq = torch.linalg.vector_norm(V.float(), dim=0).pow(2)
    if pg is not None:
        dist.all_reduce(sq, group=pg)
    elif dist.is_initialized():
        dist.all_reduce(sq)
    return V / sq.sqrt().clamp(min=eps).to(V.dtype)


def _qint(x, q):
    return int(round(x / q) * q) if q > 1 else int(round(x))


def _rms(t, eps=1e-12):
    return t.float().pow(2).mean().sqrt().clamp(min=eps)


def _rms_dist(t, pg, eps=1e-12):
    ss = t.float().pow(2).sum()
    n = torch.tensor(t.numel(), device=t.device, dtype=torch.float32)
    if pg is not None:
        dist.all_reduce(ss, group=pg)
        dist.all_reduce(n, group=pg)
    elif dist.is_initialized():
        dist.all_reduce(ss)
        dist.all_reduce(n)
    return (ss / n).sqrt().clamp(min=eps)


class GSDion(Optimizer):
    """Guarded Scale-Dion: Dion + scale control + safeguards."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        mu: float = 0.95,
        init_rank: int = 64,
        rank_min: int = 16,
        rank_max: int = 256,
        rank_quantize: int = 8,
        # feature flags
        rank_normalize: bool = True,
        use_adaptive_scalar: bool = True,
        scalar_beta2: float = 0.999,
        scalar_min: float = 0.5,
        scalar_max: float = 2.0,
        use_rms_matching: bool = True,
        use_trust_region: bool = True,
        trust_tau: float = 1.0,
        use_residual_rank: bool = True,
        tau_up: float = 0.15,
        tau_down: float = 0.05,
        hysteresis_k: int = 5,
        rank_warmup_steps: int = 10,
        warmup_rank: int = 128,
        # refresh
        refresh_period: int = 100,
        refresh_spike: float = 0.5,
        # pre-balance (MuonEq)
        use_pre_balance: bool = False,
        balance_beta: float = 0.99,
        # anchor (H2)
        use_anchor: bool = False,
        anchor_alpha: float = 0.1,
        anchor_period: int = 50,
        # misc
        qbuf_max_cols: int = 256,
        weight_decay: float = 0.0,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        defaults = dict(
            lr=lr, mu=mu, init_rank=init_rank,
            rank_min=rank_min, rank_max=rank_max, rank_quantize=rank_quantize,
            rank_normalize=rank_normalize,
            use_adaptive_scalar=use_adaptive_scalar, scalar_beta2=scalar_beta2,
            scalar_min=scalar_min, scalar_max=scalar_max,
            use_rms_matching=use_rms_matching,
            use_trust_region=use_trust_region, trust_tau=trust_tau,
            use_residual_rank=use_residual_rank,
            tau_up=tau_up, tau_down=tau_down, hysteresis_k=hysteresis_k,
            rank_warmup_steps=rank_warmup_steps, warmup_rank=warmup_rank,
            refresh_period=refresh_period, refresh_spike=refresh_spike,
            use_pre_balance=use_pre_balance, balance_beta=balance_beta,
            use_anchor=use_anchor, anchor_alpha=anchor_alpha, anchor_period=anchor_period,
            qbuf_max_cols=qbuf_max_cols, weight_decay=weight_decay,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)
        self._mesh = device_mesh
        self._pg = device_mesh.get_group() if device_mesh else None

    def _is_mat(self, p): return p.ndim == 2 and min(p.shape) >= 2
    def _loc(self, t): return t._local_tensor if isinstance(t, DTensor) else t
    def _fshape(self, p): return tuple(p.shape)
    def _ws(self):
        if self._pg: return dist.get_world_size(self._pg)
        return dist.get_world_size() if dist.is_initialized() else 1
    def _ar(self, t):
        if self._pg: dist.all_reduce(t, group=self._pg)
        elif self._mesh is None and dist.is_initialized(): dist.all_reduce(t)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": self._adamw(g)
            else: self._gsdion(g)
        return loss

    def _adamw(self, g):
        betas = g.get("betas", g.get("adamw_betas", (0.9, 0.95)))
        eps = g.get("eps", g.get("adamw_eps", 1e-8))
        lr, wd = g.get("lr", self.defaults["lr"]), g.get("weight_decay", 0.0)
        for p in g["params"]:
            if p.grad is None: continue
            s = self.state[p]
            if len(s) == 0:
                s["step"] = 0; s["m1"] = torch.zeros_like(p); s["m2"] = torch.zeros_like(p)
            s["step"] += 1; t = s["step"]
            if wd: p.data.mul_(1 - lr * wd)
            s["m1"].mul_(betas[0]).add_(p.grad, alpha=1 - betas[0])
            s["m2"].mul_(betas[1]).addcmul_(p.grad, p.grad, value=1 - betas[1])
            bc1, bc2 = 1 - betas[0] ** t, 1 - betas[1] ** t
            p.data.addcdiv_(s["m1"], (s["m2"].sqrt() / math.sqrt(bc2)).add_(eps), value=-lr / bc1)

    def _gsdion(self, g):
        lr = g["lr"]; mu = g["mu"]
        init_rank = g["init_rank"]; rmin = g["rank_min"]; rmax = g["rank_max"]; rq = g["rank_quantize"]
        do_rnorm = g["rank_normalize"]
        do_scalar = g["use_adaptive_scalar"]; sb2 = g["scalar_beta2"]
        s_lo, s_hi = g["scalar_min"], g["scalar_max"]
        do_rms = g["use_rms_matching"]
        do_tr = g["use_trust_region"]; tr_tau = g["trust_tau"]
        do_resrank = g["use_residual_rank"]
        tau_up = g["tau_up"]; tau_dn = g["tau_down"]; hyst_k = g["hysteresis_k"]
        rwarm_steps = g["rank_warmup_steps"]; rwarm_rank = g["warmup_rank"]
        ref_period = g["refresh_period"]; ref_spike = g["refresh_spike"]
        do_prebal = g["use_pre_balance"]; bal_beta = g["balance_beta"]
        do_anc = g["use_anchor"]; anc_alpha = g["anchor_alpha"]; anc_per = g["anchor_period"]
        qmax = g["qbuf_max_cols"]; wd = g["weight_decay"]

        for p in g["params"]:
            if p.grad is None: continue
            grad = self._loc(p.grad); pl = self._loc(p); fs = self._fshape(p)
            if not self._is_mat(p):
                if wd: pl.mul_(1 - lr * wd)
                pl.add_(grad, alpha=-lr); continue

            st = self.state[p]
            mf, n = fs; ml = pl.shape[0]
            dmin, dmax = min(mf, n), max(mf, n)
            ss = math.sqrt(dmin / dmax)
            rc = min(dmin, rmax)
            if rc <= 0: continue

            if len(st) == 0:
                tr = mf > n
                bc = min(rc, qmax)
                ir = min(init_rank, rc, bc)
                ws = self._ws()
                if ws > 1 and isinstance(p, DTensor):
                    assert mf % ws == 0 and ml == mf // ws
                gen = torch.Generator(device=pl.device)
                gen.manual_seed(0x4753_4469 + n * 1000 + bc)
                d0 = ml if tr else n
                Qbuf = F.normalize(torch.randn(d0, bc, device=pl.device, dtype=pl.dtype, generator=gen), dim=0)
                re = min(rmin, rc, bc)
                st.update({
                    "M": torch.zeros_like(pl), "Qbuf": Qbuf,
                    "r": int(max(1, min(max(re, ir), rc, bc))),
                    "tr": tr, "step": 0,
                    "v_scalar": 0.0,
                    "s_ref": None,
                    "q_ema": 0.1,
                    "good_streak": 0,
                    "Q_anchor": None,
                })
                if do_prebal:
                    st["Dr_ema"] = torch.ones(ml, device=pl.device, dtype=torch.float32)
                    st["Dc_ema"] = torch.ones(n, device=pl.device, dtype=torch.float32)

            st["step"] += 1
            M = st["M"]; Qbuf = st["Qbuf"]; r = st["r"]; tr = st["tr"]; dt = M.dtype
            re = int(max(1, min(rmin, rc, Qbuf.shape[1])))
            r = int(min(max(re, r if do_resrank else init_rank), Qbuf.shape[1], rc))

            # -- pre-balance (MuonEq) --
            if do_prebal:
                row_sq = grad.float().pow(2).sum(dim=1)
                col_sq = grad.float().pow(2).sum(dim=0)
                self._ar(col_sq)
                if tr: self._ar(row_sq)
                st["Dr_ema"].mul_(bal_beta).add_(row_sq, alpha=1 - bal_beta)
                st["Dc_ema"].mul_(bal_beta).add_(col_sq, alpha=1 - bal_beta)
                Dr_inv = st["Dr_ema"].clamp(min=1e-12).rsqrt()
                Dc_inv = st["Dc_ema"].clamp(min=1e-12).rsqrt()
                grad_used = (grad.float() * Dr_inv.unsqueeze(1) * Dc_inv.unsqueeze(0)).to(dt)
            else:
                grad_used = grad

            # -- adaptive scalar: track RMS(grad) with bias correction --
            grad_rms = (_rms_dist(grad, self._pg) if tr else _rms(grad)).item()
            if do_scalar:
                st["v_scalar"] = sb2 * st["v_scalar"] + (1 - sb2) * (grad_rms ** 2)
                v_hat = st["v_scalar"] / (1 - sb2 ** st["step"])  # bias correction
                s_raw = 1.0 / math.sqrt(v_hat + 1e-8)
                # set reference on first step
                if st["s_ref"] is None:
                    st["s_ref"] = s_raw
                # bounded relative scalar around 1
                s_rel = s_raw / st["s_ref"]
                s_t = max(s_lo, min(s_hi, s_rel))
            else:
                s_t = 1.0; s_raw = 1.0

            # -- momentum + B --
            M.mul_(mu).add_(grad_used)
            B = M + grad_used

            # -- power iteration + Cholesky QR --
            if not tr:
                Qp = Qbuf[:n, :r]
                Ph = B @ Qp
                Pf = Ph.float()
                gram = Pf.T @ Pf; self._ar(gram)
                gram.diagonal().add_(1e-6)
                try:
                    L = torch.linalg.cholesky(gram)
                    Po = torch.linalg.solve_triangular(L, Pf.T, upper=False).T
                except torch.linalg.LinAlgError:
                    Po, _ = torch.linalg.qr(Pf)
                Pd = Po.to(dt)
                R = (B.float().T @ Po).to(dt); self._ar(R)
                approx = Pd @ R.T
                Qnew = _col_norm(R)
                if do_anc and st["Q_anchor"] is not None and anc_alpha > 0:
                    ar = min(r, st["Q_anchor"].shape[1])
                    Qnew[:, :ar] = (1 - anc_alpha) * Qnew[:, :ar] + anc_alpha * st["Q_anchor"][:n, :ar]
                    Qnew = _col_norm(Qnew)
                D_raw = Pd @ Qnew.T
                Qbuf[:n, :r].copy_(Qnew)
            else:
                Qp = Qbuf[:ml, :r]
                Phf = (B.float().T @ Qp.float()); self._ar(Phf)
                Po, _ = torch.linalg.qr(Phf, mode="reduced")
                Pd = Po.to(dt)
                R = B @ Pd
                approx = R @ Pd.T
                Qnew = _col_norm_dist(R, self._pg)
                if do_anc and st["Q_anchor"] is not None and anc_alpha > 0:
                    ar = min(r, st["Q_anchor"].shape[1])
                    Qnew[:, :ar] = (1 - anc_alpha) * Qnew[:, :ar] + anc_alpha * st["Q_anchor"][:ml, :ar]
                    Qnew = _col_norm_dist(Qnew, self._pg)
                D_raw = Qnew @ Pd.T
                Qbuf[:ml, :r].copy_(Qnew)

            if torch.isnan(D_raw).any() or torch.isinf(D_raw).any():
                D_raw = torch.zeros_like(D_raw)

            # -- residual quality --
            resid_sq = (B - approx).float().pow(2).sum()
            denom_sq = B.float().pow(2).sum()
            self._ar(resid_sq); self._ar(denom_sq)
            q_t = (resid_sq.sqrt() / (denom_sq.sqrt() + 1e-12)).item()
            st["q_ema"] = 0.95 * st["q_ema"] + 0.05 * q_t

            # -- rank normalization --
            if do_rnorm and r > 0:
                D_norm = D_raw / math.sqrt(r)
            else:
                D_norm = D_raw

            # -- RMS matching --
            if do_rms:
                rms_M = (_rms_dist(M, self._pg) if tr else _rms(M)).item()
                rms_D = (_rms_dist(D_norm, self._pg) if tr else _rms(D_norm)).item()
                c_t = rms_M / (rms_D + 1e-12)
            else:
                c_t = 1.0

            # -- trust-region clipping --
            if do_tr:
                alpha_t = min(1.0, tr_tau / (q_t + 1e-12))
            else:
                alpha_t = 1.0

            # -- final multiplier --
            m_t = alpha_t * c_t * s_t

            # -- compute actual delta_W for logging --
            delta_W = D_norm * (lr * ss * m_t)
            dw_fro = delta_W.float().norm().item()
            dw_rms = _rms(delta_W).item()

            # -- error feedback --
            M.copy_(B - (1.0 - mu) * approx)

            # -- weight update --
            if wd: pl.mul_(1 - lr * wd)
            pl.add_(D_norm, alpha=-(lr * ss * m_t))

            # -- anchor refresh --
            if do_anc and (st["step"] % anc_per == 0 or st["Q_anchor"] is None):
                st["Q_anchor"] = Qbuf.clone()

            # -- periodic refresh --
            do_refresh = (st["step"] % ref_period == 0) or (q_t > ref_spike)
            if do_refresh and st["step"] > rwarm_steps:
                gen = torch.Generator(device=pl.device)
                gen.manual_seed(0x5246 + st["step"])
                d0 = Qbuf.shape[0]
                Qbuf[:, :r].copy_(F.normalize(
                    torch.randn(d0, r, device=pl.device, dtype=dt, generator=gen), dim=0))

            # -- residual-driven rank --
            if do_resrank and st["step"] > rwarm_steps:
                if q_t > tau_up:
                    st["good_streak"] = 0
                    r_next = min(r * 2, rc, Qbuf.shape[1])
                elif q_t < tau_dn:
                    st["good_streak"] += 1
                    r_next = max(r // 2, re) if st["good_streak"] >= hyst_k else r
                    if st["good_streak"] >= hyst_k: st["good_streak"] = 0
                else:
                    st["good_streak"] = 0; r_next = r
                r_next = _qint(min(max(re, r_next), rc, Qbuf.shape[1]), rq)
                r_next = int(min(max(re, r_next), rc, Qbuf.shape[1]))
                if r_next > r:
                    d0 = Qbuf.shape[0]
                    Qbuf[:, r:r_next].copy_(F.normalize(
                        torch.randn(d0, r_next - r, device=pl.device, dtype=dt), dim=0))
                st["r"] = r_next
            elif st["step"] <= rwarm_steps:
                des = max(re, int(min(rwarm_rank, rc, Qbuf.shape[1])))
                if des > r:
                    d0 = Qbuf.shape[0]
                    Qbuf[:, r:des].copy_(F.normalize(
                        torch.randn(d0, des - r, device=pl.device, dtype=dt), dim=0))
                st["r"] = des

            # -- store all diagnostics --
            st["_q_t"] = q_t
            st["_s_t"] = s_t
            st["_s_raw"] = s_raw
            st["_c_t"] = c_t
            st["_alpha_t"] = alpha_t
            st["_m_t"] = m_t
            st["_dw_fro"] = dw_fro
            st["_dw_rms"] = dw_rms
            st["_grad_rms"] = grad_rms
            st["_update_rms"] = _rms(D_norm).item()

    # -- diagnostics --
    def get_diagnostics(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": continue
            for p in g["params"]:
                if not self._is_mat(p): continue
                st = self.state[p]
                pf = f"p{i}"
                for k in ["r", "_q_t", "_s_t", "_s_raw", "_c_t", "_alpha_t", "_m_t",
                           "_dw_fro", "_dw_rms", "_grad_rms", "_update_rms", "q_ema"]:
                    if k in st: out[f"{pf}/{k.lstrip('_')}"] = st[k]
                i += 1
        return out

    def get_rank(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": continue
            for p in g["params"]:
                if self._is_mat(p) and "r" in self.state[p]: out[f"p{i}"] = self.state[p]["r"]
                i += 1
        return out

    def get_anchor_drift(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": continue
            for p in g["params"]:
                if not self._is_mat(p): continue
                st = self.state[p]
                Qa = st.get("Q_anchor")
                if Qa is not None and "Qbuf" in st:
                    r = st.get("r", 0)
                    d0 = min(st["Qbuf"].shape[0], Qa.shape[0])
                    rc = min(r, Qa.shape[1])
                    d = (st["Qbuf"][:d0, :rc] - Qa[:d0, :rc]).norm().item()
                    out[f"p{i}"] = d if math.isfinite(d) else 0.0
                i += 1
        return out

    def state_dict(self):
        sd = super().state_dict()
        if self._ws() > 1: sd["_gsdion_ws"] = self._ws()
        return sd

    def load_state_dict(self, sd):
        saved = sd.pop("_gsdion_ws", None)
        if saved is not None and saved != self._ws():
            raise RuntimeError(f"Checkpoint ws={saved} != current {self._ws()}")
        super().load_state_dict(sd)
