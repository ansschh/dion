"""
QCDion — Quality-Controlled Dion.

Plain Dion update geometry preserved exactly. No rank normalization,
no gradient-based scaling. Intervenes only when approximation quality
signals degrade:

  1. Quality gate: g_t = clip(exp(-rho*(z_t - z_bar)), [1/gamma, gamma])
     where z_t = a*q_t + b*d_t + c*e_t is a composite quality score from:
       q_t = ||M - proj||_F / ||M||_F       (approx residual)
       d_t = ||Q_t Q_t^T - Q_{t-1} Q_{t-1}^T||_F  (subspace drift)
       e_t = ||EF_residual||_F / ||M||_F    (error-feedback burden)
     g_t ≈ 1 when quality is normal; shrinks step on quality spikes.

  2. Quality-triggered extra power iteration:
     If q_t > tau_extra, do one extra power step (better approximation).

  3. Quality-triggered rank boost:
     If q_t > tau_rank for K consecutive steps, temporarily double rank.
     Fall back once quality stabilizes.

Weight decay is handled separately from the gate (no LR coupling).
FSDP2/DTensor compatible.
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


class QCDion(Optimizer):
    """Quality-Controlled Dion."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        mu: float = 0.95,
        init_rank: int = 64,
        rank_min: int = 16,
        rank_max: int = 256,
        # quality gate
        use_quality_gate: bool = True,
        gate_rho: float = 0.3,
        gate_gamma: float = 1.15,
        gate_beta: float = 0.99,
        gate_w_q: float = 1.0,
        gate_w_d: float = 0.3,
        gate_w_e: float = 0.3,
        # extra power iteration
        use_extra_power: bool = True,
        tau_extra: float = 0.2,
        # quality-triggered rank boost
        use_rank_boost: bool = True,
        tau_rank: float = 0.25,
        boost_k: int = 5,
        # misc
        qbuf_max_cols: int = 256,
        weight_decay: float = 0.0,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        defaults = dict(
            lr=lr, mu=mu, init_rank=init_rank,
            rank_min=rank_min, rank_max=rank_max,
            use_quality_gate=use_quality_gate,
            gate_rho=gate_rho, gate_gamma=gate_gamma, gate_beta=gate_beta,
            gate_w_q=gate_w_q, gate_w_d=gate_w_d, gate_w_e=gate_w_e,
            use_extra_power=use_extra_power, tau_extra=tau_extra,
            use_rank_boost=use_rank_boost, tau_rank=tau_rank, boost_k=boost_k,
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
            else: self._qcdion(g)
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

    def _power_step(self, B, Qbuf, r, tr, ml, n, dt):
        """One power iteration + Cholesky QR. Returns (D, approx, Qnew, R)."""
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
            D = Pd @ Qnew.T
            return D, approx, Qnew, R
        else:
            Qp = Qbuf[:ml, :r]
            Phf = (B.float().T @ Qp.float()); self._ar(Phf)
            Po, _ = torch.linalg.qr(Phf, mode="reduced")
            Pd = Po.to(dt)
            R = B @ Pd
            approx = R @ Pd.T
            Qnew = _col_norm_dist(R, self._pg)
            D = Qnew @ Pd.T
            return D, approx, Qnew, R

    def _qcdion(self, g):
        lr = g["lr"]; mu = g["mu"]
        init_rank = g["init_rank"]; rmin = g["rank_min"]; rmax = g["rank_max"]
        do_gate = g["use_quality_gate"]
        g_rho = g["gate_rho"]; g_gamma = g["gate_gamma"]; g_beta = g["gate_beta"]
        w_q, w_d, w_e = g["gate_w_q"], g["gate_w_d"], g["gate_w_e"]
        do_extra = g["use_extra_power"]; tau_extra = g["tau_extra"]
        do_boost = g["use_rank_boost"]; tau_rank = g["tau_rank"]; boost_k = g["boost_k"]
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
                gen.manual_seed(0x5143_4469 + n * 1000 + bc)
                d0 = ml if tr else n
                Qbuf = F.normalize(torch.randn(d0, bc, device=pl.device, dtype=pl.dtype, generator=gen), dim=0)
                re = min(rmin, rc, bc)
                st.update({
                    "M": torch.zeros_like(pl), "Qbuf": Qbuf,
                    "r": int(max(1, min(max(re, ir), rc, bc))),
                    "r_base": int(max(1, min(max(re, ir), rc, bc))),
                    "tr": tr, "step": 0,
                    "z_ema": 0.1,
                    "bad_streak": 0,
                    "Q_prev_proj": None,
                })

            st["step"] += 1
            M = st["M"]; Qbuf = st["Qbuf"]; r = st["r"]; tr = st["tr"]; dt = M.dtype
            r_base = st["r_base"]

            M.mul_(mu).add_(grad)
            B = M + grad

            # -- standard power step --
            D, approx, Qnew, R = self._power_step(B, Qbuf, r, tr, ml, n, dt)

            # -- quality signals --
            M_fro_sq = M.float().pow(2).sum(); self._ar(M_fro_sq)
            M_fro = M_fro_sq.sqrt().item() + 1e-12

            resid_sq = (B - approx).float().pow(2).sum(); self._ar(resid_sq)
            q_t = (resid_sq.sqrt().item()) / (M_fro + 1e-12)

            # subspace drift
            if tr:
                Q_proj = Qnew @ Qnew.T  # (ml, ml) sharded
            else:
                Q_proj = Qnew @ Qnew.T  # (n, n) replicated
            if st["Q_prev_proj"] is not None and Q_proj.shape == st["Q_prev_proj"].shape:
                drift_sq = (Q_proj - st["Q_prev_proj"]).float().pow(2).sum()
                if tr: self._ar(drift_sq)
                d_t = drift_sq.sqrt().item()
            else:
                d_t = 0.0
            st["Q_prev_proj"] = Q_proj.clone()

            # EF burden
            ef_resid = B - (1.0 - mu) * approx - M  # what EF leaves behind
            ef_sq = ef_resid.float().pow(2).sum(); self._ar(ef_sq)
            e_t = ef_sq.sqrt().item() / (M_fro + 1e-12)

            z_t = w_q * q_t + w_d * d_t + w_e * e_t

            # -- extra power iteration if quality is bad --
            if do_extra and q_t > tau_extra:
                Qbuf_slice = Qbuf[:Qnew.shape[0], :r]
                Qbuf_slice.copy_(Qnew)
                D, approx, Qnew, R = self._power_step(B, Qbuf, r, tr, ml, n, dt)
                st["_extra_step"] = True
            else:
                st["_extra_step"] = False

            # -- quality gate --
            if do_gate:
                old_z = st["z_ema"]
                st["z_ema"] = g_beta * old_z + (1 - g_beta) * z_t
                dev = z_t - st["z_ema"]
                g_raw = math.exp(-g_rho * dev)
                g_t = max(1.0 / g_gamma, min(g_gamma, g_raw))
            else:
                g_t = 1.0

            # -- NaN guard --
            if torch.isnan(D).any() or torch.isinf(D).any():
                D = torch.zeros_like(D)
                g_t = 0.0

            # -- update Qbuf --
            if tr:
                Qbuf[:ml, :r].copy_(Qnew)
            else:
                Qbuf[:n, :r].copy_(Qnew)

            # -- error feedback (standard Dion) --
            M.copy_(B - (1.0 - mu) * approx)

            # -- weight update: plain Dion geometry, gate only modulates magnitude --
            if wd: pl.mul_(1 - lr * wd)  # weight decay separate from gate
            pl.add_(D, alpha=-(lr * ss * g_t))

            # -- quality-triggered rank boost --
            if do_boost:
                if q_t > tau_rank:
                    st["bad_streak"] += 1
                    if st["bad_streak"] >= boost_k:
                        r_new = min(r * 2, rc, Qbuf.shape[1])
                        if r_new > r:
                            d0 = Qbuf.shape[0]
                            Qbuf[:, r:r_new].copy_(F.normalize(
                                torch.randn(d0, r_new - r, device=pl.device, dtype=dt), dim=0))
                        st["r"] = r_new
                        st["bad_streak"] = 0
                else:
                    st["bad_streak"] = 0
                    # fall back to base rank if quality is good
                    if r > r_base and q_t < tau_rank * 0.5:
                        st["r"] = r_base

            # -- diagnostics --
            st["_q_t"] = q_t
            st["_d_t"] = d_t
            st["_e_t"] = e_t
            st["_z_t"] = z_t
            st["_g_t"] = g_t
            st["_D_fro"] = D.float().norm().item()

    # -- diagnostics --
    def get_diagnostics(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": continue
            for p in g["params"]:
                if not self._is_mat(p): continue
                st = self.state[p]
                pf = f"p{i}"
                for k in ["r", "_q_t", "_d_t", "_e_t", "_z_t", "_g_t", "_D_fro",
                           "z_ema", "_extra_step", "bad_streak"]:
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

    def state_dict(self):
        sd = super().state_dict()
        if self._ws() > 1: sd["_qcdion_ws"] = self._ws()
        return sd

    def load_state_dict(self, sd):
        saved = sd.pop("_qcdion_ws", None)
        if saved is not None and saved != self._ws():
            raise RuntimeError(f"Checkpoint ws={saved} != current {self._ws()}")
        super().load_state_dict(sd)
