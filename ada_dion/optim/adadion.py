"""
AdaDion – Adaptive-rank Dion with Anchor regularization (H2: ANCHOR-MBP).

Base: Dion-style low-rank power iteration with Cholesky QR, error feedback,
adaptive rank via effective-rank EMA + quality control.

Anchor (H2 from Naganuma 2025): after a periodic "global" orthonormalization
step at time t_g, cache the resulting basis U*.  Between global steps the
local basis is penalised for drifting from U*:

    min  <G_k, U_k S_k V_k^T>  +  mu/2 ||S_k||^2
         + alpha/2 ||P_{U_k} - P_{U*}^{(k)}||^2_F        (Eq. 7)

In practice we implement the penalty as a soft pull: after each local power
iteration produces Q_new, we mix toward the anchor:

    Q_used = (1 - alpha) * Q_new  +  alpha * Q_anchor
    Q_used = col_normalize(Q_used)

Q_anchor is refreshed (overwritten by Q_new) every `anchor_period` steps,
simulating the periodic global step.  alpha controls regularization strength.
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


def _col_normalize(V: Tensor, eps: float = 1e-12) -> Tensor:
    norms = torch.linalg.vector_norm(V.float(), dim=0).clamp(min=eps)
    return V / norms.to(V.dtype)


def _col_normalize_distributed(V: Tensor, pg, eps: float = 1e-12) -> Tensor:
    sq = torch.linalg.vector_norm(V.float(), dim=0).pow(2)
    if pg is not None:
        dist.all_reduce(sq, group=pg)
    elif dist.is_initialized():
        dist.all_reduce(sq)
    return V / sq.sqrt().clamp(min=eps).to(V.dtype)


def _quantize_int(x: float, q: int) -> int:
    if q <= 1:
        return int(round(x))
    return int(round(x / q) * q)


class AdaDion(Optimizer):
    """Dion + adaptive rank + anchor regularization (H2: ANCHOR-MBP)."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        mu: float = 0.95,
        init_rank: int = 64,
        # adaptive rank
        adaptive_rank: bool = True,
        erank_ema_beta: float = 0.9,
        rho: float = 1.5,
        rank_min: int = 16,
        rank_max: int = 256,
        rank_quantize: int = 8,
        rank_warmup_steps: int = 10,
        warmup_rank: int = 128,
        rank_step_up: int = 16,
        rank_step_down: int = 8,
        # quality control
        use_quality_control: bool = True,
        aerr_ema_beta: float = 0.95,
        aerr_target: float = 0.08,
        aerr_up_margin: float = 0.15,
        aerr_down_margin: float = 0.15,
        # anchor (H2)
        use_anchor: bool = True,
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
            adaptive_rank=adaptive_rank, erank_ema_beta=erank_ema_beta,
            rho=rho, rank_min=rank_min, rank_max=rank_max,
            rank_quantize=rank_quantize, rank_warmup_steps=rank_warmup_steps,
            warmup_rank=warmup_rank, rank_step_up=rank_step_up,
            rank_step_down=rank_step_down,
            use_quality_control=use_quality_control,
            aerr_ema_beta=aerr_ema_beta, aerr_target=aerr_target,
            aerr_up_margin=aerr_up_margin, aerr_down_margin=aerr_down_margin,
            use_anchor=use_anchor, anchor_alpha=anchor_alpha,
            anchor_period=anchor_period,
            qbuf_max_cols=qbuf_max_cols, weight_decay=weight_decay,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)
        self._device_mesh = device_mesh
        self._pg = device_mesh.get_group() if device_mesh is not None else None

    # -- helpers ----------------------------------------------------------

    def _is_matrix(self, p):
        return p.ndim == 2 and p.shape[0] >= 2 and p.shape[1] >= 2

    def _local(self, t):
        return t._local_tensor if isinstance(t, DTensor) else t

    def _full_shape(self, p):
        return tuple(p.shape)

    def _ws(self):
        if self._pg is not None:
            return dist.get_world_size(self._pg)
        return dist.get_world_size() if dist.is_initialized() else 1

    def _ar(self, t):
        if self._pg is not None:
            dist.all_reduce(t, group=self._pg)
        elif self._device_mesh is None and dist.is_initialized():
            dist.all_reduce(t)

    # -- step dispatch ----------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            if g.get("algorithm") == "adamw":
                self._adamw(g)
            else:
                self._adadion(g)
        return loss

    # -- adamw for scalar params ------------------------------------------

    def _adamw(self, g):
        betas = g.get("betas", g.get("adamw_betas", (0.9, 0.95)))
        eps = g.get("eps", g.get("adamw_eps", 1e-8))
        lr, wd = g.get("lr", self.defaults["lr"]), g.get("weight_decay", 0.0)
        for p in g["params"]:
            if p.grad is None:
                continue
            s = self.state[p]
            if len(s) == 0:
                s["step"] = 0
                s["m1"] = torch.zeros_like(p)
                s["m2"] = torch.zeros_like(p)
            s["step"] += 1
            t = s["step"]
            if wd:
                p.data.mul_(1 - lr * wd)
            s["m1"].mul_(betas[0]).add_(p.grad, alpha=1 - betas[0])
            s["m2"].mul_(betas[1]).addcmul_(p.grad, p.grad, value=1 - betas[1])
            bc1, bc2 = 1 - betas[0] ** t, 1 - betas[1] ** t
            p.data.addcdiv_(s["m1"], (s["m2"].sqrt() / math.sqrt(bc2)).add_(eps),
                            value=-lr / bc1)

    # -- main adadion update ----------------------------------------------

    def _adadion(self, g):
        lr = g["lr"]
        mu = g["mu"]
        init_rank = g["init_rank"]
        adaptive = g["adaptive_rank"]
        erank_beta = g["erank_ema_beta"]
        rho = g["rho"]
        rmin, rmax = g["rank_min"], g["rank_max"]
        rq = g["rank_quantize"]
        rwarm_steps, rwarm_rank = g["rank_warmup_steps"], g["warmup_rank"]
        su, sd = g["rank_step_up"], g["rank_step_down"]
        use_qc = g["use_quality_control"]
        aerr_beta = g["aerr_ema_beta"]
        aerr_tgt = g["aerr_target"]
        aerr_up, aerr_dn = g["aerr_up_margin"], g["aerr_down_margin"]
        use_anc = g["use_anchor"]
        anc_alpha = g["anchor_alpha"]
        anc_period = g["anchor_period"]
        qmax = g["qbuf_max_cols"]
        wd = g["weight_decay"]

        for p in g["params"]:
            if p.grad is None:
                continue
            grad = self._local(p.grad)
            pl = self._local(p)
            fs = self._full_shape(p)

            if not self._is_matrix(p):
                if wd:
                    pl.mul_(1 - lr * wd)
                pl.add_(grad, alpha=-lr)
                continue

            st = self.state[p]
            mf, n = fs
            ml = pl.shape[0]
            dmin, dmax = min(mf, n), max(mf, n)
            ss = math.sqrt(dmin / dmax)
            rc = min(dmin, rmax)
            if rc <= 0:
                continue

            # -- init --
            if len(st) == 0:
                tr = mf > n
                bc = min(rc, qmax if adaptive else init_rank)
                ir = min(init_rank, rc, bc)
                ws = self._ws()
                if ws > 1 and isinstance(p, DTensor):
                    assert mf % ws == 0 and ml == mf // ws
                gen = torch.Generator(device=pl.device)
                gen.manual_seed(0x4469_6F6E + n * 1000 + bc)
                dim0 = ml if tr else n
                Qbuf = F.normalize(torch.randn(dim0, bc, device=pl.device,
                                               dtype=pl.dtype, generator=gen), dim=0)
                re = min(rmin, rc, bc)
                st["M"] = torch.zeros_like(pl)
                st["Qbuf"] = Qbuf
                st["r"] = int(max(1, min(max(re, ir), rc, bc)))
                st["tr"] = tr
                st["erank_ema"] = None
                st["aerr_ema"] = None
                st["Q_anchor"] = None   # cached basis from last "global" step
                st["step"] = 0

            st["step"] += 1
            M = st["M"]
            Qbuf = st["Qbuf"]
            r = st["r"]
            tr = st["tr"]
            dt = M.dtype

            re = int(max(1, min(rmin, rc, Qbuf.shape[1])))
            r = int(min(max(re, r if adaptive else init_rank), Qbuf.shape[1], rc))

            B = M + grad

            # -- power iteration + Cholesky QR --
            if not tr:
                Qp = Qbuf[:n, :r]
                Ph = B @ Qp
                Pf = Ph.float()
                gram = Pf.T @ Pf
                self._ar(gram)
                gram.diagonal().add_(1e-6)
                try:
                    L = torch.linalg.cholesky(gram)
                    Po = torch.linalg.solve_triangular(L, Pf.T, upper=False).T
                except torch.linalg.LinAlgError:
                    Po, _ = torch.linalg.qr(Pf)
                Pd = Po.to(dt)
                R = (B.float().T @ Po).to(dt)
                self._ar(R)
                approx = Pd @ R.T
                Qnew = _col_normalize(R)

                # -- H2 anchor: pull Qnew toward cached global basis --
                if use_anc and st["Q_anchor"] is not None and anc_alpha > 0:
                    Qa = st["Q_anchor"]
                    ar = min(r, Qa.shape[1])
                    Qnew[:, :ar] = (1 - anc_alpha) * Qnew[:, :ar] + anc_alpha * Qa[:n, :ar]
                    Qnew = _col_normalize(Qnew)

                update = Pd @ Qnew.T
                Qbuf[:n, :r].copy_(Qnew)
            else:
                Qp = Qbuf[:ml, :r]
                Phf = (B.float().T @ Qp.float())
                self._ar(Phf)
                Po, _ = torch.linalg.qr(Phf, mode="reduced")
                Pd = Po.to(dt)
                R = B @ Pd
                approx = R @ Pd.T
                Qnew = _col_normalize_distributed(R, self._pg)

                # -- H2 anchor --
                if use_anc and st["Q_anchor"] is not None and anc_alpha > 0:
                    Qa = st["Q_anchor"]
                    ar = min(r, Qa.shape[1])
                    Qnew[:, :ar] = (1 - anc_alpha) * Qnew[:, :ar] + anc_alpha * Qa[:ml, :ar]
                    Qnew = _col_normalize_distributed(Qnew, self._pg)

                update = Qnew @ Pd.T
                Qbuf[:ml, :r].copy_(Qnew)

            # -- NaN guard --
            if torch.isnan(update).any() or torch.isinf(update).any():
                update = torch.zeros_like(update)

            # -- error feedback --
            M.copy_(B - (1.0 - mu) * approx)

            # -- weight update --
            if wd:
                pl.mul_(1 - lr * wd)
            pl.add_(update, alpha=-(lr * ss))

            # -- refresh anchor every anchor_period steps (simulates global step) --
            if use_anc and (st["step"] % anc_period == 0 or st["Q_anchor"] is None):
                st["Q_anchor"] = Qbuf.clone()

            # -- adaptive rank --
            if adaptive:
                if st["step"] <= rwarm_steps:
                    des = int(min(rwarm_rank, rc, Qbuf.shape[1]))
                    des = max(re, des)
                    if des > r:
                        d0 = Qbuf.shape[0]
                        Qbuf[:, r:des].copy_(F.normalize(
                            torch.randn(d0, des - r, device=pl.device, dtype=dt), dim=0))
                    st["r"] = des
                else:
                    sig = torch.linalg.vector_norm(R.float(), dim=0)
                    if tr:
                        sq = sig.pow(2)
                        self._ar(sq)
                        sig = sq.sqrt()
                    sig = sig + 1e-12
                    pr = sig / (sig.sum() + 1e-12)
                    ent = -(pr * torch.log(pr + 1e-12)).sum()
                    er = torch.exp(ent).item()
                    old = st["erank_ema"]
                    st["erank_ema"] = er if old is None else erank_beta * old + (1 - erank_beta) * er
                    des = int(round(rho * st["erank_ema"]))

                    if use_qc:
                        nsq = (B - approx).float().pow(2).sum()
                        dsq = B.float().pow(2).sum()
                        self._ar(nsq)
                        self._ar(dsq)
                        ae = (nsq.sqrt() / (dsq.sqrt() + 1e-12)).item()
                        oa = st["aerr_ema"]
                        st["aerr_ema"] = ae if oa is None else aerr_beta * oa + (1 - aerr_beta) * ae
                        if ae > max(aerr_tgt, st["aerr_ema"] * (1 + aerr_up)):
                            des = max(des, r + su)
                        if ae < min(0.5 * aerr_tgt, st["aerr_ema"] * (1 - aerr_dn)):
                            des = min(des, r - sd)

                    des = _quantize_int(min(max(re, des), rc, Qbuf.shape[1]), rq)
                    des = min(max(re, des), rc, Qbuf.shape[1])
                    rn = min(des, r + su) if des > r else (max(des, r - sd) if des < r else r)
                    rn = int(min(max(re, rn), rc, Qbuf.shape[1]))
                    if rn > r:
                        d0 = Qbuf.shape[0]
                        Qbuf[:, r:rn].copy_(F.normalize(
                            torch.randn(d0, rn - r, device=pl.device, dtype=dt), dim=0))
                    st["r"] = rn

    # -- checkpoint -------------------------------------------------------

    def state_dict(self):
        sd = super().state_dict()
        ws = self._ws()
        if ws > 1:
            sd["_adadion_ws"] = ws
        return sd

    def load_state_dict(self, sd):
        saved = sd.pop("_adadion_ws", None)
        if saved is not None and saved != self._ws():
            raise RuntimeError(
                f"Checkpoint world_size={saved} != current {self._ws()}")
        super().load_state_dict(sd)

    # -- diagnostics ------------------------------------------------------

    def get_rank(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw":
                continue
            for p in g["params"]:
                if self._is_matrix(p) and "r" in self.state[p]:
                    out[f"param_{i}"] = self.state[p]["r"]
                i += 1
        return out

    def get_effective_rank(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw":
                continue
            for p in g["params"]:
                if self._is_matrix(p):
                    v = self.state[p].get("erank_ema")
                    if v is not None:
                        out[f"param_{i}"] = v
                i += 1
        return out

    def get_aerr(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw":
                continue
            for p in g["params"]:
                if self._is_matrix(p):
                    v = self.state[p].get("aerr_ema")
                    if v is not None:
                        out[f"param_{i}"] = v
                i += 1
        return out

    def get_anchor_drift(self):
        """||Q_current - Q_anchor||_F per matrix param (0 = just refreshed)."""
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw":
                continue
            for p in g["params"]:
                if not self._is_matrix(p):
                    continue
                st = self.state[p]
                Qa = st.get("Q_anchor")
                if Qa is not None and "Qbuf" in st:
                    r = st.get("r", 0)
                    d0 = min(st["Qbuf"].shape[0], Qa.shape[0])
                    rc = min(r, Qa.shape[1])
                    d = (st["Qbuf"][:d0, :rc] - Qa[:d0, :rc]).norm().item()
                    out[f"param_{i}"] = d if math.isfinite(d) else 0.0
                i += 1
        return out
