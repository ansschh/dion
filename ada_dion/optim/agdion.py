"""
AGDion — Anchor-Gap Dion.

Plain Dion with two proposal mechanisms (Naganuma 2025):

  H2 (ANCHOR): After a global refresh at step t_g, cache basis U*.
      Between refreshes, pull local basis toward U*:
        Q_used = (1-alpha) * Q_new + alpha * Q_anchor
      Communication-free.

  H3 (GAP): Penalize tail energy outside the rank-r subspace.
      After computing Q_new, project momentum residual:
        tail = ||(I - Q Q^T) M||_F^2 / ||M||_F^2
      When tail is high, take an extra power step to improve
      the rank-r approximation. This keeps momentum energy
      concentrated in the chosen subspace.

  Event-triggered refresh: Instead of fixed-period global steps,
      refresh Q (simulate global orthonormalization) only when
      drift or tail exceeds threshold. Cheap steps otherwise.

      drift_t = ||Q_t Q_t^T - Q_anchor Q_anchor^T||_F
      tail_t  = ||(I - Q_t Q_t^T) M_t||_F / ||M_t||_F
      if drift_t > tau_drift or tail_t > tau_tail: refresh

Preserves plain Dion update geometry. FSDP2/DTensor compatible.
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


class AGDion(Optimizer):
    """Anchor-Gap Dion: Dion + H2 anchor + H3 gap + event-triggered refresh."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        mu: float = 0.95,
        init_rank: int = 64,
        rank_min: int = 16,
        rank_max: int = 256,
        # H2: anchor
        use_anchor: bool = True,
        anchor_alpha: float = 0.1,
        # H3: gap (tail control)
        use_gap: bool = True,
        tau_tail_extra: float = 0.3,
        # event-triggered refresh
        use_event_refresh: bool = True,
        tau_drift: float = 2.0,
        tau_tail_refresh: float = 0.4,
        max_period: int = 200,
        min_period: int = 10,
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
            use_anchor=use_anchor, anchor_alpha=anchor_alpha,
            use_gap=use_gap, tau_tail_extra=tau_tail_extra,
            use_event_refresh=use_event_refresh,
            tau_drift=tau_drift, tau_tail_refresh=tau_tail_refresh,
            max_period=max_period, min_period=min_period,
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
            else: self._agdion(g)
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

    def _agdion(self, g):
        lr = g["lr"]; mu = g["mu"]
        init_rank = g["init_rank"]; rmin = g["rank_min"]; rmax = g["rank_max"]
        do_anc = g["use_anchor"]; anc_alpha = g["anchor_alpha"]
        do_gap = g["use_gap"]; tau_tail_ex = g["tau_tail_extra"]
        do_evref = g["use_event_refresh"]
        tau_drift = g["tau_drift"]; tau_tail_ref = g["tau_tail_refresh"]
        max_per = g["max_period"]; min_per = g["min_period"]
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
                gen.manual_seed(0x4147_4469 + n * 1000 + bc)
                d0 = ml if tr else n
                Qbuf = F.normalize(torch.randn(d0, bc, device=pl.device, dtype=pl.dtype, generator=gen), dim=0)
                re = min(rmin, rc, bc)
                r_init = int(max(1, min(max(re, ir), rc, bc)))
                st.update({
                    "M": torch.zeros_like(pl), "Qbuf": Qbuf,
                    "r": r_init, "tr": tr, "step": 0,
                    "Q_anchor": None,
                    "last_refresh": 0,
                })

            st["step"] += 1
            M = st["M"]; Qbuf = st["Qbuf"]; r = st["r"]; tr = st["tr"]; dt = M.dtype

            # momentum
            M.mul_(mu).add_(grad)
            B = M + grad

            # -- standard power step --
            D, approx, Qnew, R = self._power_step(B, Qbuf, r, tr, ml, n, dt)

            # -- H3 GAP: measure tail energy --
            # tail = ||(I - QQ^T)M||_F / ||M||_F
            if tr:
                proj_M = Qnew @ (Qnew.T @ M)  # (ml, n) via (ml, r) @ (r, n)
            else:
                proj_M = M @ (Qnew @ Qnew.T)  # (ml, n) via (ml, n) @ (n, n) -- expensive for large n
                # cheaper: proj_M = (M @ Qnew) @ Qnew.T
                proj_M = (M @ Qnew) @ Qnew.T
            tail_sq = (M - proj_M).float().pow(2).sum()
            M_sq = M.float().pow(2).sum()
            self._ar(tail_sq); self._ar(M_sq)
            tail_t = (tail_sq.sqrt() / (M_sq.sqrt() + 1e-12)).item()

            # if tail is high, do an extra power step (H3 intervention)
            did_extra = False
            if do_gap and tail_t > tau_tail_ex:
                if tr:
                    Qbuf[:ml, :r].copy_(Qnew)
                else:
                    Qbuf[:n, :r].copy_(Qnew)
                D, approx, Qnew, R = self._power_step(B, Qbuf, r, tr, ml, n, dt)
                did_extra = True

            # -- H2 ANCHOR: pull Q toward cached global basis --
            drift_t = 0.0
            if do_anc and st["Q_anchor"] is not None and anc_alpha > 0:
                Qa = st["Q_anchor"]
                d0 = Qnew.shape[0]
                ar = min(r, Qa.shape[1])
                # measure drift
                drift_sq = (Qnew[:, :ar] - Qa[:d0, :ar]).float().pow(2).sum()
                if tr: self._ar(drift_sq)
                drift_t = drift_sq.sqrt().item()
                # anchor pull
                Qnew[:, :ar] = (1 - anc_alpha) * Qnew[:, :ar] + anc_alpha * Qa[:d0, :ar]
                if not tr:
                    Qnew = _col_norm(Qnew)
                else:
                    Qnew = _col_norm_dist(Qnew, self._pg)
                # recompute D with anchored Q
                if not tr:
                    # need to recompute with anchored Qnew
                    # D = P_orth @ Qnew^T but P_orth is from the power step
                    # simplest: just recompute the update direction
                    D = (B @ Qnew) @ Qnew.T  # low-rank projection
                    # normalize like Dion: D = P_orth @ col_norm(R)^T
                    # actually reuse the existing Pd from power step
                    pass  # keep D from power step, anchor only affects Qbuf warm-start
                # The anchor primarily affects the NEXT step via Qbuf warm-start

            # -- NaN guard --
            if torch.isnan(D).any() or torch.isinf(D).any():
                D = torch.zeros_like(D)

            # -- update Qbuf --
            if tr:
                Qbuf[:ml, :r].copy_(Qnew)
            else:
                Qbuf[:n, :r].copy_(Qnew)

            # -- error feedback --
            M.copy_(B - (1.0 - mu) * approx)

            # -- weight update (plain Dion geometry, no scaling) --
            if wd: pl.mul_(1 - lr * wd)
            pl.add_(D, alpha=-(lr * ss))

            # -- event-triggered refresh --
            steps_since = st["step"] - st["last_refresh"]
            do_refresh = False

            if do_evref:
                if drift_t > tau_drift and steps_since >= min_per:
                    do_refresh = True
                if tail_t > tau_tail_ref and steps_since >= min_per:
                    do_refresh = True
            # always refresh at max_period
            if steps_since >= max_per:
                do_refresh = True

            if do_refresh:
                st["Q_anchor"] = Qbuf.clone()
                st["last_refresh"] = st["step"]
            elif st["Q_anchor"] is None:
                st["Q_anchor"] = Qbuf.clone()
                st["last_refresh"] = st["step"]

            # -- diagnostics --
            st["_drift"] = drift_t
            st["_tail"] = tail_t
            st["_did_extra"] = did_extra
            st["_did_refresh"] = do_refresh
            st["_steps_since_refresh"] = steps_since

    # -- diagnostics --
    def get_diagnostics(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": continue
            for p in g["params"]:
                if not self._is_mat(p): continue
                st = self.state[p]
                pf = f"p{i}"
                for k in ["r", "_drift", "_tail", "_did_extra", "_did_refresh",
                           "_steps_since_refresh"]:
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

    def get_drift(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": continue
            for p in g["params"]:
                if self._is_mat(p) and "_drift" in self.state[p]:
                    out[f"p{i}"] = self.state[p]["_drift"]
                i += 1
        return out

    def get_tail(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": continue
            for p in g["params"]:
                if self._is_mat(p) and "_tail" in self.state[p]:
                    out[f"p{i}"] = self.state[p]["_tail"]
                i += 1
        return out

    def state_dict(self):
        sd = super().state_dict()
        if self._ws() > 1: sd["_agdion_ws"] = self._ws()
        return sd

    def load_state_dict(self, sd):
        saved = sd.pop("_agdion_ws", None)
        if saved is not None and saved != self._ws():
            raise RuntimeError(f"Checkpoint ws={saved} != current {self._ws()}")
        super().load_state_dict(sd)
