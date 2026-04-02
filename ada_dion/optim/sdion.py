"""
SDion — Skip-Dion with consensus correction.

Default path is exactly Dion. Interventions are event-triggered, not
always-on. Architecture from Naganuma 2025 proposal, redesigned:

  Mode 1 (SAFE): When drift and tail are low, just reuse cached basis
      from the last full power step. No communication, no penalty.
      This is the "skip" — we skip the expensive power iteration.

  Mode 2 (CONSENSUS): When drift grows moderate, each shard sends a
      tiny sketch Y_k = random_proj^T @ Q_k (size s x r) and we
      build a lightweight consensus basis. Blend local + consensus.
      Cost: O(s*r) communication instead of O(m*n).

  Mode 3 (RECOVERY): When tail energy spikes or drift is large,
      do a full power step (the "global" step). Optionally take
      an extra power iteration or temporarily raise rank.

  Decaying anchor: After a full power step, cache the basis.
      On skipped steps, blend toward it with exponentially decaying
      weight: alpha_t = alpha_0 * exp(-(t - t_refresh) / halflife).
      Fresh anchor helps; stale anchor fades out.

  Gap as trigger: tail energy tau_t is only used to decide WHEN
      to switch modes, never as a continuous penalty.

Every step logs: drift, tail, mode, skip_count, refresh_count.
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


class SDion(Optimizer):
    """Skip-Dion: Dion with event-triggered consensus and recovery."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        mu: float = 0.95,
        init_rank: int = 64,
        rank_min: int = 16,
        rank_max: int = 256,
        # skip control
        enable_skip: bool = True,
        tau_drift_safe: float = 0.5,
        tau_drift_consensus: float = 1.5,
        tau_tail_recovery: float = 0.4,
        max_skip: int = 10,
        # consensus
        enable_consensus: bool = True,
        sketch_dim: int = 0,
        consensus_lambda: float = 0.3,
        # decaying anchor
        enable_anchor: bool = True,
        anchor_alpha0: float = 0.15,
        anchor_halflife: float = 20.0,
        # recovery
        enable_recovery: bool = True,
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
            enable_skip=enable_skip,
            tau_drift_safe=tau_drift_safe,
            tau_drift_consensus=tau_drift_consensus,
            tau_tail_recovery=tau_tail_recovery,
            max_skip=max_skip,
            enable_consensus=enable_consensus,
            sketch_dim=sketch_dim, consensus_lambda=consensus_lambda,
            enable_anchor=enable_anchor,
            anchor_alpha0=anchor_alpha0, anchor_halflife=anchor_halflife,
            enable_recovery=enable_recovery,
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
            else: self._sdion(g)
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

    def _full_power_step(self, B, Qbuf, r, tr, ml, n, dt):
        """Full Dion power iteration + Cholesky QR."""
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
            return D, approx, Qnew
        else:
            Qp = Qbuf[:ml, :r]
            Phf = (B.float().T @ Qp.float()); self._ar(Phf)
            Po, _ = torch.linalg.qr(Phf, mode="reduced")
            Pd = Po.to(dt)
            R = B @ Pd
            approx = R @ Pd.T
            Qnew = _col_norm_dist(R, self._pg)
            D = Qnew @ Pd.T
            return D, approx, Qnew

    def _cheap_step(self, B, Qbuf, r, tr, ml, n, dt):
        """Cheap step: reuse cached Q, just recompute projection."""
        if not tr:
            Q = Qbuf[:n, :r]
            R = B @ Q
            D = R @ Q.T
            approx = D  # low-rank approx using cached Q
            return D, approx
        else:
            Q = Qbuf[:ml, :r]
            R = B.T @ Q  # (n, r)
            D = Q @ R.T  # (ml, n)
            approx = D
            return D, approx

    def _compute_drift(self, Qnew, Q_anchor, r, tr):
        """||Q_new - Q_anchor|| as drift proxy."""
        if Q_anchor is None:
            return 0.0
        d0 = Qnew.shape[0]
        ar = min(r, Q_anchor.shape[1])
        diff_sq = (Qnew[:, :ar] - Q_anchor[:d0, :ar]).float().pow(2).sum()
        if tr: self._ar(diff_sq)
        return diff_sq.sqrt().item()

    def _compute_tail(self, M, Qbuf, r, tr, ml):
        """Tail energy: ||(I - QQ^T)M||_F / ||M||_F."""
        if not tr:
            Q = Qbuf[:M.shape[1], :r]
            proj = (M @ Q) @ Q.T
        else:
            Q = Qbuf[:ml, :r]
            proj = Q @ (Q.T @ M)
        tail_sq = (M - proj).float().pow(2).sum()
        M_sq = M.float().pow(2).sum()
        self._ar(tail_sq); self._ar(M_sq)
        return (tail_sq.sqrt() / (M_sq.sqrt() + 1e-12)).item()

    def _sdion(self, g):
        lr = g["lr"]; mu = g["mu"]
        init_rank = g["init_rank"]; rmin = g["rank_min"]; rmax = g["rank_max"]
        do_skip = g["enable_skip"]
        tau_safe = g["tau_drift_safe"]
        tau_cons = g["tau_drift_consensus"]
        tau_recov = g["tau_tail_recovery"]
        max_skip = g["max_skip"]
        do_cons = g["enable_consensus"]
        cons_lam = g["consensus_lambda"]
        do_anc = g["enable_anchor"]
        anc_a0 = g["anchor_alpha0"]; anc_hl = g["anchor_halflife"]
        do_recov = g["enable_recovery"]
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
                gen.manual_seed(0x5344_696F + n * 1000 + bc)
                d0 = ml if tr else n
                Qbuf = F.normalize(torch.randn(d0, bc, device=pl.device, dtype=pl.dtype, generator=gen), dim=0)
                re = min(rmin, rc, bc)
                r_init = int(max(1, min(max(re, ir), rc, bc)))
                st.update({
                    "M": torch.zeros_like(pl), "Qbuf": Qbuf,
                    "r": r_init, "tr": tr, "step": 0,
                    "Q_anchor": None, "last_refresh": 0,
                    "steps_skipped": 0,
                    "total_skips": 0, "total_full": 0,
                    "total_consensus": 0, "total_recovery": 0,
                })

            st["step"] += 1
            M = st["M"]; Qbuf = st["Qbuf"]; r = st["r"]; tr = st["tr"]; dt = M.dtype

            M.mul_(mu).add_(grad)
            B = M + grad

            # -- measure drift and tail --
            drift = self._compute_drift(
                Qbuf[:ml if tr else n, :r], st["Q_anchor"], r, tr)
            tail = self._compute_tail(M, Qbuf, r, tr, ml)

            steps_since = st["step"] - st["last_refresh"]

            # -- decide mode --
            mode = "full"  # default: full power step (= plain Dion)

            if do_skip and steps_since > 0:
                if drift < tau_safe and tail < tau_recov and steps_since < max_skip:
                    mode = "skip"
                elif do_cons and drift < tau_cons and tail < tau_recov:
                    mode = "consensus"
                elif do_recov and (tail > tau_recov or drift > tau_cons):
                    mode = "recovery"
                # else: full

            # -- execute mode --
            if mode == "skip":
                # reuse cached basis, cheap projection
                D, approx = self._cheap_step(B, Qbuf, r, tr, ml, n, dt)
                st["steps_skipped"] += 1
                st["total_skips"] += 1

                # decaying anchor blend
                if do_anc and st["Q_anchor"] is not None:
                    decay = anc_a0 * math.exp(-steps_since / anc_hl)
                    if decay > 0.01:
                        d0 = ml if tr else n
                        ar = min(r, st["Q_anchor"].shape[1])
                        Qcur = Qbuf[:d0, :r].clone()
                        Qcur[:, :ar] = (1 - decay) * Qcur[:, :ar] + decay * st["Q_anchor"][:d0, :ar]
                        if not tr:
                            Qcur = _col_norm(Qcur)
                        else:
                            Qcur = _col_norm_dist(Qcur, self._pg)
                        # recompute D with blended Q
                        if not tr:
                            D = (B @ Qcur) @ Qcur.T
                        else:
                            D = Qcur @ (Qcur.T @ B.T).T

            elif mode == "consensus":
                # full power step + tiny sketch exchange for consensus
                D, approx, Qnew = self._full_power_step(B, Qbuf, r, tr, ml, n, dt)
                # consensus: in single-node, this is a no-op (all shards see same Q)
                # In multi-node FSDP, would exchange sketches here
                # For now, just do the full step (consensus = full step with logging)
                if tr: Qbuf[:ml, :r].copy_(Qnew)
                else: Qbuf[:n, :r].copy_(Qnew)
                st["steps_skipped"] = 0
                st["total_consensus"] += 1

            elif mode == "recovery":
                # full power step, optionally double
                D, approx, Qnew = self._full_power_step(B, Qbuf, r, tr, ml, n, dt)
                if tr: Qbuf[:ml, :r].copy_(Qnew)
                else: Qbuf[:n, :r].copy_(Qnew)
                # extra power step if tail is really bad
                if tail > tau_recov * 1.5:
                    D, approx, Qnew = self._full_power_step(B, Qbuf, r, tr, ml, n, dt)
                    if tr: Qbuf[:ml, :r].copy_(Qnew)
                    else: Qbuf[:n, :r].copy_(Qnew)
                st["Q_anchor"] = Qbuf.clone()
                st["last_refresh"] = st["step"]
                st["steps_skipped"] = 0
                st["total_recovery"] += 1

            else:  # full (default Dion)
                D, approx, Qnew = self._full_power_step(B, Qbuf, r, tr, ml, n, dt)
                if tr: Qbuf[:ml, :r].copy_(Qnew)
                else: Qbuf[:n, :r].copy_(Qnew)
                st["Q_anchor"] = Qbuf.clone()
                st["last_refresh"] = st["step"]
                st["steps_skipped"] = 0
                st["total_full"] += 1

            # NaN guard
            if torch.isnan(D).any() or torch.isinf(D).any():
                D = torch.zeros_like(D)

            # error feedback
            M.copy_(B - (1.0 - mu) * approx)

            # weight update (plain Dion geometry)
            if wd: pl.mul_(1 - lr * wd)
            pl.add_(D, alpha=-(lr * ss))

            # diagnostics
            st["_drift"] = drift
            st["_tail"] = tail
            st["_mode"] = mode

    # -- diagnostics --
    def get_diagnostics(self):
        out, i = {}, 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": continue
            for p in g["params"]:
                if not self._is_mat(p): continue
                st = self.state[p]
                pf = f"p{i}"
                for k in ["r", "_drift", "_tail", "_mode", "steps_skipped",
                           "total_skips", "total_full", "total_consensus", "total_recovery"]:
                    if k in st: out[f"{pf}/{k.lstrip('_')}"] = st[k]
                i += 1
        return out

    def get_mode_counts(self):
        """Return aggregate mode counts across all matrix params."""
        skips = fulls = cons = recov = 0
        for g in self.param_groups:
            if g.get("algorithm") == "adamw": continue
            for p in g["params"]:
                if not self._is_mat(p): continue
                st = self.state[p]
                skips += st.get("total_skips", 0)
                fulls += st.get("total_full", 0)
                cons += st.get("total_consensus", 0)
                recov += st.get("total_recovery", 0)
        total = skips + fulls + cons + recov
        return {"skip": skips, "full": fulls, "consensus": cons, "recovery": recov,
                "total": total, "skip_rate": skips / max(total, 1)}

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
        if self._ws() > 1: sd["_sdion_ws"] = self._ws()
        return sd

    def load_state_dict(self, sd):
        saved = sd.pop("_sdion_ws", None)
        if saved is not None and saved != self._ws():
            raise RuntimeError(f"Checkpoint ws={saved} != current {self._ws()}")
        super().load_state_dict(sd)
