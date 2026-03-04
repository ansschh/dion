"""
AdaDion: Adaptive low-rank optimizer with consensus + anchor regularization.

Extends Dion with three innovations:
  1. Consensus regularization — Gram-based orthonormalization (replaces QR)
  2. Anchor regularization — EMA anchor subspace stabilizes the low-rank basis
  3. Gap monitoring — eigenvalue gap (tau) detects poorly conditioned subspaces

Supports `algorithm="adamw"` param-group routing for scalar parameters
(same pattern as Muon/Dion/Dion2 in the microsoft/dion package).
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


def _mat_inv_sqrt(S: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute S^{-1/2} via eigendecomposition for a symmetric PSD matrix.

    Uses float32 for numerical stability and adds diagonal regularization
    to handle ill-conditioned matrices from bf16 accumulation.
    """
    orig_dtype = S.dtype
    S_f32 = S.float() if S.dtype != torch.float32 else S
    # Regularize: add small diagonal to prevent ill-conditioning
    S_f32 = S_f32 + eps * torch.eye(S_f32.shape[-1], device=S_f32.device)
    try:
        eigvals, eigvecs = torch.linalg.eigh(S_f32)
    except torch._C._LinAlgError:
        # Fallback: if eigh fails, return identity (skip normalization)
        return torch.eye(S.shape[-1], device=S.device, dtype=orig_dtype)
    eigvals = eigvals.clamp(min=eps)
    result = eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-2, -1)
    return result.to(orig_dtype)


def _orthonormalize_gram(Y: Tensor, eps: float = 1e-7) -> Tensor:
    """Orthonormalize columns of Y via Gram-based method: Y @ (Y^T Y)^{-1/2}.

    Falls back to QR if Gram method produces NaN (ill-conditioned input).
    """
    S = Y.T @ Y  # r x r
    S_inv_sqrt = _mat_inv_sqrt(S, eps=eps)
    result = Y @ S_inv_sqrt
    if torch.isnan(result).any() or torch.isinf(result).any():
        # Fallback: QR decomposition (always stable)
        Q, _ = torch.linalg.qr(Y.float())
        return Q.to(Y.dtype)
    return result


class AdaDion(Optimizer):
    """
    AdaDion optimizer: adaptive low-rank with anchor regularization.

    For matrix parameters (2D with both dims >= 2), uses low-rank subspace
    tracking with momentum, Gram-based orthonormalization, anchor mixing,
    and eigenvalue gap monitoring.

    For scalar parameters (1D, embeddings, norms), routes to AdamW
    when `algorithm="adamw"` is set in the param group.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        mu: float = 0.95,
        rank_fraction: float = 0.25,
        anchor_lambda: float = 0.1,
        anchor_rho: float = 0.99,
        tau_hi: float = 0.8,
        tau_lo: float = 0.3,
        weight_decay: float = 0.0,
        refresh_period: int = 100,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            mu=mu,
            rank_fraction=rank_fraction,
            anchor_lambda=anchor_lambda,
            anchor_rho=anchor_rho,
            tau_hi=tau_hi,
            tau_lo=tau_lo,
            weight_decay=weight_decay,
            refresh_period=refresh_period,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        super().__init__(params, defaults)

    def _is_matrix_param(self, p: Tensor) -> bool:
        """Check if parameter should use the low-rank subspace algorithm."""
        return p.ndim == 2 and p.shape[0] >= 2 and p.shape[1] >= 2

    def _get_local_tensor(self, p: Tensor) -> Tensor:
        """Get the local shard for DTensor, or the tensor itself."""
        if hasattr(p, "_local_tensor"):
            return p._local_tensor
        return p

    def _get_full_shape(self, p: Tensor) -> tuple[int, ...]:
        """Get the unsharded shape for DTensor, or the tensor shape."""
        # For DTensor, p.shape gives the global (unsharded) shape
        return tuple(p.shape)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("algorithm") == "adamw":
                self._step_adamw(group)
            else:
                self._step_adadion(group)

        return loss

    def _step_adamw(self, group: dict) -> None:
        """AdamW update for scalar parameters."""
        betas = group.get("betas", group.get("adamw_betas", (0.9, 0.95)))
        eps = group.get("eps", group.get("adamw_eps", 1e-8))
        lr = group.get("lr", self.defaults["lr"])
        wd = group.get("weight_decay", 0.0)

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            step = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            # Decoupled weight decay
            if wd != 0:
                p.data.mul_(1 - lr * wd)

            # Update biased first and second moment estimates
            exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
            exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

            # Bias correction
            bc1 = 1 - betas[0] ** step
            bc2 = 1 - betas[1] ** step
            step_size = lr / bc1

            denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)

    def _step_adadion(self, group: dict) -> None:
        """AdaDion update for matrix parameters."""
        lr = group["lr"]
        mu = group["mu"]
        rank_fraction = group["rank_fraction"]
        anchor_lambda = group["anchor_lambda"]
        anchor_rho = group["anchor_rho"]
        tau_lo = group["tau_lo"]
        weight_decay = group["weight_decay"]
        refresh_period = group["refresh_period"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = self._get_local_tensor(p.grad)
            p_local = self._get_local_tensor(p)
            full_shape = self._get_full_shape(p)

            if not self._is_matrix_param(p):
                # Fallback: simple SGD with momentum for non-matrix params
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p_local)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad)
                if weight_decay != 0:
                    p_local.mul_(1 - lr * weight_decay)
                p_local.add_(buf, alpha=-lr)
                continue

            state = self.state[p]
            m_local, n = p_local.shape[0], full_shape[1]
            r = max(1, int(rank_fraction * min(full_shape[0], n)))

            if len(state) == 0:
                state["M"] = torch.zeros_like(p_local)
                state["Q"] = torch.randn(n, r, device=p_local.device, dtype=p_local.dtype)
                # Orthonormalize initial Q
                state["Q"] = _orthonormalize_gram(state["Q"])
                state["Q_anc"] = None
                state["step"] = 0
                state["rank"] = r
                state["tau"] = 1.0
                state["energy"] = 0.0

            state["step"] += 1
            M = state["M"]
            Q = state["Q"]

            # Step 0: Momentum update
            M.mul_(mu).add_(grad)

            # Step 1: Power iteration (warm-started)
            # Z = M @ Q  (m_local x r)
            Z = M @ Q
            # Y = M^T @ Z  (n x r) — needs all-reduce in distributed
            Y = M.T @ Z
            # In distributed setting, all_reduce Y here
            if hasattr(p, "_spec") and hasattr(p._spec, "mesh"):
                try:
                    torch.distributed.all_reduce(Y)
                except Exception:
                    pass  # Non-distributed or no process group

            # Step 2: Gram-based orthonormalization
            S = Y.T @ Y  # r x r
            Q_new = _orthonormalize_gram(Y)

            # Safety: if Q_new has NaN/Inf, fall back to previous Q
            if torch.isnan(Q_new).any() or torch.isinf(Q_new).any():
                Q_new = Q.clone()

            # Step 3: Right factor
            R = M @ Q_new  # m_local x r
            # In distributed setting, all_reduce R
            if hasattr(p, "_spec") and hasattr(p._spec, "mesh"):
                try:
                    torch.distributed.all_reduce(R)
                except Exception:
                    pass

            # Step 4: Anchor mixing
            Q_bar = Q_new
            if state["Q_anc"] is not None:
                Q_bar = (1 - anchor_lambda) * Q_new + anchor_lambda * state["Q_anc"]
                Q_bar = _orthonormalize_gram(Q_bar)

            # Step 5: Anchor EMA update
            if state["Q_anc"] is None:
                state["Q_anc"] = Q_new.clone()
            else:
                Q_anc_new = anchor_rho * state["Q_anc"] + (1 - anchor_rho) * Q_new
                Q_anc_new = _orthonormalize_gram(Q_anc_new)
                # Only update if orthonormalization succeeded
                if not (torch.isnan(Q_anc_new).any() or torch.isinf(Q_anc_new).any()):
                    state["Q_anc"] = Q_anc_new

            # Step 6: Gap monitoring
            tau = state["tau"]  # default: keep previous value
            try:
                S_f32 = S.float() + 1e-7 * torch.eye(S.shape[0], device=S.device)
                eigvals = torch.linalg.eigvalsh(S_f32)
                eigvals_sorted = eigvals.sort(descending=True).values
                if eigvals_sorted.numel() >= 2:
                    a, b = eigvals_sorted[0], eigvals_sorted[1]
                    tau = 1 - b / (a + 1e-8)
                else:
                    tau = 1.0
                state["tau"] = tau.item() if isinstance(tau, Tensor) else tau

                # Compute energy captured
                total_energy = eigvals.sum().item()
                if total_energy > 1e-12:
                    top_r = eigvals_sorted[:r].sum().item()
                    state["energy"] = top_r / total_energy
                else:
                    state["energy"] = 0.0
            except torch._C._LinAlgError:
                # Ill-conditioned matrix — keep previous tau/energy
                pass

            # Refresh Q if gap is too low or periodic refresh
            if tau < tau_lo or (state["step"] % refresh_period == 0):
                Q_new = torch.randn(n, r, device=p_local.device, dtype=p_local.dtype)
                Q_new = _orthonormalize_gram(Q_new)
                Q_bar = Q_new

            # Step 7: Error feedback + weight update
            # Low-rank update: Update = R @ Q_bar^T  (m_local x n)
            Update = R @ Q_bar.T

            # Normalize via (R^T R)^{-1/2} for orthonormal scaling
            RtR = R.T @ R  # r x r
            # In distributed, all_reduce RtR
            if hasattr(p, "_spec") and hasattr(p._spec, "mesh"):
                try:
                    torch.distributed.all_reduce(RtR)
                except Exception:
                    pass
            RtR_inv_sqrt = _mat_inv_sqrt(RtR)
            U = Update @ Q_bar @ RtR_inv_sqrt @ Q_bar.T  # Normalized

            # Safety: skip update if NaN/Inf (numerical failure)
            if torch.isnan(U).any() or torch.isinf(U).any():
                state["Q"] = Q_bar
                continue

            # Error feedback: remove projected component from momentum
            M.sub_(Update)

            # Weight update
            if weight_decay != 0:
                p_local.mul_(1 - lr * weight_decay)
            p_local.add_(U, alpha=-lr)

            # Store updated Q
            state["Q"] = Q_bar

    # ------------------------------------------------------------------
    # Diagnostic methods
    # ------------------------------------------------------------------

    def get_anchor_drift(self) -> dict[str, float]:
        """Return ||Q - Q_anc||_F per matrix parameter."""
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm") == "adamw":
                continue
            for p in group["params"]:
                if not self._is_matrix_param(p):
                    continue
                state = self.state[p]
                if "Q" in state and state.get("Q_anc") is not None:
                    drift = (state["Q"] - state["Q_anc"]).norm().item()
                    # Return 0.0 if drift is NaN (numerical issue)
                    result[f"param_{idx}"] = drift if math.isfinite(drift) else 0.0
                idx += 1
        return result

    def get_tail_ratio(self) -> dict[str, float]:
        """Return tau (eigenvalue gap ratio) per matrix parameter."""
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm") == "adamw":
                continue
            for p in group["params"]:
                if not self._is_matrix_param(p):
                    continue
                state = self.state[p]
                if "tau" in state:
                    result[f"param_{idx}"] = state["tau"]
                idx += 1
        return result

    def get_rank(self) -> dict[str, int]:
        """Return current rank per matrix parameter."""
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm") == "adamw":
                continue
            for p in group["params"]:
                if not self._is_matrix_param(p):
                    continue
                state = self.state[p]
                if "rank" in state:
                    result[f"param_{idx}"] = state["rank"]
                idx += 1
        return result

    def get_energy_captured(self) -> dict[str, float]:
        """Return energy captured ratio per matrix parameter."""
        result = {}
        idx = 0
        for group in self.param_groups:
            if group.get("algorithm") == "adamw":
                continue
            for p in group["params"]:
                if not self._is_matrix_param(p):
                    continue
                state = self.state[p]
                if "energy" in state:
                    result[f"param_{idx}"] = state["energy"]
                idx += 1
        return result
