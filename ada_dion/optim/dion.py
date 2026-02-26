r"""
Dion optimizer with DTensor/FSDP2 awareness.

Algorithm (per 2D weight matrix, from the Dion paper arXiv:2504.05295):
    1. B <- M + G                      (accumulate gradient into momentum)
    2. P = B @ Q                       (project momentum onto right basis)
    3. P = QR_orthonormalize(P)        (orthonormalize left factors)
    4. R = B^T @ P                     (compute new right factors)
    5. M <- B - (1-mu) * P @ R^T       (error feedback)
    6. Q <- col_norm(R)                (refresh right basis)
    7. X <- X - lr * sqrt(max(1,m/n)) * P @ Q^T  (scaled parameter update)

State per parameter: M (momentum, m x n), Q (right basis, n x r)

Note: Our `beta` parameter = paper's `(1-mu)`. With beta=0.05, this
corresponds to mu=0.95 in the reference.

References:
  - Paper: https://arxiv.org/abs/2504.05295
  - Reference: https://github.com/microsoft/dion
"""
from __future__ import annotations

from typing import Optional, Callable

import torch
from torch.optim import Optimizer

from .ortho import col_norm, orthonormalize_qr, shape_scale
from .dtensor_utils import get_local_tensor, get_full_shape, ensure_2d, restore_shape


class Dion(Optimizer):
    """
    Dion optimizer for matrix parameters with DTensor/FSDP2 support.

    Implements centralized Dion (Algorithm 1) from the Dion paper: low-rank
    power iteration with error feedback for communication-efficient distributed
    orthonormalization.

    Args:
        params: iterable of parameters (must be 2D+ tensors)
        lr: learning rate
        rank_frac: fraction of min(m,n) to use as rank r (default: 0.25)
        beta: error feedback coefficient, = paper's (1-mu) (default: 0.05)
        weight_decay: decoupled weight decay coefficient (default: 0.0)
        eps: epsilon for numerical stability (default: 1e-12)
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-2,
        rank_frac: float = 0.25,
        beta: float = 0.05,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0 < rank_frac <= 1.0):
            raise ValueError("rank_frac must be in (0, 1]")
        if not (0 < beta <= 1.0):
            raise ValueError("beta must be in (0, 1]")
        defaults = dict(
            lr=lr,
            rank_frac=rank_frac,
            beta=beta,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

        # Diagnostic storage
        self._last_col_norm_diag = []
        self._last_residual_norms = []

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._last_col_norm_diag = []
        self._last_residual_norms = []

        for group in self.param_groups:
            lr: float = group["lr"]
            rank_frac: float = group["rank_frac"]
            beta: float = group["beta"]
            wd: float = group["weight_decay"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get local shard for DTensor compatibility
                local_p = get_local_tensor(p)
                local_grad = get_local_tensor(p.grad).detach()

                if local_p.ndim < 2:
                    raise ValueError(
                        f"Dion expects matrix parameters (ndim>=2). Got {tuple(local_p.shape)}"
                    )
                if local_grad.is_sparse:
                    raise RuntimeError("Dion does not support sparse gradients.")

                # Reshape to 2D
                local_p_2d, orig_shape = ensure_2d(local_p)
                grad_2d, _ = ensure_2d(local_grad)
                m, n = local_p_2d.shape

                # Rank computed from min(m, n), capped at min(m, n)
                # Reference: https://github.com/microsoft/dion — r = rank_fraction * min(m, n)
                min_dim = min(m, n)
                r = max(1, min(min_dim, int(round(min_dim * rank_frac))))

                # Use full (unsharded) shape for scaling factor
                full_shape = get_full_shape(p)
                if len(full_shape) >= 2:
                    full_m = full_shape[0]
                    full_n = full_shape[1:].numel() if len(full_shape) > 2 else full_shape[1]
                else:
                    full_m, full_n = m, n

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["M"] = torch.zeros(
                        (m, n), device=local_p.device, dtype=torch.float32
                    )
                    # Reference uses plain randn (no QR at init)
                    state["V"] = torch.randn(
                        (n, r), device=local_p.device, dtype=torch.float32
                    )

                M: torch.Tensor = state["M"]
                V: torch.Tensor = state["V"]

                # Step 1: Accumulate gradient (B = M + G)
                M.add_(grad_2d.to(torch.float32))

                # Step 2: Power iteration
                P = M @ V       # (m, r)
                U = orthonormalize_qr(P)  # (m, r)
                W = M.T @ U     # (n, r)

                # Step 3: Error feedback: M <- M - beta * U @ W^T
                M.addmm_(U, W.T, beta=1.0, alpha=-beta)

                # Step 4: Refresh right basis
                V = col_norm(W, eps=eps)

                # Step 5: Scaled parameter update using fused addmm_
                # Avoids materializing full (m, n) matrix O = U @ V^T
                scale = shape_scale(full_m, full_n)
                scaled_lr = lr * scale

                # Weight decay (applied before update, matching reference)
                if wd != 0.0:
                    local_p.mul_(1.0 - lr * wd)

                # X <- X - scaled_lr * U @ V^T  (fused, no full m×n intermediate)
                U_cast = U.to(dtype=local_p_2d.dtype)
                V_cast = V.to(dtype=local_p_2d.dtype)
                if local_p_2d.ndim == 2 and orig_shape == local_p_2d.shape:
                    # Direct fused update (common case: already 2D, no reshape needed)
                    local_p_2d.addmm_(U_cast, V_cast.T, beta=1.0, alpha=-scaled_lr)
                else:
                    # Fallback for reshaped tensors
                    upd = U_cast @ V_cast.T
                    upd = restore_shape(upd, orig_shape)
                    local_p.add_(upd, alpha=-scaled_lr)

                # Write back state
                state["M"] = M
                state["V"] = V

                # Diagnostics: residual norm of M after error feedback
                self._last_residual_norms.append(M.norm().item())
                # V is column-normalized (not orthonormal), so we report
                # the max off-diagonal magnitude of V^T @ V as a diagnostic
                vtv = V.T @ V
                diag_mean = vtv.diag().mean().item()
                self._last_col_norm_diag.append(diag_mean)

        return loss

    # ------------------------------------------------------------------
    # Diagnostic methods
    # ------------------------------------------------------------------

    def get_column_norm_quality(self) -> list[float]:
        """Return mean diagonal of V^T V for each param (should be ~1.0)."""
        return self._last_col_norm_diag

    def get_residual_norms(self) -> list[float]:
        """Return ||M||_F for each parameter from last step (after error feedback)."""
        return self._last_residual_norms

    def get_ranks(self) -> list[int]:
        """Return the rank r used for each parameter."""
        ranks = []
        for group in self.param_groups:
            rank_frac = group["rank_frac"]
            for p in group["params"]:
                local_p = get_local_tensor(p)
                if local_p.ndim >= 2:
                    m, n = ensure_2d(local_p)[0].shape
                    min_dim = min(m, n)
                    ranks.append(max(1, min(min_dim, int(round(min_dim * rank_frac)))))
        return ranks
