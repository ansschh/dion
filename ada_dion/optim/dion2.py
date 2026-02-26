"""
Dion2 optimizer: alpha-fraction selection + Newton-Schulz on subset + selective decay.

Implements the algorithm from "Dion2: A Simple Method to Shrink Matrix in Muon"
(arXiv 2512.16928).

Algorithm per step (for each 2D weight matrix):
    1. M <- M + G                                 (accumulate gradient)
    2. K = select_alpha_fraction(M, alpha)         (select along shorter dimension)
    3. M_sel = M[K, :] or M[:, K]                 (extract selected submatrix)
    4. U_sel = NS_zeroth_power(M_sel)              (orthonormalize the subset)
    5. M[K, :] *= mu  (or M[:, K] *= mu)          (SELECTIVE decay on selected slices)
    6. X[K, :] -= lr * sqrt(max(1,m/n)) * U_sel   (sparse parameter update)

Key design decisions (from paper ablations):
  - Selection along SHORTER dimension (rows if m<=n, columns if m>n)
  - Selective decay is CRITICAL — only selected slices get decayed
  - Weight decay is applied to the ENTIRE tensor (not just selected slices)

References:
  - Paper: https://arxiv.org/abs/2512.16928
  - Reference: https://github.com/microsoft/dion/blob/main/dion/dion2.py

DTensor-aware: operates on local shards under FSDP2.
"""
from __future__ import annotations

from typing import Optional, Callable, Literal

import torch
from torch.optim import Optimizer

from .ortho import newton_schulz_zeroth_power, shape_scale, DION2_NS_COEFFS
from .dtensor_utils import get_local_tensor, get_full_shape, ensure_2d, restore_shape


class Dion2(Optimizer):
    """
    Dion2 optimizer for matrix parameters with DTensor/FSDP2 support.

    Selects an alpha-fraction of rows/columns (along the shorter dimension)
    at each step, orthonormalizes only that submatrix via Newton-Schulz,
    applies selective decay only to selected slices, and updates only
    the selected parameter slices.

    Args:
        params: iterable of parameters (must be 2D+ tensors)
        lr: learning rate
        alpha: fraction of rows/cols to select per step (default: 0.25)
        selection: selection method, "top_l1" or "random" (default: "top_l1")
        mu: momentum/decay coefficient for selective decay (default: 0.95)
        ns_steps: number of Newton-Schulz iterations (default: 5)
        weight_decay: decoupled weight decay coefficient (default: 0.0)
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
        alpha: float = 0.25,
        selection: Literal["top_l1", "random"] = "top_l1",
        mu: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if selection not in ("top_l1", "random"):
            raise ValueError(f"selection must be 'top_l1' or 'random', got {selection!r}")
        if not (0.0 <= mu < 1.0):
            raise ValueError("mu must be in [0, 1)")
        if ns_steps < 1:
            raise ValueError("ns_steps must be >= 1")
        defaults = dict(
            lr=lr,
            alpha=alpha,
            selection=selection,
            mu=mu,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Diagnostic storage
        self._last_selected_counts = []
        self._last_selected_fractions = []
        self._last_select_dims = []

    def _select_indices_top_l1(
        self, M: torch.Tensor, k: int, dim: int
    ) -> torch.Tensor:
        """Select top-k slices by L1 norm along the given dimension."""
        # Sum absolute values along the OTHER dimension
        sum_dim = 1 if dim == 0 else 0
        norms = M.abs().sum(dim=sum_dim)  # (m,) if dim=0, (n,) if dim=1
        _, indices = norms.topk(k, sorted=False)
        return indices

    def _select_indices_random(
        self, M: torch.Tensor, k: int, dim: int
    ) -> torch.Tensor:
        """Select k slices uniformly at random along the given dimension."""
        size = M.shape[dim]
        indices = torch.randperm(size, device=M.device)[:k]
        return indices

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._last_selected_counts = []
        self._last_selected_fractions = []
        self._last_select_dims = []

        for group in self.param_groups:
            lr: float = group["lr"]
            alpha: float = group["alpha"]
            selection: str = group["selection"]
            mu: float = group["mu"]
            ns_steps: int = group["ns_steps"]
            wd: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get local shard for DTensor compatibility
                local_p = get_local_tensor(p)
                local_grad = get_local_tensor(p.grad).detach()

                if local_p.ndim < 2:
                    raise ValueError(
                        f"Dion2 expects matrix parameters (ndim>=2). Got {tuple(local_p.shape)}"
                    )
                if local_grad.is_sparse:
                    raise RuntimeError("Dion2 does not support sparse gradients.")

                # Reshape to 2D
                local_p_2d, orig_shape = ensure_2d(local_p)
                grad_2d, _ = ensure_2d(local_grad)
                m, n = local_p_2d.shape

                # Use full (unsharded) shape for scaling factor
                full_shape = get_full_shape(p)
                if len(full_shape) >= 2:
                    full_m = full_shape[0]
                    full_n = full_shape[1:].numel() if len(full_shape) > 2 else full_shape[1]
                else:
                    full_m, full_n = m, n

                # Determine selection dimension: shorter side
                # Reference: "For all experiments, we select the submatrix along
                # the shorter dimension of the momentum matrix."
                select_dim = 0 if m <= n else 1
                select_size = m if select_dim == 0 else n
                k = max(1, int(round(select_size * alpha)))

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["M"] = torch.zeros(
                        (m, n), device=local_p.device, dtype=torch.float32
                    )

                M: torch.Tensor = state["M"]

                # Step 1: Accumulate gradient into momentum
                M.add_(grad_2d.to(torch.float32))

                # Step 2: Select alpha-fraction along shorter dimension
                if selection == "top_l1":
                    indices = self._select_indices_top_l1(M, k, select_dim)
                else:
                    indices = self._select_indices_random(M, k, select_dim)

                # Step 3: Extract selected submatrix
                if select_dim == 0:
                    M_sel = M[indices, :]  # (k, n)
                else:
                    M_sel = M[:, indices]  # (m, k)

                # Step 4: Orthonormalize the subset via Newton-Schulz
                U_sel = newton_schulz_zeroth_power(
                    M_sel, steps=ns_steps, coeffs=DION2_NS_COEFFS
                )

                # Step 5: Selective decay — ONLY on selected slices
                if select_dim == 0:
                    M.index_copy_(0, indices, M.index_select(0, indices) * mu)
                else:
                    M.index_copy_(1, indices, M.index_select(1, indices) * mu)

                # Step 6: Sparse parameter update
                scale = shape_scale(full_m, full_n)
                upd_sel = (scale * U_sel).to(dtype=local_p.dtype)

                # Decoupled weight decay on ENTIRE tensor (not just selected slices)
                # Reference: torch._foreach_mul_(X, 1 - base_lr * weight_decay)
                if wd != 0.0:
                    local_p_2d.mul_(1.0 - lr * wd)

                # Apply sparse update
                if select_dim == 0:
                    local_p_2d.index_add_(0, indices, upd_sel, alpha=-lr)
                else:
                    local_p_2d.index_add_(1, indices, upd_sel, alpha=-lr)

                # Write back state
                state["M"] = M

                # Diagnostics
                self._last_selected_counts.append(k)
                self._last_selected_fractions.append(k / select_size)
                self._last_select_dims.append(select_dim)

        return loss

    # ------------------------------------------------------------------
    # Diagnostic methods
    # ------------------------------------------------------------------

    def get_alpha(self) -> float:
        """Return the configured alpha value."""
        if self.param_groups:
            return self.param_groups[0]["alpha"]
        return 0.0

    def get_selected_fractions(self) -> list[float]:
        """Return actual selected fraction for each param from last step."""
        return self._last_selected_fractions

    def get_selected_counts(self) -> list[int]:
        """Return number of selected rows/cols for each param from last step."""
        return self._last_selected_counts

    def get_select_dims(self) -> list[int]:
        """Return which dimension was selected (0=rows, 1=cols) for each param."""
        return self._last_select_dims

    def get_sparsity(self) -> float:
        """Return average sparsity (1 - fraction updated) across params."""
        if not self._last_selected_fractions:
            return 0.0
        return 1.0 - sum(self._last_selected_fractions) / len(self._last_selected_fractions)

    def get_momentum_norms(self) -> list[float]:
        """Return Frobenius norms of all momentum buffers."""
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if "M" in state:
                    norms.append(state["M"].norm().item())
        return norms
