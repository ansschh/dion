"""
Vanilla Muon optimizer: momentum + Newton-Schulz orthonormalization every step.

Algorithm per step (for each 2D weight matrix):
    1. M <- mu * M + (1 - mu) * G     (EMA momentum accumulation)
    2. If nesterov: update = (1-mu)*G + mu*M   (Nesterov look-ahead)
       Else: update = M
    3. U <- NS_zeroth_power(update)    (Newton-Schulz orthonormalization)
    4. X <- X - lr * sqrt(max(1, m/n)) * U  (scaled parameter update)

References:
  - KellerJordan/Muon: https://github.com/KellerJordan/Muon/blob/master/muon.py
  - PyTorch torch.optim.Muon: https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py
  - Blog: https://kellerjordan.github.io/posts/muon/

DTensor-aware: operates on local shards under FSDP2.
"""
from __future__ import annotations

from typing import Optional, Callable

import torch
from torch.optim import Optimizer

from .ortho import newton_schulz_zeroth_power, shape_scale, MUON_NS_COEFFS
from .dtensor_utils import get_local_tensor, get_full_shape, ensure_2d, restore_shape


class Muon(Optimizer):
    """
    Vanilla Muon optimizer for matrix parameters.

    Applies Newton-Schulz orthonormalization to the momentum buffer every step,
    then uses the orthonormalized matrix as the update direction.

    Args:
        params: iterable of parameters (must be 2D+ tensors)
        lr: learning rate
        mu: momentum coefficient (default: 0.95)
        nesterov: use Nesterov momentum (default: True, matches PyTorch official)
        ns_steps: number of Newton-Schulz iterations (default: 5)
        weight_decay: decoupled weight decay coefficient (default: 0.0)
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.02,
        mu: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if not (0.0 <= mu < 1.0):
            raise ValueError("mu must be in [0, 1)")
        if ns_steps < 1:
            raise ValueError("ns_steps must be >= 1")
        defaults = dict(
            lr=lr,
            mu=mu,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["mu"]
            nesterov: bool = group["nesterov"]
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
                        f"Muon expects matrix parameters (ndim>=2). Got {tuple(local_p.shape)}"
                    )
                if local_grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")

                # Reshape to 2D
                local_p_2d, orig_shape = ensure_2d(local_p)
                grad_2d, _ = ensure_2d(local_grad)

                # Use full (unsharded) shape for scaling factor
                full_shape = get_full_shape(p)
                if len(full_shape) >= 2:
                    full_m = full_shape[0]
                    full_n = full_shape[1:].numel() if len(full_shape) > 2 else full_shape[1]
                else:
                    full_m, full_n = local_p_2d.shape

                # Initialize state
                state = self.state[p]
                if len(state) == 0:
                    state["M"] = torch.zeros_like(local_p_2d, dtype=torch.float32)

                M: torch.Tensor = state["M"]
                g = grad_2d.to(torch.float32)

                # Step 1: EMA momentum: M <- mu * M + (1 - mu) * G
                # Equivalent to M.lerp_(g, 1 - mu)
                M.mul_(mu).add_(g, alpha=1.0 - mu)

                # Step 2: Nesterov look-ahead (or plain momentum)
                if nesterov:
                    # update = (1 - mu) * G + mu * M
                    # = G + mu * (M - G)
                    # Equivalent to: g.lerp_(M, mu) but we don't want to mutate g
                    update = g.lerp(M, mu)
                else:
                    update = M

                # Step 3: Newton-Schulz orthonormalization
                U = newton_schulz_zeroth_power(update, steps=ns_steps, coeffs=MUON_NS_COEFFS)

                # Step 4: Scaled update using full (unsharded) dimensions
                scale = shape_scale(full_m, full_n)
                upd = (scale * U).to(dtype=local_p.dtype)

                # Reshape back
                upd = restore_shape(upd, orig_shape)

                # Decoupled weight decay
                if wd != 0.0:
                    local_p.mul_(1.0 - lr * wd)

                # Apply update
                local_p.add_(upd, alpha=-lr)

                state["M"] = M

        return loss

    # ------------------------------------------------------------------
    # Diagnostic methods (for benchmarking / telemetry)
    # ------------------------------------------------------------------

    def get_momentum_norms(self) -> list[float]:
        """Return Frobenius norms of all momentum buffers."""
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if "M" in state:
                    norms.append(state["M"].norm().item())
        return norms
