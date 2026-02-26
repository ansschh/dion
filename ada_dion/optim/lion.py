r"""
Minimal Lion optimizer (sign-based).

Ported from: A:\muon research\muonbp_vs_dion\optim\lion.py

Used as the scalar optimizer for parameters that shouldn't use spectral
methods (embeddings, norms, biases, LM head).
"""
from __future__ import annotations

from typing import Optional, Callable

import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """
    Minimal Lion optimizer (sign-based momentum).

    Used for scalar parameters in hybrid optimizer setups where matrix
    parameters use Muon/Dion/Dion2 and everything else uses Lion or AdamW.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-4,
        betas=(0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            wd: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)

                m = state["m"]
                m.mul_(beta1).add_(g.to(torch.float32), alpha=(1.0 - beta1))

                # decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                update = torch.sign(m).to(dtype=p.dtype)
                p.add_(update, alpha=-lr)

                # second moment smoothing
                m.mul_(beta2)

                state["m"] = m

        return loss
