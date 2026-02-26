r"""
Orthogonalization utilities for Muon-style optimizers.

Contains:
  - newton_schulz_zeroth_power: Quintic Newton-Schulz iterations for approximate
    orthonormalization (the core primitive for Muon/Dion2).
  - orthonormalize_qr: QR-based column orthonormalization (used by Dion).
  - col_norm: Column L2 normalization.
  - shape_scale: Muon/Dion scaling factor sqrt(max(1, m/n)).

References:
  - Muon: https://github.com/KellerJordan/Muon
  - PyTorch torch.optim.Muon: https://github.com/pytorch/pytorch/blob/v2.9.0/torch/optim/_muon.py
  - Microsoft Dion: https://github.com/microsoft/dion
"""
from __future__ import annotations

import math
from typing import Optional

import torch


# Canonical Muon/KellerJordan Newton-Schulz coefficients.
# Single tuple used uniformly for all iterations.
# Source: https://github.com/KellerJordan/Muon/blob/master/muon.py
# Also: PyTorch official torch.optim.Muon
MUON_NS_COEFFS = [(3.4445, -4.7750, 2.0315)]

# Microsoft Dion2 Newton-Schulz coefficients (per-iteration tuned).
# Source: https://github.com/microsoft/dion/blob/main/dion/newton_schulz_triton.py
DION2_NS_COEFFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


def newton_schulz_zeroth_power(
    x: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs=MUON_NS_COEFFS,
) -> torch.Tensor:
    """
    Approximate the "zeroth power" / orthonormalized version of x via
    Newton-Schulz iterations.

    Output has the same shape as x and (approximately) orthonormal rows/cols.

    NOTE:
      - We normalize by Frobenius norm to ensure spectral norm <= 1
        (since ||A||_2 <= ||A||_F), which is the precondition for convergence.
      - We compute A = X X^T on the smaller side by transposing when rows > cols.

    Args:
        x: (m, n) matrix
        steps: number of iterations
        eps: numerical epsilon for normalization
        coeffs: list of (a,b,c) tuples (cycled if steps > len(coeffs))
    """
    if x.ndim != 2:
        raise ValueError(f"newton_schulz_zeroth_power expects a 2D tensor, got {x.shape}")

    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T

    # normalize by Frobenius norm
    fro = torch.linalg.norm(x, ord="fro")
    x = x / (fro + eps)

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        A = x @ x.T
        B = b * A + c * (A @ A)
        x = a * x + (B @ x)

    if transpose:
        x = x.T
    return x


def col_norm(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Column-normalize w: each column has L2 norm 1."""
    if w.ndim != 2:
        raise ValueError(f"col_norm expects 2D tensor, got {w.shape}")
    norms = torch.linalg.norm(w, dim=0, keepdim=True)  # (1, r)
    return w / (norms + eps)


def orthonormalize_qr(p: torch.Tensor) -> torch.Tensor:
    """Orthonormalize columns using QR decomposition: returns Q (reduced)."""
    if p.ndim != 2:
        raise ValueError(f"orthonormalize_qr expects 2D tensor, got {p.shape}")
    q, _ = torch.linalg.qr(p, mode="reduced")
    return q


def shape_scale(m: int, n: int) -> float:
    """
    Muon/Dion scaling factor sqrt(max(1, m/n)) for a weight matrix of shape (m, n).

    For tall matrices (m > n): scales up by sqrt(m/n).
    For square or wide matrices (m <= n): scale is 1.0 (no downscaling).

    Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    """
    return math.sqrt(max(1.0, float(m) / float(n)))


# Keep old name as alias for backward compatibility
shape_scale_sqrt_out_in = shape_scale
