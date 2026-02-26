"""
DTensor/FSDP2 utilities for spectral optimizers.

Under FSDP2, model parameters are DTensors with Shard(0) placement.
Gradients are also DTensors with the same placement.

Our optimizers need to operate on the LOCAL SHARD of each parameter.
This module provides helpers to:
  - Extract local tensors from DTensors
  - Query shard info (dimension, local shape, full shape)
  - Type-check for DTensor

Key insight: Under FSDP2, each rank holds rows [start:end, :] of the weight.
The momentum buffers are also only the local shard. Operations like
Newton-Schulz and Dion's power iteration are shard-local -- no extra
communication beyond what FSDP already provides.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def is_dtensor(t: torch.Tensor) -> bool:
    """Check if a tensor is a DTensor."""
    try:
        from torch.distributed.tensor import DTensor
        return isinstance(t, DTensor)
    except ImportError:
        return False


def get_local_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Extract the local shard from a DTensor, or return the tensor as-is.

    Under FSDP2, parameters and gradients are DTensors. The local shard
    is accessible via ._local_tensor and is a regular torch.Tensor.
    """
    if is_dtensor(t):
        return t._local_tensor
    return t


def get_full_shape(t: torch.Tensor) -> torch.Size:
    """
    Get the full (unsharded) shape of a tensor.

    DTensors report their full logical shape via .shape, not the local shard shape.
    For regular tensors, this is just .shape.
    """
    return t.shape


def get_local_shape(t: torch.Tensor) -> torch.Size:
    """
    Get the local shard shape of a tensor.

    For DTensors, this is the actual allocated shape on this rank.
    For regular tensors, same as .shape.
    """
    if is_dtensor(t):
        return t._local_tensor.shape
    return t.shape


def get_shard_info(t: torch.Tensor) -> Optional[Tuple[int, int, int]]:
    """
    Get shard placement info for a DTensor.

    Returns (shard_dim, num_shards, shard_idx) if sharded, or None if not a DTensor.
    For FSDP2, typically shard_dim=0.
    """
    if not is_dtensor(t):
        return None
    try:
        placements = t.placements
        for i, p in enumerate(placements):
            if hasattr(p, 'dim'):
                # This is a Shard placement
                return (p.dim, t.device_mesh.size(i), t.device_mesh.get_local_rank(i))
    except (AttributeError, RuntimeError):
        pass
    return None


def ensure_2d(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """
    Reshape a tensor to 2D for matrix operations.

    Handles:
      - 2D: returned as-is
      - 4D conv: [C_out, C_in, kH, kW] -> [C_out, C_in*kH*kW]
      - Other nD: [d0, d1, ...] -> [d0, d1*d2*...]

    Returns (tensor_2d, original_shape).
    """
    orig_shape = t.shape
    if t.ndim == 2:
        return t, orig_shape
    if t.ndim < 2:
        raise ValueError(f"Cannot reshape {t.ndim}D tensor to 2D")
    return t.reshape(t.shape[0], -1), orig_shape


def restore_shape(t: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    """Restore tensor from 2D back to original shape."""
    if t.shape == orig_shape:
        return t
    return t.reshape(orig_shape)
