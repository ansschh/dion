"""
HybridOptimizersContainer: TorchTitan integration for Muon/Dion/Dion2.

Subclasses TorchTitan's OptimizersContainer to manage a single optimizer
per model_part that handles both spectral (matrix) and AdamW (scalar) params
via per-param-group `algorithm="adamw"` support in the microsoft/dion package.

Interface compatibility:
  - self.optimizers = list of optimizers (visible to LRSchedulersContainer)
  - step() / zero_grad() delegate to self.optimizers
  - state_dict() / load_state_dict() use TorchTitan's standard checkpoint API
  - __iter__ / __len__ delegate to self.optimizers for LRScheduler compat
"""
from __future__ import annotations

import functools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.optim import Optimizer

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable

from torch.distributed.device_mesh import init_device_mesh
from dion import Muon, Dion, Dion2
from .param_grouper import group_params_for_hybrid


def _get_device_mesh():
    """Get or create a device mesh for distributed optimizers."""
    try:
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                return init_device_mesh("cuda", (world_size,))
    except Exception:
        pass
    return None


class HybridOptimizersContainer(OptimizersContainer):
    """
    Manages a single optimizer (Muon/Dion/Dion2) per model_part that uses
    the official microsoft/dion package's built-in `algorithm="adamw"`
    per-param-group support for scalar parameters.

    The optimizers are exposed as self.optimizers so that
    LRSchedulersContainer can create LambdaLR schedulers for them.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        # Override name for the spectral optimizer
        name: str = "Muon"

        # --- Muon-specific ---
        mu: float = 0.95
        nesterov: bool = False
        adjust_lr: str = "spectral_norm"

        # --- Dion-specific ---
        rank_fraction: float = 0.25
        power_iters: int = 1

        # --- Dion2-specific ---
        fraction: float = 0.25
        ef_decay: float = 0.95

        # --- AdaDion-specific ---
        anchor_lambda: float = 0.1
        anchor_rho: float = 0.99
        tau_hi: float = 0.8
        tau_lo: float = 0.3
        refresh_period: int = 100

        # --- Scalar optimizer (AdamW param groups) ---
        scalar_lr: float = 3e-4
        scalar_weight_decay: float = 0.01
        scalar_beta1: float = 0.9
        scalar_beta2: float = 0.95
        scalar_eps: float = 1e-8

    def __init__(
        self, config: Config, *, model_parts: list[nn.Module]
    ) -> None:
        # DO NOT call super().__init__() — it would create standard AdamW.
        # We manually set up everything.

        self.model_parts = model_parts
        self.optimizers: list[Optimizer] = []
        all_params: list[nn.Parameter] = []

        for model in model_parts:
            groups = group_params_for_hybrid(model)

            # Build param groups for a single optimizer
            param_groups = []

            # Matrix params — use the spectral optimizer natively
            if groups.matrix_params:
                param_groups.append({
                    "params": groups.matrix_params,
                })

            # Embeddings — AdamW
            if groups.embed_params:
                param_groups.append({
                    "params": groups.embed_params,
                    "algorithm": "adamw",
                    "lr": config.scalar_lr,
                    "weight_decay": config.scalar_weight_decay,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "eps": config.scalar_eps,
                })

            # Output / LM head — AdamW with scaled LR
            if groups.output_params:
                d_model = groups.output_params[0].shape[-1]
                lm_head_lr = config.scalar_lr / math.sqrt(d_model)
                param_groups.append({
                    "params": groups.output_params,
                    "algorithm": "adamw",
                    "lr": lm_head_lr,
                    "weight_decay": config.scalar_weight_decay,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "eps": config.scalar_eps,
                })

            # Norms / biases — AdamW with no weight decay
            if groups.norm_params:
                param_groups.append({
                    "params": groups.norm_params,
                    "algorithm": "adamw",
                    "lr": config.scalar_lr,
                    "weight_decay": 0.0,
                    "betas": (config.scalar_beta1, config.scalar_beta2),
                    "eps": config.scalar_eps,
                })

            if param_groups:
                opt = self._create_optimizer(config, param_groups)
                self.optimizers.append(opt)
            else:
                # Fallback: dummy optimizer if no params
                opt = torch.optim.AdamW(
                    [torch.nn.Parameter(torch.empty(0))],
                    lr=config.lr,
                )
                self.optimizers.append(opt)

            # Collect all params for base Optimizer.__init__
            all_params.extend(groups.matrix_params)
            all_params.extend(groups.embed_params)
            all_params.extend(groups.output_params)
            all_params.extend(groups.norm_params)

        self._validate_length(len(self.model_parts))
        # Initialize base Optimizer with all params (for grad clipping compat)
        Optimizer.__init__(self, all_params, {"lr": config.lr})

    @staticmethod
    def _create_optimizer(
        config: "HybridOptimizersContainer.Config",
        param_groups: list[dict],
    ) -> Optimizer:
        """Create the appropriate optimizer based on config.name."""
        name = config.name
        mesh = _get_device_mesh()
        if name == "Muon":
            return Muon(
                param_groups,
                distributed_mesh=mesh,
                lr=config.lr,
                mu=config.mu,
                weight_decay=config.weight_decay,
                nesterov=config.nesterov,
                adjust_lr=config.adjust_lr,
            )
        elif name == "Dion":
            return Dion(
                param_groups,
                outer_shard_mesh=mesh,
                lr=config.lr,
                rank_fraction=config.rank_fraction,
                weight_decay=config.weight_decay,
            )
        elif name == "Dion2":
            return Dion2(
                param_groups,
                distributed_mesh=mesh,
                lr=config.lr,
                fraction=config.fraction,
                ef_decay=config.ef_decay,
                weight_decay=config.weight_decay,
            )
        elif name == "AdaDion":
            from ada_dion.optim import AdaDion
            return AdaDion(
                param_groups,
                lr=config.lr,
                mu=config.mu,
                rank_fraction=config.rank_fraction,
                anchor_lambda=config.anchor_lambda,
                anchor_rho=config.anchor_rho,
                tau_hi=config.tau_hi,
                tau_lo=config.tau_lo,
                weight_decay=config.weight_decay,
                refresh_period=config.refresh_period,
            )
        else:
            raise ValueError(
                f"Unknown matrix optimizer: {name!r}. "
                f"Supported: 'Muon', 'Dion', 'Dion2', 'AdaDion'"
            )

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            f"Expected {expected_length} optimizers, got {len(self.optimizers)}"
        )

    # ------------------------------------------------------------------
    # Iteration — LRSchedulersContainer iterates over this
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    # ------------------------------------------------------------------
    # Training loop interface
    # ------------------------------------------------------------------

    def step(self, *args, **kwargs) -> None:
        """Step all optimizers."""
        for opt in self.optimizers:
            opt.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients on all optimizers."""
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    # ------------------------------------------------------------------
    # Checkpoint state dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Save state dicts for all optimizers."""
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        sd = {}
        for model, opt in zip(self.model_parts, self.optimizers):
            for k, v in func(model, opt).items():
                sd[k] = v
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dicts for all optimizers."""
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def init_cache_state_dict(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Diagnostic interface
    # ------------------------------------------------------------------

    def get_optimizers(self) -> list[Optimizer]:
        """Return the optimizers for diagnostic access."""
        return self.optimizers
