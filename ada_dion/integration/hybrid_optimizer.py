"""
HybridOptimizersContainer: TorchTitan integration for Muon/Dion/Dion2.

Subclasses TorchTitan's OptimizersContainer to manage two optimizer paths
per model_part:
  - Matrix optimizer (Muon, Dion, or Dion2) for 2D weight matrices
  - Scalar optimizer (AdamW) for embeddings, norms, biases, LM head

Interface compatibility:
  - self.optimizers = list of matrix optimizers (visible to LRSchedulersContainer)
  - self._scalar_optimizers = list of scalar optimizers (internal, own LR schedule)
  - step() calls both
  - zero_grad() calls both
  - state_dict() / load_state_dict() save/load both with prefixed keys
  - __iter__ / __len__ delegate to self.optimizers for LRScheduler compat
"""
from __future__ import annotations

import functools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable

from ..optim.muon import Muon
from ..optim.dion import Dion
from ..optim.dion2 import Dion2
from .param_grouper import group_params_for_hybrid


class HybridOptimizersContainer(OptimizersContainer):
    """
    Manages matrix optimizer (Muon/Dion/Dion2) + scalar optimizer (AdamW)
    per model_part. Integrates with TorchTitan's training loop.

    The matrix optimizers are exposed as self.optimizers so that
    LRSchedulersContainer can create LambdaLR schedulers for them.
    The scalar optimizers are managed internally with their own LR
    scheduling (same warmup/cosine shape, different base LR).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        # Override name for the matrix optimizer
        name: str = "Muon"

        # --- Matrix optimizer hyperparameters ---
        # Muon
        mu: float = 0.95
        ns_steps: int = 5
        # Dion
        rank_frac: float = 0.25
        dion_beta: float = 0.05
        # Dion2
        alpha: float = 0.25
        selection: str = "top_l1"
        dion2_mu: float = 0.95

        # --- Scalar optimizer hyperparameters ---
        scalar_lr: float = 3e-4
        scalar_weight_decay: float = 0.01
        scalar_beta1: float = 0.9
        scalar_beta2: float = 0.95
        scalar_eps: float = 1e-8

    _scalar_optimizers: list[torch.optim.AdamW]
    _scalar_schedulers: list[LambdaLR]

    def __init__(
        self, config: Config, *, model_parts: list[nn.Module]
    ) -> None:
        # DO NOT call super().__init__() — it would create standard AdamW.
        # We manually set up everything.

        self.model_parts = model_parts
        self.optimizers: list[Optimizer] = []
        self._scalar_optimizers: list[torch.optim.AdamW] = []
        self._scalar_schedulers: list[LambdaLR] = []
        all_params: list[nn.Parameter] = []

        for model in model_parts:
            groups = group_params_for_hybrid(model)

            # --- Matrix optimizer ---
            matrix_params = groups.matrix_params
            if matrix_params:
                m_opt = self._create_matrix_optimizer(config, matrix_params)
                self.optimizers.append(m_opt)
            else:
                # Fallback: create a dummy AdamW if no matrix params
                m_opt = torch.optim.AdamW(
                    [torch.nn.Parameter(torch.empty(0))],
                    lr=config.lr,
                )
                self.optimizers.append(m_opt)

            # --- Scalar optimizer (AdamW for embed + output + norm) ---
            scalar_param_groups = []

            # Embeddings — base scalar LR
            if groups.embed_params:
                scalar_param_groups.append({
                    "params": groups.embed_params,
                    "lr": config.scalar_lr,
                    "weight_decay": config.scalar_weight_decay,
                })

            # Output / LM head — scaled LR: scalar_lr / sqrt(dim)
            if groups.output_params:
                # Estimate d_model from the output weight shape
                d_model = groups.output_params[0].shape[-1]
                lm_head_lr = config.scalar_lr / math.sqrt(d_model)
                scalar_param_groups.append({
                    "params": groups.output_params,
                    "lr": lm_head_lr,
                    "weight_decay": config.scalar_weight_decay,
                })

            # Norms / biases — no weight decay
            if groups.norm_params:
                scalar_param_groups.append({
                    "params": groups.norm_params,
                    "lr": config.scalar_lr,
                    "weight_decay": 0.0,
                })

            if scalar_param_groups:
                s_opt = torch.optim.AdamW(
                    scalar_param_groups,
                    lr=config.scalar_lr,
                    betas=(config.scalar_beta1, config.scalar_beta2),
                    eps=config.scalar_eps,
                    weight_decay=config.scalar_weight_decay,
                )
                self._scalar_optimizers.append(s_opt)
            else:
                self._scalar_optimizers.append(None)

            # Collect all params for base Optimizer.__init__
            all_params.extend(matrix_params)
            all_params.extend(groups.embed_params)
            all_params.extend(groups.output_params)
            all_params.extend(groups.norm_params)

        self._validate_length(len(self.model_parts))
        # Initialize base Optimizer with all params (for grad clipping compat)
        Optimizer.__init__(self, all_params, {"lr": config.lr})

    @staticmethod
    def _create_matrix_optimizer(
        config: "HybridOptimizersContainer.Config",
        params: list[nn.Parameter],
    ) -> Optimizer:
        """Create the appropriate matrix optimizer based on config.name."""
        name = config.name
        if name == "Muon":
            return Muon(
                params,
                lr=config.lr,
                mu=config.mu,
                ns_steps=config.ns_steps,
                weight_decay=config.weight_decay,
            )
        elif name == "Dion":
            return Dion(
                params,
                lr=config.lr,
                rank_frac=config.rank_frac,
                beta=config.dion_beta,
                weight_decay=config.weight_decay,
            )
        elif name == "Dion2":
            return Dion2(
                params,
                lr=config.lr,
                alpha=config.alpha,
                selection=config.selection,
                mu=config.dion2_mu,
                ns_steps=config.ns_steps,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown matrix optimizer: {name!r}. "
                f"Supported: 'Muon', 'Dion', 'Dion2'"
            )

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            f"Expected {expected_length} matrix optimizers, got {len(self.optimizers)}"
        )
        assert expected_length == len(self._scalar_optimizers), (
            f"Expected {expected_length} scalar optimizers, got {len(self._scalar_optimizers)}"
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
        """Step both matrix and scalar optimizers."""
        for m_opt in self.optimizers:
            m_opt.step(*args, **kwargs)
        for s_opt in self._scalar_optimizers:
            if s_opt is not None:
                s_opt.step(*args, **kwargs)
        # Step scalar LR schedulers if they exist
        for sched in self._scalar_schedulers:
            sched.step()

    def zero_grad(self, *args, **kwargs) -> None:
        """Zero gradients on both matrix and scalar optimizers."""
        for m_opt in self.optimizers:
            m_opt.zero_grad(*args, **kwargs)
        for s_opt in self._scalar_optimizers:
            if s_opt is not None:
                s_opt.zero_grad(*args, **kwargs)

    # ------------------------------------------------------------------
    # LR scheduling for scalar optimizer
    # ------------------------------------------------------------------

    def setup_scalar_lr_schedule(
        self,
        lr_lambda,
    ) -> None:
        """
        Set up LR scheduling for scalar optimizers.

        Call this after TorchTitan's LRSchedulersContainer is built,
        passing the same lr_lambda function so both matrix and scalar
        optimizers follow the same warmup/decay shape.
        """
        self._scalar_schedulers = []
        for s_opt in self._scalar_optimizers:
            if s_opt is not None:
                self._scalar_schedulers.append(LambdaLR(s_opt, lr_lambda))

    # ------------------------------------------------------------------
    # Checkpoint state dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Save state dicts for both matrix and scalar optimizers."""
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

        # Matrix optimizer state
        matrix_sd = {}
        for sd in map(func, self.model_parts, self.optimizers):
            for k, v in sd.items():
                matrix_sd[f"matrix.{k}"] = v

        # Scalar optimizer state (simple state_dict since params may not
        # cover the full model)
        scalar_sd = {}
        for i, s_opt in enumerate(self._scalar_optimizers):
            if s_opt is not None:
                for k, v in s_opt.state_dict().items():
                    scalar_sd[f"scalar.{i}.{k}"] = v

        return {**matrix_sd, **scalar_sd}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dicts for both matrix and scalar optimizers."""
        # Separate matrix and scalar keys
        matrix_sd = {
            k[len("matrix."):]: v
            for k, v in state_dict.items()
            if k.startswith("matrix.")
        }
        scalar_sds: dict[int, dict] = {}
        for k, v in state_dict.items():
            if k.startswith("scalar."):
                parts = k.split(".", 2)  # scalar.{i}.{key}
                idx = int(parts[1])
                real_key = parts[2]
                if idx not in scalar_sds:
                    scalar_sds[idx] = {}
                scalar_sds[idx][real_key] = v

        # Load matrix optimizer state
        if matrix_sd:
            func = functools.partial(
                set_optimizer_state_dict,
                optim_state_dict=matrix_sd,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
            list(map(func, self.model_parts, self.optimizers))

        # Load scalar optimizer state
        for i, s_opt in enumerate(self._scalar_optimizers):
            if s_opt is not None and i in scalar_sds:
                s_opt.load_state_dict(scalar_sds[i])

    def init_cache_state_dict(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Diagnostic interface
    # ------------------------------------------------------------------

    def get_matrix_optimizers(self) -> list[Optimizer]:
        """Return the matrix optimizers for diagnostic access."""
        return self.optimizers

    def get_scalar_optimizers(self) -> list:
        """Return the scalar optimizers for diagnostic access."""
        return self._scalar_optimizers
