"""
Config registry for AdaDion V2 optimizer experiments.

Usage:
    python -m torchtitan.train \
        --module ortho_matrix.ada_dion_v2 \
        --config llama3_320m_adadion_v2_dion_equivalent \
        --training.steps 2000
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer

from ..common.base_container import BaseHybridOptimizersContainer
from ..common.model_configs import model_registry_320m
from ..common.training_configs import (
    base_metrics_config,
    base_training_config,
    debug_trainer_base,
)


# ======================================================================
# AdaDion V2 optimizer container
# ======================================================================

class AdaDionV2Container(BaseHybridOptimizersContainer):
    """AdaDion V2 (Dion + adaptive rank) + AdamW scalar optimizer container."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseHybridOptimizersContainer.Config):
        name: str = "AdaDionV2"
        # Dion pass-through
        rank_fraction: float = 0.5
        # Adaptive rank
        adaptive_rank: bool = False
        init_rank_fraction: float = 0.25
        rank_fraction_max: float = 0.7
        erank_ema_beta: float = 0.9
        rank_scale: float = 1.5
        rank_min: int = 16
        rank_quantize: int = 8
        rank_step_up: int = 16
        rank_step_down: int = 8
        use_quality_control: bool = False
        aerr_ema_beta: float = 0.95
        aerr_target: float = 0.08
        aerr_up_margin: float = 0.15
        aerr_down_margin: float = 0.15
        adapt_step: int = 1

    @staticmethod
    def _create_optimizer(
        config: "AdaDionV2Container.Config",
        param_groups: list[dict],
        mesh: DeviceMesh | None = None,
    ) -> Optimizer:
        from .adadion_v2 import AdaDionV2

        # When adaptive, allocate Q at rank_fraction_max (max rank).
        # init_rank_fraction determines the starting active rank.
        rf = config.rank_fraction_max if config.adaptive_rank else config.rank_fraction

        return AdaDionV2(
            param_groups,
            outer_shard_mesh=mesh,
            lr=config.lr,
            rank_fraction_max=rf,
            weight_decay=config.weight_decay,
            # Adaptive rank args
            adaptive_rank=config.adaptive_rank,
            init_rank_fraction=config.init_rank_fraction,
            erank_ema_beta=config.erank_ema_beta,
            rank_scale=config.rank_scale,
            rank_min=config.rank_min,
            rank_quantize=config.rank_quantize,
            rank_step_up=config.rank_step_up,
            rank_step_down=config.rank_step_down,
            use_quality_control=config.use_quality_control,
            aerr_ema_beta=config.aerr_ema_beta,
            aerr_target=config.aerr_target,
            aerr_up_margin=config.aerr_up_margin,
            aerr_down_margin=config.aerr_down_margin,
            adapt_step=config.adapt_step,
        )


# ======================================================================
# 320M configs
# ======================================================================

def _320m_trainer_base(**optimizer_kwargs) -> Trainer.Config:
    """Shared 320M trainer config."""
    return Trainer.Config(
        model_spec=model_registry_320m(),
        optimizer=AdaDionV2Container.Config(**optimizer_kwargs),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=610,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=base_training_config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=base_metrics_config(),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
            selective_ac_option="2",
        ),
        checkpoint=CheckpointManager.Config(
            interval=1000,
            last_save_model_only=True,
        ),
        validator=Validator.Config(
            freq=100,
            steps=20,
        ),
    )


def llama3_320m_adadion_v2_dion_equivalent() -> Trainer.Config:
    """LLaMA3 320M with AdaDion V2 configured to be identical to best Dion config.

    adaptive_rank=False so it delegates entirely to dion.Dion.
    Must produce identical val_loss to llama3_320m_dion.
    """
    return _320m_trainer_base(
        lr=0.012,
        rank_fraction=0.5,
        adaptive_rank=False,
        weight_decay=0.1,
        scalar_lr=0.012,
        scalar_beta1=0.95,
        scalar_beta2=0.95,
        scalar_eps=1e-8,
        scalar_weight_decay=0.1,
        output_head_lr_scaling=False,
    )


def llama3_320m_adadion_v2() -> Trainer.Config:
    """LLaMA3 320M with AdaDion V2 adaptive rank, using Dion-matched HPs."""
    return _320m_trainer_base(
        lr=0.012,
        rank_fraction=0.5,
        adaptive_rank=True,
        init_rank_fraction=0.5,
        rank_fraction_max=0.7,
        erank_ema_beta=0.5,
        rank_scale=1.5,
        rank_min=16,
        rank_quantize=8,
        weight_decay=0.1,
        scalar_lr=0.012,
        scalar_beta1=0.95,
        scalar_beta2=0.95,
        scalar_eps=1e-8,
        scalar_weight_decay=0.1,
        output_head_lr_scaling=False,
    )


def llama3_320m_adadion_v2_dion_lr() -> Trainer.Config:
    """LLaMA3 320M with AdaDion V2 adaptive rank, Dion lr=0.012."""
    return _320m_trainer_base(
        lr=0.012,
        rank_fraction=0.5,
        adaptive_rank=True,
        init_rank_fraction=0.5,
        rank_fraction_max=0.7,
        erank_ema_beta=0.5,
        rank_scale=1.5,
        rank_min=16,
        rank_quantize=8,
        weight_decay=0.1,
        scalar_lr=0.012,
        scalar_beta1=0.95,
        scalar_beta2=0.95,
        scalar_eps=1e-8,
        scalar_weight_decay=0.1,
        output_head_lr_scaling=False,
    )


# ======================================================================
# Debug config
# ======================================================================

def llama3_debug_2gpu_adadion_v2() -> Trainer.Config:
    """Debug config for 2 GPUs — tiny model (dim=256, 6 layers), 50 steps."""
    config = debug_trainer_base()
    config.optimizer = AdaDionV2Container.Config(
        lr=0.012,
        rank_fraction=0.5,
        adaptive_rank=True,
        init_rank_fraction=0.25,
        rank_fraction_max=0.7,
        erank_ema_beta=0.5,
        rank_scale=1.5,
        rank_min=8,
        rank_quantize=8,
        weight_decay=0.1,
        scalar_lr=0.012,
        scalar_weight_decay=0.1,
    )
    config.training.steps = 50
    config.training.local_batch_size = 4
    return config


def llama3_debug_adadion_v2() -> Trainer.Config:
    """Tiny debug model with AdaDion V2 — for local validation."""
    config = debug_trainer_base()
    config.optimizer = AdaDionV2Container.Config(
        lr=0.02,
        rank_fraction=0.5,
        adaptive_rank=True,
        init_rank_fraction=0.25,
        rank_fraction_max=0.7,
        erank_ema_beta=0.9,
        rank_scale=1.5,
        rank_min=8,
        rank_quantize=8,
        weight_decay=0.0,
        scalar_lr=3e-4,
        scalar_weight_decay=0.01,
    )
    return config
