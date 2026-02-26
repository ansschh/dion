"""
Config registry for ada-dion experiments.

Provides config functions for each optimizer x model combination,
following TorchTitan's config_registry pattern.

Usage:
    python -m torchtitan.train \
        --module ada_dion.integration.config_registry \
        --config llama3_160m_muon \
        --training.steps 2000

Each function returns a Trainer.Config with the appropriate
HybridOptimizersContainer.Config (or standard OptimizersContainer.Config
for the AdamW baseline).
"""
from __future__ import annotations

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    FeedForward,
    GQAttention,
    RoPE,
)
from torchtitan.models.llama3 import (
    Llama3Model,
    Llama3TransformerBlock,
    parallelize_llama,
)
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiling import ProfilingConfig
from torchtitan.trainer import Trainer

from .hybrid_optimizer import HybridOptimizersContainer


# ======================================================================
# LLaMA3 160M model config
# ======================================================================
# ~160M params: dim=768, n_layers=12, n_heads=12, n_kv_heads=4
# hidden_dim ~= 2048 (via compute_ffn_hidden_dim with multiple_of=256)

_LLAMA3_160M = Llama3Model.Config(
    dim=768,
    n_layers=12,
    vocab_size=128256,
    layer=Llama3TransformerBlock.Config(
        feed_forward=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(768, multiple_of=256),
        ),
        attention=GQAttention.Config(
            n_heads=12,
            n_kv_heads=4,
            attn_backend="sdpa",
            rope_backend="complex",
        ),
    ),
    rope=RoPE.Config(
        dim=768 // 12,  # head_dim = dim // n_heads
        max_seq_len=8192,
        theta=500000,
        backend="complex",
        scaling="none",
    ),
)


def _model_registry_160m() -> ModelSpec:
    """Create a ModelSpec for LLaMA3 160M."""
    return ModelSpec(
        name="llama3",
        flavor="160M",
        model=_LLAMA3_160M,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


# ======================================================================
# Common training and scheduling configs
# ======================================================================

def _base_training_config(steps: int = 10000) -> TrainingConfig:
    return TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=steps,
        dtype="bfloat16",
        max_norm=1.0,
    )


def _base_lr_scheduler_config() -> LRSchedulersContainer.Config:
    return LRSchedulersContainer.Config(
        warmup_steps=200,
        decay_type="cosine",
        min_lr_factor=0.1,
    )


def _base_metrics_config() -> MetricsProcessor.Config:
    return MetricsProcessor.Config(
        log_freq=10,
        enable_tensorboard=True,
        enable_wandb=True,
    )


# ======================================================================
# AdamW baseline
# ======================================================================

def llama3_160m_adamw() -> Trainer.Config:
    """LLaMA3 160M with AdamW optimizer (baseline)."""
    return Trainer.Config(
        model_spec=_model_registry_160m(),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=3e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
        ),
        lr_scheduler=_base_lr_scheduler_config(),
        training=_base_training_config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=_base_metrics_config(),
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


# ======================================================================
# Muon
# ======================================================================

def llama3_160m_muon() -> Trainer.Config:
    """LLaMA3 160M with Muon (Newton-Schulz) + AdamW scalar."""
    return Trainer.Config(
        model_spec=_model_registry_160m(),
        optimizer=HybridOptimizersContainer.Config(
            name="Muon",
            lr=0.02,
            mu=0.95,
            ns_steps=5,
            weight_decay=0.0,
            scalar_lr=3e-4,
            scalar_weight_decay=0.01,
        ),
        lr_scheduler=_base_lr_scheduler_config(),
        training=_base_training_config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=_base_metrics_config(),
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


# ======================================================================
# Dion
# ======================================================================

def llama3_160m_dion() -> Trainer.Config:
    """LLaMA3 160M with Dion (low-rank power iteration) + AdamW scalar."""
    return Trainer.Config(
        model_spec=_model_registry_160m(),
        optimizer=HybridOptimizersContainer.Config(
            name="Dion",
            lr=0.02,
            rank_frac=0.25,
            dion_beta=0.05,
            weight_decay=0.0,
            scalar_lr=3e-4,
            scalar_weight_decay=0.01,
        ),
        lr_scheduler=_base_lr_scheduler_config(),
        training=_base_training_config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=_base_metrics_config(),
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


# ======================================================================
# Dion2
# ======================================================================

def llama3_160m_dion2() -> Trainer.Config:
    """LLaMA3 160M with Dion2 (alpha-fraction selection + NS) + AdamW scalar."""
    return Trainer.Config(
        model_spec=_model_registry_160m(),
        optimizer=HybridOptimizersContainer.Config(
            name="Dion2",
            lr=0.02,
            alpha=0.25,
            selection="top_l1",
            dion2_mu=0.95,
            ns_steps=5,
            weight_decay=0.0,
            scalar_lr=3e-4,
            scalar_weight_decay=0.01,
        ),
        lr_scheduler=_base_lr_scheduler_config(),
        training=_base_training_config(),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=_base_metrics_config(),
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


# ======================================================================
# Debug / quick validation configs
# ======================================================================

def llama3_debug_muon() -> Trainer.Config:
    """Tiny debug model with Muon — for local validation / fake_backend."""
    return Trainer.Config(
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=ModelSpec(
            name="llama3",
            flavor="debugmodel",
            model=Llama3Model.Config(
                dim=256,
                n_layers=6,
                vocab_size=2048,
                layer=Llama3TransformerBlock.Config(
                    feed_forward=FeedForward.Config(
                        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
                    ),
                    attention=GQAttention.Config(
                        n_heads=16,
                        attn_backend="sdpa",
                        rope_backend="complex",
                    ),
                ),
                rope=RoPE.Config(
                    dim=256 // 16,
                    max_seq_len=131072,
                    theta=500000,
                    backend="complex",
                    scaling="llama",
                ),
            ),
            parallelize_fn=parallelize_llama,
            pipelining_fn=pipeline_llm,
            build_loss_fn=build_cross_entropy_loss,
            post_optimizer_build_fn=None,
            state_dict_adapter=Llama3StateDictAdapter,
        ),
        optimizer=HybridOptimizersContainer.Config(
            name="Muon",
            lr=0.02,
            mu=0.95,
            ns_steps=5,
            weight_decay=0.0,
            scalar_lr=3e-4,
            scalar_weight_decay=0.01,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
    )


def llama3_debug_dion() -> Trainer.Config:
    """Tiny debug model with Dion — for local validation."""
    config = llama3_debug_muon()
    config.optimizer = HybridOptimizersContainer.Config(
        name="Dion",
        lr=0.02,
        rank_frac=0.25,
        dion_beta=0.05,
        weight_decay=0.0,
        scalar_lr=3e-4,
        scalar_weight_decay=0.01,
    )
    return config


def llama3_debug_dion2() -> Trainer.Config:
    """Tiny debug model with Dion2 — for local validation."""
    config = llama3_debug_muon()
    config.optimizer = HybridOptimizersContainer.Config(
        name="Dion2",
        lr=0.02,
        alpha=0.25,
        selection="top_l1",
        dion2_mu=0.95,
        ns_steps=5,
        weight_decay=0.0,
        scalar_lr=3e-4,
        scalar_weight_decay=0.01,
    )
    return config
