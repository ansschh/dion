"""
Parameter grouper for hybrid optimizer setups.

Classifies model parameters into groups based on TorchTitan's Llama3 naming
conventions. This ensures IDENTICAL grouping across all optimizers —
critical for fair benchmarking (otherwise you're comparing grouping choices,
not optimizer algorithms).

Groups:
  - matrix_params: 2D weights in transformer blocks (attention + MLP projections)
  - embed_params: token embedding weights
  - output_params: LM head / output projection weights
  - norm_params: RMSNorm weights, biases, any 1D parameters
"""
from __future__ import annotations

from typing import NamedTuple

import torch.nn as nn


class ParamGroups(NamedTuple):
    """Named groups of parameters for hybrid optimizer setup."""
    matrix_params: list[nn.Parameter]
    embed_params: list[nn.Parameter]
    output_params: list[nn.Parameter]
    norm_params: list[nn.Parameter]


def group_params_for_hybrid(model: nn.Module) -> ParamGroups:
    """
    Classify model parameters into groups for hybrid optimizer.

    Uses TorchTitan Llama3 naming conventions:
      - tok_embeddings.weight          -> embed_params
      - output.weight                  -> output_params
      - layers.*.attention.{wq,wk,wv,wo}.weight  -> matrix_params (2D)
      - layers.*.feed_forward.{w1,w2,w3}.weight   -> matrix_params (2D)
      - *.norm.weight, *.bias, 1D      -> norm_params

    For non-Llama models, falls back to dimension-based classification:
      - 2D params in transformer layers -> matrix_params
      - Named "embed" or "embedding"    -> embed_params
      - Named "output" or "lm_head"     -> output_params
      - Everything else (1D, bias)      -> norm_params

    Returns:
        ParamGroups with the four parameter lists.
    """
    matrix_params = []
    embed_params = []
    output_params = []
    norm_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()

        # Token embeddings — treat as collection of 1D vectors, NOT matrix
        if _is_embedding(name_lower):
            embed_params.append(param)
        # Output / LM head — needs scaled LR
        elif _is_output_head(name_lower):
            output_params.append(param)
        # 2D weight matrices in transformer blocks — use spectral optimizer
        elif param.ndim >= 2 and _is_transformer_weight(name_lower):
            matrix_params.append(param)
        # Everything else (norms, biases, 1D params)
        else:
            norm_params.append(param)

    return ParamGroups(
        matrix_params=matrix_params,
        embed_params=embed_params,
        output_params=output_params,
        norm_params=norm_params,
    )


def _is_embedding(name: str) -> bool:
    """Check if parameter name indicates an embedding layer."""
    return any(
        token in name
        for token in ("tok_embeddings", "wte", "wpe", "embed_tokens", "embedding")
    )


def _is_output_head(name: str) -> bool:
    """Check if parameter name indicates the output/LM head."""
    # TorchTitan Llama3 uses "output.weight" for the LM head
    parts = name.split(".")
    return any(
        token in parts
        for token in ("output", "lm_head", "head")
    )


def _is_transformer_weight(name: str) -> bool:
    """Check if parameter is a transformer block weight matrix."""
    # Explicitly match known transformer weight patterns
    transformer_patterns = (
        # Llama3 / TorchTitan naming
        "attention.wq", "attention.wk", "attention.wv", "attention.wo",
        "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
        # GPT-style naming
        "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj",
        # Generic transformer naming
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )
    return any(pattern in name for pattern in transformer_patterns)


def count_params(groups: ParamGroups) -> dict[str, int]:
    """Count parameters in each group."""
    return {
        "matrix": sum(p.numel() for p in groups.matrix_params),
        "embed": sum(p.numel() for p in groups.embed_params),
        "output": sum(p.numel() for p in groups.output_params),
        "norm": sum(p.numel() for p in groups.norm_params),
        "total": sum(
            p.numel()
            for group in groups
            for p in group
        ),
    }
