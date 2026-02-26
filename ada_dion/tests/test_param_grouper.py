"""Tests for the parameter grouper."""
import pytest
import torch
import torch.nn as nn

from ada_dion.integration.param_grouper import group_params_for_hybrid, count_params


class MockLlamaBlock(nn.Module):
    """Minimal mock that mimics Llama3 parameter naming."""

    def __init__(self, dim=64, hidden_dim=128):
        super().__init__()
        self.tok_embeddings = nn.Embedding(1000, dim)
        self.layers = nn.ModuleList([
            self._make_layer(dim, hidden_dim) for _ in range(2)
        ])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, 1000, bias=False)

    def _make_layer(self, dim, hidden_dim):
        layer = nn.Module()
        # Attention
        attention = nn.Module()
        attention.wq = nn.Linear(dim, dim, bias=False)
        attention.wk = nn.Linear(dim, dim, bias=False)
        attention.wv = nn.Linear(dim, dim, bias=False)
        attention.wo = nn.Linear(dim, dim, bias=False)
        layer.attention = attention
        layer.attention_norm = nn.LayerNorm(dim)
        # Feed-forward
        feed_forward = nn.Module()
        feed_forward.w1 = nn.Linear(dim, hidden_dim, bias=False)
        feed_forward.w2 = nn.Linear(hidden_dim, dim, bias=False)
        feed_forward.w3 = nn.Linear(dim, hidden_dim, bias=False)
        layer.feed_forward = feed_forward
        layer.ffn_norm = nn.LayerNorm(dim)
        return layer

    def forward(self, x):
        return x


class TestParamGrouper:
    @pytest.fixture
    def model(self):
        return MockLlamaBlock(dim=64, hidden_dim=128)

    def test_embedding_classified(self, model):
        groups = group_params_for_hybrid(model)
        # tok_embeddings.weight should be in embed_params
        assert len(groups.embed_params) == 1  # tok_embeddings.weight

    def test_output_classified(self, model):
        groups = group_params_for_hybrid(model)
        assert len(groups.output_params) == 1  # output.weight

    def test_matrix_params_classified(self, model):
        groups = group_params_for_hybrid(model)
        # 2 layers x (wq + wk + wv + wo + w1 + w2 + w3) = 2 x 7 = 14
        assert len(groups.matrix_params) == 14

    def test_norm_params_classified(self, model):
        groups = group_params_for_hybrid(model)
        # 2 layers x (attention_norm + ffn_norm) + top-level norm
        # Each LayerNorm has weight + bias = 2 params
        # So: 2 * 2 * 2 + 1 * 2 = 10 norm params
        assert len(groups.norm_params) >= 5  # at least the weights

    def test_all_params_accounted(self, model):
        groups = group_params_for_hybrid(model)
        total_grouped = (
            len(groups.matrix_params)
            + len(groups.embed_params)
            + len(groups.output_params)
            + len(groups.norm_params)
        )
        total_model = sum(1 for p in model.parameters() if p.requires_grad)
        assert total_grouped == total_model, (
            f"Grouped {total_grouped} but model has {total_model} params"
        )

    def test_count_params(self, model):
        groups = group_params_for_hybrid(model)
        counts = count_params(groups)
        assert counts["total"] > 0
        assert counts["matrix"] > 0
        assert counts["embed"] > 0
        assert counts["output"] > 0
        assert counts["norm"] >= 0

    def test_no_overlap(self, model):
        """No parameter should appear in multiple groups."""
        groups = group_params_for_hybrid(model)
        all_params = (
            groups.matrix_params
            + groups.embed_params
            + groups.output_params
            + groups.norm_params
        )
        param_ids = [id(p) for p in all_params]
        assert len(param_ids) == len(set(param_ids)), "Duplicate params in groups"
