"""Tests for the HybridOptimizersContainer."""
import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Minimal model that mimics Llama3 naming for param grouper."""

    def __init__(self, dim=32, hidden_dim=64, vocab_size=100):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        # One transformer layer
        self.layers = nn.ModuleList([nn.Module()])
        attention = nn.Module()
        attention.wq = nn.Linear(dim, dim, bias=False)
        attention.wk = nn.Linear(dim, dim, bias=False)
        attention.wv = nn.Linear(dim, dim, bias=False)
        attention.wo = nn.Linear(dim, dim, bias=False)
        self.layers[0].attention = attention
        self.layers[0].attention_norm = nn.LayerNorm(dim)

        feed_forward = nn.Module()
        feed_forward.w1 = nn.Linear(dim, hidden_dim, bias=False)
        feed_forward.w2 = nn.Linear(hidden_dim, dim, bias=False)
        feed_forward.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.layers[0].feed_forward = feed_forward
        self.layers[0].ffn_norm = nn.LayerNorm(dim)

        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        return x


class TestHybridOptimizersContainerUnit:
    """Unit tests that don't require TorchTitan imports."""

    def test_param_grouper_with_simple_model(self):
        from ada_dion.integration.param_grouper import group_params_for_hybrid

        model = SimpleModel()
        groups = group_params_for_hybrid(model)

        # Should have matrix params (attention + ff weights)
        assert len(groups.matrix_params) == 7  # wq, wk, wv, wo, w1, w2, w3
        # Embedding
        assert len(groups.embed_params) == 1
        # Output
        assert len(groups.output_params) == 1
        # Norms: attention_norm(w,b) + ffn_norm(w,b) + top norm(w,b) = 6
        assert len(groups.norm_params) >= 3

    def test_optimizers_can_step_on_groups(self):
        """Test that our optimizers work on grouped params."""
        from ada_dion.optim.muon import Muon
        from ada_dion.optim.dion import Dion
        from ada_dion.optim.dion2 import Dion2
        from ada_dion.integration.param_grouper import group_params_for_hybrid

        model = SimpleModel()
        groups = group_params_for_hybrid(model)

        # Test each matrix optimizer
        for opt_cls, kwargs in [
            (Muon, {"lr": 0.02, "mu": 0.95}),
            (Dion, {"lr": 0.02, "rank_frac": 0.25}),
            (Dion2, {"lr": 0.02, "alpha": 0.5, "mu": 0.95}),
        ]:
            # Reset model
            model = SimpleModel()
            groups = group_params_for_hybrid(model)

            m_opt = opt_cls(groups.matrix_params, **kwargs)
            s_opt = torch.optim.AdamW(
                groups.embed_params + groups.output_params + groups.norm_params,
                lr=3e-4,
            )

            # Assign gradients
            for p in model.parameters():
                if p.requires_grad:
                    p.grad = torch.randn_like(p) * 0.01

            # Step both
            m_opt.step()
            s_opt.step()

            # Verify no NaN in any parameter
            for name, p in model.named_parameters():
                assert not torch.isnan(p).any(), f"NaN in {name} with {opt_cls.__name__}"

    def test_all_params_get_updated(self):
        """Both matrix and scalar params should change after a step."""
        from ada_dion.optim.muon import Muon
        from ada_dion.integration.param_grouper import group_params_for_hybrid

        model = SimpleModel()
        groups = group_params_for_hybrid(model)

        # Save original values
        originals = {name: p.data.clone() for name, p in model.named_parameters()}

        m_opt = Muon(groups.matrix_params, lr=0.02, mu=0.95)
        s_opt = torch.optim.AdamW(
            groups.embed_params + groups.output_params + groups.norm_params,
            lr=3e-4,
        )

        # Assign non-zero gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p) * 0.1

        m_opt.step()
        s_opt.step()

        # Check at least some params changed
        changed = 0
        for name, p in model.named_parameters():
            if not torch.equal(p.data, originals[name]):
                changed += 1

        total = sum(1 for _ in model.parameters())
        assert changed > 0, "No parameters were updated"
        assert changed == total, f"Only {changed}/{total} params updated"
