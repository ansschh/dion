"""Tests for the Dion optimizer."""
import pytest
import torch
import torch.nn as nn

from ada_dion.optim.dion import Dion


class TestDionBasics:
    def test_single_step_no_nan(self):
        p = nn.Parameter(torch.randn(32, 16))
        opt = Dion([p], lr=0.02, rank_frac=0.25, beta=0.05)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()

    def test_parameter_changes(self):
        p = nn.Parameter(torch.randn(32, 16))
        p_before = p.data.clone()
        opt = Dion([p], lr=0.02, rank_frac=0.25, beta=0.05)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.equal(p.data, p_before)

    def test_state_initialized(self):
        """After one step, state should have M and V."""
        p = nn.Parameter(torch.randn(32, 16))
        opt = Dion([p], lr=0.02, rank_frac=0.25)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        state = opt.state[p]
        assert "M" in state
        assert "V" in state
        # M shape should match param
        assert state["M"].shape == (32, 16)
        # V shape: (n, r) where r = round(min(32,16) * 0.25) = round(4) = 4
        assert state["V"].shape == (16, 4)

    def test_v_column_normalization(self):
        """V should have unit-norm columns after col_norm."""
        p = nn.Parameter(torch.randn(64, 32))
        opt = Dion([p], lr=0.02, rank_frac=0.25, beta=0.05)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()

        V = opt.state[p]["V"]
        col_norms = V.norm(dim=0)
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5), \
            f"V columns should have unit norm, got norms: {col_norms}"

    def test_error_feedback_reduces_momentum(self):
        """Error feedback should reduce the momentum norm."""
        p = nn.Parameter(torch.randn(32, 16))
        opt = Dion([p], lr=0.02, rank_frac=0.25, beta=0.5)  # High beta for strong feedback
        p.grad = torch.ones_like(p)  # Large gradient
        opt.step()

        # After error feedback, M should be smaller than just G
        M_norm = opt.state[p]["M"].norm().item()
        G_norm = torch.ones(32, 16).norm().item()
        assert M_norm < G_norm, "Error feedback should reduce momentum norm"

    def test_rank_uses_min_dim(self):
        """Rank should be computed from min(m, n), not just n."""
        # Wide matrix: m=16, n=64, min=16, rank_frac=0.5 -> r=8
        p_wide = nn.Parameter(torch.randn(16, 64))
        opt_wide = Dion([p_wide], lr=0.02, rank_frac=0.5)
        p_wide.grad = torch.randn_like(p_wide)
        opt_wide.step()
        assert opt_wide.get_ranks()[0] == 8  # min(16, 64) * 0.5 = 8

        # Tall matrix: m=64, n=16, min=16, rank_frac=0.5 -> r=8
        p_tall = nn.Parameter(torch.randn(64, 16))
        opt_tall = Dion([p_tall], lr=0.02, rank_frac=0.5)
        p_tall.grad = torch.randn_like(p_tall)
        opt_tall.step()
        assert opt_tall.get_ranks()[0] == 8  # min(64, 16) * 0.5 = 8

    def test_different_rank_fracs(self):
        for rf in [0.1, 0.25, 0.5, 1.0]:
            p = nn.Parameter(torch.randn(32, 16))
            opt = Dion([p], lr=0.02, rank_frac=rf)
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            assert not torch.isnan(p).any(), f"NaN with rank_frac={rf}"

    def test_weight_decay(self):
        p = nn.Parameter(torch.ones(16, 8))
        opt = Dion([p], lr=0.02, rank_frac=0.25, weight_decay=0.1)
        p.grad = torch.zeros_like(p)
        norm_before = p.data.norm().item()
        opt.step()
        norm_after = p.data.norm().item()
        assert norm_after < norm_before

    def test_fused_update_no_full_matrix(self):
        """Verify step completes successfully (fused addmm_ path)."""
        p = nn.Parameter(torch.randn(64, 32))
        opt = Dion([p], lr=0.02, rank_frac=0.25)
        p.grad = torch.randn_like(p)
        p_before = p.data.clone()
        opt.step()
        assert not torch.equal(p.data, p_before)
        assert not torch.isnan(p).any()


class TestDionConvergence:
    def test_loss_decreases(self):
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 16))
        opt = Dion([p], lr=0.01, rank_frac=0.25, beta=0.05)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease"
