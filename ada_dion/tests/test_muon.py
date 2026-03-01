"""Tests for the Muon optimizer (official microsoft/dion package)."""
import pytest
import torch
import torch.nn as nn

from dion import Muon


class TestMuonBasics:
    def test_single_step_no_nan(self):
        p = nn.Parameter(torch.randn(32, 16))
        opt = Muon([p], lr=0.02, mu=0.95)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()

    def test_parameter_changes(self):
        p = nn.Parameter(torch.randn(32, 16))
        p_before = p.data.clone()
        opt = Muon([p], lr=0.02, mu=0.95)
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        assert not torch.equal(p.data, p_before)

    def test_weight_decay(self):
        """Weight decay should shrink parameters."""
        p = nn.Parameter(torch.ones(16, 8))
        opt = Muon([p], lr=0.02, mu=0.95, weight_decay=0.1)
        p.grad = torch.zeros_like(p)
        norm_before = p.data.norm().item()
        opt.step()
        norm_after = p.data.norm().item()
        assert norm_after < norm_before

    def test_multiple_shapes(self):
        """Test with different matrix shapes."""
        for shape in [(64, 32), (32, 64), (128, 128)]:
            p = nn.Parameter(torch.randn(*shape))
            opt = Muon([p], lr=0.02, mu=0.95)
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
            assert not torch.isnan(p).any(), f"NaN for shape {shape}"


class TestMuonConvergence:
    def test_loss_decreases(self):
        """Verify loss decreases over multiple steps."""
        torch.manual_seed(42)
        p = nn.Parameter(torch.randn(32, 16))
        opt = Muon([p], lr=0.01, mu=0.9)

        losses = []
        for _ in range(50):
            opt.zero_grad()
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease"
